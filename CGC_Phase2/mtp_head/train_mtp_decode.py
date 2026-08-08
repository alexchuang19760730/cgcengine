"""跨模型 Decode-mode MTP 训练 (统一版, 支持大模型 sequential 模式).

支持任意在 model_registry 中注册的模型 (gemma4 / dsv4 / qwen3vl).
通过 --model-name 选择模型, 所有架构参数/EOS/checkpoint 路径从 registry 自动获取.

关键修复: 用 decode hidden (非 prefill hidden) 收集训练数据.
关键修复: 修复 train_device 被误传为 lr 的 bug.
新增: --sequential 模式 — 单进程 + device_map="auto" 跨多 GPU 加载大模型 (DSV4 149GB).
新增: 训练阶段只加载 embed/lm_head 权重 (不加载完整模型), 大幅节省 VRAM 和时间.

用法:
  # Gemma4 (sglang 在 GPU 0-3, 训练用 GPU 4-7)
  python train_mtp_decode.py --model-name gemma4 --gpu-base 4 --world-size 4

  # DSV4 Flash (需先停 sglang, 全 GPU, sequential 模式)
  python train_mtp_decode.py --model-name dsv4 --sequential

  # Qwen3-VL-2B (小模型, 单卡)
  python train_mtp_decode.py --model-name qwen3vl --world-size 1

  # 只收集数据 (不训练)
  python train_mtp_decode.py --model-name gemma4 --mode collect --gpu-base 4

  # 只训练 (已有 shard)
  python train_mtp_decode.py --model-name dsv4 --mode train --sequential
"""
from __future__ import annotations

import json
import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional


# ============================================================================
# Model Registry 集成
# ============================================================================

def _get_repo_root() -> str:
    """自动检测 repo root (本地 /root/flashkv0516 或 Host1)."""
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "/root/flashkv0516",
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "app", "shared")):
            return c
    return candidates[0]


REPO_ROOT = _get_repo_root()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, os.path.join(REPO_ROOT, "app", "shared"))
    sys.path.insert(0, os.path.join(REPO_ROOT, "CGC_Phase2", "mtp_head"))


def _ensure_path():
    """确保 sys.path 包含所有需要的目录 (mp.spawn 子进程不继承父进程 sys.path)."""
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for p in [_repo, os.path.join(_repo, "app", "shared"),
              os.path.join(_repo, "CGC_Phase2", "mtp_head")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def get_model_config(model_name: str):
    """从 registry 获取模型配置."""
    _ensure_path()
    from app.shared.model_registry import get_model_config as _get
    return _get(model_name)


# ============================================================================
# 模型加载 (统一, 支持所有模型)
# ============================================================================

def load_base_model(model_path: str, device: str = "cuda"):
    """加载 base model + tokenizer (通过统一 model_loader).

    device="auto" 时使用 device_map="auto" 跨多 GPU 加载 (大模型).
    """
    _ensure_path()
    from model_loader import load_base_model as _load, get_embed_weight, get_lm_head_weight
    model, tokenizer = _load(model_path, device=device)
    embed = get_embed_weight(model)
    lm_head = get_lm_head_weight(model)
    if embed is None:
        raise RuntimeError(f"Cannot find embed_tokens for model at {model_path}")
    if lm_head is None:
        print("[WARN] lm_head not found, will use embed_weights (tied)")
    return model, tokenizer


def load_embed_and_lm_head_only(model_path: str, device: str = "cuda"):
    """只加载 embed_tokens 和 lm_head 权重 (训练阶段, 避免加载完整模型).

    对于大模型 (如 DSV4 149GB), 训练时只需 embed + lm_head (~2GB),
    无需加载完整模型权重.
    """
    _ensure_path()
    import glob
    from transformers import AutoTokenizer, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 检查是否 tied embeddings
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

    embed_weight = None
    lm_head_weight = None

    # 方法 1: 从 safetensors index 精准加载
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    if os.path.exists(index_file):
        from safetensors.torch import load_file
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

        # 找到 embed_tokens 和 lm_head 所在的文件
        files_to_load = set()
        embed_key = None
        lm_head_key = None

        for k, v in weight_map.items():
            k_lower = k.lower()
            if k.endswith(".weight"):
                # embed: standard "embed_tokens.weight" or DSV4 "embed.weight"
                if "embed_tokens" in k_lower or k_lower == "embed.weight":
                    embed_key = k
                    files_to_load.add(v)
                # lm_head: standard "lm_head.weight" or DSV4 "head.weight"
                if "lm_head" in k_lower or k_lower == "head.weight":
                    lm_head_key = k
                    files_to_load.add(v)

        print(f"[load_weights] embed_key={embed_key}, lm_head_key={lm_head_key}, "
              f"files={files_to_load}, tied={tie_word_embeddings}", flush=True)

        for f_name in files_to_load:
            f_path = os.path.join(model_path, f_name)
            state = load_file(f_path, device="cpu")
            if embed_key and embed_key in state:
                embed_weight = state[embed_key].to(device)
                print(f"[load_weights] Loaded embed: {embed_key} from {f_name} "
                      f"shape={embed_weight.shape}", flush=True)
            if lm_head_key and lm_head_key in state:
                lm_head_weight = state[lm_head_key].to(device)
                print(f"[load_weights] Loaded lm_head: {lm_head_key} from {f_name} "
                      f"shape={lm_head_weight.shape}", flush=True)
            del state

    elif st_files:
        from safetensors.torch import load_file
        for f_path in st_files:
            state = load_file(f_path, device="cpu")
            for k, v in state.items():
                k_lower = k.lower()
                if k.endswith(".weight"):
                    if "embed_tokens" in k_lower or k_lower == "embed.weight":
                        embed_weight = v.to(device)
                        print(f"[load_weights] Loaded embed: {k} from {os.path.basename(f_path)} "
                              f"shape={embed_weight.shape}", flush=True)
                    if "lm_head" in k_lower or k_lower == "head.weight":
                        lm_head_weight = v.to(device)
                        print(f"[load_weights] Loaded lm_head: {k} from {os.path.basename(f_path)} "
                              f"shape={lm_head_weight.shape}", flush=True)
            del state

    # Fallback: try pytorch_model.bin
    if embed_weight is None:
        bin_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))
        if bin_files:
            for f_path in bin_files:
                state = torch.load(f_path, map_location="cpu", weights_only=True)
                for k, v in state.items():
                    k_lower = k.lower()
                    if k.endswith(".weight"):
                        if "embed_tokens" in k_lower or k_lower == "embed.weight":
                            embed_weight = v.to(device)
                        if "lm_head" in k_lower or k_lower == "head.weight":
                            lm_head_weight = v.to(device)
                del state

    if embed_weight is None:
        raise RuntimeError(f"Cannot find embed_tokens weight in {model_path}")

    if lm_head_weight is None and tie_word_embeddings:
        lm_head_weight = embed_weight
        print("[load_weights] Using tied embeddings (lm_head = embed)", flush=True)
    elif lm_head_weight is None:
        print("[WARN] lm_head not found, using embed (tied assumption)", flush=True)
        lm_head_weight = embed_weight

    return embed_weight, lm_head_weight, tokenizer


def load_mtp_head(model_name: str, mtp_checkpoint: str, device: str, lm_head_weight):
    """加载 MTP head (通过 model_registry 自动选择正确配置)."""
    _ensure_path()
    from model import create_mtp_head_by_model_name
    mtp = create_mtp_head_by_model_name(model_name)
    mtp.set_shared_lm_head(lm_head_weight)
    if mtp_checkpoint and os.path.exists(mtp_checkpoint):
        ckpt = torch.load(mtp_checkpoint, weights_only=False, map_location="cpu")
        mtp.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[mtp] Loaded checkpoint: {mtp_checkpoint}")
    else:
        print(f"[mtp] No checkpoint for {model_name}, starting from scratch")
    mtp.to(device).to(torch.bfloat16)
    return mtp


# ============================================================================
# 数据收集 (decode-mode, 跨模型通用, 支持 multi-GPU)
# ============================================================================

def extract_text(entry: dict) -> str:
    if "text" in entry and entry["text"]:
        return entry["text"]
    if "messages" in entry:
        return " ".join(m.get("content", "") for m in entry["messages"])
    if "instruction" in entry:
        text = entry.get("instruction", "")
        if entry.get("input"):
            text += "\n" + entry["input"]
        if entry.get("output"):
            text += "\n" + entry["output"]
        return text
    return ""


def collect_decode_data(
    base_model,
    tokenizer,
    mtp_head,
    text: str,
    num_chain: int = 4,
    gen_length: int = 50,
    device: str = "cuda",
    eos_tokens: set[int] | None = None,
    compute_device: str | None = None,
    embed_weights=None,
    lm_head_weight=None,
) -> list[dict]:
    """Decode-mode 数据收集 — 从生成序列中收集 decode hidden.

    训练样本: (decode_hidden[i], token[i]) -> token[i+1]
    跨模型通用: eos_tokens 从 model_registry 传入.
    支持多 GPU: compute_device 指定 MTP head 计算设备.
    """
    _ensure_path()
    from model_loader import get_embed_weight, get_lm_head_weight

    if eos_tokens is None:
        eos_tokens = {1}

    if compute_device is None:
        compute_device = device

    input_ids = tokenizer.encode(text, add_special_tokens=False)
    # 过滤 EOS tokens
    input_ids = [t for t in input_ids if t not in eos_tokens]
    if len(input_ids) < 3:
        return []

    if len(input_ids) > 256:
        input_ids = input_ids[:256]

    input_tensor = torch.tensor([input_ids], device=device)

    if embed_weights is None:
        embed_weights = get_embed_weight(base_model)
    if lm_head_weight is None:
        lm_head_weight = get_lm_head_weight(base_model)
        if lm_head_weight is None:
            lm_head_weight = embed_weights  # tied embeddings

    # 多 GPU 模型: 将权重移到 compute_device (只移动一次)
    if str(embed_weights.device) != compute_device:
        embed_weights = embed_weights.to(compute_device)
    if str(lm_head_weight.device) != compute_device:
        lm_head_weight = lm_head_weight.to(compute_device)

    # 1. Prefill (use_cache=True)
    with torch.no_grad():
        outputs = base_model(input_tensor, output_hidden_states=True, use_cache=True)
    kv_cache = outputs.past_key_values
    prefill_last_hidden = outputs.hidden_states[-1][0, -1]  # [hidden]
    if str(prefill_last_hidden.device) != compute_device:
        prefill_last_hidden = prefill_last_hidden.to(compute_device)

    # 2. 生成第一个 token
    with torch.no_grad():
        first_logits = F.linear(prefill_last_hidden, lm_head_weight)
    first_token = int(first_logits.argmax().item())

    # 3. 逐 token decode forward, 收集 decode hidden
    decode_hiddens = []
    decode_tokens = []
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 1

    current_token = first_token
    for step in range(gen_length):
        with torch.no_grad():
            decode_out = base_model(
                torch.tensor([[current_token]], device=device),
                past_key_values=kv_cache,
                output_hidden_states=True,
                use_cache=True,
            )
        decode_hidden = decode_out.hidden_states[-1][0, 0]  # [hidden] decode hidden
        if str(decode_hidden.device) != compute_device:
            decode_hidden = decode_hidden.to(compute_device)
        kv_cache = decode_out.past_key_values

        with torch.no_grad():
            next_logits = F.linear(decode_hidden, lm_head_weight)
        next_token = int(next_logits.argmax().item())

        decode_hiddens.append(decode_hidden)
        decode_tokens.append(current_token)

        if next_token == eos_token_id or next_token in eos_tokens:
            break
        current_token = next_token

    if len(decode_hiddens) < num_chain + 1:
        return []

    # 4. 构造链式训练样本
    samples = []
    mtp_head.eval()

    for i in range(len(decode_hiddens) - num_chain):
        chain_hidden = []
        chain_tokens = []
        chain_next_tokens = []

        current_hidden = decode_hiddens[i]  # decode hidden (正确!)

        for k in range(num_chain):
            token_id = decode_tokens[i + k]
            next_token_id = decode_tokens[i + k + 1] if i + k + 1 < len(decode_tokens) else token_id

            with torch.no_grad():
                token_embed = embed_weights[token_id]
                h_3d = current_hidden.unsqueeze(0).unsqueeze(0)
                e_3d = token_embed.unsqueeze(0).unsqueeze(0)
                concat = torch.cat([h_3d, e_3d], dim=-1)
                x = mtp_head.proj(concat)
                x = mtp_head.norm1(x)
                x = x + mtp_head.attn(x)
                x = mtp_head.norm2(x)
                x = x + mtp_head.mlp(x)
                mtp_out = mtp_head.norm_out(x)

            chain_hidden.append(current_hidden.cpu())
            chain_tokens.append(token_id)
            chain_next_tokens.append(next_token_id)
            current_hidden = mtp_out[0, 0]

        samples.append({
            "hidden_states": torch.stack(chain_hidden),
            "token_ids": torch.tensor(chain_tokens),
            "next_token_ids": torch.tensor(chain_next_tokens),
        })

    return samples


# ============================================================================
# Phase 1a: 多卡并行收集 (小/中模型)
# ============================================================================

def collect_worker(
    rank: int,
    world_size: int,
    model_name: str,
    corpus_path: str,
    base_model_path: str,
    mtp_checkpoint: str,
    shard_dir: str,
    num_chain: int,
    max_samples_per_worker: int,
    gen_length: int,
    gpu_base: int,
):
    # mp.spawn 子进程不继承父进程 sys.path, 必须重新设置
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    _shared = os.path.join(_repo, "app", "shared")
    if _shared not in sys.path:
        sys.path.insert(0, _shared)
    _mtp_dir = os.path.join(_repo, "CGC_Phase2", "mtp_head")
    if _mtp_dir not in sys.path:
        sys.path.insert(0, _mtp_dir)

    actual_gpu = gpu_base + rank
    torch.cuda.set_device(actual_gpu)
    device = f"cuda:{actual_gpu}"

    # 从 registry 获取配置
    cfg = get_model_config(model_name)
    print(f"[rank {rank}] Loading {cfg.display_name} on {device}...", flush=True)

    base_model, tokenizer = load_base_model(base_model_path, device)
    base_model.eval()

    from model_loader import get_embed_weight, get_lm_head_weight
    embed_weights = get_embed_weight(base_model)
    lm_head_weight = get_lm_head_weight(base_model)
    if lm_head_weight is None:
        lm_head_weight = embed_weights

    mtp = load_mtp_head(model_name, mtp_checkpoint, device, lm_head_weight)
    mtp.eval()

    with open(corpus_path) as f:
        all_lines = f.readlines()
    total = len(all_lines)
    start = rank * total // world_size
    end = (rank + 1) * total // world_size
    my_lines = all_lines[start:end]
    print(f"[rank {rank}] corpus[{start}:{end}] = {len(my_lines)} entries, target {max_samples_per_worker} samples", flush=True)

    all_samples = []
    t0 = time.time()
    for i, line in enumerate(my_lines):
        if len(all_samples) >= max_samples_per_worker:
            break
        try:
            entry = json.loads(line)
        except Exception:
            continue
        text = extract_text(entry)
        if not text or len(text) < 20:
            continue
        try:
            samples = collect_decode_data(
                base_model, tokenizer, mtp, text,
                num_chain=num_chain, gen_length=gen_length, device=device,
                eos_tokens=cfg.eos_tokens,
                compute_device=device,
                embed_weights=embed_weights,
                lm_head_weight=lm_head_weight,
            )
            all_samples.extend(samples)
        except Exception as e:
            continue
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(my_lines) - i - 1) / rate if rate > 0 else 0
            print(f"[rank {rank}] {i+1}/{len(my_lines)} entries, {len(all_samples)} samples, {rate:.1f} entry/s, ETA {eta:.0f}s", flush=True)

    shard_path = f"{shard_dir}/shard_{rank}.pt"
    torch.save(all_samples, shard_path)
    elapsed = time.time() - t0
    print(f"[rank {rank}] DONE: {len(all_samples)} samples in {elapsed:.0f}s -> {shard_path}", flush=True)


# ============================================================================
# Phase 1b: Sequential 收集 (大模型, device_map="auto")
# ============================================================================

def collect_sequential(
    model_name: str,
    corpus_path: str,
    base_model_path: str,
    mtp_checkpoint: str,
    shard_dir: str,
    num_chain: int,
    max_samples: int,
    gen_length: int,
    num_shards: int = 4,
):
    """Sequential collect for large models (DSV4) — single process, device_map='auto'.

    大模型无法放入单 GPU, 使用 device_map="auto" 跨多 GPU 加载.
    单进程运行, 避免每个 worker 都加载完整模型.
    可选: 将结果分多个 shard 保存 (避免单 shard 过大).
    """
    _ensure_path()
    cfg = get_model_config(model_name)
    print(f"[sequential] Loading {cfg.display_name} with device_map='auto'...", flush=True)

    # 用 device_map="auto" 加载, 跨所有可用 GPU
    base_model, tokenizer = load_base_model(base_model_path, device="auto")
    base_model.eval()

    from model_loader import get_embed_weight, get_lm_head_weight
    embed_weights = get_embed_weight(base_model)
    lm_head_weight = get_lm_head_weight(base_model)
    if lm_head_weight is None:
        lm_head_weight = embed_weights

    # 确定 compute_device: lm_head 所在的 GPU (通常是最后一个 GPU)
    if lm_head_weight is not None:
        compute_device = str(lm_head_weight.device)
    else:
        compute_device = "cuda:0"
    print(f"[sequential] compute_device={compute_device}", flush=True)

    # 确定 input_device: 模型第一个参数所在的 GPU
    input_device = str(next(base_model.parameters()).device)
    print(f"[sequential] input_device={input_device}", flush=True)

    # MTP head 放在 compute_device
    mtp = load_mtp_head(model_name, mtp_checkpoint, compute_device, lm_head_weight)
    mtp.eval()

    # 将 embed/lm_head 移到 compute_device
    if str(embed_weights.device) != compute_device:
        embed_weights = embed_weights.to(compute_device)
    if str(lm_head_weight.device) != compute_device:
        lm_head_weight = lm_head_weight.to(compute_device)

    with open(corpus_path) as f:
        all_lines = f.readlines()
    print(f"[sequential] corpus: {len(all_lines)} entries, target {max_samples} samples, "
          f"gen_length={gen_length}", flush=True)

    all_samples = []
    t0 = time.time()
    shard_idx = 0
    samples_per_shard = max_samples // num_shards if num_shards > 1 else max_samples

    for i, line in enumerate(all_lines):
        if len(all_samples) >= max_samples:
            break
        try:
            entry = json.loads(line)
        except Exception:
            continue
        text = extract_text(entry)
        if not text or len(text) < 20:
            continue
        try:
            samples = collect_decode_data(
                base_model, tokenizer, mtp, text,
                num_chain=num_chain, gen_length=gen_length,
                device=input_device,
                eos_tokens=cfg.eos_tokens,
                compute_device=compute_device,
                embed_weights=embed_weights,
                lm_head_weight=lm_head_weight,
            )
            all_samples.extend(samples)
        except Exception as e:
            print(f"[sequential] Error on entry {i}: {e}", flush=True)
            continue

        # 定期保存 shard (避免内存爆炸)
        if len(all_samples) >= (shard_idx + 1) * samples_per_shard and shard_idx < num_shards - 1:
            shard_path = f"{shard_dir}/shard_{shard_idx}.pt"
            torch.save(all_samples[shard_idx * samples_per_shard:(shard_idx + 1) * samples_per_shard], shard_path)
            print(f"[sequential] Saved shard_{shard_idx}.pt "
                  f"({len(all_samples[shard_idx * samples_per_shard:(shard_idx + 1) * samples_per_shard])} samples)", flush=True)
            shard_idx += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(all_lines) - i - 1) / rate if rate > 0 else 0
            print(f"[sequential] {i+1}/{len(all_lines)} entries, {len(all_samples)} samples, "
                  f"{rate:.1f} entry/s, ETA {eta:.0f}s", flush=True)

    # 保存剩余样本
    remaining = all_samples[shard_idx * samples_per_shard:] if shard_idx > 0 else all_samples
    shard_path = f"{shard_dir}/shard_{shard_idx}.pt"
    torch.save(remaining, shard_path)
    elapsed = time.time() - t0
    print(f"[sequential] DONE: {len(all_samples)} samples in {elapsed:.0f}s -> {shard_path}", flush=True)

    # 释放大模型显存
    del base_model
    del mtp
    torch.cuda.empty_cache()
    print(f"[sequential] Model unloaded, VRAM freed", flush=True)


# ============================================================================
# Phase 2: 训练 (只用 embed+lm_head 权重, 不加载完整模型)
# ============================================================================

def train_on_shards(
    model_name: str,
    base_model_path: str,
    mtp_checkpoint: str,
    shard_dir: str,
    output_dir: str,
    num_chain: int = 4,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    device: str = "cuda:0",
    sequential: bool = False,
):
    cfg = get_model_config(model_name)
    print(f"[train] Model: {cfg.display_name} ({model_name})", flush=True)

    # 训练阶段: 只加载 embed + lm_head 权重 (不加载完整模型!)
    # 这对大模型 (DSV4 149GB) 尤其重要, 从 ~150GB 降到 ~2GB
    print(f"[train] Loading embed+lm_head weights only (no full model) on {device}...", flush=True)
    embed_weights, lm_head_weight, tokenizer = load_embed_and_lm_head_only(base_model_path, device)
    print(f"[train] embed shape={embed_weights.shape}, lm_head shape={lm_head_weight.shape}", flush=True)

    # MTP head 在同一个 device
    mtp = load_mtp_head(model_name, mtp_checkpoint, device, lm_head_weight)
    mtp.train()

    shard_paths = sorted(Path(shard_dir).glob("shard_*.pt"))
    print(f"[train] Merging {len(shard_paths)} shards...", flush=True)
    all_samples = []
    for p in shard_paths:
        shard = torch.load(p, weights_only=False)
        print(f"  {p.name}: {len(shard)} samples", flush=True)
        all_samples.extend(shard)
    print(f"[train] Total: {len(all_samples)} samples", flush=True)

    if len(all_samples) == 0:
        print("[train] No samples! Aborting.", flush=True)
        return

    optimizer = torch.optim.AdamW(mtp.parameters(), lr=lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(all_samples) // batch_size
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    avg_loss = 0.0
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        random.shuffle(all_samples)
        t0 = time.time()

        for i in range(0, len(all_samples), batch_size):
            batch = all_samples[i:i + batch_size]
            if len(batch) < 2:
                continue

            hidden = torch.stack([s["hidden_states"] for s in batch]).to(device)
            tokens = torch.stack([s["token_ids"] for s in batch]).to(device)
            next_tokens = torch.stack([s["next_token_ids"] for s in batch]).to(device)
            hidden = hidden.to(torch.bfloat16)

            loss = 0
            current_hidden = hidden[:, 0, :]

            for k in range(num_chain):
                token_embed = F.embedding(tokens[:, k], embed_weights)
                concat_input = torch.cat(
                    [current_hidden.unsqueeze(1), token_embed.unsqueeze(1)], dim=-1
                )
                x = mtp.proj(concat_input)
                x = mtp.norm1(x)
                x = x + mtp.attn(x)
                x = mtp.norm2(x)
                x = x + mtp.mlp(x)
                mtp_hidden = mtp.norm_out(x)

                logits = F.linear(mtp_hidden[:, 0, :], lm_head_weight)
                loss += F.cross_entropy(logits, next_tokens[:, k])
                current_hidden = mtp_hidden[:, 0, :].detach()

            loss = loss / num_chain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                elapsed = time.time() - t0
                rate = num_batches / elapsed
                eta = (len(all_samples) // batch_size - num_batches) / rate if rate > 0 else 0
                log(f"  Epoch {epoch+1} batch {num_batches}/{len(all_samples)//batch_size}: loss={loss.item():.4f}, {rate:.1f} batch/s, ETA {eta:.0f}s")

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0
        log(f"[train] Epoch {epoch+1}: avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")

    output_path = Path(output_dir) / f"mtp_head_{model_name}_decode.pt"
    torch.save({
        "model_state_dict": mtp.state_dict(),
        "epoch": epochs,
        "loss": avg_loss,
        "num_chain": num_chain,
        "training": f"{model_name}_decode_chained_multi",
        "num_samples": len(all_samples),
        "hidden_type": "decode",
        "model": cfg.display_name,
        "model_name": model_name,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
    }, output_path)
    log(f"[train] Saved: {output_path}")

    # 评估 (训练集上, 仅参考)
    mtp.eval()
    chain_correct = 0
    chain_total = 0
    with torch.no_grad():
        for s in all_samples[:200]:
            hidden_states = s["hidden_states"].to(device).to(torch.bfloat16)
            token_ids = s["token_ids"].to(device)
            next_token_ids = s["next_token_ids"].to(device)
            current_hidden = hidden_states[0]
            for k in range(num_chain):
                token_embed = F.embedding(token_ids[k], embed_weights)
                h_3d = current_hidden.unsqueeze(0).unsqueeze(0)
                e_3d = token_embed.unsqueeze(0).unsqueeze(0)
                concat_input = torch.cat([h_3d, e_3d], dim=-1)
                x = mtp.proj(concat_input)
                x = mtp.norm1(x)
                x = x + mtp.attn(x)
                x = mtp.norm2(x)
                x = x + mtp.mlp(x)
                mtp_out = mtp.norm_out(x)
                logits = F.linear(mtp_out[0, 0], lm_head_weight)
                pred = logits.argmax().item()
                if pred == next_token_ids[k].item():
                    chain_correct += 1
                chain_total += 1
                current_hidden = mtp_out[0, 0]
    if chain_total > 0:
        log(f"[train] Chain accept rate ({num_chain} steps, train data): {chain_correct}/{chain_total} = {chain_correct/chain_total:.1%}")
        log(f"[train] NOTE: this is on TRAINING data. Real accept must be tested on NEW prompts.")

    # 导出 slim 版
    slim_sd = {k: v for k, v in mtp.state_dict().items() if "lm_head" not in k and "embed" not in k}
    slim_path = Path(output_dir) / f"mtp_head_{model_name}_decode_slim.pt"
    torch.save({
        "model_state_dict": slim_sd,
        "num_chain": num_chain,
        "training": f"{model_name}_decode_chained_multi",
        "hidden_type": "decode",
        "model": cfg.display_name,
        "model_name": model_name,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
    }, slim_path)
    log(f"[train] Saved slim (no lm_head): {slim_path} ({slim_path.stat().st_size / 1e6:.0f}MB)")

    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-model MTP Head Decode Training")
    parser.add_argument("--model-name", required=True,
                        help="Model name from registry: gemma4, dsv4, qwen3vl (or aliases)")
    parser.add_argument("--base-model", default="",
                        help="Base model path (empty=use registry default)")
    parser.add_argument("--mtp-checkpoint", default="",
                        help="Existing MTP checkpoint (empty=start from scratch)")
    parser.add_argument("--corpus", default="",
                        help="Corpus JSONL path (empty=use registry default)")
    parser.add_argument("--output", default="",
                        help="Output directory (empty=use registry default)")
    parser.add_argument("--num-chain", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--gpu-base", type=int, default=0,
                        help="Base GPU index (0 for full, 4 if sglang on 0-3)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gen-length", type=int, default=50,
                        help="Tokens to generate per corpus entry")
    parser.add_argument("--shard-dir", default="",
                        help="Shard directory (empty=use registry default)")
    parser.add_argument("--mode", default="all", choices=["all", "collect", "train"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential mode: single process, device_map='auto' for large models (DSV4)")
    parser.add_argument("--num-shards", type=int, default=4,
                        help="Number of shards to save in sequential mode (0=auto)")
    args = parser.parse_args()

    # 从 registry 获取配置, 填充默认值
    cfg = get_model_config(args.model_name)
    base_model = args.base_model or cfg.base_model_path
    corpus_path = args.corpus or cfg.get_corpus_path()
    output_dir = args.output or os.path.join("/data", f"mtp_head_{cfg.name}_decode")
    shard_dir = args.shard_dir or cfg.get_shard_dir()

    print(f"[main] Model: {cfg.display_name} ({cfg.name})", flush=True)
    print(f"[main] hidden={cfg.hidden_size}, vocab={cfg.vocab_size}, EOS={sorted(cfg.eos_tokens)}", flush=True)
    print(f"[main] base_model={base_model}", flush=True)
    print(f"[main] corpus={corpus_path}", flush=True)
    print(f"[main] output={output_dir}", flush=True)
    print(f"[main] shard_dir={shard_dir}", flush=True)
    print(f"[main] sequential={args.sequential}", flush=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(shard_dir).mkdir(parents=True, exist_ok=True)

    if args.sequential:
        # === Sequential 模式 (大模型, device_map="auto") ===
        if args.mode in ("all", "collect"):
            for old in Path(shard_dir).glob("shard_*.pt"):
                old.unlink()

            print(f"[main] Phase 1 (sequential): {cfg.display_name} decode-mode collect "
                  f"with device_map='auto', ~{args.max_samples} samples, "
                  f"gen_length={args.gen_length}", flush=True)
            t0 = time.time()
            collect_sequential(
                model_name=cfg.name,
                corpus_path=corpus_path,
                base_model_path=base_model,
                mtp_checkpoint=args.mtp_checkpoint,
                shard_dir=shard_dir,
                num_chain=args.num_chain,
                max_samples=args.max_samples,
                gen_length=args.gen_length,
                num_shards=args.num_shards,
            )
            print(f"[main] Phase 1 (sequential) done in {time.time()-t0:.0f}s", flush=True)

        if args.mode in ("all", "train"):
            train_device = f"cuda:{args.gpu_base}" if args.gpu_base > 0 else "cuda:0"
            print(f"[main] Phase 2: train on shards (device={train_device}, sequential={args.sequential})", flush=True)
            train_on_shards(
                model_name=cfg.name,
                base_model_path=base_model,
                mtp_checkpoint=args.mtp_checkpoint,
                shard_dir=shard_dir,
                output_dir=output_dir,
                num_chain=args.num_chain,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=train_device,
                sequential=True,
            )

    else:
        # === 并行模式 (小/中模型, mp.spawn) ===
        max_per_worker = args.max_samples // args.world_size + 1

        if args.mode in ("all", "collect"):
            for old in Path(shard_dir).glob("shard_*.pt"):
                old.unlink()

            print(f"[main] Phase 1: {cfg.display_name} decode-mode collect with {args.world_size} GPUs "
                  f"(GPU {args.gpu_base}-{args.gpu_base + args.world_size - 1}), "
                  f"~{max_per_worker} samples/worker, gen_length={args.gen_length}", flush=True)
            t0 = time.time()
            mp.spawn(
                collect_worker,
                args=(args.world_size, cfg.name, corpus_path, base_model, args.mtp_checkpoint,
                      shard_dir, args.num_chain, max_per_worker, args.gen_length,
                      args.gpu_base),
                nprocs=args.world_size,
                join=True,
            )
            print(f"[main] Phase 1 done in {time.time()-t0:.0f}s", flush=True)

        if args.mode in ("all", "train"):
            train_device = f"cuda:{args.gpu_base}"
            print(f"[main] Phase 2: train on shards (device={train_device})", flush=True)
            train_on_shards(
                model_name=cfg.name,
                base_model_path=base_model,
                mtp_checkpoint=args.mtp_checkpoint,
                shard_dir=shard_dir,
                output_dir=output_dir,
                num_chain=args.num_chain,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=train_device,
                sequential=False,
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
