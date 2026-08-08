#!/usr/bin/env python3
"""Universal MTP training data collector — extracts REAL decode hidden states.

Replaces the broken collect_logits.py which used sglang API (couldn't get
full logits or hidden states, line 192 was `pass`).

This script uses transformers to load ANY registered model directly and
collects decode-mode hidden states via autoregressive generation:

  1. Prefill: model(prompt, use_cache=True) -> KV cache + prefill hidden
  2. Generate first token (argmax of lm_head(prefill_hidden))
  3. Decode loop: for each step, collect (decode_hidden, current_token)
  4. Create chain samples: (hidden[i:i+chain], token[i:i+chain]) -> next_token[i+1:i+chain+1]

Also extracts embed_tokens + lm_head weights (saved as embed_head.pt).

Model-agnostic: works for all models in model_registry (Gemma4, DSV4, Qwen3-VL).
Uses model_loader for universal embed/lm_head extraction.

Usage (on Host1/Host2 with GPUs):
  # Single GPU
  python3 collect_hidden_states.py \
    --model gemma4 \
    --output-dir /data/mtp_train_data/gemma4 \
    --num-samples 50000

  # Multi-GPU (GPUs 4-7, sglang on 0-3)
  python3 collect_hidden_states.py \
    --model gemma4 \
    --output-dir /data/mtp_train_data/gemma4 \
    --num-samples 50000 \
    --world-size 4 --gpu-base 4

  # Custom model path
  python3 collect_hidden_states.py \
    --model dsv4 \
    --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2 \
    --output-dir /data/mtp_train_data/dsv4

Output:
  /data/mtp_train_data/gemma4/shard_000000.pt   # list of chain samples
  /data/mtp_train_data/gemma4/shard_000001.pt
  ...
  /data/mtp_train_data/gemma4/embed_head.pt     # embed + lm_head weights
  /data/mtp_train_data/gemma4/meta.json         # collection metadata
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# === Path setup ===
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "app", "shared")]:
    if p not in sys.path:
        sys.path.insert(0, p)

GGUF_PY_ROOT = os.path.join(
    REPO_ROOT,
    "ComputeGraphCompiler-main",
    "Backend",
    "Llama.cpp",
    "llama.cpp",
    "gguf-py",
)
if os.path.isdir(GGUF_PY_ROOT) and GGUF_PY_ROOT not in sys.path:
    sys.path.insert(0, GGUF_PY_ROOT)


# === Diverse prompt corpus ===
CODE_PROMPTS = [
    "def binary_search(arr, target):\n    ",
    "class LinkedList:\n    def __init__(self):\n        ",
    "import numpy as np\n\ndef matrix_multiply(a, b):\n    ",
    "async def fetch_data(url):\n    ",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ",
    "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\nasync def root():\n    ",
    "def merge_sort(arr):\n    ",
    "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        ",
    "def dfs(graph, node, visited):\n    ",
    "def bfs(graph, start):\n    ",
    "import pandas as pd\n\ndef clean_data(df):\n    ",
    "def train_model(X, y):\n    from sklearn.ensemble import RandomForestClassifier\n    ",
    "class Singleton:\n    _instance = None\n    def __new__(cls):\n        ",
    "def retry(func, max_retries=3):\n    ",
    "def memoize(func):\n    cache = {}\n    def wrapper(*args):\n        ",
    "class Observer:\n    def update(self, subject):\n        ",
    "def parse_json(text):\n    import json\n    ",
    "def validate_email(email):\n    import re\n    ",
    "def chunk_list(lst, size):\n    ",
    "def flatten_dict(d, parent_key='', sep='.'):\n    ",
    "export function debounce(fn, delay) {\n  ",
    "const useState = (initial) => {\n  ",
    "async function fetchData(apiUrl) {\n  ",
    "class Vector3 {\n  constructor(x, y, z) {\n    ",
    "function binarySearch(arr, target) {\n  ",
]

QA_PROMPTS = [
    "What is the time complexity of quicksort?",
    "Explain the difference between TCP and UDP.",
    "How does garbage collection work in Python?",
    "What is a closure in JavaScript?",
    "Explain the CAP theorem.",
    "What is the difference between SQL and NoSQL?",
    "How does HTTPS encryption work?",
    "What is a microservice architecture?",
    "Explain eventual consistency.",
    "What is the actor model in concurrency?",
    "How does a B-tree index work?",
    "What is the difference between process and thread?",
    "Explain the difference between async/await and threads.",
    "What is a reverse proxy and when to use it?",
    "How does JWT authentication work?",
    "What is the difference between GraphQL and REST?",
    "Explain the MapReduce programming model.",
    "What is a Bloom filter?",
    "How does consistent hashing work?",
    "What is the difference between mutex and semaphore?",
]

COMPLETION_PROMPTS = [
    "The main advantage of using a linked list over an array is",
    "In a microservices architecture, service discovery is important because",
    "The key difference between supervised and unsupervised learning is",
    "When designing a distributed system, the CAP theorem states that",
    "The purpose of a load balancer in a web application is to",
    "In object-oriented programming, encapsulation means",
    "The time complexity of inserting an element into a hash table is",
    "A decorator in Python is a function that",
    "The main benefit of using Docker containers is",
    "In a REST API, the POST method is used to",
    "The difference between WHERE and HAVING in SQL is",
    "In Python, a generator differs from a list because",
    "The primary use case for Redis in a web application is",
    "In Kubernetes, a pod is the smallest deployable unit that",
    "The advantage of using WebSockets over HTTP polling is",
]

CREATIVE_PROMPTS = [
    "Write a short story about a robot learning to paint.",
    "Describe a futuristic city in the year 2150.",
    "Write a poem about the ocean.",
    "Create a dialogue between two AI systems meeting for the first time.",
    "Write a product description for a smart water bottle.",
    "Compose a tweet announcing a new open-source library.",
    "Write a blog post title about the future of quantum computing.",
    "Create a character description for a sci-fi novel protagonist.",
    "Write a press release headline for a Mars colony achievement.",
    "Describe the experience of flying through a nebula.",
]


def generate_prompts(num_samples: int, seed: int = 42) -> list[str]:
    """Generate diverse prompt list with code/QA/completion/creative distribution."""
    random.seed(seed)
    prompts = []
    categories = [
        (CODE_PROMPTS, 0.40),
        (QA_PROMPTS, 0.25),
        (COMPLETION_PROMPTS, 0.20),
        (CREATIVE_PROMPTS, 0.15),
    ]
    for pool, ratio in categories:
        count = int(num_samples * ratio)
        for _ in range(count):
            base = random.choice(pool)
            variant = random.random()
            if variant < 0.3:
                prefixes = ["Please ", "Can you ", "", "Help me ", ""]
                base = random.choice(prefixes) + base
            elif variant < 0.5:
                base = f"You are a helpful assistant. {base}"
            prompts.append(base)
    random.shuffle(prompts)
    return prompts[:num_samples]


def _is_gguf_model_path(model_path: str) -> bool:
    return str(model_path or "").strip().lower().endswith(".gguf")


def _gguf_n_gpu_layers(device: str) -> int:
    override = os.environ.get("EDGE_GGUF_COLLECT_N_GPU_LAYERS", "").strip()
    if override:
        try:
            return int(override)
        except Exception:
            pass
    lowered = str(device or "").strip().lower()
    if lowered.startswith("cpu"):
        return 0
    return -1


def _load_embed_head_from_gguf(model_path: str, model_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    from app.shared.model_registry import get_model_config
    import gguf
    import numpy as np

    cfg = get_model_config(model_name)
    reader = gguf.GGUFReader(model_path)
    tensors = {str(tensor.name): tensor for tensor in reader.tensors}

    def _dequantize_tensor(name: str) -> Optional[torch.Tensor]:
        tensor = tensors.get(name)
        if tensor is None:
            return None
        qtype = gguf.GGMLQuantizationType(tensor.tensor_type)
        arr = gguf.dequantize(tensor.data, qtype)
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        out = torch.from_numpy(arr.astype(np.float32, copy=False))
        return out

    embed_weight = _dequantize_tensor("token_embd.weight")
    if embed_weight is None:
        raise RuntimeError("token_embd.weight not found or not dequantizable in GGUF")
    if tuple(embed_weight.shape) != (cfg.vocab_size, cfg.hidden_size):
        raise RuntimeError(
            f"GGUF token_embd.weight shape mismatch: got {tuple(embed_weight.shape)}, "
            f"expected {(cfg.vocab_size, cfg.hidden_size)}"
        )

    lm_head_weight = _dequantize_tensor("output.weight")
    if lm_head_weight is None:
        lm_head_weight = embed_weight
    elif tuple(lm_head_weight.shape) != (cfg.vocab_size, cfg.hidden_size):
        raise RuntimeError(
            f"GGUF output.weight shape mismatch: got {tuple(lm_head_weight.shape)}, "
            f"expected {(cfg.vocab_size, cfg.hidden_size)}"
        )

    return embed_weight, lm_head_weight


def _save_embed_head(
    *,
    path: str,
    embed_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    hidden_size: int,
    vocab_size: int,
    model_name: str,
    extra: Optional[dict] = None,
) -> None:
    embed_cpu = embed_weight.cpu().to(torch.float16)
    lm_head_cpu = lm_head_weight.cpu().to(torch.float16)
    tied = torch.equal(embed_cpu, lm_head_cpu)
    payload = {
        "embed_weight": embed_cpu,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "model_name": model_name,
        "storage_dtype": "float16",
        "lm_head_tied_to_embed": tied,
    }
    if tied:
        payload["lm_head_weight"] = None
    else:
        payload["lm_head_weight"] = lm_head_cpu
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _get_resume_state(output_dir: str, append: bool) -> tuple[int, int, int, bool]:
    """Return collection resume state for append mode.

    Returns:
        next_shard_idx, total_collected, total_failed, has_embed_head
    """
    if not append:
        return 0, 0, 0, False

    existing_shards = sorted(Path(output_dir).glob("shard_*.pt"))
    next_shard_idx = 0
    if existing_shards:
        try:
            next_shard_idx = max(int(p.stem.split("_")[-1]) for p in existing_shards) + 1
        except Exception:
            next_shard_idx = len(existing_shards)

    total_collected = 0
    total_failed = 0
    meta_path = os.path.join(output_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            total_collected = int(meta.get("total_samples", 0) or 0)
            total_failed = int(meta.get("total_failed", 0) or 0)
        except Exception:
            pass

    has_embed_head = os.path.exists(os.path.join(output_dir, "embed_head.pt"))
    return next_shard_idx, total_collected, total_failed, has_embed_head


def collect_one_prompt_gguf(
    loop,
    prompt: str,
    eos_tokens: set[int],
    *,
    gen_length: int = 50,
    num_chain: int = 4,
    max_input_length: int = 256,
) -> list[dict]:
    prompt_tokens = list(
        loop.llm.tokenize(
            prompt.encode("utf-8"),
            add_bos=False,
            special=True,
        )
    )
    prompt_tokens = [tok for tok in prompt_tokens if tok not in eos_tokens]
    if len(prompt_tokens) < 3:
        return []
    if len(prompt_tokens) > max_input_length:
        prompt_tokens = prompt_tokens[:max_input_length]
        prompt = loop.llm.detokenize(prompt_tokens).decode("utf-8", errors="ignore")

    try:
        current_token = int(loop.prefill(prompt))
    except Exception:
        return []

    decode_hiddens: list[torch.Tensor] = []
    decode_tokens: list[int] = []

    for _ in range(gen_length):
        hidden, logits = loop._decode_single(current_token, pos=loop.n_past)
        loop.n_past += 1
        next_token = int(logits.argmax())

        decode_hiddens.append(torch.from_numpy(hidden.astype("float32", copy=False)).cpu())
        decode_tokens.append(int(current_token))

        if next_token in eos_tokens:
            break
        current_token = next_token

    if len(decode_hiddens) < num_chain + 1:
        return []

    samples = []
    for i in range(len(decode_hiddens) - num_chain):
        end = i + num_chain
        if end >= len(decode_tokens):
            break
        samples.append({
            "hidden_states": torch.stack(decode_hiddens[i:end]),
            "token_ids": torch.tensor(decode_tokens[i:end]),
            "next_token_ids": torch.tensor(decode_tokens[i + 1:end + 1]),
        })
    return samples


# === Core: collect decode hidden states for one prompt ===

def collect_one_prompt(
    model,
    tokenizer,
    embed_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    prompt: str,
    eos_tokens: set[int],
    gen_length: int = 50,
    num_chain: int = 4,
    max_input_length: int = 256,
    device: str = "cuda",
    input_device: str = "",
) -> list[dict]:
    """Collect decode hidden states for a single prompt.

    Returns list of chain samples:
      [{"hidden_states": [chain, hidden], "token_ids": [chain], "next_token_ids": [chain]}]

    Key: uses DECODE hidden states (not prefill), because at inference time
    the MTP head receives decode hidden states from the autoregressive loop.
    Training on prefill hidden states causes accept=0% mismatch.

    When device="auto", model is loaded with device_map="auto" across multiple GPUs.
    Input tensors go on input_device (default: model's first param device),
    and hidden states are moved to lm_head_weight.device for logits computation.
    """
    # Determine input device (for device_map="auto" support)
    if not input_device:
        input_device = str(next(model.parameters()).device)
    logits_device = lm_head_weight.device

    # 1. Tokenize
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Filter EOS / special tokens
    input_ids = [t for t in input_ids if t not in eos_tokens]
    if len(input_ids) < 3:
        return []
    if len(input_ids) > max_input_length:
        input_ids = input_ids[:max_input_length]

    input_tensor = torch.tensor([input_ids], device=input_device)

    # 2. Prefill (get KV cache + prefill hidden for first token generation)
    try:
        with torch.no_grad():
            outputs = model(input_tensor, output_hidden_states=True, use_cache=True)
    except Exception:
        # Some models need labels or other args; try minimal forward
        with torch.no_grad():
            outputs = model(input_tensor, output_hidden_states=True, use_cache=True)

    kv_cache = outputs.past_key_values
    prefill_last_hidden = outputs.hidden_states[-1][0, -1].to(logits_device)  # [hidden]

    # 3. Generate first token (greedy = temperature 0)
    with torch.no_grad():
        first_logits = F.linear(prefill_last_hidden, lm_head_weight)
    first_token = int(first_logits.argmax().item())

    # 4. Decode loop — collect decode hidden states
    decode_hiddens: list[torch.Tensor] = []
    decode_tokens: list[int] = []
    current_token = first_token
    decode_out = None

    for step in range(gen_length):
        try:
            with torch.no_grad():
                decode_out = model(
                    torch.tensor([[current_token]], device=input_device),
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                    use_cache=True,
                )
        except Exception:
            break

        decode_hidden = decode_out.hidden_states[-1][0, 0].to(logits_device)  # [hidden]
        kv_cache = decode_out.past_key_values

        # Generate next token
        with torch.no_grad():
            next_logits = F.linear(decode_hidden, lm_head_weight)
        next_token = int(next_logits.argmax().item())

        decode_hiddens.append(decode_hidden.cpu())
        decode_tokens.append(current_token)

        if next_token in eos_tokens:
            break
        current_token = next_token

    # Free KV cache
    del kv_cache, outputs
    if decode_out is not None:
        del decode_out
    torch.cuda.empty_cache() if input_device.startswith("cuda") else None

    if len(decode_hiddens) < num_chain + 1:
        return []

    # 5. Create chain training samples
    samples = []
    for i in range(len(decode_hiddens) - num_chain):
        end = i + num_chain
        if end >= len(decode_tokens):
            break
        chain_hidden = torch.stack(decode_hiddens[i:end])           # [chain, hidden]
        chain_tokens = torch.tensor(decode_tokens[i:end])            # [chain]
        chain_next = torch.tensor(decode_tokens[i + 1:end + 1])     # [chain] shifted

        samples.append({
            "hidden_states": chain_hidden,
            "token_ids": chain_tokens,
            "next_token_ids": chain_next,
        })

    return samples


# === Single-GPU collection ===

def collect_single_gpu(
    model_name: str,
    model_path: str,
    output_dir: str,
    prompts: list[str],
    num_chain: int = 4,
    gen_length: int = 50,
    shard_size: int = 500,
    device: str = "cuda",
    seed: int = 42,
    append: bool = False,
) -> int:
    """Collect hidden states on a single GPU."""
    from app.shared.model_registry import get_model_config
    from app.shared.model_loader import load_base_model, get_embed_weight, get_lm_head_weight

    cfg = get_model_config(model_name)
    eos_tokens = cfg.eos_tokens

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model
    print(f"[collect] Loading {cfg.display_name} from {model_path} on {device}...", flush=True)
    model, tokenizer = load_base_model(model_path, device=device, dtype="bfloat16")
    model.eval()

    embed_weight = get_embed_weight(model)
    lm_head_weight = get_lm_head_weight(model)
    if lm_head_weight is None:
        lm_head_weight = embed_weight  # tied embeddings
        print("[collect] Using tied embeddings (embed = lm_head)", flush=True)

    print(f"[collect] embed: {embed_weight.shape}, lm_head: {lm_head_weight.shape}", flush=True)
    print(f"[collect] hidden_size={cfg.hidden_size}, vocab={cfg.vocab_size}", flush=True)

    # 2. Save embed + lm_head weights (for training)
    shard_idx, total_collected, total_failed, has_embed_head = _get_resume_state(output_dir, append)
    embed_head_path = os.path.join(output_dir, "embed_head.pt")
    if append and has_embed_head:
        print(f"[collect] Reusing existing embed_head.pt for append mode", flush=True)
    else:
        _save_embed_head(
            path=embed_head_path,
            embed_weight=embed_weight,
            lm_head_weight=lm_head_weight,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
            model_name=model_name,
        )
        print(f"[collect] Saved embed_head.pt ({os.path.getsize(embed_head_path) / 1e9:.1f}GB)", flush=True)

    # 3. Collect
    shard_data: list[dict] = []
    t0 = time.time()

    # Determine input device (for device_map="auto" support)
    input_device = str(next(model.parameters()).device) if device == "auto" else device
    print(f"[collect] input_device={input_device}, logits_device={lm_head_weight.device}", flush=True)

    for i, prompt in enumerate(prompts):
        try:
            samples = collect_one_prompt(
                model, tokenizer, embed_weight, lm_head_weight,
                prompt, eos_tokens,
                gen_length=gen_length, num_chain=num_chain,
                device=device,
                input_device=input_device,
            )
            if samples:
                shard_data.extend(samples)
                total_collected += len(samples)
            else:
                total_failed += 1
        except Exception as e:
            total_failed += 1
            if total_failed <= 5:
                print(f"[collect] Error on prompt {i}: {e}", flush=True)

        # Save shard
        if len(shard_data) >= shard_size:
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
            torch.save(shard_data, shard_path)
            elapsed = time.time() - t0
            rate = total_collected / elapsed if elapsed > 0 else 0
            print(
                f"[collect] Shard {shard_idx}: {len(shard_data)} samples "
                f"(total: {total_collected}, failed: {total_failed}, "
                f"rate: {rate:.1f}/s, elapsed: {elapsed:.0f}s)",
                flush=True,
            )
            shard_data = []
            shard_idx += 1

        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"[collect] Progress: {i + 1}/{len(prompts)} "
                f"(samples: {total_collected}, failed: {total_failed}, "
                f"rate: {rate:.1f} prompts/s)",
                flush=True,
            )

    # Save final shard
    if shard_data:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)
        shard_idx += 1

    # 4. Save metadata
    meta = {
        "model_name": model_name,
        "model_path": model_path,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "num_chain": num_chain,
        "gen_length": gen_length,
        "total_samples": total_collected,
        "total_failed": total_failed,
        "num_shards": shard_idx,
        "collection_time_sec": time.time() - t0,
        "prompt_distribution": {"code": 0.40, "qa": 0.25, "completion": 0.20, "creative": 0.15},
        "hidden_type": "decode",
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[collect] Done! {total_collected} samples in {shard_idx} shards", flush=True)
    print(f"[collect] Output: {output_dir}/", flush=True)
    print(f"[collect] Time: {time.time() - t0:.0f}s", flush=True)

    # Free model
    del model
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    return total_collected


def collect_single_gguf(
    model_name: str,
    model_path: str,
    output_dir: str,
    prompts: list[str],
    num_chain: int = 4,
    gen_length: int = 50,
    shard_size: int = 500,
    max_input_length: int = 256,
    device: str = "cuda:0",
    seed: int = 42,
    append: bool = False,
) -> int:
    from app.shared.model_registry import get_model_config
    from CGC_Phase2.mtp_verify_loop import MTPVerifyLoop

    del seed
    cfg = get_model_config(model_name)
    eos_tokens = cfg.eos_tokens
    os.makedirs(output_dir, exist_ok=True)

    print(f"[collect] Loading GGUF {cfg.display_name} from {model_path}...", flush=True)
    print(f"[collect] GGUF runtime device={device}, n_gpu_layers={_gguf_n_gpu_layers(device)}", flush=True)

    shard_idx, total_collected, total_failed, has_embed_head = _get_resume_state(output_dir, append)
    embed_head_path = os.path.join(output_dir, "embed_head.pt")
    if append and has_embed_head:
        print(f"[collect] Reusing existing embed_head.pt for append mode", flush=True)
    else:
        embed_weight, lm_head_weight = _load_embed_head_from_gguf(model_path, model_name)
        _save_embed_head(
            path=embed_head_path,
            embed_weight=embed_weight,
            lm_head_weight=lm_head_weight,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
            model_name=model_name,
            extra={
                "source_format": "gguf",
                "source_model_path": model_path,
            },
        )
        print(f"[collect] Saved embed_head.pt ({os.path.getsize(embed_head_path) / 1e9:.1f}GB)", flush=True)

    loop = MTPVerifyLoop(
        model_path=model_path,
        mtp_checkpoint=None,
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        n_gpu_layers=_gguf_n_gpu_layers(device),
        n_ctx=int(os.environ.get("EDGE_GGUF_COLLECT_N_CTX", "4096") or "4096"),
        n_batch=int(os.environ.get("EDGE_GGUF_COLLECT_N_BATCH", "256") or "256"),
        n_ubatch=int(os.environ.get("EDGE_GGUF_COLLECT_N_UBATCH", "128") or "128"),
        n_threads=int(os.environ.get("EDGE_GGUF_COLLECT_N_THREADS", "8") or "8"),
        n_threads_batch=int(os.environ.get("EDGE_GGUF_COLLECT_N_THREADS_BATCH", "8") or "8"),
        flash_attn=bool(int(os.environ.get("EDGE_GGUF_COLLECT_FLASH_ATTN", "1") or "1")),
        offload_kqv=True,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
        use_ngram_fallback=False,
    )

    shard_data: list[dict] = []
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        try:
            samples = collect_one_prompt_gguf(
                loop,
                prompt,
                eos_tokens,
                gen_length=gen_length,
                num_chain=num_chain,
                max_input_length=max_input_length,
            )
            if samples:
                shard_data.extend(samples)
                total_collected += len(samples)
            else:
                total_failed += 1
        except Exception as e:
            total_failed += 1
            if total_failed <= 5:
                print(f"[collect] GGUF error on prompt {i}: {e}", flush=True)

        if len(shard_data) >= shard_size:
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
            torch.save(shard_data, shard_path)
            elapsed = time.time() - t0
            rate = total_collected / elapsed if elapsed > 0 else 0
            print(
                f"[collect] Shard {shard_idx}: {len(shard_data)} samples "
                f"(total: {total_collected}, failed: {total_failed}, "
                f"rate: {rate:.1f}/s, elapsed: {elapsed:.0f}s)",
                flush=True,
            )
            shard_data = []
            shard_idx += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"[collect] GGUF progress: {i + 1}/{len(prompts)} "
                f"(samples: {total_collected}, failed: {total_failed}, "
                f"rate: {rate:.1f} prompts/s)",
                flush=True,
            )

    if shard_data:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.pt")
        torch.save(shard_data, shard_path)
        shard_idx += 1

    meta = {
        "model_name": model_name,
        "model_path": model_path,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "num_chain": num_chain,
        "gen_length": gen_length,
        "total_samples": total_collected,
        "total_failed": total_failed,
        "num_shards": shard_idx,
        "collection_time_sec": time.time() - t0,
        "prompt_distribution": {"code": 0.40, "qa": 0.25, "completion": 0.20, "creative": 0.15},
        "hidden_type": "decode",
        "source_format": "gguf",
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[collect] GGUF done! {total_collected} samples in {shard_idx} shards", flush=True)
    print(f"[collect] Output: {output_dir}/", flush=True)
    print(f"[collect] Time: {time.time() - t0:.0f}s", flush=True)
    return total_collected


# === Multi-GPU collection (torch.multiprocessing) ===

def _collect_worker(
    rank: int,
    world_size: int,
    model_name: str,
    model_path: str,
    output_dir: str,
    prompts: list[str],
    num_chain: int,
    gen_length: int,
    shard_size: int,
    gpu_base: int,
    seed: int,
):
    """Worker process for multi-GPU data collection."""
    import torch.multiprocessing as mp

    actual_gpu = gpu_base + rank
    torch.cuda.set_device(actual_gpu)
    device = f"cuda:{actual_gpu}"

    # Split prompts across workers
    my_prompts = prompts[rank::world_size]
    my_output_dir = os.path.join(output_dir, f"worker_{rank}")
    os.makedirs(my_output_dir, exist_ok=True)

    print(f"[worker {rank}] GPU {device}, {len(my_prompts)} prompts", flush=True)

    count = collect_single_gpu(
        model_name=model_name,
        model_path=model_path,
        output_dir=my_output_dir,
        prompts=my_prompts,
        num_chain=num_chain,
        gen_length=gen_length,
        shard_size=shard_size,
        device=device,
        seed=seed + rank,
    )

    print(f"[worker {rank}] Done: {count} samples", flush=True)
    return count


def collect_multi_gpu(
    model_name: str,
    model_path: str,
    output_dir: str,
    prompts: list[str],
    world_size: int,
    gpu_base: int,
    num_chain: int = 4,
    gen_length: int = 50,
    shard_size: int = 500,
    seed: int = 42,
) -> int:
    """Multi-GPU parallel data collection."""
    import torch.multiprocessing as mp

    os.makedirs(output_dir, exist_ok=True)
    ctx = mp.get_context("spawn")

    workers = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_collect_worker,
            args=(rank, world_size, model_name, model_path, output_dir,
                  prompts, num_chain, gen_length, shard_size, gpu_base, seed),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # Merge worker outputs + copy embed_head.pt
    total = 0
    for rank in range(world_size):
        worker_dir = os.path.join(output_dir, f"worker_{rank}")
        if not os.path.isdir(worker_dir):
            continue
        # Copy embed_head.pt from worker 0
        if rank == 0:
            embed_head_src = os.path.join(worker_dir, "embed_head.pt")
            if os.path.exists(embed_head_src):
                import shutil
                shutil.copy2(embed_head_src, os.path.join(output_dir, "embed_head.pt"))

        # Count shards
        for f in sorted(Path(worker_dir).glob("shard_*.pt")):
            total += len(torch.load(f, weights_only=False))

    print(f"[collect] Multi-GPU done: {total} total samples across {world_size} workers", flush=True)
    return total


# === CLI ===

def main():
    parser = argparse.ArgumentParser(
        description="Universal MTP data collector — extracts REAL decode hidden states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU
  python3 collect_hidden_states.py --model gemma4 --num-samples 50000

  # Multi-GPU (GPUs 4-7)
  python3 collect_hidden_states.py --model gemma4 --num-samples 50000 --world-size 4 --gpu-base 4

  # Custom path
  python3 collect_hidden_states.py --model dsv4 --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2
        """,
    )
    parser.add_argument("--model", required=True, help="Model name: gemma4 | dsv4 | qwen3vl (or alias)")
    parser.add_argument("--model-path", default="", help="Model path (default: from model_registry)")
    parser.add_argument("--output-dir", default="", help="Output dir (default: /data/mtp_train_data/<model>)")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of prompts to process")
    parser.add_argument("--num-chain", type=int, default=4, help="Chain length for training samples")
    parser.add_argument("--gen-length", type=int, default=50, help="Max tokens to generate per prompt")
    parser.add_argument("--shard-size", type=int, default=500, help="Samples per shard file")
    parser.add_argument("--max-input-length", type=int, default=256, help="Max input prompt length (tokens)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--append", action="store_true", help="Append new shards into an existing output dir and reuse embed_head.pt if present")

    # GPU options
    parser.add_argument("--device", default="cuda:0", help="Single-device hint. GGUF collector treats non-cpu as llama.cpp GPU offload.")
    parser.add_argument("--world-size", type=int, default=0, help="Multi-GPU worker count (0 = single GPU)")
    parser.add_argument("--gpu-base", type=int, default=0, help="Base GPU index for multi-GPU")

    args = parser.parse_args()

    # Resolve model config
    from app.shared.model_registry import get_model_config
    cfg = get_model_config(args.model)
    model_path = args.model_path or cfg.base_model_path
    output_dir = args.output_dir or cfg.get_shard_dir()

    print(f"=== Universal MTP Data Collector ===", flush=True)
    print(f"  Model: {cfg.display_name} ({cfg.name})", flush=True)
    print(f"  Path: {model_path}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  hidden_size={cfg.hidden_size}, vocab={cfg.vocab_size}", flush=True)
    print(f"  Samples: {args.num_samples}, chain={args.num_chain}, gen={args.gen_length}", flush=True)

    # Generate prompts
    prompts = generate_prompts(args.num_samples, seed=args.seed)
    print(f"  Generated {len(prompts)} diverse prompts", flush=True)

    use_gguf = _is_gguf_model_path(model_path)
    print(f"  Source format: {'gguf' if use_gguf else 'transformers'}", flush=True)

    # Collect
    if use_gguf and args.world_size > 1:
        print("ERROR: GGUF collector does not support multi-worker mode yet.", flush=True)
        sys.exit(2)
    if args.world_size > 1:
        total = collect_multi_gpu(
            model_name=cfg.name,
            model_path=model_path,
            output_dir=output_dir,
            prompts=prompts,
            world_size=args.world_size,
            gpu_base=args.gpu_base,
            num_chain=args.num_chain,
            gen_length=args.gen_length,
            shard_size=args.shard_size,
            seed=args.seed,
        )
    elif use_gguf:
        total = collect_single_gguf(
            model_name=cfg.name,
            model_path=model_path,
            output_dir=output_dir,
            prompts=prompts,
            num_chain=args.num_chain,
            gen_length=args.gen_length,
            shard_size=args.shard_size,
            max_input_length=args.max_input_length,
            device=args.device,
            seed=args.seed,
            append=args.append,
        )
    else:
        total = collect_single_gpu(
            model_name=cfg.name,
            model_path=model_path,
            output_dir=output_dir,
            prompts=prompts,
            num_chain=args.num_chain,
            gen_length=args.gen_length,
            shard_size=args.shard_size,
            device=args.device,
            seed=args.seed,
            append=args.append,
        )

    if total == 0:
        print("WARNING: No samples collected. Check model loading and GPU availability.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
