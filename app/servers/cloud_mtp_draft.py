#!/usr/bin/env python3
"""Cloud MTP Draft Server — 路径 B 核心组件 (v2: decode hidden states).

在云端运行训练好的 .pt MTP head, 利用 sglang 的 return_hidden_states API 激活.

关键修复 (v2):
  - 使用 DECODE hidden states (hs[1]) 而非 PREFILL hidden states (hs[0][-1])
  - 训练数据收集的是 decode hidden states, 用 prefill 会导致 accept=0%
  - sglang max_new_tokens=1 返回:
    hs[0] = prefill hidden states (list of lists, 每个input position一个)
    hs[1] = decode hidden state (flat list of 2048 floats, 生成token的)
  - rejection 时做 fresh forward pass 获取干净的 decode hidden state

工作流程:
  1. 接收 prompt, 发送到 sglang (max_new_tokens=1, return_hidden_states=True)
  2. 从 hs[1] 获取 decode hidden state
  3. MTP head 链式生成 4 个 draft tokens
  4. 发送 current_seq + draft_tokens 到 sglang (return_logprob=True)
  5. 从 logprobs 验证 draft tokens
  6. 全部接受: 从 hs[1] 获取 decode hidden state (bonus token的)
  7. 有拒绝: 做 fresh forward pass 获取 corrected token 的 decode hidden state
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any, Optional

import httpx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger("cloud_mtp")

app = FastAPI(title="CGC Cloud MTP Draft Server v2", version="2.0")

SGLANG_URL = "http://127.0.0.1:30003"
HTTP_TIMEOUT = 60.0

_mtp_head: Optional[nn.Module] = None
_embed_weight: Optional[torch.Tensor] = None
_lm_head_weight: Optional[torch.Tensor] = None
_device: torch.device = torch.device("cpu")

_http_client: Optional[httpx.AsyncClient] = None

_stats = {
    "total_requests": 0,
    "total_tokens_generated": 0,
    "total_draft_tokens": 0,
    "total_accepted": 0,
    "total_rejected": 0,
    "total_rounds": 0,
    "total_mtp_ms": 0.0,
    "total_sglang_ms": 0.0,
    "total_request_ms": 0.0,
}


# === MTP Head Model ===

class MTPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, L, H = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        return self.o_proj(out)


class MTPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class MTPHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int, head_dim: int, intermediate_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm1 = RMSNorm(hidden_size)
        self.attn = MTPAttention(hidden_size, num_heads, head_dim)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = MTPMLP(hidden_size, intermediate_size)
        self.norm_out = RMSNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, token_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, token_embed], dim=-1)
        x = self.proj(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return self.norm_out(x)

    def predict(self, mtp_hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(mtp_hidden[:, 0, :], lm_head_weight)
        pred_token = logits.argmax(dim=-1)
        return logits, pred_token


# === Hidden state extraction ===

def extract_decode_hidden(hs_data: list) -> Optional[torch.Tensor]:
    """Extract DECODE hidden state from sglang hidden_states response.

    sglang return_hidden_states with max_new_tokens=1:
      hs[0] = prefill hidden states: [[2048 floats], ...] (one per input position)
      hs[1] = decode hidden state: [2048 floats] (flat list, for the generated token)

    The MTP head was trained on DECODE hidden states, so we must use hs[1].
    """
    if not hs_data or len(hs_data) < 2:
        # Fallback: if only 1 element, try hs[0][-1] (last prefill position)
        if hs_data and len(hs_data) == 1:
            hs0 = hs_data[0]
            if isinstance(hs0, list) and len(hs0) > 0:
                last = hs0[-1]
                if isinstance(last, list) and len(last) > 0 and isinstance(last[0], (int, float)):
                    logger.debug("Using prefill hidden state as fallback (no decode hs)")
                    return torch.tensor(last, dtype=torch.float32)
        return None

    # hs[1] should be the decode hidden state
    decode_hs = hs_data[1]
    if isinstance(decode_hs, list) and len(decode_hs) > 0:
        if isinstance(decode_hs[0], (int, float)):
            # Flat list of floats → decode hidden state
            return torch.tensor(decode_hs, dtype=torch.float32)
        elif isinstance(decode_hs[0], list) and len(decode_hs[0]) > 0:
            # List of lists → take last entry
            last = decode_hs[-1]
            if isinstance(last, list) and len(last) > 0 and isinstance(last[0], (int, float)):
                return torch.tensor(last, dtype=torch.float32)

    return None


# === Initialization ===

def load_mtp_head(checkpoint_path: str, device: torch.device) -> tuple[MTPHead, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    if isinstance(config, str):
        config = eval(config)

    hidden_size = config.get("hidden_size", 2048)
    vocab_size = config.get("vocab_size", 151936)
    num_heads = config.get("num_heads", 16)
    head_dim = config.get("head_dim", 128)
    intermediate_size = config.get("intermediate_size", 5632)

    mtp = MTPHead(hidden_size, vocab_size, num_heads, head_dim, intermediate_size)
    mtp.load_state_dict(ckpt["model_state_dict"])
    mtp = mtp.to(device).eval()

    logger.info(f"MTP head loaded: {checkpoint_path}")
    logger.info(f"  config: hidden={hidden_size}, vocab={vocab_size}, heads={num_heads}, "
                f"head_dim={head_dim}, inter={intermediate_size}")
    logger.info(f"  params: {sum(p.numel() for p in mtp.parameters()) / 1e6:.1f}M")
    logger.info(f"  device: {device}")

    return mtp, config


def load_embed_weights(model_path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load embed_tokens and lm_head weights from model safetensors."""
    from safetensors.torch import load_file
    import glob
    import os

    st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not st_files:
        st_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    logger.info(f"Loading embeddings from {len(st_files)} safetensors files...")

    embed_weight = None
    lm_head_weight = None

    for st_file in st_files:
        sd = load_file(st_file)
        for key, val in sd.items():
            # Search for embed_tokens (handles VL model paths like model.language_model.embed_tokens)
            if "embed_tokens" in key and embed_weight is None and val.dim() == 2:
                embed_weight = val
                logger.info(f"  Found embed_tokens: {key} {val.shape}")
            # Search for lm_head
            if "lm_head" in key and lm_head_weight is None and val.dim() == 2:
                lm_head_weight = val
                logger.info(f"  Found lm_head: {key} {val.shape}")
        del sd

    if embed_weight is None:
        raise RuntimeError("Could not find embed_tokens.weight in model files")

    if lm_head_weight is None:
        logger.info("  lm_head not found, using tied weights (embed_tokens)")
        lm_head_weight = embed_weight

    embed_weight = embed_weight.to(device).float()
    lm_head_weight = lm_head_weight.to(device).float()

    return embed_weight, lm_head_weight


# === Spec decode ===

async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client


async def sglang_generate(payload: dict) -> dict:
    client = await get_http_client()
    resp = await client.post(f"{SGLANG_URL}/generate", json=payload)
    resp.raise_for_status()
    return resp.json()


def mtp_draft_chain(
    current_hidden: torch.Tensor,
    current_token: int,
    num_draft: int = 4,
) -> list[int]:
    """Generate draft tokens using MTP head chain."""
    draft_tokens = []
    h = current_hidden.unsqueeze(0).unsqueeze(0).to(_device)  # [1, 1, hidden]

    with torch.no_grad():
        for k in range(num_draft):
            tok = torch.tensor([current_token], device=_device)
            token_embed = F.embedding(tok, _embed_weight)
            e = token_embed.unsqueeze(0)

            mtp_hidden = _mtp_head(h, e)
            logits, pred = _mtp_head.predict(mtp_hidden, _lm_head_weight)

            pred_token = pred.item()
            draft_tokens.append(pred_token)

            h = mtp_hidden.detach()
            current_token = pred_token

    return draft_tokens


@app.post("/generate")
async def generate(request: Request):
    """Spec decode generate: MTP draft + sglang verify (v2: decode hidden states)."""
    t0 = time.time()
    data = await request.json()

    prompt_ids = data.get("input_ids", [])
    max_tokens = data.get("max_tokens", 50)
    temperature = data.get("temperature", 0.0)
    num_draft = data.get("num_draft", 4)

    if not prompt_ids:
        return JSONResponse({"error": "input_ids is empty"}, status_code=400)

    total_mtp_ms = 0.0
    total_sglang_ms = 0.0
    total_accepted = 0
    total_rejected = 0
    total_rounds = 0

    # === Step 1: Initial forward pass — get first token + DECODE hidden state ===
    # Use max_new_tokens=2 to get hs[1] (decode hidden state):
    #   hs[0] = prefill hidden states (list of lists, per input position)
    #   hs[1] = decode hidden state (flat list of 2048 floats, for first generated token)
    sglang_t0 = time.time()
    result = await sglang_generate({
        "input_ids": prompt_ids,
        "sampling_params": {"max_new_tokens": 2, "temperature": temperature},
        "return_hidden_states": True,
    })
    sglang_ms = (time.time() - sglang_t0) * 1000
    total_sglang_ms += sglang_ms

    output_ids = result.get("output_ids", [])
    if not output_ids:
        return JSONResponse({"error": "No output from sglang"}, status_code=500)

    first_token = output_ids[0]
    all_tokens = [first_token]

    # Extract DECODE hidden state (hs[1], NOT hs[0][-1] which is prefill)
    meta = result.get("meta_info", {})
    hs_data = meta.get("hidden_states", [])
    current_hidden = extract_decode_hidden(hs_data)

    if current_hidden is None:
        logger.warning("No decode hidden state, falling back to regular generation")
        result = await sglang_generate({
            "input_ids": prompt_ids,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": temperature},
        })
        total_ms = (time.time() - t0) * 1000
        out_ids = result.get("output_ids", [])
        return JSONResponse({
            "output_ids": out_ids,
            "text": result.get("text", ""),
            "stats": {
                "tokens": len(out_ids), "accepted": 0, "rejected": 0, "accept_rate": 0,
                "rounds": 0, "mtp_ms": 0, "sglang_ms": total_sglang_ms,
                "total_ms": round(total_ms, 1),
                "tps": round(len(out_ids) / (total_ms / 1000), 1) if total_ms > 0 else 0,
                "fallback": True,
            }
        })

    logger.debug(f"Initial decode hidden: norm={current_hidden.norm():.4f}, shape={current_hidden.shape}")

    current_token = first_token
    current_seq = list(prompt_ids) + [first_token]

    # === Step 2: Spec decode loop ===
    while len(all_tokens) < max_tokens:
        # a. MTP draft
        mtp_t0 = time.time()
        draft_tokens = mtp_draft_chain(current_hidden, current_token, num_draft=num_draft)
        mtp_ms = (time.time() - mtp_t0) * 1000
        total_mtp_ms += mtp_ms

        # b. Verify: send current_seq + draft_tokens to sglang
        # Use max_new_tokens=2 to get decode hidden state (hs[1])
        input_ids = current_seq + draft_tokens
        sglang_t0 = time.time()
        result = await sglang_generate({
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 2, "temperature": temperature},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 1,
            "return_hidden_states": True,
        })
        sglang_ms = (time.time() - sglang_t0) * 1000
        total_sglang_ms += sglang_ms

        meta = result.get("meta_info", {})
        input_top_logprobs = meta.get("input_top_logprobs", [])
        output_ids_verify = result.get("output_ids", [])
        hs_data = meta.get("hidden_states", [])

        # c. Verify each draft token using logprobs
        accepted_tokens = []
        rejected_at = -1
        corrected_token = -1

        for k, draft_tok in enumerate(draft_tokens):
            idx = len(current_seq) + k

            if idx >= len(input_top_logprobs):
                break

            top_entry = input_top_logprobs[idx]
            if top_entry is None or len(top_entry) == 0:
                rejected_at = k
                corrected_token = output_ids_verify[0] if output_ids_verify else -1
                break

            top1_token_id = top_entry[0][1]
            if top1_token_id == draft_tok:
                accepted_tokens.append(draft_tok)
            else:
                rejected_at = k
                corrected_token = top1_token_id
                break

        # d. Process results
        total_accepted += len(accepted_tokens)
        total_rejected += len(draft_tokens) - len(accepted_tokens)

        for tok in accepted_tokens:
            all_tokens.append(tok)
            current_seq.append(tok)

        if rejected_at >= 0 and corrected_token >= 0:
            all_tokens.append(corrected_token)
            current_seq.append(corrected_token)

        # e. Update hidden state for next round
        if rejected_at == -1:
            # All accepted — use decode hidden state from verify call
            # hs[1] is the decode hidden state for the first output token (bonus token)
            current_hidden = extract_decode_hidden(hs_data)

            if current_hidden is not None and output_ids_verify:
                # Add bonus tokens (we got 2 from max_new_tokens=2)
                for bonus_tok in output_ids_verify:
                    if len(all_tokens) >= max_tokens:
                        break
                    all_tokens.append(bonus_tok)
                    current_seq.append(bonus_tok)
                # Use first bonus token for next MTP draft (hs[1] is its decode hidden state)
                current_token = output_ids_verify[0]
            elif current_hidden is not None:
                current_token = accepted_tokens[-1] if accepted_tokens else current_token
            else:
                current_hidden = None
        else:
            # Rejection — prefill included wrong draft tokens, decode hidden state is contaminated
            # Do a fresh forward pass with corrected sequence to get clean decode hidden state
            if len(all_tokens) < max_tokens:
                sglang_t0 = time.time()
                fresh_result = await sglang_generate({
                    "input_ids": current_seq,
                    "sampling_params": {"max_new_tokens": 2, "temperature": temperature},
                    "return_hidden_states": True,
                })
                sglang_ms = (time.time() - sglang_t0) * 1000
                total_sglang_ms += sglang_ms

                fresh_meta = fresh_result.get("meta_info", {})
                fresh_hs = fresh_meta.get("hidden_states", [])
                current_hidden = extract_decode_hidden(fresh_hs)

                # Add the fresh token as bonus
                fresh_output = fresh_result.get("output_ids", [])
                if fresh_output:
                    all_tokens.append(fresh_output[0])
                    current_seq.append(fresh_output[0])
                    current_token = fresh_output[0]
            else:
                current_hidden = None

        if current_hidden is None:
            # Last resort: do one more forward pass
            if len(all_tokens) < max_tokens:
                sglang_t0 = time.time()
                result = await sglang_generate({
                    "input_ids": current_seq,
                    "sampling_params": {"max_new_tokens": 2, "temperature": temperature},
                    "return_hidden_states": True,
                })
                sglang_ms = (time.time() - sglang_t0) * 1000
                total_sglang_ms += sglang_ms

                meta = result.get("meta_info", {})
                hs_data = meta.get("hidden_states", [])
                current_hidden = extract_decode_hidden(hs_data)
                out = result.get("output_ids", [])
                if out:
                    all_tokens.append(out[0])
                    current_seq.append(out[0])
                    current_token = out[0]
            else:
                break

        total_rounds += 1

        if len(all_tokens) >= max_tokens:
            break

    # Trim
    all_tokens = all_tokens[:max_tokens]

    total_ms = (time.time() - t0) * 1000
    total_draft = total_accepted + total_rejected
    accept_rate = total_accepted / total_draft if total_draft > 0 else 0
    tps = len(all_tokens) / (total_ms / 1000) if total_ms > 0 else 0

    _stats["total_requests"] += 1
    _stats["total_tokens_generated"] += len(all_tokens)
    _stats["total_draft_tokens"] += total_draft
    _stats["total_accepted"] += total_accepted
    _stats["total_rejected"] += total_rejected
    _stats["total_rounds"] += total_rounds
    _stats["total_mtp_ms"] += total_mtp_ms
    _stats["total_sglang_ms"] += total_sglang_ms
    _stats["total_request_ms"] += total_ms

    return JSONResponse({
        "output_ids": all_tokens,
        "stats": {
            "tokens": len(all_tokens),
            "accepted": total_accepted,
            "rejected": total_rejected,
            "accept_rate": round(accept_rate, 4),
            "rounds": total_rounds,
            "mtp_ms": round(total_mtp_ms, 1),
            "sglang_ms": round(total_sglang_ms, 1),
            "total_ms": round(total_ms, 1),
            "tps": round(tps, 1),
        }
    })


@app.get("/health")
async def health():
    total = _stats["total_accepted"] + _stats["total_rejected"]
    return {
        "status": "ok",
        "sglang_url": SGLANG_URL,
        "mtp_loaded": _mtp_head is not None,
        "embed_shape": str(_embed_weight.shape) if _embed_weight is not None else None,
        "device": str(_device),
        "stats": {
            **_stats,
            "accept_rate": round(_stats["total_accepted"] / total, 4) if total > 0 else 0,
            "avg_tps": round(_stats["total_tokens_generated"] / max(_stats["total_request_ms"] / 1000, 0.001), 1),
        }
    }


@app.post("/generate_plain")
async def generate_plain(request: Request):
    """Plain generation baseline."""
    t0 = time.time()
    data = await request.json()

    prompt_ids = data.get("input_ids", [])
    max_tokens = data.get("max_tokens", 50)
    temperature = data.get("temperature", 0.0)

    result = await sglang_generate({
        "input_ids": prompt_ids,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": temperature},
    })

    total_ms = (time.time() - t0) * 1000
    output_ids = result.get("output_ids", [])

    return JSONResponse({
        "output_ids": output_ids,
        "text": result.get("text", ""),
        "stats": {
            "tokens": len(output_ids),
            "total_ms": round(total_ms, 1),
            "tps": round(len(output_ids) / (total_ms / 1000), 1) if total_ms > 0 else 0,
        }
    })


def main():
    global SGLANG_URL, HTTP_TIMEOUT, _mtp_head, _embed_weight, _lm_head_weight, _device

    parser = argparse.ArgumentParser(description="CGC Cloud MTP Draft Server v2")
    parser.add_argument("--port", type=int, default=30070)
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30003")
    parser.add_argument("--checkpoint", default="/data/mtp_output/qwen3vl/mtp_head_qwen3vl_decode.pt")
    parser.add_argument("--model-path", default="/data/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda:1", help="Device for MTP head (cuda:0, cuda:1, cpu)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    SGLANG_URL = args.sglang_url
    HTTP_TIMEOUT = args.timeout
    _device = torch.device(args.device)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("Loading MTP head...")
    _mtp_head, config = load_mtp_head(args.checkpoint, _device)

    logger.info("Loading model embeddings...")
    _embed_weight, _lm_head_weight = load_embed_weights(args.model_path, _device)
    logger.info(f"  embed: {_embed_weight.shape}, lm_head: {_lm_head_weight.shape}")

    logger.info(f"Starting Cloud MTP Draft Server v2 on port {args.port}")
    logger.info(f"  sglang: {SGLANG_URL}")
    logger.info(f"  device: {_device}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
