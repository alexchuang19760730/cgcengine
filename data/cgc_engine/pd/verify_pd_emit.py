#!/usr/bin/env python3
"""驗證 PD emit 端點: 同一文件 → 兩台 Mac emit → Windows 收集.

用法:
  py verify_pd_emit.py \\
    --gemma4-url http://192.168.101.X:8080 \\
    --qwen36-url http://192.168.101.Y:8080 \\
    --file prompt.txt \\
    --output hidden_pair.npz

流程:
  1. 讀取文本文件
  2. 並行 POST /v1/cgc/emit 到 Mac A (Gemma4) 和 Mac B (Qwen3.6)
  3. 驗證返回的 hidden state (base64 → tensor)
  4. 打印統計信息
  5. 保存到 .npz (可用於 MoT-h 訓練)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from typing import Optional

import aiohttp
import torch


def decode_hidden_state(b64: str, seq_len: int, hidden_dim: int) -> torch.Tensor:
    """base64 → float32 tensor [seq_len, hidden_dim]."""
    raw = base64.b64decode(b64)
    expected = seq_len * hidden_dim * 4  # float32 = 4 bytes
    if len(raw) != expected:
        raise ValueError(
            f"hidden state bytes mismatch: got {len(raw)}, expected {expected} "
            f"(seq_len={seq_len}, hidden_dim={hidden_dim})")
    return torch.frombuffer(raw, dtype=torch.float32).reshape(seq_len, hidden_dim).clone()


async def emit(session: aiohttp.ClientSession, url: str, prompt: str,
               request_id: str, max_seq_len: int = 4096,
               timeout: float = 120.0) -> dict:
    """POST /v1/cgc/emit 到一台 Mac, 返回 response dict."""
    endpoint = url.rstrip("/") + "/v1/cgc/emit"
    payload = {
        "prompt": prompt,
        "request_id": request_id,
        "max_seq_len": max_seq_len,
    }
    print(f"  → POST {endpoint} (request_id={request_id})")
    t0 = time.time()
    async with session.post(endpoint, json=payload, timeout=timeout) as resp:
        elapsed = time.time() - t0
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(
                f"emit failed: HTTP {resp.status} from {url}\n{text}")
        data = await resp.json()
        data["_elapsed_s"] = elapsed
        return data


def validate_response(resp: dict, expected_model: str,
                       expected_hidden: int) -> torch.Tensor:
    """驗證 emit response 並返回 hidden tensor."""
    if not resp.get("success"):
        raise RuntimeError(f"emit returned error: {resp.get('error')}")

    model_id = resp.get("model_id", "")
    if expected_model not in model_id:
        print(f"  ⚠️  model_id mismatch: expected '{expected_model}', "
              f"got '{model_id}'")

    seq_len = resp["seq_len"]
    hidden_dim = resp["hidden_dim"]
    if hidden_dim != expected_hidden:
        raise RuntimeError(
            f"hidden_dim mismatch: expected {expected_hidden}, got {hidden_dim}")

    b64 = resp["hidden_state_b64"]
    hidden = decode_hidden_state(b64, seq_len, hidden_dim)
    return hidden


def print_stats(name: str, hidden: torch.Tensor, prefill_ms: float):
    """打印 hidden state 統計."""
    print(f"\n  {name}:")
    print(f"    shape: {tuple(hidden.shape)}")
    print(f"    dtype: {hidden.dtype}")
    print(f"    mean:  {hidden.mean().item():.6f}")
    print(f"    std:   {hidden.std().item():.6f}")
    print(f"    min:   {hidden.min().item():.4f}")
    print(f"    max:   {hidden.max().item():.4f}")
    nan_count = torch.isnan(hidden).sum().item()
    inf_count = torch.isinf(hidden).sum().item()
    print(f"    NaN:   {nan_count}")
    print(f"    Inf:   {inf_count}")
    print(f"    prefill: {prefill_ms:.1f}ms")
    if nan_count > 0 or inf_count > 0:
        print(f"  ❌ {name} 有 NaN/Inf!")
    else:
        print(f"  ✅ {name} 數值正常")


async def main():
    parser = argparse.ArgumentParser(
        description="驗證 PD emit: 同一文件 → 兩台 Mac → 收集 hidden state")
    parser.add_argument("--gemma4-url", required=True,
                        help="Mac A (Gemma4) TurboFieldfare URL, e.g. http://192.168.101.X:8080")
    parser.add_argument("--qwen36-url", required=True,
                        help="Mac B (Qwen3.6) TurboFieldfare URL, e.g. http://192.168.101.Y:8080")
    parser.add_argument("--file", required=True,
                        help="輸入文本文件路徑")
    parser.add_argument("--output", default="hidden_pair.npz",
                        help="輸出 .npz 文件路徑 (default: hidden_pair.npz)")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="最大序列長度 (default: 4096)")
    args = parser.parse_args()

    # 1. 讀取文件
    with open(args.file, "r", encoding="utf-8") as f:
        prompt = f.read()
    print(f"輸入文件: {args.file}")
    print(f"  字符數: {len(prompt)}")
    print(f"  前 100 字符: {prompt[:100]!r}")

    # 2. 並行 emit
    print(f"\n並行 emit 到兩台 Mac...")
    request_id = f"verify_{int(time.time())}"
    async with aiohttp.ClientSession() as session:
        tasks = [
            emit(session, args.gemma4_url, prompt,
                 request_id=f"{request_id}_gemma4",
                 max_seq_len=args.max_seq_len),
            emit(session, args.qwen36_url, prompt,
                 request_id=f"{request_id}_qwen36",
                 max_seq_len=args.max_seq_len),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. 處理結果
    gemma4_resp = results[0]
    qwen36_resp = results[1]

    if isinstance(gemma4_resp, Exception):
        print(f"\n❌ Gemma4 emit 失敗: {gemma4_resp}")
        gemma4_resp = None
    if isinstance(qwen36_resp, Exception):
        print(f"\n❌ Qwen3.6 emit 失敗: {qwen36_resp}")
        qwen36_resp = None

    if gemma4_resp is None or qwen36_resp is None:
        print("\n至少一台 Mac emit 失敗, 無法比較.")
        sys.exit(1)

    # 4. 驗證 + 統計
    print("\n" + "=" * 60)
    print("  驗證結果")
    print("=" * 60)

    try:
        h_gemma4 = validate_response(gemma4_resp, "gemma4", 2816)
        print_stats("Gemma4 (source)", h_gemma4, gemma4_resp["prefill_ms"])
    except Exception as e:
        print(f"  ❌ Gemma4 驗證失敗: {e}")
        h_gemma4 = None

    try:
        h_qwen36 = validate_response(qwen36_resp, "qwen36", 2048)
        print_stats("Qwen3.6 (target)", h_qwen36, qwen36_resp["prefill_ms"])
    except Exception as e:
        print(f"  ❌ Qwen3.6 驗證失敗: {e}")
        h_qwen36 = None

    # 5. 比較
    if h_gemma4 is not None and h_qwen36 is not None:
        print("\n" + "=" * 60)
        print("  跨模型比較")
        print("=" * 60)
        print(f"  Gemma4 seq_len: {h_gemma4.shape[0]}")
        print(f"  Qwen3.6 seq_len: {h_qwen36.shape[0]}")
        if h_gemma4.shape[0] == h_qwen36.shape[0]:
            print(f"  ✅ 序列長度一致 ({h_gemma4.shape[0]})")
            print(f"  Gemma4 hidden_dim: {h_gemma4.shape[1]}")
            print(f"  Qwen3.6 hidden_dim: {h_qwen36.shape[1]}")
            print(f"  維度差: {h_gemma4.shape[1] - h_qwen36.shape[1]} (需要 MoT-h 翻譯)")
        else:
            print(f"  ⚠️  序列長度不一致: {h_gemma4.shape[0]} vs {h_qwen36.shape[0]}")
            print(f"     (可能因 tokenizer 不同)")

    # 6. 保存
    if h_gemma4 is not None and h_qwen36 is not None:
        save_data = {
            "prompt": prompt,
            "h_gemma4": h_gemma4.numpy(),
            "h_qwen36": h_qwen36.numpy(),
            "gemma4_model_id": gemma4_resp["model_id"],
            "qwen36_model_id": qwen36_resp["model_id"],
            "gemma4_prefill_ms": gemma4_resp["prefill_ms"],
            "qwen36_prefill_ms": qwen36_resp["prefill_ms"],
            "gemma4_finished_layer": gemma4_resp["finished_layer"],
            "qwen36_finished_layer": qwen36_resp["finished_layer"],
        }
        torch.save(save_data, args.output)
        print(f"\n✅ 已保存到 {args.output}")
        print(f"   Gemma4: {h_gemma4.shape} → MoT-h 訓練用 source")
        print(f"   Qwen3.6: {h_qwen36.shape} → MoT-h 訓練用 target")

    print("\n下一步:")
    print("  1. 用多個文件跑此腳本, 採集訓練對")
    print("  2. 訓練 MoT-h: py train_mot_h.py --data hidden_pair.npz")
    print("  3. 端到端測試: 啟動 coordinator.py, 跑 Gemma4 emit → MoT-h → Qwen3.6 resume")


if __name__ == "__main__":
    asyncio.run(main())
