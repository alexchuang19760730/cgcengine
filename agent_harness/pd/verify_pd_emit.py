import os, sys
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#!/usr/bin/env python3
"""椹楄瓑 PD emit 绔粸: 鍚屼竴鏂囦欢 鈫?鍏╁彴 Mac emit 鈫?Windows 鏀堕泦.

鐢ㄦ硶:
  py verify_pd_emit.py \\
    --gemma4-url http://192.168.101.X:8080 \\
    --qwen36-url http://192.168.101.Y:8080 \\
    --file prompt.txt \\
    --output hidden_pair.npz

娴佺▼:
  1. 璁€鍙栨枃鏈枃浠?
  2. 涓﹁ POST /v1/cgc/emit 鍒?Mac A (Gemma4) 鍜?Mac B (Qwen3.6)
  3. 椹楄瓑杩斿洖鐨?hidden state (base64 鈫?tensor)
  4. 鎵撳嵃绲辫▓淇℃伅
  5. 淇濆瓨鍒?.npz (鍙敤鏂?MoT-h 瑷撶反)
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
    """base64 鈫?float32 tensor [seq_len, hidden_dim]."""
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
    """POST /v1/cgc/emit 鍒颁竴鍙?Mac, 杩斿洖 response dict."""
    endpoint = url.rstrip("/") + "/v1/cgc/emit"
    payload = {
        "prompt": prompt,
        "request_id": request_id,
        "max_seq_len": max_seq_len,
    }
    print(f"  鈫?POST {endpoint} (request_id={request_id})")
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
    """椹楄瓑 emit response 涓﹁繑鍥?hidden tensor."""
    if not resp.get("success"):
        raise RuntimeError(f"emit returned error: {resp.get('error')}")

    model_id = resp.get("model_id", "")
    if expected_model not in model_id:
        print(f"  鈿狅笍  model_id mismatch: expected '{expected_model}', "
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
    """鎵撳嵃 hidden state 绲辫▓."""
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
        print(f"  鉂?{name} 鏈?NaN/Inf!")
    else:
        print(f"  鉁?{name} 鏁稿€兼甯?)


async def main():
    parser = argparse.ArgumentParser(
        description="椹楄瓑 PD emit: 鍚屼竴鏂囦欢 鈫?鍏╁彴 Mac 鈫?鏀堕泦 hidden state")
    parser.add_argument("--gemma4-url", required=True,
                        help="Mac A (Gemma4) TurboFieldfare URL, e.g. http://192.168.101.X:8080")
    parser.add_argument("--qwen36-url", required=True,
                        help="Mac B (Qwen3.6) TurboFieldfare URL, e.g. http://192.168.101.Y:8080")
    parser.add_argument("--file", required=True,
                        help="杓稿叆鏂囨湰鏂囦欢璺緫")
    parser.add_argument("--output", default="hidden_pair.npz",
                        help="杓稿嚭 .npz 鏂囦欢璺緫 (default: hidden_pair.npz)")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="鏈€澶у簭鍒楅暦搴?(default: 4096)")
    args = parser.parse_args()

    # 1. 璁€鍙栨枃浠?
    with open(args.file, "r", encoding="utf-8") as f:
        prompt = f.read()
    print(f"杓稿叆鏂囦欢: {args.file}")
    print(f"  瀛楃鏁? {len(prompt)}")
    print(f"  鍓?100 瀛楃: {prompt[:100]!r}")

    # 2. 涓﹁ emit
    print(f"\n涓﹁ emit 鍒板叐鍙?Mac...")
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

    # 3. 铏曠悊绲愭灉
    gemma4_resp = results[0]
    qwen36_resp = results[1]

    if isinstance(gemma4_resp, Exception):
        print(f"\n鉂?Gemma4 emit 澶辨晽: {gemma4_resp}")
        gemma4_resp = None
    if isinstance(qwen36_resp, Exception):
        print(f"\n鉂?Qwen3.6 emit 澶辨晽: {qwen36_resp}")
        qwen36_resp = None

    if gemma4_resp is None or qwen36_resp is None:
        print("\n鑷冲皯涓€鍙?Mac emit 澶辨晽, 鐒℃硶姣旇純.")
        sys.exit(1)

    # 4. 椹楄瓑 + 绲辫▓
    print("\n" + "=" * 60)
    print("  椹楄瓑绲愭灉")
    print("=" * 60)

    try:
        h_gemma4 = validate_response(gemma4_resp, "gemma4", 2816)
        print_stats("Gemma4 (source)", h_gemma4, gemma4_resp["prefill_ms"])
    except Exception as e:
        print(f"  鉂?Gemma4 椹楄瓑澶辨晽: {e}")
        h_gemma4 = None

    try:
        h_qwen36 = validate_response(qwen36_resp, "qwen36", 2048)
        print_stats("Qwen3.6 (target)", h_qwen36, qwen36_resp["prefill_ms"])
    except Exception as e:
        print(f"  鉂?Qwen3.6 椹楄瓑澶辨晽: {e}")
        h_qwen36 = None

    # 5. 姣旇純
    if h_gemma4 is not None and h_qwen36 is not None:
        print("\n" + "=" * 60)
        print("  璺ㄦā鍨嬫瘮杓?)
        print("=" * 60)
        print(f"  Gemma4 seq_len: {h_gemma4.shape[0]}")
        print(f"  Qwen3.6 seq_len: {h_qwen36.shape[0]}")
        if h_gemma4.shape[0] == h_qwen36.shape[0]:
            print(f"  鉁?搴忓垪闀峰害涓€鑷?({h_gemma4.shape[0]})")
            print(f"  Gemma4 hidden_dim: {h_gemma4.shape[1]}")
            print(f"  Qwen3.6 hidden_dim: {h_qwen36.shape[1]}")
            print(f"  缍害宸? {h_gemma4.shape[1] - h_qwen36.shape[1]} (闇€瑕?MoT-h 缈昏)")
        else:
            print(f"  鈿狅笍  搴忓垪闀峰害涓嶄竴鑷? {h_gemma4.shape[0]} vs {h_qwen36.shape[0]}")
            print(f"     (鍙兘鍥?tokenizer 涓嶅悓)")

    # 6. 淇濆瓨
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
        print(f"\n鉁?宸蹭繚瀛樺埌 {args.output}")
        print(f"   Gemma4: {h_gemma4.shape} 鈫?MoT-h 瑷撶反鐢?source")
        print(f"   Qwen3.6: {h_qwen36.shape} 鈫?MoT-h 瑷撶反鐢?target")

    print("\n涓嬩竴姝?")
    print("  1. 鐢ㄥ鍊嬫枃浠惰窇姝よ叧鏈? 鎺￠泦瑷撶反灏?)
    print("  2. 瑷撶反 MoT-h: py train_mot_h.py --data hidden_pair.npz")
    print("  3. 绔埌绔脯瑭? 鍟熷嫊 coordinator.py, 璺?Gemma4 emit 鈫?MoT-h 鈫?Qwen3.6 resume")


if __name__ == "__main__":
    asyncio.run(main())
