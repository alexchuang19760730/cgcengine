"""MTP 首 token 预测集成 — cloud prefill → emit hidden → Mac MTP 预测首 token.

流程:
  1. Mac 发 prompt 到 cloud
  2. Cloud prefill → 返回 hidden_states (最后一层, norm 后)
  3. Mac MTP head forward(hidden, embed) → 预测首 token
  4. accept 50%: 正确 → 立即返回用户 (TTFT = prefill + 传输 + 1ms)
  5. reject 50%: 回退 cloud 首 token (TTFT = prefill + 传输 + decode)

简化实现: 不需要单独 emit 通道, 在 cloud API 响应中附带 hidden。
"""
from __future__ import annotations

import json
import time
import requests
import numpy as np
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import torch

import sys
sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2")
from mtp_patched_v3 import MTPHead, load_mtp_head


class MTPFirstTokenPredictor:
    """MTP 首 token 预测器 — 集成到 edge_first_proxy."""

    def __init__(self, mtp_checkpoint: str, target_model_path: str):
        print("[mtp-predict] Loading MTP head...")
        self.mtp = load_mtp_head(mtp_checkpoint)

        print("[mtp-predict] Loading target model (for embed + lm_head)...")
        from mlx_lm import load
        self.target_model, self.tokenizer = load(target_model_path)
        self.inner = self.target_model.language_model.model
        self.embed = self.inner.embed_tokens
        self.lm_head_w = self.embed.weight  # tied

        # 连接池 (keep-alive)
        self.session = requests.Session()
        self.cloud_url = "http://47.95.250.55:30001"

        # 预测缓存 (warm cache)
        self._cache = {}  # prompt_hash → first_token
        self._cache_max = 2000

        print("[mtp-predict] Ready")

    def predict_first_token_cloud(self, messages: list) -> dict:
        """完整流程: cloud prefill → MTP 预测首 token.

        Returns:
            {
                "first_token": int,
                "first_token_text": str,
                "method": "mtp" | "cache" | "cloud_fallback",
                "ttft_ms": float,
                "accept": bool (是否 MTP 预测正确),
            }
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_hash = hash(prompt)

        # 1. 检查缓存
        if prompt_hash in self._cache:
            cached = self._cache[prompt_hash]
            return {
                "first_token": cached,
                "first_token_text": self.tokenizer.decode([cached]),
                "method": "cache",
                "ttft_ms": 0,
                "accept": True,
            }

        t0 = time.time()

        # 2. Cloud prefill (获取 hidden + 首 token)
        # 用 cloud API prefill, 获取 hidden_states
        # 简化: 用 cloud generate 1 token + hidden
        try:
            # 方案 A: cloud 返回 hidden (需要自定义 API)
            # 方案 B: Mac 本地 prefill 获取 hidden (不经过 cloud)
            # 方案 C: cloud prefill + Mac MTP 预测 (需要 hidden 传输)

            # 当前用方案 B: Mac 本地 prefill (简化, 不需要 cloud emit)
            # 后续可改为方案 C (cloud prefill → emit hidden → Mac)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            ids = [t for t in ids if t not in (151644, 151645)]
            y = mx.array(ids, mx.uint32)

            # Mac 本地 prefill (获取 hidden)
            from mlx_lm.models.qwen3 import create_attention_mask
            h = self.embed(y[None])
            cache = [None] * len(self.inner.layers)
            mask = create_attention_mask(h, cache[0])
            for layer, c in zip(self.inner.layers, cache):
                h = layer(h, mask, c)
            hidden = self.inner.norm(h)[:, -1:, :]  # norm 后 (和 PyTorch 一致)

            # Target 首 token (ground truth)
            target_logits = hidden @ self.lm_head_w.T
            target_token = int(mx.argmax(target_logits[0, 0]).item())

            # MTP 预测首 token
            last_embed = self.embed(y[None])[:, -1:, :]
            mtp_out = self.mtp(hidden, last_embed, cache=None)
            mtp_logits = mtp_out @ self.lm_head_w.T
            mtp_token = int(mx.argmax(mtp_logits[0, 0]).item())

            ttft_ms = (time.time() - t0) * 1000
            accept = (mtp_token == target_token)

            # 记录缓存
            if len(self._cache) < self._cache_max:
                self._cache[prompt_hash] = target_token

            return {
                "first_token": mtp_token,
                "first_token_text": self.tokenizer.decode([mtp_token]),
                "method": "mtp",
                "ttft_ms": ttft_ms,
                "accept": accept,
                "target_token": target_token,
                "target_token_text": self.tokenizer.decode([target_token]),
            }

        except Exception as e:
            # 回退 cloud
            print(f"[mtp-predict] Error: {e}, fallback to cloud")
            return {
                "first_token": -1,
                "first_token_text": "",
                "method": "cloud_fallback",
                "ttft_ms": (time.time() - t0) * 1000,
                "accept": False,
            }

    def predict_with_cloud_prefill(self, messages: list) -> dict:
        """方案 C: cloud prefill → emit hidden → Mac MTP 预测.

        需要 cloud 端支持返回 hidden_states (自定义 API)。
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_hash = hash(prompt)

        # 1. 缓存
        if prompt_hash in self._cache:
            cached = self._cache[prompt_hash]
            return {
                "first_token": cached,
                "first_token_text": self.tokenizer.decode([cached]),
                "method": "cache",
                "ttft_ms": 0,
                "accept": True,
            }

        t0 = time.time()

        # 2. Cloud prefill (获取 hidden + 首 token)
        # 调用 cloud API, 获取 hidden_states
        # 注意: 标准 OpenAI API 不返回 hidden, 需要自定义 endpoint
        try:
            resp = self.session.post(
                f"{self.cloud_url}/v1/chat/completions",
                json={
                    "model": "Qwen3-VL-2B-Instruct",
                    "messages": messages,
                    "max_tokens": 1,
                    "stream": False,
                    "return_hidden": True,  # 自定义参数
                },
                timeout=10,
            )
            data = resp.json()

            if "hidden_states" in data:
                # Cloud 返回了 hidden
                hidden_np = np.array(data["hidden_states"], dtype=np.float32)
                hidden = mx.array(hidden_np).reshape(1, 1, -1)

                # 首 token (cloud 的)
                target_token = data["choices"][0]["message"]["content"]
                target_token_id = self.tokenizer.encode(target_token, add_special_tokens=False)[-1]

                # MTP 预测
                ids = [t for t in self.tokenizer.encode(prompt, add_special_tokens=False) if t not in (151644, 151645)]
                last_embed = self.embed(mx.array(ids, mx.uint32)[None])[:, -1:, :]
                mtp_out = self.mtp(hidden, last_embed, cache=None)
                mtp_logits = mtp_out @ self.lm_head_w.T
                mtp_token = int(mx.argmax(mtp_logits[0, 0]).item())

                ttft_ms = (time.time() - t0) * 1000
                accept = (mtp_token == target_token_id)

                if len(self._cache) < self._cache_max:
                    self._cache[prompt_hash] = target_token_id

                return {
                    "first_token": mtp_token,
                    "first_token_text": self.tokenizer.decode([mtp_token]),
                    "method": "mtp_cloud",
                    "ttft_ms": ttft_ms,
                    "accept": accept,
                }
            else:
                # Cloud 不支持 return_hidden, 用 cloud 首 token
                target_token = data["choices"][0]["message"]["content"]
                ttft_ms = (time.time() - t0) * 1000
                return {
                    "first_token": -1,
                    "first_token_text": target_token,
                    "method": "cloud_only",
                    "ttft_ms": ttft_ms,
                    "accept": False,
                }

        except Exception as e:
            print(f"[mtp-predict] Cloud error: {e}")
            return {
                "first_token": -1,
                "first_token_text": "",
                "method": "error",
                "ttft_ms": (time.time() - t0) * 1000,
                "accept": False,
            }


def benchmark_mtp_first_token():
    """测试 MTP 首 token 预测."""
    print("\n=== MTP 首 token 预测测试 ===\n")

    predictor = MTPFirstTokenPredictor(
        mtp_checkpoint="/tmp/mtp_head_final.pt",
        target_model_path="/Users/alexchuang/models/Qwen3-VL-2B-bf16",
    )

    prompts = [
        [{"role": "user", "content": "The capital of France is"}],
        [{"role": "user", "content": "Once upon a time"}],
        [{"role": "user", "content": "Write a Python function"}],
        [{"role": "user", "content": "Explain quantum computing"}],
        [{"role": "user", "content": "Hello, how are you?"}],
        [{"role": "user", "content": "What is 2 + 2?"}],
        [{"role": "user", "content": "France is"}],
        [{"role": "user", "content": "Translate hello to French"}],
        [{"role": "user", "content": "What is the largest planet?"}],
        [{"role": "user", "content": "How do you make a pancake?"}],
    ]

    accept_count = 0
    total_ttft = 0

    print(f"{'Prompt':40s} {'MTP':>10s} {'Target':>10s} {'Accept':>8s} {'TTFT':>8s}")
    print("-" * 80)

    for messages in prompts:
        result = predictor.predict_first_token_cloud(messages)
        prompt_short = messages[0]["content"][:38]
        mtp_text = result["first_token_text"][:8]
        tgt_text = result.get("target_token_text", "")[:8]
        accept = "✓" if result["accept"] else "✗"
        ttft = f"{result['ttft_ms']:.0f}ms"

        if result["accept"]:
            accept_count += 1
        total_ttft += result["ttft_ms"]

        print(f"{prompt_short:40s} {mtp_text:>10s} {tgt_text:>10s} {accept:>8s} {ttft:>8s}")

    print(f"\nAccept: {accept_count}/{len(prompts)} = {accept_count/len(prompts):.0%}")
    print(f"平均 TTFT: {total_ttft/len(prompts):.0f}ms")
    print(f"方法: {result['method']}")

    # 测试 warm cache
    print("\n=== Warm Cache 测试 ===")
    result = predictor.predict_first_token_cloud(prompts[0])
    print(f"重复 prompt: TTFT={result['ttft_ms']:.0f}ms, method={result['method']}")


if __name__ == "__main__":
    benchmark_mtp_first_token()
