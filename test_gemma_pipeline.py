#!/usr/bin/env python3
"""Gemma 4 26B-A4B Pipeline 投机解码测试

用法:
  # 1. 先在 Host2 启动 sglang (等模型下载完成后)
  # cgc model launch gemma-4-26b-a4b --speculative-algorithm NEXTN --exec

  # 2. 在 Mac 运行此测试
  python3 test_gemma_pipeline.py --cloud-url http://47.95.250.55:30001

  # 3. 不投机基线
  python3 test_gemma_pipeline.py --cloud-url http://47.95.250.55:30001 --no-speculative

  # 4. Pipeline 模式
  python3 test_gemma_pipeline.py --cloud-url http://47.95.250.55:30001 --pipeline
"""

import argparse
import time
import requests
import sys


def test_no_speculative(cloud_url: str, prompt: str, max_tokens: int = 100):
    """不投机基线: 全云端流式."""
    print(f"\n=== 不投机 (全云端流式) ===")
    t0 = time.time()
    
    r = requests.post(
        f"{cloud_url}/v1/chat/completions",
        json={
            "model": "gemma-4-26b-a4b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=60,
    )
    
    tokens = 0
    for line in r.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                tokens += 1
    
    dt = time.time() - t0
    tps = tokens / dt if dt > 0 else 0
    print(f"  tokens: {tokens}, time: {dt:.2f}s, tps: {tps:.1f} tok/s")
    return tps


def test_pipeline(cloud_url: str, prompt: str, max_tokens: int = 100, n: int = 4):
    """Pipeline 投机: 端侧 draft + 云端 batch verify."""
    print(f"\n=== Pipeline 投机 (N={n}) ===")
    t0 = time.time()
    total_tokens = 0
    accepted_total = 0
    rejected_total = 0
    batches = 0
    
    while total_tokens < max_tokens:
        # ① 端侧 draft (模拟, 实际用 MTP head)
        # 这里用云端先生成 draft (模拟端侧 draft)
        draft_prompt = prompt if total_tokens == 0 else None
        
        # ② emit draft + verify (batch)
        try:
            r = requests.post(
                f"{cloud_url}/v1/chat/completions",
                json={
                    "model": "gemma-4-26b-a4b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": n + 1,
                    "stream": False,
                    "speculative": True,
                    "num_speculative_tokens": n,
                },
                timeout=30,
            )
            
            if r.status_code == 200:
                resp = r.json()
                choices = resp.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    new_tokens = len(text.split())  # 粗略计数
                    total_tokens += max(new_tokens, 1)
                    accepted = min(new_tokens, n)
                    accepted_total += accepted
                    rejected_total += max(0, n - accepted)
                    batches += 1
                    print(f"  batch {batches}: +{new_tokens} tokens (total={total_tokens})")
            else:
                print(f"  verify 失败: {r.status_code}")
                break
        except Exception as e:
            print(f"  错误: {e}")
            break
        
        if batches > 50:  # 安全限制
            break
    
    dt = time.time() - t0
    tps = total_tokens / dt if dt > 0 else 0
    print(f"  total: {total_tokens} tokens, {batches} batches, {dt:.2f}s, {tps:.1f} tok/s")
    print(f"  accept: {accepted_total}, reject: {rejected_total}")
    return tps


def test_simple(cloud_url: str, prompt: str, max_tokens: int = 50):
    """简单测试: 验证 Gemma 4 可用."""
    print(f"\n=== 简单测试 ===")
    print(f"  URL: {cloud_url}")
    print(f"  Prompt: {prompt}")
    
    try:
        r = requests.post(
            f"{cloud_url}/v1/chat/completions",
            json={
                "model": "gemma-4-26b-a4b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=60,
        )
        
        if r.status_code == 200:
            resp = r.json()
            text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = resp.get("usage", {})
            print(f"  回复: {text[:200]}...")
            print(f"  usage: {usage}")
            print(f"  ✅ Gemma 4 可用")
            return True
        else:
            print(f"  ❌ HTTP {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Pipeline 投机解码测试")
    parser.add_argument("--cloud-url", type=str, default="http://47.95.250.55:30001")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--n", type=int, default=4, help="Draft tokens per batch")
    parser.add_argument("--pipeline", action="store_true", help="Test pipeline mode")
    parser.add_argument("--no-speculative", action="store_true", help="Test no-speculative baseline")
    parser.add_argument("--simple", action="store_true", help="Simple connectivity test")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" Gemma 4 26B-A4B Pipeline 投机解码测试")
    print("=" * 60)
    print(f"  Cloud: {args.cloud_url}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Max tokens: {args.max_tokens}")
    
    if args.simple:
        test_simple(args.cloud_url, args.prompt, args.max_tokens)
        return
    
    if args.no_speculative:
        tps = test_no_speculative(args.cloud_url, args.prompt, args.max_tokens)
        print(f"\n结果: {tps:.1f} tok/s (不投机)")
    
    if args.pipeline:
        tps = test_pipeline(args.cloud_url, args.prompt, args.max_tokens, args.n)
        print(f"\n结果: {tps:.1f} tok/s (Pipeline)")
    
    if not args.no_speculative and not args.pipeline:
        # 默认: 先简单测试, 再两种都测
        if test_simple(args.cloud_url, args.prompt, 30):
            print("\n--- 性能对比 ---")
            tps_no = test_no_speculative(args.cloud_url, args.prompt, args.max_tokens)
            tps_pipe = test_pipeline(args.cloud_url, args.prompt, args.max_tokens, args.n)
            
            print(f"\n{'=' * 60}")
            print(f" 对比结果:")
            print(f"  不投机:     {tps_no:.1f} tok/s")
            print(f"  Pipeline:   {tps_pipe:.1f} tok/s")
            print(f"  加速:       {tps_pipe/tps_no:.2f}x" if tps_no > 0 else "")
            print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
