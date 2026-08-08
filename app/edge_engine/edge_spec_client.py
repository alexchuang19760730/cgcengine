#!/usr/bin/env python3
"""Edge Spec Decode Client — 端侧 NGRAM draft + 云端 verify.

路径 B 端侧组件: 在 Mac 上运行 NGRAM draft, 发送云端 verify.

工作流程:
  1. 接收 prompt text
  2. Tokenize → prompt_ids
  3. 向云端请求第一个 token (regular generate, 获取 prefill)
  4. 循环:
     a. NGRAM 生成 N 个 draft tokens
     b. 发送 prompt_ids + draft_tokens 到云端 /verify
     c. 解析 accept/reject 结果
     d. 输出 accepted tokens + corrected token
     e. 更新 NGRAM suffix tree
  5. 返回完整输出 + 统计

用法:
  python3 edge_spec_client.py --verify-url http://39.106.118.206:30060 \\
      --tokenizer /path/to/tokenizer.json \\
      --prompt "def fibonacci(n):" --max-tokens 50
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter, defaultdict
from typing import Optional


class NgramDraftModel:
    """NGRAM suffix tree draft model.

    维护 token 序列的后缀树, 给定 context 返回最可能的后续 tokens.
    无需模型, 纯内存查找, 延迟 < 0.1ms.
    """

    def __init__(self, ngram_size: int = 3, max_draft: int = 4):
        self.ngram_size = ngram_size
        self.max_draft = max_draft
        # suffix_tree[ngram_tuple] = Counter({next_token: count})
        self.suffix_tree: dict[tuple, Counter] = defaultdict(Counter)

    def add_sequence(self, tokens: list[int]):
        """Add a token sequence to the suffix tree."""
        for n in range(1, self.ngram_size + 1):
            for i in range(len(tokens) - n):
                key = tuple(tokens[i:i + n])
                self.suffix_tree[key][tokens[i + n]] += 1

    def draft(self, context: list[int]) -> list[int]:
        """Generate draft tokens given context.

        策略: 从最长 ngram 开始匹配, 逐个生成 draft token.
        """
        if not context:
            return []

        draft = []
        current_context = list(context)

        for _ in range(self.max_draft):
            found = False
            # Try from longest to shortest ngram
            for n in range(min(self.ngram_size, len(current_context)), 0, -1):
                key = tuple(current_context[-n:])
                if key in self.suffix_tree and self.suffix_tree[key]:
                    # Most common next token
                    next_tok = self.suffix_tree[key].most_common(1)[0][0]
                    draft.append(next_tok)
                    current_context.append(next_tok)
                    found = True
                    break

            if not found:
                break

        return draft


class EdgeSpecDecoder:
    """端侧 spec decode: NGRAM draft + cloud verify.

    用法:
        decoder = EdgeSpecDecoder(verify_url, tokenizer)
        result = decoder.generate("def fibonacci(n):", max_tokens=50)
        print(result['text'])
        print(f"accept_rate={result['accept_rate']:.1%}, tps={result['tps']:.1f}")
    """

    def __init__(
        self,
        verify_url: str,
        tokenizer_path: str,
        ngram_size: int = 3,
        max_draft: int = 4,
        sglang_url: str = "",
    ):
        self.verify_url = verify_url
        self.sglang_url = sglang_url or verify_url.replace(":30060", ":30003")
        self.ngram = NgramDraftModel(ngram_size=ngram_size, max_draft=max_draft)
        self.max_draft = max_draft

        # Load tokenizer
        self.tokenizer = None
        self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, path: str):
        """Load tokenizer from file path."""
        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(path)
            print(f"[edge] Tokenizer loaded from {path}")
        except Exception as e:
            print(f"[edge] WARNING: Could not load tokenizer from {path}: {e}")
            print(f"[edge] Will use cloud-side tokenization (/verify_text)")
            self.tokenizer = None

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self.tokenizer:
            return self.tokenizer.encode(text).ids
        return []

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self.tokenizer:
            return self.tokenizer.decode(ids)
        return ""

    def _http_post(self, url: str, data: dict, timeout: float = 30.0) -> dict:
        """HTTP POST helper."""
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _http_get(self, url: str, timeout: float = 10.0) -> dict:
        """HTTP GET helper."""
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
    ) -> dict:
        """Generate text using spec decode: NGRAM draft + cloud verify.

        Returns:
            {
                "text": str,
                "token_ids": list[int],
                "total_tokens": int,
                "accepted": int,
                "rejected": int,
                "accept_rate": float,
                "rounds": int,
                "ttft_ms": float,
                "total_ms": float,
                "tps": float,
                "speedup": float,  # vs plain generation
            }
        """
        t0 = time.time()

        # 1. Tokenize prompt
        prompt_ids = self.encode(prompt)
        if not prompt_ids and not self.tokenizer:
            # Use cloud-side tokenization
            return self._generate_text_mode(prompt, max_tokens, temperature)

        # 2. Get first token from cloud (regular generate)
        # This also warms up the sglang prefill cache
        first_result = self._http_post(
            f"{self.sglang_url}/generate",
            {
                "input_ids": prompt_ids,
                "sampling_params": {"max_new_tokens": 1, "temperature": temperature},
            },
        )
        first_token = first_result.get("output_ids", [])[0]
        all_tokens = [first_token]
        ttft = time.time() - t0

        # Add prompt + first token to NGRAM
        self.ngram.add_sequence(prompt_ids + [first_token])

        # 3. Spec decode loop
        total_accepted = 0
        total_rejected = 0
        total_rounds = 0
        current_ids = prompt_ids + [first_token]

        while len(all_tokens) < max_tokens:
            # a. NGRAM draft
            draft_tokens = self.ngram.draft(current_ids)

            if not draft_tokens:
                # No NGRAM match, get one token from cloud
                result = self._http_post(
                    f"{self.sglang_url}/generate",
                    {
                        "input_ids": current_ids,
                        "sampling_params": {"max_new_tokens": 1, "temperature": temperature},
                    },
                )
                next_token = result.get("output_ids", [])[0]
                all_tokens.append(next_token)
                current_ids.append(next_token)
                self.ngram.add_sequence(current_ids)
                total_rounds += 1
                continue

            # b. Send to cloud verify
            verify_result = self._http_post(
                f"{self.verify_url}/verify",
                {
                    "prompt_ids": current_ids,
                    "draft_tokens": draft_tokens,
                },
            )

            if not verify_result.get("success"):
                # Verify failed, fall back to regular generation
                result = self._http_post(
                    f"{self.sglang_url}/generate",
                    {
                        "input_ids": current_ids,
                        "sampling_params": {"max_new_tokens": 1, "temperature": temperature},
                    },
                )
                next_token = result.get("output_ids", [])[0]
                all_tokens.append(next_token)
                current_ids.append(next_token)
                self.ngram.add_sequence(current_ids)
                total_rounds += 1
                continue

            # c. Process verify result
            accepted = verify_result.get("accepted_tokens", [])
            accepted_count = verify_result.get("accepted_count", 0)
            rejected_at = verify_result.get("rejected_at", -1)
            corrected_token = verify_result.get("corrected_token", -1)

            # Add accepted tokens
            for tok in accepted:
                all_tokens.append(tok)
                current_ids.append(tok)

            total_accepted += accepted_count
            total_rejected += len(draft_tokens) - accepted_count

            # Add corrected token (if rejected)
            if rejected_at >= 0 and corrected_token >= 0:
                all_tokens.append(corrected_token)
                current_ids.append(corrected_token)

            # d. Update NGRAM with the new tokens
            self.ngram.add_sequence(current_ids)

            total_rounds += 1

            # Check if we're done
            if len(all_tokens) >= max_tokens:
                break

            # If all draft tokens were accepted and no correction,
            # we need to get at least one more token from the cloud
            if rejected_at == -1:
                # All accepted — get the bonus token from the verify response
                # The verify response's sglang call generated 1 token (max_new_tokens=1)
                # But we don't have it in the verify response...
                # We need to do a regular generate to get the next token
                # Actually, let's just continue to the next draft round
                # The NGRAM should be able to draft more tokens now
                pass

        # Trim to max_tokens
        all_tokens = all_tokens[:max_tokens]

        total_ms = (time.time() - t0) * 1000
        total_generated = len(all_tokens)
        total_draft = total_accepted + total_rejected
        accept_rate = total_accepted / total_draft if total_draft > 0 else 0

        # Calculate plain generation baseline (for speedup)
        # Plain: each token requires one forward pass
        # Spec: each round verifies N tokens in one forward pass
        # Speedup ≈ (tokens per round) / (1 + draft overhead)
        # But we measure actual speedup by comparing tps

        text = self.decode(all_tokens)

        return {
            "text": text,
            "token_ids": all_tokens,
            "total_tokens": total_generated,
            "accepted": total_accepted,
            "rejected": total_rejected,
            "accept_rate": round(accept_rate, 3),
            "rounds": total_rounds,
            "ttft_ms": round(ttft * 1000, 1),
            "total_ms": round(total_ms, 1),
            "tps": round(total_generated / (total_ms / 1000), 1) if total_ms > 0 else 0,
        }

    def _generate_text_mode(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """Fallback: use cloud-side tokenization via /verify_text."""
        t0 = time.time()

        # Get first token from cloud
        first_result = self._http_post(
            f"{self.sglang_url}/generate",
            {
                "text": prompt,
                "sampling_params": {"max_new_tokens": 1, "temperature": temperature},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 1,
            },
        )

        meta = first_result.get("meta_info", {})
        input_lp = meta.get("input_token_logprobs", [])
        output_ids = first_result.get("output_ids", [])

        prompt_ids = [e[1] for e in input_lp if e and len(e) >= 2]
        first_token = output_ids[0] if output_ids else -1

        if first_token < 0 or not prompt_ids:
            return {
                "text": "",
                "token_ids": [],
                "total_tokens": 0,
                "accepted": 0,
                "rejected": 0,
                "accept_rate": 0,
                "rounds": 0,
                "ttft_ms": 0,
                "total_ms": 0,
                "tps": 0,
                "error": "Failed to get first token",
            }

        all_tokens = [first_token]
        ttft = time.time() - t0
        self.ngram.add_sequence(prompt_ids + [first_token])

        current_ids = prompt_ids + [first_token]
        total_accepted = 0
        total_rejected = 0
        total_rounds = 0

        while len(all_tokens) < max_tokens:
            draft_tokens = self.ngram.draft(current_ids)

            if not draft_tokens:
                result = self._http_post(
                    f"{self.sglang_url}/generate",
                    {
                        "input_ids": current_ids,
                        "sampling_params": {"max_new_tokens": 1, "temperature": temperature},
                    },
                )
                next_token = result.get("output_ids", [])[0]
                all_tokens.append(next_token)
                current_ids.append(next_token)
                self.ngram.add_sequence(current_ids)
                total_rounds += 1
                continue

            verify_result = self._http_post(
                f"{self.verify_url}/verify",
                {
                    "prompt_ids": current_ids,
                    "draft_tokens": draft_tokens,
                },
            )

            if not verify_result.get("success"):
                result = self._http_post(
                    f"{self.sglang_url}/generate",
                    {
                        "input_ids": current_ids,
                        "sampling_params": {"max_new_tokens": 1, "temperature": temperature},
                    },
                )
                next_token = result.get("output_ids", [])[0]
                all_tokens.append(next_token)
                current_ids.append(next_token)
                self.ngram.add_sequence(current_ids)
                total_rounds += 1
                continue

            accepted = verify_result.get("accepted_tokens", [])
            accepted_count = verify_result.get("accepted_count", 0)
            rejected_at = verify_result.get("rejected_at", -1)
            corrected_token = verify_result.get("corrected_token", -1)

            for tok in accepted:
                all_tokens.append(tok)
                current_ids.append(tok)

            total_accepted += accepted_count
            total_rejected += len(draft_tokens) - accepted_count

            if rejected_at >= 0 and corrected_token >= 0:
                all_tokens.append(corrected_token)
                current_ids.append(corrected_token)

            self.ngram.add_sequence(current_ids)
            total_rounds += 1

            if len(all_tokens) >= max_tokens:
                break

        all_tokens = all_tokens[:max_tokens]
        total_ms = (time.time() - t0) * 1000
        total_generated = len(all_tokens)
        total_draft = total_accepted + total_rejected
        accept_rate = total_accepted / total_draft if total_draft > 0 else 0

        # Decode using cloud
        decode_result = self._http_post(
            f"{self.sglang_url}/generate",
            {
                "input_ids": all_tokens,
                "sampling_params": {"max_new_tokens": 0, "temperature": temperature},
            },
        )
        text = decode_result.get("text", "")

        return {
            "text": text,
            "token_ids": all_tokens,
            "total_tokens": total_generated,
            "accepted": total_accepted,
            "rejected": total_rejected,
            "accept_rate": round(accept_rate, 3),
            "rounds": total_rounds,
            "ttft_ms": round(ttft * 1000, 1),
            "total_ms": round(total_ms, 1),
            "tps": round(total_generated / (total_ms / 1000), 1) if total_ms > 0 else 0,
        }

    def bench_plain(self, prompt: str, max_tokens: int = 50) -> dict:
        """Plain generation baseline (no spec decode) for comparison."""
        t0 = time.time()
        prompt_ids = self.encode(prompt)

        if not prompt_ids:
            # Use text mode
            result = self._http_post(
                f"{self.sglang_url}/generate",
                {
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0},
                },
            )
        else:
            result = self._http_post(
                f"{self.sglang_url}/generate",
                {
                    "input_ids": prompt_ids,
                    "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.0},
                },
            )

        total_ms = (time.time() - t0) * 1000
        output_ids = result.get("output_ids", [])
        text = result.get("text", self.decode(output_ids))

        return {
            "text": text,
            "token_ids": output_ids,
            "total_tokens": len(output_ids),
            "total_ms": round(total_ms, 1),
            "tps": round(len(output_ids) / (total_ms / 1000), 1) if total_ms > 0 else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Edge Spec Decode Client")
    parser.add_argument("--verify-url", default="http://127.0.0.1:30060",
                        help="Cloud verify server URL")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30003",
                        help="sglang server URL (for first token + fallback)")
    parser.add_argument("--tokenizer", default="",
                        help="Tokenizer JSON file path")
    parser.add_argument("--prompt", default="def fibonacci(n):",
                        help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--max-draft", type=int, default=4)
    parser.add_argument("--bench-plain", action="store_true",
                        help="Also run plain generation for comparison")
    args = parser.parse_args()

    decoder = EdgeSpecDecoder(
        verify_url=args.verify_url,
        tokenizer_path=args.tokenizer,
        ngram_size=args.ngram_size,
        max_draft=args.max_draft,
        sglang_url=args.sglang_url,
    )

    # Bench prompts
    prompts = [args.prompt] if args.prompt != "def fibonacci(n):" else [
        "def fibonacci(n):",
        "def bubble_sort(arr):",
        "class LinkedList:",
        "import numpy as np\n",
        "def binary_search(arr, target):",
        "async def fetch_data(url):",
        "def train_model(X, y):",
        "export default function App() {",
    ]

    print(f"\n{'='*60}")
    print(f"Edge Spec Decode (NGRAM + Cloud Verify)")
    print(f"{'='*60}")
    print(f"Verify URL: {args.verify_url}")
    print(f"sglang URL: {args.sglang_url}")
    print(f"NGRAM size: {args.ngram_size}, Max draft: {args.max_draft}")
    print(f"Prompts: {len(prompts)}")
    print()

    all_spec_results = []
    all_plain_results = []

    for prompt in prompts:
        print(f"Prompt: {prompt[:50]}...")

        # Spec decode
        spec_result = decoder.generate(prompt, max_tokens=args.max_tokens)
        all_spec_results.append(spec_result)

        print(f"  [SPEC] tokens={spec_result['total_tokens']}, "
              f"accept={spec_result['accept_rate']:.1%}, "
              f"rounds={spec_result['rounds']}, "
              f"tps={spec_result['tps']:.1f}, "
              f"ttft={spec_result['ttft_ms']:.0f}ms, "
              f"total={spec_result['total_ms']:.0f}ms")

        # Plain baseline
        if args.bench_plain:
            # Reset NGRAM for fair comparison
            decoder.ngram = NgramDraftModel(args.ngram_size, args.max_draft)
            plain_result = decoder.bench_plain(prompt, max_tokens=args.max_tokens)
            all_plain_results.append(plain_result)
            print(f"  [PLAIN] tokens={plain_result['total_tokens']}, "
                  f"tps={plain_result['tps']:.1f}, "
                  f"total={plain_result['total_ms']:.0f}ms")

            if spec_result['tps'] > 0 and plain_result['tps'] > 0:
                speedup = spec_result['tps'] / plain_result['tps']
                print(f"  [SPEEDUP] {speedup:.2f}x")

        print()

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")

    total_spec_tps = sum(r['tps'] for r in all_spec_results) / len(all_spec_results)
    total_accept = sum(r['accepted'] for r in all_spec_results)
    total_draft = sum(r['accepted'] + r['rejected'] for r in all_spec_results)
    avg_accept = total_accept / total_draft if total_draft > 0 else 0

    print(f"Spec decode: avg_tps={total_spec_tps:.1f}, avg_accept={avg_accept:.1%}")

    if all_plain_results:
        total_plain_tps = sum(r['tps'] for r in all_plain_results) / len(all_plain_results)
        print(f"Plain: avg_tps={total_plain_tps:.1f}")
        if total_plain_tps > 0:
            print(f"Speedup: {total_spec_tps / total_plain_tps:.2f}x")

    # Health check
    try:
        health = decoder._http_get(f"{args.verify_url}/health")
        print(f"\nVerify server stats: {json.dumps(health.get('stats', {}), indent=2)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
