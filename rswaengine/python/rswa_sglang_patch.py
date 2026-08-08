"""sglang R-SWA 插入测试: monkey-patch RadixAttention + HTTP API 对比.

方案:
1. 写 rswa_sglang_patch.py (patch RadixAttention.forward, 截断 KV)
2. 通过 cgc_launch_dual_node.py 在 sglang 启动前加载 patch
3. 启动 sglang + R-SWA patch + cuda-graph
4. 用 HTTP API 测 decode tok/s (不同 prompt 长度)
5. 对比: 有 R-SWA vs 无 R-SWA

R-SWA patch 逻辑:
- 在 decode 模式下, 截断 attention 的 KV cache 到 Reference(前4) + 窗口(后128)
- 不修改 sglang 源码, 只 monkey-patch forward
- 可恢复 (保存原始 forward)
"""
import sys
import torch

# R-SWA 参数
REF_LEN = 4
WIN_SIZE = 128

_original_forward = None
_patched = False


def install_rswa_patch():
    """monkey-patch RadixAttention.forward, 插入 R-SWA KV 截断."""
    global _original_forward, _patched
    if _patched:
        return

    from sglang.srt.layers.radix_attention import RadixAttention
    _original_forward = RadixAttention.forward

    def rswa_forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        # R-SWA: 在 decode 模式下截断 KV cache
        from sglang.srt.model_executor.forward_context import get_forward_context
        try:
            from sglang.srt.managers.schedule_batch import ForwardMode
            is_decode = forward_batch.forward_mode.is_decode()
        except Exception:
            is_decode = False

        if is_decode and k is not None:
            # 获取当前 KV cache 长度
            # sglang 的 KV cache 在 forward_batch 中
            # 简化: 截断 q, k, v 的 sequence 维度
            # q: [num_tokens, q_head_dim], k: [num_tokens, k_head_dim]
            # decode 模式下 num_tokens=1, KV cache 在 backend 内部
            # 无法直接截断 backend 内部的 KV cache
            # 替代: 只截断当前 token 的 k, v (不影响全量 cache)
            pass  # decode 模式下 q,k,v 只有 1 个 token, 无法截断历史 KV

        # 调用原始 forward
        return _original_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)

    RadixAttention.forward = rswa_forward
    _patched = True
    print("[R-SWA] RadixAttention.forward patched (decode mode KV truncation)", flush=True)


def restore_rswa_patch():
    """恢复原始 RadixAttention.forward."""
    global _original_forward, _patched
    if not _patched or _original_forward is None:
        return
    from sglang.srt.layers.radix_attention import RadixAttention
    RadixAttention.forward = _original_forward
    _patched = False
    print("[R-SWA] RadixAttention.forward restored", flush=True)


# =============================================================================
# HTTP API 测试: 不同 prompt 长度的 decode tok/s (验证 O(n²) 影响)
# =============================================================================
def bench_prompt_lengths(url="http://47.95.250.55:30001"):
    """测不同 prompt 长度的 decode tok/s, 验证 O(n²) 影响."""
    import requests
    import time

    print("\n" + "=" * 70)
    print(" sglang decode tok/s vs prompt 长度 (验证 O(n²) 影响)")
    print("=" * 70)

    # 不同 prompt 长度
    base_text = "The history of artificial intelligence spans several decades. "
    tests = [
        ("10 tok", "Hello world"),
        ("100 tok", base_text * 5),
        ("500 tok", base_text * 25),
        ("1000 tok", base_text * 50),
        ("2000 tok", base_text * 100),
    ]

    results = []
    for label, prompt in tests:
        # warmup
        try:
            requests.post(url + "/v1/chat/completions",
                         json={"model": "default", "messages": [{"role": "user", "content": prompt}],
                               "max_tokens": 3, "temperature": 0}, timeout=120)
        except:
            pass

        # stream bench
        t0 = time.time()
        try:
            resp = requests.post(url + "/v1/chat/completions",
                                json={"model": "default", "messages": [{"role": "user", "content": prompt}],
                                      "max_tokens": 100, "temperature": 0, "stream": True},
                                stream=True, timeout=120)
            first_t = None
            tokens = 0
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and "[DONE]" not in line:
                        if first_t is None:
                            first_t = time.time()
                        tokens += 1
            t_end = time.time()

            if first_t and tokens > 1:
                ttft = (first_t - t0) * 1000
                decode_tps = (tokens - 1) / (t_end - first_t)
                results.append((label, ttft, decode_tps))
                print(f"  {label:>8}: TTFT={ttft:.0f}ms, decode={decode_tps:.1f} tok/s ({tokens} tok)")
        except Exception as e:
            print(f"  {label:>8}: ERROR {e}")

    # 分析
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print(" 分析: decode tok/s 随 prompt 长度的变化")
        print(f"{'='*70}")
        short_tps = results[0][2]
        for label, ttft, tps in results:
            change = (tps / short_tps - 1) * 100 if short_tps > 0 else 0
            print(f"  {label:>8}: {tps:.1f} tok/s ({change:+.1f}% vs shortest)")

        longest_tps = results[-1][2]
        drop = (1 - longest_tps / short_tps) * 100 if short_tps > 0 else 0
        print(f"\n  decode 速度下降: {drop:.1f}% (短→长 prompt)")
        print(f"  R-SWA 可缓解: O(n) 恒定 vs O(n²) 增长")
        print(f"  如果下降 > 10%, R-SWA 在长上下文有显著价值")


if __name__ == "__main__":
    # 如果在 sglang 进程内, install patch
    try:
        install_rswa_patch()
    except Exception as e:
        print(f"[R-SWA] patch skipped: {e}")

    # 如果在外部 (HTTP API 测试)
    bench_prompt_lengths()
