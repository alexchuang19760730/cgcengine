"""投机 decode + R-SWA 叠加测试 (MLX, monkey-patch 插入, 不修改原代码).

测试方案:
1. 加载 Qwen3-VL-2B + 0.5B draft (mlx_lm)
2. 测 baseline: 标准投机 decode tok/s (无 R-SWA)
3. monkey-patch: 在 attention 中插入 R-SWA (截断 KV 到 Reference + 窗口)
4. 测 R-SWA + 投机 decode tok/s
5. 恢复原始 attention
6. 对比

R-SWA 插入方式: 在 cache.update_and_fetch 后, 截断 keys/values 到
  Reference (前 4 tokens) + 窗口 (后 128 tokens) = 恒定 132 范围
不修改 mlx_lm 源码, 只 patch __call__, 测完恢复.
"""
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate

# R-SWA 参数
REF_LEN = 4      # Reference 永久区 (系统 prompt)
WIN_SIZE = 128    # 窗口滑动区


def install_rswa_patch(model):
    """monkey-patch model 的 attention 层, 插入 R-SWA KV 截断.

    返回 restore 函数 (调用后恢复原始 attention).
    """
    from mlx_lm.models.qwen2 import scaled_dot_product_attention

    # 找到所有 attention 层 (Qwen3-VL: language_model.layers)
    if hasattr(model, 'language_model'):
        layers = model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise RuntimeError("Cannot find decoder layers")
    originals = []

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        original_call = attn.__call__
        originals.append((attn, original_call))

        def make_rswa_call(orig, attn_self):
            def rswa_call(x, mask=None, cache=None):
                B, L, D = x.shape
                queries = attn_self.q_proj(x)
                keys = attn_self.k_proj(x)
                values = attn_self.v_proj(x)

                queries = queries.reshape(B, L, attn_self.n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, attn_self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, attn_self.n_kv_heads, -1).transpose(0, 2, 1, 3)

                if cache is not None:
                    queries = attn_self.rope(queries, offset=cache.offset)
                    keys = attn_self.rope(keys, offset=cache.offset)
                    keys, values = cache.update_and_fetch(keys, values)
                else:
                    queries = attn_self.rope(queries)
                    keys = attn_self.rope(keys)

                # === R-SWA 插入: 截断 KV 到 Reference + 窗口 ===
                seq_len = keys.shape[2]
                if seq_len > REF_LEN + WIN_SIZE:
                    # 保留前 REF_LEN (Reference) + 后 WIN_SIZE (窗口)
                    keys = mx.concatenate([
                        keys[:, :, :REF_LEN, :],
                        keys[:, :, -WIN_SIZE:, :]
                    ], axis=2)
                    values = mx.concatenate([
                        values[:, :, :REF_LEN, :],
                        values[:, :, -WIN_SIZE:, :]
                    ], axis=2)

                output = scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=attn_self.scale, mask=mask
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return attn_self.o_proj(output)

            return rswa_call

        attn.__call__ = make_rswa_call(original_call, attn)

    def restore():
        for attn, orig in originals:
            attn.__call__ = orig

    return restore


def bench_spec_decode(model, tokenizer, draft_model, prompt, max_tokens=50, label=""):
    """测投机 decode tok/s."""
    # warmup
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=3, draft_model=draft_model, num_draft_tokens=16):
        pass

    t0 = time.time()
    tokens = []
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                                 draft_model=draft_model, num_draft_tokens=16):
        tokens.append(resp.token if hasattr(resp, 'token') else resp)
    dt = time.time() - t0
    tps = len(tokens) / dt if dt > 0 else 0
    print(f"  {label}: {tps:.1f} tok/s ({len(tokens)} tok / {dt:.2f}s)")
    return tps


def main():
    print("=" * 70)
    print(" 投机 decode + R-SWA 叠加测试 (MLX, monkey-patch, 不修改原代码)")
    print("=" * 70)

    # 加载模型
    print("\n加载 Qwen3-VL-2B + 0.5B draft...")
    model, tokenizer = load("/Users/alexchuang/models/Qwen3-VL-2B-bf16")
    draft_model, _ = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print("  加载完成")

    # 测试 prompt (不同长度)
    prompts = [
        ("短 prompt", "Hello world"),
        ("中 prompt", "Explain how photosynthesis works in simple terms for a beginner who has never heard of biology before please be detailed"),
        ("长 prompt", "Please write a detailed essay about the history of artificial intelligence from its inception in the 1950s through the modern era of large language models. Cover the key milestones, important researchers, major breakthroughs, and the evolution of neural networks from perceptrons to transformers. Include discussion of symbolic AI, expert systems, machine learning, deep learning, and the current state of generative AI. " * 3),
    ]

    results = {}

    for label, prompt in prompts:
        print(f"\n--- {label} ---")

        # 1. baseline: 标准投机 decode (无 R-SWA)
        base_tps = bench_spec_decode(model, tokenizer, draft_model, prompt, label="baseline (标准投机)")

        # 2. R-SWA + 投机 decode
        print("  安装 R-SWA patch...")
        restore = install_rswa_patch(model)
        rswa_tps = bench_spec_decode(model, tokenizer, draft_model, prompt, label="R-SWA + 投机")
        restore()
        print("  R-SWA patch 已恢复")

        results[label] = (base_tps, rswa_tps)

    # 汇总
    print(f"\n{'='*70}")
    print(f" 汇总对比")
    print(f"{'='*70}")
    print(f"  {'Prompt':<15} {'标准投机':>10} {'R-SWA+投机':>12} {'变化':>8}")
    print(f"  {'-'*50}")
    for label, (base, rswa) in results.items():
        change = (rswa / base - 1) * 100 if base > 0 else 0
        print(f"  {label:<15} {base:>8.1f} t/s {rswa:>10.1f} t/s {change:>+6.1f}%")


if __name__ == "__main__":
    main()
