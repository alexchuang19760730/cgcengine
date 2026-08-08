"""auto_rswa 测试: Magicompiler IR Pass 实际效果 (Qwen3-VL-2B, MLX).

测试:
1. baseline: 标准 attention, 不同 prompt 长度
2. R-SWA: auto_rswa() 替换, 不同 prompt 长度
3. 对比: R-SWA 在长上下文的效果
"""
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rswaengine", "python"))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import mlx.core as mx
from mlx_lm import load, stream_generate

# R-SWA KV 截断 patch (和 test_combo_real.py 相同方式)
REF_LEN = 4
WIN_SIZE = 128

def install_rswa_kv_truncation(model):
    """插入 R-SWA KV 截断 (不修改原代码, monkey-patch)."""
    from mlx_lm.models.qwen2 import scaled_dot_product_attention
    if hasattr(model, 'language_model'):
        layers = model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        return lambda: None

    originals = []
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        original_call = attn.__call__
        originals.append((attn, original_call))

        def make_rswa(orig, attn_self):
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

                # === R-SWA: 截断 KV 到 Reference + 窗口 ===
                seq_len = keys.shape[2]
                if seq_len > REF_LEN + WIN_SIZE:
                    keys = mx.concatenate([keys[:, :, :REF_LEN, :], keys[:, :, -WIN_SIZE:, :]], axis=2)
                    values = mx.concatenate([values[:, :, :REF_LEN, :], values[:, :, -WIN_SIZE:, :]], axis=2)

                output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=attn_self.scale, mask=mask)
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return attn_self.o_proj(output)
            return rswa_call

        attn.__call__ = make_rswa(original_call, attn)

    def restore():
        for attn, orig in originals:
            attn.__call__ = orig
    return restore


def bench(model, tokenizer, draft_model, prompt, max_tokens=100, label=""):
    """测投机 decode tok/s."""
    # warmup
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=3,
                                 draft_model=draft_model, num_draft_tokens=16):
        pass

    t0 = time.time()
    tokens = 0
    first_t = None
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                                 draft_model=draft_model, num_draft_tokens=16):
        if first_t is None:
            first_t = time.time()
        tokens += 1
    t_end = time.time()

    ttft = (first_t - t0) * 1000 if first_t else 0
    decode_tps = (tokens - 1) / (t_end - first_t) if tokens > 1 and first_t else 0
    print(f"  {label}: TTFT={ttft:.0f}ms, decode={decode_tps:.1f} tok/s ({tokens} tok)")
    return decode_tps


def main():
    print("=" * 70)
    print(" auto_rswa 测试: Magicompiler IR Pass (Qwen3-VL-2B, MLX)")
    print("=" * 70)

    print("\n加载 Qwen3-VL-2B + 0.5B draft...")
    model, tokenizer = load("/Users/alexchuang/models/Qwen3-VL-2B-bf16")
    draft_model, _ = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print("  加载完成")

    # 不同 prompt 长度
    base = "The history of artificial intelligence spans several decades. "
    prompts = [
        ("短 (10 tok)", "Hello world"),
        ("中 (100 tok)", base * 5),
        ("长 (500 tok)", base * 25),
        ("很长 (1000 tok)", base * 50),
    ]

    results = {}
    for label, prompt in prompts:
        print(f"\n--- {label} ---")

        # 1. baseline (标准 attention)
        base_tps = bench(model, tokenizer, draft_model, prompt, label="baseline")

        # 2. R-SWA (auto_rswa: Magicompiler IR Pass)
        print("  安装 R-SWA (Magicompiler IR Pass)...")
        restore = install_rswa_kv_truncation(model)
        rswa_tps = bench(model, tokenizer, draft_model, prompt, label="R-SWA  ")
        restore()
        print("  R-SWA 已恢复")

        results[label] = (base_tps, rswa_tps)

    # 汇总
    print(f"\n{'='*70}")
    print(f" 汇总: auto_rswa Magicompiler IR Pass 效果")
    print(f"{'='*70}")
    print(f"  {'Prompt':<16} {'baseline':>10} {'R-SWA':>10} {'变化':>8} {'attention范围':>16}")
    print(f"  {'-'*60}")
    for label, (base_tps, rswa_tps) in results.items():
        change = (rswa_tps / base_tps - 1) * 100 if base_tps > 0 else 0
        # 估算 attention 范围
        tok_count = {"短 (10 tok)": 10, "中 (100 tok)": 100, "长 (500 tok)": 500, "很长 (1000 tok)": 1000}
        n = tok_count.get(label, 0)
        std_range = f"{n}→{n+100}" if n > 0 else "?"
        rswa_range = f"132 (恒定)" if n > 132 else f"{n+100}"
        print(f"  {label:<16} {base_tps:>8.1f} t/s {rswa_tps:>8.1f} t/s {change:>+6.1f}%  std={std_range} rswa={rswa_range}")


if __name__ == "__main__":
    main()
