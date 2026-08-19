#!/usr/bin/env python3
"""coherence 樣本：code + prose 各生成，logits 退化檢查（NaN / 熵崩潰 / 重複）。"""
import sys, time, math, os
import mlx.core as mx

def check_logits_stats(logprob_list, token_list):
    """檢查每步 logprob 向量（[V]）：NaN、top-1 機率、熵、token 重複度。"""
    nan_steps = 0
    ent_list = []
    p1_list = []
    for lp in logprob_list:
        if lp is None:
            continue
        a = mx.array(lp).astype(mx.float32)
        if mx.any(mx.isnan(a)).item():
            nan_steps += 1
            continue
        # 熵（全 vocab softmax）
        p = mx.softmax(a, axis=-1)
        ent = float(-mx.sum(p * mx.log(p + 1e-12)).item())
        ent_list.append(ent)
        # top-1 機率
        p1_list.append(float(mx.max(p).item()))
    n = len(ent_list)
    tail = token_list[-12:]
    uniq = len(set(tail)) if tail else 0
    if n == 0:
        return {"nan": nan_steps, "steps": 0, "mean_entropy": float('nan'),
                "uniq_tail12": uniq, "tail_len": len(tail)}
    return {
        "nan": nan_steps,
        "steps": n,
        "mean_entropy": sum(ent_list) / n,
        "min_entropy": min(ent_list),
        "mean_top1_p": sum(p1_list) / n,
        "uniq_tail12": uniq,
        "tail_len": len(tail),
    }

CODE_PROMPT = "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
PROSE_PROMPT = ("Explain in three sentences why the sky is blue during the day "
                "but appears red at sunset.")

def run(model_path, prompt, label, max_tokens=60):
    from mlx_lm import load, generate
    print(f"\n{'='*70}\n[{label}] model={os.path.basename(model_path)}\n{prompt!r}\n{'-'*70}", flush=True)
    t0 = time.perf_counter()
    out = generate(model_path, prompt=prompt, max_tokens=max_tokens, verbose=False)
    wall = time.perf_counter() - t0
    print(f"output ({max_tokens} tok, {wall:.1f}s):\n{out}\n{'-'*70}", flush=True)
    return out

def run_with_logits(model_path, prompt, label, max_tokens=24):
    """自寫 decode 迴圈抓每步 (token, logprob)（檢查退化）。"""
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    import mlx.core as mx
    model, tok = load(model_path, lazy=True)
    ids = mx.array(tok.encode(prompt))
    logprob_list = []
    tokens_out = []
    t0 = time.perf_counter()
    for tok_id, lp in generate_step(ids, model, max_tokens=max_tokens):
        tokens_out.append(int(tok_id))
        logprob_list.append(lp)
    wall = time.perf_counter() - t0
    text = tok.decode(tokens_out)
    stats = check_logits_stats(logprob_list, tokens_out)
    print(f"\n[{label}] {max_tokens} tok in {wall:.1f}s  stats={stats}")
    print(f"text: {text!r}")
    return stats, text

def main():
    models = {
        "qwen36-2bit": "/Users/alexchuang/Documents/flashkv0516/models/mlx/qwen36-2bit",
        "gemma4-int2": "/Users/alexchuang/Documents/flashkv0516/models/mlx/gemma4-int2",
    }
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = {}
    for name, path in models.items():
        if only and only not in name:
            continue
        print(f"\n### {name} ###", flush=True)
        st_code, _ = run_with_logits(path, CODE_PROMPT, f"{name} code", max_tokens=24)
        st_prose, _ = run_with_logits(path, PROSE_PROMPT, f"{name} prose", max_tokens=24)
        results[name] = {"code": st_code, "prose": st_prose}
    # 摘要
    print(f"\n{'='*70}\nSUMMARY")
    for name, r in results.items():
        print(f"{name}: code={r['code']}")
        print(f"{name}: prose={r['prose']}")

if __name__ == "__main__":
    main()
