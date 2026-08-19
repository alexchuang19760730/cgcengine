#!/usr/bin/env python3

import argparse
import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_int_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    return out


def _make_prompt_tokens(tokenizer, context_len: int) -> List[int]:
    base = tokenizer.encode("hello", add_special_tokens=True)
    if not isinstance(base, list):
        base = list(base)
    if len(base) >= context_len:
        return base[:context_len]
    pad = base[-1] if len(base) > 0 else 0
    return base + [pad] * (context_len - len(base))


def _run_once(
    model,
    tokenizer,
    prompt_tokens: List[int],
    gen_tokens: int,
    sampler: Optional[Any],
) -> Dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import stream_generate

    mx.reset_peak_memory()

    t0 = time.perf_counter()
    first: Optional[Dict[str, Any]] = None
    last: Optional[Dict[str, Any]] = None

    kwargs: Dict[str, Any] = {"max_tokens": int(gen_tokens)}
    if sampler is not None:
        kwargs["sampler"] = sampler

    for resp in stream_generate(model, tokenizer, prompt_tokens, **kwargs):
        payload = {
            "prompt_tokens": int(resp.prompt_tokens),
            "prompt_tps": float(resp.prompt_tps),
            "generation_tokens": int(resp.generation_tokens),
            "generation_tps": float(resp.generation_tps),
            "peak_memory_gb": float(resp.peak_memory),
            "finish_reason": resp.finish_reason,
        }
        if first is None:
            first = payload
        last = payload

    elapsed_s = time.perf_counter() - t0
    if first is None or last is None:
        raise RuntimeError("mlx_lm.stream_generate returned no responses")

    return {
        "elapsed_s": float(elapsed_s),
        "first": first,
        "final": last,
    }


def _summarize(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    xs = [float(r[key]) for r in rows]
    if len(xs) == 0:
        return {}
    xs_sorted = sorted(xs)
    mid = len(xs_sorted) // 2
    p50 = xs_sorted[mid] if (len(xs_sorted) % 2 == 1) else 0.5 * (xs_sorted[mid - 1] + xs_sorted[mid])
    return {
        "n": len(xs),
        "mean": float(statistics.mean(xs)),
        "p50": float(p50),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _write_report(path: str, results: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Meta-Llama-3-8B-Instruct-8bit")
    p.add_argument("--contexts", default="128,512,1024,2048,4096,8192")
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--apply-cgc-hooks", action="store_true", default=True)
    p.add_argument("--no-cgc-hooks", action="store_true", default=False)
    p.add_argument("--enable-ortho-kda", action="store_true", default=False)
    p.add_argument("--ortho-kda-base-dim", type=int, default=32)
    p.add_argument("--report-path", default="/tmp/cgc_engine_benchmark_mlx_lm_context.json")
    args = p.parse_args()

    results: Dict[str, Any] = {
        "ok": True,
        "env": {
            "hostname": os.uname().nodename,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "args": {
            "model": args.model,
            "contexts": _parse_int_list(args.contexts),
            "gen_tokens": int(args.gen_tokens),
            "warmup_runs": int(args.warmup_runs),
            "runs": int(args.runs),
            "seed": int(args.seed),
            "apply_cgc_hooks": bool(args.apply_cgc_hooks and not args.no_cgc_hooks),
            "enable_ortho_kda": bool(args.enable_ortho_kda),
            "ortho_kda_base_dim": int(args.ortho_kda_base_dim),
        },
        "cases": {},
    }

    if platform.system() != "Darwin":
        results["cases"]["mlx_lm_context_bench"] = {
            "status": "SKIP",
            "reason": "MLX is macOS-only; NVIDIA/Linux is not applicable",
        }
    else:
        try:
            import mlx.core as mx
            import mlx_lm

            if args.apply_cgc_hooks and not args.no_cgc_hooks:
                from cgc_engine.cgc.mlx_ops_hook import MLXOpsHook

                hook = MLXOpsHook.get_instance()
                hook.enable_ortho_kda = bool(args.enable_ortho_kda)
                hook.ortho_kda_base_dim = int(args.ortho_kda_base_dim)
                hook.apply_hooks()

            mx.random.seed(int(args.seed))
            model, tokenizer = mlx_lm.load(args.model, lazy=True)

            sampler: Optional[Any] = None
            try:
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(temp=0.0)
            except Exception:
                sampler = None

            contexts = _parse_int_list(args.contexts)
            per_ctx: List[Dict[str, Any]] = []

            for ctx in contexts:
                ctx0 = time.perf_counter()
                try:
                    prompt_tokens = _make_prompt_tokens(tokenizer, int(ctx))

                    for _ in range(int(args.warmup_runs)):
                        _ = _run_once(
                            model,
                            tokenizer,
                            prompt_tokens,
                            min(8, int(args.gen_tokens)),
                            sampler,
                        )

                    runs: List[Dict[str, Any]] = []
                    for i in range(int(args.runs)):
                        out = _run_once(model, tokenizer, prompt_tokens, int(args.gen_tokens), sampler)
                        runs.append(
                            {
                                "run": int(i),
                                "elapsed_s": float(out["elapsed_s"]),
                                "prompt_tps": float(out["first"]["prompt_tps"]),
                                "decode_tps": float(out["final"]["generation_tps"]),
                                "peak_memory_gb": float(out["final"]["peak_memory_gb"]),
                            }
                        )

                    per_ctx.append(
                        {
                            "status": "PASS",
                            "context": int(ctx),
                            "elapsed_s": float(time.perf_counter() - ctx0),
                            "prompt_tps": _summarize(runs, "prompt_tps"),
                            "decode_tps": _summarize(runs, "decode_tps"),
                            "peak_memory_gb": _summarize(runs, "peak_memory_gb"),
                            "runs": runs,
                        }
                    )
                    _write_report(args.report_path, results)
                except Exception:
                    results["ok"] = False
                    per_ctx.append(
                        {
                            "status": "FAIL",
                            "context": int(ctx),
                            "elapsed_s": float(time.perf_counter() - ctx0),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    _write_report(args.report_path, results)
                    break

            results["cases"]["mlx_lm_context_bench"] = {
                "status": "PASS" if results["ok"] else "FAIL",
                "contexts": per_ctx,
            }
            _write_report(args.report_path, results)
        except Exception:
            results["ok"] = False
            results["cases"]["mlx_lm_context_bench"] = {
                "status": "FAIL",
                "traceback": traceback.format_exc(),
            }
            _write_report(args.report_path, results)

    _write_report(args.report_path, results)

    print(json.dumps({"ok": results["ok"], "report_path": args.report_path}, ensure_ascii=False))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
