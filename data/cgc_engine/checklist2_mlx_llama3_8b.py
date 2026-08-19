#!/usr/bin/env python3

import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Dict


def _run_case(name: str, fn, results: Dict[str, Any]) -> None:
    t0 = time.perf_counter()
    try:
        payload = fn()
        results["cases"][name] = {
            "status": "PASS",
            "elapsed_s": time.perf_counter() - t0,
            "payload": payload,
        }
    except RuntimeError as e:
        results["cases"][name] = {
            "status": "FAIL",
            "elapsed_s": time.perf_counter() - t0,
            "error": str(e),
        }
        results["ok"] = False
    except Exception:
        results["cases"][name] = {
            "status": "FAIL",
            "elapsed_s": time.perf_counter() - t0,
            "traceback": traceback.format_exc(),
        }
        results["ok"] = False


def _skip_case(name: str, reason: str, results: Dict[str, Any]) -> None:
    results["cases"][name] = {"status": "SKIP", "reason": reason}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Meta-Llama-3-8B-Instruct-8bit")
    p.add_argument("--prompt", default="Write a haiku about compilers.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--warmup-tokens", type=int, default=1)
    p.add_argument("--steady-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--apply-cgc-hooks", action="store_true", default=True)
    p.add_argument("--no-cgc-hooks", action="store_true", default=False)
    p.add_argument("--enable-ortho-kda", action="store_true", default=False)
    p.add_argument("--ortho-kda-base-dim", type=int, default=32)
    p.add_argument("--report-path", default="/tmp/cgc_engine_checklist2_mlx_llama3_8b_report.json")
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
            "max_tokens": args.max_tokens,
            "warmup_tokens": int(args.warmup_tokens),
            "steady_tokens": int(args.steady_tokens) if args.steady_tokens is not None else None,
            "seed": args.seed,
            "apply_cgc_hooks": bool(args.apply_cgc_hooks and not args.no_cgc_hooks),
            "enable_ortho_kda": bool(args.enable_ortho_kda),
            "ortho_kda_base_dim": int(args.ortho_kda_base_dim),
        },
        "cases": {},
    }

    if platform.system() != "Darwin":
        _skip_case("mlx_lm_load_generate", "MLX is macOS-only; NVIDIA/Linux is not applicable", results)
    else:
        def _run():
            import mlx.core as mx
            import mlx_lm

            if args.apply_cgc_hooks and not args.no_cgc_hooks:
                from cgc_engine.cgc.mlx_ops_hook import MLXOpsHook

                hook = MLXOpsHook.get_instance()
                hook.enable_ortho_kda = bool(args.enable_ortho_kda)
                hook.ortho_kda_base_dim = int(args.ortho_kda_base_dim)
                hook.apply_hooks()

            mx.random.seed(int(args.seed))
            steady_tokens = int(args.steady_tokens) if args.steady_tokens is not None else int(args.max_tokens)

            t_total0 = time.perf_counter()
            t0 = time.perf_counter()
            model, tokenizer = mlx_lm.load(args.model, lazy=True)
            load_s = time.perf_counter() - t0

            gen_kwargs: Dict[str, Any] = {
                "prompt": args.prompt,
                "verbose": False,
            }
            try:
                from mlx_lm.sample_utils import make_sampler

                gen_kwargs["sampler"] = make_sampler(temp=0.0)
            except Exception:
                pass

            t_warm = time.perf_counter()
            _ = mlx_lm.generate(model, tokenizer, **{**gen_kwargs, "max_tokens": int(args.warmup_tokens)})
            warmup_s = time.perf_counter() - t_warm
            e2e_first_s = time.perf_counter() - t_total0

            t_steady = time.perf_counter()
            out = mlx_lm.generate(model, tokenizer, **{**gen_kwargs, "max_tokens": steady_tokens})
            steady_s = time.perf_counter() - t_steady

            if not isinstance(out, str) or len(out.strip()) == 0:
                raise RuntimeError("mlx_lm.generate returned empty output")

            return {
                "load_s": float(load_s),
                "e2e_first_s": float(e2e_first_s),
                "warmup_s": float(warmup_s),
                "steady_s": float(steady_s),
                "steady_tokens": int(steady_tokens),
                "steady_tokens_per_sec": float(int(steady_tokens) / max(steady_s, 1e-9)),
                "output_preview": out[:200],
            }

        _run_case("mlx_lm_load_generate", _run, results)

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": results["ok"], "report_path": args.report_path}, ensure_ascii=False))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
