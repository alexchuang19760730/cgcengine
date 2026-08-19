#!/usr/bin/env python3

import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple


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


def _run_step67(
    *,
    input_shape: Tuple[int, int, int],
    backend: str,
    device: str,
) -> Dict[str, Any]:
    from cgc_engine.agent.llm_auto_pipeline import LLMAutoPipeline

    pipe = LLMAutoPipeline(output_dir="/tmp/llm_auto_pipeline_output_checklist2_mlx_step67")
    res = pipe.run(
        mode="mlx_step67",
        backend=str(backend),
        model="",
        gguf_path=None,
        contexts=[],
        input_shape=list(input_shape),
        gen_tokens=0,
        warmup_runs=1,
        runs=3,
        enable_hooks=False,
        enable_ortho_kda=False,
        ortho_kda_base_dim=0,
        seed=0,
    )
    if not res.ok:
        raise RuntimeError(res.error_msg or "pipeline failed")
    return {
        "ok": bool(res.ok),
        "native": res.native,
        "optimized": res.optimized,
        "speedup_ratio": res.speedup_ratio,
        "memory_saving_ratio": res.memory_saving_ratio,
        "steps": res.steps,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--backend", default="mlx")
    p.add_argument("--input-shape", type=int, nargs=3, default=[2, 256, 1024])
    p.add_argument("--report-path", default="/tmp/cgc_engine_checklist2_mlx_step67_report.json")
    args = p.parse_args()

    results: Dict[str, Any] = {
        "ok": True,
        "env": {
            "hostname": os.uname().nodename,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "args": {
            "device": args.device,
            "backend": args.backend,
            "input_shape": args.input_shape,
        },
        "cases": {},
    }

    if platform.system() != "Darwin":
        _skip_case("mlx_step6_step7", "MLX is macOS-only; NVIDIA/Linux is not applicable", results)
    else:
        _run_case(
            "mlx_step6_step7",
            lambda: _run_step67(input_shape=tuple(args.input_shape), backend=args.backend, device=args.device),
            results,
        )

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": results["ok"], "report_path": args.report_path}, ensure_ascii=False))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
