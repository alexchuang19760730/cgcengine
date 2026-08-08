#!/usr/bin/env python3

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mlx", choices=["mlx", "llama.cpp", "llama_cpp", "vllm"])
    p.add_argument("--model", default="mlx-community/Meta-Llama-3-8B-Instruct-8bit")
    p.add_argument("--gguf-path", default=None)
    p.add_argument("--contexts", default="128,512,1024,2048,4096,8192")
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-hooks", action="store_true", default=True)
    p.add_argument("--no-hooks", action="store_true", default=False)
    p.add_argument("--enable-ortho-kda", action="store_true", default=False)
    p.add_argument("--ortho-kda-base-dim", type=int, default=32)
    p.add_argument("--report-path", default="/tmp/cgc_engine_llm_pipeline_report.json")
    args = p.parse_args()

    results: Dict[str, Any] = {
        "ok": True,
        "env": {
            "hostname": os.uname().nodename,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "args": vars(args),
        "pipeline": {},
    }

    if platform.system() != "Darwin" and str(args.backend) in ("mlx", "mlx_lm", "mlx-lm"):
        results["ok"] = True
        results["pipeline"] = {
            "ok": True,
            "steps": {"step0_scenario": {"status": "SKIP", "reason": "MLX is macOS-only"}},
        }
    else:
        from cgc_engine.agent.llm_auto_pipeline import LLMAutoPipeline

        contexts = [int(x.strip()) for x in str(args.contexts).split(",") if x.strip()]
        enable_hooks = bool(args.enable_hooks and not args.no_hooks)

        pipe = LLMAutoPipeline(output_dir="/tmp/llm_auto_pipeline_output_checklist")
        out = pipe.run(
            backend=str(args.backend),
            model=str(args.model),
            gguf_path=str(args.gguf_path) if args.gguf_path is not None else None,
            contexts=contexts,
            gen_tokens=int(args.gen_tokens),
            warmup_runs=int(args.warmup_runs),
            runs=int(args.runs),
            enable_hooks=enable_hooks,
            enable_ortho_kda=bool(args.enable_ortho_kda),
            ortho_kda_base_dim=int(args.ortho_kda_base_dim),
            seed=int(args.seed),
        )
        results["ok"] = bool(out.ok)
        results["pipeline"] = out.__dict__

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": results["ok"], "report_path": args.report_path}, ensure_ascii=False))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
