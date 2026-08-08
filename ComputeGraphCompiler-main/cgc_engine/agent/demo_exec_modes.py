import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_contexts(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def _summarize_result(d: Dict[str, Any]) -> Dict[str, Any]:
    steps = d.get("steps") or {}
    native = d.get("native") or {}
    optimized = d.get("optimized") or {}
    return {
        "ok": bool(d.get("ok", False)),
        "mode": d.get("mode"),
        "exec_mode": d.get("exec_mode"),
        "backend": d.get("backend"),
        "native_status": native.get("status"),
        "optimized_status": optimized.get("status"),
        "native_inject": native.get("inject"),
        "optimized_inject": optimized.get("inject"),
        "fullgraph_status": (steps.get("step6_fullgraph_compile") or {}).get("status"),
        "error_msg": d.get("error_msg", ""),
    }


def main() -> int:
    p = argparse.ArgumentParser(prog="demo_exec_modes")
    p.add_argument("--backend", type=str, required=True, choices=["mlx", "vllm", "llama.cpp", "llama_cpp"])
    p.add_argument("--pipeline-mode", type=str, default="llm", choices=["llm", "mlx-step67"])
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--gguf-path", type=str, default=None)
    p.add_argument("--contexts", type=str, default="128")
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--warmup-runs", type=int, default=0)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--enable-hooks", action="store_true", default=False)
    p.add_argument("--enable-ortho-kda", action="store_true", default=False)
    p.add_argument("--ortho-kda-base-dim", type=int, default=64)
    p.add_argument("--input-shape", type=int, nargs=3, default=[2, 32, 64])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modes", type=str, default="native,inject,compile")
    p.add_argument("--output-dir", type=str, default="/tmp/llm_auto_pipeline_demo_exec_modes")
    args = p.parse_args()

    from cgc_engine.agent.llm_auto_pipeline import LLMAutoPipeline

    out_dir = Path(str(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = str(args.pipeline_mode)
    contexts = [] if mode == "mlx-step67" else _parse_contexts(str(args.contexts))
    modes = [str(x).strip() for x in str(args.modes).split(",") if str(x).strip()]

    pipe = LLMAutoPipeline(output_dir=str(out_dir))
    results: List[Dict[str, Any]] = []
    for m in modes:
        report_path = out_dir / f"report_{str(args.backend).replace('.', '_')}_{m}.json"
        r = pipe.run(
            mode="mlx_step67" if mode == "mlx-step67" else "llm",
            exec_mode=str(m),
            backend="llama.cpp" if str(args.backend) in ("llama.cpp", "llama_cpp") else str(args.backend),
            model=str(args.model),
            gguf_path=str(args.gguf_path) if args.gguf_path is not None else None,
            contexts=contexts,
            input_shape=[int(x) for x in args.input_shape] if mode == "mlx-step67" else None,
            gen_tokens=0 if mode == "mlx-step67" else int(args.gen_tokens),
            warmup_runs=int(args.warmup_runs),
            runs=int(args.runs),
            enable_hooks=bool(args.enable_hooks),
            enable_ortho_kda=bool(args.enable_ortho_kda),
            ortho_kda_base_dim=int(args.ortho_kda_base_dim),
            seed=int(args.seed),
            enable_llm1=False,
            enable_skvm_verify=False,
            enable_fullgraph_aot=False,
        )
        pipe.write_report(r, str(report_path))
        results.append(json.loads(report_path.read_text(encoding="utf-8")))

    summary = [_summarize_result(x) for x in results]
    print(json.dumps({"output_dir": str(out_dir), "reports": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
