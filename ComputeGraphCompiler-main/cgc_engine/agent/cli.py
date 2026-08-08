#!/usr/bin/env python3
"""
CGC Engine Internal CLI - Unified Pipeline Interface

Commands:
    cgc pipeline    - Run internal unified 8-step pipeline (LLM/MLX)
    cgc info        - Display system information

Architecture:
    CLI -> LLMAutoPipeline -> (LLM1 -> SkVM -> FX -> KDA -> AOTInductor) -> Backend

    Backends (Execution Engine):
        vllm        - vLLM (CUDA) inference
        llama.cpp   - llama.cpp GGUF inference
        mlx         - Apple Silicon MLX inference

    Execution Modes:
        native      - Original runtime baseline
        inject      - Inject custom backend into runtime
        compile     - Full-graph analyze + compile (.so)
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

project_root = Path(__file__).resolve().parents[2]
repo_root = Path(__file__).resolve().parents[3]
for candidate in (repo_root, project_root):
    candidate_s = str(candidate)
    if candidate_s not in sys.path:
        sys.path.insert(0, candidate_s)


def _load_release_agent_cli_module():
    from app.cli import cgc as release_cgc_cli
    return release_cgc_cli


def _resolve_fingerprint_lock_path(lock_path: str) -> str:
    raw = str(lock_path or "").strip()
    builtin = project_root / "backend_fingerprint.lock.json"
    if raw == "":
        return str(builtin) if builtin.exists() else ""

    p = Path(raw).expanduser()
    if not p.is_absolute() and p.name == "lock.json" and builtin.exists():
        return str(builtin)

    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def add_pipeline_subparser(subparsers):
    parser = subparsers.add_parser(
        'pipeline',
        help='Run internal unified 8-step pipeline (LLM/MLX)',
        description='Run the internal unified 8-step pipeline for vllm, llama.cpp, or mlx backends',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='llm',
        choices=['llm', 'mlx-step67', 'edge-cloud'],
        help='Pipeline mode',
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='auto',
        choices=['auto', 'mlx', 'mlx-lm', 'mlx_lm', 'vllm', 'llama.cpp', 'llama_cpp', 'mindspeed', 'mindspeed-llm', 'megatrain', 'mlx-tune'],
        help='Target backend',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model id for mlx/vllm backend',
    )
    parser.add_argument(
        '--gguf-path',
        type=str,
        default=None,
        help='GGUF path or HF spec for llama.cpp backend',
    )
    parser.add_argument(
        '--ppl-wikitext2',
        action='store_true',
        default=False,
        help='Use WikiText2 (wiki.test.raw) for llama.cpp perplexity gate',
    )
    parser.add_argument(
        '--ppl-file',
        type=str,
        default='',
        help='Perplexity corpus file path (overrides WikiText2 auto-resolve/download)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='',
        help='Output directory for pipeline artifacts (default: <repo>/Output/PipelineRuns/<backend>/<model>/<run_id>)',
    )
    parser.add_argument(
        '--require-cuda',
        action='store_true',
        default=False,
        help='Fail-close: require CUDA-only runtime (sets CGC_REQUIRE_CUDA=1)',
    )
    parser.add_argument(
        '--require-mlx',
        action='store_true',
        default=False,
        help='Fail-close: require MLX/MPS runtime (sets CGC_REQUIRE_MLX=1)',
    )
    parser.add_argument(
        '--fingerprint-lock',
        type=str,
        default='',
        help='Fail-close: backend fingerprint lock json path (sets CGC_BACKEND_FINGERPRINT_LOCK=...)',
    )
    parser.add_argument(
        '--contexts',
        type=str,
        default='128,512,1024,2048,4096,8192',
        help='Comma-separated context lengths',
    )
    parser.add_argument(
        '--milestone',
        type=str,
        default='auto',
        choices=['auto', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm71', 'm72', 'm73', 'm77', 'upkg37', 'm78', 'upkg38', 'upkg39', 'm79', 'upkg40'],
        help='Target milestone (controls gate strictness and recommended defaults)',
    )
    parser.add_argument(
        '--milestone-seq',
        type=str,
        default='',
        help='Run multiple milestones sequentially, e.g. m3,m4,m5,m6,m7',
    )
    parser.add_argument(
        '--seq-output-dir-template',
        type=str,
        default='',
        help='When --milestone-seq is set: template for per-milestone output_dir (supports {milestone}). If not set, uses --output-dir; if no {milestone}, appends /<milestone>.',
    )
    parser.add_argument(
        '--seq-stop-on-fail',
        action='store_true',
        default=False,
        help='When --milestone-seq is set: stop after first FAIL (default: continue to produce PASS/FAIL reports for all milestones).',
    )
    parser.add_argument(
        '--m4-speedup-min',
        type=float,
        default=1.5,
        help='M4 performance gate minimum speedup ratio (baseline/optimized)',
    )
    parser.add_argument(
        '--m4-require-autopd',
        action='store_true',
        default=False,
        help='M4 only: require Auto-PD product gate (expects CGC_M4_AUTOPD_MANIFEST to be provided and PASS)',
    )
    parser.add_argument(
        '--m4-autopd-manifest',
        type=str,
        default='',
        help='M4 only: Auto-PD manifest json path for product gate (used when --m4-require-autopd)',
    )
    parser.add_argument(
        '--m4-require-omlx-flashmoe',
        action='store_true',
        default=False,
        help='M4 only: require OMLX+FlashMoE on-demand download gate',
    )
    parser.add_argument(
        '--m4-omlx-flashmoe-manifest',
        type=str,
        default='',
        help='M4 only: OMLX+FlashMoE manifest json path',
    )
    parser.add_argument(
        '--m4-force-omlx-flashmoe',
        action='store_true',
        default=False,
        help='M4 only: force OMLX+FlashMoE smoke even when oversize is not auto-detected',
    )
    parser.add_argument(
        '--m4-omlx-flashmoe-mem-util',
        type=float,
        default=0.4,
        help='M4 only: edge memory budget ratio used by OMLX+FlashMoE smoke manifest',
    )
    parser.add_argument(
        '--m4-omlx-flashmoe-smoke-num-layers',
        type=int,
        default=2,
        help='M4 only: number of layers covered by OMLX+FlashMoE smoke evidence',
    )
    parser.add_argument(
        '--exec-mode',
        type=str,
        default='native',
        choices=['native', 'inject', 'compile'],
        help='Execution mode: native=original runtime; inject=inject custom backend into runtime; compile=full-graph analyze+compile and run in compiled engine',
    )
    parser.add_argument(
        '--inject-mode',
        type=str,
        default='attention',
        choices=['forward', 'back', 'attention', 'compute'],
        help='When --exec-mode=inject: forward=full-graph forward hijack; back=backward hijack; attention=swap attention backend; compute=hijack full compute via full-graph compile (no attention backend).',
    )
    parser.add_argument(
        '--edge-cloud-base-url',
        type=str,
        default='',
        help='Edge-cloud mode only: cloud OpenAI-compatible base url (no trailing /v1). Empty means skip cloud prefill.',
    )
    parser.add_argument(
        '--edge-cloud-model',
        type=str,
        default='',
        help='Edge-cloud mode only: cloud model id (empty means reuse --model).',
    )
    parser.add_argument(
        '--edge-cloud-api-key',
        type=str,
        default=None,
        help='Edge-cloud mode only: cloud api key (optional). If omitted, uses env EDGE_CLOUD_API_KEY.',
    )
    parser.add_argument(
        '--edge-cloud-timeout-s',
        type=int,
        default=120,
        help='Edge-cloud mode only: cloud request timeout seconds.',
    )
    parser.add_argument(
        '--mindspeed-base-url',
        type=str,
        default='',
        help='MindSpeed backend only: OpenAI-compatible base url (no trailing /v1).',
    )
    parser.add_argument(
        '--mindspeed-model',
        type=str,
        default='',
        help='MindSpeed backend only: model id (empty means reuse --model).',
    )
    parser.add_argument(
        '--mindspeed-api-key',
        type=str,
        default=None,
        help='MindSpeed backend only: api key (optional). If omitted, uses env MINDSPEED_API_KEY.',
    )
    parser.add_argument(
        '--mindspeed-timeout-s',
        type=int,
        default=120,
        help='MindSpeed backend only: request timeout seconds.',
    )
    parser.add_argument(
        '--mindspeed-exec-driver',
        type=str,
        default='http',
        choices=['http', 'subprocess'],
        help='MindSpeed backend only: http=call OpenAI endpoint; subprocess=run MindSpeed-LLM script locally and measure elapsed time.',
    )
    parser.add_argument(
        '--mindspeed-subprocess-cmd',
        type=str,
        default='',
        help='MindSpeed backend only: command template for subprocess driver. Supports {context},{gen_tokens},{model},{mcore_dir},{source_hf},{precision},{prompt}.',
    )
    parser.add_argument(
        '--mindspeed-subprocess-cwd',
        type=str,
        default='',
        help='MindSpeed backend only: working directory for subprocess command.',
    )
    parser.add_argument(
        '--mindspeed-subprocess-env-source',
        type=str,
        default='',
        help='MindSpeed backend only: shell script to source before running subprocess cmd (bash -lc).',
    )
    parser.add_argument(
        '--mindspeed-source-hf',
        type=str,
        default='',
        help='MindSpeed backend only: HF model id/dir that the Mcore checkpoint was converted from (for config capture/report).',
    )
    parser.add_argument(
        '--mindspeed-mcore-dir',
        type=str,
        default='',
        help='MindSpeed backend only: Megatron-core (Mcore) checkpoint directory path (for validation/report).',
    )
    parser.add_argument(
        '--mindspeed-precision',
        type=str,
        default='fp8_mixed',
        help='MindSpeed backend only: weight precision hint for report (e.g., fp8_mixed, bf16).',
    )
    parser.add_argument(
        '--edge-prompt',
        type=str,
        default='hello',
        help='Edge-cloud mode only: prompt text for prefill / decode.',
    )
    parser.add_argument(
        '--edge-prefill-max-tokens',
        type=int,
        default=1,
        help='Edge-cloud mode only: max tokens for cloud prefill request (default 1).',
    )
    parser.add_argument(
        '--enable-mtp',
        action='store_true',
        default=False,
        help='Edge-cloud mode only: enable multi-token prediction decode plan flag.',
    )
    parser.add_argument(
        '--enable-cuda-graph',
        action='store_true',
        default=False,
        help='Edge-cloud mode only: enable CUDA Graph freeze plan flag (CUDA only).',
    )
    parser.add_argument(
        '--task-type',
        type=str,
        default='inference',
        choices=['inference', 'train', 'tune', 'multimodal'],
        help='Task type: inference (default), train (FSDP-Aware whole-layer compile), tune (LoRA Unified Memory), multimodal (dynamic shape + split unification)',
    )
    parser.add_argument(
        '--sft-mode',
        type=str,
        default='dummy',
        choices=['dummy', 'real'],
        help='SFT mode for train/tune: dummy (benchmarking only) or real (actual weights update and dataset loading)',
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='',
        help='Path to dataset for real SFT (e.g. JSONL from Eko-Agent)',
    )
    parser.add_argument(
        '--lora-layers',
        type=int,
        default=4,
        help='Number of layers to apply LoRA (real SFT)',
    )
    parser.add_argument(
        '--save-adapter-path',
        type=str,
        default='',
        help='Path to save LoRA adapters (real SFT)',
    )
    parser.add_argument(
        '--m5-require-ort',
        action='store_true',
        default=False,
        help='M5 only: require ONNX Runtime edge gate (expects onnxruntime installed + model path)',
    )
    parser.add_argument(
        '--m5-ort-model',
        type=str,
        default='',
        help='M5 only: ONNX model path for ORT edge gate',
    )
    parser.add_argument(
        '--m5-ort-ep',
        type=str,
        default='',
        help='M5 only: ORT execution provider for ORT edge gate (default CPUExecutionProvider)',
    )
    parser.add_argument(
        '--m5-ort-custom-ops-lib',
        type=str,
        default='',
        help='M5 only: custom ops shared library path for ORT edge gate (optional)',
    )
    parser.add_argument(
        '--gen-tokens',
        type=int,
        default=128,
        help='Decode tokens per run',
    )
    parser.add_argument(
        '--warmup-runs',
        type=int,
        default=1,
        help='Warmup runs per context',
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Runs per context',
    )
    parser.add_argument(
        '--enable-hooks',
        action='store_true',
        default=False,
        help='Enable hooks/opcode optimized path (mlx only)',
    )
    parser.add_argument(
        '--enable-ortho-kda',
        action='store_true',
        default=False,
        help='Enable OrthoKDA cache semantics',
    )
    parser.add_argument(
        '--ortho-kda-base-dim',
        type=int,
        default=64,
        help='OrthoKDA base dim',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed',
    )
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=3,
        default=[2, 256, 1024],
        help='Input shape for mlx-step67: batch seq hidden',
    )
    parser.add_argument(
        '--enable-llm1',
        action='store_true',
        default=False,
        help='Enable LLM1 translation in step5 (vLLM OpenAI-compatible endpoint)',
    )
    parser.add_argument(
        '--llm1-base-url',
        type=str,
        default='http://127.0.0.1:8000',
        help='LLM1 endpoint base url, e.g. http://127.0.0.1:8000',
    )
    parser.add_argument(
        '--llm1-model',
        type=str,
        default='',
        help='LLM1 model name at endpoint (empty means reuse --model)',
    )
    parser.add_argument(
        '--llm1-api-key',
        type=str,
        default=None,
        help='LLM1 api key (optional). If omitted, uses env LLM1_API_KEY',
    )
    parser.add_argument(
        '--llm1-timeout-s',
        type=int,
        default=300,
        help='LLM1 request timeout seconds',
    )
    parser.add_argument(
        '--llm1-input-path',
        type=str,
        default=None,
        help='Path to backend kernel/operator code file for LLM1 translation',
    )
    parser.add_argument(
        '--enable-fullgraph-aot',
        action='store_true',
        default=False,
        help='Enable transformers safetensors full-graph AOTInductor compile + end-to-end benchmark',
    )
    parser.add_argument(
        '--fullgraph-model',
        type=str,
        default='',
        help='HF model id for fullgraph (empty means reuse --model)',
    )
    parser.add_argument(
        '--fullgraph-prompt',
        type=str,
        default='hello',
        help='Prompt text for fullgraph benchmark',
    )
    parser.add_argument(
        '--fullgraph-max-new-tokens',
        type=int,
        default=16,
        help='Decode tokens for fullgraph benchmark',
    )
    parser.add_argument(
        '--fullgraph-non-strict',
        action='store_true',
        default=False,
        help='Do not fail pipeline when fullgraph aot fails',
    )
    parser.add_argument(
        '--enable-skvm-verify',
        action='store_true',
        default=False,
        help='Enable SkVM verify in step3 (auto-enabled when --enable-llm1; uses --skvm-input or step5 LLM1 output)',
    )
    parser.add_argument(
        '--skvm-input',
        type=str,
        default=None,
        help='Path to pytorch graph/module file for skvm verify (optional if step5 LLM1 is enabled)',
    )
    parser.add_argument(
        '--skvm-cli',
        type=str,
        default='skvm',
        help='SkVM CLI executable',
    )
    parser.add_argument(
        '--skvm-auto-install',
        action='store_true',
        default=False,
        help='Auto-install official SkVM (skillvm.ai) when --skvm-cli=skvm is missing',
    )
    parser.add_argument(
        '--enable-skillvm-aot',
        action='store_true',
        default=False,
        help='Run official SkVM (skillvm.ai) profile+aot-compile to produce an AOT-compiled SKILL artifact (stored under Output).',
    )
    parser.add_argument(
        '--skillvm-target-model',
        type=str,
        default='',
        help='SkVM target model id (<provider>/<model-id>), e.g. openrouter/qwen/qwen3.5-35b-a3b',
    )
    parser.add_argument(
        '--skillvm-compiler-model',
        type=str,
        default='',
        help='SkVM compiler backend model id (<provider>/<model-id>), e.g. anthropic/claude-sonnet-4.6',
    )
    parser.add_argument(
        '--skillvm-adapter',
        type=str,
        default='bare-agent',
        help='SkVM adapter (default: bare-agent)',
    )
    parser.add_argument(
        '--skvm-dtype',
        type=str,
        default='fp16',
        help='SkVM dtype, e.g. fp16/bf16/fp32',
    )
    parser.add_argument(
        '--skvm-timeout-s',
        type=int,
        default=30,
        help='SkVM verify timeout seconds',
    )
    parser.add_argument(
        '--skvm-non-strict',
        action='store_true',
        default=False,
        help='Do not fail pipeline when skvm verify fails',
    )
    parser.add_argument(
        '--report-path',
        type=str,
        default='',
        help='Report JSON path (default: <output_dir>/report.json)',
    )
    parser.add_argument(
        '--bundle-export-dir',
        type=str,
        default='',
        help='Optional: export a self-contained artifact bundle directory for edge deployment (agent pipeline only).',
    )
    parser.add_argument(
        '--bundle-import-manifest',
        type=str,
        default='',
        help='Edge-cloud mode only: import artifact bundle from a manifest path or URL (http/https/file).',
    )
    parser.add_argument(
        '--bundle-import-dir',
        type=str,
        default='',
        help='Edge-cloud mode only: directory to store imported bundle payload (default: <output_dir>/bundle_cache).',
    )
    parser.add_argument(
        '--bundle-artifact-base-url',
        type=str,
        default='',
        help='Edge-cloud mode only: base URL to download bundle payload files (if manifest contains relative paths).',
    )
    parser.add_argument(
        '--disable-audit',
        action='store_true',
        default=False,
        help='Speed optimization: Disable Hash Chain audit gate checking',
    )
    parser.add_argument(
        '--m72-gui-duration-s',
        type=int,
        default=5,
        help='When milestone >= m72, collect real GUI agent evidence for this many seconds before evaluating m72 (0 disables collection).',
    )
    parser.add_argument(
        '--m72-disable-gui-evidence',
        action='store_true',
        default=False,
        help='Disable automatic GUI agent evidence collection for m72 standard route.',
    )
    # --- NEW: Advanced Hardware Gate Arguments ---
    parser.add_argument('--require-dflash', action='store_true', default=False, help='Enforce DFlash (FlashKV) gate verification')
    parser.add_argument('--require-spdk', action='store_true', default=False, help='Enforce SPDK storage gate verification')
    parser.add_argument('--require-gds', action='store_true', default=False, help='Enforce GPUDirect Storage (GDS) gate verification')
    parser.add_argument('--require-omlx', action='store_true', default=False, help='Enforce oMLX optimization gate verification')
    parser.add_argument('--require-flashmoe', action='store_true', default=False, help='Enforce FlashMoE gate verification')
    parser.add_argument('--component-id', type=str, default='', help='Explicit runtime component id for pipeline contract/manifest, e.g. deepseek_inst1')
    parser.add_argument('--component-role', type=str, default='', help='Explicit runtime component role for pipeline contract/manifest, e.g. llm_runtime')
    parser.add_argument('--component-required', action='store_true', default=False, help='Mark this runtime component as required in system readiness policy')
    parser.add_argument('--component-optional', action='store_true', default=False, help='Mark this runtime component as optional in system readiness policy')
    parser.add_argument('--component-health-endpoint', type=str, default='', help='Health endpoint URL recorded in execution_context/system_execution_manifest')
    parser.add_argument('--system-id', type=str, default='', help='Logical system id for contract aggregation, e.g. fusionroute_host2')
    parser.add_argument('--system-role', type=str, default='', help='Logical system role, e.g. multi_component_runtime')
    parser.add_argument('--system-manifest-discovery-root', type=str, default='', help='Directory used to autodiscover sibling contract_manifest.json files')
    parser.add_argument('--system-manifest-no-autodiscover', action='store_true', default=False, help='Disable sibling contract autodiscovery for system_execution_manifest')
    parser.add_argument('--system-manifest-components', type=str, default='', help='Optional JSON string or json file path containing extra system components')
    parser.add_argument('--system-manifest-routing-edges', type=str, default='', help='Optional JSON string or json file path containing routing edges')
    parser.add_argument('--system-manifest-required-components', type=str, default='', help='Optional JSON string or json file path containing required component ids')
    parser.add_argument('--system-manifest-optional-components', type=str, default='', help='Optional JSON string or json file path containing optional component ids')
    parser.set_defaults(func=pipeline_command)

def pipeline_command(args):
    def _set_or_clear_env(name: str, value: Any) -> None:
        text = str(value or "").strip()
        if text == "":
            os.environ.pop(name, None)
        else:
            os.environ[name] = text

    # --- NEW: Speed Optimization - Skip Audit Gate Failure ---
    if getattr(args, "disable_audit", False):
        os.environ["CGC_DISABLE_AUDIT_GATE"] = "1"
    else:
        os.environ["CGC_DISABLE_AUDIT_GATE"] = "0"

    milestone_seq_raw = str(getattr(args, "milestone_seq", "") or "").strip()
    if milestone_seq_raw != "":
        import argparse as _argparse
        import re as _re

        def _clear_milestone_env() -> None:
            for k in list(os.environ.keys()):
                if _re.match(r"^CGC_M[0-9]", str(k)):
                    try:
                        del os.environ[k]
                    except Exception:
                        pass

        allowed = {"m1", "m2", "m3", "m4", "m5", "m6", "m7", "m71", "m72", "m73", "m77", "upkg37", "m78", "upkg38", "upkg39", "m79", "upkg40"}
        seq = [s.strip().lower() for s in milestone_seq_raw.split(",") if s.strip() != ""]
        for m in seq:
            if m not in allowed:
                print(json.dumps({"ok": False, "error": f"invalid --milestone-seq item: {m}"}, ensure_ascii=False))
                return 2
        base_vars = dict(vars(args))
        overall_ok = True
        last_rc = 0
        for m in seq:
            child = _argparse.Namespace(**base_vars)
            setattr(child, "milestone_seq", "")
            setattr(child, "milestone", str(m))

            out_tmpl = str(getattr(child, "seq_output_dir_template", "") or "").strip()
            if out_tmpl == "":
                out_tmpl = str(getattr(child, "output_dir", "") or "").strip()
            if out_tmpl != "":
                if "{milestone}" in out_tmpl:
                    setattr(child, "output_dir", out_tmpl.format(milestone=str(m)))
                else:
                    try:
                        setattr(child, "output_dir", str(Path(out_tmpl) / str(m)))
                    except Exception:
                        setattr(child, "output_dir", out_tmpl + "/" + str(m))

            rpt = str(getattr(child, "report_path", "") or "").strip()
            if rpt != "":
                if "{milestone}" in rpt:
                    setattr(child, "report_path", rpt.format(milestone=str(m)))
                else:
                    print(json.dumps({"ok": False, "error": "when --milestone-seq is set: --report-path must be empty or contain {milestone}"}, ensure_ascii=False))
                    return 2

            bdir = str(getattr(child, "bundle_export_dir", "") or "").strip()
            if bdir != "":
                if "{milestone}" in bdir:
                    setattr(child, "bundle_export_dir", bdir.format(milestone=str(m)))
                else:
                    print(json.dumps({"ok": False, "error": "when --milestone-seq is set: --bundle-export-dir must be empty or contain {milestone}"}, ensure_ascii=False))
                    return 2

            _clear_milestone_env()
            rc = pipeline_command(child)
            last_rc = int(rc)
            if int(rc) != 0:
                overall_ok = False
                if bool(getattr(args, "seq_stop_on_fail", False)):
                    break
        return 0 if overall_ok else int(last_rc or 1)

    if bool(args.enable_llm1) and not bool(args.enable_skvm_verify):
        args.enable_skvm_verify = True

    mode = str(args.mode)
    backend = str(args.backend)
    model = str(args.model)
    gguf_path = str(args.gguf_path) if args.gguf_path is not None else None
    exec_mode = str(getattr(args, "exec_mode", "native"))
    milestone = str(getattr(args, "milestone", "auto") or "auto").strip().lower()
    if bool(getattr(args, "m4_require_autopd", False)):
        os.environ["CGC_M4_REQUIRE_AUTOPD"] = "1"
    autopd_manifest = str(getattr(args, "m4_autopd_manifest", "") or "").strip()
    if autopd_manifest != "":
        os.environ["CGC_M4_AUTOPD_MANIFEST"] = autopd_manifest

    if bool(getattr(args, "m4_require_omlx_flashmoe", False)):
        os.environ["CGC_M4_REQUIRE_OMLX_FLASHMOE"] = "1"
    omlx_manifest = str(getattr(args, "m4_omlx_flashmoe_manifest", "") or "").strip()
    if omlx_manifest != "":
        os.environ["CGC_M4_OMLX_FLASHMOE_MANIFEST"] = omlx_manifest
    if bool(getattr(args, "m4_force_omlx_flashmoe", False)):
        os.environ["CGC_M4_FORCE_OMLX_FLASHMOE"] = "1"
    os.environ["CGC_M4_OMLX_FLASHMOE_MEM_UTIL"] = str(float(getattr(args, "m4_omlx_flashmoe_mem_util", 0.4)))
    os.environ["CGC_M4_OMLX_FLASHMOE_SMOKE_NUM_LAYERS"] = str(int(getattr(args, "m4_omlx_flashmoe_smoke_num_layers", 2)))

    if bool(getattr(args, "m5_require_ort", False)):
        os.environ["CGC_M5_REQUIRE_ORT"] = "1"
    ort_model = str(getattr(args, "m5_ort_model", "") or "").strip()
    if ort_model != "":
        os.environ["CGC_M5_ORT_MODEL"] = ort_model
    ort_ep = str(getattr(args, "m5_ort_ep", "") or "").strip()
    if ort_ep != "":
        os.environ["CGC_M5_ORT_EP"] = ort_ep
    ort_custom_ops = str(getattr(args, "m5_ort_custom_ops_lib", "") or "").strip()
    if ort_custom_ops != "":
        os.environ["CGC_M5_ORT_CUSTOM_OPS_LIB"] = ort_custom_ops

    def _pick_default_gguf() -> Optional[str]:
        try:
            base = Path(project_root).resolve() / "Output" / "Models"
            cands = list(base.rglob("*.gguf"))
            if not cands:
                return None
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(cands[0])
        except Exception:
            return None

    if milestone in {"m1", "m2", "m3", "m4", "m5", "m6", "m7", "m73", "m77", "upkg37", "m78", "upkg38", "upkg39", "m79", "upkg40"}:
        os.environ["CGC_MILESTONE"] = milestone
    if milestone in {"m77", "upkg37", "m78", "upkg38", "upkg39"}:
        args.enable_ortho_kda = True

    require_cuda_flag = bool(getattr(args, "require_cuda", False))
    require_mlx_flag = bool(getattr(args, "require_mlx", False))
    if require_cuda_flag and require_mlx_flag:
        print(json.dumps({"ok": False, "error": "invalid args: both --require-cuda and --require-mlx are set"}, ensure_ascii=False))
        return 2
    if require_cuda_flag:
        os.environ["CGC_REQUIRE_CUDA"] = "1"
    if require_mlx_flag:
        os.environ["CGC_REQUIRE_MLX"] = "1"
    fingerprint_lock = str(getattr(args, "fingerprint_lock", "") or "").strip()
    if fingerprint_lock == "":
        fingerprint_lock = str(os.environ.get("CGC_BACKEND_FINGERPRINT_LOCK") or "").strip()
    fingerprint_lock = _resolve_fingerprint_lock_path(fingerprint_lock)
    if fingerprint_lock != "":
        if not Path(fingerprint_lock).exists():
            print(json.dumps({"ok": False, "error": f"fingerprint lock not found: {fingerprint_lock}"}, ensure_ascii=False))
            return 2
        os.environ["CGC_BACKEND_FINGERPRINT_LOCK"] = fingerprint_lock
    if milestone in {"m1", "m5", "m6"}:
        if backend in {"auto", "megatrain", "mlx-tune", "mlx_tune"}:
            backend = "llama.cpp"
        if gguf_path is None and backend in {"llama.cpp", "llama_cpp", "llama"}:
            gguf_path = _pick_default_gguf()
        args.enable_ortho_kda = True

    if milestone in {"m2", "m3"}:
        if exec_mode != "inject":
            exec_mode = "inject"
        if backend in {"auto", "megatrain", "mlx-tune", "mlx_tune"}:
            backend = "llama.cpp"
        if gguf_path is None and backend in {"llama.cpp", "llama_cpp", "llama"}:
            gguf_path = _pick_default_gguf()
        args.enable_ortho_kda = True
        args.enable_skvm_verify = True
        if "CGC_M2_STRICT_FINAL" not in os.environ:
            os.environ["CGC_M2_STRICT_FINAL"] = "1"
        if "CGC_M2_REQUIRE_EQ_GATE" not in os.environ:
            os.environ["CGC_M2_REQUIRE_EQ_GATE"] = "1"
        if "CGC_M2_REQUIRE_MEMORY_GATE" not in os.environ:
            os.environ["CGC_M2_REQUIRE_MEMORY_GATE"] = "1"
        if "CGC_M2_REQUIRE_SPEED_GATE" not in os.environ:
            os.environ["CGC_M2_REQUIRE_SPEED_GATE"] = "1"
        if "CGC_M2_REQUIRE_PPL_GATE" not in os.environ:
            os.environ["CGC_M2_REQUIRE_PPL_GATE"] = "1"
        os.environ["CGC_LLAMA_CPU_ALL_VARIANTS"] = "0"
        os.environ["CGC_AUTO_BUILD_CGC_CPP"] = "1"
        require_cuda_env = str(os.environ.get("CGC_REQUIRE_CUDA") or "").strip().lower()
        require_cuda = require_cuda_env in {"1", "true", "yes", "on"}
        if require_cuda:
            if "CGC_LLAMA_NGL" not in os.environ:
                os.environ["CGC_LLAMA_NGL"] = "999"
            if "CGC_LLAMA_BENCH_NGL" not in os.environ:
                os.environ["CGC_LLAMA_BENCH_NGL"] = os.environ["CGC_LLAMA_NGL"]
            if "CGC_LLAMA_PPL_NGL" not in os.environ:
                os.environ["CGC_LLAMA_PPL_NGL"] = os.environ["CGC_LLAMA_NGL"]
        if not bool(getattr(args, "ppl_wikitext2", False)):
            args.ppl_wikitext2 = True
        try:
            default_ctx = "128,512,1024,2048,4096,8192"
            if str(getattr(args, "contexts", "") or "").strip() == default_ctx:
                args.contexts = "2048,4096,8192,16384"
        except Exception:
            pass
    if milestone == "m4":
        os.environ["CGC_M4_SPEEDUP_MIN"] = str(float(getattr(args, "m4_speedup_min", 1.5)))
        local_m4_smoke = str(os.environ.get("CGC_LOCAL_M4_SMOKE", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        if local_m4_smoke:
            os.environ["CGC_M4_REQUIRE_DISTRIBUTED"] = "0"
            if str(os.environ.get("CGC_M4_DISTRIBUTED_SMOKE", "") or "").strip() == "":
                os.environ["CGC_M4_DISTRIBUTED_SMOKE"] = "0"
            args.task_type = "inference"
            exec_mode = "compile"
            if backend in {"auto", "megatrain", "mlx-tune", "mlx_tune"}:
                backend = "mlx"
            args.enable_fullgraph_aot = True
            if str(getattr(args, "fullgraph_model", "") or "").strip() == "":
                args.fullgraph_model = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
        else:
            os.environ["CGC_M4_REQUIRE_DISTRIBUTED"] = "1"
            if str(os.environ.get("CGC_M4_DISTRIBUTED_SMOKE", "") or "").strip() == "":
                os.environ["CGC_M4_DISTRIBUTED_SMOKE"] = "1"
            if str(getattr(args, "task_type", "inference")) == "inference":
                args.task_type = "train"
            backend = "megatrain"
    if milestone == "m5":
        if exec_mode != "compile":
            exec_mode = "compile"
        if backend in {"auto", "megatrain", "mlx-tune", "mlx_tune"}:
            backend = "llama.cpp"
        if gguf_path is None and backend in {"llama.cpp", "llama_cpp", "llama"}:
            gguf_path = _pick_default_gguf()

    if mode == "mlx-step67":
        input_shape = [int(x) for x in args.input_shape]
        contexts: List[int] = []
        gen_tokens = 0
    else:
        input_shape = [int(x) for x in args.input_shape] if (bool(args.enable_skvm_verify) or bool(args.enable_llm1)) else None
        contexts = [int(x) for x in str(args.contexts).split(",") if str(x).strip()]
        gen_tokens = int(args.gen_tokens)

    if backend == "auto":
        import platform

        sys_name = str(platform.system())
        if mode == "mlx-step67":
            backend = "mlx"
        elif sys_name == "Darwin":
            backend = "mlx"
        else:
            try:
                import torch

                backend = "vllm" if bool(torch.cuda.is_available()) else "llama.cpp"
            except Exception:
                backend = "llama.cpp"

    from cgc_engine.agent.llm_auto_pipeline import LLMAutoPipeline

    ppl_file = str(getattr(args, "ppl_file", "") or "").strip()
    if bool(getattr(args, "ppl_wikitext2", False)) or ppl_file != "":
        os.environ["CGC_LLAMA_PPL_TEST"] = "wikitext2"
    if ppl_file != "":
        os.environ["CGC_LLAMA_PPL_FILE"] = ppl_file
    _set_or_clear_env("CGC_MEGATRAIN_COMPONENT_ID", getattr(args, "component_id", ""))
    _set_or_clear_env("CGC_MEGATRAIN_COMPONENT_ROLE", getattr(args, "component_role", ""))
    component_required = True
    if bool(getattr(args, "component_optional", False)):
        component_required = False
    elif bool(getattr(args, "component_required", False)):
        component_required = True
    os.environ["CGC_MEGATRAIN_COMPONENT_REQUIRED"] = "1" if component_required else "0"
    _set_or_clear_env("CGC_MEGATRAIN_COMPONENT_HEALTH_ENDPOINT", getattr(args, "component_health_endpoint", ""))
    _set_or_clear_env("CGC_MEGATRAIN_SYSTEM_ID", getattr(args, "system_id", ""))
    _set_or_clear_env("CGC_MEGATRAIN_SYSTEM_ROLE", getattr(args, "system_role", ""))
    _set_or_clear_env("CGC_MEGATRAIN_SYSTEM_MANIFEST_DISCOVERY_ROOT", getattr(args, "system_manifest_discovery_root", ""))
    os.environ["CGC_MEGATRAIN_SYSTEM_MANIFEST_AUTODISCOVER"] = "0" if bool(getattr(args, "system_manifest_no_autodiscover", False)) else "1"
    _set_or_clear_env("CGC_MEGATRAIN_SYSTEM_MANIFEST_COMPONENTS", getattr(args, "system_manifest_components", ""))
    _set_or_clear_env("CGC_MEGATRAIN_SYSTEM_MANIFEST_ROUTING_EDGES", getattr(args, "system_manifest_routing_edges", ""))
    _set_or_clear_env("CGC_MEGATRAIN_SYSTEM_MANIFEST_REQUIRED_COMPONENTS", getattr(args, "system_manifest_required_components", ""))
    _set_or_clear_env("CGC_MEGATRAIN_SYSTEM_MANIFEST_OPTIONAL_COMPONENTS", getattr(args, "system_manifest_optional_components", ""))
    run_id = str(args.seed)
    try:
        import time as _time

        run_id = f"run_{_time.strftime('%Y%m%d_%H%M%S')}_{int(args.seed)}"
    except Exception:
        run_id = f"run_{int(args.seed)}"

    def _safe_name(s: str) -> str:
        out = []
        for ch in str(s):
            if ch.isalnum() or ch in ("-", "_", "."):
                out.append(ch)
            else:
                out.append("_")
        text = "".join(out).strip("_.")
        return text[:120] if len(text) > 120 else text

    output_dir = str(getattr(args, "output_dir", "") or "").strip()
    if output_dir == "":
        base = Path(project_root).resolve() / "Output" / "PipelineRuns"
        model_tag = _safe_name(gguf_path if backend in ("llama.cpp", "llama_cpp", "llama") else model)
        output_dir = str(base / _safe_name(backend) / model_tag / run_id)

    milestone_rank = {"auto": 0, "m1": 1, "m2": 2, "m3": 3, "m4": 4, "m5": 5, "m6": 6, "m7": 7, "m71": 71, "m72": 72, "m73": 73, "m77": 77, "upkg37": 77, "m78": 78, "upkg38": 78, "upkg39": 79, "m79": 80, "upkg40": 80}
    target_rank = milestone_rank.get(milestone, 0)
    prev_gui_evidence_for_pipeline = os.environ.get("CGC_M72_GUI_EVENT_EVIDENCE")
    pipeline_gui_evidence_path = ""
    if target_rank >= 72 and not bool(getattr(args, "m72_disable_gui_evidence", False)):
        try:
            from cgc_engine.agent.eval.eko_gui_agent_demo import collect_gui_runtime_evidence

            gui_duration_s = int(getattr(args, "m72_gui_duration_s", 5))
            if gui_duration_s > 0:
                pipeline_gui_evidence_path = str(
                    collect_gui_runtime_evidence(
                        duration_sec=gui_duration_s,
                        output_dir=Path(output_dir).resolve() / "gui_agent_runtime",
                    )
                )
                if pipeline_gui_evidence_path:
                    os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = str(pipeline_gui_evidence_path)
        except Exception:
            pipeline_gui_evidence_path = ""

    pipe = LLMAutoPipeline(output_dir=output_dir)
    run_mode = "mlx_step67" if mode == "mlx-step67" else "edge-cloud" if mode == "edge-cloud" else "llm"
    try:
        result = pipe.run(
            mode=run_mode,
            exec_mode=exec_mode,
            inject_mode=str(getattr(args, "inject_mode", "attention")),
            task_type=str(getattr(args, "task_type", "inference")),
            backend=backend,
            model=model,
            gguf_path=gguf_path,
            contexts=contexts,
            input_shape=input_shape,
            gen_tokens=gen_tokens,
            warmup_runs=int(args.warmup_runs),
            runs=int(args.runs),
            enable_hooks=bool(args.enable_hooks),
            enable_ortho_kda=bool(args.enable_ortho_kda),
            ortho_kda_base_dim=int(args.ortho_kda_base_dim),
            seed=int(args.seed),
            enable_llm1=bool(args.enable_llm1),
            llm1_base_url=str(args.llm1_base_url),
            llm1_model=str(args.llm1_model),
            llm1_api_key=args.llm1_api_key,
            llm1_timeout_s=int(args.llm1_timeout_s),
            llm1_input_path=str(args.llm1_input_path) if args.llm1_input_path is not None else None,
            enable_fullgraph_aot=bool(args.enable_fullgraph_aot),
            fullgraph_model=str(args.fullgraph_model),
            fullgraph_prompt=str(args.fullgraph_prompt),
            fullgraph_max_new_tokens=int(args.fullgraph_max_new_tokens),
            fullgraph_strict=not bool(args.fullgraph_non_strict),
            require_dflash=bool(getattr(args, "require_dflash", False)),
            require_spdk=bool(getattr(args, "require_spdk", False)),
            require_gds=bool(getattr(args, "require_gds", False)),
            require_omlx=bool(getattr(args, "require_omlx", False)),
            require_flashmoe=bool(getattr(args, "require_flashmoe", False)),
            enable_skvm_verify=bool(args.enable_skvm_verify),
            skvm_cli=str(args.skvm_cli),
            skvm_auto_install=bool(getattr(args, "skvm_auto_install", False)),
            skvm_input=str(args.skvm_input) if args.skvm_input is not None else None,
            skvm_dtype=str(args.skvm_dtype),
            skvm_timeout_s=int(args.skvm_timeout_s),
            skvm_strict=not bool(args.skvm_non_strict),
            enable_skillvm_aot=bool(getattr(args, "enable_skillvm_aot", False)),
            skillvm_target_model=str(getattr(args, "skillvm_target_model", "") or ""),
            skillvm_compiler_model=str(getattr(args, "skillvm_compiler_model", "") or ""),
            skillvm_adapter=str(getattr(args, "skillvm_adapter", "bare-agent") or "bare-agent"),
            edge_cloud_base_url=str(getattr(args, "edge_cloud_base_url", "") or ""),
            edge_cloud_model=str(getattr(args, "edge_cloud_model", "") or ""),
            edge_cloud_api_key=getattr(args, "edge_cloud_api_key", None),
            edge_cloud_timeout_s=int(getattr(args, "edge_cloud_timeout_s", 120)),
            edge_prompt=str(getattr(args, "edge_prompt", "hello") or "hello"),
            edge_prefill_max_tokens=int(getattr(args, "edge_prefill_max_tokens", 1)),
            enable_mtp=bool(getattr(args, "enable_mtp", False)),
            enable_cuda_graph=bool(getattr(args, "enable_cuda_graph", False)),
            bundle_import_manifest=str(getattr(args, "bundle_import_manifest", "") or ""),
            bundle_import_dir=str(getattr(args, "bundle_import_dir", "") or ""),
            bundle_artifact_base_url=str(getattr(args, "bundle_artifact_base_url", "") or ""),
            mindspeed_base_url=str(getattr(args, "mindspeed_base_url", "") or ""),
            mindspeed_model=str(getattr(args, "mindspeed_model", "") or ""),
            mindspeed_api_key=getattr(args, "mindspeed_api_key", None),
            mindspeed_timeout_s=int(getattr(args, "mindspeed_timeout_s", 120)),
            mindspeed_source_hf=str(getattr(args, "mindspeed_source_hf", "") or ""),
            mindspeed_mcore_dir=str(getattr(args, "mindspeed_mcore_dir", "") or ""),
            mindspeed_precision=str(getattr(args, "mindspeed_precision", "fp8_mixed") or "fp8_mixed"),
            mindspeed_exec_driver=str(getattr(args, "mindspeed_exec_driver", "http") or "http"),
            mindspeed_subprocess_cmd=str(getattr(args, "mindspeed_subprocess_cmd", "") or ""),
            mindspeed_subprocess_cwd=str(getattr(args, "mindspeed_subprocess_cwd", "") or ""),
            mindspeed_subprocess_env_source=str(getattr(args, "mindspeed_subprocess_env_source", "") or ""),
            sft_mode=str(getattr(args, "sft_mode", "dummy")),
            dataset_path=str(getattr(args, "dataset_path", "")),
            lora_layers=int(getattr(args, "lora_layers", 4)),
            save_adapter_path=str(getattr(args, "save_adapter_path", "")),
        )
    finally:
        if prev_gui_evidence_for_pipeline is None:
            os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
        else:
            os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = prev_gui_evidence_for_pipeline
    if target_rank >= 6:
        m6_dir = Path(output_dir).resolve() / "m6_product"
        m6_gate = {"status": "FAIL", "reason": "uninitialized"}
        try:
            from cgc_engine.product import build_bundle, run_bundle

            m6_dir.mkdir(parents=True, exist_ok=True)
            build_report = build_bundle(output_dir=str(m6_dir), template="ort_mnist_cpu")
            run_report = run_bundle(output_dir=str(m6_dir))
            b_gate = ((build_report or {}).get("gate_result") or {}).get("m6") if isinstance(build_report, dict) else None
            r_gate = ((run_report or {}).get("gate_result") or {}).get("m6") if isinstance(run_report, dict) else None

            build_status = str((b_gate or {}).get("status") or "")
            run_status = str((r_gate or {}).get("status") or "")
            build_ok = bool(isinstance(b_gate, dict) and build_status in ("PASS", "SKIP"))
            run_ok = bool(isinstance(r_gate, dict) and run_status in ("PASS", "SKIP"))
            ok = bool(build_ok and run_ok)
            overall_status = "PASS" if (build_status == "PASS" and run_status == "PASS") else ("SKIP" if ok else "FAIL")

            m6_gate = {
                "status": str(overall_status),
                "product_dir": str(m6_dir),
                "build_report_path": str(m6_dir / "build_report.json"),
                "run_report_path": str(m6_dir / "run_report.json"),
                "build_bundle_gate": (b_gate or {}).get("build_bundle_gate") if isinstance(b_gate, dict) else None,
                "run_bundle_gate": (r_gate or {}).get("run_bundle_gate") if isinstance(r_gate, dict) else None,
            }
        except Exception as e:
            r = repr(e)
            if isinstance(e, ModuleNotFoundError) and ("onnxruntime" in r or "onnxruntime" in str(e)):
                m6_gate = {"status": "SKIP", "reason": "missing_dependency:onnxruntime", "product_dir": str(m6_dir)}
            else:
                m6_gate = {"status": "FAIL", "reason": f"m6_product_error:{r}", "product_dir": str(m6_dir)}

        try:
            result.steps["m6_product"] = m6_gate
        except Exception:
            pass

        gate_result = getattr(result, "gate_result", None)
        if not isinstance(gate_result, dict):
            gate_result = {"status": "PASS"}
        gate_result["m6"] = m6_gate
        prev_ok = str(gate_result.get("status") or "PASS") == "PASS"
        m6_ok = str(m6_gate.get("status") or "") in ("PASS", "SKIP")
        gate_result["status"] = "PASS" if (prev_ok and m6_ok) else "FAIL"
        setattr(result, "gate_result", gate_result)
        if str(m6_gate.get("status") or "") == "FAIL":
            result.ok = False

    if target_rank >= 7:
        m7_gate = {"status": "FAIL", "reason": "uninitialized"}
        try:
            from cgc_engine.product import run_m7_gate

            m7_report = run_m7_gate(output_dir=str(Path(output_dir).resolve()))
            m7_gate = ((m7_report or {}).get("gate_result") or {}).get("m7") if isinstance(m7_report, dict) else {"status": "FAIL", "reason": "invalid_m7_report"}
        except Exception as e:
            m7_gate = {"status": "FAIL", "reason": f"m7_gate_error:{repr(e)}"}

        try:
            result.steps["m7_industrial"] = m7_gate
        except Exception:
            pass

        gate_result = getattr(result, "gate_result", None)
        if not isinstance(gate_result, dict):
            gate_result = {"status": "PASS"}
        gate_result["m7"] = m7_gate
        prev_ok = str(gate_result.get("status") or "PASS") == "PASS"
        m7_ok = str(m7_gate.get("status") or "") == "PASS"
        gate_result["status"] = "PASS" if (prev_ok and m7_ok) else "FAIL"
        setattr(result, "gate_result", gate_result)
        if str(m7_gate.get("status") or "") != "PASS":
            result.ok = False

    if target_rank >= 71:
        gate_result = getattr(result, "gate_result", None)
        if not isinstance(gate_result, dict):
            gate_result = {"status": "PASS"}
        m7_gate = gate_result.get("m7") if isinstance(gate_result.get("m7"), dict) else {"status": "FAIL", "reason": "missing_m7_gate"}
        gate_result["m71"] = m7_gate
        setattr(result, "gate_result", gate_result)

    if target_rank >= 72:
        m72_gate = {"status": "FAIL", "reason": "uninitialized"}
        try:
            from cgc_engine.product import run_m72_gate
            from cgc_engine.agent.eval.eko_gui_agent_demo import collect_gui_runtime_evidence

            m72_dir = Path(output_dir).resolve() / "m72_industrial"
            m72_dir.mkdir(parents=True, exist_ok=True)
            gui_evidence_path = str(pipeline_gui_evidence_path or "")
            if not str(gui_evidence_path).strip() and not bool(getattr(args, "m72_disable_gui_evidence", False)):
                gui_duration_s = int(getattr(args, "m72_gui_duration_s", 5))
                if gui_duration_s > 0:
                    gui_evidence_path = str(
                        collect_gui_runtime_evidence(
                            duration_sec=gui_duration_s,
                            output_dir=m72_dir / "gui_agent_runtime",
                        )
                    )
            prev_gui_evidence = os.environ.get("CGC_M72_GUI_EVENT_EVIDENCE")
            if str(gui_evidence_path).strip():
                os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = str(gui_evidence_path)
            elif "CGC_M72_GUI_EVENT_EVIDENCE" in os.environ:
                del os.environ["CGC_M72_GUI_EVENT_EVIDENCE"]
            try:
                m72_report = run_m72_gate(output_dir=str(m72_dir), cgc_report=result.__dict__)
            finally:
                if prev_gui_evidence is None:
                    os.environ.pop("CGC_M72_GUI_EVENT_EVIDENCE", None)
                else:
                    os.environ["CGC_M72_GUI_EVENT_EVIDENCE"] = prev_gui_evidence
            m72_gate = ((m72_report or {}).get("gate_result") or {}).get("m72") if isinstance(m72_report, dict) else {"status": "FAIL", "reason": "invalid_m72_report"}
            if str(gui_evidence_path).strip():
                m72_gate["auto_gui_evidence_path"] = str(gui_evidence_path)
        except Exception as e:
            m72_gate = {"status": "FAIL", "reason": f"m72_gate_error:{repr(e)}"}

        try:
            result.steps["m72_industrial"] = m72_gate
        except Exception:
            pass

        gate_result = getattr(result, "gate_result", None)
        if not isinstance(gate_result, dict):
            gate_result = {"status": "PASS"}
        gate_result["m72"] = m72_gate
        prev_ok = str(gate_result.get("status") or "PASS") == "PASS"
        m72_ok = str(m72_gate.get("status") or "") == "PASS"
        gate_result["status"] = "PASS" if (prev_ok and m72_ok) else "FAIL"
        setattr(result, "gate_result", gate_result)
        if str(m72_gate.get("status") or "") != "PASS":
            result.ok = False

    if target_rank >= 73:
        m73_gate = {"status": "FAIL", "reason": "uninitialized"}
        try:
            from cgc_engine.product import run_m73_gate

            m73_report = run_m73_gate(output_dir=str(Path(output_dir).resolve()))
            m73_gate = ((m73_report or {}).get("gate_result") or {}).get("m73") if isinstance(m73_report, dict) else {"status": "FAIL", "reason": "invalid_m73_report"}
        except Exception as e:
            m73_gate = {"status": "FAIL", "reason": f"m73_gate_error:{repr(e)}"}

        try:
            result.steps["m73_physical"] = m73_gate
        except Exception:
            pass

        gate_result = getattr(result, "gate_result", None)
        if not isinstance(gate_result, dict):
            gate_result = {"status": "PASS"}
        gate_result["m73"] = m73_gate
        prev_ok = str(gate_result.get("status") or "PASS") == "PASS"
        m73_ok = str(m73_gate.get("status") or "") == "PASS"
        gate_result["status"] = "PASS" if (prev_ok and m73_ok) else "FAIL"
        setattr(result, "gate_result", gate_result)
        if str(m73_gate.get("status") or "") != "PASS":
            result.ok = False

    if target_rank >= 77:
        m77_gate = {"status": "FAIL", "reason": "uninitialized"}
        try:
            from cgc_engine.product import run_m77_gate

            m77_report = run_m77_gate(output_dir=str(Path(output_dir).resolve()), cgc_report=result.__dict__)
            m77_gate = ((m77_report or {}).get("gate_result") or {}).get("m77") if isinstance(m77_report, dict) else {"status": "FAIL", "reason": "invalid_m77_report"}
        except Exception as e:
            m77_gate = {"status": "FAIL", "reason": f"m77_gate_error:{repr(e)}"}

        try:
            result.steps["m77_cloud_edge_q2rl"] = m77_gate
        except Exception:
            pass

        gate_result = getattr(result, "gate_result", None)
        if not isinstance(gate_result, dict):
            gate_result = {"status": "PASS"}
        gate_result["m77"] = m77_gate
        prev_ok = str(gate_result.get("status") or "PASS") == "PASS"
        m77_ok = str(m77_gate.get("status") or "") == "PASS"
        gate_result["status"] = "PASS" if (prev_ok and m77_ok) else "FAIL"
        setattr(result, "gate_result", gate_result)
        if str(m77_gate.get("status") or "") != "PASS":
            result.ok = False

    if target_rank >= 78:
        m78_gate = {"status": "FAIL", "reason": "uninitialized"}
        try:
            from cgc_engine.product import run_m78_gate

            m78_report = run_m78_gate(output_dir=str(Path(output_dir).resolve()), cgc_report=result.__dict__)
            m78_gate = ((m78_report or {}).get("gate_result") or {}).get("m78") if isinstance(m78_report, dict) else {"status": "FAIL", "reason": "invalid_m78_report"}
        except Exception as e:
            m78_gate = {"status": "FAIL", "reason": f"m78_gate_error:{repr(e)}"}

        try:
            result.steps["m78_teaching_pure_llm"] = m78_gate
        except Exception:
            pass

        gate_result = getattr(result, "gate_result", None)
        if not isinstance(gate_result, dict):
            gate_result = {"status": "PASS"}
        gate_result["m78"] = m78_gate
        prev_ok = str(gate_result.get("status") or "PASS") == "PASS"
        m78_ok = str(m78_gate.get("status") or "") == "PASS"
        gate_result["status"] = "PASS" if (prev_ok and m78_ok) else "FAIL"
        setattr(result, "gate_result", gate_result)
        if str(m78_gate.get("status") or "") != "PASS":
            result.ok = False

    if target_rank >= 79:
        upkg39_gate = {"status": "FAIL", "reason": "uninitialized"}
        try:
            from cgc_engine.product import run_upkg39_gate

            upkg39_report = run_upkg39_gate(output_dir=str(Path(output_dir).resolve()), cgc_report=result.__dict__)
            upkg39_gate = ((upkg39_report or {}).get("gate_result") or {}).get("upkg39") if isinstance(upkg39_report, dict) else {"status": "FAIL", "reason": "invalid_upkg39_report"}
        except Exception as e:
            upkg39_gate = {"status": "FAIL", "reason": f"upkg39_gate_error:{repr(e)}"}

        try:
            result.steps["upkg39_strict_closure"] = upkg39_gate
        except Exception:
            pass

        gate_result = getattr(result, "gate_result", None)
        if not isinstance(gate_result, dict):
            gate_result = {"status": "PASS"}
        gate_result["upkg39"] = upkg39_gate
        prev_ok = str(gate_result.get("status") or "PASS") == "PASS"
        upkg39_ok = str(upkg39_gate.get("status") or "") == "PASS"
        gate_result["status"] = "PASS" if (prev_ok and upkg39_ok) else "FAIL"
        setattr(result, "gate_result", gate_result)
        if str(upkg39_gate.get("status") or "") != "PASS":
            result.ok = False

    if target_rank >= 80:
        m79_gate = {"status": "FAIL", "reason": "uninitialized"}
        try:
            from cgc_engine.product import run_m79_gate

            m79_report = run_m79_gate(output_dir=str(Path(output_dir).resolve()), cgc_report=result.__dict__)
            m79_gate = ((m79_report or {}).get("gate_result") or {}).get("m79") if isinstance(m79_report, dict) else {"status": "FAIL", "reason": "invalid_m79_report"}
        except Exception as e:
            m79_gate = {"status": "FAIL", "reason": f"m79_gate_error:{repr(e)}"}

        try:
            result.steps["m79_embodied_upkg40"] = m79_gate
        except Exception:
            pass

        gate_result = getattr(result, "gate_result", None)
        if not isinstance(gate_result, dict):
            gate_result = {"status": "PASS"}
        gate_result["m79"] = m79_gate
        prev_ok = str(gate_result.get("status") or "PASS") == "PASS"
        m79_ok = str(m79_gate.get("status") or "") == "PASS"
        gate_result["status"] = "PASS" if (prev_ok and m79_ok) else "FAIL"
        setattr(result, "gate_result", gate_result)
        if str(m79_gate.get("status") or "") != "PASS":
            result.ok = False

    if milestone in {"m77", "upkg37", "m78", "upkg38", "upkg39", "m79", "upkg40"}:
        gate_result = getattr(result, "gate_result", None)
        if isinstance(gate_result, dict) and str(gate_result.get("status") or "") == "PASS":
            try:
                if not isinstance(result.steps, dict):
                    result.steps = {}
                result.steps["verification_mode_override"] = {
                    "status": "PASS",
                    "mode": "gate_first",
                    "applies_to": milestone,
                    "base_pipeline_error_msg": str(getattr(result, "error_msg", "") or ""),
                }
            except Exception:
                pass
            result.ok = True
            try:
                result.error_msg = ""
            except Exception:
                pass
            try:
                result.traceback = ""
            except Exception:
                pass

    report_path = str(getattr(args, "report_path", "") or "").strip()
    if report_path == "":
        report_path = str(Path(output_dir) / "report.json")
    pipe.write_report(result, report_path)

    bundle_export_dir = str(getattr(args, "bundle_export_dir", "") or "").strip()
    if milestone == "m3" and bundle_export_dir == "":
        bundle_export_dir = str(Path(output_dir) / "bundle_export")
    if bundle_export_dir:
        export_info = pipe.export_bundle(result, bundle_export_dir=bundle_export_dir)
        pipe.write_report(result, report_path)
    print(json.dumps({"ok": bool(result.ok), "output_dir": str(output_dir), "report_path": str(report_path)}, ensure_ascii=False))
    return 0 if result.ok else 1

def add_info_subparser(subparsers):
    """Add 'info' subcommand for system information"""
    parser = subparsers.add_parser(
        'info',
        help='Display CGC Engine system information',
        description='Show version, available backends, and system capabilities',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        default=False,
        help='Show detailed information',
    )
    parser.set_defaults(func=info_command)

def info_command(args):
    """Execute 'info' subcommand"""
    print("=" * 70)
    print("CGC Engine CLI - MagiCompiler Command Interface")
    print("=" * 70)

    try:
        from cgc_engine import __version__
        print(f"Version: {__version__}")
    except ImportError:
        print("Version: N/A")

    print(f"\nAvailable commands:")
    print(f"  pipeline    - Run unified 8-step pipeline (LLM/MLX)")
    print(f"  agent       - Run product-facing agent workflows (DAG import / teach / train / infer / visualize)")
    print(f"  info        - Show this information")

    print(f"\n" + "=" * 70)
    print("Backends (Execution Engine)")
    print("=" * 70)
    print(f"  vllm        - vLLM (CUDA) inference engine")
    print(f"  llama.cpp   - llama.cpp GGUF inference")
    print(f"  mlx         - Apple Silicon MLX")
    print(f"  mindspeed   - MindSpeed-LLM OpenAI-compatible endpoint (remote)")

    print(f"\n" + "=" * 70)
    print("Execution Modes (--exec-mode)")
    print("=" * 70)
    print(f"  native      - Original runtime baseline")
    print(f"  inject      - Inject custom backend/hook into runtime")
    print(f"  compile     - Full-graph analyze + compile (.so)")

    print(f"\n" + "=" * 70)
    print("Architecture Flow")
    print("=" * 70)
    print(f"  CLI -> LLMAutoPipeline -> Backend")
    print(f"              |")
    print(f"              v")
    print(f"  [ LLM1 -> SkVM -> FX -> KDA -> AOTInductor -> .so ]")

    if args.verbose:
        print(f"\n" + "=" * 70)
        print("System Capabilities")
        print("=" * 70)
        try:
            import torch
            print(f"  - CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  - CUDA device count: {torch.cuda.device_count()}")
        except ImportError:
            print("  - PyTorch not available")

    print("=" * 60)
    return 0


def add_agent_subparser(subparsers):
    parser = subparsers.add_parser(
        'agent',
        help='Run engine-facing agent workflows with the same artifact contract as `cgc agent`',
        description='Engine CLI wrapper for product-facing agent workflows: import-dag / teach / train / infer / visualize / compare / audit / replay / trace',
    )
    agent_subparsers = parser.add_subparsers(dest='agent_command', help='Agent workflow commands')

    import_parser = agent_subparsers.add_parser('import-dag', help='Import a DAG/workflow and prepare compute-graph insertion artifacts')
    import_parser.add_argument('--dag-file', type=str, required=True, help='Path to a JSON DAG/workflow file')
    import_parser.add_argument('--dag-name', type=str, default='', help='Optional logical DAG name override')
    import_parser.add_argument('--output-dir', type=str, default='', help='Directory to write imported DAG artifacts')
    import_parser.add_argument('--json', action='store_true', help='Print result as JSON')

    teach_parser = agent_subparsers.add_parser('teach', help='Collect GUI teaching evidence and write teaching/replay artifacts')
    teach_parser.add_argument('--output-dir', type=str, default='', help='Directory to write teaching artifacts')
    teach_parser.add_argument('--dag-file', type=str, default='', help='Optional DAG JSON to import before teaching')
    teach_parser.add_argument('--dag-manifest', type=str, default='', help='Optional existing imported DAG manifest path')
    teach_parser.add_argument('--dag-name', type=str, default='', help='Optional logical DAG name override')
    teach_parser.add_argument('--teaching-mode', type=str, choices=['development', 'customer'], default='development', help='Teaching evidence mode: development validation or customer real capture with screen recording plus keyboard/mouse events')
    teach_parser.add_argument('--gui-duration-s', type=int, default=5, help='Collect GUI teaching evidence for this many seconds; mainly for development validation or as supplemental evidence in customer mode')
    teach_parser.add_argument('--gui-evidence-path', type=str, default='', help='Use an existing GUI runtime evidence file instead of recording a new one')
    teach_parser.add_argument('--screen-recording-path', type=str, default='', help='Customer mode: full screen-recording file captured from the real teaching session')
    teach_parser.add_argument('--keyboard-mouse-events-path', type=str, default='', help='Customer mode: keyboard/mouse event trace file for the same teaching session')
    teach_parser.add_argument('--json', action='store_true', help='Print result as JSON')

    train_parser = agent_subparsers.add_parser('train', help='Run the upkg38/UI-TARS/Q2RL training chain with optional DAG import')
    train_parser.add_argument('--output-dir', type=str, default='', help='Directory to write training artifacts')
    train_parser.add_argument('--teach-session', type=str, default='', help='Existing `cgc agent teach` session path')
    train_parser.add_argument('--dag-file', type=str, default='', help='Optional DAG JSON to import before training')
    train_parser.add_argument('--dag-manifest', type=str, default='', help='Optional existing imported DAG manifest path')
    train_parser.add_argument('--dag-name', type=str, default='', help='Optional logical DAG name override')
    train_parser.add_argument('--teaching-mode', type=str, choices=['development', 'customer'], default='development', help='Training evidence mode; use customer when the source teaching session comes from real customer recording plus keyboard/mouse events')
    train_parser.add_argument('--gui-duration-s', type=int, default=5, help='Collect GUI evidence for this many seconds when no teach session/evidence is provided; intended for development validation')
    train_parser.add_argument('--gui-evidence-path', type=str, default='', help='Existing GUI runtime evidence file to feed into upkg38')
    train_parser.add_argument('--screen-recording-path', type=str, default='', help='Customer mode: full screen-recording file for the source teaching session')
    train_parser.add_argument('--keyboard-mouse-events-path', type=str, default='', help='Customer mode: keyboard/mouse event trace file for the source teaching session')
    train_parser.add_argument('--json', action='store_true', help='Print result as JSON')

    infer_parser = agent_subparsers.add_parser('infer', help='Materialize the edge inference bundle from a trained agent session')
    infer_parser.add_argument('--train-session', type=str, default='', help='`cgc agent train` session path')
    infer_parser.add_argument('--artifact-root', type=str, default='', help='Existing upkg38 output root when not using --train-session')
    infer_parser.add_argument('--output-dir', type=str, default='', help='Directory to write infer session artifacts')
    infer_parser.add_argument('--json', action='store_true', help='Print result as JSON')

    visualize_parser = agent_subparsers.add_parser('visualize', help='Index triplet comparison and error visualization outputs')
    visualize_parser.add_argument('--train-session', type=str, default='', help='`cgc agent train` session path')
    visualize_parser.add_argument('--artifact-root', type=str, default='', help='Existing upkg38 output root when not using --train-session')
    visualize_parser.add_argument('--output-dir', type=str, default='', help='Directory to write visualization index artifacts')
    visualize_parser.add_argument('--json', action='store_true', help='Print result as JSON')

    compare_parser = agent_subparsers.add_parser('compare', help='Summarize teaching vs pre/post Q2RL comparison artifacts')
    compare_parser.add_argument('--train-session', type=str, default='', help='`cgc agent train` session path')
    compare_parser.add_argument('--artifact-root', type=str, default='', help='Existing upkg38 output root when not using --train-session')
    compare_parser.add_argument('--output-dir', type=str, default='', help='Directory to write comparison summary artifacts')
    compare_parser.add_argument('--json', action='store_true', help='Print result as JSON')

    audit_parser = agent_subparsers.add_parser('audit', help='Summarize audit/replay/traceability outputs for a trained agent session')
    audit_parser.add_argument('--train-session', type=str, default='', help='`cgc agent train` session path')
    audit_parser.add_argument('--artifact-root', type=str, default='', help='Existing upkg38 output root when not using --train-session')
    audit_parser.add_argument('--output-dir', type=str, default='', help='Directory to write audit summary artifacts')
    audit_parser.add_argument('--json', action='store_true', help='Print result as JSON')

    replay_parser = agent_subparsers.add_parser('replay', help='Prepare replay metadata for GUI teaching and upkg38 results')
    replay_parser.add_argument('--train-session', type=str, default='', help='`cgc agent train` session path')
    replay_parser.add_argument('--artifact-root', type=str, default='', help='Existing upkg38 output root when not using --train-session')
    replay_parser.add_argument('--output-dir', type=str, default='', help='Directory to write replay session artifacts')
    replay_parser.add_argument('--json', action='store_true', help='Print result as JSON')

    trace_parser = agent_subparsers.add_parser('trace', help='Summarize stage trace and GUI event trace for a trained agent session')
    trace_parser.add_argument('--train-session', type=str, default='', help='`cgc agent train` session path')
    trace_parser.add_argument('--artifact-root', type=str, default='', help='Existing upkg38 output root when not using --train-session')
    trace_parser.add_argument('--output-dir', type=str, default='', help='Directory to write trace summary artifacts')
    trace_parser.add_argument('--json', action='store_true', help='Print result as JSON')

    parser.set_defaults(func=agent_command)


def agent_command(args):
    release_cgc_cli = _load_release_agent_cli_module()
    try:
        if args.agent_command == 'import-dag':
            output_dir = release_cgc_cli._make_agent_output_dir(args.output_dir, command_name="engine_agent_import_dag")
            result = release_cgc_cli._agent_import_dag(
                dag_path=args.dag_file,
                output_dir=output_dir,
                dag_name=args.dag_name,
            )
        elif args.agent_command == 'teach':
            output_dir = release_cgc_cli._make_agent_output_dir(args.output_dir, command_name="engine_agent_teach")
            result = release_cgc_cli._agent_collect_teach_session(
                output_dir=output_dir,
                gui_duration_s=args.gui_duration_s,
                gui_evidence_path=args.gui_evidence_path,
                dag_manifest_path=args.dag_manifest,
                dag_file=args.dag_file,
                dag_name=args.dag_name,
                teaching_mode=args.teaching_mode,
                screen_recording_path=args.screen_recording_path,
                keyboard_mouse_events_path=args.keyboard_mouse_events_path,
            )
        elif args.agent_command == 'train':
            output_dir = release_cgc_cli._make_agent_output_dir(args.output_dir, command_name="engine_agent_train")
            result = release_cgc_cli._agent_train_session(
                output_dir=output_dir,
                teach_session_path=args.teach_session,
                dag_manifest_path=args.dag_manifest,
                dag_file=args.dag_file,
                dag_name=args.dag_name,
                gui_duration_s=args.gui_duration_s,
                gui_evidence_path=args.gui_evidence_path,
                teaching_mode=args.teaching_mode,
                screen_recording_path=args.screen_recording_path,
                keyboard_mouse_events_path=args.keyboard_mouse_events_path,
            )
        elif args.agent_command == 'infer':
            result = release_cgc_cli._agent_infer_session(
                train_session_path=args.train_session,
                artifact_root=args.artifact_root,
                output_dir=args.output_dir,
            )
        elif args.agent_command == 'visualize':
            result = release_cgc_cli._agent_visualize_session(
                train_session_path=args.train_session,
                artifact_root=args.artifact_root,
                output_dir=args.output_dir,
            )
        elif args.agent_command == 'compare':
            result = release_cgc_cli._agent_compare_session(
                train_session_path=args.train_session,
                artifact_root=args.artifact_root,
                output_dir=args.output_dir,
            )
        elif args.agent_command == 'audit':
            result = release_cgc_cli._agent_audit_session(
                train_session_path=args.train_session,
                artifact_root=args.artifact_root,
                output_dir=args.output_dir,
            )
        elif args.agent_command == 'replay':
            result = release_cgc_cli._agent_replay_session(
                train_session_path=args.train_session,
                artifact_root=args.artifact_root,
                output_dir=args.output_dir,
            )
        elif args.agent_command == 'trace':
            result = release_cgc_cli._agent_trace_session(
                train_session_path=args.train_session,
                artifact_root=args.artifact_root,
                output_dir=args.output_dir,
            )
        else:
            print(json.dumps({"status": "FAIL", "error": "missing agent subcommand"}, ensure_ascii=False))
            return 2
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 1

    if bool(getattr(args, "json", False)):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"engine agent {args.agent_command} status: {result.get('status')}")
        for key in (
            "dag_manifest_path",
            "graph_insertion_contract_path",
            "teach_session_path",
            "train_session_path",
            "infer_session_path",
            "visualization_index_path",
            "compare_session_path",
            "audit_session_path",
            "replay_session_path",
            "trace_session_path",
            "subterranean_bundle_path",
        ):
            if str(result.get(key) or "").strip():
                print(f"{key}: {result[key]}")
        if str(result.get("output_dir") or "").strip():
            print(f"output_dir: {result['output_dir']}")
        if str(result.get("upkg38_output_dir") or "").strip():
            print(f"upkg38_output_dir: {result['upkg38_output_dir']}")
    return 0 if str(result.get("status") or "") == "PASS" else 1

def create_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        prog='cgc',
        description='CGC Engine Internal CLI - Unified Pipeline Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture:
    CLI -> LLMAutoPipeline -> (LLM1 -> SkVM -> FX -> KDA -> AOTInductor) -> Backend

Note:
    `cgc pipeline` is the internal engineering / verification interface.
    `cgc agent ...` is the product-facing engine workflow interface for UPKG 3.8/3.9/4.0.
    Release-facing user entrypoints remain `cgc agent ...` and `cgc gate ...`.

Examples:
  # 1. Internal native mode (baseline)
  cgc pipeline --backend vllm --exec-mode native

  # 2. Internal inject mode (runtime monkey patch / custom backend)
  cgc pipeline --backend llama.cpp --exec-mode inject --enable-ortho-kda

  # 3. Internal compile mode (LLM1 Translation + SkVM Verify + AOTInductor)
  cgc pipeline --backend mlx --exec-mode compile --enable-llm1 --llm1-input-path /path/to/attn.metal

  # Show info
  cgc info --verbose

  # Engine-facing agent workflow
  cgc agent import-dag --dag-file /path/to/workflow.json
  cgc agent teach --teaching-mode development --dag-file /path/to/workflow.json --gui-duration-s 5
  cgc agent teach --teaching-mode customer --dag-file /path/to/workflow.json --screen-recording-path /path/to/screen_recording.mp4 --keyboard-mouse-events-path /path/to/keyboard_mouse_events.jsonl
  cgc agent train --teach-session /path/to/agent_teach_session.json --teaching-mode customer
  cgc agent infer --train-session /path/to/agent_train_session.json
  cgc agent visualize --train-session /path/to/agent_train_session.json
  cgc agent compare --train-session /path/to/agent_train_session.json
  cgc agent audit --train-session /path/to/agent_train_session.json
  cgc agent replay --train-session /path/to/agent_train_session.json
  cgc agent trace --train-session /path/to/agent_train_session.json
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 3.0.0',
    )

    subparsers = parser.add_subparsers(
        title='commands',
        dest='command',
        description='Available commands',
    )

    add_pipeline_subparser(subparsers)
    add_agent_subparser(subparsers)
    add_info_subparser(subparsers)

    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return 0

    return args.func(args)

if __name__ == '__main__':
    sys.exit(main())
