#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Output" / "cli_gate_m76"


def _arg(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _build_exploration_command(args: argparse.Namespace) -> str:
    command: list[str] = [
        "python",
        "cgc_engine/cli.py",
        "model",
        "verify",
        "--model",
        args.model,
        "--gate",
        args.gate,
    ]
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.bundle:
        command.extend(["--bundle", args.bundle])
    if _arg(args, "prompt"):
        command.extend(["--prompt", str(_arg(args, "prompt"))])
    if args.strict:
        command.append("--strict")
    if args.deepep:
        command.append("--deepep")
    if args.l20n:
        command.append("--l20n")
    if _arg(args, "task_type"):
        command.extend(["--task-type", str(_arg(args, "task_type"))])
    if _arg(args, "fusion_config"):
        command.extend(["--fusion-config", str(_arg(args, "fusion_config"))])
    if args.eplb:
        command.append("--eplb")
        command.extend(["--expert-replica-factor", str(args.expert_replica_factor)])
    if args.waterfill:
        command.append("--waterfill")
        command.extend(["--waterfill-epsilon", str(args.waterfill_epsilon)])
    if args.lplb:
        command.append("--lplb")
        command.extend(["--lplb-parallelism", str(args.lplb_parallelism)])
    if args.enable_speculative:
        command.append("--enable-speculative")
    if args.speculative_mode:
        command.extend(["--speculative-mode", args.speculative_mode])
    if args.dspark_budget is not None:
        command.extend(["--dspark-budget", str(args.dspark_budget)])
    if args.jetspec_branches is not None:
        command.extend(["--jetspec-branches", str(args.jetspec_branches)])
    return " ".join(shlex.quote(part) for part in command)


def _runtime_env_mapping(args: argparse.Namespace) -> dict[str, str]:
    env = {
        "CGC_M76_DEV_MODE": "1",
        "CGC_REQUIRE_FORMAL_EVIDENCE": "1",
        "CGC_FORMAL_SUITE": "swe_bench_verified_500",
    }
    if args.deepep:
        env["CGC_M76_ENABLE_DEEPEP"] = "1"
        env["CGC_REQUESTED_DISPATCH_BACKEND"] = "deepep"
    if args.l20n:
        env["CGC_SERVICE_TOPOLOGY_BACKEND"] = "ray_cluster_dual_host"
    if _arg(args, "pd_mode"):
        env["CGC_PD_MODE"] = str(_arg(args, "pd_mode"))
    if args.profile and args.profile.startswith("ep") and "_tp" in args.profile:
        env["CGC_DEEPEP_PARALLEL_PROFILE"] = args.profile
    return env


def _manifest_annotations(args: argparse.Namespace, exploration_command: str) -> dict[str, Any]:
    annotations: dict[str, Any] = {
        "source_cli": "python cgc_engine/cli.py model verify",
        "source_gate": args.gate,
        "source_model": args.model,
        "source_profile": args.profile or "",
        "source_bundle": args.bundle or "",
        "source_prompt": str(_arg(args, "prompt") or ""),
        "source_strict": bool(args.strict),
        "source_alias": str(_arg(args, "source_alias") or ""),
        "exploration_command": exploration_command,
        "requested_capabilities": {
            "deepep": bool(args.deepep),
            "l20n": bool(args.l20n),
            "task_type": str(_arg(args, "task_type") or ""),
            "fusion_config": str(_arg(args, "fusion_config") or ""),
            "pd_mode": str(_arg(args, "pd_mode") or ""),
            "eplb": bool(args.eplb),
            "waterfill": bool(args.waterfill),
            "lplb": bool(args.lplb),
            "enable_speculative": bool(args.enable_speculative),
            "speculative_mode": args.speculative_mode or "",
            "dspark_budget": int(args.dspark_budget) if args.dspark_budget is not None else None,
            "jetspec_branches": int(args.jetspec_branches) if args.jetspec_branches is not None else None,
        },
        "formalization_policy": {
            "mode": "manifest-first",
            "preferred_entrypoint": "cgc m76-dev",
            "fallback_entrypoint": "cgc_engine.product.run_m76_gate",
            "notes": [
                "Only flags with an existing runtime env consumer are promoted into env.",
                "Speculative DSpark/JetSpec/Fusion flags are preserved as manifest annotations until a stable release-facing runtime contract exists.",
            ],
        },
    }
    if args.eplb:
        annotations["requested_capabilities"]["expert_replica_factor"] = int(args.expert_replica_factor)
    if args.waterfill:
        annotations["requested_capabilities"]["waterfill_epsilon"] = float(args.waterfill_epsilon)
    if args.lplb:
        annotations["requested_capabilities"]["lplb_parallelism"] = int(args.lplb_parallelism)
    return annotations


def _formal_artifacts(output_dir: Path) -> dict[str, str]:
    return {
        "runtime_evidence_dir": str((output_dir / "runtime_evidence").resolve()),
        "m76_report": str((output_dir / "m76_heterogeneous" / "m76_report.json").resolve()),
        "summary": str((output_dir / "m76_heterogeneous" / "summary.json").resolve()),
        "latest": str((output_dir / "m76_heterogeneous" / "latest.json").resolve()),
        "system_execution_manifest": "resolved from the latest manifest-first evidence root during run_m76_gate()",
    }


def _preferred_command(output_dir: Path, env: dict[str, str]) -> str:
    prefix = " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())
    return f"{prefix} cgc m76-dev --output-dir {shlex.quote(str(output_dir))}".strip()


def _fallback_command(output_dir: Path, env: dict[str, str]) -> str:
    prefix = " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())
    output_dir_literal = json.dumps(str(output_dir))
    inline = (
        "from cgc_engine.product import run_m76_gate; "
        "import json; "
        f"print(json.dumps(run_m76_gate(output_dir={output_dir_literal}), ensure_ascii=False, indent=2))"
    )
    return f"{prefix} python3 -c {shlex.quote(inline)}".strip()


def _mapping_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    exploration_command = _build_exploration_command(args)
    env = _runtime_env_mapping(args)
    manifest_annotations = _manifest_annotations(args, exploration_command)
    payload = {
        "bridge": "gate6_model_verify_to_m76_manifest_first",
        "source_command": exploration_command,
        "preferred_formal_command": _preferred_command(output_dir, env),
        "fallback_formal_command": _fallback_command(output_dir, env),
        "runtime_env": env,
        "manifest_annotations": manifest_annotations,
        "formal_artifacts": _formal_artifacts(output_dir),
    }
    return payload


def _write_mapping(mapping: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = (output_dir / "gate6_exploration_to_m76_manifest_mapping.json").resolve()
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return mapping_path


def _run_fallback(mapping: dict[str, Any], output_dir: Path) -> int:
    env = os.environ.copy()
    env.update(mapping["runtime_env"])
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from cgc_engine.product import run_m76_gate

    result = run_m76_gate(output_dir=str(output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("ok")) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map Gate 6.0 `cgc_engine/cli.py model verify` exploration flags to m76-dev manifest-first formal commands.",
    )
    add_cli_arguments(parser)
    return parser


def add_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", required=True, help="Exploration-side model token or symbolic scenario name.")
    parser.add_argument("--gate", default="6.0", help="Exploration gate value. Default: 6.0")
    parser.add_argument("--profile", default=None, help="Optional exploration profile. `epXX_tpYY` maps to CGC_DEEPEP_PARALLEL_PROFILE.")
    parser.add_argument("--bundle", default=None, help="Optional exploration bundle label.")
    parser.add_argument("--prompt", default=None, help="Optional prompt recorded as exploration annotation.")
    parser.add_argument("--strict", action="store_true", default=False, help="Preserve strict-mode intent in manifest annotations.")
    parser.add_argument("--task-type", default=None, help="Optional exploration task type alias recorded into manifest annotations.")
    parser.add_argument("--fusion-config", default=None, help="Optional exploration fusion config alias recorded into manifest annotations.")
    parser.add_argument("--pd-mode", default=None, help="Optional PD mode promoted into formal runtime env when provided.")
    parser.add_argument("--source-alias", default=None, help="Optional alias label describing which stable CLI entrypoint produced this mapping.")
    parser.add_argument("--deepep", action="store_true", default=False, help="Promote DeepEP intent into formal runtime env.")
    parser.add_argument("--l20n", action="store_true", default=False, help="Promote dual-node topology intent into formal runtime env.")
    parser.add_argument("--eplb", action="store_true", default=False, help="Preserve EPLB intent as manifest annotations.")
    parser.add_argument("--waterfill", action="store_true", default=False, help="Preserve Waterfill intent as manifest annotations.")
    parser.add_argument("--lplb", action="store_true", default=False, help="Preserve LPLB intent as manifest annotations.")
    parser.add_argument("--expert-replica-factor", type=int, default=2, dest="expert_replica_factor")
    parser.add_argument("--waterfill-epsilon", type=float, default=0.001, dest="waterfill_epsilon")
    parser.add_argument("--lplb-parallelism", type=int, default=4, dest="lplb_parallelism")
    parser.add_argument("--enable-speculative", action="store_true", default=False, help="Preserve speculative intent as manifest annotations.")
    parser.add_argument("--speculative-mode", choices=["dspark", "jetspec", "fusion"], default=None)
    parser.add_argument("--dspark-budget", type=int, default=None)
    parser.add_argument("--jetspec-branches", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Formal output root for m76 manifest-first artifacts.")
    parser.add_argument("--print-json", action="store_true", default=False, help="Only print the mapping payload as JSON.")
    parser.add_argument("--run-fallback", action="store_true", default=False, help="Execute `run_m76_gate()` after writing the mapping artifact.")
    return parser


def run_namespace(args: argparse.Namespace) -> int:
    mapping = _mapping_payload(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    mapping_path = _write_mapping(mapping, output_dir)

    if args.print_json:
        print(json.dumps({"mapping_path": str(mapping_path), **mapping}, ensure_ascii=False, indent=2))
    else:
        print(f"[mapping] {mapping_path}")
        print(f"[source]   {mapping['source_command']}")
        print(f"[formal]   {mapping['preferred_formal_command']}")
        print(f"[fallback] {mapping['fallback_formal_command']}")

    if args.run_fallback:
        return _run_fallback(mapping, output_dir)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_namespace(args)


if __name__ == "__main__":
    raise SystemExit(main())
