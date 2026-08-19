import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _status(ok: bool, **extra: Any) -> Dict[str, Any]:
    payload = {"status": "PASS" if ok else "FAIL"}
    payload.update(extra)
    return payload


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_trueorthkda_gate(payload: Dict[str, Any]) -> Dict[str, Any]:
    direct_gate = payload.get("trueorthkda_gate")
    if isinstance(direct_gate, dict):
        return direct_gate
    steps = payload.get("steps")
    if isinstance(steps, dict):
        nested_gate = steps.get("trueorthkda_gate")
        if isinstance(nested_gate, dict):
            return nested_gate
    gate_result = payload.get("gate_result")
    if isinstance(gate_result, dict):
        nested_gate = gate_result.get("trueorthkda_gate")
        if isinstance(nested_gate, dict):
            return nested_gate
    return {}


def _run_check(check_name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        if not isinstance(result, dict):
            return _status(False, check=check_name, error="invalid_result", detail=repr(result))
        result.setdefault("check", check_name)
        return result
    except Exception as exc:
        return _status(False, check=check_name, error=str(exc), exception_type=type(exc).__name__)


def _validate_dflash_contract(backend_fingerprint_path: Path, cli_path: Path) -> Dict[str, Any]:
    backend_source = _read_text(backend_fingerprint_path)
    cli_source = _read_text(cli_path)
    backend_markers = [
        'os.environ.get("CGC_REQUIRE_DFLASH", "0") == "1"',
        'os.environ.get("CGC_DFLASH_ENABLED", "0") == "1"',
        "STRICT GATE FAIL: DFlash (FlashKV) is required",
    ]
    cli_markers = [
        "--require-dflash",
    ]
    missing_backend = [marker for marker in backend_markers if marker not in backend_source]
    missing_cli = [marker for marker in cli_markers if marker not in cli_source]
    runtime_env_markers = {
        "CGC_REQUIRE_DFLASH": os.environ.get("CGC_REQUIRE_DFLASH", ""),
        "CGC_DFLASH_ENABLED": os.environ.get("CGC_DFLASH_ENABLED", ""),
    }
    return _status(
        not missing_backend and not missing_cli,
        backend_file=str(backend_fingerprint_path),
        cli_file=str(cli_path),
        missing_backend_markers=missing_backend,
        missing_cli_markers=missing_cli,
        runtime_env_markers=runtime_env_markers,
    )


def _validate_trueorthkda_contract(llm_auto_pipeline_path: Path) -> Dict[str, Any]:
    source = _read_text(llm_auto_pipeline_path)
    markers = [
        'os.environ.get("CGC_REQUIRE_TRUEORTHOKDA")',
        "STRICT GATE FAIL: require trueorthkda but enable_ortho_kda is false",
        '"enable_ortho_kda": True',
        '"ortho_kda_base_dim": int(ortho_kda_base_dim)',
    ]
    missing = [marker for marker in markers if marker not in source]
    return _status(
        not missing,
        file=str(llm_auto_pipeline_path),
        missing_markers=missing,
    )


def _find_trueorthkda_evidence() -> Optional[Path]:
    output_root = WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "Output"
    candidate_roots = [
        output_root / "PipelineRuns" / "llama.cpp",
        output_root / "manual_m7_seq",
        output_root / "cli_gate_m1",
        output_root / "cli_gate_m2",
        output_root / "cli_gate_m3",
        output_root / "cli_gate_m5",
        output_root / "cli_gate_m6",
        WORKSPACE_ROOT / "temp" / "test",
    ]
    candidates: List[Path] = []
    for root in candidate_roots:
        if not root.exists():
            continue
        candidates.extend(root.rglob("report.json"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = _read_json(path)
        except Exception:
            continue
        gate = _extract_trueorthkda_gate(payload)
        if isinstance(gate, dict) and str(gate.get("status") or "") == "PASS":
            return path
    return None


def _load_trueorthkda_evidence() -> Dict[str, Any]:
    evidence_path = _find_trueorthkda_evidence()
    if evidence_path is None:
        return _status(False, reason="missing_trueorthkda_runtime_evidence")
    payload = _read_json(evidence_path)
    gate = _extract_trueorthkda_gate(payload)
    ok = bool(
        str(gate.get("status") or "") == "PASS"
        and bool(gate.get("require_trueorthkda"))
        and bool(gate.get("enable_ortho_kda"))
    )
    return _status(
        ok,
        evidence_path=str(evidence_path),
        trueorthkda_gate=gate,
    )


def _load_optional_edge_report() -> Dict[str, Any]:
    report_path = WORKSPACE_ROOT / "cgc_report_m74.json"
    if not report_path.exists():
        return _status(True, report="not_provided", optional=True)
    payload = _read_json(report_path)
    network = payload.get("network") or {}
    edge_memory = payload.get("edge_memory") or {}
    experience = payload.get("experience") or {}
    ok = bool(
        float(network.get("bandwidth_mbps") or 0.0) > 0.0
        and bool(network.get("chunk_streaming_enabled"))
        and float(edge_memory.get("vram_write_ms") or 0.0) > 0.0
        and float(experience.get("generation_tps") or 0.0) > 0.0
    )
    return _status(ok, report_path=str(report_path), report=payload, optional=True)


def run_m74_gate(*, output_dir: str) -> Dict[str, Any]:
    output_root = Path(str(output_dir)).expanduser().resolve()
    m74_dir = (output_root / "m74_dflash_kda").resolve()
    m74_dir.mkdir(parents=True, exist_ok=True)

    backend_fingerprint_path = (WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "cgc_engine" / "agent" / "backend_fingerprint.py").resolve()
    llm_auto_pipeline_path = (WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "cgc_engine" / "agent" / "llm_auto_pipeline.py").resolve()
    cli_path = (WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "cgc_engine" / "agent" / "cli.py").resolve()

    checks: Dict[str, Dict[str, Any]] = {
        "dflash_contract": _run_check("dflash_contract", _validate_dflash_contract, backend_fingerprint_path, cli_path),
        "trueorthkda_contract": _run_check("trueorthkda_contract", _validate_trueorthkda_contract, llm_auto_pipeline_path),
        "trueorthkda_runtime": _run_check("trueorthkda_runtime", _load_trueorthkda_evidence),
        "edge_runtime_evidence": _run_check("edge_runtime_evidence", _load_optional_edge_report),
    }

    passed_checks = [name for name, result in checks.items() if str(result.get("status") or "") == "PASS"]
    failed_checks = [name for name, result in checks.items() if str(result.get("status") or "") == "FAIL"]
    ok = not failed_checks

    gate = {
        "status": "PASS" if ok else "FAIL",
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
    }
    report = {
        "ok": ok,
        "milestone": "m74",
        "scope": "verification_only",
        "public_entrypoint": "cgc gate m74",
        "gate_result": {"m74": gate},
    }
    report_path = (m74_dir / "m74_report.json").resolve()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": ok, "gate_result": {"m74": gate}, "report_path": str(report_path)}


def main() -> None:
    print("=" * 60)
    print("🚀 CGC Engine M7.4 Verification Gate: DFlash + TrueOrthoKDA")
    print("=" * 60)
    report = run_m74_gate(output_dir=str((WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "Output" / "cli_gate_m74").resolve()))
    gate = ((report.get("gate_result") or {}).get("m74") or {})
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not bool(report.get("ok")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
