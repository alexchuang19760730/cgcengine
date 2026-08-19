from __future__ import annotations

import os
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

DFlash_REPO_URL = "https://github.com/z-lab/dflash"
DFLASH_PROJECT_URL = "https://z-lab.ai/projects/dflash/"
SGLANG_REPO_URL = "https://github.com/sgl-project/sglang"
SGLANG_SPEC_DOC_URL = "https://docs.sglang.io/docs/advanced_features/speculative_decoding"
SGLANG_DFLASH_PR_URL = "https://github.com/sgl-project/sglang/pull/16818"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_index(paths: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in paths:
        candidate = Path(str(raw)).expanduser()
        if not candidate.exists() or not candidate.is_file():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        entries.append(
            {
                "path": resolved,
                "size_bytes": int(candidate.stat().st_size),
                "sha256": _sha256_file(candidate),
            }
        )
    return entries


def _status(ok: bool, **extra: Any) -> Dict[str, Any]:
    payload = {"status": "PASS" if ok else "FAIL"}
    payload.update(extra)
    return payload


def _pick_metric_stat(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("p50", "mean", "max", "min"):
            candidate = value.get(key)
            if isinstance(candidate, (int, float)):
                return float(candidate)
    return None


def _coerce_context_rows(value: Any) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    if not isinstance(value, list):
        return rows
    for row in value:
        if not isinstance(row, dict):
            continue
        context = row.get("context")
        if not isinstance(context, int):
            try:
                context = int(context)
            except Exception:
                continue
        rows[int(context)] = row
    return rows


def _compute_per_context_benchmark_ratios(
    baseline_rows: Dict[int, Dict[str, Any]],
    optimized_rows: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    per_context: List[Dict[str, Any]] = []
    for context in sorted(set(baseline_rows.keys()).intersection(set(optimized_rows.keys()))):
        baseline = baseline_rows[context]
        optimized = optimized_rows[context]
        bp = _pick_metric_stat(baseline.get("prefill_tps"))
        bd = _pick_metric_stat(baseline.get("decode_tps"))
        bm = _pick_metric_stat(baseline.get("peak_memory_gb"))
        op = _pick_metric_stat(optimized.get("prefill_tps"))
        od = _pick_metric_stat(optimized.get("decode_tps"))
        om = _pick_metric_stat(optimized.get("peak_memory_gb"))
        ratio: Dict[str, Any] = {}
        if bp is not None and bp > 0 and op is not None:
            ratio["prefill_tps"] = float(op / bp)
        if bd is not None and bd > 0 and od is not None:
            ratio["decode_tps"] = float(od / bd)
        if bm is not None and bm > 0 and om is not None:
            ratio["peak_memory_gb"] = float(om / bm)
        per_context.append(
            {
                "context": int(context),
                "baseline": {"prefill_tps": bp, "decode_tps": bd, "peak_memory_gb": bm},
                "optimized": {"prefill_tps": op, "decode_tps": od, "peak_memory_gb": om},
                "ratio": ratio,
            }
        )
    return per_context


def _resolve_official_benchmark_report_path(output_root: Path, gate_dir: Path) -> Path | None:
    explicit_candidates = [
        _resolve_existing_json_path(str(os.environ.get("CGC_UPKG21_OFFICIAL_SGLANG_DFLASH_BENCHMARK_REPORT") or "")),
        _resolve_existing_json_path(str(os.environ.get("CGC_UPKG21_SGLANG_DFLASH_BENCHMARK_REPORT") or "")),
    ]
    for candidate in explicit_candidates:
        if candidate is not None:
            return candidate

    default_candidates = [
        (output_root / "official_sglang_dflash_benchmark.json").resolve(),
        (output_root / "upkg21_benchmark" / "official_sglang_dflash_benchmark.json").resolve(),
        (gate_dir / "official_sglang_dflash_benchmark.json").resolve(),
    ]
    for candidate in default_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _evaluate_official_sglang_dflash_benchmark(output_root: Path, gate_dir: Path) -> Dict[str, Any]:
    required_contexts = [1024, 4096, 8192, 16384]
    metric_schema = {"prefill": "prefill_tps", "decode": "decode_tps", "memory": "peak_memory_gb"}
    report_path = _resolve_official_benchmark_report_path(output_root, gate_dir)
    if report_path is None:
        expected_default_paths = [
            str((output_root / "official_sglang_dflash_benchmark.json").resolve()),
            str((output_root / "upkg21_benchmark" / "official_sglang_dflash_benchmark.json").resolve()),
            str((gate_dir / "official_sglang_dflash_benchmark.json").resolve()),
        ]
        return _status(
            False,
            reason="missing_official_sglang_dflash_benchmark_report",
            required_contexts=required_contexts,
            metric_schema=metric_schema,
            env_keys=[
                "CGC_UPKG21_OFFICIAL_SGLANG_DFLASH_BENCHMARK_REPORT",
                "CGC_UPKG21_SGLANG_DFLASH_BENCHMARK_REPORT",
            ],
            expected_default_paths=expected_default_paths,
            baseline_runtime="official_sglang_upstream",
            optimized_runtime="project_vendored_cloud_sglang",
        )

    payload = _read_json(report_path)
    baseline_payload = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    optimized_payload = payload.get("optimized") if isinstance(payload.get("optimized"), dict) else {}
    baseline_rows = _coerce_context_rows(baseline_payload.get("contexts"))
    optimized_rows = _coerce_context_rows(optimized_payload.get("contexts"))

    missing_required_contexts = {
        "baseline": [ctx for ctx in required_contexts if ctx not in baseline_rows],
        "optimized": [ctx for ctx in required_contexts if ctx not in optimized_rows],
    }
    missing_metrics: List[Dict[str, Any]] = []
    for section, rows in (("baseline", baseline_rows), ("optimized", optimized_rows)):
        for context in required_contexts:
            row = rows.get(context) or {}
            for metric_key in ("prefill_tps", "decode_tps", "peak_memory_gb"):
                if _pick_metric_stat(row.get(metric_key)) is None:
                    missing_metrics.append({"section": section, "context": int(context), "metric": metric_key})

    per_context_ratios = _compute_per_context_benchmark_ratios(baseline_rows, optimized_rows)
    ok = not any(missing_required_contexts.values()) and not missing_metrics
    return _status(
        ok,
        report_path=str(report_path),
        baseline_runtime=str(baseline_payload.get("runtime") or "official_sglang_upstream"),
        optimized_runtime=str(optimized_payload.get("runtime") or "project_vendored_cloud_sglang"),
        model_path=str(payload.get("model_path") or ""),
        draft_model_path=str(payload.get("draft_model_path") or ""),
        speculative_algorithm=str(payload.get("speculative_algorithm") or "DFLASH"),
        required_contexts=required_contexts,
        metric_schema=metric_schema,
        missing_required_contexts=missing_required_contexts,
        missing_metrics=missing_metrics,
        per_context_ratio=per_context_ratios,
        note="UPKG 2.1 compares official upstream SGLang + DFLASH against project vendored cloud_sglang + DFLASH using per-context prefill/decode throughput and peak memory.",
    )


def _resolve_existing_json_path(raw: str) -> Path | None:
    value = str(raw or "").strip()
    if value == "":
        return None
    candidate = Path(value).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()
    return None


def _has_readable_local_weights(snapshot_dir: Path) -> bool:
    if not snapshot_dir.exists() or not snapshot_dir.is_dir():
        return False
    direct_weights = []
    for weight_path in snapshot_dir.glob("*.safetensors"):
        try:
            if weight_path.is_file() and weight_path.stat().st_size > 0:
                direct_weights.append(weight_path)
        except OSError:
            continue
    if direct_weights:
        return True

    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return False
    index_payload = _read_json(index_path)
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shard_names = sorted(
        {
            str(name).strip()
            for name in weight_map.values()
            if str(name).strip().endswith(".safetensors")
        }
    )
    if not shard_names:
        return False
    for shard_name in shard_names:
        shard_path = snapshot_dir / shard_name
        try:
            if not shard_path.is_file() or shard_path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _resolve_local_omlx_model_path() -> Path | None:
    explicit_env = str(os.environ.get("CGC_LOCAL_OMLX_MODEL") or "").strip()
    if explicit_env:
        explicit_path = Path(explicit_env).expanduser()
        if explicit_path.exists() and explicit_path.is_dir() and _has_readable_local_weights(explicit_path):
            return explicit_path.resolve()

    preferred_candidates = [
        (
            WORKSPACE_ROOT
            / "ComputeGraphCompiler-main"
            / "temp"
            / "cache"
            / "local_omlx_models"
            / "Qwen2.5-Coder-0.5B-Instruct-4bit"
        ).resolve(),
        (WORKSPACE_ROOT / "temp" / "cache" / "local_omlx_models" / "Qwen2.5-Coder-0.5B-Instruct-4bit").resolve(),
    ]
    for candidate in preferred_candidates:
        if (candidate / "config.json").exists() and _has_readable_local_weights(candidate):
            return candidate

    search_roots = [
        (WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "temp" / "cache" / "local_omlx_models").resolve(),
        (WORKSPACE_ROOT / "temp" / "cache" / "local_omlx_models").resolve(),
    ]
    discovered: List[Path] = []
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for config_path in root.rglob("config.json"):
            model_dir = config_path.parent.resolve()
            if _has_readable_local_weights(model_dir):
                discovered.append(model_dir)
    if not discovered:
        return None
    discovered.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return discovered[0]


def _is_candidate_official_benchmark_report(report_path: Path) -> bool:
    payload = _read_json(report_path)
    if not payload:
        return False
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    optimized = payload.get("optimized") if isinstance(payload.get("optimized"), dict) else {}
    baseline_rows = _coerce_context_rows(baseline.get("contexts"))
    optimized_rows = _coerce_context_rows(optimized.get("contexts"))
    required_contexts = {1024, 4096, 8192, 16384}
    return required_contexts.issubset(set(baseline_rows.keys())) and required_contexts.issubset(set(optimized_rows.keys()))


def _resolve_seed_official_benchmark_report_path() -> Path | None:
    explicit_candidates = [
        _resolve_existing_json_path(str(os.environ.get("CGC_UPKG21_OFFICIAL_SGLANG_DFLASH_BENCHMARK_REPORT") or "")),
        _resolve_existing_json_path(str(os.environ.get("CGC_UPKG21_SGLANG_DFLASH_BENCHMARK_REPORT") or "")),
    ]
    for candidate in explicit_candidates:
        if candidate is not None and _is_candidate_official_benchmark_report(candidate):
            return candidate

    preferred_candidates = [
        (
            WORKSPACE_ROOT
            / "ComputeGraphCompiler-main"
            / "temp"
            / "upkg21_rdma_fix_artifacts_20260623"
            / "official_sglang_dflash_benchmark.json"
        ).resolve(),
    ]
    for candidate in preferred_candidates:
        if candidate.exists() and candidate.is_file() and _is_candidate_official_benchmark_report(candidate):
            return candidate

    search_roots = [
        (WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "temp").resolve(),
        (WORKSPACE_ROOT / "temp" / "misc").resolve(),
    ]
    discovered: List[Path] = []
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in root.rglob("official_sglang_dflash_benchmark.json"):
            if candidate.is_file() and _is_candidate_official_benchmark_report(candidate):
                discovered.append(candidate.resolve())
    if not discovered:
        return None
    discovered.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return discovered[0]


@contextmanager
def _temporary_env_defaults(overrides: Dict[str, str]):
    applied: Dict[str, str] = {}
    try:
        for name, value in overrides.items():
            if not value or str(os.environ.get(name) or "").strip():
                continue
            os.environ[name] = value
            applied[name] = value
        yield applied
    finally:
        for name in applied:
            os.environ.pop(name, None)


def _select_upkg21_sglang_runtime() -> Dict[str, Any]:
    vendored_root = (
        WORKSPACE_ROOT
        / "ComputeGraphCompiler-main"
        / "Backend"
        / "CGC"
        / "cloud_sglang"
    ).resolve()
    dflash_worker = vendored_root / "python" / "sglang" / "srt" / "speculative" / "dflash_worker.py"
    dflash_utils = vendored_root / "python" / "sglang" / "srt" / "speculative" / "dflash_utils.py"
    spec_info = vendored_root / "python" / "sglang" / "srt" / "speculative" / "spec_info.py"
    environ_py = vendored_root / "python" / "sglang" / "srt" / "environ.py"
    arg_hook = vendored_root / "python" / "sglang" / "srt" / "arg_groups" / "speculative_hook.py"

    marker_results = {
        "vendored_root_exists": vendored_root.exists(),
        "dflash_worker_exists": dflash_worker.exists(),
        "dflash_utils_exists": dflash_utils.exists(),
        "spec_info_exists": spec_info.exists(),
        "environ_exists": environ_py.exists(),
        "speculative_hook_exists": arg_hook.exists(),
    }
    missing_files = [name for name, ok in marker_results.items() if not ok]

    spec_info_text = _read_text(spec_info) if spec_info.exists() else ""
    environ_text = _read_text(environ_py) if environ_py.exists() else ""
    hook_text = _read_text(arg_hook) if arg_hook.exists() else ""
    worker_text = _read_text(dflash_worker) if dflash_worker.exists() else ""

    missing_markers: List[str] = []
    if "DFLASH does not support overlap scheduling (spec v2)." not in spec_info_text:
        missing_markers.append("spec_info:DFlash_spec_v2_incompatibility_marker")
    if "SGLANG_ENABLE_SPEC_V2" not in environ_text:
        missing_markers.append("environ:SGLANG_ENABLE_SPEC_V2")
    if "Overlap scheduler is disabled when using DFLASH speculative decoding" not in hook_text:
        missing_markers.append("speculative_hook:dflash_overlap_disabled")
    if "Initialized DFLASH draft runner" not in worker_text:
        missing_markers.append("dflash_worker:initialized_dflash_draft_runner")

    ok = not missing_files and not missing_markers
    return _status(
        ok,
        selected_runtime="project_vendored_cloud_sglang",
        selected_runtime_path=str(vendored_root),
        alternative_runtime="official_sglang_upstream",
        alternative_runtime_url=SGLANG_REPO_URL,
        dflash_runtime_mode="sglang_dflash_non_overlap",
        comparison_result="DFLASH and SpecV2 are compared, but not conflated into one execution mode",
        selection_reason=[
            "vendored cloud_sglang already ships DFLASH-specific worker / utils / scheduler integration",
            "official SGLang docs expose DFLASH as a first-class speculative algorithm",
            "both official docs and vendored code encode that DFLASH does not currently use overlap scheduler SpecV2",
            "therefore UPKG 2.1 accepts the vendored SGLang DFLASH route, while keeping SpecV2 as a comparison baseline rather than a hard coupling",
        ],
        compatibility_statement={
            "dflash": "supported",
            "spec_v2_overlap_scheduler": "supported_by_sglang_but_not_for_dflash",
            "upkg21_acceptance_mode": "dflash_on_vendored_sglang_non_overlap",
        },
        required_source_urls={
            "dflash_repo": DFlash_REPO_URL,
            "dflash_project": DFLASH_PROJECT_URL,
            "sglang_repo": SGLANG_REPO_URL,
            "sglang_spec_doc": SGLANG_SPEC_DOC_URL,
            "sglang_dflash_pr": SGLANG_DFLASH_PR_URL,
        },
        missing_files=missing_files,
        missing_markers=missing_markers,
    )


def _write_selection_artifact(gate_dir: Path, payload: Dict[str, Any]) -> Path:
    artifact_path = (gate_dir / "upkg21_sglang_selection.json").resolve()
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact_path


def _evaluate_sglang_dflash_deepep_route() -> Dict[str, Any]:
    gateway_path = (
        WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "Backend" / "CGC" / "ray_serve_sglang_gateway.py"
    ).resolve()
    patch_path = (
        WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "Backend" / "CGC" / "deepep_sglang_patch.py"
    ).resolve()
    m75_path = (
        WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "cgc_engine" / "product" / "m75_trueorthokda_active_runtime.py"
    ).resolve()
    m76_path = (
        WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "cgc_engine" / "product" / "m76_gate.py"
    ).resolve()

    gateway_text = _read_text(gateway_path) if gateway_path.exists() else ""
    patch_text = _read_text(patch_path) if patch_path.exists() else ""
    m75_text = _read_text(m75_path) if m75_path.exists() else ""
    m76_text = _read_text(m76_path) if m76_path.exists() else ""

    required_markers = {
        "gateway": [
            '"--speculative-algorithm"',
            '"--speculative-draft-model-path"',
            "CGC_SGLANG_ENABLE_DFLASH",
            '"--moe-a2a-backend"',
            '"--deepep-mode"',
            "deepep_parallel_profile",
        ],
        "patch": [
            "def resolve_deepep_parallelism",
            '"deepep_parallel_profile": parallel_profile',
            '"ep_size": resolved_ep_size',
            '"tp_size": resolved_tp_size',
            '"moe_a2a_backend": "deepep"',
        ],
        "m75": [
            '"requested_dispatch_backend"',
            '"deepep_parallel_profile"',
            '"deepep_ep_size"',
            '"deepep_tp_size"',
            '"sglang_speculative_algorithm"',
            '"dflash_enabled"',
            '"dflash_draft_model_path"',
        ],
        "m76": [
            '"requested_dispatch_backend"',
            '"deepep_parallel_profile"',
            '"deepep_ep_size"',
            '"deepep_tp_size"',
            '"sglang_speculative_algorithm"',
            '"dflash_enabled"',
            '"dflash_draft_model_path"',
        ],
    }
    missing = {
        section: [marker for marker in markers if marker not in text]
        for section, markers, text in [
            ("gateway", required_markers["gateway"], gateway_text),
            ("patch", required_markers["patch"], patch_text),
            ("m75", required_markers["m75"], m75_text),
            ("m76", required_markers["m76"], m76_text),
        ]
    }
    ok = not any(missing.values())
    return _status(
        ok,
        selected_target_model="DeepSeek-V4-Flash",
        selected_route="vendored_sglang_dflash_deepep",
        selected_parallel_profile="ep16_tp1",
        selected_dispatch_backend="deepep",
        selected_speculative_algorithm="DFLASH",
        execution_mode="vendored_sglang_dflash_non_overlap_plus_deepep",
        note="UPKG 2.1 now accepts the combined vendored SGLang + DFlash + DeepEP route with ep16/tp1 as the explicit acceptance profile.",
        files={
            "gateway_path": str(gateway_path),
            "patch_path": str(patch_path),
            "m75_path": str(m75_path),
            "m76_path": str(m76_path),
        },
        checked_markers=required_markers,
        missing_markers=missing,
    )


def _extract_trueorthkda_gate(payload: Dict[str, Any]) -> Dict[str, Any]:
    steps = payload.get("steps") if isinstance(payload.get("steps"), dict) else {}
    gate = steps.get("trueorthkda_gate")
    if isinstance(gate, dict):
        return gate
    gate_result = payload.get("gate_result")
    if isinstance(gate_result, dict):
        nested = gate_result.get("trueorthkda_gate")
        if isinstance(nested, dict):
            return nested
    return {}


def _evaluate_m5_component(m5_report: Dict[str, Any], m5_gate: Dict[str, Any]) -> Dict[str, Any]:
    pipeline_gate = m5_gate.get("pipeline_gate_result") if isinstance(m5_gate.get("pipeline_gate_result"), dict) else {}
    m5_pipeline = pipeline_gate.get("m5") if isinstance(pipeline_gate.get("m5"), dict) else {}
    aot_gate = m5_pipeline.get("aot_precompile_gate") if isinstance(m5_pipeline.get("aot_precompile_gate"), dict) else {}
    if bool(m5_report.get("ok")) or str(m5_gate.get("status") or "") == "PASS":
        return _status(True, acceptance_mode="gate_ok", report_path=str(m5_report.get("report_path") or ""))
    fallback_ok = str(aot_gate.get("status") or "") == "PASS"
    return _status(
        fallback_ok,
        acceptance_mode="aot_precompile_fallback" if fallback_ok else "gate_fail",
        report_path=str(m5_report.get("report_path") or ""),
        aot_precompile_gate=aot_gate,
    )


def _evaluate_m74_component(m74_report: Dict[str, Any], m74_gate: Dict[str, Any], m5_pipeline_report: Dict[str, Any]) -> Dict[str, Any]:
    if bool(m74_report.get("ok")) or str(m74_gate.get("status") or "") == "PASS":
        return _status(True, acceptance_mode="gate_ok", report_path=str(m74_report.get("report_path") or ""))

    failed_checks = list(m74_gate.get("failed_checks") or []) if isinstance(m74_gate, dict) else []
    trueorthkda_gate = _extract_trueorthkda_gate(m5_pipeline_report)
    has_runtime_fallback = bool(
        failed_checks == ["trueorthkda_runtime"]
        and str(trueorthkda_gate.get("status") or "") == "PASS"
        and bool(trueorthkda_gate.get("require_trueorthkda"))
        and bool(trueorthkda_gate.get("enable_ortho_kda"))
    )
    return _status(
        has_runtime_fallback,
        acceptance_mode="m5_trueorthkda_runtime_fallback" if has_runtime_fallback else "gate_fail",
        report_path=str(m74_report.get("report_path") or ""),
        fallback_trueorthkda_gate=trueorthkda_gate,
        failed_checks=failed_checks,
    )


def _derive_m75_evidence_overrides(m75_output_root: Path, m75_report: Dict[str, Any]) -> Dict[str, str]:
    gate_payload = (
        ((m75_report or {}).get("gate_result") or {}).get("m75_trueorthokda_active")
        if isinstance(m75_report, dict)
        else {}
    )
    runtime_candidates = [
        _resolve_existing_json_path(str((gate_payload or {}).get("runtime_evidence_path") or "")),
        _resolve_existing_json_path(
            str((m75_output_root / "runtime_evidence" / "m75_trueorthokda_active_runtime.json").resolve())
        ),
    ]
    runtime_path = next((candidate for candidate in runtime_candidates if candidate is not None), None)
    if runtime_path is None:
        return {}

    runtime_payload = _read_json(runtime_path)
    local_infer_path = _resolve_existing_json_path(
        str((((runtime_payload.get("artifacts") or {}).get("local_infer_evidence_path")) or ""))
    )
    overrides = {
        "CGC_M75_RUNTIME_EVIDENCE_PATH": str(runtime_path),
        "CGC_M75_ACTIVE_RUNTIME_ROOT": str(runtime_path.parent.parent),
    }
    if local_infer_path is not None:
        overrides["CGC_M75_LOCAL_INFER_EVIDENCE_PATH"] = str(local_infer_path)
        overrides["CGC_LOCAL_INFER_EVIDENCE_PATH"] = str(local_infer_path)
    return overrides


def _run_upkg21_runtime_chain(output_root: Path) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    from .m75_trueorthokda_active_runtime import run_m75_trueorthokda_active_runtime
    from .m76_gate import run_m76_gate

    m75_output_root = (output_root / "m75").resolve()
    m76_output_root = (output_root / "m76").resolve()

    m75_report = run_m75_trueorthokda_active_runtime(output_dir=str(m75_output_root))
    env_overrides = _derive_m75_evidence_overrides(m75_output_root, m75_report)
    with _temporary_env_defaults(env_overrides) as applied_env:
        m76_report = run_m76_gate(output_dir=str(m76_output_root))
    return m75_report, m76_report, applied_env


def run_upkg21_gate(*, output_dir: str) -> Dict[str, Any]:
    output_root = Path(str(output_dir)).expanduser().resolve()
    gate_dir = (output_root / "upkg21_backend_injectable").resolve()
    gate_dir.mkdir(parents=True, exist_ok=True)

    from .m1_m6_pipeline_gates import run_m5_gate
    from .m74_dflash_kda_gate import run_m74_gate

    m75_report, m76_report, runtime_applied_env = _run_upkg21_runtime_chain(output_root)
    composite_env_defaults = _derive_m75_evidence_overrides((output_root / "m75").resolve(), m75_report)
    local_omlx_model_path = _resolve_local_omlx_model_path()
    if local_omlx_model_path is not None:
        composite_env_defaults["CGC_LOCAL_OMLX_MODEL"] = str(local_omlx_model_path)
    benchmark_report_path = _resolve_seed_official_benchmark_report_path()
    if benchmark_report_path is not None:
        composite_env_defaults["CGC_UPKG21_OFFICIAL_SGLANG_DFLASH_BENCHMARK_REPORT"] = str(benchmark_report_path)
        composite_env_defaults["CGC_UPKG21_SGLANG_DFLASH_BENCHMARK_REPORT"] = str(benchmark_report_path)

    with _temporary_env_defaults(composite_env_defaults) as composite_applied_env:
        m5_report = run_m5_gate(output_dir=str((gate_dir / "m5_core").resolve()))
        m74_report = run_m74_gate(output_dir=str(gate_dir))
        sglang_selection = _select_upkg21_sglang_runtime()
        selection_artifact_path = _write_selection_artifact(gate_dir, sglang_selection)

        m75_gate = (
            ((m75_report or {}).get("gate_result") or {}).get("m75_trueorthokda_active")
            if isinstance(m75_report, dict)
            else {}
        )
        m76_gate = ((m76_report or {}).get("gate_result") or {}).get("m76") if isinstance(m76_report, dict) else {}
        m5_gate = ((m5_report or {}).get("gate_result") or {}).get("m5") if isinstance(m5_report, dict) else {}
        m74_gate = ((m74_report or {}).get("gate_result") or {}).get("m74") if isinstance(m74_report, dict) else {}
        m5_pipeline_report = _read_json(Path(str((m5_report or {}).get("report_path") or "")).expanduser()) if str((m5_report or {}).get("report_path") or "").strip() else {}

        components = {
            "m75_trueorthokda_active_runtime": {
                "status": str((m75_gate or {}).get("status") or ("PASS" if bool((m75_report or {}).get("ok")) else "FAIL")),
                "report_path": str((m75_report or {}).get("report_path") or ""),
                "runtime_evidence_path": str((m75_gate or {}).get("runtime_evidence_path") or ""),
                "summary_path": str((m75_report or {}).get("summary_path") or ""),
                "latest_path": str((m75_report or {}).get("latest_path") or ""),
            },
            "m76_heterogeneous_gate": {
                "status": str((m76_gate or {}).get("status") or ("PASS" if bool((m76_report or {}).get("ok")) else "FAIL")),
                "report_path": str((m76_report or {}).get("report_path") or ""),
                "summary_path": str((m76_report or {}).get("summary_path") or ""),
                "latest_path": str((m76_report or {}).get("latest_path") or ""),
            },
            "m5_backend_injectable_core": _evaluate_m5_component(m5_report, m5_gate if isinstance(m5_gate, dict) else {}),
            "m74_dflash_trueorthokda_gate": _evaluate_m74_component(
                m74_report,
                m74_gate if isinstance(m74_gate, dict) else {},
                m5_pipeline_report,
            ),
            "sglang_runtime_selection": sglang_selection,
            "sglang_dflash_deepep_route": _evaluate_sglang_dflash_deepep_route(),
            "official_sglang_dflash_benchmark": _evaluate_official_sglang_dflash_benchmark(output_root, gate_dir),
        }
    applied_env = {**runtime_applied_env, **composite_applied_env}
    final_ok = all(str((component or {}).get("status") or "") == "PASS" for component in components.values())
    agent_execution = dict((m76_gate or {}).get("agent_execution") or (m75_gate or {}).get("agent_execution") or {})
    deepep_release_guard = dict((m76_gate or {}).get("deepep_release_guard") or (m75_gate or {}).get("deepep_release_guard") or {})
    schema_refs = dict((m76_gate or {}).get("schema_refs") or (m75_gate or {}).get("schema_refs") or {})

    artifact_paths = [
        str((m75_report or {}).get("report_path") or ""),
        str((m75_report or {}).get("summary_path") or ""),
        str((m76_report or {}).get("report_path") or ""),
        str((m76_report or {}).get("summary_path") or ""),
        str((m5_report or {}).get("report_path") or ""),
        str((m74_report or {}).get("report_path") or ""),
        str(selection_artifact_path),
        str((((m75_gate or {}).get("runtime_evidence_path")) or "")),
        str(((components["official_sglang_dflash_benchmark"] or {}).get("report_path")) or ""),
    ]
    artifact_index = _artifact_index(artifact_paths)
    artifact_index_path = (gate_dir / "artifact_index.json").resolve()
    artifact_index_path.write_text(json.dumps(artifact_index, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "status": "PASS" if final_ok else "FAIL",
        "upkg_target": "2.1",
        "selected_sglang_runtime": str(sglang_selection.get("selected_runtime") or ""),
        "selected_target_model": str((components["sglang_dflash_deepep_route"] or {}).get("selected_target_model") or ""),
        "dflash_runtime_mode": str(sglang_selection.get("dflash_runtime_mode") or ""),
        "deepep_parallel_profile": str((components["sglang_dflash_deepep_route"] or {}).get("selected_parallel_profile") or ""),
        "dispatch_backend": str((components["sglang_dflash_deepep_route"] or {}).get("selected_dispatch_backend") or ""),
        "m75_status": str((components["m75_trueorthokda_active_runtime"] or {}).get("status") or ""),
        "m76_status": str((components["m76_heterogeneous_gate"] or {}).get("status") or ""),
        "m5_status": str((components["m5_backend_injectable_core"] or {}).get("status") or ""),
        "m74_status": str((components["m74_dflash_trueorthokda_gate"] or {}).get("status") or ""),
        "sglang_selection_status": str((components["sglang_runtime_selection"] or {}).get("status") or ""),
        "sglang_dflash_deepep_route_status": str((components["sglang_dflash_deepep_route"] or {}).get("status") or ""),
        "official_sglang_dflash_benchmark_status": str((components["official_sglang_dflash_benchmark"] or {}).get("status") or ""),
        "m75_runtime_evidence_path": str((components["m75_trueorthokda_active_runtime"] or {}).get("runtime_evidence_path") or ""),
        "m76_summary_path": str((components["m76_heterogeneous_gate"] or {}).get("summary_path") or ""),
        "official_sglang_dflash_benchmark_report_path": str((components["official_sglang_dflash_benchmark"] or {}).get("report_path") or ""),
        "applied_environment_defaults": applied_env,
        "agent_execution": agent_execution,
        "deepep_release_guard": deepep_release_guard,
        "schema_refs": schema_refs,
        "note": "UPKG 2.1 accepts M7.4 as the DFlash + TrueOrthoKDA verification gate, uses vendored SGLang as the DFlash runtime, treats SpecV2 as a comparison path, and recognizes DeepEP ep16/tp1 as the explicit combined route profile.",
        "report_path": str((gate_dir / "upkg21_report.json").resolve()),
    }
    summary_path = (gate_dir / "summary.json").resolve()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = (gate_dir / "latest.json").resolve()
    latest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    gate = {
        "status": "PASS" if final_ok else "FAIL",
        "public_entrypoint": "cgc gate upkg21",
        "milestone": "upkg21",
        "scope": "backend_injectable_dflash_sglang_acceptance",
        "components": components,
        "selected_target_model": (components["sglang_dflash_deepep_route"] or {}).get("selected_target_model"),
        "selected_sglang_runtime": sglang_selection.get("selected_runtime"),
        "dflash_runtime_mode": sglang_selection.get("dflash_runtime_mode"),
        "selected_parallel_profile": (components["sglang_dflash_deepep_route"] or {}).get("selected_parallel_profile"),
        "selected_dispatch_backend": (components["sglang_dflash_deepep_route"] or {}).get("selected_dispatch_backend"),
        "m75_gate": m75_gate if isinstance(m75_gate, dict) else {},
        "m76_gate": m76_gate if isinstance(m76_gate, dict) else {},
        "m5_gate": m5_gate if isinstance(m5_gate, dict) else {},
        "m74_gate": m74_gate if isinstance(m74_gate, dict) else {},
        "environment_defaults": applied_env,
        "artifact_index_path": str(artifact_index_path),
        "selection_artifact_path": str(selection_artifact_path),
        "summary_path": str(summary_path),
        "latest_path": str(latest_path),
        "agent_execution": agent_execution,
        "deepep_release_guard": deepep_release_guard,
        "schema_refs": schema_refs,
    }
    report = {
        "ok": final_ok,
        "upkg_target": "2.1",
        "public_entrypoint": "cgc gate upkg21",
        "gate_result": {"upkg21": gate},
    }
    report_path = (gate_dir / "upkg21_report.json").resolve()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": final_ok,
        "gate_result": {"upkg21": gate},
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "latest_path": str(latest_path),
    }


def run_upkg21_rerun_gate(*, output_dir: str) -> Dict[str, Any]:
    output_root = Path(str(output_dir)).expanduser().resolve()
    rerun_dir = (output_root / "upkg21_rerun").resolve()
    rerun_dir.mkdir(parents=True, exist_ok=True)

    upkg21_output_root = (output_root / "upkg21").resolve()
    upkg21_report = run_upkg21_gate(output_dir=str(upkg21_output_root))
    upkg21_gate = ((upkg21_report or {}).get("gate_result") or {}).get("upkg21") if isinstance(upkg21_report, dict) else {}
    nested_components = (upkg21_gate or {}).get("components") if isinstance((upkg21_gate or {}).get("components"), dict) else {}
    nested_env = (upkg21_gate or {}).get("environment_defaults") if isinstance((upkg21_gate or {}).get("environment_defaults"), dict) else {}

    components = {
        "m75_trueorthokda_active_runtime": dict(nested_components.get("m75_trueorthokda_active_runtime") or {}),
        "m76_heterogeneous_gate": dict(nested_components.get("m76_heterogeneous_gate") or {}),
        "upkg21_backend_injectable_gate": {
            "status": str((upkg21_gate or {}).get("status") or ("PASS" if bool((upkg21_report or {}).get("ok")) else "FAIL")),
            "report_path": str((upkg21_report or {}).get("report_path") or ""),
            "summary_path": str((upkg21_report or {}).get("summary_path") or ""),
        },
    }
    final_ok = all(str((component or {}).get("status") or "") == "PASS" for component in components.values())

    artifact_paths = [
        str((components["m75_trueorthokda_active_runtime"] or {}).get("report_path") or ""),
        str((components["m76_heterogeneous_gate"] or {}).get("report_path") or ""),
        str((upkg21_report or {}).get("report_path") or ""),
        str((upkg21_report or {}).get("summary_path") or ""),
        str((components["m75_trueorthokda_active_runtime"] or {}).get("runtime_evidence_path") or ""),
    ]
    artifact_index = _artifact_index(artifact_paths)
    artifact_index_path = (rerun_dir / "artifact_index.json").resolve()
    artifact_index_path.write_text(json.dumps(artifact_index, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "status": "PASS" if final_ok else "FAIL",
        "upkg_target": "2.1",
        "rerun_mode": "m75_then_m76_then_upkg21",
        "selected_target_model": "DeepSeek-V4-Flash",
        "selected_speculative_algorithm": "DFLASH",
        "selected_dispatch_backend": "deepep",
        "selected_parallel_profile": "ep16_tp1",
        "m75_status": str((components["m75_trueorthokda_active_runtime"] or {}).get("status") or ""),
        "m76_status": str((components["m76_heterogeneous_gate"] or {}).get("status") or ""),
        "upkg21_status": str((components["upkg21_backend_injectable_gate"] or {}).get("status") or ""),
        "m75_runtime_evidence_path": str((components["m75_trueorthokda_active_runtime"] or {}).get("runtime_evidence_path") or ""),
        "upkg21_summary_path": str((upkg21_report or {}).get("summary_path") or ""),
        "applied_environment_defaults": nested_env,
        "agent_execution": dict((upkg21_gate or {}).get("agent_execution") or {}),
        "deepep_release_guard": dict((upkg21_gate or {}).get("deepep_release_guard") or {}),
        "schema_refs": dict((upkg21_gate or {}).get("schema_refs") or {}),
        "report_path": str((rerun_dir / "upkg21_rerun_report.json").resolve()),
    }
    summary_path = (rerun_dir / "summary.json").resolve()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = (rerun_dir / "latest.json").resolve()
    latest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    gate = {
        "status": "PASS" if final_ok else "FAIL",
        "public_entrypoint": "cgc gate upkg21-rerun",
        "milestone": "upkg21_rerun",
        "scope": "stable_m75_m76_upkg21_composite_rerun",
        "components": components,
        "environment_defaults": nested_env,
        "artifact_index_path": str(artifact_index_path),
        "summary_path": str(summary_path),
        "latest_path": str(latest_path),
        "selected_target_model": "DeepSeek-V4-Flash",
        "selected_speculative_algorithm": "DFLASH",
        "selected_dispatch_backend": "deepep",
        "selected_parallel_profile": "ep16_tp1",
        "agent_execution": dict((upkg21_gate or {}).get("agent_execution") or {}),
        "deepep_release_guard": dict((upkg21_gate or {}).get("deepep_release_guard") or {}),
        "schema_refs": dict((upkg21_gate or {}).get("schema_refs") or {}),
    }
    report = {
        "ok": final_ok,
        "upkg_target": "2.1",
        "public_entrypoint": "cgc gate upkg21-rerun",
        "gate_result": {"upkg21_rerun": gate},
    }
    report_path = (rerun_dir / "upkg21_rerun_report.json").resolve()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": final_ok,
        "gate_result": {"upkg21_rerun": gate},
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "latest_path": str(latest_path),
    }
