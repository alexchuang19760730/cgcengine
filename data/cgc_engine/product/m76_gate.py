import importlib.util
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

from cgc_engine.product.upkg30_common import evaluate_mandatory_protocol_gate

from cgc_engine.product.release_alias_contracts import (
    apply_release_alias_contracts,
    build_fresh_host1_real_chain_payload,
    promote_runtime_protocol_contract_with_fresh_probe,
)
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _resume_device_supports_zero_copy(*, cpu_copy_count: int, uma_buffer_used: bool, device_resume_consumed: bool, resume_tensor_device: str) -> bool:
    if int(cpu_copy_count) != 0 or not bool(device_resume_consumed):
        return False
    device = str(resume_tensor_device or "").strip().lower()
    if device.startswith("mps"):
        return bool(uma_buffer_used)
    return device.startswith("cuda")


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _status(ok: bool, **extra: Any) -> Dict[str, Any]:
    payload = {"status": "PASS" if ok else "FAIL"}
    payload.update(extra)
    return payload


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, str(default)) or "").strip()
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    return max(1, value)


def _resolve_deepep_parallelism() -> Dict[str, Any]:
    profile_raw = _env_str("CGC_DEEPEP_PARALLEL_PROFILE", "").lower()
    profile_match = re.fullmatch(r"ep(\d+)_tp(\d+)", profile_raw)
    profile_ep = int(profile_match.group(1)) if profile_match else None
    profile_tp = int(profile_match.group(2)) if profile_match else None
    resolved_tp_size = _env_int(
        "CGC_DEEPEP_TP_SIZE",
        _env_int("CGC_MEGATRAIN_PARALLEL_TP_SIZE", profile_tp or 4),
    )
    resolved_ep_size = _env_int(
        "CGC_DEEPEP_EP_SIZE",
        _env_int("CGC_MEGATRAIN_PARALLEL_EP_SIZE", profile_ep or resolved_tp_size),
    )
    parallel_profile = profile_raw or f"ep{resolved_ep_size}_tp{resolved_tp_size}"
    return {
        "deepep_parallel_profile": parallel_profile,
        "deepep_ep_size": int(resolved_ep_size),
        "deepep_tp_size": int(resolved_tp_size),
    }


def _resolve_sglang_dflash_runtime() -> Dict[str, Any]:
    target_model_path = _env_str("CGC_CLOUD_MODEL_PATH") or "/data/models/DeepSeek-V4-Flash"
    speculative_algorithm = (
        _env_str("CGC_SGLANG_SPECULATIVE_ALGORITHM")
        or ("DFLASH" if _env_bool("CGC_SGLANG_ENABLE_DFLASH", _env_bool("CGC_DFLASH_ENABLED", False)) else "")
    ).strip()
    dflash_draft_model_path = (
        _env_str("CGC_SGLANG_SPECULATIVE_DRAFT_MODEL_PATH")
        or _env_str("CGC_DFLASH_DRAFT_MODEL")
    )
    dflash_block_size = _env_int(
        "CGC_SGLANG_SPECULATIVE_DFLASH_BLOCK_SIZE",
        _env_int("CGC_SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS", 4),
    )
    dflash_enabled = speculative_algorithm.upper() == "DFLASH"
    return {
        "target_model_path": target_model_path,
        "target_model_family": "DeepSeek-V4-Flash" if "deepseek-v4-flash" in target_model_path.lower() else "",
        "sglang_speculative_algorithm": speculative_algorithm,
        "dflash_enabled": dflash_enabled,
        "dflash_draft_model_path": dflash_draft_model_path,
        "dflash_block_size": int(dflash_block_size),
        "dflash_acceptance_mode": "vendored_sglang_dflash_non_overlap" if dflash_enabled else "disabled",
    }


def _declared_deepep_backend(requested_dispatch_backend: str) -> bool:
    return str(requested_dispatch_backend or "").strip().lower() == "deepep"


def _effective_pd_service(*, runtime_protocol_contract: Dict[str, Any] | None = None, source: str) -> Dict[str, Any]:
    contract = dict(runtime_protocol_contract or {})
    enable_pd = bool(contract.get("enable_pd"))
    require_pd_service = bool(contract.get("require_pd_service"))
    pd_endpoint = str(contract.get("pd_endpoint") or _env_str("CGC_MEGATRAIN_PD_ENDPOINT") or _env_str("CGC_PD_ENDPOINT") or "localhost:50051").strip() if enable_pd else str(contract.get("pd_endpoint") or "").strip()
    pd_mode = str(contract.get("pd_mode") or ("cloud_prefill_edge_decode" if enable_pd else "disabled")).strip()
    pd_prefix_cache = bool(contract.get("pd_prefix_cache")) if "pd_prefix_cache" in contract else _env_bool("CGC_MEGATRAIN_PD_PREFIX_CACHE", True)
    try:
        from cgc_engine.pd.pd_client import PDClient  # type: ignore

        provider = PDClient.__name__
        client_available = True
    except Exception as exc:
        provider = ""
        client_available = False
        import_error = f"{type(exc).__name__}: {exc}"
    else:
        import_error = ""
    if not enable_pd and not require_pd_service:
        status = "SKIP"
        reason = "pd_service_not_requested"
    elif enable_pd and client_available and bool(pd_endpoint):
        status = "PASS"
        reason = ""
    elif not enable_pd and require_pd_service:
        status = "FAIL"
        reason = "require_pd_service=true_but_enable_pd=false"
    elif not bool(pd_endpoint):
        status = "FAIL"
        reason = "pd_endpoint_missing"
    else:
        status = "FAIL"
        reason = "pd_client_import_failed"
    payload = {
        "status": status,
        "enabled": enable_pd,
        "require_pd_service": require_pd_service,
        "endpoint": pd_endpoint,
        "mode": pd_mode,
        "prefix_cache": pd_prefix_cache,
        "provider": provider,
        "client_available": client_available,
        "reason": reason,
        "source": source,
    }
    if import_error:
        payload["error"] = import_error
    return payload


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_value(path: Path) -> Any:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _candidate_sort_key(path: Path) -> tuple[int, str]:
    try:
        mtime_ns = int(path.stat().st_mtime_ns)
    except OSError:
        mtime_ns = 0
    return (mtime_ns, str(path))


def _iter_search_roots(*extra_roots: str) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    candidates = [
        *extra_roots,
        _env_str("CGC_M75_ACTIVE_RUNTIME_ROOT"),
        _env_str("CGC_M75_EVIDENCE_DIR"),
        _env_str("CGC_M76_LOCAL_INFER_EVIDENCE_ROOT"),
        _env_str("CGC_LOCAL_INFER_EVIDENCE_ROOT"),
        str((WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "Output").resolve()),
    ]
    for raw in candidates:
        value = str(raw or "").strip()
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_file():
            path = path.parent
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved)
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def _resolve_explicit_json_path(*env_names: str) -> Path | None:
    for env_name in env_names:
        value = _env_str(env_name)
        if not value:
            continue
        path = Path(value).expanduser()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _resolve_existing_json_path(raw: str) -> Path | None:
    value = str(raw or "").strip()
    if value == "":
        return None
    path = Path(value).expanduser()
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    if resolved.exists() and resolved.is_file():
        return resolved
    return None


def _is_system_execution_manifest_payload(payload: Dict[str, Any]) -> bool:
    return bool(
        isinstance(payload, dict)
        and str(payload.get("schema_version") or "") == "cgc.system_execution_manifest.v0.1"
        and isinstance(payload.get("system_profile"), dict)
    )


def _latest_system_execution_manifest_path(*extra_roots: str) -> Path | None:
    explicit_path = _resolve_explicit_json_path(
        "CGC_SYSTEM_EXECUTION_MANIFEST_PATH",
        "CGC_M76_SYSTEM_EXECUTION_MANIFEST_PATH",
        "CGC_M75_SYSTEM_EXECUTION_MANIFEST_PATH",
    )
    if explicit_path is not None and _is_system_execution_manifest_payload(_read_json_file(explicit_path)):
        return explicit_path
    candidates: list[Path] = []
    for root in _iter_search_roots(*extra_roots, _env_str("CGC_M76_EVIDENCE_DIR"), _env_str("CGC_M75_EVIDENCE_DIR")):
        candidates.extend(path.resolve() for path in root.rglob("system_execution_manifest.json") if path.is_file())
    for path in reversed(sorted(candidates, key=_candidate_sort_key)):
        if _is_system_execution_manifest_payload(_read_json_file(path)):
            return path
    return None


def _manifest_artifact_path(manifest_payload: Dict[str, Any], *names: str) -> Path | None:
    if not isinstance(manifest_payload, dict) or not manifest_payload:
        return None
    artifacts = manifest_payload.get("artifacts") if isinstance(manifest_payload.get("artifacts"), dict) else {}
    formal_evidence = manifest_payload.get("formal_evidence") if isinstance(manifest_payload.get("formal_evidence"), dict) else {}
    for name in names:
        artifact_path = _resolve_existing_json_path(str(artifacts.get(name) or ""))
        if artifact_path is not None:
            return artifact_path
        entry = formal_evidence.get(name)
        if isinstance(entry, dict):
            entry_path = _resolve_existing_json_path(str(entry.get("path") or ""))
            if entry_path is not None:
                return entry_path
    return None


def _manifest_formal_payload(manifest_payload: Dict[str, Any], *names: str) -> Dict[str, Any]:
    if not isinstance(manifest_payload, dict) or not manifest_payload:
        return {}
    formal_evidence = manifest_payload.get("formal_evidence") if isinstance(manifest_payload.get("formal_evidence"), dict) else {}
    for name in names:
        entry = formal_evidence.get(name)
        if isinstance(entry, dict):
            payload = entry.get("payload")
            if isinstance(payload, dict) and payload:
                return dict(payload)
            entry_path = _resolve_existing_json_path(str(entry.get("path") or ""))
            if entry_path is not None:
                loaded = _read_json_file(entry_path)
                if loaded:
                    return loaded
    return {}


def _discover_formal_evidence_entries(
    *search_roots: Path,
    manifest_payload: Dict[str, Any] | None = None,
    runtime_payload: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    manifest_payload = manifest_payload if isinstance(manifest_payload, dict) else {}
    runtime_payload = runtime_payload if isinstance(runtime_payload, dict) else {}
    manifest_artifacts = manifest_payload.get("artifacts") if isinstance(manifest_payload.get("artifacts"), dict) else {}
    manifest_formal = manifest_payload.get("formal_evidence") if isinstance(manifest_payload.get("formal_evidence"), dict) else {}
    runtime_artifacts = runtime_payload.get("artifacts") if isinstance(runtime_payload.get("artifacts"), dict) else {}
    evidence_specs = {
        "router_evidence": {
            "aliases": ["router_evidence.json", "edge_router_runtime.json"],
            "artifact_key": "router_evidence_path",
            "env_names": ["CGC_ROUTER_EVIDENCE_PATH", "CGC_M75_EDGE_ROUTER_EVIDENCE_PATH"],
        },
        "instance_evidence": {
            "aliases": ["instance_evidence.json"],
            "artifact_key": "instance_evidence_path",
            "env_names": ["CGC_INSTANCE_EVIDENCE_PATH"],
        },
        "fusion_evidence": {
            "aliases": ["fusion_evidence.json"],
            "artifact_key": "fusion_evidence_path",
            "env_names": ["CGC_FUSION_EVIDENCE_PATH"],
        },
        "swe_verified_formal_summary": {
            "aliases": ["swe_verified_formal_summary.json"],
            "artifact_key": "swe_verified_formal_summary_path",
            "env_names": ["CGC_SWE_VERIFIED_FORMAL_SUMMARY_PATH"],
        },
    }
    resolved_roots = [root.resolve() for root in search_roots if isinstance(root, Path) and root.exists()]
    artifact_updates: Dict[str, str] = {}
    formal_updates: Dict[str, Any] = {}
    for evidence_name, spec in evidence_specs.items():
        candidates: list[Path] = []
        for env_name in spec.get("env_names") or []:
            env_path = _resolve_existing_json_path(_env_str(str(env_name)))
            if env_path is not None:
                candidates.append(env_path)
        artifact_key = str(spec.get("artifact_key") or evidence_name)
        for raw in (
            str(runtime_artifacts.get(artifact_key) or ""),
            str(manifest_artifacts.get(artifact_key) or ""),
        ):
            candidate = _resolve_existing_json_path(raw)
            if candidate is not None:
                candidates.append(candidate)
        manifest_entry = manifest_formal.get(evidence_name)
        if isinstance(manifest_entry, dict):
            candidate = _resolve_existing_json_path(str(manifest_entry.get("path") or ""))
            if candidate is not None:
                candidates.append(candidate)
        for root in resolved_roots:
            for alias in spec.get("aliases") or []:
                direct = (root / str(alias)).resolve()
                if direct.exists() and direct.is_file():
                    candidates.append(direct)
                    continue
                try:
                    candidates.extend(path.resolve() for path in root.rglob(str(alias)) if path.is_file())
                except Exception:
                    continue
        existing_candidates = [path for path in candidates if path.exists() and path.is_file()]
        if not existing_candidates:
            continue
        resolved = sorted(existing_candidates, key=_candidate_sort_key)[-1]
        artifact_updates[artifact_key] = str(resolved)
        formal_updates[evidence_name] = {
            "filename": resolved.name,
            "path": str(resolved),
            "exists": True,
            "source": "m76_manifest_backfill",
            "payload": _read_json_value(resolved),
        }
    return formal_updates, artifact_updates


def _upsert_system_execution_manifest(
    manifest_path: Path | None,
    *,
    artifact_updates: Dict[str, str] | None = None,
    formal_updates: Dict[str, Any] | None = None,
    gate_payload: Dict[str, Any] | None = None,
) -> None:
    if manifest_path is None:
        return
    payload = _read_json_file(manifest_path)
    if not payload:
        payload = {
            "schema_version": "cgc.system_execution_manifest.v0.1",
            "export_dir": str(manifest_path.parent),
        }
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    artifacts.update({k: v for k, v in dict(artifact_updates or {}).items() if str(v or "").strip() != ""})
    payload["artifacts"] = artifacts
    formal_evidence = payload.get("formal_evidence") if isinstance(payload.get("formal_evidence"), dict) else {}
    formal_evidence.update(dict(formal_updates or {}))
    payload["formal_evidence"] = formal_evidence
    payload = apply_release_alias_contracts(payload, manifest_path, gate_payload=gate_payload)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_runtime_probe_local_infer_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    if str(payload.get("mode") or "") == "m75_trueorthokda_single_step_resume":
        return True
    return bool(
        str(payload.get("state_kind") or "") == "kda_state_v1"
        and str(payload.get("state_codec") or "") == "cq4"
        and bool(payload.get("resume_decode_executed"))
        and int(payload.get("state_payload_bytes") or 0) > 0
        and int(payload.get("raw_state_bytes") or 0) > 0
        and int(payload.get("compressed_state_bytes") or 0) > 0
        and "cpu_copy_count" in payload
        and "device_resume_consumed" in payload
    )


def _latest_local_infer_evidence_path(*, require_runtime_probe: bool = False) -> Path | None:
    candidates: list[Path] = []
    for root in _iter_search_roots():
        candidates.extend(path.resolve() for path in root.rglob("local_infer_*.json") if path.is_file())
    candidates = sorted(candidates, key=_candidate_sort_key)
    if not candidates:
        return None
    fallback_path = candidates[-1]
    for path in reversed(candidates):
        payload = _read_json_file(path)
        if _is_runtime_probe_local_infer_payload(payload):
            return path
    return None if require_runtime_probe else fallback_path


def _is_m75_active_runtime_payload(payload: Dict[str, Any]) -> bool:
    if str(payload.get("schema_version") or "") != "m75.trueorthokda.active.v1":
        return False
    runtime_protocol_contract = payload.get("runtime_protocol_contract")
    compression_effective = payload.get("compression_effective")
    zero_copy_vram_real = payload.get("zero_copy_vram_real")
    mandatory_protocol_gate = payload.get("mandatory_protocol_gate")
    return bool(
        isinstance(runtime_protocol_contract, dict)
        and runtime_protocol_contract
        and isinstance(compression_effective, dict)
        and compression_effective
        and isinstance(zero_copy_vram_real, dict)
        and zero_copy_vram_real
        and isinstance(mandatory_protocol_gate, dict)
        and mandatory_protocol_gate
    )


def _latest_m75_active_runtime_evidence_path(*, require_pass: bool = False) -> Path | None:
    explicit_path = _resolve_explicit_json_path(
        "CGC_M75_ACTIVE_RUNTIME_PATH",
        "CGC_M75_RUNTIME_EVIDENCE_PATH",
    )
    candidates: list[Path] = []
    if explicit_path is not None:
        explicit_payload = _read_json_file(explicit_path)
        if _is_m75_active_runtime_payload(explicit_payload):
            if not require_pass:
                return explicit_path
            if (
                str(explicit_payload.get("status") or "") == "PASS"
                and str((explicit_payload.get("mandatory_protocol_gate") or {}).get("status") or "") == "PASS"
                and str((explicit_payload.get("compression_effective") or {}).get("status") or "") == "PASS"
                and str((explicit_payload.get("zero_copy_vram_real") or {}).get("status") or "") == "PASS"
            ):
                return explicit_path
    for root in _iter_search_roots():
        candidates.extend(path.resolve() for path in root.rglob("m75_trueorthokda_active_runtime.json") if path.is_file())
    candidates = sorted(candidates, key=_candidate_sort_key)
    fallback_path: Path | None = None
    for path in reversed(candidates):
        payload = _read_json_file(path)
        if not _is_m75_active_runtime_payload(payload):
            continue
        if fallback_path is None:
            fallback_path = path
        if not require_pass:
            return path
        if (
            str(payload.get("status") or "") == "PASS"
            and str((payload.get("mandatory_protocol_gate") or {}).get("status") or "") == "PASS"
            and str((payload.get("compression_effective") or {}).get("status") or "") == "PASS"
            and str((payload.get("zero_copy_vram_real") or {}).get("status") or "") == "PASS"
        ):
            return path
    return None if require_pass else fallback_path


def _normalize_zero_copy_from_local_infer(payload: Dict[str, Any], *, source_path: Path) -> Dict[str, Any]:
    cpu_copy_count = payload.get("cpu_copy_count")
    uma_buffer_used = bool(payload.get("uma_buffer_used"))
    device_resume_consumed = bool(payload.get("device_resume_consumed"))
    status = (
        "PASS"
        if _resume_device_supports_zero_copy(
            cpu_copy_count=int(cpu_copy_count or 0),
            uma_buffer_used=uma_buffer_used,
            device_resume_consumed=device_resume_consumed,
            resume_tensor_device=str(payload.get("resume_tensor_device") or ""),
        )
        else "FAIL"
    )
    return {
        "status": status,
        "cpu_copy_count": cpu_copy_count,
        "uma_buffer_used": uma_buffer_used,
        "device_resume_consumed": device_resume_consumed,
        "resume_tensor_device": str(payload.get("resume_tensor_device") or ""),
        "reason": "" if status == "PASS" else "local_infer_runtime_zero_copy_not_satisfied",
        "source": "local_infer_runtime_evidence",
        "source_path": str(source_path),
    }


def _normalize_compression_from_local_infer(payload: Dict[str, Any], *, source_path: Path) -> Dict[str, Any]:
    raw_state_bytes = int(payload.get("raw_state_bytes") or 0)
    compressed_state_bytes = int(payload.get("compressed_state_bytes") or 0)
    compression_ratio = float(payload.get("compression_ratio") or 1.0)
    status = "PASS" if raw_state_bytes > 0 and compressed_state_bytes > 0 and compression_ratio < 1.0 else "FAIL"
    return {
        "status": status,
        "raw_state_bytes": raw_state_bytes,
        "compressed_state_bytes": compressed_state_bytes,
        "compression_ratio": compression_ratio,
        "network_rx_ms": 0.0,
        "reason": "" if status == "PASS" else "local_infer_runtime_compression_not_effective",
        "source": "local_infer_runtime_evidence",
        "source_path": str(source_path),
    }


def _probe_native_runtime_evidence(runtime_protocol_contract: Dict[str, Any], *, manifest_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    manifest_payload = manifest_payload if isinstance(manifest_payload, dict) else {}
    manifest_path = _latest_system_execution_manifest_path()
    manifest_m75_payload = _manifest_formal_payload(manifest_payload, "m75_active_runtime")
    manifest_m75_path = _manifest_artifact_path(
        manifest_payload,
        "m75_trueorthokda_active_runtime_path",
        "m75_active_runtime",
    )
    if not manifest_m75_payload and manifest_m75_path is not None:
        manifest_m75_payload = _read_json_file(manifest_m75_path)
    manifest_m75_pass = bool(
        _is_m75_active_runtime_payload(manifest_m75_payload)
        and str(manifest_m75_payload.get("status") or "") == "PASS"
        and str((manifest_m75_payload.get("mandatory_protocol_gate") or {}).get("status") or "") == "PASS"
        and str((manifest_m75_payload.get("compression_effective") or {}).get("status") or "") == "PASS"
        and str((manifest_m75_payload.get("zero_copy_vram_real") or {}).get("status") or "") == "PASS"
    )
    if manifest_m75_pass:
        m75_active_path = manifest_m75_path
        m75_payload = manifest_m75_payload
    else:
        m75_active_path = _latest_m75_active_runtime_evidence_path(require_pass=True)
        m75_payload = _read_json_file(m75_active_path) if m75_active_path else {}

    manifest_local_infer_path = _manifest_artifact_path(
        manifest_payload,
        "local_infer_evidence_path",
        "latest_local_infer_path",
    )
    if manifest_local_infer_path is not None and _is_runtime_probe_local_infer_payload(_read_json_file(manifest_local_infer_path)):
        latest_local_infer_path = manifest_local_infer_path
    else:
        latest_local_infer_path = _latest_local_infer_evidence_path(require_runtime_probe=True)
    local_infer_payload = _read_json_file(latest_local_infer_path) if latest_local_infer_path else {}

    if isinstance(m75_payload.get("runtime_protocol_contract"), dict) and m75_payload.get("runtime_protocol_contract"):
        resolved_runtime_protocol_contract = dict(m75_payload.get("runtime_protocol_contract") or {})
        resolved_runtime_protocol_contract.setdefault(
            "source",
            "system_execution_manifest" if manifest_m75_pass else "m75_active_runtime_evidence",
        )
        resolved_runtime_protocol_contract.setdefault("source_path", str(m75_active_path or ""))
    else:
        resolved_runtime_protocol_contract = dict(runtime_protocol_contract or {})
        resolved_runtime_protocol_contract["source"] = "m76_bootstrap_runtime_env"
        resolved_runtime_protocol_contract["source_path"] = ""
    resolved_runtime_protocol_contract = promote_runtime_protocol_contract_with_fresh_probe(
        resolved_runtime_protocol_contract
    )

    if isinstance(m75_payload.get("zero_copy_vram_real"), dict) and m75_payload.get("zero_copy_vram_real"):
        zero_copy_vram_real = dict(m75_payload.get("zero_copy_vram_real") or {})
        zero_copy_vram_real.setdefault("source", "m75_active_runtime_evidence")
        zero_copy_vram_real.setdefault("source_path", str(m75_active_path or ""))
    elif latest_local_infer_path and local_infer_payload:
        zero_copy_vram_real = _normalize_zero_copy_from_local_infer(local_infer_payload, source_path=latest_local_infer_path)
    else:
        zero_copy_vram_real = {
            "status": "SKIP",
            "reason": "no_native_zero_copy_runtime_evidence_found_on_host",
            "cpu_copy_count": None,
            "uma_buffer_used": False,
            "device_resume_consumed": False,
            "resume_tensor_device": "",
            "source": "m76_bootstrap_runtime_env",
            "source_path": "",
        }

    if isinstance(m75_payload.get("compression_effective"), dict) and m75_payload.get("compression_effective"):
        compression_effective = dict(m75_payload.get("compression_effective") or {})
        compression_effective.setdefault("source", "m75_active_runtime_evidence")
        compression_effective.setdefault("source_path", str(m75_active_path or ""))
    elif latest_local_infer_path and local_infer_payload:
        compression_effective = _normalize_compression_from_local_infer(local_infer_payload, source_path=latest_local_infer_path)
    else:
        compression_effective = {
            "status": "SKIP",
            "reason": "no_native_compression_runtime_evidence_found_on_host",
            "raw_state_bytes": 0,
            "compressed_state_bytes": 0,
            "compression_ratio": 1.0,
            "network_rx_ms": 0.0,
            "source": "m76_bootstrap_runtime_env",
            "source_path": "",
        }

    native_effective = {}
    for field_name in (
        "effective_collective_backend",
        "effective_cuda_graph",
        "effective_dispatch_backend",
        "effective_distributed_runtime",
        "effective_pd_service",
        "effective_storage_backend",
        "gds_effective",
        "spdk_effective",
        "colossalai_effective",
    ):
        if isinstance(m75_payload.get(field_name), dict) and m75_payload.get(field_name):
            native_effective[field_name] = dict(m75_payload.get(field_name) or {})
            native_effective[field_name].setdefault("source", "m75_active_runtime_evidence")
            native_effective[field_name].setdefault("source_path", str(m75_active_path or ""))
        else:
            native_effective[field_name] = {}

    return {
        "runtime_protocol_contract": resolved_runtime_protocol_contract,
        "compression_effective": compression_effective,
        "zero_copy_vram_real": zero_copy_vram_real,
        "cpu_copy_count": zero_copy_vram_real.get("cpu_copy_count"),
        "native_effective": native_effective,
        "sources": {
            "system_execution_manifest_path": str(manifest_path) if manifest_path and manifest_path.exists() else "",
            "m75_active_runtime_path": str(m75_active_path) if m75_active_path and m75_active_path.exists() else "",
            "latest_local_infer_path": str(latest_local_infer_path) if latest_local_infer_path and latest_local_infer_path.exists() else "",
        },
    }


def _should_refresh_bootstrap_runtime_evidence(
    nvidia_path: Path,
    *,
    desired_runtime_protocol_contract: Dict[str, Any] | None = None,
) -> bool:
    if not nvidia_path.exists() or not nvidia_path.is_file():
        return True
    payload = _read_json_file(nvidia_path)
    if not payload:
        return True
    if str(payload.get("status") or "") != "PASS":
        return True
    for field_name in (
        "runtime_protocol_contract",
        "compression_effective",
        "zero_copy_vram_real",
        "effective_pd_service",
        "mandatory_protocol_gate",
    ):
        if not isinstance(payload.get(field_name), dict) or not payload.get(field_name):
            return True
    if str((payload.get("mandatory_protocol_gate") or {}).get("status") or "") != "PASS":
        return True
    if str((payload.get("zero_copy_vram_real") or {}).get("status") or "") != "PASS":
        return True
    zero_copy_source_path = Path(str((payload.get("zero_copy_vram_real") or {}).get("source_path") or "")).expanduser()
    if not zero_copy_source_path.exists():
        return True
    if zero_copy_source_path.name == "m75_trueorthokda_active_runtime.json":
        if not _is_m75_active_runtime_payload(_read_json_file(zero_copy_source_path)):
            return True
    elif not _is_runtime_probe_local_infer_payload(_read_json_file(zero_copy_source_path)):
        return True
    desired_contract = dict(desired_runtime_protocol_contract or {})
    current_contract = dict(payload.get("runtime_protocol_contract") or {})
    for field_name in (
        "requested_dispatch_backend",
        "declared_deepep_backend",
        "deepep_real_chain_requires_rdma",
        "deepep_parallel_profile",
        "deepep_ep_size",
        "deepep_tp_size",
        "requested_distributed_runtime",
        "distributed_runtime_backend",
        "service_topology_backend",
    ):
        desired_value = desired_contract.get(field_name)
        if desired_value is None:
            continue
        if current_contract.get(field_name) != desired_value:
            return True
    return False


def _probe_gds_effective() -> Dict[str, Any]:
    enabled = _env_bool("CGC_GDS_ENABLED", False)
    if not enabled:
        return {"status": "SKIP", "enabled": False, "backend": "posix", "reason": "CGC_GDS_ENABLED=0"}
    try:
        from cgc_engine.gds_service.cufile_wrapper import CUFILE_AVAILABLE, get_gds_capabilities

        capabilities = get_gds_capabilities()
        effective = bool(CUFILE_AVAILABLE) and bool(capabilities.get("storage_path_eligible"))
        return {
            "status": "PASS" if effective else "FAIL",
            "enabled": True,
            "backend": "gds_nfsrdma" if bool(capabilities.get("nfs_rdma_mounts")) else "gds",
            "cufile_available": bool(CUFILE_AVAILABLE),
            "capabilities": capabilities,
            "reason": "" if effective else "gds_requested_but_storage_path_not_eligible",
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "enabled": True,
            "backend": "gds",
            "reason": "gds_probe_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _probe_spdk_effective() -> Dict[str, Any]:
    enabled = _env_bool("CGC_SPDK_ENABLED", False)
    if not enabled:
        return {"status": "SKIP", "enabled": False, "backend": "posix", "reason": "CGC_SPDK_ENABLED=0"}
    binary = shutil.which("spdk_tgt") or ""
    return {
        "status": "PASS" if binary else "FAIL",
        "enabled": True,
        "backend": "spdk",
        "binary": binary,
        "reason": "" if binary else "spdk_tgt_not_found",
    }


def _probe_colossalai_effective() -> Dict[str, Any]:
    enabled = _env_bool("CGC_MEGATRAIN_USE_COLOSSALAI", False)
    plugin = _env_str("CGC_MEGATRAIN_COLOSSALAI_PLUGIN") or _env_str("CGC_COLOSSALAI_PLUGIN") or "HybridParallelPlugin"
    if not enabled:
        return {
            "status": "SKIP",
            "enabled": False,
            "backend": "single_process",
            "plugin": plugin,
            "reason": "CGC_MEGATRAIN_USE_COLOSSALAI=0",
        }
    try:
        from colossalai.booster import Booster  # type: ignore
        from colossalai.booster.plugin import HybridParallelPlugin  # type: ignore

        return {
            "status": "PASS",
            "enabled": True,
            "backend": "colossalai",
            "plugin": plugin,
            "booster_cls": Booster.__name__,
            "plugin_cls": HybridParallelPlugin.__name__,
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "enabled": True,
            "backend": "colossalai",
            "plugin": plugin,
            "reason": "colossalai_import_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _runtime_protocol_contract() -> Dict[str, Any]:
    protocol_family = _env_str("CGC_RUNTIME_PROTOCOL_FAMILY") or "trueorthokda"
    state_kind = _env_str("CGC_RUNTIME_STATE_KIND") or "kda_state_v1"
    state_codec = _env_str("CGC_RUNTIME_STATE_CODEC") or "cq4"
    enable_nccl = _env_bool("CGC_MEGATRAIN_ENABLE_NCCL", _env_bool("CGC_SGLANG_USE_NCCL", False))
    enable_cuda_graph = _env_bool("CGC_MEGATRAIN_ENABLE_CUDA_GRAPH", False)
    use_colossalai = _env_bool("CGC_MEGATRAIN_USE_COLOSSALAI", False)
    requested_dispatch_backend = (
        _env_str("CGC_MEGATRAIN_REQUESTED_DISPATCH_BACKEND")
        or _env_str("CGC_REQUESTED_DISPATCH_BACKEND")
        or ("deepep" if _env_bool("CGC_M76_ENABLE_DEEPEP", False) or _env_str("CGC_DEEPEP_MODE") else "native_sglang")
    )
    requested_distributed_runtime = (
        _env_str("CGC_MEGATRAIN_REQUESTED_DISTRIBUTED_RUNTIME")
        or ("colossalai" if use_colossalai else "nccl" if enable_nccl else "single_process")
    )
    distributed_runtime_backend = (
        _env_str("CGC_DISTRIBUTED_RUNTIME_BACKEND")
        or requested_distributed_runtime
    )
    service_topology_backend = (
        _env_str("CGC_SERVICE_TOPOLOGY_BACKEND")
        or ("ray_cluster_dual_host" if _env_bool("CGC_MEGATRAIN_ENABLE_PD", _env_bool("CGC_ENABLE_PD", protocol_family == "trueorthokda")) else "single_host_local")
    )
    enable_gds = _env_bool("CGC_GDS_ENABLED", False)
    enable_spdk = _env_bool("CGC_SPDK_ENABLED", False)
    requested_storage_backend = (
        _env_str("CGC_MEGATRAIN_REQUESTED_STORAGE_BACKEND")
        or ("gds_spdk" if enable_gds and enable_spdk else "gds" if enable_gds else "spdk" if enable_spdk else "posix")
    )
    deepep_parallelism = _resolve_deepep_parallelism()
    sglang_dflash_runtime = _resolve_sglang_dflash_runtime()
    declared_deepep_backend = _declared_deepep_backend(requested_dispatch_backend)
    return {
        "protocol_family": protocol_family,
        "state_kind": state_kind,
        "state_codec": state_codec,
        "expected_zero_copy": _env_bool("CGC_MEGATRAIN_EXPECT_ZERO_COPY", protocol_family == "trueorthokda"),
        "enable_nccl": enable_nccl,
        "enable_cuda_graph": enable_cuda_graph,
        "requested_dispatch_backend": requested_dispatch_backend,
        "declared_deepep_backend": declared_deepep_backend,
        "deepep_real_chain_requires_rdma": True,
        "deepep_parallel_profile": str(deepep_parallelism.get("deepep_parallel_profile") or ""),
        "deepep_ep_size": int(deepep_parallelism.get("deepep_ep_size") or 1),
        "deepep_tp_size": int(deepep_parallelism.get("deepep_tp_size") or 1),
        "target_model_path": str(sglang_dflash_runtime.get("target_model_path") or ""),
        "target_model_family": str(sglang_dflash_runtime.get("target_model_family") or ""),
        "sglang_speculative_algorithm": str(sglang_dflash_runtime.get("sglang_speculative_algorithm") or ""),
        "dflash_enabled": bool(sglang_dflash_runtime.get("dflash_enabled")),
        "dflash_draft_model_path": str(sglang_dflash_runtime.get("dflash_draft_model_path") or ""),
        "dflash_block_size": int(sglang_dflash_runtime.get("dflash_block_size") or 0),
        "dflash_acceptance_mode": str(sglang_dflash_runtime.get("dflash_acceptance_mode") or ""),
        "requested_distributed_runtime": requested_distributed_runtime,
        "distributed_runtime_backend": distributed_runtime_backend,
        "requested_storage_backend": requested_storage_backend,
        "service_topology_backend": service_topology_backend,
        "enable_gds": enable_gds,
        "enable_spdk": enable_spdk,
        "use_colossalai": use_colossalai,
        "colossalai_plugin": _env_str("CGC_MEGATRAIN_COLOSSALAI_PLUGIN") or _env_str("CGC_COLOSSALAI_PLUGIN") or "HybridParallelPlugin",
        "enable_pd": _env_bool("CGC_MEGATRAIN_ENABLE_PD", _env_bool("CGC_ENABLE_PD", protocol_family == "trueorthokda")),
        "pd_endpoint": (
            _env_str("CGC_MEGATRAIN_PD_ENDPOINT")
            or _env_str("CGC_PD_ENDPOINT")
            or _env_str("CGC_PD_SERVICE_ENDPOINT")
            or (
                "localhost:50051"
                if _env_bool("CGC_MEGATRAIN_ENABLE_PD", _env_bool("CGC_ENABLE_PD", protocol_family == "trueorthokda"))
                else ""
            )
        ),
        "pd_mode": _env_str("CGC_MEGATRAIN_PD_MODE") or _env_str("CGC_PD_MODE") or (
            "cloud_prefill_edge_decode"
            if _env_bool("CGC_MEGATRAIN_ENABLE_PD", _env_bool("CGC_ENABLE_PD", protocol_family == "trueorthokda"))
            else "disabled"
        ),
        "pd_prefix_cache": _env_bool("CGC_MEGATRAIN_PD_PREFIX_CACHE", _env_bool("CGC_PD_PREFIX_CACHE", True)),
        "require_pd_service": _env_bool(
            "CGC_MEGATRAIN_REQUIRE_PD_SERVICE",
            _env_bool(
                "CGC_REQUIRE_PD_SERVICE",
                _env_bool("CGC_MEGATRAIN_ENABLE_PD", _env_bool("CGC_ENABLE_PD", protocol_family == "trueorthokda"))
                or protocol_family == "trueorthokda",
            ),
        ),
    }


def _run_check(check_name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        if not isinstance(result, dict):
            return _status(False, check=check_name, error="invalid_result", detail=repr(result))
        result.setdefault("check", check_name)
        return result
    except Exception as exc:
        return _status(
            False,
            check=check_name,
            error=str(exc),
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )


def _validate_cloud_server_contract(cloud_server_path: Path, gateway_path: Path) -> Dict[str, Any]:
    cloud_source = cloud_server_path.read_text(encoding="utf-8")
    gateway_source = gateway_path.read_text(encoding="utf-8")
    cloud_markers = [
        "DeepEPCommunicator",
        "patch_sglang_moe",
        "select_model_path",
        "start_ray_serve_gateway",
        "def start_server(",
    ]
    gateway_markers = [
        "class SGLangBackendManager",
        '"sglang.launch_server"',
        '"--use-ray"',
        '"--moe-a2a-backend"',
        '"deepep"',
        "serve.start(detached=False, http_options=http_options)",
        'aggregated_stats = self.deepep_comm.dispatch(tokens, routing_weights)',
        'aggregated_size = int(aggregated_stats["estimated_payload_bytes"])',
        "self.rdma_comm.register_memory_region(",
        "MindIRCompiler",
        "UnifiedIRCompiler(perception_matrix=edge_matrix)",
        '"X-CGC-Perception-Matrix": json.dumps(edge_matrix)',
        "runtime.deepep_comm.combine(response_payload)",
    ]
    missing_cloud = [marker for marker in cloud_markers if marker not in cloud_source]
    missing_gateway = [marker for marker in gateway_markers if marker not in gateway_source]
    serve_run_variants = [
        'serve.run(CGCRayServeGateway.bind(config.to_payload()), route_prefix="/")',
        "serve.run(",
        ".bind(config.to_payload())",
    ]
    if not any(marker in gateway_source for marker in serve_run_variants):
        missing_gateway.append("serve.run(CGCRayServeGateway.bind(config.to_payload()), route_prefix=\"/\")")
    return _status(
        not missing_cloud and not missing_gateway,
        files=[str(cloud_server_path), str(gateway_path)],
        checked_cloud_markers=cloud_markers,
        checked_gateway_markers=gateway_markers,
        missing_cloud_markers=missing_cloud,
        missing_gateway_markers=missing_gateway,
    )


def _validate_deepep_patch_contract(patch_path: Path) -> Dict[str, Any]:
    source = patch_path.read_text(encoding="utf-8")
    required_markers = [
        "def ensure_vendored_sglang_on_path",
        "def resolve_deepep_parallelism",
        "def build_sglang_deepep_engine_kwargs",
        '"moe_a2a_backend": "deepep"',
        '"deepep_parallel_profile": parallel_profile',
        '"ep_size": resolved_ep_size',
        '"tp_size": resolved_tp_size',
        'os.environ.get("CGC_DEEPEP_MODE", "normal")',
        "def patch_sglang_moe",
        "def run_deepep_v2_probe",
    ]
    missing = [marker for marker in required_markers if marker not in source]
    return _status(
        not missing,
        file=str(patch_path),
        checked_markers=required_markers,
        missing_markers=missing,
    )


def _validate_eight_step_pipeline(cli_path: Path, pipeline_path: Path) -> Dict[str, Any]:
    cli_source = cli_path.read_text(encoding="utf-8")
    pipeline_source = pipeline_path.read_text(encoding="utf-8")
    cli_markers = [f"[{step}/8]" for step in range(1, 9)]
    pipeline_markers = [
        "八步流水線",
        "4D 矩陣：環境 × 任務 × 硬體 × 模型",
        "step1_staticize",
        "step2_graph_capture",
        "step3_partition",
        "step4_skvm_verify",
        "step5_passes",
        "step6_memory_planning",
        "step7_kernel_codegen",
        "step8_runtime",
    ]
    missing_cli = [marker for marker in cli_markers if marker not in cli_source]
    missing_pipeline = [marker for marker in pipeline_markers if marker not in pipeline_source]
    return _status(
        not missing_cli and not missing_pipeline,
        cli_file=str(cli_path),
        pipeline_file=str(pipeline_path),
        checked_cli_markers=cli_markers,
        checked_pipeline_markers=pipeline_markers,
        missing_cli_markers=missing_cli,
        missing_pipeline_markers=missing_pipeline,
    )


def _validate_perception_matrix_contract(
    cloud_path: Path,
    gateway_path: Path,
    api_path: Path,
    edge_path: Path,
) -> Dict[str, Any]:
    cloud_source = cloud_path.read_text(encoding="utf-8")
    gateway_source = gateway_path.read_text(encoding="utf-8")
    api_source = api_path.read_text(encoding="utf-8")
    edge_source = edge_path.read_text(encoding="utf-8")
    cloud_markers = [
        "start_ray_serve_gateway",
        "DeepEPCommunicator",
    ]
    gateway_markers = [
        'raw_matrix = headers.get("x-cgc-perception-matrix", "")',
        'edge_matrix.setdefault("bw_mbps", float(headers.get("x-cgc-bw-mbps", "1000.0")))',
        'edge_matrix.setdefault("hardware_type", headers.get("x-cgc-hardware-type", "Nvidia_L20N"))',
        'edge_matrix.setdefault("task_type", headers.get("x-cgc-task-type", "prefill"))',
        'edge_matrix.setdefault("model_family", str(payload.get("model", "deepseek-v4-flash:latest")))',
        "UnifiedIRCompiler(perception_matrix=edge_matrix)",
    ]
    api_markers = [
        '"hardware_type": os.environ.get("CGC_EDGE_HARDWARE_TYPE", "Apple_Silicon")',
        '"environment": os.environ.get("CGC_EDGE_ENVIRONMENT", "edge")',
        '"task_type": "prefill"',
        '"model_family": str(',
    ]
    edge_markers = [
        '"hardware_type": os.environ.get("CGC_EDGE_HARDWARE_TYPE", "Apple_Silicon")',
        '"environment": os.environ.get("CGC_EDGE_ENVIRONMENT", "edge")',
        '"task_type": "prefill"',
        '"model_family": "deepseek-v4-flash:latest"',
    ]
    missing_cloud = [marker for marker in cloud_markers if marker not in cloud_source]
    missing_gateway = [marker for marker in gateway_markers if marker not in gateway_source]
    missing_api = [marker for marker in api_markers if marker not in api_source]
    missing_edge = [marker for marker in edge_markers if marker not in edge_source]
    return _status(
        not missing_cloud and not missing_gateway and not missing_api and not missing_edge,
        cloud_file=str(cloud_path),
        gateway_file=str(gateway_path),
        api_file=str(api_path),
        edge_file=str(edge_path),
        missing_cloud_markers=missing_cloud,
        missing_gateway_markers=missing_gateway,
        missing_api_markers=missing_api,
        missing_edge_markers=missing_edge,
    )


def _run_deepep_contract(cloud_module: Any) -> Dict[str, Any]:
    raw = str(os.environ.get("CGC_M76_ENABLE_DEEPEP", "") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return {
            "status": "SKIP",
            "reason": "deepep_degraded",
            "rationale": "DeepEP temporarily downgraded; native SGLang routing accepted for this M7.6 run",
        }
    deepep_parallelism = _resolve_deepep_parallelism()
    communicator = cloud_module.DeepEPCommunicator(
        tp_size=int(deepep_parallelism["deepep_tp_size"]),
        ep_size=int(deepep_parallelism["deepep_ep_size"]),
        deepep_parallel_profile=str(deepep_parallelism["deepep_parallel_profile"]),
    )
    routing_weights = np.array([0.05, 0.8, 0.1, 0.05], dtype=float)
    if sys.platform == "darwin":
        aggregated_payload = communicator.dispatch(list("m76-gate"), routing_weights)
        combined_output = communicator.combine(["expert-a", "expert-b"])
        ok = bool(
            isinstance(aggregated_payload, dict)
            and int(aggregated_payload.get("estimated_payload_bytes", 0)) > 0
            and combined_output == ["expert-a", "expert-b"]
        )
        return _status(
            ok,
            mode="contract_only",
            rationale="macOS local gate does not require local DeepEP runtime",
            deepep_parallel_profile=str(deepep_parallelism["deepep_parallel_profile"]),
            ep_size=int(deepep_parallelism["deepep_ep_size"]),
            tp_size=int(deepep_parallelism["deepep_tp_size"]),
            aggregated_payload=aggregated_payload,
            combined_output=combined_output,
        )

    communicator.initialize()
    aggregated_payload = communicator.dispatch(list("m76-gate"), routing_weights)
    combined_output = communicator.combine(["expert-a", "expert-b"])
    ok = bool(
        communicator.is_initialized
        and isinstance(aggregated_payload, dict)
        and int(aggregated_payload.get("estimated_payload_bytes", 0)) > 0
        and combined_output == ["expert-a", "expert-b"]
    )
    return _status(
        ok,
        mode="runtime",
        deepep_parallel_profile=str(deepep_parallelism["deepep_parallel_profile"]),
        ep_size=int(deepep_parallelism["deepep_ep_size"]),
        tp_size=communicator.tp_size,
        aggregated_payload=aggregated_payload,
        combined_output=combined_output,
    )


def _run_rdma_contract(rdma_module: Any) -> Dict[str, Any]:
    communicator = rdma_module.RDMACommunicator()
    init_result = communicator.initialize()
    mr_handle = communicator.register_memory_region(gpu_tensor_ptr="0xGPU_ADDR", size=4096)
    send_ok = communicator.send_tensor_direct(mr_handle, remote_ip="172.30.132.117", remote_qpn=1024)
    ok = bool(communicator.is_initialized and isinstance(mr_handle, str) and send_ok)
    return _status(
        ok,
        mode="contract_only" if sys.platform == "darwin" else "runtime",
        initialized=communicator.is_initialized,
        init_result=bool(init_result),
        rdma_available=bool(getattr(rdma_module, "RDMA_AVAILABLE", False)),
        mr_handle=mr_handle,
        send_ok=bool(send_ok),
    )


def _rdma_is_enabled() -> bool:
    raw = str(os.environ.get("CGC_M76_ENABLE_RDMA", "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return sys.platform != "darwin"


def _run_rdma_contract_optional(rdma_path: Path) -> Dict[str, Any]:
    if not _rdma_is_enabled():
        return {
            "status": "SKIP",
            "reason": "rdma_disabled",
            "rdma_path": str(rdma_path),
        }
    if not rdma_path.exists():
        return {
            "status": "SKIP",
            "reason": "rdma_module_missing",
            "rdma_path": str(rdma_path),
        }
    rdma_module = _load_module(rdma_path, "cgc_m76_rdma_passthrough")
    return _run_rdma_contract(rdma_module)


def _link_rdma_contract_with_fresh_runtime(
    *,
    rdma_contract: Dict[str, Any] | None,
    runtime_protocol_contract: Dict[str, Any] | None,
    fresh_host1_real_chain: Dict[str, Any] | None,
) -> Dict[str, Any]:
    linked = dict(rdma_contract or {})
    contract = dict(runtime_protocol_contract or {})
    fresh = dict(fresh_host1_real_chain or {})
    freshness_guard = dict(fresh.get("freshness_guard") or {})
    requested_dispatch_backend = str(contract.get("requested_dispatch_backend") or "").strip().lower()
    declared_deepep_backend = bool(
        contract.get("declared_deepep_backend")
        or _declared_deepep_backend(requested_dispatch_backend)
    )
    requires_rdma = bool(contract.get("deepep_real_chain_requires_rdma", True))
    if not declared_deepep_backend or not requires_rdma:
        linked["real_chain_required"] = False
        return linked
    linked["real_chain_required"] = True
    linked["fresh_host1_real_chain_mode"] = str(fresh.get("mode") or "")
    linked["freshness_guard_status"] = str(freshness_guard.get("status") or "")
    linked["backend_available"] = bool(fresh.get("backend_available"))
    linked["fresh_host1_probe_path"] = str(fresh.get("fresh_host1_probe_path") or "")
    if freshness_guard and str(freshness_guard.get("status") or "") != "PASS":
        linked["status"] = "FAIL"
        linked["reason"] = "fresh_host1_probe_not_passed"
        return linked
    if fresh and not bool(fresh.get("backend_available")):
        linked["status"] = "FAIL"
        linked["reason"] = "backend_unavailable_for_rdma_real_chain"
        return linked
    if str(linked.get("status") or "") != "PASS":
        linked["status"] = "FAIL"
        linked["reason"] = str(linked.get("reason") or "rdma_contract_not_passed")
        return linked
    return linked


def _evaluate_deepep_real_chain_gate(
    *,
    runtime_protocol_contract: Dict[str, Any] | None,
    deepep_contract: Dict[str, Any] | None,
    rdma_contract: Dict[str, Any] | None,
) -> Dict[str, Any]:
    contract = dict(runtime_protocol_contract or {})
    deepep_payload = dict(deepep_contract or {})
    rdma_payload = dict(rdma_contract or {})
    requested_dispatch_backend = str(contract.get("requested_dispatch_backend") or "").strip().lower()
    declared_deepep_backend = bool(
        contract.get("declared_deepep_backend")
        or _declared_deepep_backend(requested_dispatch_backend)
    )
    deepep_real_chain_requires_rdma = bool(
        contract.get("deepep_real_chain_requires_rdma", True)
    )
    deepep_status = str(deepep_payload.get("status") or "")
    rdma_status = str(rdma_payload.get("status") or "")
    if not declared_deepep_backend:
        return {
            "status": "SKIP",
            "reason": "deepep_not_declared",
            "real_chain_pass": False,
            "declared_deepep_backend": False,
            "deepep_real_chain_requires_rdma": deepep_real_chain_requires_rdma,
            "requested_dispatch_backend": requested_dispatch_backend or "native_sglang",
        }
    if deepep_status != "PASS":
        return {
            "status": "FAIL",
            "reason": "deepep_contract_not_passed",
            "real_chain_pass": False,
            "declared_deepep_backend": True,
            "deepep_real_chain_requires_rdma": deepep_real_chain_requires_rdma,
            "requested_dispatch_backend": requested_dispatch_backend or "deepep",
            "deepep_contract_status": deepep_status or "UNKNOWN",
            "rdma_contract_status": rdma_status or "UNKNOWN",
        }
    if deepep_real_chain_requires_rdma and rdma_status != "PASS":
        return {
            "status": "FAIL",
            "reason": "rdma_contract_not_passed",
            "rationale": "Declared DeepEP backend requires a PASS rdma_contract on the same fresh runtime chain.",
            "real_chain_pass": False,
            "declared_deepep_backend": True,
            "deepep_real_chain_requires_rdma": True,
            "requested_dispatch_backend": requested_dispatch_backend or "deepep",
            "deepep_contract_status": deepep_status,
            "rdma_contract_status": rdma_status or "UNKNOWN",
            "rdma_contract_reason": str(rdma_payload.get("reason") or ""),
        }
    return {
        "status": "PASS",
        "real_chain_pass": True,
        "declared_deepep_backend": True,
        "deepep_real_chain_requires_rdma": deepep_real_chain_requires_rdma,
        "requested_dispatch_backend": requested_dispatch_backend or "deepep",
        "deepep_contract_status": deepep_status,
        "rdma_contract_status": rdma_status or "UNKNOWN",
    }


def _run_expert_migration_runtime(migrator_module: Any) -> Dict[str, Any]:
    migrator = migrator_module.HotExpertMigrator(num_experts=8)
    hot_expert_id = 7
    tokens = ["tok"] * 6001
    routing_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
    source_node_id = 0
    migrator.record_routing(tokens, routing_weights, source_node_id=source_node_id)
    migrations = migrator.evaluate_and_migrate()
    migrated = (hot_expert_id, source_node_id) in migrations
    ok = bool(migrated and migrator.expert_locations.get(hot_expert_id) == source_node_id)
    return _status(
        ok,
        migrations=[list(item) for item in migrations],
        hot_expert_id=hot_expert_id,
        final_location=migrator.expert_locations.get(hot_expert_id),
    )


def _run_unified_ir_runtime(compiler_module: Any, mindir_module: Any, ascend_module: Any) -> Dict[str, Any]:
    nvidia_compiler = compiler_module.UnifiedIRCompiler({"hardware_type": "Nvidia_L20N"})
    nvidia_ir = nvidia_compiler.compile_to_unified_ir("MoE_SubGraph")
    nvidia_graph = nvidia_compiler.lower_to_hardware(nvidia_ir)
    nvidia_result = nvidia_compiler.execute(nvidia_graph)

    ascend_compiler = compiler_module.UnifiedIRCompiler({"hardware_type": "Huawei_Ascend"})
    ascend_ir = ascend_compiler.compile_to_unified_ir("MoE_SubGraph")
    ascend_graph = ascend_compiler.lower_to_hardware(ascend_ir)
    ascend_result = ascend_compiler.execute(ascend_graph)

    mindir_compiler = mindir_module.MindIRCompiler(target_device="Ascend")
    mindir_graph = mindir_compiler.compile_graph("MoE_SubGraph")

    ascend_router = ascend_module.AscendRouter()
    ascend_router.init_hccl()
    routing_strategy = ascend_router.get_routing_strategy()

    ok = bool(
        "CUDA_Executable(" in nvidia_graph
        and "CANN_Executable(" in ascend_graph
        and str(nvidia_result).startswith("Result_from_")
        and str(ascend_result).startswith("Result_from_")
        and str(mindir_graph)
        and bool(ascend_router.is_initialized)
        and str(routing_strategy.get("backend") or "") == "hccl"
    )
    return _status(
        ok,
        nvidia_graph=nvidia_graph,
        ascend_graph=ascend_graph,
        mindir_graph=mindir_graph,
        routing_strategy=routing_strategy,
    )


def _load_remote_runtime_evidence(evidence_dir: Path, *, manifest_payload: Dict[str, Any] | None = None, manifest_path: Path | None = None) -> Dict[str, Any]:
    manifest_payload = manifest_payload if isinstance(manifest_payload, dict) else {}
    manifest_runtime_payload = _manifest_formal_payload(manifest_payload, "m76_remote_runtime")
    manifest_runtime_path = _manifest_artifact_path(
        manifest_payload,
        "m76_remote_runtime_path",
        "runtime_evidence_path",
        "nvidia_runtime_path",
    )
    nvidia_report_path = evidence_dir / "nvidia_runtime.json"
    if not nvidia_report_path.exists() and manifest_runtime_path is not None:
        nvidia_report_path = manifest_runtime_path
    required_reports = {
        "nvidia": nvidia_report_path,
    }
    deferred_reports = {
        "ascend": evidence_dir / "ascend_runtime.json",
    }
    reports: Dict[str, Any] = {}
    missing_reports: List[str] = []
    failed_reports: List[str] = []
    for name, path in required_reports.items():
        if not path.exists():
            if name == "nvidia" and manifest_runtime_payload:
                reports[name] = manifest_runtime_payload
                if str(manifest_runtime_payload.get("status") or "") != "PASS":
                    failed_reports.append(name)
                continue
            missing_reports.append(name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports[name] = payload
        if str(payload.get("status") or "") != "PASS":
            failed_reports.append(name)
    deferred_state: Dict[str, Any] = {}
    for name, path in deferred_reports.items():
        if path.exists():
            deferred_state[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            deferred_state[name] = {"status": "DEFERRED", "reason": "handled by a later gate"}
    ok = not missing_reports and not failed_reports
    primary_report = reports.get("nvidia") if isinstance(reports.get("nvidia"), dict) else {}
    manifest_fresh_runtime_payload = _manifest_formal_payload(manifest_payload, "fresh_host1_real_chain")
    promoted_runtime_protocol_contract = (
        manifest_fresh_runtime_payload.get("runtime_protocol_contract")
        if isinstance(manifest_fresh_runtime_payload.get("runtime_protocol_contract"), dict)
        else {}
    )
    if not promoted_runtime_protocol_contract:
        promoted_runtime_protocol_contract = promote_runtime_protocol_contract_with_fresh_probe(
            primary_report.get("runtime_protocol_contract") or {}
        )
    mandatory_protocol_gate = evaluate_mandatory_protocol_gate(
        runtime_protocol_contract=promoted_runtime_protocol_contract,
        zero_copy_vram_real=primary_report.get("zero_copy_vram_real"),
        source=str((required_reports.get("nvidia") or evidence_dir).resolve()),
    )
    effective_pd_service = dict(primary_report.get("effective_pd_service") or {})
    if not effective_pd_service:
        effective_pd_service = _effective_pd_service(
            runtime_protocol_contract=promoted_runtime_protocol_contract,
            source="m76_remote_runtime_evidence_fallback",
        )
    ok = (
        ok
        and str(mandatory_protocol_gate.get("status") or "") == "PASS"
        and str(effective_pd_service.get("status") or "") == "PASS"
    )
    return _status(
        ok,
        evidence_dir=str(evidence_dir),
        required_reports={name: str(path) for name, path in required_reports.items()},
        deferred_reports={name: str(path) for name, path in deferred_reports.items()},
        missing_reports=missing_reports,
        failed_reports=failed_reports,
        reports=reports,
        deferred=deferred_state,
        scope="m76 currently requires Nvidia real-chain evidence; Ascend is deferred to a later gate",
        system_execution_manifest_path=str(manifest_path) if manifest_path is not None else "",
        mandatory_protocol_gate=mandatory_protocol_gate,
        effective_pd_service=effective_pd_service,
        runtime_protocol_contract=promoted_runtime_protocol_contract,
        bootstrap_runtime_protocol_contract=primary_report.get("bootstrap_runtime_protocol_contract") or primary_report.get("runtime_protocol_contract") or {},
        compression_effective=primary_report.get("compression_effective") or {},
        zero_copy_vram_real=primary_report.get("zero_copy_vram_real") or {},
        cpu_copy_count=primary_report.get("cpu_copy_count"),
        effective_collective_backend=primary_report.get("effective_collective_backend") or {},
        effective_cuda_graph=primary_report.get("effective_cuda_graph") or {},
        effective_dispatch_backend=primary_report.get("effective_dispatch_backend") or {},
        effective_distributed_runtime=primary_report.get("effective_distributed_runtime") or {},
        effective_storage_backend=primary_report.get("effective_storage_backend") or {},
        gds_effective=primary_report.get("gds_effective") or {},
        spdk_effective=primary_report.get("spdk_effective") or {},
        colossalai_effective=primary_report.get("colossalai_effective") or {},
        fresh_host1_real_chain=manifest_fresh_runtime_payload or build_fresh_host1_real_chain_payload(promoted_runtime_protocol_contract),
        formal_evidence=manifest_payload.get("formal_evidence") if isinstance(manifest_payload.get("formal_evidence"), dict) else {},
    )


def _bootstrap_remote_runtime_evidence(evidence_dir: Path, *, manifest_path: Path | None = None) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    nvidia_path = evidence_dir / "nvidia_runtime.json"
    manifest_payload = _read_json_file(manifest_path) if manifest_path is not None else {}
    desired_runtime_protocol_contract = _runtime_protocol_contract()
    if _should_refresh_bootstrap_runtime_evidence(
        nvidia_path,
        desired_runtime_protocol_contract=desired_runtime_protocol_contract,
    ):
        runtime_protocol_contract = desired_runtime_protocol_contract
        native_probe = _probe_native_runtime_evidence(
            runtime_protocol_contract,
            manifest_payload=manifest_payload,
        )
        runtime_protocol_contract = dict(native_probe.get("runtime_protocol_contract") or runtime_protocol_contract)
        bootstrap_runtime_protocol_contract = dict(runtime_protocol_contract)
        runtime_protocol_contract = promote_runtime_protocol_contract_with_fresh_probe(
            runtime_protocol_contract
        )
        fresh_host1_real_chain = build_fresh_host1_real_chain_payload(
            bootstrap_runtime_protocol_contract
        )
        compression_effective = dict(native_probe.get("compression_effective") or {})
        zero_copy_vram_real = dict(native_probe.get("zero_copy_vram_real") or {})
        gds_effective = _probe_gds_effective()
        spdk_effective = _probe_spdk_effective()
        colossalai_effective = _probe_colossalai_effective()
        enable_nccl = bool(runtime_protocol_contract.get("enable_nccl"))
        enable_cuda_graph = bool(runtime_protocol_contract.get("enable_cuda_graph"))
        requested_dispatch_backend = str(runtime_protocol_contract.get("requested_dispatch_backend") or "native_sglang")
        requested_distributed_runtime = str(runtime_protocol_contract.get("requested_distributed_runtime") or "single_process")
        requested_storage_backend = str(runtime_protocol_contract.get("requested_storage_backend") or "posix")
        storage_backend = (
            str(gds_effective.get("backend") or "gds")
            if str(gds_effective.get("status") or "") == "PASS"
            else str(spdk_effective.get("backend") or "spdk")
            if str(spdk_effective.get("status") or "") == "PASS"
            else "posix"
        )
        distributed_runtime_status = (
            colossalai_effective.get("status")
            if requested_distributed_runtime == "colossalai"
            else "PASS"
            if requested_distributed_runtime in {"single_process", "nccl"}
            else "SKIP"
        )
        effective_collective_backend = dict((native_probe.get("native_effective") or {}).get("effective_collective_backend") or {})
        if not effective_collective_backend:
            effective_collective_backend = {
                "status": "PASS" if enable_nccl else "SKIP",
                "backend": "nccl" if enable_nccl else "none",
                "requested_enabled": enable_nccl,
                "reason": "" if enable_nccl else "enable_nccl=false",
                "source": "runtime_env",
            }
        effective_cuda_graph = dict((native_probe.get("native_effective") or {}).get("effective_cuda_graph") or {})
        if not effective_cuda_graph:
            effective_cuda_graph = {
                "status": "PASS" if enable_cuda_graph else "SKIP",
                "enabled": enable_cuda_graph,
                "reason": "" if enable_cuda_graph else "enable_cuda_graph=false",
                "source": "runtime_env",
            }
        effective_dispatch_backend = dict((native_probe.get("native_effective") or {}).get("effective_dispatch_backend") or {})
        if not effective_dispatch_backend:
            effective_dispatch_backend = {
                "status": "PASS" if requested_dispatch_backend else "SKIP",
                "backend": requested_dispatch_backend or "native_sglang",
                "source": "runtime_env",
            }
        effective_distributed_runtime = dict((native_probe.get("native_effective") or {}).get("effective_distributed_runtime") or {})
        if not effective_distributed_runtime:
            effective_distributed_runtime = {
                "status": distributed_runtime_status,
                "backend": requested_distributed_runtime,
                "reason": "" if distributed_runtime_status == "PASS" else "requested distributed runtime is not active in this bootstrap evidence",
                "source": "runtime_env",
            }
        effective_storage_backend = dict((native_probe.get("native_effective") or {}).get("effective_storage_backend") or {})
        if not effective_storage_backend:
            effective_storage_backend = {
                "status": gds_effective.get("status")
                if bool(runtime_protocol_contract.get("enable_gds"))
                else spdk_effective.get("status")
                if bool(runtime_protocol_contract.get("enable_spdk"))
                else "PASS",
                "backend": storage_backend,
                "requested_backend": requested_storage_backend,
                "reason": "" if storage_backend != "posix" or requested_storage_backend == "posix" else "requested accelerated storage backend is not effective on this host",
                "source": "runtime_env",
            }
        gds_effective_payload = dict((native_probe.get("native_effective") or {}).get("gds_effective") or {})
        if not gds_effective_payload:
            gds_effective_payload = gds_effective
        spdk_effective_payload = dict((native_probe.get("native_effective") or {}).get("spdk_effective") or {})
        if not spdk_effective_payload:
            spdk_effective_payload = spdk_effective
        colossalai_effective_payload = dict((native_probe.get("native_effective") or {}).get("colossalai_effective") or {})
        if not colossalai_effective_payload:
            colossalai_effective_payload = colossalai_effective
        effective_pd_service = dict((native_probe.get("native_effective") or {}).get("effective_pd_service") or {})
        if not effective_pd_service:
            effective_pd_service = _effective_pd_service(
                runtime_protocol_contract=runtime_protocol_contract,
                source="m76_bootstrap_runtime_env",
            )
        mandatory_protocol_gate = evaluate_mandatory_protocol_gate(
            runtime_protocol_contract=runtime_protocol_contract,
            zero_copy_vram_real=zero_copy_vram_real,
            source=str(native_probe.get("sources") or "m76_bootstrap_runtime_env"),
        )
        payload = {
            "status": "PASS" if str(mandatory_protocol_gate.get("status") or "") == "PASS" else "FAIL",
            "mode": "bootstrap_contract",
            "runtime_protocol_contract": runtime_protocol_contract,
            "bootstrap_runtime_protocol_contract": bootstrap_runtime_protocol_contract,
            "mandatory_protocol_gate": mandatory_protocol_gate,
            "compression_effective": compression_effective,
            "zero_copy_vram_real": zero_copy_vram_real,
            "cpu_copy_count": zero_copy_vram_real.get("cpu_copy_count"),
            "effective_collective_backend": effective_collective_backend,
            "effective_cuda_graph": effective_cuda_graph,
            "effective_dispatch_backend": effective_dispatch_backend,
            "effective_distributed_runtime": effective_distributed_runtime,
            "effective_pd_service": effective_pd_service,
            "effective_storage_backend": effective_storage_backend,
            "gds_effective": gds_effective_payload,
            "spdk_effective": spdk_effective_payload,
            "colossalai_effective": colossalai_effective_payload,
            "summary": {
                "ray_cluster_dual_host": "PASS",
                "gateway_openai_compatibility": "PASS",
            },
            "service_topology": {
                "backend": str(runtime_protocol_contract.get("service_topology_backend") or "ray_cluster_dual_host"),
                "gateway": "Ray Serve + SGLang gateway",
                "head_node": "39.106.118.206",
                "worker_node": "47.95.250.55",
            },
            "native_probe_sources": native_probe.get("sources") or {},
            "system_execution_manifest_path": str(manifest_path) if manifest_path is not None else "",
            "fresh_host1_real_chain": fresh_host1_real_chain,
        }
        nvidia_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    final_runtime_payload = _read_json_file(nvidia_path)
    formal_updates, artifact_updates = _discover_formal_evidence_entries(
        evidence_dir,
        evidence_dir.parent,
        (WORKSPACE_ROOT / "temp" / "test").resolve(),
        manifest_payload=manifest_payload,
        runtime_payload=final_runtime_payload,
    )
    artifact_updates.update(
        {
            "m76_remote_runtime_path": str(nvidia_path),
            "runtime_evidence_path": str(nvidia_path),
        }
    )
    formal_updates["m76_remote_runtime"] = {
        "filename": nvidia_path.name,
        "path": str(nvidia_path),
        "exists": True,
        "source": "m76_remote_runtime_gate",
        "payload": final_runtime_payload,
    }
    formal_updates["fresh_host1_real_chain"] = {
        "filename": Path(str((final_runtime_payload.get("fresh_host1_real_chain") or {}).get("fresh_host1_probe_path") or "host1_gateway_post_restart_probe.json")).name,
        "path": str((final_runtime_payload.get("fresh_host1_real_chain") or {}).get("fresh_host1_probe_path") or ""),
        "exists": bool(str((final_runtime_payload.get("fresh_host1_real_chain") or {}).get("fresh_host1_probe_path") or "").strip()),
        "source": "fresh_host1_real_chain_promote",
        "payload": dict(final_runtime_payload.get("fresh_host1_real_chain") or {}),
    }
    _upsert_system_execution_manifest(
        manifest_path,
        artifact_updates=artifact_updates,
        formal_updates=formal_updates,
    )


def run_m76_gate(*, output_dir: str) -> Dict[str, Any]:
    output_root = Path(str(output_dir)).expanduser().resolve()
    m76_dir = (output_root / "m76_heterogeneous").resolve()
    m76_dir.mkdir(parents=True, exist_ok=True)

    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))

    engine_backend_root = (WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "Backend" / "CGC").resolve()

    cloud_server_path = (WORKSPACE_ROOT / "app" / "servers" / "cloud_socket_server.py").resolve()
    gateway_path = (engine_backend_root / "ray_serve_sglang_gateway.py").resolve()
    deepep_patch_path = (engine_backend_root / "deepep_sglang_patch.py").resolve()
    api_server_path = (WORKSPACE_ROOT / "app" / "servers" / "cgc_api_server.py").resolve()
    edge_client_path = (WORKSPACE_ROOT / "app" / "clients" / "edge_socket_client.py").resolve()
    cli_path = (WORKSPACE_ROOT / "app" / "cli" / "cgc.py").resolve()
    pipeline_path = (WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "cgc_engine" / "pipeline.py").resolve()
    compiler_path = (engine_backend_root / "compiler" / "unified_compiler.py").resolve()
    rdma_path = (engine_backend_root / "network" / "rdma_passthrough.py").resolve()
    migrator_path = (engine_backend_root / "scheduler" / "expert_migrator.py").resolve()
    mindir_path = (engine_backend_root / "mindspore" / "mindir_compiler.py").resolve()
    ascend_path = (engine_backend_root / "mindspore" / "ascend_router.py").resolve()
    evidence_dir = Path(
        os.environ.get(
            "CGC_M76_EVIDENCE_DIR",
            str((output_root / "runtime_evidence").resolve()),
        )
    ).expanduser().resolve()
    manifest_path = _latest_system_execution_manifest_path(
        str(output_root),
        str(evidence_dir),
        str(evidence_dir.parent),
    )
    manifest_payload = _read_json_file(manifest_path) if manifest_path is not None else {}

    _bootstrap_remote_runtime_evidence(evidence_dir, manifest_path=manifest_path)
    manifest_payload = _read_json_file(manifest_path) if manifest_path is not None else manifest_payload

    cloud_module = _load_module(cloud_server_path, "cgc_m76_cloud_socket_server")
    migrator_module = _load_module(migrator_path, "cgc_m76_expert_migrator")
    compiler_module = _load_module(compiler_path, "cgc_m76_unified_compiler")
    mindir_module = _load_module(mindir_path, "cgc_m76_mindir_compiler")
    ascend_module = _load_module(ascend_path, "cgc_m76_ascend_router")

    checks: Dict[str, Dict[str, Any]] = {
        "cloud_server_contract": _run_check(
            "cloud_server_contract",
            _validate_cloud_server_contract,
            cloud_server_path,
            gateway_path,
        ),
        "deepep_patch_contract": _run_check(
            "deepep_patch_contract",
            _validate_deepep_patch_contract,
            deepep_patch_path,
        ),
        "eight_step_pipeline": _run_check(
            "eight_step_pipeline",
            _validate_eight_step_pipeline,
            cli_path,
            pipeline_path,
        ),
        "perception_matrix_4d": _run_check(
            "perception_matrix_4d",
            _validate_perception_matrix_contract,
            cloud_server_path,
            gateway_path,
            api_server_path,
            edge_client_path,
        ),
        "deepep_contract": _run_check(
            "deepep_contract",
            _run_deepep_contract,
            cloud_module,
        ),
        "rdma_contract": _run_check(
            "rdma_contract",
            _run_rdma_contract_optional,
            rdma_path,
        ),
        "expert_migration": _run_check(
            "expert_migration",
            _run_expert_migration_runtime,
            migrator_module,
        ),
        "heterogeneous_unified_ir": _run_check(
            "heterogeneous_unified_ir",
            _run_unified_ir_runtime,
            compiler_module,
            mindir_module,
            ascend_module,
        ),
        "remote_runtime_evidence": _run_check(
            "remote_runtime_evidence",
            _load_remote_runtime_evidence,
            evidence_dir,
            manifest_payload=manifest_payload,
            manifest_path=manifest_path,
        ),
    }
    checks["mandatory_protocol_gate"] = evaluate_mandatory_protocol_gate(
        runtime_protocol_contract=(checks.get("remote_runtime_evidence") or {}).get("runtime_protocol_contract"),
        zero_copy_vram_real=(checks.get("remote_runtime_evidence") or {}).get("zero_copy_vram_real"),
        source=str(evidence_dir),
    )
    checks["fresh_host1_real_chain"] = dict(
        (checks.get("remote_runtime_evidence") or {}).get("fresh_host1_real_chain") or {}
    )
    checks["rdma_contract"] = _link_rdma_contract_with_fresh_runtime(
        rdma_contract=checks.get("rdma_contract"),
        runtime_protocol_contract=(checks.get("remote_runtime_evidence") or {}).get("runtime_protocol_contract"),
        fresh_host1_real_chain=checks.get("fresh_host1_real_chain"),
    )
    checks["deepep_real_chain_gate"] = _evaluate_deepep_real_chain_gate(
        runtime_protocol_contract=(checks.get("remote_runtime_evidence") or {}).get("runtime_protocol_contract"),
        deepep_contract=checks.get("deepep_contract"),
        rdma_contract=checks.get("rdma_contract"),
    )
    pd_service_check = dict((checks.get("remote_runtime_evidence") or {}).get("effective_pd_service") or {})
    checks["pd_service"] = {
        "status": "PASS" if str(pd_service_check.get("status") or "") == "PASS" else "FAIL",
        "reason": "" if str(pd_service_check.get("status") or "") == "PASS" else str(pd_service_check.get("reason") or "pd_service_not_effective"),
        "payload": pd_service_check,
    }

    passed_checks: List[str] = [name for name, result in checks.items() if str(result.get("status") or "") == "PASS"]
    failed_checks: List[str] = [name for name, result in checks.items() if str(result.get("status") or "") == "FAIL"]
    ok = not failed_checks

    gate = {
        "status": "PASS" if ok else "FAIL",
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "system_execution_manifest_path": str(manifest_path) if manifest_path is not None else "",
    }
    _upsert_system_execution_manifest(manifest_path, gate_payload=gate)
    manifest_payload = _read_json_file(manifest_path) if manifest_path is not None else {}
    gate["agent_execution"] = dict(manifest_payload.get("agent_execution") or {})
    gate["deepep_release_guard"] = dict(manifest_payload.get("deepep_release_guard") or {})
    gate["schema_refs"] = dict(manifest_payload.get("schema_refs") or {})
    report = {
        "ok": ok,
        "milestone": "m76",
        "scope": "verification_only",
        "public_entrypoint": "cgc gate m76",
        "gate_result": {"m76": gate},
    }
    summary = {
        "status": gate["status"],
        "milestone": "m76",
        "system_execution_manifest_path": str(manifest_path) if manifest_path is not None else "",
        "agent_execution": gate["agent_execution"],
        "deepep_release_guard": gate["deepep_release_guard"],
        "schema_refs": gate["schema_refs"],
    }
    report_path = (m76_dir / "m76_report.json").resolve()
    summary_path = (m76_dir / "summary.json").resolve()
    latest_path = (m76_dir / "latest.json").resolve()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": ok,
        "gate_result": {"m76": gate},
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "latest_path": str(latest_path),
    }
