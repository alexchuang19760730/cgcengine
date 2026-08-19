import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict

from cgc_engine.product.release_alias_contracts import apply_release_alias_contracts
from cgc_engine.product.upkg30_common import evaluate_mandatory_protocol_gate


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _status(ok: bool, **extra: Any) -> Dict[str, Any]:
    payload = {"status": "PASS" if ok else "FAIL"}
    payload.update(extra)
    return payload


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _resume_device_supports_zero_copy(*, cpu_copy_count: int, uma_buffer_used: bool, device_resume_consumed: bool, resume_tensor_device: str) -> bool:
    if int(cpu_copy_count) != 0 or not bool(device_resume_consumed):
        return False
    device = str(resume_tensor_device or "").strip().lower()
    if device.startswith("mps"):
        return bool(uma_buffer_used)
    return device.startswith("cuda")


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


def _run_check(check_name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        if not isinstance(result, dict):
            return _status(False, check=check_name, error="invalid_result", detail=repr(result))
        result.setdefault("check", check_name)
        return result
    except Exception as exc:
        return _status(False, check=check_name, error=str(exc), exception_type=type(exc).__name__)


def _validate_report_schema(schema_path: Path) -> Dict[str, Any]:
    if not schema_path.exists():
        return _status(False, reason="missing_report_schema", schema_path=str(schema_path))
    payload = _read_json(schema_path)
    required_properties = {
        "schema_version",
        "gate",
        "status",
        "runtime_protocol_contract",
        "mandatory_protocol_gate",
        "true_state_transport",
        "edge_state_resume_decode",
        "compression_effective",
        "zero_copy_vram_real",
        "cpu_copy_count",
        "effective_collective_backend",
        "effective_cuda_graph",
        "effective_dispatch_backend",
        "effective_distributed_runtime",
        "effective_storage_backend",
        "gds_effective",
        "spdk_effective",
        "colossalai_effective",
        "artifacts",
    }
    properties = set((payload.get("properties") or {}).keys())
    missing = sorted(required_properties - properties)
    return _status(
        not missing,
        schema_path=str(schema_path),
        schema_id=payload.get("$id"),
        missing_properties=missing,
    )


def _validate_true_state_transport_contract(cloud_socket_server_path: Path) -> Dict[str, Any]:
    source = _read_text(cloud_socket_server_path)
    required_markers = [
        '"state_kind"',
        '"state_codec"',
        '"state_meta"',
        '"kda_state_v1"',
    ]
    anti_markers = [
        "_build_kda_payload(cloud_text)",
        '"text": kda_payload["text"]',
    ]
    missing = [marker for marker in required_markers if marker not in source]
    blocking_anti_markers = [marker for marker in anti_markers if marker in source]
    return _status(
        not missing and not blocking_anti_markers,
        file=str(cloud_socket_server_path),
        missing_markers=missing,
        blocking_anti_markers=blocking_anti_markers,
    )


def _validate_edge_state_resume_decode_contract(
    api_server_path: Path,
    local_infer_path: Path,
    kda_state_runtime_path: Path,
) -> Dict[str, Any]:
    api_source = _read_text(api_server_path)
    infer_source = _read_text(local_infer_path)
    runtime_source = _read_text(kda_state_runtime_path)
    required_api_markers = [
        '"state_kind"',
        "resume_from_kda_state",
        "local_resume",
    ]
    required_infer_markers = [
        "resume_from_kda_state",
        "resume_one_token_from_kda_state",
    ]
    required_runtime_markers = [
        "decode_one_step_kda_aot",
        "resume_one_token_from_kda_state",
    ]
    missing_api = [marker for marker in required_api_markers if marker not in api_source]
    missing_infer = [marker for marker in required_infer_markers if marker not in infer_source]
    missing_runtime = [marker for marker in required_runtime_markers if marker not in runtime_source]
    return _status(
        not missing_api and not missing_infer and not missing_runtime,
        api_file=str(api_server_path),
        local_infer_file=str(local_infer_path),
        kda_state_runtime_file=str(kda_state_runtime_path),
        missing_api_markers=missing_api,
        missing_local_infer_markers=missing_infer,
        missing_kda_state_runtime_markers=missing_runtime,
    )


def _load_runtime_evidence(evidence_path: Path) -> Dict[str, Any]:
    manifest_path = _latest_system_execution_manifest_path(
        str(evidence_path.parent),
        str(evidence_path.parent.parent),
    )
    manifest_payload = _read_json_safely(manifest_path) if manifest_path is not None else {}
    manifest_runtime_payload = _manifest_formal_payload(manifest_payload, "m75_active_runtime")
    manifest_runtime_path = _manifest_artifact_path(
        manifest_payload,
        "m75_trueorthokda_active_runtime_path",
        "m75_active_runtime",
    )
    if evidence_path.exists():
        payload = _read_json(evidence_path)
    elif manifest_runtime_payload:
        payload = manifest_runtime_payload
        evidence_path = manifest_runtime_path or evidence_path
    elif manifest_runtime_path is not None:
        payload = _read_json(manifest_runtime_path)
        evidence_path = manifest_runtime_path
    else:
        return _status(False, reason="missing_runtime_evidence", evidence_path=str(evidence_path))
    required_sections = [
        "runtime_protocol_contract",
        "mandatory_protocol_gate",
        "true_state_transport",
        "edge_state_resume_decode",
        "compression_effective",
        "zero_copy_vram_real",
        "effective_collective_backend",
        "effective_cuda_graph",
        "effective_dispatch_backend",
        "effective_distributed_runtime",
        "effective_storage_backend",
        "gds_effective",
        "spdk_effective",
        "colossalai_effective",
        "artifacts",
    ]
    missing_sections = [section for section in required_sections if section not in payload]
    if missing_sections:
        return _status(
            False,
            reason="missing_runtime_sections",
            evidence_path=str(evidence_path),
            missing_sections=missing_sections,
        )

    transport = payload.get("true_state_transport") or {}
    edge_resume = payload.get("edge_state_resume_decode") or {}
    compression = payload.get("compression_effective") or {}
    zero_copy = payload.get("zero_copy_vram_real") or {}
    cpu_copy_count = payload.get("cpu_copy_count")

    transport_ok = bool(
        str(transport.get("status") or "") == "PASS"
        and str(transport.get("state_kind") or "") == "kda_state_v1"
        and bool(transport.get("state_bytes_from_runtime"))
    )
    edge_resume_ok = bool(
        str(edge_resume.get("status") or "") == "PASS"
        and bool(edge_resume.get("resume_decode_executed"))
        and int(edge_resume.get("resume_output_tokens") or 0) >= 1
    )
    compression_ok = bool(
        str(compression.get("status") or "") == "PASS"
        and int(compression.get("raw_state_bytes") or 0) > 0
        and int(compression.get("compressed_state_bytes") or 0) > 0
        and float(compression.get("compression_ratio") or 0.0) > 0.0
        and float(compression.get("compression_ratio") or 0.0) < 1.0
    )
    zero_copy_ok = bool(
        str(zero_copy.get("status") or "") == "PASS"
        and bool(zero_copy.get("device_resume_consumed"))
        and _resume_device_supports_zero_copy(
            cpu_copy_count=int(cpu_copy_count if cpu_copy_count is not None else zero_copy.get("cpu_copy_count") or 0),
            uma_buffer_used=bool(zero_copy.get("uma_buffer_used")),
            device_resume_consumed=bool(zero_copy.get("device_resume_consumed")),
            resume_tensor_device=str(zero_copy.get("resume_tensor_device") or ""),
        )
    )
    ok = transport_ok and edge_resume_ok and compression_ok and zero_copy_ok
    return _status(
        ok,
        evidence_path=str(evidence_path),
        transport=transport,
        edge_resume=edge_resume,
        compression=compression,
        zero_copy=zero_copy,
        missing_sections=[],
    )


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
        _env_str("CGC_M75_LOCAL_INFER_EVIDENCE_ROOT"),
        _env_str("CGC_M76_LOCAL_INFER_EVIDENCE_ROOT"),
        _env_str("CGC_LOCAL_INFER_EVIDENCE_ROOT"),
        str((WORKSPACE_ROOT / "ComputeGraphCompiler-main" / "temp").resolve()),
        str((WORKSPACE_ROOT / "temp").resolve()),
        str((WORKSPACE_ROOT / "temp" / "test").resolve()),
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


def _read_json_safely(path: Path) -> Dict[str, Any]:
    try:
        payload = _read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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
    explicit_candidates = [
        _env_str("CGC_SYSTEM_EXECUTION_MANIFEST_PATH"),
        _env_str("CGC_M75_SYSTEM_EXECUTION_MANIFEST_PATH"),
        _env_str("CGC_M76_SYSTEM_EXECUTION_MANIFEST_PATH"),
    ]
    for raw in explicit_candidates:
        explicit_path = _resolve_existing_json_path(raw)
        if explicit_path and _is_system_execution_manifest_payload(_read_json_safely(explicit_path)):
            return explicit_path
    candidates: list[Path] = []
    search_roots = _iter_search_roots(
        *extra_roots,
        _env_str("CGC_M75_EVIDENCE_DIR"),
        _env_str("CGC_M76_EVIDENCE_DIR"),
        _env_str("CGC_MEGATRAIN_SYSTEM_MANIFEST_DISCOVERY_ROOT"),
        _env_str("CGC_RUNTIME_CONTRACT_ROOT"),
    )
    for root in search_roots:
        try:
            matches = [path.resolve() for path in root.rglob("system_execution_manifest.json") if path.is_file()]
        except OSError:
            continue
        candidates.extend(matches)
    for path in reversed(sorted(candidates, key=_candidate_sort_key)):
        if _is_system_execution_manifest_payload(_read_json_safely(path)):
            return path
    return None


def _manifest_artifact_path(manifest_payload: Dict[str, Any], *names: str) -> Path | None:
    if not isinstance(manifest_payload, dict) or not manifest_payload:
        return None
    artifacts = manifest_payload.get("artifacts") if isinstance(manifest_payload.get("artifacts"), dict) else {}
    formal_evidence = manifest_payload.get("formal_evidence") if isinstance(manifest_payload.get("formal_evidence"), dict) else {}
    for name in names:
        path = _resolve_existing_json_path(str(artifacts.get(name) or ""))
        if path is not None:
            return path
        entry = formal_evidence.get(name)
        if isinstance(entry, dict):
            path = _resolve_existing_json_path(str(entry.get("path") or ""))
            if path is not None:
                return path
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
                loaded = _read_json_safely(entry_path)
                if loaded:
                    return loaded
    return {}


def _discover_formal_evidence_entries(*search_roots: Path, manifest_payload: Dict[str, Any] | None = None) -> tuple[Dict[str, Any], Dict[str, str]]:
    manifest_payload = manifest_payload if isinstance(manifest_payload, dict) else {}
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
    manifest_artifacts = manifest_payload.get("artifacts") if isinstance(manifest_payload.get("artifacts"), dict) else {}
    formal_evidence = manifest_payload.get("formal_evidence") if isinstance(manifest_payload.get("formal_evidence"), dict) else {}
    resolved_roots = [root.resolve() for root in search_roots if isinstance(root, Path) and root.exists()]
    artifact_updates: Dict[str, str] = {}
    formal_updates: Dict[str, Any] = {}
    for evidence_name, spec in evidence_specs.items():
        candidates: list[Path] = []
        for env_name in spec.get("env_names") or []:
            candidate = _resolve_existing_json_path(_env_str(str(env_name)))
            if candidate is not None:
                candidates.append(candidate)
        artifact_candidate = _resolve_existing_json_path(str(manifest_artifacts.get(spec.get("artifact_key") or "") or ""))
        if artifact_candidate is not None:
            candidates.append(artifact_candidate)
        manifest_entry = formal_evidence.get(evidence_name)
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
        artifact_key = str(spec.get("artifact_key") or evidence_name)
        artifact_updates[artifact_key] = str(resolved)
        formal_updates[evidence_name] = {
            "filename": resolved.name,
            "path": str(resolved),
            "exists": True,
            "source": "m75_manifest_backfill",
            "payload": _read_json_safely(resolved),
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
    payload = _read_json_safely(manifest_path)
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


def _latest_local_infer_evidence_path(*extra_roots: str, require_runtime_probe: bool = False) -> Path | None:
    explicit_path = _resolve_explicit_json_path(
        "CGC_LOCAL_INFER_EVIDENCE_PATH",
        "CGC_M75_LOCAL_INFER_EVIDENCE_PATH",
        "CGC_M76_LOCAL_INFER_EVIDENCE_PATH",
    )
    if explicit_path is not None:
        explicit_payload = _read_json_safely(explicit_path)
        if _is_runtime_probe_local_infer_payload(explicit_payload):
            return explicit_path
        if not require_runtime_probe:
            return explicit_path
    candidates: list[Path] = []
    search_roots = _iter_search_roots(
        *extra_roots,
        _env_str("CGC_M76_EVIDENCE_DIR"),
        _env_str("CGC_MEGATRAIN_SYSTEM_MANIFEST_DISCOVERY_ROOT"),
        _env_str("CGC_RUNTIME_CONTRACT_ROOT"),
    )
    for root in search_roots:
        for pattern in ("local_infer_*.json", "local_infer_runtime_probe.json"):
            try:
                matches = [path.resolve() for path in root.rglob(pattern) if path.is_file()]
            except OSError:
                continue
            candidates.extend(matches)
    candidates = sorted(candidates, key=_candidate_sort_key)
    if not candidates:
        return None
    fallback_path = candidates[-1]
    for path in reversed(candidates):
        payload = _read_json_safely(path)
        if _is_runtime_probe_local_infer_payload(payload):
            return path
    return None if require_runtime_probe else fallback_path


def _bootstrap_active_runtime_evidence(runtime_evidence_path: Path) -> Dict[str, Any]:
    manifest_path = _latest_system_execution_manifest_path(
        str(runtime_evidence_path.parent),
        str(runtime_evidence_path.parent.parent),
    )
    manifest_payload = _read_json_safely(manifest_path) if manifest_path is not None else {}
    manifest_local_infer_path = _manifest_artifact_path(
        manifest_payload,
        "local_infer_evidence_path",
        "latest_local_infer_path",
    )
    if manifest_local_infer_path is not None and _is_runtime_probe_local_infer_payload(_read_json_safely(manifest_local_infer_path)):
        latest_evidence_path = manifest_local_infer_path
    else:
        latest_evidence_path = None
    try:
        if latest_evidence_path is None:
            latest_evidence_path = _latest_local_infer_evidence_path(
                str(runtime_evidence_path.parent),
                str(runtime_evidence_path.parent.parent),
                _env_str("CGC_M75_EVIDENCE_DIR"),
                _env_str("CGC_M75_LOCAL_INFER_EVIDENCE_ROOT"),
                _env_str("CGC_M76_LOCAL_INFER_EVIDENCE_ROOT"),
                require_runtime_probe=True,
            )
    except TypeError:
        if latest_evidence_path is None:
            latest_evidence_path = _latest_local_infer_evidence_path()
            if latest_evidence_path is not None and not _is_runtime_probe_local_infer_payload(_read_json_safely(latest_evidence_path)):
                latest_evidence_path = None
    if latest_evidence_path is None or not latest_evidence_path.exists():
        return _status(False, reason="missing_runtime_probe_local_infer_evidence")
    local_payload = _read_json(latest_evidence_path)
    state_kind = str(local_payload.get("state_kind") or "")
    state_codec = str(local_payload.get("state_codec") or "")
    state_meta = local_payload.get("state_meta") if isinstance(local_payload.get("state_meta"), dict) else {}
    payload_size = int(local_payload.get("state_payload_bytes") or 0)
    trace_id = str(local_payload.get("trace_id") or "")
    resume_decode_executed = bool(local_payload.get("resume_decode_executed"))
    raw_state_bytes = int(local_payload.get("raw_state_bytes") or state_meta.get("raw_state_bytes") or payload_size)
    compressed_state_bytes = int(local_payload.get("compressed_state_bytes") or state_meta.get("compressed_state_bytes") or payload_size)
    compression_ratio = float(local_payload.get("compression_ratio") or state_meta.get("compression_ratio") or 1.0)
    cpu_copy_count = int(local_payload.get("cpu_copy_count") or 0)
    uma_buffer_used = bool(local_payload.get("uma_buffer_used"))
    device_resume_consumed = bool(local_payload.get("device_resume_consumed"))
    resume_tensor_device = str(local_payload.get("resume_tensor_device") or "cpu")
    protocol_family = _env_str("CGC_RUNTIME_PROTOCOL_FAMILY") or "trueorthokda"
    enable_nccl = _env_bool("CGC_MEGATRAIN_ENABLE_NCCL", _env_bool("CGC_SGLANG_USE_NCCL", False))
    enable_cuda_graph = _env_bool("CGC_MEGATRAIN_ENABLE_CUDA_GRAPH", False)
    requested_dispatch_backend = (
        _env_str("CGC_MEGATRAIN_REQUESTED_DISPATCH_BACKEND")
        or _env_str("CGC_REQUESTED_DISPATCH_BACKEND")
        or ("deepep" if _env_bool("CGC_M76_ENABLE_DEEPEP", False) or _env_str("CGC_DEEPEP_MODE") else "native_sglang")
    )
    use_colossalai = _env_bool("CGC_MEGATRAIN_USE_COLOSSALAI", False)
    requested_distributed_runtime = (
        _env_str("CGC_MEGATRAIN_REQUESTED_DISTRIBUTED_RUNTIME")
        or ("colossalai" if use_colossalai else "nccl" if enable_nccl else "single_process")
    )
    distributed_runtime_backend = (
        _env_str("CGC_DISTRIBUTED_RUNTIME_BACKEND")
        or requested_distributed_runtime
    )
    enable_gds = _env_bool("CGC_GDS_ENABLED", False)
    enable_spdk = _env_bool("CGC_SPDK_ENABLED", False)
    requested_storage_backend = (
        _env_str("CGC_MEGATRAIN_REQUESTED_STORAGE_BACKEND")
        or ("gds_spdk" if enable_gds and enable_spdk else "gds" if enable_gds else "spdk" if enable_spdk else "posix")
    )
    deepep_parallelism = _resolve_deepep_parallelism()
    expected_zero_copy = _env_bool("CGC_MEGATRAIN_EXPECT_ZERO_COPY", protocol_family == "trueorthokda")
    enable_pd = _env_bool("CGC_MEGATRAIN_ENABLE_PD", _env_bool("CGC_ENABLE_PD", protocol_family == "trueorthokda"))
    pd_endpoint = (
        _env_str("CGC_MEGATRAIN_PD_ENDPOINT")
        or _env_str("CGC_PD_ENDPOINT")
        or _env_str("CGC_PD_SERVICE_ENDPOINT")
        or ("localhost:50051" if enable_pd else "")
    )
    service_topology_backend = (
        _env_str("CGC_SERVICE_TOPOLOGY_BACKEND")
        or ("ray_cluster_dual_host" if enable_pd else "single_host_local")
    )
    pd_mode = _env_str("CGC_MEGATRAIN_PD_MODE") or _env_str("CGC_PD_MODE") or ("cloud_prefill_edge_decode" if enable_pd else "disabled")
    pd_prefix_cache = _env_bool("CGC_MEGATRAIN_PD_PREFIX_CACHE", _env_bool("CGC_PD_PREFIX_CACHE", True))
    require_pd_service = _env_bool("CGC_MEGATRAIN_REQUIRE_PD_SERVICE", _env_bool("CGC_REQUIRE_PD_SERVICE", enable_pd or protocol_family == "trueorthokda"))
    gds_effective = _probe_gds_effective()
    spdk_effective = _probe_spdk_effective()
    colossalai_effective = _probe_colossalai_effective()
    collective_status = "PASS" if enable_nccl and resume_tensor_device.startswith("cuda") else "SKIP" if not enable_nccl else "FAIL"
    collective_backend = "nccl" if collective_status == "PASS" else "none"
    cuda_graph_status = "PASS" if enable_cuda_graph and resume_tensor_device.startswith("cuda") else "SKIP" if not enable_cuda_graph else "FAIL"
    distributed_runtime_status = (
        colossalai_effective.get("status")
        if requested_distributed_runtime == "colossalai"
        else collective_status
        if requested_distributed_runtime == "nccl"
        else "PASS"
    )
    storage_backend = (
        str(gds_effective.get("backend") or "gds")
        if str(gds_effective.get("status") or "") == "PASS"
        else str(spdk_effective.get("backend") or "spdk")
        if str(spdk_effective.get("status") or "") == "PASS"
        else "posix"
    )
    storage_status = (
        gds_effective.get("status")
        if enable_gds
        else spdk_effective.get("status")
        if enable_spdk
        else "PASS"
    )
    transport_status = "PASS" if state_kind == "kda_state_v1" and state_codec == "cq4" and payload_size > 0 else "FAIL"
    edge_resume_status = "PASS" if resume_decode_executed else "FAIL"
    compression_status = "PASS" if raw_state_bytes > 0 and compressed_state_bytes > 0 and compression_ratio < 1.0 else "FAIL"
    sglang_dflash_runtime = _resolve_sglang_dflash_runtime()
    zero_copy_status = (
        "PASS"
        if _resume_device_supports_zero_copy(
            cpu_copy_count=cpu_copy_count,
            uma_buffer_used=uma_buffer_used,
            device_resume_consumed=device_resume_consumed,
            resume_tensor_device=resume_tensor_device,
        )
        else "FAIL"
    )
    runtime_payload = {
        "schema_version": "m75.trueorthokda.active.v1",
        "gate": "m75_trueorthokda_active",
        "status": "PASS" if all(status == "PASS" for status in (transport_status, edge_resume_status, compression_status, zero_copy_status)) else "FAIL",
        "runtime_protocol_contract": {
            "protocol_family": protocol_family,
            "state_kind": state_kind,
            "state_codec": state_codec,
            "expected_zero_copy": expected_zero_copy,
            "enable_nccl": enable_nccl,
            "enable_cuda_graph": enable_cuda_graph,
            "requested_dispatch_backend": requested_dispatch_backend,
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
            "colossalai_plugin": str(colossalai_effective.get("plugin") or "HybridParallelPlugin"),
            "enable_pd": enable_pd,
            "pd_endpoint": pd_endpoint,
            "pd_mode": pd_mode,
            "pd_prefix_cache": pd_prefix_cache,
            "require_pd_service": require_pd_service,
        },
        "true_state_transport": {
            "status": transport_status,
            "state_kind": state_kind,
            "state_codec": state_codec,
            "state_bytes_from_runtime": payload_size > 0,
            "trace_id": trace_id,
        },
        "edge_state_resume_decode": {
            "status": edge_resume_status,
            "resume_decode_executed": resume_decode_executed,
            "resume_output_tokens": 1 if resume_decode_executed else 0,
            "resume_latency_ms": 0.0,
        },
        "compression_effective": {
            "status": compression_status,
            "raw_state_bytes": raw_state_bytes,
            "compressed_state_bytes": compressed_state_bytes,
            "compression_ratio": compression_ratio,
            "network_rx_ms": 0.0,
            "reason": "" if compression_ratio < 1.0 else "current state transport did not produce a compressed payload smaller than raw state bytes",
        },
        "zero_copy_vram_real": {
            "status": zero_copy_status,
            "cpu_copy_count": cpu_copy_count,
            "uma_buffer_used": uma_buffer_used,
            "device_resume_consumed": device_resume_consumed,
            "resume_tensor_device": resume_tensor_device,
            "reason": (
                ""
                if _resume_device_supports_zero_copy(
                    cpu_copy_count=cpu_copy_count,
                    uma_buffer_used=uma_buffer_used,
                    device_resume_consumed=device_resume_consumed,
                    resume_tensor_device=resume_tensor_device,
                )
                else "current resume path still materializes CPU-side bytes/tensors before local resume"
            ),
        },
        "cpu_copy_count": cpu_copy_count,
        "effective_collective_backend": {
            "status": collective_status,
            "backend": collective_backend,
            "requested_enabled": enable_nccl,
            "world_size": int(str(os.environ.get("WORLD_SIZE", "1") or "1").strip() or "1"),
            "resume_tensor_device": resume_tensor_device,
            "reason": "" if collective_status == "PASS" else "collective backend not active on this local resume path",
        },
        "effective_cuda_graph": {
            "status": cuda_graph_status,
            "enabled": enable_cuda_graph,
            "resume_tensor_device": resume_tensor_device,
            "reason": "" if cuda_graph_status == "PASS" else "cuda graph is disabled or this resume path is not executing on cuda",
        },
        "effective_dispatch_backend": {
            "status": "PASS" if requested_dispatch_backend else "SKIP",
            "backend": requested_dispatch_backend or "native_sglang",
            "source": "runtime_env",
        },
        "effective_distributed_runtime": {
            "status": distributed_runtime_status,
            "backend": requested_distributed_runtime,
            "use_colossalai": use_colossalai,
            "reason": "" if distributed_runtime_status == "PASS" else "requested distributed runtime is not effective on this active runtime path",
        },
        "effective_pd_service": _effective_pd_service(
            runtime_protocol_contract={
                "enable_pd": enable_pd,
                "pd_endpoint": pd_endpoint,
                "pd_mode": pd_mode,
                "pd_prefix_cache": pd_prefix_cache,
                "require_pd_service": require_pd_service,
            },
            source="m75_active_runtime_env",
        ),
        "effective_storage_backend": {
            "status": storage_status,
            "backend": storage_backend,
            "requested_backend": requested_storage_backend,
            "reason": "" if storage_status == "PASS" else "requested storage backend is not effective on this host/runtime",
        },
        "gds_effective": gds_effective,
        "spdk_effective": spdk_effective,
        "colossalai_effective": colossalai_effective,
        "artifacts": {
            "cloud_log": "/root/flashkv0516/cloud-deepseek-phase-a.log",
            "edge_log": f"local edge serve trace_id={trace_id} via CGC_CLOUD_TARGETS tunnel forward",
            "runtime_evidence_path": str(runtime_evidence_path),
            "local_infer_evidence_path": str(latest_evidence_path),
        },
        "local_infer_summary": {
            "mode": str(local_payload.get("mode") or ""),
            "state_source": str(local_payload.get("state_source") or ""),
            "seed_token_id": int(local_payload.get("seed_token_id") or 0),
            "edge_token_id": int(local_payload.get("edge_token_id") or 0),
            "state_meta": state_meta,
        },
    }
    runtime_payload["mandatory_protocol_gate"] = evaluate_mandatory_protocol_gate(
        runtime_protocol_contract=runtime_payload.get("runtime_protocol_contract"),
        zero_copy_vram_real=runtime_payload.get("zero_copy_vram_real"),
        source=str(latest_evidence_path),
    )
    if str(runtime_payload["mandatory_protocol_gate"].get("status") or "") != "PASS":
        runtime_payload["status"] = "FAIL"
    _write_json(runtime_evidence_path, runtime_payload)
    formal_updates, artifact_updates = _discover_formal_evidence_entries(
        runtime_evidence_path.parent,
        runtime_evidence_path.parent.parent,
        (WORKSPACE_ROOT / "temp" / "test").resolve(),
        manifest_payload=manifest_payload,
    )
    artifact_updates.update(
        {
            "m75_trueorthokda_active_runtime_path": str(runtime_evidence_path),
            "local_infer_evidence_path": str(latest_evidence_path),
        }
    )
    formal_updates["m75_active_runtime"] = {
        "filename": runtime_evidence_path.name,
        "path": str(runtime_evidence_path),
        "exists": True,
        "source": "m75_trueorthokda_active_runtime_gate",
        "payload": runtime_payload,
    }
    _upsert_system_execution_manifest(
        manifest_path,
        artifact_updates=artifact_updates,
        formal_updates=formal_updates,
    )
    return _status(
        True,
        runtime_evidence_path=str(runtime_evidence_path),
        local_infer_evidence_path=str(latest_evidence_path),
        mandatory_protocol_gate=runtime_payload.get("mandatory_protocol_gate"),
        system_execution_manifest_path=str(manifest_path) if manifest_path is not None else "",
    )


def run_m75_trueorthokda_active_runtime(*, output_dir: str) -> Dict[str, Any]:
    output_root = Path(str(output_dir)).expanduser().resolve()
    gate_dir = (output_root / "m75_trueorthokda_active").resolve()
    gate_dir.mkdir(parents=True, exist_ok=True)

    cloud_socket_server_path = (WORKSPACE_ROOT / "app" / "servers" / "cloud_socket_server.py").resolve()
    api_server_path = (WORKSPACE_ROOT / "app" / "servers" / "cgc_api_server.py").resolve()
    local_infer_path = (WORKSPACE_ROOT / "app" / "edge_engine" / "local_infer.py").resolve()
    kda_state_runtime_path = (WORKSPACE_ROOT / "app" / "edge_engine" / "kda_state_runtime.py").resolve()
    schema_path = (
        WORKSPACE_ROOT
        / "ComputeGraphCompiler-main"
        / "docs"
        / "gate_whitepapers"
        / "CGC_M75_TRUEORTHOKDA_ACTIVE_RUNTIME_REPORT_SCHEMA_v1.0.json"
    ).resolve()
    runtime_evidence_path = (
        output_root
        / "runtime_evidence"
        / "m75_trueorthokda_active_runtime.json"
    ).resolve()
    bootstrap_result = _bootstrap_active_runtime_evidence(runtime_evidence_path)
    manifest_path = _latest_system_execution_manifest_path(
        str(runtime_evidence_path.parent),
        str(output_root),
    )

    checks: Dict[str, Dict[str, Any]] = {
        "report_schema": _run_check("report_schema", _validate_report_schema, schema_path),
        "true_state_transport": _run_check(
            "true_state_transport",
            _validate_true_state_transport_contract,
            cloud_socket_server_path,
        ),
        "edge_state_resume_decode": _run_check(
            "edge_state_resume_decode",
            _validate_edge_state_resume_decode_contract,
            api_server_path,
            local_infer_path,
            kda_state_runtime_path,
        ),
        "runtime_evidence": _run_check(
            "runtime_evidence",
            _load_runtime_evidence,
            runtime_evidence_path,
        ),
    }
    runtime_evidence_payload = _read_json_safely(runtime_evidence_path)
    checks["mandatory_protocol_gate"] = evaluate_mandatory_protocol_gate(
        runtime_protocol_contract=runtime_evidence_payload.get("runtime_protocol_contract"),
        zero_copy_vram_real=runtime_evidence_payload.get("zero_copy_vram_real"),
        source=str(runtime_evidence_path),
    )

    passed_checks = [name for name, result in checks.items() if str(result.get("status") or "") == "PASS"]
    failed_checks = [name for name, result in checks.items() if str(result.get("status") or "") == "FAIL"]
    ok = not failed_checks

    gate = {
        "status": "PASS" if ok else "FAIL",
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "report_schema_path": str(schema_path),
        "runtime_evidence_path": str(runtime_evidence_path),
        "system_execution_manifest_path": str(manifest_path) if manifest_path is not None else "",
        "bootstrap": bootstrap_result,
    }
    _upsert_system_execution_manifest(manifest_path, gate_payload=gate)
    manifest_payload = _read_json_safely(manifest_path) if manifest_path is not None else {}
    gate["agent_execution"] = dict(manifest_payload.get("agent_execution") or {})
    gate["deepep_release_guard"] = dict(manifest_payload.get("deepep_release_guard") or {})
    gate["schema_refs"] = dict(manifest_payload.get("schema_refs") or {})
    report = {
        "ok": ok,
        "milestone": "m75_trueorthokda_active",
        "scope": "active_runtime",
        "public_entrypoint": "cgc gate m75-trueorthokda-active",
        "gate_result": {"m75_trueorthokda_active": gate},
    }
    summary = {
        "status": gate["status"],
        "milestone": "m75_trueorthokda_active",
        "runtime_evidence_path": str(runtime_evidence_path),
        "system_execution_manifest_path": str(manifest_path) if manifest_path is not None else "",
        "agent_execution": gate["agent_execution"],
        "deepep_release_guard": gate["deepep_release_guard"],
        "schema_refs": gate["schema_refs"],
    }
    report_path = (gate_dir / "m75_trueorthokda_active_report.json").resolve()
    summary_path = (gate_dir / "summary.json").resolve()
    latest_path = (gate_dir / "latest.json").resolve()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": ok,
        "gate_result": {"m75_trueorthokda_active": gate},
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "latest_path": str(latest_path),
    }


def main() -> None:
    print("=" * 60)
    print("🚀 CGC Engine M7.5 TrueOrthoKDA Active Runtime Gate")
    print("=" * 60)
    report = run_m75_trueorthokda_active_runtime(
        output_dir=str(
            (
                WORKSPACE_ROOT
                / "ComputeGraphCompiler-main"
                / "Output"
                / "cli_gate_m75_trueorthokda_active"
            ).resolve()
        )
    )
    gate = ((report.get("gate_result") or {}).get("m75_trueorthokda_active") or {})
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not bool(report.get("ok")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


def run_m75_trueorthokda_active_gate(*, output_dir: str) -> Dict[str, Any]:
    """Compatibility alias for older gate-oriented imports."""
    return run_m75_trueorthokda_active_runtime(output_dir=output_dir)
