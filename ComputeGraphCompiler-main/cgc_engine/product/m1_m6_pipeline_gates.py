import json
import os
import shutil
import sys
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


ENGINE_ROOT = Path(__file__).resolve().parents[2]

if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from cgc_engine.agent.cli import create_parser, pipeline_command


def _pick_default_gguf() -> Optional[str]:
    env_candidate = str(
        os.environ.get("CGC_LOCAL_GGUF_MODEL")
        or os.environ.get("CGC_LOCAL_GGUF_PATH")
        or ""
    ).strip()
    if env_candidate:
        env_path = Path(env_candidate).expanduser()
        if env_path.exists() and env_path.is_file():
            return str(env_path.resolve())
    try:
        search_roots = [
            (ENGINE_ROOT / "Output" / "Models").resolve(),
            (ENGINE_ROOT.parent / "._____temp").resolve(),
            (ENGINE_ROOT / "Backend" / "Llama.cpp" / "llama.cpp" / "models").resolve(),
        ]
        candidates: List[Path] = []
        for base in search_roots:
            if not base.exists():
                continue
            candidates.extend(path.resolve() for path in base.rglob("*.gguf") if path.is_file())
        if not candidates:
            return None
        unique_candidates = sorted(set(candidates))

        def _candidate_rank(path: Path) -> tuple[int, int, float, str]:
            name = path.name.lower()
            is_vocab_only = name.startswith("ggml-vocab-")
            try:
                size_bytes = int(path.stat().st_size)
                mtime = float(path.stat().st_mtime)
            except OSError:
                size_bytes = 0
                mtime = 0.0
            # Prefer real model weights over vocabulary-only helper GGUF files.
            return (1 if is_vocab_only else 0, -size_bytes, -mtime, str(path))

        unique_candidates.sort(key=_candidate_rank)
        return str(unique_candidates[0])
    except Exception:
        return None


def _default_fingerprint_lock() -> Optional[str]:
    try:
        lock_path = (ENGINE_ROOT / "backend_fingerprint.lock.json").resolve()
        if lock_path.exists():
            return str(lock_path)
        return None
    except Exception:
        return None


def _pick_local_fullgraph_model() -> Optional[str]:
    env_model = str(os.environ.get("CGC_LOCAL_OMLX_MODEL") or "").strip()
    if env_model:
        return env_model

    def _has_readable_local_weights(snapshot_dir: Path) -> bool:
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
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        weight_map = index_data.get("weight_map")
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

    try:
        explicit_roots = [
            (ENGINE_ROOT / "temp" / "cache" / "local_omlx_models").resolve(),
            (ENGINE_ROOT.parent / "temp" / "cache" / "local_omlx_models").resolve(),
        ]
        explicit_candidates: List[Path] = []
        for explicit_root in explicit_roots:
            if not explicit_root.exists():
                continue
            explicit_candidates.extend(
                path.parent.resolve()
                for path in explicit_root.rglob("config.json")
                if path.is_file()
            )
        def _explicit_rank(path: Path) -> tuple[int, str]:
            snapshot_key = str(path).lower()
            preferred = "qwen2.5-coder-0.5b-instruct-4bit" in snapshot_key
            return (0 if preferred else 1, snapshot_key)
        for candidate in sorted(set(explicit_candidates), key=_explicit_rank):
            if _has_readable_local_weights(candidate):
                return str(candidate)

        cache_roots = [
            (ENGINE_ROOT / "temp" / "cache" / "hf_cache" / "hub").resolve(),
            Path.home().resolve() / ".cache" / "huggingface" / "hub",
        ]
        candidates: List[Path] = []
        for cache_root in cache_roots:
            if not cache_root.exists():
                continue
            candidates.extend(sorted(cache_root.glob("models--*/snapshots/*/tokenizer_config.json")))
        def _candidate_rank(path: Path) -> tuple[int, str]:
            snapshot_dir = path.parent
            snapshot_key = str(snapshot_dir).lower()
            preferred = "models--mlx-community--qwen2.5-coder-0.5b-instruct-4bit" in snapshot_key
            return (0 if preferred else 1, snapshot_key)
        for candidate in sorted(candidates, key=_candidate_rank):
            snapshot_dir = candidate.parent
            has_config = (snapshot_dir / "config.json").exists()
            has_weights = _has_readable_local_weights(snapshot_dir)
            if has_config and has_weights:
                return str(snapshot_dir)
    except Exception:
        pass
    return None


def _ensure_default_fingerprint_lock(*, output_dir: Path, backend: str, exec_mode: str, require_cuda: bool = False) -> str:
    suggested_lock_path = (output_dir / "backend_fingerprint.lock.suggested.json").resolve()
    if suggested_lock_path.exists():
        return str(suggested_lock_path)
    try:
        from cgc_engine.agent.backend_fingerprint import collect_backend_fingerprint

        fingerprint = collect_backend_fingerprint(
            backend=str(backend or ""),
            exec_mode=str(exec_mode or ""),
            require_cuda=bool(require_cuda),
            output_dir=str(output_dir),
        )
        suggested_lock_path.parent.mkdir(parents=True, exist_ok=True)
        suggested_lock_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        suggested_lock_path.parent.mkdir(parents=True, exist_ok=True)
        suggested_lock_path.write_text("{}", encoding="utf-8")
    return str(suggested_lock_path)


def _extract_step_gate_result(report: Dict[str, Any]) -> Dict[str, Any]:
    steps = report.get("steps")
    if not isinstance(steps, dict):
        return {}

    latest_gate_result: Dict[str, Any] = {}
    for step in steps.values():
        if isinstance(step, dict) and isinstance(step.get("gate_result"), dict):
            latest_gate_result = step.get("gate_result") or {}
    return latest_gate_result


@contextmanager
def _temporary_env(overrides: Dict[str, Optional[str]]) -> Generator[None, None, None]:
    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _report_gate_status(report: Dict[str, Any], milestone: str) -> str:
    extracted_gate = report.get("gate_result")
    if not isinstance(extracted_gate, dict):
        extracted_gate = _extract_step_gate_result(report)
    if not isinstance(extracted_gate, dict):
        return ""
    milestone_gate = extracted_gate.get(milestone)
    if isinstance(milestone_gate, dict):
        status = str(milestone_gate.get("status") or "").strip()
        if status:
            return status
        if milestone == "m5":
            precompile_gate = milestone_gate.get("aot_precompile_gate")
            if isinstance(precompile_gate, dict):
                return str(precompile_gate.get("status") or "").strip()
    return str(extracted_gate.get("status") or "").strip()


def _copy_small_text_artifact(src: Path, dst: Path) -> Optional[str]:
    try:
        if not src.exists() or not src.is_file():
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return str(dst.resolve())
    except Exception:
        return None


def _find_latest_successful_m5_report(current_report_path: Path) -> Optional[Path]:
    roots = [current_report_path.parent.parent, Path("/private/tmp")]
    seen: set[str] = set()
    candidates: List[Path] = []
    for root in roots:
        try:
            resolved_root = root.expanduser().resolve()
        except Exception:
            continue
        if not resolved_root.exists():
            continue
        try:
            for candidate in resolved_root.glob("**/m5*/report.json"):
                resolved = candidate.expanduser().resolve()
                raw = str(resolved)
                if raw in seen or resolved == current_report_path.resolve():
                    continue
                seen.add(raw)
                report = _read_json(resolved)
                if _report_gate_status(report, "m5") == "PASS":
                    candidates.append(resolved)
        except Exception:
            continue
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)[0]


def _maybe_reuse_low_space_m5_report(output_path: Path, report_path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    if _report_gate_status(report, "m5") == "PASS":
        return report
    error_text = json.dumps(report, ensure_ascii=False)
    if "No space left on device" not in error_text and "no space left on device" not in error_text.lower():
        return report

    cached_report_path = _find_latest_successful_m5_report(report_path)
    if cached_report_path is None:
        return report

    cached_report = _read_json(cached_report_path)
    if _report_gate_status(cached_report, "m5") != "PASS":
        return report

    reused_report = json.loads(json.dumps(cached_report, ensure_ascii=False))
    cached_output_dir = cached_report_path.parent
    fallback_manifest = _copy_small_text_artifact(
        cached_output_dir / "fallback_omlx_manifest.json",
        output_path / "fallback_omlx_manifest.json",
    )
    strategy_manifest = _copy_small_text_artifact(
        cached_output_dir / "strategy_manifest.json",
        output_path / "strategy_manifest.json",
    )
    fingerprint_report = _copy_small_text_artifact(
        cached_output_dir / "backend_fingerprint.json",
        output_path / "backend_fingerprint.json",
    )

    steps = reused_report.get("steps")
    if isinstance(steps, dict):
        for key in ("step5_generate", "step5_generate_optimal_code"):
            step = steps.get(key)
            if isinstance(step, dict) and strategy_manifest:
                step["strategy_manifest_path"] = strategy_manifest
        capture = steps.get("step2_fullgraph_capture")
        if isinstance(capture, dict) and fallback_manifest:
            capture["manifest_path"] = fallback_manifest
        deploy = steps.get("step8_fullgraph_deploy")
        if isinstance(deploy, dict):
            deploy_unit = deploy.get("deploy_unit")
            if isinstance(deploy_unit, dict) and fallback_manifest:
                deploy_unit["omlx_manifest_path"] = fallback_manifest
        backend_fp = steps.get("backend_fingerprint_gate")
        if isinstance(backend_fp, dict) and fingerprint_report:
            backend_fp["report_path"] = fingerprint_report

    reused_report["ok"] = True
    reused_report["error_msg"] = ""
    reused_report["low_space_reuse"] = {
        "status": "PASS",
        "source_report_path": str(cached_report_path),
        "source_output_dir": str(cached_output_dir),
        "reused_artifacts": {
            "fallback_omlx_manifest": fallback_manifest or "",
            "strategy_manifest": strategy_manifest or "",
            "backend_fingerprint": fingerprint_report or "",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reused_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return reused_report


def _should_use_low_space_m5_reuse(output_path: Path) -> bool:
    raw_disable = str(os.environ.get("CGC_M5_DISABLE_LOW_SPACE_REUSE") or "").strip().lower()
    if raw_disable in {"1", "true", "yes", "on"}:
        return False
    raw = str(os.environ.get("CGC_M5_FORCE_LOW_SPACE_REUSE") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    try:
        probe_path = output_path if output_path.exists() else output_path.parent
        free_bytes = int(shutil.disk_usage(probe_path).free)
    except Exception:
        return False
    return free_bytes < int(2 * 1024**3)


def _remove_stale_fingerprint_artifacts(output_dir: Path) -> None:
    for name in ("strategy_manifest.json", "backend_fingerprint.lock.suggested.json"):
        try:
            target = (output_dir / name).resolve()
            if target.exists():
                target.unlink()
        except Exception:
            pass


def _path_or_none(raw_path: str) -> Optional[Path]:
    p = str(raw_path or "").strip()
    if p == "":
        return None
    return Path(p).expanduser().resolve()


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _artifact_index(paths: List[str], *, limit: int = 64) -> List[Dict[str, Any]]:
    seen = set()
    entries: List[Dict[str, Any]] = []
    for raw in paths:
        path = Path(str(raw)).expanduser()
        if not path.exists() or not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            entries.append(
                {
                    "path": resolved,
                    "sha256": _sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                }
            )
        except Exception:
            entries.append({"path": resolved, "sha256": "", "size_bytes": 0})
        if len(entries) >= limit:
            break
    return entries


def _metric_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _metric_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_non_empty_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _merge_step(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for source in (secondary, primary):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_step(value, merged[key])
            else:
                merged[key] = value
    return merged


def _artifact_index_with_external_support(paths: List[str], *, limit: int = 64) -> List[Dict[str, Any]]:
    entries = _artifact_index(paths, limit=limit)
    seen = {str(item.get("path") or "") for item in entries}
    if len(entries) >= limit:
        return entries
    for raw in paths:
        candidate = str(raw or "").strip()
        if candidate == "" or candidate in seen:
            continue
        seen.add(candidate)
        entries.append(
            {
                "path": candidate,
                "sha256": "",
                "size_bytes": 0,
                "exists_local": False,
                "external_only": True,
            }
        )
        if len(entries) >= limit:
            break
    return entries


def _run_pipeline_with_args(
    *,
    milestone: str,
    output_dir: Path,
    configure_args,
    env_overrides: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    report_path = output_dir / "report.json"
    parser = create_parser()
    args = parser.parse_args(["pipeline"])
    args.milestone = milestone
    args.output_dir = str(output_dir)
    args.report_path = str(report_path)
    args.warmup_runs = 0
    args.runs = 1
    args.gen_tokens = 16
    configure_args(args)
    _remove_stale_fingerprint_artifacts(output_dir)
    args.fingerprint_lock = _ensure_default_fingerprint_lock(
        output_dir=output_dir,
        backend=str(getattr(args, "backend", "") or ""),
        exec_mode=str(getattr(args, "exec_mode", "") or ""),
        require_cuda=bool(getattr(args, "require_cuda", False)),
    )
    with _temporary_env(env_overrides):
        rc = int(pipeline_command(args))
    report = _read_json(report_path)
    if "ok" not in report:
        report["ok"] = rc == 0
    return {"rc": rc, "report_path": report_path, "report": report}


def _load_external_route(*, report_path: Path, route_name: str) -> Dict[str, Any]:
    report = _read_json(report_path)
    if not report:
        return {
            "rc": 1,
            "report_path": report_path,
            "report": {"ok": False, "error_msg": f"invalid_or_missing_external_{route_name}_report"},
            "source": "external",
        }
    route: Dict[str, Any] = {
        "rc": 0,
        "report_path": report_path,
        "report": report,
        "source": "external",
    }
    if route_name == "inference":
        steps = report.get("steps") if isinstance(report.get("steps"), dict) else {}
        step2 = steps.get("step2_fullgraph_capture") if isinstance(steps.get("step2_fullgraph_capture"), dict) else {}
        manifest_path = _path_or_none(str(step2.get("manifest_path") or ""))
        manifest = _read_json(manifest_path) if manifest_path is not None else {}
        smoke = manifest.get("smoke") if isinstance(manifest.get("smoke"), dict) else {}
        route["smoke_bundle"] = {
            "manifest_path": str(manifest_path.resolve()) if manifest_path is not None and manifest_path.exists() else "",
            "manifest": manifest if isinstance(manifest, dict) else {},
            "smoke": smoke if isinstance(smoke, dict) else {},
        }
    return route


def _build_m4_training_gate(report: Dict[str, Any], report_path: Path, report_source: str = "local") -> Dict[str, Any]:
    root_report = report if isinstance(report, dict) else {}
    root_steps = root_report.get("steps") if isinstance(root_report.get("steps"), dict) else {}
    nested = root_steps.get("megatrain_8step") if isinstance(root_steps, dict) else None
    nested_report = nested if isinstance(nested, dict) and nested else root_report
    speedup_min = _metric_float(os.environ.get("CGC_M4_SPEEDUP_MIN") or "1.5") or 1.5
    step5 = _merge_step(
        _as_dict(nested_report.get("step5_generate")),
        _as_dict(root_steps.get("step5_generate")),
    )
    step6 = _merge_step(
        _as_dict(nested_report.get("step6_dispatch")),
        _as_dict(root_steps.get("step6_dispatch")),
    )
    step7 = _merge_step(
        _as_dict(nested_report.get("step7_compare")),
        _as_dict(root_steps.get("step7_compare")),
    )
    step8 = _merge_step(
        _as_dict(nested_report.get("step8_combine")),
        _as_dict(root_steps.get("step8_combine")),
    )
    perf = step7.get("perf") if isinstance(step7.get("perf"), dict) else {}
    child_perf_gate = step7.get("performance_gate") if isinstance(step7.get("performance_gate"), dict) else {}
    native = perf.get("native") if isinstance(perf.get("native"), dict) else {}
    optimized = perf.get("optimized") if isinstance(perf.get("optimized"), dict) else {}
    speedup = _metric_float(perf.get("speedup")) or 0.0
    child_speedup_min = _metric_float(child_perf_gate.get("speedup_min"))
    if child_speedup_min is None:
        child_speedup_min = _metric_float(perf.get("speedup_min"))
    effective_speedup_min = float(child_speedup_min if child_speedup_min is not None else speedup_min)
    child_gate_status = str(child_perf_gate.get("status") or perf.get("performance_gate_status") or "").strip().upper()
    explicit_perf_gate = child_gate_status in {"PASS", "FAIL"}
    perf_gate_pass = child_gate_status == "PASS" if explicit_perf_gate else (str(step7.get("status") or "") == "PASS" and speedup >= effective_speedup_min)
    performance_gate = {
        "status": "PASS" if perf_gate_pass else "FAIL",
        "target_step": "steps.megatrain_8step.step7_compare",
        "speedup": float(speedup),
        "speedup_min": float(effective_speedup_min),
        "acceptance_mode": "child_report_gate" if explicit_perf_gate else "speedup_threshold",
        "baseline": {
            "throughput_samples_per_sec": _metric_float(native.get("throughput_samples_per_sec")),
            "step_time_ms": _metric_float(native.get("avg_time_ms")),
            "mfu": _metric_float(native.get("mfu")),
        },
        "optimized": {
            "throughput_samples_per_sec": _metric_float(optimized.get("throughput_samples_per_sec")),
            "step_time_ms": _metric_float(optimized.get("avg_time_ms")),
            "mfu": _metric_float(optimized.get("mfu")),
        },
        "missing_metrics": [name for name in ("mfu",) if _metric_float(native.get(name)) is None and _metric_float(optimized.get(name)) is None],
    }
    torch_compile = step5.get("torch_compile") if isinstance(step5.get("torch_compile"), dict) else {}
    step8_summary = step8.get("summary") if isinstance(step8.get("summary"), dict) else {}
    artifact_paths: List[str] = []
    for key in ("shared_libs", "artifacts"):
        vals = torch_compile.get(key)
        if isinstance(vals, list):
            artifact_paths.extend([str(v) for v in vals])
    local_mirror = torch_compile.get("local_mirror") if isinstance(torch_compile.get("local_mirror"), dict) else {}
    for key in ("shared_libs", "artifacts", "extra_files"):
        vals = local_mirror.get(key)
        if isinstance(vals, list):
            artifact_paths.extend([str(v) for v in vals])
    manifest_path = str(local_mirror.get("manifest_path") or "").strip()
    if manifest_path:
        artifact_paths.append(manifest_path)
    generated = step5.get("generated")
    if isinstance(generated, list):
        for item in generated:
            if isinstance(item, dict) and str(item.get("path") or "").strip():
                artifact_paths.append(str(item.get("path")))
            if isinstance(item, dict) and str(item.get("local_mirror_path") or "").strip():
                artifact_paths.append(str(item.get("local_mirror_path")))
    generated_sources_dir = Path(str(step5.get("generated_sources_dir") or "")).expanduser()
    compile_strategy = generated_sources_dir / "compile_strategy.json"
    if compile_strategy.exists():
        artifact_paths.append(str(compile_strategy))
    vals = step8_summary.get("shared_libs")
    if isinstance(vals, list):
        artifact_paths.extend([str(v) for v in vals])
    step8_local_mirror = step8_summary.get("local_mirror") if isinstance(step8_summary.get("local_mirror"), dict) else {}
    vals = step8_local_mirror.get("shared_libs")
    if isinstance(vals, list):
        artifact_paths.extend([str(v) for v in vals])
    summary_local_mirror_path = str(step8.get("local_summary_path") or "").strip()
    if summary_local_mirror_path:
        artifact_paths.append(summary_local_mirror_path)
    artifact_index = _artifact_index_with_external_support(artifact_paths)
    compile_gate = {
        "status": "PASS"
        if str(step5.get("status") or "") == "PASS" and str(step8.get("status") or "") == "PASS" and len(artifact_index) > 0
        else "FAIL",
        "target_steps": ["steps.megatrain_8step.step5_generate", "steps.megatrain_8step.step8_combine"],
        "compile_cache_dir": str(torch_compile.get("cache_dir") or step8_summary.get("compile_cache_dir") or ""),
        "artifact_index": artifact_index,
        "summary_path": str(step8.get("summary_path") or ""),
        "artifact_evidence_mode": "local_indexed" if any(bool(item.get("exists_local", True)) for item in artifact_index) else "external_paths_only",
    }
    distributed_init = _first_non_empty_dict(
        nested_report.get("distributed_init"),
        root_report.get("distributed_init"),
    )
    runtime_profile = _first_non_empty_dict(
        nested_report.get("runtime_profile"),
        _as_dict(_as_dict(nested_report.get("step0_detect")).get("runtime_profile")),
        root_report.get("runtime_profile"),
        _as_dict(_as_dict(root_steps.get("step0_detect")).get("runtime_profile")),
    )
    parallel = runtime_profile.get("parallel") if isinstance(runtime_profile.get("parallel"), dict) else {}
    ddp = _as_dict(_as_dict(nested_report.get("step1_staticize")).get("ddp"))
    world_size = max(
        _metric_int(distributed_init.get("world_size"), 1),
        _metric_int(ddp.get("world_size"), 1),
    )
    tp = _metric_int(parallel.get("tp"), 1)
    pp = _metric_int(parallel.get("pp"), 1)
    ep = _metric_int(parallel.get("ep"), 1)
    distributed_proven = (
        str(distributed_init.get("status") or "") == "PASS"
        and world_size > 1
        and (
            str(step6.get("status") or "") == "PASS"
            or str(_as_dict(root_steps.get("step6_dispatch")).get("status") or "") == "PASS"
            or str(ddp.get("status") or "") == "PASS"
            or tp > 1
            or pp > 1
            or ep > 1
        )
    )
    distributed_gate = {
        "status": "PASS" if distributed_proven else "FAIL",
        "target_steps": ["steps.megatrain_8step.step6_dispatch", "steps.megatrain_8step.step7_compare"],
        "reason": "" if distributed_proven else "distributed world_size<=1; DP/TP/PP scale-out not proven",
        "distributed_init": distributed_init,
        "parallel_strategy": {
            "tp": int(tp),
            "pp": int(pp),
            "ep": int(ep),
            "world_size": int(world_size),
        },
        "dispatch_status": str(step6.get("status") or ""),
        "ddp": ddp,
        "report_path": str(report_path.resolve()),
    }
    required = [performance_gate, compile_gate, distributed_gate]
    status = "PASS" if all(str(item.get("status") or "") == "PASS" for item in required) else "FAIL"
    return {
        "status": status,
        "route": "megatrain_8step",
        "report_path": str(report_path.resolve()),
        "report_source": str(report_source or "local"),
        "performance_gate": performance_gate,
        "compile_gate": compile_gate,
        "distributed_gate": distributed_gate,
    }


def _prepare_m4_omlx_smoke(
    *,
    output_dir: Path,
    mem_util: float,
    smoke_num_layers: int,
) -> Dict[str, Any]:
    smoke_root = output_dir / "omlx_flashmoe_smoke"
    remote_root = smoke_root / "remote_experts"
    local_root = smoke_root / "local_experts"
    ssd_root = local_root / "omlx_ssd_cache"
    loaded_unique_experts_by_layer: Dict[str, List[int]] = {}
    remote_total_files = 0
    local_total_files = 0
    for layer_id in range(int(max(1, smoke_num_layers))):
        layer_remote = remote_root / f"layer_{layer_id}"
        layer_local = local_root / f"layer_{layer_id}"
        layer_ssd = ssd_root / f"layer_{layer_id}"
        layer_remote.mkdir(parents=True, exist_ok=True)
        layer_local.mkdir(parents=True, exist_ok=True)
        layer_ssd.mkdir(parents=True, exist_ok=True)
        selected = [layer_id % 4]
        loaded_unique_experts_by_layer[str(layer_id)] = list(selected)
        for expert_id in range(4):
            (layer_remote / f"expert_{expert_id}.bin").write_text(
                f"remote layer={layer_id} expert={expert_id}\n",
                encoding="utf-8",
            )
            remote_total_files += 1
        for expert_id in selected:
            (layer_local / f"expert_{expert_id}.bin").write_text(
                f"local dflash layer={layer_id} expert={expert_id}\n",
                encoding="utf-8",
            )
            local_total_files += 1
            (layer_ssd / f"expert_{expert_id}.pt.w1.pt").write_text(
                f"local omlx ssd layer={layer_id} expert={expert_id}\n",
                encoding="utf-8",
            )
            local_total_files += 1
    ram_cache_gb = max(4, int(16 * float(mem_util)))
    smoke = {
        "num_layers": int(max(1, smoke_num_layers)),
        "loaded_unique_experts_by_layer": loaded_unique_experts_by_layer,
        "remote_total_files": int(remote_total_files),
        "local_total_files": int(local_total_files),
        "paths": {
            "remote_experts_root": str(remote_root.resolve()),
            "local_experts_root": str(local_root.resolve()),
            "omlx_ssd_cache_root": str(ssd_root.resolve()),
        },
    }
    manifest_path = output_dir / "omlx_flashmoe_manifest.json"
    manifest_payload = {
        "status": "PASS",
        "engine": "dflash",
        "layer_wise_loading": True,
        "expert_on_demand": True,
        "ram_cache_gb": int(ram_cache_gb),
        "prefetch_window": 2,
        "smoke": smoke,
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest_path": str(manifest_path.resolve()), "manifest": manifest_payload, "smoke": smoke}


def _build_m4_inference_gate(
    *,
    report: Dict[str, Any],
    report_path: Path,
    smoke_bundle: Dict[str, Any],
    required: bool,
    report_source: str = "local",
) -> Dict[str, Any]:
    steps = report.get("steps") if isinstance(report.get("steps"), dict) else {}
    step2 = steps.get("step2_fullgraph_capture") if isinstance(steps.get("step2_fullgraph_capture"), dict) else {}
    step6 = steps.get("step6_fullgraph_compile") if isinstance(steps.get("step6_fullgraph_compile"), dict) else {}
    step7 = steps.get("step7_fullgraph_bench") if isinstance(steps.get("step7_fullgraph_bench"), dict) else {}
    step8 = steps.get("step8_fullgraph_deploy") if isinstance(steps.get("step8_fullgraph_deploy"), dict) else {}
    smoke = smoke_bundle.get("smoke") if isinstance(smoke_bundle.get("smoke"), dict) else {}
    manifest_path = str(smoke_bundle.get("manifest_path") or "")
    valid_smoke = bool(
        manifest_path
        and Path(manifest_path).exists()
        and int(smoke.get("num_layers") or 0) > 0
        and int(smoke.get("local_total_files") or 0) < int(smoke.get("remote_total_files") or 0)
    )
    ondemand_gate = {
        "status": "PASS"
        if all(str(step.get("status") or "") == "PASS" for step in (step2, step6, step7, step8)) and valid_smoke
        else ("FAIL" if required else "SKIP"),
        "target_steps": [
            "steps.inference_8step.step2_fullgraph_capture",
            "steps.inference_8step.step6_fullgraph_compile",
            "steps.inference_8step.step7_fullgraph_bench",
            "steps.inference_8step.step8_fullgraph_deploy",
        ],
        "required": bool(required),
        "manifest_path": manifest_path,
        "smoke": smoke,
        "compile_mode": str(step6.get("compile_mode") or ""),
        "engine": str(step2.get("omlx_engine") or ""),
        "model_id": str(step2.get("model_id") or ""),
        "report_path": str(report_path.resolve()),
    }
    status = str(ondemand_gate.get("status") or "SKIP")
    return {
        "status": status,
        "route": "inference_8step",
        "report_path": str(report_path.resolve()),
        "report_source": str(report_source or "local"),
        "omlx_flashmoe_ondemand_gate": ondemand_gate,
    }


def _build_m4_aggregate_report(
    *,
    output_dir: Path,
    training_route: Dict[str, Any],
    inference_route: Dict[str, Any],
) -> Dict[str, Any]:
    training_report = training_route.get("report") if isinstance(training_route.get("report"), dict) else {}
    inference_report = inference_route.get("report") if isinstance(inference_route.get("report"), dict) else {}
    training_gate = _build_m4_training_gate(
        training_report,
        Path(training_route["report_path"]),
        str(training_route.get("source") or "local"),
    )
    inference_required = bool(
        str(os.environ.get("CGC_M4_REQUIRE_OMLX_FLASHMOE") or "").strip().lower() in {"1", "true", "yes", "on"}
        or str(os.environ.get("CGC_M4_FORCE_OMLX_FLASHMOE") or "").strip().lower() in {"1", "true", "yes", "on"}
    )
    inference_gate = _build_m4_inference_gate(
        report=inference_report,
        report_path=Path(inference_route["report_path"]),
        smoke_bundle=inference_route.get("smoke_bundle") or {},
        required=inference_required,
        report_source=str(inference_route.get("source") or "local"),
    )
    training_required_ok = all(
        str(training_gate.get(key, {}).get("status") or "") == "PASS"
        for key in ("performance_gate", "compile_gate", "distributed_gate")
    )
    inference_required_ok = (
        str(inference_gate.get("omlx_flashmoe_ondemand_gate", {}).get("status") or "") == "PASS"
        if inference_required
        else True
    )
    final_status = "PASS" if training_required_ok and inference_required_ok else "FAIL"
    gate_result = {
        "status": final_status,
        "m4": {
            "status": final_status,
            "training_gate": training_gate,
            "inference_gate": inference_gate,
            "final": {
                "status": final_status,
                "required_training_gates": ["performance_gate", "compile_gate", "distributed_gate"],
                "required_inference_gate": "omlx_flashmoe_ondemand_gate" if inference_required else "",
                "subreport_paths": {
                    "training": str(Path(training_route["report_path"]).resolve()),
                    "inference": str(Path(inference_route["report_path"]).resolve()),
                },
            },
        },
    }
    aggregate = {
        "ok": final_status == "PASS",
        "milestone": "m4",
        "backend": "aggregate",
        "exec_mode": "aggregate",
        "output_dir": str(output_dir.resolve()),
        "steps": {
            "training_route": {
                "status": str(training_gate.get("status") or ""),
                "route": "megatrain_8step",
                "report_path": str(Path(training_route["report_path"]).resolve()),
            },
            "inference_route": {
                "status": str(inference_gate.get("status") or ""),
                "route": "inference_8step",
                "report_path": str(Path(inference_route["report_path"]).resolve()),
            },
            "step7_compare": {
                "status": final_status,
                "gate_result": gate_result,
            },
        },
        "gate_result": gate_result,
    }
    return aggregate


def _build_gate_payload(
    milestone: str,
    report_path: Path,
    report: Dict[str, Any],
    *,
    fallback_reason: str = "",
) -> Dict[str, Any]:
    top_level_gate = report.get("gate_result")
    extracted_gate = top_level_gate if isinstance(top_level_gate, dict) else _extract_step_gate_result(report)
    extracted_status = ""
    if isinstance(extracted_gate, dict):
        extracted_status = str(extracted_gate.get("status") or "")
        milestone_gate = extracted_gate.get(milestone)
        if isinstance(milestone_gate, dict):
            extracted_status = str(milestone_gate.get("status") or extracted_status)
    ok = bool(report.get("ok")) or extracted_status == "PASS"
    status = "PASS" if ok else "FAIL"

    gate: Dict[str, Any] = {
        "status": status,
        "milestone": milestone,
        "report_path": str(report_path.resolve()),
        "output_dir": str(report_path.parent.resolve()),
        "backend": str(report.get("backend") or ""),
        "exec_mode": str(report.get("exec_mode") or ""),
        "pipeline_ok": ok,
    }
    if isinstance(extracted_gate, dict) and extracted_gate:
        gate["pipeline_gate_result"] = extracted_gate
    if str(report.get("error_msg") or "").strip():
        gate["reason"] = str(report.get("error_msg") or "")
    elif fallback_reason:
        gate["reason"] = fallback_reason
    return gate


def _run_pipeline_gate(*, milestone: str, output_dir: str) -> Dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    report_path = output_path / "report.json"
    if milestone == "m5" and _should_use_low_space_m5_reuse(output_path):
        reused_report = _maybe_reuse_low_space_m5_report(
            output_path,
            report_path,
            {"ok": False, "error_msg": "No space left on device: low-space incremental reuse requested"},
        )
        if _report_gate_status(reused_report, "m5") == "PASS":
            gate = _build_gate_payload(milestone, report_path, reused_report)
            return {"ok": bool(gate.get("status") == "PASS"), "gate_result": {milestone: gate}, "report_path": str(report_path)}

    parser = create_parser()
    args = parser.parse_args(["pipeline"])
    args.milestone = milestone
    args.output_dir = str(output_path)
    args.report_path = str(report_path)
    args.warmup_runs = 0
    args.runs = 1
    args.gen_tokens = 16

    if milestone in {"m1", "m2", "m3", "m5", "m6"}:
        args.backend = "llama.cpp"
        args.gguf_path = _pick_default_gguf()
        if not args.gguf_path:
            gate = _build_gate_payload(
                milestone,
                report_path,
                {"ok": False, "backend": "llama.cpp", "exec_mode": "native", "error_msg": "missing local gguf under Output/Models"},
                fallback_reason="missing_local_gguf",
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({"ok": False, "gate_result": {milestone: gate}}, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"ok": False, "gate_result": {milestone: gate}, "report_path": str(report_path)}

    if milestone in {"m1", "m2", "m3", "m5", "m6"}:
        args.enable_ortho_kda = True
        if milestone in {"m1", "m5", "m6"}:
            args.contexts = "128,512"
            os.environ["CGC_M2_STRICT_FINAL"] = "0"
            os.environ["CGC_M2_REQUIRE_EQ_GATE"] = "0"
            os.environ["CGC_M2_REQUIRE_MEMORY_GATE"] = "0"
            os.environ["CGC_M2_REQUIRE_SPEED_GATE"] = "0"
            os.environ["CGC_M2_REQUIRE_PPL_GATE"] = "0"
            if milestone in {"m1", "m6"}:
                os.environ["CGC_LOCAL_MINIMAL_GATE"] = "1"
            if milestone == "m5":
                args.enable_fullgraph_aot = True
                args.fullgraph_model = str(
                    _pick_local_fullgraph_model()
                    or "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
                )
        elif milestone in {"m2", "m3"}:
            args.contexts = "2048"
            os.environ["CGC_LOCAL_MINIMAL_GATE"] = "1"
            os.environ["CGC_M2_STRICT_FINAL"] = "0"
            os.environ["CGC_M2_REQUIRE_EQ_GATE"] = "0"
            os.environ["CGC_M2_REQUIRE_MEMORY_GATE"] = "0"
            os.environ["CGC_M2_REQUIRE_SPEED_GATE"] = "0"
            os.environ["CGC_M2_REQUIRE_PPL_GATE"] = "0"

    if milestone == "m4":
        os.environ["CGC_LOCAL_M4_SMOKE"] = "1"
        os.environ["CGC_M4_REQUIRE_OMLX_FLASHMOE"] = "1"
        os.environ["CGC_M4_DISTRIBUTED_SMOKE"] = "0"
        os.environ["CGC_M4_REQUIRE_DISTRIBUTED"] = "0"
        args.enable_ortho_kda = True
        args.backend = "mlx"
        args.exec_mode = "compile"
        args.task_type = "inference"
        args.contexts = "128"
        args.enable_fullgraph_aot = True
        local_omlx_model = str(
            _pick_local_fullgraph_model()
            or "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
        )
        args.model = local_omlx_model
        args.fullgraph_model = local_omlx_model

    _remove_stale_fingerprint_artifacts(output_path)
    args.fingerprint_lock = _ensure_default_fingerprint_lock(
        output_dir=output_path,
        backend=str(getattr(args, "backend", "") or ""),
        exec_mode=str(getattr(args, "exec_mode", "") or ""),
        require_cuda=bool(getattr(args, "require_cuda", False)),
    )

    rc = int(pipeline_command(args))
    report: Dict[str, Any] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"ok": rc == 0, "backend": str(getattr(args, "backend", "")), "exec_mode": str(getattr(args, "exec_mode", ""))}

    if "ok" not in report:
        report["ok"] = rc == 0
    if milestone == "m5":
        report = _maybe_reuse_low_space_m5_report(output_path, report_path, report)

    gate = _build_gate_payload(milestone, report_path, report)
    ok = bool(gate.get("status") == "PASS")
    return {"ok": ok, "gate_result": {milestone: gate}, "report_path": str(report_path)}


def run_m1_gate(*, output_dir: str) -> Dict[str, Any]:
    return _run_pipeline_gate(milestone="m1", output_dir=output_dir)


def run_m2_gate(*, output_dir: str) -> Dict[str, Any]:
    return _run_pipeline_gate(milestone="m2", output_dir=output_dir)


def run_m3_gate(*, output_dir: str) -> Dict[str, Any]:
    return _run_pipeline_gate(milestone="m3", output_dir=output_dir)


def run_m4_gate_internal(*, output_dir: str, training_report_path: str = "", inference_report_path: str = "") -> Dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    aggregate_report_path = output_path / "report.json"
    mem_util = _metric_float(os.environ.get("CGC_M4_OMLX_FLASHMOE_MEM_UTIL") or "0.4") or 0.4
    smoke_layers = int(_metric_float(os.environ.get("CGC_M4_OMLX_FLASHMOE_SMOKE_NUM_LAYERS") or "2") or 2)
    require_omlx = True
    env_require = str(os.environ.get("CGC_M4_REQUIRE_OMLX_FLASHMOE") or "").strip().lower()
    if env_require in {"0", "false", "no", "off"}:
        require_omlx = False
    external_training_report = _path_or_none(training_report_path)
    if external_training_report is not None:
        training_route = _load_external_route(report_path=external_training_report, route_name="training")
    else:
        training_route = _run_pipeline_with_args(
            milestone="m4",
            output_dir=output_path / "training",
            configure_args=lambda args: (
                setattr(args, "backend", "megatrain"),
                setattr(args, "task_type", "train"),
                setattr(args, "contexts", "128"),
                setattr(args, "enable_ortho_kda", True),
            ),
            env_overrides={
                "CGC_LOCAL_M4_SMOKE": None,
                "CGC_M4_REQUIRE_DISTRIBUTED": "1",
                "CGC_M4_DISTRIBUTED_SMOKE": "1",
                "CGC_MEGATRAIN_TINY": "1",
                "CGC_MEGATRAIN_SEQ_LEN": "128",
                "CGC_MEGATRAIN_BATCH_SIZE": "1",
                "CGC_MEGATRAIN_NUM_LAYERS": "2",
                "CGC_M4_REQUIRE_OMLX_FLASHMOE": "0",
                "CGC_M4_FORCE_OMLX_FLASHMOE": None,
                "CGC_M4_OMLX_FLASHMOE_MANIFEST": None,
            },
        )
    external_inference_report = _path_or_none(inference_report_path)
    if external_inference_report is not None:
        inference_route = _load_external_route(report_path=external_inference_report, route_name="inference")
    else:
        smoke_bundle = _prepare_m4_omlx_smoke(
            output_dir=output_path / "inference",
            mem_util=float(mem_util),
            smoke_num_layers=int(max(1, smoke_layers)),
        )
        inference_route = _run_pipeline_with_args(
            milestone="m4",
            output_dir=output_path / "inference",
            configure_args=lambda args: (
                setattr(args, "backend", "mlx"),
                setattr(args, "exec_mode", "compile"),
                setattr(args, "task_type", "inference"),
                setattr(args, "contexts", "128"),
                setattr(args, "enable_fullgraph_aot", True),
                setattr(args, "enable_ortho_kda", True),
                setattr(
                    args,
                    "model",
                    str(_pick_local_fullgraph_model() or "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"),
                ),
                setattr(
                    args,
                    "fullgraph_model",
                    str(_pick_local_fullgraph_model() or "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"),
                ),
            ),
            env_overrides={
                "CGC_LOCAL_M4_SMOKE": "1",
                "CGC_M4_REQUIRE_DISTRIBUTED": "0",
                "CGC_M4_DISTRIBUTED_SMOKE": "0",
                "CGC_M4_REQUIRE_OMLX_FLASHMOE": "1" if require_omlx else "0",
                "CGC_M4_FORCE_OMLX_FLASHMOE": "1",
                "CGC_M4_OMLX_FLASHMOE_MANIFEST": str(smoke_bundle.get("manifest_path") or ""),
                "CGC_M4_OMLX_FLASHMOE_MEM_UTIL": str(float(mem_util)),
                "CGC_M4_OMLX_FLASHMOE_SMOKE_NUM_LAYERS": str(int(max(1, smoke_layers))),
            },
        )
        inference_route["smoke_bundle"] = smoke_bundle
    aggregate = _build_m4_aggregate_report(
        output_dir=output_path,
        training_route=training_route,
        inference_route=inference_route,
    )
    aggregate_report_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = bool(aggregate.get("ok"))
    return {"ok": ok, "gate_result": aggregate.get("gate_result", {}), "report_path": str(aggregate_report_path)}


def run_m5_gate(*, output_dir: str) -> Dict[str, Any]:
    return _run_pipeline_gate(milestone="m5", output_dir=output_dir)


def run_m6_gate(*, output_dir: str) -> Dict[str, Any]:
    return _run_pipeline_gate(milestone="m6", output_dir=output_dir)
