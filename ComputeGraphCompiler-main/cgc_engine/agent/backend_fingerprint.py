from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _is_truthy(v: Any) -> bool:
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "on"}


def _sha256_file(path: str) -> Optional[str]:
    p = Path(str(path))
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _pkg_info(mod_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": str(mod_name)}
    try:
        spec = importlib.util.find_spec(mod_name)
        out["spec_found"] = bool(spec is not None)
        out["origin"] = str(getattr(spec, "origin", "") or "") if spec is not None else ""
    except Exception as e:
        out["spec_found"] = False
        out["origin"] = ""
        out["spec_error"] = repr(e)

    try:
        mod = importlib.import_module(mod_name)
        out["import_ok"] = True
        out["file"] = str(getattr(mod, "__file__", "") or "")
        out["version"] = str(getattr(mod, "__version__", "") or "")
    except Exception as e:
        out["import_ok"] = False
        out["import_error"] = repr(e)
        out["file"] = ""
        out["version"] = ""
    return out


def _get_nested(d: Any, keys: Tuple[str, ...]) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _norm_fingerprint_value(keys: Tuple[str, ...], v: Any) -> Any:
    if not isinstance(v, str):
        return v
    if len(keys) > 0 and keys[-1] in {"file", "origin", "_C_file", "executable", "path"}:
        try:
            return os.path.realpath(v)
        except Exception:
            return v
    return v


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _builtin_backend_fingerprint_lock_path() -> str:
    p = _repo_root() / "backend_fingerprint.lock.json"
    if p.exists() and p.is_file():
        return str(p)
    return ""


def resolve_backend_fingerprint_lock_path(lock_path: Optional[str]) -> str:
    raw = str(lock_path or "").strip()
    builtin = _builtin_backend_fingerprint_lock_path()
    if raw == "":
        return builtin

    p = Path(raw).expanduser()
    # Legacy callers often pass a bare `lock.json`; prefer the repo-maintained
    # fingerprint baseline so cwd changes do not silently bind the wrong file.
    if not p.is_absolute() and p.name == "lock.json" and builtin:
        return builtin

    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def collect_backend_fingerprint(*, backend: str, exec_mode: str, require_cuda: bool, output_dir: str) -> Dict[str, Any]:
    fp: Dict[str, Any] = {
        "backend": str(backend),
        "exec_mode": str(exec_mode),
        "require_cuda": bool(require_cuda),
        "python": {
            "executable": str(sys.executable),
            "version": str(sys.version),
            "platform": str(platform.platform()),
        },
        "env": {
            "CGC_REQUIRE_CUDA": str(os.environ.get("CGC_REQUIRE_CUDA") or ""),
            "CGC_REQUIRE_MLX": str(os.environ.get("CGC_REQUIRE_MLX") or ""),
            "CGC_VLLM_USE_VENDOR": str(os.environ.get("CGC_VLLM_USE_VENDOR") or ""),
            "CGC_VLLM_REQUIRE_KDA": str(os.environ.get("CGC_VLLM_REQUIRE_KDA") or ""),
            "CUDA_VISIBLE_DEVICES": str(os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
        },
        "packages": {
            "torch": _pkg_info("torch"),
            "transformers": _pkg_info("transformers"),
            "vllm": _pkg_info("vllm"),
        },
        "artifacts": {},
    }

    try:
        import torch  # type: ignore

        fp["packages"]["torch"]["cuda_available"] = bool(torch.cuda.is_available())
        fp["packages"]["torch"]["cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
        try:
            fp["packages"]["torch"]["mps_available"] = bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
        except Exception:
            fp["packages"]["torch"]["mps_available"] = False
        try:
            fp["packages"]["torch"]["cuda_device_count"] = int(torch.cuda.device_count())
        except Exception:
            fp["packages"]["torch"]["cuda_device_count"] = 0
        try:
            if bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0:
                fp["packages"]["torch"]["cuda_device_0"] = str(torch.cuda.get_device_name(0))
        except Exception:
            pass
    except Exception:
        pass

    if str(backend) == "vllm":
        try:
            import vllm  # type: ignore

            fp["packages"]["vllm"]["version"] = str(getattr(vllm, "__version__", "") or "")
            fp["packages"]["vllm"]["file"] = str(getattr(vllm, "__file__", "") or "")
        except Exception:
            pass
        try:
            mod = importlib.import_module("vllm._C")
            fp["packages"]["vllm"]["has__C"] = True
            fp["packages"]["vllm"]["_C_file"] = str(getattr(mod, "__file__", "") or "")
        except Exception as e:
            fp["packages"]["vllm"]["has__C"] = False
            fp["packages"]["vllm"]["_C_error"] = repr(e)

    try:
        import cgc_cpp  # type: ignore

        fp["packages"]["cgc_cpp"] = _pkg_info("cgc_cpp")
        try:
            fp["packages"]["cgc_cpp"]["platform_name"] = str(cgc_cpp.get_platform_name())
        except Exception:
            pass
    except Exception:
        fp["packages"]["cgc_cpp"] = _pkg_info("cgc_cpp")

    try:
        from cgc_engine.gds_service.cufile_wrapper import (
            CUFILE_AVAILABLE as _CUFILE_AVAILABLE,
            get_gds_backend as _get_gds_backend,
            get_gds_capabilities as _get_gds_capabilities,
        )

        fp["packages"]["gds"] = {
            "import_ok": True,
            "cufile_available": bool(_CUFILE_AVAILABLE),
            "backend": str(_get_gds_backend()),
            "capabilities": _get_gds_capabilities(),
        }
    except Exception as e:
        fp["packages"]["gds"] = {
            "import_ok": False,
            "import_error": repr(e),
        }

    try:
        out_dir = Path(str(output_dir)).resolve()
        for name in ("strategy_manifest.json", "bundle_manifest.json"):
            p = out_dir / name
            if p.exists():
                fp["artifacts"][name] = {"path": str(p), "sha256": _sha256_file(str(p))}
    except Exception:
        pass

    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = {
            "llama_cli": repo_root / "cgc_run_m2_attention_verify" / "build_llama" / "bin" / "llama-cli",
            "llama_bench": repo_root / "cgc_run_m2_attention_verify" / "build_llama" / "bin" / "llama-bench",
            "llama_perplexity": repo_root / "cgc_run_m2_attention_verify" / "build_llama" / "bin" / "llama-perplexity",
            "ggml_cgc": repo_root / "cgc_run_m2_attention_verify" / "build_llama" / "bin" / "ggml-cgc",
            "ggml_backends_llama_bench": repo_root / "cgc_run_m2_attention_verify" / "ggml_backends" / "llama-bench",
        }
        for k, p in candidates.items():
            if p.exists() and p.is_file():
                fp["artifacts"][k] = {"path": str(p), "sha256": _sha256_file(str(p))}
    except Exception:
        pass

    return fp


def validate_backend_fingerprint(
    fp: Dict[str, Any],
    *,
    backend: str,
    require_cuda: bool,
    require_mlx: bool,
    lock_path: str,
    lock_required: bool,
) -> Dict[str, Any]:
    errors = []

    if bool(require_cuda) and bool(require_mlx):
        errors.append("STRICT GATE FAIL: CGC_REQUIRE_CUDA and CGC_REQUIRE_MLX are both set")

    if bool(require_mlx):
        if str(platform.system()) != "Darwin":
            errors.append("STRICT GATE FAIL: require MLX but platform is not Darwin")
        torch_info = ((fp.get("packages") or {}).get("torch") or {}) if isinstance(fp.get("packages"), dict) else {}
        if not bool(torch_info.get("import_ok")):
            errors.append("STRICT GATE FAIL: require MLX but torch not importable")
        if bool(torch_info.get("import_ok")) and not bool(torch_info.get("mps_available")):
            errors.append("STRICT GATE FAIL: require MLX but torch.backends.mps.is_available() is false")
        if str(backend) in {"vllm"}:
            errors.append("STRICT GATE FAIL: require MLX but backend is vllm (CUDA-only backend)")

    # === NEW: DFlash / FlashKV Gate ===
    # Validate DFlash configuration if requested via environment variable.
    require_dflash = os.environ.get("CGC_REQUIRE_DFLASH", "0") == "1"
    dflash_enabled = os.environ.get("CGC_DFLASH_ENABLED", "0") == "1"
    dflash_draft_model = os.environ.get("CGC_DFLASH_DRAFT_MODEL", "")
    
    if require_dflash:
        if not dflash_enabled:
            errors.append(
                "STRICT GATE FAIL: DFlash (FlashKV) is required but CGC_DFLASH_ENABLED is not set to 1. "
                "You must enable DFlash for this pipeline execution."
            )
        if not dflash_draft_model:
            errors.append(
                "STRICT GATE FAIL: DFlash (FlashKV) is enabled but no draft model is specified. "
                "You must set CGC_DFLASH_DRAFT_MODEL (e.g., z-lab/Qwen3.5-4B-DFlash)."
            )

    # === NEW: Advanced Hardware & Acceleration Gates (SPDK, GDS, oMLX, FlashMoE) ===
    require_spdk = os.environ.get("CGC_REQUIRE_SPDK", "0") == "1"
    spdk_enabled = os.environ.get("CGC_SPDK_ENABLED", "0") == "1"
    
    require_gds = os.environ.get("CGC_REQUIRE_GDS", "0") == "1"
    gds_enabled = os.environ.get("CGC_GDS_ENABLED", "0") == "1"
    
    require_omlx = os.environ.get("CGC_REQUIRE_OMLX", "0") == "1"
    omlx_enabled = os.environ.get("CGC_OMLX_ENABLED", "0") == "1"
    
    require_flashmoe = os.environ.get("CGC_REQUIRE_FLASHMOE", "0") == "1"
    flashmoe_enabled = os.environ.get("CGC_FLASHMOE_ENABLED", "0") == "1"

    if require_spdk and not spdk_enabled:
        errors.append("STRICT GATE FAIL: SPDK (Storage Performance Development Kit) is required but CGC_SPDK_ENABLED is not set to 1.")
        
    if require_gds:
        gds_info = ((fp.get("packages") or {}).get("gds") or {}) if isinstance(fp.get("packages"), dict) else {}
        gds_caps = gds_info.get("capabilities") if isinstance(gds_info, dict) else {}
        if not gds_enabled:
            errors.append("STRICT GATE FAIL: GPUDirect Storage (GDS) is required but CGC_GDS_ENABLED is not set to 1.")
        if not require_cuda:
            errors.append("STRICT GATE FAIL: GPUDirect Storage (GDS) requires CUDA, but CGC_REQUIRE_CUDA is not 1.")
        if not bool(gds_info.get("import_ok")):
            errors.append("STRICT GATE FAIL: GPUDirect Storage (GDS) capability probe is not importable.")
        if not bool(gds_info.get("cufile_available")):
            errors.append("STRICT GATE FAIL: libcufile/cuda.bindings.cufile is not available for GPUDirect Storage.")
        if isinstance(gds_caps, dict):
            if not bool(gds_caps.get("storage_path_eligible")):
                errors.append("STRICT GATE FAIL: no NVMe or NFSoRDMA storage path is eligible for GPUDirect Storage.")
            if bool(gds_caps.get("nfs_mounts")) and not bool(gds_caps.get("nfs_rdma_mounts")):
                errors.append("STRICT GATE FAIL: NFS is mounted but not over RDMA; this is not NFSoRDMA direct-to-GPU.")
            if bool(gds_caps.get("nfs_rdma_mounts")) and not bool(gds_caps.get("rdma_devices")):
                errors.append("STRICT GATE FAIL: NFSoRDMA mount is declared, but no RDMA device is present on this host.")

    if require_omlx:
        if not omlx_enabled:
            errors.append("STRICT GATE FAIL: oMLX (Optimized MLX) is required but CGC_OMLX_ENABLED is not set to 1.")
        if not require_mlx:
            errors.append("STRICT GATE FAIL: oMLX requires MLX backend, but CGC_REQUIRE_MLX is not 1.")

    if require_flashmoe and not flashmoe_enabled:
        errors.append("STRICT GATE FAIL: FlashMoE is required but CGC_FLASHMOE_ENABLED is not set to 1.")

    if require_cuda:
        if str(backend) == "vllm":
            torch_info = ((fp.get("packages") or {}).get("torch") or {}) if isinstance(fp.get("packages"), dict) else {}
            if not bool(torch_info.get("import_ok")):
                errors.append("STRICT GATE FAIL: require CUDA but torch not importable for vllm backend")
            if bool(torch_info.get("import_ok")) and not bool(torch_info.get("cuda_available")):
                errors.append("STRICT GATE FAIL: require CUDA but torch.cuda.is_available() is false")

    if str(backend) == "vllm":
        vllm_info = ((fp.get("packages") or {}).get("vllm") or {}) if isinstance(fp.get("packages"), dict) else {}
        if not bool(vllm_info.get("import_ok")):
            errors.append(f"STRICT GATE FAIL: vllm import failed: {vllm_info.get('import_error')}")
        if not bool(vllm_info.get("has__C")):
            errors.append(f"STRICT GATE FAIL: vllm._C missing: {vllm_info.get('_C_error')}")

        use_vendor = _is_truthy(((fp.get("env") or {}).get("CGC_VLLM_USE_VENDOR") if isinstance(fp.get("env"), dict) else ""))
        if use_vendor:
            vfile = str(vllm_info.get("file") or "")
            if "Backend/Vllm" not in vfile and "/Backend/Vllm" not in vfile:
                errors.append("STRICT GATE FAIL: CGC_VLLM_USE_VENDOR=1 but imported vllm is not from vendor path")

    lock_path = str(lock_path or "").strip()
    lock_required = bool(lock_required)
    if lock_required and lock_path == "":
        errors.append(
            "STRICT GATE FAIL: CGC_BACKEND_FINGERPRINT_LOCK_REQUIRED is 1, but CGC_BACKEND_FINGERPRINT_LOCK is not set."
        )

    if lock_path != "":
        p = Path(lock_path)
        if p.exists() and p.is_file():
            try:
                lock = json.loads(p.read_text(encoding="utf-8"))
                keys: Tuple[Tuple[str, ...], ...] = (
                    ("python", "executable"),
                    ("packages", "torch", "version"),
                    ("packages", "torch", "file"),
                    ("packages", "torch", "origin"),
                    ("packages", "transformers", "version"),
                    ("packages", "transformers", "file"),
                    ("packages", "transformers", "origin"),
                    ("packages", "vllm", "version"),
                    ("packages", "vllm", "file"),
                    ("packages", "vllm", "origin"),
                    ("packages", "vllm", "_C_file"),
                    ("packages", "cgc_cpp", "version"),
                    ("packages", "cgc_cpp", "file"),
                    ("packages", "cgc_cpp", "origin"),
                    ("artifacts", "strategy_manifest.json", "sha256"),
                    ("artifacts", "bundle_manifest.json", "sha256"),
                    ("artifacts", "llama_cli", "sha256"),
                    ("artifacts", "llama_bench", "sha256"),
                    ("artifacts", "llama_perplexity", "sha256"),
                    ("artifacts", "ggml_cgc", "sha256"),
                    ("artifacts", "ggml_backends_llama_bench", "sha256"),
                )
                for k in keys:
                    want = _get_nested(lock, k)
                    got = _get_nested(fp, k)
                    want_n = _norm_fingerprint_value(k, want)
                    got_n = _norm_fingerprint_value(k, got)
                    if want_n not in (None, "", "ANY") and str(want_n) != str(got_n):
                        errors.append(f"STRICT GATE FAIL: backend fingerprint mismatch: {'.'.join(k)}")
            except Exception as e:
                errors.append(f"STRICT GATE FAIL: invalid backend fingerprint lock: {repr(e)}")
        else:
            errors.append("STRICT GATE FAIL: backend fingerprint lock path does not exist")

    status = "PASS" if len(errors) == 0 else "FAIL"
    return {"status": status, "errors": errors, "lock_path": lock_path}


def write_suggested_lock(output_dir: Optional[str], fp: Dict[str, Any]) -> str:
    out_dir = Path(str(output_dir or "/tmp/llm_auto_pipeline_output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "backend_fingerprint.lock.suggested.json"
    p.write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def generate_and_verify_fingerprint(
    output_dir: Optional[str],
    *,
    backend: str,
    exec_mode: str,
    require_cuda: bool,
    require_mlx: bool,
    # --- NEW: Inject Advanced CLI Options ---
    disable_audit: bool = False,
    require_dflash: bool = False,
    require_spdk: bool = False,
    require_gds: bool = False,
    require_omlx: bool = False,
    require_flashmoe: bool = False,
) -> Dict[str, Any]:
    # Pass CLI options to environment for validation
    if require_dflash: os.environ["CGC_REQUIRE_DFLASH"] = "1"
    if require_spdk: os.environ["CGC_REQUIRE_SPDK"] = "1"
    if require_gds: os.environ["CGC_REQUIRE_GDS"] = "1"
    if require_omlx: os.environ["CGC_REQUIRE_OMLX"] = "1"
    if require_flashmoe: os.environ["CGC_REQUIRE_FLASHMOE"] = "1"

    fp = collect_backend_fingerprint(
        backend=str(backend),
        exec_mode=str(exec_mode),
        require_cuda=bool(require_cuda),
        output_dir=str(output_dir or "/tmp/llm_auto_pipeline_output"),
    )
    suggest_lock_path = write_suggested_lock(output_dir, fp)
    lock_path = resolve_backend_fingerprint_lock_path(os.environ.get("CGC_BACKEND_FINGERPRINT_LOCK"))
    if lock_path != "":
        os.environ["CGC_BACKEND_FINGERPRINT_LOCK"] = lock_path
    lock_required_env = str(os.environ.get("CGC_BACKEND_FINGERPRINT_LOCK_REQUIRED") or "1").strip().lower()
    lock_required = lock_required_env in {"1", "true", "yes", "on"}
    gate = validate_backend_fingerprint(
        fp,
        backend=str(backend),
        require_cuda=bool(require_cuda),
        require_mlx=bool(require_mlx),
        lock_path=lock_path,
        lock_required=bool(lock_required),
    )
    if str(gate.get("status") or "") != "PASS":
        err = None
        if isinstance(gate.get("errors"), list) and len(gate["errors"]) > 0:
            err = str(gate["errors"][0])
        if not err:
            err = "STRICT GATE FAIL: backend fingerprint gate failed"
        raise RuntimeError(err)
    return {
        "status": "PASS",
        "lock_path": lock_path,
        "suggest_lock_path": str(suggest_lock_path),
        "fingerprint": fp,
    }
