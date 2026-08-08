from __future__ import annotations

import os
import platform
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EdgeBuildResult:
    status: str
    command: List[str]
    generated_at: str
    output_path: str
    executable_path: str
    python_bin: str
    builder: str
    platform: str
    host_platform: str
    host_arch: str
    package_format: str
    size_bytes: int
    executable_size_bytes: int
    artifact_sha256: str
    executable_sha256: str
    output_exists: bool
    supported_platforms: List[str]


def _candidate_python_bins(repo_root: Path) -> List[Path]:
    candidates: List[Path] = []
    env_python = str(os.environ.get("CGC_BUILD_PYTHON") or "").strip()
    if env_python:
        candidates.append(Path(env_python).expanduser())

    repo_candidates = [
        repo_root / ".venv-cgc" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python",
        repo_root / ".venv-cgc" / "Scripts" / "python.exe",
        repo_root / ".venv" / "Scripts" / "python.exe",
    ]
    candidates.extend(repo_candidates)
    candidates.append(Path(sys.executable))

    seen = set()
    unique: List[Path] = []
    for candidate in candidates:
        raw = str(candidate.expanduser())
        if raw in seen:
            continue
        seen.add(raw)
        unique.append(Path(raw))
    return unique


def _python_has_module(python_bin: Path, module_name: str) -> bool:
    if not python_bin.exists():
        return False
    cmd = [
        str(python_bin),
        "-c",
        (
            "import importlib.util, sys; "
            f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def resolve_nuitka_python(repo_root: Path) -> Path:
    checked: List[str] = []
    for python_bin in _candidate_python_bins(repo_root):
        checked.append(str(python_bin))
        if _python_has_module(python_bin, "nuitka"):
            return python_bin
    checked_str = ", ".join(checked) if checked else "<none>"
    raise RuntimeError(
        "Nuitka is not bundled in the managed CGC Edge Engine Python environments. "
        f"Checked: {checked_str}. Preinstall Nuitka in the packaged environment instead of "
        "installing it dynamically at runtime."
    )


def _python_module_path(python_bin: Path, module_name: str) -> Path | None:
    if not python_bin.exists():
        return None
    cmd = [
        str(python_bin),
        "-c",
        (
            "import importlib.util, pathlib; "
            f"spec = importlib.util.find_spec('{module_name}'); "
            "path = '';\n"
            "if spec is not None:\n"
            "    if spec.origin:\n"
            "        path = str(pathlib.Path(spec.origin).resolve())\n"
            "    elif spec.submodule_search_locations:\n"
            "        path = str(pathlib.Path(list(spec.submodule_search_locations)[0]).resolve())\n"
            "print(path)"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    raw = str(result.stdout or "").strip()
    return Path(raw) if raw else None


def _output_name_for_platform(platform_name: str) -> str:
    if platform_name == "win32":
        return "cgc.exe"
    if platform_name == "darwin":
        return "cgc.app"
    return "cgc"


def _normalized_platform(platform_name: str) -> str:
    if platform_name == "win32":
        return "windows"
    if platform_name == "darwin":
        return "macos"
    return "linux"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_format_for_platform(platform_name: str) -> str:
    if platform_name == "win32":
        return "exe"
    if platform_name == "darwin":
        return "app_bundle"
    return "elf"


def _path_size_bytes(target: Path) -> int:
    resolved = target.expanduser().resolve()
    if not resolved.exists():
        return 0
    if resolved.is_file():
        return int(resolved.stat().st_size)
    total = 0
    for path in resolved.rglob("*"):
        if path.is_file():
            total += int(path.stat().st_size)
    return int(total)


def _hash_file(target: Path) -> str:
    hasher = hashlib.sha256()
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _path_sha256(target: Path) -> str:
    resolved = target.expanduser().resolve()
    if not resolved.exists():
        return ""
    if resolved.is_file():
        return _hash_file(resolved)
    hasher = hashlib.sha256()
    for path in sorted((p for p in resolved.rglob("*") if p.is_file()), key=lambda p: str(p.relative_to(resolved))):
        rel = str(path.relative_to(resolved)).replace("\\", "/")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(path.stat().st_size).encode("ascii"))
        hasher.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    return hasher.hexdigest()


def _build_disk_floor_bytes() -> int:
    # Repeat Darwin rebuilds stay below the initial peak after ccache warms up,
    # so keep some safety margin without blocking every iterative local rebuild.
    return int(3.5 * 1024**3) if sys.platform == "darwin" else 3 * 1024**3


def _embed_mlx_runtime_enabled() -> bool:
    return str(os.environ.get("CGC_BUILD_EMBED_MLX_RUNTIME") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ensure_build_disk_headroom(output_dir: Path) -> None:
    usage = shutil.disk_usage(output_dir)
    required = _build_disk_floor_bytes()
    if usage.free >= required:
        return
    raise RuntimeError(
        "Insufficient free disk space for CGC Edge Engine build. "
        f"Free={usage.free / 1024**3:.2f} GiB, required>={required / 1024**3:.2f} GiB. "
        "Clean old build artifacts before retrying."
    )


def _build_command(*, python_bin: Path, output_dir: Path, output_name: str) -> List[str]:
    embed_mlx_runtime = _embed_mlx_runtime_enabled()
    mlx_module_file = _python_module_path(python_bin, "mlx") if embed_mlx_runtime else None
    if mlx_module_file is None:
        mlx_metallib = None
    elif mlx_module_file.is_dir():
        mlx_metallib = mlx_module_file / "lib" / "mlx.metallib"
    else:
        mlx_metallib = mlx_module_file.parent / "lib" / "mlx.metallib"
    base = [
        str(python_bin),
        "-m",
        "nuitka",
        "--remove-output",
        f"--output-dir={output_dir}",
        f"--output-filename={output_name}",
        "--include-module=fastapi",
        "--include-module=uvicorn",
        "--include-module=app.servers.cgc_api_server",
        "--include-module=app.servers.internal_proxy_server",
        "--nofollow-import-to=torch",
        "--nofollow-import-to=mlx",
        "--nofollow-import-to=mlx_lm",
        "--nofollow-import-to=mlx_vlm",
        "--nofollow-import-to=mlx_audio",
        "--nofollow-import-to=mlx_lm.tokenizer_utils",
        "--nofollow-import-to=modelscope",
        "--nofollow-import-to=torchvision",
        "--nofollow-import-to=torchaudio",
        "--nofollow-import-to=sklearn",
        "--nofollow-import-to=accelerate",
        "--nofollow-import-to=cv2",
        "--nofollow-import-to=onnx",
        "--nofollow-import-to=onnxruntime",
        "--nofollow-import-to=pycountry",
        "--nofollow-import-to=openai_harmony",
        "--nofollow-import-to=transformers.models",
        "--nofollow-import-to=transformers.modeling_gguf_pytorch_utils",
        "--nofollow-import-to=transformers.integrations.ggml",
        "--nofollow-import-to=omlx.tests",
        "--nofollow-import-to=omlx.eval",
        "--nofollow-import-to=openai",
        "--nofollow-import-to=sqlalchemy",
        "--nofollow-import-to=scipy",
        "--nofollow-import-to=pandas",
        "--nofollow-import-to=datasets",
        "--nofollow-import-to=opentelemetry",
        "--nofollow-import-to=vllm",
        "--nofollow-import-to=matplotlib",
        "--nofollow-import-to=PIL",
        "--assume-yes-for-downloads",
        "app/cli/cgc_edge.py",
    ]
    if embed_mlx_runtime:
        base.extend(
            [
                "--include-module=mlx._reprlib_fix",
                "--include-module=mlx_lm.models.qwen2",
                "--include-package=mlx_lm.chat_templates",
                "--include-package=mlx_lm.tool_parsers",
                "--include-module=app.edge_engine.mlx_tokenizer_shim",
            ]
        )
        base = [arg for arg in base if arg not in {"--nofollow-import-to=mlx", "--nofollow-import-to=mlx_lm"}]
    if mlx_metallib is not None and mlx_metallib.exists():
        base.insert(-1, f"--include-data-files={mlx_metallib}=mlx/lib/mlx.metallib")
    if sys.platform == "darwin":
        return [
            *base[:3],
            "--mode=app",
            *base[3:],
        ]
    return [
        *base[:3],
        "--standalone",
        "--onefile",
        *base[3:],
    ]


def _resolve_built_artifact(*, output_dir: Path, output_name: str) -> Tuple[Path, Path]:
    direct_path = (output_dir / output_name).resolve()
    if sys.platform != "darwin":
        return direct_path, direct_path

    if direct_path.exists() and direct_path.suffix == ".app":
        executables = sorted(
            path for path in (direct_path / "Contents" / "MacOS").glob("*") if path.is_file()
        )
        executable = executables[0] if executables else direct_path
        return direct_path, executable.resolve()

    app_dirs = sorted(
        [path for path in output_dir.glob("*.app") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not app_dirs:
        return direct_path, direct_path

    bundle_path = app_dirs[0].resolve()
    executables = sorted(
        path for path in (bundle_path / "Contents" / "MacOS").glob("*") if path.is_file()
    )
    executable = executables[0] if executables else bundle_path
    return bundle_path, executable.resolve()


def _patch_mlx_bundle_runtime(bundle_path: Path) -> None:
    if sys.platform != "darwin" or bundle_path.suffix != ".app":
        return

    macos_dir = bundle_path / "Contents" / "MacOS"
    mlx_core = macos_dir / "mlx" / "core.so"
    libmlx = macos_dir / "libmlx.dylib"
    if not mlx_core.exists() or not libmlx.exists():
        return

    mlx_lib_dir = macos_dir / "mlx" / "lib"
    mlx_lib_dir.mkdir(parents=True, exist_ok=True)
    bundle_libmlx = mlx_lib_dir / "libmlx.dylib"
    shutil.copy2(libmlx, bundle_libmlx)

    subprocess.run(
        [
            "install_name_tool",
            "-change",
            "@executable_path/libmlx.dylib",
            "@loader_path/lib/libmlx.dylib",
            str(mlx_core),
        ],
        check=True,
    )
    # Keep a single bundled copy of libmlx to avoid shipping the same 21 MB dylib twice.
    if libmlx.exists():
        libmlx.unlink()
    for path in sorted(macos_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".so", ".dylib"} and path.name not in {"Python", "cgc.app"}:
            continue
        subprocess.run(["strip", "-x", str(path)], check=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(bundle_libmlx)], check=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(mlx_core)], check=True)
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(bundle_path)], check=True)


def build_edge_engine(*, repo_root: Path, output_dir: Path, quiet: bool = False) -> EdgeBuildResult:
    repo_root = repo_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_build_disk_headroom(output_dir)

    python_bin = resolve_nuitka_python(repo_root)
    output_name = _output_name_for_platform(sys.platform)
    command = _build_command(
        python_bin=python_bin,
        output_dir=output_dir,
        output_name=output_name,
    )
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        check=False,
        capture_output=quiet,
        text=quiet,
    )
    if completed.returncode != 0:
        error_detail = ""
        if quiet:
            stderr = str(completed.stderr or "").strip()
            stdout = str(completed.stdout or "").strip()
            if stderr:
                error_detail = stderr
            elif stdout:
                error_detail = stdout
        raise RuntimeError(
            "CGC Edge Engine build failed."
            + (f" details: {error_detail[:2000]}" if error_detail else "")
        )
    artifact_path, executable_path = _resolve_built_artifact(output_dir=output_dir, output_name=output_name)
    _patch_mlx_bundle_runtime(artifact_path)
    normalized_platform = _normalized_platform(sys.platform)
    package_format = _package_format_for_platform(sys.platform)
    artifact_exists = bool(artifact_path.exists())
    artifact_size_bytes = _path_size_bytes(artifact_path)
    executable_size_bytes = _path_size_bytes(executable_path)
    artifact_sha256 = _path_sha256(artifact_path)
    executable_sha256 = _path_sha256(executable_path)
    return EdgeBuildResult(
        status="PASS",
        command=command,
        generated_at=_utc_now(),
        output_path=str(artifact_path),
        executable_path=str(executable_path),
        python_bin=str(python_bin),
        builder="nuitka",
        platform=normalized_platform,
        host_platform=str(platform.system()),
        host_arch=str(platform.machine()),
        package_format=package_format,
        size_bytes=artifact_size_bytes,
        executable_size_bytes=executable_size_bytes,
        artifact_sha256=artifact_sha256,
        executable_sha256=executable_sha256,
        output_exists=artifact_exists,
        supported_platforms=["windows", "macos", "linux"],
    )
