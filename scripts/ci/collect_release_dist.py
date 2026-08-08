#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_artifact(source: Path, target: Path) -> None:
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        # Preserve macOS app bundle signatures and metadata during dist collection.
        if source.suffix == ".app":
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ditto", str(source), str(target)], check=True)
            return
        shutil.copytree(source, target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def find_artifact_source(build_artifacts_dir: Path, artifact_name: str, platform_name: str) -> Path | None:
    candidates = sorted(build_artifacts_dir.rglob(artifact_name))
    if not candidates:
        return None
    preferred = [path for path in candidates if platform_name in str(path).lower()]
    return (preferred or candidates)[0].resolve()


def path_size_bytes(target: Path) -> int:
    if not target.exists():
        return 0
    if target.is_file():
        return int(target.stat().st_size)
    total = 0
    for path in target.rglob("*"):
        if path.is_file():
            total += int(path.stat().st_size)
    return int(total)


def create_release_archive(*, platform_name: str, source: Path, release_assets_dir: Path) -> Path:
    release_assets_dir.mkdir(parents=True, exist_ok=True)
    if platform_name == "linux":
        archive_path = (release_assets_dir / "cgc-linux.tar.gz").resolve()
        with tarfile.open(archive_path, "w:gz") as handle:
            handle.add(source, arcname=source.name)
        return archive_path
    archive_path = (release_assets_dir / f"cgc-{platform_name}.zip").resolve()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_dir():
                    continue
                handle.write(path, arcname=str(path.relative_to(source.parent)))
        else:
            handle.write(source, arcname=source.name)
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect per-platform CGC M8.4 build artifacts into CGC_Release/dist")
    parser.add_argument("--matrix-dir", required=True, help="Directory containing build_matrix.json and per-platform reports")
    parser.add_argument("--build-artifacts-dir", required=True, help="Directory containing downloaded/stashed build outputs")
    parser.add_argument("--dist-dir", required=True, help="Destination dist root, typically CGC_Release/dist")
    parser.add_argument("--manifest-file", default="", help="Optional explicit output path for build_matrix_manifest.json")
    parser.add_argument("--release-assets-dir", default="", help="Optional output directory for release asset archives")
    parser.add_argument(
        "--required-platform",
        dest="required_platforms",
        action="append",
        default=[],
        help="Required platform name; repeatable. Defaults to windows/macos/linux.",
    )
    args = parser.parse_args()

    matrix_dir = Path(args.matrix_dir).expanduser().resolve()
    build_artifacts_dir = Path(args.build_artifacts_dir).expanduser().resolve()
    dist_dir = Path(args.dist_dir).expanduser().resolve()
    required_platforms = [str(x).strip() for x in (args.required_platforms or []) if str(x).strip()] or [
        "windows",
        "macos",
        "linux",
    ]
    manifest_file = (
        Path(args.manifest_file).expanduser().resolve()
        if str(args.manifest_file or "").strip()
        else (dist_dir / "build_matrix_manifest.json").resolve()
    )
    release_assets_dir = (
        Path(args.release_assets_dir).expanduser().resolve()
        if str(args.release_assets_dir or "").strip()
        else (dist_dir / "release_assets").resolve()
    )

    matrix_payload = read_json((matrix_dir / "build_matrix.json").resolve())
    ensure_clean_dir(dist_dir)
    release_assets_dir.mkdir(parents=True, exist_ok=True)

    missing_platforms: List[str] = []
    invalid_platforms: List[str] = []
    platforms: Dict[str, Dict[str, Any]] = {}

    for platform_name in required_platforms:
        report_path = (matrix_dir / f"{platform_name}.json").resolve()
        report_payload = read_json(report_path) if report_path.exists() else {}
        artifact_source: Path | None = None
        dist_artifact_path = (dist_dir / platform_name).resolve()
        archive_path = None
        if not report_path.exists():
            missing_platforms.append(platform_name)
        elif str(report_payload.get("status") or "") != "PASS":
            invalid_platforms.append(platform_name)
        else:
            artifact_name = Path(str(report_payload.get("output_path") or "")).name
            artifact_source = find_artifact_source(build_artifacts_dir, artifact_name, platform_name)
            if artifact_source is None:
                missing_platforms.append(platform_name)
            else:
                if artifact_source.is_dir():
                    dist_artifact_path = (dist_dir / platform_name / artifact_source.name).resolve()
                else:
                    dist_artifact_path = (dist_dir / platform_name / artifact_source.name).resolve()
                dist_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                copy_artifact(artifact_source, dist_artifact_path)
                archive_path = create_release_archive(
                    platform_name=platform_name,
                    source=dist_artifact_path,
                    release_assets_dir=release_assets_dir,
                )
        platforms[platform_name] = {
            "status": str(report_payload.get("status") or "FAIL"),
            "report_path": str(report_path),
            "report_exists": report_path.exists(),
            "source_artifact_path": str(artifact_source) if artifact_source is not None else "",
            "dist_artifact_path": str(dist_artifact_path) if dist_artifact_path.exists() else "",
            "dist_artifact_exists": dist_artifact_path.exists(),
            "release_asset_path": str(archive_path) if archive_path is not None and archive_path.exists() else "",
            "release_asset_exists": bool(archive_path is not None and archive_path.exists()),
            "package_format": str(report_payload.get("package_format") or ""),
            "size_bytes": int(report_payload.get("size_bytes") or 0),
            "executable_size_bytes": int(report_payload.get("executable_size_bytes") or 0),
            "artifact_sha256": str(report_payload.get("artifact_sha256") or ""),
            "executable_sha256": str(report_payload.get("executable_sha256") or ""),
            "release_asset_size_bytes": path_size_bytes(archive_path) if archive_path is not None else 0,
        }

    manifest_payload = {
        "status": "PASS" if not missing_platforms and not invalid_platforms else "FAIL",
        "generated_at": utc_now(),
        "matrix_dir": str(matrix_dir),
        "matrix_file": str((matrix_dir / "build_matrix.json").resolve()),
        "matrix_status": str(matrix_payload.get("status") or ""),
        "build_artifacts_dir": str(build_artifacts_dir),
        "dist_dir": str(dist_dir),
        "release_assets_dir": str(release_assets_dir),
        "required_platforms": required_platforms,
        "missing_platforms": missing_platforms,
        "invalid_platforms": invalid_platforms,
        "platforms": platforms,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(manifest_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
