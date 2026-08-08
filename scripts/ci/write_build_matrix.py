#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge per-platform CGC M8.4 build reports into build_matrix.json")
    parser.add_argument("--input-dir", required=True, help="Directory containing windows.json / macos.json / linux.json")
    parser.add_argument("--output-file", default="", help="Optional explicit output path for build_matrix.json")
    parser.add_argument(
        "--required-platform",
        dest="required_platforms",
        action="append",
        default=[],
        help="Required platform name; repeatable. Defaults to windows/macos/linux.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    required_platforms = [str(x).strip() for x in (args.required_platforms or []) if str(x).strip()] or [
        "windows",
        "macos",
        "linux",
    ]
    output_file = (
        Path(args.output_file).expanduser().resolve()
        if str(args.output_file or "").strip()
        else (input_dir / "build_matrix.json").resolve()
    )

    platform_reports: Dict[str, str] = {}
    missing_platforms = []
    invalid_platforms = []
    report_summaries: Dict[str, Dict[str, Any]] = {}

    for platform_name in required_platforms:
        report_path = (input_dir / f"{platform_name}.json").resolve()
        payload = read_json(report_path) if report_path.exists() else {}
        if not report_path.exists():
            missing_platforms.append(platform_name)
        elif str(payload.get("status") or "") != "PASS":
            invalid_platforms.append(platform_name)
        else:
            platform_reports[platform_name] = str(report_path)
        report_summaries[platform_name] = {
            "report_path": str(report_path),
            "exists": report_path.exists(),
            "status": str(payload.get("status") or ""),
            "platform": str(payload.get("platform") or ""),
            "package_format": str(payload.get("package_format") or ""),
            "size_bytes": int(payload.get("size_bytes") or 0),
            "executable_size_bytes": int(payload.get("executable_size_bytes") or 0),
        }

    matrix_payload = {
        "status": "PASS" if not missing_platforms and not invalid_platforms else "FAIL",
        "generated_at": utc_now(),
        "input_dir": str(input_dir),
        "required_platforms": required_platforms,
        "platform_reports": platform_reports,
        "missing_platforms": missing_platforms,
        "invalid_platforms": invalid_platforms,
        "report_summaries": report_summaries,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(matrix_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
