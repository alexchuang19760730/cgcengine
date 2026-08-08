#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml


MATRIX_ONLY_CHECK_IDS = {
    "build_release_matrix_contract",
    "build_dist_manifest_contract",
    "windows_artifact_size_budget_contract",
    "macos_artifact_size_budget_contract",
    "linux_artifact_size_budget_contract",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an M8.4-only gate config for CI matrix aggregation")
    parser.add_argument("--base-config", required=True, help="Path to the base m8_gate.yaml")
    parser.add_argument("--output-config", required=True, help="Path to write the rendered matrix-only config")
    parser.add_argument("--matrix-dir", required=True, help="Directory containing windows.json / macos.json / linux.json")
    parser.add_argument("--dist-dir", required=True, help="Directory containing CGC_Release/dist and build_matrix_manifest.json")
    args = parser.parse_args()

    base_config = Path(args.base_config).expanduser().resolve()
    output_config = Path(args.output_config).expanduser().resolve()
    matrix_dir = str(Path(args.matrix_dir).expanduser().resolve())
    dist_dir = str(Path(args.dist_dir).expanduser().resolve())

    config = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise SystemExit("base config is not a mapping")

    sections = list(config.get("sections") or [])
    selected_section: Dict[str, Any] = {}
    for section in sections:
        if isinstance(section, dict) and str(section.get("name") or "") == "m84_release_build":
            selected_section = dict(section)
            break
    if not selected_section:
        raise SystemExit("m84_release_build not found in base config")

    filtered_checks: List[Dict[str, Any]] = []
    for raw_check in list(selected_section.get("checks") or []):
        if not isinstance(raw_check, dict):
            continue
        check_id = str(raw_check.get("id") or "")
        if check_id not in MATRIX_ONLY_CHECK_IDS:
            continue
        check = dict(raw_check)
        check["matrix_dir"] = matrix_dir
        if check_id == "build_dist_manifest_contract":
            check["dist_dir"] = dist_dir
        filtered_checks.append(check)
    selected_section["checks"] = filtered_checks

    acceptance_contract = config.get("acceptance_contract") if isinstance(config.get("acceptance_contract"), dict) else {}
    rendered = {
        "name": str(config.get("name") or "CGC_M8_Productization_Gate"),
        "description": "Focused M8.4 matrix aggregation gate",
        "version": str(config.get("version") or "1.0"),
        "acceptance_contract": {
            "m84_cgc_build_release_acceptance": acceptance_contract.get("m84_cgc_build_release_acceptance", {}),
        },
        "sections": [selected_section],
        "output": config.get("output") or {
            "report_file": "report.json",
            "summary_file": "summary.json",
            "pass_fail_strategy": "all_sections_must_pass",
        },
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(rendered, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(str(output_config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
