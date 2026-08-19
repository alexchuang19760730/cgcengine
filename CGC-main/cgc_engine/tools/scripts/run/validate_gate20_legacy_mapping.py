#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
GATE20_DIR = (
    REPO_ROOT
    / "docs"
    / "technical_whitepapers"
    / "CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation"
)
GATE20_MAP = (
    GATE20_DIR
    / "CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_gate_map.json"
)
GATE20_README = GATE20_DIR / "README.md"


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if not GATE20_MAP.is_file():
        return fail(f"gate_map not found: {GATE20_MAP}")
    if not GATE20_README.is_file():
        return fail(f"README not found: {GATE20_README}")

    gate_map = load_json(GATE20_MAP)
    capabilities = gate_map.get("capabilities", [])
    capability_ids = {cap.get("capability_id") for cap in capabilities}

    legacy_mapping = gate_map.get("legacy_capability_mapping")
    if not isinstance(legacy_mapping, dict):
        return fail("legacy_capability_mapping is missing or not an object")

    entries = legacy_mapping.get("entries")
    if not isinstance(entries, list) or not entries:
        return fail("legacy_capability_mapping.entries is missing or empty")

    valid_mapping_kinds = {
        "same_capability_id",
        "folded_into_broader_bucket",
        "split_across_multiple_buckets",
    }
    valid_statuses = {"done", "proof", "target", "stub", "mixed"}

    seen_legacy_ids = set()
    for index, entry in enumerate(entries, start=1):
        legacy_id = entry.get("legacy_capability_id")
        mapping_kind = entry.get("mapping_kind")
        current_buckets = entry.get("current_gate_map_bucket")
        status_after_audit = entry.get("status_after_audit")

        if not legacy_id or not isinstance(legacy_id, str):
            return fail(f"entry #{index} has invalid legacy_capability_id")
        if legacy_id in seen_legacy_ids:
            return fail(f"duplicate legacy_capability_id: {legacy_id}")
        seen_legacy_ids.add(legacy_id)

        if mapping_kind not in valid_mapping_kinds:
            return fail(f"{legacy_id} has invalid mapping_kind: {mapping_kind}")
        if status_after_audit not in valid_statuses:
            return fail(
                f"{legacy_id} has invalid status_after_audit: {status_after_audit}"
            )
        if not isinstance(current_buckets, list) or not current_buckets:
            return fail(f"{legacy_id} has empty current_gate_map_bucket")

        for bucket in current_buckets:
            if bucket not in capability_ids:
                return fail(f"{legacy_id} points to missing capability bucket: {bucket}")

        if mapping_kind == "same_capability_id" and legacy_id not in capability_ids:
            return fail(
                f"{legacy_id} is same_capability_id but missing from capabilities[]"
            )

    readme = GATE20_README.read_text(encoding="utf-8")
    if "## Legacy ID 映射" not in readme:
        return fail("README is missing '## Legacy ID 映射' section")

    missing_from_readme = [
        entry["legacy_capability_id"] for entry in entries if entry["legacy_capability_id"] not in readme
    ]
    if missing_from_readme:
        return fail(
            "README is missing legacy ids: " + ", ".join(sorted(missing_from_readme))
        )

    print("PASS")
    print(f"checked_entries={len(entries)}")
    print(f"capability_count={len(capabilities)}")
    print(f"readme_path={GATE20_README}")
    print(f"gate_map_path={GATE20_MAP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
