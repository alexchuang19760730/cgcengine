#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path


_PROGRESS_RE = re.compile(
    r"Downloading \[(?P<name>[^\]]+)\]:\s+"
    r"(?P<pct>\d+)%\|.*?\|\s*"
    r"(?P<done>[\d.]+)(?P<done_unit>[KMG])/"
    r"(?P<total>[\d.]+)(?P<total_unit>[KMG])\s*"
    r"\[(?P<elapsed>[0-9:]+)<(?P<eta>[0-9:]+),\s*"
    r"(?P<speed>[\d.]+)(?P<speed_unit>[KMG]?B/s)\]"
)


def _to_bytes(value: float, unit: str) -> float:
    scale = {"K": 1024.0, "M": 1024.0**2, "G": 1024.0**3}
    return value * scale[unit]


def _format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or math.isinf(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _parse_hms(text: str) -> int:
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 1:
        return parts[0]
    raise ValueError(f"bad duration: {text}")


def _read_last_progress(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        match = _PROGRESS_RE.search(line)
        if not match:
            continue
        done_bytes = _to_bytes(float(match.group("done")), match.group("done_unit"))
        total_bytes = _to_bytes(float(match.group("total")), match.group("total_unit"))
        speed_unit = match.group("speed_unit")
        speed_factor = 1.0
        if speed_unit.startswith("KB"):
            speed_factor = 1024.0
        elif speed_unit.startswith("MB"):
            speed_factor = 1024.0**2
        elif speed_unit.startswith("GB"):
            speed_factor = 1024.0**3
        speed_bps = float(match.group("speed")) * speed_factor
        elapsed_sec = _parse_hms(match.group("elapsed"))
        eta_sec = _parse_hms(match.group("eta"))
        return {
            "name": match.group("name"),
            "progress_pct": int(match.group("pct")),
            "downloaded_bytes": int(done_bytes),
            "total_bytes": int(total_bytes),
            "speed_bps": speed_bps,
            "elapsed_sec": elapsed_sec,
            "eta_sec": eta_sec,
            "source": "log",
        }
    return None


def _estimate_from_file(file_path: Path, total_bytes: int | None) -> dict | None:
    if not file_path.exists() or total_bytes is None or total_bytes <= 0:
        return None
    stat = file_path.stat()
    downloaded_bytes = stat.st_size
    elapsed_sec = max(time.time() - stat.st_mtime, 1.0)
    speed_bps = downloaded_bytes / elapsed_sec if downloaded_bytes > 0 else 0.0
    eta_sec = (total_bytes - downloaded_bytes) / speed_bps if speed_bps > 0 else math.inf
    return {
        "name": file_path.name,
        "progress_pct": int((downloaded_bytes / total_bytes) * 100),
        "downloaded_bytes": downloaded_bytes,
        "total_bytes": total_bytes,
        "speed_bps": speed_bps,
        "elapsed_sec": int(elapsed_sec),
        "eta_sec": None if math.isinf(eta_sec) else int(eta_sec),
        "source": "file",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a large model download and estimate ETA.")
    parser.add_argument("--log", help="Progress log file path.")
    parser.add_argument("--file", help="Partial or final download file path.")
    parser.add_argument("--expected-bytes", type=int, help="Expected total size in bytes.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    log_snapshot = _read_last_progress(Path(args.log)) if args.log else None
    total_bytes = args.expected_bytes or (log_snapshot or {}).get("total_bytes")
    file_snapshot = _estimate_from_file(Path(args.file), total_bytes) if args.file else None
    snapshot = log_snapshot or file_snapshot

    if snapshot is None:
        raise SystemExit("no progress data available")

    if file_snapshot and log_snapshot:
        snapshot["local_file_bytes"] = file_snapshot["downloaded_bytes"]

    remaining_bytes = max(int(snapshot["total_bytes"]) - int(snapshot["downloaded_bytes"]), 0)
    snapshot["remaining_bytes"] = remaining_bytes
    snapshot["downloaded_human"] = _format_bytes(snapshot["downloaded_bytes"])
    snapshot["total_human"] = _format_bytes(snapshot["total_bytes"])
    snapshot["remaining_human"] = _format_bytes(remaining_bytes)
    snapshot["speed_human"] = f"{_format_bytes(snapshot['speed_bps'])}/s" if snapshot["speed_bps"] else "0B/s"
    snapshot["elapsed_human"] = _format_seconds(snapshot.get("elapsed_sec"))
    snapshot["eta_human"] = _format_seconds(snapshot.get("eta_sec"))

    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print(f"name: {snapshot['name']}")
    print(f"source: {snapshot['source']}")
    print(
        f"progress: {snapshot['progress_pct']}% "
        f"({snapshot['downloaded_human']} / {snapshot['total_human']})"
    )
    print(f"remaining: {snapshot['remaining_human']}")
    print(f"speed: {snapshot['speed_human']}")
    print(f"elapsed: {snapshot['elapsed_human']}")
    print(f"eta: {snapshot['eta_human']}")
    if "local_file_bytes" in snapshot:
        print(f"local_file: {_format_bytes(snapshot['local_file_bytes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
