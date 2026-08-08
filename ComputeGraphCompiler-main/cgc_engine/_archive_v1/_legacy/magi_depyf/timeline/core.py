# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_timeline_dir: Path | None = None
_timeline_jsonl_path: Path | None = None
_event_index: int = 0
_start_time: float | None = None


def start_timeline(timeline_dir: str | Path) -> None:
    global _timeline_dir, _timeline_jsonl_path, _event_index, _start_time
    timeline_path = Path(timeline_dir)
    timeline_path.mkdir(parents=True, exist_ok=True)
    _timeline_dir = timeline_path
    _timeline_jsonl_path = timeline_path / "timeline.jsonl"
    _event_index = 0
    _start_time = None


def end_timeline() -> None:
    global _timeline_dir, _timeline_jsonl_path, _event_index, _start_time
    _timeline_dir = None
    _timeline_jsonl_path = None
    _event_index = 0
    _start_time = None


def _format_relative_time(delta_seconds: float) -> str:
    if delta_seconds < 1:
        return f"{delta_seconds * 1000:.1f}ms"
    if delta_seconds < 60:
        return f"{delta_seconds:.2f}s"
    mins = int(delta_seconds // 60)
    secs = int(delta_seconds % 60)
    return f"{mins}m{secs}s"


def emit_event(
    name: str,
    message: str | None = None,
    files: dict[str, str] | Callable[[], dict[str, str]] | None = None,
    attributes: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
) -> None:
    global _event_index, _start_time
    if _timeline_dir is None or _timeline_jsonl_path is None:
        return

    now = time.time()
    if _start_time is None:
        _start_time = now

    resolved_files = files() if callable(files) else (files or {})
    resolved_attributes = attributes() if callable(attributes) else (attributes or {})
    if message is not None:
        resolved_attributes = {**resolved_attributes, "message": message}

    event_idx = _event_index
    dt_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    rel_str = _format_relative_time(now - _start_time)

    record = {
        "index": event_idx,
        "name": name,
        "relative_time": rel_str,
        "files": list(resolved_files.keys()),
        "attributes": resolved_attributes,
        "datetime": dt_str,
        "timestamp": now,
    }

    _timeline_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with _timeline_jsonl_path.open("a", encoding="utf-8") as f_jsonl:
        f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    _write_event_files(event_idx, name, resolved_files, resolved_attributes)
    _event_index += 1


def _write_event_files(event_idx: int, event_name: str, files: dict[str, str], attributes: dict[str, Any]) -> None:
    if _timeline_dir is None or not files:
        return

    event_dir = _timeline_dir / "files" / f"{event_idx:04d}_{event_name}"
    sub = attributes.get("subgraph")
    if sub:
        event_dir = event_dir / str(sub)
    event_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in files.items():
        path = event_dir / filename
        path.write_text(content, encoding="utf-8")

    attr_path = event_dir / "attributes.json"
    attr_path.write_text(json.dumps(attributes, indent=2, ensure_ascii=False), encoding="utf-8")
