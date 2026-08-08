from __future__ import annotations

import os
import sys
from pathlib import Path


def engine_root_for(anchor: str) -> Path:
    return Path(anchor).resolve().parents[2]


def app_root_for(anchor: str) -> Path:
    engine_root = engine_root_for(anchor)
    workspace_root = engine_root.parent
    candidates = [
        str(os.environ.get("CGC_APP_ROOT", "") or "").strip(),
        str((workspace_root / "flashkv0516" / "app").resolve()),
        str((workspace_root / "app").resolve()),
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path.resolve()
    return (workspace_root / "flashkv0516" / "app").resolve()


def extend_pythonpath_for(anchor: str) -> None:
    engine_root = engine_root_for(anchor)
    app_root = app_root_for(anchor)
    for candidate in (app_root.parent, engine_root, engine_root.parent):
        raw = str(candidate.resolve())
        if raw not in sys.path:
            sys.path.insert(0, raw)
