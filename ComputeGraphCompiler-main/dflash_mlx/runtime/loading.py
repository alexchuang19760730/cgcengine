from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _ensure_omlx_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    omlx_root = repo_root / "Backend" / "oMLX"
    omlx_root_str = str(omlx_root)
    if omlx_root_str not in sys.path:
        sys.path.insert(0, omlx_root_str)


@dataclass
class TargetBundle:
    model: Any
    tokenizer: Any
    target_ops: Any
    meta: dict[str, Any]


@dataclass
class DraftBundle:
    model_name: str
    draft_quant: str | None = None


def _read_config(model_name: str) -> dict[str, Any]:
    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_target_bundle(model_name: str) -> TargetBundle:
    _ensure_omlx_path()
    from omlx.utils.model_loading import apply_post_load_transforms, load_text_model

    model, tokenizer = load_text_model(model_name)
    model = apply_post_load_transforms(model)
    meta = {"config": _read_config(model_name)}
    return TargetBundle(
        model=model,
        tokenizer=tokenizer,
        target_ops=None,
        meta=meta,
    )


def load_draft_bundle(
    draft_model_path: str,
    *,
    draft_quant: str | None = None,
) -> tuple[DraftBundle, dict[str, Any]]:
    bundle = DraftBundle(model_name=draft_model_path, draft_quant=draft_quant)
    meta = {"draft_model_path": draft_model_path, "draft_quant": draft_quant}
    return bundle, meta
