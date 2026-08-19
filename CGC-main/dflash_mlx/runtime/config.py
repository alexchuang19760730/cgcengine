from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    prefix_cache: bool = True
    prefix_cache_max_entries: int = 4
    prefix_cache_max_bytes: int = 8 * 1024**3
    prefix_cache_l2: bool = False
    prefix_cache_l2_dir: str = ""
    prefix_cache_l2_max_bytes: int = 0
    draft_window_size: int = 1024
    draft_sink_size: int = 64
    verify_mode: str = "adaptive"


def runtime_config_from_defaults(**kwargs: object) -> RuntimeConfig:
    cfg = RuntimeConfig()
    for key, value in kwargs.items():
        if value is None:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
