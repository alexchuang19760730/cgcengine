from __future__ import annotations

from dataclasses import dataclass

from .config import RuntimeConfig


@dataclass
class RuntimeContext:
    runtime: RuntimeConfig


def build_runtime_context(cfg: RuntimeConfig) -> RuntimeContext:
    return RuntimeContext(runtime=cfg)
