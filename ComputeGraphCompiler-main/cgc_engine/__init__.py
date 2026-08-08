from importlib import import_module
from typing import Any

__all__ = ["CGCEngine", "CGCEngineConfig", "compile", "run_cgc_with_kda"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".core_engine", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
