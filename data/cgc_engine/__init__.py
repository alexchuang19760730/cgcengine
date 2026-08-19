"""cgc-engine — CGC 端雲協同引擎."""

try:
    from .api import magi_compile, magi_register_custom_op
except Exception:
    # 无 torch / 后端不可用（如仅做 GGUF 静态分析）时降级为空 API，
    # 保证 magicompiler_integration 等模块在 torch 缺失环境下仍可导入。
    magi_compile = None
    magi_register_custom_op = None

try:
    from ._version import __version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["magi_compile", "magi_register_custom_op", "__version__"]
