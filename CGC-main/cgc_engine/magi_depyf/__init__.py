import importlib as _importlib
import sys as _sys

_mod = _importlib.import_module("cgc_engine._legacy.magi_depyf")
_sys.modules[__name__] = _mod

