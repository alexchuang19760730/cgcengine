import importlib as _importlib
import sys as _sys

_mod = _importlib.import_module("cgc_engine._legacy.piecewise_graph")
_sys.modules[__name__] = _mod

