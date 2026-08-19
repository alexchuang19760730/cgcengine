"""
CGC C++ Backend Module

This module provides the Python interface to the compiled C++ backend.
The actual .so files are in the build/ directory.
"""

import os
import sys

_BUILD_DIR = os.path.join(os.path.dirname(__file__), "build")

if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

try:
    import cgc_cpp as cgc_cpp_module
    cgc_cpp = cgc_cpp_module
except ImportError as e:
    import warnings
    warnings.warn(f"CGC C++ backend not available: {e}")
    cgc_cpp = None

__all__ = ["cgc_cpp"]