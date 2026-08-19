"""
vLLM Attention Backend for KDA / CGC Engine
"""
__version__ = "1.0.0"

try:
    from .cgc_kda_backend import CGCKDAImpl, CGCKDABackend, get_kda_backend
    from .kda_backend_stub import CGCKDAAttentionBackend
    __all__ = ["CGCKDAImpl", "CGCKDABackend", "CGCKDAAttentionBackend", "get_kda_backend"]
except ImportError as e:
    print(f"[CGC-KDA] Import error: {e}")
    __all__ = []
