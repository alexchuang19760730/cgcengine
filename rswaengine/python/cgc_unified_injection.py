# =============================================================================
# cgc_unified_injection.py
# -----------------------------------------------------------------------------
# rswaengine 的 Python 桥接层（ctypes）：把 C/C++ 引擎核心（libcgc_unified.so）
# 暴露给 Python，供 sglang / vllm monkey-patch adapter 消费统一 IR 与
# R-SWA + 窗口 OrthoKDA 的真实注意力计算。
#
# 设计要点（对应白皮书 §7 / §8）：
#   - inject_unified_ir_for_role() 调用 C 端同名函数，该函数会构建 cgc_strategy_t
#     并回调 cgc_inject_strategy（自动启用 kda_replace_mode），即「统一注入 <-> 策略」
#     的双向联动在 Python 侧同样可见。
#   - RSWAManager 封装 rswa_manager + window_ortho_kv_compressor 的 C API，
#     供 adapter 在框架注意力层里做实际计算（含 Reference 标准注意力 + 窗口正交压缩）。
#
# 仅依赖 ctypes + numpy；torch / sglang / vllm 为可选（adapter 侧使用）。
# =============================================================================
from __future__ import annotations

import ctypes
import os
import numpy as np

# -----------------------------------------------------------------------------
# 加载共享库
# -----------------------------------------------------------------------------
def _find_lib() -> str:
    cand = [
        os.environ.get("RSWAENGINE_LIB", ""),
        os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "libcgc_unified.so"),
        os.path.join(os.path.dirname(__file__), "..", "cpp", "build_cuda", "libcgc_unified_cuda.so"),
    ]
    for p in cand:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "libcgc_unified.so not found. Run `bash cpp/build_unified.sh` first.\n"
        f"Looked in: {[c for c in cand if c]}"
    )

_LIB = ctypes.CDLL(_find_lib())

# libc free（用于释放 C 端返回的 summary 字符串）
import sys as _sys
_LIBC = ctypes.CDLL("libc.dylib" if _sys.platform == "darwin" else "libc.so.6")
_LIBC.free.argtypes = [ctypes.c_void_p]

# -----------------------------------------------------------------------------
# 枚举（与 C 端保持一致）
# -----------------------------------------------------------------------------
CGC_INJECT_ROLE_PREFILL, CGC_INJECT_ROLE_DECODE, CGC_INJECT_ROLE_EDGE_RESUME = 0, 1, 2
CGC_BACKEND_ADAPTER_SGLANG, CGC_BACKEND_ADAPTER_VLLM, CGC_BACKEND_ADAPTER_CLOUD_SGLANG = 0, 1, 2
RSWA_LAYER_REFERENCE_STD, RSWA_LAYER_WINDOW_ORTHO, RSWA_LAYER_WINDOW_STD = 0, 1, 2

# -----------------------------------------------------------------------------
# ctypes 结构体（ABI 须与 C 端一致；见 rswa_manager.h / unified_ir.h）
# -----------------------------------------------------------------------------
class RswaConfig(ctypes.Structure):
    _fields_ = [
        ("num_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("reference_len", ctypes.c_int32),
        ("window_size", ctypes.c_int32),
        ("ortho_base_dim", ctypes.c_int32),
        ("decay_rate", ctypes.c_float),
        ("hybrid_every", ctypes.c_int32),
        ("enable_window_ortho", ctypes.c_int32),
    ]


class CgcUnifiedIrConfig(ctypes.Structure):
    _fields_ = [
        ("rswa", RswaConfig),
        ("role", ctypes.c_int32),
        ("adapter", ctypes.c_int32),
        ("enable_window_ortho", ctypes.c_uint8),
        ("enable_reference_std", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8 * 2),
        ("num_layers", ctypes.c_int32),
    ]


# -----------------------------------------------------------------------------
# C 函数签名绑定
# -----------------------------------------------------------------------------
_LIB.cgc_init.restype = ctypes.c_int
_LIB.cgc_get_kda_replace_mode.restype = ctypes.c_int

_LIB.rswa_create.restype = ctypes.c_void_p
_LIB.rswa_create.argtypes = [ctypes.POINTER(RswaConfig)]
_LIB.rswa_destroy.argtypes = [ctypes.c_void_p]
_LIB.rswa_set_reference.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_LIB.rswa_feed_token.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_LIB.rswa_feed_token.restype = ctypes.c_int
_LIB.rswa_visible_count.argtypes = [ctypes.c_void_p]
_LIB.rswa_visible_count.restype = ctypes.c_int32
_LIB.rswa_visibility_mask.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8)]
_LIB.rswa_layer_role.argtypes = [ctypes.c_void_p, ctypes.c_int32]
_LIB.rswa_layer_role.restype = ctypes.c_int
_LIB.rswa_window_compressor.argtypes = [ctypes.c_void_p]
_LIB.rswa_window_compressor.restype = ctypes.c_void_p
_LIB.rswa_reference_attention.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_LIB.rswa_window_attention.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_LIB.rswa_combined_attention.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

_LIB.wokdc_current_dim.argtypes = [ctypes.c_void_p]
_LIB.wokdc_current_dim.restype = ctypes.c_int
_LIB.wokdc_get_state.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
]
_LIB.wokdc_set_state.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
_LIB.wokdc_create.restype = ctypes.c_void_p
_LIB.wokdc_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
_LIB.wokdc_set_window_capacity.argtypes = [ctypes.c_void_p, ctypes.c_int]

_LIB.cgc_inject_unified_ir_for_role.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(CgcUnifiedIrConfig), ctypes.c_int,
]
_LIB.cgc_inject_unified_ir_for_role.restype = ctypes.c_int
_LIB.cgc_unified_ir_summary.argtypes = [ctypes.c_int]
_LIB.cgc_unified_ir_summary.restype = ctypes.c_void_p


# -----------------------------------------------------------------------------
# 统一注入入口（Python 友好封装）
# -----------------------------------------------------------------------------
def inject_unified_ir_for_role(
    role: int,
    adapter: int,
    *,
    num_layers: int,
    reference_len: int,
    window_size: int,
    ortho_base_dim: int,
    num_heads: int = 1,
    head_dim: int = 64,
    decay_rate: float = 0.01,
    hybrid_every: int = 0,
    enable_window_ortho: bool = True,
    model=None,
) -> int:
    """调用 C 端 cgc_inject_unified_ir_for_role；返回 cgc_error_t。
    该函数会构建 cgc_strategy_t 并回调 cgc_inject_strategy（联动 kda_replace_mode）。"""
    cfg = CgcUnifiedIrConfig()
    cfg.rswa.num_heads = num_heads
    cfg.rswa.head_dim = head_dim
    cfg.rswa.reference_len = reference_len
    cfg.rswa.window_size = window_size
    cfg.rswa.ortho_base_dim = ortho_base_dim
    cfg.rswa.decay_rate = decay_rate
    cfg.rswa.hybrid_every = hybrid_every
    cfg.rswa.enable_window_ortho = 1 if enable_window_ortho else 0
    cfg.role = role
    cfg.adapter = adapter
    cfg.enable_window_ortho = 1 if enable_window_ortho else 0
    cfg.enable_reference_std = 1  # 硬约束：Reference 恒标准注意力
    cfg.num_layers = num_layers
    model_ptr = ctypes.c_void_p(0) if model is None else ctypes.c_void_p(id(model))
    return _LIB.cgc_inject_unified_ir_for_role(role, model_ptr, ctypes.byref(cfg), adapter)


def get_kda_replace_mode() -> bool:
    return _LIB.cgc_get_kda_replace_mode() != 0


def unified_ir_summary(role: int) -> str:
    p = _LIB.cgc_unified_ir_summary(role)
    if not p:
        return ""
    s = ctypes.cast(p, ctypes.c_char_p).value.decode("utf-8")
    _LIBC.free(p)
    return s


# -----------------------------------------------------------------------------
# RSWAManager —— adapter 实际注意力计算句柄
# -----------------------------------------------------------------------------
class RSWAManager:
    """封装 C 端 rswa_manager_t + 窗口 OrthoKDA 压缩器，供 monkey-patch adapter 调用。"""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        reference_len: int,
        window_size: int,
        ortho_base_dim: int,
        decay_rate: float = 0.01,
        hybrid_every: int = 0,
        enable_window_ortho: bool = True,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ortho_base_dim = ortho_base_dim
        cfg = RswaConfig()
        cfg.num_heads = num_heads
        cfg.head_dim = head_dim
        cfg.reference_len = reference_len
        cfg.window_size = window_size
        cfg.ortho_base_dim = ortho_base_dim
        cfg.decay_rate = decay_rate
        cfg.hybrid_every = hybrid_every
        cfg.enable_window_ortho = 1 if enable_window_ortho else 0
        self._ptr = _LIB.rswa_create(ctypes.byref(cfg))
        if not self._ptr:
            raise RuntimeError("rswa_create failed")

    def set_reference(self, k: np.ndarray, v: np.ndarray):
        k = np.ascontiguousarray(k, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        _LIB.rswa_set_reference(self._ptr, k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def feed_token(self, k: np.ndarray, v: np.ndarray) -> int:
        k = np.ascontiguousarray(k, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        return _LIB.rswa_feed_token(self._ptr, k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                    v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def visible_count(self) -> int:
        return _LIB.rswa_visible_count(self._ptr)

    def visibility_mask(self, seq_len: int) -> np.ndarray:
        mask = np.zeros(seq_len, dtype=np.uint8)
        _LIB.rswa_visibility_mask(self._ptr, ctypes.c_int32(seq_len),
                                  mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return mask

    def layer_role(self, layer_idx: int) -> int:
        return _LIB.rswa_layer_role(self._ptr, ctypes.c_int32(layer_idx))

    def combined_attention(self, Q: np.ndarray) -> np.ndarray:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        out = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
        _LIB.rswa_combined_attention(self._ptr, Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out

    def window_compressor(self):
        return _LIB.rswa_window_compressor(self._ptr)

    def get_state(self):
        wok = self.window_compressor()
        if not wok:
            return None
        K = np.zeros((self.num_heads, self.ortho_base_dim, self.head_dim), dtype=np.float32)
        V = np.zeros_like(K)
        decay = np.zeros(self.ortho_base_dim, dtype=np.float32)
        cdim = ctypes.c_int(0)
        _LIB.wokdc_get_state(
            wok,
            K.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            decay.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(cdim),
        )
        return {"K": K, "V": V, "decay": decay, "current_dim": int(cdim.value)}

    def set_state(self, state: dict):
        wok = self.window_compressor()
        if not wok:
            return
        K = np.ascontiguousarray(state["K"], dtype=np.float32)
        V = np.ascontiguousarray(state["V"], dtype=np.float32)
        decay = np.ascontiguousarray(state["decay"], dtype=np.float32)
        _LIB.wokdc_set_state(
            wok,
            K.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            decay.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(state.get("current_dim", 0)),
        )

    def destroy(self):
        if self._ptr:
            _LIB.rswa_destroy(self._ptr)
            self._ptr = None

    def __del__(self):
        self.destroy()


# -----------------------------------------------------------------------------
# 自检（无需 torch / sglang / vllm）
# -----------------------------------------------------------------------------
def _self_test():
    _LIB.cgc_init()
    print("[py] inject_unified_ir_for_role(sglang/decode) ...")
    rc = inject_unified_ir_for_role(
        CGC_INJECT_ROLE_DECODE, CGC_BACKEND_ADAPTER_SGLANG,
        num_layers=32, reference_len=4, window_size=8,
        ortho_base_dim=8, num_heads=2, head_dim=8,
    )
    print(f"[py]   cgc_error_t = {rc}; kda_replace_mode = {get_kda_replace_mode()} "
          f"(expect True via linkage)")
    print(f"[py]   summary = {unified_ir_summary(CGC_INJECT_ROLE_DECODE)}")

    rng = np.random.default_rng(0)
    mgr = RSWAManager(num_heads=2, head_dim=8, reference_len=4,
                      window_size=8, ortho_base_dim=8)
    for _ in range(4):
        mgr.set_reference(rng.standard_normal((2, 8)).astype(np.float32),
                          rng.standard_normal((2, 8)).astype(np.float32))
    for _ in range(20):
        mgr.feed_token(rng.standard_normal((2, 8)).astype(np.float32),
                       rng.standard_normal((2, 8)).astype(np.float32))
    Q = rng.standard_normal((2, 8)).astype(np.float32)
    out = mgr.combined_attention(Q)
    assert np.all(np.isfinite(out)), "combined attention not finite"
    print(f"[py] RSWAManager.combined_attention OK (norm={float(np.linalg.norm(out)):.4f}); "
          f"visible={mgr.visible_count()}")
    st = mgr.get_state()
    print(f"[py] window compressor state: current_dim={st['current_dim']}")
    mgr.destroy()
    print("[py] self-test PASS ✅")


if __name__ == "__main__":
    _self_test()
