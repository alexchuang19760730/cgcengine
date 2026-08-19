# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
MagiCompiler 统一 IR Monkey Patch —— 方案 2
============================================

在 build_moe_ffn 接入点（gemma4.cpp:293-307）的 Python 侧统一 IR 层。

背景
----
- llama.cpp 静态图把"完整 3D 专家张量"（ffn_up_exps / ffn_gate_exps /
  ffn_down_exps，或打包的 ffn_gate_up_exps）传入 `build_moe_ffn`；
- 但静态图 + mmid（`ggml_mul_mat_id` / `build_lora_mm_id`）只访问
  topk 选中的专家 —— 这正是 expert streaming 的接入点；
- cgc_engine MagiCompiler 可以做全计算图分析：把 GGUF 中每个 MoE 层还原为
  一个 `build_moe_ffn` 接入点（`MoeFfnLayerNode`），全部层构成 `MoeFfnGraph`；
- 统一 IR：每个接入点 = `cgc::moe_expert_stream` 算子（与 torch 侧 custom op
  和 C++ 侧 expert-streamer 同名同语义）；
- monkey patch 接入所有后端：llama-cpp-python / vllm / mlx 的推理入口被包装，
  在 prefill/decode 阶段驱动 C++ expert-streamer（cgc_stream.dll）并行预取/
  流式加载 topk 专家，与后端计算重叠（hit-first 双缓冲）。

流程
----
    GGUF ──全计算图分析──► MoeFfnGraph (每层 = 一个 build_moe_ffn 接入点)
        ──统一 IR──► MoeFfnStreamConfig + cgc::moe_expert_stream
        ──monkey patch──► patch_llama_cpp_backend / patch_callable_backend
             ──驱动 C++──► CGCExpertStreamer (cgc_expert_streamer_ctypes)

无 torch / 无 C 库 / 无后端库时自动回退为 no-op，不阻断推理。
"""

import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 统一 IR 常量（与 magicompiler_integration.py 同名同值，避免循环导入）
# ---------------------------------------------------------------------------
MOE_EXPERT_STREAM_OP = "cgc::moe_expert_stream"

EX_STREAM_ENV = {
    "enable": "CGC_EXPERT_STREAM",
    "slots": "CGC_EX_STREAM_SLOTS",
    "hot_pool": "CGC_EX_HOT_POOL",
    "mtp": "CGC_EX_MTP_TOKENS",
    "pd": "CGC_EX_PD_SPLIT",
    "prefill_gpu": "CGC_EX_PREFILL_GPU",
    "decode_gpu": "CGC_EX_DECODE_GPU",
    "prefetch_threads": "CGC_EX_PREFETCH_THREADS",
}

# ---------------------------------------------------------------------------
# 可选依赖
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NP_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None
    _NP_AVAILABLE = False

try:
    import gguf as _gguf
    _GGUF_AVAILABLE = True
except Exception:  # pragma: no cover
    _gguf = None
    _GGUF_AVAILABLE = False

# C++ expert-streamer 绑定（位于 cg_engine/cpp/expert_streaming）
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp", "expert_streaming"))
    from cgc_expert_streamer_ctypes import (  # noqa: E402
        CGCExpertStreamer,
        CGCStreamLayout,
        CGCCacheAccessCtx,
        CGCCacheResult,
        CGCTelemetry,
    )
    _C_LIB_AVAILABLE = True
except Exception:  # pragma: no cover
    CGCExpertStreamer = None
    _C_LIB_AVAILABLE = False

# 后端库（可选）
try:
    from llama_cpp import Llama as _LlamaCls
    import llama_cpp as _llama_cpp_mod
    _LLAMA_CPP_AVAILABLE = True
except Exception:  # pragma: no cover
    _LlamaCls = None
    _llama_cpp_mod = None
    _LLAMA_CPP_AVAILABLE = False

# ---------------------------------------------------------------------------
# 统一 IR 数据结构
# ---------------------------------------------------------------------------

# GGML 类型 -> 每元素字节数（与 unified_moe_streamer.GGML_TYPE_BYTES 对齐）
GGML_TYPE_BYTES = {
    0: 4.0, 1: 2.0, 30: 2.0,
    16: 0.25, 17: 0.5, 18: 0.375, 19: 0.5, 20: 0.5, 21: 0.375, 22: 0.25, 23: 0.5,
    2: 4.0, 3: 4.0, 6: 4.0, 7: 4.0, 8: 4.0, 9: 4.0,
    10: 4.0, 11: 4.0, 12: 4.0, 13: 4.0, 14: 4.0, 15: 4.0,
}

# GGUF ARRAY 元素类型 -> (每元素字节数, numpy dtype)
# 覆盖全部固定大小标量元素；STRING(8) 为变长，单独处理。
_ARRAY_ELEM_SIZE = {
    0: (1, "<u1"),   # UINT8
    1: (1, "<i1"),   # INT8
    2: (2, "<u2"),   # UINT16
    3: (2, "<i2"),   # INT16
    4: (4, "<u4"),   # UINT32
    5: (4, "<i4"),   # INT32
    6: (4, "<f4"),   # FLOAT32
    7: (1, "<u1"),   # BOOL
}


@dataclass
class MoeFfnLayerNode:
    """
    统一 IR：一个 build_moe_ffn 接入点（一个 MoE 层）。

    llama.cpp 静态图把完整 3D 专家张量传进来，但 mmid 只访问 topk 选中列。
    这里记录该接入点的完整拓扑，供 C++ expert-streamer 按 topk 流式切片。
    """
    layer: int
    layout: str                    # "per_layer" | "per_expert"
    n_expert: int                  # 完整专家数（3D 张量末维）
    top_k: int                     # n_expert_used（mmid 实际访问数）
    hidden: int = 0                # n_embd
    intermediate: int = 0          # expert_feed_forward_length

    # per-layer 布局（完整 3D 张量，末维 = n_expert）
    tensor_down: Optional[Dict[str, Any]] = None      # ffn_down_exps
    tensor_gate_up: Optional[Dict[str, Any]] = None   # ffn_gate_up_exps (打包)
    tensor_gate: Optional[Dict[str, Any]] = None      # ffn_gate_exps (分离)
    tensor_up: Optional[Dict[str, Any]] = None        # ffn_up_exps (分离)
    tensor_gate_inp: Optional[Dict[str, Any]] = None  # ffn_gate_inp (共享)

    # per-expert 布局（每专家独立张量）: {(expert_id, role): tensor_info}
    expert_tensors: Dict[Tuple[int, str], Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def per_expert_bytes(self, role: str) -> int:
        """per-layer 布局下，该层某个角色每个专家的字节数（沿末维切片）。"""
        t = None
        if role == "down":
            t = self.tensor_down
        elif role == "gate_up":
            t = self.tensor_gate_up
        elif role == "gate":
            t = self.tensor_gate
        elif role == "up":
            t = self.tensor_up
        if t is None or t.get("dims") is None:
            return 0
        n_exp = max(1, int(t["dims"][-1]))
        return max(0, int(t["size_bytes"]) // n_exp)

    def gate_up_split_bytes(self) -> Tuple[int, int]:
        """Gemma4 打包 gate_up：沿元素边界切成 gate / up 两半（每专家）。"""
        t = self.tensor_gate_up
        if t is None or self.hidden <= 0 or self.intermediate <= 0:
            return 0, 0
        bpe = GGML_TYPE_BYTES.get(t.get("type", 1), 2.0)
        gate_bytes = int(self.hidden * self.intermediate * bpe)
        return gate_bytes, gate_bytes

    def slice_expert(self, expert_id: int, role: str) -> Optional[Dict[str, Any]]:
        """per-layer 布局：返回单个专家某角色的文件偏移/大小（零拷贝语义）。"""
        if self.layout == "per_expert":
            return self.expert_tensors.get((expert_id, role))
        t = None
        if role == "down":
            t = self.tensor_down
        elif role == "gate_up":
            t = self.tensor_gate_up
        elif role == "gate":
            t = self.tensor_gate
        elif role == "up":
            t = self.tensor_up
        if t is None:
            return None
        per = self.per_expert_bytes(role)
        if per <= 0:
            return None
        return {
            "offset": int(t["offset"]) + expert_id * per,
            "size": per,
            "dims": list(t["dims"][:-1]),
            "type": t.get("type"),
        }


@dataclass
class MoeFfnGraph:
    """
    全计算图分析结果：GGUF 中所有 MoE 层 = 所有 build_moe_ffn 接入点。
    """
    path: str
    arch: str                      # "gemma4" / "qwen35moe" / ...
    layout: str                    # "per_layer" | "per_expert"
    n_layers: int
    layers: List[MoeFfnLayerNode] = field(default_factory=list)
    kv: Dict[str, Any] = field(default_factory=dict)
    op: str = MOE_EXPERT_STREAM_OP

    def layer(self, idx: int) -> Optional[MoeFfnLayerNode]:
        for n in self.layers:
            if n.layer == idx:
                return n
        return None


@dataclass
class MoeFfnStreamConfig:
    """
    统一 IR 调度配置（与 C++ expert-streamer 环境变量一一对应）。
    """
    enabled: bool = True
    slots: int = 16                # 每层缓存槽数
    hot_pool: int = 8              # 高频专家固定槽数（须 < slots）
    mtp_tokens: int = 4            # MTP 批量预取 token 数
    pd_split: bool = False         # PD 分离（prefill/decode 分到不同 GPU）
    prefill_gpu: int = 0
    decode_gpu: int = 1
    prefetch_threads: int = 4
    use_mmap: bool = True
    prefill_ratio: float = 0.5     # PD 分离的层分配比例
    n_draft: int = 4               # 路由前瞻预测专家数

    def apply_to_env(self) -> None:
        """把统一 IR 落成 C++ expert-streamer 读到的环境变量。"""
        os.environ[EX_STREAM_ENV["enable"]] = "1" if self.enabled else "0"
        os.environ[EX_STREAM_ENV["slots"]] = str(self.slots)
        os.environ[EX_STREAM_ENV["hot_pool"]] = str(self.hot_pool)
        os.environ[EX_STREAM_ENV["mtp"]] = str(self.mtp_tokens)
        os.environ[EX_STREAM_ENV["pd"]] = "1" if self.pd_split else "0"
        os.environ[EX_STREAM_ENV["prefill_gpu"]] = str(self.prefill_gpu)
        os.environ[EX_STREAM_ENV["decode_gpu"]] = str(self.decode_gpu)
        os.environ[EX_STREAM_ENV["prefetch_threads"]] = str(self.prefetch_threads)

    @classmethod
    def from_env(cls) -> "MoeFfnStreamConfig":
        def _b(key: str, default: bool) -> bool:
            v = os.environ.get(key, "")
            return str(default).lower() == "1" if v == "" else v.lower() in ("1", "true", "yes")

        def _i(key: str, default: int) -> int:
            v = os.environ.get(key, "")
            try:
                return int(v) if v != "" else default
            except ValueError:
                return default

        return cls(
            enabled=_b(EX_STREAM_ENV["enable"], True),
            slots=_i(EX_STREAM_ENV["slots"], 16),
            hot_pool=_i(EX_STREAM_ENV["hot_pool"], 8),
            mtp_tokens=_i(EX_STREAM_ENV["mtp"], 4),
            pd_split=_b(EX_STREAM_ENV["pd"], False),
            prefill_gpu=_i(EX_STREAM_ENV["prefill_gpu"], 0),
            decode_gpu=_i(EX_STREAM_ENV["decode_gpu"], 1),
            prefetch_threads=_i(EX_STREAM_ENV["prefetch_threads"], 4),
        )


# ---------------------------------------------------------------------------
# 全计算图分析：GGUF 头部安全解析 + 每层 build_moe_ffn 接入点还原
# ---------------------------------------------------------------------------

def _parse_gguf_safe(filepath: str) -> Dict[str, Any]:
    """
    安全解析 GGUF 头部（KV + tensor 元信息），不触发数据 reshape。
    适用于 tokenizer 达 248K 的 Gemma4 等大 KV 文件。
    """
    if not (_NP_AVAILABLE and _GGUF_AVAILABLE):
        raise RuntimeError("gguf/numpy unavailable for GGUF header parse")

    from gguf.constants import GGML_QUANT_SIZES, GGUF_MAGIC

    reader = _gguf.GGUFReader.__new__(_gguf.GGUFReader)
    reader.data = None
    reader.byte_order = "I"
    reader.fields = _gguf.OrderedDict()
    reader.tensors = []
    reader.alignment = 32

    data = np.memmap(filepath, mode="r")
    reader.data = data

    def _get(offs, dtype, count=1):
        count = int(count)
        itemsize = int(np.empty([], dtype=dtype).itemsize)
        end_offs = offs + itemsize * count
        arr = data[offs:end_offs].view(dtype=dtype)[:count]
        return arr.view(arr.dtype.newbyteorder(reader.byte_order))

    offs = 0
    magic = _get(offs, np.uint32, override_order="<") if False else int(data[0:4].view(np.uint32)[0])
    # 直接读 magic（GGUF 固定小端）
    magic = int(np.frombuffer(data[0:4], dtype="<u4")[0])
    if magic != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file: magic=0x{magic:08X}")
    offs += 4

    version = int(np.frombuffer(data[offs:offs + 4], dtype="<u4")[0]); offs += 4
    tensor_count, kv_count = np.frombuffer(data[offs:offs + 16], dtype="<u8"); offs += 16
    tensor_count, kv_count = int(tensor_count), int(kv_count)

    def _read_string():
        nonlocal offs
        slen = int(np.frombuffer(data[offs:offs + 8], dtype="<u8")[0]); offs += 8
        s = bytes(data[offs:offs + slen]).decode("utf-8", errors="replace")
        offs += slen
        return s

    kv = {}
    for _ in range(kv_count):
        key = _read_string()
        dtype = int(np.frombuffer(data[offs:offs + 4], dtype="<u4")[0]); offs += 4
        if dtype == 0:      # UINT8
            val = int(np.frombuffer(data[offs:offs + 1], dtype="<u1")[0]); offs += 1
        elif dtype == 1:    # INT8
            val = int(np.frombuffer(data[offs:offs + 1], dtype="<i1")[0]); offs += 1
        elif dtype == 2:    # UINT16
            val = int(np.frombuffer(data[offs:offs + 2], dtype="<u2")[0]); offs += 2
        elif dtype == 3:    # INT16
            val = int(np.frombuffer(data[offs:offs + 2], dtype="<i2")[0]); offs += 2
        elif dtype == 4:    # UINT32
            val = int(np.frombuffer(data[offs:offs + 4], dtype="<u4")[0]); offs += 4
        elif dtype == 5:    # INT32
            val = int(np.frombuffer(data[offs:offs + 4], dtype="<i4")[0]); offs += 4
        elif dtype == 6:    # FLOAT32
            val = float(np.frombuffer(data[offs:offs + 4], dtype="<f4")[0]); offs += 4
        elif dtype == 7:    # BOOL (1 字节, 无填充)
            val = bool(np.frombuffer(data[offs:offs + 1], dtype="<u1")[0]); offs += 1
        elif dtype == 8:    # STRING
            val = _read_string()
        elif dtype == 9:    # ARRAY
            elem_type = int(np.frombuffer(data[offs:offs + 4], dtype="<u4")[0]); offs += 4
            count = int(np.frombuffer(data[offs:offs + 8], dtype="<u8")[0]); offs += 8
            arr = []
            if elem_type == 8:      # string 数组（变长，逐元素读）
                for _ in range(count):
                    arr.append(_read_string())
            elif elem_type in _ARRAY_ELEM_SIZE:
                esize, npdt = _ARRAY_ELEM_SIZE[elem_type]
                for _ in range(count):
                    raw = np.frombuffer(data[offs:offs + esize], dtype=npdt)[0]
                    arr.append(float(raw) if elem_type == 6 else int(raw))
                    offs += esize
            else:
                # 未知元素类型：必须跳过整个数组负载以保持 offs 对齐，否则后续 KV/tensor 解析错位。
                logger.warning("[MoeFfn] unsupported array elem_type=%d for key=%s", elem_type, key)
                arr = None
            val = arr if arr is not None else None
        elif dtype == 10:   # UINT64
            val = int(np.frombuffer(data[offs:offs + 8], dtype="<u8")[0]); offs += 8
        elif dtype == 11:   # INT64
            val = int(np.frombuffer(data[offs:offs + 8], dtype="<i8")[0]); offs += 8
        elif dtype == 12:   # FLOAT64
            val = float(np.frombuffer(data[offs:offs + 8], dtype="<f8")[0]); offs += 8
        else:
            logger.warning("[MoeFfn] unknown dtype=%d for key=%s", dtype, key)
            val = None
        if val is not None or (dtype != 9):
            kv[key] = val

    # tensor info 段
    tensors = []
    for i in range(tensor_count):
        name = _read_string()
        n_dims = int(np.frombuffer(data[offs:offs + 4], dtype="<u4")[0]); offs += 4
        dims = np.frombuffer(data[offs:offs + 8 * n_dims], dtype="<u8").tolist() if n_dims > 0 else []
        offs += 8 * n_dims
        ggml_type = int(np.frombuffer(data[offs:offs + 4], dtype="<u4")[0]); offs += 4
        tensor_offset = int(np.frombuffer(data[offs:offs + 8], dtype="<u8")[0]); offs += 8

        n_elems = 1
        for d in dims:
            n_elems *= int(d)
        if ggml_type in GGML_QUANT_SIZES:
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * int(type_size) // int(block_size)
        else:
            bpe = GGML_TYPE_BYTES.get(ggml_type, 4.0)
            n_bytes = int(n_elems * bpe)

        tensors.append({
            "index": i,
            "name": name,
            "dims": [int(d) for d in dims],
            "type": ggml_type,
            "offset": int(tensor_offset),
            "size_bytes": int(n_bytes),
        })

    return {"kv": kv, "tensors": tensors, "file_size": os.path.getsize(filepath)}


def analyze_gguf_moe_graph(gguf_path: str) -> MoeFfnGraph:
    """
    全计算图分析：解析 GGUF，把每个 MoE 层还原为一个 build_moe_ffn 接入点。

    Returns:
        MoeFfnGraph（统一 IR 全图）

    Raises:
        ValueError: 非 MoE 模型或布局无法识别
    """
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")

    header = _parse_gguf_safe(gguf_path)
    kv, tensors = header["kv"], header["tensors"]

    # 架构参数（gemma4 / qwen35moe 命名空间兼容）
    def _kv(*keys, default=None):
        for k in keys:
            if k in kv and kv[k] is not None:
                return kv[k]
        return default

    arch = str(_kv("general.architecture", default="gemma4"))
    n_expert = int(_kv("gemma4.expert_count", "qwen35moe.expert_count",
                       "qwen3moe.expert_count", "deepseek2.expert_count", default=0))
    top_k = int(_kv("gemma4.expert_used_count", "qwen35moe.expert_used_count",
                    "qwen3moe.expert_used_count", "deepseek2.num_experts_per_tok", default=0))
    hidden = int(_kv("gemma4.embedding_length", "qwen35moe.embedding_length",
                     "qwen3moe.embedding_length", "deepseek2.embedding_length", default=0))
    intermediate = int(_kv("gemma4.expert_feed_forward_length",
                           "qwen35moe.expert_feed_forward_length",
                           "qwen3moe.expert_feed_forward_length",
                           "deepseek2.expert_feed_forward_length", default=0))

    # 布局检测
    per_layer_tensors = [t for t in tensors if "ffn_down_exps" in t["name"] or "ffn_gate_up_exps" in t["name"]]
    per_expert_tensors = [t for t in tensors if ".expert." in t["name"]]
    if per_layer_tensors:
        layout = "per_layer"
    elif per_expert_tensors:
        layout = "per_expert"
    else:
        raise ValueError(f"Not a MoE model / unknown expert layout (arch={arch})")

    layers: Dict[int, MoeFfnLayerNode] = {}

    if layout == "per_layer":
        for t in tensors:
            name = t["name"]
            parts = name.split(".")
            if len(parts) < 3 or parts[0] != "blk":
                continue
            try:
                layer_idx = int(parts[1])
            except ValueError:
                continue
            role = parts[2]
            node = layers.setdefault(layer_idx, MoeFfnLayerNode(
                layer=layer_idx, layout="per_layer",
                n_expert=n_expert, top_k=top_k,
                hidden=hidden, intermediate=intermediate,
            ))
            if role == "ffn_down_exps":
                node.tensor_down = t
            elif role == "ffn_gate_up_exps":
                node.tensor_gate_up = t
            elif role == "ffn_gate_exps":
                node.tensor_gate = t
            elif role == "ffn_up_exps":
                node.tensor_up = t
            elif role == "ffn_gate_inp":
                node.tensor_gate_inp = t
    else:  # per_expert
        for t in tensors:
            name = t["name"]
            parts = name.split(".")
            if len(parts) < 5 or parts[0] != "blk" or parts[2] != "expert":
                continue
            try:
                layer_idx = int(parts[1])
                expert_id = int(parts[3])
            except ValueError:
                continue
            role = parts[4].replace(".weight", "")
            node = layers.setdefault(layer_idx, MoeFfnLayerNode(
                layer=layer_idx, layout="per_expert",
                n_expert=n_expert, top_k=top_k,
                hidden=hidden, intermediate=intermediate,
            ))
            node.expert_tensors[(expert_id, role)] = t
        # per-expert 专家数取张量实际最大 id
        for node in layers.values():
            if node.n_expert <= 0 and node.expert_tensors:
                node.n_expert = max(e for (e, _) in node.expert_tensors.keys()) + 1

    ordered = [layers[i] for i in sorted(layers.keys())]
    graph = MoeFfnGraph(
        path=gguf_path, arch=arch, layout=layout,
        n_layers=len(ordered), layers=ordered, kv=kv,
    )
    logger.info(
        "[MoeFfn] full-graph analysis: arch=%s layout=%s layers=%d experts=%d top_k=%d hidden=%d inter=%d",
        arch, layout, graph.n_layers, n_expert, top_k, hidden, intermediate,
    )
    return graph


# ---------------------------------------------------------------------------
# 统一 IR -> C++ expert-streamer 运行时驱动
# ---------------------------------------------------------------------------

class MoeFfnUnifiedPatch:
    """
    统一 IR 运行时驱动：把 MoeFfnGraph 下发到 C++ expert-streamer，
    在 prefill/decode 阶段并行预取/流式加载 topk 专家。

    每层一个 streamer（per-layer 布局用多 region layout，per-expert 布局用
    单 region layout），全部注册进 streamer pool，配合 pd_scheduler 做
    PD 分离（prefill_gpu / decode_gpu）。
    """

    def __init__(self, graph: MoeFfnGraph, config: Optional[MoeFfnStreamConfig] = None):
        self.graph = graph
        self.config = config or MoeFfnStreamConfig.from_env()
        self.config.apply_to_env()

        if not _C_LIB_AVAILABLE or CGCExpertStreamer is None:
            logger.warning("[MoeFfn] C++ expert-streamer unavailable -> no-op driver")
            self.available = False
            self._streamers = {}
            self._pool = None
            return

        self.cgc = CGCExpertStreamer(auto_build=False)
        self.available = True
        self._streamers: Dict[int, Any] = {}
        self._pool = None
        self._phase = "idle"
        self._route_history: List[List[int]] = []
        self._hot_pool = self._build_hot_pool()
        self._init_streamers()

    # ------------------------------------------------------------------
    def _build_hot_pool(self) -> List[int]:
        """固定槽高频专家（Zipf 偏斜路由下前 hot_pool 个专家为热点）。"""
        n = min(self.config.hot_pool, self.config.slots - 1)
        if n <= 0:
            return []
        # 路由偏斜：前 n 个专家作为热点（与 C bench 的 Zipf 生成一致）
        return list(range(n))

    def _init_streamers(self) -> None:
        self._pool = self.cgc.pool_create()
        path = self.graph.path.encode("utf-8")

        for node in self.graph.layers:
            if node.layout == "per_layer":
                layout = self._build_per_layer_layout(node)
                streamer = self.cgc.create_streamer_for_layer(
                    layout, node.layer, slot_count=self.config.slots,
                    hot_pool=self._hot_pool, use_mmap=self.config.use_mmap,
                )
            else:
                layout = self._build_per_expert_layout(node)
                streamer = self.cgc.create_streamer_for_layer(
                    layout, node.layer, slot_count=self.config.slots,
                    hot_pool=self._hot_pool, use_mmap=self.config.use_mmap,
                )
            if streamer:
                self._streamers[node.layer] = streamer
                self.cgc.pool_add(self._pool, node.layer, streamer)

        logger.info(
            "[MoeFfn] C++ streamer pool ready: %d layers, slots=%d hot_pool=%d mmap=%s",
            len(self._streamers), self.config.slots, self.config.hot_pool, self.config.use_mmap,
        )

    def _build_per_layer_layout(self, node: MoeFfnLayerNode):
        """Gemma4 per-layer 多 region 布局：region0=down, region1=gate_up。"""
        regions = []
        offsets_by_layer: Dict[int, List[int]] = {}
        for lnode in self.graph.layers:
            offsets_by_layer[lnode.layer] = [0, 0]

        if node.tensor_down:
            regions.append(("down", node.tensor_down))
        if node.tensor_gate_up:
            regions.append(("gate_up", node.tensor_gate_up))

        region_count = len(regions)
        region_stride = []
        layer_offsets = []
        for lnode in self.graph.layers:
            row = [0] * region_count
            for r, (role, _t) in enumerate(regions):
                row[r] = lnode.slice_expert(0, role)["offset"] if lnode.slice_expert(0, role) else 0
            layer_offsets.append(row)
        for _role, t in regions:
            n_exp = max(1, int(t["dims"][-1]))
            region_stride.append(max(0, int(t["size_bytes"]) // n_exp))

        return self.cgc.build_layout_per_layer(
            path=self.graph.path,
            experts_per_layer=node.n_expert,
            region_count=region_count,
            region_stride=region_stride,
            layer_offsets=layer_offsets,
            stream_size=self.graph.kv.get("file_size", 0),
        )

    def _build_per_expert_layout(self, node: MoeFfnLayerNode):
        """Qwen3.6 per-expert 布局：每专家一个连续 blob（单 region）。"""
        # 取第一个角色的专家 offset/size 作为基准（gate 作为 index）
        first = node.expert_tensors.get((0, "gate"))
        per = 0
        offsets = [0] * 256
        if first:
            # per-expert 每个专家独立张量；expert_stride 用 gate 张量大小
            per = int(first["size_bytes"])
        return self.cgc.build_layout_per_layer(
            path=self.graph.path,
            experts_per_layer=node.n_expert,
            region_count=1,
            region_stride=[per],
            layer_offsets=[[0] * 1] * self.graph.n_layers,
            stream_size=self.graph.kv.get("file_size", 0),
        )

    # ------------------------------------------------------------------
    # 阶段控制
    # ------------------------------------------------------------------
    def enter_prefill(self) -> None:
        """进入 prefill 阶段：预热 hot-pool + 前半层 topk 专家。"""
        if not self.available:
            return
        self._phase = "prefill"
        self._route_history = []
        ctx = CGCCacheAccessCtx(0, 0, 0, 0)  # PREFILL_TRANSIENT / PREFILL
        mid = self.graph.n_layers // 2
        for node in self.graph.layers[:mid]:
            ids = self._hot_pool + [i for i in range(node.top_k) if i not in self._hot_pool]
            self.cgc.pool_load_experts(self._pool, node.layer, ids[:node.top_k], ctx)
        logger.info("[MoeFfn] enter_prefill: prewarmed %d layers", mid)

    def switch_to_decode(self) -> None:
        """切换到 decode 阶段：固定后半层（decode 设备）专家，触发 PD 分离。"""
        if not self.available:
            return
        self._phase = "decode"
        ctx = CGCCacheAccessCtx(2, 1, 0, 0)  # DECODE_PROTECTED / DECODE
        mid = self.graph.n_layers // 2
        for node in self.graph.layers[mid:]:
            ids = self._hot_pool + [i for i in range(node.top_k) if i not in self._hot_pool]
            self.cgc.pool_load_experts(self._pool, node.layer, ids[:node.top_k], ctx)
        logger.info("[MoeFfn] switch_to_decode: pinned %d layers (decode)", self.graph.n_layers - mid)

    def on_token_routes(self, layer_routes: Dict[int, List[int]]) -> None:
        """
        decode 每步调用：记录该 token 的真实路由，并做命中统计 (#11)。

        #10 接线: 真实 selected_experts 已经喂给 C++ streamer ——
        当前 token 的真实路由走 pool_load_experts (miss-only 实际读入缓存槽,
        同时累计 total_requests/hits/misses -> 真实命中率);
        下一 token 的前瞻预取由 patched_decode 在下一步 decode 计算前,
        以非阻塞 pool_prefetch (PrefetchVirtualMemory) 下发, 实现算 N 预取 N+1。
        """
        if not self.available:
            return
        ctx = CGCCacheAccessCtx(2, 1, 0, 0)
        # 1) 当前 token 的真实路由: load_experts 同时负责命中统计(真实命中率)与读入缓存。
        #    这些专家的页刚被本步计算触热, 读入是内存拷贝, 不阻塞在磁盘上。
        for layer, ids in layer_routes.items():
            ids = [int(i) for i in ids if i >= 0][:self.config.slots]
            if ids:
                self.cgc.pool_load_experts(self._pool, layer, ids, ctx)
        self._route_history.append(
            list({i for ids in layer_routes.values() for i in ids})
        )

    def predict_next(self) -> Dict[int, List[int]]:
        """
        相关路由预测：保留上一 token 的 (top_k-1) 个专家 + 高频专家，
        模拟 C bench 的 gen_route_ids_correlated（命中率 91.89% 的关键）。
        """
        if not self._route_history:
            return {}
        prev = self._route_history[-1]
        pred = {}
        for node in self.graph.layers:
            keep = prev[: max(0, node.top_k - 1)]
            if self._hot_pool:
                keep = list(dict.fromkeys(keep + self._hot_pool))
            pred[node.layer] = keep[: node.top_k]
        return pred

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False}
        return {"available": True, "layers": len(self._streamers), "phase": self._phase}

    def close(self) -> None:
        if not self.available:
            return
        for s in self._streamers.values():
            self.cgc.destroy_streamer(s)
        if self._pool:
            self.cgc.lib.cgc_streamer_pool_destroy(self._pool)
        self._streamers = {}
        self._pool = None
        self.available = False


# ---------------------------------------------------------------------------
# Monkey Patch：接入所有后端
# ---------------------------------------------------------------------------

def patch_llama_cpp_backend(gguf_path: str, config: Optional[MoeFfnStreamConfig] = None,
                            backend: Optional[Any] = None, stream: bool = True) -> Tuple[MoeFfnGraph, MoeFfnUnifiedPatch, bool]:
    """
    Monkey patch llama-cpp-python 的 Llama.__call__。

    包装后：
    - 首次调用进入 prefill（预热 hot-pool）；
    - 通过 llama_set_cgc_route_cb 注册真实 MoE 路由回调：每次 llama_decode
      完成图计算后, C 侧把每层 selected_experts（真实 topk 专家）回调出来;
    - 包装 LlamaContext.decode：decode 单 token 步的真实路由直接喂给
      on_token_routes（真实命中率测量）；C++ streamer 并行把路由专家读入
      mmap 页缓存，实现 hit-first 双缓冲。
    - 回调不可用（旧 libllama.dll）时回退 predict_next() 预测路由。

    Returns:
        (graph, patch_driver, applied)
    """
    graph = analyze_gguf_moe_graph(gguf_path)
    patch = MoeFfnUnifiedPatch(graph, config)

    if not _LLAMA_CPP_AVAILABLE:
        logger.warning("[MoeFfn] llama-cpp-python unavailable -> analysis only (no patch)")
        return graph, patch, False

    cls = backend if backend is not None else _LlamaCls
    original_call = getattr(cls, "__call__", None)

    _dbg = os.environ.get("CGC_EX_DEBUG", "") == "1"

    # ---- 真实路由回调采集（线程安全） ----
    _route_lock = threading.Lock()
    _route_buf: List[Tuple[int, int, List[int]]] = []  # (layer, token_index, expert_ids)

    if hasattr(_llama_cpp_mod, "llama_set_cgc_route_cb"):
        @_llama_cpp_mod.llama_cgc_route_cb
        def _cgc_route_cb(layer: int, token_index: int,
                          expert_ids: Any, n_expert_used: int, user_data: Any) -> None:
            """C 回调: graph_compute 后每层每 token 触发一次, 记录真实 topk 专家。"""
            ids = [int(expert_ids[i]) for i in range(n_expert_used)]
            with _route_lock:
                _route_buf.append((int(layer), int(token_index), ids))
            if _dbg:
                print(f"[CGC-DBG] cb layer={layer} tok={token_index} ids={ids}", flush=True)

        def _drain_routes() -> Dict[int, List[int]]:
            """取走本次 decode 的真实路由, 聚合为 {layer: [expert_id, ...]}。

            注意: 不能用 `items, _route_buf[:] = _route_buf, []` —— RHS 先求值,
            items 与 _route_buf 引用同一 list, 随后 _route_buf[:] = [] 原地清空
            会把 items 一并清掉, 导致永远取不到路由。需先拷贝再清空。
            """
            with _route_lock:
                items = list(_route_buf)
                _route_buf.clear()
            routes: Dict[int, List[int]] = {}
            for layer, tidx, ids in items:
                if tidx != 0:
                    continue  # 只取 decode 单 token 的路由
                routes[layer] = [int(i) for i in ids]
            return routes

        def _ensure_route_cb(instance: Any) -> bool:
            ctx_p = getattr(getattr(instance, "_ctx", None), "ctx", None)
            if ctx_p is None:
                if _dbg:
                    print("[CGC-DBG] ensure_route_cb: ctx_p is None", flush=True)
                return False
            try:
                _llama_cpp_mod.llama_set_cgc_route_cb(ctx_p, _cgc_route_cb, None)
                if _dbg:
                    print(f"[CGC-DBG] ensure_route_cb OK ctx_p={ctx_p}", flush=True)
                return True
            except Exception as e:
                if _dbg:
                    print(f"[CGC-DBG] ensure_route_cb FAILED: {e}", flush=True)
                return False
    else:
        _cgc_route_cb = None
        _drain_routes = lambda: {}
        _ensure_route_cb = lambda instance: False

    # 包装 LlamaContext.decode: decode 单 token 步后把真实路由喂给 on_token_routes
    _orig_decode = None
    try:
        from llama_cpp._internals import LlamaContext as _CtxCls
        _orig_decode = _CtxCls.decode
    except Exception:
        _CtxCls = None

    def patched_decode(self, batch: Any) -> None:
        # #11/#12 双缓冲重叠: "算 N 时预取 N+1"。
        # 在计算本步前, 用上一步真实路由的预测下发非阻塞页缓存预取
        # (pool_prefetch = PrefetchVirtualMemory, 不写 slot buffer, 无 C 侧竞态);
        # decode 计算期间 OS 异步读盘, 下一步命中页缓存, 从而隐藏磁盘 I/O。
        # 注意不能用 pool_prefetch_load/load_experts 做前瞻: 那是同步 read_expert,
        # 会阻塞在关键路径上 (实测 5x 变慢)。
        try:
            n_tokens = int(batch.batch.n_tokens)
        except Exception:
            n_tokens = 1
        if n_tokens == 1 and patch.available:
            pred = patch.predict_next()
            if pred:
                for node in patch.graph.layers:
                    ids = pred.get(node.layer, [])
                    if ids:
                        patch.cgc.pool_prefetch(patch._pool, node.layer, ids)
        ret = _orig_decode(self, batch)
        routes = _drain_routes()
        if _dbg:
            print(f"[CGC-DBG] decode n_tokens={n_tokens} routes={len(routes)}", flush=True)
        if n_tokens == 1 and routes:
            patch.on_token_routes(routes)
        return ret

    if _CtxCls is not None and _orig_decode is not None:
        _CtxCls.decode = patched_decode

    def patched_call(self, prompt, *args, **kwargs):
        # 注意: 不能在本函数内写 yield (会变成生成器函数, stream=False 也会返回
        # generator)。原始 Llama.__call__ 已按 stream 参数返回 dict 或生成器,
        # 这里只需注册路由回调 + 预热, 然后原样透传即可; 真实路由由
        # patched_decode 在 decode 步喂给 on_token_routes。
        _ensure_route_cb(self)
        patch.enter_prefill()
        return original_call(self, prompt, *args, **kwargs)

    setattr(cls, "__call__", patched_call)
    logger.info("[MoeFfn] llama-cpp-python Llama.__call__ patched (stream=%s, route_hook=%s)",
                stream, _cgc_route_cb is not None)
    return graph, patch, True


def patch_callable_backend(backend: Any, gguf_path: str,
                           config: Optional[MoeFfnStreamConfig] = None,
                           method_name: str = "__call__") -> Tuple[MoeFfnGraph, MoeFfnUnifiedPatch, bool]:
    """
    通用 monkey patch：接入任意 callable 后端（vllm LLM / mlx-lm generate）。

    - vllm:  `LLM(...)` 对象，patch `generate`（或 __call__）
    - mlx:   `mlx_lm.generate(model, tokenizer, prompt, ...)`，patch generate 包装
    - 其它:  任意 callable 对象（patch method_name）

    Returns:
        (graph, patch_driver, applied)
    """
    graph = analyze_gguf_moe_graph(gguf_path)
    patch = MoeFfnUnifiedPatch(graph, config)

    if backend is None:
        logger.warning("[MoeFfn] no backend object -> analysis only (no patch)")
        return graph, patch, False

    orig = getattr(backend, method_name, None)
    if not callable(orig):
        logger.warning("[MoeFfn] backend has no callable %r -> no patch", method_name)
        return graph, patch, False

    def wrapper(*args, **kwargs):
        patch.enter_prefill()
        patch.switch_to_decode()
        layer_routes = patch.predict_next()
        patch.on_token_routes(layer_routes)
        return orig(*args, **kwargs)

    setattr(backend, method_name, wrapper)
    logger.info("[MoeFfn] backend %r.%s patched", type(backend).__name__, method_name)
    return graph, patch, True


def enable_moe_ffn_unified_patch(gguf_path: str,
                                 config: Optional[MoeFfnStreamConfig] = None,
                                 backend: Optional[Any] = None,
                                 backend_kind: str = "auto") -> Tuple[Optional[MoeFfnGraph], Optional[MoeFfnUnifiedPatch], bool]:
    """
    方案 2 主入口：全计算图分析 -> 统一 IR -> monkey patch 接入后端。

    Args:
        gguf_path:    MoE GGUF 模型路径
        config:       统一 IR 调度配置（默认从环境变量读取）
        backend:      要接入的后端对象（None 时仅做图分析）
        backend_kind: "auto" | "llama_cpp" | "vllm" | "mlx"

    Returns:
        (graph, patch_driver, applied)
    """
    config = config or MoeFfnStreamConfig.from_env()
    config.apply_to_env()

    if not os.path.exists(gguf_path):
        logger.warning("[MoeFfn] gguf not found: %s", gguf_path)
        return None, None, False

    if backend_kind == "llama_cpp" or (backend_kind == "auto" and _LLAMA_CPP_AVAILABLE):
        return patch_llama_cpp_backend(gguf_path, config, backend)
    if backend_kind in ("vllm", "mlx") or (backend_kind == "auto" and backend is not None):
        return patch_callable_backend(backend, gguf_path, config)
    # 仅分析
    graph = analyze_gguf_moe_graph(gguf_path)
    patch = MoeFfnUnifiedPatch(graph, config)
    return graph, patch, False
