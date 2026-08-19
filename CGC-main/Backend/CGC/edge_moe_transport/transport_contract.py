"""transport_contract.py — 端→云层接续 hidden_states + partial_kv 正式 ABI 契约

本契约对应 Gate 2.0 能力 hidden_states_partial_kv_abi：
  端侧执行前 max_local_layer 层后，将 (finished_layer, hidden_states, partial_kv,
  layer_metadata) 序列化为 EdgeCloudLayerHandoff 通过 CQ4/NFSoRDMA 上传云侧；
  云侧 deepseek_v4.py forward loop 入口消费该 handoff，从 finished_layer+1 接续 Prefill。

ABI 稳定性：版本号写入 schema_version，向后兼容由 deserializer 处理。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np


class HandoffSerializationFormat(Enum):
    """端云 handoff 序列化格式"""
    NUMPY_ZEROCOPY = "numpy_zerocopy"  # 生产路径：numpy.save + zlib + header
    SAFETENSORS = "safetensors"  # 备选：safetensors 跨语言
    MSGPACK = "msgpack"  # 兼容路径：msgpack + numpy array


class HandoffValidationError(Exception):
    """handoff 反序列化/校验失败"""


@dataclass
class LayerMetadata:
    """单层执行元数据，用于云侧校验路由一致性"""
    layer_id: int
    layer_norm_eps: float
    layer_norm_weight_hash: str  # 端侧该层 RMSNorm 权重哈希（前 8 字节十六进制），云侧校验同模型同层
    moe_topk_idx_shape: Optional[Tuple[int, ...]] = None  # 若该层为 MoE，记录 topk_idx shape
    moe_topk_weights_dtype: Optional[str] = None  # topk_weights dtype


@dataclass
class EdgeCloudLayerHandoff:
    """端→云层接续正式 ABI 数据结构

    字段对应能力 hidden_states_partial_kv_abi 的 DOPDResumePayloadV2 中间态：
      - finished_layer: 端侧已完成的最后一层 index（0-based）
      - hidden_states: 端侧最后一层输出 hidden_states，shape=[num_tokens, hidden_dim], dtype=float16/bf16
      - partial_kv: 端侧已生成的 KV cache（可选，仅 R-SWA 双层 KV 场景）
      - layer_metadata_list: 每层元数据，用于云侧路由一致性校验
      - model_id: 目标模型 ID，云侧校验同模型
      - schema_version: ABI 版本号
    """
    finished_layer: int
    hidden_states: np.ndarray  # shape=[num_tokens, hidden_dim]
    partial_kv: Optional[Dict[str, np.ndarray]] = None  # {"layer_i_k": ..., "layer_i_v": ...}
    layer_metadata_list: list = field(default_factory=list)  # List[LayerMetadata]
    model_id: str = ""
    schema_version: str = "v2"
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self, expected_model_id: Optional[str] = None) -> None:
        """校验 handoff 字段一致性，失败抛 HandoffValidationError"""
        if self.finished_layer < 0:
            raise HandoffValidationError(f"finished_layer < 0: {self.finished_layer}")
        if not isinstance(self.hidden_states, np.ndarray):
            raise HandoffValidationError(
                f"hidden_states must be numpy.ndarray, got {type(self.hidden_states)}"
            )
        if self.hidden_states.ndim != 2:
            raise HandoffValidationError(
                f"hidden_states must be 2D [num_tokens, hidden_dim], got shape={self.hidden_states.shape}"
            )
        # 允许的 dtype：float16 / float32 + 可选 bfloat16（来自 ml_dtypes）
        _allowed_dtypes = (np.float16, np.float32)
        try:
            import ml_dtypes  # type: ignore
            _allowed_dtypes = (np.float16, np.float32, ml_dtypes.bfloat16)
        except ImportError:
            pass
        if self.hidden_states.dtype not in _allowed_dtypes:
            raise HandoffValidationError(
                f"hidden_states dtype must be float16/bf16/fp32, got {self.hidden_states.dtype}"
            )
        if expected_model_id and self.model_id and self.model_id != expected_model_id:
            raise HandoffValidationError(
                f"model_id mismatch: handoff={self.model_id} expected={expected_model_id}"
            )
        # layer_metadata_list 校验：layer_id 必须连续 0..finished_layer
        if self.layer_metadata_list:
            ids = [m.layer_id for m in self.layer_metadata_list]
            expected_ids = list(range(0, self.finished_layer + 1))
            if ids != expected_ids:
                raise HandoffValidationError(
                    f"layer_metadata_list layer_id not contiguous: got={ids} expected={expected_ids}"
                )

    def to_cloud_resume_kwargs(self) -> Dict[str, Any]:
        """转换为云侧 deepseek_v4.py forward loop 入口消费的 kwargs

        云侧 forward 入口检查 forward_batch.cgc_edge_resume_handoff，若存在则：
          - start_layer = handoff.finished_layer + 1
          - hidden_states = handoff.hidden_states (覆盖输入)
          - 跳过前 finished_layer 层
        """
        return {
            "cgc_edge_resume_from_layer": self.finished_layer + 1,
            "cgc_edge_resume_hidden_states": self.hidden_states,
            "cgc_edge_resume_model_id": self.model_id,
            "cgc_edge_resume_schema_version": self.schema_version,
        }


# ============================================================================
# 序列化 / 反序列化
# ============================================================================

_MAGIC = b"CGCEDGELAYERHANDOFF"
_SCHEMA_VERSION = "v2"


def serialize_handoff(
    handoff: EdgeCloudLayerHandoff,
    fmt: HandoffSerializationFormat = HandoffSerializationFormat.NUMPY_ZEROCOPY,
) -> bytes:
    """序列化 handoff 为字节流

    NUMPY_ZEROCOPY 格式：
      [magic 20B][header_len 4B][header json][hidden_states bytes][partial_kv bytes]
    """
    if fmt != HandoffSerializationFormat.NUMPY_ZEROCOPY:
        raise NotImplementedError(f"format {fmt} not implemented yet")

    # header: schema + tensors 元数据
    header = {
        "schema_version": _SCHEMA_VERSION,
        "model_id": handoff.model_id,
        "finished_layer": handoff.finished_layer,
        "hidden_states_shape": list(handoff.hidden_states.shape),
        "hidden_states_dtype": str(handoff.hidden_states.dtype),
        "partial_kv_keys": list(handoff.partial_kv.keys()) if handoff.partial_kv else [],
        "partial_kv_meta": (
            {k: [list(v.shape), str(v.dtype)] for k, v in handoff.partial_kv.items()}
            if handoff.partial_kv else {}
        ),
        "layer_metadata_list": [
            asdict(m) if hasattr(m, "__dataclass_fields__") else dict(m)
            for m in handoff.layer_metadata_list
        ],
        "extra": handoff.extra,
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")

    # tensors: hidden_states + partial_kv 拼接
    hs_bytes = handoff.hidden_states.tobytes(order="C")
    pkv_bytes = b""
    if handoff.partial_kv:
        pkv_chunks = []
        for k in header["partial_kv_keys"]:
            pkv_chunks.append(handoff.partial_kv[k].tobytes(order="C"))
        pkv_bytes = b"".join(pkv_chunks)

    payload = hs_bytes + pkv_bytes
    return _MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + payload


def deserialize_handoff(data: bytes) -> EdgeCloudLayerHandoff:
    """反序列化字节流为 handoff"""
    if not data.startswith(_MAGIC):
        raise HandoffValidationError(f"magic mismatch: expected {_MAGIC!r}")
    offset = len(_MAGIC)
    (header_len,) = struct.unpack_from("<I", data, offset)
    offset += 4
    header = json.loads(data[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    schema_version = header.get("schema_version", "v1")
    if schema_version != _SCHEMA_VERSION:
        raise HandoffValidationError(
            f"schema_version mismatch: data={schema_version} runtime={_SCHEMA_VERSION}"
        )

    # hidden_states
    hs_shape = tuple(header["hidden_states_shape"])
    hs_dtype = np.dtype(header["hidden_states_dtype"])
    hs_nbytes = int(np.prod(hs_shape)) * hs_dtype.itemsize
    hidden_states = np.frombuffer(data, dtype=hs_dtype, count=int(np.prod(hs_shape)), offset=offset).reshape(hs_shape)
    offset += hs_nbytes

    # partial_kv
    partial_kv = None
    if header["partial_kv_keys"]:
        partial_kv = {}
        for k in header["partial_kv_keys"]:
            shape, dtype_str = header["partial_kv_meta"][k]
            shape = tuple(shape)
            dtype = np.dtype(dtype_str)
            count = int(np.prod(shape))
            arr = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape)
            offset += count * dtype.itemsize
            partial_kv[k] = arr

    # layer_metadata_list
    layer_metadata_list = []
    for m_dict in header.get("layer_metadata_list", []):
        layer_metadata_list.append(LayerMetadata(**m_dict))

    return EdgeCloudLayerHandoff(
        finished_layer=header["finished_layer"],
        hidden_states=hidden_states,
        partial_kv=partial_kv,
        layer_metadata_list=layer_metadata_list,
        model_id=header.get("model_id", ""),
        schema_version=schema_version,
        extra=header.get("extra", {}),
    )
