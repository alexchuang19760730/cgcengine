"""Backend.CGC.edge_moe_transport — 端云一层一层张量传输运行时

本模块实现 Gate 2.0 端云分层传输路径的运行时组件：
  - transport_contract: 端→云 hidden_states + partial_kv 正式 ABI 契约
  - cq4_session: CQ4 端云协议承载层
  - nfsordma_transport: NFSoRDMA 张量传输实现

Scope: 端↔云跨节点层粒度张量传输，与云内 DeepEP EP ElasticBuffer 路径明确区分。
       云内 EP 路径见 cloud_internal_deepep_ep_moe_elastic_buffer 能力。
"""

from .transport_contract import (
    EdgeCloudLayerHandoff,
    HandoffSerializationFormat,
    HandoffValidationError,
    deserialize_handoff,
    serialize_handoff,
)

__all__ = [
    "EdgeCloudLayerHandoff",
    "HandoffSerializationFormat",
    "HandoffValidationError",
    "serialize_handoff",
    "deserialize_handoff",
]
