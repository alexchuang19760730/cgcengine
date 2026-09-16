"""cgc-engine/pd — PD 分離協調服務 (Gemma4 → Qwen3.6 跨模型 MoT-h).

架構:
  Mac A (Gemma4 prefill)                Mac B (Qwen3.6 decode)
  ──────────────────────                ──────────────────────
  TurboFieldfare                        TurboFieldfare
    │ POST /v1/cgc/emit                    │ POST /v1/cgc/resume
    │ (prefill 30 層, emit 末層 hidden)    │ (recv 翻譯後 hidden,
    ▼                                     │  Wk/Wv 還原 KV cache, decode)
  hidden_src [seq, 2816]                hidden_tgt [seq, 2048]
    │                                     ▲
    └──► coordinator.py ──► MoT-h ────────┘
         (FastAPI:9000)    (2816→2048)

整合模塊:
  - master_slave_sync: Master/Slave 代碼同步（slave 定時對齊 master）
  - compute_sharing: 端端/端雲算力共享框架
  - pd_config: PD 統一配置（服務端/邊緣/同步/算力共享）
"""
from .protocol import (
    HiddenStatePacket,
    PDMode,
    SourceModel,
    TargetModel,
    EmitRequest,
    EmitResponse,
    ResumeRequest,
)

# 整合模塊（可選導入，依賴額外包時不影響核心 PD 功能）
try:
    from .pd_config import PDConfig
    _PD_CONFIG_AVAILABLE = True
except ImportError:
    _PD_CONFIG_AVAILABLE = False

try:
    from .master_slave_sync import (
        CodeSyncer, SyncConfig, SyncResult, SyncRole, SyncMethod,
        start_master_server,
    )
    _SYNC_AVAILABLE = True
except ImportError:
    _SYNC_AVAILABLE = False

try:
    from .compute_sharing import (
        ComputeScheduler, ComputeNode, ComputeTask,
        TaskType, NodeRole, SharingPolicy, TaskStatus,
        GPUInfo, ModelInstance, ScheduleResult,
        start_compute_node,
    )
    _COMPUTE_SHARING_AVAILABLE = True
except ImportError:
    _COMPUTE_SHARING_AVAILABLE = False

__all__ = [
    # 核心 PD
    "HiddenStatePacket",
    "PDMode",
    "SourceModel",
    "TargetModel",
    "EmitRequest",
    "EmitResponse",
    "ResumeRequest",
    # 統一配置
    "PDConfig",
    # Master/Slave 同步
    "CodeSyncer",
    "SyncConfig",
    "SyncResult",
    "SyncRole",
    "SyncMethod",
    "start_master_server",
    # 算力共享
    "ComputeScheduler",
    "ComputeNode",
    "ComputeTask",
    "TaskType",
    "NodeRole",
    "SharingPolicy",
    "TaskStatus",
    "GPUInfo",
    "ModelInstance",
    "ScheduleResult",
    "start_compute_node",
    # 可用性標記
    "_PD_CONFIG_AVAILABLE",
    "_SYNC_AVAILABLE",
    "_COMPUTE_SHARING_AVAILABLE",
]
