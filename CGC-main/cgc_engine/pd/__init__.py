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

__all__ = [
    "HiddenStatePacket",
    "PDMode",
    "SourceModel",
    "TargetModel",
    "EmitRequest",
    "EmitResponse",
    "ResumeRequest",
]
