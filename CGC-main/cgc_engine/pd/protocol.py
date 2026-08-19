#!/usr/bin/env python3
"""cgc-engine/pd — PD 分離 + MoT-h 跨模型翻譯的 wire protocol.

此文件定義 TurboFieldfare (Swift) 端需要實現的 HTTP 接口契約。
Python coordinator 通過這些接口與 Mac A (Gemma4 prefill) 和 Mac B (Qwen3.6 decode) 溝通。

架構 (Gemma4 → Qwen3.6 跨模型 MoT-h)
======================================

  Mac A (Gemma4 26B-A4B)                Mac B (Qwen3.6-35B-A3B)
  ──────────────────────                ──────────────────────
  TurboFieldfare                        TurboFieldfare
  hidden=2816, layers=30                hidden=2048, layers=40
    │                                     ▲
    │ POST /v1/cgc/emit                    │ POST /v1/cgc/resume
    │ (prefill 30 層, 不 decode)           │ (recv 翻譯後 hidden,
    │ return hidden [seq, 2816]            │  Wk/Wv 還原 KV cache,
    ▼                                     │  decode 輸出 token)
  hidden_src                             hidden_tgt
  [seq_len, 2816]                        [seq_len, 2048]
    │                                     ▲
    │  ┌──────────────────────────────┐   │
    └─►│  Python coordinator          │───┘
       │  MoT-h 翻譯 (2816 → 2048)    │
       │  + 通道映射 (30層 → 40層)     │
       │  + Context Replay (KV 還原)  │
       └──────────────────────────────┘

Swift 端需實現的兩個新 endpoint
================================

1. POST /v1/cgc/emit  (Mac A, Gemma4)
   - 輸入: {prompt, max_tokens=0, ...}
   - 行為: prefill 全部 30 層, 不 decode
   - 輸出: {hidden_state, seq_len, model_info, request_id}

2. POST /v1/cgc/resume (Mac B, Qwen3.6)
   - 輸入: {hidden_state, seq_len, max_tokens, temperature, ...}
   - 行為: 用 hidden_state + 原生 Wk/Wv 還原 KV cache → decode
   - 輸出: SSE stream (OpenAI chunk 格式)

hidden_state wire format
========================
傳輸格式: base64 encoded float32 (little-endian)
  - shape: [seq_len, hidden_dim]
  - dtype: float32 (小端)
  - encoding: base64(raw_bytes)
  - JSON 中作為字符串傳傳輸

範例:
  hidden_state_b64 = base64.b64encode(tensor.numpy().astype('<f4').tobytes()).decode()
  # 或從 base64 還原:
  raw = base64.b64decode(hidden_state_b64)
  tensor = torch.frombuffer(raw, dtype=torch.float32).reshape(seq_len, hidden_dim)
"""
from __future__ import annotations

import base64
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# 模型定義
# ---------------------------------------------------------------------------
class SourceModel(enum.Enum):
    """Prefill 側模型 (雲端)."""
    GEMMA4_26B_A4B = "gemma4-26b-a4b"
    DEEPSEEK_V4_FLASH = "dsv4-flash"

    @property
    def hidden_size(self) -> int:
        return {
            SourceModel.GEMMA4_26B_A4B: 2816,
            SourceModel.DEEPSEEK_V4_FLASH: 7168,
        }[self]

    @property
    def num_layers(self) -> int:
        return {
            SourceModel.GEMMA4_26B_A4B: 30,
            SourceModel.DEEPSEEK_V4_FLASH: 61,
        }[self]


class TargetModel(enum.Enum):
    """Decode 側模型 (端側)."""
    QWEN36_35B_A3B = "qwen36-35b-a3b"

    @property
    def hidden_size(self) -> int:
        return {TargetModel.QWEN36_35B_A3B: 2048}[self]

    @property
    def num_layers(self) -> int:
        return {TargetModel.QWEN36_35B_A3B: 40}[self]


TARGET_MODELS = {m.value: m for m in TargetModel}


# ---------------------------------------------------------------------------
# PD 模式
# ---------------------------------------------------------------------------
class PDMode(enum.Enum):
    """PD 分離模式."""
    PASSTHROUGH = "passthrough"  # 同模型 hidden state 直傳 (Phase 1 驗證用)
    TRANSLATE = "translate"      # 跨模型 MoT-h 翻譯 (Phase 2, Gemma4→Qwen3.6)
    COLLECT = "collect"          # 只 prefill emit, 採集訓練對 (不 decode)


# ---------------------------------------------------------------------------
# Wire format: hidden state 序列化/反序列化
# ---------------------------------------------------------------------------
def encode_hidden_state(tensor: torch.Tensor) -> str:
    """將 hidden state tensor 編碼為 base64 字符串.

    Args:
        tensor: [seq_len, hidden_dim] float32 tensor

    Returns:
        base64 encoded string (float32 little-endian)
    """
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    # 確保連續內存 + little-endian float32
    raw = tensor.contiguous().numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


def decode_hidden_state(
    b64_str: str, seq_len: int, hidden_dim: int
) -> torch.Tensor:
    """從 base64 字符串解碼 hidden state tensor.

    Args:
        b64_str: base64 encoded float32 bytes
        seq_len: 序列長度
        hidden_dim: 隱藏維度

    Returns:
        [seq_len, hidden_dim] float32 tensor
    """
    raw = base64.b64decode(b64_str)
    expected = seq_len * hidden_dim * 4  # float32 = 4 bytes
    if len(raw) != expected:
        raise ValueError(
            f"hidden state bytes mismatch: got {len(raw)}, expected {expected} "
            f"(seq_len={seq_len}, hidden_dim={hidden_dim})"
        )
    return torch.frombuffer(raw, dtype=torch.float32).reshape(seq_len, hidden_dim).clone()


# ---------------------------------------------------------------------------
# 數據結構
# ---------------------------------------------------------------------------
@dataclass
class ModelInfo:
    """模型元信息."""
    model_id: str           # e.g. "gemma4-26b-a4b"
    hidden_size: int        # e.g. 2816
    num_layers: int         # e.g. 30
    dtype: str = "float32"  # wire format 固定 float32


@dataclass
class HiddenStatePacket:
    """hidden state 傳輸封包 — emit/resume 共用.

    這是整個 PD 管線的核心數據載體。
    """
    hidden_state_b64: str       # base64 encoded float32 [seq_len, hidden_dim]
    seq_len: int                # 序列長度
    hidden_dim: int             # 隱藏維度 (emit 端 = source, resume 端 = target)
    finished_layer: int         # 已完成的層數 (emit 端 = 30, 全 prefill)
    model_info: ModelInfo       # 模型元信息
    request_id: str = ""        # 請求 ID (用於跨節點 step 對齊)
    timestamp: float = field(default_factory=time.time)

    def to_tensor(self) -> torch.Tensor:
        """解碼為 tensor [seq_len, hidden_dim]."""
        return decode_hidden_state(self.hidden_state_b64, self.seq_len, self.hidden_dim)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        model_info: ModelInfo,
        finished_layer: int,
        request_id: str = "",
    ) -> "HiddenStatePacket":
        """從 tensor 構造封包."""
        seq_len, hidden_dim = tensor.shape
        return cls(
            hidden_state_b64=encode_hidden_state(tensor),
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            finished_layer=finished_layer,
            model_info=model_info,
            request_id=request_id or str(uuid.uuid4())[:8],
        )

    def to_dict(self) -> dict:
        """序列化為 JSON 可傳輸的 dict."""
        return {
            "hidden_state_b64": self.hidden_state_b64,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "finished_layer": self.finished_layer,
            "model_info": {
                "model_id": self.model_info.model_id,
                "hidden_size": self.model_info.hidden_size,
                "num_layers": self.model_info.num_layers,
                "dtype": self.model_info.dtype,
            },
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HiddenStatePacket":
        """從 dict 反序列化."""
        return cls(
            hidden_state_b64=d["hidden_state_b64"],
            seq_len=d["seq_len"],
            hidden_dim=d["hidden_dim"],
            finished_layer=d["finished_layer"],
            model_info=ModelInfo(
                model_id=d["model_info"]["model_id"],
                hidden_size=d["model_info"]["hidden_size"],
                num_layers=d["model_info"]["num_layers"],
                dtype=d["model_info"].get("dtype", "float32"),
            ),
            request_id=d.get("request_id", ""),
            timestamp=d.get("timestamp", time.time()),
        )


# ---------------------------------------------------------------------------
# Emit Request/Response (Mac A — Gemma4 prefill)
# ---------------------------------------------------------------------------
@dataclass
class EmitRequest:
    """POST /v1/cgc/emit 請求體.

    Swift 端收到後:
      1. tokenize prompt
      2. prefill 全部層 (Gemma4 = 30 層)
      3. 取末層 hidden state [seq_len, 2816]
      4. 返回 EmitResponse
    """
    prompt: str
    max_tokens: int = 0         # emit 模式不 decode, 固定 0
    temperature: float = 0.0
    request_id: str = ""

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "request_id": self.request_id or str(uuid.uuid4())[:8],
        }


@dataclass
class EmitResponse:
    """POST /v1/cgc/emit 響應體."""
    packet: HiddenStatePacket    # 末層 hidden state
    prefill_latency_ms: float    # prefill 耗時
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "packet": self.packet.to_dict(),
            "prefill_latency_ms": self.prefill_latency_ms,
            "success": self.success,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmitResponse":
        return cls(
            packet=HiddenStatePacket.from_dict(d["packet"]),
            prefill_latency_ms=d["prefill_latency_ms"],
            success=d.get("success", True),
            error=d.get("error", ""),
        )


# ---------------------------------------------------------------------------
# Resume Request/Response (Mac B — Qwen3.6 decode)
# ---------------------------------------------------------------------------
@dataclass
class ResumeRequest:
    """POST /v1/cgc/resume 請求體.

    Swift 端收到後:
      1. 解碼 hidden_state_b64 → tensor [seq_len, 2048]
      2. 用 Qwen3.6 原生 Wk/Wv 還原 KV cache (40 層)
         - 通道層: 直接用翻譯後 hidden @ Wk/Wv
         - 非通道層: Context Replay 補全 (或 MVP 先全用同一個 hidden)
      3. decode: 以 KV cache 為上下文, 自回歸生成 max_tokens 個 token
      4. SSE stream 返回 token (OpenAI chunk 格式)
    """
    hidden_state_b64: str       # 翻譯後的 Qwen3.6 hidden state
    seq_len: int
    hidden_dim: int             # = 2048 (Qwen3.6)
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True
    request_id: str = ""

    def to_dict(self) -> dict:
        return {
            "hidden_state_b64": self.hidden_state_b64,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "request_id": self.request_id or str(uuid.uuid4())[:8],
        }


# ---------------------------------------------------------------------------
# Swift 端接口契約摘要 (給 TurboFieldfare 開發者)
# ---------------------------------------------------------------------------
SWIFT_ENDPOINT_CONTRACT = """
=== TurboFieldfare Swift 端需實現的接口契約 ===

1. POST /v1/cgc/emit  (Mac A — Gemma4 26B-A4B prefill)
   ──────────────────────────────────────────────────
   Request JSON:
   {
     "prompt": "Write a Python function to...",
     "max_tokens": 0,
     "temperature": 0.0,
     "request_id": "abc12345"
   }

   Response JSON:
   {
     "packet": {
       "hidden_state_b64": "<base64 string>",   // float32 [seq_len, 2816]
       "seq_len": 128,
       "hidden_dim": 2816,
       "finished_layer": 30,                     // 全 30 層 prefill
       "model_info": {
         "model_id": "gemma4-26b-a4b",
         "hidden_size": 2816,
         "num_layers": 30,
         "dtype": "float32"
       },
       "request_id": "abc12345",
       "timestamp": 1234567890.123
     },
     "prefill_latency_ms": 45.2,
     "success": true,
     "error": ""
   }

   實現要點:
   - TurboFieldfare 已有 lastHiddenState: [Float] (RealForwardRunner.swift:1141)
   - 只需 prefill 不 decode, 取末層 hidden state
   - 將 [Float] 轉為 [UInt8] (float32 little-endian) → base64 encode
   - Swift: Data(bytes: hiddenState.withUnsafeBufferPointer { Data(buffer: $0) })
   - 確保 float 是 little-endian (Apple Silicon 原生即小端)


2. POST /v1/cgc/resume  (Mac B — Qwen3.6-35B-A3B decode)
   ──────────────────────────────────────────────────
   Request JSON:
   {
     "hidden_state_b64": "<base64 string>",   // float32 [seq_len, 2048]
     "seq_len": 128,
     "hidden_dim": 2048,
     "max_tokens": 512,
     "temperature": 0.7,
     "top_p": 0.9,
     "stream": true,
     "request_id": "abc12345"
   }

   Response: SSE stream (OpenAI chat completions chunk 格式)
   data: {"choices":[{"delta":{"content":"def"}}]}
   data: {"choices":[{"delta":{"content":" fibonacci"}}]}
   ...
   data: [DONE]

   實現要點:
   - 解碼 base64 → [Float] → Metal buffer
   - 用 Qwen3.6 原生 Wk/Wv 還原 KV cache:
     for layer in 0..<40:
       K[layer] = hidden_state @ Wk[layer]   // [seq_len, kv_dim]
       V[layer] = hidden_state @ Wv[layer]   // [seq_len, kv_dim]
   - (MVP: 所有層用同一個 hidden_state; 後續: 通道映射 + Context Replay)
   - 設置 KV cache 後, 從最後一個 token 開始 decode
   - SSE stream 輸出 (復用現有 /v1/chat/completions 的 stream 邏輯)


3. 環境變量 (Swift 端讀取)
   CGC_PD_ROLE=emit|resume    # 角色切換
   CGC_PD_MODEL=gemma4|qwen36 # 模型選擇
"""
