"""cq4_session.py — CQ4 端云协议承载层

CQ4 (Cloud-Quantum-Queue-4) 是 Gate 2.0 端云切换协议的承载层，负责：
  - 端↔云层接续 handoff 的可靠传输会话管理
  - 4 段 QoS 优先级（control / hidden_states / kv / telemetry）
  - 与 NFSoRDMA / HTTP fallback 的统一接口

对应能力 cq4_transport_plane。
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .transport_contract import EdgeCloudLayerHandoff, deserialize_handoff, serialize_handoff


class CQ4QoSClass(Enum):
    """CQ4 四段 QoS 优先级"""
    CONTROL = 0       # 控制消息（highest priority）
    HIDDEN_STATES = 1 # hidden_states 张量
    KV_CACHE = 2      # partial_kv 张量
    TELEMETRY = 3     # 遥测/trace


class CQ4SessionState(Enum):
    IDLE = "idle"
    HANDSHAKING = "handshaking"
    ESTABLISHED = "established"
    CLOSED = "closed"


@dataclass
class CQ4SessionConfig:
    """CQ4 会话配置"""
    cloud_endpoint: str = "http://127.0.0.1:7777"  # 云侧 cgc_api_server
    transport_backend: str = "http"  # "http" | "nfsordma"
    timeout_s: float = 5.0
    max_retries: int = 3
    session_id: str = ""


class CQ4Session:
    """CQ4 端云会话

    一个会话对应一次端→云层接续请求。会话内可发送多段 QoS 分级消息。
    """

    def __init__(self, config: Optional[CQ4SessionConfig] = None):
        self.config = config or CQ4SessionConfig()
        self.state = CQ4SessionState.IDLE
        self._lock = threading.Lock()
        self._sent_bytes = 0
        self._sent_messages = 0
        self._last_error: Optional[str] = None

    def open(self) -> None:
        """打开会话（握手）"""
        with self._lock:
            if self.state == CQ4SessionState.ESTABLISHED:
                return
            self.state = CQ4SessionState.HANDSHAKING
        # 握手：向云侧 /cgc/cq4/session/open 发送 POST
        try:
            self._http_post(
                f"{self.config.cloud_endpoint}/cgc/cq4/session/open",
                payload={"session_id": self.config.session_id, "transport": self.config.transport_backend},
                timeout=self.config.timeout_s,
            )
            with self._lock:
                self.state = CQ4SessionState.ESTABLISHED
        except Exception as e:
            with self._lock:
                self.state = CQ4SessionState.CLOSED
                self._last_error = str(e)
            raise

    def send_handoff(self, handoff: EdgeCloudLayerHandoff, qos: CQ4QoSClass = CQ4QoSClass.HIDDEN_STATES) -> Dict[str, Any]:
        """发送端→云层接续 handoff

        Returns:
            云侧响应（含 cloud_request_id, accepted_layer 等）
        """
        if self.state != CQ4SessionState.ESTABLISHED:
            self.open()

        payload = serialize_handoff(handoff)
        # 分段发送（简化版：单包；生产可按 QoS 拆包走不同 priority queue）
        resp = self._http_post(
            f"{self.config.cloud_endpoint}/cgc/cq4/handoff",
            payload=payload,
            timeout=self.config.timeout_s,
            headers={
                "Content-Type": "application/octet-stream",
                "X-CQ4-QoS": qos.value,
                "X-CQ4-Session": self.config.session_id,
                "X-CQ4-Schema": handoff.schema_version,
            },
        )
        with self._lock:
            self._sent_bytes += len(payload)
            self._sent_messages += 1
        return resp

    def close(self) -> None:
        with self._lock:
            self.state = CQ4SessionState.CLOSED

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state.value,
                "sent_bytes": self._sent_bytes,
                "sent_messages": self._sent_messages,
                "last_error": self._last_error,
            }

    def _http_post(self, url: str, payload: Any, timeout: float, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """HTTP POST 实现（带重试）"""
        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                if isinstance(payload, (dict, list)):
                    data = json.dumps(payload).encode("utf-8")
                    hdrs = {"Content-Type": "application/json"}
                elif isinstance(payload, (bytes, bytearray)):
                    data = bytes(payload)
                    hdrs = {}
                else:
                    raise TypeError(f"unsupported payload type: {type(payload)}")
                if headers:
                    hdrs.update(headers)
                req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError:
                        return {"raw": body}
            except Exception as e:
                last_err = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
        raise ConnectionError(f"CQ4 POST {url} failed after {self.config.max_retries} retries: {last_err}")


def receive_handoff_from_request_body(body: bytes) -> EdgeCloudLayerHandoff:
    """云侧从 HTTP request body 反序列化 handoff

    供 cgc_api_server 的 /cgc/cq4/handoff handler 调用。
    """
    return deserialize_handoff(body)
