"""base.py — 验证器基础设施"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerificationStatus(str, Enum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class VerificationResult:
    capability: str
    status: VerificationStatus = VerificationStatus.PENDING
    evidence: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "status": self.status.value,
            "evidence": list(self.evidence),
            "metrics": dict(self.metrics),
            "error": self.error,
            "duration_ms": round(self.duration_ms, 3),
        }


class BaseVerifier:
    """所有验证器的基类"""

    capability: str = "base"

    def __init__(self, args: Any) -> None:
        self.args = args
        self.result = VerificationResult(capability=self.capability)

    def _start(self) -> float:
        self.result.evidence.append(f"[{self.capability}] verification start")
        return time.time()

    def _finish(self, start: float, status: VerificationStatus, error: Optional[str] = None) -> VerificationResult:
        self.result.status = status
        self.result.error = error
        self.result.duration_ms = (time.time() - start) * 1000.0
        self.result.evidence.append(
            f"[{self.capability}] verification {status.value} in {self.result.duration_ms:.2f} ms"
        )
        return self.result

    def _add_evidence(self, msg: str) -> None:
        self.result.evidence.append(msg)

    def _add_metric(self, key: str, value: Any) -> None:
        self.result.metrics[key] = value

    def verify(self) -> VerificationResult:  # pragma: no cover - abstract
        raise NotImplementedError
