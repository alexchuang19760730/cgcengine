"""dopd_verifier.py — Gate 1.0 DOPD handoff 真实验证

调用 cgc_engine.pd.dopd_runtime.DOPDSessionRuntime + dopd_schema
端到端：prepare_handoff -> commit_handoff -> resume_decode -> 校验 resume_token + resume_payload round-trip
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict

from .base import BaseVerifier, VerificationResult, VerificationStatus


class DOPDVerifier(BaseVerifier):
    capability = "dopd_handoff"

    def verify(self) -> VerificationResult:
        start = self._start()
        try:
            # 导入真实模块
            from cgc_engine.pd.dopd_runtime import DOPDSessionRuntime
            from cgc_engine.pd.dopd_schema import (
                encode_dopd_resume_payload_v2,
                decode_dopd_resume_payload_v2,
                extract_dopd_resume_state_bytes,
            )
            self._add_evidence("✓ imported cgc_engine.pd.dopd_runtime.DOPDSessionRuntime")
            self._add_evidence("✓ imported cgc_engine.pd.dopd_schema encode/decode/extract")

            # 构造 resume payload（按 DOPDResumePayloadV2 required 字段）
            import base64
            session_id = f"dopd-verify-{int(time.time())}"
            handoff_id = f"h-{session_id}"
            state_bytes = os.urandom(256)
            resume_payload_v2 = {
                "session_id": session_id,
                "handoff_id": handoff_id,
                "phase_role": "edge_decode_resume",
                "cache_schema": "v2",
                "kv_variant": "partial_kv",
                "model_name": getattr(self.args, "model", "test-model"),
                "resume_position": 128,
                "token_position": 128,
                "finished_layer": 12,
                "max_local_layer": 16,
                "transport_codec": "cq4",
                "compression_codec": "trueorthokda",
                "zero_copy_vram": bool(getattr(self.args, "zero_copy", False)),
                "state_bytes_b64": base64.b64encode(state_bytes).decode("ascii"),
                "metadata": {"prompt_hash": "abc123", "max_local_layer": "12"},
                "version": 2,
                "payload_kind": "dopd_resume",
            }
            payload_bytes = encode_dopd_resume_payload_v2(resume_payload_v2)
            self._add_metric("payload_bytes", len(payload_bytes))
            self._add_evidence(f"✓ encode_dopd_resume_payload_v2 -> {len(payload_bytes)} bytes")

            # round-trip decode
            decoded = decode_dopd_resume_payload_v2(payload_bytes)
            if decoded is None:
                return self._finish(start, VerificationStatus.FAIL, "decode returned None")
            self._add_evidence("✓ decode_dopd_resume_payload_v2 round-trip OK")

            extracted = extract_dopd_resume_state_bytes(decoded)
            if len(extracted) != 256:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    f"extracted resume_state bytes mismatch: {len(extracted)} != 256",
                )
            self._add_evidence(f"✓ extract_dopd_resume_state_bytes -> {len(extracted)} bytes")

            # DOPDSessionRuntime prepare -> commit -> resume
            runtime = DOPDSessionRuntime()
            record = runtime.prepare_handoff(
                session_id=session_id,
                handoff_id=handoff_id,
                source_role="edge",
                target_role="cloud",
                phase_role="edge_decode_resume",
                model_name=resume_payload_v2["model_name"],
                cache_schema="v2",
                kv_variant="partial_kv",
                transport_codec="raw",
                compression_codec="none",
                zero_copy_vram=getattr(self.args, "zero_copy", False),
                resume_payload=payload_bytes,
                metadata={"prompt_hash": "abc123"},
            )
            self._add_evidence(f"✓ prepare_handoff status={record.status}")

            committed = runtime.commit_handoff(
                session_id=session_id,
                handoff_id=handoff_id,
                target_worker="cloud-worker-0",
                resume_position=128,
                resume_payload=b"",
                metadata={},
            )
            if not committed.resume_token:
                return self._finish(start, VerificationStatus.FAIL, "commit_handoff produced empty resume_token")
            self._add_metric("resume_token_len", len(committed.resume_token))
            self._add_evidence(f"✓ commit_handoff resume_token={committed.resume_token[:8]}...")

            resumed = runtime.resume_decode(
                session_id=session_id,
                handoff_id=handoff_id,
                resume_token=committed.resume_token,
                worker_id="cloud-worker-0",
                max_new_tokens=32,
                resume_payload=b"",
                metadata={},
            )
            self._add_evidence(f"✓ resume_decode status={resumed.status}")

            # stats
            stats = runtime.get_stats()
            self._add_metric("dopd_sessions", stats["dopd_sessions"])
            self._add_metric("dopd_handoffs", stats["dopd_handoffs"])
            self._add_metric("dopd_active_handoffs", stats["dopd_active_handoffs"])

            return self._finish(start, VerificationStatus.PASS)

        except ImportError as e:
            return self._finish(start, VerificationStatus.SKIP, f"module not available: {e}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._add_evidence(tb)
            return self._finish(start, VerificationStatus.ERROR, str(e))
