"""trueorthokda_verifier.py — Gate 1.0 TrueOrthoKDA + CQ4 压缩真实验证"""
from __future__ import annotations

from .base import BaseVerifier, VerificationStatus
from .workspace_paths import extend_pythonpath_for


class TrueOrthoKDAVerifier(BaseVerifier):
    capability = "trueorthokda_kv_cq4_compression"

    def verify(self):
        start = self._start()
        try:
            extend_pythonpath_for(__file__)
            from app.edge_engine.kda_state_runtime import (
                build_real_kda_state_from_request,
                inspect_kda_state_bytes,
                resume_one_token_from_kda_state,
            )

            state_bundle = build_real_kda_state_from_request(
                {"prompt": "trueorthokda gate10 smoke"},
                trace_id="gate10-trueorthokda-smoke",
            )
            state_summary = inspect_kda_state_bytes(
                state_kind=str(state_bundle.get("state_kind") or ""),
                state_codec=str(state_bundle.get("state_codec") or ""),
                state_bytes=state_bundle["state_bytes"],
            )
            resume = resume_one_token_from_kda_state(
                state_kind=str(state_bundle.get("state_kind") or ""),
                state_codec=str(state_bundle.get("state_codec") or ""),
                state_bytes=state_bundle["state_bytes"],
                trace_id="gate10-trueorthokda-smoke-resume",
            )
            zero_copy = (
                resume.get("zero_copy_runtime")
                if isinstance(resume.get("zero_copy_runtime"), dict)
                else {}
            )

            self._add_metric("state_kind", state_bundle.get("state_kind"))
            self._add_metric("state_codec", state_bundle.get("state_codec"))
            self._add_metric("compression_ratio", state_summary.get("compression_ratio"))
            self._add_metric("raw_state_bytes", state_summary.get("raw_state_bytes"))
            self._add_metric("compressed_state_bytes", state_summary.get("compressed_state_bytes"))
            self._add_metric("resume_tensor_device", resume.get("resume_tensor_device"))
            self._add_metric("zero_copy_runtime", zero_copy)
            self._add_metric("state_summary", state_summary)
            self._add_evidence(
                "✓ built real KDA state:"
                f" kind={state_bundle.get('state_kind')} codec={state_bundle.get('state_codec')}"
                f" compression_ratio={state_summary.get('compression_ratio')}"
            )
            self._add_evidence(
                "✓ resume decode executed:"
                f" device={resume.get('resume_tensor_device')}"
                f" zero_copy={zero_copy.get('device_resume_consumed')}"
            )

            if str(state_bundle.get("state_kind") or "") != "kda_state_v1":
                return self._finish(start, VerificationStatus.FAIL, "state_kind is not kda_state_v1")
            if str(state_bundle.get("state_codec") or "") != "cq4":
                return self._finish(start, VerificationStatus.FAIL, "state codec is not cq4")
            if int(state_summary.get("prompt_len") or 0) <= 0:
                return self._finish(start, VerificationStatus.FAIL, "prompt_len <= 0")
            if float(state_summary.get("compression_ratio") or 1.0) >= 1.0:
                return self._finish(start, VerificationStatus.FAIL, "compression_ratio >= 1.0")
            if not bool(resume.get("resume_decode_executed")):
                return self._finish(start, VerificationStatus.FAIL, "resume_decode_executed=false")
            if not bool(zero_copy.get("device_resume_consumed")):
                return self._finish(start, VerificationStatus.FAIL, "device_resume_consumed=false")

            return self._finish(start, VerificationStatus.PASS)
        except ImportError as exc:
            return self._finish(start, VerificationStatus.SKIP, f"module not available: {exc}")
        except Exception as exc:
            import traceback

            self._add_evidence(traceback.format_exc())
            return self._finish(start, VerificationStatus.ERROR, str(exc))
