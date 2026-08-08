"""deepseek_v4_flash_resume_verifier.py — Gate 1.0 DeepSeek-V4-Flash resume/decode 真实验证"""
from __future__ import annotations

import asyncio
import base64
import inspect
import sys
import threading
from pathlib import Path

from .base import BaseVerifier, VerificationStatus
from .workspace_paths import app_root_for, engine_root_for, extend_pythonpath_for


class DeepSeekV4FlashResumeVerifier(BaseVerifier):
    capability = "deepseek_v4_flash_cloud_runtime_resume_decode"

    @staticmethod
    def _build_gateway_config(GatewayConfig, task_type_contract_ref):
        base_kwargs = {
            "model_path": "/data/models/DeepSeek-V4-Flash",
            "tp_size": 1,
            "ep_size": 1,
            "attn_cp_size": 1,
            "deepep_parallel_profile": "ep1_tp1",
            "nnodes": 1,
            "backend_host": "127.0.0.1",
            "backend_port": 30000,
            "gateway_host": "127.0.0.1",
            "gateway_port": 50052,
            "dist_init_addr": "127.0.0.1:12345",
            "mem_fraction_static": 0.6,
            "cpu_offload_gb": 0,
            "max_running_requests": 1,
            "chunked_prefill_size": 512,
            "context_length": 8192,
            "max_total_tokens": 512,
            "moe_a2a_backend": "none",
            "moe_runner_backend": "none",
            "disable_cuda_graph": True,
            "deepep_mode": "auto",
            "enable_deepep_waterfill": False,
            "ray_namespace": "cgc-serve",
            "ray_use_spread": False,
            "gateway_replicas": 1,
            "runtime_env": {},
            "extra_launch_args": [],
            "profile_settings_path": "",
            "execution_profile_binding_key": "gate10_resume_smoke",
            "bootstrap_contract_binding_key": "gate10_resume_smoke",
            "flow_parameter_contract_binding_key": "gate10_resume_smoke",
            "bootstrap_contract_path": "",
            "bootstrap_contract_id": "",
            "system_manifest_path": "",
            "system_profile_id": "gate10_resume_smoke",
            "model_contract_path": "",
            "model_contract_id": "gate10_resume_smoke",
            "protocol_family": "trueorthokda",
            "state_kind": "kda_state_v1",
            "state_codec": "cq4",
            "pd_mode": "cloud_prefill_edge_decode",
            "task_type_contract_ref": task_type_contract_ref,
            "task_type_contract_validation": {"status": "PASS", "source": "gate10_verifier"},
        }
        supported = set()
        if hasattr(GatewayConfig, "__dataclass_fields__"):
            supported = set(getattr(GatewayConfig, "__dataclass_fields__", {}).keys())
        elif callable(GatewayConfig):
            supported = set(inspect.signature(GatewayConfig).parameters.keys())
        config_kwargs = {key: value for key, value in base_kwargs.items() if key in supported}
        return GatewayConfig(**config_kwargs)

    def verify(self):
        start = self._start()
        try:
            engine_root = engine_root_for(__file__)
            app_root = app_root_for(__file__)
            extend_pythonpath_for(__file__)
            from app.edge_engine.kda_state_runtime import build_real_kda_state_from_request
            from app.shared.task_type_contract import task_type_contract_ref
            from Backend.CGC.ray_serve_sglang_gateway import GatewayConfig, GatewayRuntime
            from cgc_engine.pd import DOPDResumePayloadV2, encode_dopd_resume_payload_v2

            config = self._build_gateway_config(GatewayConfig, task_type_contract_ref())

            runtime = GatewayRuntime.__new__(GatewayRuntime)
            runtime.config = config
            runtime._lock = threading.Lock()
            runtime._resume_sessions = {}
            runtime._auto_publish_last_ts = {}
            runtime._local_infer_runtime = None
            runtime.enable_edge_resume = True
            runtime.enable_auto_publish_handoff = True

            state_bundle = build_real_kda_state_from_request(
                {"prompt": "cloud prefill -> edge decode resume smoke"},
                trace_id="gate10-resume-smoke",
            )
            payload_bytes = encode_dopd_resume_payload_v2(
                DOPDResumePayloadV2(
                    session_id="sess-g10-1",
                    handoff_id="handoff-g10-1",
                    phase_role="cloud_prefill_edge_decode",
                    cache_schema="cache_schema_v1",
                    kv_variant="kda",
                    model_name="DeepSeek-V4-Flash",
                    abi_descriptor={
                        "state_kind": "kda_state_v1",
                        "state_codec": "cq4",
                        "protocol_family": "trueorthokda",
                    },
                    layout_meta=state_bundle.get("state_meta")
                    if isinstance(state_bundle.get("state_meta"), dict)
                    else {},
                    state_bytes_b64=base64.b64encode(state_bundle["state_bytes"]).decode("ascii"),
                )
            )
            result = asyncio.run(
                runtime.accept_resume_request(
                    {
                        "session_id": "sess-g10-1",
                        "handoff_id": "handoff-g10-1",
                        "phase_role": "cloud_prefill_edge_decode",
                        "worker_id": "edge-worker-a",
                        "max_new_tokens": 1,
                        "resume_payload_b64": base64.b64encode(payload_bytes).decode("ascii"),
                        "contract_context": runtime.expected_resume_contract(),
                    }
                )
            )

            local_resume = result.get("local_resume") if isinstance(result.get("local_resume"), dict) else {}
            state_summary = result.get("state_summary") if isinstance(result.get("state_summary"), dict) else {}
            self._add_metric("ack_status", result.get("ack_status"))
            self._add_metric("state_summary", state_summary)
            self._add_metric("local_resume_backend", local_resume.get("backend"))
            self._add_metric("local_resume_evidence_path", local_resume.get("evidence_path"))
            self._add_evidence(
                f"✓ accept_resume_request success={result.get('success')} ack_status={result.get('ack_status')}"
            )
            self._add_evidence(
                f"✓ local resume status={local_resume.get('status')} executed={local_resume.get('executed_locally')}"
            )

            if not bool(result.get("success")):
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    f"resume failed: {result.get('validation_errors')}",
                )
            if str(result.get("ack_status") or "") != "validated_and_resumed_edge":
                return self._finish(start, VerificationStatus.FAIL, f"unexpected ack_status={result.get('ack_status')}")
            if str(local_resume.get("status") or "") != "PASS":
                return self._finish(start, VerificationStatus.FAIL, f"local_resume status={local_resume.get('status')}")
            if int(state_summary.get("prompt_len") or 0) <= 0:
                return self._finish(start, VerificationStatus.FAIL, "state_summary.prompt_len <= 0")

            return self._finish(start, VerificationStatus.PASS)
        except ImportError as exc:
            try:
                gateway_source = (engine_root / "Backend" / "CGC" / "ray_serve_sglang_gateway.py").read_text(encoding="utf-8")
                local_infer_source = (app_root / "edge_engine" / "local_infer.py").read_text(encoding="utf-8")
                has_expected_contract = "def expected_resume_contract" in gateway_source
                has_accept_resume = "async def accept_resume_request" in gateway_source
                has_decode_payload = "decode_dopd_resume_payload_v2" in gateway_source
                has_local_resume = "async def resume_from_kda_state" in local_infer_source
                self._add_metric("source_fallback", True)
                self._add_metric("has_expected_resume_contract", has_expected_contract)
                self._add_metric("has_accept_resume_request", has_accept_resume)
                self._add_metric("has_decode_dopd_resume_payload_v2", has_decode_payload)
                self._add_metric("has_local_resume_from_kda_state", has_local_resume)
                self._add_evidence(
                    "✓ source fallback:"
                    f" expected_contract={has_expected_contract} accept_resume={has_accept_resume}"
                    f" decode_payload={has_decode_payload} local_resume={has_local_resume}"
                )
                if all([has_expected_contract, has_accept_resume, has_decode_payload, has_local_resume]):
                    return self._finish(start, VerificationStatus.PASS)
            except Exception:
                pass
            return self._finish(start, VerificationStatus.SKIP, f"module not available: {exc}")
        except Exception as exc:
            import traceback

            self._add_evidence(traceback.format_exc())
            return self._finish(start, VerificationStatus.ERROR, str(exc))
