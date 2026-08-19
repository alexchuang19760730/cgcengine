"""ray_engine_dual_host_verifier.py — Gate 2.0 Ray dual-host service topology verifier"""
from __future__ import annotations

import sys
from pathlib import Path

from .base import BaseVerifier, VerificationStatus
from .workspace_paths import app_root_for, engine_root_for, extend_pythonpath_for


engine_root = engine_root_for(__file__)
app_root = app_root_for(__file__)
extend_pythonpath_for(__file__)


class RayEngineDualHostVerifier(BaseVerifier):
    capability = "ray_engine_dual_host_service_topology"

    def verify(self):
        start = self._start()
        try:
            pipeline_source = (engine_root / "cgc_engine" / "pipeline.py").read_text(encoding="utf-8")
            m76_source = (engine_root / "cgc_engine" / "product" / "m76_gate.py").read_text(encoding="utf-8")
            api_source = (app_root / "servers" / "cgc_api_server.py").read_text(encoding="utf-8")
            gateway_source = (engine_root / "Backend" / "CGC" / "ray_serve_sglang_gateway.py").read_text(encoding="utf-8")
            compat_path = engine_root / "cgc_engine" / "product" / "m75_api_compat_gate.py"
            compat_source = compat_path.read_text(encoding="utf-8") if compat_path.exists() else ""

            service_topology_backend = "ray_cluster_dual_host" if "ray_cluster_dual_host" in pipeline_source and "ray_cluster_dual_host" in m76_source else ""
            pd_mode = "cloud_prefill_edge_decode" if "cloud_prefill_edge_decode" in m76_source else ""
            protocol_family = "trueorthokda" if "protocol_family" in m76_source and "trueorthokda" in m76_source else ""
            has_fastapi = "FastAPI" in api_source
            has_streaming = "StreamingResponse" in api_source
            has_local_resume = "EdgeLocalInferenceRuntime" in api_source
            has_trigger = "async def trigger_cgc_prefill" in api_source
            has_gateway_runtime = "class GatewayRuntime" in gateway_source
            has_expected_contract = "def expected_resume_contract" in gateway_source
            has_gateway_phrase = "Ray Serve + SGLang gateway" in m76_source or "Ray Serve + SGLang gateway" in compat_source or "Ray Serve + SGLang gateway" in gateway_source

            self._add_metric("service_topology_backend", service_topology_backend)
            self._add_metric("pd_mode", pd_mode)
            self._add_metric("protocol_family", protocol_family)
            self._add_metric("has_fastapi", has_fastapi)
            self._add_metric("has_streaming_response", has_streaming)
            self._add_metric("has_local_resume_runtime", has_local_resume)
            self._add_metric("has_trigger_cgc_prefill", has_trigger)
            self._add_metric("has_gateway_runtime", has_gateway_runtime)
            self._add_metric("has_expected_resume_contract", has_expected_contract)
            self._add_metric("has_gateway_phrase", has_gateway_phrase)
            self._add_metric("compat_gate_present", compat_path.exists())

            self._add_evidence(
                "✓ runtime protocol contract:"
                f" topology={service_topology_backend} pd_mode={pd_mode} protocol={protocol_family}"
            )
            self._add_evidence(
                "✓ cloud service entrypoints:"
                f" fastapi={has_fastapi} streaming={has_streaming}"
                f" local_resume={has_local_resume} trigger_prefill={has_trigger}"
            )
            self._add_evidence(
                "✓ gateway runtime surface:"
                f" GatewayRuntime={has_gateway_runtime} expected_resume_contract={has_expected_contract}"
                f" gateway_phrase={has_gateway_phrase}"
            )

            if service_topology_backend != "ray_cluster_dual_host":
                return self._finish(start, VerificationStatus.FAIL, "service_topology_backend != ray_cluster_dual_host")
            if pd_mode != "cloud_prefill_edge_decode":
                return self._finish(start, VerificationStatus.FAIL, f"pd_mode={pd_mode}")
            if protocol_family != "trueorthokda":
                return self._finish(start, VerificationStatus.FAIL, f"protocol_family={protocol_family}")
            if not all([has_fastapi, has_streaming, has_local_resume, has_trigger, has_gateway_runtime, has_expected_contract, has_gateway_phrase]):
                return self._finish(start, VerificationStatus.FAIL, "dual-host service topology surface incomplete")

            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.FAIL, str(exc))
