"""Gate 2.2 DeepEP L20N capability verifiers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from .base import BaseVerifier, VerificationStatus
from .workspace_paths import app_root_for, engine_root_for


class _G22DeepEPL20NBaseVerifier(BaseVerifier):
    def _workspace_root(self) -> Path:
        return engine_root_for(__file__).parent

    def _engine_root(self) -> Path:
        return engine_root_for(__file__)

    def _app_root(self) -> Path:
        return app_root_for(__file__)

    def _read_engine_source(self, rel_path: str) -> str:
        return (self._engine_root() / rel_path).read_text(encoding="utf-8")

    def _read_app_source(self, rel_path: str) -> str:
        return (self._app_root() / rel_path).read_text(encoding="utf-8")

    def _read_example_json(self, filename: str) -> Dict[str, Any]:
        return json.loads(
            (
                self._engine_root()
                / "docs"
                / "technical_whitepapers"
                / "examples"
                / filename
            ).read_text(encoding="utf-8")
        )

    @staticmethod
    def _contains_all(source: str, markers: Iterable[str]) -> bool:
        return all(marker in source for marker in markers)

    def _load_dualnode_bundle(self) -> Dict[str, Dict[str, Any]]:
        return {
            "bootstrap": self._read_example_json(
                "dualnode_blackwell_deepep_ep16_tp1_runtime_bootstrap_contract.example.json"
            ),
            "system_manifest": self._read_example_json(
                "dualnode_deepseek_v4_flash_qwen35_dflash_system_manifest.example.json"
            ),
            "profile_settings": self._read_example_json(
                "dualnode_deepseek_v4_flash_qwen35_dflash_profile_settings.example.json"
            ),
        }


class G22DeepEPL20NDualNodeVerifier(_G22DeepEPL20NBaseVerifier):
    capability = "g22_deepep_l20n_dualnode_16gpus"

    def verify(self):
        start = self._start()
        try:
            deepep_source = self._read_engine_source("Backend/CGC/deepep_sglang_patch.py")
            bundle = self._load_dualnode_bundle()
            bootstrap = bundle["bootstrap"]
            manifest = bundle["system_manifest"]
            profile = bundle["profile_settings"]

            runtime_defaults = bootstrap.get("runtime_defaults") or {}
            extra_args = runtime_defaults.get("extra_args") or []
            env = runtime_defaults.get("env") or {}
            components = manifest.get("components") or []
            runtime_component = next(
                (item for item in components if item.get("component_id") == "vendored_sglang_runtime"),
                {},
            )
            runtime_shape = profile.get("runtime_shape") or {}
            launch_env_defaults = profile.get("launch_env_defaults") or {}

            has_dualnode_contract = (
                (bootstrap.get("capability_summary") or {}).get("topology") == "dualnode_16gpu"
                and (bootstrap.get("capability_summary") or {}).get("deepep_parallel_profile") == "ep16_tp1"
                and "--nnodes" in extra_args
                and "2" in extra_args
                and str(env.get("CGC_SGLANG_NNODES") or "") == "2"
            )
            has_dualnode_manifest = (
                str((manifest.get("system_profile") or {}).get("routing_topology_profile") or "")
                == "single_runtime_dualnode_deepep"
                and int(runtime_component.get("parallel_ep_size") or 0) == 16
                and int(runtime_component.get("parallel_nnodes") or 0) == 2
            )
            has_profile_binding = (
                int(runtime_shape.get("ep_size") or 0) == 16
                and int(runtime_shape.get("nnodes") or 0) == 2
                and str((profile.get("distributed_binding") or {}).get("parallel_profile") or "") == "ep16_tp1"
                and str(launch_env_defaults.get("CGC_SGLANG_NNODES") or "") == "2"
            )
            has_patch_entrypoints = self._contains_all(
                deepep_source,
                [
                    "def resolve_deepep_parallelism(",
                    "def build_sglang_deepep_engine_kwargs(",
                    'or profile_tp',
                    'or profile_ep',
                ],
            )

            self._add_metric("has_dualnode_contract", has_dualnode_contract)
            self._add_metric("has_dualnode_manifest", has_dualnode_manifest)
            self._add_metric("has_profile_binding", has_profile_binding)
            self._add_metric("has_patch_entrypoints", has_patch_entrypoints)
            self._add_evidence(
                "[g22_deepep_l20n_dualnode_16gpus] dualnode bundle formalizes ep16/tp1 + nnodes=2 and vendored runtime component parallel_nnodes=2"
            )
            self._add_evidence(
                f"[g22_deepep_l20n_dualnode_16gpus] patch_entrypoints={has_patch_entrypoints}"
            )

            if not all(
                [has_dualnode_contract, has_dualnode_manifest, has_profile_binding, has_patch_entrypoints]
            ):
                return self._finish(start, VerificationStatus.FAIL, "dualnode l20n contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))


class G22DeepEPL20NMegatrainVerifier(_G22DeepEPL20NBaseVerifier):
    capability = "g22_deepep_l20n_megatrain_8step"

    def verify(self):
        start = self._start()
        try:
            pipeline_source = self._read_engine_source("cgc_engine/pipeline.py")
            m76_source = self._read_engine_source("cgc_engine/product/m76_gate.py")
            bundle = self._load_dualnode_bundle()
            runtime_shape = (bundle["profile_settings"].get("runtime_shape") or {})

            has_megatrain_route = self._contains_all(
                pipeline_source,
                [
                    "def run_megatrain_eight_step_pipeline(",
                    "return MegatrainEightStepPipeline(config).run()",
                    "class HarnessAgentPipeline:",
                ],
            )
            has_eight_step_contract = self._contains_all(
                m76_source,
                [
                    "def _validate_eight_step_pipeline(",
                    'cli_markers = [f"[{step}/8]" for step in range(1, 9)]',
                    '"step7_kernel_codegen"',
                    '"step8_runtime"',
                ],
            )
            has_l20n_shape = (
                int(runtime_shape.get("ep_size") or 0) == 16
                and int(runtime_shape.get("nnodes") or 0) == 2
                and str((bundle["profile_settings"].get("launch_env_defaults") or {}).get("CGC_CLOUD_MODEL_PATH") or "")
                != ""
            )

            self._add_metric("has_megatrain_route", has_megatrain_route)
            self._add_metric("has_eight_step_contract", has_eight_step_contract)
            self._add_metric("has_l20n_shape", has_l20n_shape)
            self._add_evidence(
                "[g22_deepep_l20n_megatrain_8step] pipeline exposes run_megatrain_eight_step_pipeline and m76 validates the full 8-step marker set"
            )
            if not all([has_megatrain_route, has_eight_step_contract, has_l20n_shape]):
                return self._finish(start, VerificationStatus.FAIL, "megatrain 8step l20n contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))


class G22DeepEPL20NInferenceVerifier(_G22DeepEPL20NBaseVerifier):
    capability = "g22_deepep_l20n_inference_8step"

    def verify(self):
        start = self._start()
        try:
            pipeline_source = self._read_engine_source("cgc_engine/pipeline.py")
            gates_source = self._read_engine_source("cgc_engine/product/m1_m6_pipeline_gates.py")
            manifest = self._load_dualnode_bundle()["system_manifest"]

            has_inference_route = self._contains_all(
                gates_source,
                [
                    '"route": "inference_8step"',
                    '"steps.inference_8step.step2_fullgraph_capture"',
                    '"steps.inference_8step.step8_fullgraph_deploy"',
                    '"omlx_flashmoe_ondemand_gate"',
                ],
            )
            has_inference_pipeline = self._contains_all(
                pipeline_source,
                [
                    'task_type="inference"',
                    'model_name="moe_harness"',
                    'report_filename=str(self.config.get("report_filename", "harness_moe_report.json"))',
                ],
            )
            has_dualnode_runtime = any(
                item.get("component_id") == "vendored_sglang_runtime"
                and int(item.get("parallel_nnodes") or 0) == 2
                and int(item.get("parallel_ep_size") or 0) == 16
                for item in (manifest.get("components") or [])
            )

            self._add_metric("has_inference_route", has_inference_route)
            self._add_metric("has_inference_pipeline", has_inference_pipeline)
            self._add_metric("has_dualnode_runtime", has_dualnode_runtime)
            self._add_evidence(
                "[g22_deepep_l20n_inference_8step] inference_8step gate paths and dualnode deepep runtime component are both present"
            )
            if not all([has_inference_route, has_inference_pipeline, has_dualnode_runtime]):
                return self._finish(start, VerificationStatus.FAIL, "inference 8step l20n contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))


class G22BootstrapDeepEPCompatVerifier(_G22DeepEPL20NBaseVerifier):
    capability = "g22_deepep_bootstrap_deepep_compat"

    def verify(self):
        start = self._start()
        try:
            validator_source = self._read_app_source("shared/profile_bundle_validator.py")
            bundle = self._load_dualnode_bundle()
            bootstrap = bundle["bootstrap"]
            manifest = bundle["system_manifest"]
            profile = bundle["profile_settings"]

            has_bundle_validator = self._contains_all(
                validator_source,
                [
                    "def validate_profile_bundle(",
                    'bootstrap_contract_path: str = ""',
                    "resolved_bootstrap_path = _resolve_contract_path(",
                    "bootstrap_contract = _load_json_file(resolved_bootstrap_path)",
                ],
            )
            has_bootstrap_binding = (
                str(profile.get("bootstrap_contract_path") or "")
                == "dualnode_blackwell_deepep_ep16_tp1_runtime_bootstrap_contract.example.json"
                and str(profile.get("bootstrap_contract_binding_key") or "")
                == "dualnode_blackwell_deepep_ep16_tp1_runtime_v1"
                and str(bootstrap.get("requested_dispatch_backend") or "") == "deepep"
            )
            has_system_ref = (
                str(((profile.get("system_profile_ref") or {}).get("source_path")) or "")
                == "dualnode_deepseek_v4_flash_qwen35_dflash_system_manifest.example.json"
                and str((((manifest.get("system_profile") or {}).get("environment_bootstrap_ref") or {}).get("requested_dispatch_backend")) or "")
                == "deepep"
            )

            self._add_metric("has_bundle_validator", has_bundle_validator)
            self._add_metric("has_bootstrap_binding", has_bootstrap_binding)
            self._add_metric("has_system_ref", has_system_ref)
            self._add_evidence(
                "[g22_deepep_bootstrap_deepep_compat] profile_settings -> bootstrap_contract -> system_manifest reference chain is explicit and validator reads the same triplet"
            )
            if not all([has_bundle_validator, has_bootstrap_binding, has_system_ref]):
                return self._finish(start, VerificationStatus.FAIL, "bootstrap deepep compatibility contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))


class G22SystemProfileL20NVerifier(_G22DeepEPL20NBaseVerifier):
    capability = "g22_deepep_system_profile_l20n"

    def verify(self):
        start = self._start()
        try:
            gateway_source = self._read_engine_source("Backend/CGC/ray_serve_sglang_gateway.py")
            manifest = self._load_dualnode_bundle()["system_manifest"]
            system_profile = manifest.get("system_profile") or {}

            has_system_profile = (
                str(system_profile.get("deployment_mode") or "") == "cloud_cluster"
                and str((system_profile.get("hardware_profile") or {}).get("hardware_topology") or "")
                == "2x8_blackwell_sm120"
                and str((system_profile.get("component_families") or {}).get("route_policy_family") or "")
                == "deepep_ep16_tp1_dualnode"
            )
            has_gateway_manifest_binding = self._contains_all(
                gateway_source,
                [
                    "system_manifest_path = _resolve_contract_path(",
                    "system_manifest = _load_json_file(system_manifest_path)",
                    'edge_matrix.setdefault("hardware_type", headers.get("x-cgc-hardware-type", "Nvidia_L20N"))',
                    '"system_manifest_path": str(self.config.system_manifest_path or "")',
                ],
            )

            self._add_metric("has_system_profile", has_system_profile)
            self._add_metric("has_gateway_manifest_binding", has_gateway_manifest_binding)
            self._add_evidence(
                "[g22_deepep_system_profile_l20n] system manifest carries L20N dualnode profile and gateway threads system_manifest_path into runtime contract context"
            )
            if not all([has_system_profile, has_gateway_manifest_binding]):
                return self._finish(start, VerificationStatus.FAIL, "system profile l20n contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))


class G22UPKL20NOptimizationVerifier(_G22DeepEPL20NBaseVerifier):
    capability = "g22_deepep_upk_l20n_optimization"

    def verify(self):
        start = self._start()
        try:
            app_cli_source = self._read_app_source("cli/cgc.py")
            profile = self._load_dualnode_bundle()["profile_settings"]
            has_upk_bundle_writer = self._contains_all(
                app_cli_source,
                [
                    "def _write_profile_settings_bundle(",
                    '"profile_settings_path": profile_settings_path',
                    'fields["execution_profile_binding_key"] = str(execution)',
                    "validate_profile_bundle(",
                    '"upkg_target": "2.2"',
                ],
            )
            has_execution_binding = (
                str(profile.get("execution_profile_binding_key") or "") == "dualnode_dsv4_qwen_dflash_exec_v1"
                and str(profile.get("flow_parameter_contract_binding_key") or "") == "dualnode_dsv4_qwen_dflash_flow_v1"
                and str((profile.get("distributed_binding") or {}).get("parallel_profile") or "") == "ep16_tp1"
            )

            self._add_metric("has_upk_bundle_writer", has_upk_bundle_writer)
            self._add_metric("has_execution_binding", has_execution_binding)
            self._add_evidence(
                "[g22_deepep_upk_l20n_optimization] app CLI writes profile_settings/execution_profile_binding_key and validates the bundle against system_manifest + bootstrap contract"
            )
            if not all([has_upk_bundle_writer, has_execution_binding]):
                return self._finish(start, VerificationStatus.FAIL, "upk l20n optimization contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))


class G22StateABIL20NVerifier(_G22DeepEPL20NBaseVerifier):
    capability = "g22_deepep_state_abi_l20n"

    def verify(self):
        start = self._start()
        try:
            schema_source = self._read_engine_source("cgc_engine/pd/dopd_schema.py")
            profile = self._load_dualnode_bundle()["profile_settings"]

            has_state_abi = self._contains_all(
                schema_source,
                [
                    "class DOPDResumePayloadV2:",
                    "finished_layer: int = 0",
                    "max_local_layer: int = 0",
                    'transport_codec: str = "cq4"',
                    "def encode_dopd_resume_payload_v2(",
                    "def decode_dopd_resume_payload_v2(",
                ],
            )
            has_l20n_binding = (
                str((profile.get("system_profile_ref") or {}).get("profile_id") or "")
                == "dualnode_deepseek_v4_flash_qwen35_dflash_deepep"
                and str((profile.get("runtime_shape") or {}).get("default_runtime_extra_args") or "").find("deepep") != -1
            )

            self._add_metric("has_state_abi", has_state_abi)
            self._add_metric("has_l20n_binding", has_l20n_binding)
            self._add_evidence(
                "[g22_deepep_state_abi_l20n] DOPDResumePayloadV2 exposes layer-granularity ABI fields and the L20N profile binds the same dualnode deepep runtime shape"
            )
            if not all([has_state_abi, has_l20n_binding]):
                return self._finish(start, VerificationStatus.FAIL, "state abi l20n contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))
