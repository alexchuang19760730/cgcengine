"""Gate 2.1 fusion-governance capability verifiers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from .base import BaseVerifier, VerificationStatus
from .workspace_paths import app_root_for, engine_root_for


class _G21FusionBaseVerifier(BaseVerifier):
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

    def _load_upkg21_bundle(self) -> Dict[str, Dict[str, Any]]:
        return {
            "bootstrap": self._read_example_json(
                "host2_blackwell_sglang_runtime_bootstrap_contract.example.json"
            ),
            "system_manifest": self._read_example_json(
                "host2_upkg21_dflash_benchmark_system_manifest.example.json"
            ),
            "profile_settings": self._read_example_json(
                "host2_upkg21_dflash_benchmark_profile_settings.example.json"
            ),
        }


class G21EightStepPipelineGovernanceVerifier(_G21FusionBaseVerifier):
    capability = "g21_eight_step_pipeline_governance_integration"

    def verify(self):
        start = self._start()
        try:
            m76_source = self._read_engine_source("cgc_engine/product/m76_gate.py")
            pipeline_source = self._read_engine_source("cgc_engine/pipeline.py")
            upkg21_source = self._read_engine_source("cgc_engine/product/upkg21_gate.py")

            has_m76_governance_validator = self._contains_all(
                m76_source,
                [
                    "def _validate_eight_step_pipeline(",
                    'cli_markers = [f"[{step}/8]" for step in range(1, 9)]',
                    '"step7_kernel_codegen"',
                    '"step8_runtime"',
                ],
            )
            has_pipeline_eight_step_route = self._contains_all(
                pipeline_source,
                [
                    "def run_megatrain_eight_step_pipeline(",
                    "return MegatrainEightStepPipeline(config).run()",
                    "八步流水線",
                    "4D 矩陣：環境 × 任務 × 硬體 × 模型",
                ],
            )
            has_upkg21_governance_integration = self._contains_all(
                upkg21_source,
                [
                    "m76_report = run_m76_gate(output_dir=str(m76_output_root))",
                    '"m76_heterogeneous_gate": {',
                    '"agent_execution": agent_execution',
                    '"schema_refs": schema_refs',
                ],
            )

            self._add_metric("has_m76_governance_validator", has_m76_governance_validator)
            self._add_metric("has_pipeline_eight_step_route", has_pipeline_eight_step_route)
            self._add_metric("has_upkg21_governance_integration", has_upkg21_governance_integration)
            self._add_evidence(
                "[g21_eight_step_pipeline_governance_integration] m76 validates the formal 8-step marker set and upkg21 imports m76 into the product governance chain"
            )

            if not all(
                [
                    has_m76_governance_validator,
                    has_pipeline_eight_step_route,
                    has_upkg21_governance_integration,
                ]
            ):
                return self._finish(start, VerificationStatus.FAIL, "g21 eight-step governance contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))


class G21UPKFusionBindingVerifier(_G21FusionBaseVerifier):
    capability = "g21_upk_binding_for_fusion_variants"

    def verify(self):
        start = self._start()
        try:
            engine_cli_source = self._read_engine_source("cgc_engine/cli.py")
            app_cli_source = self._read_app_source("cli/cgc.py")
            validator_source = self._read_app_source("shared/profile_bundle_validator.py")
            bundle = self._load_upkg21_bundle()
            profile = bundle["profile_settings"]
            manifest = bundle["system_manifest"]
            bootstrap = bundle["bootstrap"]
            binding_ref = (manifest.get("system_profile") or {}).get("profile_binding_ref") or {}

            has_fusion_variant_surface = self._contains_all(
                engine_cli_source,
                [
                    "choices=['dspark', 'jetspec', 'fusion']",
                    "if args.speculative_mode == 'fusion':",
                    "checks.append('fusion_dspark_jetspec_valid')",
                    "checks.append('verified500_closure_valid')",
                    'speculative_mode="fusion"',
                    "dspark_budget=64",
                    "jetspec_branches=8",
                ],
            )
            has_upk_binding_writer = self._contains_all(
                app_cli_source,
                [
                    '"execution_profile_binding_keys"',
                    '"bootstrap_contract_binding_keys"',
                    '"flow_parameter_contract_binding_keys"',
                    "def _write_profile_settings_bundle(",
                    "validate_profile_bundle(",
                    '"profile_settings_path": profile_settings_path',
                ],
            )
            has_example_bundle_binding = (
                str(profile.get("execution_profile_binding_key") or "")
                == str(binding_ref.get("execution_profile_binding_key") or "")
                and str(profile.get("bootstrap_contract_binding_key") or "")
                == str(binding_ref.get("bootstrap_contract_binding_key") or "")
                and str(profile.get("flow_parameter_contract_binding_key") or "")
                == str(binding_ref.get("flow_parameter_contract_binding_key") or "")
                and str(profile.get("bootstrap_contract_path") or "")
                == "host2_blackwell_sglang_runtime_bootstrap_contract.example.json"
                and str((profile.get("system_profile_ref") or {}).get("source_path") or "")
                == "host2_upkg21_dflash_benchmark_system_manifest.example.json"
                and str(bootstrap.get("bootstrap_contract_id") or "") == "host2_blackwell_sglang_runtime_v1"
            )
            has_bundle_validator = self._contains_all(
                validator_source,
                [
                    "def validate_profile_bundle(",
                    "resolved_bootstrap_path = _resolve_contract_path(",
                    "resolved_system_manifest_path = _resolve_contract_path(",
                    "task_type_contract_validation",
                ],
            )

            self._add_metric("has_fusion_variant_surface", has_fusion_variant_surface)
            self._add_metric("has_upk_binding_writer", has_upk_binding_writer)
            self._add_metric("has_example_bundle_binding", has_example_bundle_binding)
            self._add_metric("has_bundle_validator", has_bundle_validator)
            self._add_evidence(
                "[g21_upk_binding_for_fusion_variants] CLI exposes dspark/jetspec/fusion variants, while app CLI writes and validates execution/bootstrap/flow binding keys for the upkg21 bundle"
            )

            if not all(
                [
                    has_fusion_variant_surface,
                    has_upk_binding_writer,
                    has_example_bundle_binding,
                    has_bundle_validator,
                ]
            ):
                return self._finish(start, VerificationStatus.FAIL, "g21 upk fusion binding contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))


class G21StateABIExtensionHookVerifier(_G21FusionBaseVerifier):
    capability = "g21_state_abi_extension_hook"

    def verify(self):
        start = self._start()
        try:
            schema_source = self._read_engine_source("cgc_engine/pd/dopd_schema.py")
            pipeline_source = self._read_engine_source("cgc_engine/pipeline.py")
            jetspec_engine_source = self._read_engine_source(
                "Backend/CGC/vendored/jetspec/jetspec/inference_engine/engine.py"
            )
            jetspec_frontier_source = self._read_engine_source(
                "Backend/CGC/vendored/jetspec/jetspec/tree/layer_conditional/path_conditional_refresh.py"
            )

            has_state_payload = self._contains_all(
                schema_source,
                [
                    "class DOPDResumePayloadV2:",
                    "abi_descriptor: Dict[str, Any] = field(default_factory=dict)",
                    "finished_layer: int = 0",
                    "max_local_layer: int = 0",
                    "hidden_states_ref: str = \"\"",
                    "partial_kv_ref: str = \"\"",
                    "def encode_dopd_resume_payload_v2(",
                    "def decode_dopd_resume_payload_v2(",
                ],
            )
            has_abi_hook_surface = self._contains_all(
                pipeline_source,
                [
                    '"state_abi_policy": str(getattr(config, "state_abi_policy", "") or "")',
                    'if "deepseek_v2_to_v4" in state_abi_policy:',
                    'notes.append("runtime_plugin=deepseek_abi_bridge")',
                    'state_abi_policy: str = "deepseek_v2_to_v4_min_state_abi_v1_2"',
                    "qk_nope_head_dim: int = 128",
                    "legacy_o_proj_in_dim: int = 16384",
                ],
            )
            has_tree_verify_frontier_hooks = self._contains_all(
                jetspec_engine_source,
                [
                    "tree_kv = self._batched_tree_verify_forward(",
                    "accepted_path, acc, correction = tree_accept(",
                    "self._append_tree_path_kv(",
                    "def _batched_tree_verify_forward(",
                ],
            )
            has_frontier_cache_hook = self._contains_all(
                jetspec_frontier_source,
                [
                    "batched-frontier KV reuse",
                    "tree-attention + batched-frontier KV-reuse hook",
                    "Genuinely vLLM-required.",
                ],
            )

            self._add_metric("has_state_payload", has_state_payload)
            self._add_metric("has_abi_hook_surface", has_abi_hook_surface)
            self._add_metric("has_tree_verify_frontier_hooks", has_tree_verify_frontier_hooks)
            self._add_metric("has_frontier_cache_hook", has_frontier_cache_hook)
            self._add_evidence(
                "[g21_state_abi_extension_hook] DOPD resume payload carries ABI-aligned edge/cloud state, pipeline surfaces a deepseek ABI bridge policy, and vendored JetSpec exposes tree-verify plus frontier-KV hook anchors"
            )

            if not all(
                [
                    has_state_payload,
                    has_abi_hook_surface,
                    has_tree_verify_frontier_hooks,
                    has_frontier_cache_hook,
                ]
            ):
                return self._finish(start, VerificationStatus.FAIL, "g21 state abi hook contract incomplete")
            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            return self._finish(start, VerificationStatus.ERROR, str(exc))
