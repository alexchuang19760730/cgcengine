"""g23_trueorthokda_adapter_verifier.py — g23 TrueOrthoKDA 适配验证器"""
from __future__ import annotations

import inspect
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator

import torch

from .base import BaseVerifier, VerificationStatus
from .workspace_paths import extend_pythonpath_for


@contextmanager
def _patched_env(updates: Dict[str, str]) -> Iterator[None]:
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class G23TrueOrthoKDAAdapterVerifier(BaseVerifier):
    """验证 g23 TrueOrthoKDA adapter 的当前真实闭环。

    校验内容：
      1. KDA state runtime 生成 versioned + compressed 的可恢复状态
      2. resume_one_token_from_kda_state 可直接消费该状态
      3. UnifiedIRCompiler 在启用 TrueOrthoKDA 时会把 attention lower 为 ortho_kda
      4. RSWA adapter 仍能维护 Reference KV + Output KV 的双层管理
    """

    capability = "g23_trueorthokda_adapter"

    def verify(self):
        start = self._start()
        try:
            extend_pythonpath_for(__file__)
            from app.edge_engine.kda_state_runtime import (
                build_real_kda_state_from_request,
                inspect_kda_state_bytes,
                resume_one_token_from_kda_state,
            )
            from Backend.CGC.compiler.unified_compiler import UnifiedIRCompiler
            from cgc_engine.rswa_integration.rswa_prefill_pool_adapter import (
                CGCUnlimitedRSWAAttention,
            )

            with _patched_env({"CGC_ENABLE_ORTHO_KDA": "1", "CGC_ENABLE_RSWA": "0"}):
                compiler = UnifiedIRCompiler({"hardware_type": "Nvidia_L20N"})
                ir = compiler.compile_to_unified_ir(
                    model_graph={
                        "arch": "g23_trueorthokda_adapter",
                        "layers_block_type": ["attention", "attention"],
                        "hidden_size": 16,
                        "num_hidden_layers": 2,
                    }
                )
                lowered = compiler.lower_to_hardware(ir)

            ortho_layers = [layer for layer in ir.layers if layer.layer_type == "attention"]
            ortho_backends = [layer.kernel_backend for layer in ortho_layers]
            injectable_backends = [
                str(layer.get("kernel_backend") or "")
                for layer in lowered.get("injectable_layers", [])
            ]

            state_bundle = build_real_kda_state_from_request(
                {"prompt": "g23 trueorthokda adapter smoke"},
                trace_id="g23-trueorthokda-adapter",
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
                trace_id="g23-trueorthokda-adapter-resume",
            )
            zero_copy = (
                resume.get("zero_copy_runtime")
                if isinstance(resume.get("zero_copy_runtime"), dict)
                else {}
            )

            with tempfile.TemporaryDirectory(prefix="g23_trueorthokda_adapter_") as tmpdir:
                init_sig = inspect.signature(CGCUnlimitedRSWAAttention.__init__)
                init_kwargs = {"dim": 16, "num_heads": 4, "window_size": 4}
                if "init_projs" in init_sig.parameters:
                    init_kwargs["init_projs"] = True
                attention = CGCUnlimitedRSWAAttention(**init_kwargs)
                attention.prefill_pool.storage_path = Path(tmpdir)
                attention.prefill_pool.storage_path.mkdir(parents=True, exist_ok=True)

                token_ids = torch.arange(6, dtype=torch.long)
                ref_k = torch.randn(1, 4, 6, 4, dtype=torch.bfloat16)
                ref_v = torch.randn(1, 4, 6, 4, dtype=torch.bfloat16)
                chunk_id = attention.add_reference_chunk(token_ids, ref_k, ref_v)
                x = torch.randn(1, 3, 16)
                _, new_k, new_v = attention.forward(x, use_reference=True, update_output_kv=True)
                pool_info = attention.get_pool_info()

            self._add_metric("state_kind", state_bundle.get("state_kind"))
            self._add_metric("state_codec", state_bundle.get("state_codec"))
            self._add_metric("state_summary", state_summary)
            self._add_metric("resume_tensor_device", resume.get("resume_tensor_device"))
            self._add_metric("zero_copy_runtime", zero_copy)
            self._add_metric("ortho_backends", ortho_backends)
            self._add_metric("injectable_backends", injectable_backends)
            self._add_metric("rswa_reference_chunk_id", chunk_id)
            self._add_metric("rswa_reference_tokens", int(ref_k.shape[2]))
            self._add_metric("rswa_window_tokens", int(new_k.shape[2]))
            self._add_metric("rswa_pool_info", pool_info)
            self._add_evidence(
                f"[g23_trueorthokda_adapter] kda state built: "
                f"schema={state_summary.get('schema_version')} codec={state_bundle.get('state_codec')} "
                f"compression={state_summary.get('compression_ratio')}"
            )
            self._add_evidence(
                f"[g23_trueorthokda_adapter] resume executed: "
                f"device={resume.get('resume_tensor_device')} zero_copy={zero_copy.get('device_resume_consumed')}"
            )
            self._add_evidence(
                f"[g23_trueorthokda_adapter] unified compiler lowered attention to {ortho_backends}"
            )
            self._add_evidence(
                f"[g23_trueorthokda_adapter] rswa adapter keeps reference/output kv split: "
                f"reference_tokens={ref_k.shape[2]} output_window={new_k.shape[2]}"
            )

            if str(state_bundle.get("state_kind") or "") != "kda_state_v1":
                return self._finish(start, VerificationStatus.FAIL, "state_kind is not kda_state_v1")
            if str(state_bundle.get("state_codec") or "") != "cq4":
                return self._finish(start, VerificationStatus.FAIL, "state codec is not cq4")
            if int(state_summary.get("schema_version") or 0) != 1:
                return self._finish(start, VerificationStatus.FAIL, "schema_version != 1")
            if str(state_summary.get("state_source") or "") != "prefill_prefix_cache_kda_aot":
                return self._finish(start, VerificationStatus.FAIL, "unexpected state_source")
            if float(state_summary.get("compression_ratio") or 1.0) >= 1.0:
                return self._finish(start, VerificationStatus.FAIL, "compression_ratio >= 1.0")
            if not bool(resume.get("resume_decode_executed")):
                return self._finish(start, VerificationStatus.FAIL, "resume_decode_executed=false")
            if not bool(zero_copy.get("device_resume_consumed")):
                return self._finish(start, VerificationStatus.FAIL, "device_resume_consumed=false")
            if not ortho_backends or any(backend != "ortho_kda" for backend in ortho_backends):
                return self._finish(start, VerificationStatus.FAIL, f"unexpected ortho backends: {ortho_backends}")
            if "ortho_kda" not in injectable_backends:
                return self._finish(start, VerificationStatus.FAIL, "ortho_kda not exposed in lowered injectable layers")
            if int(new_k.shape[2]) != attention.window_size or int(new_v.shape[2]) != attention.window_size:
                return self._finish(start, VerificationStatus.FAIL, "output kv window mismatch")
            if int((pool_info.get("pool_status") or {}).get("hot_chunks") or 0) < 1:
                return self._finish(start, VerificationStatus.FAIL, "reference kv hot chunk missing")

            return self._finish(start, VerificationStatus.PASS)
        except ImportError as exc:
            return self._finish(start, VerificationStatus.SKIP, f"module not available: {exc}")
        except Exception as exc:
            import traceback

            self._add_evidence(traceback.format_exc())
            return self._finish(start, VerificationStatus.ERROR, str(exc))
