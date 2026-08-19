"""edge_omlx_flashmoe_verifier.py — Gate 1.0 OMLX + FlashMoE memory threshold 真实验证"""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import torch

from .base import BaseVerifier, VerificationStatus
from .workspace_paths import extend_pythonpath_for


class EdgeOMLXFlashMoEVerifier(BaseVerifier):
    capability = "edge_omlx_flashmoe_memory_threshold_decision"

    def verify(self):
        start = self._start()
        try:
            extend_pythonpath_for(__file__)
            from app.edge_engine.local_infer import EdgeLocalInferenceRuntime
            from cgc_engine.flash_moe import FlashMoEClient
            from cgc_engine.omlx import OMLXClient

            runtime = EdgeLocalInferenceRuntime()
            runtime._can_attempt_local = lambda model, use_omlx, use_flashmoe: True  # type: ignore[method-assign]
            runtime._get_vram_watermark = lambda: {  # type: ignore[method-assign]
                "total_mb": 16000.0,
                "used_mb": 4000.0,
                "available_mb": 12000.0,
                "source": "gate10_verifier",
            }

            decision1 = runtime._resolve_layer_decision(
                model="DeepSeek-V4-Flash",
                use_omlx=True,
                use_flashmoe=True,
                num_total_layers=24,
                layer_mem_mb=400.0,
                safety_threshold=0.75,
            )
            decision2 = runtime._resolve_layer_decision(
                model="DeepSeek-V4-Flash",
                use_omlx=True,
                use_flashmoe=True,
                num_total_layers=24,
                layer_mem_mb=400.0,
                safety_threshold=0.75,
            )
            self._add_metric("layer_decision", decision1)
            self._add_evidence(
                f"✓ deterministic layer decision={decision1.get('decision')} max_local_layer={decision1.get('max_local_layer')}"
            )
            if decision1 != decision2:
                return self._finish(start, VerificationStatus.FAIL, "layer decision is not deterministic")
            if str(decision1.get("decision") or "") != "split":
                return self._finish(start, VerificationStatus.FAIL, f"unexpected decision={decision1.get('decision')}")
            if int(decision1.get("max_local_layer") or 0) != 20:
                return self._finish(start, VerificationStatus.FAIL, f"unexpected max_local_layer={decision1.get('max_local_layer')}")

            temp_root = Path(tempfile.mkdtemp(prefix="cgc_omlx_flashmoe_"))
            omlx = OMLXClient(
                model_dir=str((temp_root / "omlx_model").resolve()),
                num_experts=8,
                expert_dim=64,
                intermediate_dim=128,
                gpu_cache_size=2,
                ssd_cache_dir=str((temp_root / "omlx_ssd").resolve()),
            )
            predicted = omlx.predict_experts(torch.randn(1, 64), top_k=2)
            cached_experts = omlx.get_cached_experts()
            self._add_metric("omlx_predicted_shape", list(predicted.shape))
            self._add_metric("omlx_cached_experts", cached_experts)
            self._add_evidence(f"✓ OMLX predicted experts={predicted.flatten().tolist()} cached={cached_experts}")
            if list(predicted.shape) != [1, 2]:
                return self._finish(start, VerificationStatus.FAIL, f"unexpected predicted shape={list(predicted.shape)}")
            if len(cached_experts) != 2:
                return self._finish(start, VerificationStatus.FAIL, f"unexpected cached expert count={len(cached_experts)}")

            flash = FlashMoEClient(
                expert_dir=str((temp_root / "flash_experts").resolve()),
                backend="cpu",
            )
            flash.expert_dim = 32
            flash.intermediate_dim = 64
            expert_weights = flash.load_experts([0, 1])
            self._add_metric("flashmoe_loaded_shape", list(expert_weights.shape))
            self._add_metric("flashmoe_backend", flash.info().get("backend"))
            self._add_evidence(f"✓ FlashMoE loaded experts tensor shape={list(expert_weights.shape)}")
            if list(expert_weights.shape) != [2, 64, 32]:
                return self._finish(start, VerificationStatus.FAIL, f"unexpected expert tensor shape={list(expert_weights.shape)}")

            return self._finish(start, VerificationStatus.PASS)
        except ImportError as exc:
            return self._finish(start, VerificationStatus.SKIP, f"module not available: {exc}")
        except Exception as exc:
            import traceback

            self._add_evidence(traceback.format_exc())
            return self._finish(start, VerificationStatus.ERROR, str(exc))
