"""layer_adaptive_verifier.py — Gate 2.0 层自适应校验

校验 max_local_layer / finished_layer 配置一致性：
- max_local_layer 必须 <= model_config.num_layers
- finished_layer continuation 必须 max_local_layer < num_layers
- 端云分工拓扑合理性
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

from .base import BaseVerifier, VerificationStatus


class LayerAdaptiveVerifier(BaseVerifier):
    capability = "layer_adaptive"

    def verify(self) -> VerificationResult:
        start = self._start()
        try:
            max_local_layer = getattr(self.args, "max_local_layer", None)
            finished_layer = getattr(self.args, "finished_layer", False)

            # 简化：从 args.model 推断层数（实际可从 profile bundle 读取）
            num_layers = self._infer_num_layers()
            self._add_metric("max_local_layer", max_local_layer)
            self._add_metric("finished_layer", finished_layer)
            self._add_metric("inferred_num_layers", num_layers)

            if max_local_layer is None:
                self._add_evidence("max_local_layer not set, using default split")
                return self._finish(start, VerificationStatus.PASS)

            if not isinstance(max_local_layer, int) or max_local_layer <= 0:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    f"max_local_layer must be positive int, got {max_local_layer}",
                )

            if num_layers > 0 and max_local_layer > num_layers:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    f"max_local_layer ({max_local_layer}) > num_layers ({num_layers})",
                )
            self._add_evidence(f"✓ max_local_layer={max_local_layer} <= num_layers={num_layers}")

            if finished_layer:
                if max_local_layer >= num_layers:
                    return self._finish(
                        start,
                        VerificationStatus.FAIL,
                        "finished_layer continuation requires max_local_layer < num_layers",
                    )
                self._add_evidence(
                    f"✓ finished_layer continuation: edge runs [0, {max_local_layer}), "
                    f"cloud continues [{max_local_layer}, {num_layers})"
                )

            # 模拟端云分工比例合理性
            edge_ratio = max_local_layer / num_layers if num_layers > 0 else 0.0
            self._add_metric("edge_layer_ratio", round(edge_ratio, 3))
            if edge_ratio < 0.1 or edge_ratio > 0.9:
                self._add_evidence(
                    f"⚠ edge_layer_ratio={edge_ratio:.3f} outside [0.1, 0.9] recommended range"
                )
            else:
                self._add_evidence(f"✓ edge_layer_ratio={edge_ratio:.3f} within recommended range")

            return self._finish(start, VerificationStatus.PASS)

        except Exception as e:
            import traceback
            self._add_evidence(traceback.format_exc())
            return self._finish(start, VerificationStatus.ERROR, str(e))

    def _infer_num_layers(self) -> int:
        """从 args.model 推断层数（保守返回 32 作默认）"""
        model = getattr(self.args, "model", "")
        # 简化映射，实际应从 profile bundle 读取
        layer_map = {
            "deepseek": 61,
            "tmax": 32,
            "qwen": 28,
            "llama": 32,
            "mistral": 32,
        }
        for k, v in layer_map.items():
            if k in str(model).lower():
                return v
        return 32
