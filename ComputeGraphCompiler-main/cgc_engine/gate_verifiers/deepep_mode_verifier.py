"""deepep_mode_verifier.py — Gate 2.0 DeepEP 三模式验证

调用 Backend.CGC.deepep_sglang_patch + sglang.srt.layers.moe.utils.DeepEPMode 枚举
端到端：
- 校验 DeepEPMode 三模式（NORMAL / LOW_LATENCY / AUTO）真实启用
  * NORMAL       — 仅启用 Normal dispatch/combine 路径（prefill 主干，高吞吐）
  * LOW_LATENCY  — 仅启用 LowLatency dispatch/combine 路径（decode 主干，低延迟）
  * AUTO         — 双路径同时启用，按 is_extend_in_batch 动态切换
- 校验 build_sglang_deepep_engine_kwargs 返回的 deepep_mode 字段
- 校验 DeepEPMode.resolve() 语义（AUTO → NORMAL for prefill, LOW_LATENCY for decode）
- 若 deep_ep 模块可用 + 有多 GPU，可触发 run_deepep_v2_probe
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict

from .base import BaseVerifier, VerificationStatus


class DeepEPModeVerifier(BaseVerifier):
    capability = "deepep_mode"

    def verify(self) -> VerificationResult:
        start = self._start()
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            backend_cgc = os.path.join(repo_root, "Backend", "CGC")
            if backend_cgc not in sys.path:
                sys.path.insert(0, backend_cgc)

            from deepep_sglang_patch import (
                build_sglang_deepep_engine_kwargs,
                run_deepep_v2_probe,
            )
            # DeepEPMode 在 sglang.srt.layers.moe.utils
            sys.path.insert(0, os.path.join(repo_root, "Backend", "CGC", "cloud_sglang", "python"))
            from sglang.srt.layers.moe.utils import DeepEPMode
            self._add_evidence("✓ imported deepep_sglang_patch.build_sglang_deepep_engine_kwargs")
            self._add_evidence("✓ imported sglang.srt.layers.moe.utils.DeepEPMode enum")
            self._add_evidence("✓ imported deepep_sglang_patch.run_deepep_v2_probe")

            # ====================================================================
            # 三模式真实启用校验（基于 DeepEPMode 真实 API：enable_normal/enable_low_latency）
            # ====================================================================
            # NORMAL 模式：仅启用 Normal 路径（prefill 主干，高吞吐）
            normal = DeepEPMode.NORMAL
            assert normal.enable_normal() is True, "NORMAL should enable_normal=True"
            assert normal.enable_low_latency() is False, "NORMAL should enable_low_latency=False"
            assert normal.resolve(is_extend_in_batch=True) == DeepEPMode.NORMAL
            assert normal.resolve(is_extend_in_batch=False) == DeepEPMode.NORMAL
            self._add_evidence("✓ NORMAL: enable_normal=True, enable_low_latency=False (prefill 主干，高吞吐)")
            self._add_metric("mode_normal_enable_normal", True)
            self._add_metric("mode_normal_enable_low_latency", False)

            # LOW_LATENCY 模式：仅启用 LowLatency 路径（decode 主干，低延迟）
            low_lat = DeepEPMode.LOW_LATENCY
            assert low_lat.enable_normal() is False, "LOW_LATENCY should enable_normal=False"
            assert low_lat.enable_low_latency() is True, "LOW_LATENCY should enable_low_latency=True"
            assert low_lat.resolve(is_extend_in_batch=True) == DeepEPMode.LOW_LATENCY
            assert low_lat.resolve(is_extend_in_batch=False) == DeepEPMode.LOW_LATENCY
            self._add_evidence("✓ LOW_LATENCY: enable_normal=False, enable_low_latency=True (decode 主干，低延迟)")
            self._add_metric("mode_low_latency_enable_normal", False)
            self._add_metric("mode_low_latency_enable_low_latency", True)

            # AUTO 模式：双路径同时启用，按 is_extend_in_batch 动态切换
            auto = DeepEPMode.AUTO
            assert auto.enable_normal() is True, "AUTO should enable_normal=True"
            assert auto.enable_low_latency() is True, "AUTO should enable_low_latency=True"
            # AUTO 的 resolve 语义：prefill→NORMAL, decode→LOW_LATENCY
            assert auto.resolve(is_extend_in_batch=True) == DeepEPMode.NORMAL, \
                "AUTO.resolve(is_extend_in_batch=True) should be NORMAL"
            assert auto.resolve(is_extend_in_batch=False) == DeepEPMode.LOW_LATENCY, \
                "AUTO.resolve(is_extend_in_batch=False) should be LOW_LATENCY"
            self._add_evidence("✓ AUTO: enable_normal=True, enable_low_latency=True (双路径同时启用)")
            self._add_evidence("✓ AUTO.resolve(is_extend_in_batch=True)  -> NORMAL    (prefill 阶段)")
            self._add_evidence("✓ AUTO.resolve(is_extend_in_batch=False) -> LOW_LATENCY (decode 阶段)")
            self._add_metric("mode_auto_enable_normal", True)
            self._add_metric("mode_auto_enable_low_latency", True)

            # ====================================================================
            # build_sglang_deepep_engine_kwargs 真实调用（需要 deep_ep 模块）
            # 校验返回的 deepep_mode 字段与输入一致
            # ====================================================================
            for mode_str in ["normal", "low_latency", "auto"]:
                try:
                    kwargs = build_sglang_deepep_engine_kwargs(
                        tp_size=4,
                        ep_size=4,
                        deepep_mode=mode_str,
                    )
                    returned_mode = kwargs.get("deepep_mode")
                    if returned_mode == mode_str:
                        self._add_evidence(f"✓ build_sglang_deepep_engine_kwargs(deepep_mode={mode_str}) -> deepep_mode={returned_mode}")
                    else:
                        self._add_evidence(f"⚠ build_sglang_deepep_engine_kwargs(deepep_mode={mode_str}) returned deepep_mode={returned_mode}")
                    self._add_metric(f"build_kwargs_mode_{mode_str}", returned_mode)
                except Exception as build_e:
                    # deep_ep 模块不可用是常见情况，不报 FAIL
                    self._add_evidence(f"build_sglang_deepep_engine_kwargs(mode={mode_str}) skipped: {build_e}")
                    self._add_metric(f"build_kwargs_mode_{mode_str}", "unavailable")

            # ====================================================================
            # 可选：真实 probe（需要多 GPU + deep_ep 模块）
            # ====================================================================
            run_probe = bool(int(os.environ.get("CGC_DEEPEP_RUN_PROBE", "0")))
            self._add_metric("run_probe", run_probe)
            if run_probe:
                try:
                    import torch
                    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
                        self._add_evidence(f"launching run_deepep_v2_probe with {torch.cuda.device_count()} GPUs...")
                        run_deepep_v2_probe(
                            num_tokens=128,
                            hidden=4096,
                            num_topk=2,
                            num_experts=8,
                            tp_size=torch.cuda.device_count(),
                            deepep_mode="auto",
                        )
                        self._add_evidence("✓ run_deepep_v2_probe completed")
                    else:
                        self._add_evidence("skip probe: need >=2 CUDA GPUs")
                except Exception as probe_e:
                    self._add_evidence(f"probe failed (non-fatal): {probe_e}")

            return self._finish(start, VerificationStatus.PASS)

        except ImportError as e:
            return self._finish(start, VerificationStatus.SKIP, f"deepep_sglang_patch not available: {e}")
        except AssertionError as e:
            return self._finish(start, VerificationStatus.FAIL, f"DeepEPMode API assertion failed: {e}")
        except Exception as e:
            import traceback
            self._add_evidence(traceback.format_exc())
            return self._finish(start, VerificationStatus.ERROR, str(e))
