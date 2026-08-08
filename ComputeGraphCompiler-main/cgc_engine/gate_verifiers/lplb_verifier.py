"""lplb_verifier.py — Gate 2.2 LPLB 线性规划负载均衡器验证

调用 cgc_engine.lplb_solver（新增实现）
端到端：
- 构造模拟负载（256 experts, 8 GPUs, Zipf 分布）
- 调用 solve_lplb 求解
- 校验：variance_after < variance_before、solver_time < 150ms、assignment 合法
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict

from .base import BaseVerifier, VerificationStatus


class LPLBVerifier(BaseVerifier):
    capability = "lplb_linear_programming"

    def verify(self) -> VerificationResult:
        start = self._start()
        try:
            from cgc_engine.lplb_solver import solve_lplb, LPLBResult
            self._add_evidence("✓ imported cgc_engine.lplb_solver.solve_lplb")

            import numpy as np

            # 模拟 Zipf 分布负载（典型 MoE 场景）
            num_experts = 256
            num_gpus = 8
            num_replicas = getattr(self.args, "expert_replica_factor", 2)
            lplb_parallelism = getattr(self.args, "lplb_parallelism", 4)

            rng = np.random.default_rng(42)
            loads = rng.zipf(1.5, num_experts).astype(np.float64)
            loads = loads / loads.sum() * 1000.0
            capacities = np.full(num_gpus, loads.sum() / num_gpus * 1.2, dtype=np.float64)

            self._add_metric("num_experts", num_experts)
            self._add_metric("num_gpus", num_gpus)
            self._add_metric("num_replicas", num_replicas)
            self._add_metric("lplb_parallelism", lplb_parallelism)
            self._add_metric("variance_before", float(np.var(loads / capacities.mean())))

            # 调用求解器
            result = solve_lplb(
                loads=loads,
                capacities=capacities,
                num_replicas=num_replicas,
                use_gpu=True,
                max_iterations=100,
                tolerance=1e-6,
            )

            self._add_metric("solver_kind", result.solver_kind)
            self._add_metric("solver_time_ms", round(result.solver_time_ms, 3))
            self._add_metric("variance_after", result.variance_after)
            self._add_metric("variance_reduction_pct", round(
                (1 - result.variance_after / max(result.variance_before, 1e-9)) * 100, 3
            ))
            self._add_metric("iterations", result.iterations)
            self._add_metric("optimal", result.optimal)
            self._add_evidence(
                f"✓ solver={result.solver_kind}, time={result.solver_time_ms:.2f}ms, "
                f"variance {result.variance_before:.6f} -> {result.variance_after:.6f}"
            )

            # 校验 assignment 合法性
            if len(result.assignment) != num_experts * num_replicas:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    f"assignment length {len(result.assignment)} != {num_experts * num_replicas}",
                )
            if result.assignment.min() < 0 or result.assignment.max() >= num_gpus:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    f"assignment gpu_id out of range [0, {num_gpus})",
                )
            self._add_evidence("✓ assignment validity OK (all gpu_ids in range)")

            # 校验 variance 降低
            if result.variance_after >= result.variance_before:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    f"variance not reduced: {result.variance_before} -> {result.variance_after}",
                )
            self._add_evidence("✓ variance reduction OK")

            # 校验求解时间 < 150ms
            if result.solver_time_ms >= 150.0:
                return self._finish(
                    start,
                    VerificationStatus.FAIL,
                    f"solver time {result.solver_time_ms:.2f}ms >= 150ms target",
                )
            self._add_evidence(f"✓ solver time {result.solver_time_ms:.2f}ms < 150ms target")

            # GPU 并行校验
            try:
                import torch
                if torch.cuda.is_available():
                    self._add_metric("gpu_solver_available", result.solver_kind == "gpu_ipm")
                    self._add_evidence(
                        f"✓ GPU parallel solver: {'used' if result.solver_kind == 'gpu_ipm' else 'fallback to cpu'}"
                    )
            except ImportError:
                pass

            return self._finish(start, VerificationStatus.PASS)

        except ImportError as e:
            return self._finish(start, VerificationStatus.SKIP, f"lplb_solver not available: {e}")
        except Exception as e:
            import traceback
            self._add_evidence(traceback.format_exc())
            return self._finish(start, VerificationStatus.ERROR, str(e))
