"""g23_cloud_l20n_tp4_verifier.py — g23 云端 L20N 双 TP4 适配验证器"""
from __future__ import annotations

from pathlib import Path

from .base import BaseVerifier, VerificationStatus


class G23CloudL20NTP4Verifier(BaseVerifier):
    """以 source-contract 方式校验 L20N + TP4 云端适配代码面。

    该能力目前不直接声明 PCIe runtime 观测闭环，只验证：
      1. pipeline 对 L20N 平台有标准化入口
      2. gateway / m76 对 Nvidia_L20N 有明确默认硬件标记
      3. SGLang backend launch command 显式声明 TP/EP/Ray 参数
      4. DeepSeek-V4 / SGLang backend 构建函数存在于当前代码面
    """

    capability = "g23_cloud_l20n_tp4_adaptation"

    def verify(self):
        start = self._start()
        try:
            workspace_root = Path(__file__).resolve().parents[3]
            pipeline_source = (workspace_root / "ComputeGraphCompiler-main" / "cgc_engine" / "pipeline.py").read_text(encoding="utf-8")
            gateway_source = (workspace_root / "ComputeGraphCompiler-main" / "Backend" / "CGC" / "ray_serve_sglang_gateway.py").read_text(encoding="utf-8")
            m76_source = (workspace_root / "ComputeGraphCompiler-main" / "cgc_engine" / "product" / "m76_gate.py").read_text(encoding="utf-8")

            has_l20n_normalizer = all(
                marker in pipeline_source
                for marker in ['if lowered in {"l20n", "l20", "nvidia_l20n"}:', 'return "l20n"']
            )
            has_l20n_default = 'headers.get("x-cgc-hardware-type", "Nvidia_L20N")' in gateway_source and '"Nvidia_L20N"' in m76_source
            has_tp4_launch = all(
                marker in gateway_source
                for marker in ['class SGLangBackendManager', '--tp-size', '--ep-size', '--use-ray']
            )
            has_deepseek_v4_route = 'return "deepseek_v4"' in pipeline_source and 'model_name in {"deepseek_v4", "deepseek_v4_flash_pro"}' in pipeline_source
            has_parallel_builder = all(
                marker in m76_source
                for marker in ["def _resolve_deepep_parallelism", "def build_sglang_deepep_engine_kwargs"]
            )

            self._add_metric("has_l20n_normalizer", has_l20n_normalizer)
            self._add_metric("has_l20n_default", has_l20n_default)
            self._add_metric("has_tp4_launch", has_tp4_launch)
            self._add_metric("has_deepseek_v4_route", has_deepseek_v4_route)
            self._add_metric("has_parallel_builder", has_parallel_builder)
            self._add_evidence(f"[g23_cloud_l20n_tp4_adaptation] l20n_normalizer={has_l20n_normalizer}")
            self._add_evidence(f"[g23_cloud_l20n_tp4_adaptation] l20n_default_marker={has_l20n_default}")
            self._add_evidence(f"[g23_cloud_l20n_tp4_adaptation] tp4_launch_contract={has_tp4_launch}")
            self._add_evidence(f"[g23_cloud_l20n_tp4_adaptation] deepseek_v4_route={has_deepseek_v4_route}")
            self._add_evidence(f"[g23_cloud_l20n_tp4_adaptation] parallel_builder_contract={has_parallel_builder}")

            if not all([has_l20n_normalizer, has_l20n_default, has_tp4_launch, has_deepseek_v4_route, has_parallel_builder]):
                return self._finish(start, VerificationStatus.FAIL, "l20n tp4 source contract incomplete")

            return self._finish(start, VerificationStatus.PASS)
        except Exception as exc:
            import traceback

            self._add_evidence(traceback.format_exc())
            return self._finish(start, VerificationStatus.ERROR, str(exc))
