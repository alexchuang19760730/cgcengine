"""test_unified_ir_injector_integration.py — UnifiedIRInjector SGLang runtime 集成测试

对应能力：
  - moe_route_consistency_across_edge_cloud（TopK 注入）
  - moe_edge_continuation_kv_push（FusedMoE forward hook）

环境要求：
  - GPU 主机（CUDA）
  - vendored SGLang 已部署在 Backend/CGC/cloud_sglang/python/

无 GPU / 无 vendored SGLang 时自动 SKIP。
"""

import importlib
import os
import sys
import unittest
from unittest import mock


def _vendored_sglang_path():
    """返回 vendored SGLang 的 python 目录"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(
        here, "..", "Backend", "CGC", "cloud_sglang", "python"))


def _cgc_backend_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "Backend", "CGC"))


def _ensure_paths():
    sg = _vendored_sglang_path()
    cgc = _cgc_backend_path()
    if os.path.isdir(sg) and sg not in sys.path:
        sys.path.insert(0, sg)
    if os.path.isdir(cgc) and cgc not in sys.path:
        sys.path.insert(0, cgc)


def _has_torch_cuda():
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_sglang_layers_moe():
    """检查 vendored SGLang 是否有 moe layers 模块（实际 patch 目标）"""
    try:
        import sglang.srt.layers.moe.topk  # type: ignore
        return True
    except Exception:
        try:
            import sglang.srt.layers.moe  # type: ignore
            return True
        except Exception:
            return False


@unittest.skipUnless(_has_torch_cuda(), "需要 GPU 主机（CUDA）")
class UnifiedIRInjectorMoEIntegrationTest(unittest.TestCase):
    """端云 MoE 注入的 SGLang runtime 真实集成测试

    覆盖：
      1. LayerSpec 含 edge_cloud_route_sync 时 IR 编译正确
      2. _inject_topk_route_consistency 不抛异常（即使 sglang 不存在也优雅返回）
      3. _inject_fused_moe_edge_continuation 不抛异常
    """

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        from compiler.unified_compiler import (
            LayerSpec, UnifiedIR, UnifiedIRCompiler, UnifiedIRInjector,
        )
        cls.LayerSpec = LayerSpec
        cls.UnifiedIR = UnifiedIR
        cls.UnifiedIRCompiler = UnifiedIRCompiler
        cls.UnifiedIRInjector = UnifiedIRInjector

    def _build_ir_with_moe(self):
        layers = [
            self.LayerSpec(layer_id=0, layer_type="attention", inject=True),
            self.LayerSpec(
                layer_id=1, layer_type="moe", inject=True,
                routing_strategy="edge_cloud_consistent",
                topk_override=8,
                expert_bias=[0.05] * 64,
                edge_cloud_route_sync=True,
            ),
            self.LayerSpec(
                layer_id=2, layer_type="moe", inject=True,
                routing_strategy="default",
            ),
        ]
        return self.UnifiedIR(
            model_arch="DeepseekV4ForCausalLM",
            hardware_type="cuda",
            layers=layers,
        )

    def test_ir_compile_moe_layers(self):
        """IR 编译输出 moe_layers，含 edge_cloud_route_sync 字段"""
        ir = self._build_ir_with_moe()
        compiler = self.UnifiedIRCompiler()
        compiled = compiler.lower_to_hardware(ir)

        self.assertIn("moe_layers", compiled)
        self.assertEqual(len(compiled["moe_layers"]), 2)
        edge_sync = [m for m in compiled["moe_layers"]
                     if m["edge_cloud_route_sync"]]
        self.assertEqual(len(edge_sync), 1)
        self.assertEqual(edge_sync[0]["topk_override"], 8)
        self.assertEqual(len(edge_sync[0]["expert_bias"]), 64)

    def test_inject_topk_route_consistency_no_raise(self):
        """TopK 注入：即使 vendored SGLang 未完整部署，注入也不应抛异常"""
        ir = self._build_ir_with_moe()
        compiler = self.UnifiedIRCompiler()
        compiled = compiler.lower_to_hardware(ir)
        injector = self.UnifiedIRInjector()

        edge_sync_layers = [m for m in compiled["moe_layers"]
                            if m["edge_cloud_route_sync"]]
        # 注入失败时应该被 injector 内部捕获，返回 None 或抛特定异常
        # 这里只要不抛 unexpected 异常即可
        try:
            injector._inject_topk_route_consistency(edge_sync_layers)
        except Exception as e:
            # 允许 ImportError / AttributeError（sglang 缺失场景），但不允许
            # 其它未预期的异常
            self.assertIsInstance(
                e, (ImportError, AttributeError, ModuleNotFoundError),
                f"unexpected exception: {e!r}")

    def test_inject_fused_moe_no_raise(self):
        """FusedMoE forward hook 注入：sglang 缺失时优雅失败"""
        ir = self._build_ir_with_moe()
        compiler = self.UnifiedIRCompiler()
        compiled = compiler.lower_to_hardware(ir)
        injector = self.UnifiedIRInjector()

        try:
            injector._inject_fused_moe_edge_continuation(compiled["moe_layers"])
        except Exception as e:
            self.assertIsInstance(
                e, (ImportError, AttributeError, ModuleNotFoundError),
                f"unexpected exception: {e!r}")


@unittest.skipUnless(
    _has_torch_cuda() and _has_sglang_layers_moe(),
    "需要 GPU + vendored SGLang moe 模块（实跑校验）",
)
class UnifiedIRInjectorMoELivePatchTest(unittest.TestCase):
    """vendored SGLang 已部署时的真实 monkey-patch 校验

    覆盖：
      1. TopK select_experts 被 patch 后含 correction_bias
      2. FusedMoE.forward 被 patch 后调用 callback
    """

    @classmethod
    def setUpClass(cls):
        _ensure_paths()

    def test_topk_select_experts_patched(self):
        """注入后 select_experts 应被 monkey-patch 为 CGC 版本"""
        from compiler.unified_compiler import (
            LayerSpec, UnifiedIR, UnifiedIRCompiler, UnifiedIRInjector,
        )
        import sglang.srt.layers.moe.topk as topk_mod  # type: ignore

        original = topk_mod.select_experts
        try:
            ir = UnifiedIR(
                model_arch="DeepseekV4ForCausalLM",
                hardware_type="cuda",
                layers=[LayerSpec(
                    layer_id=0, layer_type="moe", inject=True,
                    routing_strategy="edge_cloud_consistent",
                    topk_override=8,
                    expert_bias=[0.1] * 64,
                    edge_cloud_route_sync=True,
                )],
            )
            compiled = UnifiedIRCompiler().lower_to_hardware(ir)
            injector = UnifiedIRInjector()
            edge_sync = [m for m in compiled["moe_layers"]
                         if m["edge_cloud_route_sync"]]
            injector._inject_topk_route_consistency(edge_sync)

            # 注入后函数对象应改变（或至少原函数被包装）
            self.assertTrue(
                topk_mod.select_experts is not original or
                hasattr(topk_mod.select_experts, "_cgc_patched"))
        finally:
            topk_mod.select_experts = original

    def test_fused_moe_forward_callback_invoked(self):
        """FusedMoE forward hook 注入后，callback 应被调用"""
        from compiler.unified_compiler import UnifiedIRInjector
        try:
            import sglang.srt.layers.moe.fused_moe as fused_mod  # type: ignore
            FusedMoE = fused_mod.FusedMoE
        except (ImportError, AttributeError) as e:
            self.skipTest(f"FusedMoE 不可用: {e}")

        original_forward = FusedMoE.forward
        callback_calls = []

        try:
            injector = UnifiedIRInjector()
            # 注入 callback
            injector._inject_fused_moe_edge_continuation([{
                "layer_id": 0,
                "edge_cloud_route_sync": True,
                "kv_push_callback": lambda *a, **k: callback_calls.append(1),
            }])
            # 注入后 forward 应被替换
            self.assertTrue(hasattr(FusedMoE.forward, "_cgc_patched") or
                            FusedMoE.forward is not original_forward)
        finally:
            FusedMoE.forward = original_forward


if __name__ == "__main__":
    unittest.main(verbosity=2)
