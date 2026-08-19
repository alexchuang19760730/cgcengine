"""test_sglang_spec_plugin_integration.py — vendored SGLang 部署环境的 spec plugin 实跑校验

对应能力：
  - jetspec_sglang_verify_stack（draft→verify 端到端）
  - dspark_sglang_verify_stack
  - cgc_custom_spec_algo_registry

环境要求：
  - vendored SGLang 已部署在 Backend/CGC/cloud_sglang/python/
  - spec_registry.register_algorithm 可用

无 vendored SGLang 时自动 SKIP（部分纯逻辑测试不 SKIP）。
"""

import importlib
import os
import sys
import unittest


def _vendored_sglang_path():
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


def _has_vendored_sglang():
    _ensure_paths()
    try:
        import sglang.srt.speculative.spec_registry  # type: ignore
        return True
    except Exception:
        return False


class SGLangSpecPluginPureLogicTest(unittest.TestCase):
    """不依赖 vendored SGLang 的纯逻辑测试"""

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        from vendored.sglang_spec_plugin import (
            DraftProposal, build_target_verify_input, accept_tokens_after_verify,
        )
        cls.DraftProposal = DraftProposal
        cls.build_target_verify_input = staticmethod(build_target_verify_input)
        cls.accept_tokens_after_verify = staticmethod(accept_tokens_after_verify)

    def test_draft_proposal_dataclass(self):
        """DraftProposal 数据结构字段完整"""
        p = self.DraftProposal(
            draft_tokens=[1, 2, 3],
            source="jetspec",
            verify_length_hint=3,
        )
        self.assertEqual(p.draft_tokens, [1, 2, 3])
        self.assertEqual(p.source, "jetspec")
        self.assertEqual(p.verify_length_hint, 3)

    def test_build_target_verify_input(self):
        """build_target_verify_input 输出 SGLang target worker 期望的格式"""
        p = self.DraftProposal(draft_tokens=[10, 20, 30], source="dspark")
        vi = self.build_target_verify_input(p, target_token_id=5)
        self.assertEqual(vi["draft_tokens"], [5, 10, 20, 30])
        self.assertEqual(vi["verify_length_hint"], 3)
        self.assertIn("tree_mask", vi)

    def test_accept_tokens_after_verify_full_match(self):
        """全部匹配 → 接受全部 draft tokens

        target_logits[0] 对应 target_token_id（不参与 acceptance）
        target_logits[1..N] 对应 draft_tokens[0..N-1] 的 verify logits
        """
        import torch
        p = self.DraftProposal(draft_tokens=[10, 20, 30], source="jetspec")
        # target_logits: 4 行
        #   row 0: target_token_id 的 logits（不参与）
        #   row 1: argmax=10 → 匹配 draft[0]
        #   row 2: argmax=20 → 匹配 draft[1]
        #   row 3: argmax=30 → 匹配 draft[2]
        logits = torch.zeros(4, 100)
        logits[0, 5] = 1.0   # target_token_id (不参与)
        logits[1, 10] = 1.0  # 匹配 draft[0]
        logits[2, 20] = 1.0  # 匹配 draft[1]
        logits[3, 30] = 1.0  # 匹配 draft[2]
        accepted, n = self.accept_tokens_after_verify(logits, p)
        self.assertEqual(accepted, [10, 20, 30])
        self.assertEqual(n, 3)

    def test_accept_tokens_after_verify_partial_match(self):
        """部分匹配 → 截断

        draft=[10, 20, 30]
        target_logits[1].argmax=10 → 匹配 draft[0]
        target_logits[2].argmax=99 → 不匹配 draft[1]=20 → 截断，accepted=[10, 99]
        """
        import torch
        p = self.DraftProposal(draft_tokens=[10, 20, 30], source="jetspec")
        # target_logits: 4 行（target_token_id + 3 draft verify）
        logits = torch.zeros(4, 100)
        logits[0, 5] = 1.0   # target_token_id (不参与)
        logits[1, 10] = 1.0  # 匹配 draft[0]
        logits[2, 99] = 1.0  # 不匹配 draft[1]=20 → 截断
        logits[3, 30] = 1.0  # 不会到达
        accepted, n = self.accept_tokens_after_verify(logits, p)
        self.assertEqual(accepted, [10, 99])
        self.assertEqual(n, 2)


@unittest.skipUnless(_has_vendored_sglang(),
                    "需要 vendored SGLang（实跑校验）")
class SGLangSpecPluginRegistryTest(unittest.TestCase):
    """vendored SGLang 已部署时校验 register_cgc_spec_algos"""

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        from vendored.sglang_spec_plugin import register_cgc_spec_algos
        from sglang.srt.speculative import spec_registry  # type: ignore
        cls.register_cgc_spec_algos = staticmethod(register_cgc_spec_algos)
        cls.spec_registry = spec_registry

    def test_register_algos_returns_registered(self):
        """register_cgc_spec_algos 返回 registered=True 或 already registered"""
        res = self.register_cgc_spec_algos()
        # 首次注册返回 registered=True；已注册时返回 already registered（视为成功）
        self.assertTrue(
            res.get("registered") or res.get("reason") == "already registered",
            f"register failed: {res}")

    def test_jetspec_in_registry(self):
        """JETSPEC 算法在 _REGISTRY 中"""
        # 先尝试注册（已注册会抛 ValueError，预期）
        try:
            self.register_cgc_spec_algos()
        except ValueError:
            pass  # 已注册，正常
        algo = self.spec_registry.get_spec("JETSPEC")
        self.assertIsNotNone(algo, "JETSPEC not in registry")
        self.assertEqual(algo.name, "JETSPEC")
        self.assertTrue(algo.is_eagle())
        self.assertTrue(algo.supports_target_verify_for_draft())

    def test_dspark_in_registry(self):
        """DSPARK 算法在 _REGISTRY 中"""
        try:
            self.register_cgc_spec_algos()
        except ValueError:
            pass
        algo = self.spec_registry.get_spec("DSPARK")
        self.assertIsNotNone(algo, "DSPARK not in registry")
        self.assertEqual(algo.name, "DSPARK")
        self.assertTrue(algo.is_eagle())
        self.assertTrue(algo.supports_target_verify_for_draft())

    def test_create_worker_returns_draft_worker(self):
        """create_worker 返回 Factory，Factory() 返回 draft worker 实例"""
        try:
            self.register_cgc_spec_algos()
        except ValueError:
            pass

        # 构造 mock server_args
        class MockServerArgs:
            disable_overlap_schedule = True
            speculative_algorithm = "JETSPEC"
            speculative_num_draft_tokens = 4
            model_path = "/tmp/mock"
            spec_draft_model_path = "/tmp/mock_draft"

        jetspec_algo = self.spec_registry.get_spec("JETSPEC")
        factory = jetspec_algo.create_worker(MockServerArgs())
        # create_worker 返回 Factory 实例
        from vendored.sglang_spec_plugin import (
            JetSpecDraftWorkerFactory, DSparkDraftWorkerFactory,
        )
        self.assertIsInstance(factory, JetSpecDraftWorkerFactory)
        # Factory 调用后返回真正的 worker
        worker = factory()
        from vendored.sglang_spec_plugin import JetSpecDraftWorker
        self.assertIsInstance(worker, JetSpecDraftWorker)

        dspark_algo = self.spec_registry.get_spec("DSPARK")
        factory2 = dspark_algo.create_worker(MockServerArgs())
        self.assertIsInstance(factory2, DSparkDraftWorkerFactory)
        worker2 = factory2()
        from vendored.sglang_spec_plugin import DSparkDraftWorker
        self.assertIsInstance(worker2, DSparkDraftWorker)

    def test_reserved_name_collision(self):
        """注册保留名（如 EAGLE）应失败"""
        from sglang.srt.speculative.spec_registry import register_algorithm
        with self.assertRaises(ValueError):
            @register_algorithm("EAGLE")
            def _factory(server_args):
                return None


@unittest.skipUnless(_has_vendored_sglang(),
                    "需要 vendored SGLang（实跑校验）")
class SGLangSpecPluginEndToEndTest(unittest.TestCase):
    """draft→verify 端到端调用（不启动完整 SGLang server）"""

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        from vendored.sglang_spec_plugin import (
            register_cgc_spec_algos, run_draft_verify_round,
        )
        cls.register_cgc_spec_algos = staticmethod(register_cgc_spec_algos)
        cls.run_draft_verify_round = staticmethod(run_draft_verify_round)

    def test_run_draft_verify_round_jetspec(self):
        """单次 draft→verify 调用：JetSpec（mock draft + target worker）"""
        try:
            self.register_cgc_spec_algos()
        except ValueError:
            pass

        import torch
        from vendored.sglang_spec_plugin import DraftProposal

        # mock draft_worker：propose 返回固定 DraftProposal
        class MockDraftWorker:
            def propose(self, *, hidden_states=None):
                return DraftProposal(
                    draft_tokens=[10, 20, 30, 40],
                    source="jetspec",
                    verify_length_hint=4,
                )

        # mock target_worker：forward 返回固定 logits
        # target_logits shape = [num_tokens, vocab]
        # row 0 = target_token_id 的 logits（不参与）
        # row 1..N = draft verify logits
        class MockTargetWorker:
            def forward(self, draft_tokens, forward_batch=None):
                n = len(draft_tokens)
                logits = torch.zeros(n, 100)
                # 全匹配 draft_tokens[1:]（row 0 是 target_token_id）
                for i, t in enumerate(draft_tokens):
                    if i == 0:
                        continue  # target_token_id 行不参与 acceptance
                    logits[i, t] = 1.0
                return logits

        result = self.run_draft_verify_round(
            draft_worker=MockDraftWorker(),
            target_worker=MockTargetWorker(),
            hidden_states=torch.zeros(1, 8),
            target_token_id=5,
        )
        self.assertIn("accepted_tokens", result)
        self.assertIn("accepted_length", result)
        self.assertGreaterEqual(result["accepted_length"], 1)

    def test_run_draft_verify_round_dspark(self):
        """单次 draft→verify 调用：DSpark（mock draft + target worker）"""
        try:
            self.register_cgc_spec_algos()
        except ValueError:
            pass

        import torch
        from vendored.sglang_spec_plugin import DraftProposal

        class MockDraftWorker:
            def propose(self, *, hidden_states=None, gpu_load_factor=0.0):
                return DraftProposal(
                    draft_tokens=[10, 20, 30],
                    source="dspark",
                    verify_length_hint=3,
                )

        class MockTargetWorker:
            def forward(self, draft_tokens, forward_batch=None):
                n = len(draft_tokens)
                logits = torch.zeros(n, 100)
                for i, t in enumerate(draft_tokens):
                    if i == 0:
                        continue
                    logits[i, t] = 1.0
                return logits

        result = self.run_draft_verify_round(
            draft_worker=MockDraftWorker(),
            target_worker=MockTargetWorker(),
            hidden_states=torch.zeros(1, 8),
            target_token_id=5,
        )
        self.assertGreaterEqual(result["accepted_length"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
