"""CLI-Universe 主引擎模块 - 论文三阶段流水线（arXiv:2606.22883）

三阶段流水线（Figure 1）：
  Step 1: Task Blueprint Construction（construct_blueprints）
    - Candidate spec → evidence-guided refinement → blueprint formation
    - 三维评分（creativity/technical_grounding/feasibility），top-k 保留
    - Rubric 门控验证（人类72%→91%，LLM75%→93%）

  Step 2: Environment Realization（realize_environments）
    - Asset materialization（download/adapt/synthesize）
    - Docker assembly（pinned versions, env vars, services, permissions）
    - Smoke test（deps/services/filesystem/e2e，失败丢弃）

  Step 3: Validation & Executable Filtering（validate_and_filter）
    - Rubric-gated tests（test agent 独立构建，correctness/determinism/edge cases）
    - Solution construction（solution agent 使用 internal_hint，角色隔离）
    - Hint-Conditional Filtering：no-hint fail + with-hint succeed 才保留
    - Fail-to-Pass Checking：initial fail → after-solution pass（双向检查）

关键论文数据：
  - 端到端候选保留率：33.6%（约2/3被丢弃）
  - 最终数据集：CLI-Universe-6K（6000条成功轨迹）
  - 教师模型：Kimi-K2.6（最优，TB2.0 33.4% for 32B），DeepSeek-V4-Pro 备选
  - 表2a：仅用成功轨迹（6K）训练比全量10K（含失败）好5.2分
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .skill_taxonomy import TaskTaxonomy, TaskBlueprint, DifficultyLevel
from .scenario_retriever import EvidenceGuidedResearcher
from .task_generator import BlueprintGenerator, CandidateTask
from .environment_validator import EnvironmentRealizer, DockerEnvironment, SmokeTestResult
from .quality_filter import (
    ExecutableFilter, FilterResult, TestCase, TrajectoryStep, SolutionTrajectory,
)


@dataclass
class PipelineStatistics:
    """三阶段流水线统计信息 - 对应论文图2d五阶段过滤累积"""
    total_candidates_generated: int = 0
    candidates_after_scoring: int = 0
    blueprints_after_rubric: int = 0
    environments_after_smoke: int = 0
    tasks_after_test_construction: int = 0
    tasks_after_hint_filter: int = 0
    final_tasks_after_failpass: int = 0
    target_sft_trajectories: int = 6000

    stage1_blueprint_stats: Dict[str, float] = field(default_factory=dict)
    stage2_environment_stats: Dict[str, float] = field(default_factory=dict)
    stage3_filter_stats: Dict[str, float] = field(default_factory=dict)

    @property
    def end_to_end_retention_rate(self) -> float:
        if self.total_candidates_generated == 0:
            return 0.0
        return self.final_tasks_after_failpass / self.total_candidates_generated

    def __repr__(self) -> str:
        return (
            f"PipelineStatistics(candidates={self.total_candidates_generated}, "
            f"final={self.final_tasks_after_failpass}, "
            f"retention={self.end_to_end_retention_rate:.1%}, "
            f"sft_trajectories={self.target_sft_trajectories})"
        )


@dataclass
class SynthesisResult:
    """完整合成结果"""
    result_id: str
    candidates: List[CandidateTask]
    blueprints: List[TaskBlueprint]
    environments: List[DockerEnvironment]
    filter_results: List[FilterResult]
    passed_results: List[FilterResult]
    successful_trajectories: List[SolutionTrajectory]
    statistics: PipelineStatistics
    sft_dataset_path: Optional[str] = None
    teacher_model_used: str = "kimi-k2.6"

    def __repr__(self) -> str:
        return (
            f"SynthesisResult(id={self.result_id[:8]}, "
            f"final={len(self.passed_results)} tasks, "
            f"trajectories={len(self.successful_trajectories)}, "
            f"retention={self.statistics.end_to_end_retention_rate:.1%})"
        )


class CLIUniverseEngine:
    """CLI-Universe 可验证任务合成引擎 - 论文三阶段流水线

    流水线入口：
      - synthesize()：一键执行完整三阶段流水线
      - construct_blueprints()：仅执行 Step 1
      - realize_environments()：仅执行 Step 2
      - validate_and_filter()：仅执行 Step 3
      - integrate_with_tmax()：接入 TMAX Outcome-Only RL 训练
      - export_sft_dataset()：导出 CLI-Universe-6K 轨迹数据集

    教师模型（论文 Table 1）：
      - kimi-k2.6（默认，最优）
      - deepseek-v4-pro（备选）
    """

    def __init__(
        self,
        teacher_model: str = "kimi-k2.6",
        seed: int = 42,
        target_retention_rate: float = 0.336,
        sft_size: int = 6000,
    ):
        """
        Args:
            teacher_model: 教师模型，论文最优 "kimi-k2.6"，备选 "deepseek-v4-pro"
            seed: 随机种子
            target_retention_rate: 端到端目标保留率（论文33.6%）
            sft_size: SFT数据集大小（CLI-Universe-6K）
        """
        if teacher_model not in ("kimi-k2.6", "deepseek-v4-pro"):
            print(f"[Engine] Warning: unknown teacher model '{teacher_model}', "
                  f"falling back to kimi-k2.6")
            teacher_model = "kimi-k2.6"

        self.teacher_model = teacher_model
        self.seed = seed
        self.target_retention_rate = target_retention_rate
        self.sft_size = sft_size
        self.rng = __import__("random").Random(seed)

        self.taxonomy = TaskTaxonomy()
        self.researcher = EvidenceGuidedResearcher(seed=seed)
        self.blueprint_gen = BlueprintGenerator(
            taxonomy=self.taxonomy,
            researcher=self.researcher,
            seed=seed,
        )
        self.env_realizer = EnvironmentRealizer(seed=seed+1)
        self.quality_filter = ExecutableFilter(
            teacher_model=teacher_model,
            seed=seed+2,
        )

        self._candidates: List[CandidateTask] = []
        self._blueprints: List[TaskBlueprint] = []
        self._environments: List[DockerEnvironment] = []
        self._filter_results: List[FilterResult] = []
        self._passed_results: List[FilterResult] = []
        self._successful_trajectories: List[SolutionTrajectory] = []
        self._stats = PipelineStatistics(target_sft_trajectories=sft_size)

    def construct_blueprints(self, num_blueprints: int = 100) -> List[TaskBlueprint]:
        """Step 1: Task Blueprint Construction - 论文 Section 3.2

        流程：
        1. 基于 TaskAnchor 四维采样生成候选想法
        2. 三维评分（creativity/technical_grounding/feasibility）
        3. top-scoring 候选进入 evidence-guided refinement
        4. 注入真实技术证据（3.45× more turns）
        5. 形成 TaskBlueprint，rubric 验证（人类72%→91%，LLM75%→93%）

        Returns:
            通过 rubric 验证的 TaskBlueprint 列表
        """
        print("[Stage 1/3] Task Blueprint Construction")
        print("  = Candidate generation + 3D scoring + top-k selection")
        print("  = Evidence-guided refinement (3.45x more turns)")
        print("  = Blueprint formation + rubric validation")

        blueprints = self.blueprint_gen.construct_blueprints(num_blueprints=num_blueprints)
        candidates = self.blueprint_gen.get_candidates()

        self._candidates = candidates
        self._blueprints = blueprints
        self._stats.total_candidates_generated = len(candidates)
        self._stats.candidates_after_scoring = len(candidates)
        self._stats.blueprints_after_rubric = len(blueprints)
        self._stats.stage1_blueprint_stats = self.blueprint_gen.get_statistics()

        print(f"  [Stage1] Generated {len(candidates)} scored candidates, "
              f"{len(blueprints)} blueprints passed rubric "
              f"({len(blueprints)/max(len(candidates),1):.1%} acceptance)")
        return blueprints

    def realize_environments(
        self,
        blueprints: Optional[List[TaskBlueprint]] = None,
    ) -> List[DockerEnvironment]:
        """Step 2: Environment Realization - 论文 Section 3.3

        流程：
        1. Asset Materialization：下载/适配/合成所需资产（注入受控故障）
        2. Docker Assembly：pinned versions、env vars、services、permissions
        3. Smoke Test：deps/services/filesystem/e2e 四项检查，失败丢弃

        Returns:
            通过 smoke test 的 DockerEnvironment 列表
        """
        if blueprints is None:
            blueprints = self._blueprints
        if not blueprints:
            blueprints = self.construct_blueprints()

        print("[Stage 2/3] Environment Realization")
        print("  = Asset materialization (download/adapt/synthesize)")
        print("  = Docker assembly (pinned versions, services, permissions)")
        print("  = Smoke test (deps/services/filesystem/e2e), failures discarded")

        envs = self.env_realizer.realize_environments(blueprints)
        self._environments = envs
        self._stats.environments_after_smoke = len(envs)
        self._stats.stage2_environment_stats = self.env_realizer.get_statistics()

        print(f"  [Stage2] Realized {len(envs)} environments passed smoke test "
              f"({len(envs)/max(len(blueprints),1):.1%} pass)")
        return envs

    def validate_and_filter(
        self,
        blueprints: Optional[List[TaskBlueprint]] = None,
        environments: Optional[List[DockerEnvironment]] = None,
    ) -> List[FilterResult]:
        """Step 3: Validation & Executable Filtering - 论文 Section 3.4

        四个关键组件，角色隔离严格保证：
        1. RubricGatedTester：test agent 独立构建测试（看不到 hint）
           - 覆盖 correctness/determinism/edge cases，迭代精化直到稳定
        2. SolutionConstructor：solution agent 构建解答（看不到 test 输出）
           - with-hint rollout（教师模型，应成功）
           - no-hint rollout（模拟学生，应失败）
        3. HintConditionalFilter：no-hint fail + with-hint succeed 才保留
           - 移除 trivial-to-solve 任务（平凡可解）
        4. FailToPassChecker：initial fail → after-solution pass（双向检查）
           - 初始环境测试必须失败（证明故障存在）
           - 执行 solution 后测试必须通过（证明修复有效）

        Returns:
            通过所有过滤器的 FilterResult 列表
        """
        if environments is None:
            environments = self._environments
        if not environments:
            environments = self.realize_environments(blueprints)
        if blueprints is None:
            blueprints = self._blueprints

        bp_by_id = {bp.blueprint_id: bp for bp in blueprints}
        bp_env_pairs = []
        for env in environments:
            bp = bp_by_id.get(env.blueprint_id)
            if bp:
                bp_env_pairs.append((bp, env))

        print("[Stage 3/3] Validation & Executable Filtering")
        print("  = Rubric-gated test construction (test agent, role-isolated)")
        print("  = Solution construction (solution agent with hint, no test visibility)")
        print("  = Hint-Conditional Filter: no-hint FAIL + with-hint SUCCEED")
        print("  = Fail-to-Pass Check: initial FAIL -> after-solution PASS")

        passed = self.quality_filter.validate_and_filter(bp_env_pairs)
        all_results = self.quality_filter.get_all_results()
        trajs = self.quality_filter.get_successful_trajectories(limit=self.sft_size)

        self._filter_results = all_results
        self._passed_results = passed
        self._successful_trajectories = trajs
        self._stats.tasks_after_test_construction = sum(
            1 for r in all_results if r.stage_retained.get("stage3_test_construction")
        )
        self._stats.tasks_after_hint_filter = sum(
            1 for r in all_results if r.stage_retained.get("stage4_hint_conditional")
        )
        self._stats.final_tasks_after_failpass = len(passed)
        self._stats.stage3_filter_stats = self.quality_filter.get_statistics()

        print(f"  [Stage3] {len(passed)} tasks passed all filters "
              f"({len(passed)/max(len(bp_env_pairs),1):.1%} stage pass, "
              f"{self._stats.end_to_end_retention_rate:.1%} end-to-end)")
        return passed

    def synthesize(
        self,
        num_initial_candidates: int = 200,
        export_sft: bool = True,
        sft_export_path: Optional[str] = None,
    ) -> SynthesisResult:
        """一键执行完整三阶段流水线

        Args:
            num_initial_candidates: 初始候选数量（最终约保留 num * 33.6%）
            export_sft: 是否导出 SFT 数据集
            sft_export_path: SFT 导出路径

        Returns:
            SynthesisResult 包含所有产物和统计
        """
        print("=" * 70)
        print("CLI-Universe Verifiable Task Synthesis Engine (arXiv:2606.22883)")
        print(f"Teacher model: {self.teacher_model}")
        print(f"Target end-to-end retention: {self.target_retention_rate:.1%}")
        print("=" * 70)

        blueprints = self.construct_blueprints(num_blueprints=num_initial_candidates)
        environments = self.realize_environments(blueprints)
        passed = self.validate_and_filter(blueprints, environments)

        sft_path = None
        if export_sft:
            sft_path = self.export_sft_dataset(sft_export_path)

        result = SynthesisResult(
            result_id=uuid.uuid4().hex,
            candidates=self._candidates,
            blueprints=self._blueprints,
            environments=self._environments,
            filter_results=self._filter_results,
            passed_results=self._passed_results,
            successful_trajectories=self._successful_trajectories,
            statistics=self._stats,
            sft_dataset_path=sft_path,
            teacher_model_used=self.teacher_model,
        )

        print("=" * 70)
        print(f"Synthesis complete: {len(passed)} high-quality CLI tasks")
        print(f"SFT trajectories: {len(self._successful_trajectories)} "
              f"(CLI-Universe-{min(len(self._successful_trajectories), self.sft_size)}K)")
        print(f"End-to-end retention: {self._stats.end_to_end_retention_rate:.1%} "
              f"(paper: 33.6%)")
        if sft_path:
            print(f"SFT dataset exported to: {sft_path}")
        print("=" * 70)
        return result

    def integrate_with_tmax(
        self,
        trajectories: Optional[List[SolutionTrajectory]] = None,
        model_size: str = "9b",
        run_training: bool = True,
    ):
        """整合 TMAX Outcome-Only RL 训练 - 论文 Section 4 / Table 2b

        训练流程（TMAX）：
        1. SFT 预热：使用 CLI-Universe-6K 成功轨迹做监督微调
        2. Outcome-only RL：二元奖励（成功=1，失败=0），无过程监督
        3. PPO 算法优化策略

        论文结果（Table 2b, TB 2.0 expected score）：
          - 9B model: >27%
          - 32B model: >33.4% (SOTA for ≤32B open-source)

        Args:
            trajectories: 成功轨迹，None则使用流水线产出
            model_size: 模型大小 "9b" 或 "32b"
            run_training: 是否实际执行训练

        Returns:
            (trainer, metrics)
        """
        from .rl_trainer import TMAXRLTrainer, TMAXRLConfig

        if trajectories is None:
            trajectories = self._successful_trajectories
        if not trajectories:
            self.synthesize(export_sft=False)
            trajectories = self._successful_trajectories

        print(f"[TMAX Integration] Initializing TMAX RL trainer ({model_size})")
        print(f"  SFT warmup data: {len(trajectories)} successful trajectories")
        print(f"  Reward: outcome-only (success=1, fail=0), no process supervision")
        print(f"  Target TB 2.0: "
              f"{'>27%' if model_size == '9b' else '>33.4%'} for {model_size}")

        config = TMAXRLConfig(
            base_model=f"tmax-{model_size}",
            teacher_model=self.teacher_model,
            seed=self.seed,
        )
        trainer = TMAXRLTrainer(config=config)

        metrics = None
        if run_training:
            metrics = trainer.train_pipeline(trajectories)
            print(f"[TMAX] Training complete. Final metrics: {metrics}")

        return trainer, metrics

    def export_sft_dataset(self, output_path: Optional[str] = None) -> str:
        """导出 SFT 训练数据集 - CLI-Universe-6K（trajectories.jsonl）

        论文表2a关键发现：
        "Training on 6K successful trajectories outperforms training on the full
        10K set (including failures) by 5.2 points on Terminal-Bench 2.0."

        因此我们只导出成功轨迹（with_hint=True 且 final_success=True）。

        数据格式（JSONL，每行一个样本）：
        {
          "instruction": <user_query>,
          "output": <完整 shell 解答脚本 from trajectory steps>,
          "metadata": {
            "trajectory_id": ...,
            "blueprint_id": ...,
            "difficulty": ...,
            "num_steps": ...,
            "domain": ..., "skill_type": ...,
            "capability": ..., "pillar": ...,
            "scores": {creativity, technical_grounding, feasibility},
            "teacher_model": ...
          }
        }
        """
        if not self._successful_trajectories:
            raise RuntimeError("Pipeline must run and produce trajectories before export. "
                             "Call synthesize() or validate_and_filter() first.")

        if output_path is None:
            output_dir = os.path.join(".", "cli_universe_sft_dataset")
            os.makedirs(output_dir, exist_ok=True)
            n = min(len(self._successful_trajectories), self.sft_size)
            output_path = os.path.join(output_dir, f"trajectories_{n}.jsonl")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        bp_by_id = {bp.blueprint_id: bp for bp in self._blueprints}

        with open(output_path, 'w', encoding='utf-8') as f:
            for traj in self._successful_trajectories:
                bp = bp_by_id.get(traj.blueprint_id)
                cmd_list = [s.command for s in traj.steps if s.command]
                solution_script = "#!/bin/bash\nset -euo pipefail\ncd /workspace/task\n\n"
                solution_script += "\n".join(cmd_list) + "\n"

                metadata = {
                    "trajectory_id": traj.trajectory_id,
                    "blueprint_id": traj.blueprint_id,
                    "num_steps": traj.total_steps,
                    "used_hint": traj.used_hint,
                    "teacher_model": self.teacher_model,
                }
                if bp:
                    metadata.update({
                        "difficulty": bp.difficulty.value,
                        "domain": bp.anchor.domain.value,
                        "skill_type": bp.anchor.skill_type.value,
                        "capability": bp.anchor.capability.value,
                        "engineering_pillar": bp.anchor.engineering_pillar.value,
                        "estimated_turns": bp.estimated_turns,
                        "scores": {
                            "creativity": bp.creativity_score,
                            "technical_grounding": bp.technical_grounding_score,
                            "feasibility": bp.feasibility_score,
                        },
                    })

                user_query = bp.user_query if bp else ""
                sample = {
                    "instruction": user_query,
                    "input": "",
                    "output": solution_script,
                    "metadata": metadata,
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        n_exported = min(len(self._successful_trajectories), self.sft_size)
        print(f"[Export] Exported {n_exported} successful trajectories to: {output_path}")
        print(f"  (Only successful trajectories exported - paper Table 2a: +5.2pt vs 10K full set)")
        return output_path

    def get_statistics(self) -> PipelineStatistics:
        return self._stats

    def get_successful_trajectories(self) -> List[SolutionTrajectory]:
        return list(self._successful_trajectories)

    def print_pipeline_summary(self):
        """打印五阶段保留率统计（对应论文图2d）"""
        stats = self._stats
        print("=" * 60)
        print("Pipeline Retention Summary (cf. paper Figure 2d)")
        print("=" * 60)
        print(f"  Input candidates:        {stats.total_candidates_generated}")
        print(f"  After scoring/top-k:     {stats.candidates_after_scoring} "
              f"({stats.candidates_after_scoring/max(stats.total_candidates_generated,1):.1%})")
        print(f"  After blueprint rubric:  {stats.blueprints_after_rubric} "
              f"({stats.blueprints_after_rubric/max(stats.total_candidates_generated,1):.1%})")
        print(f"  After smoke test:        {stats.environments_after_smoke} "
              f"({stats.environments_after_smoke/max(stats.total_candidates_generated,1):.1%})")
        print(f"  After test construction: {stats.tasks_after_test_construction} "
              f"({stats.tasks_after_test_construction/max(stats.total_candidates_generated,1):.1%})")
        print(f"  After hint-conditional:  {stats.tasks_after_hint_filter} "
              f"({stats.tasks_after_hint_filter/max(stats.total_candidates_generated,1):.1%})")
        print(f"  After fail-to-pass:      {stats.final_tasks_after_failpass} "
              f"({stats.end_to_end_retention_rate:.1%})  <-- Final (paper: 33.6%)")
        print(f"  SFT trajectories:        {len(self._successful_trajectories)} "
              f"(CLI-Universe-6K)")
        print("=" * 60)

    def __repr__(self) -> str:
        return (
            f"CLIUniverseEngine(teacher={self.teacher_model}, "
            f"candidates={len(self._candidates)}, "
            f"blueprints={len(self._blueprints)}, "
            f"final={len(self._passed_results)}, "
            f"retention={self._stats.end_to_end_retention_rate:.1%})"
        )
