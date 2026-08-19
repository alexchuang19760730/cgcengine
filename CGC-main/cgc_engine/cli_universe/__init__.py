"""CLI-Universe: 面向终端智能体的可验证任务合成引擎

论文精准复现（arXiv:2606.22883）：
"CLI-Universe: Towards Verifiable Task Synthesis Engine for Terminal Agents"
Nanjing University · StepFun · ZODA · Shanghai AI Lab · HUST

三阶段流水线（论文 Figure 1, Section 3.2-3.4）：
  Step 1: Task Blueprint Construction（任务蓝图构建）
    - Candidate spec → 3D scoring (creativity/technical_grounding/feasibility)
    - Evidence-guided refinement (3.45x more solver turns, -13.3pt pass rate)
    - Blueprint rubric validation (human 72%→91%, LLM 75%→93% acceptance)
  Step 2: Environment Realization（环境物化）
    - Asset materialization (download/adapt/synthesize)
    - Docker assembly (pinned versions, services, permissions, env vars)
    - Smoke test (deps/services/filesystem/e2e; failures discarded)
  Step 3: Validation & Executable Filtering（验证与可执行过滤）
    - Rubric-gated tests (test agent role-isolated, correctness/determinism/edge cases)
    - Solution construction (solution agent with hint, NO test visibility)
    - Hint-Conditional Filter: no-hint FAIL + with-hint SUCCEED (remove trivial tasks)
    - Fail-to-Pass Check: initial FAIL → after-solution PASS (bidirectional verify)

关键实证数据：
  - 端到端候选保留率：33.6%（约2/3被丢弃，论文图2d五阶段过滤）
  - 最终数据集：CLI-Universe-6K（6000条高保真成功轨迹）
  - 教师模型：Kimi-K2.6（最优，TB2.0 33.4% for 32B），DeepSeek-V4-Pro（31.2%）
  - 表2a关键发现：仅保留成功轨迹（6K成功 > 10K未过滤全量，+5.2pt on TB2.0）
  - TMAX训练：SFT预热 + Outcome-only PPO RL（二元奖励，无过程监督）
  - TB 2.0 expected score: 9B >27%, 32B >33.4% (SOTA for ≤32B open-source)
"""

from .engine import CLIUniverseEngine, SynthesisResult, PipelineStatistics
from .skill_taxonomy import (
    TaskTaxonomy, Domain, SkillType, Capability, EngineeringPillar,
    TaskAnchor, TaskBlueprint, DifficultyLevel
)
from .scenario_retriever import EvidenceGuidedResearcher, ResearchEvidence
from .task_generator import BlueprintGenerator, CandidateTask
from .environment_validator import (
    EnvironmentRealizer, DockerEnvironment, SmokeTestResult,
    MaterializedAsset
)
from .quality_filter import (
    ExecutableFilter, RubricGatedTester, SolutionConstructor,
    HintConditionalFilter, FailToPassChecker,
    FilterResult, TestCase, TrajectoryStep, SolutionTrajectory
)
from .rl_trainer import TMAXRLTrainer, TMAXRLConfig, TrainingMetrics
from .fusionroute_agent import (
    FusionRouteAgentOrchestrator, AgentRole, TaskType,
    AgentInstance, AgentTask, RoutingDecision,
    HealthChecker, TenantManager, MiniCPM5RouterSimulator,
    create_fusionroute_agent,
)
from .agent_benchmarks import (
    OSWorldValidator, WebArenaValidator, AgentBenchmarkOrchestrator,
    BenchmarkDomain, WebArenaDomain, BenchmarkResult, BenchmarkSummary,
)
from .agent_model import (
    AgentModelBackend, RealTMAXPlanner, RealUITARSExecutor,
    create_real_agent_orchestrator,
)

__version__ = "1.0.0-paper"
__paper__ = "arXiv:2606.22883"
__all__ = [
    "CLIUniverseEngine", "SynthesisResult", "PipelineStatistics",
    "TaskTaxonomy", "Domain", "SkillType", "Capability", "EngineeringPillar",
    "TaskAnchor", "TaskBlueprint", "DifficultyLevel",
    "EvidenceGuidedResearcher", "ResearchEvidence",
    "BlueprintGenerator", "CandidateTask",
    "EnvironmentRealizer", "DockerEnvironment", "SmokeTestResult", "MaterializedAsset",
    "ExecutableFilter", "RubricGatedTester", "SolutionConstructor",
    "HintConditionalFilter", "FailToPassChecker",
    "FilterResult", "TestCase", "TrajectoryStep", "SolutionTrajectory",
    "TMAXRLTrainer", "TMAXRLConfig", "TrainingMetrics",
    "FusionRouteAgentOrchestrator", "AgentRole", "TaskType",
    "AgentInstance", "AgentTask", "RoutingDecision",
    "HealthChecker", "TenantManager", "MiniCPM5RouterSimulator",
    "create_fusionroute_agent",
    "OSWorldValidator", "WebArenaValidator", "AgentBenchmarkOrchestrator",
    "BenchmarkDomain", "WebArenaDomain", "BenchmarkResult", "BenchmarkSummary",
]
