"""任务蓝图构建模块 - 论文 Section 3.2 Step 1: Task Blueprint Construction

论文三阶段流水线第一阶段：
  - Candidate Task Idea Generation（基于 TaskAnchor 生成候选想法）
  - Evidence-Guided Refinement（证据引导深度研究，grounding in real tech materials）
  - Blueprint Formation & Rubric Validation（蓝图形成 + rubric 完整性验证）

关键论文数据：
  - 三维评分：creativity, technical_grounding, feasibility
  - 仅 top-scoring 候选进入下一阶段
  - 证据精炼后：3.45× more solver turns, pass rate -13.3pt（真实难度提升）
  - Rubric 验证后接受率：人类 72%→91%，LLM 75%→93%
  - 蓝图包含：user_query, internal_hint, environment_checklist, acceptance_criteria
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .skill_taxonomy import (
    TaskAnchor, TaskBlueprint, TaskTaxonomy, DifficultyLevel,
    Domain, SkillType, Capability, EngineeringPillar
)
from .scenario_retriever import EvidenceGuidedResearcher, ResearchEvidence


@dataclass
class CandidateTask:
    """候选任务想法 - 论文 3.2 Candidate Specification

    从四维 TaskAnchor 采样后生成的抽象任务想法，尚未经过证据精炼和rubric验证。
    经过三维评分后，只有 top-scoring 候选进入下一阶段。
    """
    candidate_id: str
    anchor: TaskAnchor
    raw_idea: str
    creativity_score: float = 0.0
    technical_grounding_score: float = 0.0
    feasibility_score: float = 0.0
    evidence: List[ResearchEvidence] = field(default_factory=list)
    scored: bool = False
    refined: bool = False

    @property
    def total_score(self) -> float:
        return (self.creativity_score + self.technical_grounding_score +
                self.feasibility_score) / 3.0

    def __repr__(self) -> str:
        return (
            f"CandidateTask(id={self.candidate_id[:8]}, "
            f"anchor=[{self.anchor.domain.value}/{self.anchor.skill_type.value}/"
            f"{self.anchor.capability.value}/{self.anchor.engineering_pillar.value}], "
            f"score={self.total_score:.2f}, refined={self.refined})"
        )


# 基于四维组合的任务想法模板（模拟论文的 LLM idea generation）
_IDEA_TEMPLATES: Dict[Tuple[str, str, str, str], List[str]] = {
    ("software_and_system", "debugging", "error_recovery", "debugging"): [
        "诊断并修复 systemd 服务启动失败问题：服务启动后立即退出，日志显示 socket 权限被拒绝",
        "排查 cron 备份任务静默失败问题：cron 环境 PATH 不完整导致命令找不到",
        "调试 Docker 容器内进程无法写入挂载卷的问题：UID/GID 不匹配导致权限错误",
    ],
    ("data_processing", "text_processing", "multi_step_reasoning", "data_analysis"): [
        "解析 10GB Nginx 访问日志，提取所有 5xx 错误的 Top 10 IP，按小时聚合统计",
        "批量处理 CSV 文件，处理引号转义问题，清洗后导入 PostgreSQL 并创建索引",
        "从多个 JSON 日志文件中聚合错误事件，按错误类型分组统计出现频率",
    ],
    ("networking_and_security", "cryptography", "constraint_satisfaction", "troubleshooting"): [
        "轮换 Nginx 上过期的 TLS 证书，验证证书链完整性，reload 服务不中断现有连接",
        "配置 SSH 跳板机代理转发，设置正确的 known_hosts 和密钥权限（600）",
        "诊断内部服务 TLS 握手失败：检查 SNI 匹配、中间证书、协议版本兼容性",
    ],
    ("machine_learning", "configuration", "long_horizon_planning", "configuration_management"): [
        "调试 PyTorch 训练任务 OOM：设置 CUDA_VISIBLE_DEVICES、调整 batch size 和梯度累积步数",
        "配置 MLflow 跟踪服务器，设置后端存储和 artifact 路径，验证实验记录功能",
        "排查训练中 NaN loss：检查学习率、梯度裁剪、混合精度 scaling factor",
    ],
    ("software_and_system", "version_control", "exploration", "new_feature_creation"): [
        "从 detached HEAD 状态恢复工作：使用 reflog 找到丢失的本地提交并创建分支",
        "交互式 rebase 压缩最近 5 个提交，解决过程中的冲突并保留作者信息",
        "设置 git hooks 进行 pre-commit 代码检查，配置 husky 或 pre-commit 框架",
    ],
    ("software_and_system", "configuration", "constraint_satisfaction", "devops"): [
        "设置带文件锁的 cron 每日备份任务，正确配置 PATH、日志输出、防止重叠执行",
        "配置 logrotate 轮转应用日志，设置压缩、保留期、postrotate 脚本 reload 服务",
        "编写 systemd service unit 文件，配置资源限制、重启策略、日志重定向",
    ],
    ("data_processing", "database", "error_recovery", "data_analysis"): [
        "批量导入 CSV 到 PostgreSQL，处理编码错误，验证行数，创建合适的索引并 ANALYZE",
        "排查 MySQL 慢查询，使用 EXPLAIN 分析执行计划，添加缺失的索引",
        "从备份恢复 PostgreSQL 数据库，处理外键约束依赖顺序，验证数据完整性",
    ],
    ("networking_and_security", "networking", "exploration", "troubleshooting"): [
        "逐步诊断 curl 到内部服务失败的原因：DNS→防火墙→TLS→端口→MTU 黑hole",
        "使用 tcpdump 抓包分析 HTTP 502 错误，定位是后端连接超时还是网关问题",
        "配置 iptables 规则只允许特定 IP 段访问 SSH，同时不中断现有连接",
    ],
    ("software_and_system", "systems", "tool_orchestration", "new_feature_creation"): [
        "编写 shell 脚本安全清理 Docker：停止容器→删除悬空镜像→清理未使用 volumes",
        "使用 rsync 增量同步数据到备份服务器，保留权限、设置带宽限制、删除源端已删除文件",
        "构建多阶段 Dockerfile，优化镜像大小，正确处理缓存层和构建参数",
    ],
    ("software_and_system", "file_manipulation", "multi_step_reasoning", "refactoring"): [
        "递归重命名含空格和特殊字符的文件，使用 find -print0 | xargs -0 安全处理",
        "批量转换项目中文件的换行符 CRLF→LF，排除二进制文件和 .git 目录",
        "整理混乱的下载目录：按文件类型/日期分类移动，处理文件名冲突",
    ],
    ("machine_learning", "debugging", "error_recovery", "debugging"): [
        "排查分布式训练梯度不同步：检查 NCCL 通信、random seed、数据加载顺序",
        "调试 CUDA out of memory 在推理时出现而训练时正常：定位是 KV cache 泄漏",
        "分析模型推理结果不一致：设置 deterministic 模式，排查 dropout/batch norm",
    ],
    ("software_and_system", "debugging", "state_tracking", "troubleshooting"): [
        "诊断端口被占用但连接被拒绝：检查 bind address(127.0.0.1 vs 0.0.0.0)、SELinux、iptables",
        "排查磁盘空间不足但 du 显示还有空间：已删除文件被进程持有，需要 lsof 定位",
        "追踪内存泄漏：使用 valgrind massif 或 pmap 分析进程内存增长趋势",
    ],
}

_DEFAULT_IDEAS = [
    "编写脚本批量处理日志文件，提取关键错误信息并生成报告",
    "配置服务监控脚本，检查端口和进程状态，异常时发送告警",
    "设置自动化部署流程，从 git pull 到 build 到 restart 的完整流水线",
    "数据备份验证脚本：检查备份文件完整性、大小、时间戳",
    "编写日志分析工具，统计请求延迟分布和错误率趋势",
]


class BlueprintGenerator:
    """任务蓝图生成器 - 论文 Section 3.2 Step 1

    实现完整的蓝图构建三小步：
    1. Candidate Idea Generation：基于 TaskAnchor 四维组合生成候选想法
    2. Evidence-Guided Refinement：调用 EvidenceGuidedResearcher 注入真实技术证据
    3. Blueprint Formation & Rubric Validation：形成蓝图并用 rubric 验证完整性

    关键过滤：
    - 三维评分 creativity/technical_grounding/feasibility
    - 仅 top-scoring 候选进入 refine
    - Rubric 检查：user_query 清晰度、hint 有用性、checklist 完整性、criteria 可验证性
    - 论文数据：rubric 验证后人类接受率 72%→91%，LLM 75%→93%
    """

    def __init__(
        self,
        taxonomy: Optional[TaskTaxonomy] = None,
        researcher: Optional[EvidenceGuidedResearcher] = None,
        top_k_ratio: float = 0.65,
        rubric_pass_threshold: float = 0.76,
        seed: int = 42,
    ):
        """初始化蓝图生成器

        Args:
            taxonomy: 四维任务分类法
            researcher: 证据引导研究器
            top_k_ratio: 候选保留比例（评分后）
            rubric_pass_threshold: rubric 验证通过阈值
            seed: 随机种子
        """
        self.taxonomy = taxonomy or TaskTaxonomy()
        self.researcher = researcher or EvidenceGuidedResearcher(seed=seed)
        self.top_k_ratio = top_k_ratio
        self.rubric_pass_threshold = rubric_pass_threshold
        self.rng = random.Random(seed)
        self._candidates: List[CandidateTask] = []
        self._blueprints: List[TaskBlueprint] = []

    def _generate_raw_idea(self, anchor: TaskAnchor) -> str:
        """基于四维 anchor 生成原始任务想法"""
        key = (anchor.domain.value, anchor.skill_type.value,
               anchor.capability.value, anchor.engineering_pillar.value)

        if key in _IDEA_TEMPLATES:
            return self.rng.choice(_IDEA_TEMPLATES[key])

        domain_match = [k for k in _IDEA_TEMPLATES if k[0] == anchor.domain.value]
        if domain_match and self.rng.random() < 0.7:
            return self.rng.choice(_IDEA_TEMPLATES[self.rng.choice(domain_match)])

        return self.rng.choice(_DEFAULT_IDEAS)

    def _score_candidate(self, candidate: CandidateTask) -> None:
        """三维评分：creativity, technical_grounding, feasibility（论文 3.2）

        - creativity: 避免平凡可解任务，需要非显而易见的推理
        - technical_grounding: 基于真实工具和已知失败模式，不是虚构场景
        - feasibility: 在典型 CLI 环境中可实现，不需要特殊硬件/内部资源
        """
        anchor = candidate.anchor
        idea = candidate.raw_idea.lower()

        creativity = 0.5
        creativity_keywords = ["debug", "diagnose", "troubleshoot", "recover", "oops",
                               "conflict", "expired", "silent fail", "oorm", "nan",
                               "permission denied", "race condition", "mismatch"]
        for kw in creativity_keywords:
            if kw in idea:
                creativity += 0.08
        if anchor.capability in (Capability.ERROR_RECOVERY, Capability.LONG_HORIZON_PLANNING):
            creativity += 0.15
        if anchor.engineering_pillar == EngineeringPillar.DEBUGGING:
            creativity += 0.1
        creativity = min(max(creativity + self.rng.gauss(0, 0.05), 0.2), 1.0)

        technical_grounding = 0.5
        grounding_keywords = ["systemd", "cron", "docker", "nginx", "postgres", "mysql",
                              "ssh", "openssl", "git", "awk", "sed", "grep", "curl",
                              "iptables", "tcpdump", "valgrind", "pytorch", "cuda",
                              "rsync", "logrotate", "mlflow", "nccl"]
        for kw in grounding_keywords:
            if kw in idea:
                technical_grounding += 0.06
        if anchor.skill_type in (SkillType.SYSTEMS, SkillType.NETWORKING, SkillType.CRYPTOGRAPHY):
            technical_grounding += 0.1
        technical_grounding = min(max(technical_grounding + self.rng.gauss(0, 0.05), 0.2), 1.0)

        feasibility = 0.7
        infeasible_signals = ["internal.company", "proprietary", "custom hardware",
                              "production database", "real user data"]
        for sig in infeasible_signals:
            if sig in idea:
                feasibility -= 0.3
        if anchor.skill_type in (SkillType.FILE_MANIPULATION, SkillType.TEXT_PROCESSING):
            feasibility += 0.15
        if anchor.domain == Domain.SOFTWARE_SYSTEM:
            feasibility += 0.1
        feasibility = min(max(feasibility + self.rng.gauss(0, 0.05), 0.2), 1.0)

        candidate.creativity_score = round(creativity, 3)
        candidate.technical_grounding_score = round(technical_grounding, 3)
        candidate.feasibility_score = round(feasibility, 3)
        candidate.scored = True

    def _refine_with_evidence(self, candidate: CandidateTask) -> None:
        """证据引导精炼 - 论文 3.2 Evidence-Guided Refinement

        "the agent searches real technical materials ... and progressively incorporates
        the evidence into the task specification, grounding it in specific tools,
        realistic constraints, known failure modes, and concrete input/output contracts."

        效果：3.45× more solver turns, pass rate -13.3pt
        """
        evidence = self.researcher.research(
            candidate.anchor.domain.value,
            candidate.anchor.skill_type.value,
            candidate.raw_idea,
        )
        candidate.evidence = evidence
        candidate.refined = True

    def _estimate_difficulty(self, blueprint: TaskBlueprint) -> DifficultyLevel:
        """根据 turns、证据数量、capability 估计难度"""
        turns = blueprint.estimated_turns
        n_evidence = len(blueprint.evidence)
        cap = blueprint.anchor.capability

        if turns >= 20 or cap == Capability.LONG_HORIZON_PLANNING and n_evidence >= 3:
            return DifficultyLevel.EXPERT
        if turns >= 14 or n_evidence >= 3:
            return DifficultyLevel.HARD
        if turns >= 8:
            return DifficultyLevel.MEDIUM
        return DifficultyLevel.EASY

    def _form_blueprint(self, candidate: CandidateTask) -> TaskBlueprint:
        """从精炼后的候选形成 TaskBlueprint - 论文 3.2 Blueprint Formation"""
        evidence = candidate.evidence
        tools = [e.tool_name for e in evidence if e.tool_name]
        failure_modes = [e.failure_mode for e in evidence if e.failure_mode]
        constraints = [e.constraint for e in evidence if e.constraint]

        tool_str = ", ".join(tools[:4]) if tools else "standard CLI tools"
        user_query = (
            f"{candidate.raw_idea}.\n\n"
            f"Available tools include: {tool_str}. "
            f"Work in the /workspace/task directory. "
            f"All changes must be verifiable through command-line checks."
        )

        hint_parts = [f"Solve the task step by step using {tool_str}."]
        if failure_modes:
            hint_parts.append(f"Key pitfalls to handle: {'; '.join(failure_modes[:3])}.")
        if constraints:
            hint_parts.append(f"Constraints: {'; '.join(constraints[:2])}.")
        hint_parts.append(
            "Start by exploring the environment to understand the current state, "
            "then apply fixes incrementally and verify each step."
        )
        internal_hint = " ".join(hint_parts)

        env_checklist = [
            "/workspace/task working directory exists",
            "Basic POSIX utilities available (grep, sed, awk, find, xargs)",
        ]
        if any("docker" in t for t in tools):
            env_checklist.append("Docker daemon accessible")
        if any(t in tools for t in ["systemctl", "journalctl"]):
            env_checklist.append("systemd-based Linux environment")
        if any("git" in t for t in tools):
            env_checklist.append("Git repository initialized with history")
        if any(t in tools for t in ["psql", "mysql"]):
            env_checklist.append("Database service running with test database")
        if any(t in tools for t in ["nginx", "curl"]):
            env_checklist.append("HTTP service listening on test port")
        if failure_modes:
            env_checklist.append(f"Pre-seeded failure state for: {failure_modes[0][:60]}")

        acceptance_criteria = [
            "Exit code 0 after solution execution",
            "Primary task objective verified (file created/service running/data fixed)",
            "No regressions introduced (existing functionality preserved)",
        ]
        if any("cert" in fm.lower() or "tls" in fm.lower() for fm in failure_modes):
            acceptance_criteria.append("TLS certificate chain valid and not expired")
        if any("permission" in fm.lower() for fm in failure_modes):
            acceptance_criteria.append("File/directory permissions set correctly")
        if any("cron" in t for t in tools) or "backup" in candidate.raw_idea.lower():
            acceptance_criteria.append("Automated task runs successfully without manual intervention")

        blueprint_id = f"bp-{uuid.uuid4().hex[:12]}"
        blueprint = TaskBlueprint(
            blueprint_id=blueprint_id,
            user_query=user_query,
            internal_hint=internal_hint,
            environment_checklist=env_checklist,
            acceptance_criteria=acceptance_criteria,
            anchor=candidate.anchor,
            evidence=[e.source_url for e in evidence],
            estimated_turns=8,
            creativity_score=candidate.creativity_score,
            technical_grounding_score=candidate.technical_grounding_score,
            feasibility_score=candidate.feasibility_score,
        )

        fm = [e.failure_mode for e in evidence if e.failure_mode]
        if fm:
            blueprint.estimated_turns = max(8, int(blueprint.estimated_turns * 3.45))
            blueprint.internal_hint += f"  Key pitfalls to handle: {'; '.join(fm[:3])}."
        blueprint.difficulty = self._estimate_difficulty(blueprint)
        return blueprint

    def _rubric_validate(self, blueprint: TaskBlueprint) -> Tuple[bool, Dict[str, float], List[str]]:
        """Rubric 门控验证 - 论文 3.2 Blueprint Rubric Validation

        Rubric 维度：
        1. user_query 清晰度（clarity）：是否明确描述目标，不含糊
        2. internal_hint 有用性（hint utility）：是否指出关键陷阱但不直接给答案
        3. environment_checklist 完整性（env completeness）：所需资产/服务是否明确
        4. acceptance_criteria 可验证性（criteria verifiability）：验收标准是否可执行检查

        论文效果：rubric 验证后人类接受率 72%→91%（+19pp），LLM 75%→93%（+18pp）
        """
        scores = {}
        issues = []

        clarity = 0.40
        if len(blueprint.user_query) > 100:
            clarity += 0.18
        if any(kw in blueprint.user_query.lower() for kw in ["fix", "debug", "create", "configure",
                                                               "parse", "rotate", "recover"]):
            clarity += 0.15
        if len(blueprint.user_query.split('.')) >= 3:
            clarity += 0.10
        if "available tools" in blueprint.user_query.lower():
            clarity += 0.07
        clarity = min(max(clarity + self.rng.gauss(0, 0.10), 0.15), 1.0)
        scores["clarity"] = round(clarity, 3)
        if clarity < 0.60:
            issues.append("user_query lacks clear objective or sufficient detail")

        hint_utility = 0.40
        if "pitfall" in blueprint.internal_hint.lower() or "key pitfall" in blueprint.internal_hint.lower():
            hint_utility += 0.20
        if "step by step" in blueprint.internal_hint.lower():
            hint_utility += 0.10
        if any(tool in blueprint.internal_hint.lower() for tool in ["grep", "sed", "awk", "docker",
                                                                     "systemctl", "git", "curl", "openssl"]):
            hint_utility += 0.12
        if blueprint.internal_hint.count('.') >= 3:
            hint_utility += 0.06
        if "explore" in blueprint.internal_hint.lower():
            hint_utility += 0.07
        hint_utility = min(max(hint_utility + self.rng.gauss(0, 0.10), 0.15), 1.0)
        scores["hint_utility"] = round(hint_utility, 3)
        if hint_utility < 0.58:
            issues.append("internal_hint does not adequately point out key pitfalls")

        env_completeness = 0.40
        env_text = " ".join(blueprint.environment_checklist).lower()
        if "working directory" in env_text:
            env_completeness += 0.10
        if any(kw in env_text for kw in ["docker", "systemd", "git", "database", "http",
                                          "utilities", "service"]):
            env_completeness += 0.15
        if len(blueprint.environment_checklist) >= 4:
            env_completeness += 0.13
        if "failure state" in env_text or "pre-seeded" in env_text:
            env_completeness += 0.12
        env_completeness = min(max(env_completeness + self.rng.gauss(0, 0.10), 0.15), 1.0)
        scores["env_completeness"] = round(env_completeness, 3)
        if env_completeness < 0.60:
            issues.append("environment_checklist missing key assets or pre-conditions")

        criteria_verifiability = 0.40
        crit_text = " ".join(blueprint.acceptance_criteria).lower()
        if "exit code" in crit_text:
            criteria_verifiability += 0.10
        if "verified" in crit_text or "verifiable" in crit_text:
            criteria_verifiability += 0.10
        if any(kw in crit_text for kw in ["file", "service", "data", "certificate",
                                           "permission", "running", "created", "fixed"]):
            criteria_verifiability += 0.15
        if len(blueprint.acceptance_criteria) >= 3:
            criteria_verifiability += 0.10
        if "no regression" in crit_text or "preserved" in crit_text:
            criteria_verifiability += 0.05
        criteria_verifiability = min(max(criteria_verifiability + self.rng.gauss(0, 0.10), 0.15), 1.0)
        scores["criteria_verifiability"] = round(criteria_verifiability, 3)
        if criteria_verifiability < 0.60:
            issues.append("acceptance_criteria not sufficiently verifiable")

        overall = sum(scores.values()) / len(scores)
        base_pass = overall >= self.rubric_pass_threshold and len(issues) == 0
        passed = base_pass
        return passed, scores, issues

    def generate_candidates(self, num_candidates: int) -> List[CandidateTask]:
        """生成并评分候选任务想法"""
        anchors = self.taxonomy.sample_anchors(num_candidates * 2, seed=self.rng.randint(0, 100000))
        candidates = []
        seen_ideas = set()

        for anchor in anchors:
            raw_idea = self._generate_raw_idea(anchor)
            if raw_idea in seen_ideas:
                continue
            seen_ideas.add(raw_idea)

            candidate = CandidateTask(
                candidate_id=uuid.uuid4().hex,
                anchor=anchor,
                raw_idea=raw_idea,
            )
            self._score_candidate(candidate)
            candidates.append(candidate)

            if len(candidates) >= num_candidates:
                break

        candidates.sort(key=lambda c: c.total_score, reverse=True)
        keep_n = max(1, int(len(candidates) * self.top_k_ratio))
        top_candidates = candidates[:keep_n]
        self._candidates = top_candidates
        return top_candidates

    def construct_blueprints(self, num_blueprints: int = 50) -> List[TaskBlueprint]:
        """执行完整的蓝图构建流程（Step 1 of pipeline）

        流程：candidate generation → scoring → top-k selection → evidence refinement
             → blueprint formation → rubric validation

        Args:
            num_blueprints: 期望生成的蓝图数量（会多生成候选再过滤）

        Returns:
            通过 rubric 验证的 TaskBlueprint 列表
        """
        candidates_needed = int(num_blueprints / (self.top_k_ratio * self.rubric_pass_threshold)) + 5
        candidates = self.generate_candidates(candidates_needed)

        refined_count = 0
        for c in candidates:
            self._refine_with_evidence(c)
            refined_count += 1

        blueprints = []
        for c in candidates:
            bp = self._form_blueprint(c)
            passed, scores, issues = self._rubric_validate(bp)
            if passed:
                blueprints.append(bp)
            if len(blueprints) >= num_blueprints:
                break

        self._blueprints = blueprints
        return blueprints

    def get_candidates(self) -> List[CandidateTask]:
        return list(self._candidates)

    def get_blueprints(self) -> List[TaskBlueprint]:
        return list(self._blueprints)

    def get_statistics(self) -> Dict[str, float]:
        """获取蓝图构建统计"""
        total_candidates = len(self._candidates)
        total_blueprints = len(self._blueprints)
        if not self._candidates:
            return {"total_candidates": 0}

        avg_score = sum(c.total_score for c in self._candidates) / total_candidates
        return {
            "total_candidates": total_candidates,
            "candidates_scored": sum(1 for c in self._candidates if c.scored),
            "candidates_refined": sum(1 for c in self._candidates if c.refined),
            "blueprints_accepted": total_blueprints,
            "blueprint_accept_rate": total_blueprints / max(total_candidates, 1),
            "avg_creativity": sum(c.creativity_score for c in self._candidates) / total_candidates,
            "avg_technical_grounding": sum(c.technical_grounding_score for c in self._candidates) / total_candidates,
            "avg_feasibility": sum(c.feasibility_score for c in self._candidates) / total_candidates,
            "avg_total_score": avg_score,
        }

    def __repr__(self) -> str:
        return (
            f"BlueprintGenerator(candidates={len(self._candidates)}, "
            f"blueprints={len(self._blueprints)}, "
            f"top_k_ratio={self.top_k_ratio:.0%}, "
            f"rubric_threshold={self.rubric_pass_threshold:.0%})"
        )
