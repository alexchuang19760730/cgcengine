"""任务分类法 - 论文四维正交分类

论文 Section 3.2: "We anchor idea generation on three orthogonal dimensions beyond domain:
(1) skill type (algorithmic/systems/configuration/cryptography/...),
(2) capability (exploration/error recovery/constraint satisfaction/long-horizon planning),
(3) engineering pillar (new feature creation/debugging/DevOps/refactoring)"

加上 domain 维度共四维，采样组合作为 anchor point。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import random


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class Domain(Enum):
    """论文领域维度：基于 TB 2.0 基准四大超类"""
    SOFTWARE_SYSTEM = "software_and_system"
    DATA_PROCESSING = "data_processing"
    MACHINE_LEARNING = "machine_learning"
    NETWORKING_SECURITY = "networking_and_security"


class SkillType(Enum):
    """技能类型维度 - 论文 Appendix A 定义的专业技术知识"""
    ALGORITHMIC = "algorithmic"
    SYSTEMS = "systems"
    CONFIGURATION = "configuration"
    CRYPTOGRAPHY = "cryptography"
    FILE_MANIPULATION = "file_manipulation"
    TEXT_PROCESSING = "text_processing"
    NETWORKING = "networking"
    DEBUGGING = "debugging"
    DATABASE = "database"
    VERSION_CONTROL = "version_control"


class Capability(Enum):
    """能力（推理行为）维度 - 论文定义任务应引发的推理行为"""
    EXPLORATION = "exploration"
    ERROR_RECOVERY = "error_recovery"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    LONG_HORIZON_PLANNING = "long_horizon_planning"
    STATE_TRACKING = "state_tracking"
    TOOL_ORCHESTRATION = "tool_orchestration"
    MULTI_STEP_REASONING = "multi_step_reasoning"


class EngineeringPillar(Enum):
    """工程支柱维度 - 任务所代表的工作形式"""
    NEW_FEATURE = "new_feature_creation"
    DEBUGGING = "debugging"
    DEVOPS = "devops"
    REFACTORING = "refactoring"
    DATA_ANALYSIS = "data_analysis"
    TROUBLESHOOTING = "troubleshooting"
    CONFIGURATION_MANAGEMENT = "configuration_management"


@dataclass
class TaskAnchor:
    """四维组合锚点 - 论文 3.2: "sample combinations as anchor points" """
    domain: Domain
    skill_type: SkillType
    capability: Capability
    engineering_pillar: EngineeringPillar


@dataclass
class TaskBlueprint:
    """任务蓝图 - 论文 3.2 Blueprint Formation 产物
    
    包含：
    - user_query: 面向用户的任务指令
    - internal_hint: 内部hint（不向用户显示，用于解答构建）
    - environment_checklist: 环境资产清单
    - acceptance_criteria: 验收标准
    - anchor: 四维分类锚点
    - evidence: 证据引用
    - estimated_turns: 预估solver turns（精炼后约 3.45×）
    """
    blueprint_id: str
    user_query: str
    internal_hint: str
    environment_checklist: List[str]
    acceptance_criteria: List[str]
    anchor: TaskAnchor
    evidence: List[str] = field(default_factory=list)
    estimated_turns: int = 8
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    creativity_score: float = 0.0
    technical_grounding_score: float = 0.0
    feasibility_score: float = 0.0


class TaskTaxonomy:
    """四维任务分类法 - 论文 3.2 的结构化能力空间
    
    关键特征：
    - 四维度正交组合，确保能力空间广泛且可控覆盖
    - 每个 anchor 生成的任务约束到特定能力模式
    - 在技能级别保证多样性，而不仅是表面话题
    """

    def __init__(self):
        self._define_domain_skill_map()

    def _define_domain_skill_map(self):
        """定义每个 domain 内合理的 skill_type 组合"""
        self.domain_skills: Dict[Domain, List[SkillType]] = {
            Domain.SOFTWARE_SYSTEM: [
                SkillType.SYSTEMS, SkillType.DEBUGGING, SkillType.FILE_MANIPULATION,
                SkillType.VERSION_CONTROL, SkillType.CONFIGURATION, SkillType.TEXT_PROCESSING,
            ],
            Domain.DATA_PROCESSING: [
                SkillType.ALGORITHMIC, SkillType.TEXT_PROCESSING, SkillType.FILE_MANIPULATION,
                SkillType.DATABASE, SkillType.SYSTEMS,
            ],
            Domain.MACHINE_LEARNING: [
                SkillType.SYSTEMS, SkillType.CONFIGURATION, SkillType.DEBUGGING,
                SkillType.FILE_MANIPULATION, SkillType.ALGORITHMIC,
            ],
            Domain.NETWORKING_SECURITY: [
                SkillType.NETWORKING, SkillType.CRYPTOGRAPHY, SkillType.CONFIGURATION,
                SkillType.SYSTEMS, SkillType.DEBUGGING,
            ],
        }

    def sample_anchor(self, rng: Optional[random.Random] = None) -> TaskAnchor:
        """从四维空间采样一个组合锚点"""
        rng = rng or random
        domain = rng.choice(list(Domain))
        valid_skills = self.domain_skills[domain]
        skill_type = rng.choice(valid_skills)
        capability = rng.choice(list(Capability))
        pillar = rng.choice(list(EngineeringPillar))
        return TaskAnchor(
            domain=domain,
            skill_type=skill_type,
            capability=capability,
            engineering_pillar=pillar,
        )

    def sample_anchors(self, n: int, seed: int = 42) -> List[TaskAnchor]:
        """采样 n 个多样化锚点"""
        rng = random.Random(seed)
        anchors = []
        seen = set()
        attempts = 0
        while len(anchors) < n and attempts < n * 20:
            anchor = self.sample_anchor(rng)
            key = (anchor.domain, anchor.skill_type, anchor.capability, anchor.engineering_pillar)
            if key not in seen:
                seen.add(key)
                anchors.append(anchor)
            attempts += 1
        return anchors

    def total_combinations(self) -> int:
        """理论总组合数"""
        total = 0
        for skills in self.domain_skills.values():
            total += len(skills) * len(Capability) * len(EngineeringPillar)
        return total

    def describe_anchor(self, anchor: TaskAnchor) -> str:
        return (
            f"[{anchor.domain.value}/{anchor.skill_type.value}/"
            f"{anchor.capability.value}/{anchor.engineering_pillar.value}]"
        )

    def __repr__(self) -> str:
        return (
            f"TaskTaxonomy(domains={len(Domain)}, skill_types={len(SkillType)}, "
            f"capabilities={len(Capability)}, pillars={len(EngineeringPillar)}, "
            f"total_combinations={self.total_combinations()})"
        )