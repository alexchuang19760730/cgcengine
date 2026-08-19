# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Optimizer Analytics - 學習器分析追蹤系統

功能：
- 追蹤 Agent 生成的結果
- 追蹤 Feedback 收集結果
- 追蹤是否使用 Feedback 進行優化
- 追蹤優化生成成功率
- 生成分析報告

使用方式：
    from cgc_engine.agent.analytics import OptimizerAnalytics, LayerType

    analytics = OptimizerAnalytics()

    # 記錄 Agent 生成
    analytics.record_agent_generation(
        layer=LayerType.COMPUTATION,
        generated=True,
        strategy_params={...}
    )

    # 記錄 Feedback 收集
    analytics.record_feedback_collection(
        layer=LayerType.SCHEDULING,
        collected=True,
        feedback_score=0.85
    )

    # 記錄是否使用 Feedback
    analytics.record_feedback_usage(
        layer=LayerType.STORAGE,
        used=True,
        improvement_score=0.12
    )

    # 生成報告
    report = analytics.generate_report()
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import time
import json
import logging
from pathlib import Path

if TYPE_CHECKING:
    from .scheduling_layer import SchedulerStrategy, SchedulingFeedback
    from .storage_layer import StorageStrategy, StorageFeedback
    from .space_builder import OptimizationSpace

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """層類型"""
    COMPUTATION = "computation"
    SCHEDULING = "scheduling"
    STORAGE = "storage"


class EventType(Enum):
    """事件類型"""
    AGENT_GENERATION = "agent_generation"
    FEEDBACK_COLLECTION = "feedback_collection"
    FEEDBACK_USAGE = "feedback_usage"
    OPTIMIZATION_SUCCESS = "optimization_success"
    OPTIMIZATION_FAILURE = "optimization_failure"


@dataclass
class AnalyticsEvent:
    """分析事件"""
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    layer: str = ""
    success: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class LayerStats:
    """層統計數據"""
    layer: str = ""
    agent_generations: int = 0
    agent_successes: int = 0
    agent_success_rate: float = 0.0

    feedback_collections: int = 0
    feedback_successes: int = 0
    feedback_success_rate: float = 0.0

    feedback_usages: int = 0
    feedback_used: int = 0
    feedback_usage_rate: float = 0.0

    avg_improvement_score: float = 0.0
    avg_feedback_score: float = 0.0

    last_generation_time: float = 0.0
    last_feedback_time: float = 0.0


class OptimizerAnalytics:
    """
    學習器分析追蹤器

    統一追蹤三層學習器的各項指標
    """

    def __init__(self):
        self.events: List[AnalyticsEvent] = []
        self.layer_stats: Dict[str, LayerStats] = {
            LayerType.COMPUTATION.value: LayerStats(layer=LayerType.COMPUTATION.value),
            LayerType.SCHEDULING.value: LayerStats(layer=LayerType.SCHEDULING.value),
            LayerType.STORAGE.value: LayerStats(layer=LayerType.STORAGE.value),
        }
        self.start_time = time.time()

    def record_agent_generation(
        self,
        layer: LayerType,
        generated: bool,
        strategy_params: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """
        記錄 Agent 生成結果

        Args:
            layer: 層類型
            generated: 是否成功生成
            strategy_params: 生成策略的參數
            duration_ms: 生成耗時
        """
        event = AnalyticsEvent(
            event_type=EventType.AGENT_GENERATION.value,
            layer=layer.value,
            success=generated,
            details=strategy_params or {},
            duration_ms=duration_ms,
        )
        self.events.append(event)

        stats = self.layer_stats[layer.value]
        stats.agent_generations += 1
        stats.last_generation_time = event.timestamp
        if generated:
            stats.agent_successes += 1

        stats.agent_success_rate = (
            stats.agent_successes / stats.agent_generations
            if stats.agent_generations > 0 else 0.0
        )

        logger.info(
            f"[Analytics] Agent generation ({layer.value}): "
            f"success={generated}, rate={stats.agent_success_rate:.2%}"
        )

    def record_feedback_collection(
        self,
        layer: LayerType,
        collected: bool,
        feedback_score: float = 0.0,
        feedback_source: str = "llama.cpp",
        duration_ms: float = 0.0,
    ) -> None:
        """
        記錄 Feedback 收集結果

        Args:
            layer: 層類型
            collected: 是否成功收集
            feedback_score: 反饋分數
            feedback_source: 反饋來源 (llama.cpp/vllm)
            duration_ms: 收集耗時
        """
        event = AnalyticsEvent(
            event_type=EventType.FEEDBACK_COLLECTION.value,
            layer=layer.value,
            success=collected,
            details={
                "feedback_score": feedback_score,
                "feedback_source": feedback_source,
            },
            duration_ms=duration_ms,
        )
        self.events.append(event)

        stats = self.layer_stats[layer.value]
        stats.feedback_collections += 1
        stats.last_feedback_time = event.timestamp
        if collected:
            stats.feedback_successes += 1
            stats.avg_feedback_score = (
                (stats.avg_feedback_score * (stats.feedback_successes - 1) + feedback_score)
                / stats.feedback_successes
            )

        stats.feedback_success_rate = (
            stats.feedback_successes / stats.feedback_collections
            if stats.feedback_collections > 0 else 0.0
        )

        logger.info(
            f"[Analytics] Feedback collection ({layer.value}): "
            f"success={collected}, score={feedback_score:.2f}, rate={stats.feedback_success_rate:.2%}"
        )

    def record_feedback_usage(
        self,
        layer: LayerType,
        used: bool,
        improvement_score: float = 0.0,
        strategy_applied: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        記錄是否使用 Feedback 進行優化

        Args:
            layer: 層類型
            used: 是否使用反饋
            improvement_score: 改進分數
            strategy_applied: 應用的策略參數
        """
        event = AnalyticsEvent(
            event_type=EventType.FEEDBACK_USAGE.value,
            layer=layer.value,
            success=used,
            details={
                "improvement_score": improvement_score,
                "strategy_applied": strategy_applied or {},
            },
        )
        self.events.append(event)

        stats = self.layer_stats[layer.value]
        stats.feedback_usages += 1
        if used:
            stats.feedback_used += 1
            stats.avg_improvement_score = (
                (stats.avg_improvement_score * (stats.feedback_used - 1) + improvement_score)
                / stats.feedback_used
            )

        stats.feedback_usage_rate = (
            stats.feedback_used / stats.feedback_usages
            if stats.feedback_usages > 0 else 0.0
        )

        logger.info(
            f"[Analytics] Feedback usage ({layer.value}): "
            f"used={used}, improvement={improvement_score:.2%}, rate={stats.feedback_usage_rate:.2%}"
        )

    def record_optimization_result(
        self,
        layer: LayerType,
        success: bool,
        final_score: float = 0.0,
        error_message: Optional[str] = None,
    ) -> None:
        """
        記錄優化結果

        Args:
            layer: 層類型
            success: 是否成功
            final_score: 最終分數
            error_message: 錯誤訊息
        """
        event_type = (
            EventType.OPTIMIZATION_SUCCESS.value
            if success else EventType.OPTIMIZATION_FAILURE.value
        )
        event = AnalyticsEvent(
            event_type=event_type,
            layer=layer.value,
            success=success,
            details={
                "final_score": final_score,
                "error_message": error_message,
            },
        )
        self.events.append(event)

        logger.info(
            f"[Analytics] Optimization ({layer.value}): "
            f"success={success}, final_score={final_score:.2f}"
        )

    def get_layer_stats(self, layer: LayerType) -> LayerStats:
        """獲取指定層的統計數據"""
        return self.layer_stats[layer.value]

    def get_all_stats(self) -> Dict[str, LayerStats]:
        """獲取所有層的統計數據"""
        return self.layer_stats.copy()

    def get_summary(self) -> Dict[str, Any]:
        """獲取總結報告"""
        total_events = len(self.events)
        total_generations = sum(s.agent_generations for s in self.layer_stats.values())
        total_feedbacks = sum(s.feedback_collections for s in self.layer_stats.values())
        total_usages = sum(s.feedback_usages for s in self.layer_stats.values())

        return {
            "total_events": total_events,
            "total_agent_generations": total_generations,
            "total_feedback_collections": total_feedbacks,
            "total_feedback_usages": total_usages,
            "uptime_seconds": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
        }

    def generate_report(self) -> Dict[str, Any]:
        """生成完整分析報告"""
        summary = self.get_summary()
        layers = {}

        for layer_type in LayerType:
            stats = self.layer_stats[layer_type.value]
            layers[layer_type.value] = {
                "agent_generation": {
                    "total": stats.agent_generations,
                    "successes": stats.agent_successes,
                    "success_rate": stats.agent_success_rate,
                    "last_time": stats.last_generation_time,
                },
                "feedback_collection": {
                    "total": stats.feedback_collections,
                    "successes": stats.feedback_successes,
                    "success_rate": stats.feedback_success_rate,
                    "avg_score": stats.avg_feedback_score,
                    "last_time": stats.last_feedback_time,
                },
                "feedback_usage": {
                    "total": stats.feedback_usages,
                    "used": stats.feedback_used,
                    "usage_rate": stats.feedback_usage_rate,
                    "avg_improvement": stats.avg_improvement_score,
                },
            }

        return {
            "summary": summary,
            "layers": layers,
        }

    def save_report(self, filepath: str) -> None:
        """保存報告到檔案"""
        report = self.generate_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"[Analytics] Report saved to {filepath}")

    def get_recent_events(
        self,
        layer: Optional[LayerType] = None,
        event_type: Optional[EventType] = None,
        limit: int = 10,
    ) -> List[AnalyticsEvent]:
        """獲取最近的事件"""
        events = self.events

        if layer:
            events = [e for e in events if e.layer == layer.value]
        if event_type:
            events = [e for e in events if e.event_type == event_type.value]

        return events[-limit:]

    def reset(self) -> None:
        """重置所有統計"""
        self.events.clear()
        for stats in self.layer_stats.values():
            stats.agent_generations = 0
            stats.agent_successes = 0
            stats.agent_success_rate = 0.0
            stats.feedback_collections = 0
            stats.feedback_successes = 0
            stats.feedback_success_rate = 0.0
            stats.feedback_usages = 0
            stats.feedback_used = 0
            stats.feedback_usage_rate = 0.0
            stats.avg_improvement_score = 0.0
            stats.avg_feedback_score = 0.0
            stats.last_generation_time = 0.0
            stats.last_feedback_time = 0.0
        self.start_time = time.time()
        logger.info("[Analytics] Reset complete")


class FeedbackDecisionMaker:
    """
    Feedback 決策器

    決定是否使用 Feedback 進行優化
    """

    def __init__(
        self,
        min_feedback_score: float = 0.6,
        min_improvement_threshold: float = 0.05,
    ):
        self.min_feedback_score = min_feedback_score
        self.min_improvement_threshold = min_improvement_threshold
        self.analytics: Optional[OptimizerAnalytics] = None

    def should_use_feedback(
        self,
        feedback_score: float,
        current_score: float,
        target_score: float,
        layer: LayerType,
    ) -> tuple[bool, str]:
        """
        決定是否使用 Feedback

        Args:
            feedback_score: 反饋分數
            current_score: 當前分數
            target_score: 目標分數
            layer: 層類型

        Returns:
            (是否使用, 原因)
        """
        improvement = (target_score - current_score) / current_score if current_score > 0 else 0

        if feedback_score < self.min_feedback_score:
            reason = f"Feedback score {feedback_score:.2f} < threshold {self.min_feedback_score}"
            logger.info(f"[FeedbackDecision] Not using: {reason}")
            return False, reason

        if improvement < self.min_improvement_threshold:
            reason = f"Expected improvement {improvement:.2%} < threshold {self.min_improvement_threshold}"
            logger.info(f"[FeedbackDecision] Not using: {reason}")
            return False, reason

        reason = f"Feedback score {feedback_score:.2f} and improvement {improvement:.2%} both acceptable"
        logger.info(f"[FeedbackDecision] Using feedback: {reason}")
        return True, reason

    def bind_analytics(self, analytics: OptimizerAnalytics) -> None:
        """綁定 Analytics"""
        self.analytics = analytics


def create_analytics_with_recommended_settings() -> OptimizerAnalytics:
    """創建使用推薦設置的 Analytics"""
    return OptimizerAnalytics()
