#!/usr/bin/env python3
"""AcceptanceTracker — MTP 投机接受率追踪 + 三态自适应状态机.

设计文档: CGC_EDGE_CLOUD_ARCHITECTURE_V2.md 第五章

三态:
  ENABLED  — 全力 MTP (steps=5, topk=2), accept_rate > 55%
  DEGRADED — 短 draft (steps=2, topk=1), accept_rate 35-50%
  DISABLED — 关闭 MTP, 直连云 (parallel preflight 仍工作), accept_rate < 35%

迟滞防抖: 连续 20 次请求满足阈值才转换状态, 防止抖动.

核心原则: 框架永远不强制开启 MTP 推测; 低接受率自动降级, 避免负优化.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional


# ── 状态枚举 ──────────────────────────────────────────────────

STATE_ENABLED = "ENABLED"
STATE_DEGRADED = "DEGRADED"
STATE_DISABLED = "DISABLED"

ALL_STATES = (STATE_ENABLED, STATE_DEGRADED, STATE_DISABLED)

# ── 每状态对应的 MTP 配置 ─────────────────────────────────────

STATE_MTP_CONFIG = {
    STATE_ENABLED:  {"steps": 5, "topk": 2, "speculate": True,  "min_confidence": 0.55},
    STATE_DEGRADED: {"steps": 2, "topk": 1, "speculate": True,  "min_confidence": 0.70},
    STATE_DISABLED: {"steps": 0, "topk": 0, "speculate": False, "min_confidence": 1.01},
}

# ── 迟滞阈值 ──────────────────────────────────────────────────

# 状态转换需要连续 N 次满足条件
CONSECUTIVE_TRANSITION_COUNT = 20

# accept_rate 阈值
THRESHOLD_DISABLE_FROM_DEGRADED = 0.35   # DEGRADED → DISABLED
THRESHOLD_DEGRADE_FROM_ENABLED = 0.40    # ENABLED → DEGRADED
THRESHOLD_ENABLE_FROM_DEGRADED = 0.50    # DEGRADED → ENABLED
THRESHOLD_ENABLE_FROM_DISABLED = 0.55    # DISABLED → ENABLED

# 滚动窗口大小
WINDOW_SIZE = 50

# per-family 最小样本数 (低于此数不单独追踪, 用全局)
MIN_FAMILY_SAMPLES = 10


class AcceptanceTracker:
    """投机接受率追踪器, 线程安全.

    用法:
      tracker = AcceptanceTracker()
      tracker.record(hit=True, family="fix")    # 记录一次投机结果
      tracker.record(hit=False, family="debug")

      if tracker.should_speculate(family="fix"):
          # 执行投机
          ...

      config = tracker.get_mtp_config()
      # config = {"steps": 5, "topk": 2, "speculate": True}
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        transition_count: int = CONSECUTIVE_TRANSITION_COUNT,
    ):
        self._lock = threading.Lock()

        # 全局滚动窗口 (1=hit, 0=miss)
        self._window: deque[int] = deque(maxlen=window_size)

        # per-family 滚动窗口
        self._family_windows: dict[str, deque[int]] = {}

        # 连续计数器 (用于状态转换)
        self._consecutive_low: int = 0    # 连续低 accept
        self._consecutive_high: int = 0   # 连续高 accept

        # 当前状态
        self._state: str = STATE_ENABLED

        # 配置
        self._window_size = window_size
        self._transition_count = transition_count

        # 统计
        self._total_hits = 0
        self._total_misses = 0
        self._state_transitions: list[dict] = []

        # 状态转换回调 (设置后, 每次状态变化时调用)
        self._on_transition_cb = None

    # ── 核心接口 ──────────────────────────────────────────────

    def record(self, hit: bool, family: str = "generic") -> None:
        """记录一次投机结果.

        Args:
            hit: True=命中, False=未命中
            family: prompt family (fix/debug/write/...)
        """
        with self._lock:
            val = 1 if hit else 0
            self._window.append(val)

            # per-family 追踪
            if family not in self._family_windows:
                self._family_windows[family] = deque(maxlen=self._window_size)
            self._family_windows[family].append(val)

            # 全局统计
            if hit:
                self._total_hits += 1
                self._consecutive_high += 1
                self._consecutive_low = 0
            else:
                self._total_misses += 1
                self._consecutive_low += 1
                self._consecutive_high = 0

            # 状态机转换
            self._maybe_transition()

    def should_speculate(self, family: str = "generic") -> bool:
        """是否应该对当前请求执行投机.

        逻辑:
        1. 全局状态为 DISABLED → False
        2. 全局状态为 DEGRADED → 只对高 accept 的 family 投机
        3. 全局状态为 ENABLED → True (除非该 family 独立 accept 极低)
        """
        with self._lock:
            state = self._state
            config = STATE_MTP_CONFIG[state]

            if not config["speculate"]:
                return False

            # per-family 检查: 如果该 family 有足够样本且 accept 极低, 跳过
            family_window = self._family_windows.get(family)
            if family_window and len(family_window) >= MIN_FAMILY_SAMPLES:
                family_rate = sum(family_window) / len(family_window)
                if state == STATE_DEGRADED and family_rate < THRESHOLD_DISABLE_FROM_DEGRADED:
                    return False
                if state == STATE_ENABLED and family_rate < THRESHOLD_DEGRADE_FROM_ENABLED:
                    # 即使全局 ENABLED, 该 family 持续低 accept 也跳过
                    return False

            return True

    def get_mtp_config(self) -> dict:
        """获取当前状态对应的 MTP 配置."""
        with self._lock:
            return dict(STATE_MTP_CONFIG[self._state])

    def get_min_confidence(self) -> float:
        """获取当前状态对应的最小投机 confidence threshold.

        ENABLED:  0.55 (宽松, 多投机)
        DEGRADED: 0.70 (保守, 只对高 confidence 投机)
        DISABLED: 1.01 (不可能达到, 实际关闭)
        """
        with self._lock:
            return STATE_MTP_CONFIG[self._state]["min_confidence"]

    def set_on_transition(self, cb) -> None:
        """设置状态转换回调: cb(old_state, new_state, accept_rate)."""
        self._on_transition_cb = cb

    # ── 状态机 ────────────────────────────────────────────────

    def _maybe_transition(self) -> None:
        """检查是否应该转换状态 (调用者需持有 _lock)."""
        current = self._state
        rate = self._current_rate()

        if current == STATE_ENABLED:
            if self._consecutive_low >= self._transition_count and rate < THRESHOLD_DEGRADE_FROM_ENABLED:
                self._transition_to(STATE_DEGRADED, rate)
            elif self._consecutive_low >= self._transition_count * 2 and rate < THRESHOLD_DISABLE_FROM_DEGRADED:
                # 极端情况: 直接从 ENABLED 跳到 DISABLED
                self._transition_to(STATE_DISABLED, rate)

        elif current == STATE_DEGRADED:
            if self._consecutive_low >= self._transition_count and rate < THRESHOLD_DISABLE_FROM_DEGRADED:
                self._transition_to(STATE_DISABLED, rate)
            elif self._consecutive_high >= self._transition_count and rate > THRESHOLD_ENABLE_FROM_DEGRADED:
                self._transition_to(STATE_ENABLED, rate)

        elif current == STATE_DISABLED:
            if self._consecutive_high >= self._transition_count and rate > THRESHOLD_ENABLE_FROM_DISABLED:
                self._transition_to(STATE_ENABLED, rate)

    def _transition_to(self, new_state: str, rate: float) -> None:
        """执行状态转换 (调用者需持有 _lock)."""
        old_state = self._state
        self._state = new_state
        # 重置连续计数器
        self._consecutive_low = 0
        self._consecutive_high = 0
        # 记录转换历史
        transition_record = {
            "time": time.time(),
            "from": old_state,
            "to": new_state,
            "accept_rate": round(rate, 3),
            "total_samples": len(self._window),
        }
        self._state_transitions.append(transition_record)
        # 保留最近 50 条转换记录
        if len(self._state_transitions) > 50:
            self._state_transitions = self._state_transitions[-50:]
        # 回调通知 (不持有 lock 调用, 避免死锁)
        if self._on_transition_cb:
            try:
                self._on_transition_cb(old_state, new_state, rate)
            except Exception:
                pass

    # ── 查询 ──────────────────────────────────────────────────

    def _current_rate(self) -> float:
        """当前全局 accept rate (调用者需持有 _lock)."""
        if not self._window:
            return 1.0  # 默认乐观 (无数据时不降级)
        return sum(self._window) / len(self._window)

    def get_state(self) -> str:
        """获取当前状态."""
        with self._lock:
            return self._state

    def get_accept_rate(self, family: Optional[str] = None) -> float:
        """获取 accept rate (全局或 per-family)."""
        with self._lock:
            if family:
                fw = self._family_windows.get(family)
                if not fw or len(fw) < MIN_FAMILY_SAMPLES:
                    return self._current_rate()
                return sum(fw) / len(fw)
            return self._current_rate()

    def get_family_rates(self) -> dict[str, float]:
        """获取所有 family 的 accept rate."""
        with self._lock:
            result = {}
            for family, window in self._family_windows.items():
                if len(window) >= MIN_FAMILY_SAMPLES:
                    result[family] = round(sum(window) / len(window), 3)
            return result

    def get_status(self) -> dict:
        """获取完整状态快照 (用于 /health 端点)."""
        with self._lock:
            rate = self._current_rate()
            return {
                "state": self._state,
                "mtp_config": dict(STATE_MTP_CONFIG[self._state]),
                "global_accept_rate": round(rate, 3),
                "global_samples": len(self._window),
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "consecutive_low": self._consecutive_low,
                "consecutive_high": self._consecutive_high,
                "family_rates": {
                    f: round(sum(w) / len(w), 3)
                    for f, w in self._family_windows.items()
                    if len(w) >= MIN_FAMILY_SAMPLES
                },
                "family_sample_counts": {
                    f: len(w) for f, w in self._family_windows.items()
                },
                "recent_transitions": self._state_transitions[-5:] if self._state_transitions else [],
                "thresholds": {
                    "degrade_from_enabled": THRESHOLD_DEGRADE_FROM_ENABLED,
                    "disable_from_degraded": THRESHOLD_DISABLE_FROM_DEGRADED,
                    "enable_from_degraded": THRESHOLD_ENABLE_FROM_DEGRADED,
                    "enable_from_disabled": THRESHOLD_ENABLE_FROM_DISABLED,
                    "transition_count": self._transition_count,
                    "window_size": self._window_size,
                },
            }

    def reset(self) -> None:
        """重置追踪器 (用于测试)."""
        with self._lock:
            self._window.clear()
            self._family_windows.clear()
            self._consecutive_low = 0
            self._consecutive_high = 0
            self._state = STATE_ENABLED
            self._total_hits = 0
            self._total_misses = 0
            self._state_transitions.clear()


# ── 全局单例 ──────────────────────────────────────────────────

_global_tracker: Optional[AcceptanceTracker] = None
_global_lock = threading.Lock()


def get_acceptance_tracker() -> AcceptanceTracker:
    """获取全局 AcceptanceTracker 单例."""
    global _global_tracker
    if _global_tracker is None:
        with _global_lock:
            if _global_tracker is None:
                _global_tracker = AcceptanceTracker()
    return _global_tracker
