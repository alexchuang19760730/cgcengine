"""无缝切换器: 云↔本地自动切换 (十步流水线运行时扩展).

属于 4D 感知矩阵的运行时层:
  - 十步流水线 (启动时): 检测硬件 + 选择初始路由
  - SeamlessSwitcher (运行时): 持续监控 + 动态切换

切换触发:
  1. 内存不足 → 本地切云 (OOM 前预防)
  2. 内存恢复 → 云回本地 (省成本)
  3. 网络断开 → 云切本地 (可用性)
  4. 网络恢复 → 本地回云 (质量/速度)
  5. decode 太慢 → 本地切云 (体验)

KV cache 迁移:
  - 本地 MLX KV → 序列化 → 传云
  - 云 KV → 序列化 → 注入本地 MLX
"""
from __future__ import annotations

import os
import time
import json
import threading
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


class SwitchMode(str, Enum):
    """运行模式."""
    LOCAL = "local"           # Mac 本地 oMLX
    CLOUD = "cloud"           # 全云
    # LAYER_SPLIT 已废弃 (2026-07-25): Mac 参与 forward 是负优化
    # 端云 PD 分离 (cloud prefill → edge decode) 由 route_decision Step 7.5 决策
    OFFLINE = "offline"       # 离线降级 (本地小模型)


class SwitchReason(str, Enum):
    """切换原因."""
    MEM_CRITICAL = "内存不足,预防OOM"
    MEM_RECOVERED = "内存恢复,回切本地"
    MODEL_TOO_LARGE = "模型太大,本地放不下"
    NET_TIMEOUT = "网络超时,切本地"
    NET_RECOVERED = "网络恢复,回切云"
    DECODE_TOO_SLOW = "decode太慢,切云"
    USER_PRIVACY = "用户选择隐私模式"
    USER_MANUAL = "用户手动切换"
    INITIAL = "初始路由"


@dataclass
class SwitchEvent:
    """切换事件."""
    timestamp: float
    from_mode: str
    to_mode: str
    reason: str
    kv_migrated: bool = False
    kv_size_mb: float = 0.0
    switch_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SwitchThresholds:
    """切换阈值配置."""
    # 内存 (GB)
    mem_critical_gb: float = 1.0       # 低于 → 切云
    mem_safe_gb: float = 3.0           # 高于 → 回本地
    mem_check_interval_s: float = 5.0  # 检查间隔

    # 网络 (ms)
    rtt_critical_ms: float = 500.0     # 超过 → 切本地
    rtt_safe_ms: float = 200.0         # 低于 → 回云
    net_check_interval_s: float = 10.0

    # decode 速度 (tok/s)
    decode_min_tps: float = 5.0        # 低于 → 切云

    # KV cache 迁移
    kv_migration_enabled: bool = True
    kv_max_transfer_mb: float = 500.0  # 超过则不迁移 (太大)


class SeamlessSwitcher:
    """云↔本地无缝切换器.

    运行时持续监控 4D 感知矩阵变化,自动触发切换.

    用法:
        switcher = SeamlessSwitcher(hardware_info, cloud_endpoint)
        switcher.start()  # 启动后台监控
        # 每次请求前检查
        mode = switcher.get_current_mode()
        if switcher.should_switch(model_info):
            event = switcher.switch(model_info)
    """

    def __init__(
        self,
        hardware_info=None,
        cloud_endpoint: str = "",
        thresholds: Optional[SwitchThresholds] = None,
        on_switch_callback: Optional[Callable[[SwitchEvent], None]] = None,
    ):
        self.hw = hardware_info
        self.cloud_endpoint = cloud_endpoint
        self.thresholds = thresholds or SwitchThresholds()
        self.on_switch_callback = on_switch_callback

        # 当前状态
        self.current_mode: SwitchMode = SwitchMode.CLOUD
        self.current_reason: str = SwitchReason.INITIAL.value

        # 历史记录
        self.switch_history: list[SwitchEvent] = []
        self.last_hw_check: float = 0.0
        self.last_net_check: float = 0.0

        # KV cache (本地)
        self.local_kv_cache: Optional[dict] = None
        self.local_kv_seq_len: int = 0

        # 后台线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._running: bool = False

        # 用户覆盖 (手动模式)
        self._user_override: Optional[SwitchMode] = None

    def set_initial_mode(self, mode: SwitchMode, reason: str = ""):
        """设置初始模式 (十步流水线 Step 7.5 调用)."""
        self.current_mode = mode
        self.current_reason = reason or SwitchReason.INITIAL.value
        logger.info(f"[switcher] Initial mode: {mode.value} ({reason})")

    def start(self):
        """启动后台监控线程."""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SeamlessSwitcher"
        )
        self._monitor_thread.start()
        logger.info("[switcher] Monitor started")

    def stop(self):
        """停止监控."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("[switcher] Monitor stopped")

    def _monitor_loop(self):
        """后台监控循环."""
        while self._running:
            try:
                now = time.time()

                # 内存检查
                if now - self.last_hw_check > self.thresholds.mem_check_interval_s:
                    self._check_memory()
                    self.last_hw_check = now

                # 网络检查
                if now - self.last_net_check > self.thresholds.net_check_interval_s:
                    self._check_network()
                    self.last_net_check = now

            except Exception as e:
                logger.error(f"[switcher] Monitor error: {e}")

            time.sleep(1.0)

    def _check_memory(self):
        """检查内存,触发切换."""
        if self._user_override:
            return  # 用户手动覆盖,不自动切换

        try:
            from app.shared.hardware_sensing import detect_memory, detect_os
            os_name = self.hw.os_name if self.hw else detect_os()[0]
            _, avail_mem = detect_memory(os_name)

            if self.current_mode == SwitchMode.LOCAL:
                # 本地模式 → 检查是否需要切云
                if avail_mem < self.thresholds.mem_critical_gb:
                    logger.warning(
                        f"[switcher] Memory critical: {avail_mem}GB < {self.thresholds.mem_critical_gb}GB → switch to cloud"
                    )
                    self._trigger_switch(
                        SwitchMode.CLOUD,
                        SwitchReason.MEM_CRITICAL,
                        avail_mem=avail_mem,
                    )

            elif self.current_mode == SwitchMode.CLOUD:
                # 云模式 → 检查是否可以回本地
                if avail_mem > self.thresholds.mem_safe_gb:
                    logger.info(
                        f"[switcher] Memory recovered: {avail_mem}GB > {self.thresholds.mem_safe_gb}GB → consider local"
                    )
                    # 不立即切换,等下次请求时由 should_switch 决定

        except Exception as e:
            logger.error(f"[switcher] Memory check error: {e}")

    def _check_network(self):
        """检查网络,触发切换."""
        if self._user_override:
            return

        try:
            from app.shared.hardware_sensing import measure_rtt
            rtt = measure_rtt(self.cloud_endpoint.replace("http://", "").replace("https://", "").split(":")[0])

            if self.current_mode == SwitchMode.CLOUD:
                if rtt > self.thresholds.rtt_critical_ms:
                    logger.warning(
                        f"[switcher] Network timeout: RTT={rtt}ms > {self.thresholds.rtt_critical_ms}ms → switch to local"
                    )
                    self._trigger_switch(
                        SwitchMode.OFFLINE,
                        SwitchReason.NET_TIMEOUT,
                        rtt=rtt,
                    )

            elif self.current_mode == SwitchMode.OFFLINE:
                if rtt < self.thresholds.rtt_safe_ms:
                    logger.info(
                        f"[switcher] Network recovered: RTT={rtt}ms < {self.thresholds.rtt_safe_ms}ms → back to cloud"
                    )
                    self._trigger_switch(
                        SwitchMode.CLOUD,
                        SwitchReason.NET_RECOVERED,
                        rtt=rtt,
                    )

        except Exception as e:
            logger.error(f"[switcher] Network check error: {e}")

    def _trigger_switch(self, to_mode: SwitchMode, reason: SwitchReason, model_info=None, **context):
        """触发模式切换 (含 KV cache 迁移)."""
        from_mode = self.current_mode

        # LAYER_SPLIT 已废弃: 模型放不下直接切云, 不再下载部分层
        # KV cache 迁移
        kv_migrated = False
        kv_size = 0.0
        t0 = time.time()

        if self.thresholds.kv_migration_enabled:
            try:
                kv_data = self._migrate_kv(from_mode, to_mode)
                if kv_data:
                    kv_migrated = True
                    kv_size = kv_data.get("size_mb", 0)
            except Exception as e:
                logger.error(f"[switcher] KV migration failed: {e}")

        switch_time = (time.time() - t0) * 1000

        # 更新状态
        self.current_mode = to_mode
        self.current_reason = reason.value

        # 记录事件
        event = SwitchEvent(
            timestamp=time.time(),
            from_mode=from_mode.value,
            to_mode=to_mode.value,
            reason=reason.value,
            kv_migrated=kv_migrated,
            kv_size_mb=kv_size,
            switch_time_ms=switch_time,
        )
        self.switch_history.append(event)

        logger.info(
            f"[switcher] {from_mode.value} → {to_mode.value}: {reason.value} "
            f"(KV: {'migrated' if kv_migrated else 'no'}, {switch_time:.0f}ms)"
        )

        # 回调通知
        if self.on_switch_callback:
            try:
                self.on_switch_callback(event)
            except:
                pass

    def _migrate_kv(self, from_mode: SwitchMode, to_mode: SwitchMode) -> Optional[dict]:
        """迁移 KV cache.

        本地 → 云: 导出 MLX KV → 序列化 → 传云
        云 → 本地: 从云获取 KV → 反序列化 → 注入 MLX
        """
        if from_mode == SwitchMode.LOCAL and to_mode == SwitchMode.CLOUD:
            # 本地 → 云: 导出本地 KV
            return self._export_local_kv()

        elif from_mode == SwitchMode.CLOUD and to_mode == SwitchMode.LOCAL:
            # 云 → 本地: 从云获取 KV
            return self._import_cloud_kv()

        return None

    def _export_local_kv(self) -> Optional[dict]:
        """导出本地 MLX KV cache.

        MLX KV cache 格式: list[KVCache], 每个 KVCache 有 keys/values
        """
        if self.local_kv_cache is None:
            return None

        try:
            import mlx.core as mx
            import numpy as np

            # 序列化 KV cache
            kv_data = {
                "seq_len": self.local_kv_seq_len,
                "layers": [],
                "size_mb": 0,
            }

            total_bytes = 0
            for layer_kv in self.local_kv_cache:
                if layer_kv is None:
                    kv_data["layers"].append(None)
                    continue

                # KVCache 有 keys 和 values
                k = layer_kv.get("keys") if isinstance(layer_kv, dict) else getattr(layer_kv, "keys", None)
                v = layer_kv.get("values") if isinstance(layer_kv, dict) else getattr(layer_kv, "values", None)

                if k is not None and v is not None:
                    # MLX → numpy → bytes
                    k_np = np.array(mx.eval(k))
                    v_np = np.array(mx.eval(v))
                    kv_data["layers"].append({
                        "keys": k_np.tolist(),
                        "values": v_np.tolist(),
                        "shape": list(k_np.shape),
                        "dtype": str(k_np.dtype),
                    })
                    total_bytes += k_np.nbytes + v_np.nbytes
                else:
                    kv_data["layers"].append(None)

            kv_data["size_mb"] = total_bytes / 1e6

            # 检查大小
            if kv_data["size_mb"] > self.thresholds.kv_max_transfer_mb:
                logger.warning(
                    f"[switcher] KV cache too large: {kv_data['size_mb']:.0f}MB > {self.thresholds.kv_max_transfer_mb}MB, skip migration"
                )
                return None

            logger.info(f"[switcher] Exported local KV: {kv_data['size_mb']:.1f}MB, seq_len={kv_data['seq_len']}")
            return kv_data

        except Exception as e:
            logger.error(f"[switcher] Export KV error: {e}")
            return None

    def _import_cloud_kv(self) -> Optional[dict]:
        """从云获取 KV cache 并注入本地.

        通过 HTTP API 从云端获取序列化的 KV cache.
        """
        try:
            import requests

            resp = requests.post(
                f"{self.cloud_endpoint}/v1/cgc/kv/export",
                json={"session_id": getattr(self, "session_id", "")},
                timeout=10,
            )

            if resp.status_code == 200:
                kv_data = resp.json()
                logger.info(f"[switcher] Imported cloud KV: {kv_data.get('size_mb', 0):.1f}MB")

                # 注入本地 MLX
                self._inject_kv_to_mlx(kv_data)
                return kv_data
            else:
                logger.warning(f"[switcher] Cloud KV export failed: {resp.status_code}")
                return None

        except Exception as e:
            logger.error(f"[switcher] Import cloud KV error: {e}")
            return None

    def _inject_kv_to_mlx(self, kv_data: dict):
        """将 KV cache 注入本地 MLX 模型."""
        try:
            import mlx.core as mx
            import numpy as np

            layers_data = kv_data.get("layers", [])
            self.local_kv_cache = []
            self.local_kv_seq_len = kv_data.get("seq_len", 0)

            for layer_data in layers_data:
                if layer_data is None:
                    self.local_kv_cache.append(None)
                    continue

                k_np = np.array(layer_data["keys"], dtype=np.float32)
                v_np = np.array(layer_data["values"], dtype=np.float32)

                self.local_kv_cache.append({
                    "keys": mx.array(k_np),
                    "values": mx.array(v_np),
                })

            logger.info(f"[switcher] Injected KV to local: {len(layers_data)} layers, seq={self.local_kv_seq_len}")

        except Exception as e:
            logger.error(f"[switcher] Inject KV error: {e}")

    def set_local_kv(self, kv_cache: list, seq_len: int):
        """设置本地 KV cache (由推理引擎调用)."""
        self.local_kv_cache = kv_cache
        self.local_kv_seq_len = seq_len

    def should_switch(self, model_info=None) -> Optional[tuple[SwitchMode, SwitchReason, int]]:
        """检查是否需要切换 (每次请求前调用).

        Returns:
            (to_mode, reason, P) 如果需要切换, None 如果不需要
            P = 0 (layer-split 已废弃, 保留参数为兼容性)
        """
        if self._user_override:
            return None  # 用户手动覆盖

        try:
            from app.shared.hardware_sensing import detect_memory, detect_os

            os_name = self.hw.os_name if self.hw else detect_os()[0]
            _, avail_mem = detect_memory(os_name)

            # 内存检查
            if self.current_mode == SwitchMode.LOCAL:
                if avail_mem < self.thresholds.mem_critical_gb:
                    # 内存严重不足 → 切云
                    return (SwitchMode.CLOUD, SwitchReason.MEM_CRITICAL, 0)

                if model_info and model_info.model_size_gb > avail_mem:
                    # 模型放不下 → 直接切云 (layer-split 已废弃)
                    logger.info(
                        f"[switcher] Model too large ({model_info.model_size_gb}GB > {avail_mem}GB) "
                        f"→ cloud (layer-split deprecated)"
                    )
                    return (SwitchMode.CLOUD, SwitchReason.MODEL_TOO_LARGE, 0)

            elif self.current_mode == SwitchMode.CLOUD:
                if avail_mem > self.thresholds.mem_safe_gb:
                    if model_info and model_info.model_size_gb < avail_mem:
                        # 内存恢复 + 模型放得下 → 回本地
                        return (SwitchMode.LOCAL, SwitchReason.MEM_RECOVERED, 0)
                    # 内存恢复但放不下 → 继续云 (layer-split 已废弃)

            return None

        except Exception as e:
            logger.error(f"[switcher] should_switch error: {e}")
            return None

    def get_current_mode(self) -> SwitchMode:
        """获取当前模式."""
        return self._user_override or self.current_mode

    def user_override(self, mode: Optional[SwitchMode]):
        """用户手动覆盖模式 (None = 恢复自动)."""
        self._user_override = mode
        if mode:
            logger.info(f"[switcher] User override: {mode.value}")
        else:
            logger.info("[switcher] User override removed, back to auto")

    def get_status(self) -> dict:
        """获取切换器状态."""
        return {
            "current_mode": self.get_current_mode().value,
            "current_reason": self.current_reason,
            "user_override": self._user_override.value if self._user_override else None,
            "switch_count": len(self.switch_history),
            "last_switch": self.switch_history[-1].to_dict() if self.switch_history else None,
            "local_kv_seq_len": self.local_kv_seq_len,
            "thresholds": asdict(self.thresholds),
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516")

    print("=" * 60)
    print("SeamlessSwitcher 测试 (云↔本地无缝切换)")
    print("=" * 60)

    from app.shared.hardware_sensing import detect_all
    from app.shared.route_decision import MODEL_PRESETS, compute_route

    hw = detect_all()
    print(f"硬件: {hw.cpu_brand}, {hw.available_mem_gb}GB, {hw.compute_tier}")

    # 初始路由
    model = MODEL_PRESETS["qwen3-vl-2b-4bit"]
    route = compute_route(hw, model)
    print(f"模型: {model.name}")
    print(f"初始路由: {route.mode} P={route.P}")

    # 创建切换器
    switcher = SeamlessSwitcher(
        hardware_info=hw,
        cloud_endpoint="http://47.95.250.55:30001",
    )

    # 设置初始模式
    initial_mode = SwitchMode.LOCAL if route.mode in ("local_only", "pd_separation") else SwitchMode.CLOUD
    switcher.set_initial_mode(initial_mode, f"初始路由: {route.mode}")
    print(f"初始模式: {switcher.get_current_mode().value}")

    # 模拟内存不足
    print("\n--- 模拟内存不足 (available < 1GB) ---")
    switcher.thresholds.mem_critical_gb = 10.0  # 调高阈值触发切换
    switcher.thresholds.mem_safe_gb = 20.0
    result = switcher.should_switch(model)
    if result:
        to_mode, reason = result
        print(f"  需要切换: {switcher.get_current_mode().value} → {to_mode.value} ({reason.value})")
        switcher._trigger_switch(to_mode, reason)
    print(f"  当前模式: {switcher.get_current_mode().value}")

    # 模拟内存恢复
    print("\n--- 模拟内存恢复 (available > 3GB) ---")
    switcher.thresholds.mem_critical_gb = 0.5
    switcher.thresholds.mem_safe_gb = 1.0
    result = switcher.should_switch(model)
    if result:
        to_mode, reason = result
        print(f"  需要切换: {switcher.get_current_mode().value} → {to_mode.value} ({reason.value})")
        switcher._trigger_switch(to_mode, reason)
    print(f"  当前模式: {switcher.get_current_mode().value}")

    # 状态
    print(f"\n--- 切换器状态 ---")
    status = switcher.get_status()
    print(f"  切换次数: {status['switch_count']}")
    if status['last_switch']:
        ls = status['last_switch']
        print(f"  最后切换: {ls['from_mode']} → {ls['to_mode']} ({ls['reason']})")

    # 用户覆盖
    print(f"\n--- 用户手动覆盖 (隐私模式) ---")
    switcher.user_override(SwitchMode.LOCAL)
    print(f"  当前模式: {switcher.get_current_mode().value}")
    result = switcher.should_switch(model)
    print(f"  should_switch: {result} (用户覆盖,不自动切换)")
    switcher.user_override(None)
    print(f"  恢复自动: {switcher.get_current_mode().value}")
