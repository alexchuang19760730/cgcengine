#!/usr/bin/env python3
"""SPDK io_uring 升级验证"""
import sys
sys.path.insert(0, ".")

print("=== liburing 验证 ===")
import liburing
print("liburing: available ✅")
print("版本:", getattr(liburing, "__version__", "unknown"))

print()
print("=== SPDK 升级验证 ===")
from cgc_engine.spdk_adapter.spdk_io_manager import SPDKIOManager, SPDKConfig, SPDK_AVAILABLE
print("SPDK_AVAILABLE:", SPDK_AVAILABLE)
mode = "liburing (io_uring 异步I/O)" if SPDK_AVAILABLE else "thread-pool (降级)"
print("模式:", mode)

cfg = SPDKConfig()
mgr = SPDKIOManager(cfg)
mgr.start(num_workers=4)
print("SPDKIOManager.start(): OK")
actual_mode = "liburing" if SPDK_AVAILABLE else "thread-pool"
print("实际模式:", actual_mode)
mgr.stop()
print("SPDKIOManager.stop(): OK")

print()
print("=== FlashMoE + SPDK 集成 ===")
from cgc_engine.flash_moe.distributed_expert_store import SPDK_AVAILABLE as FM_SPDK
print("FlashMoE SPDK_AVAILABLE:", FM_SPDK)
fm_mode = "liburing ✅" if FM_SPDK else "thread-pool ⚠️"
print("FlashMoE SPDK 模式:", fm_mode)

print()
print("=== 总结 ===")
status = "✅ io_uring 异步I/O (liburing)" if SPDK_AVAILABLE else "⚠️ 线程池降级"
print("SPDK 状态:", status)
print("注意: io_uring ≠ 真正 SPDK NVMe")
print("真正 SPDK NVMe 需要: SPDK C 库 + NVMe 设备绑定 (vfio/uio)")
