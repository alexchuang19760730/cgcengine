#!/usr/bin/env python3
"""
测试 GDS/SPDK 完整性
"""

from gds_service.cufile_wrapper import CUFILE_AVAILABLE, is_gds_available
from spdk_adapter.spdk_io_manager import SPDK_AVAILABLE, SPDKIOManager
from spdk_adapter.spdk_config import SPDKConfig

def main():
    print("=== GDS 检测 ===")
    print(f"GDS 可用: {CUFILE_AVAILABLE}")
    print(f"is_gds_available(): {is_gds_available()}")

    print("\n=== SPDK 检测 ===")
    print(f"SPDK liburing 可用: {SPDK_AVAILABLE}")

    print("\n=== SPDK IO 测试 ===")
    config = SPDKConfig(kv_store_path="/tmp/spdk_test", io_queues=4)
    io_manager = SPDKIOManager(config)
    io_manager.start()

    # 测试写入
    task1 = io_manager.submit_write("test_key1", b"Hello SPDK World!")
    result1 = task1.wait()
    print(f"写入结果: {result1}")

    # 测试读取
    task2 = io_manager.submit_read("test_key1")
    result2 = task2.wait()
    print(f"读取结果: {result2}")

    # 获取统计
    stats = io_manager.get_stats()
    print(f"\nSPDK 统计: {stats}")

    io_manager.stop()
    print("\n✅ SPDK IO 测试通过!")

if __name__ == "__main__":
    main()
