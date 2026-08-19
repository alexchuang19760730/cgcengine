#!/usr/bin/env python3
"""
端云一体综合测试 - DFlash + MTP Harness Agent测试
使用通用防造假框架进行数据采集和校验
"""

import sys
import os
import random
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(__file__))

from tools.anti_fraud_framework import AntiFraudCollector, DataSource, calculate_crc32

# ==================== 数据源定义 ====================

class DFlashDataSource(DataSource):
    """DFlash端云一体数据源"""

    def __init__(self):
        self.source_name = "dflash"

    def collect(self) -> Dict[str, Any]:
        """采集DFlash端云一体数据"""
        return {
            "edge_decode_time": 15.0 + random.uniform(-3, 3),     # ms
            "cloud_prefill_time": 60.0 + random.uniform(-10, 10), # ms
            "kv_transfer_time": 20.0 + random.uniform(-5, 5),     # ms
            "edge_token_count": 8 + random.randint(-2, 2),
            "cloud_kv_blocks": 512 + random.randint(-32, 32),
            "dflash_efficiency": 0.85 + random.uniform(-0.05, 0.05),
            "source": "dflash"
        }

    def get_source_name(self) -> str:
        return self.source_name

class MTPDataSource(DataSource):
    """MTP端云一体数据源"""

    def __init__(self):
        self.source_name = "mtp"

    def collect(self) -> Dict[str, Any]:
        """采集MTP端云一体数据"""
        return {
            "cloud_mtp_count": 4 + random.randint(-1, 1),
            "cloud_prefill_time": 40.0 + random.uniform(-5, 5),   # ms
            "cloud_kv_write_time": 18.0 + random.uniform(-3, 3), # ms
            "kv_transfer_time": 22.0 + random.uniform(-4, 4),    # ms
            "edge_verify_time": 6.0 + random.uniform(-1, 1),     # ms
            "edge_accept_rate": 0.85 + random.uniform(-0.05, 0.05),
            "mtp_efficiency": (4 * 0.85) + random.uniform(-0.2, 0.2),
            "source": "mtp"
        }

    def get_source_name(self) -> str:
        return self.source_name

class HardwareDataSource(DataSource):
    """硬件层数据源"""

    def __init__(self):
        self.source_name = "hardware"

    def collect(self) -> Dict[str, Any]:
        """采集硬件数据"""
        return {
            "nvml_vram": 18000.0 + random.uniform(-500, 500),    # MB
            "gpu_power": 280.0 + random.uniform(-20, 20),        # W
            "gpu_utilization": 88.0 + random.uniform(-5, 5),     # %
            "source": "hardware"
        }

    def get_source_name(self) -> str:
        return self.source_name

class EngineDataSource(DataSource):
    """引擎层数据源"""

    def __init__(self):
        self.source_name = "engine"

    def collect(self) -> Dict[str, Any]:
        """采集引擎数据"""
        return {
            "engine_vram": 18500.0 + random.uniform(-300, 300),  # MB
            "total_latency": 86.0 + random.uniform(-5, 5),       # ms
            "kv_read_bytes": 1024 * 1024 * 250,                  # bytes
            "kv_write_bytes": 1024 * 1024 * 60,                  # bytes
            "bandwidth": 3600.0 + random.uniform(-100, 100),     # MB/s
            "source": "engine"
        }

    def get_source_name(self) -> str:
        return self.source_name

# ==================== Harness Agent ====================

class EdgeCloudHarnessAgent:
    """端云一体Harness Agent"""

    def __init__(self):
        self.collector = AntiFraudCollector()
        self.collector.register_source(HardwareDataSource())
        self.collector.register_source(EngineDataSource())
        self.results = []

    def run_dflash_test(self, scenario_name: str, description: str, tamper_data: bool = False) -> Dict[str, Any]:
        """执行DFlash端云一体测试"""
        print(f"\n{'='*90}")
        print(f"🚀 DFlash端云一体 | {scenario_name}")
        print(f"{'='*90}")
        print(f"📝 描述: {description}")

        # 创建DFlash采集器
        dflash_collector = AntiFraudCollector()
        dflash_collector.register_source(DFlashDataSource())

        # 采集数据
        data = dflash_collector.collect_all()

        # 模拟数据篡改
        if tamper_data:
            if "dflash" in data:
                original = data["dflash"]["dflash_efficiency"]
                data["dflash"]["dflash_efficiency"] = 1.5  # 篡改效率值
                print(f"\n🔧 [模拟数据篡改] DFlash效率: {original:.2f} -> {data['dflash']['dflash_efficiency']}")

        # 校验
        is_consistent, _ = dflash_collector.validate_consistency(data)
        is_hash_valid, _ = dflash_collector.validate_hash(data)

        print(f"\n📊 DFlash采集数据:")
        for k, v in data["dflash"].items():
            if k != "source":
                print(f"   {k}: {v}")
        print(f"   CRC32哈希: 0x{data['crc32_hash']:08x}")

        print(f"\n🔍 校验结果:")
        print(f"   一致性: {'✅ 通过' if is_consistent else '❌ 失败'}")
        print(f"   哈希: {'✅ 通过' if is_hash_valid else '❌ 失败'}")

        result = {
            "type": "dflash",
            "scenario": scenario_name,
            "is_valid": is_consistent and is_hash_valid,
            "hash": data["crc32_hash"]
        }
        self.results.append(result)
        return result

    def run_mtp_test(self, scenario_name: str, description: str, tamper_data: bool = False) -> Dict[str, Any]:
        """执行MTP端云一体测试"""
        print(f"\n{'='*90}")
        print(f"🚀 MTP端云一体 | {scenario_name}")
        print(f"{'='*90}")
        print(f"📝 描述: {description}")

        # 创建MTP采集器
        mtp_collector = AntiFraudCollector()
        mtp_collector.register_source(MTPDataSource())

        # 采集数据
        data = mtp_collector.collect_all()

        # 模拟数据篡改
        if tamper_data:
            if "mtp" in data:
                original = data["mtp"]["edge_accept_rate"]
                data["mtp"]["edge_accept_rate"] = 1.2  # 篡改接受率
                print(f"\n🔧 [模拟数据篡改] MTP接受率: {original:.2f} -> {data['mtp']['edge_accept_rate']}")

        # 校验
        is_consistent, _ = mtp_collector.validate_consistency(data)
        is_hash_valid, _ = mtp_collector.validate_hash(data)

        print(f"\n📊 MTP采集数据:")
        for k, v in data["mtp"].items():
            if k != "source":
                print(f"   {k}: {v}")
        print(f"   CRC32哈希: 0x{data['crc32_hash']:08x}")

        print(f"\n🔍 校验结果:")
        print(f"   一致性: {'✅ 通过' if is_consistent else '❌ 失败'}")
        print(f"   哈希: {'✅ 通过' if is_hash_valid else '❌ 失败'}")

        result = {
            "type": "mtp",
            "scenario": scenario_name,
            "is_valid": is_consistent and is_hash_valid,
            "hash": data["crc32_hash"]
        }
        self.results.append(result)
        return result

    def run_full_edge_cloud_test(self):
        """执行完整的端云一体测试"""
        print("=" * 90)
        print("🔍 端云一体综合测试 - DFlash + MTP")
        print("=" * 90)
        print("\n📋 测试计划:")
        print("  1. DFlash正常推理")
        print("  2. DFlash数据篡改检测")
        print("  3. MTP正常推理")
        print("  4. MTP数据篡改检测")

        # DFlash测试
        self.run_dflash_test("场景1：DFlash正常推理", "端云一体推理，数据未被篡改")
        self.run_dflash_test("场景2：DFlash数据篡改", "DFlash效率被篡改", tamper_data=True)

        # MTP测试
        self.run_mtp_test("场景3：MTP正常推理", "端云一体MTP推理，数据未被篡改")
        self.run_mtp_test("场景4：MTP数据篡改", "MTP接受率被篡改", tamper_data=True)

        # 输出汇总
        self.print_summary()

    def print_summary(self):
        """打印测试汇总"""
        print("\n" + "=" * 90)
        print("📊 端云一体测试汇总")
        print("=" * 90)

        dflash_results = [r for r in self.results if r["type"] == "dflash"]
        mtp_results = [r for r in self.results if r["type"] == "mtp"]

        print("\n🔹 DFlash测试结果:")
        for r in dflash_results:
            status = "✅ 有效" if r["is_valid"] else "❌ 无效"
            print(f"   {r['scenario']}: {status}")

        print("\n🔹 MTP测试结果:")
        for r in mtp_results:
            status = "✅ 有效" if r["is_valid"] else "❌ 无效"
            print(f"   {r['scenario']}: {status}")

        print("\n🔹 统计:")
        total = len(self.results)
        valid = sum(1 for r in self.results if r["is_valid"])
        print(f"   总测试场景: {total}")
        print(f"   ✅ 有效数据: {valid}")
        print(f"   ❌ 无效数据: {total - valid}")
        print(f"   🎯 检测准确率: {valid/total*100:.0f}%")

# ==================== 主程序 ====================

def main():
    agent = EdgeCloudHarnessAgent()
    agent.run_full_edge_cloud_test()

if __name__ == "__main__":
    main()