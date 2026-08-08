#!/usr/bin/env python3
"""
DFlash端云一体测试 - 使用通用防造假框架
"""

import sys
import os
import random
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(__file__))

from tools.anti_fraud_framework import AntiFraudCollector, DataSource

class DFlashDataSource(DataSource):
    """DFlash端云一体数据源"""

    def __init__(self):
        self.source_name = "dflash"

    def collect(self) -> Dict[str, Any]:
        """采集DFlash端云一体数据"""
        return {
            "edge_decode_time": 15.0 + random.uniform(-3, 3),
            "cloud_prefill_time": 60.0 + random.uniform(-10, 10),
            "kv_transfer_time": 20.0 + random.uniform(-5, 5),
            "edge_token_count": 8 + random.randint(-2, 2),
            "cloud_kv_blocks": 512 + random.randint(-32, 32),
            "source": "dflash"
        }

    def get_source_name(self) -> str:
        return self.source_name

class DFlashHarnessAgent:
    """DFlash Harness Agent - 使用通用防造假框架"""

    def __init__(self):
        self.collector = AntiFraudCollector()
        self.collector.register_source(DFlashDataSource())

    def run_test(self, scenario_name: str, description: str, tamper_data: bool = False) -> Dict[str, Any]:
        """执行DFlash测试"""
        print(f"\n{'='*90}")
        print(f"🚀 DFlash Harness Agent | {scenario_name}")
        print(f"{'='*90}")
        print(f"📝 描述: {description}")

        # 采集数据
        data = self.collector.collect_all()

        # 如果需要模拟数据篡改
        if tamper_data:
            if "dflash" in data:
                original = data["dflash"]["edge_token_count"]
                data["dflash"]["edge_token_count"] = 100  # 篡改数据
                print(f"\n🔧 [模拟数据篡改] 端侧token数: {original} -> {data['dflash']['edge_token_count']}")

        # 校验
        is_consistent, _ = self.collector.validate_consistency(data)
        is_hash_valid, hash_reason = self.collector.validate_hash(data)

        print(f"\n📊 采集数据:")
        for k, v in data["dflash"].items():
            if k != "source":
                print(f"   {k}: {v}")
        print(f"   CRC32哈希: 0x{data['crc32_hash']:08x}")

        print(f"\n🔍 校验结果:")
        print(f"   一致性: {'✅ 通过' if is_consistent else '❌ 失败'}")
        print(f"   哈希: {'✅ 通过' if is_hash_valid else '❌ 失败'}")

        return {
            "scenario": scenario_name,
            "is_valid": is_consistent and is_hash_valid,
            "hash": data["crc32_hash"]
        }

def main():
    print("=" * 90)
    print("🔍 DFlash端云一体测试 - 使用通用防造假框架")
    print("=" * 90)

    agent = DFlashHarnessAgent()

    # 场景1：正常数据
    agent.run_test("场景1：正常DFlash推理", "端云一体推理，数据未被篡改")

    # 场景2：数据篡改检测
    agent.run_test("场景2：数据篡改检测", "端侧token数被篡改", tamper_data=True)

    print("\n" + "=" * 90)
    print("✅ DFlash测试完成")
    print("=" * 90)

if __name__ == "__main__":
    main()