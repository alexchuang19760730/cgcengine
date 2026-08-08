#!/usr/bin/env python3
"""
MTP端云一体测试 - 使用通用防造假框架
"""

import sys
import os
import random
import time
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from tools.anti_fraud_framework import AntiFraudCollector, DataSource, calculate_crc32

class MTPDataSource(DataSource):
    """MTP端云一体数据源"""

    def __init__(self):
        self.source_name = "mtp"

    def collect(self) -> Dict[str, Any]:
        """采集MTP端云一体数据"""
        return {
            "cloud_mtp_count": 4 + random.randint(-1, 1),
            "cloud_prefill_time": 40.0 + random.uniform(-5, 5),
            "cloud_kv_write_time": 18.0 + random.uniform(-3, 3),
            "kv_transfer_time": 22.0 + random.uniform(-4, 4),
            "edge_verify_time": 6.0 + random.uniform(-1, 1),
            "edge_accept_rate": 0.85 + random.uniform(-0.05, 0.05),
            "source": "mtp"
        }

    def get_source_name(self) -> str:
        return self.source_name

class MTPHarnessAgent:
    """MTP Harness Agent - 使用通用防造假框架"""

    def __init__(self):
        self.collector = AntiFraudCollector()
        self.collector.register_source(MTPDataSource())

    def run_test(self, scenario_name: str, description: str, tamper_data: bool = False) -> Dict[str, Any]:
        """执行MTP测试"""
        print(f"\n{'='*90}")
        print(f"🚀 MTP Harness Agent | {scenario_name}")
        print(f"{'='*90}")
        print(f"📝 描述: {description}")

        # 采集数据
        data = self.collector.collect_all()

        # 如果需要模拟数据篡改
        if tamper_data:
            if "mtp" in data:
                original = data["mtp"]["edge_accept_rate"]
                data["mtp"]["edge_accept_rate"] = 1.5  # 篡改数据
                print(f"\n🔧 [模拟数据篡改] MTP接受率: {original:.2f} -> {data['mtp']['edge_accept_rate']}")

        # 校验
        is_consistent, consistency_reason = self.collector.validate_consistency(data)
        is_hash_valid, hash_reason = self.collector.validate_hash(data)

        print(f"\n📊 采集数据:")
        for k, v in data["mtp"].items():
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
    print("🔍 MTP端云一体测试 - 使用通用防造假框架")
    print("=" * 90)

    agent = MTPHarnessAgent()

    # 场景1：正常数据
    agent.run_test("场景1：正常MTP推理", "MTP云端预测4token + 端侧验证")

    # 场景2：数据篡改检测
    agent.run_test("场景2：数据篡改检测", "MTP接受率被篡改", tamper_data=True)

    print("\n" + "=" * 90)
    print("✅ MTP测试完成")
    print("=" * 90)

if __name__ == "__main__":
    main()