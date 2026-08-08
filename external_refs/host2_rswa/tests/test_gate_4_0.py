#!/usr/bin/env python3
"""
CGC Gate 4.0 Embodied 测试文件
验证端云协同深度融合与具身智能支持能力
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cgc_engine.core import CGCEngine

class Gate40Test:
    def __init__(self):
        self.engine = CGCEngine()
        self.gate_id = "CGC_Gate_4.0_embodied"
        self.gate_version = "4.0"
    
    def test_initialization(self):
        """测试 Gate 4.0 初始化"""
        print(f"[TEST] 初始化 Gate 4.0 ({self.gate_id})...")
        self.engine.initialize_gate(self.gate_id, self.gate_version)
        assert self.engine.get_gate_status(self.gate_id) == "active"
        print(f"[PASS] Gate 4.0 初始化成功")
    
    def test_capabilities(self):
        """测试所有能力"""
        capabilities = [
            ("cli_unified_command_set", "统一 CLI 指令集"),
            ("train_inference_unification", "训练推理一体化"),
            ("self_harness_loop", "Self-Harness 三阶段闭环"),
            ("edge_cloud_collaboration", "端云协同深度融合"),
            ("embodied_intelligence", "具身智能支持"),
            ("sensor_fusion", "传感器融合"),
            ("robot_control", "机器人控制接口"),
        ]
        
        print(f"[TEST] 验证 Gate 4.0 能力矩阵...")
        for cap_id, cap_name in capabilities:
            result = self.engine.check_capability(self.gate_id, cap_id)
            assert result["status"] == "done", f"能力 {cap_name} 未完成"
            print(f"  [PASS] {cap_name}: {result['status']}")
    
    def test_cli_commands(self):
        """测试 CLI 指令集"""
        commands = ["train", "infer", "deploy", "tune", "bench", "validate", "monitor", "audit", "ops"]
        print(f"[TEST] 验证 CLI 指令集...")
        for cmd in commands:
            result = self.engine.execute_cli(self.gate_id, cmd, "--help")
            assert result["success"] == True
            print(f"  [PASS] cgc {cmd}")
    
    def test_self_harness(self):
        """测试 Self-Harness 三阶段闭环"""
        print(f"[TEST] 验证 Self-Harness 三阶段闭环...")
        phases = ["phase1_init", "phase2_execute", "phase3_optimize"]
        for phase in phases:
            result = self.engine.run_self_harness_phase(self.gate_id, phase)
            assert result["status"] == "completed"
            print(f"  [PASS] {phase}: {result['status']}")
        
        # 测试闭环反馈
        feedback = self.engine.get_self_harness_feedback(self.gate_id)
        assert feedback["loop_complete"] == True
        print(f"  [PASS] 闭环反馈: 完成")
    
    def test_edge_cloud_sync(self):
        """测试端云状态同步"""
        print(f"[TEST] 验证端云状态同步...")
        result = self.engine.sync_edge_cloud_state(self.gate_id)
        assert result["latency_ms"] < 50, f"同步延迟超过 50ms: {result['latency_ms']}ms"
        assert result["success"] == True
        print(f"  [PASS] 端云同步延迟: {result['latency_ms']}ms")
    
    def run_all_tests(self):
        """运行所有测试"""
        print(f"\n{'='*60}")
        print(f"  CGC Gate 4.0 Embodied 测试套件")
        print(f"{'='*60}")
        
        self.test_initialization()
        self.test_capabilities()
        self.test_cli_commands()
        self.test_self_harness()
        self.test_edge_cloud_sync()
        
        print(f"\n{'='*60}")
        print(f"  ✅ Gate 4.0 所有测试通过!")
        print(f"{'='*60}")
        return True

if __name__ == "__main__":
    test = Gate40Test()
    test.run_all_tests()
