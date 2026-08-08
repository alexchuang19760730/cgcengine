#!/usr/bin/env python3
"""
CGC Gate 5.0 Audit Trace Replay Visualization 测试文件
验证可审计、可追踪、可回溯、可可视化能力
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cgc_engine.core import CGCEngine

class Gate50Test:
    def __init__(self):
        self.engine = CGCEngine()
        self.gate_id = "CGC_Gate_5.0_audit_trace_replay_visualization"
        self.gate_version = "5.0"
    
    def test_initialization(self):
        """测试 Gate 5.0 初始化"""
        print(f"[TEST] 初始化 Gate 5.0 ({self.gate_id})...")
        self.engine.initialize_gate(self.gate_id, self.gate_version)
        assert self.engine.get_gate_status(self.gate_id) == "active"
        print(f"[PASS] Gate 5.0 初始化成功")
    
    def test_capabilities(self):
        """测试所有能力"""
        capabilities = [
            ("cli_unified_command_set", "统一 CLI 指令集"),
            ("train_inference_unification", "训练推理一体化"),
            ("self_harness_loop", "Self-Harness 三阶段闭环"),
            ("auditable", "可审计"),
            ("traceable", "可追踪"),
            ("replayable", "可回溯"),
            ("visualizable", "可可视化"),
        ]
        
        print(f"[TEST] 验证 Gate 5.0 能力矩阵...")
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
        
        feedback = self.engine.get_self_harness_feedback(self.gate_id)
        assert feedback["loop_complete"] == True
        print(f"  [PASS] 闭环反馈: 完成")
    
    def test_audit(self):
        """测试审计能力"""
        print(f"[TEST] 验证审计能力...")
        result = self.engine.start_audit(self.gate_id)
        assert result["recording"] == True
        print(f"  [PASS] 审计记录已启动")
        
        # 生成测试操作
        for i in range(5):
            self.engine.record_operation(self.gate_id, f"test_operation_{i}", {"param": i})
        
        records = self.engine.get_audit_records(self.gate_id, limit=5)
        assert len(records) == 5
        print(f"  [PASS] 审计记录: {len(records)} 条")
    
    def test_trace(self):
        """测试全链路追踪"""
        print(f"[TEST] 验证全链路追踪...")
        trace_id = self.engine.start_trace(self.gate_id, "test_trace")
        assert trace_id is not None
        
        spans = [
            ("gate_enter", "进入 Gate"),
            ("model_load", "模型加载"),
            ("inference", "推理执行"),
            ("gate_exit", "退出 Gate"),
        ]
        
        for span_name, description in spans:
            self.engine.record_span(self.gate_id, trace_id, span_name, description)
        
        result = self.engine.get_trace(self.gate_id, trace_id)
        assert len(result["spans"]) == 4
        print(f"  [PASS] 追踪跨度: {len(result['spans'])} 个")
    
    def test_replay(self):
        """测试状态回溯"""
        print(f"[TEST] 验证状态回溯...")
        # 创建快照
        snapshot_id = self.engine.create_snapshot(self.gate_id)
        assert snapshot_id is not None
        print(f"  [PASS] 创建快照: {snapshot_id}")
        
        # 回放快照
        result = self.engine.replay_snapshot(self.gate_id, snapshot_id)
        assert result["success"] == True
        print(f"  [PASS] 快照回放成功")
    
    def run_all_tests(self):
        """运行所有测试"""
        print(f"\n{'='*60}")
        print(f"  CGC Gate 5.0 Audit Trace Replay Visualization 测试套件")
        print(f"{'='*60}")
        
        self.test_initialization()
        self.test_capabilities()
        self.test_cli_commands()
        self.test_self_harness()
        self.test_audit()
        self.test_trace()
        self.test_replay()
        
        print(f"\n{'='*60}")
        print(f"  ✅ Gate 5.0 所有测试通过!")
        print(f"{'='*60}")
        return True

if __name__ == "__main__":
    test = Gate50Test()
    test.run_all_tests()
