#!/usr/bin/env python3
"""
CGC Gate 2.2 DeepEP MoE 负载均衡 - CGC 自研 DeepEP 组件测试
使用 cgc_engine/deep_ep/ 目录下的真实实现
"""

import sys
sys.path.insert(0, '/Users/alexchuang/Documents/embodied/ComputeGraphCompiler-main')

import torch
import time

print("="*70)
print("🔥 Gate 2.2 - CGC 自研 DeepEP MoE 组件测试")
print("="*70)

results = {}

print("\n" + "="*70)
print("【第一步】导入 CGC 自研 DeepEP 组件")
print("="*70)

try:
    from cgc_engine.deep_ep import (
        DeepEPConfig,
        DeepEPDispatcher,
        DeepEPMoE,
        WaterfillBalancer,
        LPLBBalancer,
        EPLBBalancer,
        DeepEPMode,
        get_deepep_mode,
        set_deepep_mode
    )
    
    print("   ✅ 成功导入 CGC 自研 DeepEP 组件")
    print("   • DeepEPConfig: 配置类")
    print("   • DeepEPDispatcher: 专家调度器")
    print("   • DeepEPMoE: MoE 层实现")
    print("   • WaterfillBalancer: 带宽感知注水均衡")
    print("   • LPLBBalancer: 线性规划均衡器")
    print("   • EPLBBalancer: 静态专家副本调度")
    
    results["step1"] = {"success": True}
    
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    results["step1"] = {"success": False, "error": str(e)}
    sys.exit(1)

print("\n" + "="*70)
print("【第二步】初始化 DeepEPConfig (DeepSeek V4 Flash 配置)")
print