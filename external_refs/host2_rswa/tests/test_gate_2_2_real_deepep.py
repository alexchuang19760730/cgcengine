#!/usr/bin/env python3
"""
CGC Gate 2.2 DeepEP MoE 负载均衡 - 真实 CGC DeepEP 组件测试
直接使用 cgc_engine.deep_ep 中的真实实现
"""

import sys
sys.path.insert(0, '/Users/alexchuang/Documents/embodied/ComputeGraphCompiler-main')

import torch
import numpy as np

print("="*70)
print("🔥 Gate 2.2 - 真实 CGC DeepEP 组件测试")
print("="*70)

results = {}

print("\n" + "="*70)
print("【第一步】导入 CGC 真实 DeepEP 组件")
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
        set_deepep_mode,
    )
    
    print("   ✅ 成功导入 CGC 真实 DeepEP 组件")
    print("   • DeepEPConfig: DeepEP 配置类")
    print("   • DeepEPDispatcher: 专家调度器")
    print("   • DeepEPMoE: MoE 专家并行实现")
    print("   • WaterfillBalancer: 带宽感知注水均衡")
    print("   • LPLBBalancer: 线性规划负载均衡")
    print("   • EPLBBalancer: 静态专家副本调度")
    print("   • DeepEPMode: DeepEP 模式枚举")
    
    results["step1"] = {
        "success": True, 
        "components": [
            "DeepEPConfig", 
            "DeepEPDispatcher", 
            "DeepEPMoE",
            "WaterfillBalancer",
            "LPLBBalancer",
            "EPLBBalancer",
            "DeepEPMode"
        ]
    }
    
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    results["step1"] = {"success": False, "error": str(e)}
    sys.exit(1)

print("\n" + "="*70)
print("【第二步】检查并设置 DeepEP 运行模式")
print("="*70)

try:
    current_mode = get_deepep_mode()
    print(f"   • 当前模式: {current_mode}")
    
    set_deepep_mode(DeepEPMode.DEEP_SEEK)