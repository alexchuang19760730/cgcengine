#!/usr/bin/env python3
"""
端云推理测试脚本
===============

测试 Phi-MoE 和 Qwen 7B 模型的端云推理流程

架构：
- 云侧 (10.100.200.65): Prefill + MoE 计算
- 端侧 (本地): Decode

测试模型：
1. Phi-3.5-MoE-instruct-Q4_K_M.gguf (24GB)
2. qwen2.5-7b-q4_k_m.gguf (4.4GB)
"""

import sys
import os
import json
from typing import Dict, Any, List

def create_test_script():
    """创建服务器端测试脚本"""
    script = '''
import sys
import os
sys.path.insert(0, '/home/gs01/cgc_engine')

from cgc.unified_inference_engine import UnifiedInferenceEngine, ModelConfig, BackendType
import time

def test_phi_moe():
    """测试 Phi-MoE 模型"""
    print("="*70)
    print("测试 1: Phi-3.5-MoE-instruct-Q4_K_M.gguf")
    print("="*70)
    
    config = ModelConfig(
        model_path="/home/gs01/cgc_engine/models/Phi-3.5-MoE-instruct-Q4_K_M.gguf",
        model_type="moe",
        num_experts=16,
        expert_dim=4096,
        top_k=2,
    )
    
    engine = UnifiedInferenceEngine(config)
    
    if engine.initialize():
        print("✅ 引擎初始化成功")
        
        prompts = [
            "什么是量子计算？",
            "解释一下机器学习的基本概念",
            "写一首关于春天的诗",
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"\\n[{i+1}] 输入: {prompt}")
            start = time.time()
            result = engine.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
                use_moe=True,
            )
            elapsed = time.time() - start
            print(f"输出: {result.text[:100]}...")
            print(f"耗时: {elapsed:.2f}s")
            print(f"专家: {result.expert_ids}")
    else:
        print("❌ 引擎初始化失败")

def test_qwen_7b():
    """测试 Qwen 7B 模型"""
    print("\\n" + "="*70)
    print("测试 2: qwen2.5-7b-q4_k_m.gguf")
    print("="*70)
    
    config = ModelConfig(
        model_path="/home/gs01/cgc_engine/models/qwen2.5-7b-q4_k_m.gguf",
        model_type="dense",
        num_experts=1,
        top_k=1,
    )
    
    engine = UnifiedInferenceEngine(config)
    
    if engine.initialize():
        print("✅ 引擎初始化成功")
        
        prompts = [
            "什么是人工智能？",
            "解释一下深度学习",
            "推荐一本好书",
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"\\n[{i+1}] 输入: {prompt}")
            start = time.time()
            result = engine.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
                use_moe=False,
            )
            elapsed = time.time() - start
            print(f"输出: {result.text[:100]}...")
            print(f"耗时: {elapsed:.2f}s")
    else:
        print("❌ 引擎初始化失败")

if __name__ == "__main__":
    test_phi_moe()
    test_qwen_7b()
    print("\\n" + "="*70)
    print("✅ 端云推理测试完成")
    print("="*70)
'''
    return script

def run_remote_test():
    """在服务器上运行测试"""
    import subprocess
    
    # 创建测试脚本
    test_script = create_test_script()
    
    # 将脚本传输到服务器并执行
    cmd = f'''export SSHPASS='za34tq4cpg' && sshpass -e ssh -o StrictHostKeyChecking=no gs01@10.100.200.65 "cd /home/gs01/cgc_engine && python3 -c '{test_script}'"'''
    
    print("🚀 正在服务器上运行端云推理测试...")
    print("=" * 70)
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("❌ 测试执行失败")
        print("错误输出:", result.stderr)
    
    return result.returncode == 0

def test_end_to_end():
    """端到端测试"""
    print("\n" + "="*70)
    print("端云一体推理测试")
    print("="*70)
    
    # 1. 检查服务器连接
    print("\n[1/3] 检查服务器连接...")
    import subprocess
    result = subprocess.run(
        '''export SSHPASS='za34tq4cpg' && sshpass -e ssh -o StrictHostKeyChecking=no gs01@10.100.200.65 "echo OK"''',
        shell=True,
        capture_output=True,
        text=True,
    )
    
    if "OK" in result.stdout:
        print("✅ 服务器连接正常")
    else:
        print("❌ 服务器连接失败")
        return False
    
    # 2. 检查模型文件
    print("\n[2/3] 检查模型文件...")
    result = subprocess.run(
        '''export SSHPASS='za34tq4cpg' && sshpass -e ssh -o StrictHostKeyChecking=no gs01@10.100.200.65 "ls -la /home/gs01/cgc_engine/models/"''',
        shell=True,
        capture_output=True,
        text=True,
    )
    
    if "Phi-3.5-MoE" in result.stdout and "qwen2.5-7b" in result.stdout:
        print("✅ 模型文件就绪")
        print(result.stdout)
    else:
        print("❌ 模型文件缺失")
        return False
    
    # 3. 运行推理测试
    print("\n[3/3] 运行推理测试...")
    return run_remote_test()

if __name__ == "__main__":
    success = test_end_to_end()
    
    if success:
        print("\n🎉 端云推理测试成功！")
    else:
        print("\n❌ 端云推理测试失败")
        sys.exit(1)
