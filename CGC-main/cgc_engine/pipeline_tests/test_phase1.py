#!/usr/bin/env python3
"""
MagiCompiler Phase 1 测试脚本
测试 torch.compile + CUDA Graph 集成
"""

import os
import sys
import time
import torch
from vllm import LLM, SamplingParams


def test_cuda_graph_basic():
    """测试 CUDA Graph 基础功能"""
    print("=" * 60)
    print("测试 1: CUDA Graph 基础功能")
    print("=" * 60)
    
    # 创建简单模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(512, 512).cuda()
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            return self.relu(self.linear(x))
    
    model = SimpleModel()
    input_tensor = torch.randn(1, 128, 512).cuda()
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)
    
    # Eager 模式性能测试
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()
    eager_time = (time.time() - start) * 1000 / 100
    print(f"Eager 模式: {eager_time:.2f} ms/iter")
    
    # CUDA Graph 模式性能测试
    graph = torch.cuda.CUDAGraph()
    output_placeholder = torch.empty_like(model(input_tensor))
    
    with torch.cuda.graph(graph):
        output_placeholder.copy_(model(input_tensor))
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    graph_time = (time.time() - start) * 1000 / 100
    print(f"CUDA Graph 模式: {graph_time:.2f} ms/iter")
    print(f"加速比: {eager_time/graph_time:.2f}x")
    
    return eager_time, graph_time


def test_torch_compile():
    """测试 torch.compile 功能"""
    print("\n" + "=" * 60)
    print("测试 2: torch.compile 功能")
    print("=" * 60)
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(512, 1024).cuda()
            self.linear2 = torch.nn.Linear(1024, 512).cuda()
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.linear1(x))
            return self.linear2(x)
    
    model = SimpleModel()
    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    input_tensor = torch.randn(1, 128, 512).cuda()
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = compiled_model(input_tensor)
    
    # 测试性能
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = compiled_model(input_tensor)
    torch.cuda.synchronize()
    compile_time = (time.time() - start) * 1000 / 100
    print(f"torch.compile 模式: {compile_time:.2f} ms/iter")
    
    return compile_time


def test_vllm_basic():
    """测试 vLLM 基础功能"""
    print("\n" + "=" * 60)
    print("测试 3: vLLM 基础功能")
    print("=" * 60)
    
    try:
        llm = LLM(
            model="/home/gs01/models/Qwen/Qwen2___5-7B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=2048,
        )
        
        prompts = ["Hello, my name is", "The quick brown fox"]
        sampling_params = SamplingParams(max_tokens=16)
        
        # 预热
        _ = llm.generate(["Warmup"], sampling_params)
        
        # 测试
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = (time.time() - start) * 1000
        
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Output: {output.outputs[0].text}")
        
        print(f"\n推理耗时: {elapsed:.2f} ms")
        return True
    except Exception as e:
        print(f"vLLM 测试失败: {e}")
        return False


def test_magi_engine():
    """测试 MagiEngine 集成"""
    print("\n" + "=" * 60)
    print("测试 4: MagiEngine 集成")
    print("=" * 60)
    
    try:
        from magi_engine import create_magi_engine
        
        engine = create_magi_engine(
            model_name_or_path="/home/gs01/models/Qwen/Qwen2___5-7B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=2048,
        )
        
        # 预热
        engine.warmup()
        
        # 生成测试
        prompts = ["Hello, my name is", "The quick brown fox"]
        outputs = engine.generate(prompts, SamplingParams(max_tokens=16))
        
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Output: {output.outputs[0].text}")
        
        # 基准测试
        results = engine.benchmark(
            prompts=["Hello, world!"],
            sampling_params=SamplingParams(max_tokens=64),
            num_iterations=3
        )
        
        return results
    except Exception as e:
        import traceback
        print(f"MagiEngine 测试失败: {e}")
        traceback.print_exc()
        return None


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("MagiCompiler Phase 1 测试套件")
    print("torch.compile + CUDA Graph 集成测试")
    print("=" * 80)
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return
    
    print(f"✅ CUDA 可用 (设备: {torch.cuda.get_device_name(0)})")
    print(f"✅ PyTorch 版本: {torch.__version__}")
    
    # 运行测试
    test_cuda_graph_basic()
    test_torch_compile()
    test_vllm_basic()
    test_magi_engine()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
