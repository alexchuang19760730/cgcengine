"""
vLLM Custom Backend 集成示例
展示如何把新架构用在你的代码中
"""

from cgc.vllm_custom_backend_system import (
    VLLMBackend,
    VLLMComputeRequest,
    ComputeType,
    register_vllm_backend,
    set_current_vllm_backend,
    get_current_vllm_backend,
    create_magicompiler_backend,
    vllm_compute_with_backend,
)


def example_integration():
    """
    完整集成示例
    
    对应你原来的代码现在怎么用，现在也怎么用
    """
    print("=" * 60)
    print("vLLM Custom Backend System 完整示例")
    print("=" * 60)
    print()
    
    # ┌─────────────────────────────────────────────────┐
    # 1. 创建并注册 MagiCompiler backend
    # └─────────────────────────────────────────────────┘
    print("📦 步骤 1: 注册 MagiCompiler backend")
    magi_backend = create_magicompiler_backend()
    print()
    
    # ┌─────────────────────────────────────────────────┐
    # 2. 你的 vLLM 模型代码（几乎不用改）
    # └─────────────────────────────────────────────────┘
    print("🚀 步骤 2: 用 backend 执行 vLLM compute")
    
    import torch
    from typing import Optional
    
    # 模拟你的 vLLM 模型
    class MockVLLMModel:
        def forward(self, input_ids: torch.Tensor):
            print(f"  → vLLM native forward")
            return torch.randn_like(input_ids.shape[0], input_ids.shape[1], 32000)
    
    model = MockVLLMModel()
    input_ids = torch.randint(0, 32000, (1, 128))
    
    # ┌─────────────────────────────────────────────────┐
    # 3. 用 vllm_compute_with_backend 包装
    #（唯一需要改的地方）
    # └─────────────────────────────────────────────────┘
    print("📝 步骤 3: 用 backend 包装")
    
    # 原来的代码（不用改）
    # output = model.forward(input_ids)
    
    # 改成：
    req = VLLMComputeRequest(
        compute_type=ComputeType.FULL_FORWARD,
        input_ids=input_ids
    )
    output = vllm_compute_with_backend(
        model.forward, req, input_ids
    )
    
    print(f"✅ 计算完成: {output.shape}")
    print()
    
    # ┌─────────────────────────────────────────────────┐
    # 4. 查看统计
    # └─────────────────────────────────────────────────┘
    print("📊 步骤 4: 查看统计")
    stats = magi_backend.get_stats()
    print(f"  Compute 次数: {stats.compute_count}")
    print()
    
    print("=" * 60)


def example_with_llamacpp_backend():
    """llama.cpp backend 示例"""
    from cgc.vllm_custom_backend_system import LlamaCppVLLMBackend
    
    llama_backend = LlamaCppVLLMBackend()
    register_vllm_backend(llama_backend)
    set_current_vllm_backend(llama_backend.name)
    print(f"✅ LlamaCpp backend 已激活")


class YourOriginalCode():
    """
    你的原始代码（几乎不用改）
    只需要把 model.forward 改成用 vllm_compute_with_backend 包装
    """
    def __init__(self):
        from cgc.cgc.vllm_custom_backend_system import (
            VLLMComputeRequest,
            ComputeType,
            vllm_compute_with_backend,
        )
        
        # 你的 vLLM 模型
        # from vllm import LLM, SamplingParams
        
    def generate_with_backend(self, prompt: str):
        """
        用 backend 增强的 generate
        
        原来:
            outputs = self.llm.generate(prompt, sampling_params)
        
        现在（几乎一样：
        """
        # 模拟 vLLM 原生调用
        pass


if __name__ == "__main__":
    example_integration()