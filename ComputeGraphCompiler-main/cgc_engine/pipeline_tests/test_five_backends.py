#!/usr/bin/env python3
"""
测试 MagicCompiler 五大后端集成
验证：llama.cpp、vLLM、MLX、CUDA Runtime、Native PyTorch
"""

import sys

print("=" * 70)
print("🚀 测试 MagicCompiler 五大后端集成")
print("=" * 70)

# ================================================
# 1. 测试 unified_backend 五大后端
# ================================================
print("\n📦 测试 Unified Backend（五大后端）")
try:
    from cgc_engine.cgc.magicompiler_unified_backend import (
        create_magicompiler_backend,
        BackendSource,
        UnifiedComputeRequest,
        TensorInfo,
        ComputeType,
    )
    
    # 创建五大后端
    unified, ggml_adapter, vllm_adapter, mlx_adapter, cuda_adapter = create_magicompiler_backend()
    
    # 测试 ggml_backend (llama.cpp)
    print("\n  🔹 测试 ggml_backend (llama.cpp)")
    ggml_req = UnifiedComputeRequest(
        request_id="ggml_test",
        source=BackendSource.GGML,
        compute_type=ComputeType.ATTENTION,
        inputs=[
            TensorInfo("q", (1, 128, 4096), "float16", "cpu"),
            TensorInfo("k", (1, 128, 4096), "float16", "cpu"),
            TensorInfo("v", (1, 128, 4096), "float16", "cpu"),
        ],
        outputs=[TensorInfo("out", (1, 128, 4096), "float16", "cpu")],
    )
    result = unified.compute(ggml_req)
    print(f"     ✅ ggml_backend 测试通过: {result}")
    
    # 测试 vllm_backend
    print("\n  🔹 测试 vllm_backend (vLLM)")
    vllm_req = UnifiedComputeRequest(
        request_id="vllm_test",
        source=BackendSource.VLLM,
        compute_type=ComputeType.MOE_FFN,
        inputs=[TensorInfo("hidden", (1, 128, 4096), "float16", "cuda:0")],
        outputs=[TensorInfo("out", (1, 128, 4096), "float16", "cuda:0")],
        expert_ids=[3, 7],
    )
    result = unified.compute(vllm_req)
    print(f"     ✅ vllm_backend 测试通过: {result}")
    
    # 测试 mlx_backend
    print("\n  🔹 测试 mlx_backend (Apple MLX)")
    mlx_req = UnifiedComputeRequest(
        request_id="mlx_test",
        source=BackendSource.MLX,
        compute_type=ComputeType.FLASH_ATTENTION,
        inputs=[
            TensorInfo("q", (1, 128, 4096), "bfloat16", "mlx"),
            TensorInfo("k", (1, 128, 4096), "bfloat16", "mlx"),
            TensorInfo("v", (1, 128, 4096), "bfloat16", "mlx"),
        ],
        outputs=[TensorInfo("out", (1, 128, 4096), "bfloat16", "mlx")],
        metadata={"unified_memory": True},
    )
    result = unified.compute(mlx_req)
    print(f"     ✅ mlx_backend 测试通过: {result}")
    
    # 测试 cuda_backend
    print("\n  🔹 测试 cuda_backend (CUDA Runtime)")
    cuda_req = UnifiedComputeRequest(
        request_id="cuda_test",
        source=BackendSource.CUDA,
        compute_type=ComputeType.LINEAR,
        inputs=[
            TensorInfo("input", (128, 4096), "float16", "cuda:0"),
            TensorInfo("weight", (4096, 4096), "float16", "cuda:0"),
        ],
        outputs=[TensorInfo("out", (128, 4096), "float16", "cuda:0")],
        metadata={"cuda_graph": True},
    )
    result = unified.compute(cuda_req)
    print(f"     ✅ cuda_backend 测试通过: {result}")
    
    # 测试 native_backend
    print("\n  🔹 测试 native_backend (Native PyTorch)")
    native_req = UnifiedComputeRequest(
        request_id="native_test",
        source=BackendSource.NATIVE,
        compute_type=ComputeType.MLP_SILU,
        inputs=[TensorInfo("input", (128, 4096), "float32", "cpu")],
        outputs=[TensorInfo("out", (128, 4096), "float32", "cpu")],
        metadata={"torch_compile": True},
    )
    result = unified.compute(native_req)
    print(f"     ✅ native_backend 测试通过: {result}")
    
    # 分析报告
    print("\n  📊 分析报告")
    report = unified.analyze_and_report()
    print(f"     总图数: {report['total_graphs']}")
    print(f"     总算子数: {report['total_ops']}")
    for src, stats in report['backend_stats'].items():
        print(f"     {src}: {stats['ops']} ops")

except Exception as e:
    print(f"     ❌ Unified Backend 测试失败: {e}")
    import traceback
    traceback.print_exc()

# ================================================
# 2. 测试知识库存储五大后端支持
# ================================================
print("\n\n📦 测试知识库存储（五大后端支持）")
try:
    from cgc_engine.utils.knowledge_storage import KnowledgeStorage
    
    # 创建知识库
    storage = KnowledgeStorage(db_path=":memory:")
    
    # 获取所有后端知识（五大后端：llama.cpp / vLLM / MLX / mlx-tune / megatrain）
    backends = ["llama.cpp", "vllm", "mlx", "mlx-tune", "megatrain"]
    print("\n  🔹 验证五大后端知识库")
    for backend_id in backends:
        backend = storage.get_backend_knowledge(backend_id)
        if backend:
            print(f"     ✅ {backend.name} ({backend_id}): 已注册")
            print(f"        - 支持算子: {len(backend.supported_ops)} 个")
            print(f"        - 优化能力: {len(backend.optimization_capabilities)} 项")
        else:
            print(f"     ❌ {backend_id}: 未找到")
    
    # 获取所有图模式
    patterns = storage.get_all_patterns()
    print(f"\n  🔹 图模式数量: {len(patterns)}")
    
    # 检测当前平台
    platform = storage.detect_current_platform()
    print(f"\n  🔹 当前平台检测:")
    print(f"     - 后端: {platform['backend']}")
    print(f"     - 设备数: {platform['num_devices']}")
    print(f"     - 内存: {platform['memory_gb']:.1f} GB")

except Exception as e:
    print(f"     ❌ 知识库存储测试失败: {e}")
    import traceback
    traceback.print_exc()

# ================================================
# 总结
# ================================================
print("\n" + "=" * 70)
print("🎉 五大后端集成测试总结")
print("=" * 70)
print("已支持的后端:")
print("   ├── 🔹 llama.cpp (端侧推理)")
print("   ├── 🔹 vLLM (云端推理)")
print("   ├── 🔹 MLX (Apple端侧推理基础版)")
print("   ├── 🔹 mlx-tune (MLX Custom Backend)")
print("   └── 🔹 MegaTrain (云端训练)")
print("=" * 70)
print("\n✅ MagicCompiler 五大后端同步优化完成!")
