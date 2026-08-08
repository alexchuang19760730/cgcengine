#!/usr/bin/env python3
"""
vLLM Custom Backend - 完整测试脚本

测试场景：
1. 注册自定义后端
2. Hook vLLM 执行流程
3. 验证 compute 调用是否被正确路由
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_backend import (
    VLLMBackend,
    BackendManager,
    register_backend,
    get_backend,
    VLLMEngineHook,
    MagiCompilerBackend,
    NativeVLLMBackend,
)

def test_backend_registration():
    """测试后端注册机制"""
    print("\n=== 测试 1: 后端注册 ===")
    
    # 清除之前的注册
    BackendManager._registered_backends = {}
    BackendManager._current_backend = None
    
    # 注册 MagiCompiler 后端
    backend = MagiCompilerBackend()
    backend.init()
    register_backend(backend)
    
    # 验证注册
    backends = BackendManager.list_backends()
    print(f"已注册后端: {backends}")
    assert "magi-compiler" in backends, "MagiCompiler 后端未注册成功"
    
    # 获取当前后端
    current = get_backend()
    assert current is not None, "当前后端为 None"
    assert current.get_name() == "magi-compiler", "当前后端名称不正确"
    
    print("✓ 后端注册测试通过")

def test_backend_capabilities():
    """测试后端能力查询"""
    print("\n=== 测试 2: 后端能力 ===")
    
    backend = get_backend()
    caps = backend.get_capabilities()
    
    print(f"后端能力: {caps}")
    
    # 验证关键能力
    assert caps.get("prefill") == True, "缺少 prefill 能力"
    assert caps.get("decode") == True, "缺少 decode 能力"
    assert caps.get("kda") == True, "缺少 KDA 能力"
    
    print("✓ 后端能力测试通过")

def test_vllm_hook():
    """测试 vLLM Hook 机制"""
    print("\n=== 测试 3: vLLM Hook ===")
    
    # 测试 Hook
    VLLMEngineHook.hook()
    
    # 验证 Hook 是否成功
    from vllm.v1.executor.abstract import Executor
    assert Executor.execute_model.__name__ == "hooked_execute_model", "Hook 未成功"
    
    print("✓ vLLM Hook 测试通过")

def test_backend_switching():
    """测试后端切换"""
    print("\n=== 测试 4: 后端切换 ===")
    
    # 注册原生后端
    native_backend = NativeVLLMBackend()
    register_backend(native_backend)
    
    # 切换到原生后端
    success = BackendManager.set_backend("native-vllm")
    assert success, "后端切换失败"
    
    current = get_backend()
    assert current.get_name() == "native-vllm", "后端切换不正确"
    
    # 切换回 MagiCompiler
    success = BackendManager.set_backend("magi-compiler")
    assert success, "切换回 MagiCompiler 失败"
    
    current = get_backend()
    assert current.get_name() == "magi-compiler", "未切换回 MagiCompiler"
    
    print("✓ 后端切换测试通过")

def test_compute_routing():
    """测试 compute 路由"""
    print("\n=== 测试 5: Compute 路由 ===")
    
    # 创建一个模拟的 scheduler_output
    class MockSchedulerOutput:
        batch_size = 4
        seq_len = 128
    
    # 获取后端并调用 compute
    backend = get_backend()
    result = backend.compute(MockSchedulerOutput(), non_block=False)
    
    # 检查统计信息
    stats = backend.get_stats()
    print(f"Compute 调用次数: {stats['compute_calls']}")
    print(f"处理的 token 总数: {stats['total_tokens']}")
    
    assert stats['compute_calls'] == 1, "Compute 调用次数不正确"
    assert stats['total_tokens'] == 4 * 128, "Token 计数不正确"
    
    print("✓ Compute 路由测试通过")

def test_unhook():
    """测试取消 Hook"""
    print("\n=== 测试 6: 取消 Hook ===")
    
    # 保存原始方法
    from vllm.v1.executor.abstract import Executor
    original_method = Executor.execute_model
    
    # 取消 Hook
    VLLMEngineHook.unhook()
    
    # 验证是否恢复
    assert Executor.execute_model.__name__ != "hooked_execute_model", "Hook 未取消"
    
    print("✓ 取消 Hook 测试通过")

def main():
    """运行所有测试"""
    print("=" * 70)
    print("vLLM Custom Backend 完整测试")
    print("=" * 70)
    
    try:
        test_backend_registration()
        test_backend_capabilities()
        test_vllm_hook()
        test_backend_switching()
        test_compute_routing()
        test_unhook()
        
        print("\n" + "=" * 70)
        print("✓ 所有测试通过！")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()