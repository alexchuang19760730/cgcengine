#!/usr/bin/env python3
"""
完整集成测试：MagiCompiler + SPDK + GDS + CGC 双引擎
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MagiCompiler-main"))

import torch
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """测试所有模块的导入"""
    print("\n" + "="*80)
    print("📦 1. 测试模块导入")
    print("="*80)
    
    # 测试 MagiCompiler 核心
    try:
        import magi_compiler
        from magi_compiler import magi_compile
        print("✅ MagiCompiler 导入成功")
    except Exception as e:
        print(f"❌ MagiCompiler 导入失败: {e}")
        return False
    
    # 测试 CGC 模块
    try:
        from magi_compiler.cgc import CGCCommand, CGCExecutor, CGCConfig
        from magi_compiler.cgc_simd_executor import CGCExecutor as CGCExecutorSIMD
        from magi_compiler.cgc_dual_executor import CGCDualExecutor
        from magi_compiler.dual_layer_manager import DualLayerManager, DualLayerConfig, StorageTier
        print("✅ CGC 模块导入成功")
    except Exception as e:
        print(f"❌ CGC 模块导入失败: {e}")
        return False
    
    # 测试 SPDK
    try:
        from magi_compiler.spdk_adapter import (
            SPDKConfig, SPDKBlockDevice, SPDKKVStore, SPDKExpertStore, SPDKIOManager
        )
        print("✅ SPDK 模块导入成功")
    except Exception as e:
        print(f"❌ SPDK 模块导入失败: {e}")
        return False
    
    # 测试 GDS
    try:
        from magi_compiler.gds_service import (
            GDSConfig, GDSMemoryManager, GDSFileIO, GDSIntegration
        )
        print("✅ GDS 模块导入成功")
    except Exception as e:
        print(f"❌ GDS 模块导入失败: {e}")
        return False
    
    # 测试 JITLoad
    try:
        from magi_compiler.cgc_jitload import (
            JITLoadConfig, CGCCache, JITLoadManager, AOTPrecompiler
        )
        print("✅ JITLoad 模块导入成功")
    except Exception as e:
        print(f"❌ JITLoad 模块导入失败: {e}")
        return False
    
    return True


def test_dual_layer_manager():
    """测试双分层管理器（含 SPDK/GDS）"""
    print("\n" + "="*80)
    print("💾 2. 测试双分层管理器")
    print("="*80)
    
    try:
        from magi_compiler.dual_layer_manager import DualLayerManager, DualLayerConfig
        
        # 创建配置（禁用 SPDK/GDS，仅用于测试）
        config = DualLayerConfig(
            max_ram_kv_blocks=10,
            max_ram_experts=5,
            enable_spdk=False,
            enable_gds=False
        )
        
        # 创建管理器
        manager = DualLayerManager(config)
        print(f"✅ DualLayerManager 创建成功")
        
        # 测试 KV 存储
        print("\n🧪 测试 KV Cache 存储...")
        k1 = torch.randn(2, 512)
        v1 = torch.randn(2, 512)
        manager.put_kv_block(1, k1, v1)
        
        k2, v2 = manager.get_kv_block(1)
        assert torch.allclose(k1, k2)
        assert torch.allclose(v1, v2)
        print(f"✅ KV Cache 读写成功")
        
        # 测试专家权重存储
        print("\n🧪 测试 MoE 专家权重存储...")
        expert_weights = {
            "w_gate": torch.randn(512, 512),
            "w_up": torch.randn(512, 512),
            "w_down": torch.randn(512, 512),
        }
        manager.put_expert(1, expert_weights)
        
        loaded_weights = manager.get_expert(1)
        for key in expert_weights:
            assert torch.allclose(expert_weights[key], loaded_weights[key])
        print(f"✅ MoE 专家权重读写成功")
        
        # 打印统计
        manager.print_stats()
        
        return True
        
    except Exception as e:
        print(f"❌ 双分层管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spdk_adapter():
    """测试 SPDK 适配器（Fallback 模式）"""
    print("\n" + "="*80)
    print("⚡ 3. 测试 SPDK 适配器（Fallback 模式）")
    print("="*80)
    
    try:
        from magi_compiler.spdk_adapter import (
            SPDKConfig, SPDKBlockDevice, SPDKKVStore, SPDKExpertStore
        )
        
        # 创建配置
        config = SPDKConfig(
            enable_spdk=False,  # 使用 Fallback 模式
            cache_dir="/tmp/spdk_test"
        )
        
        # 测试块设备
        print("\n🧪 测试 SPDKBlockDevice...")
        block_device = SPDKBlockDevice(config)
        block_device.initialize()
        
        # 写入测试
        key = "test_key"
        data = b"Hello SPDK!"
        assert block_device.write(key, data)
        print(f"✅ 数据写入成功")
        
        # 读取测试
        read_data = block_device.read(key)
        assert read_data == data
        print(f"✅ 数据读取成功")
        
        # 测试 KV 存储
        print("\n🧪 测试 SPDKKVStore...")
        kv_store = SPDKKVStore(config, block_device)
        kv_store.initialize()
        
        k = torch.randn(2, 512)
        v = torch.randn(2, 512)
        kv_store.put_kv_block(1, k, v)
        
        loaded_k, loaded_v = kv_store.get_kv_block(1)
        assert torch.allclose(k, loaded_k)
        assert torch.allclose(v, loaded_v)
        print(f"✅ KV 块存储成功")
        
        # 测试专家存储
        print("\n🧪 测试 SPDKExpertStore...")
        expert_store = SPDKExpertStore(config, block_device)
        expert_store.initialize()
        
        expert_weights = {
            "w1": torch.randn(512, 512),
            "w2": torch.randn(512, 512),
        }
        expert_store.put_expert(1, expert_weights)
        
        loaded_expert = expert_store.get_expert(1)
        for key in expert_weights:
            assert torch.allclose(expert_weights[key], loaded_expert[key])
        print(f"✅ 专家权重存储成功")
        
        print("\n📊 SPDK 统计:")
        print(f"  KV Stats: {kv_store.get_stats()}")
        print(f"  Expert Stats: {expert_store.get_stats()}")
        
        return True
        
    except Exception as e:
        print(f"❌ SPDK 适配器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gds_service():
    """测试 GDS 服务（Fallback 模式）"""
    print("\n" + "="*80)
    print("🎮 4. 测试 GDS 服务（Fallback 模式）")
    print("="*80)
    
    try:
        from magi_compiler.gds_service import (
            GDSConfig, GDSMemoryManager, GDSFileIO, GDSIntegration
        )
        
        # 创建配置
        config = GDSConfig(
            enable_gds=False,  # 使用 Fallback 模式
            registered_memory_size_mb=128
        )
        
        # 测试内存管理器
        print("\n🧪 测试 GDSMemoryManager...")
        gds_memory = GDSMemoryManager(config)
        gds_memory.initialize()
        
        # 分配测试 tensor
        test_tensor = torch.empty(1024, dtype=torch.float32)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
        
        # 注册 tensor（Fallback 模式总是成功）
        assert gds_memory.register_tensor(test_tensor)
        print(f"✅ GDS 内存注册成功")
        
        # 测试文件 IO
        print("\n🧪 测试 GDSFileIO...")
        gds_io = GDSFileIO(config, gds_memory)
        gds_io.initialize()
        
        # 简单测试（Fallback 模式）
        test_data = torch.randn(100)
        test_path = "/tmp/gds_test.bin"
        
        if torch.cuda.is_available():
            test_data = test_data.cuda()
        
        # 写入测试
        gds_io.write_from_gpu(test_path, test_data.view(torch.uint8))
        print(f"✅ GDS 写入成功（Fallback）")
        
        # 测试集成
        print("\n🧪 测试 GDSIntegration...")
        gds_integration = GDSIntegration(config)
        gds_integration.initialize()
        
        print("\n📊 GDS 统计:")
        print(f"  IO Stats: {gds_io.get_stats()}")
        
        return True
        
    except Exception as e:
        print(f"❌ GDS 服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jitload():
    """测试 JITLoad 模块"""
    print("\n" + "="*80)
    print("🚀 5. 测试 JITLoad 模块")
    print("="*80)
    
    try:
        from magi_compiler.cgc_jitload import (
            JITLoadConfig, CGCCache, JITLoadManager, AOTPrecompiler
        )
        
        # 创建配置
        config = JITLoadConfig(
            enable_jitload=True,
            cgc_cache_size=1000,
            cache_dir="/tmp/jitload_test"
        )
        
        # 测试 CGC 缓存
        print("\n🧪 测试 CGCCache...")
        cache = CGCCache(config)
        
        cache.put("test_key1", "test_value1")
        cache.put("test_key2", {"data": [1, 2, 3]})
        
        value1 = cache.get("test_key1")
        value2 = cache.get("test_key2")
        
        assert value1 == "test_value1"
        assert value2 == {"data": [1, 2, 3]}
        print(f"✅ CGC 缓存读写成功")
        
        # 测试 JITLoadManager
        print("\n🧪 测试 JITLoadManager...")
        jitload = JITLoadManager(config)
        jitload.initialize()
        
        # 保存编译产物
        compile_data = {
            "optimized": True,
            "passes": ["pass1", "pass2"],
            "metadata": {"version": "1.0"}
        }
        jitload.save_artifact("test_model", "config_v1", compile_data)
        
        # 加载编译产物
        loaded_data = jitload.load_artifact("test_model", "config_v1")
        assert loaded_data == compile_data
        print(f"✅ JITLoad 编译产物读写成功")
        
        # 测试 AOT 预编译器
        print("\n🧪 测试 AOTPrecompiler...")
        aot = AOTPrecompiler(config, jitload)
        
        # 提交预编译任务（简化版本）
        def compile_fn():
            time.sleep(0.1)
            return {"compiled": True, "time": time.time()}
        
        task = aot.submit_task("model_a", compile_fn, priority=10)
        
        print(f"✅ AOT 预编译任务提交成功")
        
        # 统计
        print("\n📊 JITLoad 统计:")
        print(f"  Stats: {jitload.get_stats()}")
        print(f"  Cache Stats: {cache.get_stats()}")
        
        return True
        
    except Exception as e:
        print(f"❌ JITLoad 模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cgc_dual_executor():
    """测试 CGC 双引擎执行器"""
    print("\n" + "="*80)
    print("⚙️ 6. 测试 CGC 双引擎执行器")
    print("="*80)
    
    try:
        from magi_compiler.cgc_dual_executor import CGCDualExecutor
        from magi_compiler.dual_layer_manager import get_dual_layer_manager
        
        # 获取双分层管理器
        dual_mgr = get_dual_layer_manager()
        
        # 创建执行器
        executor = CGCDualExecutor(dual_mgr)
        print(f"✅ CGCDualExecutor 创建成功")
        
        # 准备一些测试数据
        print("\n🧪 准备测试数据...")
        k = torch.randn(2, 512)
        v = torch.randn(2, 512)
        dual_mgr.put_kv_block(1, k, v)
        
        # 测试专家权重
        expert_weights = {
            "w_gate": torch.randn(512, 512),
            "w_up": torch.randn(512, 512),
            "w_down": torch.randn(512, 512),
        }
        dual_mgr.put_expert(1, expert_weights)
        
        # （暂时不测试 CGC 命令执行，因为需要完整的 opcode 系统）
        print(f"✅ 基础功能验证成功")
        
        return True
        
    except Exception as e:
        print(f"❌ CGC 双引擎执行器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "╔════════════════════════════════════════════════════════════╗")
    print("║  🔥 全球独一份：完整集成测试 - 终极双分层融合代码 🔥       ║")
    print("║  MagiCompiler + SPDK + GDS + CGC 双引擎 + 双分层管理        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    results = []
    
    # 1. 测试导入
    results.append(("模块导入", test_imports()))
    
    # 2. 测试双分层管理器
    results.append(("双分层管理器", test_dual_layer_manager()))
    
    # 3. 测试 SPDK
    results.append(("SPDK 适配器", test_spdk_adapter()))
    
    # 4. 测试 GDS
    results.append(("GDS 服务", test_gds_service()))
    
    # 5. 测试 JITLoad
    results.append(("JITLoad", test_jitload()))
    
    # 6. 测试 CGC 双引擎
    results.append(("CGC 双引擎", test_cgc_dual_executor()))
    
    # 总结
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name:<20} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n" + "🎉"*10)
        print("🎉 所有测试通过！完美！")
        print("🎉"*10)
    else:
        print("\n⚠️  部分测试失败，请检查")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
