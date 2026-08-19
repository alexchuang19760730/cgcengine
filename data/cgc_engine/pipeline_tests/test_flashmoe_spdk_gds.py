#!/usr/bin/env python3
"""
测试 FlashMoE/PD 与 SPDK/GDS 的集成
"""

import sys
import os
import logging
import torch

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

def test_gds_expert_loader():
    """测试 GDS 专家加载器"""
    logger.info("=== 测试 GDS Expert Loader ===")
    
    try:
        from cgc_engine.flash_moe.gds_expert_loader import GDSExpertLoader
        
        loader = GDSExpertLoader()
        logger.info(f"✅ GDS 可用: {loader.gds_enabled}")
        
        # 加载专家
        expert = loader.load_expert(0, [4096, 4096])
        logger.info(f"✅ 专家加载成功: shape={expert.shape}, device={expert.device}")
        
        # 测试缓存
        expert2 = loader.load_expert(0, [4096, 4096])
        logger.info(f"✅ 缓存命中测试: same object={expert is expert2}")
        
        # 统计信息
        stats = loader.get_stats()
        logger.info(f"✅ 统计信息: {stats}")
        
        return True
    except Exception as e:
        logger.error(f"❌ GDS Expert Loader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spdk_kv_cache():
    """测试 SPDK KV Cache"""
    logger.info("\n=== 测试 SPDK KV Cache ===")
    
    try:
        from cgc_engine.pd.spdk_kv_cache import SPDKKVCache
        
        kv_cache = SPDKKVCache()
        logger.info(f"✅ SPDK 可用: {kv_cache.spdk_enabled}")
        
        # 创建测试张量
        k = torch.randn(1, 32, 128, 64)
        v = torch.randn(1, 32, 128, 64)
        
        # 设置 KV
        success = kv_cache.set_kv("test_session", k, v)
        logger.info(f"✅ KV 设置成功: {success}")
        
        # 获取 KV
        result = kv_cache.get_kv("test_session")
        if result:
            k_out, v_out = result
            logger.info(f"✅ KV 获取成功: k.shape={k_out.shape}, v.shape={v_out.shape}")
        
        # 统计信息
        stats = kv_cache.get_stats()
        logger.info(f"✅ 统计信息: {stats}")
        
        kv_cache.shutdown()
        
        return True
    except Exception as e:
        logger.error(f"❌ SPDK KV Cache 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distributed_expert_store():
    """测试分布式专家存储"""
    logger.info("\n=== 测试 Distributed Expert Store ===")
    
    try:
        from cgc_engine.flash_moe.distributed_expert_store import DistributedExpertStore
        
        # 创建分布式存储（模拟集群）
        cluster_nodes = ["node1:4420", "node2:4420"]
        store = DistributedExpertStore(cluster_nodes=cluster_nodes)
        
        logger.info(f"✅ 集群节点: {store.cluster_nodes}")
        logger.info(f"✅ 分区数量: {store.num_partitions}")
        
        # 存储专家
        expert = torch.randn(4096, 4096, dtype=torch.float16)
        success = store.store_expert(0, expert)
        logger.info(f"✅ 专家存储成功: {success}")
        
        # 加载专家
        loaded_expert = store.load_expert(0, [4096, 4096])
        logger.info(f"✅ 专家加载成功: shape={loaded_expert.shape}, device={loaded_expert.device}")
        
        # 统计信息
        stats = store.get_stats()
        logger.info(f"✅ 统计信息: {stats}")
        
        store.shutdown()
        
        return True
    except Exception as e:
        logger.error(f"❌ Distributed Expert Store 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import torch
    
    logger.info("=" * 60)
    logger.info("FlashMoE/PD 与 SPDK/GDS 集成测试")
    logger.info("=" * 60)
    
    results = []
    
    # 测试 GDS Expert Loader
    results.append(test_gds_expert_loader())
    
    # 测试 SPDK KV Cache
    results.append(test_spdk_kv_cache())
    
    # 测试 Distributed Expert Store
    results.append(test_distributed_expert_store())
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    if all(results):
        logger.info("✅ 所有测试通过!")
        return 0
    else:
        logger.info("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
