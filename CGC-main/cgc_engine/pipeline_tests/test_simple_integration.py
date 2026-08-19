#!/usr/bin/env python3
"""
简单的 GDS/SPDK 集成测试
"""

import sys
import os
import logging

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gds():
    """测试 GDS"""
    logger.info("=== 测试 GDS ===")
    try:
        from gds_service.cufile_wrapper import CUFILE_AVAILABLE, is_gds_available
        from gds_service.gds_manager import GDSManager
        
        logger.info(f"GDS 可用: {CUFILE_AVAILABLE}")
        logger.info(f"is_gds_available(): {is_gds_available()}")
        
        gds = GDSManager()
        logger.info(f"GDS Manager info: {gds.info()}")
        
        # 测试权重加载
        try:
            weight = gds.load_weight_from_pd("/test/model.weight", [1024, 1024])
            logger.info(f"✅ 权重加载成功: shape={weight.shape}, device={weight.device}")
        except Exception as e:
            logger.warning(f"⚠️ 权重加载测试失败（预期）: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ GDS 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spdk():
    """测试 SPDK"""
    logger.info("\n=== 测试 SPDK ===")
    try:
        from spdk_adapter.spdk_io_manager import SPDKIOManager, SPDK_AVAILABLE
        from spdk_adapter.spdk_config import SPDKConfig
        
        logger.info(f"SPDK liburing 可用: {SPDK_AVAILABLE}")
        
        # 创建 SPDK IO Manager
        config = SPDKConfig(kv_store_path="/tmp/test_spdk", io_queues=4)
        io_manager = SPDKIOManager(config)
        io_manager.start()
        
        # 测试写入
        task1 = io_manager.submit_write("test_key1", b"Hello SPDK Integration!")
        result1 = task1.wait()
        logger.info(f"✅ 写入结果: {result1}")
        
        # 测试读取
        task2 = io_manager.submit_read("test_key1")
        result2 = task2.wait()
        logger.info(f"✅ 读取结果: {result2}")
        
        # 统计信息
        stats = io_manager.get_stats()
        logger.info(f"SPDK 统计: {stats}")
        
        io_manager.stop()
        
        return True
    except Exception as e:
        logger.error(f"❌ SPDK 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("=" * 60)
    logger.info("GDS/SPDK 简单集成测试")
    logger.info("=" * 60)
    
    results = []
    
    # 测试 GDS
    results.append(test_gds())
    
    # 测试 SPDK
    results.append(test_spdk())
    
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
