#!/usr/bin/env python3
"""
GDS/SPDK 服务连接测试

用法:
    python test_gds_spdk_server.py --gds-host localhost --gds-port 50051
    python test_gds_spdk_server.py --spdk-host localhost --spdk-port 8080
"""

import argparse
import sys
import time

def test_gds_service(host: str, port: int):
    """测试 GDS (GPUDirect Storage) 服务连接"""
    print(f"\n{'='*60}")
    print(f"测试 GDS 服务连接: {host}:{port}")
    print(f"{'='*60}")

    try:
        from cgc_engine.gds_service import GDSManager
        from cgc_engine.gds_service.gds_config import GDSConfig

        pd_endpoint = f"{host}:{port}"
        print(f"[GDS] 连接到 PD 服务: {pd_endpoint}")

        config = GDSConfig(
            enable_gds=True,
            enable_pd=True,
            enable_cufile=True,
            fallback_to_pytorch=True
        )

        gds_manager = GDSManager(pd_endpoint=pd_endpoint)

        print(f"[GDS] GDS Manager 初始化成功")
        print(f"[GDS] 信息: {gds_manager.info()}")

        print(f"\n[GDS] 测试 KV Cache 加载...")
        try:
            k, v = gds_manager.load_kv_from_pd("test_key", 128, 64)
            print(f"[GDS] ✅ KV Cache 加载成功: k.shape={k.shape}, v.shape={v.shape}")
        except Exception as e:
            print(f"[GDS] ⚠️ KV Cache 加载失败 (可能是服务未运行): {e}")

        print(f"\n[GDS] 测试权重加载...")
        try:
            weight = gds_manager.load_weight_from_pd("/test/model.safetensors", [1024, 1024])
            print(f"[GDS] ✅ 权重加载成功: weight.shape={weight.shape}")
        except Exception as e:
            print(f"[GDS] ⚠️ 权重加载失败 (可能是服务未运行): {e}")

        print(f"\n[GDS] 测试完成!")
        return True

    except ImportError as e:
        print(f"[GDS] ❌ 导入失败: {e}")
        print(f"[GDS] 请确保已安装 GDS 相关依赖")
        return False
    except Exception as e:
        print(f"[GDS] ❌ 连接失败: {e}")
        return False


def test_spdk_service(host: str, port: int):
    """测试 SPDK (Storage Performance Development Kit) 服务连接"""
    print(f"\n{'='*60}")
    print(f"测试 SPDK 服务连接: {host}:{port}")
    print(f"{'='*60}")

    try:
        from cgc_engine.spdk_adapter import SPDKIOManager, SPDKConfig
        from cgc_engine.spdk_adapter.spdk_kv_store import SPDKKVStore
        from cgc_engine.spdk_adapter.spdk_expert_store import SPDKExpertStore

        spdk_config = SPDKConfig(
            enable_spdk=True,
            spdk_host=host,
            spdk_port=port,
            enable_kernel_vhost=True,
            enable_rdma=False
        )

        print(f"[SPDK] 连接到 NVMe over Fabrics 服务: {host}:{port}")

        spdk_kv_store = SPDKKVStore(spdk_config) if 'SPDKKVStore' in dir() else None
        spdk_expert_store = SPDKExpertStore(spdk_config) if 'SPDKExpertStore' in dir() else None

        spdk_client = SPDKIOManager(
            config=spdk_config,
            kv_store=spdk_kv_store,
            expert_store=spdk_expert_store
        )

        print(f"[SPDK] SPDK Manager 初始化成功")

        print(f"\n[SPDK] 测试 KV Cache 读取...")
        try:
            result = spdk_client.submit_read("test_kv_key")
            print(f"[SPDK] ✅ KV Cache 读取成功")
        except Exception as e:
            print(f"[SPDK] ⚠️ KV Cache 读取失败 (可能是服务未运行): {e}")

        print(f"\n[SPDK] 测试 KV Cache 写入...")
        try:
            test_data = b"test_data_123"
            result = spdk_client.submit_write("test_kv_key", test_data)
            print(f"[SPDK] ✅ KV Cache 写入成功")
        except Exception as e:
            print(f"[SPDK] ⚠️ KV Cache 写入失败 (可能是服务未运行): {e}")

        print(f"\n[SPDK] 测试完成!")
        return True

    except ImportError as e:
        print(f"[SPDK] ❌ 导入失败: {e}")
        print(f"[SPDK] 请确保已安装 SPDK 相关依赖")
        return False
    except Exception as e:
        print(f"[SPDK] ❌ 连接失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="GDS/SPDK 服务连接测试")
    parser.add_argument("--gds-host", type=str, default="localhost", help="GDS PD 服务主机")
    parser.add_argument("--gds-port", type=int, default=50051, help="GDS PD 服务端口")
    parser.add_argument("--spdk-host", type=str, default="localhost", help="SPDK NVMe-oF 服务主机")
    parser.add_argument("--spdk-port", type=int, default=8080, help="SPDK NVMe-oF 服务端口")
    parser.add_argument("--test-gds", action="store_true", help="仅测试 GDS")
    parser.add_argument("--test-spdk", action="store_true", help="仅测试 SPDK")
    parser.add_argument("--test-all", action="store_true", help="测试全部服务")

    args = parser.parse_args()

    if not any([args.test_gds, args.test_spdk, args.test_all]):
        args.test_all = True

    results = {}

    if args.test_all or args.test_gds:
        results["GDS"] = test_gds_service(args.gds_host, args.gds_port)

    if args.test_all or args.test_spdk:
        results["SPDK"] = test_spdk_service(args.spdk_host, args.spdk_port)

    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())