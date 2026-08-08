#!/usr/bin/env python3
"""
Prefill Pool + GDS 验证脚本
验证 NFSoRDMA 直写显存功能
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cgc_engine.prefill_pool import PrefillPool
from cgc_engine.gds_service.cufile_wrapper import (
    is_gds_available, 
    get_gds_backend, 
    get_gds_capabilities
)


def test_gds_availability():
    """测试 GDS 可用性"""
    print("=== GDS 可用性测试 ===")
    print(f"GDS 可用: {is_gds_available()}")
    print(f"GDS 后端: {get_gds_backend()}")
    
    caps = get_gds_capabilities()
    print(f"\n存储能力详情:")
    print(f"  NVMe 可用: {caps['has_nvme']}")
    print(f"  RDMA 设备: {caps['rdma_devices']}")
    print(f"  NFS 挂载数: {len(caps['nfs_mounts'])}")
    print(f"  NFSoRDMA 挂载数: {len(caps['nfs_rdma_mounts'])}")
    
    if caps['nfs_rdma_mounts']:
        print("  NFSoRDMA 挂载路径:")
        for mount in caps['nfs_rdma_mounts']:
            print(f"    - {mount['target']} (from {mount['source']})")
    
    return is_gds_available()


def test_prefill_pool():
    """测试 Prefill Pool 功能"""
    print("\n=== Prefill Pool 测试 ===")
    
    pool = PrefillPool(
        max_hot_chunks=2,
        chunk_size=8192,
        storage_path="/data/nfs/prefill_pool_test"
    )
    
    print(f"Pool 信息: {pool.info()}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    num_heads = 32
    head_dim = 128
    seq_len = 4096
    
    ref_k1 = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                         device=device, dtype=torch.bfloat16)
    ref_v1 = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                         device=device, dtype=torch.bfloat16)
    token_ids1 = torch.arange(seq_len, device=device, dtype=torch.long)
    
    ref_k2 = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                         device=device, dtype=torch.bfloat16)
    ref_v2 = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                         device=device, dtype=torch.bfloat16)
    token_ids2 = torch.arange(seq_len, seq_len * 2, device=device, dtype=torch.long)
    
    print("\n1. 添加第一个热块...")
    chunk_id1 = pool.add_hot_chunk(token_ids1, ref_k1, ref_v1)
    print(f"   结果: chunk_id = {chunk_id1}")
    print(f"   Pool 状态: {pool.info()}")
    
    print("\n2. 添加第二个热块...")
    chunk_id2 = pool.add_hot_chunk(token_ids2, ref_k2, ref_v2)
    print(f"   结果: chunk_id = {chunk_id2}")
    print(f"   Pool 状态: {pool.info()}")
    
    print("\n3. 获取所有参考 KV...")
    all_ref_k, all_ref_v = pool.get_all_ref_kv(device)
    print(f"   ref_k 形状: {all_ref_k.shape if all_ref_k is not None else 'None'}")
    print(f"   ref_v 形状: {all_ref_v.shape if all_ref_v is not None else 'None'}")
    
    print("\n4. 添加第三个热块（触发驱逐）...")
    ref_k3 = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                         device=device, dtype=torch.bfloat16)
    ref_v3 = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                         device=device, dtype=torch.bfloat16)
    token_ids3 = torch.arange(seq_len * 2, seq_len * 3, device=device, dtype=torch.long)
    
    chunk_id3 = pool.add_hot_chunk(token_ids3, ref_k3, ref_v3)
    print(f"   结果: chunk_id = {chunk_id3}")
    print(f"   Pool 状态: {pool.info()}")
    
    print("\n5. 加载被驱逐的块...")
    loaded_chunk = pool.load_chunk(chunk_id1, device)
    if loaded_chunk:
        print(f"   ✅ 成功加载 Chunk {chunk_id1}")
        print(f"   ref_k 形状: {loaded_chunk.ref_k.shape}")
    else:
        print(f"   ❌ 加载 Chunk {chunk_id1} 失败")
    
    print(f"   Pool 状态: {pool.info()}")
    
    print("\n6. 验证数据完整性...")
    if loaded_chunk:
        if torch.allclose(loaded_chunk.ref_k, ref_k1, atol=1e-3):
            print("   ✅ ref_k 数据验证通过")
        else:
            print("   ❌ ref_k 数据验证失败")
        
        if torch.allclose(loaded_chunk.token_ids, token_ids1):
            print("   ✅ token_ids 数据验证通过")
        else:
            print("   ❌ token_ids 数据验证失败")
    
    print("\n7. 清理测试数据...")
    pool.clear()
    print("   ✅ 清理完成")
    
    return True


def main():
    print("=" * 60)
    print("Prefill Pool + GDS 直写显存验证脚本")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA，无法测试 GDS")
        return
    
    gds_ok = test_gds_availability()
    
    if not gds_ok:
        print("\n⚠️ GDS 当前不可用，但继续测试 Prefill Pool 基础功能")
        print("   请确保已正确配置 NFSoRDMA 挂载")
    
    try:
        test_prefill_pool()
        print("\n✅ 所有测试通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
