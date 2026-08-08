#!/bin/bash
# run_full_integration.sh - 完整集成测试脚本
# 
# 流程:
# 1. 配置 NFSoRDMA 挂载
# 2. 验证 GDS 可用性
# 3. 运行 R-SWA + Prefill Pool 集成测试

set -e

echo "========================================"
echo "R-SWA + Prefill Pool 完整集成测试"
echo "========================================"

# 步骤 1: 配置 NFSoRDMA
echo ""
echo "=== 步骤 1: 配置 NFSoRDMA 挂载 ==="
if [ ! -d "/data/nfs" ]; then
    mkdir -p /data/nfs
fi

# 检查是否已有 NFSoRDMA 挂载
if ! mount | grep -q "/data/nfs.*proto=rdma"; then
    echo "正在配置 NFSoRDMA..."
    # 卸载现有挂载（如果存在）
    umount /data/nfs 2>/dev/null || true
    
    # 使用 RDMA 协议挂载
    NFS_SERVER="39.106.118.206"
    NFS_EXPORT="/export/cgc_data"
    
    echo "挂载 NFS: $NFS_SERVER:$NFS_EXPORT -> /data/nfs"
    mount -t nfs -o proto=rdma,port=20048 "$NFS_SERVER:$NFS_EXPORT" /data/nfs
    
    # 验证
    if mount | grep -q "/data/nfs.*proto=rdma"; then
        echo "✅ NFSoRDMA 挂载成功"
    else
        echo "❌ NFSoRDMA 挂载失败"
        exit 1
    fi
else
    echo "✅ NFSoRDMA 已配置"
fi

# 步骤 2: 验证 GDS
echo ""
echo "=== 步骤 2: 验证 GDS 可用性 ==="
python3 << 'PYTHON'
import sys
sys.path.insert(0, '/root/flashkv0516/ComputeGraphCompiler-main')

from cgc_engine.gds_service.cufile_wrapper import (
    is_gds_available, 
    get_gds_backend, 
    get_gds_capabilities,
    CUFILE_AVAILABLE
)

print(f"GDS 可用: {is_gds_available()}")
print(f"GDS 后端: {get_gds_backend()}")
print(f"CUFILE_AVAILABLE: {CUFILE_AVAILABLE}")

caps = get_gds_capabilities()
print(f"\n存储能力:")
print(f"  NVMe: {caps['has_nvme']}")
print(f"  RDMA 设备: {caps['rdma_devices']}")
print(f"  NFS 挂载数: {len(caps['nfs_mounts'])}")
print(f"  NFSoRDMA 挂载数: {len(caps['nfs_rdma_mounts'])}")

if caps['nfs_rdma_mounts']:
    for m in caps['nfs_rdma_mounts']:
        print(f"    - {m['target']}")

if not is_gds_available():
    print("\n⚠️ GDS 不可用，但继续测试...")
PYTHON

# 步骤 3: 创建测试目录
echo ""
echo "=== 步骤 3: 创建测试目录 ==="
mkdir -p /data/nfs/prefill_pool_test

# 步骤 4: 运行 R-SWA 集成测试
echo ""
echo "=== 步骤 4: 运行 R-SWA + Prefill Pool 集成测试 ==="
python3 << 'PYTHON'
import sys
sys.path.insert(0, '/root/flashkv0516/ComputeGraphCompiler-main')

import torch

# 检查 CUDA
if not torch.cuda.is_available():
    print("❌ 未检测到 CUDA")
    sys.exit(1)

print("✅ CUDA 可用")
print(f"GPU: {torch.cuda.get_device_name(0)}")

from cgc_engine.rswa_integration import RSWAPrefillPoolEngine

# 初始化引擎
print("\n初始化 R-SWA 引擎...")
engine = RSWAPrefillPoolEngine(
    dim=4096,
    num_heads=32,
    window_size=128,
    max_hot_chunks=2,
    chunk_size=4096,
)

print(f"\n引擎信息: {engine.info()}")

# 添加参考文本
print("\n添加参考文本到 Prefill Pool...")
reference_texts = [
    "法律文档: 中华人民共和国刑法 第三百零二条 盗窃、侮辱、故意毁坏尸体、尸骨、骨灰的，处三年以下有期徒刑、拘役或者管制。",
    "技术规范: GPU Direct Storage (GDS) 是 NVIDIA 提供的一项技术，允许 GPU 直接从存储设备读取数据，无需经过 CPU 内存中转。",
    "历史知识: 唐朝（618年—907年），是中国历史上继隋朝之后的大一统中原王朝，共历二十一帝，享国二百八十九年。",
]

chunk_ids = engine.prefill_reference(reference_texts)
print(f"已添加 {len(chunk_ids)} 个参考块")

# 执行推理
print("\n执行推理...")
query = "什么是 GDS 技术？"
response = engine.infer(query, max_tokens=50)
print(f"查询: {query}")
print(f"响应: {response[:100]}...")

# 检查 Pool 状态
print(f"\nPool 状态: {engine.attention.get_pool_info()}")

# 清理
engine.attention.clear_pool()
print("\n✅ 集成测试完成!")
PYTHON

echo ""
echo "========================================"
echo "集成测试全部完成!"
echo "========================================"
