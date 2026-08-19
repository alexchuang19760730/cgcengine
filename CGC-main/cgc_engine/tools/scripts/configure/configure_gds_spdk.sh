#!/bin/bash
# GDS 和 SPDK 配置脚本

echo "============================================================"
echo "GDS (GPUDirect Storage) 和 SPDK 配置"
echo "============================================================"

# 1. 检查当前状态
echo ""
echo "1. 检查当前状态..."
echo "   - CUDA 版本:"
nvcc --version 2>/dev/null || echo "     nvcc 未找到"
echo "   - NVIDIA 驱动:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "     nvidia-smi 未找到"

# 2. 检查 GDS / NFSoRDMA
echo ""
echo "2. 检查 GDS (GPUDirect Storage) / NFSoRDMA..."
if [ -f "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so" ]; then
    echo "   ✅ libcufile.so 存在"
else
    echo "   ❌ libcufile.so 不存在"
fi

# 检查 nvidia-fs
if command -v nvidia-fs &> /dev/null; then
    echo "   - nvidia-fs 状态:"
    nvidia-fs status 2>&1 || true
else
    echo "   ❌ nvidia-fs 未安装"
fi

if [ -f "/etc/cufile.json" ]; then
    echo "   ✅ /etc/cufile.json 存在"
else
    echo "   ❌ /etc/cufile.json 不存在"
fi

echo "   - RDMA 链路:"
rdma link show 2>&1 || true

echo "   - IB 设备:"
ibv_devices 2>&1 || true

echo "   - NFSoRDMA 相关模块:"
lsmod | egrep 'rpcrdma|xprtrdma|svcrdma|mlx5_ib|nvidia_fs|nvidia_peermem|nv_peer_mem' || true

echo "   - 当前 NFS 挂载:"
mount | egrep ' nfs|rdma ' || true

echo "   - NFS 挂载细节:"
nfsstat -m 2>&1 || true

# 3. 尝试配置 GDS
echo ""
echo "3. 配置 GDS..."
if [ -d "/usr/local/cuda" ]; then
    # 创建 cuFile 配置目录
    sudo mkdir -p /etc/cufile.d 2>/dev/null || true
    
    # 创建 GDS 配置文件
    cat << 'EOF' | sudo tee /etc/cufile.d/gds.conf 2>/dev/null || true
# GDS Configuration
fs.gdsEnabled = 1
fs.gdsSyncXfer = 1
fs.gdsAioXfer = 1
EOF
    
    if [ -f "/etc/cufile.d/gds.conf" ]; then
        echo "   ✅ GDS 配置文件已创建: /etc/cufile.d/gds.conf"
        cat /etc/cufile.d/gds.conf
    fi
else
    echo "   ❌ CUDA 未安装在 /usr/local/cuda"
fi

# 4. 检查 SPDK
echo ""
echo "4. 检查 SPDK..."
if command -v spdk_tgt &> /dev/null; then
    echo "   ✅ SPDK 已安装"
    spdk_tgt --version 2>&1 | head -1 || true
else
    echo "   ❌ SPDK 未安装"
    echo ""
    echo "   安装 SPDK 的步骤:"
    echo "   sudo apt-get update"
    echo "   sudo apt-get install -y spdk libspdk-dev"
    echo "   sudo /usr/share/spdk/scripts/setup.sh"
fi

# 5. 检查大页内存 (SPDK 需要)
echo ""
echo "5. 检查大页内存..."
HUGEPAGES=$(cat /proc/meminfo | grep HugePages_Total | awk '{print $2}')
if [ "$HUGEPAGES" -gt 0 ]; then
    echo "   ✅ 大页内存已配置: $HUGEPAGES 页"
else
    echo "   ❌ 大页内存未配置"
    echo "   配置命令: sudo echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"
fi

# 6. 检查存储路径能力
echo ""
echo "6. 检查存储路径能力..."
NVME_COUNT=$(ls -la /dev/nvme* 2>/dev/null | wc -l)
if [ "$NVME_COUNT" -gt 0 ]; then
    echo "   ✅ 找到本地 NVMe 设备:"
    ls -la /dev/nvme* 2>/dev/null || true
else
    echo "   ⚠️ 未找到本地 NVMe 设备"
    echo "   注意: 若目标是 GDS + NFSoRDMA 直写显存，可不依赖本地 NVMe，但必须看到 RDMA-NFS 挂载"
fi

# 7. 检查 IOMMU
echo ""
echo "7. 检查 IOMMU (VFIO 需要)..."
if [ -d "/sys/class/iommu" ]; then
    IOMMU_GROUPS=$(ls /sys/class/iommu/ 2>/dev/null | wc -l)
    if [ "$IOMMU_GROUPS" -gt 0 ]; then
        echo "   ✅ IOMMU 已启用: $IOMMU_GROUPS 个组"
    else
        echo "   ❌ IOMMU 未启用"
        echo "   需要在 BIOS 中启用 VT-d/IOMMU"
    fi
else
    echo "   ❌ IOMMU 不可用"
fi

# 8. 检查 VFIO
echo ""
echo "8. 检查 VFIO..."
if [ -d "/dev/vfio" ]; then
    echo "   ✅ VFIO 设备存在"
    ls -la /dev/vfio/ 2>/dev/null || true
else
    echo "   ❌ VFIO 未配置"
    echo "   加载命令: sudo modprobe vfio vfio_iommu_type1 vfio_pci"
fi

echo ""
echo "============================================================"
echo "配置建议:"
echo "============================================================"
echo ""
echo "GDS 配置:"
echo "  1. 确保 NVIDIA 驱动支持 GDS"
echo "  2. 运行: sudo nvidia-fs enable (如果可用)"
echo "  3. 配置 /etc/cufile.json 绑定 RDMA 网卡与目标挂载点"
echo "  4. 如果走 NFS，必须使用 nfs4.2 + proto=rdma 挂载，不能停留在 TCP-NFS"
echo "  5. 重启系统使配置生效"
echo ""
echo "SPDK 配置:"
echo "  1. 安装: sudo apt-get install -y spdk libspdk-dev"
echo "  2. 配置大页内存: sudo echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"
echo "  3. 运行 SPDK 设置: sudo /usr/share/spdk/scripts/setup.sh"
echo "  4. SPDK 只优化存储服务端 NVMe 吞吐，不等于客户端 NFSoRDMA 直写显存"
echo ""
echo "============================================================"
