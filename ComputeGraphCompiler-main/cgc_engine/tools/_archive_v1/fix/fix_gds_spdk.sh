#!/bin/bash
# GDS 和 SPDK 修复脚本

echo "============================================================"
echo "GDS 和 SPDK 修复脚本"
echo "============================================================"

# 1. 创建 GDS 配置
echo ""
echo "1. 配置 GDS..."
sudo mkdir -p /etc/cufile.d

cat << 'EOF' | sudo tee /etc/cufile.d/gds.conf
# GDS Configuration for GPUDirect Storage
fs.gdsEnabled = 1
fs.gdsSyncXfer = 1
fs.gdsAioXfer = 1
fs.poll_mode_timeout_ns = 100000000
fs.max_pinned_memory_ratio = 0.8
EOF

echo "   ✅ GDS 配置文件已创建"
cat /etc/cufile.d/gds.conf

# 2. 安装 nvidia-gds
echo ""
echo "2. 安装 nvidia-gds..."
if ! command -v nvidia-fs &> /dev/null; then
    echo "   正在安装 nvidia-gds..."
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-gds 2>&1 | grep -E "(Setting up|Unpacking|Processing)" || echo "   安装完成或已安装"
else
    echo "   ✅ nvidia-gds 已安装"
fi

# 3. 检查 SPDK
echo ""
echo "3. 检查 SPDK..."
if ! command -v spdk_tgt &> /dev/null; then
    echo "   正在安装 SPDK..."
    sudo apt-get install -y spdk libspdk-dev 2>&1 | grep -E "(Setting up|Unpacking|Processing)" || echo "   安装完成或已安装"
else
    echo "   ✅ SPDK 已安装"
fi

# 4. 配置大页内存 (SPDK 需要)
echo ""
echo "4. 配置大页内存..."
CURRENT_HUGEPAGES=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages)
if [ "$CURRENT_HUGEPAGES" -eq 0 ]; then
    echo "   正在配置 1024 个 2MB 大页..."
    sudo echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
    echo "   ✅ 大页内存已配置"
else
    echo "   ✅ 大页内存已配置: $CURRENT_HUGEPAGES 页"
fi

# 5. 挂载 hugetlbfs (如果未挂载)
echo ""
echo "5. 检查 hugetlbfs..."
if ! mount | grep -q hugetlbfs; then
    echo "   正在挂载 hugetlbfs..."
    sudo mkdir -p /dev/hugepages
    sudo mount -t hugetlbfs nodev /dev/hugepages
    echo "   ✅ hugetlbfs 已挂载"
else
    echo "   ✅ hugetlbfs 已挂载"
fi

# 6. 加载 VFIO 模块
echo ""
echo "6. 加载 VFIO 模块..."
sudo modprobe vfio
sudo modprobe vfio_iommu_type1
sudo modprobe vfio_pci
echo "   ✅ VFIO 模块已加载"

# 7. 验证配置
echo ""
echo "7. 验证配置..."
echo ""
echo "   GDS 配置:"
if [ -f "/etc/cufile.d/gds.conf" ]; then
    echo "   ✅ /etc/cufile.d/gds.conf 存在"
fi

echo ""
echo "   大页内存:"
grep HugePages /proc/meminfo | head -2

echo ""
echo "   VFIO 设备:"
ls -la /dev/vfio/ 2>/dev/null | head -3 || echo "   未找到 VFIO 设备"

echo ""
echo "============================================================"
echo "修复完成！"
echo "============================================================"
echo ""
echo "注意事项:"
echo "1. GDS 需要支持的文件系统 (XFS, EXT4)"
echo "2. SPDK 需要 NVMe SSD 设备 (当前系统无 NVMe)"
echo "3. 建议重启系统使所有配置生效"
echo ""
echo "验证命令:"
echo "  - GDS: python3 -c \"import cufile; print(cufile.get_gds_status())\""
echo "  - SPDK: sudo /usr/share/spdk/scripts/setup.sh status"
echo ""
echo "============================================================"