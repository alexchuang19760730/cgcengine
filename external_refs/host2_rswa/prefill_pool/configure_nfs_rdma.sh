#!/bin/bash
# configure_nfs_rdma.sh - 配置 NFSoRDMA 挂载

set -e

NFS_SERVER="39.106.118.206"
NFS_EXPORT="/export/cgc_data"
LOCAL_MOUNT="/data/nfs"

echo "=== NFSoRDMA 配置脚本 ==="

if [ "$(id -u)" -ne 0 ]; then
    echo "❌ 请以 root 身份运行"
    exit 1
fi

if mount | grep -q "$LOCAL_MOUNT"; then
    echo "1. 卸载现有挂载..."
    umount "$LOCAL_MOUNT"
    echo "   ✅ 卸载完成"
fi

mkdir -p "$LOCAL_MOUNT"

echo "2. 使用 RDMA 协议挂载 NFS..."
mount -t nfs -o proto=rdma,port=20048 "$NFS_SERVER:$NFS_EXPORT" "$LOCAL_MOUNT"

echo "3. 验证挂载..."
if mount | grep -q "$LOCAL_MOUNT" && mount | grep -q "proto=rdma"; then
    echo "   ✅ NFSoRDMA 挂载成功"
    mount | grep "$LOCAL_MOUNT"
else
    echo "   ❌ NFSoRDMA 挂载失败"
    exit 1
fi

echo "4. 配置开机自动挂载..."
FSTAB_ENTRY="$NFS_SERVER:$NFS_EXPORT $LOCAL_MOUNT nfs proto=rdma,port=20048 0 0"

if grep -q "$NFS_SERVER:$NFS_EXPORT" /etc/fstab; then
    echo "   ⚠️ fstab 中已存在该挂载，跳过"
else
    echo "$FSTAB_ENTRY" >> /etc/fstab
    echo "   ✅ 已添加到 fstab"
fi

echo "5. 测试写入性能..."
dd if=/dev/zero of="$LOCAL_MOUNT/test_rdma.tmp" bs=1G count=1 conv=fdatasync 2>&1 | tail -1
rm -f "$LOCAL_MOUNT/test_rdma.tmp"

echo ""
echo "✅ NFSoRDMA 配置完成!"
echo "   挂载点: $LOCAL_MOUNT"
echo "   服务器: $NFS_SERVER:$NFS_EXPORT"
