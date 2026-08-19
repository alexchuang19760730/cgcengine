#!/bin/bash
#==============================================================================
# Harness Agent GDS/SPDK 策略测试 - 服务器部署脚本
#
# 功能：
# 1. 真实硬件环境测试 - 在 Linux + CUDA 环境下测试真实的 GDS/SPDK 加速
# 2. 性能基准测试 - 对比 GDS/SPDK 与标准 IO 的性能差异
# 3. 分布式部署 - 在双卡 5090 上测试 NVMe over Fabrics
#
# 针对：
# - 端侧 llama.cpp
# - 云侧 vLLM
#==============================================================================

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SERVER_SCRIPT="$PROJECT_ROOT/test_harness_gds_spdk_strategy.py"
RESULTS_FILE="/tmp/harness_gds_spdk_results.json"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境
check_environment() {
    log_info "检查运行环境..."

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    log_info "Python3: $(python3 --version)"

    # 检查 CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU 检测到:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        log_warn "未检测到 NVIDIA GPU"
    fi

    # 检查 CUDA 库
    if [ -d "/usr/local/cuda" ]; then
        log_info "CUDA 安装在: /usr/local/cuda"
    elif [ -d "/usr/local/cuda-13.0" ]; then
        log_info "CUDA 安装在: /usr/local/cuda-13.0"
        export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
    else
        log_warn "未找到 CUDA 安装"
    fi

    # 检查 liburing
    if python3 -c "import liburing" 2>/dev/null; then
        log_info "liburing: 可用"
    else
        log_warn "liburing: 不可用"
    fi

    # 检查 cuFile
    if [ -f "/usr/local/cuda/lib64/libcufile.so" ] || [ -f "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcufile.so" ]; then
        log_info "libcufile.so: 可用"
    else
        log_warn "libcufile.so: 不可用"
    fi
}

# 安装依赖
install_dependencies() {
    log_info "安装依赖..."

    cd "$PROJECT_ROOT"

    # 安装 Python 依赖
    pip3 install torch numpy --quiet

    log_info "依赖安装完成"
}

# 创建测试目录
setup_directories() {
    log_info "创建测试目录..."

    mkdir -p /data/models
    mkdir -p /data/flashmoe_experts
    mkdir -p /data/spdk_kv_cache

    log_info "测试目录创建完成"
}

# 运行基准测试
run_benchmark() {
    log_info "运行 Harness Agent GDS/SPDK 策略测试..."

    cd "$PROJECT_ROOT"

    # 设置环境变量
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    # 运行测试
    python3 "$SERVER_SCRIPT"

    # 检查结果
    if [ -f "$RESULTS_FILE" ]; then
        log_info "测试结果已保存到: $RESULTS_FILE"

        # 打印摘要
        echo ""
        log_info "=== 测试结果摘要 ==="
        python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)

print('\\n配置:')
print(f\"  GPU 数量: {data['config']['num_gpus']}\")
print(f\"  GDS 启用: {data['config']['enable_gds']}\")
print(f\"  SPDK 启用: {data['config']['enable_spdk']}\")

if 'gds' in data['benchmarks']:
    gds = data['benchmarks']['gds']
    if 'gds_vs_pytorch' in gds:
        print(f\"\\nGDS 加速比: {gds['gds_vs_pytorch'].get('speedup', 1.0):.2f}x\")

if 'spdk' in data['benchmarks']:
    spdk = data['benchmarks']['spdk']
    if 'spdk_vs_standard' in spdk:
        print(f\"SPDK 写入加速比: {spdk['spdk_vs_standard'].get('write_speedup', 1.0):.2f}x\")
        print(f\"SPDK 读取加速比: {spdk['spdk_vs_standard'].get('read_speedup', 1.0):.2f}x\")

if 'distributed' in data['benchmarks']:
    dist = data['benchmarks']['distributed']
    if 'parallel_loading' in dist:
        print(f\"双卡并行加载加速比: {dist['parallel_loading'].get('speedup', 1.0):.2f}x\")
"
    else
        log_error "测试结果文件未生成"
        exit 1
    fi
}

# 清理
cleanup() {
    log_info "清理测试数据..."

    rm -rf /data/flashmoe_experts/test_*
    rm -rf /data/spdk_kv_cache/test_*
    rm -rf /data/spdk_kv_cache/standard

    log_info "清理完成"
}

# 主函数
main() {
    echo "=============================================="
    echo "Harness Agent GDS/SPDK 策略测试"
    echo "=============================================="

    check_environment
    install_dependencies
    setup_directories
    run_benchmark

    echo ""
    log_info "测试完成!"
}

# 显示用法
usage() {
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  check     - 检查运行环境"
    echo "  setup     - 安装依赖和创建目录"
    echo "  run       - 运行基准测试"
    echo "  cleanup   - 清理测试数据"
    echo "  all       - 执行所有步骤 (默认)"
    echo ""
    echo "示例:"
    echo "  $0 all          # 执行完整测试流程"
    echo "  $0 check        # 仅检查环境"
    echo "  $0 run          # 仅运行测试"
}

# 解析参数
case "${1:-all}" in
    check)
        check_environment
        ;;
    setup)
        install_dependencies
        setup_directories
        ;;
    run)
        run_benchmark
        ;;
    cleanup)
        cleanup
        ;;
    all)
        main
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        log_error "未知命令: $1"
        usage
        exit 1
        ;;
esac
