#!/bin/bash
set -e

echo "============================================================"
echo "  🔥 完整自動化 - 代碼同步 + 代理設定 + 服務器真實測試"
echo "  代理: export all_proxy=http://127.0.0.1:7897"
echo "============================================================"

LOCAL_DIR="/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/"
SERVER_USER="gs01"
SERVER_IP="10.100.200.65"
SERVER_DIR="/home/gs01/MagiCompiler"

echo ""
echo "[步驟 1/5] 同步代碼到服務器..."
rsync -avz \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '*.pyc' \
    --exclude '*.DS_Store' \
    "$LOCAL_DIR" \
    "${SERVER_USER}@${SERVER_IP}:${SERVER_DIR}/"

echo ""
echo "[步驟 2/5] 在服務器上設定代理並執行環境配置..."
ssh "${SERVER_USER}@${SERVER_IP}" << 'EOF'
    set -e
    echo ""
    echo "✅ 已登入服務器 $(hostname)"
    
    export all_proxy=http://127.0.0.1:7897
    export http_proxy=http://127.0.0.1:7897
    export https_proxy=http://127.0.0.1:7897
    echo "✅ 代理已設定: all_proxy=http://127.0.0.1:7897"
    
    cd /home/gs01/MagiCompiler
    chmod +x server/run_real_benchmark.sh
    
    echo ""
    echo "✅ 執行真實 GPU 基準測試..."
    bash server/run_real_benchmark.sh
EOF

echo ""
echo "============================================================"
echo "  ✅ 全部完成！從服務器下載測試報告..."
echo "============================================================"

rsync -avz \
    "${SERVER_USER}@${SERVER_IP}:${SERVER_DIR}/real_deepseek_v4_benchmark_report.json" \
    ./real_deepseek_v4_benchmark_report_from_server.json

echo ""
echo "✅ 報告已下載到本機: ./real_deepseek_v4_benchmark_report_from_server.json"
