#!/bin/bash
# Mac Edge Stack Launch Script for CGC Gate 6.0
# Mac → Host2 (47.95.250.55) via SSH tunnel
# TTFT target: 30ms (edge-first MTP draft model)
# Decode target: 50 token/s (cloud SGLang DSV4+MTP)

set -e

REPO_ROOT="/Users/alexchuang/Documents/flashkv0516"
PYTHON_BIN="/Users/alexchuang/.workbuddy/binaries/python/envs/cgc_edge/bin/python3"

# Kill any existing processes on our ports
pkill -f "uvicorn.*18000" 2>/dev/null || true
pkill -f "uvicorn.*14000" 2>/dev/null || true
pkill -f "edge_first_proxy" 2>/dev/null || true
pkill -f "cgc_api_server" 2>/dev/null || true
sleep 1

# === Environment Variables ===
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/ComputeGraphCompiler-main:${PYTHONPATH:-}"

# Cloud connection (via SSH tunnel to Host2)
export CGC_CLOUD_HTTP_HOST="127.0.0.1"
export CGC_CLOUD_HTTP_PORT="50053"
export CGC_EDGE_API_PORT="18000"
export CGC_FUSION_CONFIG="gate_6_0"

# Edge-first proxy settings
export EDGE_FIRST_ENABLED="1"
export CLOUD_URL="http://127.0.0.1:30000"
export EDGE_FIRST_SPECULATION_MIN_CONFIDENCE="0.84"
export EDGE_FIRST_ENABLE_WARMUP="1"
export EDGE_FIRST_WARMUP_TTL_SEC="600"

# Gate 6.0 contracts
export CGC_ENABLE_CQ4="1"
export CGC_ENABLE_GDS="1"
export CGC_ZERO_COPY_VRAM="1"
export CGC_CLOUD_PREFILL_EDGE_DECODE="1"
export CGC_ENABLE_ORTHO_KDA="1"
export CGC_KV_DIFF_ALGORITHM="lz4"
export CGC_ENABLE_WARMUP="1"
export CGC_ENABLE_DEEP_EP="1"
export CGC_LOAD_BALANCING="eplb,waterfill,lplb"
export CGC_NUM_EXPERTS="8"
export CGC_EXPERT_CAPACITY="128"
export CGC_ENABLE_RSWA="1"
export CGC_RSWA_WINDOW_SIZE="8192"
export CGC_ENABLE_PREFILL_POOL="1"
export CGC_PREFILL_POOL_SIZE="1024"
export CGC_ENABLE_DYNAMIC_EXPANSION="1"
export CGC_ENABLE_JETSPEC="1"
export CGC_ENABLE_NFS_RDMA="1"

echo "=========================================================="
echo "🚀 CGC Mac Edge Stack - Gate 6.0"
echo "   Cloud: Host2 (47.95.250.55) via SSH tunnel"
echo "   SGLang: localhost:30000 → Host2:30000"
echo "   Gateway: localhost:50053 → Host2:30000"
echo "   Edge API: http://127.0.0.1:18000"
echo "   Edge Proxy (edge-first): http://127.0.0.1:14000"
echo "   TTFT target: 30ms | Decode target: 50 tok/s"
echo "=========================================================="

# Start cgc_api_server on port 18000
echo "[1/2] Starting CGC API Server on port 18000..."
cd "${REPO_ROOT}"
nohup "${PYTHON_BIN}" -m uvicorn app.servers.cgc_api_server:app \
    --host 0.0.0.0 --port 18000 --log-level info \
    > /tmp/cgc_api_server_mac.log 2>&1 &
API_PID=$!
echo "   PID: ${API_PID}"

sleep 2

# Start edge_first_proxy on port 14000 (replaces internal proxy)
echo "[2/2] Starting Edge-First Proxy on port 14000..."
nohup "${PYTHON_BIN}" "${REPO_ROOT}/app/servers/edge_first_proxy.py" \
    --port 14000 --host 127.0.0.1 \
    --cloud-url http://127.0.0.1:30000 \
    > /tmp/cgc_edge_first_proxy_mac.log 2>&1 &
PROXY_PID=$!
echo "   PID: ${PROXY_PID}"

sleep 2

# Verify services are running
echo ""
echo "=== Service Status ==="
if kill -0 ${API_PID} 2>/dev/null; then
    echo "✅ CGC API Server (port 18000) - Running (PID: ${API_PID})"
else
    echo "❌ CGC API Server failed to start. Check /tmp/cgc_api_server_mac.log"
    tail -20 /tmp/cgc_api_server_mac.log
fi

if kill -0 ${PROXY_PID} 2>/dev/null; then
    echo "✅ Edge-First Proxy (port 14000) - Running (PID: ${PROXY_PID})"
else
    echo "❌ Edge-First Proxy failed to start. Check /tmp/cgc_edge_first_proxy_mac.log"
    tail -20 /tmp/cgc_edge_first_proxy_mac.log
fi

echo ""
echo "=== Usage ==="
echo "  cgc run deepseek-v4-flash:latest --prompt 'Hello'"
echo "  cgc claude"
echo ""
echo "  Edge API: http://127.0.0.1:18000/v1/chat/completions"
echo "  Edge Proxy: http://127.0.0.1:14000/v1/chat/completions"
echo ""
echo "Logs:"
echo "  API: /tmp/cgc_api_server_mac.log"
echo "  Proxy: /tmp/cgc_edge_first_proxy_mac.log"
