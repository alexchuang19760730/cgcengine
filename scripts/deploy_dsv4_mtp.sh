#!/bin/bash
# DSV4 MTP Deployment Script
# Deploys DeepSeek V4 Flash with native NEXTN MTP on TP8
# Usage: bash deploy_dsv4_mtp.sh

set -e

export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_ENDPOINT=https://hf-mirror.com

MODEL_PATH="/data/models/DeepSeek-V4-Flash"
PORT=30001
TP=8
LOG_FILE="/tmp/dsv4_sglang_mtp.log"

echo "=== DSV4 NEXTN MTP Deployment ==="
echo "Model: $MODEL_PATH"
echo "TP: $TP"
echo "Port: $PORT"
echo "Log: $LOG_FILE"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Kill any existing sglang on this port
echo "Stopping existing sglang processes..."
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 3

# Wait for GPU memory to free up
echo "Waiting for GPU memory to free..."
for i in $(seq 1 10); do
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -8 | awk '{sum+=$1} END {print sum}')
    echo "  Total GPU memory in use: ${GPU_MEM} MiB"
    if [ "$GPU_MEM" -lt 10000 ]; then
        echo "  GPUs are free!"
        break
    fi
    echo "  Waiting... ($i/10)"
    sleep 5
done

echo ""
echo "Launching DSV4 with NEXTN MTP..."
echo "Command:"
echo "  python3 -m sglang.launch_server \\"
echo "    --model-path $MODEL_PATH \\"
echo "    --speculative-algorithm NEXTN \\"
echo "    --speculative-num-steps 4 \\"
echo "    --speculative-eagle-topk 1 \\"
echo "    --speculative-num-draft-tokens 4 \\"
echo "    --tp $TP \\"
echo "    --host 0.0.0.0 --port $PORT \\"
echo "    --trust-remote-code \\"
echo "    --mem-fraction-static 0.60 \\"
echo "    --cuda-graph-max-bs 4"
echo ""

nohup /data/venv_gemma4/bin/python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp $TP \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --mem-fraction-static 0.60 \
    --cuda-graph-max-bs 4 \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Waiting for server to start (checking every 10s)..."

for i in $(seq 1 60); do
    sleep 10
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo ""
        echo "=== SERVER IS READY ==="
        echo "Endpoint: http://localhost:$PORT"
        
        # Quick test
        echo ""
        echo "Running quick test..."
        curl -s "http://localhost:$PORT/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "default",
                "prompt": "Write a Python function to calculate fibonacci numbers:",
                "max_tokens": 50,
                "temperature": 0
            }' | python3 -m json.tool 2>/dev/null || echo "(test request sent)"
        
        echo ""
        echo "=== DEPLOYMENT SUCCESSFUL ==="
        exit 0
    fi
    
    # Check if process died
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "=== SERVER PROCESS DIED ==="
        echo "Last 30 lines of log:"
        tail -30 "$LOG_FILE"
        exit 1
    fi
    
    echo -n "  [$i/60] "
    tail -1 "$LOG_FILE" 2>/dev/null | head -c 100
    echo ""
done

echo ""
echo "=== TIMEOUT: Server did not start in 600s ==="
echo "Last 30 lines of log:"
tail -30 "$LOG_FILE"
exit 1
