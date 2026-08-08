#!/bin/bash
# OrthoKDA recovery and test script for Host1
# Run this AFTER Host1 has been rebooted
set -e

echo "=== Step 0: Check GPU status ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo ""

echo "=== Step 1: Check sglang version ==="
python3 -c "import sglang; print('sglang version:', sglang.__version__)"
echo ""

echo "=== Step 2: Check SEQRLEN_FIX patch ==="
SCHED_FILE="/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/schedule_batch.py"
grep -n "SEQRLEN_FIX\|fill_ids.*=.*fill_ids.*or" "$SCHED_FILE" || echo "No SEQRLEN_FIX found"
echo ""

echo "=== Step 3: Check sitecustomize.py ==="
if [ -f /data/sitecustomize.py ]; then
    echo "sitecustomize.py exists at /data/"
    cat /data/sitecustomize.py
else
    echo "No sitecustomize.py at /data/"
fi
echo ""

echo "=== Step 4: Check orthokda_patch_loader.py ==="
if [ -f /data/orthokda_patch_loader.py ]; then
    echo "orthokda_patch_loader.py exists at /data/"
    head -5 /data/orthokda_patch_loader.py
else
    echo "No orthokda_patch_loader.py at /data/"
fi
echo ""

echo "=== Step 5: Kill any existing sglang processes ==="
pkill -9 -f sglang 2>/dev/null || true
pkill -9 -f launch_server 2>/dev/null || true
sleep 2
echo "Done"
echo ""

echo "=== Step 6: Start PURE sglang (no OrthoKDA, no patches) ==="
echo "Starting sglang with Gemma4 26B..."
nohup python3 -m sglang.launch_server \
    --model-path /data/models/gemma-4-26b-a4b-it \
    --tp 8 \
    --port 30001 \
    --host 0.0.0.0 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --trust-remote-code \
    --skip-server-warmup \
    > /tmp/sglang_pure.log 2>&1 &

echo "sglang PID: $!"
echo "Waiting for server to start..."
echo "Log: /tmp/sglang_pure.log"
echo ""

# Wait for server to be ready
for i in $(seq 1 120); do
    if curl -s http://localhost:30001/health 2>/dev/null | grep -q "ok\|true\|healthy"; then
        echo "Server is ready after ${i}s!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Server failed to start within 120s"
        echo "=== Last 30 lines of log ==="
        tail -30 /tmp/sglang_pure.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Step 7: Send test request ==="
curl -s http://localhost:30001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4-26b-a4b-it",
        "messages": [{"role": "user", "content": "Hello, say one word."}],
        "max_tokens": 10
    }' 2>&1

echo ""
echo "=== Step 8: Check if server is still alive ==="
if curl -s http://localhost:30001/health 2>/dev/null | grep -q "ok\|true\|healthy"; then
    echo "SUCCESS: Pure sglang works!"
    echo "Now killing pure sglang to test with OrthoKDA..."
    pkill -9 -f sglang 2>/dev/null || true
    sleep 3
else
    echo "FAILURE: Server crashed after request!"
    echo "=== Last 50 lines of log ==="
    tail -50 /tmp/sglang_pure.log
    exit 1
fi

echo ""
echo "=== Step 9: Start sglang WITH OrthoKDA ==="
ORTHO_KDA_ENABLED=1 PYTHONPATH=/data nohup python3 -m sglang.launch_server \
    --model-path /data/models/gemma-4-26b-a4b-it \
    --tp 8 \
    --port 30001 \
    --host 0.0.0.0 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --trust-remote-code \
    --skip-server-warmup \
    > /tmp/sglang_orthokda.log 2>&1 &

echo "sglang PID: $!"
echo "Waiting for OrthoKDA server to start..."
echo "Log: /tmp/sglang_orthokda.log"
echo ""

for i in $(seq 1 120); do
    if curl -s http://localhost:30001/health 2>/dev/null | grep -q "ok\|true\|healthy"; then
        echo "OrthoKDA server is ready after ${i}s!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: OrthoKDA server failed to start within 120s"
        echo "=== Last 30 lines of log ==="
        tail -30 /tmp/sglang_orthokda.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Step 10: Send test request to OrthoKDA server ==="
curl -s http://localhost:30001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4-26b-a4b-it",
        "messages": [{"role": "user", "content": "Hello, say one word."}],
        "max_tokens": 10
    }' 2>&1

echo ""
echo "=== Step 11: Check OrthoKDA logs ==="
grep -i "orthokda\|ORTHO\|ERROR\|error" /tmp/sglang_orthokda.log | head -20

echo ""
echo "=== Step 12: Check if OrthoKDA server is still alive ==="
if curl -s http://localhost:30001/health 2>/dev/null | grep -q "ok\|true\|healthy"; then
    echo "SUCCESS: OrthoKDA server works!"
else
    echo "FAILURE: OrthoKDA server crashed!"
    echo "=== Last 50 lines of log ==="
    tail -50 /tmp/sglang_orthokda.log
fi

echo ""
echo "=== DONE ==="
