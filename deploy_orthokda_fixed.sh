#!/bin/bash
# Fix fill_ids bug in schedule_batch.py and restart sglang with OrthoKDA

# 1. Fix the fill_ids bug
SCHED_FILE="/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/schedule_batch.py"
if grep -q "req.fill_ids = req.fill_ids" "$SCHED_FILE"; then
    sed -i 's/req.fill_ids = req.fill_ids or \[0\] \* req.extend_input_len/req.full_untruncated_fill_ids = req.full_untruncated_fill_ids or array("q", [0] * req.extend_input_len)/' "$SCHED_FILE"
    echo "FIXED fill_ids bug in schedule_batch.py"
else
    echo "fill_ids bug not found (already fixed?)"
fi

# 2. Kill old sglang
pkill -9 -f sglang 2>/dev/null
sleep 2
fuser -k 30001/tcp 2>/dev/null
sleep 1

# 3. Start sglang with OrthoKDA
export ORTHO_KDA_ENABLED=1
export ORTHO_BASE_DIM=128
export ORTHO_REF_LEN=4
export ORTHO_WINDOW_SIZE=128
export RSWA_MAX_ATTN_LEN=0
export PYTHONPATH=/data:$PYTHONPATH

cd /data
nohup python3 -m sglang.launch_server \
    --model-path /data/models/gemma-4-26b-a4b-it \
    --tp 8 --port 30001 --host 0.0.0.0 \
    --cuda-graph-max-bs 1 \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --trust-remote-code \
    --disable-cuda-graph-padding \
    > /tmp/orthokda_sglang.log 2>&1 &

echo "STARTED PID:$!"

# 4. Wait for ready
for i in $(seq 1 30); do
    sleep 5
    if curl -s --connect-timeout 2 http://127.0.0.1:30001/v1/models 2>/dev/null | grep -q gemma; then
        echo "READY after $((i*5))s"
        # 5. Send test request
        echo "=== TEST ==="
        curl -s --connect-timeout 10 -X POST http://127.0.0.1:30001/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"/data/models/gemma-4-26b-a4b-it","messages":[{"role":"user","content":"Hello, how are you?"}],"max_tokens":20,"temperature":0}'
        echo ""
        echo "=== LOGS ==="
        grep -E "OrthoKDA|ERROR|Traceback" /tmp/orthokda_sglang.log | tail -10
        break
    fi
    if grep -q "kill_process_tree" /tmp/orthokda_sglang.log 2>/dev/null; then
        echo "CRASHED after $((i*5))s"
        grep -A 10 "Scheduler hit an exception" /tmp/orthokda_sglang.log | head -15
        break
    fi
    echo "wait $i/30"
done
