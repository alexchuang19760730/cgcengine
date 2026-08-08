#!/bin/bash
# deploy_host1_8gpu.sh - Deploy 8 TP1+FP8 sglang instances on Host1
# GPU 0-7, ports 30000-30007, with load balancer + proxy

set -e

export FLASHINFER_DISABLE_VERSION_CHECK=1

echo "=== Step 1: Kill old processes ==="
pkill -f 'sglang' 2>/dev/null || true
pkill -f 'launch_server' 2>/dev/null || true
pkill -f 'edge_first_proxy' 2>/dev/null || true
pkill -f 'load_balancer' 2>/dev/null || true
sleep 3
echo "Old processes killed"

echo "=== Step 2: Check GPU status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader

echo "=== Step 3: Check model config ==="
python3 -c "
import json
c=json.load(open('/data/models/gemma-4-26b-a4b-it/config.json'))
print('dtype:', c.get('torch_dtype'))
print('hidden:', c.get('hidden_size'))
print('experts:', c.get('num_local_experts'))
print('per_tok:', c.get('num_experts_per_tok'))
print('vocab:', c.get('vocab_size'))
print('layers:', c.get('num_hidden_layers'))
" 2>&1

echo "=== Step 4: Create sglang instance script ==="
cat > /tmp/sglang_instance.sh << 'INSTANCE_SCRIPT'
#!/bin/bash
# Usage: sglang_instance.sh <gpu_id> <port>
GPU_ID=$1
PORT=$2
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=$GPU_ID

exec python3 -m sglang.launch_server \
  --model-path /data/models/gemma-4-26b-a4b-it \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /data/models/gemma-4-26b-a4b-it-assistant \
  --speculative-num-steps 5 --speculative-eagle-topk 2 --speculative-num-draft-tokens 6 \
  --trust-remote-code --attention-backend triton --sampling-backend pytorch \
  --quantization fp8 \
  --port $PORT --tp 1 --mem-fraction-static 0.80 --log-level info
INSTANCE_SCRIPT
chmod +x /tmp/sglang_instance.sh
echo "Instance script created"

echo "=== Step 5: Starting 8 sglang instances (GPU 0-7, port 30000-30007) ==="
for i in $(seq 0 7); do
  PORT=$((30000 + i))
  echo "Starting GPU $i on port $PORT..."
  nohup setsid bash /tmp/sglang_instance.sh $i $PORT > /tmp/sglang_gpu${i}.log 2>&1 < /dev/null &
  disown
  sleep 3
done
echo "All 8 instances launched"

echo "=== Step 6: Wait for instances to start (30s) ==="
sleep 30

echo "=== Step 7: Check instance status ==="
RUNNING=0
for i in $(seq 0 7); do
  PORT=$((30000 + i))
  if curl -s --max-time 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q "ok\|true\|healthy"; then
    echo "GPU $i (port $PORT): HEALTHY"
    RUNNING=$((RUNNING + 1))
  else
    echo "GPU $i (port $PORT): NOT_READY (check /tmp/sglang_gpu${i}.log)"
    tail -3 /tmp/sglang_gpu${i}.log 2>/dev/null
  fi
done
echo "=== $RUNNING/8 instances healthy ==="

echo "=== Step 8: GPU memory usage ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== Step 9: Check for FP8 errors ==="
for i in $(seq 0 7); do
  ERR=$(grep -i "error\|exception\|traceback" /tmp/sglang_gpu${i}.log 2>/dev/null | head -1)
  if [ -n "$ERR" ]; then
    echo "GPU $i ERROR: $ERR"
  fi
done

echo "=== Deploy complete ==="
