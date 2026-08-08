#!/bin/bash
# deploy_16gpu.sh - Deploy 16 TP1+FP8 sglang instances across Host1 + Host2
# Host1: GPU 0-7, ports 30000-30007
# Host2: GPU 0-7, ports 30000-30007

set -e

echo "=== Killing old processes ==="
pkill -f 'sglang' 2>/dev/null || true
pkill -f 'launch_server' 2>/dev/null || true
pkill -f 'edge_first_proxy' 2>/dev/null || true
pkill -f 'load_balancer' 2>/dev/null || true
sleep 3

echo "=== GPU status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader

echo "=== Model config ==="
python3 -c "
import json
c=json.load(open('/data/models/gemma-4-26b-a4b-it/config.json'))
print('dtype:', c.get('torch_dtype'))
print('hidden:', c.get('hidden_size'))
print('experts:', c.get('num_local_experts'))
print('per_tok:', c.get('num_experts_per_tok'))
print('vocab:', c.get('vocab_size'))
print('layers:', c.get('num_hidden_layers'))
"

echo "=== Model size ==="
du -sh /data/models/gemma-4-26b-a4b-it/
du -sh /data/models/gemma-4-26b-a4b-it-assistant/

# Create startup script template
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

echo "=== Starting 8 sglang instances on Host1 (GPU 0-7, port 30000-30007) ==="
for i in $(seq 0 7); do
  PORT=$((30000 + i))
  echo "Starting GPU $i on port $PORT..."
  nohup setsid bash /tmp/sglang_instance.sh $i $PORT > /tmp/sglang_gpu${i}.log 2>&1 < /dev/null &
  disown
  sleep 2  # Stagger startup to avoid GPU memory contention
done

echo "=== All 8 instances launched on Host1 ==="
echo "Waiting 10s before status check..."
sleep 10

echo "=== Process status ==="
ps aux | grep launch_server | grep -v grep | wc -l
echo "instances running"

echo "=== GPU memory ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== Testing Host2 SSH ==="
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@47.95.250.55 "echo HOST2_OK" 2>&1 || echo "HOST2_SSH_FAILED"

echo "=== Deploy script complete ==="
