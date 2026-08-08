#!/bin/bash
# Deploy aiohttp proxy + MTP parameter changes to Host1
# Usage: bash deploy_host1.sh

set -e

HOST1="root@39.106.118.206"
PASS="Gen@song@2026622"
SSH="sshpass -p $PASS /usr/bin/ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o PreferredAuthentications=password -o PubkeyAuthentication=no $HOST1"
SCP="sshpass -p $PASS /usr/bin/scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no"

echo "=== 1. Copy updated edge_first_proxy.py to Host1 ==="
$SCP /Users/alexchuang/Documents/flashkv0516/app/servers/edge_first_proxy.py $HOST1:/root/flashkv0516/app/servers/edge_first_proxy.py
echo "  Done"

echo "=== 2. Check/install aiohttp in proxy venv ==="
$SSH '/root/flashkv0516/venv/bin/python3 -c "import aiohttp; print(f\"aiohttp {aiohttp.__version__}\")" 2>&1 || /root/flashkv0516/venv/bin/pip install aiohttp 2>&1'

echo "=== 3. Restart edge_first_proxy with aiohttp backend ==="
$SSH 'fuser -k 30002/tcp 2>/dev/null; sleep 2; cd /root/flashkv0516 && PYTHONPATH=/root/flashkv0516 DSV4_TOKENIZER_PATH=/data/models/gemma-4-26b-a4b-it CLOUD_URL=http://localhost:30001 EDGE_FIRST_ENABLED=1 EDGE_FIRST_SPECULATION_MIN_CONFIDENCE=0.55 nohup /root/flashkv0516/venv/bin/python3 app/servers/edge_first_proxy.py --port 30002 --cloud-url http://localhost:30001 --host 0.0.0.0 > /tmp/edge_proxy_aiohttp.log 2>&1 & sleep 3; curl -s http://localhost:30002/health | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"status\"],d.get(\"cache_sizes\"))"'

echo "=== 4. Deploy complete ==="
echo "  Proxy: http://39.106.118.206:30002 (internal only)"
echo "  sglang: http://39.106.118.206:30001"
