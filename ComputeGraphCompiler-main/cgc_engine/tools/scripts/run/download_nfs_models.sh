#!/bin/bash
# Download UI-TARS-7B-DPO to /data/models/ for CGC FusionRoute Agent :50073
# Usage: bash download_models.sh
# Run on Host1 (39.106.118.206) as root
# NOTE: /nfs is on root partition which is 100% full; use /data/models/ (527GB free local SSD)

set -e
echo "=== CGC FusionRoute - Model Download Script ==="
echo "UI-TARS-7B-DPO target: /data/models/UI-TARS-7B-DPO (:50073 UITARS executor)"
echo "TMAX-9B target:        /data/models/TMAX-9B (:50063 TMAX planner)"
echo ""

mkdir -p /data/models /data/hf_tmp
export TMPDIR=/data/hf_tmp
export TEMP=/data/hf_tmp
export TMP=/data/hf_tmp

# Check hf CLI
if ! command -v hf &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install -U huggingface_hub
fi

export HF_ENDPOINT=https://hf-mirror.com

if [ -d "/data/models/UI-TARS-7B-DPO" ] && [ -f "/data/models/UI-TARS-7B-DPO/model-00001-of-00004.safetensors" ]; then
    echo "UI-TARS-7B-DPO already downloaded at /data/models/UI-TARS-7B-DPO/"
    du -sh /data/models/UI-TARS-7B-DPO/
else
    echo "Starting UI-TARS-7B-DPO download (~16GB) to /data/models/UI-TARS-7B-DPO ..."
    echo "Log: /tmp/uitars7b_download.log"
    nohup hf download bytedance-research/UI-TARS-7B-DPO \
        --local-dir /data/models/UI-TARS-7B-DPO \
        --max-workers 4 \
        > /tmp/uitars7b_download.log 2>&1 &
    echo "Download PID: $!"
    echo "Monitor: tail -f /tmp/uitars7b_download.log"
fi

echo ""
echo "=== TMAX-9B download instructions (:50063) ==="
echo "TMAX-9B is NOT on HuggingFace. Get it from:"
echo "  https://github.com/hamishivi/tmax"
echo "Check README for Google Drive/AI2 weight download links"
echo "Place extracted weights at: /data/models/TMAX-9B/"
echo ""
echo "After both models are downloaded, AgentModelBackend auto-detects them."
