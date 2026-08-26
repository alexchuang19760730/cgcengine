#!/bin/bash
# One-click deploy to HarmonyOS PC
set -e

[ -z "$1" ] && { echo "Usage: $0 <user@harmonyos-pc>"; exit 1; }

REMOTE="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deploying to $REMOTE ==="
scp -r "$SCRIPT_DIR" "$REMOTE:~/llama-cpp/" 2>&1
ssh "$REMOTE" "cd ~/llama-cpp/harmonyos && chmod +x build.sh && ./build.sh" 2>&1
echo "=== Done ==="
echo "Run: ssh $REMOTE 'cd ~/llama-cpp/harmonyos && ./run.sh -m ~/models/model.gguf'"
