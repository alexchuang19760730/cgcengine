#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-}"
if [[ -z "$ROOT" ]]; then
  echo "Usage: $0 <DSTMC_root_dir>" >&2
  exit 2
fi

ROOT="$(cd "$ROOT" && pwd)"

ASCEND="$ROOT/Ascend_system"
NVIDIA="$ROOT/Nvidia_system"
DS4="$ROOT/ds4_system"
MAC="$ROOT/Mac_system"

echo "export CGC_DSTMC=\"$ROOT\""
echo "export CGC_ASCEND_SYSTEM=\"$ASCEND\""
echo "export CGC_NVIDIA_SYSTEM=\"$NVIDIA\""
echo "export CGC_DS4_SYSTEM=\"$DS4\""
echo "export CGC_MAC_SYSTEM=\"$MAC\""
echo "export CGC_PAPERS_DIR=\"$ASCEND/papers\""
echo "export CGC_WEIGHTS_DIR=\"$ASCEND/weights\""
