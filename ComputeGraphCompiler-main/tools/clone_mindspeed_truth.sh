#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-}"
if [[ -z "$DEST" ]]; then
  echo "Usage: $0 <dest_dir>" >&2
  exit 2
fi

mkdir -p "$DEST"

if [[ -d "$DEST/MindSpeed-LLM/.git" ]]; then
  (cd "$DEST/MindSpeed-LLM" && git fetch --all --prune)
else
  git clone https://gitcode.com/ascend/MindSpeed-LLM.git "$DEST/MindSpeed-LLM"
fi

if [[ -d "$DEST/MindSpeed/.git" ]]; then
  (cd "$DEST/MindSpeed" && git fetch --all --prune)
else
  git clone https://gitcode.com/ascend/MindSpeed.git "$DEST/MindSpeed"
fi

(cd "$DEST/MindSpeed-LLM" && git rev-parse HEAD)
(cd "$DEST/MindSpeed" && git rev-parse HEAD)
