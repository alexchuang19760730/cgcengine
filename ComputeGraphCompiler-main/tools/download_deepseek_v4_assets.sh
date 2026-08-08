#!/usr/bin/env bash
set -euo pipefail

SYSTEM="${1:-}"
MODEL="${2:-}"
DEST="${3:-}"

if [[ -z "$SYSTEM" || -z "$MODEL" || -z "$DEST" ]]; then
  echo "Usage: $0 <ascend|nvidia|mac> <v4_flash|v4_pro|paper> <dest_dir>" >&2
  exit 2
fi

mkdir -p "$DEST"

if [[ "$MODEL" == "paper" ]]; then
  URL="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf"
  OUT="$DEST/DeepSeek_V4.pdf"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$OUT" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$OUT" "$URL"
  else
    echo "Missing downloader (curl or wget)" >&2
    exit 2
  fi
  echo "$OUT"
  exit 0
fi

REPO=""
case "$MODEL" in
  v4_flash) REPO="deepseek-ai/DeepSeek-V4-Flash" ;;
  v4_pro) REPO="deepseek-ai/DeepSeek-V4-Pro" ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" --local-dir "$DEST" --local-dir-use-symlinks False
  exit 0
fi

if command -v modelscope >/dev/null 2>&1; then
  modelscope download --model "$REPO" --local-dir "$DEST"
  exit 0
fi

echo "Missing downloader: huggingface-cli or modelscope" >&2
exit 2
