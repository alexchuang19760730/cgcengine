#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# 在 M4（Apple Silicon）上跑 MLX LoRA 微调：用 sft_data_ft 微调 gemma4。
#
# 前提（M4 一次性）：
#   python3 -m venv ~/.venvs/mlx && source ~/.venvs/mlx/bin/activate
#   pip install -U mlx-lm mlx
#   huggingface-cli download <MLX_MODEL>   # 或已有本地 MLX 权重目录
#
# 用法: ./finetune_gemma4.sh [data_dir] [adapter_dir]
# ============================================================
TB_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=config.env
source "$TB_LOOP_DIR/config.env"

DATA_DIR="${1:-$TB_LOOP_DIR/sft_data_ft}"
ADAPTER_DIR="${2:-$TB_LOOP_DIR/finetune/adapters/gemma4-tb-lora}"
mkdir -p "$(dirname "$ADAPTER_DIR")"

# 检查数据
if [[ ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/valid.jsonl" ]]; then
    echo "error: $DATA_DIR 需要 train.jsonl + valid.jsonl" >&2
    exit 1
fi
python3 -c "
import json
for f in ('$DATA_DIR/train.jsonl', '$DATA_DIR/valid.jsonl'):
    n = sum(1 for _ in open(f, encoding='utf-8'))
    last = json.loads(open(f, encoding='utf-8').readlines()[-1])['messages'][-1]['role']
    print(f'{f}: {n} examples, last_role={last}')
    assert last == 'assistant', 'last message must be assistant'
"

echo "== 数据: $DATA_DIR  (train=$(wc -l < "$DATA_DIR/train.jsonl") valid=$(wc -l < "$DATA_DIR/valid.jsonl"))"
echo "== 模型: $MLX_MODEL"
echo "== 适配器输出: $ADAPTER_DIR"

# mlx-lm 0.31.x 用 --num-layers；更老版本是 --lora-layers。自动探测。
if python3 -m mlx_lm.lora --help 2>/dev/null | grep -q -- '--num-layers'; then
    LAYERS_ARG=(--num-layers "$MLX_LORA_LAYERS")
else
    LAYERS_ARG=(--lora-layers "$MLX_LORA_LAYERS")
fi

python3 -m mlx_lm.lora \
    --model "$MLX_MODEL" \
    --train \
    --fine-tune-type lora \
    --data "$DATA_DIR" \
    "${LAYERS_ARG[@]}" \
    --batch-size "$MLX_BATCH_SIZE" \
    --iters "$MLX_ITERS" \
    --learning-rate "$MLX_LR" \
    --mask-prompt \
    --grad-checkpoint \
    --steps-per-report "$MLX_STEPS_PER_REPORT" \
    --steps-per-eval "$MLX_STEPS_PER_EVAL" \
    --val-batches 1 \
    --save-every "$MLX_SAVE_EVERY" \
    --adapter-path "$ADAPTER_DIR" \
    --seed 42

echo
echo "=== 完成。适配器: $ADAPTER_DIR ==="
echo "  服务（对比用）: mlx_lm.server --model $MLX_MODEL --adapter-path $ADAPTER_DIR --port $MLX_FT_PORT"
