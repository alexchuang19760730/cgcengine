#!/bin/bash
# 在 Host2 上训练 MTP Head
#
# 用法:
#   bash launch_mtp_train.sh
#
# 前置:
#   1. Host2 有 Qwen3-VL-2B-Instruct 模型 (/data2/models/Qwen3-VL-2B-Instruct)
#   2. 有训练语料 (JSONL 格式, 每行 {"text": "..."} 或 {"messages": [...]})
#   3. GPU 可用 (单卡 RTX PRO 5000, 24GB)

set -e

# === 配置 ===
BASE_MODEL="/data2/models/Qwen3-VL-2B-Instruct"
CORPUS_PATH="${1:-/data/mtp_corpus.jsonl}"  # 训练语料路径
DATA_DIR="/data/mtp_training_data"          # 收集的 hidden states 存储目录
OUTPUT_DIR="/data/mtp_head_output"          # 训练输出目录
MAX_SAMPLES=500000                          # 最大样本数
EPOCHS=3
BATCH_SIZE=32
LR=1e-4

# === Step 1: 数据收集 ===
echo "=========================================="
echo "Step 1: Collecting training data"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "Corpus: $CORPUS_PATH"
echo "Output: $DATA_DIR"
echo "Max samples: $MAX_SAMPLES"

if [ ! -f "$CORPUS_PATH" ]; then
    echo "ERROR: Corpus file not found: $CORPUS_PATH"
    echo "Please prepare a JSONL corpus with format:"
    echo '  {"text": "some text..."}'
    echo '  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}'
    echo ""
    echo "Suggested data sources:"
    echo "  - ShareGPT (https://huggingface.co/datasets/anon8231489683/ShareGPT_Vicuna_unfiltered)"
    echo "  - OpenOrca (https://huggingface.co/datasets/Open-Orca/OpenOrca)"
    echo "  - Alpaca (https://huggingface.co/datasets/tatsu-lab/alpaca)"
    echo "  - WikiText (https://huggingface.co/datasets/wikitext)"
    exit 1
fi

mkdir -p "$DATA_DIR"

python3 /root/flashkv0516/CGC_Phase2/mtp_head/collect_data.py \
    --model-path "$BASE_MODEL" \
    --corpus-path "$CORPUS_PATH" \
    --output-dir "$DATA_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --device cuda

echo ""
echo "Data collection done. Samples saved to $DATA_DIR"

# === Step 2: 训练 ===
echo ""
echo "=========================================="
echo "Step 2: Training MTP Head"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

python3 /root/flashkv0516/CGC_Phase2/mtp_head/train.py \
    --base-model "$BASE_MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device cuda

echo ""
echo "Training done. Checkpoint saved to $OUTPUT_DIR"

# === Step 3: 评估 ===
echo ""
echo "=========================================="
echo "Step 3: Evaluating MTP Head"
echo "=========================================="

python3 /root/flashkv0516/CGC_Phase2/mtp_head/eval.py \
    --base-model "$BASE_MODEL" \
    --mtp-checkpoint "$OUTPUT_DIR/mtp_head_final.pt" \
    --device cuda

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo "MTP head checkpoint: $OUTPUT_DIR/mtp_head_final.pt"
echo "Eval results: $OUTPUT_DIR/eval_results.json"
echo ""
echo "Next steps:"
echo "  1. Convert to MLX: python convert_mlx.py --mtp-checkpoint $OUTPUT_DIR/mtp_head_final.pt"
echo "  2. Test on Mac: use as draft_model in mlx_lm.stream_generate()"
echo "  3. Integrate with PD separation: cloud prefill → emit hidden_L → Mac MTP head 首包预测"
