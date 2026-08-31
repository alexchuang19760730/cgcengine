#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# 微调前后 Terminal-Bench 分数对比（M4 上跑）：
#   1) base 轮：未微调的 gemma4（mlx_lm.server 端口 $MLX_BASE_PORT）
#   2) 微调：mlx_lm.lora 用 sft_data_ft 训练
#   3) ft 轮：加载适配器的 gemma4（mlx_lm.server 端口 $MLX_FT_PORT）
#   4) compare_rounds.py 输出逐任务 pass/fail 对比
#
# 前提：两个 mlx_lm.server 已在跑（或本脚本用 --auto-server 帮您拉起）：
#   终端1: mlx_lm.server --model <MLX_MODEL> --port $MLX_BASE_PORT
#   终端2: mlx_lm.server --model <MLX_MODEL> --adapter-path <ADAPTER_DIR> --port $MLX_FT_PORT
#
# 用法: ./compare_finetune.sh [--skip-train] [--only-round base|ft]
# ============================================================
TB_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=config.env
source "$TB_LOOP_DIR/config.env"

SKIP_TRAIN=0
ONLY_ROUND=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-train) SKIP_TRAIN=1; shift ;;
        --only-round) ONLY_ROUND="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

BASE_URL="$TB_GEMMA4_BASE_URL"
# 容器内通过 host.docker.internal 访问宿主 M4（与 TB_GEMMA4_BASE_URL 同约定）
FT_URL="http://host.docker.internal:$MLX_FT_PORT/v1"
ADAPTER_DIR="$TB_LOOP_DIR/finetune/adapters/gemma4-tb-lora"

[[ "$ONLY_ROUND" == "" || "$ONLY_ROUND" == "base" ]] && {
    echo "================ 1/4 base 轮（未微调 gemma4）================"
    bash "$TB_LOOP_DIR/finetune/eval_round.sh" base "$BASE_URL"
}

if [[ "$ONLY_ROUND" == "" || "$ONLY_ROUND" == "ft" ]]; then
    if [[ "$SKIP_TRAIN" == "0" ]]; then
        echo "================ 2/4 微调（sft_data_ft → LoRA）================"
        bash "$TB_LOOP_DIR/finetune/finetune_gemma4.sh"
    else
        echo "== [--skip-train] 跳过微调，直接用已有适配器: $ADAPTER_DIR"
    fi
    echo "================ 3/4 ft 轮（gemma4 + LoRA 适配器）================"
    bash "$TB_LOOP_DIR/finetune/eval_round.sh" ft "$FT_URL"
fi

echo "================ 4/4 对比 ================"
"$TB_VENV_PY" "$TB_LOOP_DIR/learning/compare_rounds.py" \
    "$TB_RESULTS_DIR/compare/base/base" \
    "$TB_RESULTS_DIR/compare/ft/ft" || true
