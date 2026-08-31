#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# 用 freebuff2api 驱动 codebuff agent 生成 Terminal-Bench 成功轨迹，
# 并整理成 MLX LoRA 可用的 SFT 数据集。
#
# 流程：
#   1) tb run 用 CodebuffApiAgent（host 侧迭代 agent）跑任务，
#      每步调 freebuff2api（codebuff 云端 agent），轨迹写入 agent-logs/
#   2) build_sft_dataset.py 只保留 is_resolved=true 的任务，
#      转成 train.jsonl / valid.jsonl（messages 对话格式）
#
# 用法: ./gen_sft.sh [run_id]
# ============================================================
TB_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "$TB_LOOP_DIR/config.env"

RUN_ID="${1:-gen_1}"
OUT="$TB_RESULTS_DIR/sft_gen/$RUN_ID"
mkdir -p "$(dirname "$OUT")"

echo "==> [1/2] tb run: codebuff-api agent, n_tasks=$SFT_N_TASKS, model=$SFT_MODEL"
# shellcheck disable=SC2086
PYTHONPATH="$TB_LOOP_DIR" "$TB_TB_BIN" run \
    -d "$TB_DATASET" \
    --agent-import-path "tb_loop.agents.codebuff_api_agent:CodebuffApiAgent" \
    -m "openai/$SFT_MODEL" \
    -k model_name="$SFT_MODEL" \
    -k api_key="$SFT_API_KEY" \
    -k base_url="$SFT_API_BASE_URL" \
    -k max_steps="$SFT_MAX_STEPS" \
    -k max_tokens="$SFT_MAX_TOKENS" \
    --n-tasks "$SFT_N_TASKS" \
    --n-concurrent 1 \
    --output-path "$(dirname "$OUT")" \
    --run-id "$RUN_ID" \
    ${TB_TASK_IDS:+--task-id "$TB_TASK_IDS"}

echo "==> [2/2] build SFT dataset (正例 + ${SFT_INCLUDE_FAILED:+反例})"
"$TB_VENV_PY" "$TB_LOOP_DIR/learning/build_sft_dataset.py" \
    "$OUT" --out-dir "$SFT_DATA_DIR" ${SFT_INCLUDE_FAILED:+--include-failed}

echo
echo "=== 完成。数据集: $SFT_DATA_DIR ==="
echo "  训练: mlx_lm.lora --train --data $SFT_DATA_DIR --model <gemma4 基础模型>"
