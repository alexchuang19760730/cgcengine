#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# Terminal-Bench × gemma4 × prime-agent 评估-学习循环主入口
#
# 每轮：
#   1) tb run 用 gemma4 + prime-agent 跑一轮任务（容器内 prime-agent，
#      harness 状态已从 tb_loop/harness 注入）
#   2) extract_failures.py 提取失败任务
#   3) refine_harness.sh 在 host 侧把失败教训 /refine 进 harness
#   4) compare_rounds.py 与上一轮对比
#
# 用法: ./run_round.sh [起始轮次]
# ============================================================
TB_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "$TB_LOOP_DIR/config.env"

FIRST_ROUND="${1:-1}"
PREV_RUN_DIR=""
PREV_ROUND_OUT=""

mkdir -p "$TB_RESULTS_DIR" "$TB_HARNESS_DIR"

for ((round = FIRST_ROUND; round <= TB_ROUNDS; round++)); do
    echo
    echo "================ ROUND $round ================"
    ROUND_OUT="$TB_RESULTS_DIR/round_$round"
    RUN_ID="round_$round"
    mkdir -p "$ROUND_OUT"

    # 1) 评估 ------------------------------------------------------------
    echo "[round $round] tb run: dataset=$TB_DATASET n_tasks=$TB_N_TASKS model=$TB_MODEL_PATTERN"
    # shellcheck disable=SC2086
    PYTHONPATH="$TB_LOOP_DIR" "$TB_TB_BIN" run \
        -d "$TB_DATASET" \
        --agent-import-path "tb_loop.agents.prime_agent_adapter:PrimeAgentAgent" \
        -m "openai/$TB_GEMMA4_MODEL" \
        -k model_name="$TB_GEMMA4_MODEL" \
        -k api_key="$TB_GEMMA4_API_KEY" \
        -k base_url="$TB_GEMMA4_BASE_URL" \
        -k model_prefix="$TB_MODEL_PREFIX" \
        -k harness_dir="$TB_HARNESS_DIR" \
        -k max_turns="$TB_MAX_TURNS" \
        -k max_tokens="$TB_MAX_TOKENS" \
        -k timeout_ms="$TB_TIMEOUT_MS" \
        -k max_continuations="$TB_MAX_CONTINUATIONS" \
        --n-tasks "$TB_N_TASKS" \
        --n-concurrent "$TB_N_CONCURRENT" \
        --output-path "$ROUND_OUT" \
        --run-id "$RUN_ID" \
        ${TB_TASK_IDS:+--task-id "$TB_TASK_IDS"}

    RUN_DIR="$ROUND_OUT/$RUN_ID"

    # 2) 提取失败 ---------------------------------------------------------
    echo
    echo "[round $round] extract failures..."
    "$TB_VENV_PY" "$TB_LOOP_DIR/learning/extract_failures.py" \
        "$RUN_DIR" --out "$ROUND_OUT/failures.json"

    # 3) host 侧 refine harness ------------------------------------------
    echo
    echo "[round $round] refine harness (max $TB_MAX_REFINE_FAILURES failures)..."
    bash "$TB_LOOP_DIR/learning/refine_harness.sh" \
        "$ROUND_OUT/failures.json" "$TB_MAX_REFINE_FAILURES"

    # 4) 与上一轮对比 -----------------------------------------------------
    if [[ -n "$PREV_RUN_DIR" ]]; then
        echo
        echo "[round $round] compare with previous round:"
        "$TB_VENV_PY" "$TB_LOOP_DIR/learning/compare_rounds.py" \
            "$PREV_RUN_DIR" "$RUN_DIR" || true
    fi

    # 5) 教训级归因 + 有害回滚（上一轮 refine 的条目变化 × 本轮成败）--------
    if [[ -n "$PREV_RUN_DIR" && -n "$PREV_ROUND_OUT" \
          && -f "$PREV_ROUND_OUT/harness_state.before.json" \
          && -f "$PREV_ROUND_OUT/harness_state.after.json" ]]; then
        echo
        echo "[round $round] 教训级归因 (round $((round-1)) vs $round):"
        "$TB_VENV_PY" "$TB_LOOP_DIR/learning/attribution.py" \
            --before "$PREV_ROUND_OUT/harness_state.before.json" \
            --after "$PREV_ROUND_OUT/harness_state.after.json" \
            --round1 "$PREV_RUN_DIR" --round2 "$RUN_DIR" \
            --min-observations "${TB_MIN_OBSERVATIONS:-2}" \
            --out "$ROUND_OUT/attribution.json" || true
        if [[ "${RUN_ROLLBACK:-0}" == "1" ]] && [[ -f "$ROUND_OUT/attribution.json" ]]; then
            echo
            echo "[round $round] 自动回滚有害 refinement:"
            bash "$TB_LOOP_DIR/learning/rollback_harmful.sh" \
                "$ROUND_OUT/attribution.json" --execute || true
        fi
    fi

    PREV_RUN_DIR="$RUN_DIR"
    PREV_ROUND_OUT="$ROUND_OUT"
done

echo
echo "=== 完成。结果目录: $TB_RESULTS_DIR ==="
for d in "$TB_RESULTS_DIR"/round_*; do
    [[ -d "$d" ]] || continue
    echo "  $d/failures.json"
done
