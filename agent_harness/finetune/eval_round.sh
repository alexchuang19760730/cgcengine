#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# 用指定 gemma4 端点跑一轮 Terminal-Bench 评估（prime-agent adapter）。
# 微调前后对比用：先跑 base，再跑 ft，最后 compare_rounds.py。
#
# 用法: ./eval_round.sh <round_name> <base_url>
#   例: ./eval_round.sh base http://127.0.0.1:1234/v1
#       ./eval_round.sh ft    http://127.0.0.1:1235/v1
# ============================================================
TB_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=config.env
source "$TB_LOOP_DIR/config.env"

ROUND_NAME="${1:?usage: eval_round.sh <round_name> <base_url>}"
BASE_URL="${2:?usage: eval_round.sh <round_name> <base_url>}"

# 健康检查（mlx_lm.server 的 /v1/models 或 /health 均可；有任一 200 就算活）
# 注意：BASE_URL 可能是 host.docker.internal（容器内地址），宿主侧探测时映射回 127.0.0.1
PROBE_URL="${BASE_URL//host.docker.internal/127.0.0.1}"
if ! curl -sf -m 5 "$PROBE_URL/models" >/dev/null 2>&1 \
   && ! curl -sf -m 5 "${PROBE_URL%/v1}/health" >/dev/null 2>&1; then
    echo "error: $BASE_URL 不可达。先启动 mlx_lm.server（见 serve.sh start）" >&2
    exit 1
fi

RUN_OUT="$TB_RESULTS_DIR/compare/$ROUND_NAME"
mkdir -p "$(dirname "$RUN_OUT")"
echo "== [$ROUND_NAME] tb run: n_tasks=$TB_N_TASKS model=$TB_GEMMA4_MODEL @ $BASE_URL"

# shellcheck disable=SC2086
PYTHONPATH="$TB_LOOP_DIR" "$TB_TB_BIN" run \
    -d "$TB_DATASET" \
    --agent-import-path "tb_loop.agents.prime_agent_adapter:PrimeAgentAgent" \
    -m "openai/$TB_GEMMA4_MODEL" \
    -k model_name="$TB_GEMMA4_MODEL" \
    -k api_key="$TB_GEMMA4_API_KEY" \
    -k base_url="$BASE_URL" \
    -k model_prefix="$TB_MODEL_PREFIX" \
    -k harness_dir="$TB_HARNESS_DIR" \
    -k max_turns="$TB_MAX_TURNS" \
    -k max_tokens="$TB_MAX_TOKENS" \
    -k timeout_ms="$TB_TIMEOUT_MS" \
    -k max_continuations="$TB_MAX_CONTINUATIONS" \
    --n-tasks "$TB_N_TASKS" \
    --n-concurrent "$TB_N_CONCURRENT" \
    --output-path "$RUN_OUT" \
    --run-id "$ROUND_NAME" \
    ${TB_TASK_IDS:+--task-id "$TB_TASK_IDS"}

echo "== [$ROUND_NAME] 完成: $RUN_OUT/$ROUND_NAME"
