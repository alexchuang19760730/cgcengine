#!/usr/bin/env bash
# batch_run.sh — 跑全部 76 个 Terminal-Bench 任务
# 用法: ./scripts/batch_run.sh [round_name]
# 依赖: config.env, local_rehearsal.py, freebuff2api 或本地模型
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TB_LOOP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TB_LOOP_DIR"

source config.env

ROUND="${1:-batch_$(date +%Y%m%d_%H%M)}"
TASKS_DIR="datasets/terminal-bench-core-0.1.1/tasks"
OUT_DIR="results/$ROUND"
mkdir -p "$OUT_DIR"

echo "=== Batch run: $ROUND ==="
echo "Tasks dir: $TASKS_DIR"
echo "Output: $OUT_DIR"
echo ""

# 获取所有任务
ALL_TASKS=()
for d in "$TASKS_DIR"/*/; do
  [ -f "$d/task.yaml" ] && ALL_TASKS+=("$(basename "$d")")
done
echo "Total tasks: ${#ALL_TASKS[@]}"

# 跳过已完成的
PENDING=()
SKIPPED=0
for task in "${ALL_TASKS[@]}"; do
  if [ -f "$OUT_DIR/$task/results.json" ]; then
    SKIPPED=$((SKIPPED + 1))
  else
    PENDING+=("$task")
  fi
done
echo "Already done: $SKIPPED"
echo "Pending: ${#PENDING[@]}"
echo ""

if [ ${#PENDING[@]} -eq 0 ]; then
  echo "All tasks completed!"
  exit 0
fi

# 逐任务跑
PASSED=0
FAILED=0
for i in "${!PENDING[@]}"; do
  task="${PENDING[$i]}"
  idx=$((i + 1))
  total=${#PENDING[@]}
  echo "--- [$idx/$total] $task ---"

  # 获取 API key
  API_KEY="${SFT_API_KEY:-}"
  if [ -z "$API_KEY" ]; then
    echo "ERROR: SFT_API_KEY not set. Set it in config.env or environment."
    exit 1
  fi

  # 跑任务
  python "$SCRIPT_DIR/local_rehearsal.py" \
    --task "$task" \
    --tasks-dir "$TASKS_DIR" \
    --output "$OUT_DIR" \
    --api-base-url "$SFT_API_BASE_URL" \
    --api-key "$API_KEY" \
    --model "$SFT_MODEL" \
    --max-steps 12 \
    --command-timeout 120 \
    --rewrite-app \
    2>&1 | tail -5

  # 检查结果
  if [ -f "$OUT_DIR/$task/results.json" ]; then
    resolved=$(python -c "import json; print(json.load(open('$OUT_DIR/$task/results.json')).get('is_resolved', False))" 2>/dev/null || echo "False")
    if [ "$resolved" = "True" ]; then
      echo "  ✅ PASS"
      PASSED=$((PASSED + 1))
    else
      echo "  ❌ FAIL"
      FAILED=$((FAILED + 1))
    fi
  else
    echo "  ⚠️  No result (crash?)"
    FAILED=$((FAILED + 1))
  fi
  echo ""
done

echo "=== Batch $ROUND complete ==="
echo "Passed: $PASSED / $total"
echo "Failed: $FAILED / $total"
echo "Skipped (already done): $SKIPPED"
