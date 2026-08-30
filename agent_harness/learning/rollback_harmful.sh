#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# 按归因报告自动回滚"有害" refinement（教训级归因的闭环）。
#
# 回滚单位是 refinement id（prime-agent 的 /refine rollback <id> --global
# 会反向重放该 refinement 的所有 edits，见 refinement.ts rollbackProposal）。
#
# 默认 dry-run（只打印将执行的回滚）；加 --execute 才真正调用 prime-agent。
#
# 用法:
#   ./rollback_harmful.sh <attribution_report.json> [--execute]
# ============================================================
REPORT="${1:?用法: rollback_harmful.sh <report.json> [--execute]}"
EXECUTE=0
[[ "${2:-}" == "--execute" ]] && EXECUTE=1

TB_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=config.env
source "$TB_LOOP_DIR/config.env"

# 找一个能用的 python（Windows 上 python3 可能是坏的 WindowsApps 桩）
PY=""
if command -v python3 >/dev/null 2>&1 && python3 -c 'pass' >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1 && python -c 'pass' >/dev/null 2>&1; then
    PY=python
elif [[ -x "$TB_LOOP_DIR/../.tb-loop-venv/Scripts/python" ]]; then
    PY="$TB_LOOP_DIR/../.tb-loop-venv/Scripts/python"
else
    echo "error: 找不到可用的 python（尝试 python3 / python / .tb-loop-venv）" >&2
    exit 2
fi

export PRIME_AGENT_CODING_AGENT_DIR="$TB_HARNESS_DIR"

REFINE_MODEL="${TB_REFINE_MODEL_PATTERN:-freebuff-codebuff/${SFT_MODEL}}"
if [[ "$REFINE_MODEL" == freebuff-codebuff/* ]]; then
    probe="${SFT_API_BASE_URL//host.docker.internal/127.0.0.1}"
    if ! curl -sf -m 8 -H "Authorization: Bearer $SFT_API_KEY" "$probe/models" >/dev/null 2>&1; then
        echo "WARN: freebuff2api ($probe) 不可达 -> 回退本地模型 $TB_MODEL_PATTERN"
        REFINE_MODEL="$TB_MODEL_PATTERN"
    fi
fi

IDS="$("$PY" - "$REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
for rid in report.get("rollback", []):
    print(rid)
PY
)"

if [[ -z "$IDS" ]]; then
    echo "== 报告里没有需要回滚的有害 refinement"
    exit 0
fi

echo "== 将回滚以下有害 refinement（model=$REFINE_MODEL）:"
echo "$IDS" | sed 's/^/   - /'
if [[ "$EXECUTE" != "1" ]]; then
    echo
    echo "== dry-run：未执行。确认后加 --execute 真正回滚"
    exit 0
fi

while IFS= read -r rid; do
    [[ -z "$rid" ]] && continue
    echo
    echo "== /refine rollback $rid --global"
    "$PY" - "$rid" "$REFINE_MODEL" <<'PY'
import os, subprocess, sys
rid, model = sys.argv[1], sys.argv[2]
prompt = f"/refine rollback {rid} --global"
cmd = [
    "prime-agent", "-p", "--offline",
    "--model", model,
    "--autonomous",
    "--autonomous-max-turns", os.environ.get("TB_MAX_TURNS", "12"),
    "--autonomous-max-tokens", os.environ.get("TB_MAX_TOKENS", "30000"),
    "--autonomous-timeout-ms", os.environ.get("TB_TIMEOUT_MS", "600000"),
    "--autonomous-max-continuations", os.environ.get("TB_MAX_CONTINUATIONS", "3"),
    prompt,
]
r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
if r.returncode != 0:
    print(f"  WARN rc={r.returncode}: {(r.stderr or r.stdout or '')[-500:]}")
else:
    out = (r.stdout or "").strip()
    print(f"  OK: {out[-400:]}")
PY
done <<< "$IDS"

echo
echo "回滚后 harness 状态:"
find "$TB_HARNESS_DIR" -name "harness_state.json" -exec ls -la {} \;
