#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# engine_loop 的 T1 蒸餾：把已跑過的迴圈證據餵給 prime-agent /refine，
# 要求它輸出 JSONL（decision / lesson），產物落在 distill/out/<ts>/。
#
# 沿用 tb_loop/learning/refine_harness.sh 的三個既有設計（PLAN §5）：
#   1. 餵「完整證據」而不是失敗摘要——由 distill/collect_evidence.py 收集
#      （未引用 episode 的原始觀測 ＋ 最新 decision 的 ruled_out ＋ 既有 lesson
#       ＋ 本輪文件的路徑）。第 2 項是關鍵：負面知識是最常被重用的部分。
#   2. 限定輸出類型：只提煉 memory/decision，不建 skill/subagent、不改系統提示詞。
#   3. 顯式 `/refine --global`：跨會話作用域。
#
# 只換兩件事（PLAN §5）：scope 標記 `[engine]`，以及要求輸出 JSONL 而非散文。
#
# 產物**不直接進 traces/**：T1 是半自動。候選先落在 distill/out/<ts>/，
# 經人審後才由 --accept 追加進 traces/decisions.jsonl 與 traces/lessons.jsonl。
# 理由：`traces/` 是訓練資料的唯一出口，而 validate.py 擋的是 schema，
# 擋不了「一條讀起來合理但其實沒證據的規訓」。
#
# 用法:
#   refine_engine.sh --dry-run            # 只印 prompt（不呼叫任何模型）——可離線驗證
#   refine_engine.sh                      # 呼叫 prime-agent，寫候選
#   refine_engine.sh --accept <ts>        # 人審後，把某次的候選追加進 traces/
# ============================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$ENGINE/../.." && pwd)"
TEMPLATE="$HERE/prompt/refine_engine.md"
OUT_ROOT="$HERE/out"
COLLECT="$HERE/collect_evidence.py"

PY="${PY:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "error: no usable python3" >&2; exit 2; }

# --accept 寫進哪裡。做成可覆寫的**唯一**理由是：否則「追加」這條路徑無法被測試
# （測試會去改真的 traces/），而一個無法被測試的寫入路徑正是這個 repo 最不想留的東西。
# 生產用法不需要設它。
TRACES_DIR="${TRACES_DIR:-$ENGINE/traces}"

# 蒸餾用的模型**沒有預設值**：用哪個模型蒸餾是一個判斷（它決定了蒸餾品質與成本），
# 不該由腳本猜一個。--dry-run 不需要它；真跑之前要 export REFINE_ENGINE_MODEL。
# （tb_loop 那份的預設是 `freebuff-codebuff/<SFT_MODEL>`，那是它的 scope 的答案，
#   不是這個 scope 的——照抄會把一個沒被檢驗過的選擇偽裝成慣例。）
REFINE_MODEL="${REFINE_ENGINE_MODEL:-}"
MAX_TURNS="${REFINE_ENGINE_MAX_TURNS:-12}"
MAX_TOKENS="${REFINE_ENGINE_MAX_TOKENS:-30000}"
TIMEOUT_MS="${REFINE_ENGINE_TIMEOUT_MS:-600000}"
MAX_CONT="${REFINE_ENGINE_MAX_CONTINUATIONS:-3}"

if [[ "${1:-}" == "--accept" ]]; then
    TS="${2:?usage: refine_engine.sh --accept <ts>}"
    D="$OUT_ROOT/$TS"
    [[ -d "$D" ]] || { echo "error: no such run: $D" >&2; exit 2; }
    for f in lesson decision; do
        src="$D/${f}s.candidate.jsonl"
        [[ -s "$src" ]] || continue
        dst="$TRACES_DIR/${f}s.jsonl"
        echo "==> 追加 $(wc -l < "$src" | tr -d ' ') 筆 $f -> ${dst#$REPO/}"
        mkdir -p "$TRACES_DIR"
        cat "$src" >> "$dst"
    done
    echo "==> 追加完成。**先跑驗證器再提交**："
    echo "    $PY $TRACES_DIR/validate.py"
    echo "    $PY $ENGINE/harness_engine/build_memories.py"
    exit 0
fi

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

EVIDENCE="$("$PY" "$COLLECT")"
NEXT_IDS="$("$PY" "$COLLECT" --next-ids)"

PROMPT="$("$PY" - "$TEMPLATE" "$EVIDENCE" "$NEXT_IDS" <<'EOF'
import sys
tpl, evidence, next_ids = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(tpl, encoding="utf-8").read()
if "{{EVIDENCE}}" not in text or "{{NEXT_IDS}}" not in text:
    raise SystemExit("prompt template lost its placeholders -- {{EVIDENCE}}/{{NEXT_IDS}} must both exist")
sys.stdout.write(text.replace("{{EVIDENCE}}", evidence).replace("{{NEXT_IDS}}", next_ids))
EOF
)"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "$PROMPT"
    echo "" >&2
    echo "[dry-run] prompt bytes: ${#PROMPT}   model would be: $REFINE_MODEL" >&2
    echo "[dry-run] prompt 首行: $(echo "$PROMPT" | head -1)" >&2
    exit 0
fi

if [[ -z "$REFINE_MODEL" ]]; then
    echo "error: REFINE_ENGINE_MODEL is unset. This script will not guess a distillation model --" >&2
    echo "       the choice decides quality and cost and belongs to the operator." >&2
    echo "       Try:  REFINE_ENGINE_MODEL=<provider>/<model> $0" >&2
    exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$OUT_ROOT/$TS"
mkdir -p "$OUT"
echo "==> run $TS  model=$REFINE_MODEL  out=${OUT#$REPO/}"

export PRIME_AGENT_CODING_AGENT_DIR="$ENGINE/harness_engine"

# stdin=DEVNULL：无 TTY/管道环境下 prime-agent 会等 stdin 挂起（tb_loop 已踩过）
set +e
prime-agent -p --offline \
    --model "$REFINE_MODEL" \
    --autonomous \
    --autonomous-max-turns "$MAX_TURNS" \
    --autonomous-max-tokens "$MAX_TOKENS" \
    --autonomous-timeout-ms "$TIMEOUT_MS" \
    --autonomous-max-continuations "$MAX_CONT" \
    "$PROMPT" | tee "$OUT/raw.stdout" >/dev/null
RC=$?
set -e
echo "==> prime-agent rc=$RC"

# 把 stdout 裡的 JSONL 行分離出來（模型可能夾雜散文或 code fence）
"$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
raw = (out / "raw.stdout")
lines = raw.read_text(encoding="utf-8", errors="replace").splitlines() if raw.exists() else []
buckets = {"lesson": [], "decision": []}
rejected = []
for ln in lines:
    s = ln.strip()
    if not (s.startswith("{") and s.endswith("}")):
        continue
    try:
        rec = json.loads(s)
    except Exception:
        rejected.append(s[:120]); continue
    t = rec.get("type")
    if t in buckets:
        buckets[t].append(rec)
    else:
        rejected.append(s[:120])
for kind, rows in buckets.items():
    p = out / f"{kind}s.candidate.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"  {kind}s: {len(rows)} -> {p.name}")
(out / "rejected.txt").write_text("\n".join(rejected) + "\n" if rejected else "", encoding="utf-8")
if rejected:
    print(f"  rejected (not a lesson/decision JSON object): {len(rejected)} line(s) -> rejected.txt")
print("  0 筆也是合法結果：證據不支持新規訓時，正確的輸出就是什麼都不寫。")
EOF

# 用 traces/ 自己的驗證器驗候選——**和正式 record 一起驗**，不是只驗候選檔。
# 只驗候選檔會讓跨 record 的完整性檢查（superseded_by 指到誰、counterexample 指的 episode
# 存不存在）失去參照，於是「id 指向不存在的東西」這一類缺陷會在候選階段全數漏掉。
if [[ -s "$OUT/lessons.candidate.jsonl" || -s "$OUT/decisions.candidate.jsonl" ]]; then
    set +e
    "$PY" "$ENGINE/traces/validate.py" "$ENGINE"/traces/*.jsonl \
        "$OUT"/lessons.candidate.jsonl "$OUT"/decisions.candidate.jsonl 2>&1 | tail -20
    set -e
fi
echo "==> 候選在 ${OUT#$REPO/}；人審後：$0 --accept $TS"
