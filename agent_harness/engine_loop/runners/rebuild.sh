#!/bin/bash
# runners/rebuild.sh — 建置引擎，並把「這批二進位檔是哪一批」變成一等公民 (PLAN §3)
#
#   rebuild.sh --dry-run     # 只顯示閘門判定與**當前**指紋，不建置、不寫任何檔
#   rebuild.sh               # 閘門 -> cmake --build -> 寫 build.json
#   rebuild.sh --check       # 不建置：重算指紋並與 build.json 比對，漂移則 exit 1
#   rebuild.sh --force       # 跳過閘門（**你要為此負責**；見下）
#
# 兩個設計決定，都有具體的事故在後面：
#
# --- 1. 閘門必須 abort，不能只印出來 -------------------------------------------------
# `cmake --build` 在這個 repo 裡**不是**本機動作：建置產物被納入版控，所以它會蓋掉
# `src/llama.cpp/build/bin/` 底下所有檔案 —— 而其他 session 可能正在用那些 .dylib 跑量測。
# 2026-09-17 實測：把 `lsof -nP -iTCP:8080` 與 `cmake --build` 寫在**同一行**，lsof 明明印出
# 別條線的 server 在聽，build 還是跑了 ⇒ 蓋掉他們正在 map 的 `libggml-metal`／`libllama`，
# 於是那一輪的 A/B **橫跨兩個 build**。而這個 repo 的所有比較都建立在「同一個 build」上。
#
# 所以這裡的閘門是「非空就 exit 1」，而且檢查與動作在同一個分支裡 —— 印出來不等於擋下來。
#
# --- 2. 指紋用委派，不重寫 ------------------------------------------------------------
# build 指紋的唯一定義在 `scripts/check/decode_sweep.py:build_fingerprint()`。
# `ab_interleave.py:42` 已經因為自己抄了一份而漂移過一次：抄的那份只 hash 三個手挑的檔案
# （llama-server + libggml-metal + libllama），漏掉 `libggml-base`（scheduler 與 CGC_OA_ASYNC
# 閘門所在）⇒ 兩次 scheduler 不同的 run 拿到**相同**的可比較指紋，被那個「只在同一指紋內比較」
# 的工具蓋章認定可比。見 lesson `eng-mh-0007`。
# 這支腳本因此只做一件事：import 那個模組，要它的答案。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$(cd "$HERE/.." && pwd)/config.env"
# shellcheck source=../config.env
[ -r "$CFG" ] && . "$CFG"
: "${ENGINE_REPO_ROOT:=$(cd "$HERE/../.." && pwd)}"
: "${ENGINE_FP_MODULE:=$ENGINE_REPO_ROOT/scripts/check/decode_sweep.py}"

BUILD_DIR="${ENGINE_BUILD_DIR:-$ENGINE_REPO_ROOT/src/llama.cpp/build}"
TARGET="${ENGINE_TARGET:-llama-server}"
JOBS="${ENGINE_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
OUT="${ENGINE_BUILD_JSON:-$(cd "$HERE/.." && pwd)/build.json}"
PY="${PY:-python3}"

DRY=0 CHECK=0 FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --check)   CHECK=1 ;;
        --force)   FORCE=1 ;;
        --out)     OUT="$2"; shift ;;
        --target)  TARGET="$2"; shift ;;
        -j)        JOBS="$2"; shift ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# 指紋：唯一的實作在別處，這裡只負責問
# ---------------------------------------------------------------------------
fingerprint() {
    "$PY" - "$ENGINE_FP_MODULE" <<'PY'
import importlib.util, json, sys
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("decode_sweep", path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as exc:                                  # noqa: BLE001
    print(f"error: cannot load the fingerprint authority {path}: {exc}", file=sys.stderr)
    raise SystemExit(3)
fn = getattr(mod, "build_fingerprint", None)
if fn is None:
    print(f"error: {path} no longer defines build_fingerprint() -- the delegation is stale, "
          f"fix it rather than reimplementing the rule here", file=sys.stderr)
    raise SystemExit(3)
print(json.dumps(fn(), ensure_ascii=False, sort_keys=True))
PY
}

git_head() { git -C "$ENGINE_REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown"; }

# ---------------------------------------------------------------------------
# --check：不建置，只問「現在的二進位檔是不是 build.json 說的那一批」
# ---------------------------------------------------------------------------
if [ "$CHECK" = 1 ]; then
    if [ ! -f "$OUT" ]; then
        echo "  error: $OUT 不存在 ⇒ 沒有任何已記錄的指紋可比" >&2; exit 1
    fi
    now="$(fingerprint)" || exit $?
    was="$("$PY" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["fingerprint"], ensure_ascii=False, sort_keys=True))' "$OUT")"
    if [ "$now" = "$was" ]; then
        n="$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["fingerprint"]))' "$OUT")"
        echo "  OK: build.json 的指紋（${n} 鍵）等於當前二進位檔"
        exit 0
    fi
    echo "  DRIFT: 二進位檔已不是 build.json 記錄的那一批" >&2
    "$PY" - "$OUT" "$now" <<'PY' >&2
import json, sys
was = json.load(open(sys.argv[1]))["fingerprint"]; now = json.loads(sys.argv[2])
for k in sorted(set(was) | set(now)):
    a, b = was.get(k), now.get(k)
    if a != b:
        print(f"    {k}: {a} -> {b}" + ("   (key set changed)" if (k not in was or k not in now) else ""))
PY
    echo "  ⇒ 在任何量測之前先跑 rebuild.sh；一個橫跨兩個 build 的比較不是比較。" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 閘門
# ---------------------------------------------------------------------------
gate_fail=0
gate_notes=()

port="${CGC_SERVER_PORT:-8080}"
if command -v lsof >/dev/null 2>&1; then
    listeners="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | tail -n +2)"
    if [ -n "$listeners" ]; then
        gate_fail=1
        gate_notes+=("port $port 上有 listener（別人的 server 正在服務）")
    fi
fi

# 量測行程：建置會蓋掉它們正在 map 的 dylib。名字刻意含 llama-* 與本 loop 的執行器。
meas="$(pgrep -f '[r]un_ids_dst_capture|[d]ecode_sweep|[k]nifeedge_matrix|[a]b_interleave|[m]123_oracle_gate|[p]refill_idle_sweep' 2>/dev/null || true)"
if [ -n "$meas" ]; then
    gate_fail=1
    gate_notes+=("有量測行程在跑：$(echo $meas | tr '\n' ' ')")
fi

# 殘留引擎：身分由 ps 決定（見 runners/preflight.sh 的說明），不是命令列文字
resid="$("$HERE/preflight.sh" --json 2>/dev/null | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(" ".join(d["residual_pids"]))' 2>/dev/null || true)"
if [ -n "$resid" ]; then
    gate_fail=1
    gate_notes+=("有殘留引擎行程：$resid")
fi

other_builders="$(pgrep -f '[c]make|[n]inja' 2>/dev/null || true)"
if [ -n "$other_builders" ]; then
    gate_fail=1
    gate_notes+=("另一個建置正在跑：$(echo $other_builders | tr '\n' ' ')")
fi

# ★ dry-run 不因為閘門關著而 exit 1：它的工作就是**告訴你判定**。
#   第一次寫的時候兩者共用同一個 exit，於是閘門鏈裡「檢查建置前置」這一條會在
#   別條線正在量測時變紅 —— 而那個紅燈的意義是「現在不要建置」，不是「這支工具壞了」。
#   把兩件事分開：dry-run 報告並回 0；真正的建置才拒絕（這是 2026-09-17 13:18 實測到的
#   —— 那一輪閘門真的擋下了一次，別條線正在跑 m123_oracle_gate + decode_sweep）。
if [ "$gate_fail" = 1 ] && [ "$DRY" != 1 ] && [ "$FORCE" != 1 ]; then
    echo "error: 建置閘門擋下（NOT safe to build）" >&2
    for n in "${gate_notes[@]}"; do echo "  - $n" >&2; done
    echo "  ⇒ 建置產物被納入版控，所以 cmake --build 是對『機器上所有正在跑的實驗』的一次寫入。" >&2
    echo "     等上述都結束，或明確用 --force（你要為此負責）。" >&2
    exit 1
fi
[ "$gate_fail" = 1 ] && [ "$FORCE" = 1 ] && [ "$DRY" != 1 ] \
    && echo "  ⚠ --force: 跳過閘門（${gate_notes[*]}）"

# ---------------------------------------------------------------------------
# --dry-run：閘門 + 當前指紋，不建置、不寫檔
# ---------------------------------------------------------------------------
if [ "$DRY" = 1 ]; then
    echo "== rebuild --dry-run =="
    echo "  build dir : $BUILD_DIR"
    echo "  target    : $TARGET   jobs: $JOBS"
    echo "  out       : $OUT"
    echo "  git HEAD  : $(git_head)"
    echo "  閘門      : $([ "$gate_fail" = 1 ] && echo "會擋（$(printf '%s; ' "${gate_notes[@]}")）" || echo 通過)"
    echo "  當前指紋（委派 ${ENGINE_FP_MODULE}）:"
    fingerprint | "$PY" -c 'import json,sys; [print(f"    {k:18s} {v}") for k,v in json.load(sys.stdin).items()]'
    echo "  （dry-run：沒有建置、沒有寫任何檔案）"
    exit 0
fi

# ---------------------------------------------------------------------------
# 建置
# ---------------------------------------------------------------------------
[ -f "$BUILD_DIR/CMakeCache.txt" ] || {
    echo "error: $BUILD_DIR 尚未 configure（找不到 CMakeCache.txt）" >&2
    echo "       先跑 scripts/build_fork_llama.sh 或照該檔的 cmake 參數 configure。" >&2
    exit 2
}
echo "==> cmake --build $BUILD_DIR --target $TARGET -j $JOBS"
cmake --build "$BUILD_DIR" --target "$TARGET" -j "$JOBS" || {
    echo "error: build 失敗（指紋未更新 —— 一份寫著成功但其實沒建的東西比沒有更糟）" >&2
    exit 1
}

fp="$(fingerprint)" || exit $?
"$PY" - "$OUT" "$fp" "$(git_head)" "$TARGET" "$JOBS" "$BUILD_DIR" <<'PY'
import datetime, json, os, sys
out, fp, head, target, jobs, build_dir = sys.argv[1:7]
doc = {
    "fingerprint": json.loads(fp),
    "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "git_head": head,
    "target": target,
    "jobs": int(jobs),
    "build_dir": build_dir,
    # 哪一支產生的指紋。沒有這一欄，「這批數字屬於哪批二進位檔」就無法被追溯到規則的版本。
    "fingerprint_source": "scripts/check/decode_sweep.py:build_fingerprint()",
}
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
    fh.write("\n")
print(f"  wrote {out}")
PY
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("  fingerprint:", len(d["fingerprint"]), "keys @", d["recorded_at"], "HEAD", d["git_head"][:9])' "$OUT"
