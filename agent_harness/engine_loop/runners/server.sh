#!/bin/bash
# runners/server.sh — engine loop 的啟動口 (PLAN §3)
#
#   server.sh --check        # 不啟動：檢查前置是否存在、印出**將要執行的完整命令**
#   server.sh --dry-run      # 同上，但另外跑一次 preflight（前置 + 殘留 + 埠）
#   server.sh                # 經閘門後 exec scripts/run_server.sh
#
# 這支腳本刻意**很薄**，因為 PLAN §3 的紅線是「`scripts/run_server.sh` 是生產腳本，不得複製
# 進 engine_loop」。所以這裡沒有一行 llama-server 的參數、沒有一個 profile 的定義、
# 沒有一個模型的預設值 —— 那些全部留在權威那裡，這裡只負責：
#
#   ① 把 engine loop 需要的東西**明確化**（ctx 夠不夠裝 30.6k token 的閉環 prompt）；
#   ② 在前置不存在時**在啟動前**失敗（而不是在 server 起來一半時）；
#   ③ 把「我即將用哪一批二進位檔」印出來（接 runners/rebuild.sh 的 build.json）。
#
# 為什麼 ① 值得一支腳本：`run_server.sh` 的 ctx 預設是 4096/8192，而 `distill/closed_loop.py`
# 的四臂 prompt 實測是 **30,594 tokens**（2026-09-17 `/tokenize`）。ctx 不夠時請求會被截斷，
# **而回答仍然看起來像回事** —— 這正是「一個數字什麼時候可以引用」那一族問題的入口。
# 所以這一格是硬檢查，不是提醒。
#
# profile 的**合法值**不在這裡驗：`scripts/run_server.sh` 自己的 `case` 分支就會對未知值報錯
# 並結束。抄一份合法值清單來這裡，就是製造第二份真相（而第二份的失效方式是安靜的：
# 權威新增 profile 後這裡仍然「驗得過」，只是驗的是舊規則）。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$(cd "$HERE/.." && pwd)/config.env"
# shellcheck source=../config.env
[ -r "$CFG" ] && . "$CFG"
: "${ENGINE_REPO_ROOT:=$(cd "$HERE/../.." && pwd)}"
: "${ENGINE_SERVER_SH:=$ENGINE_REPO_ROOT/scripts/run_server.sh}"
: "${ENGINE_SERVER:=$ENGINE_REPO_ROOT/src/llama.cpp/build/bin/llama-server}"
: "${ENGINE_MODEL_ROOT:=$ENGINE_REPO_ROOT/models/gguf}"
: "${ENGINE_MIN_CTX:=40960}"
: "${ENGINE_LOOP_DIR:=$ENGINE_REPO_ROOT/agent_harness/engine_loop}"

PROFILE="${ENGINE_PROFILE:-}"
RUNTIME_PROFILE="${ENGINE_RUNTIME_PROFILE:-}"
CTX="${CGC_SERVER_CTX:-}"
PORT="${CGC_SERVER_PORT:-8080}"
MODEL="${ENGINE_MODEL:-}"
DETACH=0
CHECK=0
DRY=0
FORCE=0
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --profile)         PROFILE="$2"; shift ;;
        --runtime-profile) RUNTIME_PROFILE="$2"; shift ;;
        --ctx)             CTX="$2"; shift ;;
        --port)            PORT="$2"; shift ;;
        --model)           MODEL="$2"; shift ;;
        --detach)          DETACH=1 ;;
        --check)           CHECK=1 ;;
        --dry-run)         DRY=1 ;;
        --force)           FORCE=1 ;;
        --) shift; EXTRA+=("$@"); break ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) EXTRA+=("$1") ;;   # 其餘原樣轉給 run_server.sh（它才是參數的權威）
    esac
    shift
done

fails=0
note() { printf '  %-6s %s\n' "$1" "$2"; }
ok()   { note "ok"   "$1"; }
bad()  { note "FAIL" "$1"; fails=$((fails+1)); }
warn() { note "warn" "$1"; }

if [ "$CHECK" = 1 ] || [ "$DRY" = 1 ]; then
    echo "== server --check =="
fi

# ---------------------------------------------------------------------------
# 前置存在性
# ---------------------------------------------------------------------------
[ -x "$ENGINE_SERVER_SH" ] && ok "權威啟動器 $ENGINE_SERVER_SH" \
                           || bad "權威啟動器不存在或不可執行：$ENGINE_SERVER_SH"
[ -x "$ENGINE_SERVER" ]    && ok "引擎二進位檔 $ENGINE_SERVER" \
                           || bad "引擎二進位檔不存在（先跑 runners/rebuild.sh）：$ENGINE_SERVER"

shopt -s nullglob
ggufs=("$ENGINE_MODEL_ROOT"/*.gguf)
shopt -u nullglob
if [ "${#ggufs[@]}" -gt 0 ]; then
    ok "模型目錄有 ${#ggufs[@]} 個 .gguf（哪一個由 run_server.sh 依 runtime profile 決定）"
else
    bad "模型目錄裡沒有 .gguf：$ENGINE_MODEL_ROOT"
fi
if [ -n "$MODEL" ]; then
    [ -f "$MODEL" ] && ok "指定的模型存在：$MODEL" || bad "指定的模型不存在：$MODEL"
fi

# ---------------------------------------------------------------------------
# ctx：這一格是硬檢查
# ---------------------------------------------------------------------------
if [ -z "$CTX" ]; then
    bad "ctx 未設定 —— 閉環 prompt 是 ${ENGINE_MIN_CTX:-0}+ tokens，run_server.sh 的預設 4096/8192 裝不下"
    echo "         （截斷後的回覆仍然看起來像回事 ⇒ 這不是可以讓它過去的預設）"
    echo "         用法：CGC_SERVER_CTX=${ENGINE_MIN_CTX:-40960} $0"
elif [ "$CTX" -lt "$ENGINE_MIN_CTX" ]; then
    bad "ctx=$CTX < 需要的 $ENGINE_MIN_CTX"
else
    ok "ctx=${CTX}（>= ${ENGINE_MIN_CTX}）"
fi

# ---------------------------------------------------------------------------
# 這是哪一批二進位檔
# ---------------------------------------------------------------------------
if [ -f "$ENGINE_LOOP_DIR/build.json" ]; then
    python3 - "$ENGINE_LOOP_DIR/build.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
fp = d.get("fingerprint", {})
print(f"  ok     build.json: {len(fp)} 鍵 @ {d.get('recorded_at')}  HEAD {str(d.get('git_head'))[:9]}")
print(f"         server={fp.get('server')}  libggml-metal={fp.get('libggml-metal')}")
PY
else
    warn "沒有 $ENGINE_LOOP_DIR/build.json ⇒ 這批二進位檔的身分不明（跑 runners/rebuild.sh 產生）"
fi

# ---------------------------------------------------------------------------
# 埠與殘留（不啟動也要能回答）
# ---------------------------------------------------------------------------
if command -v lsof >/dev/null 2>&1; then
    busy="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tail -n +2)"
    [ -n "$busy" ] && bad "port $PORT 已被佔用" || ok "port $PORT 空"
fi

# ---------------------------------------------------------------------------
# 將要執行的命令（不是它「大概會用到」的參數，而是我們實際傳出去的那些）
# ---------------------------------------------------------------------------
export CGC_SERVER_CTX="$CTX"
[ -n "$RUNTIME_PROFILE" ] && export CGC_SERVER_RUNTIME_PROFILE="$RUNTIME_PROFILE"
[ -n "$PROFILE" ]         && export CGC_SERVER_PROFILE="$PROFILE"
[ -n "$MODEL" ]           && export CGC_SERVER_MODEL="$MODEL"
export CGC_SERVER_PORT="$PORT"

show_cmd() {
    # ★ 不能用 ${DETACH:+ --detach}：`DETACH=0` 是**非空字串**，所以 `:+` 會展開。
    #   第一次寫的時候就是這樣錯的，症狀是 `--check` 明明沒給 --detach 卻印出它。
    local detach_flag=""
    [ "$DETACH" = 1 ] && detach_flag=" --detach"
    local extra_str=""
    [ "${#EXTRA[@]}" -gt 0 ] && extra_str=" ${EXTRA[*]}"
    echo "  將執行："
    printf '    CGC_SERVER_CTX=%s \\\n' "$CTX"
    [ -n "$RUNTIME_PROFILE" ] && printf '    CGC_SERVER_RUNTIME_PROFILE=%s \\\n' "$RUNTIME_PROFILE"
    [ -n "$PROFILE" ]         && printf '    CGC_SERVER_PROFILE=%s \\\n' "$PROFILE"
    [ -n "$MODEL" ]           && printf '    CGC_SERVER_MODEL=%s \\\n' "$MODEL"
    printf '    CGC_SERVER_PORT=%s \\\n' "$PORT"
    printf '    %s%s%s\n' "$ENGINE_SERVER_SH" "$detach_flag" "$extra_str"
    echo "  （profile 的合法值由 run_server.sh 自己驗；engine_loop 不持有第二份清單）"
}

echo
if [ "$CHECK" = 1 ] || [ "$DRY" = 1 ]; then
    show_cmd
    echo
fi

if [ "$fails" -gt 0 ]; then
    echo "  ${fails} 項前置不成立 ⇒ 不啟動" >&2
    [ "$FORCE" = 1 ] || exit 1
    echo "  ⚠ --force：仍然啟動（你要為此負責）" >&2
fi

if [ "$CHECK" = 1 ]; then exit 0; fi

if [ "$DRY" = 1 ]; then
    echo "== preflight（--dry-run 才跑）=="
    "$HERE/preflight.sh" || true
    echo "  （dry-run：沒有啟動任何東西）"
    exit 0
fi

# ---------------------------------------------------------------------------
# 交給權威
# ---------------------------------------------------------------------------
echo "==> exec $ENGINE_SERVER_SH${DETACH:+ --detach}"
if [ "$DETACH" = 1 ]; then
    exec "$ENGINE_SERVER_SH" --detach "${EXTRA[@]+"${EXTRA[@]}"}"
else
    exec "$ENGINE_SERVER_SH" "${EXTRA[@]+"${EXTRA[@]}"}"
fi
