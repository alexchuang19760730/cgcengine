#!/bin/bash
# runners/preflight.sh — 「現在這台機器能不能安全地起一個引擎？」(PLAN §3)
#
#   preflight.sh                 # 只報告，不留痕跡；有殘留或記憶體不足則 exit 1
#   preflight.sh --json          # 同一件事，機器可讀
#   preflight.sh --kill          # 先 TERM 再用 SIGKILL 收掉**驗證過的**真 llama 行程
#   preflight.sh --want-ctx N    # 檢查伺服器能否容納 N 個 token 的 prompt（預設 $ENGINE_MIN_CTX）
#
# 為什麼「殘留判定」這件事值得一支獨立腳本，見 CONVENTIONS.md A16/B8 與 lesson `eng-mh-0007`
# 那一族的形狀：**一個行程的身份是它的可執行檔，不是它的命令列文字。**
#
#   `pgrep -f "build/bin/llama-server"` 比對的是**完整命令列**，所以任何提到那個路徑的行程
#   （`bash -c` wrapper、grep、正在做診斷的人自己的指令）都會被算成「殘留的 llama-server」。
#   2026-09-17 這一天，這個錯誤在三個地方各自造成一次事故：
#     - `run_server.sh` 的 `cgc_existing_llama_server_count` 把它餵給 memory guard，
#       而門檻是 0 ⇒ 多數到 1 就 `startup blocked by memory guard` + exit 1（**靜默擋掉一整臂**）
#     - `Backup/run_req2_retest.sh` 的 `pkill -9 -f` 會把 wrapper 直接 SIGKILL，
#       而等待迴圈被那種行程拖滿 90 秒
#     - 一支診斷指令本身被某個無差別的 `pkill -f` 殺掉（它的命令列裡有那個字串）
#   而 `scripts/check/e2e/issue_coverage.py:226` **早就把它記成 issue 了**。
#
# 所以這裡的判定是兩段式的：`pgrep -f` 只負責**蒐集候選**，`ps -o comm=` 的 basename 才是身分。
# 原始碼層面的對應物是 `run_server.sh` 的 `cgc_preflight_pids`（同一條判準，同一句話）。
#
# 這支腳本**預設不殺任何東西**。`--kill` 是顯式的，而且只殺通過上面那個判定的行程。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# self-anchoring：從自己的位置往上找 config.env，不依賴 caller 的 CWD
CFG="$(cd "$HERE/.." && pwd)/config.env"
# shellcheck source=../config.env
[ -r "$CFG" ] && . "$CFG"
: "${ENGINE_REPO_ROOT:=$(cd "$HERE/../.." && pwd)}"
: "${ENGINE_BIN:=$ENGINE_REPO_ROOT/src/llama.cpp/build/bin}"
: "${ENGINE_MIN_CTX:=0}"

WANT_CTX="$ENGINE_MIN_CTX"
JSON=0
KILL=0
SELFTEST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --json) JSON=1 ;;
        --kill) KILL=1 ;;
        --self-test) SELFTEST=1 ;;
        --want-ctx) WANT_CTX="$2"; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# 行程身分：候選由 pgrep 蒐集，**身分由 ps 決定**
# ---------------------------------------------------------------------------
# 這份名字清單與 run_server.sh 的 CGC_PREFLIGHT_NAMES 是同一條判準。刻意各自成一份而不是
# 互相 source：那支是生產啟動器（會在被 source 時執行很多東西），這一支必須能在任何時候
# 單獨跑而不碰任何狀態。兩份的**判準**是同一句，而判準是可被測的（見 --self-test）。
RESID_NAMES="${ENGINE_RESID_NAMES:-llama-server llama-cli llama-bench llama-simple llama-perplexity}"

# 蒐集候選：用一個刻意寬鬆的 pattern 抓所有可能相關的行程
_candidates() {
    for n in $RESID_NAMES; do
        pgrep -f "$n" 2>/dev/null || true
    done | sort -u
}

# 身分判定：basename(ps -o comm=) 必須**確切等於**一個已知的 llama 可執行檔名
_exe_of() {
    ps -o comm= -p "$1" 2>/dev/null | head -1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

_is_engine() {
    local base; base="$(basename "$(_exe_of "$1")")"
    for n in $RESID_NAMES; do [ "$base" = "$n" ] && return 0; done
    return 1
}

resid_pids() {
    local pid
    for pid in $(_candidates); do
        [ "$pid" = "$$" ] && continue
        _is_engine "$pid" && echo "$pid"
    done
}

# 對照組用的證據：每一筆候選都要說明它「算」或「不算」以及為什麼。
# 這一段是這支腳本唯一真正重要的輸出 —— 沒有它，「0 殘留」與「判定根本沒在跑」同形。
candidates_report() {
    local pid exe base
    for pid in $(_candidates); do
        [ "$pid" = "$$" ] && continue
        exe="$(_exe_of "$pid")"; base="$(basename "$exe")"
        if _is_engine "$pid"; then
            printf '    pid=%s  exe=%s  -> RESID\n' "$pid" "$exe"
        else
            printf '    pid=%s  exe=%s  -> not-an-engine（命令列提到名字，可執行檔不是）\n' "$pid" "$exe"
        fi
    done
}

# ---------------------------------------------------------------------------
# 記憶體
# ---------------------------------------------------------------------------
# 這一節只**報告**，不設門檻：門檻的權威在 run_server.sh 的 cgc_memory_guard_req()，而那裡
# 的門檻依 profile 而異（"0 35 0" / "0 40 0" / …）。在這裡再定一組就是第二份真相。
# 解析後放進變數而不是印出來：JSON 與文字兩種輸出都要用同一組數字，
# 而「同一件事算兩次」是兩個數字開始不一致的標準開頭。
MEM_FREE_MB=0 MEM_TOTAL_MB=0 MEM_FREE_PCT=0 MEM_SWAP_USED_MB=0 MEM_SWAP_TOTAL_MB=0
mem_measure() {
    local free_pages inactive page_size
    page_size="$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)"
    free_pages="$(vm_stat | awk '/Pages free/{gsub(/\./,"",$3); print $3}')"
    inactive="$(vm_stat   | awk '/Pages inactive/{gsub(/\./,"",$3); print $3}')"
    : "${free_pages:=0}"; : "${inactive:=0}"
    MEM_FREE_MB=$(( ( free_pages + inactive ) * page_size / 1048576 ))
    MEM_TOTAL_MB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))
    MEM_FREE_PCT=$(( MEM_FREE_MB * 100 / ( MEM_TOTAL_MB > 0 ? MEM_TOTAL_MB : 1 ) ))
    local su st
    su="$(sysctl -n vm.swapusage 2>/dev/null | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p')"
    st="$(sysctl -n vm.swapusage 2>/dev/null | sed -n 's/.*total = \([0-9.]*\)M.*/\1/p')"
    MEM_SWAP_USED_MB="${su%.*}";  : "${MEM_SWAP_USED_MB:=0}"
    MEM_SWAP_TOTAL_MB="${st%.*}"; : "${MEM_SWAP_TOTAL_MB:=0}"
}

# ---------------------------------------------------------------------------
# 埠
# ---------------------------------------------------------------------------
port_pids() {
    local port="${CGC_SERVER_PORT:-8080}"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | tail -n +2 | awk '{print $2}' | sort -u
    fi
}

# ---------------------------------------------------------------------------
# 建置閘門：這一節回答「現在跑 cmake --build 會不會蓋掉別人正在用的二進位檔」
# ---------------------------------------------------------------------------
# 為什麼這是一條**閘門**而不是一句提醒：2026-09-17 我把 `lsof` 與 `cmake --build` 寫在同一行，
# `lsof` 明明印出別條線的 server 在聽，build 還是跑了 ⇒ 蓋掉他們 11:28 起跑那輪正在用的
# `libggml-metal`／`libllama`，於是那一輪的 A/B **橫跨兩個 build**（而這個 repo 的所有比較
# 都建立在「同一個 build」這個前提上）。印出來不等於擋下來。
builders() {
    pgrep -f '[c]make|[n]inja' 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# --self-test：證明這個判定**兩個方向都會答**
# ---------------------------------------------------------------------------
# 判定的核心主張是「候選由命令列蒐集，身分由可執行檔決定」。這主張有兩個方向，而只驗一個方向
# 的檢查不算檢查（lesson `eng-gate-0040`）：
#   ① 命令列提到名字、但可執行檔不是      -> 必須**不**算
#   ② 可執行檔就是那個名字                -> 必須算
# 沒有 ①，一個「永遠回答有待清理」的實作也會通過；沒有 ②，一個「永遠回答乾淨」的也會。
#
# ★ 測試用**合成的名字**（覆寫 ENGINE_RESID_NAMES），不是 llama-server：
#   ② 會製造一個 argv[0] 真的叫 llama-server 的行程，而另一條線的啟動守門會在同時段把它
#   算成「殘留的 server」而擋掉自己（門檻是 0）。**為了驗自己的檢查而讓別人的臂 abort
#   不算驗證。** 合成名字走完全相同的程式路徑，卻不與任何真實服務同名。
self_test() {
    local fails=0
    local fake="cgc-selftest-engine.$$"
    echo "== preflight --self-test（合成名字 ${fake}，不碰任何真實服務名）=="
    local wpid rpid
    bash -c "echo 'build/bin/$fake --fake' >/dev/null; sleep 8" &
    wpid=$!
    ( exec -a "$fake" sleep 8 ) &
    rpid=$!
    sleep 1

    local cand resid
    # RESID_NAMES 是在載入時從 ENGINE_RESID_NAMES 定案的，所以這裡必須**直接**覆寫它，
    # 而不是再設一次 ENGINE_RESID_NAMES（那不會被重新推導 —— 第一次寫的時候就是這樣錯的，
    # 症狀是 candidates=0：覆寫看起來生效了、其實整條判定用的是預設清單）。
    cand="$( RESID_NAMES="$fake"; _candidates | sort -u )"
    resid="$( RESID_NAMES="$fake"; resid_pids | sort -u )"
    local n_cand n_resid
    n_cand="$(printf '%s\n' "$cand" | grep -c '[0-9]' || true)"
    n_resid="$(printf '%s\n' "$resid" | grep -c '[0-9]' || true)"

    check() { if [ "$2" = 1 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; fails=$((fails+1)); fi; }

    check "① 命令列提到名字的 wrapper 被蒐集為候選（pgrep 確實看得到它）" \
          "$(printf '%s\n' "$cand" | grep -qx "$wpid" && echo 1 || echo 0)"
    check "② 可執行檔就是那個名字的行程被蒐集為候選" \
          "$(printf '%s\n' "$cand" | grep -qx "$rpid" && echo 1 || echo 0)"
    check "③ ① 被排除（不是引擎）"  "$(printf '%s\n' "$resid" | grep -qx "$wpid" && echo 0 || echo 1)"
    check "④ ② 被認出（是引擎）"    "$(printf '%s\n' "$resid" | grep -qx "$rpid" && echo 1 || echo 0)"
    check "⑤ 候選 2 個、殘留 1 個（不是全收也不是全丟）" \
          "$([ "$n_cand" -ge 2 ] && [ "$n_resid" = 1 ] && echo 1 || echo 0)"
    echo "      證據: candidates=$n_cand resid=$n_resid"
    echo "      $(printf '%s' "$(_exe_of "$wpid")" | sed 's/^/wrapper exe: /')"
    echo "      $(printf '%s' "$(_exe_of "$rpid")" | sed 's/^/real    exe: /')"

    kill -TERM "$wpid" "$rpid" 2>/dev/null
    wait "$wpid" "$rpid" 2>/dev/null
    if [ "$fails" = 0 ]; then echo "  --self-test: $((5-fails))/5 通過"; return 0; fi
    echo "  --self-test: $fails 項失敗" >&2
    return 1
}

if [ "$SELFTEST" = 1 ]; then
    self_test; exit $?
fi

# ---------------------------------------------------------------------------
# 輸出
# ---------------------------------------------------------------------------
PORT_PIDS="$(port_pids)"
RESID="$(resid_pids)"
BUILDERS="$(builders)"

n_resid=$(printf '%s\n' "$RESID"    | grep -c '[0-9]' || true)
n_port=$(printf '%s\n' "$PORT_PIDS" | grep -c '[0-9]' || true)
n_build=$(printf '%s\n' "$BUILDERS" | grep -c '[0-9]' || true)

if [ "$KILL" = 1 ] && [ "$n_resid" -gt 0 ]; then
    echo "==> --kill: 只收掉通過身分判定的行程"
    for pid in $RESID; do echo "    TERM $pid  $(_exe_of "$pid")"; kill -TERM "$pid" 2>/dev/null || true; done
    for _ in $(seq 1 90); do [ "$(resid_pids | grep -c '[0-9]' || true)" = 0 ] && break; sleep 1; done
    for pid in $(resid_pids); do echo "    KILL $pid  $(_exe_of "$pid")"; kill -9 "$pid" 2>/dev/null || true; done
    sleep 2
    RESID="$(resid_pids)"; n_resid=$(printf '%s\n' "$RESID" | grep -c '[0-9]' || true)
fi

mem_measure

verdict=ok
[ "$n_resid" -gt 0 ] && verdict=residual
[ "$n_port"  -gt 0 ] && verdict=port-busy

if [ "$JSON" = 1 ]; then
    local_pids() { printf '%s' "$1" | tr ' ' ',' | sed 's/^,*//;s/,*$//'; }
    printf '{"verdict":"%s","residual_pids":[%s],"port_pids":[%s],"builders":[%s],' \
        "$verdict" "$(local_pids "$RESID")" "$(local_pids "$PORT_PIDS")" "$(local_pids "$BUILDERS")"
    printf '"free_mb":%s,"total_mb":%s,"free_pct":%s,"swap_used_mb":%s,"swap_total_mb":%s,"want_ctx":%s}\n' \
        "$MEM_FREE_MB" "$MEM_TOTAL_MB" "$MEM_FREE_PCT" "$MEM_SWAP_USED_MB" "$MEM_SWAP_TOTAL_MB" "${WANT_CTX:-0}"
else
    echo "== engine loop preflight =="
    echo "  repo        : $ENGINE_REPO_ROOT"
    echo "  engine bin  : $ENGINE_BIN"
    echo
    echo "  [1] 引擎行程（判定 = basename(ps -o comm=)）"
    if [ "$n_resid" -gt 0 ]; then
        for pid in $RESID; do printf '      RESID pid=%s  %s\n' "$pid" "$(_exe_of "$pid")"; done
    else
        echo "      無"
    fi
    local_n=$(candidates_report | grep -c . || true)
    if [ "${local_n:-0}" -gt 0 ]; then
        echo "      候選逐筆（包含「不算」的那些 —— 沒有這一段，0 與「判定沒在跑」同形）:"
        candidates_report | sed 's/^/    /'
    else
        echo "      候選逐筆: 無候選（連命令列提到名字的都沒有）"
    fi
    echo
    echo "  [2] 記憶體"
    printf '      free=%s MB / total=%s MB (%s%%)\n' "$MEM_FREE_MB" "$MEM_TOTAL_MB" "$MEM_FREE_PCT"
    printf '      swap=%s MB / %s MB\n' "$MEM_SWAP_USED_MB" "$MEM_SWAP_TOTAL_MB"
    echo "      （門檻的權威在 run_server.sh 的 cgc_memory_guard_req()，這裡不重述）"
    echo
    echo "  [3] port ${CGC_SERVER_PORT:-8080}"
    if [ "$n_port" -gt 0 ]; then
        lsof -nP -iTCP:"${CGC_SERVER_PORT:-8080}" -sTCP:LISTEN 2>/dev/null | sed 's/^/      /'
    else
        echo "      空"
    fi
    echo
    echo "  [4] 建置"
    if [ "$n_build" -gt 0 ]; then
        echo "      ⚠ 有建置在跑（${n_build}）⇒ rebuild.sh 會 abort"
    else
        echo "      無建置在跑"
    fi
    echo
    echo "  [5] prompt 容量"
    if [ "${WANT_CTX:-0}" -gt 0 ]; then
        echo "      需要 >= $WANT_CTX tokens 的 prompt；啟動時要覆寫 CGC_SERVER_CTX"
        echo "      （run_server.sh 預設 4096/8192 —— 30.6k token 的閉環 prompt 裝不下，"
        echo "        而截斷後的回覆仍然「看起來像回事」）"
    else
        echo "      未設 ENGINE_MIN_CTX ⇒ 不檢查"
    fi
    echo
    echo "  verdict: $verdict"
fi

[ "$verdict" = ok ] || exit 1
exit 0
