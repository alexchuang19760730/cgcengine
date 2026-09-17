#!/bin/bash
# wrappers/_common.sh — 五個 wrapper 共用的那一層 (PLAN §3)
#
# PLAN §3 對 wrappers/ 的規定是**負面**的：「只放『指向 scripts/check/* 的呼叫 + 輸出正規化』，
# 不放邏輯」。這一支就是把那句話實作出來的地方，所以它只做四件事：
#
#   ① 從 classes.tsv 讀出「這一類有哪些腳本」（分類的唯一來源是 wrappers/classify.py）
#   ② 前置：每個腳本都要真的存在，否則**在跑之前**失敗並點名
#   ③ 執行：把 stdout/stderr 落成一份 log，並記下「這是哪一批二進位檔跑的」
#   ④ 不留第二份真相：**不**把輸出轉成 episode record —— 那是 `traces/emit_episodes.py`
#      的工作（PLAN §5 的 T0）。wrapper 若自己再轉一次，同一個 episode 就會有兩個生產者，
#      而兩個生產者的輸出不會一致（它們會各自演化），於是「哪一份是對的」變成一個沒人能答的問題。
#
# ★ 「輸出正規化」在這裡的意思是**讓一次 run 可被追溯**，不是轉換資料格式：
#   一次 run 的產物是 {哪個腳本、什麼參數、什麼時候、哪一批二進位檔、rc、log 在哪}。
#   少了最後兩項中的任何一項，那個 run 產生的數字就不能被引用（CONVENTIONS.md 的 M 系列）。
#
# 用法（在 wrapper 裡）：
#     . "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#     w_class sweep
#     w_main "$@"

set -uo pipefail

W_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W_CONFIG="$(cd "$W_DIR/.." && pwd)/config.env"
# shellcheck source=../config.env
[ -r "$W_CONFIG" ] && . "$W_CONFIG"
: "${ENGINE_REPO_ROOT:=$(cd "$W_DIR/../../.." && pwd)}"
: "${ENGINE_CHECK_DIR:=$ENGINE_REPO_ROOT/scripts/check}"
: "${ENGINE_LOOP_DIR:=$(cd "$W_DIR/.." && pwd)}"

W_CLASSES="$W_DIR/classes.tsv"
W_RUNS_DIR="${ENGINE_RUNS_DIR:-$ENGINE_LOOP_DIR/runs}"
W_BUILD_JSON="${ENGINE_BUILD_JSON:-$ENGINE_LOOP_DIR/build.json}"

W_CLASS=""

w_class() {
    W_CLASS="$1"
    if ! grep -q "^${W_CLASS}"$'\t' "$W_CLASSES" 2>/dev/null; then
        echo "error: 未知的能力類 '${W_CLASS}'（classes.tsv 裡沒有）" >&2
        echo "       已知：$(w_all_classes | tr '\n' ' ')" >&2
        exit 2
    fi
}

w_all_classes() { grep -v '^#' "$W_CLASSES" | cut -f1 | sort -u; }

# (腳本名, MANIFEST role, 說明) —— 這是這一類的全部成員
w_members() { grep "^${W_CLASS}"$'\t' "$W_CLASSES" | cut -f2,3,4; }

w_scripts() { grep "^${W_CLASS}"$'\t' "$W_CLASSES" | cut -f2; }

w_script_path() { echo "$ENGINE_CHECK_DIR/$1"; }

# ---------------------------------------------------------------------------
# 用哪個直譯器呼叫
# ---------------------------------------------------------------------------
# ★ 這一格是被自己的自測抓出來的，不是先想到的。原本寫 `exec "$path" "$@"`，
#   而實測 `scripts/check/` 的 44 支裡有 **30 支沒有可執行位**（`.py` 幾乎都沒有）
#   ⇒ 那條路對大多數成員直接 EACCES。自測的第 3 項（「所有成員都可執行」）就是這樣變紅的。
#
# 修法不是去 chmod（那是改動別條線的生產腳本、而且會被下一次 checkout 還原），
# 而是**顯式指定直譯器** —— 這也正是 repo 自己的慣例：
# `scripts/check/ab_interleave.py:66` 呼叫 decode_bench 時用的是
# `[sys.executable, "scripts/check/decode_bench.py", ...]`。
#
# 讀 shebang 而不是只看副檔名，是為了尊重 `#!/usr/bin/env python3.13` 這類差異；
# 讀不到才退回副檔名。`env` 形式要取第二個欄位（第一個只是 `/usr/bin/env`）。
w_interp() {
    local p="$1" first rest
    first="$(head -1 "$p" 2>/dev/null || true)"
    if [ "${first#\#!}" != "$first" ]; then
        rest="${first#\#!}"; rest="${rest# }"
        case "$rest" in
            */env\ *) rest="${rest#*/env }"; echo "${rest%% *}"; return 0 ;;
            *)        echo "${rest%% *}"; return 0 ;;
        esac
    fi
    case "$p" in *.py) echo "${PY:-python3}" ;; *) echo bash ;; esac
}

# ---------------------------------------------------------------------------
# 前置存在性：**在跑之前**失敗並點名，而不是在中途
# ---------------------------------------------------------------------------
w_precheck() {
    local missing=0 n=0
    local s
    for s in $(w_scripts); do
        n=$((n + 1))
        [ -f "$(w_script_path "$s")" ] || { echo "  MISSING  scripts/check/$s" >&2; missing=$((missing + 1)); }
    done
    if [ "$n" = 0 ]; then
        echo "error: 類 '$W_CLASS' 在 classes.tsv 裡沒有任何成員 —— 這不是「沒事做」，是分類表壞了" >&2
        exit 1
    fi
    if [ "$missing" -gt 0 ]; then
        echo "error: ${missing}／${n} 個生產腳本不存在 ⇒ 不執行任何東西" >&2
        echo "       （分類表與磁碟漂移：跑 wrappers/classify.py --check）" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 這一批二進位檔的身分。**沒有它就沒有可引用的讀數**，所以預設是拒絕而不是警告。
# ---------------------------------------------------------------------------
w_fingerprint() {
    [ -f "$W_BUILD_JSON" ] || {
        echo "error: $W_BUILD_JSON 不存在 ⇒ 這一類的讀數將無法附帶 build 指紋，因此不可引用。" >&2
        echo "       先跑 runners/rebuild.sh（它產 build.json）。" >&2
        echo "       要明知故犯：ENGINE_ALLOW_NO_FINGERPRINT=1" >&2
        [ "${ENGINE_ALLOW_NO_FINGERPRINT:-0}" = 1 ] || exit 1
        echo "(no-fingerprint)" >&2
        printf '{"keys":0,"sha":"none"}'
        return 0
    }
    python3 - "$W_BUILD_JSON" <<'PY'
import hashlib, json, sys
d = json.load(open(sys.argv[1]))
fp = d.get("fingerprint", {})
blob = json.dumps(fp, sort_keys=True, ensure_ascii=False)
print('{"keys":%d,"sha":"%s"}' % (len(fp), hashlib.sha256(blob.encode()).hexdigest()[:16]))
PY
}

# ---------------------------------------------------------------------------
# 執行 + 記錄
# ---------------------------------------------------------------------------
w_run() {
    local script="$1"; shift
    local path; path="$(w_script_path "$script")"
    local ts; ts="$(date +%Y%m%d_%H%M%S)"
    local slug; slug="$(echo "$script" | tr './' '__')"
    local run_dir="$W_RUNS_DIR/${ts}_${W_CLASS}_${slug}"
    mkdir -p "$run_dir"
    local log="$run_dir/run.log"
    local fp; fp="$(w_fingerprint)"
    local fp_sha; fp_sha="$(printf '%s' "$fp" | sed -n 's/.*"sha":"\([^"]*\)".*/\1/p')"
    local fp_keys; fp_keys="$(printf '%s' "$fp" | sed -n 's/.*"keys":\([0-9]*\).*/\1/p')"

    {
        echo "# wrapper   : $W_CLASS"
        echo "# class file: class=$W_CLASS script=$script role=$(grep "^${W_CLASS}"$'\t'"$script"$'\t' "$W_CLASSES" | cut -f3)"
        echo "# argv      : $(w_interp "$path") scripts/check/$script $*"
        echo "# cwd       : $ENGINE_REPO_ROOT"
        echo "# started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# git HEAD  : $(git -C "$ENGINE_REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "# build fp  : $fp_keys keys  sha256[:16]=$fp_sha"
        echo "# ---------- output ----------"
    } > "$log"

    local t0; t0="$(date +%s)"
    local rc=0
    ( cd "$ENGINE_REPO_ROOT" && exec "$(w_interp "$path")" "$path" "$@" ) >> "$log" 2>&1 || rc=$?
    local t1; t1="$(date +%s)"

    local rec="$run_dir/run.json"
    python3 - "$rec" "$W_CLASS" "$script" "$rc" "$((t1 - t0))" "$log" "$fp_sha" "$fp_keys" "$*" "$ENGINE_REPO_ROOT" <<'PY'
import datetime, json, os, subprocess, sys
rec, cls, script, rc, secs, log, fp_sha, fp_keys, argv, repo = sys.argv[1:12]
head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
doc = {
    "wrapper": cls,
    "script": script,
    "argv": ["scripts/check/" + script] + (argv.split() if argv else []),
    "rc": int(rc),
    "seconds": int(secs),
    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "git_head": head or "unknown",
    "build_fingerprint_keys": int(fp_keys),
    "build_fingerprint_sha16": fp_sha,
    "log": os.path.relpath(log, repo),
    # 明確**不是** trace record：把它轉成 episode 是 traces/emit_episodes.py 的事（PLAN §5 T0）。
    # 記在這裡是為了讓「同一個 episode 有兩個生產者」這件事不會發生。
    "not_a_record": "this is a run manifest; traces/emit_episodes.py owns record production",
}
with open(rec, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
    fh.write("\n")
print(f"  rc={rc}  {secs}s  log={os.path.relpath(log, repo)}")
PY
    return "$rc"
}

# ---------------------------------------------------------------------------
# 共用的 --list / --dry-run / --self-test
# ---------------------------------------------------------------------------
w_list() {
    echo "  class: $W_CLASS"
    awk -F'\t' -v c="$W_CLASS" '$1==c {printf "    %-38s role=%-10s %s\n", $2, $3, $4}' "$W_CLASSES"
}

w_selftest() {
    local fails=0
    chk() { if [ "$2" = 1 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; fails=$((fails + 1)); fi; }
    echo "== $W_CLASS --self-test =="
    local n; n="$(w_scripts | grep -c . || true)"
    chk "classes.tsv 有這個類且非空（${n} 個成員）" "$([ "$n" -gt 0 ] && echo 1 || echo 0)"
    # ★ 這裡原本寫 `w_precheck >/dev/null 2>&1; chk ... "$([ $? = 0 ] && ...)"` —— 而 `$?`
    #   在那個位置讀到的是**前一個 chk 呼叫**的狀態，不是 w_precheck 的。於是這一項永遠通過。
    #   一個因為錯誤的原因而通過的斷言，比一個紅的斷言更糟（lesson eng-gate-0040 的同一形狀）。
    local pc=1; w_precheck >/dev/null 2>&1 && pc=0
    chk "所有成員都真的存在（w_precheck）" "$([ "$pc" = 0 ] && echo 1 || echo 0)"
    # 呼叫靠的是直譯器，不是可執行位：實測 44 支裡 30 支沒有 +x（見 w_interp 的註解）。
    # 所以這一項驗「直譯器解析得到」，並把沒有 +x 的數量印出來當可見的事實。
    local bad=0 nox=0 s interp
    for s in $(w_scripts); do
        [ -x "$(w_script_path "$s")" ] || nox=$((nox + 1))
        interp="$(w_interp "$(w_script_path "$s")")"
        command -v "$interp" >/dev/null 2>&1 || bad=$((bad + 1))
    done
    chk "所有成員的直譯器都解析得到（沒有 +x 的有 ${nox}／${n}，那不影響）" \
        "$([ "$bad" = 0 ] && echo 1 || echo 0)"
    # 分類表的**互斥性**由 classify.py 保證；這裡只驗「這一類不是空的、也不是全部」
    local total; total="$(grep -v '^#' "$W_CLASSES" | grep -c . || true)"
    chk "這一類不是全部（${n}／${total}）—— 一個包含所有東西的類別不叫分類" "$([ "$n" -lt "$total" ] && echo 1 || echo 0)"
    if [ "$fails" = 0 ]; then echo "  --self-test: 4/4 通過"; return 0; fi
    echo "  --self-test: $fails 項失敗" >&2
    return 1
}

# 每個 wrapper 都以 w_main "$@" 收尾
w_main() {
    if [ $# = 0 ]; then
        w_list; echo; echo "  用法：$0 --list | --dry-run <script> [args...] | --self-test | <script> [args...]"
        return 0
    fi
    case "$1" in
        --list)      w_list; return 0 ;;
        --self-test) w_selftest; return $? ;;
        --dry-run)
            w_precheck || return 1
            shift
            local s="${1:-}"; [ -n "$s" ] && shift || true
            if [ -z "$s" ]; then echo "  用法：$0 --dry-run <script> [args...]"; return 2; fi
            [ -f "$(w_script_path "$s")" ] || { echo "error: scripts/check/$s 不在這一類（或不存在）" >&2; return 1; }
            echo "  cd $ENGINE_REPO_ROOT && $(w_interp "$(w_script_path "$s")") $(w_script_path "$s") $*"
            echo "  （dry-run：沒有執行任何東西）"
            return 0 ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; return 0 ;;
    esac
    w_precheck || return 1
    local s="$1"; shift
    w_run "$s" "$@"
}
