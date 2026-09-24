#!/usr/bin/env bash
# [CGC 2026-09-24] 量測腳本的「超訂拒跑」閘門 —— 立即可做項（docs/SWAP_MISS_LINK_2026-09-24.md §7）
#
# 為什麼要有這個東西：
#   prod-new + pool 8 GiB 在 16 GB 機器上是**靜態超訂**的（model 13030 + pool 8192 = 21222
#   > 16384，超訂 4838 MiB，與實測 resident 21222 MiB 對上）。這種配置跑出來的數字活在
#   macOS 記憶體壓縮 + swap 之上，不可引用 —— MTP on + ρ 影子節點那一輪就是這樣白跑
#   50 分鐘、B 臂全 0.00 t/s（commit 3f0e97140 的實證）。
#   `scripts/check/budget_preflight.py` 提供了口徑，但**它只是一支工具**：量測腳本不會自己
#   去叫它。這支 gate 就是把「launch 前先預檢」變成一行 source 的共用面。
#
# 用法（在量測腳本 launch 之前）：
#     . scripts/check/budget_gate.sh          # 超訂 → 直接 exit 2，不產出樣本
#  或 BUDGET_GATE=warn bash your_script.sh    # 明知超訂仍要跑：警告 + 打標記後繼續
#
# 環境（全部可選，都有預設）：
#   POOL_BYTES | CGC_SERVER_EXPERT_CACHE_BYTES   pool 上限 bytes（預設 8 GiB = prod-new）
#   MODEL      | CGC_SERVER_MODEL                模型檔（預設 prod-new Nail checkpoint）
#   LOAD_MODE  | CGC_SERVER_LOAD_MODE            none|mmap|mmap+mlock（預設 none；
#                                                mmap 系 model resident=0，OS 可回收）
#   BUDGET_GATE  strict（預設）| warn | off
#   PY                                           直譯器（預設 python3）
#
# 行為：
#   strict：超訂 → exit 2 拒跑；未超訂 → export CGC_SERVER_STRICT_BUDGET=1
#           （讓 run_server.sh 自己也擋一次，兩道閘門同一口徑）
#   warn  ：超訂 → 印警告 + export CGC_BUDGET_OVERSUBSCRIBED=1（讓下游 log 帶標記）+ 繼續
#   off   ：不檢查（相容既有流程）
#
# ⚠ 給 16 GB 這台機器的現實：pool 8 GiB 一定超訂 ⇒ strict 會拒跑。要跑請 BUDGET_GATE=warn
#   （樣本會帶 OVERSUBSCRIBED 標記，將來不會被誤當乾淨樣本引用），或把 pool 降到 ≤ 3 GiB
#   （16384 − 13030 = 3354 MiB，與 commit 3f0e97140 的結論一致）。

# shellcheck shell=bash
_BG_SRC="${BASH_SOURCE[0]:-$0}"
_BG_SOURCED=0
[ "${_BG_SRC}" != "$0" ] && _BG_SOURCED=1

budget_gate() {
    local mode="${BUDGET_GATE:-strict}"
    local py="${PY:-python3}"
    local repo
    repo="$(cd "$(dirname "${_BG_SRC}")/../.." && pwd)"
    local gate_py="${repo}/scripts/check/budget_preflight.py"

    if [ "${mode}" = "off" ]; then
        echo "[budget-gate] off（BUDGET_GATE=off）：跳過超訂預檢。"
        return 0
    fi
    if [ ! -f "${gate_py}" ]; then
        echo "[budget-gate] 找不到 ${gate_py} —— 不擋，但這一趟沒有預檢。" >&2
        return 0
    fi

    local pool="${POOL_BYTES:-${CGC_SERVER_EXPERT_CACHE_BYTES:-8589934592}}"
    local model="${MODEL:-${CGC_SERVER_MODEL:-${repo}/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf}}"
    local load_mode="${LOAD_MODE:-${CGC_SERVER_LOAD_MODE:-none}}"

    # --warn-only 讓 preflight 同時印數字與判定；rc=2 才是 OVERSUBSCRIBED（見 budget_preflight.py）。
    # 注意：不能直接依賴 preflight 的預設拒跑，因為 warn 模式下我們仍要拿到它的判定繼續跑。
    local out rc
    out="$("${py}" "${gate_py}" --pool-bytes "${pool}" --model "${model}" \
                   --load-mode "${load_mode}" 2>&1)"
    rc=$?
    printf '%s\n' "${out}"

    if [ "${rc}" -eq 0 ]; then
        # 未超訂：把 run_server.sh 那道閘門也打開（同一口徑，雙保險）。
        export CGC_SERVER_STRICT_BUDGET=1
        unset CGC_BUDGET_OVERSUBSCRIBED 2>/dev/null || true
        return 0
    fi

    # rc != 0：只可能是 OVERSUBSCRIBED（preflight 的唯一非 0 出口）
    if [ "${mode}" = "warn" ]; then
        export CGC_BUDGET_OVERSUBSCRIBED=1
        unset CGC_SERVER_STRICT_BUDGET 2>/dev/null || true
        echo "[budget-gate] BUDGET_GATE=warn：明知超訂仍跑 —— 已 export CGC_BUDGET_OVERSUBSCRIBED=1，" >&2
        echo "[budget-gate]   這一趟的樣本帶 OVERSUBSCRIBED 標記，不可當乾淨基線引用。" >&2
        return 0
    fi
    echo "[budget-gate] 拒跑（exit 2）。要硬跑：BUDGET_GATE=warn（帶標記）或減 pool 到 ≤ 3 GiB。" >&2
    return 2
}

# ── self-test：用真實的 hw.memsize（16 GB）＋ 假的小模型檔，四個分支各驗一次 ──
budget_gate_self_test() {
    local py="${PY:-python3}" tmp rc pass=0 fail=0
    tmp="$(mktemp -d)"
    local small="${tmp}/small.gguf"
    "${py}" -c "import sys; open(sys.argv[1],'wb').truncate(100*1048576)" "${small}"
    local big_pool=20971520000     # 20000 MiB：100 + 20000 > 16384 ⇒ 必然超訂
    local tiny_pool=104857600      # 100 MiB：100 + 100 < 16384 ⇒ 不超訂
    local mmap_pool=17091788800    # 16300 MiB：none => 100+16300 > 16384 超訂；mmap => 0+16300 放行

    check() { # name expected_rc
        if [ "$2" = "$3" ]; then pass=$((pass+1)); echo "  ok   $1 (rc=$2)";
        else fail=$((fail+1)); echo "  FAIL $1 (rc=$2, want $3)"; fi
    }

    echo "[budget-gate] self-test：hw.memsize=$(sysctl -n hw.memsize) bytes"

    ( BUDGET_GATE=strict POOL_BYTES="${tiny_pool}" MODEL="${small}" . "${_BG_SRC}" ) >/dev/null 2>&1
    check "未超訂 strict => 放行" "$?" 0

    ( BUDGET_GATE=strict POOL_BYTES="${big_pool}" MODEL="${small}" . "${_BG_SRC}" ) >/dev/null 2>&1
    check "超訂 strict => 拒跑" "$?" 2

    ( BUDGET_GATE=warn POOL_BYTES="${big_pool}" MODEL="${small}" . "${_BG_SRC}" ) >/dev/null 2>&1
    check "超訂 warn => 放行" "$?" 0

    ( BUDGET_GATE=off POOL_BYTES="${big_pool}" MODEL="${small}" . "${_BG_SRC}" ) >/dev/null 2>&1
    check "超訂 off => 放行" "$?" 0

    # warn 分支必須真的把標記 export 出去（下游 log 靠它標污染樣本）。
    # ⚠ 這裡用「同一個 subshell 內 source 完再判斷」。第一版寫成
    #   mark="$( . gate >/dev/null 2>&1; printf '%s' "${VAR:-unset}" )"，實測靜默拿到空字串
    #   （FAIL）；同樣的寫法單獨跑卻拿得到 1 —— 機制未究，不要假裝懂。同 subshell 的寫法
    #   已實測穩定，就用它。
    if ( BUDGET_GATE=warn POOL_BYTES="${big_pool}" MODEL="${small}" . "${_BG_SRC}" >/dev/null 2>&1 && \
         [ "${CGC_BUDGET_OVERSUBSCRIBED:-0}" = "1" ] ); then
        pass=$((pass+1)); echo "  ok   warn 分支 export CGC_BUDGET_OVERSUBSCRIBED=1"
    else
        fail=$((fail+1)); echo "  FAIL warn 分支沒帶標記"
    fi

    # mmap：model resident=0 ⇒ 同一個 pool 不再超訂（口徑與 run_server.sh:1050 一致）
    ( BUDGET_GATE=strict POOL_BYTES="${mmap_pool}" MODEL="${small}" LOAD_MODE=none . "${_BG_SRC}" ) >/dev/null 2>&1
    check "同一 pool 在 load_mode=none 下超訂" "$?" 2
    ( BUDGET_GATE=strict POOL_BYTES="${mmap_pool}" MODEL="${small}" LOAD_MODE=mmap . "${_BG_SRC}" ) >/dev/null 2>&1
    check "同一 pool 在 mmap（resident=0）下放行" "$?" 0

    rm -rf "${tmp}"
    echo "[budget-gate] self-test ${pass} passed / ${fail} failed"
    [ "${fail}" -eq 0 ]
}

# 直接執行（非 source）時：--self-test 跑測試，其餘當作一次單發預檢。
if [ "${_BG_SOURCED}" -eq 0 ]; then
    if [ "${1:-}" = "--self-test" ]; then
        budget_gate_self_test
        exit $?
    fi
    budget_gate
    exit $?
fi

# source 時：直接跑一次；超訂就 exit（**不是 return**）。
# ⚠ `return` 只會結束「被 source 的那份檔案」，呼叫方會若無其事地繼續往下跑 —— 第一版就是這樣，
#   閘門判了拒跑、量測還是照跑。拒跑必須是 exit，讓呼叫方整支停下來。
budget_gate
_bg_rc=$?
if [ "${_bg_rc}" -ne 0 ]; then
    exit "${_bg_rc}"
fi
