#!/usr/bin/env bash
# [CGC 2026-09-24] 量測腳本的「超訂拒跑」閘門 —— 立即可做項（docs/SWAP_MISS_LINK_2026-09-24.md §7）
#
# 為什麼要有這個東西：
#   真實超訂（dense 常駐 + pool > 實體）跑出來的數字活在 macOS 記憶體壓縮 + swap 之上，
#   不可引用 —— MTP on + ρ 影子節點那一輪就是這樣白跑 50 分鐘、B 臂全 0.00 t/s
#   （commit 3f0e97140 的實證）。
#   [口徑 v3 2026-10-18，依 docs/L4_POOL_ANON_NONRECLAIMABLE_VERDICT_2026-10-18.md]
#   早期 v1 用「模型檔案全量 13030 + pool 8192 = 21222」（雙重計 expert，作廢）；v2（2026-10-17）
#   改 dense 555 + pool 8192 = 8747 對比 hw.memsize 16384，仍誤判 OK，且誤把 pool 當檔案 mmap
#   可回收。v3 修正四道：① 上限改引擎自報 recommendedMaxWorkingSetSize（本機 11453 MiB，
#   Backup/metal_working_set.json），不是 hw.memsize；② L4 pool 是 **Metal shared 匿名 dirty 頁**
#   （vm_allocate + newBufferWithBytesNoCopy），pread 寫入即不可回收、evict 後 madvise 只覆蓋 L3
#   不覆蓋 L4，pool 全額按不可回收計；③ 補 KV(85)+gather(356)+compute(400) 固定開銷；④ 分窄/寬
#   工作集：narrow（harness 單 prompt）靜態門放行；wide（常駐多 profile）池被 churn 全滿、冷頁
#   只在 swap（8G 實測獨佔髒足跡 ≈12.9G、swap 4.55G），須過更嚴的寬集門。
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
#                                                mmap 系 dense 常駐=0，OS 可回收）
#   DENSE_MIB  | CGC_DENSE_MIB                   非 expert 固定常駐 MiB（預設 preflight 內 555＝
#                                                denseIQ4X 實測；空值用預設，與 run_server 同口徑）
#   WORKLOAD   | CGC_BUDGET_WORKLOAD             narrow（預設，harness 單 prompt，靜態門）|
#                                                wide（常駐 server 連跑多 profile，加寬集冷頁門）
#   BUDGET_GATE  strict（預設）| warn | off
#   PY                                           直譯器（預設 python3）
#
# 行為：
#   strict：超訂 → exit 2 拒跑；未超訂 → export CGC_SERVER_STRICT_BUDGET=1
#           （讓 run_server.sh 自己也擋一次，兩道閘門同一口徑）
#   warn  ：超訂 → 印警告 + export CGC_BUDGET_OVERSUBSCRIBED=1（讓下游 log 帶標記）+ 繼續
#   off   ：不檢查（相容既有流程）
#
# 16 GB 這台機器的現實（v3 口徑，Metal ceiling 11453、可用門檻 11453×0.9=10308）：
#   固定非池常駐≈1396（dense555+kv85+gather356+compute400）。
#   靜態容量門（narrow）：8G 需求 9588（餘 720，緊但過）/ 6G 7540（餘 2768）/ 4G 5492（餘 4816）。
#   寬工作集門（wide，+2048 寬集餘量）：8G 需求 11636 > 10308 **拒跑**（常駐多主題實測必然
#   swap、tier D，見 L4 判詞）/ 6G 9588（過，餘 720）/ 4G 7540（過，餘 2768）。
#   ⇒ 常駐 replay 多 profile 請用 6G（寬集最小可行）或 4G，或等 L4 冷頁歸還（C 項）；
#     穩態速度權威仍用 harness 單 prompt（narrow）的 C17 口徑。10G 擴池撞 Metal working set OOM
#     （見 POOL10G_OOM_VERDICT）。若 dense 實測比預設大，用 DENSE_MIB 覆寫。

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

    # [CGC 2026-10-06 fix] run_server.sh 把外部覆寫鍵 CGC_SERVER_EXPERT_CACHE_BYTES 折算成
    # SERVER_ENV 裡的 CGC_EXPERT_CACHE_BYTES；matrix 的 resolve() dump 只回 CGC_EXPERT_CACHE_BYTES
    # （沒有 CGC_SERVER_* 那把）。budget gate 若只讀 CGC_SERVER_EXPERT_CACHE_BYTES，會在 matrix
    # 路徑（樞紐池已降到 ≤3GiB 卻仍被判 8GiB 超訂）誤拒。優先序維持 CGC_SERVER_* > CGC_* > 預設。
    local pool="${POOL_BYTES:-${CGC_SERVER_EXPERT_CACHE_BYTES:-${CGC_EXPERT_CACHE_BYTES:-8589934592}}}"
    local model="${MODEL:-${CGC_SERVER_MODEL:-${repo}/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf}}"
    local load_mode="${LOAD_MODE:-${CGC_SERVER_LOAD_MODE:-none}}"
    # [口徑對齊 2026-10-17] dense（非 expert）常駐 MiB，預設在 preflight 內（555，denseIQ4X 實測），
    # 與 run_server.sh CGC_DENSE_MIB 同口徑；外部可覆寫。空值＝用 preflight 預設。
    local dense="${DENSE_MIB:-${CGC_DENSE_MIB:-}}"
    # [v3 2026-10-18] 窄/寬工作集：narrow（預設，harness 單 prompt 靜態門）/ wide（常駐多 profile
    # 加寬集冷頁門）。量測腳本常駐 replay 應傳 WORKLOAD=wide 或 CGC_BUDGET_WORKLOAD=wide。
    local workload="${WORKLOAD:-${CGC_BUDGET_WORKLOAD:-narrow}}"

    # --warn-only 讓 preflight 同時印數字與判定；rc=2 才是 OVERSUBSCRIBED（見 budget_preflight.py）。
    # 注意：不能直接依賴 preflight 的預設拒跑，因為 warn 模式下我們仍要拿到它的判定繼續跑。
    local out rc dense_arg=""
    [ -n "${dense}" ] && dense_arg="--dense-mib ${dense}"
    out="$("${py}" "${gate_py}" --pool-bytes "${pool}" --model "${model}" \
                   --load-mode "${load_mode}" --workload "${workload}" ${dense_arg} 2>&1)"
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
    echo "[budget-gate] 拒跑（exit 2）。要硬跑：BUDGET_GATE=warn（帶標記）；常駐多主題請減 pool 到 6G/4G（WORKLOAD=wide），或改用 harness 單 prompt（WORKLOAD=narrow）。" >&2
    return 2
}

# ── self-test：用真實 Metal ceiling（11453，可用門檻 10308）＋ 假小模型檔，各分支驗證 ──
budget_gate_self_test() {
    local py="${PY:-python3}" tmp rc pass=0 fail=0
    tmp="$(mktemp -d)"
    local small="${tmp}/small.gguf"
    "${py}" -c "import sys; open(sys.argv[1],'wb').truncate(100*1048576)" "${small}"
    local big_pool=20971520000     # 20000 MiB：fixed 1396 + 20000 > 10308 ⇒ 必然超訂
    local tiny_pool=104857600      # 100 MiB：fixed 1396 + 0.1 ≈ 1396 < 10308 ⇒ 不超訂
    local mmap_pool=9437184000     # 9000 MiB：none => fixed1396+9000=10396 > 10308 超訂；mmap => 841+9000=9841 放行
    local p8g=8589934592           # 8 GiB：narrow 9588 放行；wide 11636 > 10308 拒
    local p6g=6442450944           # 6 GiB：wide 9588 ≤ 10308 放行

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

    # mmap：model resident=0 ⇒ 同一個 pool 不再超訂（dense 不計匿名）
    ( BUDGET_GATE=strict POOL_BYTES="${mmap_pool}" MODEL="${small}" LOAD_MODE=none . "${_BG_SRC}" ) >/dev/null 2>&1
    check "同一 pool 在 load_mode=none 下超訂" "$?" 2
    ( BUDGET_GATE=strict POOL_BYTES="${mmap_pool}" MODEL="${small}" LOAD_MODE=mmap . "${_BG_SRC}" ) >/dev/null 2>&1
    check "同一 pool 在 mmap（dense=0）下放行" "$?" 0

    # v3 窄/寬工作集：8G narrow 放行（C17 單 prompt 健康）、wide 拒（常駐多主題實測 swap 4.55G）；6G wide 放行
    ( BUDGET_GATE=strict WORKLOAD=narrow POOL_BYTES="${p8g}" MODEL="${small}" . "${_BG_SRC}" ) >/dev/null 2>&1
    check "8G narrow（harness 單 prompt）=> 放行" "$?" 0
    ( BUDGET_GATE=strict WORKLOAD=wide   POOL_BYTES="${p8g}" MODEL="${small}" . "${_BG_SRC}" ) >/dev/null 2>&1
    check "8G wide（常駐多主題）=> 拒跑" "$?" 2
    ( BUDGET_GATE=strict WORKLOAD=wide   POOL_BYTES="${p6g}" MODEL="${small}" . "${_BG_SRC}" ) >/dev/null 2>&1
    check "6G wide => 放行" "$?" 0

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
