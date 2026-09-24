#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# [CGC 2026-09-24] 等一個乾淨視窗再 build —— 不搶別條線的 binary。
#
# 背景：build 產物（libllama.dylib / llama-server）是**共用資源**。在別條線的量測
# 行程還在跑的時候 build，會有兩種傷害：
#   ① 重連結會讓「之後才啟動」的行程拿到新 binary，而「之前啟動」的還是舊的
#      ⇒ 同一次 A/B 橫跨兩個 build，該 repo 的所有比較都建立在「同一個 build」上；
#   ② 機器記憶體吃緊時（swap 近滿）build 本身會失敗或更慢。
#
# 所以這裡採「連續空窗」判定：不是看到一次空就 build，而是
#   **連續 CLEAR_N 次（每次間隔 POLL_SEC 秒）三個閘門都空** 才動手，
#   動手前再驗一次，且檢查與動作在同一個 if 分支裡（不能只是印出來）。
#
# 用法：
#   bash scripts/check/build_when_clear.sh                 # 預設等 20 分鐘
#   MAX_WAIT_MIN=5 bash scripts/check/build_when_clear.sh  # 只等 5 分鐘
#   bash scripts/check/build_when_clear.sh --now           # 不等待，立刻驗一次就決定
# ─────────────────────────────────────────────────────────────────────────────
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO}/src/llama.cpp/build"
POLL_SEC="${POLL_SEC:-20}"
CLEAR_N="${CLEAR_N:-3}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-20}"
NOW="${1:-}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ── 三個閘門：任一非空就不可 build ──────────────────────────────────────────
#   ① 8080 有 listener（server 正在服務）
#   ② 任何量測行程（server / bench / sweep / 別的 build）
gate_reason() {
    if lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "8080 有 listener"
        return 1
    fi
    local p
    p="$(pgrep -fl 'llama-server|llama-bench|run_server\.sh|decode_sweep|prod_matrix|cmake|ninja|^make' 2>/dev/null | head -3)"
    if [ -n "${p}" ]; then
        echo "量測/build 行程：$(echo "${p}" | head -1 | cut -c1-60)"
        return 1
    fi
    echo ""
    return 0
}

attempt_build() {
    # ★ 檢查與動作必須在同一個分支：這裡再驗一次，驗過才真的 build。
    local why
    if ! why="$(gate_reason)"; then
        log "放棄這次機會：${why}"
        return 2
    fi
    log "視窗乾淨，開始 build"
    if cmake --build "${BUILD_DIR}" --target llama-server llama-bench -j 8 \
         >"${REPO}/.workbuddy/build_when_clear.log" 2>&1; then
        log "build rc=0"
    else
        log "build 失敗，見 .workbuddy/build_when_clear.log"
        tail -15 "${REPO}/.workbuddy/build_when_clear.log"
        return 3
    fi
    # 驗開關真的編進去了（不是只改了原始碼）
    local n
    n="$(strings "${BUILD_DIR}/bin/libllama.dylib" 2>/dev/null \
         | grep -cE '^(CGC_EXPERT_SKIP_READRAW|CGC_POOL_MADVISE)$')"
    log "開關字串命中數 = ${n}（應為 2）"
    [ "${n}" -ge 2 ] || return 4
    log "完成：P0.g 已進 binary"
    return 0
}

if [ "${NOW}" = "--now" ]; then
    attempt_build
    exit $?
fi

log "等乾淨視窗：連續 ${CLEAR_N} 次 × ${POLL_SEC}s 全空才 build（最多 ${MAX_WAIT_MIN} 分鐘）"
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
clear_run=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
    if why="$(gate_reason)"; then
        clear_run=$((clear_run + 1))
        log "空窗 ${clear_run}/${CLEAR_N}"
        if [ "${clear_run}" -ge "${CLEAR_N}" ]; then
            attempt_build
            exit $?
        fi
    else
        if [ "${clear_run}" -gt 0 ]; then
            log "空窗中斷：${why}"
        fi
        clear_run=0
    fi
    sleep "${POLL_SEC}"
done
log "逾時 ${MAX_WAIT_MIN} 分鐘，一直沒等到乾淨視窗；未 build。"
exit 1
