#!/bin/bash
# dlk_phase2.sh — 死鎖 root-cause phase 2：活性探針 + drain 緩解驗證（2026-08-29）
#
# 背景（phase 1 A/B 定案）：
#   - residency ON/OFF 都死鎖（T2 GGML_METAL_NO_RESIDENCY=1 相同簽名）→ H1 推翻
#   - 死鎖簽名：6 cb "Executing"（handler 已觸發、status 凍結）+ 3 cb "Committed"（永不排程）
#     GPU idle（BusyWorkQueues=()）→ Metal/IOGPU driver completion 管線 mid-batch wedge
#   - 記憶體壓力調製（T2 stall 時 38% free vs 正常 63-65%）
#
# phase 2 兩臂：
#   P（probe）：N30CACHE_WATCHDOG_CAPTURE=1，stall 時自動跑
#     probe[same-queue]  — 16B fill 提交到共享 queue（排在 wedge 後）：ALIVE=tail 未堵；DEAD=queue 堵死
#     probe[fresh-queue] — 全新 command queue：ALIVE=queue 級 wedge（可換 queue 恢復）；DEAD=device/kernel 級
#   D（drain）：N30CACHE_DRAIN_EVERY=64，每 64 graph_compute 全 queue 排水 → 驗證緩解假說
#
# 每臂連續跑（製造 page-cache 污染 → 逼出間歇死鎖），bit-identical 由 accept 確認
cd /Users/alexchuang/Documents/flashkv0516 || exit 1

ARM="${1:-P}"
N_RUNS="${2:-8}"
SUMMARY=/tmp/dlk2_summary.txt
: > "$SUMMARY"

run_one() {
    local tag=$1 envs=$2
    local log=/tmp/dlk2_run_${tag}.log
    env $envs N30CACHE_WATCHDOG_CAPTURE=1 ./scripts/run_n30cache.sh -m qwen36 --mtp --dense-iq4x --steady \
        > "$log" 2>&1 &
    local pid=$!
    local waited=0
    while kill -0 $pid 2>/dev/null; do
        sleep 5; waited=$((waited+5))
        if [ $waited -ge 300 ]; then
            echo "[$tag] OUTER TIMEOUT 300s — killing" | tee -a "$SUMMARY"
            kill -9 $pid 2>/dev/null
            pkill -9 -f "build/bin/llama-speculative-simple" 2>/dev/null
            break
        fi
    done
    wait $pid 2>/dev/null
    local rc=$?
    cp /tmp/n30cache.err /tmp/dlk2_${tag}.err 2>/dev/null
    local stall=$(grep -c "Metal stall" /tmp/dlk2_${tag}.err 2>/dev/null || true)
    local probe_same=$(grep -oE "probe\[same-queue\] \w+" /tmp/dlk2_${tag}.err 2>/dev/null | tail -1)
    local probe_fresh=$(grep -oE "probe\[fresh-queue\] \w+" /tmp/dlk2_${tag}.err 2>/dev/null | tail -1)
    local spd=$(grep -oE "speed: *[0-9.]+ t/s" "$log" | tail -1)
    local acc=$(grep -oE "accept *= *[0-9.]+%" /tmp/dlk2_${tag}.err 2>/dev/null | tail -1)
    local mp=$(memory_pressure 2>/dev/null | grep "percentage" | head -1 | tr -s ' ')
    echo "[$tag] rc=$rc stall=$stall | ${probe_same:-no-probe} ${probe_fresh:-} | $spd $acc | $mp" | tee -a "$SUMMARY"
}

if [ "$ARM" = "P" ]; then
    for i in $(seq 1 "$N_RUNS"); do
        run_one "P$i" ""
    done
elif [ "$ARM" = "D" ]; then
    for i in $(seq 1 "$N_RUNS"); do
        run_one "D$i" "N30CACHE_DRAIN_EVERY=64"
    done
else
    echo "usage: $0 P|D [N_RUNS]" >&2; exit 2
fi

echo "=== 總計 ===" | tee -a "$SUMMARY"
echo "deadlocks: $(grep -cE 'stall=[1-9]' "$SUMMARY") / $N_RUNS" | tee -a "$SUMMARY"
