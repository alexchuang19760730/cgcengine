#!/bin/bash
# dlk_ab.sh — 間歇 prefill GPU 死鎖歸因 A/B（2026-08-29）
#
# 證據（watchdog + sample + ioreg）：
#   - 死鎖時 6 個 command buffer 卡 Executing、3 個 Committed，kernel GPU scheduler
#     busy=0 / BusyWorkQueues 空、recoveryCount=0 → driver 層 completion 遺失
#   - pool fill 線程全閒置、encode 無停滯；唯一活躍的 kernel 端 GPU 機制 =
#     MTLResidencySet 心跳（每 5ms 對每個 rset requestResidency →
#     IOGPUResourceSetPurgeable kernel trap；graph_compute 每 2ms reset keep_alive）
#
# 假說 H1：residency 心跳 × in-flight completion 的 driver race（記憶體壓力調製 1-in-4）
# 判定：污染 session 下 B（現況）仍死鎖、T（GGML_METAL_NO_RESIDENCY=1）= 0 → H1 成立
#
#   B = baseline（residency ON）   T = GGML_METAL_NO_RESIDENCY=1（關 rset + 心跳）
# 順序：2 輪 seed（不計）建立 page-cache 污染 → B/T 交替 ×7 → T ×2（污染更重的條件）
cd /Users/alexchuang/Documents/flashkv0516 || exit 1

SUMMARY=/tmp/dlk_summary.txt
: > "$SUMMARY"

run_one() {
    local tag=$1 envs=$2
    local log=/tmp/dlk_run_${tag}.log
    local t0=$(date +%s)
    env $envs N30CACHE_WATCHDOG=1 ./scripts/run_n30cache.sh -m qwen36 --mtp --dense-iq4x --steady \
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
    local t1=$(date +%s)
    cp /tmp/n30cache.err /tmp/dlk_${tag}.err 2>/dev/null
    cp /tmp/n30cache.out /tmp/dlk_${tag}.out 2>/dev/null
    local stall=$(grep -c "Metal stall" /tmp/dlk_${tag}.err 2>/dev/null || true)
    local spd=$(grep -oE "speed: *[0-9.]+ t/s" "$log" | tail -1)
    local acc=$(grep -oE "accept *= *[0-9.]+%" /tmp/dlk_${tag}.err 2>/dev/null | tail -1)
    local mp=$(memory_pressure 2>/dev/null | grep -E "percentage|pressure" | head -2 | tr '\n' ' ')
    echo "[$tag] rc=$rc dur=$((t1-t0))s stall=$stall $spd $acc | $mp" | tee -a "$SUMMARY"
}

# seed（不計入）
run_one seed1 ""
run_one seed2 ""

# B/T 交替 ×7
for i in 1 2 3 4 5 6 7; do
    run_one "B$i" ""
    run_one "T$i" "GGML_METAL_NO_RESIDENCY=1"
done
# T 追加 ×2（污染最重條件下）
run_one "T8" "GGML_METAL_NO_RESIDENCY=1"
run_one "T9" "GGML_METAL_NO_RESIDENCY=1"

echo "=== 總計 ===" | tee -a "$SUMMARY"
b_total=7; t_total=9
b_dl=$(grep -E "^\[B[0-9]+\].*stall=[1-9]" "$SUMMARY" | wc -l | tr -d ' ')
t_dl=$(grep -E "^\[T[0-9]+\].*stall=[1-9]" "$SUMMARY" | wc -l | tr -d ' ')
echo "B: $b_dl/$b_total deadlocks (residency ON)" | tee -a "$SUMMARY"
echo "T: $t_dl/$t_total deadlocks (NO_RESIDENCY)" | tee -a "$SUMMARY"
