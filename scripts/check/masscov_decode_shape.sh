#!/bin/bash
# masscov_decode_shape.sh — 在「decode 主導」形狀下重測 masscov 的 cur / selcold
#
# 為什麼要跑這個（這是 docs/ACCEPTMOE_ADAPTATION §5.0 之外新發現的口徑分歧）：
#   - scripts/check/pin_profiles/masscov_base_2026-09-23.txt 的 cur=0.6585 / selcold=0.3416
#     是在 pin_abba.sh 的形狀下測的：-p 2048 -n 128 -d 512 ⇒ prefill 2560 token 佔 91%，
#     decode 只有 256 token。池在 prefill 階段從空開始填 ⇒ 冷訪問被 prefill 大量貢獻。
#   - 但 route_overlap_3prompt.sh（decode 主導：-p 256 -n 512 ⇒ decode 佔 67%）實測
#     decode/pool hit = 96.2%，與 prod-new 交付 json 的 96.3% 一致 ⇒ 冷訪問只有 ~3.8%。
#   兩者差 10 倍。AcceptMoE 的「代價」與「收益」都取決於這個數是哪一個，
#   所以必須在同一個形狀下量一次，不能拿 prefill 主導的數去定 decode 的價格。
#
# 方法：同 prod-new env（pool 8 GiB）+ 同 decode 主導形狀 + CGC_MASSCOV=1，
#   CGC_MASSCOV_DUMP 寫出 per-layer cur / selcold / kN。
#
# 判據：
#   selcold ≈ 0.04  ⇒ 34% 是 prefill 污染；decode 穩態幾乎全駐留
#                   ⇒ AcceptMoE 代價降到 ~4%，但收益也同時縮水（cb 本來就不是冷訪問造成）
#   selcold ≈ 0.34  ⇒ decode 穩態真的有 34% 冷訪問；hit 96.2% 與 selcold 34% 的 10x 矛盾
#                     必須另找機制（很可能是「ensure_batch 命中判定」與「slot_table 快照時機」不同）
#
# 用法：
#   bash scripts/check/masscov_decode_shape.sh
#   PROMPT_FILE=/path/to.txt bash scripts/check/masscov_decode_shape.sh
set -u

REPO="/Users/alexchuang/Documents/flashkv-devserver"
BIN="$REPO/src/llama.cpp/build/bin/llama-bench"
MODEL="$REPO/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"
PROMPT="${PROMPT_FILE:-$REPO/scripts/prompts/calib_qwen36.txt}"
TAG="${TAG:-mcshape}"
DUMP="${DUMP:-$REPO/scripts/check/pin_profiles/masscov_${TAG}.txt}"
POOL_BYTES="${POOL_BYTES:-8589934592}"

# ---- 共用閘門：檢查與動作必須在同一分支 ----
LISTEN="$(lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null | tail -n +2)"
BUSY="$(pgrep -x llama-server; pgrep -x llama-bench; pgrep -x llama-cli; \
        pgrep -fl 'run_ids_dst_capture|decode_sweep|prod_profile|window_sentinel|llama_bench_matrix')"
if [ -n "${LISTEN}" ] || [ -n "${BUSY}" ]; then
    echo "ABORT: 視窗忙 listener=[${LISTEN}] busy=[${BUSY}]" >&2
    exit 1
fi
[ -f "$BIN" ]     || { echo "error: binary not found: $BIN" >&2; exit 2; }
[ -f "$MODEL" ]   || { echo "error: model not found: $MODEL" >&2; exit 2; }
[ -f "$PROMPT" ]  || { echo "error: prompt not found: $PROMPT" >&2; exit 2; }

# ---- 完整 prod-new env（pool_sweet_spot.sh:14 固化版）+ MASSCOV ----
# 與 route_overlap_3prompt.sh 只差 CGC_MASSCOV=1 / CGC_MASSCOV_DUMP（MASSCOV 不改行為）。
ENVS=(CGC_SERVER_MTP=0 CGC_EXPERT_CACHE_BYTES="$POOL_BYTES" \
      LLAMA_EXPERT_CACHE_ALLOW_NGL=1 LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=0 \
      LLAMA_EXPERT_CACHE_WORKERS=8 CGC_WAKE_POLL_US=15 CGC_EVICTED_RING=0 \
      CGC_N_CB=8 CGC_OA_ASYNC=1 CGC_SERVER_AUTO_ANCHOR=0 \
      CGC_SERVER_DEFAULT_MARKER_STOPS=1 CGC_GATHER_SLAB_CAP=256 \
      CGC_PREFILL_STREAM=1 CGC_DBUF=1 CGC_SPAC=1 CGC_SPAC_ALPHA=0.75 \
      CGC_SOFT_POOL_L0=0 CGC_SOFT_POOL_L1=0 CGC_LOOP_GUARD=1 \
      CGC_GLU_FUSED_DOWN=1 CGC_WATCHDOG=1 CGC_MM_BITIDENT=1)

# ── [CGC 2026-09-24] 超訂預檢閘門（docs/SWAP_MISS_LINK_2026-09-24.md §7「立即可做」）──
# pool 8 GiB + load_mode=none 在 16 GB 上是**靜態超訂** 4838 MiB（13030 + 8192 = 21222 > 16384，
# 與實測 resident 21222 MiB 對上）⇒ 這種配置跑出來的數字活在壓縮 + swap 之上，不可引用。
# 預設（BUDGET_GATE=strict）直接 exit 2 拒跑，不再造出污染樣本；要跑請
#   BUDGET_GATE=warn  → 放行但 export CGC_BUDGET_OVERSUBSCRIBED=1（樣本帶標記，不可當乾淨基線）
#   BUDGET_GATE=off   → 完全不檢查（相容既有流程）
#   或把 pool 降到 ≤ 3 GiB（16384 − 13030 = 3354 MiB）。
. "${REPO}/scripts/check/budget_gate.sh"
echo "[budget] gate: ${CGC_BUDGET_OVERSUBSCRIBED:+OVERSUBSCRIBED-ACK }BUDGET_GATE=${BUDGET_GATE:-strict}"

echo "[$(date +%H:%M:%S)] === masscov decode-shape start (pool=$((POOL_BYTES/1073741824))G) ==="
env "${ENVS[@]}" CGC_MASSCOV=1 CGC_MASSCOV_DUMP="$DUMP" \
    "$BIN" -m "$MODEL" -ngl 99 --load-mode none -t 8 \
        -expert-cache "$POOL_BYTES" --cache-type-k q8_0 --cache-type-v q8_0 \
        -b 5632 -ub 5632 -p 256 -n 512 -d 0 -r 2 \
        --prompt-file "$PROMPT" -o json \
        > "/tmp/masscov_${TAG}.json" 2> "/tmp/masscov_${TAG}.log"
RC=$?
if [ $RC -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] FAILED rc=${RC}" >&2
    tail -8 "/tmp/masscov_${TAG}.log" >&2
    exit 1
fi
echo "[$(date +%H:%M:%S)] OK dump=${DUMP}"
grep -hE "decode/pool|ROUTE-DUMP|COUNTERFACTURAL|COUNTERFACTUAL" "/tmp/masscov_${TAG}.log" | tail -8 || true
