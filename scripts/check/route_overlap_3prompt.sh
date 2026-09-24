#!/bin/bash
# route_overlap_3prompt.sh — 跨 prompt 路由重疊（AcceptMoE / 靜態放置的 go-no-go 門檻數）
#
# 為什麼要跑這個：
#   docs/ACCEPTMOE_ADAPTATION_2026-09-24.md §5.0 指出 k143=98.88% 是「同 prompt 內取前 N 名」
#   的 in-sample 上界。要判斷「靜態／混合放置」與「以駐留為資格」兩條路站不站得住，
#   必須知道：不同 prompt 的 top-K 路由集合重疊多少。
#     重疊高 ⇒ 路由是模型固有的 heavy-tail，靜態放置可行，駐留率有得上 95%。
#     重疊低 ⇒ 路由 prompt-dependent，上界虛高，只剩 AcceptMoE 的 +12~13%。
#
# 方法：同 prod-new env（pool 8 GiB）跑 3 個領域差異極大的真實文本 prompt，
#   LLAMA_EXPERT_CACHE_ROUTE_RECORD=1 記錄每層路由頻率，teardown 時
#   LLAMA_EXPERT_CACHE_ROUTE_DUMP 寫出 per-layer top-K 排名（PIN_PROFILE 格式）。
#   再用 scripts/check/accept_moe_sizing.py --overlap 兩兩比較。
#
# ⚠ 口徑：llama-bench 預設用 std::rand()%n_vocab 填 prompt（--help 明說
#   "The random fill is NOT what the server sees"），所以歷來用隨機 fill 的 bench
#   路由統計不代表真實文本。本腳本一律用 --prompt-file 真實文本。
#
# ⚠ 形狀：freq 同時含 prefill 與 decode。本輪刻意讓 decode 主導（-p 256 -n 512
#   ⇒ decode 佔 67%），因為 cb／駐留是 decode 現象。
#
# 用法：
#   bash scripts/check/route_overlap_3prompt.sh
#   TAG=myrun bash scripts/check/route_overlap_3prompt.sh
set -u

REPO="/Users/alexchuang/Documents/flashkv-devserver"
BIN="$REPO/src/llama.cpp/build/bin/llama-bench"
MODEL="$REPO/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"
OUTDIR="$REPO/scripts/check/pin_profiles"
TAG="${TAG:-ovl3}"
POOL_BYTES="${POOL_BYTES:-8589934592}"   # 8 GiB（用戶指定；= prod-new 預設值）

# ---- 共用閘門：檢查與動作必須在同一分支（否則檢查只是安慰劑）----
LISTEN="$(lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null | tail -n +2)"
BUSY="$(pgrep -x llama-server; pgrep -x llama-bench; pgrep -x llama-cli; \
        pgrep -fl 'run_ids_dst_capture|decode_sweep|prod_profile|window_sentinel|llama_bench_matrix')"
if [ -n "${LISTEN}" ] || [ -n "${BUSY}" ]; then
    echo "ABORT: 視窗忙 listener=[${LISTEN}] busy=[${BUSY}]" >&2
    exit 1
fi
[ -f "$BIN" ]   || { echo "error: binary not found: $BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "error: model not found: $MODEL" >&2; exit 2; }

# ---- 完整 prod-new env（解析自 run_server.sh，pool_sweet_spot.sh:14 已固化版）----
# 缺任一個都會 Metal OOM（見 §EN-47x）。本輪額外加 ROUTE_RECORD=1。
BASE_ENVS=(CGC_SERVER_MTP=0 CGC_EXPERT_CACHE_BYTES="$POOL_BYTES" \
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

mkdir -p "$OUTDIR"

run_one() {
    local name="$1" prompt="$2" dump="$3"
    echo "[$(date +%H:%M:%S)] === arm ${name} start (swap=$(sysctl -n vm.swapusage | awk -F'used = ' '{split($2,a," "); print a[1]}')) ==="
    env "${BASE_ENVS[@]}" LLAMA_EXPERT_CACHE_ROUTE_RECORD=1 \
        LLAMA_EXPERT_CACHE_ROUTE_DUMP="$dump" \
        "$BIN" -m "$MODEL" -ngl 99 --load-mode none -t 8 \
            -expert-cache "$POOL_BYTES" --cache-type-k q8_0 --cache-type-v q8_0 \
            -b 5632 -ub 5632 -p 256 -n 512 -d 0 -r 2 \
            --prompt-file "$prompt" -o json \
            > "/tmp/route_ovl_${TAG}_${name}.json" 2> "/tmp/route_ovl_${TAG}_${name}.log"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] arm ${name} FAILED rc=${rc}" >&2
        tail -5 "/tmp/route_ovl_${TAG}_${name}.log" >&2
        return 1
    fi
    if [ ! -s "$dump" ]; then
        echo "[$(date +%H:%M:%S)] arm ${name}: DUMP 沒寫出來" >&2
        return 1
    fi
    local experts layers
    experts=$(awk '{n+=NF} END{print n+0}' "$dump")
    layers=$(wc -l < "$dump" | tr -d ' ')
    echo "[$(date +%H:%M:%S)] arm ${name} OK dump=${dump} (${experts} experts / ${layers} layers)"
    grep -h "ROUTE-DUMP" "/tmp/route_ovl_${TAG}_${name}.log" | tail -2 || true
}

run_one code    "$REPO/scripts/prompts/calib_qwen36.txt"      "$OUTDIR/route_${TAG}_code.txt"
run_one history "$REPO/scripts/prompts/xprompt_zh_history.txt" "$OUTDIR/route_${TAG}_history.txt"
run_one bio     "$REPO/scripts/prompts/xprompt_biology.txt"    "$OUTDIR/route_${TAG}_bio.txt"

echo "=== DONE ${TAG} ==="
