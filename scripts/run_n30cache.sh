#!/bin/bash
# run_n30cache.sh — llama.cpp fork bounded-residency 生產啟動器
#
# 對應 turbo-fieldfare 的 bin/run_prod.sh，把所有經 A/B 定案的設定打包成
# 單一 CLI。每一項設定的出處見 moeexpert/LLAMACPP_EXPERT_BOUNDED_RESIDENCY_FORK_方案.md：
#
#   - -expert-cache BYTES（bounded pool）          : L2/L4，§8.38/8.44 日常定案
#   - LLAMA_EXPERT_CACHE_WORKERS=8（pool8）        : §8.12 甜蜜點（16 反效果）
#   - LLAMA_EXPERT_CACHE_ALLOW_NGL=1               : -ngl>0 + cache 必要（n99 硬 guard）
#   - LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1          : blk.0 排除 pool 修復，§8.35/8.36
#   - PIN_PROFILE / WAKE_POLL 預設 OFF             : §8.55 / §8.51 已證偽，opt-in 保留
#   - -no-mmap                                     : 凍機防護（mmap 9GB 冷頁風暴根因）
#
# 家族預設：
#   gemma4 : -ngl 30（sweet spot，§8.44/8.26；-ngl 99 不快且 content-dependent 風險）
#   qwen36 : -ngl 99（base 硬 OOM 13.2GB>11.45GB，只有 + cache 能 full offload，§8.38）
#
# 用法：
#   ./scripts/run_n30cache.sh -m gemma4 -n 128 -p "The capital of France is"
#   ./scripts/run_n30cache.sh -m qwen36 -n 128 --prompt-file /tmp/msg.txt
#   N30CACHE_BUDGET=8589934592 ./scripts/run_n30cache.sh -m qwen36 -n 128 -p "..."
#
# 可覆寫 env：N30CACHE_MODEL / N30CACHE_BUDGET / N30CACHE_NGL / N30CACHE_WORKERS /
#             N30CACHE_PIN_PROFILE / N30CACHE_WAKE_POLL_US
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/temp/llama_roadB/llama.cpp-master/build/bin/llama-simple"
BIN_SPEC="$ROOT/temp/llama_roadB/llama.cpp-master/build/bin/llama-speculative-simple"
G4="$ROOT/models/gguf/gemma-4-26B-A4B-it-UD-IQ3_S.gguf"
Q36="$ROOT/models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
# §MTP: qwen36 graft model 含 fraQtl blk.40 MTP head（Q4_K），
# 因純 UD-IQ3_XXS 沒 blk.40，無法啟用 --spec-type draft-mtp。
Q36_MTP="$ROOT/models/gguf/Qwen3.6-35B-A3B-UD-IQ3XXS-trunk_Q4K-blk40.gguf"

MODEL="${N30CACHE_MODEL:-${1:-gemma4}}"
N=128
PROMPT=""
PROMPT_FILE=""
CTX="${N30CACHE_CTX:-0}"  # 0 = model default (4096), production uses 2048
BUDGET="${N30CACHE_BUDGET:-4294967296}"     # 4GiB（qwen36 8GiB 可覆寫）
WORKERS="${N30CACHE_WORKERS:-8}"            # pool8 甜蜜點（§8.12）
WAKE_POLL_US="${N30CACHE_WAKE_POLL_US:-15}"  # §8.51：15us 為生產設定（先前測試 0 為 off，已證偽）
N_CB="${N30CACHE_N_CB:-8}"                    # §8.93：cb8 壓 p50（77→66ms），cb4 可覆寫
SEED="${N30CACHE_SEED:-}"                     # 未設 = llama-simple 預設；bit-identity 驗證請設固定值
# §MTP: MTP draft context 多吃一份 GPU buffer，預設 ctx=2048 避免 OOM。
# 實測 -c 0（model 預設 4096）+ MTP 必 OOM（kIOGPUCommandBufferCallbackErrorOutOfMemory）。
# §MTP n_max：draft tokens 數，預設 2（graft blk.40 為 Q4_K，>2 accept rate 不增反降）。
MTP=0
MTP_CTX=2048
MTP_N_MAX="${N30CACHE_MTP_N_MAX:-2}"

usage() {
    echo "usage: $0 [-m gemma4|qwen36] [-n tokens] [-p prompt | --prompt-file F] [--ngl N] [--budget BYTES] [--pin-profile F] [--no-cache] [--mtp [N]]
#   --mtp [N] 啟用 MTP draft-mtp（僅 qwen36 有效，自動切到 graft model + speculative-simple binary，-c 2048 解 OOM）；
#             N 為 --spec-draft-n-max，預設 2；可用 N30CACHE_MTP_N_MAX 覆寫
#   env: N30CACHE_N_CB / N30CACHE_SEED / N30CACHE_BUDGET / N30CACHE_NGL / N30CACHE_WORKERS / N30CACHE_MTP_N_MAX 可覆寫" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2 ;;
        -n) N="$2"; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --ngl) N30CACHE_NGL="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --pin-profile) PIN_PROFILE="$2"; shift 2 ;;
        --no-cache) NO_CACHE=1; shift ;;
        --mtp)
            MTP=1
            # 可選擇性接 N：--mtp 3
            case "${2:-}" in
                ''|*[!0-9]*) : ;;          # 下個不是數字，保持預設
                *) MTP_N_MAX="$2"; shift ;;
            esac
            shift
            ;;
        -h|--help) usage ;;
        *) usage ;;
    esac
done

# MTP 只支援 qwen36（純 UD-IQ3_XXS 沒 blk.40 MTP head）
if [ "$MTP" = 1 ] && [ "$MODEL" != "qwen36" ]; then
    echo "error: --mtp 僅支援 qwen36（需 graft model 含 blk.40 MTP head）" >&2
    exit 2
fi

case "$MODEL" in
    gemma4) M="$G4"; DEFAULT_NGL=30 ;;
    qwen36)
        DEFAULT_NGL=99
        if [ "$MTP" = 1 ]; then
            M="$Q36_MTP"
            BIN="$BIN_SPEC"
        else
            M="$Q36"
        fi
        ;;
    *) echo "error: MODEL must be gemma4|qwen36 (got $MODEL)" >&2; exit 2 ;;
esac
# §8.77/8.78: CGC_OA_ASYNC — Metal splits 走 async（免每層 callback sync）。
# 兩家族都驗證 bit-identical + 更快：qwen36 +12.6%、gemma4 +22%（§8.78）。
# 安全機制：ffn_moe_probs pin CPU（llama-context.cpp:3489）把 topk 鏈留在 CPU split，
# hook 在 CPU split 觸發，Metal split 無需 callback。
OA_ASYNC=1
# §8.81: CGC_N_CB=4 — MTL encode 平行化（fork 預設 1 是 per-op 計時安全設定，非最優）。
# qwen36 掃描：cb1 mean 106-126ms/run vs cb4 82-84ms/run（-20~25%），四臂 BIT-IDENTICAL，
# 機制是消掉串行 encode 的 >100ms 尾部尖峰（14-21/48 step → 4/48）。gemma4 短測亦 identical。
# §8.93: cb8 飽和掃描 — p50 77.6 → 66.3ms（-13%，encode-bound 直接砍 wall），9/9 bit-identical；
# cb16 無額外收益（67.9ms，encode thread 飽和）。cb8 取代 cb4 進生產。
NGL="${N30CACHE_NGL:-$DEFAULT_NGL}"

[ -f "$BIN" ] || { echo "error: binary not found: $BIN (先跑 scripts/build_cgc_llama.sh)" >&2; exit 2; }
[ -f "$M" ]  || { echo "error: model not found: $M" >&2; exit 2; }
if [ -n "$PROMPT_FILE" ]; then
    PROMPT="$(cat "$PROMPT_FILE")"
fi
[ -n "$PROMPT" ] || { echo "error: need -p prompt or --prompt-file" >&2; exit 2; }

# 生產 env 集（全部有 §8 出處；不設則對應行為 off）
ENVS=(LLAMA_EXPERT_CACHE_ALLOW_NGL=1
      CGC_EXPERT_CACHE_BYTES=$BUDGET
      LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
      LLAMA_EXPERT_CACHE_WORKERS=$WORKERS
      CGC_WAKE_POLL_US=$WAKE_POLL_US)
[ "$OA_ASYNC" = 1 ] && ENVS+=(CGC_OA_ASYNC=1)
ENVS+=(CGC_N_CB=$N_CB)
[ -n "${PIN_PROFILE:-}" ] && ENVS+=(LLAMA_EXPERT_CACHE_PIN_PROFILE="$PIN_PROFILE")

echo "=== n30cache production run ==="
echo "  model  : $MODEL ($(basename "$M"))"
echo "  ngl    : $NGL   budget: $((BUDGET/1073741824))GiB   workers: $WORKERS"
echo "  pin    : ${PIN_PROFILE:-off}   wake-poll: ${WAKE_POLL_US}us   cache: ${NO_CACHE:-0}=off"
[ "$MTP" = 1 ] && echo "  mtp    : ON (spec-type=draft-mtp, n_max=$MTP_N_MAX, ctx=$MTP_CTX)"

CACHE_ARG=""  # patched: use CGC_EXPERT_CACHE_BYTES env var instead
[ "${NO_CACHE:-0}" = 1 ] && CACHE_ARG=""
SEED_ARG=""
[ -n "$SEED" ] && SEED_ARG="-s $SEED"
CTX_ARG=""
[ "$CTX" != "0" ] && CTX_ARG="-c $CTX"
MTP_ARG=""
[ "$MTP" = 1 ] && MTP_ARG="--spec-type draft-mtp --spec-draft-n-max $MTP_N_MAX -c $MTP_CTX"
OUT=/tmp/n30cache.out; ERR=/tmp/n30cache.err
/usr/bin/time -l env "${ENVS[@]}" "$BIN" -m "$M" -n "$N" -ngl "$NGL" --no-mmap -t 8 \
    $SEED_ARG $CTX_ARG $MTP_ARG -p "$PROMPT" > "$OUT" 2> "$ERR"
RC=$?

echo "--- 結果 ---"
sed 's/\r/\n/g' "$ERR" | grep -oE "decoded *[0-9]+ tokens in *[0-9.]+ seconds?, *speed: *[0-9.]+ t/s" | tail -1 || true
grep -oE "hit rate [0-9.]+%" "$ERR" | tail -1 || true
grep -oE "cache (hits|misses)=[0-9]+" "$ERR" | tail -2 || true
# §MTP: accept rate + n_drafted + n_accept（speculative-simple 才有）
grep -oE "accept *= *[0-9.]+%" "$ERR" | tail -1 || true
grep -oE "n_(drafted|accept) *= *[0-9]+" "$ERR" | tail -2 || true
grep "maximum resident set size" "$ERR" | awk '{printf "RSS: %.2f GB\n", $1/1073741824}' || true
echo "--- 輸出前 80 字元 ---"
head -c 80 "$OUT" | tr '\n' ' '; echo
echo "exit=$RC"
