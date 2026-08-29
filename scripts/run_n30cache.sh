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
#             N30CACHE_PIN_PROFILE / N30CACHE_WAKE_POLL_US / N30CACHE_WARM
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# 2026-08-25: 跑 run 前清理上一輪殘留。segfault/OOM 可能留下 ~9.6GB 的 llama-simple（+掛住的 lldb），
# 佔滿 16GB unified 記憶體 → 新 run 直接 kIOGPUCommandBufferCallbackErrorOutOfMemory。
# 只 pkill 我們 fork 的 bin 路徑（不誤殺 IDE/其他 llama）。N30CACHE_NO_CLEAN=1 可跳過。
if [ "${N30CACHE_NO_CLEAN:-0}" != 1 ]; then
    for pat in "src/llama.cpp/build/bin/llama-simple" "src/llama.cpp/build/bin/llama-speculative-simple"; do
        pkill -9 -f "$pat" 2>/dev/null && echo "  [clean] killed stale $pat" || true
    done
fi
# 2026-08-25: BIN 指到 llama-src 新 build（含 CGC_DECODEHIT）；舊 root build 無。
BIN="$ROOT/src/llama.cpp/build/bin/llama-simple"
BIN_SPEC="$ROOT/src/llama.cpp/build/bin/llama-speculative-simple"
G4="${N30CACHE_G4:-$ROOT/models/gguf/gemma-4-26B-A4B-it-UD-IQ3_S.gguf}"
Q36="$ROOT/models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
# §MTP: qwen36 MTP 載體用 Nail model（blk.40 MTP head 為 UD-IQ3_XXS，與 trunk 同量）。
# 2026-08-25 實測：graft（blk.40 Q4_K）verify 126ms/token；Nail（blk.40 UD-IQ3_XXS）103ms/token
# 且 hit rate 更高（77.4% vs 72.1%）→ 較快。純 UD-IQ3_XXS 沒 blk.40，無法啟用 --spec-type draft-mtp。
Q36_MTP="$ROOT/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf"
# §MTP 2026-08-25 #3：head IQ2 A/B 載體（--head-iq2 / N30CACHE_HEAD_IQ2=1）。
# llama-quantize --tensor-type-file 只把 blk.40 MTP head 的 output.weight Q6_K→IQ2_S（其餘 752
# tensor byte-copy，bit-identical by construction）。獨立 option、預設 off：沒設 = 原 Q6_K head。
Q36_MTP_HEADIQ2="$ROOT/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-headIQ2.gguf"
# §MTP 2026-08-29：dense IQ4_XS 載體（--dense-iq4x / N30CACHE_DENSE_IQ4X=1），head 保留 Q6_K。
# 從 base IQ3_XXS 以 --tensor-type-file 重新量化：dense Q6_K（attn_*/ssm_out/ffn_*_shexp）→IQ4_XS、
# output.weight 釘 Q6_K（byte-copy）、其餘 tensor byte-copy（gen_denseiq4x_tt.py --keep-head）。
# 前代 headIQ2 版（27.33 t/s）出現確定性退化尾段（seed 1：1101 token 的最後 ~140 個 → 0000），
# head IQ2_S 為頭號嫌疑（target+draft 共用 lm_head）→ 重建為 Q6_K head。檔案 13020 MiB（+243 MiB）。
Q36_MTP_DENSEIQ4X="$ROOT/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"

MODEL="${N30CACHE_MODEL:-${1:-gemma4}}"
N=128
PROMPT=""
PROMPT_FILE=""
CTX="${N30CACHE_CTX:-0}"  # 0 = model default (4096), production uses 2048
BUDGET="${N30CACHE_BUDGET:-4294967296}"     # 4GiB（qwen36 8GiB 可覆寫）
WORKERS="${N30CACHE_WORKERS:-8}"            # pool8 甜蜜點（§8.12）
WAKE_POLL_US="${N30CACHE_WAKE_POLL_US:-15}"  # §8.51：15us 為生產設定（先前測試 0 為 off，已證偽）
# 優化 load（2026-08-25 對照實測，乾淨環境 + --no-mmap 下）：run 前把 model cat 進 page cache
#   load 9.04→7.08s（-22%）、prefill 1.50→1.16s、TOTAL 22→18s（-18%）。
#   注意：重開機後第一次 run 才需要（之後 page cache 已暖）；預設 off，--warm / N30CACHE_WARM=1 啟用。
WARM="${N30CACHE_WARM:-0}"
# CGC_DECODEHIT 是診斷 counter（decode hit rate，每 390 step 印一次，實測 98.08%），非 perf 設定，預設 off
DECODEHIT="${N30CACHE_DECODEHIT:-0}"
N_CB="${N30CACHE_N_CB:-8}"                    # §8.93：cb8 壓 p50（77→66ms），cb4 可覆寫
SEED="${N30CACHE_SEED:-}"                     # 未設 = llama-simple 預設；bit-identity 驗證請設固定值
LONG_PROMPT=0                                 # --long-prompt：產生確定性長 prompt（>1000 token）
STEADY=0                                      # --steady：固定 seed + 長 prompt + CGC_DECODEHIT + 長生成(-n 1100) + --ignore-eos
N_SET=0                                       # -n 顯式給過生成長度（--steady 才不會覆寫）
# --ignore-eos：越過 EOG 繼續生成（2026-08-25 實測 Qwen3 答完就 EOG，-n 1100 只出 71 token；
# 量 >1000 token steady-state t/s 必用）。llama-simple 2026-08-25 新增的 flag。
IGNORE_EOS=0
# §MTP: MTP draft context 多吃一份 GPU buffer，預設 ctx=2048 避免 OOM。
# 實測 -c 0（model 預設 4096）+ MTP 必 OOM（kIOGPUCommandBufferCallbackErrorOutOfMemory）。
# §MTP n_max：draft tokens 數，預設 2（graft blk.40 為 Q4_K，>2 accept rate 不增反降）。
MTP=0
MTP_CTX=3072
MTP_N_MAX="${N30CACHE_MTP_N_MAX:-2}"
# §MTP 2026-08-25 A/B：draft-only top-8→top-4（CGC_MTP_DRAFT_TOP4=1，只改 blk.40 MTP head 的
# routed experts，trunk 不變）。獨立 option、預設 off：沒設 = 原 top-8 draft（bit-exact）。
MTP_TOP4="${N30CACHE_MTP_TOP4:-0}"
# §MTP 2026-08-25 #3：head IQ2 A/B（載體切到 output.weight IQ2_S 的 Nail model）。獨立 option、
# 預設 off：沒設 = 原 Q6_K head（bit-exact）。
HEAD_IQ2="${N30CACHE_HEAD_IQ2:-0}"
# §MTP 2026-08-26 #1：dense IQ4_XS + head IQ2 A/B（載體切到 dense IQ4_XS / head IQ2_S 的 Nail model）。
# 獨立 option、預設 off：沒設 = 原 Q6_K head+dense（bit-exact）。與 --head-iq2 互斥（此已含 head IQ2）。
DENSE_IQ4X="${N30CACHE_DENSE_IQ4X:-0}"

usage() {
    echo "usage: $0 [-m gemma4|qwen36] [-n tokens] [-p prompt | --prompt-file F] [--ngl N] [--budget BYTES] [--pin-profile F] [--no-cache] [--warm] [--mtp [N]] [--mtp-top4] [--head-iq2] [--dense-iq4x] [--seed N] [--decodehit] [--long-prompt] [--ignore-eos] [--steady]
#   --warm 優化 load：run 前把 model cat 進 page cache（重開機後第一次 run 建議；load -22%）
#   --mtp [N] 啟用 MTP draft-mtp（僅 qwen36 有效，自動切到 graft model + speculative-simple binary，-c 2048 解 OOM）；
#             N 為 --spec-draft-n-max，預設 2；可用 N30CACHE_MTP_N_MAX 覆寫
#   --mtp-top4 啟用 MTP draft top-8→top-4 A/B（blk.40 MTP head 只路由 4 experts，trunk 不變；獨立、預設 off）
#   --head-iq2 啟用 MTP head IQ2 A/B（載體切到 output.weight IQ2_S 的 Nail model；獨立、預設 off）
#   --dense-iq4x 啟用 MTP dense IQ4_XS + head IQ2 A/B（載體切到 dense IQ4_XS / head IQ2_S 的 Nail
#             model；含 head IQ2，與 --head-iq2 互斥；獨立、預設 off）
#   --seed N 固定 seed（bit-identity / run-to-run 對照必設；等於 N30CACHE_SEED）
#   --decodehit 印 CGC_DECODEHIT（decode hit rate，每 390 step 一次）
#   --long-prompt 產生確定性長 prompt（>1000 token；同內容跨 run 完全一致，供對照）
#   --ignore-eos 越過 EOG 繼續生成（量 >1000 token steady-state 必用；--steady 自動開）
#   --steady 驗證模式：--seed 42 + --long-prompt + --decodehit + --ignore-eos + -n 1100（量 steady-state t/s + hit rate）；
#             --seed / -n 可覆寫
#   env: N30CACHE_N_CB / N30CACHE_SEED / N30CACHE_BUDGET / N30CACHE_NGL / N30CACHE_WORKERS / N30CACHE_MTP_N_MAX / N30CACHE_MTP_TOP4 / N30CACHE_HEAD_IQ2 / N30CACHE_DENSE_IQ4X / N30CACHE_WARM 可覆寫" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2 ;;
        -n) N="$2"; N_SET=1; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --ngl) N30CACHE_NGL="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --pin-profile) PIN_PROFILE="$2"; shift 2 ;;
        --no-cache) NO_CACHE=1; shift ;;
        --warm) WARM=1; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --decodehit) DECODEHIT=1; shift ;;
        --long-prompt) LONG_PROMPT=1; shift ;;
        --ignore-eos) IGNORE_EOS=1; shift ;;
        --steady) STEADY=1; shift ;;
        --mtp)
            MTP=1
            # 可選擇性接 N：--mtp 3
            case "${2:-}" in
                ''|*[!0-9]*) : ;;          # 下個不是數字，保持預設
                *) MTP_N_MAX="$2"; shift ;;
            esac
            shift
            ;;
        --mtp-top4) MTP_TOP4=1; shift ;;
        --head-iq2) HEAD_IQ2=1; shift ;;
        --dense-iq4x) DENSE_IQ4X=1; shift ;;
        -h|--help) usage ;;
        *) usage ;;
    esac
done

# --steady：steady-state 驗證模式 — 固定 seed + 長 prompt + CGC_DECODEHIT + 長生成（>1000 tok 讓 hit rate 升到頂）
if [ "$STEADY" = 1 ]; then
    [ -z "$SEED" ] && SEED=42
    LONG_PROMPT=1
    DECODEHIT=1
    IGNORE_EOS=1
    [ "$N_SET" = 0 ] && N=1100
    # steady 跑長時間（~1min+），間歇 Metal lost-wakeup 死鎖風險高 — 自動開 watchdog
    # （N30CACHE_WATCHDOG=0 顯式關閉）
    [ -z "${N30CACHE_WATCHDOG:-}" ] && N30CACHE_WATCHDOG=1
fi

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
            if [ "$DENSE_IQ4X" = 1 ]; then
                M="$Q36_MTP_DENSEIQ4X"
            elif [ "$HEAD_IQ2" = 1 ]; then
                M="$Q36_MTP_HEADIQ2"
            else
                M="$Q36_MTP"
            fi
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
# 2026-08-28: 暴露 N30CACHE_OA_ASYNC=0 覆寫供 MTP 路徑 A/B（預設仍 1 = 生產值）。
OA_ASYNC="${N30CACHE_OA_ASYNC:-1}"
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
# --long-prompt：產生確定性長 prompt（>1000 token）。循環一組不同句子（非單一段落重複）——
# 避免退化 prompt 觸發 model 立即 EOG（2026-08-25 實測：重複 fox 段落 → 只 decode 1 token 就結束）。
# 同內容跨 run 完全一致，供 bit-identity / steady-state 對照；實際 token 數以 perf print 為準。
if [ "$LONG_PROMPT" = 1 ]; then
    LONG_SENTS=(
        "The coastal observatory recorded steady winds from the northwest throughout the morning, and the tide charts suggested a calm crossing for the research vessel. "
        "Historical records indicate that the old lighthouse was rebuilt three times after storms damaged its foundations beyond repair. "
        "A team of engineers inspected the railway bridge, noting the corrosion on the lower girders and scheduling reinforcement work for the coming season. "
        "The museum's new exhibition traces the development of printing from wooden blocks to movable type and finally to industrial presses. "
        "Farmers in the valley reported an unusually abundant harvest, with the grain stores filling earlier than they had in a decade. "
        "The orchestra opened with a slow movement, and the woodwinds carried the melody while the strings provided a steady harmonic foundation. "
        "Geologists mapped the ancient riverbed, discovering fossilized shells that suggested the region was once covered by a shallow sea. "
        "The committee reviewed the proposal for the new library wing, debating the allocation of funds between reading rooms and digital archives. "
    )
    PROMPT=""
    i=0
    while [ ${#PROMPT} -lt 7000 ]; do
        PROMPT="${PROMPT}${LONG_SENTS[i % ${#LONG_SENTS[@]}]}"
        i=$((i + 1))
    done
fi
[ -n "$PROMPT" ] || { echo "error: need -p prompt or --prompt-file" >&2; exit 2; }
# --no-cache：真正關閉 expert cache。cache 由 env CGC_EXPERT_CACHE_BYTES 驅動
#（llama-simple 只認 env，CLI -expert-cache 是 MTP/speculative-simple 專用）→ NO_CACHE 時 BUDGET=0。
[ "${NO_CACHE:-0}" = 1 ] && BUDGET=0

# 生產 env 集（全部有 §8 出處；不設則對應行為 off）
ENVS=(LLAMA_EXPERT_CACHE_ALLOW_NGL=1
      CGC_EXPERT_CACHE_BYTES=$BUDGET
      LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
      LLAMA_EXPERT_CACHE_WORKERS=$WORKERS
      CGC_WAKE_POLL_US=$WAKE_POLL_US
      # §8.6x hist prefetch（08-26 回歸移除，08-27 恢復）：CGC_PREFETCH_SRC=hist = rolling window
      # (CGC_PREFETCH_WINDOW, 預設 4) 的近期 step union → 下一個 decode 的 ensure_batch 多命中，
      # 冷 miss 不再卡 hook。CGC_EVICTED_RING=0 = 停用 evicted-ring re-prefetch（A/B 定案：省 ring
      # 開銷，純 LRU）。兩者都只對非 MTP decode 生效。
      CGC_PREFETCH_SRC=hist
      CGC_EVICTED_RING=0)
[ "$OA_ASYNC" = 1 ] && ENVS+=(CGC_OA_ASYNC=1)
ENVS+=(CGC_N_CB=$N_CB)
ENVS+=(CGC_GLU_FUSED_DOWN=1)  # §8.113: fused gate+up+GLU+down, +6.5% speed
# §CGC 2026-08-28 WIN_PIN（LLAMA_EXPERT_CACHE_WIN_PIN=K）：window pin — 每層保留最近 K 個
# miss-step union 的 expert 不被 LRU 逐出。A/B 結論（steady MTP 4GiB seed 1，2026-08-28）：
# K=4 hit 51.0%、K=2 hit 51.4%（baseline 51.5%）→ 全 K 值中性無效。根因：steady 的 miss
# 為 chunked prefill 的結構性 cold routing（跨 chunk 路由重疊極低，非 recurring hot
# eviction）；decode 期 93% 步走 fast path（touch+ZERO，不 fill），ensure_batch 只在
# prefill/catch-up。且 pread 16.28s 為 8-worker 平行累計值 → 實際 prefill wall ~1-2s
# （總時間 <4%）→ miss rate 對可感知速度影響極小。保留代碼供後續 prefill I/O 優化
# 復用；預設 off（=舊純 LRU，bit-identical 已驗證）。A/B 用 N30CACHE_WIN_PIN=K 傳入。
if [ -n "${N30CACHE_WIN_PIN:-}" ]; then
    ENVS+=(LLAMA_EXPERT_CACHE_WIN_PIN=$N30CACHE_WIN_PIN)
fi
[ "$DECODEHIT" = 1 ] && ENVS+=(CGC_DECODEHIT=1)
[ -n "${PIN_PROFILE:-}" ] && ENVS+=(LLAMA_EXPERT_CACHE_PIN_PROFILE="$PIN_PROFILE")
# §CGC 2026-08-28: per-step miss timeline（llama-context.cpp STEP_DBG）：每 20 步打印
# 累計 fast-path union/cold + ensure req/hit，判定 miss 為前期集中（cold start → prewarm
# 修復有效）還是 steady churn（結構性 → prewarm 無效）。診斷用，不影響行為。
if [ -n "${N30CACHE_STEP_DBG:-}" ]; then
    ENVS+=(LLAMA_EXPERT_CACHE_STEP_DBG=1)
fi
# §CGC 2026-08-29 merge-read A/B（LLAMA_EXPERT_CACHE_NO_MERGE=1 回退 per-segment pread）
if [ -n "${N30CACHE_NO_MERGE:-}" ]; then
    ENVS+=(LLAMA_EXPERT_CACHE_NO_MERGE=1)
fi
# §CGC 2026-08-29 F_RDADVISE A/B（LLAMA_EXPERT_CACHE_NO_RDADVISE=1 關閉 read-ahead 提示）
if [ -n "${N30CACHE_NO_RDADVISE:-}" ]; then
    ENVS+=(LLAMA_EXPERT_CACHE_NO_RDADVISE=1)
fi
# §CGC 2026-08-29 deadlock watchdog（CGC_WATCHDOG=1：Metal completion 停滯 >10s 時 dump 所有
# command buffer 狀態並在 60s 取樣寬限後 abort；診斷間歇性 prefill 死鎖用，預設 off）
# steady 模式自動開（見上方 --steady 段）；N30CACHE_WATCHDOG=0 顯式關閉
if [ -n "${N30CACHE_WATCHDOG:-}" ] && [ "$N30CACHE_WATCHDOG" != "0" ]; then
    ENVS+=(CGC_WATCHDOG=1)
    [ -n "${N30CACHE_WATCHDOG_MS:-}" ] && ENVS+=(CGC_WATCHDOG_MS=$N30CACHE_WATCHDOG_MS)
fi
# §CGC 2026-08-29 LAYER_CAPS（LLAMA_EXPERT_CACHE_LAYER_CAPS="start-end:cap;..."）：
# per-layer slot 容量。MISS_DUMP 定案：steady 的 miss 79.3% 集中於 layer 40（MTP draft 層，
# il == n_layer()）—— draft 層 routing 涵蓋 250 distinct experts，遠超 71 slots 的
# LRU 抖動區；trunk 1-39 層各僅 ~60 miss（健康）。40-40:256 = draft 層全常駐。
# A/B（重開機 ABBA 4 跑，2026-08-29）：misses 11574→2622（-77%）、draft cold 32%→0%、
# file_reads -79%、accept 98.919→99.728%（+0.8pt）、速度 +0.8 t/s（ON 兩對全勝）。
# MTP steady profile 預設啟用；N30CACHE_LAYER_CAPS=0 顯式關閉回 uniform 71（A/B 用）。
# LFU-aging 置換同日定案中性無效（fast-path cold 64.0% 一 token 不動）——置換策略
# 槓桿關閉，LAYER_CAPS 是容量「重分配」而非擴充（pool 總大小不變，僅層間移 185 slots）。
if [ -n "${N30CACHE_LAYER_CAPS:-}" ]; then
    if [ "$N30CACHE_LAYER_CAPS" = "0" ]; then
        : # explicit OFF for A/B
    else
        ENVS+=(LLAMA_EXPERT_CACHE_LAYER_CAPS=$N30CACHE_LAYER_CAPS)
    fi
elif [ "$MTP" = "1" ] && [ "$STEADY" = "1" ]; then
    ENVS+=(LLAMA_EXPERT_CACHE_LAYER_CAPS="40-40:256")
fi
# §CGC 2026-08-29 routing-aware placement（2-pass oracle）：
#   pass 1: N30CACHE_ROUTE_RECORD=1 N30CACHE_ROUTE_DUMP=<file> 跑 steady → dump per-layer
#           top-K 路由頻率表（K = usable slots，LAYER_CAPS 感知）+ coverage 判定
#           （測量定案：oracle top-71 coverage mean 67.0% vs LRU live 35.6%，重尾確認）
#   pass 2: N30CACHE_PIN_PROFILE=<file> 餵回 → ensure_batch 填入的 pin 成員標記
#           slot_pinned_static（永不逐出）；層滿釘後 tail 專家 ZERO-mapped（pin_skip）
#           不再死等。預設 off。
if [ -n "${N30CACHE_ROUTE_RECORD:-}" ]; then
    ENVS+=(LLAMA_EXPERT_CACHE_ROUTE_RECORD=1)
    [ -n "${N30CACHE_ROUTE_DUMP:-}" ] && ENVS+=(LLAMA_EXPERT_CACHE_ROUTE_DUMP=$N30CACHE_ROUTE_DUMP)
fi
if [ -n "${N30CACHE_PIN_PROFILE:-}" ]; then
    ENVS+=(LLAMA_EXPERT_CACHE_PIN_PROFILE=$N30CACHE_PIN_PROFILE)
fi
# §CGC 2026-08-29 P1-3a 融合派發（CGC_MMV_FUSE=1：MoE gate+up+swiglu 三連融合 GEMV kernel，
# bit-identical 已驗證；預設 off。N30CACHE_MMV_FUSE_DBG=1 開 OPSEQ/MMV_FUSE debug 探針）
if [ -n "${N30CACHE_MMV_FUSE:-}" ] && [ "$N30CACHE_MMV_FUSE" != "0" ]; then
    ENVS+=(CGC_MMV_FUSE=1)
    [ -n "${N30CACHE_MMV_FUSE_DBG:-}" ] && ENVS+=(CGC_MMV_FUSE_DBG=1)
fi
if [ -n "${N30CACHE_WATCHDOG_CAPTURE:-}" ]; then
    ENVS+=(CGC_WATCHDOG=1 CGC_WATCHDOG_CAPTURE=1)
fi
# §CGC 2026-08-29 死鎖緩解實驗（CGC_DRAIN_EVERY=K）：每 K 個 graph_compute 全 queue 排水
# （waitUntilCompleted），強制 kernel command queue 週期性回收；預設 off
if [ -n "${N30CACHE_DRAIN_EVERY:-}" ]; then
    ENVS+=(CGC_DRAIN_EVERY=$N30CACHE_DRAIN_EVERY)
fi

echo "=== n30cache production run ==="
echo "  model  : $MODEL ($(basename "$M"))"
echo "  ngl    : $NGL   budget: $((BUDGET/1073741824))GiB   workers: $WORKERS"
echo "  pin    : ${PIN_PROFILE:-off}   wake-poll: ${WAKE_POLL_US}us   cache: ${NO_CACHE:-0}=off   warm: $WARM"
echo "  decodehit: $DECODEHIT"
echo "  seed   : ${SEED:-default}   long-prompt: $LONG_PROMPT   ignore-eos: $IGNORE_EOS   steady: $STEADY   gen: $N"
[ "$MTP" = 1 ] && echo "  mtp    : ON (spec-type=draft-mtp, n_max=$MTP_N_MAX, ctx=$MTP_CTX, top4=$MTP_TOP4)"

SEED_ARG=""
[ -n "$SEED" ] && SEED_ARG="-s $SEED"
IGNORE_EOS_ARG=""
[ "$IGNORE_EOS" = 1 ] && IGNORE_EOS_ARG="--ignore-eos"
CTX_ARG=""
[ "$CTX" != "0" ] && CTX_ARG="-c $CTX"
MTP_ARG=""
# §MTP: speculative-simple 用 common_params_parse，只認 CLI -expert-cache（不讀 CGC_EXPERT_CACHE_BYTES
# env——那是 llama-simple 的私有 patch）。MTP 不走 env 會回到全權重（weights ~8118 MiB）→ GPU OOM。
# §MTP: --temp 0 對齊 base 臂（llama-simple 為 greedy）。MTP 預設 temp sampling 會讓 target 採樣
# 偏離 draft argmax → accept 掉到 ~31%（target sample != draft argmax）→ 多 reject + checkpoint
# restore → 速度砍半（3.6 vs 6.1 t/s）。greedy 下 draft head 與 target argmax 一致 → accept 87.5%。
[ "$MTP" = 1 ] && MTP_ARG="--spec-type draft-mtp --spec-draft-n-max $MTP_N_MAX -c $MTP_CTX -expert-cache $BUDGET --temp 0"
# §MTP 2026-08-27：async prefetch bg thread（B-section hist prefetch）在 GPU 執行 MTP verify 時
# 填/逐 slots，覆寫 GPU 仍在讀的 slot → 輸出壞 + accept 掉（08-22 實測）。MTP 一律關 prefetch
# （CGC_NO_PREFETCH=1），基於歷史約束 CGC_PREFETCH_OFF 的同一目的。
if [ "$MTP" = 1 ]; then
    ENVS+=(CGC_NO_PREFETCH=1)
fi
# §MTP 2026-08-25（獨立 option，CGC_VERIFY_DECODE=1）：verify 走 decode fast path（ZERO-slot，無
# 同步 pread stall）。實測 verify 64→35.4ms/token、MTP 13.9→25.2 t/s（=base decode rate）、
# accept 97.8%→95.7%、輸出仍正確（deterministic）。此 env 是 MTP 專屬 opt-in：沒設 = 原 exact-load
# verify（bit-exact、較慢）。N30CACHE_MTP_VERIFY_DECODE=0 可關掉。
if [ "$MTP" = 1 ] && [ "${N30CACHE_MTP_VERIFY_DECODE:-1}" = 1 ]; then
    ENVS+=(CGC_VERIFY_DECODE=1)
fi
# §MTP 2026-08-25（獨立 option，CGC_DRAFT_DECODE=1）：draft 也走 decode fast path（touch + ZERO-slot，
# 無同步 pread），與 verify 同 pool residency → draft/verify 冷專家同處歸零 → accept 不掉反升
# （95.7%→97.1%）。速度持平（25.4 t/s）：step-timing 證實 draft 已非瓶頸（verify 在 bandwidth floor、
# blk.40 head 是真 GPU compute、CPU 已全藏 GPU 後）→ 27-28 t/s 需砍 MTP head 成本（model 側）。
# N30CACHE_MTP_DRAFT_DECODE=0 可關掉。
if [ "$MTP" = 1 ] && [ "${N30CACHE_MTP_DRAFT_DECODE:-1}" = 1 ]; then
    ENVS+=(CGC_DRAFT_DECODE=1)
fi
# §MTP 2026-08-25 A/B（獨立 option，CGC_MTP_DRAFT_TOP4=1）：blk.40 MTP head 的 routed experts
# 8→4（draft-only，trunk 不變）→ draft weight-read 減半。預設 off：沒設 = 原 top-8 draft（bit-exact）。
# 用途：量 top-4 draft 對 accept / t/s / 輸出的影響（品質成本）。N30CACHE_MTP_TOP4=1 或 --mtp-top4 開啟。
if [ "$MTP" = 1 ] && [ "$MTP_TOP4" = 1 ]; then
    ENVS+=(CGC_MTP_DRAFT_TOP4=1)
fi
# 優化 load：先 cat model 進 page cache（重開機後第一次 run 建議），之後 loader 的 read 全 RAM-speed
if [ "$WARM" = 1 ]; then
    echo "  warm   : pre-loading $(basename "$M") into page cache..."
    /usr/bin/time -l sh -c "cat '$M' > /dev/null" 2>&1 | grep -E "real" | sed 's/^/    /'
fi
OUT=/tmp/n30cache.out; ERR=/tmp/n30cache.err
/usr/bin/time -l env "${ENVS[@]}" "$BIN" -m "$M" -n "$N" -ngl "$NGL" --no-mmap -t 8 \
    $SEED_ARG $IGNORE_EOS_ARG $CTX_ARG $MTP_ARG -p "$PROMPT" > "$OUT" 2> "$ERR"
RC=$?

echo "--- 結果 ---"
sed 's/\r/\n/g' "$ERR" | grep -oE "decoded *[0-9]+ tokens in *[0-9.]+ seconds?, *speed: *[0-9.]+ t/s" | tail -1 || true
# load / prefill / decode 指標（llama_perf_context_print: load time / prompt eval time / eval time / total time）
sed 's/\r/\n/g' "$ERR" | grep -E "llama_perf_context_print:" | tail -4 || true
grep -oE "hit rate [0-9.]+%" "$ERR" | tail -1 || true
grep -oE "CGC-DECODEHIT: decode hit [0-9.]+% \([0-9]+/[0-9]+\)" "$ERR" | tail -1 || true
grep -oE "cache (hits|misses)=[0-9]+" "$ERR" | tail -2 || true
# §MTP: accept rate + n_drafted + n_accept（speculative-simple 才有）
grep -oE "accept *= *[0-9.]+%" "$ERR" | tail -1 || true
grep -oE "n_(drafted|accept) *= *[0-9]+" "$ERR" | tail -2 || true
grep "maximum resident set size" "$ERR" | awk '{printf "RSS: %.2f GB\n", $1/1073741824}' || true
echo "--- 輸出前 80 字元 ---"
head -c 80 "$OUT" | tr '\n' ' '; echo
echo "exit=$RC"
