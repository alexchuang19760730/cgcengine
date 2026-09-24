#!/usr/bin/env bash
# [CGC ρ-fill A/B 2026-09-23] 把「量到的 ρ」變成「真的提前發起 IO」，然後量交付 cell 的 t/s。
#
# 前面幾輪量到的東西（都是實測，不是估的）：
#   cov_uni = 0.854~0.860   —— 影子 router（pre-attn 殘差走同一個 gate）能蓋住真實 union 的比例
#   lead    = 1.31~2.27 ms  —— 影子節點在 segment 的 45.1%，領先真實 argsort 的量
#   cb      = 42~51 ms/步   —— 交付 cell 穩態下 expert fill 的 IO 總量
# 這支腳本要回答的是那三個數字**換成真的 t/s 是多少**、以及那 4.76 ms/步的插入成本
# （圖裡多一個 norm + 一個 matmul）是否把它吃回去。
#
# 四臂（缺一不可，因為 ρ 臂同時動了兩件事：圖裡多節點、以及提前發起 IO）：
#   base      什麼都不加 —— 錨點那一格
#   rho       CGC_RHO_PROBE + CGC_RHO_FILL —— 影子 router + 用它提前 prefetch
#   probe     CGC_RHO_PROBE only —— 隔離「圖裡多一個 matmul」的成本（ρ 的淨收益要減掉它）
#   seg       只細分段 —— 隔離「多 command buffer」的成本
#
# ⚠ 交替（ABAB）跑，不要先跑完一臂再跑下一臂：單臂噪音 ±27%（2026-09-23 實測），
#   而熱浸會讓「後跑的那一臂」系統性偏低。
set -u

REPO=/Users/alexchuang/Documents/flashkv-devserver
PY=/Users/alexchuang/.workbuddy/binaries/python/versions/3.13.12/bin/python3
ROUNDS="${ROUNDS:-2}"
REPS="${REPS:-1}"
TAG="${TAG:-$(date +%H%M%S)}"
EXTRA_SEG="${SEG:-1}"   # SEG=1 ⇒ ρ 臂加細分段（CGC_N_CB=16 + CGC_CB_N_MAIN=1）
# [CGC 2026-09-24] PIN_PROFILE 靜態釘住的 profile（§EN-476 實測：由 --prompt 0 的
# ROUTE_DUMP 產生，5680 專家在 load 時填好並 static pin）。
PIN_PROFILE="${PIN_PROFILE:-${REPO}/scripts/check/pin_profiles/route_top142_p0_2026-09-23.txt}"

# 交付 cell：prod-new 統一介面（2026-09-23 拍板）——每臂自己帶 arm spec：
#   off（MTP off 交付口徑基準）vs on（MTP on）vs ρ 系列（MTP on + 影子 router + 提前 fill）。
# 形狀與原 prod25-stream 交付格一致（--warm-skip 64 跳過池預熱段）。
CELL=(--reps "${REPS}" --prompt 0 --gen 128 --depths 512 \
      --batch 512 --ctx-size 4096 --warm-skip 64)

# 視窗閘門：任何 listener 或別條線的量測行程在跑就停手（建置產物與 GPU 都是共用的）。
# SKIP_WINDOW_CHECK=1 跳過（用戶明確插隊時用；結果要標記「並行污染風險」）。
if [ -z "${SKIP_WINDOW_CHECK:-}" ]; then
LISTEN="$(lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null | tail -n +2)"
BUSY="$(pgrep -x llama-server; pgrep -x llama-bench; \
        pgrep -fl 'run_ids_dst_capture|decode_sweep|prod_profile|window_sentinel|llama_bench_matrix')"
if [ -n "${LISTEN}" ] || [ -n "${BUSY}" ]; then
    echo "ABORT: 視窗忙 listener=[${LISTEN}] busy=[${BUSY}]"
    exit 1
fi
fi

# [CGC 2026-09-23] 每一臂之前都要過 prod_profile.py 的同一套閘門，否則量到的不是這個 cell 的
# 數字。這個教訓是實測買來的：第一版這支腳本只有上面的「有沒有人在跑」檢查，沒有冷卻，
# 結果 base 臂讀到 9.47（錨點 12.57）—— 我當時把它歸給「單臂噪音 ±27%」，但真正的成因是
# 前面連續跑了 6 趟 GPU 沒散熱。交替順序只擋得住「熱浸偏在後跑的那一臂」，擋不住整批都熱。
# 閘門口徑與 prod_profile.py 逐字相同（複用它自己 import 的那兩個實作，不長第二份）：
#   mem usable >= MIN_USABLE %，且 wait_nominal 等到 NOMINAL；等不到就**拒跑**，不是照跑。
MIN_USABLE="${MIN_USABLE:-30.0}"
COOLDOWN_TIMEOUT="${COOLDOWN_TIMEOUT:-420.0}"
POLL="${POLL:-5.0}"
# [CGC 2026-09-23 18:5x] swap 閘門從「> 0 拒跑」改成「> MAX_SWAP_MIB 拒跑」。
#   「> 0」在重開機後 11 分鐘就變成永遠拒跑：實測 reboot 後 6 分鐘 swap 已 1378 MiB、
#   11 分鐘 1962 MiB（macOS 常態換頁，不等於 expert pool 被換出）⇒ 12 格全 REFUSE，
#   整批 A/B 一格都沒跑。真正崩掉的那一格是 swap 5301 MiB（hit% 68.7→57.2、base 12.57→4.62）。
#   所以門檻設在兩個實測點中間：2048 MiB。並且**每臂都把 swap 印出來**（下面的 printf），
#   讓「base 與 rho 是不是在同一個 swap 狀態下比的」這件事可以被複核，而不是靠閘門背書。
MAX_SWAP_MIB="${MAX_SWAP_MIB:-2048}"
gate() {
    "${PY}" - "${MIN_USABLE}" "${COOLDOWN_TIMEOUT}" "${POLL}" "${MAX_SWAP_MIB}" <<'PYEOF'
import sys
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv-devserver/scripts/check')
import thermal_pressure as tp
from prefill_certifiability import mem_state
from profile_duo import wait_nominal

min_usable = float(sys.argv[1]); timeout = float(sys.argv[2]); poll = float(sys.argv[3])
max_swap = float(sys.argv[4])
mem = mem_state()
if mem["usable_pct"] < min_usable:
    print("REFUSE mem usable %.1f%% < %.1f%%" % (mem["usable_pct"], min_usable))
    sys.exit(1)
# [CGC skill 第 61 條] swap 才是殺手，熱壓 NOMINAL 完全擋不住它：實測 swap 5301 MiB 時
# hist 192/192 全是 NOMINAL，而 base 從 12.57 掉到 4.62（pool 被換出 ⇒ hit% 68.7→57.2）。
# 門檻（不是 0）：重開機後 macOS 常態就會累積 1.4~2.0 GiB，而那一格量到的 base 是 11.97
# （錨點 12.57）⇒ 1.4~2.0 GiB 是可比的；5301 MiB 不是。所以門檻取 2048 MiB。
sw = float(mem.get("swap_used_mib", 0.0))
if sw > max_swap:
    print("REFUSE swap %.0f MiB > %.0f (pool 被換出，hit%% 崩)" % (sw, max_swap))
    sys.exit(1)
w = wait_nominal(timeout, poll, quiet=True)
if not w["ok"]:
    print("REFUSE still %s after %.0fs (timeout %.0fs)" % (w["label"], w["waited_s"], timeout))
    sys.exit(1)
print("gate %-8s waited %3.0fs  mem %.1f%%  swap %.0f" % (w["label"], w["waited_s"], mem["usable_pct"], sw))
PYEOF
}

arm_spec() {
    case "$1" in
        base)  echo "prod-new";;                                  # MTP off 交付口徑基準
        on)    echo "prod-new:CGC_SERVER_MTP=1";;                # MTP on 基準（無 ρ）
        # [CGC 2026-09-24] PIN A/B 走 **prod25-stream**（= prod25 + PREFILL_STREAM +
        # GATHER_SLAB_CAP=256），這是 prod_profile.py 的 anchor arm，也是昨晚實跑的那一格。
        # 注意 prod_profile.py --profile 只收 server profile 名（prod25），它的 decode 軸
        # 用的是 args.profile ⇒ 拿不到 prod25-stream ⇒ 這裡直接餵 registry 名。
        pbase) echo "prod25-stream";;
        ppin)  echo "prod25-stream";;
        *)     echo "prod-new:CGC_SERVER_MTP=1";;                # ρ 系列全在 MTP on
    esac
}

arm_spec_flag() {
    case "$1" in
        base)  echo "";;                         # MTP off：不傳 spec-type（與 run_server.sh 1345 一致）
        *)     echo "--spec-type draft-mtp";;    # MTP on：draft-mtp
    esac
}

arm_env() {
    case "$1" in
        base)  echo "";;
        on)    echo "";;
        # [CGC 2026-09-24] PIN_PROFILE 靜態釘住 A/B（GAP_ELIMINATION_PLAN 第 1 步）。
        # pbase/ppin 都在 **prod25-stream + MTP on**（交付 cell）上，唯一差別是下面這個 env。
        # 不加 CGC_MASSCOV：要讓兩臂的環境差異只有 PIN_PROFILE 一個變數。
        pbase) echo "";;
        ppin)  echo "LLAMA_EXPERT_CACHE_PIN_PROFILE=${PIN_PROFILE}";;
        rho)   if [ "${EXTRA_SEG}" = "1" ]; then
                   echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1 CGC_N_CB=16 CGC_CB_N_MAIN=1"
               else
                   echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1"
               fi;;
        probe) if [ "${EXTRA_SEG}" = "1" ]; then
                   echo "CGC_RHO_PROBE=1 CGC_N_CB=16 CGC_CB_N_MAIN=1"
               else
                   echo "CGC_RHO_PROBE=1"
               fi;;
        rhons) echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1";;   # ρ 但不細分段：隔離「細分段」這個變量
        # [CGC 2026-09-23 fill 空轉] ρ 相同、只有「背景 prefetch 怎麼讀」不同：
        #   rho    = 新路徑（合併成 preadv ＋ bg 執行緒內聯，見 fill_segments_merged_serial）
        #   rholeg = 舊路徑（thread-per-segment ＋ 未合併的 0.04 MiB 小讀）
        # 這一對是「fill 空轉」修復本身的 A/B：兩臂 ids 完全相同，差別只在讀的形狀。
        rholeg) echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1 CGC_PREFETCH_LEGACY_FILL=1";;
        # [CGC 2026-09-23 rho fuse] ρ + CGC_RHO_PREFETCH_MAXQ —— 背景預取佇列深度上限。
        # §EN-471: ρ 的 80k 個 41KB 小讀把 fill_wait 頂到 15-20s。這組掃 MAXQ：
        # 若 t/s 隨 MAXQ 變小而回歸（且 #13 maxq_limit > 0），干涉是成因 ⇒ 按層批次化值得做。
        # [CGC 2026-09-24 甜點上掃] GAP_FIX_EXEC §4 第 1 步：09-23 只掃了 q4/q8/q16 且單調
        # 遞增（16 > 8 > 4）⇒ 甜點很可能在 16 以上。而「按層批次化」進樹後 MAXQ 的語意從
        # 「in-flight slots」變成「in-flight batches」（每層 1 個，40 層模型）⇒ 16 很可能偏小。
        # 補 q24 / q48 把單調性往右延伸：若 q32 或 q48 明顯高於 q16 ⇒ 甜點要重定。
        rho-q48) echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1 CGC_RHO_PREFETCH_MAXQ=48";;
        rho-q32) echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1 CGC_RHO_PREFETCH_MAXQ=32";;
        rho-q24) echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1 CGC_RHO_PREFETCH_MAXQ=24";;
        rho-q16) echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1 CGC_RHO_PREFETCH_MAXQ=16";;
        rho-q8)  echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1 CGC_RHO_PREFETCH_MAXQ=8";;
        rho-q4)  echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1 CGC_RHO_PREFETCH_MAXQ=4";;
        # [CGC 2026-09-23 判別] MAXQ=16 + legacy fill：rho-q16 已實測 9.64（干涉被保險絲壓住）。
        # 若 rhoq16-leg（thread-per-segment 小讀）也 ~9.6 ⇒ 干涉是唯一問題、MAXQ 就是答案，
        # 按層批次化（保留完整預取）不值得寫；若掉回 <9 ⇒ fill 形狀在無塞車時仍有代價。
        rhoq16-leg) echo "CGC_RHO_PROBE=1 CGC_RHO_FILL=1 CGC_RHO_PREFETCH_MAXQ=16 CGC_PREFETCH_LEGACY_FILL=1";;
        seg)   echo "CGC_N_CB=16 CGC_CB_N_MAIN=1";;
        *)     echo "";;
    esac
}

ARMS="${ARMS:-base on rho rho-q16 rho-q8 rho-q4}"

echo "== ρ-fill A/B tag=${TAG} rounds=${ROUNDS} reps=${REPS} seg=${EXTRA_SEG} arms=[${ARMS}] =="
echo
# ── [CGC 2026-09-24] 超訂預檢閘門（docs/SWAP_MISS_LINK_2026-09-24.md §7「立即可做」）──
# pool 8 GiB + load_mode=none 在 16 GB 上是**靜態超訂** 4838 MiB（13030 + 8192 = 21222 > 16384，
# 與實測 resident 21222 MiB 對上）⇒ 這種配置跑出來的數字活在壓縮 + swap 之上，不可引用。
# 預設（BUDGET_GATE=strict）直接 exit 2 拒跑，不再造出污染樣本；要跑請
#   BUDGET_GATE=warn  → 放行但 export CGC_BUDGET_OVERSUBSCRIBED=1（樣本帶標記，不可當乾淨基線）
#   BUDGET_GATE=off   → 完全不檢查（相容既有流程）
#   或把 pool 降到 ≤ 3 GiB（16384 − 13030 = 3354 MiB）。
. "${REPO}/scripts/check/budget_gate.sh"
echo "[budget] gate: ${CGC_BUDGET_OVERSUBSCRIBED:+OVERSUBSCRIBED-ACK }BUDGET_GATE=${BUDGET_GATE:-strict}"

# 熱浸是**單調累積**的，不是隨機的：實測 base 連跑兩趟 11.97 → 9.92（−17%），而 thermal
# key 兩趟都報 NOMINAL。若每輪都用同一個順序，永遠是後跑的那一臂在付熱浸 ⇒ ρ 會被
# 系統性地判慢。所以奇數輪正序、偶數輪逆序（ABBA），讓熱浸的線性分量在各臂間平均分配。
reverse() {
    local out=""
    for a in "$@"; do out="${a} ${out}"; done
    echo "${out}"
}

for r in $(seq 1 "${ROUNDS}"); do
    if [ $((r % 2)) -eq 0 ]; then
        # shellcheck disable=SC2086
        ORDER="$(reverse ${ARMS})"
    else
        ORDER="${ARMS}"
    fi
    for a in ${ORDER}; do
        wd="/tmp/rho_ab_${TAG}/${a}_r${r}"
        mkdir -p "${wd}"
        # 每一臂之前過閘門（mem + 等到 NOMINAL）。等不到就跳過這一臂，不留下一個
        # 穿著 NOMINAL 外衣的熱機器數字。
        g="$(gate)"
        case "${g}" in
            REFUSE*)
                printf "r%d  %-6s  REFUSED: %s\n" "${r}" "${a}" "${g}"
                echo "${g}" > "${wd}/refused.txt"
                continue;;
        esac
        envs="$(arm_env "${a}")"
        arm="$(arm_spec "${a}")"
        spec="$(arm_spec_flag "${a}")"
        # shellcheck disable=SC2086
        env ${envs} "${PY}" "${REPO}/scripts/check/llama_bench_matrix.py" --arms "${arm}" \
            ${spec} "${CELL[@]}" --workdir "${wd}" > "${wd}/driver.log" 2>&1
        ts="$("${PY}" -c "
import json, glob, sys
fs = sorted(glob.glob('${wd}/llama_bench*.json'))
if not fs:
    print('NA'); sys.exit()
d = json.load(open(fs[0]))
vals = [r.get('avg_ts') for r in d if isinstance(r, dict) and r.get('avg_ts')]
print('%.2f' % (sum(vals)/len(vals)) if vals else 'NA')
")"
        pf="$(grep -h 'CGC-RHO-FILL' "${wd}"/*.stderr.log 2>/dev/null | tail -1 | sed 's/.*per_layer=//')"
        th="$(grep -c 'HEAVY\|SERIOUS' "${wd}/driver.log" 2>/dev/null)"
        # [CGC 2026-09-23 19:0x] 原本寫 `^cache: hit`，但 driver.log 那一行是縮排過的
        # （`  cache: hit 58.9% ...`）⇒ 每一行都抓不到，hit% 全印 NA，而「hit% 崩」正是
        # 我們要監看的那一件事。改成容許前導空白。
        hit="$(grep -hE '^[[:space:]]*cache: hit' "${wd}/driver.log" 2>/dev/null | tail -1 | sed 's/.*hit //;s/%.*//')"
        swp="$(printf '%s' "${g}" | sed 's/.*swap //')"
        printf "r%d  %-6s  t/s=%-8s  hit%%=%-7s  rho_per_layer=%-8s  swap=%-8s | %s\n" \
               "${r}" "${a}" "${ts}" "${hit:-NA}" "${pf:-none}" "${swp}" "${g}"
    done
done

echo
echo "讀法：先比 base vs probe（插入成本），再比 probe vs rho（ρ 的淨收益）。"
echo "      seg 臂是用來扣掉細分段自己的成本 —— 若 rho 加了細分段，淨收益要減掉 (seg - base)。"
