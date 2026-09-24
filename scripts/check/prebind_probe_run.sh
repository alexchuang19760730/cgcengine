#!/bin/bash
# prebind_probe_run.sh -- 跑一次帶 `CGC-PREBIND-PROBE` 的量測，把 stderr 留下來。
#
# 為什麼不直接叫 `prod_profile.py`：它用 `subprocess.run(..., capture_output=True)`，
# 而 probe 印的是 **stderr**，會被吞掉（它只把 llama-bench 自己的 --json row 讀回來）。
# 所以這裡直接叫 `llama_bench_matrix.py`（prod_profile.py 自己也是叫它 —— 同一條路徑，
# 同一組 knobs），只是把 `--workdir` 指到自己的目錄：它會把 child 的 stderr 寫成
# `<stem>.stderr.log`，probe 的逐層行就在那裡。
#
# ★ 為什麼不自己組 llama-bench 命令（踩過的兩個坑，都寫在這裡免得再踩）：
#   1. `prod_profile.py` 的 DECODE_SHAPE 用的是 `--prompt/--gen/--depths/--batch/--reps`，
#      那是 **llama_bench_matrix 的**參數名，不是 llama-bench 的。llama-bench 自己認的是
#      `-p/--n-prompt`、`-n/--n-gen`、`-d/--n-depth`、`-b/--batch-size`、`-r/--repetitions`。
#      字面搬過去會得到 `error: invalid parameter for argument: --prompt`，0.4 s 印完 help 退出。
#      ⇒ 參數名的轉換是 llama_bench_matrix 的工作，不要自己來。
#   2. 自己指定 `-b 512` 會撞 `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`：
#      p0 + d512 + n128 = 640 > 512。arm 自己的 -b/-ub（prefill250 = 5632）才是能跑的。
#      ⇒ n_batch 由 arm 決定（`--batch` 只在真的要覆寫時才傳）。
#
# 三道閘門（缺一不跑）—— 檢查與動作在**同一個分支裡**，不分家：
#   1. 港埠 8080 沒有 listener
#   2. 沒有別的 llama-server / llama-bench 行程（建置產物與 GPU 是共用資源）
#   3. usable memory >= --min-usable（預設 30%，對齊 prod_profile.py）
# 閘門 1/2 會**等待**到 `CGC_PREBIND_WAIT` 秒；等不到就不跑，不硬闖。
#
# 用法:
#   ./scripts/check/prebind_probe_run.sh
#   CGC_PREBIND_WAIT=1800 ./scripts/check/prebind_probe_run.sh
#   CGC_PREBIND_DRY=1 ./scripts/check/prebind_probe_run.sh   # 只印命令，不跑
#
# 產出: $WORKDIR/llama_bench_<arm>_p<..>_n<..>_d<..>_r<..>.stderr.log
# 解析: python3 scripts/check/prebind_probe_parse.py <that stderr.log>
set -euo pipefail

ROOT="/Users/alexchuang/Documents/flashkv-devserver"
PY="${PYTHON:-/Users/alexchuang/.workbuddy/binaries/python/versions/3.13.12/bin/python3}"
MATRIX="$ROOT/scripts/check/llama_bench_matrix.py"

# ★ arm 為什麼是 `prod25-stream` 不是 `prefill250`：
#   ① 它才是交付 anchor 12.57 的臂（prod_profile.py 的 REF_ARM）—— 要用同一個池與同一個
#      batch 去量 p_res0，換臂量的常駐率不是交付配置的常駐率。
#   ② prefill250 在 16 GB M4 上會撞 GPU working set（11453 MB）：它額外帶 `-b 5632 -ub 5632`
#      （prod25 沒有，走 llama-bench 自己的預設 2048/512），實測
#      `CGC-METAL-FAIL: command buffer 8 failed (status 5, Insufficient Memory)` -> SIGABRT。
#      兩者 pool 都是 8 GiB，差的就是那個 compute buffer。
ARM="${CGC_PREBIND_ARM:-prod25-stream}"
WORKDIR="${CGC_PREBIND_WORKDIR:-/tmp/prebind_probe}"
WAIT_SEC="${CGC_PREBIND_WAIT:-900}"
POLL_SEC="${CGC_PREBIND_POLL:-5}"
MIN_USABLE="${CGC_PREBIND_MIN_USABLE:-30}"

# ── 閘門 1/2：等別人的量測結束 ────────────────────────────────────────────────
# ⚠ 用 `pgrep -x`（精確行程名），不能用 `pgrep -f`：後者會匹配到本腳本自己的命令列
#   （命令列裡就含 "llama-server"）⇒ 永遠「有人在用」⇒ 永遠等不到窗口。
window_free() {
    [ -z "$(lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null)" ] || return 1
    [ -z "$(pgrep -x llama-server 2>/dev/null)" ] || return 1
    [ -z "$(pgrep -x llama-bench 2>/dev/null)" ] || return 1
    return 0
}

# ── 沉降期（SETTLE）：光是「行程結束了」還不夠 ────────────────────────────────
# `ggml_metal_rsets_init: creating a residency set collection (keep_alive = 180 s)`：
# 一個 llama 行程退出後，它的 GPU 記憶體還會被 residency set 佔著最多 180 s。實測
# （2026-09-23）：別線 server 一退出就立刻起 bench，撞
#   `CGC-METAL-FAIL: command buffer 8 failed (status 5, Insufficient Memory)` -> SIGABRT
# 而 usable memory 明明有 52%。所以窗口的定義不是「沒有行程」，而是
# 「沒有行程 **而且** 最後一支走了至少 SETTLE 秒」。
# 只在真的看過別人時才要求沉降（一開始就沒人在跑就不用白等）。
SETTLE_SEC="${CGC_PREBIND_SETTLE:-180}"
waited=0
seen_other=0
last_seen=0
while [ "$waited" -lt "$WAIT_SEC" ]; do
    if window_free; then
        if [ "$seen_other" = 0 ]; then
            break
        fi
        if [ "$((waited - last_seen))" -ge "$SETTLE_SEC" ]; then
            echo "[gate] 最後一支別人走了 $((waited - last_seen))s（沉降門檻 ${SETTLE_SEC}s）" >&2
            break
        fi
    else
        if [ "$seen_other" = 0 ]; then
            echo "[gate] 有別的量測在跑，等待窗口（最多 ${WAIT_SEC}s）..." >&2
            lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null | tail -n +2 | head -3 >&2 || true
            pgrep -x llama-server >&2 || true
        fi
        seen_other=1
        last_seen="$waited"
    fi
    sleep "$POLL_SEC"
    waited=$((waited + POLL_SEC))
done

if ! window_free; then
    echo "[gate] BLOCKED：等了 ${WAIT_SEC}s 還是有別的量測在跑 -> 不跑。"
    echo "       要看是誰：lsof -nP -iTCP:8080 -sTCP:LISTEN ; pgrep -x llama-server"
    exit 3
fi
echo "[gate] 1/2 OK：8080 無 listener、無其他 llama 行程（等了 ${waited}s）" >&2

# ── 閘門 3：usable memory ────────────────────────────────────────────────────
# 用 prod_profile.py 同一個來源（prefill_certifiability.mem_state），不要有第二個門檻定義。
usable="$("$PY" -c '
import sys
sys.path.insert(0, "'"$ROOT"'/scripts/check")
from prefill_certifiability import mem_state
print("%.1f" % mem_state()["usable_pct"])
')"
echo "[gate] usable = ${usable}% (門檻 ${MIN_USABLE}%)" >&2
if ! "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' \
        "$usable" "$MIN_USABLE"; then
    echo "[gate] BLOCKED：usable ${usable}% < ${MIN_USABLE}% -> 不跑（13 GB 模型會 OOM）。"
    exit 4
fi
echo "[gate] 3 OK" >&2

# ── 跑 ───────────────────────────────────────────────────────────────────────
# env 用 merge（`run_env = dict(os.environ); run_env.update(env)`），不是 allowlist，
# 所以這兩個 probe flag 會真的到 child —— 不像 run_server.sh 那條路要另外登記。
# 形狀 = prod_profile.py 的 DECODE_SHAPE，逐字搬（k=3 => ntok=4）。
mkdir -p "$WORKDIR"

CMD=(
    "$PY" "$MATRIX"
    --arms "$ARM"
    --reps 1
    --prompt 0
    --gen 128
    --depths 512
    --ctx-size 4096
    --warm-skip 64
    --spec-type draft-mtp
    --spec-draft-n-max 3
    --workdir "$WORKDIR"
)
# ⚠ 不要預設覆寫 -b/-ub：`prod25-stream` 本身不帶 -b/-ub，走 llama-bench 的預設，
#   而那個預設是能在 8 GiB pool + 4617 MiB model 下存活的組合。prefill250 強制的
#   `-b 5632 -ub 5632` 才是 OOM 的來源（見 ARM 那段註解）。
#   真的要試別的寬度才設 CGC_PREBIND_BATCH，並記住 run_server.sh [budget] 的存活紀錄
#   （ub=4096 pool 8GiB 5/5、5120 3/3、5632 4/4、6144 0/5）。
if [ -n "${CGC_PREBIND_BATCH:-}" ]; then
    CMD+=(--batch "$CGC_PREBIND_BATCH")
fi

if [ -n "${CGC_PREBIND_DRY:-}" ]; then
    echo "[dry] env: CGC_PREBIND_PROBE=1 CGC_PREBIND_PROBE_VERBOSE=1"
    echo "[dry] cmd:"; printf '  %s\n' "${CMD[@]}"
    exit 0
fi

echo "[run] workdir = ${WORKDIR}（stderr 會落在這裡）" >&2
# CGC_RHO_PROBE：同一支 binary 上同時量 ρ（影子 router，見 qwen35moe.cpp 的註解）。
# ⚠ 開了它，圖裡會多一個 norm + 一個 matmul ⇒ **本輪的 t/s 不可引用**，只取 ρ 與幾何量。
#
# CGC_TD_CB=1：**沒有它就量不到 ρ。** 這個 repo 走的是 CGC 分段派送器
#   （ggml-backend.cpp ~2490 / ~3010），它只把「每段最後的 top-k 節點」送進 eval callback
#   （ask=false）；其它節點只有在 CGC_TD_CB 存在時才被逐段轉發。影子節點永遠不是段尾的
#   top-k ⇒ 不開這個開關，`CGC-RHO-CAP` 會是零行而 skip=100%（2026-09-23 實測）。
#   它是「張量名過濾器」，dispatcher 側轉發全部節點、dump 側才按名比對 ⇒ 設成 "1" 就不會
#   真的 dump 任何張量（沒有張量叫 "1"）。代價：它會把 async pipeline 序列化 ⇒ 只影響速度。
# CGC_RHO_PROBE_LATE=1 ⇒ 影子 router 退回「attn(L) 之後」的**舊位置**（新預設是貼著
#   `inpSA = inpL`，也就是 layer L 的最前面）。同一支 binary、同一組權重、同一個數學，
#   ⇒ **ρ 應逐位元相同，只有 GPU 上那個 node 落的位置不同**。
#   這是用來證明「提前量從哪裡來」的對照臂，不是用來重測準度的。
#   （2026-09-23 的 cov_uni=0.854 是在**舊位置**量到的；準度結論不受位置影響。）
RHO_ENV=("CGC_RHO_PROBE=1")
if [ -n "${CGC_RHO_PROBE_LATE:-}" ]; then
    RHO_ENV+=("CGC_RHO_PROBE_LATE=1")
fi

env "${RHO_ENV[@]}" CGC_PREBIND_PROBE=1 CGC_PREBIND_PROBE_VERBOSE=1 CGC_TD_CB=1 \
  "${CMD[@]}" > "$WORKDIR/driver.stdout" 2>&1
rc=$?
echo "[done] rc=$rc" >&2

# ★ llama_bench_matrix.py 用「參數」命名 log 檔 ⇒ 同參數第二次跑會**蓋掉**第一支 log
#   （2026-09-23 就這樣丟了 run 1 的原始檔）。跑完立刻另存一份帶時間戳的。
STAMP="$(date +%Y-%m-%d_%H%M%S)"
LATEST="$(ls -t ${WORKDIR}/*.stderr.log 2>/dev/null | head -1)"
if [ -n "$LATEST" ]; then
    cp "$LATEST" "${WORKDIR}/run_${STAMP}.stderr.log"
    echo "[save] ${WORKDIR}/run_${STAMP}.stderr.log" >&2
fi
echo "[next] $PY $ROOT/scripts/check/rho_probe_parse.py ${WORKDIR}/run_${STAMP}.stderr.log" >&2
