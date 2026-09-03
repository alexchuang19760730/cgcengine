#!/bin/bash
# run_server.sh — llama-server 生產啟動器（Windows 夥伴 HTTP 測試用）
#
# 與 run_n30cache.sh 同源的防護 + 生產 env，包成單一 CLI。
# 2026-08-30 教訓制度化（見 release.html §4.5/§4.7）：
#   - 啟動前清殘留行程（kernel panic 根因 = 行程疊加；N30CACHE_NO_CLEAN=1 跳過）
#   - 啟動前記憶體水位檢查（free < 25% 拒跑——4GiB wired pool + 13GB 模型）
#   - log 寫持久路徑 Backup/cgc_logs/（/tmp 會被重開機清掉，§4.5 附帶損失）
#   - curl 測 localhost 必帶 --noproxy '*'（本地代理 7897 會攔 127.0.0.1 → 502 空回應）
# 2026-08-31 架構修正：
#   - OpenAI-compatible 主服務層改以 llama-server 為準，不再以 edge_server.py 為長期主線
#   - 若 llama-server 的 chat/MTP/品質有缺口，就直接把 llama-server 路徑修到支持
#   - 本腳本因此補上 MTP server 模式，將 0000 防護與 draft-mtp 生產語義帶回正式服務入口
# 2026-09-01 OOM 修正：
#   - 本輪「當機」根因不是 prompt，而是 Metal decode 期 OutOfMemory
#   - 單看 system free memory 不足以判斷安全；target + MTP draft context 的 GPU/offload 組合
#     可能在請求進來後才把 command buffer 推爆，表現為 500 Compute error / ret=-3
#   - 因此新增 OOM-safe 參數面：只在明確指定 fallback 時啟用較保守的
#     ngl / draft-ngl / batch / ubatch / ctx / expert-cache；主線預設仍維持性能線
#
# 用法：
#   ./scripts/run_server.sh                       # 預設 qwen36 MTP（25+ t/s），port 8080
#   ./scripts/run_server.sh --detach              # 同上，但脫離父 shell（setsid），不被 SIGHUP 殺
#   CGC_SERVER_RUNTIME_PROFILE=non-mtp ./scripts/run_server.sh   # 非 MTP 基線（~8 t/s）
#   CGC_SERVER_RUNTIME_PROFILE=mtp ./scripts/run_server.sh        # 明確切回 MTP 生產配置
#   CGC_SERVER_OOM_SAFE=1 ./scripts/run_server.sh # 16GB 機器上的 fallback / 保命模式
#   CGC_SERVER_PORT=9931 ./scripts/run_server.sh  # 換 port
#   CGC_SERVER_MODEL_ROOT=/path/to/models/gguf ./scripts/run_server.sh # worktree 外掛模型目錄
#   伙伴（Windows/其他機器）：http://<Mac LAN IP>:8080/v1/chat/completions（OpenAI 相容）
#
# 鐵律：server 運行期間，本機禁止任何 13GB 級操作（llama-simple 對照/量化/HF 下載）。
set -euo pipefail

# --detach：用 Python os.setsid() 脫離父 shell process group，避免 SIGHUP/SIGINT 級聯殺死 server。
# macOS 無 setsid 命令，但 Python 的 os.setsid() 會真正建立新 session + 新 process group。
CGC_DETACHED="${CGC_DETACHED:-0}"
for arg in "$@"; do
    [ "$arg" = "--detach" ] && CGC_DETACHED=1
    [ "$arg" = "-d" ] && CGC_DETACHED=1
done
if [ "$CGC_DETACHED" = 1 ] && [ -z "${_CGC_DETACHED_MARKER:-}" ]; then
    echo "[detach] forking via Python os.setsid() (immune to parent SIGHUP)"
    _SCRIPT="$0"
    _ARGS="$@"
    # Strip --detach / -d from args passed to child
    _ARGS=$(echo "$_ARGS" | sed 's/--detach//g; s/^-d$//')
    python3 -c "
import os, sys, subprocess
pid = os.fork()
if pid > 0:
    print(f'[detach] child PID={pid}, waiting for health...')
    import time, urllib.request
    for i in range(60):
        time.sleep(2)
        try:
            r = urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)
            if b'ok' in r.read():
                print(f'[detach] server ready (PID={pid})')
                sys.exit(0)
        except Exception:
            pass
    print('[detach] 120s timeout')
    sys.exit(1)
else:
    os.setsid()
    os.environ['_CGC_DETACHED_MARKER'] = '1'
    os.environ['CGC_DETACHED'] = '1'
    os.execvp('$0', ['$0'] + '$_ARGS'.split())
"
    exit $?
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/src/llama.cpp/build/bin/llama-server"
SERVER_MINIMAL_CHAT_TEMPLATE="$ROOT/src/llama.cpp/models/templates/Nail-Qwen3.6-Minimal-Chat.jinja"
MODEL_ROOT="${CGC_SERVER_MODEL_ROOT:-$ROOT/models/gguf}"
Q36="$MODEL_ROOT/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
Q36_MTP="$MODEL_ROOT/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf"
Q36_MTP_DENSEIQ4X="$MODEL_ROOT/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"

SERVER_RUNTIME_PROFILE="${CGC_SERVER_RUNTIME_PROFILE:-auto}"
SERVER_MTP="${CGC_SERVER_MTP:-1}"
SERVER_DENSE_IQ4X="${CGC_SERVER_DENSE_IQ4X:-1}"  # denseIQ4X is the production MTP carrier
SERVER_N_CB="${CGC_SERVER_N_CB:-8}"  # §8.93: cb8 sweet spot
SERVER_GLU_FUSED_DOWN="${CGC_SERVER_GLU_FUSED_DOWN:-1}"  # §8.113: +6.5% speed
SERVER_WATCHDOG="${CGC_SERVER_WATCHDOG:-1}"  # Metal deadlock watchdog
SERVER_OA_ASYNC="${CGC_SERVER_OA_ASYNC:-1}"  # §8.77/8.78: async callback split
SERVER_PROFILE="${CGC_SERVER_PROFILE:-off}"
SERVER_CHAT_TEMPLATE="${CGC_SERVER_CHAT_TEMPLATE:-}"
SERVER_CHAT_TEMPLATE_FILE="${CGC_SERVER_CHAT_TEMPLATE_FILE:-}"
SERVER_CHAT_TEMPLATE_KWARGS="${CGC_SERVER_CHAT_TEMPLATE_KWARGS:-}"
SERVER_LOG_PROMPTS_DIR="${CGC_SERVER_LOG_PROMPTS_DIR:-}"
SERVER_REASONING="${CGC_SERVER_REASONING:-off}"
SERVER_REASONING_FORMAT="${CGC_SERVER_REASONING_FORMAT:-none}"
SERVER_SKIP_CHAT_PARSING="${CGC_SERVER_SKIP_CHAT_PARSING:-0}"
SERVER_CHAT_AB="${CGC_SERVER_CHAT_AB:-off}"
SERVER_CHAT_AB_PREFIX="${CGC_SERVER_CHAT_AB_PREFIX:-巴黎是法國的首都。}"
SERVER_CHAT_AB_MAX_TOKENS="${CGC_SERVER_CHAT_AB_MAX_TOKENS:-8}"
SERVER_CHAT_AB_STOP="${CGC_SERVER_CHAT_AB_STOP:-。}"

case "$SERVER_RUNTIME_PROFILE" in
    ''|auto|off) ;;
    mtp)
        SERVER_MTP=1
        ;;
    non-mtp|non_mtp)
        SERVER_MTP=0
        ;;
    *)
        echo "error: CGC_SERVER_RUNTIME_PROFILE must be auto|mtp|non-mtp (got $SERVER_RUNTIME_PROFILE)" >&2
        exit 2
        ;;
esac

MODEL_DEFAULT="$Q36"
if [ "$SERVER_MTP" = "1" ]; then
    if [ "$SERVER_DENSE_IQ4X" = "1" ]; then
        MODEL_DEFAULT="$Q36_MTP_DENSEIQ4X"
    else
        MODEL_DEFAULT="$Q36_MTP"
    fi
fi

MODEL="${CGC_SERVER_MODEL:-$MODEL_DEFAULT}"
PORT="${CGC_SERVER_PORT:-8080}"
HOST_BIND="${CGC_SERVER_HOST:-0.0.0.0}"
PHYS_MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
PHYS_MEM_GB=$(( PHYS_MEM_BYTES / 1024 / 1024 / 1024 ))
SERVER_OOM_SAFE="${CGC_SERVER_OOM_SAFE:-0}"

# qwen36 的 server chat 默認走最小模板，先拿掉 metadata 模板裡的空/open think 起手。
# answer-side prefill 仍維持 opt-in，避免把未驗證的回答腳手架硬塞成默認。
if [ -z "$SERVER_CHAT_TEMPLATE" ] && [ -z "$SERVER_CHAT_TEMPLATE_FILE" ]; then
    SERVER_CHAT_TEMPLATE_FILE="$SERVER_MINIMAL_CHAT_TEMPLATE"
fi
if [ -z "${CGC_SERVER_SKIP_CHAT_PARSING:-}" ]; then
    SERVER_SKIP_CHAT_PARSING=1
fi

# 可重跑 profile：把已驗證過的 QA / 長文口徑固化，讓 OA_ASYNC=0/1 只切一個變量。
# 若外部已顯式指定同名 env，則尊重外部覆寫。
case "$SERVER_PROFILE" in
    ''|off) ;;
    qa-zh)
        [ -z "${CGC_SERVER_CHAT_AB+x}" ] && SERVER_CHAT_AB="healthy-prefix"
        [ -z "${CGC_SERVER_CHAT_AB_PREFIX+x}" ] && SERVER_CHAT_AB_PREFIX="答："
        [ -z "${CGC_SERVER_CHAT_AB_MAX_TOKENS+x}" ] && SERVER_CHAT_AB_MAX_TOKENS="24"
        [ -z "${CGC_SERVER_CHAT_AB_STOP+x}" ] && SERVER_CHAT_AB_STOP="。"
        ;;
    longform-zh)
        [ -z "${CGC_SERVER_CHAT_AB+x}" ] && SERVER_CHAT_AB="custom-prefix"
        [ -z "${CGC_SERVER_CHAT_AB_PREFIX+x}" ] && SERVER_CHAT_AB_PREFIX="巴黎之所以成為法國的政治與文化中心，主要是因為"
        [ -z "${CGC_SERVER_CHAT_AB_MAX_TOKENS+x}" ] && SERVER_CHAT_AB_MAX_TOKENS="220"
        [ -z "${CGC_SERVER_CHAT_AB_STOP+x}" ] && SERVER_CHAT_AB_STOP="<|end|>,<|output|>,<|user|>"
        ;;
    *)
        echo "error: CGC_SERVER_PROFILE must be off|qa-zh|longform-zh (got $SERVER_PROFILE)" >&2
        exit 2
        ;;
esac

case "$SERVER_CHAT_AB" in
    ''|off) ;;
    healthy-prefix)
            # QA 路徑使用極短前綴，直接把回答拉進正文，避免 first-token 落回 <think>。
            # 現行 minimal template 只對 prefill_mode=always 生效，因此不要再傳未使用的 short_qa_zh。
            if [ -z "$SERVER_CHAT_TEMPLATE_KWARGS" ]; then
                SERVER_CHAT_TEMPLATE_KWARGS="{\"assistant_prefill\":\"答：\",\"assistant_prefill_mode\":\"always\"}"
            fi
            ;;
    custom-prefix)
            # 通用 prefill 模式：用很短的 assistant 起手把模型從 <think> 起手拉回正文。
            # 由外部以 CGC_SERVER_CHAT_AB_PREFIX 提供內容，例如：
            #   "答案是" / "At dawn, the lighthouse keeper "
            if [ -z "$SERVER_CHAT_TEMPLATE_KWARGS" ]; then
                SERVER_CHAT_TEMPLATE_KWARGS="{\"assistant_prefill\":\"$SERVER_CHAT_AB_PREFIX\",\"assistant_prefill_mode\":\"always\"}"
            fi
            ;;
    *)
        echo "error: CGC_SERVER_CHAT_AB must be off|healthy-prefix|custom-prefix (got $SERVER_CHAT_AB)" >&2
        exit 2
        ;;
esac

CTX_DEFAULT=2048
NGL_DEFAULT=99
DRAFT_NGL_DEFAULT=""
BATCH_DEFAULT=""
UBATCH_DEFAULT=""
BUDGET_DEFAULT="${N30CACHE_BUDGET:-4294967296}"   # 4GiB expert pool（與生產線一致）
SPEC_DRAFT_N_MAX="${CGC_SERVER_MTP_N_MAX:-3}"  # MTP draft tokens
SERVER_LAYER_CAPS="${CGC_SERVER_LAYER_CAPS:-}"  # Layer caps for expert cache

if [ "$SERVER_MTP" = "1" ]; then
    CTX_DEFAULT=3072
    if [ "$SERVER_OOM_SAFE" = "1" ] && [ "$PHYS_MEM_GB" -le 16 ]; then
        CTX_DEFAULT=1024
        NGL_DEFAULT=8
        DRAFT_NGL_DEFAULT=0
        BATCH_DEFAULT=64
        UBATCH_DEFAULT=32
        BUDGET_DEFAULT=0
    fi
fi

CTX="${CGC_SERVER_CTX:-$CTX_DEFAULT}"
SERVER_NGL="${CGC_SERVER_NGL:-$NGL_DEFAULT}"
SERVER_DRAFT_NGL="${CGC_SERVER_DRAFT_NGL:-$DRAFT_NGL_DEFAULT}"
SERVER_BATCH="${CGC_SERVER_BATCH:-$BATCH_DEFAULT}"
SERVER_UBATCH="${CGC_SERVER_UBATCH:-$UBATCH_DEFAULT}"
BUDGET="${CGC_SERVER_EXPERT_CACHE_BYTES:-$BUDGET_DEFAULT}"
LOG_DIR="$ROOT/Backup/cgc_logs"
LOG="$LOG_DIR/llama_server_$(date +%Y%m%d_%H%M%S).log"

[ -x "$BIN" ] || { echo "error: llama-server 不存在：$BIN（cmake -DLLAMA_BUILD_SERVER=ON 後構建）" >&2; exit 1; }
if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    echo "" >&2
    echo "  目前模型目錄：$MODEL_ROOT" >&2
    echo "  若 devserver worktree 沒放模型，可設：" >&2
    echo "    CGC_SERVER_MODEL_ROOT=/path/to/models/gguf" >&2
    echo "" >&2
    echo "  GGUF 不進 git（>100MB）。從 Hugging Face 下載：" >&2
    echo "    hf download Alexchuang/cgcengine-models \"$(basename "$MODEL")\" --local-dir models/gguf" >&2
    echo "  下載後驗證：cd models/gguf && shasum -a 256 -c SHA256SUMS" >&2
    echo "  全部模型清單：models/gguf/MANIFEST.md" >&2
    exit 1
fi

# [防護 1] 清殘留（§4.5：殭屍 server 是 0000 退化與 kernel panic 的共同土壤）
# pattern 用「build/bin/llama-*」子字串：行程可能是絕對路徑或相對路徑啟動（sandbox 用相對），
# 絕對路徑 pattern 比對不到相對路徑行程 → port 衝突 → 新行程秒退（2026-08-30 實測踩過）。
if [ "${N30CACHE_NO_CLEAN:-0}" != 1 ]; then
    for pat in "build/bin/llama-server" "build/bin/llama-simple" "build/bin/llama-speculative-simple"; do
        pkill -9 -f "$pat" 2>/dev/null && echo "  [clean] killed stale $pat" || true
    done
    sleep 1
fi

# [防護 2] 記憶體水位（模型 13.2GB --no-mmap + 4GiB wired pool；低水位強制拒跑）
FREE_PCT=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/{print int($2)}')
if [ -n "${FREE_PCT:-}" ] && [ "$FREE_PCT" -lt 25 ]; then
    echo "error: 系統可用記憶體僅 ${FREE_PCT}%（<25%）——16GB 機上疊 13GB 模型會 kernel panic（§4.5）。關掉其他重工行程再跑。" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
ln -sf "$LOG" "$LOG_DIR/llama_server_latest.log"

# [防護 3] 單 slot + 非 unified KV：與 llama-simple 行為對齊（np=auto 的 4 slots 會把 context 切 512/流）
# -expert-cache 是單刮號參數（common args 註冊形式，--expert-cache 不認）
# [防護 5] -sps 0 禁用 KV slot LCP 復用（2026-08-30 晚間曾誤判為 0000 根因；後續實證推翻：
#   -sps 0 上線後 task 0 首請求照樣 iota。真正根因見下。保留 -sps 0 作為減少干擾變數的防護）。
# [防護 6 / 2026-08-30 0000 真正根因] 拿掉 CGC_OA_ASYNC（原此處設 1）：
#   async Metal split 下 ggml-alloc 會把 top-k ids buffer 回收給同 step 後續 tensor（gather
#   用的 iota/arange 索引）→ hook 快照讀到 iota(0..N) 線性序列 → 錯誤專家 → garbage
#   logits → 輸出 0000。log 證據：llama_server_20260830_210223.log（-sps 0 已生效、
#   無任何 slot reuse）task 0 pmax=37 起全層 iota。OA_ASYNC 的 +12.6% 速度不值得換正確性；
#   要恢復需在 C++ 端為 ids 張量加同步（OPEN 項）。
echo "[start] $MODEL  port=$PORT  ctx=$CTX  ngl=$SERVER_NGL  budget=${BUDGET}B"
if [ "$SERVER_MTP" = "1" ]; then
    echo "[mode]  MTP ON (draft-mtp, n_max=$SPEC_DRAFT_N_MAX, denseIQ4X=$SERVER_DENSE_IQ4X)"
    if [ -n "$SERVER_DRAFT_NGL" ]; then
        echo "[mode]  draft_ngl=$SERVER_DRAFT_NGL"
    fi
    if [ "$SERVER_OOM_SAFE" = "1" ] && [ "$PHYS_MEM_GB" -le 16 ]; then
        echo "[mode]  OOM-safe fallback active for ${PHYS_MEM_GB}GB unified memory"
    fi
else
    echo "[mode]  MTP OFF (non-MTP baseline, ~8 t/s)"
fi
if [ -n "$SERVER_LAYER_CAPS" ]; then
    echo "[perf]  layer_caps=$SERVER_LAYER_CAPS"
elif [ "$SERVER_MTP" = "1" ]; then
    echo "[perf]  layer_caps=40-40:256 (default)"
fi
if [ -n "$SERVER_BATCH" ] || [ -n "$SERVER_UBATCH" ]; then
    echo "[perf]  batch=${SERVER_BATCH:-auto} ubatch=${SERVER_UBATCH:-auto}"
fi
echo "[perf]  n_cb=$SERVER_N_CB glu_fused_down=$SERVER_GLU_FUSED_DOWN watchdog=$SERVER_WATCHDOG oa_async=$SERVER_OA_ASYNC"
echo "[perf]  runtime_profile=$SERVER_RUNTIME_PROFILE model_root=$MODEL_ROOT"
if [ "$SERVER_PROFILE" != "off" ]; then
    echo "[chat]  profile=$SERVER_PROFILE"
fi
if [ -n "$SERVER_CHAT_TEMPLATE" ]; then
    echo "[chat]  template=$SERVER_CHAT_TEMPLATE"
fi
if [ -n "$SERVER_CHAT_TEMPLATE_FILE" ]; then
    echo "[chat]  template_file=$SERVER_CHAT_TEMPLATE_FILE"
fi
if [ -n "$SERVER_CHAT_TEMPLATE_KWARGS" ]; then
    echo "[chat]  template_kwargs=$SERVER_CHAT_TEMPLATE_KWARGS"
fi
if [ "$SERVER_CHAT_AB" != "off" ]; then
    echo "[chat]  ab_mode=$SERVER_CHAT_AB prefix=$SERVER_CHAT_AB_PREFIX max_tokens=$SERVER_CHAT_AB_MAX_TOKENS stop=$SERVER_CHAT_AB_STOP"
fi
if [ -n "$SERVER_LOG_PROMPTS_DIR" ]; then
    echo "[chat]  log_prompts_dir=$SERVER_LOG_PROMPTS_DIR"
fi
if [ -n "$SERVER_REASONING" ] || [ -n "$SERVER_REASONING_FORMAT" ] || [ "$SERVER_SKIP_CHAT_PARSING" = "1" ]; then
    echo "[chat]  reasoning=${SERVER_REASONING:-auto} format=${SERVER_REASONING_FORMAT:-auto} skip_chat_parsing=$SERVER_SKIP_CHAT_PARSING"
fi
echo "[log]   ${LOG}（tail -f 同路徑）"
SERVER_ARGS=(
    -m "$MODEL"
    -expert-cache "$BUDGET"
    -ngl "$SERVER_NGL"
    --no-mmap
    -t 8
    -c "$CTX"
    -np 1
    --no-kv-unified
    -sps 0
    --host "$HOST_BIND"
    --port "$PORT"
)
if [ -n "$SERVER_BATCH" ]; then
    SERVER_ARGS+=(-b "$SERVER_BATCH")
fi
if [ -n "$SERVER_UBATCH" ]; then
    SERVER_ARGS+=(-ub "$SERVER_UBATCH")
fi
if [ "$SERVER_MTP" = "1" ]; then
    SERVER_ARGS+=(
        --spec-type draft-mtp
        --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
        --temp 0
    )
    if [ -n "$SERVER_DRAFT_NGL" ]; then
        SERVER_ARGS+=(--spec-draft-ngl "$SERVER_DRAFT_NGL")
    fi
fi
if [ -n "$SERVER_CHAT_TEMPLATE" ]; then
    SERVER_ARGS+=(--chat-template "$SERVER_CHAT_TEMPLATE")
fi
if [ -n "$SERVER_CHAT_TEMPLATE_FILE" ]; then
    SERVER_ARGS+=(--chat-template-file "$SERVER_CHAT_TEMPLATE_FILE")
fi
if [ -n "$SERVER_CHAT_TEMPLATE_KWARGS" ]; then
    SERVER_ARGS+=(--chat-template-kwargs "$SERVER_CHAT_TEMPLATE_KWARGS")
fi
if [ -n "$SERVER_LOG_PROMPTS_DIR" ]; then
    SERVER_ARGS+=(--log-prompts-dir "$SERVER_LOG_PROMPTS_DIR")
fi
if [ -n "$SERVER_REASONING" ]; then
    SERVER_ARGS+=(--reasoning "$SERVER_REASONING")
fi
if [ -n "$SERVER_REASONING_FORMAT" ]; then
    SERVER_ARGS+=(--reasoning-format "$SERVER_REASONING_FORMAT")
fi
if [ "$SERVER_SKIP_CHAT_PARSING" = "1" ]; then
    SERVER_ARGS+=(--skip-chat-parsing)
fi
SERVER_ENV=(
    CGC_EXPERT_CACHE_BYTES="$BUDGET"
    LLAMA_EXPERT_CACHE_ALLOW_NGL=1
    LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
    LLAMA_EXPERT_CACHE_WORKERS=8
    CGC_WAKE_POLL_US=15
    CGC_PREFETCH_SRC=hist
    CGC_EVICTED_RING=0
    CGC_N_CB="$SERVER_N_CB"
    CGC_OA_ASYNC="$SERVER_OA_ASYNC"  # §8.77/8.78: +12.6% speed (0000 bug fixed in C++)
)
if [ "$SERVER_GLU_FUSED_DOWN" = "1" ]; then
    SERVER_ENV+=(CGC_GLU_FUSED_DOWN=1)
fi
if [ "$SERVER_WATCHDOG" = "1" ]; then
    SERVER_ENV+=(CGC_WATCHDOG=1)
fi
if [ "$SERVER_MTP" = "1" ]; then
    # MTP server 路徑對齊 run_n30cache.sh 的已驗證防護：
    # - 關 prefetch：避免 verify/draft 期背景填槽覆寫 GPU 正在讀的 slot
    # - verify/draft decode fast path：把 server 服務語義拉回生產 MTP 水位
    # - warm gate：短 prompt 避免 0000 退化；denseIQ4X 長 prompt 則不繼承短 prompt 門檻
    SERVER_ENV+=(
        CGC_NO_PREFETCH=1
        CGC_VERIFY_DECODE=1
        CGC_DRAFT_DECODE=1
    )
    if [ -n "${CGC_SERVER_WARM_NPAST:-}" ]; then
        SERVER_ENV+=(CGC_WARM_NPAST="$CGC_SERVER_WARM_NPAST")
    elif [ "$SERVER_DENSE_IQ4X" = "1" ]; then
        SERVER_ENV+=(CGC_WARM_NPAST=0)
    else
        SERVER_ENV+=(CGC_WARM_NPAST=8)
    fi
    if [ -n "$SERVER_LAYER_CAPS" ]; then
        SERVER_ENV+=(LLAMA_EXPERT_CACHE_LAYER_CAPS="$SERVER_LAYER_CAPS")
    else
        SERVER_ENV+=(LLAMA_EXPERT_CACHE_LAYER_CAPS="40-40:256")
    fi
fi
env "${SERVER_ENV[@]}" "$BIN" "${SERVER_ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!

# [防護 4] 健康輪詢（最多 120s；load 完成前 /health 不回）
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "<Mac IP>")
echo "[wait]  模型載入中（首次 ~1min）..."
for i in $(seq 1 60); do
    sleep 2
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "error: server 行程已退出——看 $LOG" >&2; exit 1; }
    if curl -s --noproxy '*' -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q "ok"; then
        echo ""
        echo "================ 連線卡（給夥伴） ================"
        echo "  Base URL   : http://$LAN_IP:$PORT/v1（OpenAI 相容）"
        echo "  測試       : curl --noproxy '*' http://127.0.0.1:$PORT/v1/models"
        echo "  Windows 伙伴 : 程式內直接指 http://$LAN_IP:$PORT/v1/chat/completions"
        echo "  Runtime    : CGC_SERVER_RUNTIME_PROFILE=mtp|non-mtp（目前 $SERVER_RUNTIME_PROFILE）"
        echo "  Regression : bash scripts/check/check_server.sh --base-url http://127.0.0.1:$PORT/v1"
        echo "  Benchmark  : python3 scripts/benchmark/benchmark_server_profiles.py --base-url http://127.0.0.1:$PORT/v1 --iterations 3"
        if [ "$SERVER_PROFILE" = "qa-zh" ]; then
            echo "  Profile    : qa-zh（中文短答；已驗證答：前綴可避開 <think>）"
            echo "  Payload    : {\"messages\":[{\"role\":\"user\",\"content\":\"請用一句中文回答：巴黎是哪個國家的首都？\"}],\"max_tokens\":$SERVER_CHAT_AB_MAX_TOKENS,\"stop\":[\"$SERVER_CHAT_AB_STOP\",\"<|end|>\",\"<|output|>\",\"<|user|>\"]}"
        elif [ "$SERVER_PROFILE" = "longform-zh" ]; then
            echo "  Profile    : longform-zh（中文長文；預設前綴可用 env 覆寫）"
            echo "  Payload    : {\"messages\":[{\"role\":\"user\",\"content\":\"請用一段中文說明巴黎為什麼是法國的政治與文化中心，避免條列，至少120字。\"}],\"max_tokens\":$SERVER_CHAT_AB_MAX_TOKENS,\"stop\":[\"<|end|>\",\"<|output|>\",\"<|user|>\"]}"
        elif [ "$SERVER_CHAT_AB" = "healthy-prefix" ]; then
            echo "  Chat A/B   : healthy-prefix（prefill=答：，驗證短答正文起手）"
            echo "  Payload    : {\"messages\":[{\"role\":\"user\",\"content\":\"請用一句中文回答：巴黎是哪個國家的首都？\"}],\"max_tokens\":24,\"stop\":[\"$SERVER_CHAT_AB_STOP\",\"<|end|>\",\"<|output|>\",\"<|user|>\"]}"
        elif [ "$SERVER_CHAT_AB" = "custom-prefix" ]; then
            echo "  Chat A/B   : custom-prefix（prefill=${SERVER_CHAT_AB_PREFIX}）"
        fi
        echo "  注意       : 本機 curl 測 localhost 必帶 --noproxy '*'（代理 7897 攔截）"
        echo "  停止       : pkill -INT -f llama-server（或 kill ${SERVER_PID}）"
        if [ "$SERVER_MTP" = "1" ]; then
            echo "  服務模式   : MTP / draft-mtp（目標 = 25+ t/s）"
            echo "  0000 防護  : CGC_NO_PREFETCH=1 + CGC_VERIFY_DECODE=1 + CGC_DRAFT_DECODE=1"
            echo "  OA_ASYNC   : ${SERVER_OA_ASYNC}（+12.6% speed, 0000 bug fixed in C++）"
            echo "  Layer Caps : 40-40:256（MTP draft layer full residency）"
        else
            echo "  服務模式   : 非 MTP 基線（約 ~8 t/s）"
        fi
        echo "=================================================="
        echo ""
        if [ "$CGC_DETACHED" = 1 ]; then
            echo "[detach] server PID=$SERVER_PID（已脫離父 shell，不受 SIGHUP 影響）"
            echo "         停止: kill -INT $SERVER_PID 或 pkill -INT -f llama-server"
            echo "         log: tail -f $LOG"
            exit 0
        else
            echo "[run]    前景運行中（Ctrl+C = 優雅關閉）。log: tail -f $LOG"
            trap 'kill -INT "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; exit 0' INT TERM
            wait "$SERVER_PID"
            exit $?
        fi
    fi
done
echo "error: 120s 內 /health 未就緒——看 $LOG" >&2
exit 1
