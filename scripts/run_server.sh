#!/bin/bash
# run_server.sh — llama-server 生產啟動器（Windows 夥伴 HTTP 測試用）
#
# 與 run_n30cache.sh 同源的防護 + 生產 env，包成單一 CLI。
# 2026-08-30 教訓制度化（見 release.html §4.5/§4.7）：
#   - 啟動前清殘留行程（kernel panic 根因 = 行程疊加；N30CACHE_NO_CLEAN=1 跳過）
#   - 啟動前記憶體水位檢查（free < 25% 拒跑——8GiB expert pool（default）+ ~13GB 模型/others）
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
#   CGC_SERVER_MTP_CLI_PARITY=1 ./scripts/run_server.sh          # 增量套用 CLI 的 MTP init（warmup / seq_rm probe）
#   CGC_SERVER_PROFILE=legacy-25plus ./scripts/run_server.sh     # 還原第一個 25+ server 狀態的 longform 啟動口徑
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
SERVER_MINIMAL_CHAT_TEMPLATE="$ROOT/src/llama.cpp/models/templates/Qwen3-nothink-ChatML.jinja"
# Edge0-35B's own chat template.  Byte-exact copy of the GGUF's tokenizer.chat_template
# (sha256 e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259) and identical
# to edge0 repo models/edge0-35b/chat_template.jinja.  See the Edge0 selector below.
SERVER_EDGE0_CHAT_TEMPLATE="$ROOT/src/llama.cpp/models/templates/Edge0-35B-ChatML.jinja"
MODEL_ROOT="${CGC_SERVER_MODEL_ROOT:-$ROOT/models/gguf}"
Q36="$MODEL_ROOT/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
Q36_MTP="$MODEL_ROOT/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf"
Q36_MTP_DENSEIQ4X="$MODEL_ROOT/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"

SERVER_RUNTIME_PROFILE="${CGC_SERVER_RUNTIME_PROFILE:-auto}"
SERVER_MTP="${CGC_SERVER_MTP:-1}"
SERVER_DENSE_IQ4X="${CGC_SERVER_DENSE_IQ4X:-1}"  # denseIQ4X is the production MTP carrier
SERVER_MTP_CLI_PARITY="${CGC_SERVER_MTP_CLI_PARITY:-0}"  # opt-in: mirror speculative-simple init path
SERVER_MTP_NO_WARMUP="${CGC_SERVER_MTP_NO_WARMUP:-0}"    # pass-through: skip CLI-style manual warmup
SERVER_NO_SEQ_RM_PROBE="${CGC_SERVER_NO_SEQ_RM_PROBE:-0}" # pass-through: skip seq_rm probe explicitly
SERVER_N_CB="${CGC_SERVER_N_CB:-8}"  # §8.93: cb8 sweet spot
SERVER_GLU_FUSED_DOWN="${CGC_SERVER_GLU_FUSED_DOWN:-1}"  # §8.113: +6.5% speed
SERVER_WATCHDOG="${CGC_SERVER_WATCHDOG:-1}"  # Metal deadlock watchdog
SERVER_OA_ASYNC="${CGC_SERVER_OA_ASYNC:-0}"  # §8.77/8.78: async callback split（2026-09-09 預設改 0：OA_ASYNC 疊加 skip0 時 0/10；單獨 8/10，不值得）
SERVER_PROFILE="${CGC_SERVER_PROFILE:-off}"
SERVER_CHAT_TEMPLATE="${CGC_SERVER_CHAT_TEMPLATE:-}"
SERVER_CHAT_TEMPLATE_FILE="${CGC_SERVER_CHAT_TEMPLATE_FILE:-}"
SERVER_CHAT_TEMPLATE_KWARGS="${CGC_SERVER_CHAT_TEMPLATE_KWARGS:-}"
SERVER_LOG_PROMPTS_DIR="${CGC_SERVER_LOG_PROMPTS_DIR:-}"
SERVER_REASONING="${CGC_SERVER_REASONING:-off}"
# 2026-09-09 FIX: when reasoning=off, reasoning-format must be "none", not "deepseek".
# The previous default (off + deepseek) was a contradictory combo that caused chat template
# mis-parsing: <think> tags were emitted as literal text, prompt-echo loops, empty content,
# and code-fence loops on Windows clients. When reasoning=off, use "none" to disable all
# reasoning-format processing. Override via CGC_SERVER_REASONING_FORMAT if needed.
SERVER_REASONING_FORMAT="${CGC_SERVER_REASONING_FORMAT:-none}"
SERVER_REASONING_PRESERVE="${CGC_SERVER_REASONING_PRESERVE:-0}"  # 2026-09-05：預設關，b66e0eaf7 原版沒開；只在明確要 preserve reasoning 時才 CGC_SERVER_REASONING_PRESERVE=1
SERVER_SKIP_CHAT_PARSING="${CGC_SERVER_SKIP_CHAT_PARSING:-0}"
SERVER_CHAT_AB="${CGC_SERVER_CHAT_AB:-off}"
SERVER_CHAT_AB_PREFIX="${CGC_SERVER_CHAT_AB_PREFIX:-巴黎是法國的首都。}"
SERVER_CHAT_AB_MAX_TOKENS="${CGC_SERVER_CHAT_AB_MAX_TOKENS:-8}"
SERVER_CHAT_AB_STOP="${CGC_SERVER_CHAT_AB_STOP:-。}"
SERVER_MEMORY_MODE="${CGC_SERVER_MEMORY_MODE:-dev}"  # dev: fail fast, prod: auto-fallback on low-memory starts

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
SERVER_AUTO_ANCHOR="${CGC_SERVER_AUTO_ANCHOR:-0}"        # 2026-09-09: 預設關閉。09-07 為防 echo 加入的 lang-matched starter，oracle 實測反而是 echo 來源之一（同配置 anchor ON→echo loop、OFF→42 正確）；需要時可設 CGC_SERVER_AUTO_ANCHOR=1 開啟
SERVER_DEFAULT_MARKER_STOPS="${CGC_SERVER_DEFAULT_MARKER_STOPS:-1}" # client 沒給 stop 時補 ChatML marker stops（截斷 scaffold 迴圈）
CGC_FORCE_TEMP0="${CGC_FORCE_TEMP0:-0}" # 2026-09-09 FIX: 預設關閉強制 temp=0。
                                  # 原預設 1 會導致某些 prompt 在 greedy 解碼下進入 echo 循環（Windows 端回報）。
                                  # 量測：temp=0.1-0.3 既能避免 IQ3_XXS 噪聲崩潰，又能打破 greedy 循環。
                                  # 若需要嚴格確定性，可設 CGC_FORCE_TEMP0=1 覆蓋。

# qwen36 的 server chat 預設走 GGUF embedded ChatML（b66e0eaf7 驗證：Nail jinja 的
# <|user|> token model 訓練不認，會重複 <|user|> 而非回答；只有明確想用 Nail jinja 的
# profile 才傳 --chat-template-file Nail-Qwen3.6-Minimal-Chat.jinja）。
# prefill 改成 profile-aware：longform 可用，qa 預設不用，避免把長文前綴誤塞到短答。
if [ -z "$SERVER_CHAT_TEMPLATE" ] && [ -z "$SERVER_CHAT_TEMPLATE_FILE" ]; then
    # 2026-09-05 預設改走 Nail jinja (v4 ChatML 風格): 支援 assistant_prefill / disable_think_scaffold。
    # b66e0eaf7 之前的「Nail jinja 不工作」是因為 v1-v3 Nail jinja 用 <|user|>,而當前 v4 已改 <|im_start|>
    # ChatML 風格,行為等同 GGUF embedded + 額外支援 prefill。Nail jinja header 含
    # "Minimal chat template for Nail-Qwen3.6-MTP" + "<think>" 兩個字串,會 dispatch 進 chat.cpp
    # Nail handler,啟用 THINK_SEED 注入 (v4 fix)。
    # CGC FIX 2026-09-05: Use Qwen3 no-think ChatML template to eliminate think loops
    SERVER_CHAT_TEMPLATE_FILE="$SERVER_MINIMAL_CHAT_TEMPLATE"
fi
if [ -z "${CGC_SERVER_SKIP_CHAT_PARSING:-}" ]; then
    SERVER_SKIP_CHAT_PARSING=0
fi

# 可重跑 profile：把已驗證過的 QA / 長文口徑固化，讓 OA_ASYNC=0/1 只切一個變量。
# 若外部已顯式指定同名 env，則尊重外部覆寫。
case "$SERVER_PROFILE" in
    ''|off) ;;
    qa-zh)
        # 2026-09-05：Nail jinja 的 <|user|> token model 訓練不認，會重複 <|user|>
        # 而非回答。改走 GGUF embedded ChatML(<|im_start|>/im_end|>)，b66e0eaf7 已驗證。
        # --reasoning off + --reasoning-format none 是 b66e0eaf7 的原版設定。
        [ -z "${CGC_SERVER_CHAT_AB+x}" ] && SERVER_CHAT_AB="off"
        [ -z "${CGC_SERVER_CHAT_AB_MAX_TOKENS+x}" ] && SERVER_CHAT_AB_MAX_TOKENS="24"
        [ -z "${CGC_SERVER_CHAT_AB_STOP+x}" ] && SERVER_CHAT_AB_STOP="。"
        # 不預設 Nail jinja → 走 GGUF embedded ChatML
        if [ -z "${CGC_SERVER_CHAT_TEMPLATE_FILE:-}" ] && [ -z "${CGC_SERVER_CHAT_TEMPLATE:-}" ]; then
            SERVER_CHAT_TEMPLATE_FILE=""
        fi
        ;;
    longform-zh)
        # 2026-09-05：跟 qa-zh 一致走 GGUF embedded ChatML，避免 Nail jinja 的 <|user|> 不被認。
        [ -z "${CGC_SERVER_CHAT_AB+x}" ] && SERVER_CHAT_AB="custom-prefix"
        [ -z "${CGC_SERVER_CHAT_AB_PREFIX+x}" ] && SERVER_CHAT_AB_PREFIX="巴黎之所以成為法國的政治與文化中心，主要是因為"
        [ -z "${CGC_SERVER_CHAT_AB_MAX_TOKENS+x}" ] && SERVER_CHAT_AB_MAX_TOKENS="220"
        [ -z "${CGC_SERVER_CHAT_AB_STOP+x}" ] && SERVER_CHAT_AB_STOP="<|end|>,<|output|>,<|user|>"
        # 不預設 Nail jinja → 走 GGUF embedded ChatML
        if [ -z "${CGC_SERVER_CHAT_TEMPLATE_FILE:-}" ] && [ -z "${CGC_SERVER_CHAT_TEMPLATE:-}" ]; then
            SERVER_CHAT_TEMPLATE_FILE=""
        fi
        ;;
    coding)
        # 2026-09-05：代碼生成 profile。
        # - 用 ```python\n# ``` code block prefill 把模型從 <think> 起手拉回代碼正文
        # - 長 max_tokens 適合函數/類生成
        # - ctx=4096 預留較長的輸入空間（檔案級 prompt）
        # - 走 GGUF embedded ChatML (跟 qa-zh / longform-zh 一致)
        # 注意：prefix 的 \n 用 literal backslash-n（不要用 $'\n' 變成真的 LF）
        #   這樣經 SERVER_CHAT_TEMPLATE_KWARGS 包成 JSON 時，JSON parser 會把 \n 轉成 LF。
        [ -z "${CGC_SERVER_CHAT_AB+x}" ] && SERVER_CHAT_AB="custom-prefix"
        [ -z "${CGC_SERVER_CHAT_AB_PREFIX+x}" ] && SERVER_CHAT_AB_PREFIX='```python\n# '
        [ -z "${CGC_SERVER_CHAT_AB_MAX_TOKENS+x}" ] && SERVER_CHAT_AB_MAX_TOKENS="512"
        [ -z "${CGC_SERVER_CHAT_AB_STOP+x}" ] && SERVER_CHAT_AB_STOP='```'
        [ -z "${CGC_SERVER_CTX+x}" ] && CTX_DEFAULT=4096
        # 不預設 Nail jinja → 走 GGUF embedded ChatML
        if [ -z "${CGC_SERVER_CHAT_TEMPLATE_FILE:-}" ] && [ -z "${CGC_SERVER_CHAT_TEMPLATE:-}" ]; then
            SERVER_CHAT_TEMPLATE_FILE=""
        fi
        ;;
    legacy-25plus)
        # 2026-09-04 第一個 25+ server 狀態：
        # - denseIQ4X + MTP + CLI parity env
        # - healthy runtime（ngl=99, ctx=3072, n_max=3）
        # - longform prefill 把輸出拉回 content 軌
        # - reasoning-format=deepseek，便於重放 semantic-gap 當時的量測口徑
        [ -z "${CGC_SERVER_MTP+x}" ] && SERVER_MTP=1
        [ -z "${CGC_SERVER_DENSE_IQ4X+x}" ] && SERVER_DENSE_IQ4X=1
        [ -z "${CGC_SERVER_MTP_CLI_PARITY+x}" ] && SERVER_MTP_CLI_PARITY=1
        [ -z "${CGC_SERVER_SKIP_CHAT_PARSING+x}" ] && SERVER_SKIP_CHAT_PARSING=0
        [ -z "${CGC_SERVER_REASONING_FORMAT+x}" ] && SERVER_REASONING_FORMAT="deepseek"
        [ -z "${CGC_SERVER_CHAT_AB+x}" ] && SERVER_CHAT_AB="custom-prefix"
        [ -z "${CGC_SERVER_CHAT_AB_PREFIX+x}" ] && SERVER_CHAT_AB_PREFIX="巴黎之所以成為法國的政治與文化中心，主要是因為"
        [ -z "${CGC_SERVER_CHAT_AB_MAX_TOKENS+x}" ] && SERVER_CHAT_AB_MAX_TOKENS="220"
        [ -z "${CGC_SERVER_CHAT_AB_STOP+x}" ] && SERVER_CHAT_AB_STOP="<|end|>,<|output|>,<|user|>"
        # 走 GGUF embedded ChatML（b66e0eaf7 設定）
        if [ -z "${CGC_SERVER_CHAT_TEMPLATE_FILE:-}" ] && [ -z "${CGC_SERVER_CHAT_TEMPLATE:-}" ]; then
            SERVER_CHAT_TEMPLATE_FILE=""
        fi
        ;;
    *)
        echo "error: CGC_SERVER_PROFILE must be off|qa-zh|longform-zh|coding|legacy-25plus (got $SERVER_PROFILE)" >&2
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
# [CGC 2026-09-06 pool budget 4GiB -> 8GiB] A/B on this 16GB Mac (MTP+denseIQ4X, ngl=99):
#   - 8GiB pool loads + runs stably (RSS ~7.7GB, health OK, ~18% free), same as the earlier
#     10GiB probe (RSS 9.64GB, 2h22m uptime) — the 16GB envelope hosts far more than 4GiB.
#   - 8GiB default (no renorm): quality = baseline exactly (qa-zh 1.0 / longform 0.882 /
#     coding 1.0 over 5 runs), decode ~11-22 t/s vs 4GiB's 7-14 (fewer fast-path guard trips).
#   - 8GiB + CGC_RN_ROUTING=1 + CGC_WCOLD_EN=1: decode 25.9 t/s / 98.7% accept but quality 0.3
#     (renorm mask mechanism bug, invariant to pool size — see docs …RenormRouting_AB §8).
# Still overridable: N30CACHE_BUDGET / CGC_SERVER_EXPERT_CACHE_BYTES / CGC_SERVER_OOM_SAFE=1.
# [CGC 2026-09-13] 8GiB -> 10GiB. The pool is a COMPUTED number of slots, not a raw byte pool:
#   capacity = budget / (41 layers * per_slot 1.465MB)   ->   8GiB = 143 slots/layer
# The routing oracle (LLAMA_EXPERT_CACHE_ROUTE_RECORD + _ROUTE_DUMP + CGC_MASSCOV) measured, at
# 143 slots on Nail denseIQ4X, hit rate 90.8%:
#   current membership mass coverage      90.3%
#   best possible top-143-by-mass         90.5%   -> placement gap 0.2pp
#   counterfactual: K=96 79.7%  K=128 87.7%  K=192 97.1%  K=256 100.0%
# So no placement heuristic can win more than 0.2pp, while capacity buys ~6.6pp per +49 slots.
# 10GiB -> 178 slots/layer -> ~95% coverage (interpolated), i.e. non-resident selected experts
# 9.8% -> ~5%, for ~+2GiB RSS (measured 7.55GiB at 143 slots, including 1.61GiB dense).
BUDGET_DEFAULT="${N30CACHE_BUDGET:-10737418240}"  # 10GiB expert pool
SPEC_DRAFT_N_MAX="${CGC_SERVER_MTP_N_MAX:-3}"  # MTP draft tokens
SERVER_LAYER_CAPS="${CGC_SERVER_LAYER_CAPS:-}"  # Layer caps for expert cache
# [2026-09-12] 記憶體模式改可覆寫。預設 none = 舊的 --no-mmap 行為（逐值相同）。
# 16GB 機器上 --no-mmap 讓 dense 段變成可被 swap 的匿名頁；mmap 則讓它變成可回收的
# clean file page。實測本機曾出現 swap used 5.8GB / free pages 61MB，這是速度主因之一。
#   CGC_SERVER_LOAD_MODE=mmap ./scripts/run_server.sh   # 讓 OS 可回收模型頁
# 合法值：auto | none | mmap | mlock | mmap+mlock
SERVER_LOAD_MODE="${CGC_SERVER_LOAD_MODE:-none}"

if [ "$SERVER_MTP" = "1" ]; then
    # [CGC 2026-09-07 ctx 3072 -> 8192] Claude Code CLI sends ~3117 tokens MINIMUM
    # (its agent system prompt + 3 tool schemas + <system-reminder> wrappers), which
    # exceeded n_ctx=3072 and made every CLI request fail with HTTP 400
    # "exceeds the available context size". 8192 fits the CLI floor + output headroom;
    # verified load + healthy runtime on the 16GB M4 Max (RSS ~7.9GB, MTP + 8GiB pool).
    # Profile-specific CTX (e.g. coding 4096) still overrides below when set.
    CTX_DEFAULT=8192
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

case "$SERVER_MEMORY_MODE" in
    dev|prod) ;;
    *)
        echo "error: CGC_SERVER_MEMORY_MODE must be dev|prod (got $SERVER_MEMORY_MODE)" >&2
        exit 2
        ;;
esac

cgc_existing_llama_server_count() {
    local count
    count=$(pgrep -f "build/bin/llama-server" 2>/dev/null | wc -l | tr -d ' ' || true)
    echo "${count:-0}"
}

cgc_memory_guard_class() {
    if [ "$SERVER_MTP" = "1" ] && [ "${SERVER_NGL:-0}" -ge 90 ] && [ "${CTX:-0}" -ge 3072 ]; then
        if [ "$SERVER_PROFILE" = "legacy-25plus" ]; then
            echo "full-mtp-known-profile"
        else
            echo "full-mtp"
        fi
    elif [ "$SERVER_MTP" = "1" ]; then
        echo "fallback-mtp"
    else
        echo "baseline"
    fi
}

cgc_memory_guard_req() {
    local klass="$1"
    case "$klass" in
        full-mtp-known-profile) echo "0 35 0" ;;   # known single-machine profile: free_pct other_llama_servers gate only
        full-mtp) echo "0 40 0" ;;                 # unknown heavy full-MTP launches need more headroom
        fallback-mtp) echo "0 20 0" ;;
        baseline) echo "0 15 0" ;;
        *) echo "0 0 999" ;;
    esac
}

cgc_memory_guard_reason() {
    local klass="$1"
    local phys_gb="$2"
    local free_pct="$3"
    local other_servers="$4"
    local req_phys="$5"
    local req_free="$6"
    local req_other="$7"
    local reasons=()
    if [ "$phys_gb" -lt "$req_phys" ]; then
        reasons+=("physical=${phys_gb}GB<${req_phys}GB")
    fi
    if [ "$free_pct" -lt "$req_free" ]; then
        reasons+=("free=${free_pct}%<${req_free}%")
    fi
    if [ "$other_servers" -gt "$req_other" ]; then
        reasons+=("other_llama_servers=${other_servers}>${req_other}")
    fi
    local joined="${reasons[*]:-unknown}"
    echo "${klass}: ${joined}"
}

cgc_apply_prod_memory_fallback() {
    echo "[guard] prod low-memory fallback: forcing OOM-safe server profile" >&2
    SERVER_OOM_SAFE=1
    if [ "$SERVER_MTP" = "1" ]; then
        CTX=1024
        SERVER_NGL=8
        SERVER_DRAFT_NGL=0
        SERVER_BATCH=64
        SERVER_UBATCH=32
        BUDGET=0
    else
        CTX="${CTX:-2048}"
        if [ "${SERVER_NGL:-0}" -gt 8 ]; then
            SERVER_NGL=8
        fi
    fi
}

[ -x "$BIN" ] || { echo "error: llama-server 不存在：${BIN}（cmake -DLLAMA_BUILD_SERVER=ON 後構建）" >&2; exit 1; }
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

# [防護 1b] sudo purge 清理記憶體（2026-09-07：16GB 機器上 35B MoE + 8GB pool 記憶體壓力大，
# purge 可釋放 inactive/purgeable 記憶體，減少 compressor 與 swap 波動）
# 用法：
#   CGC_PURGE=1 ./scripts/run_server.sh              # 執行 sudo purge（需手動輸入密碼）
#   CGC_PURGE=1 CGC_SUDO_PASSWORD=xxx ./scripts/run_server.sh  # 自動輸入密碼（不安全，僅供測試）
#   預設 CGC_PURGE=0（不執行，避免需要密碼）
CGC_PURGE="${CGC_PURGE:-0}"
if [ "$CGC_PURGE" = "1" ]; then
    if [ -n "${CGC_SUDO_PASSWORD:-}" ]; then
        echo "$CGC_SUDO_PASSWORD" | sudo -S purge 2>/dev/null && echo "  [purge] sudo purge 完成（自動密碼）" || echo "  [purge] sudo purge 失敗（密碼錯誤？）"
    else
        echo "  [purge] 執行 sudo purge（請輸入密碼）..."
        sudo purge 2>/dev/null && echo "  [purge] sudo purge 完成" || echo "  [purge] sudo purge 失敗（已跳過）"
    fi
    sleep 1
fi

# [防護 2] 記憶體水位（模型 13.2GB --no-mmap + 8GiB expert pool default；低水位先記 warning，
# 再交由下方 memory guard 統一決定是 fail fast 還是 prod fallback）
FREE_PCT=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/{print int($2)}')
if [ -n "${FREE_PCT:-}" ] && [ "$FREE_PCT" -lt 25 ]; then
    echo "[guard] warning: 系統可用記憶體僅 ${FREE_PCT}%（<25%）——後續 memory guard 會決定 fail fast 或 prod fallback" >&2
fi

# [防護 2b] 基本可運作記憶體要求：
# - 已驗證單機 profile（legacy-25plus）按 free% + other_llama_servers 判斷，不再用 physical RAM 一刀切。
# - 未知 full-MTP heavy line 仍保守要求更高 free%。
# - prod 線若 full-MTP 不滿足要求，先自動降到保命配置。
FREE_PCT="${FREE_PCT:-0}"
OTHER_LLAMA_SERVERS="$(cgc_existing_llama_server_count)"
MEM_CLASS="$(cgc_memory_guard_class)"
read -r MEM_REQ_PHYS_GB MEM_REQ_FREE_PCT MEM_REQ_OTHER <<< "$(cgc_memory_guard_req "$MEM_CLASS")"
if [ "$PHYS_MEM_GB" -lt "$MEM_REQ_PHYS_GB" ] || [ "$FREE_PCT" -lt "$MEM_REQ_FREE_PCT" ] || [ "$OTHER_LLAMA_SERVERS" -gt "$MEM_REQ_OTHER" ]; then
    MEM_REASON="$(cgc_memory_guard_reason "$MEM_CLASS" "$PHYS_MEM_GB" "$FREE_PCT" "$OTHER_LLAMA_SERVERS" "$MEM_REQ_PHYS_GB" "$MEM_REQ_FREE_PCT" "$MEM_REQ_OTHER")"
    if [ "$SERVER_MEMORY_MODE" = "prod" ] && [ "$MEM_CLASS" = "full-mtp" ]; then
        echo "[guard] prod start does not meet full-MTP memory requirement -> $MEM_REASON" >&2
        cgc_apply_prod_memory_fallback
        MEM_CLASS="$(cgc_memory_guard_class)"
        read -r MEM_REQ_PHYS_GB MEM_REQ_FREE_PCT MEM_REQ_OTHER <<< "$(cgc_memory_guard_req "$MEM_CLASS")"
    else
        echo "error: startup blocked by memory guard -> $MEM_REASON" >&2
        echo "hint: dev 線請先停掉其他 llama-server / 釋放記憶體；prod 線可設 CGC_SERVER_MEMORY_MODE=prod 走自動降級。" >&2
        exit 1
    fi
fi

if [ "$PHYS_MEM_GB" -lt "$MEM_REQ_PHYS_GB" ] || [ "$FREE_PCT" -lt "$MEM_REQ_FREE_PCT" ] || [ "$OTHER_LLAMA_SERVERS" -gt "$MEM_REQ_OTHER" ]; then
    MEM_REASON="$(cgc_memory_guard_reason "$MEM_CLASS" "$PHYS_MEM_GB" "$FREE_PCT" "$OTHER_LLAMA_SERVERS" "$MEM_REQ_PHYS_GB" "$MEM_REQ_FREE_PCT" "$MEM_REQ_OTHER")"
    echo "error: even fallback memory guard is not satisfied -> $MEM_REASON" >&2
    echo "hint: 這代表目前機器狀態連保命線都撐不住，先關掉其他重工行程後再跑。" >&2
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
    if [ "$SERVER_MTP_CLI_PARITY" = "1" ]; then
        echo "[mode]  cli_parity_init=1 (warmup / seq_rm probe mirror speculative-simple)"
    fi
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
echo "[perf]  n_cb=$SERVER_N_CB glu_fused_down=$SERVER_GLU_FUSED_DOWN watchdog=$SERVER_WATCHDOG oa_async=$SERVER_OA_ASYNC load_mode=$SERVER_LOAD_MODE"
echo "[perf]  runtime_profile=$SERVER_RUNTIME_PROFILE model_root=$MODEL_ROOT"
echo "[guard] memory_mode=$SERVER_MEMORY_MODE class=$MEM_CLASS phys=${PHYS_MEM_GB}GB free=${FREE_PCT}% other_llama_servers=$OTHER_LLAMA_SERVERS"
if [ "$SERVER_PROFILE" != "off" ]; then
    echo "[chat]  profile=$SERVER_PROFILE"
fi
if [ -n "$SERVER_CHAT_TEMPLATE" ]; then
    echo "[chat]  template=$SERVER_CHAT_TEMPLATE"
fi
# ── Edge0-35B: use the model's OWN vendored chat template ────────────────────────
# The hand-written Qwen3-nothink-ChatML.jinja above is a Nail-era artifact; it is NOT
# Edge0's template.  For an Edge0 checkpoint, fall back to the model's own template.
#
# "--reasoning off" (already set for the no-think profiles) sets
# enable_thinking=false in common/arg.cpp, and Edge0's template then renders its
# canonical DIRECT-ANSWER generation prompt
#     <|im_start|>assistant\n<think>\n\n</think>\n\n
# which is exactly the form edge0's own server uses with think=False (see
# edge0 src/edge0/server/chat.py::_chat_text).  A bare "<|im_start|>assistant\n"
# without the closed block makes the model emit its own opener, which leaks into
# content -- that is the symptom this selector removes.
#
# An explicit CGC_SERVER_CHAT_TEMPLATE_FILE / CGC_SERVER_CHAT_TEMPLATE always wins.
if [ -z "${CGC_SERVER_CHAT_TEMPLATE_FILE:-}" ] && [ -z "${CGC_SERVER_CHAT_TEMPLATE:-}" ]; then
    case "$(basename "$MODEL")" in
        Edge0-*)
            SERVER_CHAT_TEMPLATE_FILE="$SERVER_EDGE0_CHAT_TEMPLATE"
            echo "[chat]  Edge0 model -> model's own template ($(basename "$SERVER_EDGE0_CHAT_TEMPLATE"))"
            ;;
    esac
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
    --load-mode "$SERVER_LOAD_MODE"
    -t 8
    -c "$CTX"
    -np "${CGC_SERVER_CONCURRENCY:-1}"
    --no-kv-unified
    -sps 0
    --host "$HOST_BIND"
    --port "$PORT"
    --jinja
)
# [CGC 2026-09-08 KV cache Q8 quantization] reduce KV cache memory by ~45% (FP16 -> Q8_0).
# For Qwen3.6-35B-A3B at ctx=8192: saves ~0.56GB (1.25GB -> 0.69GB). Quality impact is
# negligible (<1%) for typical workloads. Set CGC_SERVER_KV_Q8=0 to disable (fall back to FP16).
SERVER_KV_Q8="${CGC_SERVER_KV_Q8:-1}"
if [ "$SERVER_KV_Q8" = "1" ]; then
    SERVER_ARGS+=(--cache-type-k q8_0 --cache-type-v q8_0)
    echo "[kv]     cache-type-k=q8_0 cache-type-v=q8_0 (saves ~0.56GB at ctx=8192)"
fi
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
if [ "${SERVER_REASONING_PRESERVE:-0}" = "1" ]; then
    SERVER_ARGS+=(--reasoning-preserve)
fi
if [ "$SERVER_SKIP_CHAT_PARSING" = "1" ]; then
    SERVER_ARGS+=(--skip-chat-parsing)
fi
# [CGC IQ3_XXS Sampling] Optimized for low-bit quantization quality
# temp 0.4 + top_p 0.8 是 06:52 生產基線 (26.25 t/s, 98.2% draft_accept) 的配置
# 可通過 CGC_SERVER_TEMP / CGC_SERVER_TOP_P 覆蓋
SERVER_ARGS+=(--temp "${CGC_SERVER_TEMP:-0.4}")
SERVER_ARGS+=(--top-k "${CGC_SERVER_TOP_K:-0}")
SERVER_ARGS+=(--top-p "${CGC_SERVER_TOP_P:-0.8}")
# [CGC 2026-09-07 IQ3_XXS repetition guard] mid-generation repetition collapse
# (coding loops like 'def test_f fibonacci():' xN) is a SAMPLING property, not an
# expert-cache one (persists at 5.6-7.1% selcold across 8/10GB pools). Measured:
# rp=1.3 breaks the echo/repeat loop and pivots to real code, at a small draft-accept
# cost on genuine content. Default 1.0 = stock behavior; override via CGC_SERVER_REPEAT_PENALTY.
SERVER_ARGS+=(--repeat-penalty "${CGC_SERVER_REPEAT_PENALTY:-1.0}")
# [CGC 2026-09-07 DRY sampling] structural repetition breaker: unlike blanket rp,
# DRY only applies its multiplier when a repeated sequence is DETECTED (Z-algorithm),
# so MTP draft acceptance on healthy text is untouched. Default multiplier 0 = disabled.
# [CGC 2026-09-08] DRY defaults hardened: multiplier 1.0 + allowed-length 10 verified
# on 7-profile replay (coding 0.3->1.0/1.0/1.0 with draft accept back at 89-94%).
# Sequence breakers: DEFAULT includes "\n" which lets newline-separated loops
# (e.g. math echo '\n答：123 × 456 等于多少') escape DRY (rep_limit < allowed_length).
# CGC_SERVER_DRY_BREAKERS (space-separated, replaces defaults; pass '' to disable all).
SERVER_ARGS+=(--dry-multiplier "${CGC_SERVER_DRY_MULTIPLIER:-1.0}")
SERVER_ARGS+=(--dry-allowed-length "${CGC_SERVER_DRY_ALLOWED_LEN:-10}")
SERVER_ARGS+=(--dry-penalty-last-n "${CGC_SERVER_DRY_LAST_N:-512}")
if [ -n "${CGC_SERVER_DRY_BREAKERS+x}" ]; then
    _IFS_OLD="$IFS"; IFS=' '; set -f
    for _b in ${CGC_SERVER_DRY_BREAKERS}; do
        SERVER_ARGS+=(--dry-sequence-breaker "$_b")
    done
    set +f; IFS="$_IFS_OLD"
fi

SERVER_ENV=(
    CGC_EXPERT_CACHE_BYTES="$BUDGET"
    LLAMA_EXPERT_CACHE_ALLOW_NGL=1
    # 2026-09-09 品質修復：L4_SKIP_LAYER0 已證實 4/10 品質殺手（blk.0 走 CPU
    # skip-load 但 graph 在 Metal → cross-backend garbage）。甜蜜點（12:44 log
    # 5976 slots = 預設 uniform caps、layer 0 在 pool 內）無 skip0 = 90-100%。
    LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=0
    LLAMA_EXPERT_CACHE_WORKERS=8
    CGC_WAKE_POLL_US=15
    # [CGC 2026-09-13] CGC_PREFETCH_SRC=hist removed: it is a placement/prefetch heuristic
    # (it re-residents LRU-evicted hot experts from a rolling window), and the routing oracle
    # bounded ALL placement work at +0.2pp of routing mass. Unset = the "step" default, which
    # prefetches this step's own (already resident) union and drops everything. Opt back in with
    # CGC_PREFETCH_SRC=hist for A/B.
    CGC_EVICTED_RING=0
    CGC_N_CB="$SERVER_N_CB"
    CGC_OA_ASYNC="$SERVER_OA_ASYNC"  # §8.77/8.78: +12.6% speed (0000 bug fixed in C++)
    CGC_SERVER_AUTO_ANCHOR="$SERVER_AUTO_ANCHOR"
    CGC_SERVER_DEFAULT_MARKER_STOPS="$SERVER_DEFAULT_MARKER_STOPS"
)
# [CGC 2026-09-13 FIX] CGC_FORCE_TEMP0 used to be exported unconditionally. The C++ side
# tests PRESENCE, not value --  static const bool cgc_force_temp0 = getenv("CGC_FORCE_TEMP0") ? true : false;
# (tools/server/server-common.cpp:1356) -- so exporting the default "0" still ENABLED the
# override. The 2026-09-09 comment above declares the default OFF and says temp 0.1-0.3 both
# avoids the IQ3_XXS noise collapse and breaks the greedy loop, but that fix never took effect:
# every request was forced to temperature=0 (and seed=0, since the same block sets it).
# Measured 2026-09-13 on Edge0-35B-Q4_0-MTP-edge0head with the old export: request temperatures
# 0.0 / 0.4 / 0.9 / 1.5 all returned BYTE-IDENTICAL text, and two independent server starts
# reproduced 44/420 MTP drafts exactly -- greediness proves the override was live.
# This is the same 0-means-unset contract the 0000-guard flags already honour (see the
# "0000 防護 : ... (1=set, 0=unset)" line in the banner), applied where it was missed.
if [ "$CGC_FORCE_TEMP0" = "1" ]; then
    SERVER_ENV+=(CGC_FORCE_TEMP0="1")
fi
# CGC P0: down-combine (Lily-style batch down projection). Pass through if externally set.
if [ -n "${CGC_DOWN_COMBINE:-}" ]; then
    SERVER_ENV+=(CGC_DOWN_COMBINE="$CGC_DOWN_COMBINE")
fi
# CGC hook profiling (diagnostic only, default off)
if [ -n "${CGC_HOOK_PROFILE:-}" ]; then
    SERVER_ENV+=(CGC_HOOK_PROFILE="$CGC_HOOK_PROFILE")
fi
# CGC logits oracle dump (diagnostic only, default off)
if [ -n "${CGC_LOGITS_ORACLE_DUMP:-}" ]; then
    SERVER_ENV+=(CGC_LOGITS_ORACLE_DUMP="$CGC_LOGITS_ORACLE_DUMP")
fi
# CGC exact path debug (diagnostic only, default off)
if [ -n "${CGC_EXACT_PATH_DBG:-}" ]; then
    SERVER_ENV+=(CGC_EXACT_PATH_DBG="$CGC_EXACT_PATH_DBG")
fi
# [CGC 2026-09-13] expert-cache diagnosis pass-throughs (diagnostic only, default off).
# These two oracles are the cheapest way to answer "is the miss traffic attackable?"
# before writing any code:
#   - ROUTE_DUMP writes a PIN_PROFILE-format file (one line per layer, top-K expert ids)
#     plus the top-K coverage verdict. Compare that coverage against K/n_expert -- this
#     model has 256 experts, so the uniform baseline is 143/256 = 55.9%, NOT the
#     "K/128" the log line hard-codes. Coverage far above baseline = heavy-tailed
#     routing = the placement lever is live; near baseline = diffuse = lever closed.
#   - CGC_MASSCOV prints the LIVE membership coverage plus the counterfactual
#     top-K-by-mass coverage at 96/128/192/256 slots: the capacity curve for free,
#     without touching the budget.
# Without these blocks the explicit `env` allowlist on the launch line silently drops
# them, so setting them in the shell appears to do nothing.
if [ -n "${LLAMA_EXPERT_CACHE_ROUTE_RECORD:-}" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_ROUTE_RECORD="$LLAMA_EXPERT_CACHE_ROUTE_RECORD")
fi
if [ -n "${LLAMA_EXPERT_CACHE_ROUTE_DUMP:-}" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_ROUTE_DUMP="$LLAMA_EXPERT_CACHE_ROUTE_DUMP")
fi
if [ -n "${CGC_MASSCOV:-}" ]; then
    SERVER_ENV+=(CGC_MASSCOV="$CGC_MASSCOV")
fi
# CGC Fast-Path Wait: wait for in-flight fills instead of ZERO-mapping (default off)
if [ -n "${CGC_FAST_WAIT:-}" ]; then
    SERVER_ENV+=(CGC_FAST_WAIT="$CGC_FAST_WAIT")
fi
if [ -n "${CGC_FAST_WAIT_US:-}" ]; then
    SERVER_ENV+=(CGC_FAST_WAIT_US="$CGC_FAST_WAIT_US")
fi
if [ -n "${CGC_FAST_WAIT_MAX:-}" ]; then
    SERVER_ENV+=(CGC_FAST_WAIT_MAX="$CGC_FAST_WAIT_MAX")
fi
# [CGC verify-strict 2026-09-13] The ZERO-slot fix now lives in the code: the fast path REFUSES a
# step whose selected expert is still cold (the exact path runs instead) instead of reading the
# reserved zero region and silently dropping that expert's contribution. These knobs are passed
# through so the strictness itself can be A/B'd from the launcher — without them in this allowlist
# `CGC_VERIFY_STRICT=0 ./scripts/run_server.sh` would be silently dropped and the A/B would report
# "no effect" that is really "no knob".
if [ -n "${CGC_VERIFY_STRICT:-}" ]; then
    SERVER_ENV+=(CGC_VERIFY_STRICT="$CGC_VERIFY_STRICT")
fi
if [ -n "${CGC_SYNCFILL_COLD:-}" ]; then
    SERVER_ENV+=(CGC_SYNCFILL_COLD="$CGC_SYNCFILL_COLD")
fi
# CGC pool-path batch cap (C++ default 8): the pool path cannot exceed cap x top_k usable slots,
# so raising it interacts with the pool size and must be visible to the launcher.
if [ -n "${CGC_POOL_MAX_TOKENS:-}" ]; then
    SERVER_ENV+=(CGC_POOL_MAX_TOKENS="$CGC_POOL_MAX_TOKENS")
fi
# CGC prev-token prefetch (default off)
# [CGC M0 decode profile 2026-09-13] Per-layer wait/cb/submit attribution of a decode step.
# Only produces output under CGC_OA_ASYNC=1 (see CGC_SERVER_OA_ASYNC): without the segmented
# dispatcher the whole 41-layer graph is one async submit and none of the three components is
# separable. CGC_DECODE_PROFILE_ALL=1 adds every layer instead of the top 8.
if [ -n "${CGC_DECODE_PROFILE:-}" ]; then
    SERVER_ENV+=(CGC_DECODE_PROFILE="$CGC_DECODE_PROFILE")
fi
if [ -n "${CGC_DECODE_PROFILE_ALL:-}" ]; then
    SERVER_ENV+=(CGC_DECODE_PROFILE_ALL="$CGC_DECODE_PROFILE_ALL")
fi
if [ -n "${CGC_PREV_TOKEN_PREFETCH:-}" ]; then
    SERVER_ENV+=(CGC_PREV_TOKEN_PREFETCH="$CGC_PREV_TOKEN_PREFETCH")
fi
if [ -n "${CGC_PREV_TOKEN_PREFETCH_SYNC:-}" ]; then
    SERVER_ENV+=(CGC_PREV_TOKEN_PREFETCH_SYNC="$CGC_PREV_TOKEN_PREFETCH_SYNC")
fi
# CGC P0: down-combine nsg override (for tuning). Pass through if externally set.
if [ -n "${CGC_DC_NSG:-}" ]; then
    SERVER_ENV+=(CGC_DC_NSG="$CGC_DC_NSG")
fi
# CGC SPAC: EMA top-K prefetch count (for tuning). Pass through if externally set.
if [ -n "${CGC_SPAC_K:-}" ]; then
    SERVER_ENV+=(CGC_SPAC_K="$CGC_SPAC_K")
fi
# CGC SPAC: EMA refresh interval (for tuning). Pass through if externally set.
if [ -n "${CGC_SPAC_REFRESH:-}" ]; then
    SERVER_ENV+=(CGC_SPAC_REFRESH="$CGC_SPAC_REFRESH")
fi
# CGC DBUF: queue capacity cap (for tuning). Pass through if externally set.
if [ -n "${CGC_DBUF_CAP:-}" ]; then
    SERVER_ENV+=(CGC_DBUF_CAP="$CGC_DBUF_CAP")
fi
# CGC MMV: original GEMV kernel threadgroup size (for tuning). Pass through if externally set.
if [ -n "${CGC_MMV_NSG:-}" ]; then
    SERVER_ENV+=(CGC_MMV_NSG="$CGC_MMV_NSG")
fi
# CGC PREFETCH: rolling window size for hist prefetch source (for tuning). Pass through if externally set.
if [ -n "${CGC_PREFETCH_WINDOW:-}" ]; then
    SERVER_ENV+=(CGC_PREFETCH_WINDOW="$CGC_PREFETCH_WINDOW")
fi
# CGC DBUF: hook-time step-ahead refill (production ON by default, b8a564d45 verified +25% coding speed).
# Set CGC_DBUF=0 to disable.
if [ "${CGC_DBUF:-1}" != "0" ]; then
    SERVER_ENV+=(CGC_DBUF=1)
fi
# [CGC 2026-09-13] DEFAULT FLIPPED TO OFF. SpAc is a placement heuristic (EMA-utility prefetch
# source plus EMA victim selection). The routing oracle measured the ceiling of ALL placement
# work at +0.2pp of routing mass — current membership 90.3% vs the best possible top-K-by-mass
# 90.5% at the same 143 slots — so it cannot pay for itself. Keep it reachable for A/B via
# CGC_SPAC=1, but do not enable it by default. Alpha, if opted in, defaults to the tuned 0.75.
if [ "${CGC_SPAC:-0}" != "0" ]; then
    SERVER_ENV+=(CGC_SPAC=1)
    if [ -n "${CGC_SPAC_ALPHA:-}" ]; then
        SERVER_ENV+=(CGC_SPAC_ALPHA="$CGC_SPAC_ALPHA")
    else
        SERVER_ENV+=(CGC_SPAC_ALPHA=0.75)
    fi
fi
# CGC Soft Pool: L0/L1 tier partition. L0 = fixed hot slots (no LRU eviction), L1 = warm LRU
# slots. L0+L1 <= n_slots; L0+L1=0 keeps the legacy uniform n_slots behavior.
# [CGC 2026-09-13] DEFAULT FLIPPED TO 0/0 (partition off). L0 is a hand-picked "fixed hot set",
# which is exactly the placement decision the routing oracle showed is worth at most 0.2pp
# (current 90.3% vs best-possible 90.5%). Opt back in with CGC_SOFT_POOL_L0=48
# CGC_SOFT_POOL_L1=48.
if [ -n "${CGC_SOFT_POOL_L0:-}" ]; then
    SERVER_ENV+=(CGC_SOFT_POOL_L0="$CGC_SOFT_POOL_L0")
else
    SERVER_ENV+=(CGC_SOFT_POOL_L0=0)
fi
if [ -n "${CGC_SOFT_POOL_L1:-}" ]; then
    SERVER_ENV+=(CGC_SOFT_POOL_L1="$CGC_SOFT_POOL_L1")
else
    SERVER_ENV+=(CGC_SOFT_POOL_L1=0)
# [CGC phrase-loop guard 2026-09-08] server-side truncation of live phrase loops
# (>=6 char block x3 consecutive, mirroring the replay quality gate). Default ON;
# set CGC_LOOP_GUARD=0 to disable. CGC_LOOP_GUARD_EVERY = check cadence (tokens).
if [ "${CGC_LOOP_GUARD:-1}" != "0" ]; then
    SERVER_ENV+=(CGC_LOOP_GUARD=1)
    [ -n "${CGC_LOOP_GUARD_EVERY:-}" ] && SERVER_ENV+=(CGC_LOOP_GUARD_EVERY="$CGC_LOOP_GUARD_EVERY")
fi
fi
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
    #
    # [2026-09-12 A/B] 這三個原本寫死，現在可覆寫。預設 1（行為與改動前逐值相同）。
    #   CGC_SERVER_VERIFY_DECODE=0 CGC_SERVER_DRAFT_DECODE=0 ./scripts/run_server.sh
    #
    # ⚠️ 陷阱：C++ 端對這些布林旗標全部用 `getenv(X) != nullptr` 判斷（presence，不是值），
    # 所以 `CGC_VERIFY_DECODE=0` 仍然是「開」。這裡因此把 0 翻成「完全不傳該變數」，
    # 傳 0 才會真的關掉。（第一版 A/B 就是踩到這個：兩臂逐位元相同，等於沒改。）
    SERVER_NO_PREFETCH="${CGC_SERVER_NO_PREFETCH:-1}"
    SERVER_VERIFY_DECODE="${CGC_SERVER_VERIFY_DECODE:-1}"
    SERVER_DRAFT_DECODE="${CGC_SERVER_DRAFT_DECODE:-1}"
    [ "$SERVER_NO_PREFETCH" = "0" ]  || SERVER_ENV+=(CGC_NO_PREFETCH=1)
    [ "$SERVER_VERIFY_DECODE" = "0" ] || SERVER_ENV+=(CGC_VERIFY_DECODE=1)
    [ "$SERVER_DRAFT_DECODE" = "0" ]  || SERVER_ENV+=(CGC_DRAFT_DECODE=1)
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
    if [ "$SERVER_MTP_CLI_PARITY" = "1" ]; then
        SERVER_ENV+=(CGC_MTP_CLI_PARITY=1)
    fi
    if [ "$SERVER_MTP_NO_WARMUP" = "1" ]; then
        SERVER_ENV+=(CGC_MTP_NO_WARMUP=1)
    fi
    # [CGC MTP sampler parity 2026-09-13] pass through the draft-sampler A/B knob. Without this
    # the explicit `env` allowlist on the launch line drops it and the server silently keeps the
    # legacy {TOP_K=10} draft chain, so an A/B would show "no effect" that is really "no knob".
    if [ -n "${CGC_MTP_SAMPLER_PARITY:-}" ]; then
        SERVER_ENV+=(CGC_MTP_SAMPLER_PARITY="$CGC_MTP_SAMPLER_PARITY")
    fi
    # [CGC MTP perf split 2026-09-13] print the spec impl's own phase timings
    # (t_begin / t_draft / t_accept, cumulative) to stderr. Always accumulated, never printed:
    # the existing line is LOG_TRC and the server runs at INFO.
    if [ -n "${CGC_MTP_PERF:-}" ]; then
        SERVER_ENV+=(CGC_MTP_PERF="$CGC_MTP_PERF")
    fi
    # [CGC 2026-09-13] CGC-IDS / CGC-HOOK dumping has NO env gate -- it fires for every batch
    # with n_tokens <= 8 (i.e. every decode step, every verify/draft batch) until a 4000-line
    # budget runs out, via synchronous fprintf(stderr) inside the decode loop. So every run
    # silently pays ~4000 lines of stderr I/O before settling. Pass the budget through so it
    # can be turned off for measurement (CGC_IDS_MAX_LINES=0) instead of only being tunable
    # by editing code.
    if [ -n "${CGC_IDS_MAX_LINES:-}" ]; then
        SERVER_ENV+=(CGC_IDS_MAX_LINES="$CGC_IDS_MAX_LINES")
    fi
    if [ "$SERVER_NO_SEQ_RM_PROBE" = "1" ]; then
        SERVER_ENV+=(CGC_NO_SEQ_RM_PROBE=1)
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
        echo "  Runtime    : CGC_SERVER_RUNTIME_PROFILE=mtp|non-mtp (current=${SERVER_RUNTIME_PROFILE})"
        echo "  Regression : bash scripts/check/check_server.sh --base-url http://127.0.0.1:$PORT/v1"
        if [ "$SERVER_PROFILE" = "qa-zh" ]; then
            echo "  Benchmark  : python3 scripts/benchmark/benchmark_server_profiles.py --base-url http://127.0.0.1:$PORT/v1 --iterations 3"
            echo "  Profile    : qa-zh（中文短答；目前重點在驗證短答起手是否會誤撞結構 token）"
            echo "  Payload    : {\"messages\":[{\"role\":\"user\",\"content\":\"請用一句中文回答：巴黎是哪個國家的首都？\"}],\"max_tokens\":$SERVER_CHAT_AB_MAX_TOKENS,\"stop\":[\"$SERVER_CHAT_AB_STOP\",\"<|end|>\",\"<|output|>\",\"<|user|>\"]}"
        elif [ "$SERVER_PROFILE" = "longform-zh" ]; then
            echo "  Benchmark  : python3 scripts/benchmark/benchmark_server_profiles.py --base-url http://127.0.0.1:$PORT/v1 --iterations 3"
            echo "  Profile    : longform-zh（中文長文；預設前綴可用 env 覆寫）"
            echo "  Payload    : {\"messages\":[{\"role\":\"user\",\"content\":\"請用一段中文說明巴黎為什麼是法國的政治與文化中心，避免條列，至少120字。\"}],\"max_tokens\":$SERVER_CHAT_AB_MAX_TOKENS,\"stop\":[\"<|end|>\",\"<|output|>\",\"<|user|>\"]}"
        elif [ "$SERVER_PROFILE" = "legacy-25plus" ]; then
            echo "  Benchmark  : python3 scripts/check/replay_server_profile.py --base-url http://127.0.0.1:$PORT/v1 --profile longform-zh"
            echo "  Compare    : python3 scripts/benchmark/compare_cli_steady_vs_server.py --server-base-url http://127.0.0.1:$PORT/v1"
            echo "  Profile    : legacy-25plus（還原第一個 25+ server 狀態的 longform 啟動口徑）"
            echo "  Payload    : {\"messages\":[{\"role\":\"user\",\"content\":\"請用一段中文說明巴黎為什麼是法國的政治與文化中心，避免條列，至少120字。\"}],\"max_tokens\":$SERVER_CHAT_AB_MAX_TOKENS,\"stop\":[\"<|end|>\",\"<|output|>\",\"<|user|>\"]}"
        elif [ "$SERVER_CHAT_AB" = "healthy-prefix" ]; then
            echo "  Benchmark  : python3 scripts/benchmark/benchmark_server_profiles.py --base-url http://127.0.0.1:$PORT/v1 --iterations 3"
            echo "  Chat A/B   : healthy-prefix（prefill=答：，僅作短答起手 A/B，不代表已通過 QA gate）"
            echo "  Payload    : {\"messages\":[{\"role\":\"user\",\"content\":\"請用一句中文回答：巴黎是哪個國家的首都？\"}],\"max_tokens\":24,\"stop\":[\"$SERVER_CHAT_AB_STOP\",\"<|end|>\",\"<|output|>\",\"<|user|>\"]}"
        elif [ "$SERVER_CHAT_AB" = "custom-prefix" ]; then
            echo "  Benchmark  : python3 scripts/check/replay_server_profile.py --base-url http://127.0.0.1:$PORT/v1 --profile longform-zh"
            echo "  Chat A/B   : custom-prefix（prefill=${SERVER_CHAT_AB_PREFIX}）"
        fi
        echo "  注意       : 本機 curl 測 localhost 必帶 --noproxy '*'（代理 7897 攔截）"
        echo "  停止       : pkill -INT -f llama-server（或 kill ${SERVER_PID}）"
        if [ "$SERVER_MTP" = "1" ]; then
            echo "  服務模式   : MTP / draft-mtp（目標 = 25+ t/s）"
            echo "  0000 防護  : CGC_NO_PREFETCH=$SERVER_NO_PREFETCH CGC_VERIFY_DECODE=$SERVER_VERIFY_DECODE CGC_DRAFT_DECODE=$SERVER_DRAFT_DECODE (1=set, 0=unset)"
            echo "  OA_ASYNC   : ${SERVER_OA_ASYNC}（+12.6% speed, 0000 bug fixed in C++）"
            echo "  Layer Caps : 40-40:256（MTP draft layer full residency）"
        else
            echo "  服務模式   : 非 MTP 基線（約 ~8 t/s）"
        fi
        echo "=================================================="
        echo ""
        if [ "$CGC_DETACHED" = 1 ]; then
            echo "[detach] server PID=${SERVER_PID}（已脫離父 shell，不受 SIGHUP 影響）"
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
