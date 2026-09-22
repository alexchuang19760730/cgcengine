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
# [CGC 2026-09-15] DEFAULTS FLIPPED 0 -> 1. These two are two of the three bit-identical
# pillars, and the whitepaper (moeexpert/doc/CGC_MTP_Bit-Identical_技術白皮書_2026-08-30.html
# §4.3) is explicit that all three are required, "缺一不可":
#     CGC_MM_BITIDENT=1  (M-invariant kernel choice)
#     CGC_MTP_NO_WARMUP=1 + CGC_NO_SEQ_RM_PROBE=1  (专家缓存污染修复)
# Root cause A in that document is exactly the cache-ON divergence: the MTP context's manual
# warmup decode and the ctx_tgt can_seq_rm probe decode both run an extra decode through the
# expert cache BEFORE the first real request, filling/evicting pool slots. The pool then holds a
# different membership than the no-cache path, the router's near-tie argsort flips, different
# experts are selected, and the outputs diverge from token 0. With the cache OFF there is no
# pool to pollute, which is why "no expert cache" always looked like the stable oracle.
# Both knobs also REMOVE work (they skip two decodes), so this is not a speed/correctness
# tradeoff -- leaving them off was paying for the bug.
SERVER_MTP_NO_WARMUP="${CGC_SERVER_MTP_NO_WARMUP:-1}"    # pass-through: skip CLI-style manual warmup
SERVER_NO_SEQ_RM_PROBE="${CGC_SERVER_NO_SEQ_RM_PROBE:-1}" # pass-through: skip seq_rm probe explicitly
SERVER_N_CB="${CGC_SERVER_N_CB:-8}"  # §8.93: cb8 sweet spot
SERVER_GLU_FUSED_DOWN="${CGC_SERVER_GLU_FUSED_DOWN:-1}"  # §8.113: +6.5% speed
SERVER_WATCHDOG="${CGC_SERVER_WATCHDOG:-1}"  # Metal deadlock watchdog
# §8.77/8.78: async callback split。2026-09-09 把這裡的預設改成 0，但**當時它從來沒有生效**：
# ggml-backend.cpp 的閘門讀的是 `getenv("CGC_OA_ASYNC") != nullptr`（存在，不是值），而這個腳本
# 無條件把它塞進 SERVER_ENV，所以 `0` 與「設成任何字串」都選到**分段**分支——「預設 0」是一個
# 從未被執行的意圖。2026-09-15 閘門改成 value-aware 之後，這個 0 第一次真的生效，乾跑實測
# `CGC_SERVER_PROFILE=off|prefill250|qa-zh|longform-zh` 全部是 `ENV CGC_OA_ASYNC=0`，
# 也就是把這些 profile 從分段靜默降到非分段（同一 build 量測：6.95 → 0.72 t/s，約 10×）。
# 預設因此回到 1，**保留歷史有效行為**；只有顯式 `CGC_SERVER_OA_ASYNC=0` 才選非分段路徑
# （它仍然有效，是 S1 診斷要用的一格）。prod25 內部另外會把 1 寫死一次（外部可覆寫），
# 所以生產口徑不受這裡影響。教訓：`traces/lessons.jsonl` eng-gate-0005。
SERVER_OA_ASYNC="${CGC_SERVER_OA_ASYNC:-1}"
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
    prod25)
        # [CGC 2026-09-15] 复原 2026-09-07 的生产档 —— 那一版是唯一有实测 25.17 t/s 的配置
        # （wall 39.7 ms/token，best 25.87，draft_accept 98.2%，3/3 profiles 品质达标）。
        # 出处：docs/archive/pre-consistency-metrics-2026-09-11/CGC_TPOT_延迟分解与优化路线图_2026-09-07.html §8。
        #
        # 与 prefill250 的分工：那个为 prefill 吞吐优化（6144 chunk + whole-layer slab streaming），
        # 代价是 6144 的 ubatch 把 compute buffer 撑大、加剧 16GB 机器的内存压力，而 decode 只需要
        # 小 batch。这个 profile 反过来：不碰 batch（用模型默认），把 decode 的 TPOT 当作目标。
        #
        # 与当前默认值的差异（就是这两处漂移让 decode 从 25 掉到 5）：
        #   CGC_SPAC=1/alpha 0.75  —— 09-13 被翻成预设关闭（理由是 placement 上限 +0.2pp），
        #                             但 SPAC 同时也是 membership 驱动，关掉后 decode 工作集不驻留
        #   CGC_SERVER_OA_ASYNC=1  —— 09-09 预设改 0（叠加 skip0 时 0/10），生产档是 1（+12.6%）
        # 其余 (verify/draft fast path, no_prefetch, N_CB=8, DBUF, workers=8, 8GiB pool,
        # draft_n_max=3, LAYER_CAPS 40-40:256) 已由通用默认值覆盖。
        [ -z "${CGC_SERVER_MTP+x}" ]            && SERVER_MTP=1
        [ -z "${CGC_SERVER_DENSE_IQ4X+x}" ]     && SERVER_DENSE_IQ4X=1
        [ -z "${CGC_SERVER_OA_ASYNC+x}" ]       && SERVER_OA_ASYNC=1
        [ -z "${CGC_SPAC+x}" ]                  && CGC_SPAC=1
        [ -z "${CGC_SPAC_ALPHA+x}" ]            && CGC_SPAC_ALPHA=0.75
        # bit-identical 三支柱（生产档 §8 全部为 1；全局预设已于 2026-09-15 翻成 1）
        [ -z "${CGC_MM_BITIDENT+x}" ]           && CGC_MM_BITIDENT=1
        [ -z "${CGC_SERVER_MTP_NO_WARMUP+x}" ]  && SERVER_MTP_NO_WARMUP=1
        [ -z "${CGC_SERVER_NO_SEQ_RM_PROBE+x}" ] && SERVER_NO_SEQ_RM_PROBE=1
        # 走 GGUF embedded ChatML（与 qa-zh / coding 一致）
        if [ -z "${CGC_SERVER_CHAT_TEMPLATE_FILE:-}" ] && [ -z "${CGC_SERVER_CHAT_TEMPLATE:-}" ]; then
            SERVER_CHAT_TEMPLATE_FILE=""
        fi
        ;;
    prefill250)
        # [CGC 2026-09-15] 可復現的 prefill 250+ tok/s 口徑（見 docs/PREFILL250_CONFIGURATION_GUIDE_2026-09-15.html
        # 與 264.78 tok/s 的實測：Backup/cgc_logs/llama_server_20260915_011146.log）。
        # 關鍵不是單一旗幟，而是「大 chunk + M2 prefill streaming + 256-expert slab」三件事同時成立：
        #   n_batch = n_ubatch = 6144  -> prefill 一次吃 6144 token，不再被切碎
        #   CGC_PREFILL_STREAM=1      -> 大 chunk 走 whole-layer slab（ne[2]=n_expert）而不是 pool
        #   CGC_GATHER_SLAB_CAP=256   -> slab 裝得下全部 256 個 expert，否則 streaming 會被拒
        # 實際的 batch/ubatch/stream 覆寫在下面的 BUDGET 解析之後（那裡 SERVER_BATCH 才被定案）。
        #
        # [CGC 2026-09-15] 把 dispatcher 旋鈕寫死，理由就是 eng-gate-0006：**profile 不寫的那一格，
        # 就是會漂移的那一格**。prefill250 從來沒設 CGC_SERVER_OA_ASYNC，所以它一路繼承全域預設；
        # 09-15 21:36 把全域預設由 0 改回 1（恢復閘門修正前的實際行為，見 eng-gate-0005）之後，
        # 這個 profile 被靜默重解析，於是 oracle gate 每一跑都報 INVALID COMPARISON：
        #
        #   ENV.CGC_OA_ASYNC: ref='0'  now='1'
        #
        # 1 不是新選擇，而是**這個 profile 一直都是的行為**：v2 ref（
        # Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident.jsonl，17:21）是在
        # presence-based 閘門下 dump 的，當時 `CGC_OA_ASYNC` 只問存在、不問值，而本腳本無條件把
        # 它塞進 SERVER_ENV ⇒ 那個 `0` 選的是**分段**分支，與今天的 `1` 同一條路。獨立證據：
        # 同一份 dump 對 v2 量到 M1 9/9 bit-identical，而非分段路徑的 digest 是 `ff68c5a2`，
        # 不是錨點 `dc055e63`——所以兩邊跑的確實是同一條路。詳見 dec-20260915-2215。
        #
        # [CGC 2026-09-16] 統一：prefill250 也開 SpAc（CGC_SPAC=1 / alpha 0.75）。
        # 在這之前 SPAC 只存在於 prod25 血統，於是「prefill 的臂」與「decode 的臂」是兩個
        # profile —— 而 decode 剩下的槓桿（M1 batched-union gather、M4 verify 真批次）需要
        # whole-layer slab，那只有這個 profile 有；prod25 的池路徑被 cgc_pool_max_tokens()
        # 夾在 n_batch<=8（llama-graph.h:18-37 的註解自己把它綁在 MTP 上）。所以「decode 用
        # prod25」等於把槓桿鎖在一個結構上走不遠的形狀裡。反過來，prefill 這邊只缺 SPAC 一個旋鈕。
        #
        # 依據（否證實驗，不是印象）：交錯 A/B/A/B、每臂等讀數回 NOMINAL、
        # `-b 512 -p 0 -n 128 -d 0,512 -r 3`（Backup/run_unified_ab.sh）：
        #   d512 配對中位 +17.2%（+1.53 / +1.66，兩次同向 10.78/10.91 vs 9.25/9.25）
        #   而且 d512 的離散由 ±2.97 崩到 ±0.04/±0.47（逐 rep 反推 1.94x 擺動 -> 0.7%）
        # ⇒ SPAC 的主要效果不是把均值推上去，是把「decode 工作集不駐留」造成的不穩定拿掉。
        # d0 上兩臂重疊（SPAC 關的那一臂自己的 reps 就跨 5.95-10.58），所以只有 d512 是決定性的。
        # 09-13 把 SPAC 翻成全域預設關的理由是「placement 上限 +0.2pp」（見 docs 的 placement 量測）
        # —— 那個論證只界定了 *placement* 的增益上限，對「decode 工作集是否駐留」不置一詞。
        #
        # 成本，以及為什麼 prefill 必須重量（這是這次修改唯一真正的風險）：
        #   (1) cgc_spac_on() 的 EMA 更新在**每個 routed step** 都跑，註解自承含大 prefill
        #       （llama-context.cpp:4736-4741「Fires on every path that reaches here — large
        #       prefill, MTP draft and trunk verify alike」），且每次取 cache->m 這把鎖；
        #   (2) spac_prefetch 預設 CGC_SPAC_REFRESH=1，即**每步**對每層的非駐留專家做 partial
        #       sort 並排入 prefetch。
        #   ⇒ 兩者都在 prefill 的熱路徑上。所以 prefill 250 的交付數字必須在**同一個 commit**
        #     內重新量過（熱閘門），而不是假定不變。
        #
        # 要回到 09-15 GA 的形狀：CGC_SPAC=0 ./scripts/run_server.sh（profile 只補預設，不寫死）。
        [ -z "${CGC_SPAC+x}" ]                  && CGC_SPAC=1
        [ -z "${CGC_SPAC_ALPHA+x}" ]            && CGC_SPAC_ALPHA=0.75
        [ -z "${CGC_SERVER_OA_ASYNC+x}" ] && SERVER_OA_ASYNC=1
        [ -z "${CGC_SERVER_MTP+x}" ] && SERVER_MTP=1
        [ -z "${CGC_SERVER_DENSE_IQ4X+x}" ] && SERVER_DENSE_IQ4X=1
        [ -z "${CGC_SERVER_CHAT_AB+x}" ] && SERVER_CHAT_AB="custom-prefix"
        [ -z "${CGC_SERVER_CHAT_AB_PREFIX+x}" ] && SERVER_CHAT_AB_PREFIX="巴黎之所以成為法國的政治與文化中心，主要是因為"
        [ -z "${CGC_SERVER_CHAT_AB_MAX_TOKENS+x}" ] && SERVER_CHAT_AB_MAX_TOKENS="220"
        [ -z "${CGC_SERVER_CHAT_AB_STOP+x}" ] && SERVER_CHAT_AB_STOP="<|end|>,<|output|>,<|user|>"
        if [ -z "${CGC_SERVER_CHAT_TEMPLATE_FILE:-}" ] && [ -z "${CGC_SERVER_CHAT_TEMPLATE:-}" ]; then
            SERVER_CHAT_TEMPLATE_FILE=""
        fi
        ;;
    *)
        echo "error: CGC_SERVER_PROFILE must be off|qa-zh|longform-zh|coding|legacy-25plus|prod25|prefill250 (got $SERVER_PROFILE)" >&2
        exit 2
        ;;
esac

# [2026-09-16] MTP_SUPPORT 是**編譯期**開關，不是執行期選項，所以「MTP 跑得起來」推不出
# 「MTP 是照原始碼那條路在跑」。少了 -DMTP_SUPPORT 時照樣會載入 draft context、draft acceptance
# 照樣可以是 1.00000，但被編掉的是三處 [CGC MTP fix]（M-RoPE 的 seq_rm、逐列安全讀取）與
# src/models/qwen35moe.cpp 的 res->t_embd 發佈 —— 也就是我們會在原始碼描述之外的路徑上除錯。
# 實測（同一個 tree，只差這一個 define）：MTP_SUPPORT 專屬字串 0/3 → 3/3。
# 這裡把「產物與原始碼不一致」變成看得見的：MTP 開著就檢查載入中的那份 libllama-common。
# 判準刻意選在「閘門能不能分辨自己有沒有效」上（CONVENTIONS B13）：不只看得到 absent，
# 也要在不該 warning 的時候不 warning，否則它會被當成背景噪音而失效。
if [ "$SERVER_MTP" = "1" ]; then
    _cgc_common="$(dirname "$BIN")/libllama-common.0.dylib"
    if [ -e "$_cgc_common" ]; then
        # [CGC 2026-09-19] `grep -c`, not `grep -q` -- this script is `set -euo pipefail` (line 34)
        # and `grep -q` closes the pipe on first match, so `strings` dies of SIGPIPE (rc=141) and
        # pipefail turns the WHOLE pipeline into a failure. The guard then fires on an artifact
        # that DOES carry -DMTP_SUPPORT. Measured on this tree: `pipefail + grep -q` -> rc=1
        # "warning: MTP=1 但 ... 沒有 -DMTP_SUPPORT" while `strings -a libllama-common.0.dylib |
        # grep -c 'MTPDBG mtp_ctor'` -> 3; without pipefail the same test is silent.
        #
        # Why this one mattered more than a cosmetic warning: every MTP=1 launch printed "產物與
        # 原始碼不一致", and that sentence is acted on -- 2026-09-19 a parallel session recorded
        # MTP numbers as unattributable because of it. A guard that always says "broken" trains
        # its reader to ignore it and costs real measurements. The exact same defect class is
        # already documented 500 lines below (CGC_METAL_LIB), which is where the pattern is from.
        _cgc_hits="$(strings -a "$_cgc_common" 2>/dev/null | grep -c 'MTPDBG mtp_ctor' || true)"
        if [ "${_cgc_hits:-0}" -eq 0 ]; then
            echo "warning: MTP=1 但 $(basename "$_cgc_common") 沒有 -DMTP_SUPPORT 編出來的程式碼。" >&2
            echo "         MTP 路徑的 [CGC MTP fix] 與 qwen35moe 的 t_embd 都會被編掉，產物與原始碼不一致。" >&2
            echo "         重建：cmake -B src/llama.cpp/build -DLLAMA_BUILD_SERVER=ON -DCMAKE_CXX_FLAGS=-DMTP_SUPPORT" >&2
            echo "               cmake --build src/llama.cpp/build -j8" >&2
        fi
        unset _cgc_hits
    fi
    unset _cgc_common
fi

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

# [CGC prefill250 2026-09-15] batch/ubatch 在上面的 case 之後才被定案，所以 profile 的
# 覆寫必須放在這裡，而不是放進 case（那裡設的值會被 317/318 行蓋掉）。
# 全部用 ${VAR+x} 守門：顯式給的環境變數永遠贏，profile 只補預設值。
# [CGC 2026-09-15] prod25：CTX/BUDGET/draft_n_max 都在 case 之後才定案（CTX 在下面幾行、
# BUDGET 在上面、SPEC_DRAFT_N_MAX 在 case 之前就被讀走），所以必須在這裡覆寫。
# 顯式環境變數永遠贏：全部用 ${VAR+x} 守門。
if [ "$SERVER_PROFILE" = "prod25" ]; then
    # 4096 而非 8192：生產檔的 ctx。decode 的 attention 成本大致與 ctx 成正比，而 8192 是為了
    # 讓 Claude Code CLI（最小 ~3117 token prompt）不撞 HTTP 400 才加的，一般 decode 不需要。
    [ -z "${CGC_SERVER_CTX+x}" ]                 && CTX=4096
    [ -z "${CGC_SERVER_EXPERT_CACHE_BYTES+x}" ]  && BUDGET=8589934592
    # batch/ubatch 留空 = 模型預設。6144 的 ubatch 只在 prefill250 有意義。
    [ -z "${CGC_SERVER_MTP_N_MAX+x}" ]           && SPEC_DRAFT_N_MAX=3
fi

if [ "$SERVER_PROFILE" = "prefill250" ]; then
    # [CGC 2026-09-16 13:3x] 6144 -> 5632. The 6144 chunk was this profile's defining number, and on
    # the current build it does not survive: at the profile's own 8 GiB pool, `ub=6144` died at req2
    # in 5 of 5 independent launches today (12:05, 13:15, 13:17, 13:32, 13:33; two were 12+ min
    # apart, three A11 fingerprints involved), every time with the same nameable cause --
    # `status 5 kIOGPUCommandBufferCallbackErrorOutOfMemory` on the second request.
    #
    # 5632 is the highest width MEASURED to survive, at the profile's own pool size, with the
    # measurement interleaved against the failing control so machine drift cannot explain it:
    #     none ub=4096 pool 8GiB  survived 5/5
    #     none ub=5120 pool 8GiB  survived 3/3
    #     none ub=5632 pool 8GiB  survived 4/4   <- this default
    #     none ub=6144 pool 8GiB  survived 0/5   <- the old default
    #     none ub=6144 pool 6GiB  survived 3/3   (the alternative; see [budget] below)
    #     mmap ub=4096/5120/6144  survived 0/3   (load-mode mmap, any width)
    # Evidence: Backup/cgc_logs/{ub_ladder,prod_choice,pool_axis,mmap_axis}/summary.tsv, all arms on
    # common_md5 7bceb3bd5320. 5632 also ranked faster than the 6 GiB-pool alternative in BOTH
    # interleaved rounds (req2 170.6 vs 153.2, then 167.2 vs 158.0), so it is not a speed sacrifice.
    #
    # It is NOT claimed to reproduce the profile's name on demand, and it does not need to: what
    # changed is that it now SURVIVES. The acceptance run on this default (13:45:10, no overrides,
    # OUTDIR=Backup/cgc_logs/prod_accept) passed all three requests at prefill
    #     268.09 / 273.55 / 256.61 t/s   (A11 a9c9dc10f057bf640e0132a83e01c12)
    # with 0 crash reports -- i.e. the profile's own target, on the shipped default, measured. But
    # the SAME fingerprint in the interleaved rounds above gave 147-180 t/s, so quote throughput
    # only with the machine state attached: the spread is environmental (identical A11 fingerprints
    # differ by up to 2x), whereas survival is binary and reproduced 5/5 across two rounds.
    # 250+ tok/s was also measured on 2026-09-15 (264.78 t/s) and once today (284.58 t/s at 12:51).
    # Keep the old width explicitly if you need it: CGC_SERVER_UBATCH=6144 CGC_SERVER_BATCH=6144.
    [ -z "${CGC_SERVER_BATCH+x}" ]      && SERVER_BATCH=5632
    [ -z "${CGC_SERVER_UBATCH+x}" ]     && SERVER_UBATCH=5632
    [ -z "${CGC_SERVER_CTX+x}" ]        && CTX=8192
    [ -z "${CGC_PREFILL_STREAM+x}" ]    && CGC_PREFILL_STREAM=1
    [ -z "${CGC_GATHER_SLAB_CAP+x}" ]   && CGC_GATHER_SLAB_CAP=256
    # [CGC 2026-09-15] expert pool 必須是 8GiB，不能用預設的 10GiB。
    #   264.78 tok/s 那筆量測（Backup/cgc_logs/llama_server_20260915_011146.log）跑的是
    #   143 slots/layer = 8GiB；profile 若沿用 10GiB（179 slots/layer），多出來的 2GB pool
    #   會把 6144-token ubatch 的 compute buffer 擠出 GPU，第一個請求就
    #       ggml_metal_synchronize: command buffer failed with status 5
    #       Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory) ret=-3
    #   —— 2026-09-15 用 CGC_SERVER_PROFILE=prefill250 實跑就是這樣炸的。
    #   這是「能在 run_server.sh 復現」的真正門檻：三要素之外還得有第四要素（pool 尺寸）。
    [ -z "${CGC_SERVER_EXPERT_CACHE_BYTES+x}" ] && BUDGET=8589934592
    # 8192 是 16GB 機器上 verified 可載入的上限；6144 的 chunk 需要它才不會被截斷
    if [ -n "${CGC_SERVER_BATCH:-}" ] || [ -n "${CGC_SERVER_UBATCH:-}" ]; then
        : # 使用者顯式指定，尊重
    fi
    # [CGC 2026-09-15] 這個 profile 自帶 MTP，而 MTP verify batch 的 n_tokens = n_draft+1 > 1 ——
    # 它是唯一一條「remap 必須一次寫滿所有 draft token 的 expert union」的路徑，也是最可疑的一條
    # （所有已完成的 probe 都沒開 MTP，所以從來沒被檢查過）。要看它就帶 CGC_MMID_MV_DBG=2：
    #     CGC_SERVER_PROFILE=prefill250 CGC_MMID_MV_DBG=2 ./scripts/run_server.sh
    # 兩條路徑會在同一個 log 裡交錯出現，用 ne02 分辨：ne02=slots(71/143) 是 pool 路徑，
    # ne02=256 是 M2 prefill whole-layer slab 路徑。
fi
LOG_DIR="$ROOT/Backup/cgc_logs"
LOG="$LOG_DIR/llama_server_$(date +%Y%m%d_%H%M%S).log"

case "$SERVER_MEMORY_MODE" in
    dev|prod) ;;
    *)
        echo "error: CGC_SERVER_MEMORY_MODE must be dev|prod (got $SERVER_MEMORY_MODE)" >&2
        exit 2
        ;;
esac

# 這個數字餵給 memory guard 的 OTHER_LLAMA_SERVERS，而那個門檻在**每一個** profile 上都是 0
# （見 cgc_memory_guard_req：full-mtp-known-profile 是 "0 35 0"、full-mtp 是 "0 40 0" …）
# ⇒ 只要多數到 1 就 "startup blocked by memory guard" + exit 1，整臂被擋掉。
#
# 所以它必須數「真的是 llama binary」的行程，不能數「命令列裡出現過那個路徑」的行程 ——
# 後者包含任何 `bash -c "... build/bin/llama-server ..."` 的 wrapper、grep、或記錄用的 echo。
# 實測（2026-09-17）：造兩個這種行程，舊版回 count=2，本版回 count=0。
#
# 判準與 cgc_preflight_pids 一致（pgrep 蒐集候選、ps 確認第一個 token 的 basename），
# 順帶涵蓋 llama-cli / llama-bench / llama-simple —— 它們同樣搶記憶體，而這正是本 guard 在問的事。
#   ※ CGC_PREFLIGHT_NAMES 的註解在 2026-09-15 記過同一個坑（一支 Doubao agent 的
#     `/bin/bash -c ... ./bin/llama-cli ...` wrapper 被誤 SIGTERM）：清理路徑修了，這裡漏了。
#   ※ cgc_preflight_pids 定義在 676 行（本函式之後），但本函式唯一的呼叫點在 799 行
#     ⇒ 執行時它已經定義好了。要在更早的地方呼叫就得先搬定義。
cgc_existing_llama_server_count() {
    cgc_preflight_pids | wc -l | tr -d ' '
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

[ -x "$BIN" ] || {
    echo "error: llama-server 不存在：${BIN}" >&2
    echo "  構建（MTP 要開就必須帶 -DMTP_SUPPORT；少了它 MTP 仍能跑，但被編掉的是修正，不是除錯）：" >&2
    echo "    cmake -B src/llama.cpp/build -DLLAMA_BUILD_SERVER=ON -DCMAKE_CXX_FLAGS=-DMTP_SUPPORT" >&2
    echo "    cmake --build src/llama.cpp/build -j8" >&2
    exit 1
}
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
#
# [CGC 2026-09-15 preflight-kill] 為什麼這裡從「清殘留」升級成「起跑前必做的 preflight」：
#   16GB unified-memory 機器上，前一支 llama-server 就算已經被 kill，Metal 端的 GPU buffer
#   還沒被回收。這時候立刻起新行程，llama-server 會在第一次 command buffer 提交時拿到
#       kIOGPUCommandBufferCallbackErrorOutOfMemory ret=-3
#   也就是「GPU 說沒記憶體、但 memory_pressure 看起來還有」的那一種炸法 —— 2026-09-15
#   第一次跑 prefill250 就是這樣炸的（當下 free 只剩 67MB、swap 6.1GB），而且炸在載入模型
#   之後，浪費掉整段 load 時間。所以這裡做四件事，缺一不可：
#     1. SIGTERM 優雅退出（讓 Metal buffer 走正常釋放路徑），等最長 TERM_WAIT
#     2. 只有「等不到」的才 SIGKILL（直接 -9 會漏：10 次崩潰累積 ~8GB，vramFree 剩 15MB）
#     3. kill 之後「等」：輪詢 pgrep 真的歸零，而不是 sleep 1 就當沒事
#     4. 等 free% 回到 SETTLE 門檻（GPU 回收是 async 的，行程消失 ≠ 記憶體回來）
#   關閉：CGC_PREFLIGHT_KILL=0（只在你確定沒有殘留行程時用，例如 CI 或手動跑第二支）
#
# pattern 用「build/bin/llama-*」子字串：行程可能是絕對路徑或相對路徑啟動（sandbox 用相對），
# 絕對路徑 pattern 比對不到相對路徑行程 → port 衝突 → 新行程秒退（2026-08-30 實測踩過）。
# [CGC 2026-09-15] 另外補上 bin/ 下的同名 binary（部分 worktree 會把 binary 複製到 bin/），
# 以及 llama-cli / llama-bench —— 它們一樣吃 GPU buffer，一樣會把新 server 擠到 OOM。
CGC_PREFLIGHT_KILL="${CGC_PREFLIGHT_KILL:-1}"
CGC_PREFLIGHT_TERM_WAIT_SEC="${CGC_PREFLIGHT_TERM_WAIT_SEC:-15}"
CGC_PREFLIGHT_MIN_FREE_PCT="${CGC_PREFLIGHT_MIN_FREE_PCT:-25}"
CGC_PREFLIGHT_SETTLE_SEC="${CGC_PREFLIGHT_SETTLE_SEC:-20}"
CGC_PREFLIGHT_PATTERNS=(
    "build/bin/llama-server"
    "build/bin/llama-simple"
    "build/bin/llama-speculative-simple"
    "build/bin/llama-cli"
    "build/bin/llama-bench"
    "bin/llama-server"
    "bin/llama-cli"
)

# 只有這些 binary 才算「殘留的 llama」。ps 拿得到 command 時用它分類，避免誤殺：
# 2026-09-15 實測，純 pattern 比對會打到「命令列裡剛好出現 llama 路徑」的上層 wrapper
# shell（一支 Doubao agent 的 `/bin/bash -c ... ./bin/llama-cli ...`），把它 SIGTERM 掉是
# 純粹的附帶傷害。這裡改成：pgrep 只負責蒐集候選，ps 負責確認「第一個 token 的 basename」
# 真的是一支 llama binary。ps 拿不到資料時保守放行（寧可多清，不要 OOM）。
CGC_PREFLIGHT_NAMES=(llama-server llama-simple llama-speculative-simple llama-cli llama-bench)
if [ -n "${CGC_PREFLIGHT_EXTRA_NAMES:-}" ]; then
    for _n in ${CGC_PREFLIGHT_EXTRA_NAMES}; do
        CGC_PREFLIGHT_NAMES+=("$_n")
    done
fi

cgc_preflight_cmd() {
    ps -o command= -p "$1" 2>/dev/null | head -1 || true
}

# `ps -o pid=,etime=,command=`：把「跑了多久」也印出來 —— 要判斷一支行程是自己的殘留還是別人
# 正在跑的量測，「年齡」是最有用的單一資訊（幾秒前才起來的，八成是別人剛發的量測）。
cgc_preflight_desc() {
    ps -o pid=,etime=,command= -p "${1}" 2>/dev/null | cut -c1-150 || true
}

# 去重後的殘留 pid 清單（pgrep 一次一個 pattern，多個 pattern 會重複命中同一支行程）
cgc_preflight_pids() {
    local p pid cmd exe base n cands
    cands=$(
        for p in "${CGC_PREFLIGHT_PATTERNS[@]}"; do
            # pgrep 沒命中會回 1；在 pipefail 下會殺掉腳本，所以必須 || true
            pgrep -f "$p" 2>/dev/null || true
        done | sort -u
    ) || true
    for pid in $cands; do
        [ -n "$pid" ] || continue
        cmd="$(cgc_preflight_cmd "$pid")"
        if [ -z "$cmd" ]; then
            echo "$pid"                       # ps 不可用 → 保守列入
            continue
        fi
        exe="${cmd%% *}"
        base="${exe##*/}"
        for n in "${CGC_PREFLIGHT_NAMES[@]}"; do
            if [ "$base" = "$n" ]; then
                echo "$pid"
                break
            fi
        done
    done
}

cgc_preflight_count() {
    local n
    n=$(cgc_preflight_pids | wc -l | tr -d ' ')
    echo "${n:-0}"
}

cgc_preflight_signal() {
    # $1 = signal；逐 pid 送，不用 pkill -f（pattern 會打到 wrapper，見上）
    local sig="$1" pid
    cgc_preflight_pids | while read -r pid; do
        [ -n "$pid" ] || continue
        kill "$sig" "$pid" 2>/dev/null || true
    done || true
}

# [CGC 2026-09-18 no-cross-kill] 預設從「送訊號清場」改成「只報告」。理由：2026-09-18 同一天裡
# 兩支正在量的 server 被外部 SIGTERM 打死（日誌 `[CGC] Received SIGTERM`），不是 watchdog（走
# GGML_ABORT）、不是 OOM —— 兇手就在這一段：pattern + `pgrep -f` **不看 port、不看 session**，
# 任何一條線起 `run_server.sh` 都會把別人的 llama 一起 SIGTERM 掉；而且這段**排在 memory guard
# 之前**，所以就算自己隨後被 guard 擋下、沒能起來，被殺的那一輪也已經死了（對方的量測整天白跑，
# 自己這輪也沒成果）。先前的「nohup／進程組連坐」假設已被證偽：第二次死亡發生在前景命令還活著時。
#
# 三個取值：
#   CGC_PREFLIGHT_KILL=1    （預設）列出撞到的行程，**不送任何訊號**；由下面的 STALE 硬檢查擋住啟動
#   CGC_PREFLIGHT_KILL=all  舊行為：SIGTERM 全部（你確定機器上只有你自己在用）
#   CGC_PREFLIGHT_KILL=0    完全略過 preflight
# NOTE：`CGC_DUMP_ENV=1` 只是把解析好的 env/argv 印出來、**不會真的起 server**，所以它絕對不能
#       有任何殺人副作用 —— llama_bench_matrix / prod_matrix / m123_oracle_gate / phase_split_ab
#       都是靠它取值（一次的「查 env」不該把別人的量測打死）。
if [ "$CGC_PREFLIGHT_KILL" != "0" ] && [ "${N30CACHE_NO_CLEAN:-0}" != "1" ] && [ "${CGC_DUMP_ENV:-}" != "1" ]; then
    PRE_N="$(cgc_preflight_count)"
    if [ "${PRE_N:-0}" -gt 0 ]; then
        if [ "$CGC_PREFLIGHT_KILL" = "all" ]; then
            echo "[preflight] 發現 ${PRE_N} 支殘留 llama 行程，先清乾淨再起 server（避免 GPU OOM ret=-3）"
            cgc_preflight_pids | while read -r pid; do
                echo "  [preflight]   $(cgc_preflight_desc "${pid}")"
            done || true
            cgc_preflight_signal -TERM
            echo "  [preflight] SIGTERM 已送（graceful，讓 Metal buffer 正常釋放）"
            # 等 graceful shutdown：TERM_WAIT 秒，每 0.5s 檢查一次
            TERM_TICKS=$(( CGC_PREFLIGHT_TERM_WAIT_SEC * 2 ))
            for i in $(seq 1 "${TERM_TICKS:-30}"); do
                sleep 0.5
                [ "$(cgc_preflight_count)" -eq 0 ] && break
            done
            # 只剩不回應的才 SIGKILL
            if [ "$(cgc_preflight_count)" -gt 0 ]; then
                cgc_preflight_signal -9
                echo "  [preflight] WARNING: SIGKILL 不回應的行程（GPU 記憶體可能漏）"
                sleep 2
            fi
        else
            echo "[preflight] 發現 ${PRE_N} 支 llama 行程（可能是別條 session 正在量測）→ 不送任何訊號" >&2
            cgc_preflight_pids | while read -r pid; do
                echo "  [preflight]   $(cgc_preflight_desc "${pid}")" >&2
            done || true
            echo "  [preflight] 確定是自己的殘留才手動清：kill -TERM <pid>（讓 Metal buffer 正常釋放，不要一開始就 -9）" >&2
            echo "  [preflight] 或明確授權清場：CGC_PREFLIGHT_KILL=all ./scripts/run_server.sh" >&2
            echo "  [preflight] 下面的 STALE 硬檢查會擋住啟動；確定要硬闖：CGC_PREFLIGHT_SKIP_STALE_CHECK=1" >&2
        fi
    else
        echo "[preflight] 無殘留 llama 行程"
    fi

    # kill 之後不等於安全：行程消失後 Metal 還要時間把 buffer 還回來。
    # 這裡輪詢到 free% >= CGC_PREFLIGHT_MIN_FREE_PCT，最多等 SETTLE_SEC 秒。
    SETTLE_TICKS=$(( CGC_PREFLIGHT_SETTLE_SEC * 2 ))
    for i in $(seq 1 "${SETTLE_TICKS:-40}"); do
        PRE_FREE="$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/{print int($2)}' || true)"
        if [ -z "${PRE_FREE:-}" ] || [ "${PRE_FREE:-0}" -ge "$CGC_PREFLIGHT_MIN_FREE_PCT" ]; then
            break
        fi
        if [ "$i" -eq 1 ]; then
            echo "[preflight] 等 GPU/系統記憶體回穩（free=${PRE_FREE}% < ${CGC_PREFLIGHT_MIN_FREE_PCT}%，最多 ${CGC_PREFLIGHT_SETTLE_SEC}s）"
        fi
        sleep 0.5
    done
    [ -n "${PRE_FREE:-}" ] && echo "[preflight] free=${PRE_FREE}%"
fi

# [CGC 2026-09-18] CGC_DUMP_ENV=1 是「只解析、不起 server」的模式：之後每一個「能不能起 server」
# 的閘門（STALE 硬檢查、memory guard）對它都沒有意義，而且會讓你在**別條 session 正在跑**的時候
# 查不到任何東西 —— 偏偏那正是最需要知道真相的時候。所以這裡立一個旗標，閘門全部讓路。
DUMP_ONLY=0
[ "${CGC_DUMP_ENV:-}" = "1" ] && DUMP_ONLY=1

# 硬性攔截：kill/等待都做完了還有殘留，就在這裡停，不要讓它炸在模型載入之後。
STALE_N="$(cgc_preflight_count)"
if [ "${STALE_N:-0}" -gt 0 ] && [ "$DUMP_ONLY" = "0" ]; then
    echo "error: 仍有 ${STALE_N} 支 llama 行程，繼續啟動極可能 GPU OOM (ret=-3)" >&2
    echo "  （preflight 預設已不再幫你清場：2026-09-18 它把別條 session 正在量的 server SIGTERM 掉了）" >&2
    cgc_preflight_pids | while read -r pid; do
        echo "  $(cgc_preflight_desc "${pid}")" >&2
    done || true
    echo "  第二欄是 etime（跑了多久）：幾秒鐘的多半是別人剛發的量測，不要殺。" >&2
    echo "  確定是自己的殘留：kill -TERM <pid>（等它自己退，不要一開始就 -9）" >&2
    echo "  確定是自己一個人在用機器：CGC_PREFLIGHT_KILL=all ./scripts/run_server.sh" >&2
    echo "  或在充分理解風險下跳過：CGC_PREFLIGHT_SKIP_STALE_CHECK=1" >&2
    [ "${CGC_PREFLIGHT_SKIP_STALE_CHECK:-0}" = "1" ] || exit 1
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
if [ "$PHYS_MEM_GB" -lt "$MEM_REQ_PHYS_GB" ] || [ "$FREE_PCT" -lt "$MEM_REQ_FREE_PCT" ] || [ "$OTHER_LLAMA_SERVERS" -gt "$MEM_REQ_OTHER" ] && [ "$DUMP_ONLY" = "0" ]; then
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

if [ "$PHYS_MEM_GB" -lt "$MEM_REQ_PHYS_GB" ] || [ "$FREE_PCT" -lt "$MEM_REQ_FREE_PCT" ] || [ "$OTHER_LLAMA_SERVERS" -gt "$MEM_REQ_OTHER" ] && [ "$DUMP_ONLY" = "0" ]; then
    MEM_REASON="$(cgc_memory_guard_reason "$MEM_CLASS" "$PHYS_MEM_GB" "$FREE_PCT" "$OTHER_LLAMA_SERVERS" "$MEM_REQ_PHYS_GB" "$MEM_REQ_FREE_PCT" "$MEM_REQ_OTHER")"
    echo "error: even fallback memory guard is not satisfied -> $MEM_REASON" >&2
    echo "hint: 這代表目前機器狀態連保命線都撐不住，先關掉其他重工行程後再跑。" >&2
    exit 1
fi

# ── 判定：這個 binary 有沒有 pool/mmap 修復？（[防護 2d] 與 [防護 2e] 都用它）
# 不用版本號、也不用腳本自己的記憶，而是問 binary：修復會讓 L4 pool 走到一個名字含 "_Pool" 的
# buffer type（ggml-metal.cpp:ggml_backend_metal_buffer_type_pool），而那個字串常數就在
# libggml-metal 裡。所以 `strings libggml-metal*.dylib | grep -q '_Pool'` 是「含/不含修復」的
# 可檢查判準。四面對照：有修復的 lib -> 1；改旗標前的舊產物
# （Backup/pre_flagalign_20260916/libggml-metal.0.19.0.dylib）-> 0。
# 可覆寫是為了讓閘門的「不含修復」那一面可以被證明：指到改旗標前的舊產物
# （Backup/pre_flagalign_20260916/libggml-metal.0.19.0.dylib）就必須拒跑。
#
# 用 `grep -c` 而不是 `grep -q`：本腳本第 34 行是 `set -euo pipefail`，而 grep -q 一命中就
# 立刻關閉 pipe，上游 strings 收 SIGPIPE（`strings: failed to flush output`，rc=141），
# pipefail 於是把整條 pipeline 判成失敗 —— 也就是說 `grep -q` 在這裡會讓**含修復的 binary
# 被判成不含修復**（實測：pipefail+grep -q rc=1、pipefail+grep -c rc=0 n=1、
# 無 pipefail+grep -q rc=0）。這是「閘門 fail-closed 的假陰性」，比漏擋更危險，因為它會讓人
# 把修好的路逕判成不可用。grep -c 讀完整條流，沒有 SIGPIPE 問題。
CGC_METAL_LIB="${CGC_METAL_LIB:-$(ls "$(dirname "$BIN")"/libggml-metal*.dylib 2>/dev/null | head -1)}"
CGC_MMAP_POOL_FIX=0
if [ -n "$CGC_METAL_LIB" ]; then
    _n_pool="$(strings -a "$CGC_METAL_LIB" 2>/dev/null | grep -c '_Pool' || true)"
    [ "${_n_pool:-0}" -gt 0 ] && CGC_MMAP_POOL_FIX=1
fi

# [防護 2d / 2026-09-16] 啟動預算：把 OOM 從形容詞變成算式。
#
# 為什麼要有這一塊：req2 的 OOM（白皮書 §11.10.5）炸在 prefill 開頭，log 只留
#   Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
# —— 那句話不含任何數字。而真正可算的三項，在載入模型之前就全部已知：
#
#   1. 模型檔位元組（stat -f%z）。--load-mode none（= --no-mmap）把它讀成匿名頁，**不可回收**；
#      mmap 則是 file-backed 的 clean page，OS 隨時可回收。所以「算不算進需求」取決於 load mode。
#      [CGC 2026-09-16 13:3x，修正] 這一條原本接著寫「而且那個修復正是 req2 OOM 的機制解」。
#      **那句被實測推翻了。** mmap × expert cache 的唯讀映射缺陷確實修好了（載入不再 SIGBUS，
#      見 [防護 2e]），但修好之後 mmap 在任何量過的寬度都不存活：
#          mmap ub=4096 / 5120 / 6144 各 1 次 → 全部 ready=no、0/3
#      （死法：載完 40 層後在第一個 ggml_metal_synchronize 以 status 5 OOM 死）。
#      而 `recommended max working set` 這條警告在 119 次有紀錄的 load_mode=none 啟動裡出現
#      **0 次**，在 5 份可判定 load_mode 的警告日誌裡是 **5/5 mmap** ⇒ mmap 讓 Metal 的工作集
#      變大。可回收的是 host 頁，被撐大的是 GPU 工作集，而這台機器撞的是後者。
#      所以：mmap 不是 OOM 的槓桿，下面也不再把它列為槓桿。
#   2. expert pool 的上限（BUDGET）。**它是上限、不是實配**：實際槽數另受
#      LLAMA_EXPERT_CACHE_LAYER_CAPS 限制（prefill250 走 "40-40:256"，載入時實測 5976 槽）。
#      實測換算：M2 whole-layer slab 在 256 experts 時是 110 MiB/層
#      （log：CGC-PREFILL-STREAM ... experts=256 slab=110.00 MiB），
#      因此 5976 槽 ≈ 5976/256 × 110 MiB ≈ 2.5 GiB —— 也就是說 8 GiB 的 BUDGET 對這台機器是
#      「喊出來的數字」，真正的池子比它小。所以下面**不**把兩者混成單一結論，兩個數都印。
#   3. KV cache。這顆是 qwen35moe，而 src/models/qwen35moe.cpp:28 是
#      is_recr_impl[i] = ((i + 1) % full_attn_interval != 0)，帶 full_attention_interval=4
#      ⇒ 41 層裡只有 i=3,7,11,…,39 這 **10 層**帶 KV，其餘 31 層是 recurrent（線性注意力）。
#      KV/token = 10 層 × n_head_kv(2) × (key 256 + value 256) × 2B = 20 KiB/token
#      ⇒ KV@ctx=8192 ≈ 160 MiB。**它不是瓶頸**；算進來只是為了不讓「先怪 KV」取代算術。
#      （§11.11.7 裁決過：這裡原本寫 164 MiB，是 KiB/MiB 換算算錯；正確是 160 MiB。）
#      注意：recurrent 快取（rs_size）**不在**這個數字裡，而且它隨 n_ubatch 成長（不是隨 ctx），
#      所以 ub=6144 除了撐大 compute buffer 之外還有這一項成本。本行不列入總和。
#
# 判準（CONVENTIONS B7）：這一段永遠印，並且在 demand > 實體記憶體 時多印 OVERSUBSCRIBED。
# 預設只警告不拒跑 —— prefill250 本來就是以這種方式在跑，硬擋會讓 D5 閘門與所有既有量測都起不來。
# 要它在超額時直接拒跑：CGC_SERVER_STRICT_BUDGET=1。
CGC_SERVER_STRICT_BUDGET="${CGC_SERVER_STRICT_BUDGET:-0}"
_MIB=1048576
MODEL_BYTES="$(stat -f%z "$MODEL" 2>/dev/null || echo 0)"
case "$SERVER_LOAD_MODE" in
    mmap|mmap+mlock) _MODEL_RESIDENT=0; _MODEL_KIND="file-backed、OS 可回收" ;;
    *)               _MODEL_RESIDENT="$MODEL_BYTES"; _MODEL_KIND="匿名頁、不可回收" ;;
esac
_b_model=$(( MODEL_BYTES    / _MIB ))
_b_res=$(( _MODEL_RESIDENT / _MIB ))
_b_pool=$(( BUDGET         / _MIB ))
_b_phys=$(( PHYS_MEM_BYTES / _MIB ))
_b_dem=$(( _b_res + _b_pool ))
echo "[budget] model       ${_b_model} MiB（load_mode=$SERVER_LOAD_MODE => ${_MODEL_KIND}）"
echo "[budget] expert pool ${_b_pool} MiB（BUDGET 是上限，實配另受 LAYER_CAPS 限制；實測 110 MiB/層@256 experts ≈ 2.5 GiB@5976 槽）"
echo "[budget] KV          ctx=${CTX}；主 context 走 hybrid memory，filter_attn = (il<40 && !is_recr(il))" >&2
echo "[budget]             （llama-model.cpp:2292-2295 + llama-memory-hybrid.cpp:48-50）=> 只有 10 層帶 K/V（i=3,7,…,39，full_attention_interval=4）；" >&2
echo "[budget]             每層 2 kv-head × (256k+256v) × 2B = 2 KiB/token => fp16 KV@8192 = 160 MiB，q8_0 ≈ 85 MiB —— 非瓶頸（算術，未實測）" >&2
echo "[budget] 靜態需求    ${_b_dem} MiB（model resident ${_b_res} + pool ${_b_pool}） vs 實體 ${_b_phys} MiB"
if [ "$_b_phys" -gt 0 ] && [ "$_b_dem" -gt "$_b_phys" ]; then
    echo "[budget] OVERSUBSCRIBED by $(( _b_dem - _b_phys )) MiB：這個啟動只能在 macOS 記憶體壓縮 + swap 之上執行。"
    echo "[budget] 槓桿：CGC_SERVER_EXPERT_CACHE_BYTES（池上限）／CGC_SERVER_UBATCH（compute buffer 與 recurrent 快取都隨 ub 成長）／CGC_SERVER_CTX（影響很小：KV@8192 只有約 160 MiB）"
    # [CGC 2026-09-16 13:3x，修正] 這裡原本把 mmap 列為「最直接的一根」。實測推翻了：修好之後
    # mmap 在 4096/5120/6144 三個寬度全部不存活（0/3，見 [防護 2d] 第 1 條與 §11.12）。
    # 它現在的正確定位是「一個已修好的缺陷，但代價是 Metal 工作集變大」——不是槓桿。
    # 保留 CGC_MMAP_POOL_FIX 的判定，是為了讓訊息說得出「你這個 binary 有沒有那個修復」。
    if [ "${CGC_MMAP_POOL_FIX:-0}" = "1" ]; then
        echo "[budget] mmap（**不是**槓桿，不要再試）：這個 binary 含 pool/mmap 修復（[防護 2e]），所以載入不會 SIGBUS，"
        echo "[budget]   但修好之後它在 4096/5120/6144 全部不存活（0/3），而且工作集警告只在 mmap 出現 ⇒ 它把 GPU 工作集撐大。"
    else
        echo "[budget] mmap（**不是**槓桿）：這個 binary **不含** pool/mmap 修復（[防護 2e]）—— 那條路會在載入階段硬崩。"
    fi
    echo "[budget] 這一項沒有任何自動補救：-ngl 是顯式指定的，-fit 不會介入（見上面的 [防護 2c]）。"
    echo "[budget] 已量測的存活設定（2026-09-16 13:0x-13:3x，同 profile、同 load_mode=none、同 binary）："
    echo "[budget]   ub=4096 pool 8GiB 存活 5/5；ub=5120 存活 3/3；ub=5632 存活 4/4（profile 現在的預設）；"
    echo "[budget]   ub=6144 pool 8GiB 存活 0/5（12:05/13:15/13:17/13:32/13:33 五次都以 status 5 OOM 死在 req2）；"
    echo "[budget]   ub=6144 pool 6GiB（CGC_SERVER_EXPERT_CACHE_BYTES=6442450944）存活 3/3。"
    echo "[budget]   機制：OOM 的邊界不是 ub 一個旋鈕，而是 (pool + compute buffer) 的和 —— 詳見 §11.12。"
    echo "[budget]   mmap 在任何量過的寬度（4096/5120/6144）都不存活 0/3：缺陷已修（載入不再 SIGBUS，"
    echo "[budget]   見 [防護 2e]），但它把 Metal 的工作集撐大（工作集警告只在 mmap 出現），所以它不是 OOM 的解。"
    echo "[budget]   吞吐不在此列，因為它不可重現：同一個 A11 指紋的兩臂可差到 2 倍（記憶體/swap 的殘留狀態），"
    echo "[budget]   存活是二元的、可重現；t/s 要引用就得在安靜的機器上做交錯 A/B。"
    if [ "$CGC_SERVER_STRICT_BUDGET" = "1" ]; then
        echo "error: CGC_SERVER_STRICT_BUDGET=1 且靜態需求 ${_b_dem} MiB 超過實體 ${_b_phys} MiB -> 拒跑。" >&2
        exit 1
    fi
else
    echo "[budget] OK：靜態需求 ${_b_dem} MiB <= 實體 ${_b_phys} MiB。"
fi

# [防護 2e / 2026-09-16] load_mode=mmap × expert cache：一個**已經修好的缺陷**，而閘門要能分辨
# 「含修復的 binary」與「不含修復的 binary」，不能一句話禁掉整條路。
#
# ── 原缺陷（2026-09-16 12:02 量測，profile=prefill250 + CGC_SERVER_LOAD_MODE=mmap）
# 行程在**載入階段**就死，health 從未轉 ok，crash report
# ~/Library/Logs/DiagnosticReports/llama-server-2026-09-16-120239.ips：
#
#   EXC_BAD_ACCESS / SIGBUS / KERN_PROTECTION_FAILURE at 0x00000003332f5ae0
#     0 __bzero + 32
#     1 fill_pool_direct(llama_expert_cache*, unsigned int, int, unsigned int) + 172
#     2 llama_expert_cache_ensure_slot(...) + 652
#     3 llama_expert_cache_prewarm_hot(llama_expert_cache*) + 880
#     4 llama_context::process_ubatch(...)
#     5 llama_context::decode(...)      <- 由 load_model 的 warmup 走進來
#
# 機制：expert cache 的 L4 pool 是「regions adopted from expert tensors」（log 原句），也就是說
# 它直接寫進模型張量的儲存。--load-mode none（=--no-mmap）時那是可寫的堆積；--load-mode mmap
# 時那是**唯讀的 file-backed 映射**。於是 src/llama-expert-cache.cpp:693-710 的 fill_pool_direct：
# pread 寫不進去 -> fill_segments_concurrent 回報失敗 -> 印「short read ... zeroing slot」
# （log 第 17 行正是這句）-> 走 :704-706 的 memset 把 slot 清零 -> SIGBUS。
# 也就是說那條「短讀取就清零」的復原路徑預設了 dst 可寫，這個預設只在 none 模式下成立。
#
# ── 修復（同日）：不動 expert cache，改動張量落點
# 讓這些張量落到一個**不是 device default** 的 buft（ggml-metal.cpp:
# ggml_backend_metal_buffer_type_pool，名字含 "_Pool"）。理由是 llama-model.cpp:1595 的 mmap
# fast path 只在 `is_default_buft`（buft == ggml_backend_dev_buffer_type(dev)）時成立：換一個
# 身分就自動落到「真的配置一個 buffer」的分支，於是 pool 的 ctx 拿到可寫的 MTLBuffer，
# 而其餘權重照樣走 zero-copy file mapping（這正是修復的價值：模型頁仍然可回收）。
# 只在 use_mmap 時選它（llama-model-loader.cpp: select_pool_buft），所以 load_mode=none 路徑
# 的 buft 選擇與以前逐位元相同。
#
# 判準（CONVENTIONS B20：相容性缺陷要用閘門擋住，而閘門要能分辨修好與沒修好）用的是上面
# 那兩行算出來的 CGC_MMAP_POOL_FIX（在 [防護 2d] 之前就判定了）。
# 要硬闖（例如就是要重現原缺陷）：CGC_SERVER_ALLOW_MMAP_EXPERT_CACHE=1。
case "$SERVER_LOAD_MODE" in
    mmap|mmap+mlock)
        if [ "${CGC_SERVER_EXPERT_CACHE_OFF:-0}" != "1" ]; then
            if [ "$CGC_MMAP_POOL_FIX" = "1" ]; then
                echo "[mmap]  load_mode=$SERVER_LOAD_MODE + expert cache：pool tensor 走自己的 '_Pool' buft（真的 MTLBuffer），不是唯讀映射；其餘權重仍是 file-backed。" >&2
            elif [ "${CGC_SERVER_ALLOW_MMAP_EXPERT_CACHE:-0}" = "1" ]; then
                echo "warning: 這個 binary 不含 pool/mmap 修復，load_mode=$SERVER_LOAD_MODE + expert cache 是已知會 SIGBUS 的組合（llama-expert-cache.cpp:704，__bzero 寫唯讀映射）；已由 CGC_SERVER_ALLOW_MMAP_EXPERT_CACHE=1 放行。" >&2
            else
                echo "error: 這個 binary 不含 pool/mmap 修復（libggml-metal 裡找不到 '_Pool' buft），load_mode=$SERVER_LOAD_MODE 與 expert cache 併用會在載入階段 SIGBUS -> 拒跑。" >&2
                echo "  判定用的檔案：${CGC_METAL_LIB:-<找不到 libggml-metal*.dylib>}" >&2
                echo "  證據：llama-server-2026-09-16-120239.ips，__bzero <- fill_pool_direct <- prewarm_hot（見腳本內註解）。" >&2
                echo "  選項：改回 CGC_SERVER_LOAD_MODE=none，或加 CGC_SERVER_EXPERT_CACHE_OFF=1（cache-free），" >&2
                echo "        或重建含修復的樹（scripts/build_fork_llama.sh），或 CGC_SERVER_ALLOW_MMAP_EXPERT_CACHE=1 硬闖。" >&2
                exit 1
            fi
        fi
        ;;
esac

mkdir -p "$LOG_DIR"
ln -sf "$LOG" "$LOG_DIR/llama_server_latest.log"

# [probe] 只印前檢階段的判定就退出，不載入模型。
# 為什麼需要它：到這裡為止的四道防護（1/2/2b/2d/2e）全部在載入之前跑完，而「放行」的那幾面
# （修復存在時的 mmap、EXPERT_CACHE_OFF 的豁免、ALLOW 的硬闖）原本只能靠真的載入 13 GB 模型
# 才看得到 rc。要量「閘門有沒有在該擋的時候擋、該放的時候放」，得能在零載入成本下逐面驗。
# 注意：[fit]/[kv] 在更後面（argv 組裝段），這個探針不覆蓋它們。
if [ "${CGC_SERVER_PROBE_ONLY:-0}" = "1" ]; then
    echo "[probe] pre-flight OK (load_mode=$SERVER_LOAD_MODE, mmap_pool_fix=$CGC_MMAP_POOL_FIX) -> exit before launch (CGC_SERVER_PROBE_ONLY=1)"
    exit 0
fi

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
# [CGC 2026-09-15] print glu_fused_down with its actual gate. The fused-glu entry point requires
# CGC_MMV_FUSE=1; without it the flag is inert (see the pass-through below), and a bare
# "glu_fused_down=1" invites the reader to believe a fusion is active when it is not.
echo "[perf]  n_cb=$SERVER_N_CB glu_fused_down=$SERVER_GLU_FUSED_DOWN(mm_fuse=${CGC_MMV_FUSE:-off}) watchdog=$SERVER_WATCHDOG oa_async=$SERVER_OA_ASYNC load_mode=$SERVER_LOAD_MODE"
# [CGC 2026-09-19] Echo the schedule-shaping diagnostics. An UNLISTED CGC_* is dropped silently, so
# without this line "the switch did nothing" and "the switch was never set" leave the same artifact
# -- which is the single most expensive confusion in this repo's history (see the CGC_OA_ASYNC note
# near the top, and the CGC_MMV_FUSE trap). These four are all in the allowlist below and were
# visible in NO log: CGC_SUBMIT_AHEAD is the zero-code ceiling probe for the whole overlap family,
# CGC_SLOT_TABLE_GPU is the S1 arm, CGC_CANON_ORDER is M1 item 4, CGC_POOL_SPLIT_DBG is the
# pool-vs-wide path discriminator. Do NOT add a knob here that the allowlist drops -- the banner
# would then lie about what was requested, which is worse than being silent.
echo "[perf]  diag: submit_ahead=${CGC_SUBMIT_AHEAD:-off} slot_table_gpu=${CGC_SLOT_TABLE_GPU:-off} canon_order=${CGC_CANON_ORDER:-off} pool_split_dbg=${CGC_POOL_SPLIT_DBG:-off} s1_dbg=${CGC_S1_DBG:-off}"
echo "[perf]  runtime_profile=$SERVER_RUNTIME_PROFILE model_root=$MODEL_ROOT"
# 這一行同時發布 launcher 自己的判定（admits=yes/no），而不是只發布輸入值。
# 理由：`harness.py show` 與 box_probe_compare.py 會把這條線跟 harness 自己的探針並排，
# 而兩個定義曾經對同一台機器給出相反答案（2026-09-20：harness 在 6636 MB 就拒絕，launcher 讀到 84%）。
# 若由 Python 端拿這些數字自行推論，就會多出「第二個 launcher 門檻定義」——而門檻只住在
# cgc_memory_guard_req 裡，不該有第二份。
LAUNCHER_ADMITS=yes
if [ "$PHYS_MEM_GB" -lt "$MEM_REQ_PHYS_GB" ] || [ "$FREE_PCT" -lt "$MEM_REQ_FREE_PCT" ] || [ "$OTHER_LLAMA_SERVERS" -gt "$MEM_REQ_OTHER" ]; then
    LAUNCHER_ADMITS=no
fi
echo "[guard] memory_mode=$SERVER_MEMORY_MODE class=$MEM_CLASS phys=${PHYS_MEM_GB}GB free=${FREE_PCT}% other_llama_servers=$OTHER_LLAMA_SERVERS req_phys=${MEM_REQ_PHYS_GB}GB req_free=${MEM_REQ_FREE_PCT}% req_other=${MEM_REQ_OTHER} admits=$LAUNCHER_ADMITS"
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

# [防護 2c / 2026-09-16] 「-fit 會幫我們挑一組裝得下的參數」是錯的 —— 在這一條路徑上它從來沒跑過，
# 而啟動訊息完全不提。這一段把它的實際狀態印出來，並且提供一個能真的讓它跑起來的開關。
#
#   common/common.h:477 讓 params.fit_params 預設 true（等同 -fit on），但
#   src/llama.cpp/common/fit.cpp:377-379 是：
#       if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
#           throw common_params_fit_exception("n_gpu_layers already set by user to " + ... + ", abort");
#       }
#   而 llama_model_default_params().n_gpu_layers 是 -1（src/llama-model.cpp:2479），這個腳本卻一律
#   顯式帶 -ngl（見下面的 SERVER_ARGS）。所以 fit 每次都 abort 成一行 warning，而它只出現在
#   server log 的第 9 行：
#       W common_fit_params: failed to fit params to free device memory:
#                            n_gpu_layers already set by user to 99, abort
#   ⇒ 真正在做記憶體判斷的是 [防護 1]（清殘留）＋ [防護 2/2b]（free% 水位）＋ [防護 2d]（啟動預算）。
#     不要把「裝不裝得下」記在 fit 頭上；它沒有參與過。
#
#   CGC_SERVER_FIT=1 會把 -ngl 從 argv 拿掉，讓 fit 真的能決定 n_gpu_layers。但 fit 只會「縮」到
#   裝得下（它不會維持 ngl=99 的形狀），那會直接改變記憶體與效能剖面，也會改掉與既有 oracle 抓
#   下來的 config stamp 的可比性 —— 所以預設關閉。注意 argv 順序：D5 閘門的 config stamp 是
#   「位置比對」ARG[i]，所以 FIT=0 時 -ngl 必須留在原本的索引上（下面用兩段 append 而不是重新排序）。
CGC_SERVER_FIT="${CGC_SERVER_FIT:-0}"

SERVER_ARGS=(
    -m "$MODEL"
)
if [ "$CGC_SERVER_FIT" = "1" ]; then
    echo "[fit]   -ngl 不帶 -> common_fit_params 會自己決定 n_gpu_layers（會改變記憶體／效能剖面）" >&2
else
    echo "[fit]   -fit 是 no-op：-ngl=$SERVER_NGL 是本腳本顯式指定的，fit 會 abort（fit.cpp:377）" >&2
    SERVER_ARGS+=(-ngl "$SERVER_NGL")
fi
SERVER_ARGS+=(
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
# [CGC 2026-09-15] Expert-cache launch switch -- and the only way to get a genuine
# cache-FREE oracle.
#
# Why this matters: -expert-cache used to be written unconditionally in the array above, so
# EVERY "reference oracle" ever dumped was itself an expert-cache-ON run. That is why
# knifeedge_matrix.py scored 19/19 for M1/M2/M3: it was comparing a cache-ON run against
# another cache-ON run. 19/19 proves the engine is deterministic, not that the hook is
# correct. Without a cache-free reference there is no ground truth to be identical to.
#
# `-expert-cache 0` is NOT a substitute. Budget 0 still walks enough of the L4 load-time path
# (skip_load stays on, tensors are placed on the CPU buffer, ne02 shrinks) for the tensor
# geometry to disagree with what the hook expects -- measured: every request HTTP 500. So the
# flag has to be OMITTED, not zeroed. Omitting it is exactly the baseline the NOGATHER A/B
# compared against (argmax 198 / logit 27.18, matching the no-cache arm).
if [ "${CGC_SERVER_EXPERT_CACHE_OFF:-0}" = "1" ]; then
    BUDGET=0
    echo "[cache]  EXPERT CACHE OFF — cache-free oracle launch (no -expert-cache flag)" >&2
else
    SERVER_ARGS+=(-expert-cache "$BUDGET")
fi
# [CGC 2026-09-08 KV cache Q8 quantization] -- numbers re-derived 2026-09-16 (the old
# "saves ~0.56GB (1.25GB -> 0.69GB)" was wrong for this model; see below).
# For Qwen3.6-35B-A3B at ctx=8192 the KV cache holds K/V for 10 of the 40 layers only:
# the main context is a hybrid memory whose attn filter is `il < n_layer() && !is_recr(il)`
# (llama-model.cpp:2292-2295 -> llama-memory-hybrid.cpp:48-50), and full_attention_interval=4
# makes exactly i=3,7,...,39 non-recurrent. Per layer per token: 2 kv-heads x (key 256 +
# value 256) x 2B = 2 KiB.  =>  fp16 160 MiB  ->  q8_0 ~85 MiB  =>  saving ~75 MiB.
# NOTE the old text counted all 41 blocks as dense-attention layers (41 x 1024 x 8192 x 2B
# = 0.69 GB) and paired it with an f32 figure; that is a dense model's shape, not this one's.
# Arithmetic only -- this fork prints no KV-size line at -lv 3, so it is not measured.
# Quality impact is negligible (<1%) for typical workloads. Set CGC_SERVER_KV_Q8=0 to
# disable (fall back to FP16).
SERVER_KV_Q8="${CGC_SERVER_KV_Q8:-1}"
if [ "$SERVER_KV_Q8" = "1" ]; then
    SERVER_ARGS+=(--cache-type-k q8_0 --cache-type-v q8_0)
    echo "[kv]     cache-type-k=q8_0 cache-type-v=q8_0 (saves ~75 MiB at ctx=8192, not 0.56GB -- see comment)"
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
    # [CGC 2026-09-16] VALUE semantics, not presence. Until today the three readers of this knob
    # (llama.cpp, llama-expert-cache.cpp, llama-context.cpp) tested `getenv(...) != nullptr`, under
    # which `0` is a NON-NULL pointer -- so this line ENABLED skip0 and the 2026-09-09 quality fix
    # below was inert for six days, while every cap_*.json and llama-bench env block recorded `"0"`
    # and read as if it were off. The readers now parse the value (unset/empty/leading `0` = off),
    # so `0` here finally means what the comment says. Do not "fix" a mode by writing `=0` at a
    # call site while the reader tests presence -- that is this bug.
    #
    # What `0` means: blk.0 STAYS IN the pool -- 40 pooled layers, not 39. skip0 is a
    # layout-matching knob, not a quality knob: `-ngl 30` (run_n30cache.sh) keeps blk.0's FFN on
    # the CPU in the base, so skip0=1 is the bit-identical setting THERE; this profile is
    # full-offload L4 (ALLOW_NGL=1, all 40 layers on Metal), where skip0=1 makes layer 0 a
    # full-weight CPU tensor read by a Metal graph -- measured 4/10 on the 15+27 probe on
    # 2026-09-09, vs 90-100% for the sweet spot (uniform caps, layer 0 in the pool).
    #
    # Flipping this line is a NUMERICS change (pooled-layer count 39<->40 changes layer 0's FFN
    # consumer path, the pool's resident bytes and layer 0's pread traffic), so it needs its own
    # M1/M2/M3 reference: a skip0-off dump is not comparable to a skip0-on one even though the
    # resolved env string is identical in both cases.
    #
    # Overridable as CGC_SERVER_SKIP0 (same shape as CGC_SERVER_WORKERS below) so the control arm
    # `CGC_SERVER_SKIP0=1` can be measured against the pre-fix reference without editing this file.
    # A literal here could not be overridden from outside at all: `env "${SERVER_ENV[@]}"` would
    # apply this line over any incoming value, so a hand-set env var would silently lose.
    LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0="${CGC_SERVER_SKIP0:-0}"
    # [CGC 2026-09-15] Was the literal 8, never swept. It is the ONLY concurrency knob on the
    # expert-fill path, and the 2026-09-15 measurements say concurrency -- not bytes -- is the
    # binding term: 57363 preads x 1.73 ms = 99.03 s of summed latency inside a 57.5 s decode
    # wall (MTP off) and 285279 preads = 553.78 s inside a 92.6 s wall (MTP on). The bytes are
    # cheap: 81.5 GiB at even 2 GB/s is 41 s. So the question is how many preads are in flight,
    # and 8 workers is an assumption nobody has tested.
    #
    # Scope note: this only affects jobs submitted through the persistent pool
    # (fill_segments_pool, i.e. llama_expert_cache_ensure_batch). The MTP fast path's cold
    # fixes go through llama_expert_cache_ensure_slot, which spawns its own 3 threads per
    # expert and does NOT use this pool -- so raising it will under-deliver until that path is
    # routed through ensure_batch too. Default stays 8, so no existing profile moves.
    LLAMA_EXPERT_CACHE_WORKERS="${CGC_SERVER_WORKERS:-8}"
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
# [CGC 2026-09-15] Bit-identical arm switches, passed through explicitly.
#
# The launch line runs the child through `env "${SERVER_ENV[@]}"`, an ALLOWLIST -- anything not
# listed is silently dropped, so setting one of these in the shell looks like it worked while
# the arm quietly becomes a rerun of the control. That is the exact trap the ROUTE_DUMP/MASSCOV
# comment below warns about, and it is why NOHOOK/NOGATHER were unreachable from run_server.sh
# even though llama.cpp:381 has honoured them since 2026-09-15.
#
#   NOHOOK   = 4th arm: hook+remap+pool+slab repoint OFF, skip_load left as-is.
#              NOHOOK == cache-free oracle  -> the fault is in remap/pool/slab repoint.
#              NOHOOK != cache-free oracle  -> the fault is BELOW the hook (CPU residency /
#                                              buffer type / loader placement).
#   NOGATHER = 2 variables at once (skip_load AND hook) -- kept for continuity with the
#              2026-09-14 A/B, but NOHOOK is the one that localises.
#   L3_NGL   = activate at ngl>0 without the L4 shrink/adoption path.
# [CGC 2026-09-15] LLAMA_EXPERT_CACHE_STEP_DBG added to the list below. It is not a new knob: it
# has existed in llama-context.cpp (the per-step miss timeline, whose 2026-08-28 reading -- cold
# ~65% in EVERY phase, hence "structural churn, not a cold start a prewarm could fix" -- is quoted
# verbatim in that comment) but was never allowlisted, so it could not be turned on through the
# launcher at all; that verdict must therefore have been taken by some other launch path, and every
# attempt since has been a silent no-op rather than a negative result.
# It matters now because it is the cheapest probe of RESIDENCY CHURN, which is what decides D3's
# premise B (publication off the hot path): a slot table whose entries move every step has to be
# republished every step, and then the segment boundary keeps its reason to exist.
for _v in LLAMA_EXPERT_CACHE_NOHOOK LLAMA_EXPERT_CACHE_NOGATHER LLAMA_EXPERT_CACHE_L3_NGL \
          LLAMA_EXPERT_CACHE_STEP_DBG \
          CGC_S1_OUT_CAP CGC_S1_OUT_LAYERS CGC_S1_TABLE_CHURN CGC_S1_CLAMP_ABORT \
          CGC_LOGITS_ORACLE_TOPN CGC_LOGITS_ORACLE_FIRST_N; do
    if [ -n "${!_v:-}" ]; then
        SERVER_ENV+=("$_v=${!_v}")
    fi
done
unset _v
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
# CGC 2026-09-18: print, per MoE layer per graph, the value of each down-combine gate condition
# and the resulting decision. Read-only (does not change the decision). It exists because the gate
# is an AND of six conditions and only ONE of them was ever quoted -- the expert tensor's type --
# and the type the loader produces is not the one the GGUF-oriented reasoning assumed.
if [ -n "${CGC_DOWN_COMBINE_AUDIT:-}" ]; then
    SERVER_ENV+=(CGC_DOWN_COMBINE_AUDIT="$CGC_DOWN_COMBINE_AUDIT")
fi
# 2026-09-18: admit the MULTI-TOKEN verify shape (n_tokens 2/4) to the fused down-combine. Read in
# BOTH llama-graph.cpp and ggml-metal-ops.cpp -- the graph-side gate and the Metal-side dispatch
# must agree, or the node lands in the graph with nothing to execute it.
if [ -n "${CGC_DC_MULTITOK:-}" ]; then
    SERVER_ENV+=(CGC_DC_MULTITOK="$CGC_DC_MULTITOK")
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
# [CGC M5 prerouter 2026-09-17] PREFETCH-ONLY decode predictor. Roadmap M5 ("only decode, only
# L+1, only the 8, only time"). Three claims the FIRST version of this comment got wrong; they
# were measured/read on 2026-09-17 and they are why the default stays OFF:
#   * `llama_expert_cache_prefetch_slot` is NOT free-slot-only. Its own header comment still says
#     "never evicts for a prediction", but the body (llama-expert-cache.cpp:1341-1381) falls back
#     to evicting the layer's LRU / lowest-utility slot when no free slot exists. A prediction can
#     therefore DISPLACE a resident expert -- which is precisely what "prefetch only, never decide
#     routing" is supposed to forbid. Not reachable in the run below (queued=0), but reachable in
#     principle, and it is the first thing to fix if M5 is ever switched on.
#   * `freq` does NOT require LLAMA_EXPERT_CACHE_ROUTE_RECORD. record_routes -- the only writer of
#     freq -- is called unconditionally at llama-context.cpp:5610; the ROUTE_RECORD flag controls
#     the route *dump*, not the accumulation. (Verified: a CGC_PREROUTER-only run reported
#     nodata=0, i.e. freq was populated.)
#   * it predicts from `freq`, and `prewarm_hot` already fills each layer's pool from the SAME
#     freq top-K before the first decode step (llama-context.cpp:1928). Predictor and prewarm read
#     the same ranking, so the prediction is already resident BY CONSTRUCTION. Measured:
#     `CGC-PREROUTER: calls=80 queued=0 scored=40 pred_total=320 hit=41 (precision 12.8%)` --
#     prefetch_slot rejected all 320. Set LLAMA_EXPERT_CACHE_PREFETCH_DBG to see which reject.
#   * the roadmap's own placement measurement gives this a 0.2pp ceiling and notes the head it
#     does have starts at layer 7, i.e. it has no head for the churn-heaviest layers 1/2. The
#     expected outcome is "recorded as an excluded dead end", so the default stays OFF.
# Without these blocks the explicit `env` allowlist on the launch line silently drops them, so
# setting them in the shell would appear to do nothing.
if [ -n "${CGC_PREROUTER:-}" ]; then
    SERVER_ENV+=(CGC_PREROUTER="$CGC_PREROUTER")
fi
# [CGC 2026-09-19 layer-ahead decode prefetch] CGC_LAYER_AHEAD_PREFETCH=1 makes each layer's hook
# queue the NEXT layer's predicted union (previous token's ids) through prefetch_slot, so the pread
# overlaps segment il+1's GPU window instead of blocking inside layer il+1's own hook. It is handled
# in llama-context.cpp expert_cache_on_topk; without this block the launch line's `env` allowlist
# drops it and "the flag did nothing" would have no visible cause (the trap this section documents).
if [ -n "${CGC_LAYER_AHEAD_PREFETCH:-}" ]; then
    SERVER_ENV+=(CGC_LAYER_AHEAD_PREFETCH="$CGC_LAYER_AHEAD_PREFETCH")
fi
# [CGC 2026-09-19 hook split] CGC_HOOK_SPLIT=1 makes the per-layer top-k hook report its own cost
# split into pre / ensure (the batched union fill) / drain / tail (publish + remap leaf). `cb` in the
# decode profile is the quantity every batching decision turns on and nothing said what is in it: the
# existing fill counters are blind to this hook's demand fills (fill_batch_usec is only accumulated by
# the blob path, fill_wait_us only counts waits on prefetch-queued slots). Same allowlist trap.
if [ -n "${CGC_HOOK_SPLIT:-}" ]; then
    SERVER_ENV+=(CGC_HOOK_SPLIT="$CGC_HOOK_SPLIT")
fi
# [CGC 2026-09-19 slab→pool handoff] CGC_SLAB_HANDOFF=<cap> experts/layer: at the first decode step
# after a slab prefill, publish the prefill's hot set into the pool (evicting, capped, synchronous),
# because the slab path repoints the FFN weights at a per-layer slab and never writes the pool. The
# one-shot prewarm_hot that exists for this is a process-level flag, so in a server it is consumed by
# the first request and later prefills get no publish at all. Same allowlist trap as the two above.
if [ -n "${CGC_SLAB_HANDOFF:-}" ]; then
    SERVER_ENV+=(CGC_SLAB_HANDOFF="$CGC_SLAB_HANDOFF")
fi
# Attribution for the line above: prefetch_slot's three exit classes (guard-reject vs already
# resident vs no-free-slot) print `PFDBG guard-reject ...` / `PFDBG drop: resident ...` per call.
# Not in the allowlist before 2026-09-17, so "queued=0" had no readable cause.
if [ -n "${LLAMA_EXPERT_CACHE_PREFETCH_DBG:-}" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_PREFETCH_DBG="$LLAMA_EXPERT_CACHE_PREFETCH_DBG")
fi
# [CGC 2026-09-22 verify-marginal attribution] LLAMA_EXPERT_CACHE_BATCH_DBG=1 prints one
# `BATCHDBG layer=<l> misses=<n> slots: e<expert>->s<slot> ...` line per ensure_batch call that had
# >=1 miss (llama-expert-cache.cpp:1174). Its consumer is the per-layer series: the question
# "is the marginal verify token paid in gather or in compute?" is a regression of a layer's
# per-round cost against its per-round MISS COUNT, and without this producer there is no miss axis
# per layer -- only the run-total split. Never forwarded through this launcher before (the engine
# has read it since 2026-09-13), so arming it looked exactly like an instrument with no effect.
if [ -n "${LLAMA_EXPERT_CACHE_BATCH_DBG:-}" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_BATCH_DBG="$LLAMA_EXPERT_CACHE_BATCH_DBG")
fi
# [CGC 2026-09-19 thrash attribution] LLAMA_EXPERT_CACHE_MISS_DUMP=<path> writes one
# "<layer> <expert>" line per DEMAND-ORDERED miss (llama-expert-cache.cpp:1061, flushed per line so a
# kill -9 still leaves a usable file). It exists to answer a question the cumulative compulsory/
# capacity counters cannot: a capacity miss is a reload, but is it a reload after 2 steps (real
# thrash, a policy can fix it) or after 200 (a working set larger than the pool, only capacity
# fixes it)? The counter split alone cannot tell those apart, and the answer decides whether the
# lever is the eviction policy or the pool geometry. Never armed through this launcher before, so
# the dump had no producer -- the same allowlist trap as the blocks above (the engine has read it
# since 2026-09-13). MISS_ATTR_LAYERS is its companion: per-layer distinct-vs-slots rows.
if [ -n "${LLAMA_EXPERT_CACHE_MISS_DUMP:-}" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_MISS_DUMP="$LLAMA_EXPERT_CACHE_MISS_DUMP")
fi
if [ -n "${LLAMA_EXPERT_CACHE_MISS_ATTR_LAYERS:-}" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_MISS_ATTR_LAYERS="$LLAMA_EXPERT_CACHE_MISS_ATTR_LAYERS")
fi
if [ -n "${CGC_PREROUTER_TOP_K:-}" ]; then
    SERVER_ENV+=(CGC_PREROUTER_TOP_K="$CGC_PREROUTER_TOP_K")
fi
# [CGC MTP instrument 2026-09-17] LLAMA_BENCH_SPEC=1 makes `llama-bench` generate through the
# speculative path (tools/llama-bench/llama-bench.cpp: test_gen_spec) instead of its bare
# `llama_decode` loop -- the instrument that makes M4 measurable at all, since the whole
# tools/llama-bench/ directory has ZERO hits for speculat/spec_type/draft/mtp.
#
# WHY A NON-CGC VAR IS IN THIS ALLOWLIST: the server does not read it and never will (the name is
# namespaced to the tool). It is here because `llama_bench_matrix.py` gets its env from
# `run_server.sh CGC_DUMP_ENV=1`, which prints exactly the SERVER_ENV array -- so a variable not
# listed here is dropped on the floor and the run looks like "the instrument has no effect".
# LLAMA_BENCH_SPEC_DRAFT_N_MAX is the same story; it is the draft-depth knob (default 3).
if [ -n "${LLAMA_BENCH_SPEC:-}" ]; then
    SERVER_ENV+=(LLAMA_BENCH_SPEC="$LLAMA_BENCH_SPEC")
fi
if [ -n "${LLAMA_BENCH_SPEC_DRAFT_N_MAX:-}" ]; then
    SERVER_ENV+=(LLAMA_BENCH_SPEC_DRAFT_N_MAX="$LLAMA_BENCH_SPEC_DRAFT_N_MAX")
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
# [CGC 2026-09-15] CGC_SYNCFILL_SERIAL=1 restores the pre-fix per-expert serial
# llama_expert_cache_ensure_slot loop in the MTP fast path (llama-context.cpp). The default is
# now the batched ensure_batch fill. Without this pass-through the knob would be dropped by the
# launch line's explicit `env` allowlist, so the A/B that decides whether the batching changed
# the numerics (it changes which experts are cold at verify read, which changes the ZERO-slot
# reads) would silently compare the new path against itself.
if [ -n "${CGC_SYNCFILL_SERIAL:-}" ]; then
    SERVER_ENV+=(CGC_SYNCFILL_SERIAL="$CGC_SYNCFILL_SERIAL")
fi
# CGC pool-path batch cap (C++ default 8): the pool path cannot exceed cap x top_k usable slots,
# so raising it interacts with the pool size and must be visible to the launcher.
if [ -n "${CGC_POOL_MAX_TOKENS:-}" ]; then
    SERVER_ENV+=(CGC_POOL_MAX_TOKENS="$CGC_POOL_MAX_TOKENS")
fi
# CGC M1 work item 1 (C++ default OFF): keep the expert tensors at FULL WIDTH and give the pool its
# own Metal allocation, which is what decouples the batch width from the pool size. Needs a load mode
# that can map the model (CGC_SERVER_LOAD_MODE=mmap), otherwise the full-width tensors are allocated
# and read resident -- that difference IS the cost being measured.
if [ -n "${CGC_POOL_SPLIT:-}" ]; then
    SERVER_ENV+=(CGC_POOL_SPLIT="$CGC_POOL_SPLIT")
fi
# Diagnostic for the above: prints every pool-region adoption and, for the first layers, what the
# graph repointed each FFN tensor to (il / kind / base / pool buffer). Without it in this list a
# "the diagnostics printed nothing" would read as "the mechanism did not run".
if [ -n "${CGC_POOL_SPLIT_DBG:-}" ]; then
    SERVER_ENV+=(CGC_POOL_SPLIT_DBG="$CGC_POOL_SPLIT_DBG")
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
# [CGC 2026-09-18 node-level GPU time] CGC_GPU_NODES=1 prints the per-NODE-KIND GPU table
# (CGC-GPUNODE:) on top of the per-layer one: each command buffer's GPUStartTime/GPUEndTime is
# attributed across the contiguous node range that buffer encoded, so the resolution goes one level
# below a layer without sampling or an MTLCounterSampleBuffer. Requires CGC_DECODE_PROFILE=1 (it
# supplies the step cadence and the `layer gpu_sum` the table self-checks against).
if [ -n "${CGC_GPU_NODES:-}" ]; then
    SERVER_ENV+=(CGC_GPU_NODES="$CGC_GPU_NODES")
fi
# [CGC 2026-09-18 node-level GPU time TRACE] CGC_GPU_NODES_TRACE=1 adds the RAW node names of the
# three hottest command buffers of each sampled step, plus a range-size histogram -- the pair that
# separates "the hot nodes are not named ffn_moe_*" from "they share a buffer with small nodes so the
# node-count split diluted them". Needs CGC_GPU_NODES=1.
if [ -n "${CGC_GPU_NODES_TRACE:-}" ]; then
    SERVER_ENV+=(CGC_GPU_NODES_TRACE="$CGC_GPU_NODES_TRACE")
fi
# [CGC 2026-09-18 node-level GPU time MATRIX] CGC_GPU_NODES_MATRIX=1 dumps one CGC-NSM line per
# command buffer (duration + how many nodes of each kind it encoded) so the per-kind cost can be
# recovered offline by least squares instead of guessed by node count. Needs CGC_GPU_NODES=1.
if [ -n "${CGC_GPU_NODES_MATRIX:-}" ]; then
    SERVER_ENV+=(CGC_GPU_NODES_MATRIX="$CGC_GPU_NODES_MATRIX")
fi
# [CGC 2026-09-18 per-NODE command buffers] CGC_CB_N_MAIN overrides the Metal encoder's
# `n_main = MAX(64, 0.1*n_nodes)` floor, i.e. how many of a segment's FIRST nodes go into the
# main thread's single command buffer. It exists to be paired with CGC_SERVER_N_CB (already a knob
# of this script, read at the top as `SERVER_N_CB`), so that the n_cb+1 buffers become one-node
# slices and CGC_GPU_NODES can attribute GPU time per NODE instead of by node count.
#
# MEASUREMENT ONLY, and the two knobs are NOT equally available. Lowering it alone is free (it
# changes no count, the buffers stay at n_cb+1). Pairing it with a LARGE CGC_SERVER_N_CB is not:
# measured 2026-09-18 01:48, n_cb=127 blocks Metal's command-buffer creation
# (`commandBufferWithUnretainedReferences` -> `_dispatch_semaphore_wait_slow`) and the server never
# reaches ready -- the granularity is capped by Metal's in-flight limit, not by the timestamp API.
# The usable pairing is n_cb <= ~16. Unset => upstream behaviour, byte for byte.
if [ -n "${CGC_CB_N_MAIN:-}" ]; then
    SERVER_ENV+=(CGC_CB_N_MAIN="$CGC_CB_N_MAIN")
fi
# [CGC 2026-09-18] CGC_GRPH_DBG=1 dumps the first 6 graph_computes' full node list
# (`CGC-GRPH[i] name=... op=... ne=[..]`). It was NOT in this allowlist, so passing it through this
# launcher did nothing -- the same silent-drop trap as CGC_VERIFY_OP_TIMING below. Its use here is
# NAME RESOLUTION, not shapes: the node-kind table buckets by a fixed prefix vocabulary
# (ggml-backend.cpp:2100-2109) and whatever matches no entry lands in "(other)". Measured, that
# bucket is the single largest one in a decode step (cntw 20.2% / ub 62.8% of segment busy), so
# "what is in (other)" cannot be answered without the raw names. The first 6 graphs are warmup
# graphs: their token counts are wrong for anything FLOPs-based, but their NAME+OP sets are the
# same trunk graph.
if [ -n "${CGC_GRPH_DBG:-}" ]; then
    SERVER_ENV+=(CGC_GRPH_DBG="$CGC_GRPH_DBG")
fi
# [CGC 2026-09-18 op-keyed attribution] CGC_GPU_OPS=1 (needs CGC_GPU_NODES=1) adds a second table
# keyed by the ggml OP instead of by node-name prefix. It exists because the name-keyed table cannot
# answer the question that decides whether merging the shape chain is worth building: does a VIEW cost
# GPU time? A name bucket mixes ops, and node COUNT is not GPU TIME. The op table prints, per op,
# `uni` = the EXACT us/node measured on the command buffers that held ONLY that op -- the positive
# control the earlier least-squares attempt did not have.
if [ -n "${CGC_GPU_OPS:-}" ]; then
    SERVER_ENV+=(CGC_GPU_OPS="$CGC_GPU_OPS")
fi
# [CGC 2026-09-18] CGC_VERIFY_OP_TIMING=1 prints the per-OP-TYPE breakdown (gate/up/down) instead of
# a per-command-buffer estimate. It was NOT in this allowlist until now, so every attempt to pass it
# through this launcher was silently dropped -- i.e. it looked exactly like "the instrument has no
# effect" (the trap this whole allowlist section exists to document). Its target predicate is
# `node->src[2]->ne[1] > 1`, so it only fires for T>1, i.e. the MTP verify batch -- which is the
# production shape (MTP on), though NOT the T=1 shape M3's leave condition is written against.
if [ -n "${CGC_VERIFY_OP_TIMING:-}" ]; then
    SERVER_ENV+=(CGC_VERIFY_OP_TIMING="$CGC_VERIFY_OP_TIMING")
fi
# [CGC 2026-09-15 GPU-side timing] CGC_GPU_TIMING=1 makes the Metal completion handlers record
# each command buffer's own GPUStartTime/GPUEndTime, and the segmented dispatcher prints
# CGC-GPUTIME: per-step wait / gpu_busy_sum / gpu_union / gap. It answers the one question
# CGC_DECODE_PROFILE cannot: the 91% `wait` is a CPU-side spin, so it cannot distinguish real
# GPU execution (lever: batch the per-expert GEMVs) from launch + completion-report latency
# (lever: restore inter-segment overlap). Same dispatcher as CGC_DECODE_PROFILE, and the same
# allowlist rule: an unlisted CGC_* is dropped silently, which looks exactly like "no effect".
if [ -n "${CGC_GPU_TIMING:-}" ]; then
    SERVER_ENV+=(CGC_GPU_TIMING="$CGC_GPU_TIMING")
fi
# [CGC 2026-09-15 remap-overlap ceiling probe] CGC_SUBMIT_AHEAD=1 restores the pre-fix submit
# order (submit segment i+1 BEFORE the top-k hook of segment i writes its remap leaf). That order
# is WRONG -- it is exactly the raciness the current order was introduced to fix, and the GPU can
# read a stale remap -> garbage output. It exists because it is the only zero-code measurement of
# the CEILING of the whole "restore inter-segment overlap" family: it deletes the CPU window
# between the poll and the commit in one switch. If decoding does not speed up here, no
# double-buffered remap design can help and the segmented dispatch is not the bottleneck.
# Must be in this allowlist: ggml-backend.cpp reads it, but an unlisted CGC_* is dropped, so the
# probe would silently do nothing (the same trap as CGC_MMV_FUSE -- see the note near the top).
if [ -n "${CGC_SUBMIT_AHEAD:-}" ]; then
    SERVER_ENV+=(CGC_SUBMIT_AHEAD="$CGC_SUBMIT_AHEAD")
fi
# [CGC 2026-09-15 S1 slot-table] CGC_SLOT_TABLE_GPU=1 replaces the host-written remap leaf with a
# GPU-side gather: the eval hook publishes the per-layer expert->slot table (I32 [1, n_expert]) and
# the graph computes `slots = get_rows(table, selected_experts)`, which is byte-for-byte what the
# hook used to write.
# SCOPE (corrected 2026-09-18): NOT single-token only. It is exactly the host leaf's condition --
# every step that would have built the leaf, i.e. single-token decode AND the small multi-token
# pool steps (MTP verify / gather). See the authoritative comment at llama-graph.cpp:2138, which
# also explains why the earlier n_tokens == 1 narrowing was wrong (build_moe_ffn's n_tokens is not
# ubatch.n_tokens, so a decode step never built a table at all).
# Same allowlist rule as CGC_SUBMIT_AHEAD and CGC_MMV_FUSE: an unlisted CGC_* is dropped silently,
# which is indistinguishable from "the change had no effect".
if [ -n "${CGC_SLOT_TABLE_GPU:-}" ]; then
    SERVER_ENV+=(CGC_SLOT_TABLE_GPU="$CGC_SLOT_TABLE_GPU")
fi
# [CGC 2026-09-18] CGC_MM_DBG=1 makes ggml-metal-ops.cpp print which mul_mat family each MUL_MAT
# takes (`MMDBG <family> ne11=.. ne00=.. t0=.. t1=..`). Needed because that family is chosen by an
# inline type list (ggml-metal-ops.cpp:2479) plus `ne11 > ne11_mm_min = 8`: a type missing from the
# list is indistinguishable from "the kernel is fine" without this trace. Verified ABSENT from the
# allowlist before today, so setting it was a silent no-op (`CGC_DUMP_ENV=1` printed nothing).
if [ -n "${CGC_MM_DBG:-}" ]; then
    SERVER_ENV+=(CGC_MM_DBG="$CGC_MM_DBG")
fi
# [CGC 2026-09-18] The three S1 arm controls (CGC_S1_KEEP_LEAF / CGC_S1_BUILD_LEAF / CGC_S1_MIN_IL)
# are allowlisted further down, next to the rest of the S1 block -- they were ALREADY there, so a
# 2026-09-18 edit of mine that added a second copy of the same three `if` blocks here was pure
# duplication and has been removed. Read the definitions there; verified transmitted with
# CGC_DUMP_ENV=1 (`ENV CGC_S1_KEEP_LEAF=1` on the p25-s1-keepleaf arm).
if [ -n "${CGC_S1_DBG:-}" ]; then
    SERVER_ENV+=(CGC_S1_DBG="$CGC_S1_DBG")
fi
# [CGC 2026-09-17 r38] CGC_IDS_LINEAR_READ=1 restores the PRE-FIX top-k snapshot in
# `expert_cache_on_topk`: a linear read of `ggml_argsort_top_k`'s strided view, which fed tokens
# >= 1 of every T >= 2 step another token's expert ids (see the point of use in
# llama-context.cpp). It exists so the fix can be A/B'd on ONE binary -- same build, same pool,
# same everything -- because the only other baseline is a 3-day-old binary in a different
# dispatch regime, and comparing across that would not be a before/after. Never for quality or
# throughput numbers; the knob reproduces a defect on purpose.
if [ -n "${CGC_IDS_LINEAR_READ:-}" ]; then
    SERVER_ENV+=(CGC_IDS_LINEAR_READ="$CGC_IDS_LINEAR_READ")
fi
# [CGC M1 work item 4 · canonical gather order] CGC_CANON_ORDER=1 orders each token's k expert
# positions by expert id before the FFN aggregation, so the fp32 add chain (which is not
# associative -- CGC_ADD_ORDER=rev moved the anchor md5) stops being a function of the pool layout.
# =2 is the IDENTITY control: the same graph nodes with perm[p] = p, i.e. it isolates "the reordering
# changed the numbers" from "the extra nodes changed the numbers". Both modes change model output,
# so neither may be quoted as quality or D5 evidence for a reference dumped in another mode.
# Same allowlist rule as the rest: an unlisted CGC_* is dropped silently, which is indistinguishable
# from "the change had no effect" -- exactly the failure this knob's first run would be misread as.
if [ -n "${CGC_CANON_ORDER:-}" ]; then
    SERVER_ENV+=(CGC_CANON_ORDER="$CGC_CANON_ORDER")
fi
# [CGC M1 work item 2 · phase split] CGC_PREFILL_THRESHOLD is the width at which a step takes the
# PREFILL graph (whole-layer slab, 256 experts, raw ids) even though the decode graph could route it.
# Default 512 (the roadmap's starting point); only has an effect when the pool is large enough to
# route wide steps. Same allowlist rule as the rest: an unlisted CGC_* is dropped silently, which is
# indistinguishable from "the threshold had no effect".
if [ -n "${CGC_PREFILL_THRESHOLD:-}" ]; then
    SERVER_ENV+=(CGC_PREFILL_THRESHOLD="$CGC_PREFILL_THRESHOLD")
fi
# [CGC 2026-09-15 S1 kernel-side ids capture] Opt-in snapshot of the ids a mul_mat_id kernel
# actually consumed, written by a tiny post-consumer kernel into a buffer the graph allocator does
# not own. Must be listed here for the same reason as CGC_SLOT_TABLE_GPU above: the launch line's
# allowlist drops anything else, which is indistinguishable from "the instrument did nothing".
if [ -n "${CGC_IDS_CAPTURE:-}" ]; then
    SERVER_ENV+=(CGC_IDS_CAPTURE="$CGC_IDS_CAPTURE")
fi
# [CGC 2026-09-16 §9.18.6] The same instrument pointed at a node's OUTPUT instead of its ids
# operand: CGC_TENSOR_CAPTURE=<a comma-separated list of EXACT node names, or `*`> snapshots the
# tensors those nodes produced, into a second destination with a wider stride. It is what turns
# §9.18.4's elimination argument into a measurement. Same allowlist trap as above -- and note the
# value is a node NAME, so a silent drop here would look exactly like "the node never ran".
if [ -n "${CGC_TENSOR_CAPTURE:-}" ]; then
    SERVER_ENV+=(CGC_TENSOR_CAPTURE="$CGC_TENSOR_CAPTURE")
fi
if [ -n "${CGC_TENSOR_CAPTURE_WORDS:-}" ]; then
    SERVER_ENV+=(CGC_TENSOR_CAPTURE_WORDS="$CGC_TENSOR_CAPTURE_WORDS")
fi
# [CGC 2026-09-16 §9.18.6 r5] WHICH END of the tensor the window reads: unset/0 = the head (element
# 0), 1 = the tail. Not a cosmetic knob. A ggml tensor is ne0-fastest, so for a (2048, T) output the
# default head window is TOKEN 0 -- the OLDEST token of the pass -- while `conv_input`, whose ne0 is
# K-1+T, has a window that spans every token. Comparing those two as if they were the same window is
# what produced "dense projections identical, conv input not": the dense verdicts were SAME AT TOKEN 0
# only. Guarded by the usual allowlist check -- an unlisted CGC_* is dropped SILENTLY, and a dropped
# tail flag looks exactly like "the tail makes no difference".
if [ -n "${CGC_TENSOR_CAPTURE_TAIL:-}" ]; then
    SERVER_ENV+=(CGC_TENSOR_CAPTURE_TAIL="$CGC_TENSOR_CAPTURE_TAIL")
fi
# [CGC 2026-09-16 §9.18.6 r6] Digest the WHOLE tensor rather than a 32-word window. Set this when the
# question is "is this tensor identical between the arms": that is what the localisation is asking,
# and it is the one thing a window cannot answer, because each node's window lands on its own
# (token, channel) coordinate and two windows of the SAME tensor give opposite verdicts.
if [ -n "${CGC_TENSOR_CAPTURE_HASH:-}" ]; then
    SERVER_ENV+=(CGC_TENSOR_CAPTURE_HASH="$CGC_TENSOR_CAPTURE_HASH")
fi
# [CGC 2026-09-17 §9.18.6 r12] The THIRD capture destination: the pool ROWS the ids select, digested
# ON THE DEVICE at the moment the consumer runs. This is the one reading that separates "the ids point
# at different experts" from "the same expert index holds different bytes" without either arm needing
# a host-visible pointer -- the existing host-side probe (CGC_MMID_MV_DBG) is measurably blind on the
# S1 arm, because it reads op->src[2]->data at ENCODE time and the ids are a GPU-computed node there.
# Same allowlist trap as above, and it bites harder here: a dropped CGC_POOL_CAPTURE produces no POOL
# rows at all, which reads exactly like "that node never ran on this arm".
if [ -n "${CGC_POOL_CAPTURE:-}" ]; then
    SERVER_ENV+=(CGC_POOL_CAPTURE="$CGC_POOL_CAPTURE")
fi
if [ -n "${CGC_POOL_CAPTURE_ROWS:-}" ]; then
    SERVER_ENV+=(CGC_POOL_CAPTURE_ROWS="$CGC_POOL_CAPTURE_ROWS")
fi
if [ -n "${CGC_POOL_CAPTURE_BYTES:-}" ]; then
    SERVER_ENV+=(CGC_POOL_CAPTURE_BYTES="$CGC_POOL_CAPTURE_BYTES")
fi
if [ -n "${CGC_S1_IDENT:-}" ]; then
    SERVER_ENV+=(CGC_S1_IDENT="$CGC_S1_IDENT")
fi
if [ -n "${CGC_S1_TAG:-}" ]; then
    SERVER_ENV+=(CGC_S1_TAG="$CGC_S1_TAG")
fi
# [CGC 2026-09-15 S1 layer-0 gate] Lowest layer index that may use the GPU slot table (default 1).
# Layer 0's MoE FFN is not offloaded in prod25 (full-size expert tensors in the BLAS buffer), so its
# mul_mat_id runs on the CPU backend and needs the ids in a CPU-visible buffer. The baseline leaf is
# a CPU-backend graph input so that holds automatically; a Metal-computed gather does not, and the
# scheduler records the cross-backend copy as "CPU#ffn_moe_slots-0# [NULL]". See the long note in
# llama-graph.cpp (build_moe_ffn) and the GGML_SCHED_DEBUG=2 evidence in Backup/cgc_logs.
if [ -n "${CGC_S1_MIN_IL:-}" ]; then
    SERVER_ENV+=(CGC_S1_MIN_IL="$CGC_S1_MIN_IL")
fi
# [CGC 2026-09-15 S1 CONTROL] CGC_S1_KEEP_LEAF=1 builds every node the S1 branch builds (the per-layer
# table, the CONT, the GET_ROWS, the VIEW) AND the host leaf, and lets mul_mat_id consume the LEAF.
# The graph therefore carries exactly the S1 extra nodes while the ids still come from the host, which
# is the only control that separates "the GPU-computed ids are wrong" from "adding these four nodes
# per layer moves the answer by itself" -- the two remaining explanations for the digest divergence,
# which need opposite fixes. It is also the only arm in which the POST readback can do its direct
# comparison: POST walks every captured ffn_moe_slots node and prints gather-vs-leaf (same=1/0), and
# both tensors are ggml_set_output so both stay readable after the synchronize.
if [ -n "${CGC_S1_KEEP_LEAF:-}" ]; then
    SERVER_ENV+=(CGC_S1_KEEP_LEAF="$CGC_S1_KEEP_LEAF")
fi
# [CGC 2026-09-15 S1 CONTROL] CGC_S1_BUILD_LEAF=1 builds the host leaf AS WELL but leaves it
# unconsumed (ids still come from the GPU gather). This is the design the header comment on
# cache_slot_table_tensors has always claimed S1 implements -- "the leaf itself is still built when
# this is on: it is simply not consumed ... a pure scheduler/dispatch change" -- while
# llama-graph.cpp's if/else never built it. With the ids ruled out (per-layer readback: the gather
# equals the host mapping at every layer) and the extra nodes ruled out (all 39 layers' nodes are
# bit-identical), the absence of this node is the only remaining difference between the real arm
# and the keep-leaf control, so the documented design is the thing to measure.
if [ -n "${CGC_S1_BUILD_LEAF:-}" ]; then
    SERVER_ENV+=(CGC_S1_BUILD_LEAF="$CGC_S1_BUILD_LEAF")
fi
# [CGC 2026-09-15 S1 backend-assignment probe] GGML_SCHED_DEBUG=1 prints one "## SPLIT #N: <backend>
# # M inputs" line per split; =2 additionally prints every node with its assigned backend and the
# reason code. It is the only instrument that answers which backend a given mul_mat_id landed on,
# and it is also ggml's own code path (ggml-backend.cpp:2271 reads the env; :986 prints), so it
# needs no CGC-side plumbing. Why it matters here: the 2026-09-15 20:18 S1 run crashed with
# SIGSEGV inside ggml_compute_forward_mul_mat_id on libggml-cpu -- a signature that appears in no
# other run of the day -- i.e. at least one mul_mat_id was scheduled on the CPU backend while its
# weight operand is pool-repointed into a Metal buffer. Cheap to read, expensive to guess.
if [ -n "${GGML_SCHED_DEBUG:-}" ]; then
    SERVER_ENV+=(GGML_SCHED_DEBUG="$GGML_SCHED_DEBUG")
fi
if [ -n "${CGC_SUBMIT_DBG:-}" ]; then
    SERVER_ENV+=(CGC_SUBMIT_DBG="$CGC_SUBMIT_DBG")
fi
# [CGC M1 union-routable 2026-09-14] Per-layer union + chosen decode path (pool vs gather),
# ranked by max union over 8 layer sweeps, with the layer's usable slot count beside it so the
# `union > usable` window (see docs/M1_POOL_GRAPH_DECOUPLE_PLAN_2026-09-14.md) is directly
# observable instead of inferred from pool geometry. Independent of CGC_OA_ASYNC: the hook runs
# on the default single-submit path too.
if [ -n "${CGC_UNION_LOG:-}" ]; then
    SERVER_ENV+=(CGC_UNION_LOG="$CGC_UNION_LOG")
fi
# [CGC M1 Metal slab 2026-09-14] Capacity of the gather slab in experts (default 64 = cap*top_k
# at cap 8). Override for A/B only -- the capacity must stay a constant, not follow the pool or
# the observed union, or the summation order stops being pool-independent.
if [ -n "${CGC_GATHER_SLAB_CAP:-}" ]; then
    SERVER_ENV+=(CGC_GATHER_SLAB_CAP="$CGC_GATHER_SLAB_CAP")
fi
# [CGC M2 whole-layer streaming 2026-09-14] Enable prefill whole-layer slab path. When on,
# prefill chunks wider than cap materialize the full 256-expert layer into a Metal slab and
# dispatch mul_mat_id against the full-width tensor. Decode stays on the pool path.
if [ -n "${CGC_PREFILL_STREAM:-}" ]; then
    SERVER_ENV+=(CGC_PREFILL_STREAM="$CGC_PREFILL_STREAM")
fi
# [CGC M1 work item 2 · phase split 2026-09-17] Say out loud which regime this launch is in.
#
# The phase split (PREFILL_GRAPH = whole-layer slab, 256 experts / DECODE_GRAPH = pool path) only
# EXISTS when the slab is armed; with it off there is no prefill graph, so prefill chunks wider
# than the decode width go through the pool path and the n_batch clamp stays on. That is the
# intended default, but it used to be visible only as an ABSENCE: the two variables that arm the
# slab are defaulted inside the `prefill250` block above, so "the knob was never set" and "the
# knob was set to off" look identical in the log -- and "why did this chunk take the pool path"
# could only be answered by re-deriving the profile from the launch line. Printed here because the
# binary cannot know: CGC_SERVER_PROFILE is a launcher-shell variable and is deliberately NOT in
# SERVER_ENV (adding it would move every existing .cap ENV fingerprint, and D5's comparability is
# read off that fingerprint -- see eng-gate-0006).
#
# Deliberately NOT auto-armed from runtime memory state: that would make two runs with identical
# .cap fingerprints have different memory layouts, which the gate cannot see. One-time re-baseline
# is the precondition if this ever becomes the default, and the profile should then write =0/=1
# explicitly rather than expressing the state as absence.
if [ -z "${CGC_PREFILL_STREAM:-}" ] || [ "${CGC_PREFILL_STREAM}" = "0" ]; then
    echo "[arm]   slab OFF (profile=${SERVER_PROFILE}): prefill chunks wider than the decode width take the POOL path with the n_batch clamp on; arm with CGC_PREFILL_STREAM=1 and CGC_GATHER_SLAB_CAP>=n_expert (256 for the 35B MoE)"
else
    echo "[arm]   slab ON  (profile=${SERVER_PROFILE}, CGC_PREFILL_STREAM=1, CGC_GATHER_SLAB_CAP=${CGC_GATHER_SLAB_CAP:-64}): prefill chunks wider than the decode width use the whole-layer slab"
fi
# [CGC M2 pool reuse 2026-09-14] Slab-fill diagnostics: CGC_M2_PROFILE prints one CGC-M2-FILL /
# CGC-M2-PROF line per (layer,kind) fill (pool vs disk bytes, ms), CGC_M2_DB_DISABLE turns the
# double-buffer off for A/B. Both must be in this allowlist: unknown CGC_* variables are dropped
# silently, so "the knob did nothing" and "the knob was never set" would look identical in the log
# (roadmap gap #4 -- it already cost two diagnostics on 2026-09-14).
if [ -n "${CGC_M2_PROFILE:-}" ]; then
    SERVER_ENV+=(CGC_M2_PROFILE="$CGC_M2_PROFILE")
fi
if [ -n "${CGC_M2_DB_DISABLE:-}" ]; then
    SERVER_ENV+=(CGC_M2_DB_DISABLE="$CGC_M2_DB_DISABLE")
fi
# [CGC decode phase decomposition 2026-09-15] CGC_PHASE_TIMING prints the per-decode-step
# build/alloc/inputs/compute breakdown (llama-context.cpp:1721, CGC-PHASE line every 32 steps).
# Needed to attribute the measured ~100 ms/token against the 5-10 ms/token budget: without this
# line the only numbers available are the process-wide teardown totals, which say how much IO
# happened but not where in the step it was spent. Same allowlist rule as above: an unlisted
# CGC_* is dropped silently, which looks exactly like "the knob had no effect".
if [ -n "${CGC_PHASE_TIMING:-}" ]; then
    SERVER_ENV+=(CGC_PHASE_TIMING="$CGC_PHASE_TIMING")
fi
# [CGC 2026-09-15] Slab-fill FAILURE diagnostics. fill_job() already reports the exact failing
# pread (offset/want/got/errno/fsize) under LLAMA_EXPERT_CACHE_PREAD_DBG, and
# LLAMA_EXPERT_CACHE_SERIAL_FILL forces the pre-threaded fill loop -- both are the direct way to
# tell "the fill list is wrong" from "the fill raced the read". They were absent from this
# allowlist, so the open layer-1 zero-row defect could not be attributed from the launcher.
if [ -n "${LLAMA_EXPERT_CACHE_PREAD_DBG:-}" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_PREAD_DBG="$LLAMA_EXPERT_CACHE_PREAD_DBG")
fi
if [ -n "${LLAMA_EXPERT_CACHE_SERIAL_FILL:-}" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_SERIAL_FILL="$LLAMA_EXPERT_CACHE_SERIAL_FILL")
fi
if [ -n "${CGC_MMID_MV_DBG:-}" ]; then
    SERVER_ENV+=(CGC_MMID_MV_DBG="$CGC_MMID_MV_DBG")
fi
# [CGC 2026-09-15] CGC_M2_DBG gates the CGC-M2-DBG per-layer line on the large-batch prefill
# branch. It used to print unconditionally (10 lines per run); it is now opt-in, so it has to be
# in this allowlist or the knob is silently dropped (see the note above).
if [ -n "${CGC_M2_DBG:-}" ]; then
    SERVER_ENV+=(CGC_M2_DBG="$CGC_M2_DBG")
fi
# [CGC 2026-09-15] CGC_SLOT_DBG gates the CGC-SLOT per-layer slot-table/remap line in
# llama-context.cpp. It used to print unconditionally (40 lines per run, synchronous fprintf
# on the decode path); it is now opt-in, so it has to be in this allowlist.
if [ -n "${CGC_SLOT_DBG:-}" ]; then
    SERVER_ENV+=(CGC_SLOT_DBG="$CGC_SLOT_DBG")
fi
# [CGC 2026-09-16] Blocker B / the two-pass ensure_batch. The invariant gate
# (llama-expert-cache.cpp: assert every batch member got a slot, slots are distinct, and the
# slot table reads back the slot that was assigned) is useless if the production entry point
# cannot set it -- the defect it detects was silent for exactly that reason: the mapping stayed
# "valid", just shifted, and only surfaced two layers later as NaN. Same argument as
# CGC_SLOT_DBG above.
if [ -n "${LLAMA_EXPERT_CACHE_BATCH_INVARIANT:-}" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_BATCH_INVARIANT="$LLAMA_EXPERT_CACHE_BATCH_INVARIANT")
fi
# [CGC 2026-09-15] P1 prefill-protect. llama-context.cpp:4987 records the measurement: with it
# on, steady-state decode is 22.2 t/s; with it off (the build default) a generation that follows
# a prefill runs 8.1-8.9 t/s, because the prefill's fill churns the very slots decode is about
# to need. `cgc_prefill_protect_on()` is pure env (`getenv(...) != nullptr`), and the launch line
# has an explicit `env` allowlist -- so without this pass-through the knob can never be set from
# run_server.sh, which is why every server-profile decode number so far has been the 8 t/s case.
if [ -n "${CGC_PREFILL_PROTECT:-}" ]; then
    SERVER_ENV+=(CGC_PREFILL_PROTECT="$CGC_PREFILL_PROTECT")
fi
# [CGC 2026-09-15] CGC-IDS / CGC-PREV-PF used to print unconditionally on the decode path
# (n_tokens=2 <= 8 spent the whole 4000-line budget inside one request). Both are opt-in now.
if [ -n "${CGC_IDS_MAX_LINES:-}" ]; then
    SERVER_ENV+=(CGC_IDS_MAX_LINES="$CGC_IDS_MAX_LINES")
fi
if [ -n "${CGC_PREV_PF_DBG:-}" ]; then
    SERVER_ENV+=(CGC_PREV_PF_DBG="$CGC_PREV_PF_DBG")
fi
# [CGC M1 Metal slab 2026-09-14] Print the operands of Metal's own buffer range check at every
# gather-path repoint (CGC-SLAB-CHECK), so a "buffer is nil" is attributable, not guessed.
if [ -n "${CGC_SLAB_DBG:-}" ]; then
    SERVER_ENV+=(CGC_SLAB_DBG="$CGC_SLAB_DBG")
fi
# [CGC M1 Metal slab 2026-09-14] Print the operands Metal itself sees when its range check fails
# (CGC-METAL-NIL), so a nil is attributable instead of inferred from our side of the check.
if [ -n "${CGC_METAL_DBG:-}" ]; then
    SERVER_ENV+=(CGC_METAL_DBG="$CGC_METAL_DBG")
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
# CGC_DBUF=0 disables it. Note HOW that worked: this block only ever pushes `CGC_DBUF=1`, and
# disabling is done by pushing NOTHING, leaving the knob absent. That omission was load-bearing
# until 2026-09-16, because the reader tested non-emptiness, so a literal `CGC_DBUF=0` meant ON and
# only the omission made "=0" behave as documented. The reader is value-aware now, so both
# spellings agree; the omission is kept because it also keeps the env block clean.
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
fi
# [CGC phrase-loop guard 2026-09-08] server-side truncation of live phrase loops
# (>=6 char block x3 consecutive, mirroring the replay quality gate). Default ON;
# set CGC_LOOP_GUARD=0 to disable. CGC_LOOP_GUARD_EVERY = check cadence (tokens).
#
# [CGC 2026-09-16 FIX] This block used to sit INSIDE the `else` branch of the CGC_SOFT_POOL_L1 test
# above -- a missing `fi`. Measured with CGC_DUMP_ENV=1: setting CGC_SOFT_POOL_L1, which is exactly
# what the comment above recommends ("opt back in with CGC_SOFT_POOL_L0=48 CGC_SOFT_POOL_L1=48"),
# silently DROPPED CGC_LOOP_GUARD from the launch env, and CGC_LOOP_GUARD=1 could not restore it:
# "push the guard" lived in the branch that only runs when CGC_SOFT_POOL_L1 is UNSET. So the one
# setting that opts the soft pool back in also switched off the phrase-loop guard -- a quality
# guard, and unreachable, not merely defaulted off. Same disease as the presence-gated knob fixed
# earlier the same day: one setting silently deciding an unrelated one. No default profile sets
# CGC_SOFT_POOL_L1, so the production launch env is unchanged by this fix (verified by fingerprint).
if [ "${CGC_LOOP_GUARD:-1}" != "0" ]; then
    SERVER_ENV+=(CGC_LOOP_GUARD=1)
    [ -n "${CGC_LOOP_GUARD_EVERY:-}" ] && SERVER_ENV+=(CGC_LOOP_GUARD_EVERY="$CGC_LOOP_GUARD_EVERY")
fi
if [ "$SERVER_GLU_FUSED_DOWN" = "1" ]; then
    SERVER_ENV+=(CGC_GLU_FUSED_DOWN=1)
fi
# [CGC 2026-09-15] CGC_MMV_FUSE (fused gate+up+swiglu, ggml-metal-ops.cpp:2659) was NOT in this
# allowlist, so it could never be set from this launcher -- while CGC_GLU_FUSED_DOWN above WAS
# appended. That is an inert combination: ggml_metal_op_can_batch_mmv_glu_down (ops.cpp:2980) is
# only reachable from inside ggml_metal_op_mul_mat_id_glu_fused, which is entered only when
# CGC_MMV_FUSE=1. So the [perf] banner below has been printing `glu_fused_down=1` for a knob that
# does nothing, and the "§8.113: +6.5% speed" it cites is not in effect. Worse, any fusion A/B
# driven through this launcher would have read as "no effect" because the arm never turned the
# fusion on. Pass both through so the pair is testable; default stays off (unchanged behaviour).
if [ -n "${CGC_MMV_FUSE:-}" ]; then
    SERVER_ENV+=(CGC_MMV_FUSE="$CGC_MMV_FUSE")
fi
if [ "$SERVER_WATCHDOG" = "1" ]; then
    SERVER_ENV+=(CGC_WATCHDOG=1)
fi
# [CGC 2026-09-19 P0-1] Two knobs that used to live in the MTP block below are NOT spec
# properties: the engine reads them through getenv() at points that exist whether or not a
# verify/draft batch does.
#   * CGC_MM_BITIDENT -> ggml-metal-ops.cpp:2470 (mul_mat kernel choice; decode's GEMV is M=1,
#     inside its M<=8 range). It is bit-identical pillar 1.
#   * CGC_NO_PREFETCH  -> llama-context.cpp:2058 (background slot prefetch, plain decode).
# While they sat inside `if [ "$SERVER_MTP" = "1" ]`, an MTP-off A/B arm silently ran a DIFFERENT
# KERNEL and a different prefetch policy, so any output difference had two candidate causes and
# could not be attributed to speculation (the confound recorded in .workbuddy/memory §EN-193).
# They are hoisted so a caller can EQUALISE the arms; nothing more.
#
# DEFAULT BEHAVIOUR IS UNCHANGED, deliberately: MTP=1 still gets both (as before), MTP=0 still gets
# neither unless the caller sets them here. Flipping the MTP=0 default would silently re-baseline
# every recorded MTP-off number in this repo (the 10.47 / 12.99 t/s family), a bigger change than
# this bug needs.
SERVER_NO_PREFETCH="${CGC_SERVER_NO_PREFETCH:-}"
if [ -z "$SERVER_NO_PREFETCH" ]; then
    if [ "$SERVER_MTP" = "1" ]; then
        SERVER_ENV+=(CGC_NO_PREFETCH=1)
        SERVER_NO_PREFETCH=1
    else
        SERVER_NO_PREFETCH=0
    fi
elif [ "$SERVER_NO_PREFETCH" != "0" ]; then
    SERVER_ENV+=(CGC_NO_PREFETCH=1)
fi
if [ -z "${CGC_MM_BITIDENT:-}" ]; then
    if [ "$SERVER_MTP" = "1" ]; then
        SERVER_ENV+=(CGC_MM_BITIDENT=1)
    fi
elif [ "${CGC_MM_BITIDENT}" != "0" ]; then
    SERVER_ENV+=(CGC_MM_BITIDENT="$CGC_MM_BITIDENT")
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
    # CGC_NO_PREFETCH is NOT set here any more: it is engine-wide policy (llama-context.cpp:2058),
    # so it moved to the hoisted block above this `if` (2026-09-19). MTP=1 default is unchanged.
    SERVER_VERIFY_DECODE="${CGC_SERVER_VERIFY_DECODE:-1}"
    SERVER_DRAFT_DECODE="${CGC_SERVER_DRAFT_DECODE:-1}"
    [ "$SERVER_VERIFY_DECODE" = "0" ] || SERVER_ENV+=(CGC_VERIFY_DECODE=1)
    [ "$SERVER_DRAFT_DECODE" = "0" ]  || SERVER_ENV+=(CGC_DRAFT_DECODE=1)
    if [ -n "${CGC_SERVER_WARM_NPAST:-}" ]; then
        SERVER_ENV+=(CGC_WARM_NPAST="$CGC_SERVER_WARM_NPAST")
    elif [ "$SERVER_DENSE_IQ4X" = "1" ]; then
        SERVER_ENV+=(CGC_WARM_NPAST=0)
    else
        SERVER_ENV+=(CGC_WARM_NPAST=8)
    fi
    if [ "$SERVER_MTP_CLI_PARITY" = "1" ]; then
        SERVER_ENV+=(CGC_MTP_CLI_PARITY=1)
    fi
    if [ "$SERVER_MTP_NO_WARMUP" = "1" ]; then
        SERVER_ENV+=(CGC_MTP_NO_WARMUP=1)
    fi
    # [CGC 2026-09-15] CGC_MM_BITIDENT=1 forces M<=8 matmuls onto the M-invariant mul_mv path
    # (ggml-metal-ops.cpp:2395). It is bit-identical pillar 1, the production §8 value, and it
    # was completely absent from this allowlist -- so the "expert cache ON is not bit-identical"
    # investigation has been running with one of its three required pillars disabled the whole
    # time. Default 1 to match production; opt out with CGC_MM_BITIDENT=0 for a speed A/B.
    # 2026-09-19: the assignment moved to the hoisted block above this `if` -- the knob decides a
    # KERNEL, so an MTP-off A/B arm must be able to carry it too. MTP=1 stays 1, as before.
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
    # [CGC M4 rejection sampling 2026-09-17] The accept rule for the MTP verify step. OFF by
    # default, and that default matters: with the greedy/gate configuration the draft and target
    # distributions are one-hot at the same token, so min(1, p_t/q) is 1 or 0 and the rule
    # degenerates to the exact-match test it replaces -- i.e. a gate arm is unaffected either way.
    # The rule only bites under sampling (temperature > 0), which is where accept was being
    # mis-measured. Nothing reaches the rejection path unless this is set.
    if [ -n "${CGC_MTP_REJECTION:-}" ]; then
        SERVER_ENV+=(CGC_MTP_REJECTION="$CGC_MTP_REJECTION")
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
# [CGC 2026-09-17] LAYER_CAPS must NOT be gated on the MTP profile. It sizes EVERY layer's pool
# region on BOTH sides (the loader's ne[2] shrink via cgc_layer_cap, and the cache's n_slots_l),
# and the decode baseline arm (p25-gputime) is MTP=0 -- so while this block sat inside
# `if [ "$SERVER_MTP" = "1" ]` the knob was silently dropped exactly where it was wanted.
# Measured 2026-09-17: with CGC_SERVER_LAYER_CAPS set, MTP=1 exported it and MTP=0 exported
# nothing; the engine printed no `LAYER_CAPS per-layer caps` census line at all, and the run
# looked exactly like "the knob has no effect". Byte-identical for every existing config:
# no env + MTP=1 still gets 40-40:256; no env + MTP=0 still gets nothing (as before).
if [ -n "$SERVER_LAYER_CAPS" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_LAYER_CAPS="$SERVER_LAYER_CAPS")
elif [ "$SERVER_MTP" = "1" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_LAYER_CAPS="40-40:256")
fi
# [CGC 2026-09-15] CGC_DUMP_ENV=1 -- print the FULLY-RESOLVED launch environment and argv, then
# exit without launching anything. Inserted here, after every profile default / override has been
# applied and immediately before the exec, so what is printed is bit-for-bit what the server would
# have received.
#
# Why this exists: the 25.17 -> 5 t/s regression was a *configuration drift*, not a code or hardware
# change (CGC_OA_ASYNC 1->0, CGC_SPAC 1->off, CGC_MM_BITIDENT dropped, two bit-identical pillars
# 1->0). Any benchmark that re-derives the production env from a hand-written list will drift the
# same way -- llama-bench in particular, which is now the production measuring stick
# (`scripts/check/llama_bench_matrix.py`). So the bench harness must not have its own copy: it asks
# this script for the env via CGC_DUMP_ENV=1. One source of truth, one place to change.
#
# Format (line-oriented, easy to diff):
#   CGCENV <KEY> <VALUE>   -- resolved scalars the bench driver needs for its own argv
#   ENV <K=V>              -- exactly the SERVER_ENV array handed to `env`
#   ARG <token>            -- exactly the argv handed to the binary (one token per line)
if [ "${CGC_DUMP_ENV:-}" = "1" ]; then
    echo "CGCENV PROFILE   $SERVER_PROFILE"
    echo "CGCENV BIN       $BIN"
    echo "CGCENV MODEL     $MODEL"
    echo "CGCENV CTX       $CTX"
    echo "CGCENV BUDGET    $BUDGET"
    echo "CGCENV NGL       $SERVER_NGL"
    echo "CGCENV LOAD_MODE $SERVER_LOAD_MODE"
    echo "CGCENV BATCH     ${SERVER_BATCH:--}"
    echo "CGCENV UBATCH    ${SERVER_UBATCH:--}"
    echo "CGCENV PORT      $PORT"
    echo "CGCENV LOG       $LOG"
    for _kv in "${SERVER_ENV[@]}"; do echo "ENV $_kv"; done
    for _a in "${SERVER_ARGS[@]}"; do echo "ARG $_a"; done
    exit 0
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
