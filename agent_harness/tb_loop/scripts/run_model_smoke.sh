#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# tb model smoke —— 带模型的 PrimeAgentAgent 变体（会真的呼叫端点）
#
# 与 run_round.sh 的差别：只跑 1 题、输出到 /tmp、**不进学习循环**（不 refine、不比对）。
# 它只回答一个问题：容器能不能装起 prime-agent，而且 agent 真的呼叫得到模型。
#   ★ oracle 变体通过**不能**推论出这件事 —— oracle 跑的是 gold solution，
#     完全不碰模型、也不用装任何 agent；两条路唯一共同的前置只有 Docker。
#
# 端点的来源有**三层**，后面的盖前面的（优先级：环境变量 > config.local.env > config.env）：
#   config.env        版控内的预设（本机 llama.cpp：host.docker.internal:1234）
#   config.local.env  ★ 本机私密覆写，gitignored。云端端点（TokenHub）写在这里，
#                     这样金钥才不会进 git（config.env 在 MANIFEST 索引里）。
#   环境变量          临时用、不落盘：TB_GEMMA4_API_KEY=... bash <本档>
#
# 用法:
#   bash agent_harness/tb_loop/scripts/run_model_smoke.sh
#   bash agent_harness/tb_loop/scripts/run_model_smoke.sh --print-only   # 只印生效设定，不碰 docker
#   TB_GEMMA4_API_KEY=... TB_GEMMA4_MODEL=hy3 bash agent_harness/tb_loop/scripts/run_model_smoke.sh
#   TB_SMOKE_TASK=raman-fitting.easy bash agent_harness/tb_loop/scripts/run_model_smoke.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB_LOOP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TB_HARNESS_ROOT="$(cd "${TB_LOOP_DIR}/.." && pwd)"
TB_REPO_ROOT="$(cd "${TB_HARNESS_ROOT}/.." && pwd)"

# ---- 先把「外面传进来的」三个值记下来 ---------------------------------------
# config.env 与 config.local.env 都是 source 进来的普通赋值，会把 export 的值盖掉。
# 不记的话，`TB_GEMMA4_API_KEY=xxx bash run_model_smoke.sh`（不想把金钥落盘的那种用法）
# 会被 config.local.env 里的空值静默盖成空，然后死于前置 3 —— 而那句讯息会指你去填档案，
# 方向完全相反。优先级：环境变量 > config.local.env > config.env。
# ★ 用 if 而不是 `[[ -n ]] && ...`：后者在条件为假时回传 1，而 set -e 会因此终止整个脚本。
_ENV_BASE_URL="${TB_GEMMA4_BASE_URL:-}"
_ENV_API_KEY="${TB_GEMMA4_API_KEY:-}"
_ENV_MODEL="${TB_GEMMA4_MODEL:-}"

# shellcheck source=../config.env
source "${TB_LOOP_DIR}/config.env"

# ---- 本机私密覆写（不存在 = no-op，行为与原本的 run_round.sh 相同）----
TB_LOCAL_ENV="${TB_LOOP_DIR}/config.local.env"
if [[ -f "${TB_LOCAL_ENV}" ]]; then
    # shellcheck source=../config.local.env
    source "${TB_LOCAL_ENV}"
    echo "[smoke] 已载入本机覆写: ${TB_LOCAL_ENV}"
else
    echo "[smoke] 未找到 ${TB_LOCAL_ENV} —— 用 config.env 的预设端点（本机 llama.cpp）"
fi

# ---- 环境变量优先（显式传进来的赢过上面两个档）----
if [[ -n "${_ENV_BASE_URL}" ]]; then TB_GEMMA4_BASE_URL="${_ENV_BASE_URL}"; fi
if [[ -n "${_ENV_API_KEY}"  ]]; then TB_GEMMA4_API_KEY="${_ENV_API_KEY}";  fi
if [[ -n "${_ENV_MODEL}"    ]]; then TB_GEMMA4_MODEL="${_ENV_MODEL}";      fi
unset _ENV_BASE_URL _ENV_API_KEY _ENV_MODEL

# ---- 打印生效设定（金钥只印尾 4 码）----------------------------------------
# 做成函数是因为 --print-only 也要用它 —— 而「可被自测」对这支脚本不是装饰：
# 三层优先序（环境变量 > config.local.env > config.env）如果只靠读程式码相信，
# 那么它坏掉的方式（静默用错端点／用错空值）与正常运作长得一样。
print_eff() {
    echo "[smoke] base_url = ${TB_GEMMA4_BASE_URL}"
    echo "[smoke] model    = ${TB_GEMMA4_MODEL}   (provider 前缀 ${TB_MODEL_PREFIX} ⇒ prime-agent 侧 ${TB_MODEL_PREFIX}/${TB_GEMMA4_MODEL})"
    if [[ -n "${TB_GEMMA4_API_KEY}" ]]; then
        echo "[smoke] api_key  = ***${TB_GEMMA4_API_KEY: -4}   (len=${#TB_GEMMA4_API_KEY})"
    else
        echo "[smoke] api_key  = (空)"
    fi
    echo "[smoke] dataset  = ${TB_DATASET}"
}

# ============================================================
# 两个「不碰 docker」的模式 —— ★ 它们必须摆在下面三道前置**之前**。
#   （第一版把 --print-only 写在 docker 检查之后，而注解却宣称「连 docker 都不问」。
#     因为这台机器的 docker 一直正常，那个假宣称在测试里看不出来 —— 注解与顺序，两边都要对。）
# ============================================================

# ---- --print-only: 只印生效设定就退出 ---------------------------------------
if [[ "${1:-}" == "--print-only" ]]; then
    print_eff
    echo "[smoke] --print-only: 未做任何前置检查、未启动 tb。"
    exit 0
fi

# ---- --probe: 一秒分辨 key/model 对不对（不必等 docker build 才知道）--------
# 为什么需要它：key 打错／范围没勾／model 名写错，在 smoke 里都要等到容器起来、
# prime-agent 装完、agent 第一次呼叫才显形（数分钟），而那时现场只剩「任务 Unresolved」。
# 这里用同一个 base_url ＋ 同一把 key ＋ 同一个 model 打一次真请求，把四类失败分开：
#   200 通 ／ 401 key 本身 ／ 403 范围没勾到 model ／ 404 路径或 model 名
# （401 与 403 在容器里的现场一模一样 —— 那是这条命令存在的主要理由。）
if [[ "${1:-}" == "--probe" ]]; then
    if [[ -z "${TB_GEMMA4_API_KEY}" ]]; then
        echo "[probe] abort: TB_GEMMA4_API_KEY 是空的 —— 先填 ${TB_LOCAL_ENV}" >&2
        exit 3
    fi
    URL="${TB_GEMMA4_BASE_URL%/}/chat/completions"
    BODY="$(mktemp)"
    CODE="$(curl -sS -m 30 -o "${BODY}" -w '%{http_code}' -X POST "${URL}" \
        -H "Authorization: Bearer ${TB_GEMMA4_API_KEY}" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${TB_GEMMA4_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4}" \
        2>/dev/null)" || CODE="000"
    echo "[probe] POST ${URL}"
    echo "[probe] model=${TB_GEMMA4_MODEL}  key=***${TB_GEMMA4_API_KEY: -4}  ⇒ http=${CODE}"
    case "${CODE}" in
        200) echo "[probe] OK：key 与 model 都能用，可以跑 smoke 了。" ;;
        401) echo "[probe] 401：key 本身不对 —— 复制时漏字／被截断，或被停用了。"
             echo "        也确认这把 key 是在**国内站**控制台建的：国内站是 tokenhub.tencentmaas.com，"
             echo "        国际站是 tokenhub-intl.tencentcloudmaas.com，两边 endpoint 不同、key 不通用。" ;;
        403) echo "[probe] 403：key 有效，但**可访问范围没勾到 ${TB_GEMMA4_MODEL}**。"
             echo "        去 API Key 管理 → 编辑 → 把 ${TB_GEMMA4_MODEL} 加进范围（或直接勾全选）。" ;;
        404) echo "[probe] 404：路径或 model 名不对。"
             echo "        model 值要照控制台「模型列表」的 model(调用参数) 栏填。" ;;
        *)   echo "[probe] 非预期状态码，看下面 body。" ;;
    esac
    head -c 400 "${BODY}" 2>/dev/null; echo
    rm -f "${BODY}"
    if [[ "${CODE}" == "200" ]]; then exit 0; else exit 4; fi
fi

# ---- 前置 1: Docker daemon -------------------------------------------------
# terminal-bench 靠 docker compose 起任务容器。daemon 不在时 tb 的报错是一层
# CalledProcessError traceback，看不出真因，所以在这里先挡。
if ! docker ps >/dev/null 2>&1; then
    echo "[smoke] abort: docker daemon 不可用。" >&2
    echo "        colima start --memory 4 --cpu 2   # 别用 profile 的 8 GiB，本机只有 16 GB" >&2
    exit 3
fi

# ---- 前置 2: docker compose 外挂 -------------------------------------------
# exit 125 = compose 指令本身没跑起来（外挂/设定），不是 compose 跑了但失败。
# 本机的坑是 Docker Desktop 被移除后 ~/.docker/cli-plugins/ 底下留下断连结，
# 而 docker 对断连结是**静默忽略**的（ls 看得到，不代表它活着）。
if ! docker compose version >/dev/null 2>&1; then
    echo "[smoke] abort: docker compose 外挂不可用（docker: unknown command 的 exit 125 多是这个）。" >&2
    echo "        侦测断连结: for f in ~/.docker/cli-plugins/*; do [ -e \"\$f\" ] || echo DANGLE \"\$f\"; done" >&2
    echo "        修法: brew install docker-compose，再把 ~/.docker/cli-plugins/docker-compose" >&2
    echo "              指到 /opt/homebrew/lib/docker/cli-plugins/docker-compose" >&2
    exit 3
fi

# ---- 前置 3: api_key 非空 --------------------------------------------------
# 空金钥一定会失败，但失败的现场在容器深处（agent 第一个模型呼叫），
# 拿真因要再把 harness 复制到 /tmp 才读得到 agent.jsonl。在这里挡便宜得多。
if [[ -z "${TB_GEMMA4_API_KEY}" ]]; then
    echo "[smoke] abort: TB_GEMMA4_API_KEY 是空的。" >&2
    echo "        填进 ${TB_LOCAL_ENV} 的这一行:" >&2
    echo "            TB_GEMMA4_API_KEY=\"...\"" >&2
    echo "        （或者不要落盘：TB_GEMMA4_API_KEY=... bash $0 —— 环境变量优先于那两个档）" >&2
    exit 3
fi

print_eff

OUT_ROOT="${TB_SMOKE_OUT:-/tmp/tb_modelsmoke}"
RUN_ID="${TB_SMOKE_RUN_ID:-model_smoke_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"
echo "[smoke] output   = ${OUT_ROOT}/${RUN_ID}"

# ---- 續跑守衛 --------------------------------------------------------------
# tb 的續跑機制是「run 目錄裡有 tb.lock 才准續」。若目錄存在但**沒有** tb.lock，
# 它會噴：ValueError: output directory <dir> exists but no lock file found.
#         Cannot resume run without lock file.
# 而那句話**完全沒說**「上一次那一輪極早就失敗了、只留下一個空目錄」——
# 那個情境幾乎一定是「第一次跑掛在建置／參數錯誤」，而不是真的要續跑。
# 實測：兩次連續失敗就是這樣疊出來的（第一次死在 DatasetConfig 參數互斥，留下空目錄；
#       第二次死在這個 ValueError）。所以這裡先判掉：
#   空目錄        ⇒ 清掉（沒有任何東西會丟）繼續跑
#   有 tb.lock    ⇒ 照 tb 的續跑語義走下去（只印一行提醒）
#   非空又沒 lock ⇒ 明確 abort，並告訴你換 run-id（不要猜、也不要自動刪別人的產物）
RUN_DIR="${OUT_ROOT}/${RUN_ID}"
if [[ -d "${RUN_DIR}" ]]; then
    if [[ -f "${RUN_DIR}/tb.lock" ]]; then
        echo "[smoke] 注意: ${RUN_DIR} 已存在且有 tb.lock —— tb 會以「續跑該輪」處理。"
    elif [[ -z "$(ls -A "${RUN_DIR}" 2>/dev/null)" ]]; then
        echo "[smoke] ${RUN_DIR} 是上一次極早失敗留下的空目錄，清掉後繼續。"
        rmdir "${RUN_DIR}"
    else
        echo "[smoke] abort: ${RUN_DIR} 存在、非空、且沒有 tb.lock。" >&2
        echo "        tb 會拒絕並說「Cannot resume run without lock file」，那句話不會告訴你原因。" >&2
        echo "        換一個 TB_SMOKE_RUN_ID，或自行確認內容後刪掉該目錄。" >&2
        exit 3
    fi
fi

# ---- 跑 -------------------------------------------------------------------
# cwd 必须是 repo 根，而 PYTHONPATH 要指 tb_loop 的**父目录**（agent_harness/）——
# --agent-import-path 要的套件名是 tb_loop.agents.…，所以解译器得看得到 tb_loop/ 那一层。
# 入口是 venv 的 tb console script，不是 python -m terminal_bench.cli.tb
# （后者在这个版本会说该 package 不能直接执行）。
# harness_dir 给**绝对路径**（tb_loop/README.md 的范例那三者没有任何 cwd 能同时成立）。
cd "${TB_REPO_ROOT}"

# ★ 金钥**不进命令列**：adapter 本来就支援从环境变数读（`api_key or os.environ["TB_GEMMA4_API_KEY"]`，
#   已用哨兵值实测：不传 `-k` 时读 env，传了则 `-k` 优先），所以这里 export 而**不要**再传 `-k api_key=`。
#   理由不是洁癖：`tb run … -k api_key=…` 会让完整金钥出现在 **`ps` 的命令列**里 ——
#   这台机器上任何本机使用者都能看到（这一轮就是靠 `ps` 看到的）。
#   同一个理由也适用于其他密文：**能走环境变数就不要走 argv。**
export TB_GEMMA4_API_KEY

# shellcheck disable=SC2086
TB_ARGS=(
    run
    -d "${TB_DATASET}"
    --agent-import-path "tb_loop.agents.prime_agent_adapter:PrimeAgentAgent"
    -m "openai/${TB_GEMMA4_MODEL}"
    -k model_name="${TB_GEMMA4_MODEL}"
    -k base_url="${TB_GEMMA4_BASE_URL}"
    -k model_prefix="${TB_MODEL_PREFIX}"
    -k harness_dir="${TB_HARNESS_DIR}"
    -k max_turns="${TB_MAX_TURNS}"
    -k max_tokens="${TB_MAX_TOKENS}"
    -k timeout_ms="${TB_TIMEOUT_MS}"
    -k max_continuations="${TB_MAX_CONTINUATIONS}"
    --n-concurrent 1
    --output-path "${OUT_ROOT}"
    --run-id "${RUN_ID}"
)

# ★ --task-id 與 --n-tasks **互斥** —— tb 的 DatasetConfig 會硬擋：
#     ValidationError: Cannot specify both task_ids and n_tasks
#   而 tb 只在 **Harness.__init__ → _init_dataset → DatasetConfig** 才發現，
#   所以症狀是一個 pydantic traceback、`rc=1`，訊息裡完全沒提是我們多傳了 --n-tasks。
#   第一版就是「永遠傳 --n-tasks」＋「有 TB_SMOKE_TASK 時再傳 --task-id」⇒ 一釘題就必炸。
#   ★ 為什麼非釘題不可：--n-tasks 1 是「按時長排序、最長優先」，它會挑到
#     建置本身就吃掉整輪的題（實測 super-benchmark-upet：apt 裝 clang/jupyter/openjdk，
#     手動重跑同一條 build 12 分鐘仍無輸出）⇒ 那時的 Unresolved 與模型／agent 路徑無關。
if [[ -n "${TB_SMOKE_TASK:-}" ]]; then
    TB_ARGS+=(--task-id "${TB_SMOKE_TASK}")
else
    TB_ARGS+=(--n-tasks "${TB_SMOKE_N_TASKS:-1}")
fi

PYTHONPATH="${TB_HARNESS_ROOT}" "${TB_TB_BIN}" "${TB_ARGS[@]}"

echo
echo "[smoke] 结果: ${OUT_ROOT}/${RUN_ID}/results.json"
echo "[smoke] 注意: 本轮会写回 ${TB_HARNESS_DIR}（logs/ sessions/ daemon-workers/）——"
echo "        那些是刻意纳管的跨轮学习状态，收工前看一次 git status 决定要不要提交。"
