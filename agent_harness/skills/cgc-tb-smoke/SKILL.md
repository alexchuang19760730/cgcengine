---
name: cgc-tb-smoke
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）跑 terminal-bench 的鏈路 smoke —— 分兩個變體：oracle 變體不需要模型（E1 第 3 條驗收），PrimeAgentAgent 變體需要一個 OpenAI 相容端點。當要在該 repo 跑 `tb run`、或 `tb run` 失敗要診斷（`unknown command: docker compose`、`docker-credential-desktop: not found`、prime-agent 裝不起來、卡住不返回）時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-tb-smoke/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-18 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# flashkv-devserver 的 tb smoke

repo：`/Users/alexchuang/Documents/flashkv-devserver`。相關目錄：`agent_harness/tb_loop/`。

**兩個變體，成本差很多：**

| 變體 | `--agent` | 需要模型？ | 驗什麼 |
|---|---|---|---|
| **oracle**（預設） | 不指定（＝`oracle`） | **不需要** | dataset → docker build → 容器啟動 → 跑 gold solution → 跑測試 → 寫 `results.json`。2026-09-17 實測 **1/1 resolved**（`raman-fitting.easy`） |
| **PrimeAgentAgent** | `--agent-import-path tb_loop.agents.prime_agent_adapter:PrimeAgentAgent` | **需要**一個 OpenAI 相容端點 ＋ 有效 key | 上面全部 **＋ 在容器裡安裝 prime-agent ＋ agent 真的呼叫模型** |

★ **oracle 通過不代表 agent 路徑能跑** —— oracle 跑的是 gold solution，**完全不碰模型、也不用安裝任何 agent**。
兩條路唯一的共同前置只有 Docker。所以「E1 第 3 條驗收通過」不能推論出 agent 那條可用。

## 指令（oracle 變體）

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
PYTHONPATH=agent_harness agent_harness/tb_loop/.venv/bin/tb run \
  -d "terminal-bench-core==0.1.1" --n-tasks 1 \
  --output-path /tmp/tb_smoke --run-id e1_smoke
```

- **入口是 venv 的 `tb` console script**，不是 `python -m terminal_bench.cli.tb`
  （後者在這個版本會 `'terminal_bench.cli.tb' is a package and cannot be directly executed`）。
- `PYTHONPATH` 要指 **`agent_harness/`** —— `tb_loop` 才是那裡的套件名。
- **`cwd` 必須是 repo 根**，而 `harness_dir` 要給**絕對路徑**。
  `tb_loop/README.md` 的範例寫 `PYTHONPATH=agent_harness .venv/bin/tb run … -k harness_dir=tb_loop/harness`，
  但那三者**沒有任何 cwd 能同時成立**（`.venv/bin/tb` 要 cwd=`agent_harness/tb_loop`；
  `PYTHONPATH=agent_harness` 要 cwd=repo 根；而 `agent_harness/tb_loop/tb_loop/` 不存在）。
  **該 README 目前仍是錯的**（2026-09-17 刻意未改：它在 `MANIFEST.jsonl` 裡，改它會多一筆索引漂移）。
- 輸出寫到 `/tmp`，不要寫進 repo（`results/` 會多出未追蹤檔）。約 1–3 分鐘（映像有快取時更快）。
- `--n-tasks 1` 會挑哪一題**不固定**（按時長排序、最長優先）—— 實測遇過
  `raman-fitting.easy`、`build-linux-kernel-qemu`、`play-zork`、**`super-benchmark-upet`**。

**★★ 做 model smoke（或任何要驗「agent 那條路」的跑）就必須釘題，不要讓它挑。**
理由：`--n-tasks 1` 是**按時長排序、最長優先** ⇒ 它會挑到**建置本身就足以吃掉整輪時間**的題。
2026-09-17 實測：它挑了 `super-benchmark-upet`，那一題的 client Dockerfile 是
`FROM python:3.10-slim-bookworm` ＋ `apt-get install -y tmux curl asciinema git build-essential
gcc cmake ccache clang jupyter openjdk-17-jdk …` ＋ `pip install -r requirements.txt`
（我手動重跑同一條 build，**12 分鐘仍無輸出**）。
結果就是 tb 報 `docker compose … build` **exit status 2**、容器從未起來、`results.json` 寫
`Unresolved 1 / Accuracy 0.00%` —— **而那個 Unresolved 與模型／agent 路徑無關，一個字都沒說。**
★ 這種 `Unresolved` 是最容易誤讀的一種：它長得跟「agent 打不到模型」一模一樣。

**怎麼挑**：看 `client/Dockerfile`（有的題放在**任務根目錄**的 `Dockerfile`，有的在 `client/` 子目錄 ——
兩種都遇過）。要選「`FROM` 一個**已快取**的基礎映像 ＋ 只有 `COPY`」的那種。實測最佳選擇：

```sh
TB_SMOKE_TASK=raman-fitting.easy bash agent_harness/tb_loop/scripts/run_model_smoke.sh
```

`raman-fitting.easy` 的 Dockerfile 只有 `FROM ghcr.io/laude-institute/t-bench/python-3-13:latest`
（**本機已快取**，187 MB）＋ `COPY task-deps/ ./` ⇒ 幾乎不用建置；任務是「用 numpy/scipy 擬合
Raman 峰、寫 `results.json`」，小且明確，且 oracle smoke 已在本機證明它 1/1。
查已快取的基礎映像：`docker images | grep -i "t-bench\|laude"`；
查已建好的任務映像：`docker images | grep '^tb__'`（命名 `tb__<task>__client`）。

**★★ 兩個 tb 的硬約束 —— 不知道它們，「釘題」這條建議就是不成立的**（2026-09-17 各踩一次）：

1. **`--task-id` 與 `--n-tasks` 互斥。** 兩個一起傳，tb 在
   `Harness.__init__ → _init_dataset → DatasetConfig` 才擋：
   `ValidationError: Cannot specify both task_ids and n_tasks`。
   訊息裡**完全沒提**是「你多傳了 `--n-tasks`」—— 它長得像 dataset 設定壞了。
   ⇒ 釘題時**必須把 `--n-tasks` 拿掉**（`run_model_smoke.sh` 已改成：有 `TB_SMOKE_TASK` 就傳
   `--task-id`，否則才傳 `--n-tasks`；並改用 **bash 陣列**組參數，不要再靠 `\` 續行拼）。
2. **run 目錄存在但沒有 `tb.lock` ⇒ tb 拒絕，且訊息不告訴你原因。**
   `ValueError: output directory <dir> exists but no lock file found. Cannot resume run without
   lock file.` —— 它說的是「不能續跑」，但真正常見的情境是**上一輪極早就失敗、只留下一個空目錄**。
   ⇒ **連續兩次失敗會疊成這個樣子**（第一次死在參數互斥、留下空目錄；第二次死在這個 ValueError，
   而且看起來像另一件事）。`run_model_smoke.sh` 已加守衛：空目錄清掉、有 `tb.lock` 就照 tb 語義續跑、
   非空又沒 lock 就明確 abort 並叫你換 run-id。
   ★ 順帶：**固定 `TB_SMOKE_RUN_ID` 是風險** —— 失敗一次就會留下目錄，下次撞上。
   預設的 `model_smoke_$(date +%Y%m%d_%H%M%S)` 反而安全。

期望的尾端輸出（oracle）：

```
Results Summary:
| Resolved Trials   | 1       |
| Accuracy          | 100.00% |
Results written to <out>/e1_smoke/results.json
```

## 帶模型的變體（PrimeAgentAgent）—— ★ 三個陷阱，兩個會讓它 hang 而不是失敗

**★ 先講入口：repo 裡現在有一支標準入口，不要每次手打那一長串 `-k`。**

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
bash agent_harness/tb_loop/scripts/run_model_smoke.sh
# 臨時換模型／題目：
#   TB_GEMMA4_MODEL=hy3 TB_SMOKE_TASK=raman-fitting.easy \
#     bash agent_harness/tb_loop/scripts/run_model_smoke.sh
```

它把端點來源做成**三層**，後面的蓋前面的 —— **優先序：環境變數 > `config.local.env` > `config.env`**。
**★ 這一層「環境變數優先」是必要的，不是裝飾**：`config.env` 與 `config.local.env` 都是 `source` 進來的
普通賦值，會把 `export` 的值蓋掉 ⇒ 若不在 `source` 之前先把環境變數記下來、之後再蓋回去，
`TB_GEMMA4_API_KEY=... bash run_model_smoke.sh`（**不想把金鑰落盤**的那種用法）會被檔案裡的空值
**靜默蓋成空**，然後死於前置 3 —— 而那句訊息會指你去填檔案，方向完全相反。

它內建三道前置（docker daemon → `docker compose version` → `api_key` 非空），
**每一道都是 abort 而不是印警告**（金鑰留空時 `exit=3`，且在啟動 tb **之前**就停），金鑰只印尾 4 碼。

**★ 兩個「不碰 docker」的模式，都擺在三道前置之前**（真的不問 docker —— 這一點用**絆線**驗過：
放一個假的 `docker` 進 PATH，它留不下痕跡。第一版把 `--print-only` 寫在 docker 檢查**之後**，
而註解卻宣稱「連 docker 都不問」；因為那台機器的 docker 一直正常，這個假宣稱在測試裡看不出來
⇒ **「測不出來」與「沒有這個問題」不一樣**）：

```bash
# (1) 只印生效設定，不碰任何前置、不啟 tb —— 用來驗「三層優先序」有沒有壞
#     （它壞掉的方式與正常運作長得一樣：靜默用了錯的端點或空值）
bash agent_harness/tb_loop/scripts/run_model_smoke.sh --print-only
#   TB_GEMMA4_MODEL=envmodel TB_GEMMA4_API_KEY=SECRET1234 ... --print-only   # 應顯示 envmodel 與 ***1234
#   TB_GEMMA4_BASE_URL=http://env.example/v1  ... --print-only              # 應顯示那個 URL

# (2) ★ 一秒分辨 key/model 對不對 —— 省下「等數分鐘才發現是 401」的那一條
bash agent_harness/tb_loop/scripts/run_model_smoke.sh --probe
#   200 通 / 401 key 本身 / 403 範圍沒勾到 model / 404 路徑或 model 名，各自印對應處置
```

**為什麼 `--probe` 值得存在**：key 打錯、範圍沒勾、model 名寫錯，在 smoke 裡都要等到容器起來、
prime-agent 裝完、agent 第一次呼叫才顯形（數分鐘），而**那時現場只剩「任務 `Unresolved`」** ——
401 與 403 在容器裡的現場一模一樣。`--probe` 用同一組 base_url/key/model 打一次真請求就把四類分開。
（**live 負對照**：餵 `TB_GEMMA4_API_KEY=bogus_key_1234` → `http=401` ＋ 正確的處置文字、`exit=4`。）

其餘旋鈕：`TB_SMOKE_OUT` / `TB_SMOKE_RUN_ID` / `TB_SMOKE_N_TASKS` / `TB_SMOKE_TASK`。

**★ key 放哪裡 —— 不要寫進 `config.env`。** `config.env` 在版控內，**而且被
`agent_harness/engine_loop/MANIFEST.jsonl` 索引**（`index_assets.py --check` 驗它的 `bytes`/`mtime`）
⇒ 金鑰寫進去會同時進 git **並**讓索引漂移。正確位置是 `agent_harness/tb_loop/config.local.env`
（`.gitignore` 已**指名**排除該路徑；該檔的 `.example` 才是要入庫的）：

```sh
cp agent_harness/tb_loop/config.local.env.example agent_harness/tb_loop/config.local.env
# 然後填 TB_GEMMA4_BASE_URL / TB_GEMMA4_API_KEY / TB_GEMMA4_MODEL
```

**手打版本**（等同那支腳本的核心；`config.local.env` 的變數要先 `export` 進環境）：

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
PYTHONPATH=agent_harness agent_harness/tb_loop/.venv/bin/tb run \
  -d "terminal-bench-core==0.1.1" \
  --agent-import-path tb_loop.agents.prime_agent_adapter:PrimeAgentAgent \
  -m openai/hy3 -k model_name=hy3 -k api_key="$TB_GEMMA4_API_KEY" \
  -k base_url=https://tokenhub.tencentmaas.com/v1 \
  -k harness_dir="$PWD/agent_harness/tb_loop/harness" \
  -k max_turns=12 -k max_tokens=30000 -k timeout_ms=600000 \
  --n-tasks 1 --output-path /tmp/tb_modelsmoke --run-id model_smoke
```

- ★★ **`-m` 的值會蓋掉 `-k model_name=`，而且會一路變成送給端點的 model id**（2026-09-17 從原始碼
  證實，**推翻**本 skill 舊版的「`-m` 只是標籤」那句話）。出處 `terminal_bench/cli/tb/runs.py:60`
  的 `_process_agent_kwargs`：先跑 `for kwarg in agent_kwargs: processed_kwargs[key] = …`，
  **然後**才 `if model_name is not None: processed_kwargs["model_name"] = model_name`
  —— 後套用 ⇒ `-m` 勝。
  ⇒ **`-m` 要給「真正要送給端點的裸 model id」**，不要加 provider 前綴。
  踩過的形狀：寫 `-m "openai/deepseek/deepseek-flash"` ⇒ adapter 拼成
  `local-gemma4/openai/deepseek/deepseek-flash` ⇒ 端點回
  `400 The model or service ID openai/deepseek/deepseek-flash does not exist`。
  **單變數判準**：同一顆容器只把 `openai/` 拿掉，`rc=0`、正常回覆。
- adapter 把 `base_url` 以 `TB_GEMMA4_BASE_URL`／`OPENAI_BASE_URL` 注入**容器內**
  ⇒ 容器裡的 `localhost` **不是**這台 Mac；指向本機 server 要用 `host.docker.internal:<port>/v1`。
  雲端 https 端點不受此限（容器有外網即可）。
- 端點：**WorkBuddy 自己不提供** OpenAI 相容的推論端點（它的 OpenAPI 是企業管理 API、
  且明說不支援個人 API Key）。目前用過的兩家（實測有效）：
  - 騰訊雲 **TokenHub**：`https://tokenhub.tencentmaas.com/v1/chat/completions`，Key 在
    `console.cloud.tencent.com/tokenhub/apikey` 建。
    ★ 2026-09-17 晚間回過 `402 The free trial quota for the service has been exhausted and
    postpaid billing is not enabled`（免費額度用盡、未開後付費）⇒ 遇到 402 就是帳號動作，不是 harness。
  - **MiniMax 開放平台**（2026-09-17 起改用這家）：
    `https://api.minimax.chat/v1`（國內站）或 `https://api.minimaxi.com/v1`（國際站），**兩者皆 200**；
    ★ `https://api.minimaxi.chat/v1` 是**舊主機名**，回 `401 invalid api key (2049)`，別用。
    8 個模型：`MiniMax-M3`（最新）/ `M2.7` / `M2.7-highspeed` / `M2.5` / `M2.5-highspeed` /
    `M2.1` / `M2.1-highspeed` / `M2`。`stream: true` 通（回 `chat.completion.chunk`）。
    ★ M3 是**思考型**：`content` 裡帶 `<think>…</think>` 區塊。

**★ 換端點只要問三件事，一分鐘測完，不必跑 smoke**（每次換家都適用）：
① `GET /v1/models` 通不通 —— 順帶拿到**真實**的模型 id（**文件會過時，`/models` 才是權威**：
TokenHub 的說明說 `qwen3.5-flash` 已下線但它仍在 `/models` 裡）；
② `stream: true` 通不通 —— **prime-agent 一定用 SSE**，這是硬要求；
③ 認證形狀（401 = key 本身不對；403 = key 有效但**可訪問範圍沒勾到模型**；404 = 路由／模型名不對）。
做法：**key 寫進臨時檔、用 python `urllib` 讀檔發請求** —— 不要把 key 放進 `curl` argv（`ps` 看得見）。

**★ 401 與 403 不是同一件事 —— 而它們在容器裡的現場一模一樣**（都是「agent 第一個模型呼叫就失敗，
任務 `Unresolved`、`/app/answer.txt` 不存在」）：

| 碼 | 意思 |
|---|---|
| **401** `{"type":"authentication_error","code":"401002"}` | key 本身不存在／簽名校驗失敗 |
| **403** | key 有效，但**「可訪問範圍」沒勾到你要用的那個模型** ⇒ 很容易誤判成 key 打錯而反覆重試 |

**先在 host 上花 1 秒分辨，不要進容器才猜**：

```bash
curl -s -m 20 -o /tmp/th.json -w 'http=%{http_code}\n' -X POST \
  "https://tokenhub.tencentmaas.com/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy3","messages":[{"role":"user","content":"ping"}],"max_tokens":1}'
# 不帶金鑰時得到 401 只證明「端點活著、路徑對」（實測 TLS 0.5 s / total 0.76 s）；
# 帶金鑰後 200 才算通；403 就是範圍沒勾到。
```

**診斷順序（照這個順序看，就不會被誤導）：**

| 症狀 | 真因 | 處置 |
|---|---|---|
| 容器裡 `bash: curl: command not found` ＋ `prime-agent install failed` | 任務映像（`ubuntu-24-04` 精簡版）**不帶 curl**，而官方安裝腳本要它 | `prime-agent-setup.sh` **已修**：安裝前 `apt-get install -y curl` |
| **tb 卡住不返回**（`max_timeout_sec: inf`），而 `INSTALL_FAIL_STATUS` **沒印出來** | tb 是用 `source install-agent.sh \|\| echo …; tmux wait -S done` 執行的；腳本裡的 **`exit 1` 在被 source 時會殺掉 tmux shell** ⇒ `tmux wait` 永遠不執行 | **已修**：改用 `return 1 2>/dev/null \|\| exit 1`。★ 判準：`INSTALL_FAIL_STATUS` 在日誌裡「出現」可能只是**那行命令本身**（`grep -n` 看它在不在命令列那一行） |
| 卡在 `Install Node.js and npm with standalone Node.js?`（或 PATH／npm／Python runtime 等同族提示） | 官方安裝器有 **5 個** yes/no 提示，`PRIME_AGENT_INSTALLER_NONINTERACTIVE` **只蓋得住 1 個**（Native 安裝）；它優先開 `/dev/tty`，而 tmux 裡開得到 | **已修**：`curl … \| setsid sh`（不給控制終端 ⇒ 五個提示全走 `return 2`＝繼續）。`setsid` 在該映像是 `/usr/bin/setsid` |
| 任務 `Unresolved`，`/app/answer.txt` 不存在 | 這是**正常的失敗**（不是卡住）——agent 的第一個模型呼叫就失敗了 | 看 prime-agent 自己的日誌拿真因（下一條） |
| 要拿 agent 自己的錯誤 | tb 的 `panes/*.txt` **不含 agent 的 stderr**，容器收尾時已刪 | 在 host 上把 `harness/` **複製到 `/tmp`**（別指向 repo 裡的，會被寫入），用同一組 `TB_GEMMA4_*` 跑一次，再讀 `<複本>/logs/agent.jsonl`。401 會長成 `{"kind":"auth","providerErrorType":"authentication_error","status":401,…}` |

**這條路上我犯過的兩個錯（別重犯）：**

1. 把 `exit` 留在被 source 的腳本裡 —— 它不會回報失敗，它會**讓 tb 永遠等下去**
   （`min_timeout_sec: 0.0 max_timeout_sec: inf`），只能手動殺。
2. `PRIME_AGENT_INSTALLER_NONINTERACTIVE=1 curl … | sh` 的賦值**只作用於 `curl`**，
   而安裝器在管線右側的 `sh` 裡 ⇒ 它看不到那些變數，症狀與「開關根本沒用」一模一樣。要用 `export`。
   （同理：`setsid` 也要放在管線**右側**：`… | setsid sh`。）

**修好的證據長相**：容器的 `panes/post-agent.txt` 裡會出現
`Installed Prime Agent 0.9.5 at …` 與 `prime-agent ready: /root/.local/bin/prime-agent`
（後者是 `agent_harness/tb_loop/agents/prime-agent-setup.sh` 自己的驗證行）。

## ★ 同一個端點還能換掉 /refine 的蒸餾器（只換一格，不是整槽）

tb_loop 的學習迴圈裡，**蒸餾器**（`learning/refine_harness.sh`）預設用 freebuff2api 的
`freebuff-codebuff/<SFT_MODEL>`；**freebuff2api 不可達時它會退回端側 gemma4 自己蒸餾自己**（品質較低）。
而 `harness/extensions/freebuff-provider.ts` 是一個**通用 OpenAI 相容 provider**
（base_url / key / model 全從 `SFT_API_BASE_URL` / `SFT_API_KEY` / `SFT_MODEL` 讀）。

**★ 2026-09-17 起有專用旋鈕，用這個 —— 不要再叫使用者改 `SFT_*`：**

```sh
REFINE_API_BASE_URL="https://tokenhub.tencentmaas.com/v1"
REFINE_API_KEY="<同一把 key>"
REFINE_API_MODEL="deepseek/deepseek-v4-flash"
# 用法（本檔不會被 refine_harness.sh 自動載入，要先 export）：
#   set -a; source agent_harness/tb_loop/config.local.env; set +a
#   bash agent_harness/tb_loop/learning/refine_harness.sh <failures.json>
```

- **未設 `REFINE_API_*` 時完全沿用 `SFT_*`** ⇒ 舊行為一字不差（已回歸測過）。
- 端點**只注入 prime-agent 的子行程**（`subprocess.run(..., env=child_env)` 覆寫 `SFT_*`）
  ⇒ `gen_sft.sh` / `CodebuffApiAgent` 那一槽**原封不動**。這是它存在的全部理由。
- **★ 優先序有個坑我第一版就踩了**：`TB_REFINE_MODEL_PATTERN` 被 `config.env` 釘死
  （`source` 是**無條件賦值**），所以它**永遠**優先於任何同名環境變數。症狀極具誤導性：
  **探測已經打對新端點（沒有 WARN），模型字串卻還是舊的** ⇒ 執行時看起來像「端點說找不到模型」。
  修法是「`REFINE_API_MODEL` 有設就直接組 `freebuff-codebuff/${REFINE_API_MODEL}`」。
  **一般化：想用環境變數覆蓋 `config.env` 的值，就必須換一個 `config.env` 不認識的變數名。**

**★ model id 去哪裡抄（不要自創）**：官方「模型列表」上
「DeepSeek-V4-Flash 0731 正式版 原廠直供」的 **model 參數 = `deepseek/deepseek-v4-flash`** ——
而那**正好是 `freebuff-provider.ts` 檔頭寫的預期值**。規格 1M/1M/384k，支援
深度思考、結構化輸出、Function Calling、Cache。同一把 TokenHub key 同時覆蓋 `hy3` 與它。
（另有別名 `deepseek-v4-flash` / `deepseek-v4-flash-0731` / `deepseek-v4-flash-202605`，
以及 V4.1-Flash 的 `deepseek/deepseek-flash`；用別名前先 `GET /v1/models` 確認。）

**★ 為什麼不要用「整槽換」（改 `SFT_*`）**：`SFT_*` 被 10 個檔讀、至少三種用途
（① 蒸餾 ② 生成 SFT 資料 ③ 直接當 tb 的 agent），而 ③ `CodebuffApiAgent` 解析的是
**codebuff 原生 DSML 工具呼叫格式**（該檔 :46、:71）⇒ 換成普通 OpenAI 相容模型會
**請求成功、卻解析不出任何工具呼叫，且沒有錯誤訊息**。詳見本節後面那張表。

**★★ 但先搞清楚「換的是哪一槽」—— `SFT_*` 不是「蒸餾器專用」，它是一整槽「雲端強模型」，
被 10 個檔讀，功能上至少三種用途。整槽換掉的代價不均等：**

| 用途 | 誰 | hy3 換上去會怎樣 |
|---|---|---|
| **① 蒸餾**（`/refine`） | `learning/refine_harness.sh` | **升級** —— teacher 越強，蒸出的教訓越好 |
| **② 生成 SFT 訓練資料** | `gen_sft.sh` → `CodebuffApiAgent` | 跟著 ③ 一起壞 |
| **③ 直接當 tb 的 agent** | `agents/codebuff_api_agent.py` | **會壞，而且壞得隱蔽**（見下） |

**③ 為什麼會壞 —— 「形狀相容 ≠ 語意相容」**：它的 HTTP 形狀**是** OpenAI 相容的
（`:187` `POST {base_url}/chat/completions` ＋ `:184` `Authorization: Bearer`），所以「指過去會不會通」
的答案是「會」。**但它解析的是 codebuff 原生工具呼叫格式**（`:46` 檔內註解自己寫著
「codebuff agent 原生工具调用格式（DSML，标记是全角竖线）」，解析器 `:71 _extract_tool_commands`）。
hy3 不會吐那種格式 ⇒ **請求會成功，但 agent 解析不出任何工具呼叫** —— 沒有錯誤訊息可看。
⇒ **判斷「能不能把 base_url 指過去」時，要看的是那支 client 怎麼解析回應，不是它怎麼發請求。**

⇒ **整槽換的代價不均等**：① 是升級、③（連帶 ②）會**靜默壞掉**。
**✅ 2026-09-17 已補上專用旋鈕**：`refine_harness.sh` 現在讀 `REFINE_API_BASE_URL` / `REFINE_API_KEY` /
`REFINE_API_MODEL`（未設時沿用 `SFT_*`）⇒ **只換①，②③ 不動**（見本節開頭）。

**★ 對照組：引擎線的蒸餾器故意不共用這一槽。** `engine_loop/distill/refine_engine.sh` 用自己的
`REFINE_ENGINE_MODEL`，且 `:93-94` **明寫拒絕猜測模型**（沒設就 `error` 退出）；它出現在
`grep -l` 的結果裡只因 `:43` 的**註解**提到 `SFT_MODEL`。
**教訓：`grep -l` 命中一個檔 ≠ 那個檔讀了那個變數（註解也會被命中）—— 要 `grep -n` 看它落在哪一行。**
（我 2026-09-17 就先據 `grep -l` 寫了一句「改 `SFT_*` 會跨線影響引擎蒸餾」，然後自己更正。）

**已驗掉的前提**：`refine_harness.sh:47` 的可達性探測打 `$SFT_API_BASE_URL/models`。
實測 TokenHub：`GET /v1/models` → **401**（路由存在、要認證 ⇒ 帶對 key 就是 200）；
`GET /models` → **404** ⇒ **base_url 必須含 `/v1`**。
**副作用是好的**：走這條會自動拿到 `reasoning: true` ＋ `contextWindow: 128000`
（freebuff-provider 自己的宣告），比 `gemma4-provider.ts` 的 `reasoning: false` / `32768`
更貼近 hy3 的實況（256k、有保留式思考）。

**★ 但要知道這個檔目前只有 `run_model_smoke.sh` 會讀**（`run_round.sh` 只 source `config.env`）
⇒ 要讓學習迴圈吃到這些值，得自己 `export`，或把 `run_round.sh` 也接上覆寫層
（`run_round.sh` **不在** `MANIFEST` 的 `CURATED` 裡 ⇒ 改它不會讓索引漂移）。

**★ 反例：不要順手把 `engine_loop/distill/llm_client.py` 也指過來。** 它打的是
**`{BASE}/completion`** —— llama.cpp 的**原生**端點，不是 `/v1/chat/completions`，
而且**完全不送 `Authorization`**。TokenHub 兩個都不接受 ⇒ 要換得改路徑 ＋ 加 Bearer 兩處，
而它是引擎線的閉環蒸餾（**跨線**）。這一條的教訓是：**「換 base_url」能不能成立，
取決於協定形狀與認證，不是取決於網址** —— 先讀那支 client 怎麼組請求，再談能不能指過去。

### ★ 怎麼證明「這條接線是通的」，而手上還沒有那把 key —— 用兩面夾

2026-09-17 實測。當「真的跑一次」只缺一個前置（這裡是 key）時，**不要等**：把它拆成兩個各自的
未知項，分別用替身消掉。**假憑證打真端點**（驗我方接線）＋ **真形狀打假端點**（驗成功路徑）：

| 案例 | 端點 | key | 實測結果 |
|---|---|---|---|
| A | 真 TokenHub | **假** | `rc=1`；**本地替身收到 0 個請求**（正確，它去了 TokenHub）；prime-agent 印 `401 The API Key does not exist or signature verification failed …` |
| B | **假**（本地 stub） | 任意 | `rc=0`；替身收到 `POST /v1/chat/completions`、`has_auth=true`、`model="hy3"`、**`stream=true`**；prime-agent 印出 stub 的 `pong` |

**A 證到哪**：extension 有被載入（provider 註冊成功）、`SFT_API_BASE_URL` 確實被用、
**Bearer 頭有送出去**（TokenHub 回的是它**自己的** 401 body，不是通用錯誤 ⇒ 它認得這個認證形狀）、
`--model freebuff-codebuff/hy3` 解析正常、錯誤有浮上來。**B 證到哪**：成功路徑全通 ——
請求 shape、`SFT_MODEL` → wire 的 `model`、SSE 串流、回應解析。
⇒ 剩下唯一變數就是那把 key。**這比「等條件齊了再說」多拿到兩倍資訊，而且兩邊都是可證的。**

假端點（stub）只要回答兩條路由就夠：`GET /v1/models`（`refine_harness.sh:47` 的探測打這個）
與 `POST /v1/chat/completions`（**stream 與非 stream 都要**，因為見下）。放在 `/tmp`。

**★ 從這組測試拿到的三個新事實**：

1. **prime-agent 一定用 streaming**（`"stream": true`）⇒ 端點必須支援 SSE。
   （TokenHub 文件的 curl 示例就是 `"stream": true`，所以沒問題。）
2. **`SFT_MODEL` 就是 wire 上的 `model` 值。**
3. **★ 401 的訊息有誤導性**：prime-agent 把認證失敗印成
   `Run /login to update credentials.` —— 對一個 **API 端點**來說這完全指錯方向
   （會讓人以為要去登入 prime-agent，而不是去換 key）。**看到 `/login` 要先想到「其實是 key」。**

**★ 跑 host 側 prime-agent 的兩條衛生規矩**：
- `PRIME_AGENT_CODING_AGENT_DIR` **絕不指向 repo 裡的 `tb_loop/harness`** —— 那是已追蹤檔，
  它會寫入（實測在該目錄下新增 `daemon-workers/<id>/`、`session-artifacts/<uuid>/`）。
  **複製到 `/tmp` 再指**；驗收方式是 `git status -- agent_harness/tb_loop/harness/` 為空。
- **★ 它起的 daemon 會活過父行程 —— 「跑完就沒了」是錯的（我 2026-09-17 先寫錯再更正）。**
  實測：一輪測試之後仍有 `prime-agent --mode daemon --daemon-socket …`（**`ppid=1`，孤兒**，
  存活 3 分 47 秒）＋ 卡住的 `-p` ＋ 兩個 worker。收工前一定要掃。
  **清法用 PID 差集，不要用 `pkill -f prime-agent`** —— 那個模式會命中**你自己**的命令列
  （你正在跑的腳本文字裡就有這串），這裡已經踩過同族的自命中兩次：

  ```sh
  snap() { ps -Ao pid=,command= | awk 'index($0,"prime-agent") && !index($0,"awk") {print $1}' | sort; }
  BEFORE="$(snap)";  # …跑測試…
  AFTER="$(snap)"; NEW="$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | tr '\n' ' ')"
  [[ -n "${NEW// /}" ]] && kill -TERM ${NEW} 2>/dev/null
  snap   # 收尾必須為空
  ```

**★★ 診斷「refine 卡住」的第一個問題：那個端點通不通（不是模型壞了）。**
實測的**判準很乾淨**（單變數，只換 base_url）：

| 端點 | 觀測 |
|---|---|
| `127.0.0.1:1`（可解析、沒人聽） | **TIMEOUT(25s)，零輸出** |
| `127.0.0.1:18099`（可解析、假端點在聽） | **rc=0**，印出回應 |
| TokenHub（可達、回 **401**，假 key） | **rc=1，秒回**，訊息清楚 |

⇒ **連得到（含回 4xx）＝快速返回；連不到（連線被拒或名稱解析不到）＝卡住不返回。**
所以「卡住」看起來像當掉，其實是**在等一個不會有回應的端點**。
（機制未驗：只觀察到 ≥25s 不返回、無輸出，是否重試退避沒量。）
**附帶**：一個 `-p` 呼叫會對端點發 **2 個** 完全相同的 `POST /v1/chat/completions`。

**★ 另外兩個會一起咬人的東西（2026-09-17 實查）**：
1. **回退那條在 host 上是壞的、而且是設計性的**：`refine_harness.sh` 探測 freebuff2api 失敗後
   回退到 `local-gemma4/<model>`，端點是 `TB_GEMMA4_BASE_URL` 預設
   `http://host.docker.internal:1234/v1`。而 **`host.docker.internal` 只在容器內解析得到**
   （`/etc/hosts` 無此條目、`curl` exit **6 = could not resolve host**），refine 卻跑在 **host** 上。
   ⇒ **兩條分支都不通。** 要修就是「host 側的那個值不能用容器內的名字」。
2. **`SFT_API_KEY` 在 `app/cloud/freebuff2api/.env` 不存在時是空字串**（config.env 的 `else` 分支
   是 `SFT_API_KEY="${SFT_API_KEY:-}"`，不是註解寫的 `"local"`）。
   **兩個缺陷會互相掩蓋**：端點不通時探測在認證**之前**就失敗 ⇒ 你永遠看不到 key 也是空的。
   只有一個失敗訊號時，要先問「**這一個訊號掩蓋了幾個缺陷**」。

## ★★ 判讀鐵律：`total_input_tokens = 0` 底下是**三個互相獨立**的缺陷（2026-09-17 實測）

同一個症狀、`failure_mode` 會換來換去，但底下是三件互不相干的事。修好任一件**不會**讓另兩件變好。
**判準不是 `failure_mode`，而是「agent 那一步實際花了多少秒」**（`run.log` 的
`Blocking command completed in <N>s`）：**10 秒 = agent 根本沒被等待**；53／163／596 秒 = 真的跑了。

> ★★ **更正：不要用 `total_input_tokens` 當判準 —— 它在這條路徑上永遠是 0。**
> tb 的 `abstract_installed_agent.py:176-179` 把
> `return AgentResult(total_input_tokens=0, total_output_tokens=0)` **寫死**，
> 所有 installed agent 都不解析用量。所以「`total_input_tokens > 0` 才算通」這個判準**本身不成立**
> （我先前拿它當 gate，白等了好幾輪）。
> **真正的判準有兩個，都在容器裡、都是硬的**：
> ① 「agent 那一步花了幾秒」（10 秒 = 沒跑；數百秒 = 真的在做事）；
> ② **容器裡有沒有產物** —— `docker exec <c> cat /app/results.json`、
>    `ls -la /prime-agent-harness/sessions/`（session jsonl 的大小就是「做了多少」的代理指標，
>    實測一輪成功的 agent 是 **544 KB**）。

### 修好上面三件事之後，會撞到的第四、第五道牆（2026-09-17 實測，順序已驗證）

- **第四道：測試階段自己建環境，被 60s 切掉。** raman 的 `run-tests.sh` 會
  `apt install curl` → 下 `uv` → `uv venv` → **`uv sync`**，走的是與 kernel bootstrap **同一條慢 PyPI 通道**，
  卡在 `Preparing packages... (4/6)` 就被 `task.yaml` 的 `max_test_timeout_sec: 60.0` 切斷
  ⇒ `failure_mode = test_timeout`、`Error parsing results … No short test summary info found`。
  槓桿：① tb 有 `--global-test-timeout-sec`；② 把測試需要的那幾個**純 python** wheel
  （pytest/pygments/iniconfig/pluggy/packaging）也放進 wheelhouse，並在 setup 腳本裡
  `export UV_FIND_LINKS="$HOME/wheels"`（setup 腳本是被 `source` 進 tmux shell 的，export 會留給測試階段）。
  ★ **不要加 `UV_OFFLINE=1`** —— 解析不到時會直接失敗；只給 find-links 是「先本機、後網路」，嚴格更優。
- **第五道：模型的遵循度。** 換 MiniMax-M3 那輪 agent 真的完成了擬合、寫出 `/app/results.json`
  （`ipython` 工具有了、工具呼叫閉環成立），但**鍵名寫錯**：任務要 `{"G": …, "2D": …}`，
  它寫成 `{"G_peak": …, "2D_peak": …}` ⇒ 即使測試不逾時，schema 也會對不上。
  ⇒ 記住：harness 修好之後，下一個失敗點的現場是「`test failed`（有內容）」，與 harness 缺陷的
  「0 token／逾時（沒內容）」完全不同 —— **不要混為一談**。

| # | 缺陷 | 症狀 | 修法 |
|---|---|---|---|
| 1 | 安裝腳本裡的 **Node.js 兜底**（`deb.nodesource.com` ＋ `apt-get install nodejs`）是安裝階段最後一條外網依賴 | 快時 43s、慢時 **>10 分鐘**（實測 10.7 kB/s，而 nodejs deb 約 25 MB ⇒ 40 分鐘） | **整段刪掉**。官方安裝器不需要 node：`install.sh:76-88` 在 `PRIME_AGENT_INSTALL_METHOD=auto` 且平台偵測成功時直接走 native 並 `return`，`:1078` 的 Node.js 提示一行都不會執行。再 `export PRIME_AGENT_INSTALL_METHOD=binary` 把路徑釘死 |
| 2 | **tb 的 agent 命令預設 `block=False`** ⇒ agent 從未被等待 | agent 階段只有 ~10 秒 | `TerminalCommand(..., block=True, max_timeout_sec=inf)`。預設值出處 `terminal/models.py:13`；`abstract_installed_agent.py:173-179` 送完就 return 一個 0/0 的 AgentResult。上游既有寫法一律 `block=True`（`claude_code_agent.py:64`、`codex_agent.py:46`） |
| 3 | **agent 的時間預算包含「安裝 agent」** | 預算被安裝吃掉 ⇒ 被讀成「模型解不出來」 | `max_agent_timeout_sec` 是**任務自己**訂的（`harness.py:639-643`；raman 的 `task.yaml:35` = 360.0），而 tb 把安裝與 agent 送進**同一個 tmux session** ⇒ 加 `--global-agent-timeout-sec`（★ 放寬後與官方榜單數字**不可直接比較**，要寫進交付文件） |

## ★★ Python kernel：`PRIME_AGENT_KERNEL_PYTHON` 有硬檢查，判準在執行檔裡（不是文件）

解析順序（官方 `docs/rlm-runtime.md` 的 Kernel Lifecycle）：
① `PRIME_AGENT_KERNEL_PYTHON`（要有 current `prime-agent-runtime`）
② `~/.prime/agent/kernel-venv/bin/python`（uv bootstrap 的）
③ XDG 位置。

- **路徑②要一個 bootstrap 標記**。執行檔裡的常數：`xPn = ".bootstrap-version"`、`E$s = ".bootstrap.lock"`、`bPn = 9`。
  離線自己建出來的 venv **沒有那個標記** ⇒ 必被判 stale ⇒ 走 bootstrap ⇒ `uv python install 3.11` 因無網而失敗
  （原文：`Failed to set up the Python kernel runtime. /root/.local/bin/uv python install 3.11 failed with exit code 1`）。
  **⇒ 必須走路徑①**：在 agent 命令裡 `export PRIME_AGENT_KERNEL_PYTHON="$HOME/.prime/agent/kernel-venv/bin/python"`。
- **路徑①有一道硬檢查，缺 default Python packages 直接拒絕**（不是 warning）。原文：
  `PRIME_AGENT_KERNEL_PYTHON points to a Python missing default Python packages`
  `(requests, httpx, yaml, tomli, dotenv, pandas, numpy, scipy, bs4, lxml, pydantic, tyro): …`
  ⇒ 那份清單裡 **numpy/scipy/pandas/lxml/pydantic-core 都是原生擴充** ⇒ wheel 綁 ABI
  ⇒ **「用映像自己的 python ＋ 只裝 pure-python wheel」在這道要求下不可能成立**（我為此白做了一版 19 MB 的 bundle）。
  可行解：bundle **自帶 linux-aarch64 的 standalone CPython 3.11**
  （URL 用 host 的 `uv python list --all-platforms --show-urls` 取，來源 `releases.astral.sh`）
  ＋ cp311 的 wheelhouse（`pip download --platform manylinux_2_28_aarch64 --platform manylinux_2_17_aarch64
  --python-version 3.11 --implementation cp --abi cp311 …`）。好處：與任務映像的 Python 版本無關
  （python-3-13 與 ubuntu-24-04 都適用），且 3.11 正是執行檔裡 `pPn = "3.11"` 期望的版本。
  實測 bundle 128 MB、容器內離線建 venv ＋ 裝完 30 個包 **1 秒**。
- **`mcp` 仍然不要裝**（用 `--no-deps` 裝 runtime）：`import rlm.mcp` **在沒有 mcp 的情況下 OK**
  （`rlm/mcp.py` 只 import 標準庫與 `.mcp_base`；mcp_base 對 mcp SDK 的 import 全在**函式內部**）。
  **這點很關鍵**：kernel shim 把 `import rlm.mcp` 放在 `bash()` 的**同一個 try 裡** ⇒ import 不過會**連 bash 一起廢**。
  裝了反而壞：mcp → pyjwt[crypto] → cryptography，其原生擴充在這台 VM 會 SIGILL。
- 診斷口徑：**「venv 存在」不等於「被接受」** —— 驗收要看 agent 那邊
  `Failed to set up the Python kernel runtime` 與 `points to a Python missing` **都是 0 次**，
  不是看安裝腳本那行 `kernel ready`。

## ★ 判準去哪挖：`strings` 那個執行檔，比讀文件快

想知道 prime-agent 的環境變數／檔名／錯誤文案，直接掃執行檔字串
（`prime-agent` 在 mirror 的 release tarball 裡，解出來約 160 MB）：

```python
import re
data = open('/tmp/pa_bin/prime-agent','rb').read()
for s in re.findall(rb'[\x20-\x7e]{5,}', data):
    if b'kernel-venv' in s or b'KERNEL_PYTHON' in s or b'bootstrap' in s:
        print(s.decode('ascii','replace'))
```

實測一次就拿到：`.bootstrap-version`、`.bootstrap.lock`、`kernel-venv`、`PRIME_AGENT_KERNEL_PYTHON`、
以及三句錯誤文案的完整字串（含那份 default packages 清單）。**文件只說「A bootstrap marker detects
stale environments」，不會告訴你那個 marker 叫什麼、內容是什麼。**

## ★ 效率規矩（我自己的傷）：秒級探針先做，端到端最後做

這條線我連著跑了 5 輪、每輪 5–10 分鐘的端到端 smoke，每輪只換一個變數 —— 這是最貴的做法。
**端到端只用來回答「整體通不通」**；凡是能用秒級探針回答的，不要用端到端去問：

- 「這個 venv 合格嗎」→ 容器內建 venv ＋ 逐個 `import`（**1 秒**），不要跑 smoke。
- 「prime-agent 要求什麼」→ `strings` 掃執行檔（**秒級**）。
- 「這一步花多久」→ 分段計時／PATH 墊片／每 2 秒採樣 `docker top`（**分鐘級**），不要靠猜。
- 「安裝腳本本身對不對」→ 把兩個 tarball `docker cp` 進去、直接 `source` 那支腳本（**~15 秒**），
  不必啟動 tb。

## 前置（缺一不可）—— 2026-09-17 三次失敗才通

```bash
colima start --memory 4 --cpu 2          # 只用 4 GiB，別用 profile 的 8 GiB（本機 16 GB）
docker ps >/dev/null && echo daemon OK
docker compose version                   # 一定要印出 Docker Compose version …
```

**本機的坑（Docker Desktop 已被移除，殘骸還在）**：

| 症狀 | 原因 | 修法 |
|---|---|---|
| `docker: unknown command: docker compose`（exit 125） | `~/.docker/cli-plugins/` 底下 **16 個連結全斷**（都指向已不存在的 `/Applications/Docker.app`），docker **靜默忽略**斷連結 | `brew install docker-compose`，再把 `~/.docker/cli-plugins/docker-compose` 指到 `/opt/homebrew/lib/docker/cli-plugins/docker-compose` |
| `error listing credentials … docker-credential-desktop: not found` | `~/.docker/config.json` 的 `credsStore: "desktop"` 指向同一個消失的 app（`auths` 本來就是空的 ⇒ 移除該鍵不損失憑證） | 備份後移除 `credsStore` |

偵測斷連結：`for f in ~/.docker/cli-plugins/*; do [ -e "$f" ] || echo DANGLE "$f"; done`
（`[ -e ]` 對斷連結是 false；而 `ls` 用 lstat，**照樣把它列出來**，所以「ls 看得到」不代表它活著。）

## 動手前／收工後（本 repo 的硬規矩）

- **★★ 絕對不要改一支「正在執行中」的腳本。** bash 對腳本檔是**增量讀取**（不是一次載入）
  ⇒ 你的編輯會讓它從**錯的位元組位移**續讀，症狀是腳本**尾端**噴出毫無道理的錯誤。
  2026-09-17 實測：把 `run_model_smoke.sh` 丟背景跑（19:52 起），19:58 改了它 ⇒
  尾端出現 `line 187: -m: command not found`、`rc=127` —— 而**已提交的那個檔案第 187 行是
  `-k timeout_ms=…`（前一行的續行反斜線還在），根本不可能變成指令開頭**。
  **受控驗證**（同一現象的最小重現）：造一支 263 行的腳本、在第 2 行後放 `sleep 8`，
  跑起來後在 sleep 期間把它換成 293 行的版本 ⇒ 輸出**截斷在第 1 行**，並噴
  `line 294: unexpected EOF while looking for matching '"'` ＋ `line 295: syntax error: unexpected end of file`；
  **完全不動它的對照組是 0 錯誤**。
  ⇒ **判準：看到「腳本尾端出現莫名其妙的 `command not found` / EOF」，先問「它跑的時候我有沒有改它」**，
  不要先去審腳本內容。★ 而且**錯誤行號是落在「新版檔案」的行數上**，所以拿舊版去對行號會對不上
  （我就是先拿舊版對 187 行，才多繞一圈）。
  要改就跑完再改，或改另一份複本再跑 —— 尤其**長時間的背景任務**（smoke 動輒 10+ 分鐘）。
- **空窗要讀日誌，不是讀 `pgrep`**：行程掃描是空的**不代表**沒人在量測。
  要先看 `<repo>/.workbuddy/memory/YYYY-MM-DD.md` 的最新段落（另一條線一天做 20–29 輪，
  取樣空窗可能只有幾十秒）。這是同一個陷阱踩過兩次。
- **用 repo 已經有的閘門，不要手寫**：`bash agent_harness/engine_loop/runners/preflight.sh`
  （`--self-test` **5/5 含正負對照**：命令列提到名字的 wrapper 被**排除**、真正以那個名字執行的行程
  被**認出**）。★ 我手寫過 `ps -Ao comm= | grep -x llama-server` —— **在 macOS 上 `comm` 不是
  basename 而是完整路徑**（`/Users/.../build/bin/llama-server`），所以它**永遠不命中**：
  那個閘門回報的「乾淨」與「有、但我比對錯欄位」長得一模一樣（lesson `eng-gate-0054`）。
  用 `pgrep -f`／`grep command` 時要寫成 `[l]lama-server` 這種自我排除形式，否則會命中**自己的命令列**。
- 檢查與動作要在同一個分支裡（`if … ; then exit; fi` 緊接動作）。
- **動別的東西之前先問**：起 llama-server（13 GB）或任何服務，即使「機器看起來很空」也一樣 ——
  這一輪就是先被叫去跑、兩分鐘後被叫停。工作目錄一律是 `flashkv-devserver`；
  `flashkv0516` 是它的**主 repo**（worktree 關係），踏進去要另外授權。
- 收工：`colima stop`，並把 `~/.colima/default/colima.yaml` 的 `cpu`／`memory` 改回 `4`／`8`。
- **macOS 沒有 `timeout`**；直接跑，或用 `gtimeout`。
- **殺不掉別人（或自己殘留）的行程時**：孤兒行程（`ppid=1`）可能不在你這個 session 的行程樹內，
  沙箱下的 `kill -9` 會靜默無效 —— 要用非沙箱的執行方式才殺得掉。
  （daemon 類的，例如 prime-agent 的 supervisor/worker，也是這樣。）

## 失敗時怎麼讀

- terminal-bench 只把 `CalledProcessError` 的 traceback 丟出來，**看不到 compose 的 stderr**。
  要手動重現那個 build：

```bash
D=~/.cache/terminal-bench/terminal-bench-core/0.1.1/<task>
T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME=tb-client \
T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME=tb-client-1 \
T_BENCH_TEST_DIR=/app T_BENCH_TASK_LOGS_PATH=/tmp/tblogs T_BENCH_CONTAINER_LOGS_PATH=/logs \
docker compose -p diag -f "$D/docker-compose.yaml" build
```

  沒有那 5 個 `T_BENCH_*` 變數會先死在 `container_name '' does not match pattern` ——
  那是**你的重現不忠實**，不是真缺陷。
- exit **125** ＝ compose 指令本身沒跑起來（外掛／設定）；exit **1** ＝ compose 跑了但失敗（憑證／Dockerfile）。
- 要看 agent 在那個瞬間在做什麼：`docker exec <container> tmux capture-pane -p -t 0`
  （容器在 run 期間是活的；**收尾後會被 down 掉，就看不到了**）。
- `run.log` 的 `Sending keys: ['…', 'Enter'] … min_timeout_sec / max_timeout_sec`
  是診斷「卡住」最直接的一行：**`max_timeout_sec: inf` 的那一步**就是會永遠等的那一步。

### ★ 先看 token 數：這個失敗是不是「關於模型」的？

- **`results.json` → `results[0].total_input_tokens` 是 0 ⇒ agent 從來沒有跟模型講過話。**
  那麼同一個物件裡的 `is_resolved: false` / `failure_mode: agent_timeout` / `Unresolved`
  **不是**關於模型或 agent 路徑的證據。
  這條線**四個實例全部屬這一類**（① 選題 ⇒ `docker compose build` exit 2；② `--task-id`/`--n-tasks` 互斥；
  ③ 殘骸目錄造成假續跑；④ 容器內安裝 agent 逾時）。
  ⇒ **引用任何失敗數字前，先問「被測物有沒有真的被執行過」。**

### 收尾後才讀得到的原文在哪：run 目錄的 `panes/`，不是 `run.log`

```
<out>/<run-id>/<task>/<trial>/panes/{pre-agent,post-agent,post-test}.txt   ← tmux pane 全文
<out>/<run-id>/<task>/<trial>/commands.txt                                 ← 送進去的按鍵
```
`run.log` 只告訴你**哪一步**卡住，pane 告訴你**為什麼**。實例：`run.log` 說
`Blocking command completed in 421.89s` ＋ `Agent timed out after 360.0s`（＝卡在安裝），
而 `post-agent.txt` 尾給出決定性的（**原文**）：

```
Downloading compiled Prime Agent
curl: (28) Operation timed out after 300003 milliseconds with 49090240 out of 59607437 bytes received
prime-agent install failed
INSTALL_FAIL_STATUS
```

官方 `install.sh` 下載 prime-agent 的 59.6 MB 編譯產物，`--max-time 300` **寫死**
（`prime_agent_curl_download -fsSL --connect-timeout 10 --max-time 300 …`），
全檔**沒有**可調該值的環境變數（`PRIME_AGENT_PROBE_TIMEOUT_SECONDS` 只管 probe，不管下載）。

### ★★「同一個 URL 在 host 與在容器內各量一次」＝分辨「上游慢」vs「容器網路慢」

實測同一個 release tarball：host **12.5 MB/s（4.55 s 抓完整 59,607,437 B）**
vs 容器 **164 KB/s（300 s 收 49,090,240 B）** ⇒ **差 80 倍，瓶頸是 colima NAT，不是上游**。
⇒ 按容器速率 59.6 MB 需約 **364 s** > cap 300 s ⇒ **重試也一定失敗**（當成偶發會白費一整輪）。

```bash
# host 端樣本（20 秒就夠，不必抓完整檔）
curl -s -o /dev/null --max-time 20 -w "%{size_download} %{speed_download}\n" "<url>"
# 容器端不必另外量：curl 的 (28) 訊息自帶已收位元組數與毫秒數
```

可用的槓桿：`PRIME_AGENT_DOWNLOAD_BASE_URL`（`install.sh:10`，base URL 可換）⇒ 指向 host 上的本地鏡像
（容器內用 `host.docker.internal:<port>` —— 與 `TB_GEMMA4_BASE_URL` 預設值同一個機制）。
鏡像需覆蓋 channel 檔（`stable` → `v0.9.5`）、`releases/v0.9.5/SHA256SUMS`、arm64 tarball。

### ★★ 已實作的做法（2026-09-17）：host 預抓 → 注入容器 → 容器內 **loopback** 供檔

**不要**起 host 的 http server 讓容器連 —— 走 `copy_to_container`（Docker API）**不經過**那個慢網路，
再用容器自己的 loopback 供檔。三個檔案就是全部：

| 檔案 | 角色 |
|---|---|
| `agent_harness/tb_loop/scripts/fetch-prime-agent-mirror.sh` | host 端抓一次並快取（`~/.cache/prime-agent-mirror{,.tar.gz}`，~112 MB）；**冪等**，`--refresh` 重抓；逐檔驗 sha256 |
| `prime_agent_adapter.py` 的 `_ship_mirror()` | 把那個 tarball 用 `session.copy_to_container(..., container_dir="/installed-agent")` 送進去 |
| `prime-agent-setup.sh` | 解鏡像 → `python3 -m http.server <port> --bind 127.0.0.1` → 設 loopback feed 的兩個 env → 跑官方安裝器 |

為什麼是官方支援的路徑（**不是 hack**）：`install.sh` 的 `prime_agent_validate_download_base_url`
對 `http://*` 只在 `PRIME_AGENT_ALLOW_INSECURE_HTTP_FOR_TESTS=1` **且**
`prime_agent_is_loopback_test_base_url`（只接受 `http://127.0.0.1:<數字埠>`）時放行 ——
上游自己的錯誤訊息就寫著「Local loopback test feeds require …」。

**實測**（同一顆容器、同一輪）：安裝 **421.89 s 逾時失敗 → 43.26 s rc=0**
（43 s 含 113 MB 的 `docker cp` 與解壓）。loopback server 的 access log 證明它只取了三個檔。

⚠️ **`--autonomous` 需要 Python kernel，不要為了省時間關掉它。**
`PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=0` 看起來很划算（那段 bootstrap 實測 **>418 s**），
但 agent 自己會報「`Status: blocked — no executable tool is available`」——
因為 **`bash()` 是透過 Python REPL 暴露的**，kernel 起不來就等於**沒有任何工具**。
```text
- Every `ipython` call returns: `Failed to set up the Python kernel runtime.
  uv is required to set up the Python kernel.`
- `bash()` is exposed only through the Python REPL, so the broken kernel also removes shell access.
```
⇒ kernel 是必需品；它慢是因為它也要下 `uv` ＋ Python ＋ 一批 wheel，**也需要同樣的鏡像／預熱待遇**。
在那之前，安裝會慢（但正確）。

**模型路徑的最小可用證明**（不必跑完整 task）：
```bash
# 在容器內，設好 loopback 與環境後
prime-agent -p --offline --model local-gemma4/deepseek/deepseek-flash 'reply with the single word: pong'
# → rc=0，輸出 pong   ← 這一條通了，代表 provider/認證/端點/串流全都通
```
`--offline` 是「Disable startup network operations」，**不**擋模型呼叫（別把它當成離線模式）。
