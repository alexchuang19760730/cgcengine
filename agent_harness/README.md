# agent_harness — 傘狀入口

這裡住著**兩個形狀相同、獎勵不同的迴圈**，加上它們共用的判準、記憶與索引。
學習發生在 harness 的**狀態**（文本），不在模型權重——這是 `tb_loop/README.md` 已驗證過的原則。

| 迴圈 | 任務 | 獎勵 | 入口 |
|---|---|---|---|
| **`tb_loop/`** | Terminal-Bench 題目 | 測試 pass / fail | `tb_loop/run_round.sh` |
| **`engine_loop/`** | 引擎的一個調校假設 | 吞吐、bit-identical 閘門、池計數不變量、logits oracle | `engine_loop/index_assets.py` ＋ `scripts/check/*` |

兩者都走同一條四階段：`evaluate → extract → refine → compare`。

---

## 目錄

```
agent_harness/
├── README.md                 ← 本檔（傘狀入口）
├── CONVENTIONS.md            ★ 判準憲章：A 數字什麼時候可以引用／B 診斷／C 讀原始碼／D 流程／E 自我約束
├── PLAN_ENGINE_LOOP_2026-09-15.md   兩個迴圈的規劃與 E0–E4 驗收
│
├── tb_loop/                  ★ 任務迴圈（Terminal-Bench × gemma4 × prime-agent）
│   ├── README.md             它的完整說明（安裝、彩排、SFT、/refine）
│   ├── config.env            所有參數；`TB_LOOP_DIR` / `TB_HARNESS_ROOT` / `TB_REPO_ROOT` 三個錨點
│   ├── run_round.sh          主入口（評估 → 學習 → 對比）
│   ├── gen_sft.sh            用 codebuff agent 產 SFT 資料
│   ├── agents/               tb 的 installed agent（prime-agent / codebuff / loopmoe）
│   ├── learning/             失敗提取、/refine、歸因、回滾、對比
│   ├── harness/              ★ 跨輪學習狀態（會被注入容器）
│   ├── finetune/             MLX LoRA 微調與前後對比
│   ├── scripts/              環境準備與彩排
│   ├── results/  datasets/  sft_data*/   每輪輸出、資料集、產出的訓練資料
│   └── docs/                 whittle-moe-whitepaper.md
│
├── engine_loop/              ★ 引擎迴圈（把它自己的資產索引化，並投影成訓練資料）
│   ├── README.md             四階段定義、三種 record、已知誠實邊界
│   ├── MANIFEST.jsonl        ← 由 index_assets.py 產生（資產清單，73 筆）
│   ├── index_assets.py       產生 MANIFEST；`--check` 驗 bytes/mtime
│   ├── traces/               ★ 訓練資料的唯一出口（episode / decision / lesson）
│   └── memory/               專案記憶的段落級索引（INDEX.jsonl）＋ 產生器
│
├── memory/                   記憶的 dated 快照（非權威；權威在 .workbuddy/memory/）
├── skills/                   skill 的 dated 快照（非權威；權威在 ~/.workbuddy/skills/）
└── docs/                     PD 相關文件（見下方「未分類」）
```

---

## 紅線

**`scripts/check/*` 與 `scripts/run_server.sh` 不得複製進來。**
它們是生產腳本，pre-commit 閘門與白皮書引用的是 `scripts/` 那一份
（`knifeedge_matrix.py` 181 KB、`decode_sweep.py` 56 KB、`m123_oracle_gate.py` 37 KB）。
複製一份就等於製造兩份真相，而且會分叉——一個 bug 要修兩處，總有一處被漏。
本目錄只放**指向它們的呼叫**（`engine_loop/index_assets.py` 記錄權威路徑）與**它們產出的 record 的投影**。

同一條規則的推論：**記憶與 skill 不放副本，只放索引**（`CONVENTIONS.md` D6）。
`memory/` 與 `skills/` 底下的實體檔是**非權威的 dated 快照**，唯一用途是跨機器搬運
（`agent_harness/scripts/auto_git_push.ps1` 週期性 `git add agent_harness` 後 push，
運送的是內容而不是指標）。

---

## 三種 record

唯一的訓練資料出口是 `engine_loop/traces/*.jsonl`，一行一 record，只有三種類型。
權威定義在 `engine_loop/traces/schema/*.schema.json`，說明在 `engine_loop/README.md` §3。

| 類型 | 是什麼 | 為什麼值錢 |
|---|---|---|
| `episode` | 一次 run、一個臂：build 指紋 ＋ 量測 ＋ artifact | 觀測本身，**不含任何詮釋** |
| `decision` | 一個判斷點：問題 ＋ 證據 ＋ 推理 ＋ **排除掉什麼** | 最高價值。`ruled_out` 是負面知識 |
| `lesson` | 可重用的規訓（＝harness 狀態） | 換一個問題也適用，能注入回 agent |

---

## 怎麼跑

```sh
cd <repo>

# ── 引擎迴圈（純讀，不啟動引擎）─────────────────────────────
python3 agent_harness/engine_loop/index_assets.py                  # 盤點
python3 agent_harness/engine_loop/index_assets.py --check          # 漂移則 exit 1（要濾 OK:/error 兩類行）
python3 agent_harness/engine_loop/traces/emit_episodes.py --stats  # 只看盤點
python3 agent_harness/engine_loop/traces/validate.py               # 三種 record 一起驗
python3 agent_harness/engine_loop/traces/selftest.py               # 證明驗證器會擋錯（必須 10/10）
python3 agent_harness/engine_loop/memory/build_memory_index.py --check
python3 agent_harness/engine_loop/memory/build_memory_index.py --query <TERM> -n 8

# ── 任務迴圈（需要 Docker Desktop 與一個 OpenAI 兼容的模型 server）──
cd agent_harness/tb_loop
bash scripts/setup_env.sh          # 一次性：venv ＋ terminal-bench ＋ host 側 prime-agent
./run_round.sh                     # 從 round 1 跑到 TB_ROUNDS
```

**索引重生的順序不可顛倒**：先 `memory/build_memory_index.py`（寫 `INDEX.jsonl`），
再 `index_assets.py`（`MANIFEST.jsonl` 會記錄 `INDEX.jsonl` 的 bytes/mtime）。
顛倒的簽名是 `INDEX.jsonl` 的 **mtime** 漂移，不是 bytes 漂移。

---

## 三個路徑錨點（`tb_loop/config.env` 定義）

| 變數 | 指向 | 用途 |
|---|---|---|
| `TB_LOOP_DIR` | `agent_harness/tb_loop` | **屬於 tb_loop** 的東西：`harness/`、`results/`、`sft_data*/`、`.venv/` |
| `TB_HARNESS_ROOT` | `agent_harness` | **不隨 tb_loop 移動**的資產：`loopmoe/`、`loopmoe_output/`、`pd_data/` |
| `TB_REPO_ROOT` | 倉庫根 | 跨過 harness 根的東西：`app/cloud/freebuff2api/.env` |

三個都要顯式宣告的理由是同一個：**一個路徑若「剛好」等於另一個，它就會在下一次搬動時靜默指錯。**
（E1 之前 `run_round.sh` 寫 `PYTHONPATH="$TB_LOOP_DIR"` 而 `TB_LOOP_DIR` 當時就是 `agent_harness/`，
所以那一行同時是對的又是錯的——它只是還沒被移動過。）

---

## 分類（2026-09-17：從「誠實清單」變成「決定」）

`PLAN_ENGINE_LOOP_2026-09-15.md` §3 的目標結構只安排了 `tb_loop/`、`engine_loop/`、`shared/`。
以下條目不屬於任何一個迴圈。E1 當時把它們記成「未分類誠實清單」——**那個措辭是對的**
（計畫沒說不等於我可以自己發明一個家），但它留了一個缺口：**沒有說它們屬於誰**。

2026-09-17 補上歸屬。判準是**消費者**（誰 import／呼叫它），不是目錄名：

| 條目 | 它是什麼 | **歸屬** | 依據（實測） |
|---|---|---|---|
| `qwen36/` | Qwen3.6-35B-A3B 推論整合層（1 檔、3 個 def/class） | **tb_loop 的依賴** | 它自己的 docstring 列出三個消費者，全部在 `tb_loop/`：`agents/loopmoe_agent_adapter.py`、`finetune/finetune_loopmoe.sh`、SFT 管線 |
| `loopmoe/` | Loop MoE 模型與訓練套件（22 檔） | **tb_loop 的依賴** | `tb_loop/finetune/finetune_loopmoe.sh` import `loopmoe.{config,models.loop_moe_model,training.weight_graft}` |
| `dsh_config/` | DSH（cordis）設定 ＋ trajectory→SFT 轉換（4 檔） | **tb_loop 的 SFT 輸入** | 它的 README 自述「整合進 agent_harness…提升 Terminal-Bench 表現」；產物是 `tb_loop/sft_data*/` 的上游 |
| `pd/` | Prefill-Decode 端雲分離（35 檔） | **引擎側的推論子系統**（不屬兩條迴路） | `qwen36/__init__.py` 把 `CGC edge_server (… + DOPD)` 列為推論後端之一 |
| `docs/` | PD 的文件（3 檔） | 同 `pd/` | 其中 2 份原本是 `pd/` 的逐位元組副本，**已於 2026-09-17 改成指標** |
| `scripts/auto_git_push.ps1` | 定時推送（`git add agent_harness` 後 push） | **跨機器搬運線** | **刻意不搬**：路徑被 `CONVENTIONS.md` D6、`engine_loop/memory/README.md`、`build_memory_index.py` 的 docstring 與 lesson `eng-bound-0004` 的 `applies_to` 引用，而其中兩份是不可手改的快照、一份是已定稿 record。搬它換不到任何東西，只會製造引用漂移 |
| `cgc_proxy.js`、`cgc_proxy.py`、`cgc_anthropic_proxy.py`、`openai_cgc_bridge.js`、`openai_cgc_bridge.py`、`litellm_config.yaml`、`_gen.js` | CGC edge proxy / bridge | **基礎設施**（不是迴路也不是資料） | 沒有 harness 側的消費者；它們服務的是對外的 API 相容層 |

**每個目錄都有自己的 README**，說明「它不是什麼」——因為誤解發生在**讀者落在那個目錄裡**的時候，
而不是在他讀到這一頁的時候。

### 重複的文件：已處理

`docs/CGC_PD_Whitepaper.md` 與 `pd/CGC_PD_Whitepaper.md` 原本位元組完全相同
（`md5 e1a34fac72c958ef2d688b5184a4a943`），`docs/DOPD_UPGRADE_PLAN.md` 與
`pd/DOPD_UPGRADE_PLAN.md` 亦然（`md5 cc6292251e904be9bfc395a2f87136c6`）。

2026-09-17 處置：`docs/` 底下那兩份**換成指標**（不是刪除）。
不刪的理由是 `docs/AGENT_HARNESS_E1_RESTRUCTURE_20260916.html` 引用過那個路徑，
刪掉會讓一份已定稿的文件說假話 —— 本 repo 對已發表文件的立場是**標註而不是改寫**。
指標讓路徑仍然解析得到，而內容只有一份。`PD_INTEGRATION_DELIVERY.md` 沒有重複，留在原處。

### 一句話說明為什麼不搬家

進來的**路徑**引用幾乎是零（`pd/` 1、`loopmoe/` 1、`qwen36/` **0**、`dsh_config/` 1），
所以搬的成本很低。但 repo 根已經有 17 個一級目錄，再開一個給 62 個檔不會讓任何人更容易找到它。
**沒有更好的家時，搬家的唯一效果是把問題換一個地址。** 所以處置是標明歸屬，而不是搬家。
真要搬，該由各支的 owner 決定去處。

⚠️ 引用判準要用**完整路徑前綴**：裸字串會嚴重誤導 —— `qwen36` 有 83 處命中，但絕大多數指的是
**模型**（`Qwen3.6-35B-A3B` 的別名）而不是這個套件；`loopmoe` 的 14 處多半是 `LOOPMOE_*` 變數名。

---

## 這份目錄刻意不自足

`index_assets.py` 是 `REPO = HERE/../..`、`build_memory_index.py` 是 `HERE/../../..`、
`emit_episodes.py` 用 `find_repo()` 往上找 `.git`——**它索引的東西全在它外面**
（`scripts/check/*`、`docs/*`、`Backup/cgc_logs`、`.workbuddy/memory/*`）。

⇒ **不要**把它拆成獨立 git repo，也不要巢狀 `.git` 或 submodule：巢狀 `.git` 會讓
`find_repo()` 停在那一層，從此看不到 `Backup/`。要一條乾淨的同步線就開一條只提交
`agent_harness/` 的 subtree 分支。

---

## 現況（E 階段）

| 階段 | 內容 | 狀態 |
|---|---|---|
| **E0** | 索引 ＋ 第一版 episode 匯出 | ✅ 122 筆 episode；資產數看 `index_assets.py --check` |
| **E1** | 結構 ＋ 憲章：資產搬進 `tb_loop/`、修 import 路徑、寫 `CONVENTIONS.md` | ✅ 三條驗收 **2 條已結清、第 3 條只差 Docker**：套件解析**實測通過**（見下方 2026-09-17 更正 —— 要用 `tb_loop/.venv` 的直譯器）、`CONVENTIONS.md` 61/61 條有指針；**`tb run --n-tasks 1` 的 smoke 未跑**，缺的是 Docker daemon（colima 已停）與 `.venv`（已存在）——不是缺模型端點 |
| **E2** | T1 蒸餾 ＋ 兩個投影（`sft_pi/`、`sft_prime/`、`harness_engine/`）＋ 結清 D6 的閉環欠帳 | ✅ 四個目錄已建、兩份投影可用 `--check` 驗（見 `docs/AGENT_HARNESS_E2_TRAINING_PROJECTIONS_20260916.html`）；**T1 未接真模型；D6 的閉環欠帳未結清** |
| **E3** | 閉環：round1（無 lesson）vs round2（注入 lesson） | ⏳ 執行器已建並離線驗證（`distill/closed_loop.py`，41 項自測；見 `docs/AGENT_HARNESS_E3_DESIGN_20260916.html`）；**模型半未跑 ⇒ 本體未結清** |
| **E4** | 治理（可選）：log 政策、`knifeedge_matrix.py` 拆分、`scripts/check/` 分層、標記 stale | ✅ 四項都到了終態，見下方 2026-09-17 的註記；每一項的證據一支指令可重跑 |

**★ 2026-09-17 更正：上面那格 E4 曾經說「`pack_evidence.py` 根本不存在」—— 那句話是假的。**
`shared/pack_evidence.py`、`sanitize.py`、`trace_schema.md` 在 `fdfc923ef` 就入庫了，
§8 的 2446 個內容尋址 evidence pack（12.79 MB）在 `becb20a0e` 入庫，而 **item 1 與 item 3 也早已落地**
（item 3 以「能力分層 ＋ `classify.py` 資料化」取代 `git mv`，因為 `scripts/check/*` 有 562 條引用、
其中 134 條來自已定稿的白皮書，搬檔會讓 175 條懸空）。這與下面那段是**同一個病**：
**閘門全綠而散文是假的** —— `index_assets.py --check` 當時報 102 筆資產全同意，
因為它管的是 MANIFEST 裡的資產，**不含本表的敘述列**。綠燈與「沒被檢查」在那裡長得一樣。

**E4 四項的終態與證據（2026-09-17）**

| # | 內容 | 終態 | 怎麼自己驗一次 |
|---|---|---|---|
| 1 | log 政策落地 | `shared/log_policy.json` 是單一權威（納管／不納管／上限／既有例外） | `python3 agent_harness/shared/check_log_policy.py --check --privacy` |
| 2 | `knifeedge_matrix.py` 拆分 | 拆成 `scripts/check/knifeedge/` 11 個模組 ＋ 薄 shim（3310 行 → 最大 571 行）；五層等價證明 | `python3 agent_harness/shared/check_module_split.py --self-test`（另四支見 map 的 `verification`） |
| 3 | `scripts/check/` 分層 | 以**資料**表達（`wrappers/classify.py` ＋ `classes.tsv`，46 個腳本 6 類），不搬檔 | `python3 agent_harness/engine_loop/wrappers/classify.py --check` |
| 4 | 標記 stale | `shared/stale_registry.json` 是單一權威；banner 跟著資料（JSON 的 `_stale`）＋ 在消費點由 wrapper 印出 | `python3 agent_harness/shared/check_stale.py --check`；`bash agent_harness/engine_loop/wrappers/gate.sh --dry-run replay_bench_compare.py` |

誠實邊界：item 1 的檢查**也是**第一次把政策變成斷言，而它一啟動就抓到 `Backup/cgc_logs` 底下有
**12 個 `.log` 原文在版控裡**（5.6 MB，而 `docs/` 與 `*.md` 對它們的引用數是 0）。那不是這條線的檔案，
所以本輪**不移出**，而是登記在 `log_policy.json` 的 `grandfathered`（每一筆印出來，多一筆就紅）。
**登記 ≠ 豁免**：它現在是一個看得見的例外，而不是一個沉默的違規。

**這張表在 2026-09-16 晚間到 09-17 上午是錯的**：它寫 E2「⏳ 未開始」，而 E2 早已完成。
成因是 E2 只更新了 `engine_loop/README.md` 與 `PLAN` §9，**漏了這個傘狀入口**。
一份說「未開始」的狀態表比一個過期的數字更糟——它會讓人不去讀已經存在的東西。

## ★ 2026-09-17 更正：E1「只結清一半」這句話本身是錯的，而且 E1 有一支回歸沒修

有人問「E1–E3 代碼都完成了、只剩下量測嗎」。逐條實測之後，答案是**不是**，而且其中一半不是量測：

**(a) E1 的第一條驗收其實已經通過 —— 只是當初用錯了直譯器。**
`PLAN` §9 與本表長期寫著「`import tb_loop.agents.prime_agent_adapter` 止於
`No module named 'terminal_bench'`」。那句話是用**系統 python3** 跑出來的；而 `tb_loop` 有
自己的 venv（`agent_harness/tb_loop/.venv`，`config.env` 的 `TB_VENV_PY` 指的就是它），
**terminal-bench 裝在那裡面**。實測：

```
$ PYTHONPATH=agent_harness agent_harness/tb_loop/.venv/bin/python \
    -c "import tb_loop.agents.prime_agent_adapter"
-> OK（解析到 agent_harness/tb_loop/agents/prime_agent_adapter.py）
```

**驗收條件是「套件解析成功」，不是「系統 python 也能解析」。** 用對直譯器之後它是綠的 ——
而「用錯直譯器量出來的紅」與「用錯閘門量出來的綠」是同一種錯（量具指錯對象）。

**(b) E1 有一支回歸沒修：`tb_loop/scripts/local_rehearsal.py` 連 `--help` 都跑不起來。**
它第 42 行的註解在 E1 時就改成了 `tb_loop.agents.…`，但**第 48 行的 import 還是搬遷前的
`from agent_harness.agents.codebuff_api_agent import CodebuffApiAgent`** —— 而上面那兩行塞進
`sys.path` 的是 `tb_loop/` 與它的父目錄（`agent_harness/`），`agent_harness` 本身不在路徑上
（它是 namespace package，只有 repo 根在路徑上時才解析得到）。於是
`ModuleNotFoundError: No module named 'agent_harness'`。
同一輪還掃到第二處同類：`agents/loopmoe_agent_adapter.py` 的 docstring 用法範例仍寫
`--agent-import-path agent_harness.agents.loopmoe_agent_adapter:…`（使用者照抄就會踩同一個錯）。
**兩處都已修**（`tb_loop.agents.…`），並全掃過 `tb_loop`：0 處殘留、135 支 `.py` 全部可編譯、
所有有 `--help` 的 entry 都 rc=0。

★ 這與 `eng-bound-0007`（E1 的 shell 錨點回歸）是**同一個形狀**：口徑（註解、README）改了，
程式沒改。而 E1 當時的掃描只看了 **shell 的路徑派生**，沒有掃 **Python 的 import 字串** ——
所以「掃描涵蓋了什麼」與「掃描漏了什麼」是兩個不同的問題，而後者不會自己說出來。

**(c) 第 3 條驗收（`tb run --n-tasks 1` smoke）缺的是 Docker，不是模型端點。**
`.venv` 與 `terminal-bench` 都在；`colima` 已停（使用者要求），所以 Docker daemon 不在。
這是一個**環境前置**，不是量測 —— 起 colima 之後就能跑。
（另註：`local_rehearsal.py` 的彩排路徑以 Windows Git Bash 為設計對象，本機的等價物是
`scripts/run_wsl2.sh`；`results/` 底下 9/3 的 rehearsal_1/2/3 就是那條線的產物。）

**(d) E2/E3 剩下的確實是量測，但要分清哪一種。**
T1 蒸餾（`refine_engine.sh`）、D6 欠帳（`closed_loop.py` 的 A vs B）、E3 本體（C vs D）
都需要一個**真的會回答的模型**；`llm_client.py` 的預設是 `127.0.0.1:8080`、
`litellm_config.yaml` 指 `127.0.0.1:8083`，兩者都是本機 server ⇒ 需要窗。
但 `LLM_BASE_URL` 是可換的：**T1 的蒸餾在語意上可以指向遠端端點**（它讀的是 harness 的紀錄），
而 **D6/E3 的閉環不行** —— 那一組實驗是圍繞本機模型的 ChatML／`<think>` scaffold 設計的，
換模型等於換一個實驗。所以「等窗」這句話對 D6/E3 成立，對 T1 不完全成立。

改動本目錄的任何敘述之前，先讀 `CONVENTIONS.md`——**它同時是 `engine_loop/sft_pi/` 的 system prompt**，
所以改它等於改動兩個迴圈未來的行為。
