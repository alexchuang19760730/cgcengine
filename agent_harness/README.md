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

## 未分類（誠實清單）

`PLAN_ENGINE_LOOP_2026-09-15.md` §3 的目標結構只安排了 `tb_loop/`、`engine_loop/`、`shared/`。
以下條目**不屬於任何一個迴圈，也沒有被規劃安排位置**，因此 E1 刻意不動它們：

| 條目 | 它是什麼 |
|---|---|
| `pd/` ＋ `docs/CGC_PD_Whitepaper.md`、`docs/DOPD_UPGRADE_PLAN.md`、`docs/PD_INTEGRATION_DELIVERY.md` | Prefill-Decode 端雲分離（另有同名副本在 `pd/` 底下，見下） |
| `loopmoe/` | Loop MoE（RDT 循環 ＋ Gated DeltaNet） |
| `qwen36/` | Qwen3.6 相關 |
| `dsh_config/` | DSH（cordis）設定與 SFT 轉換 |
| `scripts/auto_git_push.ps1` | 定時推送（`git add agent_harness` 後 push）。**刻意不搬**：它的路徑被 `CONVENTIONS.md` D6、`engine_loop/memory/README.md`、`build_memory_index.py` 的 docstring 與 lesson `eng-bound-0004` 的 `applies_to` 引用，而其中兩份是**不可手改的快照**、一份是 `traces/` 裡的已定稿 record。搬它換不到任何東西，只會製造引用漂移。 |
| `cgc_proxy.js`、`cgc_proxy.py`、`cgc_anthropic_proxy.py`、`openai_cgc_bridge.js`、`openai_cgc_bridge.py`、`litellm_config.yaml`、`_gen.js` | CGC edge proxy / bridge（Anthropic 與 OpenAI 兼容轉接） |

**已知的重複（既有，未處理）**：`docs/CGC_PD_Whitepaper.md` 與 `pd/CGC_PD_Whitepaper.md`
位元組完全相同（`md5 e1a34fac72c958ef2d688b5184a4a943`），`docs/DOPD_UPGRADE_PLAN.md` 與
`pd/DOPD_UPGRADE_PLAN.md` 亦然（`md5 cc6292251e904be9bfc395a2f87136c6`）。這是本專案最厭惡的
「兩份真相」，但歸屬在 PD 那一支，不在本次 E1 的範圍內。

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
| **E0** | 索引 ＋ 第一版 episode 匯出 | ✅ 122 筆 episode、73 筆資產 |
| **E1** | 結構 ＋ 憲章：資產搬進 `tb_loop/`、修 import 路徑、寫 `CONVENTIONS.md` | ✅ 套件解析已通（見 `docs/AGENT_HARNESS_E1_RESTRUCTURE_20260916.html`）；`CONVENTIONS.md` 已成 |
| **E2** | T1 蒸餾 ＋ 兩個投影（`sft_pi/`、`sft_prime/`、`harness_engine/`）＋ 結清 D6 的閉環欠帳 | ⏳ 未開始 |
| **E3** | 閉環：round1（無 lesson）vs round2（注入 lesson） | ⏳ 待 E2 |
| **E4** | 治理（可選）：log 政策、`knifeedge_matrix.py` 拆分、`scripts/check/` 分層、標記 stale | ⏳ 可選 |

改動本目錄的任何敘述之前，先讀 `CONVENTIONS.md`——**它同時是 `engine_loop/sft_pi/` 的 system prompt**，
所以改它等於改動兩個迴圈未來的行為。
