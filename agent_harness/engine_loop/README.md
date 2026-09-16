# engine_loop — 引擎調校迴圈

`agent_harness/` 底下有**兩個**迴圈，形狀相同、獎勵不同：

| | `tb_loop/`（既有） | `engine_loop/`（本目錄） |
|---|---|---|
| 任務 | Terminal-Bench 題目 | 引擎的一個調校假設（「S1 應該 bit-identical」） |
| 環境 | Terminal-Bench 容器 | `prod25` / `prefill250` profile × 臂 |
| 觀測 | 測試 pass/fail | 吞吐、**bit-identical 閘門**、池計數不變量、logits oracle |
| 學習 | 失敗軌跡 → `/refine` → harness 狀態（文本） | 判準數字的方法論 → `CONVENTIONS.md` + `traces/lessons.jsonl` |

兩者的共同點是：**學習發生在 harness 的狀態（文本），不在模型權重**。這是 `tb_loop/README.md`
已驗證過的原則，本目錄直接沿用。

---

## 1. 紅線

**`scripts/check/*` 與 `scripts/run_server.sh` 不得複製進來。**

它們是生產腳本：pre-commit 閘門與技術白皮書引用的是 `scripts/` 那一份
（`knifeedge_matrix.py` 181 KB、`decode_sweep.py` 34 KB、`m123_oracle_gate.py` 31 KB）。
複製一份就等於製造兩份真相，而且會分叉——一個 bug 要修兩處，總有一處被漏。
本目錄只放**指向它們的呼叫**（`index_assets.py` 記錄權威路徑）與**它們產出的 record 的投影**。

## 2. 四階段

```
evaluate  →  extract  →  refine  →  compare
（跑一個臂）  （記錄觀測）  （蒸餾成決策/教訓）（對照前後）
```

| 階段 | 本目錄的產物 | 現況 |
|---|---|---|
| **evaluate** | `index_assets.py` 指到的臂與 profile，由 `scripts/check/decode_sweep.py` 執行 | 已存在（今天就在跑） |
| **extract** | `traces/episodes.jsonl`（T0，純腳本，**已完成**） | ✅ E0 |
| **refine** | `distill/refine_engine.sh` ＋ `distill/collect_evidence.py` ＋ `distill/prompt/refine_engine.md` → `distill/out/<ts>/*.candidate.jsonl`（人審後 `--accept`） | ✅ E2（機制已建並通過離線自測；**未接真實模型跑過**） |
| **compare** | `harness_engine/`（注入用的 harness 狀態）＋ `sft_pi/` `sft_prime/`（兩份投影）＋ `distill/closed_loop.py`（四臂閉環對照） | ✅ E2（投影已產出並可重生）；執行器與實驗設計 ✅（`closed_loop_selftest.py` 33 項全過），但**模型半仍未跑** ⇒ **E3 本體未結清**，見 §9 |

### 2.1 E2 新增的四個目錄

```
engine_loop/
├── harness_engine/            prime-agent 的第二份狀態（PLAN §6.1）
│   ├── extensions -> ../../tb_loop/harness/extensions   ← 符號連結，**不是複本**（R1）
│   ├── build_memories.py      lessons.jsonl -> memories/engine/<id>.md（可重生、有 --check）
│   └── memories/engine/       一 lesson 一檔（現 111）；第一行固定 `[engine] <rule>`（/refine --global 的 scope 標記）
├── sft_pi/                    工具軌跡投影（messages + tool_calls）
│   ├── build_sft_pi.py        episode 序列 + decision.action -> train/valid.jsonl
│   └── {train,valid}.jsonl
├── sft_prime/                 (證據→教訓) 與 (狀態→下一個動作)
│   ├── build_sft_prime.py
│   └── {train,valid}.jsonl
├── distill/                   T1 蒸餾（半自動）
│   ├── collect_evidence.py    可單獨跑：--stats / --next-ids / 證據區塊
│   ├── refine_engine.sh       --dry-run（離線可驗）/ 真跑 / --accept
│   ├── prompt/refine_engine.md
│   ├── selftest.py            假 prime-agent，驗 prompt 的手遞與 --accept 路徑
│   ├── closed_loop.py         四臂閉環對照（A/B = D6 欠帳、C/D = E3）
│   ├── closed_loop_selftest.py  33 項離線自測（含「儀器必須說得出『沒有差異』」的陰性對照）
│   └── closed_loop_questions.md  8 題 + 每題的承重點
└── sft_common.py              兩份投影共用的載入／渲染／切分（**只有一份渲染器**）
```

「能推導就不要維護」在這四個目錄上是貫徹的：`memories/engine/` 由 `lessons.jsonl` 產生、
兩份 SFT 由 `traces/*.jsonl` 產生、`extensions/` 是指向而不是複本。**手改衍生物會被下一次
重生蓋掉，而那次重生看起來完全正常。**

### 2.2 E3 的執行器怎麼用（`distill/closed_loop.py`）

```sh
python3 agent_harness/engine_loop/distill/closed_loop.py --dry-run    # 不需要模型
python3 agent_harness/engine_loop/distill/closed_loop_selftest.py     # 33 項離線自測

# 真跑。需要一個 OpenAI 兼容端點；「用哪個模型」是實驗的一部分，所以腳本不給預設值。
CLOSED_LOOP_MODEL_CMD='llm -m <model>' \
  python3 agent_harness/engine_loop/distill/closed_loop.py --memories-scope first:8
```

| 參數 | 意思 |
|---|---|
| `--memories-scope all`（預設） | 注入全部 live lesson |
| `--memories-scope first:8` | `lessons.jsonl` 的**前 8 筆**（PLAN §6.3 的「那 8 條」） |
| `--memories-scope class:mh` | 單一 class（縮寫或全名） |
| `--memories-scope none` | 空。**陰性對照**：C 與 D 的 prompt 必須位元組相同 |
| `--memories-scope ids:<path>` | 明確清單，一行一個 id，`#` 可註解 |
| `--reps N`（預設 3） | 每個（題, 臂）呼叫幾次。**1 次證明不了「可重現」**，而 §9 的驗收要求的是後者 |
| `--only Q3` | 只跑一題 |

產物在 `distill/closed_loop_out/<ts>/`：

- `answers.jsonl` —— 每次呼叫一筆
- `compare.json` —— 每題的**臂內一致性** ＋ 跨臂判斷
- `manifest.json` —— 注入了哪些 id、每個 rep 的臂順序、charter 與 memories 的 sha256。
  **model cmd 只記 sha256、不記原文**：manifest 會被 commit，而命令列可能帶著 API key。

**讀 `compare.json` 的先後是固定的**：先看 `comparable`，再看跨臂欄位。`comparable: false`
（某個臂自己前後不一致）的題目，其跨臂判斷一律無效。而**「不同」不等於「改善」**——
承重點在 `closed_loop_questions.md` 的表裡，要人工核對。

## 3. 三種 record

唯一的訓練數據出口是 `traces/*.jsonl`，一行一 record，只有三種類型。權威定義在
`traces/schema/*.schema.json`，說明在 `../CONVENTIONS.md` 與 `../PLAN_ENGINE_LOOP_2026-09-15.md` §4。

| 類型 | 是什麼 | 為什麼值錢 |
|---|---|---|
| `episode` | 一次 run、一個臂：build 指紋 + 量測 + artifact | 觀測本身。**不含任何詮釋** |
| `decision` | 一個判斷點：問題 + 證據 + 推理 + **排除掉什麼** | 最高價值。`ruled_out` 是負面知識，最常被重用 |
| `lesson` | 可重用的規訓（= harness 狀態） | 換一個問題也適用，能注入回 agent |

### 三條硬規則（`traces/validate.py` 強制）

1. **`build == null ⇒ usable_as_evidence == false`。** 不知道二進位檔的數字不能比——不是「比較不準」，
   是「不可比」。E0 匯出的 100 筆裡只有 13 筆有指紋，因為 `decode_sweep.py` 直到今天才開始記錄它。
2. **有生成（`decode_tps.median` 是數字）⇒ `answer_md5` 不得為空。** 否則「答案穩定」是空真。
3. **`judgement == refuted ⇒ superseded_by` 非空。** 被推翻的假設不得當正例訓練
   （`PLAN §10 R3`：雙緩衝 remap 那個假設就是這樣被擋下來的）。

驗證器的自我證明：`traces/selftest.py` 注入 8 種違規，8/8 必須被拒。**它現在還沒被證明會擋錯之前，
不算閘門。**

## 4. 用法

```bash
cd <repo>

# 盤點：哪些資產存在、角色是什麼、能不能重放（純讀，安全）
python3 agent_harness/engine_loop/index_assets.py
python3 agent_harness/engine_loop/index_assets.py --check     # 檢查 manifest 是否 drift

# 匯出 + 驗證（純讀檔，不啟動引擎）
python3 agent_harness/engine_loop/traces/emit_episodes.py --stats           # 只看盤點，不寫檔
python3 agent_harness/engine_loop/traces/emit_episodes.py                   # 寫 episodes.jsonl 並驗證
python3 agent_harness/engine_loop/traces/validate.py                        # 三種 record 一起驗
python3 agent_harness/engine_loop/traces/selftest.py                        # 證明驗證器會擋錯
```

`emit_episodes.py` **從不啟動引擎**。它只讀 `Backup/**` 已經寫好的 JSON 與 log。
一旦它開始跑引擎，它就變成第二份會分叉的 harness。

## 5. 資料源（E0 已接的）

| 來源 | 產生者 | record 種類 |
|---|---|---|
| `Backup/phase_decomp/*.json` | `decode_sweep.py` / `ab_interleave.py` | `episode` (sweep / interleave) |
| `Backup/llama_bench/matrix_*.json` | `llama_bench_matrix.py` | `episode` (bench) |
| `Backup/knifeedge_matrix/capinv_*.json` | `knifeedge_matrix.py` | `episode` (gate，**有 verdict**） |
| `Backup/m123_oracle_gate/cap_*.json` | `m123_oracle_gate.py` | `episode` (gate，僅 provenance) |
| `Backup/knifeedge_matrix/*.jsonl` | oracle dump | `episode` (oracle) |

**未接（刻意）**：`Backup/cgc_logs/*` 原文（463 MB / ~1400 檔，單檔 > 32 MiB 只記路徑與大小，
不複製）、`harness_engine/logs/`、`sessions/`。log 只被**摘要**成 episode 的 `obs.phase_ms` /
`obs.asserts`，原文留在原地。

## 6. 體量與去隱私

- `episodes.jsonl` 100 筆約 220 KB。log 原文不進 repo。
- 絕對路徑 → `$REPO` / `$HOME`；`192.168.x.x` → `$LAN_IP`；`sk-*` → `$API_KEY`。
  `run_server.sh` 會印含區網 IP 的「連線卡」，oracle 的 `.cap` 內嵌完整 argv（含絕對模型路徑）
  —— 兩者都必須被 `sanitize()` 擋掉，而這兩者都會進 record。

## 7. 已知的誠實邊界

- **E0 的 122 筆裡只有 27 筆 `usable_as_evidence == true`（22%）。** 這不是匯出失敗，是今天之前的資料
  沒有 build 指紋、`n_rounds` 不足 3、或臂本身是探針。門檻是刻意設在那裡的。
  注意這個欄位擋的是**引用數字**，不是「能不能被引用」：層梯二分最關鍵的那一列
  （全臂、跨輪不穩定）正是 `usable_as_evidence == false`，而 `decisions.jsonl` 照樣用它。
- **`tb_loop` 的三個入口腳本曾經跑不起來——E1 已修（套件解析層面）。**
  原本 `import tb_loop` → `ModuleNotFoundError`：三支入口都寫 `--agent-import-path "tb_loop.agents.…"`，
  而 `PYTHONPATH` 指向 `agent_harness/` 自己（那時它就是 `tb_loop`），所以解譯器看不到名為
  `tb_loop` 的套件。E1 把資產搬進 `agent_harness/tb_loop/`，並讓 `PYTHONPATH` 指向其**父目錄**，
  import 字串一行未改。**已驗證**：`tb_loop` 現在是**正規套件**（`__init__.py`），
  `tb_loop.agents.{prime_agent_adapter,codebuff_api_agent,loopmoe_agent_adapter}` 與
  `tb_loop.learning.attribution` 四個模組都解析到搬遷後的位置；對照組
  （`PYTHONPATH=agent_harness/tb_loop`）仍正確地報 `No module named 'tb_loop'`。
  **仍未驗的**：`tb run --n-tasks 1` 的 smoke 沒跑——`tb_loop/.venv` 不存在、`terminal-bench`
  未安裝，且 Docker daemon 未執行。**`bash -n` 抓不到套件解析問題**，這一條靠真的 `import` 才成立。
  詳見 `docs/AGENT_HARNESS_E1_RESTRUCTURE_20260916.html`。
- **S1 的 bit-identical 閘門尚未通過。** 值層面的解釋已全部排除，但今天的兩個決定性配對把問題**重新定型**：
  - 層梯二分：`min_il=20`（20 層、約 80 個多出節點）與基準**逐位元相同且穩定**；`min_il=18/14/10/6`
    **穩定地不同**；`min_il=2` 與全臂**跨輪不穩定**（3 個值）。
  - `p25-keepleaf`（全部 S1 節點都建、`mul_mat_id` 消費 host leaf，**分段**）：**逐位元相同且穩定**。
  - `p25-slotgpu-dbg`（**真臂**，沒有 leaf）：`EQUIV` 2067 次全部 `mismatch=0`，且 `EXPECT` vs
    `POST` 在 39 層 × 6 個 build 上每一行都一致。⇒ **真臂消費的 id 就是 host 想要的那一份。**
  - `p25-buildleaf`（照 `llama-context.h` 註解做）**跑不起來**：無消費者的 graph root 拿不到
    backend，`ggml-alloc.c:623` 斷言。⇒ 那條路要劃掉。
  - **`p25-slotgpu-noasync`（閘門修好後真非分段）= `{ff68c5a2}` 穩定**，而同一 sweep 的基準
    `p25-gputime-noasync` = `{dc055e63}` 三個量全同 ⇒ **dispatcher 對基準是數值中性的**（分段關閉的
    代價是 ~10×：6.95 → 0.72 t/s，這是 S2/S3 的獎金大小）。時序假說**被否決**，但失敗的**形狀**
    變了：分段下不穩定、非分段下穩定但錯 ⇒ 不確定性來自分段 dispatcher，底下有一個確定性缺陷。
  - **`p25-keepleaf-noasync` = `{8c870183}` 穩定 ≠ 基準** ⇒ **「節點是惰性的」只成立於分段
    dispatcher**。同一個控制臂、同一個節點集合、同樣消費 leaf，只換 dispatcher 就換了答案，
    而且**從第一個 token 就是另一種回答模式**（不是最後位元），池計數同時 +16%/+18%。
    ⇒ 這個調查的控制臂本身對受測變數敏感；§9.9 的「節點不是原因」要加註條件，
    「少一個 pinned 節點造成佈局差異」這條候選失去結構前提。
  - 剩下的唯一候選是「**消費的那一刻**那個 buffer 裡是什麼」。三個現有儀器都答不了
    （POST 是 sync 之後讀、`id_oob` 是 encode 期主機讀裝置 buffer、`EQUIV` 只比兩個主機實作），
    它們的共同病是「量的東西與宣稱量的東西不同一個」，見
    `docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md` §9.13/§9.14 與 `traces/decisions.jsonl`
    `dec-20260915-2136` / `-2138` / `-2158` / `-2202`。
- **建置指紋漏了 `libggml-base`（已修）**。`build_fingerprint()` 只雜湊三個檔案
  （`llama-server` + `libggml-metal` + `libllama`），而 `ggml-backend.cpp`——排程器與 `CGC_OA_ASYNC`
  閘門——編進 **`libggml-base`**。後果是 21:34（閘門未修）與 21:41（閘門已修）兩次跑帶**完全相同**的
  指紋，被標成可比較。現改為 glob `libggml*.dylib` / `libllama*.dylib`（8 鍵，鍵名不含版本號），
  `ab_interleave.py` 改為直接委派同一份實作。**v1（3 鍵）的歷史列在這一維上不可比**——這正是要揭露的。
  教訓：`eng-mh-0007`。
- **一個「被設了但值被忽略」的閘門已修**：`ggml/src/ggml-backend.cpp:1745` 原本用
  `getenv("CGC_OA_ASYNC") != nullptr` 選分段 dispatch，所以 `CGC_OA_ASYNC=0` 選到的仍是分段分支
  （而 `run_server.sh` 無條件設定它，連「不設」都做不到）。修成 value-aware 後，prod25 設 `"1"`，
  錨定 digest `dc055e63` 不變。
- **同上，閘門修好只保住 prod25，卻把另外四個 profile 靜默改道（已修）**。乾跑實測修正前
  `off` / `prefill250` / `qa-zh` / `longform-zh` 都是 `CGC_OA_ASYNC=0`——而在修正前那個 `0` 因為閘門
  讀存在而**從未生效**（一直是分段）；修正後它第一次真的生效，等於把這些 profile 從 6.95 降到
  0.72 t/s。`prefill250` 是 oracle 閘門與 bench matrix 的口徑，所以影響落在驗證鏈上。
  `run_server.sh:109` 的預設已回到 `1`（= 它們**實際**在做的事），顯式 `0` 仍可用。
  教訓：`eng-gate-0006`（**錨 digest 不變 ≠ 行為不變**）。

## 8. 專案記憶（`memory/`）

`.workbuddy/memory/` 是這個 repo 資訊密度最高的紀錄，也是最少被用的：當日日誌到 1000+ 行，
整檔讀取會靜默截斷，而截斷後的「沒查到」與「沒寫過」同形。

`memory/` **只放衍生物**，原檔留在 host 寫入的位置（複製等於製造第二份真相，見 `CONVENTIONS.md` D6
與它 **2026-09-16 的修訂**——那個修訂沒有放寬這一條，只是另外承認了一份非權威的 dated 快照
`agent_harness/memory/`，用途僅限跨機器搬運）：

```bash
python3 agent_harness/engine_loop/memory/build_memory_index.py            # 重建 INDEX.jsonl
python3 agent_harness/engine_loop/memory/build_memory_index.py --check    # drift 則 exit 1
python3 agent_harness/engine_loop/memory/build_memory_index.py --query mmid -n 8
```

`--query` 回報 `path:line_start-line_end` 與該節的 `###` 子標題，所以下一步是精準讀取
（`Read(offset=line_start)`）。record 的 `action` / `evidence` 要引 `path:line_start`，不要複述內容。

---

## 9. E2 的驗收與兩件**沒有結清**的事（2026-09-16）

### 9.1 已建並通過離線驗證

| 產物 | 驗證方式 | 結果 |
|---|---|---|
| `harness_engine/memories/engine/` | `build_memories.py --check` | 106 檔、無漂移；第一行都是 `[engine] <rule>` |
| `harness_engine/extensions` | 符號連結解析 ＋ `--check` 會擋「變成真目錄」 | 指向 `tb_loop/harness/extensions`（單一複本） |
| `sft_pi/` | 由 `traces/*.jsonl` 產生，seed 固定可 diff | 14 條軌跡 / 38 個 tool call；每筆帶 `_provenance.reconstructed` |
| `sft_prime/` | 同上 | 141 筆（105 evidence→lesson ＋ 36 state→next-arm）；排除 2 條 refuted、1 條 superseded |
| `distill/` | `distill/selftest.py`（假 prime-agent） | 27 項檢查全過 |
| `distill/closed_loop.py` | `--dry-run`（不需模型） | 四臂的 prompt 大小／sha256／逐對 diff |

`distill/selftest.py` 抓到一個**真的 prompt 設計缺陷**並留下回歸斷言：v1 的 prompt 把
「下一個可用 id」放在最末，而證據區先列了 106 個**已在使用**的 id ⇒ 從文件中第一個看到的
id 取用就會撞號（`validate.py` 報 `duplicate lesson_id`）。修法是把可用 id 用哨兵包起來
並明文禁止重用，**並且把「文件中第一個 id ≠ 可用 id」變成一條斷言**——否則這個缺陷會在下一次
改 prompt 時無聲復發。

### 9.2 **閉環對照的模型半沒有跑**（D6 的欠帳仍然開著）

`CONVENTIONS.md` §E 要求「每次改 `CONVENTIONS.md` 都要走一次閉環對照」，
而 D6 的 2026-09-16 修訂聲稱「改變不到任何執行時行為」。那句話**目前仍然沒有產物支撐**。

- **執行器已建**：`distill/closed_loop.py`，四臂（A_preD6 / B_postD6 / C_head / D_head_mem），
  只有 A→B 回答欠帳，另兩對是反歸因用的。`--dry-run` 已證實三對各自只隔離一件事
  （A→B ＋18/−0 行、B→C ＋18/−4 行、C→D 章程 0 行／memories ＋23,785 B）。
  **第一版把 `e688f346e^` 直接對上 HEAD，結果那一對同時量了 D6 與後來的 E1 D1 更新——
  是 `--dry-run` 的逐對 diff 把它抓出來的。**
- **模型半未跑**，原因不是「做不到」，而是兩個必須先解決的條件：
  1. 本機沒有可用的模型端點（1234/1235/8080/11434/8000/5000/9000 全部無回應，
     `~/.workbuddy/models.json` 是空的）。
  2. `llama-server` 與本 repo 的 13.6 GB 模型**都在**，所以技術上可以起一個。
     **但那會與另一個 session 正在進行的 prefill/decode 量測互相干擾**——
     那些量測對 GPU 時脈與記憶體狀態極敏感（見 A16 的 τ 與熱暫態），
     起一個 13.6 GB 的 server 去回答 32 個問題，代價可能是對方整輪的數字作廢。
     **「為了結清一條欠帳而弄壞另一條線的資料」不是結清。**
- **要跑它的指令**（在沒有併行量測的時間窗）：
  ```sh
  CLOSED_LOOP_MODEL_CMD='<讀 stdin 寫 stdout 的任何模型命令>' \
    python3 agent_harness/engine_loop/distill/closed_loop.py
  ```
  然後**人工核對** `closed_loop_questions.md` 的承重點表格——字串相同只是自動部分。
- **不得宣稱 D6 修訂已結清**（PLAN §9 的 E2 欄原文即如此要求）。
