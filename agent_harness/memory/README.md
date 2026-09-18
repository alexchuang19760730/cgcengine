# agent_harness/memory — 專案記憶的**快照**（非權威）

**權威位置是 `.workbuddy/memory/`**，本目錄只是那份內容在某個日期的實體複本。

| 檔案 | 權威位置 | 主題 |
| --- | --- | --- |
| `MEMORY.md` | `.workbuddy/memory/MEMORY.md` | 索引入口：專案長期筆記的分類與入口 |
| `MEMORY_FACTS.md` | `.workbuddy/memory/MEMORY_FACTS.md` | 長期事實／陷阱／周邊介面 |
| `MEMORY_PERF.md` | `.workbuddy/memory/MEMORY_PERF.md` | profile／模型／幾何／速度／散熱 |
| `MEMORY_S1.md` | `.workbuddy/memory/MEMORY_S1.md` | S1／分歧定位（S1＝把 slot table 放 GPU 的探針臂） |
| `2026-09-15.md` | `.workbuddy/memory/2026-09-15.md` | 當日工作誌（append-only） |
| `2026-09-16.md` | `.workbuddy/memory/2026-09-16.md` | 當日工作誌（append-only） |
| `2026-09-17.md` | `.workbuddy/memory/2026-09-17.md` | 當日工作誌（append-only） |
| `2026-09-18.md` | `.workbuddy/memory/2026-09-18.md` | 當日工作誌（append-only） |

每個快照檔開頭都有 banner 標明這件事（插在 YAML frontmatter **之後**，避免弄壞 frontmatter）。

**這張表受檢查約束**：`import_harness_snapshot.py --check` 會把它的成員與實際探索到的集合逐項比對
（`存在但未宣告`／`宣告了但不存在` 都算失敗，**拿掉表格本身也算失敗**）。
理由不是強迫維護文件，而是：這張表在 2026-09-18 之前是**腐爛的**（見下方「修正紀錄」），
而**沒有任何東西會出聲**。要刪掉這張表就要同時改那個檢查的判準，那是一個顯式的決定，不是靜默的繞道。

## 為什麼要放實體檔

`.workbuddy/` 在 `.gitignore` 裡，所以記憶**從來沒有進過版控**。而
`agent_harness/scripts/auto_git_push.ps1`（定時推送）只 `git add agent_harness`——
記憶不在那棵樹下，就永遠不會被推到 `cgc0907/fusionroutemot`。

索引方案（`../engine_loop/memory/INDEX.jsonl`）推的是**指標，不是內容**：另一台機器拿到索引
也讀不到本文。要有內容就得有實體檔，所以這裡有。

## 同步義務（明確接受）＋ 一個**會發聲的**檢查

快照**不會自動更新**。要更新就重跑（順序固定）：

```sh
python3 agent_harness/scripts/import_harness_snapshot.py            # ① 刷新快照 + SNAPSHOT.jsonl
python3 agent_harness/engine_loop/memory/build_memory_index.py      # ② 先：寫 INDEX.jsonl
python3 agent_harness/engine_loop/index_assets.py                   # ③ 後：MANIFEST 記 INDEX 的 bytes/mtime
```

**檢查漂移（不寫任何檔）**：

```sh
python3 agent_harness/scripts/import_harness_snapshot.py --check      # 0 = 一致；1 = 有漂移，逐項印出
python3 agent_harness/scripts/import_harness_snapshot.py --dry-run    # 只顯示「會匯入什麼」
python3 agent_harness/scripts/import_harness_snapshot.py --self-test  # 黑箱自測 14/14（含陰性對照）
```

`--check` 分開五種漂移，因為**處置不同**：

| 標記 | 意思 | 處置 |
| --- | --- | --- |
| `[stale]` | 原檔的 sha256/bytes 與記錄不符 | 正常：重跑匯入 |
| `[new]` | 探索到原檔但沒有記錄（新的一天／新 skill） | 重跑匯入 |
| `[removed]` | 有記錄但已不是輸入（skill 被刪／改名） | 重跑匯入 |
| `[hand-edited]` | 副本與「用它自己記錄的 `snapshot_date` 重算」不符 | **違規**：有人手改了副本。先看過再匯入（匯入會蓋掉它） |
| `[readme-table]` | 兩份 README 的成員表與實際集合不符（或表格不存在） | 修表格（或照上面的說明改檢查判準） |

## 檢查為什麼**不**接進 pre-commit hook（刻意的，不是遺漏）

`.workbuddy/memory/*.md` 包含**每日 append-only 日誌**，而那是**多個 session 共寫**的（本 repo 常態）。
任何 commit 前都可能有人在上一分鐘剛 append 一行 ⇒ 硬閘門會**永遠紅**。
`agent_harness/CONVENTIONS.md:927-928` 對同一個形狀已有判決：
「每日日誌當天必變，永遠紅的閘門等於沒有閘門」。

⇒ 這個檢查走**兩條會真的發聲的路**：
① 收尾序列（`~/.workbuddy/skills/cgc-commit-gate/SKILL.md` §6.3）；
② **定時自動化**（每日跑 `--check`；有漂移就刷新＋重生索引＋commit）。

## 修正紀錄（2026-09-18）

這一節留下來，因為**本檔案自己就是那個失效模態的樣本**——一份被設計成「防線」的文件，
在沒有任何東西會出聲的情況下漂了兩處：

1. **成員表宣告 3 個檔（實際 8 個）。** `MEMORY_FACTS.md`／`MEMORY_PERF.md`／`MEMORY_S1.md`
   與 `2026-09-17.md`／`2026-09-18.md` 都是在這張表寫下之後才出現的，而表沒跟上。
   它與舊版匯入器的硬編碼清單（`SKILL_NAMES`／`MEM_FILES`）是**同一個毛病換了地方**：
   指向推導集合的手維運清單。差別只在這次沒有「漏掉」的後果，而是**說假話**。
2. **「匯入腳本住在 `Backup/`」是錯的**（該腳本 2026-09-16 已搬進 `agent_harness/scripts/`）。
   這句話在腳本搬家的當天就死了，而它留在這裡 2 天。

修法不是「把它改對然後等它再爛一次」，而是**讓爛掉變成紅燈**（`--check` 的 `[readme-table]`）。

## 誰是權威

`../engine_loop/memory/README.md`（§為什麼是索引，不是副本）記了完整的決定與三個防線。
一句話：**要引用事實，讀 `.workbuddy/memory/`；要同步到別台機器，讀這裡。**
