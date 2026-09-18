# agent_harness/skills — skill 的**快照**（非權威）

**權威位置是 `~/.workbuddy/skills/<name>/SKILL.md`**，本目錄只是那份內容在某個日期的實體複本。

| skill | 權威位置 | 什麼時候用 |
| --- | --- | --- |
| `cgc-commit-gate` | `~/.workbuddy/skills/cgc-commit-gate/SKILL.md` | 在本 repo commit 時通過 pre-commit 閘門與 D5 |
| `cgc-decision-registry` | `~/.workbuddy/skills/cgc-decision-registry/SKILL.md` | 把日誌的 §EN-／§TB- 決策節登錄成 `traces/decisions.jsonl`（schema、id 格式、改完要重生哪兩個投影） |
| `cgc-decode-attribution` | `~/.workbuddy/skills/cgc-decode-attribution/SKILL.md` | 把 decode 步歸因到 GPU／CPU／池 IO／編碼 |
| `cgc-observer-portal` | `~/.workbuddy/skills/cgc-observer-portal/SKILL.md` | 新增「可機檢的觀測入口」（目標／決策／閘門／資產四合一）時：四條硬規矩、零漂移的位置、突變式自測清單 |
| `cgc-prefill-thermal-delivery` | `~/.workbuddy/skills/cgc-prefill-thermal-delivery/SKILL.md` | 讓一個 prefill t/s 數字取得可交付地位 |
| `cgc-tb-smoke` | `~/.workbuddy/skills/cgc-tb-smoke/SKILL.md` | 跑 terminal-bench 的鏈路 smoke（oracle／帶模型兩變體） |
| `cgc-whitepaper-delivery` | `~/.workbuddy/skills/cgc-whitepaper-delivery/SKILL.md` | 寫 `docs/*.html` 技術白皮書（版式、取材清單、現場閘門輸出、收尾雙 commit） |

YAML frontmatter（`name` / `description` / `agent_created`）**完整保留在最前面**，
banner 插在它之後——所以這些副本仍然可以被 skill loader 解析。

**這張表受檢查約束**：`../scripts/import_harness_snapshot.py --check` 會把它的成員與實際探索到的
集合逐項比對（`存在但未宣告`／`宣告了但不存在` 都算失敗，**拿掉表格本身也算失敗**）。
保留表格（而不是換成指向 `SNAPSHOT.jsonl` 的一行）是因為第三欄「什麼時候用」是**不可推導**的內容；
它腐爛過一次（`cgc-tb-smoke` 在 2026-09-17 建立之後沒有被加進來），所以現在由檢查釘住。

## 待匯入清單是**推導**出來的，不是維護出來的（2026-09-16 修）

上表的**成員**不需要靠人維護——匯入器會 glob
`~/.workbuddy/skills/*/SKILL.md`，所以**新增一個 skill 不需要改匯入器的任何清單**
（沒有 `SKILL.md` 的目錄會被排除）。每天新長出來的 memory 檔同理
（glob `.workbuddy/memory/*.md`）。上表只需要跟上（而檢查會在你忘記時出聲）。

這一段值得留下來，因為**它前一版的敘述是錯的，而那個錯本身很有代表性**：

- 舊版的匯入器住在 `Backup/import_harness_snapshot.py`，被 `.gitignore:396` 排除，
  而且用兩個**硬編碼清單**（`SKILL_NAMES` 決定匯入哪些 skill、`MEM_FILES` 決定匯入哪些記憶檔）。
- 失敗方式是**安靜的**：不是報錯，是「它不在 `SNAPSHOT.jsonl` 裡」——
  這與「那個檔案不存在」長得一模一樣。所以新增 skill、或**單純過了一天**，都會被漏掉。
- 更關鍵的是：即使把名字補進清單，**那個修正本身也不會被提交**（`Backup/` 不在版控裡）
  ⇒ clone 到另一台機器之後，**連匯入器都不存在**，不只是漏一個名字。

第一個版本（同日稍早）把結論寫成「把有版控的 README 當成權威清單、兩邊都改」。
那只是把**症狀**釘住：它要求一個人在兩個地方手動同步，而這個 repo 的立場一向是
「**能推導的就不要維護**」（`index_assets.py` 對每日日誌就是這個選擇：
"so a new day is not invisible until somebody remembers to add a row"）。
真正的修法是兩件事一起：**把腳本搬進版控**（`../scripts/`，與
`auto_git_push.ps1` 同一個目錄——那整條線就是「跨機器搬運」），
**並把四個寫死的東西換成推導**（兩個清單 → glob、`SNAP_DATE` → `date.today()`、
`REPO` 的絕對路徑 → `Path(__file__).parents[2]`）。

推導之後，「被漏掉」這個模態消失了，取而代之的是**匯入器先印出它發現了什麼**
（下面是形狀示意，不是當下的數字——數字會變，別引用它）：

```
discovered 8 memory file(s), 5 skill(s):
  mem    MEMORY.md
  mem    MEMORY_FACTS.md
  ...
  skill  cgc-commit-gate
  skill  cgc-tb-smoke
  ...
```

**空清單會拒跑而不是寫出一份空索引**——空的 `SNAPSHOT.jsonl` 與「探索步驟壞了」在外表上同形，
這是本專案反覆出現的那一族（B7／B12）。

## 為什麼要放實體檔

跟 `../memory/` 同一個理由：`~/.workbuddy/skills/` 在專案外，不在版控裡，也不在
`auto_git_push.ps1` 的 `git add agent_harness` 範圍內。skill 是這個專案累積出來的方法論
（判準、陷阱、可複製的指令序列），值得跟著 repo 走。

一般化的形狀見 lesson `eng-bound-0004`：**一個只發行指標（path + hash）的索引，
在「把內容送到另一台機器」這個用途下是無效的**——可驗證性與充分性是兩條獨立的軸。
匯入器搬進版控是同一條規則的另一半：**一個要跨機器的機制，自己得先能跨機器。**

## 同步義務（明確接受）＋ 一個**會發聲的**檢查

快照**不會自動更新**。要更新就重跑（順序固定）：

```sh
python3 agent_harness/scripts/import_harness_snapshot.py            # ① 刷新快照 + SNAPSHOT.jsonl
python3 agent_harness/engine_loop/memory/build_memory_index.py      # ② 先：寫 INDEX.jsonl
python3 agent_harness/engine_loop/index_assets.py                   # ③ 後：MANIFEST 記 INDEX 的 bytes/mtime
```

```sh
python3 agent_harness/scripts/import_harness_snapshot.py --check      # 0 = 一致；1 = 有漂移
python3 agent_harness/scripts/import_harness_snapshot.py --self-test  # 黑箱自測 14/14
```

`SNAPSHOT.jsonl` 逐檔記錄來源路徑、`source_sha256`、`source_bytes`、`source_mtime` 與
`snapshot_date`。**要改 skill 請改原檔**（`~/.workbuddy/skills/...`），不要改這裡的副本——
改了副本下次匯入就被蓋掉。**而且 `--check` 會先把它標成 `[hand-edited]`**（2026-09-18 起）：
在那之前，手改副本是**完全不可觀測**的（兩個既有的 `--check` 都只驗原檔↔索引）。

檢查為什麼**不**接進 pre-commit hook、以及五種漂移的處置表，見 `../memory/README.md` 的同一節
（判決沿用 `CONVENTIONS.md:927-928`：每日日誌當天必變，永遠紅的閘門等於沒有閘門）。
