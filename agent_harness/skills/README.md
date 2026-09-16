# agent_harness/skills — skill 的**快照**（非權威）

**權威位置是 `~/.workbuddy/skills/<name>/SKILL.md`**，本目錄只是那份內容在某個日期的實體複本。

| skill | 權威位置 | 什麼時候用 |
| --- | --- | --- |
| `cgc-commit-gate` | `~/.workbuddy/skills/cgc-commit-gate/SKILL.md` | 在本 repo commit 時通過 pre-commit 閘門與 D5 |
| `cgc-decode-attribution` | `~/.workbuddy/skills/cgc-decode-attribution/SKILL.md` | 把 decode 步歸因到 GPU／CPU／池 IO／編碼 |
| `cgc-prefill-thermal-delivery` | `~/.workbuddy/skills/cgc-prefill-thermal-delivery/SKILL.md` | 讓一個 prefill t/s 數字取得可交付地位 |
| `cgc-whitepaper-delivery` | `~/.workbuddy/skills/cgc-whitepaper-delivery/SKILL.md` | 寫 `docs/*.html` 技術白皮書（版式、取材清單、現場閘門輸出、收尾雙 commit） |

YAML frontmatter（`name` / `description` / `agent_created`）**完整保留在最前面**，
banner 插在它之後——所以這些副本仍然可以被 skill loader 解析。

## 待匯入清單是**推導**出來的，不是維護出來的（2026-09-16 修）

上面那張表只是**這一刻的結果**，不是輸入。匯入器會 glob
`~/.workbuddy/skills/*/SKILL.md`，所以**新增一個 skill 不需要改任何清單**
（沒有 `SKILL.md` 的目錄會被排除）。每天新長出來的 memory 檔同理
（glob `.workbuddy/memory/*.md`）。

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
真正的修法是兩件事一起：**把腳本搬進版控**（`agent_harness/scripts/`，與
`auto_git_push.ps1` 同一個目錄——那整條線就是「跨機器搬運」），
**並把四個寫死的東西換成推導**（兩個清單 → glob、`SNAP_DATE` → `date.today()`、
`REPO` 的絕對路徑 → `Path(__file__).parents[2]`）。

推導之後，「被漏掉」這個模態消失了，取而代之的是**匯入器先印出它發現了什麼**：

```
discovered 3 memory file(s), 4 skill(s):
  mem    MEMORY.md
  mem    2026-09-15.md
  mem    2026-09-16.md
  skill  cgc-commit-gate
  skill  cgc-decode-attribution
  skill  cgc-prefill-thermal-delivery
  skill  cgc-whitepaper-delivery
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

## 同步義務（明確接受）

快照**不會自動更新**。要更新就重跑：

```sh
python3 agent_harness/scripts/import_harness_snapshot.py
```

（這支腳本以前叫 `Backup/import_harness_snapshot.py`。舊路徑已不存在——
`Backup/` 底下的東西進不了 commit。）

`SNAPSHOT.jsonl` 逐檔記錄來源路徑、`source_sha256`、`source_bytes`、`source_mtime` 與
`snapshot_date`。**要改 skill 請改原檔**（`~/.workbuddy/skills/...`），不要改這裡的副本——
改了副本下次匯入就被蓋掉。
