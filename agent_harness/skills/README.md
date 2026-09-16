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

## ★ 這個清單是**手動維護**的，而維護它的那一行不受版控

`Backup/import_harness_snapshot.py` 用一個硬編碼的 `SKILL_NAMES` 清單決定要匯入哪些 skill，
而**那個檔案住在 `Backup/`，被 `.gitignore:396` 排除**。兩件事合起來的後果：

- **新增一個 skill 卻忘了加進 `SKILL_NAMES`，它的快照不會被建立**——而且失敗方式是安靜的：
  不是報錯，是「它不在 `SNAPSHOT.jsonl` 裡」，而這與「這個 skill 不存在」長得一模一樣。
- 就算當下把 `SKILL_NAMES` 改對了，**那個修正本身不會被提交**⇒ 換一台機器 clone 之後，
  那個 skill 又會在下一次匯入時被漏掉。

所以：**這裡的清單才是權威的待匯入清單**（它有版控）；`SKILL_NAMES` 只是它的實作。
新增 skill 時要**兩邊都改**，並且在 commit message 裡明寫「`SKILL_NAMES` 的修正只存在於本機」
——`Backup/` 底下修好的東西不算交付（同一條規則的例子：`run_req2_retest.sh`、
`run_lib_ab.sh` 也都只存在於本機）。

- 2026-09-16（E1）：加入第四筆 `cgc-whitepaper-delivery`。它的快照
  `agent_harness/skills/cgc-whitepaper-delivery/SKILL.md` 是本目錄第一個**新增**的檔案，
  也是「清單必須兩邊都改」這件事的第一個實例。

## 為什麼要放實體檔

跟 `../memory/` 同一個理由：`~/.workbuddy/skills/` 在專案外，不在版控裡，也不在
`auto_git_push.ps1` 的 `git add agent_harness` 範圍內。skill 是這個專案累積出來的方法論
（判準、陷阱、可複製的指令序列），值得跟著 repo 走。

一般化的形狀見 lesson `eng-bound-0004`：**一個只發行指標（path + hash）的索引，
在「把內容送到另一台機器」這個用途下是無效的**——可驗證性與充分性是兩條獨立的軸。

## 同步義務（明確接受）

快照**不會自動更新**。要更新就重跑：

```sh
python3 Backup/import_harness_snapshot.py
```

`SNAPSHOT.jsonl` 逐檔記錄來源路徑、`source_sha256`、`source_bytes`、`source_mtime` 與
`snapshot_date`。**要改 skill 請改原檔**（`~/.workbuddy/skills/...`），不要改這裡的副本——
改了副本下次匯入就被蓋掉。
