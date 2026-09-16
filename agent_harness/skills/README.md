# agent_harness/skills — skill 的**快照**（非權威）

**權威位置是 `~/.workbuddy/skills/<name>/SKILL.md`**，本目錄只是那份內容在某個日期的實體複本。

| skill | 權威位置 | 什麼時候用 |
| --- | --- | --- |
| `cgc-commit-gate` | `~/.workbuddy/skills/cgc-commit-gate/SKILL.md` | 在本 repo commit 時通過 pre-commit 閘門與 D5 |
| `cgc-decode-attribution` | `~/.workbuddy/skills/cgc-decode-attribution/SKILL.md` | 把 decode 步歸因到 GPU／CPU／池 IO／編碼 |
| `cgc-prefill-thermal-delivery` | `~/.workbuddy/skills/cgc-prefill-thermal-delivery/SKILL.md` | 讓一個 prefill t/s 數字取得可交付地位 |

YAML frontmatter（`name` / `description` / `agent_created`）**完整保留在最前面**，
banner 插在它之後——所以這些副本仍然可以被 skill loader 解析。

## 為什麼要放實體檔

跟 `../memory/` 同一個理由：`~/.workbuddy/skills/` 在專案外，不在版控裡，也不在
`auto_git_push.ps1` 的 `git add agent_harness` 範圍內。skill 是這個專案累積出來的方法論
（判準、陷阱、可複製的指令序列），值得跟著 repo 走。

## 同步義務（明確接受）

快照**不會自動更新**。要更新就重跑：

```sh
python3 Backup/import_harness_snapshot.py
```

`SNAPSHOT.jsonl` 逐檔記錄來源路徑、`source_sha256`、`source_bytes`、`source_mtime` 與
`snapshot_date`。**要改 skill 請改原檔**（`~/.workbuddy/skills/...`），不要改這裡的副本——
改了副本下次匯入就被蓋掉。
