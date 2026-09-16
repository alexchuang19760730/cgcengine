# agent_harness/memory — 專案記憶的**快照**（非權威）

**權威位置是 `.workbuddy/memory/`**，本目錄只是那份內容在某個日期的實體複本。

| 檔案 | 權威位置 |
| --- | --- |
| `MEMORY.md` | `.workbuddy/memory/MEMORY.md` |
| `2026-09-15.md` | `.workbuddy/memory/2026-09-15.md` |
| `2026-09-16.md` | `.workbuddy/memory/2026-09-16.md` |

每個快照檔開頭都有 banner 標明這件事（插在 YAML frontmatter **之後**，避免弄壞 frontmatter）。

## 為什麼要放實體檔

`.workbuddy/` 在 `.gitignore` 裡，所以記憶**從來沒有進過版控**。而
`agent_harness/scripts/auto_git_push.ps1`（定時推送）只 `git add agent_harness`——
記憶不在那棵樹下，就永遠不會被推到 `cgc0907/fusionroutemot`。

索引方案（`../engine_loop/memory/INDEX.jsonl`）推的是**指標，不是內容**：另一台機器拿到索引
也讀不到本文。要有內容就得有實體檔，所以這裡有。

## 同步義務（明確接受）

快照**不會自動更新**。要更新就重跑：

```sh
python3 agent_harness/scripts/import_harness_snapshot.py          # 先：更新快照 + SNAPSHOT.jsonl
python3 agent_harness/engine_loop/memory/build_memory_index.py   # 後：更新索引
```

`SNAPSHOT.jsonl` 逐檔記錄來源路徑、`source_sha256`、`source_bytes`、`source_mtime` 與
`snapshot_date`，所以「這份快照對應原檔哪一版」是可查的，不靠猜。

**注意**：匯入腳本住在 `Backup/`，而 `Backup/` 被 `.gitignore:396` 排除 ⇒ 腳本本身不會被提交。

## 誰是權威

`../engine_loop/memory/README.md`（§為什麼是索引，不是副本）記了完整的決定與三個防線。
一句話：**要引用事實，讀 `.workbuddy/memory/`；要同步到別台機器，讀這裡。**
