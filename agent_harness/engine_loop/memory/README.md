# engine_loop/memory — 專案記憶的索引入口

這個目錄**不是**記憶的存放處。專案記憶的權威位置是 `.workbuddy/memory/`：

```
.workbuddy/memory/MEMORY.md          # 跨日的專案長期事實（模型身分、profile 幾何、入口指令、量測衛生）
.workbuddy/memory/YYYY-MM-DD.md      # 每日工作記錄，append-only，一天可能到 1000+ 行
```

那些檔由 host（其他 WorkBuddy session）在 loop 跑的時候持續寫入。因此本目錄**只放衍生物**。

## 為什麼是索引，不是副本（2026-09-16 修訂）

複製一份到 `agent_harness/` 會讓同一批 bytes 有第二個權威來源，而第二個來源的失效方式是安靜的：
原檔改了，副本看起來還是一樣權威。本專案對 `scripts/check/*` 已經有同一條規則（放索引、不複製），
記憶適用同一條理由。所以：

- 原檔留在 host 寫入的位置；
- 這裡只有 `INDEX.jsonl`（由腳本產生，可重建、可驗證）；
- `--check` 會重新推導並在 drift 時**失敗**。

**修訂：本目錄仍然只放索引，但 `agent_harness/memory/` 現在多放了一份 dated 實體快照。**

原先寫的是「若要把內容實體鏡像進 repo，那是另一個決定（會引入同步義務），目前刻意不做」——
那個決定在 2026-09-16 被做了。理由不是「索引不夠用」，而是一個索引服務不到的需求：

- `agent_harness/scripts/auto_git_push.ps1` 會定時把 `agent_harness/` `git add` 後推到
  `cgc0907/fusionroutemot`。索引只含 `path` 與 `sha256`，**推上去的是指標、不是內容**——
  另一台機器拿到索引也讀不到記憶本文。要讓記憶與 skill 真的進到那條推送線上，只能是實體檔。

同步義務是**明確接受**的代價，並用三個手段把「安靜失效」壓住：

| 手段 | 位置 | 擋什麼 |
| --- | --- | --- |
| 檔頭 banner | 每個快照檔開頭（frontmatter **之後**） | 讀者以為它是最新版 |
| `SNAPSHOT.jsonl` | `agent_harness/memory/`、`agent_harness/skills/` | 無法回答「這份快照對應原檔哪一版」 |
| 這一段 | 本檔 | 兩個機制重疊時不知道誰是權威 |

快照**不是**權威，`--check` 也**不**驗快照——它只驗 `.workbuddy/memory/` 與 `INDEX.jsonl` 的關係。
所以兩者要**一起**重生，順序固定（先快照、後索引）：

```sh
python3 agent_harness/scripts/import_harness_snapshot.py          # 先：更新實體快照 + SNAPSHOT.jsonl
python3 agent_harness/engine_loop/memory/build_memory_index.py   # 後：更新索引
```

## 用法

```sh
# 重建索引（記憶檔有變動時）
python3 memory/build_memory_index.py

# 驗證索引與 .workbuddy/memory/ 一致；不一致 exit 1
python3 memory/build_memory_index.py --check

# 查「我們對 X 已知什麼」——只讀索引，不載入 1000 行
python3 memory/build_memory_index.py --query mmid -n 8
python3 memory/build_memory_index.py --query "allowlist" --full
```

`--query` 會回報 `path:line_start-line_end` 與該節的子標題，所以下一步是精準讀取，例如
`Read(.workbuddy/memory/2026-09-15.md, offset=854, limit=70)`（`offset` 直接用印出來的
`line_start`，是 1-based）。

## `INDEX.jsonl` 的形狀

每行一筆，兩種 `row`：

| row | 欄位 | 用途 |
| --- | --- | --- |
| `file` | `path` `abs` `bytes` `sha256`（前 16 碼）`lines` `mtime` | 這份記憶現在是哪一版 |
| `section` | `path` `title` `line_start` `line_end` `lines` `body_bytes` `subs[]` | 一個 `##` 節及其 `###` 子標題 |

節的邊界是 `##`。`#`（檔標題）與 `###` 刻意不切：前者只出現一次，後者掛在父節上，這樣一節剛好是
一次能讀完的單位。

## 與其他機制的關係

- `traces/` 記的是**引擎**發生了什麼（episode / decision / lesson）。
- `memory/` 通的是**這個 repo 已知什麼**（人與 agent 累積的判斷）。
- 兩者不重疊：`decisions.jsonl` 回答「這一步該怎麼判」，記憶回答「這件事以前怎麼走過來的」。
  一個 decision 的 `action` 若指向某段記憶，應引 `path:line_start`，不要複述內容——複述會漂移。

## 已知邊界

- 索引只涵蓋 `.workbuddy/memory/`。`~/.workbuddy/MEMORY.md`（跨專案的個人偏好）與雲端
  profile 不在這裡，它們的 scope 不是這個 repo。
- `sha256` 只取前 16 碼，用途是「是不是同一版」，不是完整性證明。
- 每日記錄是 append-only 且持續變動，所以索引在**同一天內**會頻繁過期；`--check` 紅燈在這種
  情況下是預期行為，重跑即可。
- `agent_harness/memory/` 的實體快照是**手動 snapshot，不是同步**。它只在有人跑
  `agent_harness/scripts/import_harness_snapshot.py` 時更新。
  2026-09-16 之前那支腳本住在 `Backup/`（`.gitignore:396` ⇒ 不受版控，clone 出來的機器上
  **根本沒有它**），而且用硬編碼清單決定要匯入哪些檔——新增 skill 或**單純過了一天**都會被
  靜默漏掉。現在它進了版控，且清單改成 glob（`~/.workbuddy/skills/*/SKILL.md`、
  `.workbuddy/memory/*.md`），所以「被漏掉」這個模態不再存在。見
  `agent_harness/skills/README.md` 的「待匯入清單是推導出來的」。
