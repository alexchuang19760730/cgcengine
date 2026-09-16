# MEMORY_FACTS — 長期事實／陷阱／周邊介面

> **這是快照，不是權威副本。**
> 權威位置：`.workbuddy/memory/MEMORY_FACTS.md`（由 host 持續寫入）。
> 本檔於 2026-09-17 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 索引與漂移檢查見 `agent_harness/engine_loop/memory/INDEX.jsonl`。

> **這是 `MEMORY.md` 的主題分檔（2026-09-17 拆分），不是歷史存檔。** 動 build／載入／預算／`-ub`／
> mmap，或要碰 `agent_harness/` 之前讀本檔。這裡每一條都是「踩過一次就不該再踩」的。

## 長期事實（踩過就不該再踩）

- **`MTP_SUPPORT` 是編譯期開關且必須定義**（macOS 沒預設，靠 `scripts/build_fork_llama.sh`）——少了它被
  編掉的是**修正**（`qwen35moe.cpp:737` ＋ 四處 `[CGC MTP fix]`）。
- **SONAME 內嵌 commit 數** ⇒ 全量重建換檔名、舊檔變孤兒；載入集要從 `otool -L` 反推再 realpath，**不要
  寫死檔名**；`strings -a` 要一起印**不受 guard 保護的對照字串**。
- **錯誤路徑不得「先釋放、後記錄」**（`prefill250` req2 SIGSEGV 根因）。判準：**指令序本身就是證據** ⇒
  要保留當時的二進位（`Backup/pre_mtp_rebuild_20260916/`）；崩潰報告的 `imageOffset` 可反組譯。
- **`-ub` 是 req2 OOM 的旋鈕**：`4096` 全過、`6144` 的 req2 以 status 5 OOM 死（代價 −22%）。**機制沒修
  只有繞道。** 存活表：`4096 5/5`、`5120 3/3`、`5632 4/4`、`6144 0/5`、`6144＋池 6 GiB 3/3` ⇒ **邊界是
  (pool + compute) 的和**。出廠預設 5632。
- **`--load-mode mmap` 與 expert cache 不相容**（L4 pool 是 read-only file-backed 映射 ⇒ zeroing 寫唯讀頁
  ⇒ `SIGBUS` 在 `__bzero`）；缺陷已修但**任何量過的寬度仍是 0/5** ⇒ **不是 OOM 的槓桿，不要再試**。
- **`-ngl` 一律被顯式帶上 ⇒ `-fit` 永遠是 no-op**（`common/fit.cpp:377-379`）；`CGC_SERVER_FIT=1` 才真跑。
  **argv 是位置比對**（D5 config stamp）⇒ 不可重排。
- **`-expert-cache 8192` 是上限不是實配**（實配 ≈2.5 GiB）；啟動預算 13030＋8192 = 21222 vs 16384 MiB ⇒
  **OVERSUBSCRIBED 4838 MiB**（`[防護 2d]` 會印；`CGC_SERVER_STRICT_BUDGET=1` 超額 exit 1）。
- **`CGC_METAL_LIB` 不是載入覆蓋**（只做 `_Pool` 閘門偵測）⇒ metal 庫 A/B 只能**互換檔案**。
- **同一檔案的多個 Edit 不能並行送**（後寫覆蓋先寫，兩筆都回 success）⇒ 序列化或合併成一筆。
- `Backup/` 在 `.gitignore:396`（要 `git add -f`）；`check 8` 只看已 staged 檔。

## agent_harness（另一條線；本線只讀）

兩迴圈 `tb_loop/`／`engine_loop/`；`CONVENTIONS.md` = 判準憲章（A 可引用／B 診斷／C 讀原始碼／D 流程／
E 自我約束），同時是 `sft_pi/` 的 system prompt ⇒ 改它要走 §6.3 閉環對照。`traces/`：
`episodes/decisions/lessons.jsonl`；`engine_loop/memory/` **只放索引**，原檔在 `.workbuddy/memory/`。

- **D6**：`agent_harness/{memory,skills}/` 是**非權威 dated 快照**（只為跨機器搬運，因為
  `auto_git_push.ps1` 推送的是內容而不是指標）⇒ 要一起重生得**先**
  `scripts/import_harness_snapshot.py`、**後**索引；**兩個 `--check` 都不驗快照**。
- 快照是**手動**的，`SNAPSHOT.jsonl` 的 `source_sha256`／`source_bytes`／`source_mtime` 是它「對應原檔
  哪一版」的唯一依據 ⇒ **看到它落後是預期狀態，不是故障**（它只在有人跑那支腳本時更新）。
- **★ `auto_git_push.ps1` 在這台機器上不會跑（2026-09-17 實查）**：它是 **PowerShell**，硬編碼
  `$RepoDir = "D:\alex\flashkv0516\cgcengine_full"`、remote `cgc0907`、branch `fusionroutemot`
  ⇒ 那是**另一台 Windows 機**上的備份工作。本機沒有 `pwsh`／`powershell`、crontab 與 LaunchAgents
  都沒有相關條目，而且 `agent_harness/scripts/logs/`（腳本第一次跑就會建）**不存在** ⇒ 從未在此執行。
  **⇒ 「D6 說 `auto_git_push.ps1` 會週期性推送」≠「現在確實在推」**：GitHub 上 `fusionroutemot` 的
  最後一筆是 `1f3b0a78d "auto(agent_harness): periodic backup 2026-09-16 14:04"`（正是該腳本的訊息格式
  ⇒ **它至少成功跑過一次**），之後沒有；而本線的工作在**另一條分支** `demo/sweet-spot-windows-fix`。
  ⇒ **記憶要真的上 GitHub，得先有人在這台機器上刷新快照＋commit＋push**（快照與索引本身不會自己上傳）。
- **不要拆成獨立 git repo**：它刻意不自足（`index_assets.py` 是 `REPO=HERE/../..`、
  `build_memory_index.py` 是 `HERE/../../..`、`emit_episodes.py` 用 `find_repo()` 往上找 `.git`）。

## 記憶機制（09-17 拆分後）

- `.workbuddy/memory/` 現在是 **1 索引（`MEMORY.md`）＋ 3 主題檔（`_PERF`／`_S1`／`_FACTS`）＋每日
  append-only 日誌**。三支工具全部用 **glob** 而不是白名單（`build_memory_index.py:66` 的 `*.md`、
  `index_assets.py:352` 的 `.workbuddy/memory/*.md`、`import_harness_snapshot.py:72`）⇒ **新增／刪除
  任何 `.md` 都會讓索引與 manifest 漂移，必須重生（順序見 `MEMORY.md` 的入口）**。
- **★ `.workbuddy/` 整個在 `.gitignore:41` 被忽略，`git ls-files .workbuddy` 是空的** ⇒ **記憶本體
  不在版控內**（`git status` 看不到它，`git add` 也不會收它）。要讓記憶跨機器、或進 `auto_git_push.ps1`
  的推送線，**唯一的路是 `agent_harness/memory/` 的實體快照**（D6 修訂）。⇒ 動完記憶後，若那個推送線
  需要這批內容，得跑 `python3 agent_harness/scripts/import_harness_snapshot.py`（它會把**當日日誌全文**
  與 4 個 skill 一起快照，所以是一個要有人決定的動作，不是自動的）。
- `index_assets.py` 對記憶檔只驗**存在**（`.workbuddy/memory/` 在 `VOLATILE_PREFIXES` 內，每日日誌當天
  必變 ⇒ 驗 bytes/mtime 會是永遠紅的閘門）；`build_memory_index.py --check` 則會**重新推導並在 drift 時
  exit 1** ⇒ 記憶有動就要重生，且**不要**靠「`--check` 紅是預期」當藉口跳過。
- **重生順序固定、而且「每一次」都要按序**：先 `build_memory_index.py`、後 `index_assets.py`。
  `build_memory_index.py` 每次都會重寫 `INDEX.jsonl`（bytes 可能相同、只有 mtime 動）⇒ 若 `index_assets`
  報漂移**且差異只在 mtime**，那就是「順序寫反了」，重跑一次兩支即可，不要去找內容差異。
