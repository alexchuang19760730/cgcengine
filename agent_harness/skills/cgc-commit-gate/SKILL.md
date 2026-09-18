---
name: cgc-commit-gate
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）提交 commit 時，正確通過 pre-commit 閘門並滿足 D5 的完整流程與陷阱。當要在該 repo commit、被 pre-commit hook 擋下、不確定要不要跑 replay benchmark / M1/M2/M3 oracle、或 `check_build_tracked.sh` 印 OK 卻全是 SKIP 時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-commit-gate/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-18 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# flashkv-devserver 提交閘門

專案：`/Users/alexchuang/Documents/flashkv-devserver`
（**是 git worktree**，`.git` 是一個檔案 → `.../flashkv0516/.git/worktrees/flashkv-devserver`）

**本檔按主題編排**：§1 環境陷阱 ／ §2 閘門鏈 ／ §3 索引 ／ §4 併行 writer ／ §5 憲章（D6 與快照）
／ §6 收尾與 push ／ §7 commit 風格 ／ §8 附錄（歷史輪次索引）。
**要動手就從 §0 的指令區塊開始，卡住再查對應主題。**

### 動手前必記（只有這 8 條會真的弄壞 commit；細節在後面對應節）

1. 手動預演要自己帶 `BIN_DIR='src/llama.cpp/build/bin'`，否則閘門全 SKIP 卻印 `OK`（§1.1）。
2. 提交要 `RUN_REPLAY_BENCH=0`——這是**依 D2**（基線 stale），不是腳本預設（§2.3）。
3. 有動 `src/` ⇒ 跑 `m123_oracle_gate.py`，且**先看 `comparable` 再讀 M1/M2/M3**；
   **D5 的白皮書那一半沒有豁免**（§2.4 §2.5）。
4. 建置新鮮度看**建置輸出有沒有編譯行**，不是 exit code 或 mtime（§2.9）。
5. 索引重生順序固定、**每一次**都要：先 `build_memory_index.py`、後 `index_assets.py`（§3.1）。
6. 「commit 之後才知道的事」（hash／push／gate 輸出）⇒ 收尾通常是**兩個 commit**（§6.1）。
7. 在 `src/` 上做過實驗 ⇒ 還原要證到**產物 md5 逐位元相同** ＋ `git status --short -- src/` 空（§2.10）。
8. 動手前後各跑一次 `git status --porcelain -uall`；看到**不是你改的** modified／staged 檔
   ⇒ 停下找 owner，只新增不修改、不重生索引、不 commit（§4）。**`agent_harness/` 目前歸另一個
   session（他在做 E1），引擎層（`src/`、`scripts/check/`）歸這個 session。**

---

## 0. 30 秒版：標準提交指令

```sh
cd /Users/alexchuang/Documents/flashkv-devserver

# (a) 若有動 src/ → 先跑 D5 數值閘門（自己起 server，所以先確認 8080 空）
lsof -nP -iTCP:8080 -sTCP:LISTEN ; pgrep -fl llama-server     # 兩者都要空
python3 scripts/check/m123_oracle_gate.py --tag <標籤>        # ~40–60 s（含載入模型 8–9 s）
#     重建完成後才能跑（dump 沒有 binary 指紋，見 §2.6）

# (b) 若做過 git stash 換臂，或動過跨 dylib 邊界的 struct → 先重建再跑任何測試（§2.12）

# (c) 有動被索引的檔案 → 依序重生索引（順序固定，見 §3.1）
python3 agent_harness/engine_loop/memory/build_memory_index.py    # 先：寫 INDEX.jsonl
cd agent_harness/engine_loop && python3 index_assets.py && cd -   # 後：MANIFEST 記 INDEX 的 bytes/mtime

# (d) 預演閘門（不要盲目提交）
BIN_DIR='src/llama.cpp/build/bin' RUN_REPLAY_BENCH=0 bash scripts/check_build_tracked.sh --repo "$PWD"

# (e) 提交（RUN_REPLAY_BENCH=0 必須在 commit 指令的環境裡 —— hook 繼承環境）
RUN_REPLAY_BENCH=0 git commit -F Backup/commit_msg_*.txt

# (f) 推兩個遠端，並用 ls-remote 的真實 SHA 驗三處相同（§6.2）
```

**定調兩句**：D5 的閘門很便宜（~25 s），**不要用結構論證取代它**；「commit 之後才知道的事實」
結構上塞不進被提交的 commit，所以**收尾常常是兩個 commit**（§6.1）。

---

## 1. 環境陷阱（sandbox ／ shell ／ 工具）

### 1.1 `BIN_DIR` 沒指定 → 閘門全 SKIP 卻印 OK

`check_build_tracked.sh` 預設看 `build/bin`，**本 repo 的產物在 `src/llama.cpp/build/bin`**，
所以預設會讓追蹤／rpath／deadlock／原始碼↔binary 同步／replay 全部 SKIP，最後仍印一句無條件的 `OK`。
這是 B7（可被跳過的閘門不是閘門）的實例。**`pre-commit` hook 已經幫你帶了
`BIN_DIR='src/llama.cpp/build/bin'`**，手動跑要自己帶。

### 1.2 hook 落點在 worktree 之外，寫死路徑會偽 PASS

`git rev-parse --git-dir` → `/Users/alexchuang/Documents/flashkv0516/.git/worktrees/flashkv-devserver`，
而 hook 在 **common dir**：`/Users/alexchuang/Documents/flashkv0516/.git/hooks/pre-commit`。
它 `export BIN_DIR=...` 後 `exec "$REPO_ROOT/scripts/check_build_tracked.sh"`，其中
`REPO_ROOT="$(git rev-parse --show-toplevel)"` **在執行時解析**——刻意不寫死，否則從別的 worktree
commit 時會去驗錯的 repo（偽 PASS）。自己重裝 hook 時要保留這個特性。

### 1.3 `grep` 在這個 sandbox 會靜默失效 ⇒ 用內建 Grep 工具

實測：`grep -n "SIGTERM\|SIGINT" <file>`（BSD BRE 把 `\|` 當字面）與 `grep -n "^## "` **都回空**，
exit code 在 0/1 之間不一致，而檔案裡**明明有**那些字串（`sed -n` 與內建 Grep 都看得到）。
**一個「查不到＝沒有」的假陰性會直接變成結論**，這與 B7／B12 同族。
例外（已雙向量過，可用）：`strings -a <file> | grep -q <str>` 這種用法是好的（MTP-on rc=0、MTP-off rc=1）。

**★ 第二個實例，而且它直接產生了一個錯誤結論（2026-09-17 交付時踩到）：
`git diff <file> | grep -E "^[-+]" | grep -v "^[-+][-+]"` 會靜默漏行。**
用這條管線讀 `MANIFEST.jsonl` 的 diff 時，`scripts/run_server.sh` 的那一筆**整條不見了**
（`-`／`+` 兩行都沒有），於是我得到的結論是「另一條線改的檔沒進 manifest、我的 commit 不會污染它」——
而**事實相反**（該筆的 bytes/mtime 正是他們 11:38:21 的版本）。改用 **python** 重做同一個 diff
（`json.loads(line[1:])` 逐行印）才看到真相。⇒ **凡是「diff／log 的某些行是不是存在」會改變結論的地方，
一律用 python 或內建 Grep，不要用 bash `grep` 過濾。** 這一條與 §4 的併行判斷直接相關：
「他那筆有沒有進索引」正是決定要不要 commit 的那個問題。

**★★ 第三個實例，而且它產生的是「與輸出明文矛盾的數字」（2026-09-18）：`cmd | tail -N; echo $?`
讀到的是 `tail` 的退出碼，不是 `cmd` 的。** 我用那個形狀量一個新寫的 `--check`：

```sh
python3 agent_harness/scripts/import_harness_snapshot.py --check 2>&1 | tail -4; echo "rc=$?"
-> DRIFT: 1 項。…
rc=0                      ← 錯的：這裡的 $? 是 tail 的
python3 agent_harness/scripts/import_harness_snapshot.py --check > /tmp/o.txt 2>&1; echo $?   # rc=1 ✔
```

同族的還有 `${PIPESTATUS[0]}`（bash）。**沒有輸出可以讀的時候，這個錯誤是完全隱形的**
（例如用 `| tail` 只為了截短一段你看過的訊息，而結論只依賴 rc）——而它與「檢查真的回 0」
長得一模一樣。判準：**要量 rc 就不要接管線。**
一般化：**兩份證據互相矛盾時，先懷疑量測工具，不要先懷疑被測物。**

### 1.4 zsh 不對未加引號的參數做 word-split

`for pair in "a b"; do set -- $pair; echo $2; done` 在 zsh 下 **`$2` 是空的**，會得到
`fatal: Not a valid object name`，看起來像「那個 SHA 不存在」——**是 shell 把參數吃掉了，不是 git 的問題**。
直接寫成兩個獨立命令，別用迴圈拆字串。

**同一族：`cd` 寫在 `if` 區塊（或 `&&` 串）裡面會外洩到後續指令。** 實測
`if [ -d X ]; then cd X && ... ; fi; ls agent_harness/scripts/` ⇒ 那個 `ls` 跑在 `X` 底下，
於是列出**另一個 repo** 的內容，看起來像「本 repo 的檔案不見了」。診斷腳本時要用**絕對路徑**，
不要把 `cd` 放進條件區塊——它產出的是一份**自信的錯誤清單**，而不是報錯。

### 1.5 macOS 沒有 `timeout`／`gtimeout`；`ps` 可用但**欄位語意必須當場驗**，行程身分用 `pgrep -x`

要做「啟動 30 秒後自動收掉」的 smoke，用背景 PID ＋ 有界 sleep：

```sh
bash -c 'CGC_SERVER_PROFILE=prefill250 CGC_SERVER_UBATCH=4096 bash scripts/run_server.sh \
           >/tmp/p3.txt 2>&1 & P=$!; sleep 32; kill -TERM $P; sleep 5;
         kill -0 $P 2>/dev/null && echo STILL_ALIVE || echo EXITED'
# 判準看「執行檔 basename」，不要看命令列文字，也不要看 ps 的欄位索引（下一段）
pgrep -x llama-server ; pgrep -x llama-bench
```

判準是 log 裡同時有 `model loaded` 與 `listening on http://0.0.0.0:8080`，而且 `SIGTERM`／`SIGINT`
走優雅關閉（`[CGC] Received SIGINT — initiating graceful shutdown` ⇒ 不洩漏 Metal buffer；
`kill -9` 才會）。

**★ `pgrep -f <路徑>` 是「命令列文字比對」，不是行程身分（2026-09-17 同一類踩到三次）。**
任何**命令列裡剛好提到那個字串**的行程都會被算成「殘留的 llama-server」：`bash -c "… build/bin/llama-server …"`、
`grep llama-server`、記錄用的 `echo`，甚至**一個把它自己的 pattern 印在命令列上的監看 shell**
（實測：那種 shell 會讓閘門假性 abort）。後果不只是誤報：`run_server.sh:534` 的
`cgc_existing_llama_server_count` 把這個數字餵給 memory guard 的 `OTHER_LLAMA_SERVERS`，
而該門檻在**每個** profile 上都是 **0** ⇒ **多到 1 就 `startup blocked by memory guard` ＋ `exit 1`，
直接擋掉一整臂**（實例：日誌 `§Z6`／commit `a6125accc`）。同一類 bug 在本 repo 有**三份**獨立實作
（`run_server.sh` 的 `OTHER_LLAMA_SERVERS`、`Backup/run_req2_retest.sh` 的 `alive()`、以及本檔舊版的檢查）。

**★★ 2026-09-17 21:0x 更正：本機的 `comm` 是「完整路徑」，不是 basename ⇒ 下面那個 `$2 == "llama-server"` 是一個永遠不命中的閘門。**
實測（用別條線正在聽 8080 的 server 當樣本）：

```
ps -Ao pid=,comm=      →  47353 /Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/build/bin/llama-server
ps -Ao pid=,comm= | awk '$2=="llama-server"'      →  （空）
pgrep -x llama-server                              →  47353      ← 對的
```

代價是**靜默的假陰性**：閘門印 `GATE-PASS` 而機器上其實有一輪在跑。本輪（EN-80）我因此
**兩次**把「自己剛啟動、正在載入 13.66 GB 的行程」讀成「被系統殺掉了」，多花一輪去診斷一個不存在的失敗。

**正確寫法** —— 用 `pgrep -x`（比對 **basename**，且不是命令列文字比對 ⇒ 同時躲開 `pgrep -f` 的誤報與
`comm` 全路徑的假陰性）：

```sh
# 執行檔身分：basename 精確比對
pgrep -x llama-server ; pgrep -x llama-bench ; pgrep -x llama-speculative-simple
# 量測驅動（python 腳本）：這類只能命令列比對，所以用字元類躲掉自己那一行
pgrep -f '[d]ecode_sweep\.py|[m]123_oracle_gate\.py|[p]rod_matrix\.py|[p]hase_split_ab\.py|[d]ecode_step_profile\.py'
# 別條線的 driver 包裝（`bash -c 'sleep N; … tail …/dsp_out/driver*.log'`）——它不含任何 llama 字樣
pgrep -f '[d]sp_out/driver|[p]refill_certifiability'
```

`ps` 仍然可用，但**只信任你當場用樣本驗過的欄位語意**；`$2=="llama-server"` 在這台機器上不是那樣的語意。
下面保留舊寫法只為了對照 —— 如果你要改它，先跑一次上面的樣本驗證。

```sh
# 舊寫法（留作對照；在本機 `comm` 是完整路徑 ⇒ 第 2 行永不命中）
ps -Ao pid=,comm=,command= | awk '
    $2 == "llama-server" || $2 == "llama-bench" { print; next }
    $2 ~ /^python/ && (index($0,"decode_sweep.py") || index($0,"decode_bench.py") ||
                       index($0,"m123_oracle_gate.py")) { print; next }
    $2 == "bash" && index($0,"run_ids_dst_capture.sh") { print }'

# 現用寫法：basename 精確比對 + 字元類躲自匹配
{ pgrep -x llama-server; pgrep -x llama-bench; pgrep -x llama-speculative-simple;
  pgrep -f '[d]ecode_sweep\.py|[m]123_oracle_gate\.py|[p]rod_matrix\.py|[d]sp_out/driver'; }
```
「有輸出就停手」的閘門必須**在同一個分支裡 abort**，只印出來不算 —— 見 §4-6。

**要知道「哪些行只印在真實啟動路徑」的話，先確認探針有沒有走到那裡。** `run_server.sh` 的零成本探針
有兩個，但**都在 `[fit]` 之前就退出**：`CGC_SERVER_STRICT_BUDGET=1` 在 `[防護 2d]` 印完預算段就 `exit 1`；
`CGC_SERVER_LOAD_MODE=mmap` 在 `[防護 2e]` 就 `exit 1`。所以 `[fit]`／`[kv]` 這兩行**只能用真實啟動驗**。
反過來說，只改 `[budget]` 那一段時，這兩個探針就能在零載入成本下把四種情境都印出來——**不要為了看一行
echo 去載 13 GB 的模型**。

### 1.6 `Backup/` 與 `.workbuddy/` 進不了 commit（別浪費時間找）

`.gitignore:396` 排除 `Backup/`（含所有證據檔與 commit message 草稿）、`.workbuddy/`（含 `memory/`）、
`bin/`（`TurboFieldfareCLI-*`）、`__pycache__/`。
⇒ **message 草稿放 `Backup/` 是安全的**（它本來就不會被提交），但它不會被提交——message 內容要真的
寫進 commit。**同理：在 `Backup/` 裡修好的量測腳本不算交付**（要嘛搬進 `scripts/`，要嘛在 final reply
明說它只存在於本機）。

**`Backup/` 底下是「已追蹤」與「被忽略」混在一起的，要先分清（2026-09-17 實查）。** `.gitignore`
只影響**未追蹤**的檔案：`Backup/` 底下有 9 支量測腳本是**已追蹤**的，它們的修改會被 `git add -A`
正常收進去（也會出現在 `git status`）。實測的形狀是
`Backup/analyze_capture_nodes.py` 之類顯示 ` M`，而同一目錄的 `compare_slot_owner.py` 完全不顯示。
⇒ 判準：`git ls-files Backup/ | head` 看它是不是已追蹤。一個**已經存在的**驅動改了會被提交、
一個**新寫的**不會——所以「我在 Backup/ 修好了腳本」與「它進得了 commit」是兩件事。
（本 repo 的慣例是儀器驅動要進版控，所以新的那幾支就用 `git add -f` 收進來，並在 message 揭露。）

**★ 但「已追蹤」不等於「`git add <路徑>` 會成功」——顯式指名 `Backup/…` 一律要 `-f`（2026-09-17 第二次實測）。**
即使那個檔**已經在版控裡**，`git add Backup/compare_pool_row.py` 仍會回
`The following paths are ignored by one of your .gitignore files: Backup` ＋
`Use -f if you really want to add them`，**而且會讓整條 `git add`（同一行裡的其他路徑也一起）失敗**——
exit 1、什麼都沒 staged。所以正確的寫法是**把 `Backup/` 的路徑從主 `git add` 裡拆出來、單獨下 `-f`**：

```sh
git add -f Backup/compare_pool_row.py          # 先，單獨
git add agent_harness/... docs/... src/...     # 後，其餘（這一行不會被拖累）
git diff --cached --name-status                # 驗：staged 清單就是你要的那些
```
（判準仍是 §4：**只 stage 自己的檔**。上面那個「整條一起失敗」的特性有正面用法——它不會讓你
不小心把別人未追蹤的 `Backup/` 檔收進來。）

### 1.7 新增 C++ 測試編譯產物會讓 `git status` 不空

`scripts/check/*.cpp` 若會被編成同名無副檔名的執行檔（目前只有 `expert_cache_ensure_batch_order.cpp`），
要在 `.gitignore` 加**該檔的絕對路徑**（`/scripts/check/<name>`），
**不要用 `scripts/check/*` 這種會吞掉 source 的樣式**。`.gitignore` 本身**不在**索引裡（改它不會讓
`--check` 紅）。

---

## 2. 閘門鏈

### 2.1 誰在守著 commit：`check_build_tracked.sh`，不是 e2e gate

`precommit_e2e_gate.sh` **刻意沒有**接進 hook（`00b70bebc` 明載「Deliberately NOT done here: whether
the hook should invoke precommit_e2e_gate.sh … It needs an explicit decision」）。
所以缺少 e2e 是**既定決定**，不是遺漏——commit message 要這樣寫。

### 2.2 預演與 SKIP 的報法

預演指令見 §0(d)。判準是 **0 FAIL，且每一個 SKIP 都逐列有理由**；
`OK: 沒有 FAIL —但本次有 4 個區段被 SKIP…` 這句是腳本自己提醒你「OK 只代表沒有失敗」。
最常見的 4 個 SKIP：非 dev 分支的 @rpath、`RUN_CGC_PROD_ACCEPT`、`RUN_DEPLOY_HARMONYOS_ACCEPT`、
以及 `RUN_REPLAY_BENCH=0`。

### 2.3 `RUN_REPLAY_BENCH` 的預設有兩層，不要搞混

- 腳本自身預設 `check_build_tracked.sh:559` 是 `RUN_REPLAY_BENCH:-1` → **會跑**（然後因為沒有 8080 上
  的活 server 而 FAIL）。
- 專案慣例 **D2**（`agent_harness/CONVENTIONS.md:447`）：**`RUN_REPLAY_BENCH=0` 一律預設**。
  D2 的理由是 **`.replay_bench_baseline.json` 已 stale**（2026-09-08、commit `37305dbc6` 起沒動），
  跑它等於拿舊基線比新程式。

⇒ 提交要 `RUN_REPLAY_BENCH=0 git commit ...`（hook 繼承環境）。**引用時要說「依 D2」，
不可說成「腳本預設」。**

### 2.4 D5：它其實有**兩半**

原文是「每次 commit 附一份技術白皮書，**並**在提交前跑完 `llama-bench` + M1/M2/M3 + 最新 M2 oracle」。

**數值那一半**：`scripts/check/m123_oracle_gate.py` 只花約 23 s（含載入模型 8–9 s 時全程 40–60 s），
而且會**自己**透過 `run_server.sh` 起 server（所以 profile／env allowlist／load path 都是生產那套，
不是會漂移的手寫 argv），送同一個決定性 probe，與最新 M2 oracle 比對 logits：

```
M1 numeric identity 9/9   M2 decision agreement 9/9   M3 top-k set 9/9
(aux) full_fnv1a64 9/9    cross-tab 9/0/0/0
comparable=true  config_diffs=[]   ← 兩端配置由 run_server.sh CGC_DUMP_ENV=1 解析
```

summary 落在 `Backup/m123_oracle_gate/summary_<tag>.json`，**tag 自己取**。

**白皮書那一半沒有豁免。** 純 doc/腳本的 commit 在「`src/`」那一半可以不跑，但白皮書要出
**新日期的檔**，不要就地改既有的（本 repo `docs/` 的慣例：`PREFILL250_DECODE25_WHITEPAPER_*` 有
1745／1810／1900／2240／2355 多個版本）。**白皮書可以由別的 session 產出**；納入你的 commit 時要在
message 講清楚三件事：(a) 不是這個 session 寫的（附 mtime 與它引用的 commit）、(b) 你核對過哪些數字、
(c) 你**沒有**逐字審閱。版式與取材清單見 skill **`cgc-whitepaper-delivery`**（含「同批檔名要指名到
`_HHMM`」）。

★ **「出新檔」是預設，不是唯一選項。** 當這一輪是**同一個問題、同一支儀器的連續推進**（而不是新的
里程碑）時，正法是**就地追加節**到既有那一份——判準、義務（開頭仍要加取代標註框、節號接續、
**每次追加之後都要重跑整份自檢**）見 `cgc-whitepaper-delivery` §6.1.1。使用者的指令是決定性的
（2026-09-18：「更新…**到之前的量測工具技術白皮書 html 裡面**」）。**要在 commit message 寫明
這是「修訂既有白皮書（追加 §X–§Y）」而不是新檔**，否則下一個人會以為少交了一份。

**實務判準**：D5 的數值那一半看「有沒有動 `src/`」，不是「有沒有動 engine 邏輯」。逐個 commit 數過
`a22ebe88e` / `6ceeb281b` / `57bd90801` / `e0152777d` 全部 **0 個 `src/` 檔**，四者 message 都**沒有**
報告 M1/M2/M3；報告它的是動到 `src/` 的 `2b284667f` 與 `2162e8cbd`。
注意 `e0152777d`（subject 就叫 *the D5 gate is cheap -- run it*）本身 0 個 src 檔——它是 `2162e8cbd`
的補記 commit。⇒ **一個純 doc/腳本的 commit 技術上可以不跑，但跑它只花 23 秒**，而「省下 23 秒然後
在 message 裡用一段話解釋為何不跑」是這個 repo 已經明確判為錯的做法（`eng-gate-0016`）。**跑了就寫出來。**

### 2.5 D5 的 oracle 旋鈕是「釘住的」——它會擋下一種你以為沒事的改動

`m123_oracle_gate.py` 有 `ORACLE_PINNED_ENV = ("CGC_SERVER_BATCH=6144", "CGC_SERVER_UBATCH=6144")`
（放在 `DEFAULT_REF` 旁邊），在 `resolve_launch` 之前併入，並把**有效集合印在啟動前**。
這必要，因為 gate 說明文字原本宣稱「預設會重現 oracle 配置（batch 6144）」，而那句話**寄生在
`prefill250` 的生產預設值上**：把生產預設改成 5632 之後，gate 立刻對 `CGCENV.BATCH` / `CGCENV.UBATCH` /
`ARG[27]` / `ARG[29]` 四列差異報 **INVALID COMPARISON**，而同一份輸出裡 M1/M2/M3 都是 9/9
（印著 *printed for information, NOT a verdict*）。

⇒ **改了 `prefill250` 的生產預設（或任何被 gate 預設 profile 帶上的 knob），先看 `comparable` 欄位
再讀 M1/M2/M3**；`comparable=False` 時那三個 9/9 不是裁決。
要比另一個配置就顯式覆蓋（`--env`）並**重新基線**（`--write-ref` ＋ `--no-pin-oracle-env`）——
覆蓋之後 INVALID 會再出現一次，**那是訊號不是故障**。
**殘留**：gate 認證的是 oracle 配置的數值；出廠預設 5632 沒有自己的參考檔，只量得到跨配置的 9/9。

**第二個實例（2026-09-16，把 `CGC_SPAC=1` pin 進 `prefill250`）**：2 個 diff，形狀是
`ENV.CGC_SPAC: ref='<absent>' now='1'` ＋ `_ALPHA` 同款。這一次的處置是**重新基線 v5**，而
**證據標準是「逐位元相同」，不是「M1 = 9/9」** —— v5 與 v4 的 jsonl **md5 相同**
（`a0a0ca742ca94e843c54b39981742738`），代表 oracle rows 沒有被重推導、只是被重新蓋章；用 M1 9/9
當證據是弱述句（「我看的時候是 9/9」），用 md5 相同才是「數字沒有動，動的是戳記」。

固定順序（照抄，一次跑完不要跳）：
1. `m123_oracle_gate.py --tag <t> --allow-incomparable` → 讀 INVALID 的 diff 與 M1/M2/M3；
2. `m123_oracle_gate.py --tag <t5> --write-ref Backup/knifeedge_matrix/ref_..._v5_<why>.jsonl
   --ref-note "<為什麼、以及證據>"` → 同時給出 v5-vs-v4 的比對；
3. `md5 -q` 兩份 jsonl 對一眼（相同 ⇒ 名目重基線）；再確認新 `.cap` 的 `resolved.ENV` **有**記下新鍵；
4. 改 `DEFAULT_REF` 並在旁邊寫 dated 註解；5. **再跑一次 gate，要求 `comparable=True`**。

**不要把新鍵塞進 `DIAGNOSTIC_KEYS`。** 那個集合會把鍵從 `config_stamp` 濾掉 ⇒ `.cap` 從此不再記錄
它，下一個讀者分不出 SPAC-on 與 SPAC-off 的基線。`CGC_SLOT_TABLE_GPU` 的前例**不適用**：那是「整條
主張就是 bit-identity」的實驗旋鈕，不是生產預設。

**盤點副作用（pin 一個 env 鍵會靜默重解析每一個「預設 profile 就是它」的工具）**：
動手前跑 `grep -rn '"--profile", default=' scripts/check/*.py`。2026-09-16 的實際清單：
`decode_sweep.py`（**預設 `prefill250`**）、`m123_oracle_gate.py`（同）、`ab_interleave.py`（`prod25`，不受影響），
外加不吃 `--profile` 的 `llama_bench_matrix.py`（`prefill250` 臂）、`prefill_certifiability.py`（`--arm prefill250`）、
`run_thermal_gate.sh → run_req2_retest.sh`（交付數字那條路）。附帶：`decode_sweep` 的 `spac-on` 臂在新
base 上與 `baseline` **同義**（退化了），已在 ARMS 的註解寫明。

### 2.6 D5 的 dump 沒有 binary 指紋 ⇒ 重建完成後才跑，並對一眼時間

`Backup/m123_oracle_gate/cap_<tag>.json` 只記 `created` / `profile` / `resolved.{ARG,ENV,CGCENV}`
—— **沒有 md5、沒有 UUID**。所以若 gate 是在 `cmake --build` 收尾期間起的，那份 PASS **不可歸屬**
（實例：12:12 的 `flagalign` 與 `libllama-*-impl.dylib` 的 12:12 寫入重疊）。
處置：重建完成後才跑 gate，把 `cap_<tag>.json` 的 `created` 與最新產物 mtime 對一眼；一旦重疊就
**換新 tag 重跑，只引用新的那一份**（改用 12:18 的 `flagalign2`，舊的不引用也不刪，留在 `Backup/`）。
成本 25 s，比事後解釋便宜。

### 2.7 check 8：原始碼 ↔ 產物同步

含 `src/` 原始碼的 commit，必須把對應 build 產物一起 staged，且**binary 比 staged 原始碼新**。
注意 `llama-server` 的 mtime **不必**更新：它動態連 `libggml-base.dylib`，install name 不變就不需 relink。
只改 `.h` 也要重建並 stage 同一個 dylib（成員偏移變了，見 §2.12）。
`git status --short` 有時因為 stat cache 沒把它列出來——**用 `git diff --stat` 確認**，它會直接印
`Bin 3136592 -> 3136592 bytes`。

| 改到的原始碼 | 要一起 staged 的產物 |
|---|---|
| `src/llama.cpp/ggml/src/ggml-backend.cpp` | `src/llama.cpp/build/bin/libggml-base.0.19.0.dylib` |
| `src/llama.cpp/src/llama-expert-cache.cpp` / `.h` | `src/llama.cpp/build/bin/libllama.0.0.239.dylib` |

**check 8 曾經獨漏 `.m`。** 它的 case 樣式列了 `*src/*.cpp|*.h|*.c|*.mm|*.metal`——沒有 `.m`。
後果：改 `ggml/src/ggml-metal/ggml-metal-context.m` 會被判成「8 無 llama 原始碼變更（僅 doc/腳本/產物）」
＝**假 PASS**，並同時跳掉「產物有沒有 staged」與「binary 比原始碼新」兩條。已補 `*src/*.m`。
**新增任何會被編進 binary 的副檔名時要同步這個樣式**，並用陽性對照驗它（餵一份含該副檔名的假
staged 清單，看 `staged_src` 是不是 0）。

**check 8 會被「建置戳記」誤導。** `libggml-base` / `libggml-cpu` 內嵌 build stamp
（`strings` 差集恰好是 `8af8c99bd` → `efeade7e2-dirty`）：`git diff --stat` 印
`Bin 771480 -> 771480 bytes`（同大小不同內容），而 `cmp -l | wc -l` 可能報到 **47940** 個位元組不同
——那是戳記字串變長造成的**整體位移**，不是程式碼改動。**判準是 `strings` 的差集，不是 `cmp` 的計數。**
同族的另一個誤讀：`llama-server` 與 `libllama-server-impl.dylib` 被重寫（mtime 更新）但可以與 HEAD
**逐位元相同**（install name 沒動就不需 relink）⇒ `git diff --stat` 是空的，不要因為「它被重建過」
就預期它會出現在 staged 清單裡。

### 2.8 「改了註解要不要重跑 D5」有確定答案：看產物 md5，不要靠推理

註解改動不影響 codegen，但會**位移行號**，而 `__LINE__` 與 DWARF 行表都在二進位裡 ⇒ md5 會變、
D5 就必須重跑。做法：**先把註解壓回原本的行數**再重建，然後比對 md5。實測：
`md5 before/after comment-restore rebuild = dd4d1bc4fbd8b1528eb73d813aae8dd4`（byte-identical）。
⇒ message 寫「D5 未重跑，理由是產物逐位元相同」＋貼 md5。這比「重跑一次然後說它 pass」資訊量更高，
因為它同時證明了**沒有東西改變**。

**clang 對多行巨集的 `__LINE__` 取「收尾括號」那一行，不是巨集名那一行。**
`GGML_ABORT(...)` 的 invocation 從第 786 行開始、`);` 收在第 788 行 ⇒ 執行時印 `file:788`。
不知道這條會以為「原始碼與二進位不一致」，會浪費好幾步去追。最小驗證（實測印 `6`，不是 `4`）：

```c
#define M(...) printf("%d\n", __LINE__)
/* M( 在第 4 行、) 在第 6 行 */
M("a",
  "b",
  "c");
```

靜態查二進位內嵌行號：反組譯後找 `bl _ggml_abort`，往前找 `mov w1, #imm`
（`ggml_abort(file, line, fmt, ...)` ⇒ x0=file、w1=line、x2=fmt）。

### 2.9 建置新鮮度的判準＝「建置輸出有沒有編譯行」

`cmake --build` 的 exit code 在「已經最新」與「剛剛編好」**兩種情況都回 0**。
⇒ check 8 的證據推薦 `cd src/llama.cpp && cmake --build build --target llama -j8`：
印 `Built target llama` 且**沒有任何編譯行**＝產物由當前原始碼產生（比對 mtime 更強，訊息可貼進 body）。
**反過來也成立：一旦它印出 `Building …`，就證明磁碟上的產物不是當前原始碼產生的，且先前在同一棵樹上
取得的所有數字全部失效。** 實例：12:44 全量重建後，13:00 又編輯了 `ggml-metal-context.m`（只改註解）
而沒有重建 ⇒ 12:44–13:50 之間每一筆量測（含出廠驗收與一份 D5 PASS）都屬於舊產物；
`libggml-metal` md5 `968c36cf…` → `f2d1c961…`。check 8 的 mtime 規則會擋這一格（binary < source），
但**它在 commit 前才跑，而量測更早發生**。

### 2.10 在 `src/` 上做實驗之後，「還原」必須是可證的

陽性對照、A/B 探針這類實驗會**改動 `src/`**，而檢查 8 只看「staged 的原始碼與產物同不同步」
——**它看不出「你改了又改回來，但中間那棵樹的產物還留著」**。所以收尾要自己證，三步都要有輸出：

```sh
git checkout -- <動過的檔案>          # 還原原始碼
cmake --build build --target llama-server -j8
md5 <產物>                            # 必須與實驗前逐位元相同
git status --short -- src/            # 必須空
```

實測：`ggml-metal.metal` 的同長度改動（`1.0f`→`1.1f`）重建後，`libggml-metal` 的 md5
`f2d1c96193939bd15404ba713a6fa85d`、size 954,920 **逐位元還原** ⇒ 該 commit 因此可以合法地
**不重跑 D5**（0 個 `src/`、二進位沒變），並在 message 寫明理由。
**反過來：若還原後 md5 不同，那不只是「還原失敗」——你這一輪建立在該產物上的所有結論全部作廢。**

找「差異到底在哪」的標準工具是 `cmp -l` ＋ `otool -l`（實測：兩份產物差 80 個位元組、15 個 ≤64 KiB，
元兇是 ld64 的**內容衍生 `LC_UUID`**，它讓「只雜湊前 64 KiB」的戳記意外地有效）。
一般化規訓在 `traces/lessons.jsonl` 的 **`eng-mh-0037`**：**便宜戳記（取樣視窗／人工列舉清單／抽樣
digest）的判別力只能用陽性對照決定，不能由「它涵蓋了什麼」推論；否證本身有價值，因為它會指名
偵測力來自誰。**

### 2.11 崩潰取證（`imageOffset` / `usedImages` / 保留舊產物）

**判斷「兩臂之間到底換了哪幾個 image」要用 crash report 的 `usedImages` UUID。** 每份 `.ips` 都記錄了
當次載入的每個 image 的路徑與 UUID（本地檔用 `dwarfdump --uuid` 對照）。把兩份報告的 `usedImages`
做 diff，就能證明一次 A/B 是不是單變數——**這是最便宜的單變數證明**，而且它推翻過兩次直覺
（一次把沒改的庫當成變了，一次把其實沒變的庫當成混淆項）。注意 `libobjc.A.dylib` 會因為 faulting
thread 停在 `objc_msgSend` 內而出現／消失，那是**清單呈現**差異，不是載入差異。

**`imageOffset` 才是反組譯要用的偏移，而且前提是「當時那份二進位」還在。** `.ips` 的 `symbolLocation`
是相對**符號起點**，`imageOffset` 才是相對 **image base**。所以**重建前先備份舊產物**
（`Backup/pre_mtp_rebuild_<date>/` ＋ `MANIFEST.txt` 記 md5/size/mtime），否則事後無法證明根因
（B17 的實務前提）。

### 2.12 `git stash` 換臂之後：build tree 屬於**另一臂**；struct 加成員是連結器看不見的 ABI 破壞

做 A/B 的常見手法是 `git stash` 掉修法 → 重建 → 跑舊臂 → `git stash pop`。**pop 不會重建**，
所以 `src/llama.cpp/build/bin` 留著舊臂的產物。若這一輪又在跨 dylib 邊界的 struct 加了成員，
那就同時中了第二層：`llama_expert_cache` 這種以**指標**傳遞的 struct，新增 8 bytes 會把後續每個成員
的偏移整體推移，而**連結器只看得見符號、看不見成員偏移**——不會是連結錯誤，而是「照新標頭編的
harness 連上照舊佈局編的庫」，症狀是**崩在函式庫裡**：

```
SIGSEGV / EXC_BAD_ACCESS (KERN_INVALID_ADDRESS at 0x0)
  libllama.0.0.239.dylib  llama_expert_cache_ensure_batch + 1068
```

實例（2026-09-16）：新增 `n_hit_adopted_queued` 讓 `ever_loaded` 從 `0x608` 移到 `0x610`；
舊庫把 `0x608` 讀成 `n_slot_table_unchanged`（`size_t`，值 0）當成 data pointer 去索引。

**診斷四步，一分鐘內收斂（不要先去改測試——第一反應常是回頭審測試的初始化，那條路會一路改到
看不出問題，因為測試本身沒錯，錯的是它連到的庫）：**

```sh
# a) 符號有沒有被導出（有 = 不是連結問題）
nm -gU src/llama.cpp/build/bin/libllama.0.0.239.dylib | grep expert_cache
# b) 兩套偏移：寫個 offsetof 探針，與反組譯回推的偏移對照
#    otool -tvV <dylib>，故障位址 = 符號起始 + symbolLocation（.ips 的 symbolLocation 就是它）
# c) 這份 dylib 是新碼還是舊碼：找一個本輪新增的字串
strings src/llama.cpp/build/bin/libllama.0.0.239.dylib | grep "CGC-BATCH-INVARIANT"   # 空 = 舊碼
# d) mtime 一眼看完
stat -f "%Sm %N" -t "%Y-%m-%d %H:%M:%S" src/llama.cpp/build/bin/libllama.0.0.239.dylib
```

**規則**：動到跨 dylib 邊界的 struct ⇒ 與該標頭連結的測試／工具要和**它所連的樹同狀態重建**；
`stash pop` 之後的第一件事是重建，不是跑測試。做負對照時要**兩臂都重建庫與 harness**
（否則兩臂的 harness 會各自連錯），這樣才拿得到 HEAD 的 FAIL / 修好的 PASS。

### 2.13 你新寫的那支自測／驗證器，它的**錯誤訊息本身**也要被測

2026-09-16 E3 的實例：`agent_harness/engine_loop/distill/closed_loop.py` 新增了一個守衛
（`--memories-scope` 選到的 lesson id 在投影裡沒有對應檔 ⇒ 硬錯誤）。它在真實情況
**確實偵測到了**缺檔（另一條線剛追加 5 筆 lesson、投影還沒重生）——
但它在**渲染自己的訊息**時崩了：訊息裡用了 `Path.relative_to(REPO)`，
而自測為了不污染真實目錄，正是把那個目錄重導到 **repo 之外**（`/tmp/...`），
於是 `relative_to` 拋 `ValueError`。

結果：從外面看，「**守衛擋下了壞輸入**」與「**這支程式有 bug**」是**同一個事件** ——
一個 traceback ＋ 一個非零結束碼。自測的斷言寫的是「rc≠0 **且**訊息裡有那個 id」，所以報 FAIL。

判準與做法：

- **錯誤路徑平常不會被執行 ⇒ 它是全檔最少被驗證的一條路徑。**
  寫完守衛要**故意餵它壞輸入**，並檢查**訊息本身**，不是只檢查 rc。
- **訊息裡每一個「對路徑做相對化／格式化」的操作都要有 fallback。**
  本 repo 的常見觸發點是：測試用環境變數把目錄重導到 repo 外
  （`CLOSED_LOOP_OUT`／`CLOSED_LOOP_MEM_DIR` 這類「只為自測而存在」的旋鈕就是為此加的）。
  寫一個 `rel(p)`：在 repo 內用相對、不在就用絕對。
- 斷言要含**訊息的內容**，不只是 rc：`rc != 0 and "<那個具體 id>" in out`。
  只驗 rc 的話，一個「因為別的原因而崩」的實作也會通過 —— 而它與正確的守衛無法區分。
- **同一輪要補一個成對的陰性斷言**：把守衛的訊息檢查拿掉後，測試應該要**變紅**。
  這一條見 lesson `eng-gate-0040`（比較器必須先被證明能給出「沒有差異」）。

---

## 3. 索引（`INDEX.jsonl` / `MANIFEST.jsonl` / `CURATED`）

### 3.1 重生順序固定，而且**每一次**重生都要按序

兩者是 **byte 級快照，不是內容摘要**，改一個字就漂移。

1. **先** `memory/build_memory_index.py`（寫 `memory/INDEX.jsonl`）
2. **後** `agent_harness/engine_loop/index_assets.py`（`MANIFEST.jsonl` 記錄 `INDEX.jsonl` 的 bytes/mtime）

順序顛倒 → `index_assets --check` 報漂移；而且**只重生 manifest 不會修**，必須兩者都重跑。
推論：**memory 寫完要放在索引重生之前**。`index_assets.py` **不要加 `--out`**
（相對路徑會寫出第二份 manifest）。

**最常見的犯法是在同一輪裡的「第二次」重生把順序寫反。** 漂移的**簽名很好認——是 mtime 而不是 bytes**：

```
agent_harness/engine_loop/memory/INDEX.jsonl: mtime: manifest '2026-09-16T11:15:00' vs disk '2026-09-16T11:15:15'
```

因為 `build_memory_index.py` **每次都會重寫 `INDEX.jsonl`，即使內容一字不變**（bytes 相同、只有 mtime 動）。
看到這個簽名就**重跑一次「先 `build_memory_index.py` 再 `index_assets.py`」即可**，不要去找內容差異。

### 3.2 哪些檔案被索引、掃描範圍在哪

- **自動索引**：`scripts/check/*`（`index_assets.py:314` 的 glob）、`agent_harness/engine_loop/*`。
- **`CURATED` 逐條列出**：`docs/*`、`agent_harness/CONVENTIONS.md`、`PLAN_ENGINE_LOOP_*.md`、
  `traces/*.jsonl`、`.workbuddy/memory/*.md`、以及 **`index_assets.py` 自己**（改策展列也會改自己的 bytes）。
  所以**改了 `CONVENTIONS.md`、`PLAN_ENGINE_LOOP`、白皮書的 note、或 `index_assets.py` 自己都會讓
  `--check` 紅**。
- **不在範圍內**：`agent_harness/memory/`、`agent_harness/skills/`（§5.3 的快照）。
  在那裡新增目錄**不會**讓 manifest 漂移，要驗的是 `build_memory_index.py --check`。
- **不確定就跑 `index_assets.py --check`**，它會直接印出漂移的 path 與 bytes 變化——它就是為此存在的，
  比猜便宜。
- ★★ **`build_memory_index.py --check` 是「位置對位置」比對，所以新增／刪除一節會讓它對「別的檔案」報漂移**
  （2026-09-18 實測）。它的核心是 `for a, b in zip(have, want)` —— 算出來的 `hk`／`wk` 兩份 key 清單
  **沒有拿來配對**，於是推導清單多一列之後，**它後面每一列都在跟鄰居相比**。
  症狀：你在 `2026-09-18.md` 加**一節**，它卻報 `2026-09-17.md` 有 **16 處**漂移
  —— 看起來**完全像別條線正在寫那個檔**。
  **自證**：被標成 `(file row)` 的那列，其「欄位差異」清單裡會出現 `level`／`subs`／`title`
  （那是**節**欄位，檔案列不可能有）⇒ 一眼可辨的錯位，不必查原因。
  判準：**先逐檔比 sha256 再下結論** —— 把 `INDEX.jsonl` 的 file row 與現算的 sha256／bytes／lines
  逐檔並列，相符的那些就是**沒被動過**。本輪 8 個檔裡 7 個相符、只有我自己改的那個不符
  ⇒ 重生索引即綠，**不要**因此認定有 foreign writer，也不要去找那個「別人在改的檔」（它不存在）。
  修法（**尚未做**）：把 `zip` 換成按 `(row, path, title)` 配對 —— `hk`／`wk` 已經算好了，接上去就行。

**★★ 2026-09-18 實測更正：上面那份清單是「設計意圖」，不是「此刻的成員」。動手前要量，不要照抄。**
那次實際列了 `MANIFEST.jsonl` 的 **108 筆**，按目錄分：`scripts/` 57、`agent_harness/` 33、
`.workbuddy/` 8、`docs/` 5、`Backup/` 5。三個與上面清單**不符**的地方，而每一個都會改變處置：

| 上面的清單說 | 實測 | 後果 |
|---|---|---|
| `traces/*.jsonl` 在 CURATED | **不在**（只有 `traces/schema/*.json`、`validate.py`、`emit_episodes.py` 在） | 追加 lesson 到 `lessons.jsonl` **不會**漂移；§7 的「lesson 必須寫在索引重生之前」在這棵樹上**不觸發** |
| 自動索引 `agent_harness/engine_loop/*` | 只有**頂層**那幾支（`sft_common.py` 也**不在**） | 改 `harness_engine/`、`sft_pi/`、`sft_prime/`、`distill/` 底下的東西不必重生索引 |
| `docs/*` 在 CURATED | 只有 **5 筆**（`PREFILL250_CERTIFIABILITY_*` 等），新白皮書與入口產物都**不在** | 就地追加白皮書**不會**漂移 |

量法（10 秒，只讀不寫）：
```sh
python3 - <<'PY'
import json, collections
rows=[json.loads(l) for l in open('agent_harness/engine_loop/MANIFEST.jsonl') if l.strip()]
print(len(rows), collections.Counter(str(r.get("path","")).split("/")[0] for r in rows))
for t in ("lessons.jsonl","portal/","harness_engine/","sft_prime/"):
    print(t, [r["path"] for r in rows if t in str(r.get("path",""))][:3] or "（不在索引）")
PY
```
**為什麼這件事重要**：§4 的併行 writer 規矩是「只新增不修改、不要重生索引、不要 commit」。
但「你能不能 commit」的真正前置條件不是「樹上紅不紅」，而是
**「你動的檔在不在 `MANIFEST.jsonl` 裡」**——逐筆量過之後，如果全部不在，
你就可以在**別人的檔正在改**的同一個樹上安全地 commit（只要逐檔 `git add`，永不 `git add -A`），
而不會讓任何一格變成假綠。本輪（`ec1a389c8`）就是這個情形：42 個自己的檔全部在管轄範圍外，
唯一紅的 `index-assets` 來源是另一條線正在編輯的 `scripts/check/llama_bench_matrix.py`，
所以 commit 之後索引仍然是 13/14 —— 那**不是**這次 commit 的迴歸，而 commit message 要寫明這件事。

**`docs/` 的「新增」與「修改」差別很大**：新增一份 `docs/*.html`（D5 的白皮書附件）⇒ **不漂移**，
不必重生（實測新增後 `--check` 仍印 `manifest OK: 73 assets`）；但編輯 `CURATED` 裡**已列出的** docs 條目
（例如改某份白皮書的 note）⇒ 會紅，要重生。⇒ 省掉一輪「我加了檔案，先重跑索引吧」的無用重生。

**★ `docs/*.md` 也一樣（2026-09-17 第二次實測）**：`CURATED` 是**逐條列**
（`index_assets.py:266+`，形狀 `("docs/<檔名>", role, loop, replayable, produces_record, note)`）
而**不是 glob** ⇒ **新增 `docs/*.md` 同樣不漂移**。判準（兩步都做）：
`grep -c "<檔名>" agent_harness/engine_loop/MANIFEST.jsonl` 命中 **0** ＋ `--check` 仍印 `OK: N assets`。
反面同樣成立：**要讓一份新 docs 被索引收錄，必須手動加進 `CURATED`**（跟新腳本要加 `role`＋`note` 一樣）
——「新增不漂移」不等於「它被索引了」，這兩件事很容易混。

### 3.3 `CURATED` 的維護

**新增 `scripts/check/` 底下的腳本時要一併加進 `CURATED`**，否則它只會被自動索引、並讓
`index_assets.py` 一直印 `[info] N auto-indexed script(s) still need a role and a note`。
格式 `(path, role, loop, replayable, produces_record, note)`；`role` 用既有分類詞
（`gate` / `probe` / `measure` / `evidence` / `log` / `compare` / `arms` / `runner` / `conclusion` / `index`），
`note` 寫成「它做什麼 ＋ 讀者該拿它做什麼」。**改完要再重生一次 manifest**（它改了 `index_assets.py`
自己的 bytes）。

**`note` 會腐爛，不是只有新腳本要加。** 實例：`powermetrics_gpu_freq_parse.py` 的 note 寫著
`--selftest: 6/6`（實際 9/9）而且**沒提它已經會讀 thermal sampler**——索引在對讀者說「這支不會答散熱」。
**能力變動時要回頭改 note**，並重生 manifest。
另外 `[info] N auto-indexed … need a role and a note` 是**既有欠帳**（某輪 25 支），不是你那一次的錯，
**別為了消掉它去亂填 role**。

### 3.4 `--check` 的收尾：兩條都跑，而且要濾行

`index_assets.py --check` 的結論行被尾端的 `--out` 說明包住，**直接 `tail` 只會看到說明文字**——
要濾 `OK:` / `error` 這類行才看得到結論。兩條都要跑：

```sh
cd agent_harness/engine_loop && python3 index_assets.py --check 2>&1 | grep -E "OK:|error"
python3 agent_harness/engine_loop/memory/build_memory_index.py --check
```

---

## 4. 併行 writer（這個 repo 會同時有多個 session 在寫）

**症狀**：`git status --porcelain --untracked-files=all` 出現**不是你改的** modified／staged 檔。
實例：一個 session 連續寫入 `traces/lessons.jsonl` 與 `PLAN_ENGINE_LOOP_*.md`，而另一個 session 為了
「看一下 by-role 分佈」跑了**不帶 `--check` 的 `index_assets.py`** ⇒ 後者把對方**尚未定稿**的狀態
寫進了 `MANIFEST.jsonl`。另一個實例：某 session 正在執行 E1 的 `git mv`（600+ 個 rename 已 staged）。

**為什麼要獨立一條：失效是安靜的。** `--check` 在下一次重生之前都會說 `manifest OK`，
而 manifest 已經記著一組對方還沒打算停下來的 bytes。此時若直接 `git add -A`，manifest 會被提交在一個
**既不是前一版、也不是對方完成版**的狀態上。

**規則**

1. **動手前跑一次 `git status --porcelain --untracked-files=all`，收尾前再跑一次。**
   出現**不是你改的** modified／staged 檔 ⇒ 有別的 writer。
2. 此時**只新增不修改**：新增 `docs/*.html` 是安全的（見 §3.2）；**不要重生索引、不要 commit**。
   連「刷新 skill 快照」都先擱著——它的 banner 已經聲明不會自動跟上，`SNAPSHOT.jsonl` 也記了 provenance，
   所以延後是**被設計允許**的。

   **★ 但「追加一筆 lesson」不屬於「只新增」，而它會與 §7 直接衝突（2026-09-17 實測）。**
   `traces/*.jsonl` 在 `CURATED` 裡 ⇒ 追加一筆 lesson **會**讓 `index_assets.py --check` 多一筆漂移，
   而這一輪又不能重生索引去消它。於是 §7 的「新知識要落成 lesson 並放進同一個 commit」踩上 §4-2。
   **裁決是延後，不是硬做** —— 硬做等於把對方未定稿的位元組寫進你的 commit，那正是 §4 要防的。
   做法：把 lesson 的內容（連可複製的 `rule`／`because`／`counterexample_observed` 形狀）
   **寫進那一輪的白皮書**，並在 commit message 與白皮書各寫一行
   「lesson 未登錄、理由是樹上有別人未提交的索引位元組」，下一輪樹靜下來時補登。
   判準很好記：**先問「這個檔在 `MANIFEST.jsonl` 裡嗎」** ——
   `docs/*.html`（**新增**）不在 ⇒ 安全；`traces/*.jsonl`、`agent_harness/CONVENTIONS.md`、
   `PLAN_ENGINE_LOOP_*.md`、`.workbuddy/memory/*.md` 在 ⇒ **編輯它們會漂移**
   （§2 的「新增不漂移」只對**新增**成立）。同理，別把「`--check` 說沒漂移」讀成
   「這個檔不在索引裡」——`index_assets.py` 對 `Backup/` 與 `.workbuddy/memory/` 這兩類前綴
   只驗**存在**、不驗 bytes（`VOLATILE_PREFIXES`），所以它們的漂移會改由
   `build_memory_index.py --check` 單獨報出來。

   **★★ 2026-09-18 新增同族的第三種誤讀：`manifest OK` 也不代表那條資產「在版控裡」。**
   `index_assets.py` 對 `scripts/check/*` 驗的是**磁碟上的** existence + bytes + mtime ⇒
   一個**從未 `git add`** 的檔案照樣讓它印
   `manifest OK: N assets, existence + bytes + mtime all agree`，
   而 repo 的其他部分（白皮書、skill、`lessons.jsonl`）**已經在引用它** ——
   **引用在 repo 內是斷的，而每一道閘門都是綠的。**
   實例（同日）：`scripts/check/decode_step_profile.py`（487 行、19 函式、有 `main()`）
   **已在 `MANIFEST.jsonl` 裡註冊**，卻不在版控；`docs/M3_M4_STATUS_2026-09-17.md` 被
   我的白皮書與三份記憶快照引用，也不在版控（`docs/` 的五份 `M3*` 兄弟檔全都已追蹤）。
   修法是純新增 `git add`（零內容改動），不是改檔案。
   **30 秒判別式**：對任何要引用的路徑跑 `git ls-files --error-unmatch <path>`；
   失敗就是「不在版控」，與索引的顏色無關。
   ⇒ 通則：**「索引綠」與「在版控裡」是兩個獨立的軸。**
   （同族的第四種：`git add <path>` 對**已追蹤但被 `.gitignore` 蓋到**的路徑會失敗
   ——`Backup/` 就是。此時要先 `git ls-files --error-unmatch` 確認它**真的已追蹤**再用 `-f`；
   若沒先確認，`-f` 會把一個本來不該進版控的檔推進去。）
3. **不要為了「看一眼」而跑不帶 flag 的 `index_assets.py`**——那就是重建，會覆寫 manifest。
4. 確認對方靜止（`sessions` 表裡它的 `status` 不再是 `working`、相關檔 mtime 不再動）再重生。
5. **「誰在用這台機器」與「誰在改這些檔」是兩個不同的軸 —— 而第一個要讀日誌，不是 `pgrep`。**
   實例（2026-09-17 11:0x，agent_harness 線）：我用
   `pgrep -f "llama-server|llama-bench|decode_sweep|m123_oracle"` 全空，就推論「這是空窗」，
   於是起了 13 GB 的 `llama-server` 做實驗。**結果另一條線的日誌在 10:47–11:1x 寫著：**
   > `§9.18.7 owner 欄 —— 但只寫完，沒建置沒跑（機器被別的 session 佔著）`

   那個 session 就是我；他們的最後一節還明寫「交給下一個空窗期的第一步：`cmake --build …`」。
   ⇒ **「沒有程序在跑」不等於「沒有人在等」。** `pgrep` 只看得到**已發生**的佔用，
   日誌看得到**已宣告**的佔用。

   判斷法：讀 `.workbuddy/memory/YYYY-MM-DD.md` 的**最後一節** —— 這個 repo 的日誌明寫誰在等空窗、
   下一步要做什麼。13 GB 級的操作（`llama-server`、`cmake --build`、量化）尤其要這樣判斷。
   事後補救：停掉 → 確認 8080 空 ＋ `vm_stat` 的 `Pages free` 回到 GB 級 → 在日誌裡記下
   「我佔了哪個時段、已釋放」。

   ★ **反向也成立，而且更貴**：如果對方**正在跑** A/B 對照，你的任何請求都會污染它 ——
   不是「慢一點」，是讓他們的資料作廢。實例（同日 11:2x）：`run_ids_dst_capture.sh` 起了
   llama-server 跑 `ARMS=p25-gputime-churn,p25-slotgpu-churn`，此時**唯一正確的動作是什麼都不送**。
   要驗證自己的東西就找不需要 GPU 的那一半（本輪就只做了字串結構的自測，把行為面留給空窗）。

6. **★ 建置產物是共用資源 ⇒ `cmake --build` 是「對機器上所有正在跑的實驗」的一次寫入**，
   不是本機動作。`src/llama.cpp/build/bin/*.dylib` 與 `llama-server` **受版控**，而且是每個 session
   都會 map 的同一個檔。實例（2026-09-17 11:29）：我把 `lsof -iTCP:8080` 與 `cmake --build` 寫在
   **同一行卻沒有 abort 分支** —— `lsof` 明明印出別條線的 server 在聽 8080，build 還是跑了，
   **蓋掉他們正在 map 的 `libggml-metal`／`libllama`**，於是他們那一輪 A/B 橫跨兩個 build
   （而本 repo 的所有比較都建立在「同一個 build」這個前提上）。唯一正確的形狀是
   **檢查與動作在同一個分支、非空就 abort**：

   ```sh
   # 兩件事都要看：① 港埠有沒有 listener ② 別條線的量測行程
   # ⚠ 行程偵測用 `pgrep -x`（basename 精確比對）—— 不是 `ps ... $2=="llama-server"`，
   #   那個在本機永遠不命中（`comm` 是完整路徑）⇒ 閘門會靜默放行。見 §1.5。
   busy=0
   lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 && busy=1
   pgrep -x llama-server >/dev/null 2>&1 && busy=1
   pgrep -x llama-bench  >/dev/null 2>&1 && busy=1
   pgrep -f '[d]ecode_sweep\.py|[m]123_oracle_gate\.py|[p]rod_matrix\.py|[d]sp_out/driver' >/dev/null 2>&1 && busy=1
   if [ "$busy" = "1" ]; then
       echo "!! build aborted: another line is using the machine"; exit 3
   fi
   cmake --build src/llama.cpp/build --target llama-server -j 8
   ```

   **只印出來不算閘門** —— 那正是 11:29 的錯（lesson `eng-mh-0048`）。
   附帶：連結是**逐檔**進行的，所以在連結進行中取 binary 指紋會做出一個**從未真正跑過的 build 指紋**
   （`decode_sweep.py` 的 `build_fingerprint()` 逐檔算 md5；他們 11:28 那輪記下的是「新 metal ＋ 舊 llama」）
   ⇒ **撕裂的指紋不會給錯數字，它給錯的許可**（lesson `eng-mh-0049`）。

**推論**：`PLAN_ENGINE_LOOP_*.md` 與 `CONVENTIONS.md` 都在 `CURATED` 裡，所以它們一被別人改，
`--check` 就會紅——**紅燈不是你的錯時，不要急著重生把它消掉**，先把 owner 找出來。

**owner 查法**（唯讀查詢，不要寫）：

| 位置 | 內容 |
|---|---|
| `~/.workbuddy/workbuddy.db`（SQLite） | `sessions` 表：`id / cwd / title / status / created_at / updated_at / last_activity_at / deleted_at`。`status` 會是 `working` / `completed` / `archived` |
| `~/.workbuddy/projects/<cwd 轉義>/<sessionId>.jsonl` | 對話本體 |
| `~/.workbuddy/sessions/<pid>.json` | 目前在跑的 session 程序（含 heartbeat） |
| `~/.workbuddy/logs/<日期>/<目錄名>__<hash>.log` | 依目錄分檔的執行日誌 |

**⚠️ 一個實際會咬人的細節**：`last_activity_at` 對**自己**這個 session 而言是「使用者最後送訊息的時間」，
不是「我剛剛做了什麼」。所以別把自己的 session 誤判成別人。判別法：先看 `cwd`，再看
`last_activity_at` 對不對得上你這個 turn 的開始時間。

---

## 5. 憲章層級：D6 與快照

### 5.1 D6 說「不要把記憶複製進 `agent_harness/`」——很反直覺，因為那正是常被要求做的事

`CONVENTIONS.md` D6 明寫「專案記憶只由索引進入 loop，不複製」，`index_assets.py` 的 `CURATED` note
甚至硬寫 "never copied into the harness"。所以當要求是「把 memory/skill 匯入 agent_harness」時，
**不要默默做、也不要默默拒絕**：依憲章 §E（「條文被推翻要標 `superseded_by` 而不是刪掉」）
寫一份 **dated 修訂**——D6 對 loop 的約束不動，只多承認一份非權威快照，並**明寫未完成的義務**
（§E 要求改 CONVENTIONS.md 走一次 PLAN §6.3 的閉環對照；若 `engine_loop/sft_pi/` 與
`harness_engine/` 還不存在，就照實說它們不存在、義務尚未執行，**不可寫成已完成**）。

**改 `CONVENTIONS.md` 時同輪要一起修這些會變成假話的敘述**（實測每一處都會讓 `--check` 紅）：
`index_assets.py` 的 `CURATED` note、同一檔的 auto-indexed note、`build_memory_index.py` docstring 的
"WHY AN INDEX AND NOT A MIRROR" 段、`engine_loop/README.md` §8。
搜法：用**內建 Grep**（不是 bash `grep`）搜 `不複製|never copied|never as a copy|第二個權威`，一次掃完。

### 5.2 快照機制與重生順序

`agent_harness/memory/`（3 檔）與 `agent_harness/skills/`（3 skill）是**非權威 dated 快照**，
唯一用途是跨機器搬運（`agent_harness/scripts/auto_git_push.ps1` 會 `git add agent_harness` 後 push，
而**索引運送的指標到不了那條線**）。banner 插在 YAML frontmatter **之後**；`SNAPSHOT.jsonl` 記
`source_sha256` / `source_bytes` / `source_mtime`。**兩個 `--check` 都不驗快照。**

**★ 為什麼非有這份快照不可（2026-09-17 實查）：`.workbuddy/` 整棵在 `.gitignore:41` 內，而
`git ls-files .workbuddy` 是空的** ⇒ 記憶的**權威本體從來不在版控內**（`git status` 看不到它、
`git add` 也不收它）。所以「把記憶帶去別台機器」與「把記憶推上 GitHub」的唯一通道就是這份快照。
三個推論：(a) 快照刷新不是美化動作，它是唯一的搬運手段，沒跑它就等於記憶沒有離開本機；
(b) 一個**宣稱會定期推送**的機制不等於它在跑——`auto_git_push.ps1` 硬編碼
`D:\alex\flashkv0516\cgcengine_full` / remote `cgc0907` / branch `fusionroutemot`，本機沒有
`pwsh`、crontab 與 LaunchAgents 都無相關條目、連它第一次跑就會建的 `agent_harness/scripts/logs/`
都不存在 ⇒ 它在本機**從未執行**；引用那條推送線之前先重量它（GitHub 上 `fusionroutemot` 停在
`1f3b0a78d`，訊息 `auto(agent_harness): periodic backup 2026-09-16 14:04`）；
(c) **要真的把記憶送上去，就得手動走一次**：`import_harness_snapshot.py` → 重生索引 → commit → push。

**重生順序（比 §3.1 多一層）**：
改記憶／skill 原檔 → `agent_harness/scripts/import_harness_snapshot.py`（刷新快照）→ `build_memory_index.py`
→ `index_assets.py` → commit。

**匯入器已在版控裡，而且清單是推導的（2026-09-16 改）。** 它原本叫
`Backup/import_harness_snapshot.py`——`Backup/` 被 `.gitignore:396` 排除，所以**clone 出來的機器上
根本沒有這支腳本**；而且它用兩個硬編碼清單（`SKILL_NAMES`、`MEM_FILES`）決定要匯入什麼，
所以新增一個 skill、或**單純過了一天**，那份快照都會被**靜默**漏掉（不是報錯，是
「它不在 `SNAPSHOT.jsonl` 裡」，與「那個檔案不存在」同形）。
現在它住 `agent_harness/scripts/`（與 `auto_git_push.ps1` 同目錄——整條線就是「跨機器搬運」），
glob 兩個來源目錄，並在**空清單時拒跑**（空的 `SNAPSHOT.jsonl` 與「探索步驟壞了」同形）。
⇒ 新增 skill **不需要改任何清單**；但若你看到它印出的 `discovered N skill(s)` 少了誰，
那才是訊號。

**★ 2026-09-18 更新：它現在有 `argparse` 了 —— 而且有了 `--check`。**
這一節原本寫「沒有 `argparse`、傳 `--help` 會直接執行匯入」，那個危害已經修掉：

```sh
python3 agent_harness/scripts/import_harness_snapshot.py                # 匯入（寫檔）
python3 agent_harness/scripts/import_harness_snapshot.py --check        # ★ 只驗不寫：0=一致、1=有漂移，逐項印出
python3 agent_harness/scripts/import_harness_snapshot.py --dry-run      # 只顯示會做什麼
python3 agent_harness/scripts/import_harness_snapshot.py --self-test    # 黑箱自測 14/14（含陰性對照）
python3 agent_harness/scripts/import_harness_snapshot.py --help         # 現在真的只是印說明
```

`--check` 分開**五種**漂移，因為處置不同：`[stale]`（原檔已變）、`[new]`／`[removed]`（集合變了）、
`[hand-edited]`（**副本被手改** —— 這在 09-18 之前完全不可觀測，而匯入會靜默蓋掉它）、
`[readme-table]`（兩份 README 的成員表與實際集合不符，**或表格不存在**）。

**為什麼它不進 pre-commit hook（刻意的）**：`.workbuddy/memory/` 含**多 session 共寫**的每日
append-only 日誌 ⇒ 硬閘門會永遠紅。判決沿用 `CONVENTIONS.md:927-928`
（「每日日誌當天必變，永遠紅的閘門等於沒有閘門」）。它走收尾序列 ＋ **定時自動化**。

**它一加進來就紅了兩處，而那兩處就是它存在的理由**（2026-09-18）：
`agent_harness/memory/README.md` 宣告 3 檔（實際 8）、`agent_harness/skills/README.md` 宣告 4 個
skill（實際 5）；memory 那份還寫著「匯入腳本住在 `Backup/`」（2026-09-16 搬家當天就死了）。
⇒ 被憲章記成「三個防線」的機制，**第三道防線自己漂了兩天而沒有任何東西會出聲**。
lesson `eng-gate-0056`；`CONVENTIONS.md` D6 有 2026-09-18 的 dated 修訂。

仍然要自己重算的場合：**`--check` 是唯一權威**，下面那段手算 sha256 現在只是交叉核對
（`--check` 用的是同一組 `source_sha256`，但它同時驗集合與副本內容）：

```sh
python3 - <<'PY'
import json, hashlib
for s in ('agent_harness/skills/SNAPSHOT.jsonl', 'agent_harness/memory/SNAPSHOT.jsonl'):
    ok = tot = 0
    for l in open(s, encoding='utf-8'):
        if not l.strip(): continue
        d = json.loads(l); tot += 1
        ok += hashlib.sha256(open(d['source'], 'rb').read()).hexdigest() == d['source_sha256']
    print(f'{s}: {ok}/{tot} 相符')
PY
```

**順序有一個容易漏的轉折：記憶要先寫、再匯入。** 因為匯入會把 `.workbuddy/memory/*.md` 的
**當下狀態**封進快照——所以「先匯入、後補一節 §EN-xx」會讓快照在**同一個 turn 內**就 stale。
正確序列：**改 skill 原檔 ＋ append 記憶 → `import_harness_snapshot.py` → 逐檔重算 sha256 驗 12/12
→ 重生索引 → commit**。匯入器是幂等的，寫漏了就再跑一次（成本就是再複製 12 檔）。

**skill 原檔一改，快照就 stale ⇒ 不要讓它變成第二個 commit。** 把「改 skill」與「刷新快照」放在
**同一個** commit。（若 skill 是在交付 commit **之後**才被改的，那就難免要多一個 snapshot commit。）

### 5.3 E2 的閉環欠帳（已認領，別讓它變無主）

D6 的 dated 修訂留下「改 CONVENTIONS.md 要走一次 §6.3 的閉環對照、義務記在 E2 頭上」。
它已記在三處：`PLAN_ENGINE_LOOP_2026-09-15.md` §9 的 E2 驗收欄（含認領條件：`distill/` 與
`harness_engine/` 一落地就跑「修訂前後兩版 CONVENTIONS.md」的同一組待答問題，**沒有差異也是結果**；
對照跑完並留有產物之前不得聲稱已結清）、`CONVENTIONS.md` D6 修訂段、`.workbuddy/memory/MEMORY.md`
（**2026-09-17 起 `.workbuddy/memory/` 是 1 索引 ＋ 3 主題檔**：`MEMORY.md` 為索引，
同層 `MEMORY_PERF.md`／`MEMORY_S1.md`／`MEMORY_FACTS.md` 依主題。三支 indexer 都用 **glob**
而不是白名單 ⇒ **新增／刪除任何 `.md` 都要重生索引**，且 commit body 要一併寫清楚動了哪幾檔）。

---

## 6. 收尾序列與 push

### 6.1 收尾常常是兩個（有時三個）commit

**第二個是 resync**：commit 的結果（hash、推送成功與否、gate 的實際輸出）**只有 commit 之後才知道**，
所以「把本輪收尾寫進 `memory/YYYY-MM-DD.md`」結構上不可能滿足 §3.1 的順序。走過的實際序列：

```
RUN_REPLAY_BENCH=0 git commit ...      # 交付本身
# 此時把 commit hash / D5 / gate / push 結果 append 進 .workbuddy/memory/YYYY-MM-DD.md
python3 memory/build_memory_index.py && (cd .. && python3 index_assets.py)   # 順序照舊
git add agent_harness/engine_loop/MANIFEST.jsonl agent_harness/engine_loop/memory/INDEX.jsonl \
  && RUN_REPLAY_BENCH=0 git commit -m 'docs(index): resync ...'
```

**★ resync 不要用 `git add -A`（2026-09-17 實測）。** 這一節原本寫的就是 `git add -A`，但在一個
**有其他 session 正在寫**的樹上，它會把對方的 modified／untracked 一起 stage 進你的 resync commit
—— 而 resync 的整個賣點是「只動 `INDEX.jsonl` ＋ `MANIFEST.jsonl`」。實例（同日 21:1x）：樹上有
`agent_harness/tb_loop/agents/*`（tb_loop 線正在寫）、`Backup/cgc_logs/ids_dst_capture/arms.json`、
`ggml-metal-ops.cpp` ＋ `libggml-metal.0.19.0.dylib`、`closed_loop_out/*`，以及**當下新出現**的
`scripts/check/decode_step_profile.py` —— `git add -A` 會把它們全部收走。
⇒ **逐檔 add**（上面那一行就是答案），並在 commit 前用 `git diff --cached --name-status` 逐列看過。
（同一條理由也適用於**交付** commit：那一輪我是逐檔列出 20 個路徑 add 的。）

**★ D5 重跑失敗的第一嫌疑是記憶體，不是你的改動（同日實測）。** 交付 commit 前用新 tag 重跑 D5，
server 在**載入模型階段**就被 SIGTERM（`run_server.sh: line 1992: … Terminated: 15`）、沒有寫出 summary。
原因是**這台機器 13 GB 級量測的次數上限**：那時 `vm_stat` 的 `Pages free` 只剩 **82 MB**、
`vm.swapusage used` 是 **10351.69 MB / 11264 MB**（Swapins 111695859／Swapouts 131502717）——
同一個 session 內**連跑兩次探針（各載入一次 13 GB）＋ 兩次 gate** 疊出來的 carried swap。
⇒ 判準：跑 D5 之前先看 `vm_stat | awk '/Pages free/'` 與 `sysctl -n vm.swapusage`；
**把 13 GB 級的載入當成一種配額**，一期一兩次，第三次起環境會自己否證結果。
失敗時的正確動作是**具名交代**（哪個 tag、為什麼失敗、引用的替代證據是哪一份、為什麼它與當前樹等價），
不是反覆重跑。同行那一次的做法可照抄：引用 `en-m4m5-inert`（cap 19:36 晚於產物 mtime
19:35:52／19:35:59 ⇒ 不重疊、可歸屬），並說明 `run_server.sh` 在 gate 之後的改動只有註解與
條件式 allowlist 塊（`-n "${VAR:-}"`）⇒ 在 gate 的跑法下 `SERVER_ENV` 逐項不變。

**★ 2026-09-18 補：記憶體吃緊是「它可能失敗」的理由，不是「不要跑」的理由。**
同一天在 `vm_stat` 的 `Pages free` 只有 **179 MB**、`sysctl -n vm.swapusage` 是
`used 9233.00M / total 10240.00M`（≈90%）的情況下，`m123_oracle_gate.py --tag en-orn-ab`
仍然在 **48 s** 內 `PASS`（`comparable=true`、`config_diffs=[]`、M1/M2/M3 各 9/9、
`cap_en-orn-ab.json` 的 `created 04:44:16` 晚於產物 mtime 04:18:53／04:18:55 ⇒ 可歸屬）。
判準：**D5 是數值閘門，不是計時量測** —— 慢不影響它的有效性，只有**真的失敗**（載入期 SIGTERM、
沒有寫出 summary）才走上一段的具名交代。所以「swap 很滿」**不構成**預先跳過 D5 的正當理由
（那正是 `eng-gate-0016` 判為錯的做法）。反過來說：這一條也**不**解除「13 GB 級載入是配額」
的警告 —— 它說的是「先跑一次看結果」，不是「跑幾次都行」。

**不要**試圖把收尾 memory 塞進被提交的那個 commit（做不到），也**不要把漂移留到下一輪**
（下一個人會被 `--check` 的紅字誤導成「上一輪沒重生索引」）。**多一個 3 行的 resync commit 是正確答案。**
它只動 `MANIFEST.jsonl` + `INDEX.jsonl`，沒有 `src/` ⇒ **D5 不必重跑**（`271538f2f`／`e0152777d` 前例），
但 message 要寫明這一點。

**但「寫明」的措辭有對錯，這條與 §2.4／`eng-gate-0016` 的界線很容易踩（2026-09-17 實測）。**
§2.4 判為錯的做法是「**省下 23 秒，然後在 message 裡用一段話解釋為何不跑**」——因為那時**沒有**證據，
理由只能是省時間。resync 不屬於那一類，差別在**同一棵樹上已經有一份完整證據，而且與它同一次推送**：
把理由寫成「本 commit 無 `src/`；它所對應的那棵樹（同一組二進位、同一份 profile）的完整 D5 已在
`<交付 commit>` 上跑過並寫進那個 commit 的 message（附 `--tag`、`comparable`、M1/M2/M3），兩者一起
推上去，可查 `Backup/m123_oracle_gate/summary_<tag>.json`」。
**判準：理由裡必須出現「哪個 commit、哪個 tag、哪些數字」；只能出現「省時間」的說法就是 §2.4 的那一種。**
（若 resync 之前又動了 `agent_harness/` 的快照原檔，那仍不算 `src/`，結論不變。）

**第三個什麼時候出現**：交付 commit 之後又發生了「必須進版控的事」，例如 skill 原檔被改 ⇒ 快照要跟上；
或補寫記憶（§5.2）。**遞迴終止點是「最後那個 resync commit 自己的 hash 是唯一沒被記錄的事實」**，
那是固有的，不必再追一輪。

### 6.2 推送與 FF 驗證

**驗 fast-forward 要用 `ls-remote` 的*真實 SHA*，不要用 tracking ref。**
`git fetch <remote> <branch>` **只寫 `FETCH_HEAD`，不更新 `<remote>/<branch>`**，
所以拿 `<remote>/demo/...` 去算 ahead/behind 可能在拿舊快照下結論。

```sh
git ls-remote <remote> 'refs/heads/demo/sweet-spot-windows-fix'   # 拿遠端真實 SHA
git merge-base --is-ancestor <sha> HEAD && echo FF   # 是祖先 = 純 FF，不需 force
git rev-list --count <sha>..HEAD                     # 會推上去幾個
git push origin demo/sweet-spot-windows-fix
git push cgcengine0907 demo/sweet-spot-windows-fix
# 推完再驗一次：local / origin / cgcengine0907 三個 hash 應該完全相同
```

### 6.3 提交後驗證收尾狀態

```sh
# ── 憲章與 record（權威）────────────────────────────────────────────
python3 agent_harness/shared/check_citations.py                 # 憲章：每條都要有指針或「證據形態」標記
python3 agent_harness/shared/check_shell_cjk.py --root "$PWD/agent_harness"   # $VAR 緊接中文（見下）
python3 agent_harness/engine_loop/traces/validate.py            # 無重複 id
python3 agent_harness/engine_loop/traces/selftest.py            # 必須 10/10
# ── 衍生物（重生 == 磁碟，相對各自的輸入）────────────────────────────
python3 agent_harness/engine_loop/harness_engine/build_memories.py --check
python3 agent_harness/engine_loop/sft_pi/build_sft_pi.py --check
python3 agent_harness/engine_loop/sft_prime/build_sft_prime.py --check
# ★ 快照（agent_harness/memory + skills）。2026-09-18 之前它不被任何 --check 覆蓋 ⇒
#   過期是靜默的，而 auto_git_push.ps1 會忠實地把過期快照推上去。順序：改原檔 → import → 索引。
python3 agent_harness/scripts/import_harness_snapshot.py --check   # 0=一致；1=五種漂移逐項印出
# ── 索引（磁碟 == 索引；★ 不驗「索引涵蓋了它」）────────────────────────
python3 agent_harness/engine_loop/index_assets.py --check
python3 agent_harness/engine_loop/memory/build_memory_index.py --check
python3 agent_harness/engine_loop/wrappers/classify.py --check  # scripts/check/* 全部被分類
# ── 各支工具的自測（沒有自測的閘門等於沒有閘門）──────────────────────
python3 agent_harness/shared/sanitize.py --self-test
python3 agent_harness/shared/check_citations.py --self-test
python3 agent_harness/shared/check_shell_cjk.py --self-test
python3 agent_harness/shared/pack_evidence.py --dry-run
python3 agent_harness/engine_loop/distill/selftest.py
python3 agent_harness/engine_loop/distill/closed_loop_selftest.py
# ── engine loop 的 runner/wrapper 層（PLAN §3）──────────────────────
bash agent_harness/engine_loop/runners/preflight.sh --self-test
bash agent_harness/engine_loop/runners/rebuild.sh --dry-run     # ★ 閘門關著也回 0（它只報告）
env CGC_SERVER_CTX=40960 bash agent_harness/engine_loop/runners/server.sh --check
for w in sweep ab bench gate triage env; do bash agent_harness/engine_loop/wrappers/$w.sh --self-test; done
# ── E4 治理（2026-09-17 新增，4 條）────────────────────────────────
python3 agent_harness/shared/check_stale.py --check                  # 3 個登記項；反向查未登記的 _stale
python3 agent_harness/shared/check_log_policy.py --check --privacy   # 政策 ＋ 去隱私（全掃 2446 個 pack：7.8s）
python3 agent_harness/shared/check_module_split.py --self-test       # 拆分等價證明的陰性對照（2/2）
bash agent_harness/engine_loop/wrappers/gate.sh --dry-run replay_bench_compare.py   # stale banner 的出口
# ── 收尾 ───────────────────────────────────────────────────────────
git status --porcelain --untracked-files=all                    # 必須空（例外見 §1.7）
```

**2026-09-17 之後閘門鏈是 30 條。** 四個容易誤讀的地方：

- **`check_log_policy.py --check` 的綠燈包含一段 ⚠。** 它會印「登記在案的既有例外 12 筆
  （5831279 bytes）」—— 那是 `Backup/cgc_logs` 底下 12 個 `.log` 原文**仍在版控裡**
  （另一條線的量測證據，2026-09-16 以 `git add -f` 加入；`docs/` 與 `*.md` 的引用數是 0）。
  **綠燈的意思是「沒有新違反」，不是「政策成立」。** 要移出得由 owner 決定，見
  `agent_harness/shared/log_policy.json` 的 `grandfathered`。
- **`check_module_split.py` 的原檔修訂是「釘住的」，不是 HEAD**（2026-09-17 修）。
  它預設讀 `shared/knifeedge_split_map.json` 的 `source_rev`。在這之前預設是 `HEAD`——
  而那在**提交之前**是對的、提交之後就開始拿 shim 當原檔比。症狀極具誤導性：
  `declared_edits 指名了原檔裡沒有的名字` ＋ `原樣的複本 -> fail`，讀起來像「拆分壞了」。
  ★ 它還會**掩蓋陰性對照**：兩項都回 rc=2，於是「改一個字元的複本必須失敗」看起來是 ok 的
  （它與陽性對照同歸於盡）。⇒ **凡是拿「當下的 HEAD」當預設輸入的檢查，提交後都要再看一次。**
  紅了的第一個問題是「它讀的是哪一個修訂」，不是「我的內容錯了」；修法一律是改 map
  並重跑 `split_module.py`，不要手改產生出來的模組（lesson `eng-gate-0049`）。

- **`rebuild.sh --dry-run` 在閘門關著時仍然回 0。** 那個閘門的意義是
  「**現在不要建置**」，不是「這支工具壞了」—— 13:18 實測它真的擋下了一次（別條線正在跑
  `m123_oracle_gate` ＋ `decode_sweep`）。所以「它回 0」不等於「可以建置」；
  要問那個問題要看輸出裡的 `閘門: 會擋（…）`。
- **`server.sh --check` 沒有 `CGC_SERVER_CTX` 會故意失敗。** 那不是缺參數，是斷言：
  閉環 prompt 是 ~30.6k token，而 `run_server.sh` 的預設 4096/8192 會**截斷**它 ——
  而截斷後的回覆仍然看起來像回事。所以 ctx 是硬檢查。

**`check_shell_cjk.py` 為什麼存在（這個 repo 特別容易踩）**：`$VAR` 緊接非 ASCII 時，
bash 會把那幾個位元組**吃進變數名** ⇒ `set -u` 下 `unbound variable`，而
**`bash -n` 完全抓不到**（語法合法）。這裡的註解與訊息幾乎全是中文，所以 2026-09-17
**一天之內四個新檔各踩一次**，其中兩次只在特定分支（`n_build > 0`、`missing > 0`）才執行
⇒ 連「跑過一遍」也不保證抓到。寫法上一律用 `${VAR}`。

**`classify.py --check` 會在別人新增 `scripts/check/*` 時變紅** —— 那是對的，不是誤報：
一個沒被分類的腳本就是一個沒有入口的腳本。2026-09-17 它因為另一條線加了兩支而變紅兩次。
處置是把它們加進 `classify.py` 的表（那張表是手 judgment 的部分）。

**`check_citations.py` 驗的是憲章自己的第 7 行**（「每一條都必須能指到今天的具體證據……
指不到的條文不是憲章，是感想」）。它最容易被誤讀成「有沒有檔案」——**不是**：括號裡的
「欄位值」本來就沒有檔案可指（B25 的證據是一次 `grep` 的輸出），所以判準是
「有可解析指針 **或** 一行 `- 證據形態：…`」。缺口從來不是「沒有檔案」，是**沉默**。
改 `CONVENTIONS.md` 就要跑它；沒交代的新條文會讓它失敗。

**三條衍生物的 `--check` 各驗一件事，而且它們和上面兩條驗的不是同一件事。** `index_assets.py
--check` 與 `build_memory_index.py --check` 比的是「磁碟上的 bytes == 索引記的 bytes」——
那只證明**沒有人手改過**那份產物，不證明它**跟得上它的輸入**。要問「它有沒有隨來源前進」
只能重生一次再比。實測（2026-09-17）：`sft_prime` 落後 +47,478 B，而**上面兩條索引全綠**，
因為 `MANIFEST.jsonl` 的 80 筆**一筆都不是** `harness_engine/`、`sft_pi/`、`sft_prime/`、
`distill/` ⇒ 那四個目錄根本不在管轄範圍內。**綠燈與「沒被檢查」在那裡長得一樣。**

```
衍生物                 驗什麼                                   不驗什麼
build_memories --check  重生 == 磁碟（相對 lessons.jsonl）        不驗下游誰還在用它
build_sft_pi --check    重生 == 磁碟 ＋ PROVENANCE 的 invocation   不驗訓練品質
build_sft_prime --check 重生 == 磁碟 ＋ 輸入 sha256 有沒有前進     不驗訓練品質
index_assets --check    磁碟 == 索引（相對索引自己）               ★ 不驗「索引涵蓋了它」
```

**快照的忠實度要逐檔複核，不能看匯入器印了什麼。** 「`imported 7 file(s)`」只證明它寫了 7 個檔，
不證明那 7 個檔的內容對得上原檔（橫幅插入、編碼、截斷都可能走樣）。判準是重算
`source_sha256` 比對——這也是唯一能證明「快照對應原檔哪一版」的方法：

```sh
python3 - <<'EOF'
import json, hashlib
tot = ok = 0
for s in ('agent_harness/memory/SNAPSHOT.jsonl', 'agent_harness/skills/SNAPSHOT.jsonl'):
    for l in open(s, encoding='utf-8'):
        if not l.strip(): continue
        d = json.loads(l); tot += 1
        ok += hashlib.sha256(open(d['source'], 'rb').read()).hexdigest() == d['source_sha256']
print(f'{ok}/{tot} 相符')
EOF
```

（2026-09-16 實測：兩次刷新各 7/7 相符。若不符，第一個要問的是「那份快照是哪一輪寫的」——
`snapshot_date` 欄位就是為此存在的。）

**改到快照的來源時，順序是「改原檔 → 匯入 → 索引」，而且不能手改副本。**
具體地：`~/.workbuddy/skills/*/SKILL.md` 與 `.workbuddy/memory/*.md` 是權威，
`agent_harness/skills/` 與 `agent_harness/memory/` 底下的都是**衍生物**——手改副本會在
下一次匯入時被靜默蓋掉，而那次匯入看起來完全正常。
**`.workbuddy/memory/YYYY-MM-DD.md` 的歷史條目不要改寫**：它們描述的是*當時*的狀態
（例如「腳本住在 `Backup/`」），改寫等於篡改觀測。要更正就在**同一個檔案追加新的一節**
（`## §X.`），並在新節裡說明它取代了哪一節——本 repo 的 §E 對條文就是這個規定，
對日誌同樣適用。

---

## 7. commit 風格（從 body 讀出來的）

- subject 很長（常 90+ 字元，用 `--` 分兩段），型別前綴如 `engine(diag):`、`perf(prefill):`、`fix(ci):`。
- body 逐項列出「為什麼原本是錯的」與量到的數字，並**明寫未跑什麼、為何未跑**，不把 SKIP 折進「通過」。
- **新知識要落成 `traces/lessons.jsonl` 的 lesson 並放進同一個 commit。** 欄位固定
  `type/lesson_id/class/rule/because/counterexample_observed/applies_to/superseded_by`；
  `lesson_id` 取續號（`eng-mh-NNNN` / `eng-gate-NNNN` / `eng-bound-NNNN` / …），
  `validate.py` 會擋重複 id，`selftest.py` 要 10/10。
  **★ 它必須寫在「索引重生」之前（2026-09-18 實測）。** `traces/*.jsonl` 在 `CURATED` 裡 ⇒
  追加一筆就會讓 `index_assets.py --check` 漂移。正確順序是
  **寫 lesson → 跑閘門 → 重生索引（先 `build_memory_index.py`、後 `index_assets.py`）→ 交付 commit**；
  顛倒的話（先重生、後寫 lesson）要多重生一次，而且第一次重生出來的 `MANIFEST.jsonl` 是
  **已經 staged、但內容會立刻過期**的版本（本輪就是這樣多跑了一輪）。
  `git diff -U0 -- traces/lessons.jsonl` 的計數仍必須是 `-0 +N`。
- **`class` 是封閉 enum**：`measurement-hygiene` / `log-forensics` / `diagnosis` / `gate-integrity` /
  `source-reading` / `honest-bounds` / `performance` / `smoke`。
  自創 class（曾試 `error-path-integrity`）會被擋下並印出整份可選清單。
  **挑最接近的既值，不要為了語意精確去擴 enum。**
- **兩個欄位陷阱（2026-09-16 各踩一次）**：
  (1) **`superseded_by` 是 required（可為 null）**——「打算留 null 所以省略」會直接 FAIL
  （`missing required field 'superseded_by'`）。用程式 append 時要顯式寫 `"superseded_by": null`。
  (2) **`applies_to` 的每一項是「檔案路徑」**，不是主題標籤。寫 `"s1-divergence"` 之類會產生
  `WARN: applies_to path does not exist`（validate 仍回 OK，所以很容易漏掉 12 條 WARN 就交出去）。
  慣例值是 `agent_harness/CONVENTIONS.md`、`scripts/run_server.sh`、`src/...cpp` 這種真實路徑。
  **交出去前看一眼 warning 數量**：`-> OK` 不代表乾淨。
- **★ 不要把 JSON 寫在 shell heredoc 裡去 append lesson（第三個陷阱，2026-09-17 實測）。**
  `python3 - <<'PY'` 的 `<<'PY'` 看似有引號保護，但字串常值裡的 `\n` **會被吃掉** ⇒
  Python 得到的是「字串在中途斷行」⇒ `SyntaxError: unterminated string literal (detected at line N)`，
  而 N 指向字串**開頭**那一行，讀起來像「引號沒收」而不是「跳脫被吃」。
  正確形狀：**用編輯器／Write 工具把腳本寫成一個實體檔（`/tmp/xxx.py`），再 `python3 /tmp/xxx.py`。**
  寫完要**逐欄讀回核對**（`assert` 那個 id 在 ＋ 把 `applies_to`／`class` 印出來），
  不要只信腳本自己印的 `appended [...]`。
  （好消息：這個失敗是**原子**的——它在 `open(P,'a')` 之前就炸，所以不會留下半筆；
  判準是 `validate.py` 的記錄數沒變。）
- **改 append-only 的共享檔（`lessons.jsonl`）之後，要驗「沒有弄掉別人的行」——判準是 diff 的計數。**
  `git diff -U0 -- agent_harness/engine_loop/traces/lessons.jsonl` **必須是 `-0 +N`**
  （新增 N 筆、移除 0 筆）。用 **python 逐行解析**那個 diff，不要用 bash `grep` 過濾（§1.3 會靜默漏行）。
  要就地改一個字（例如錯字）就**只重寫那一行**再驗一次計數——整檔 read-modify-write 有
  蓋掉別條線同時 append 的風險（該檔同一小時被兩條線各寫一筆，見 lesson `eng-gate-0052` 的由來）。

---

## 8. 附錄：歷史輪次索引

日期化的實測紀錄已按主題併入上文；這裡只留「哪一輪 ＝ 哪個 commit ＝ 主題」的對照。

| 輪 | commit | 主題（本檔對應節） |
|---|---|---|
| 二 | `271538f2f` | D5 看 `src/`；`CURATED` note 腐爛；bash `grep` 靜默失效 → §2.4 §3.3 §1.3 |
| 三 | `fd246c756` `a22328adb` | resync commit 的誕生；`ls-remote` FF；zsh word-split；`cmake --build` 驗 check 8 → §6.1 §6.2 §1.4 §2.9 |
| 四 | `bc9a184e2` `8af8c99bd` | 註解 vs D5；多行巨集 `__LINE__`；`usedImages`；`imageOffset`；check 8 漏 `.m`；class enum → §2.8 §2.11 §2.7 §7 |
| 五 | `b1c3f75c1` `efeade7e2` | D5 dump 無指紋；沒有 `timeout`；零成本探針的界線 → §2.6 §1.5 |
| 六 | `89dd85b23` `2a71b332f` | `ORACLE_PINNED_ENV`；建置戳記造成的假 modified；`Backup/` 不算交付 → §2.5 §2.7 §1.6 |
| 七 | `e688f346e` … | D6 修訂；快照機制；`agent_harness/` 刻意不自足；檔案層整併 ≠ merge → §5 §4 |
| 八 | `18fa0e19b` `3155e7c9a` | `src/` 實驗的可證還原；D5 的兩半 → §2.10 §2.4 |

**相關 skill**：`cgc-decode-attribution`（decode 相位歸因）、`cgc-prefill-thermal-delivery`
（prefill t/s 的條件式交付）、`cgc-whitepaper-delivery`（white paper 版式與交付）。
