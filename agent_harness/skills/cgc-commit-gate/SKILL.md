---
name: cgc-commit-gate
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）提交 commit 時，正確通過 pre-commit 閘門並滿足 D5 的完整流程與陷阱。當要在該 repo commit、被 pre-commit hook 擋下、不確定要不要跑 replay benchmark / M1/M2/M3 oracle、或 `check_build_tracked.sh` 印 OK 卻全是 SKIP 時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-commit-gate/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-17 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
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

### 1.4 zsh 不對未加引號的參數做 word-split

`for pair in "a b"; do set -- $pair; echo $2; done` 在 zsh 下 **`$2` 是空的**，會得到
`fatal: Not a valid object name`，看起來像「那個 SHA 不存在」——**是 shell 把參數吃掉了，不是 git 的問題**。
直接寫成兩個獨立命令，別用迴圈拆字串。

**同一族：`cd` 寫在 `if` 區塊（或 `&&` 串）裡面會外洩到後續指令。** 實測
`if [ -d X ]; then cd X && ... ; fi; ls agent_harness/scripts/` ⇒ 那個 `ls` 跑在 `X` 底下，
於是列出**另一個 repo** 的內容，看起來像「本 repo 的檔案不見了」。診斷腳本時要用**絕對路徑**，
不要把 `cd` 放進條件區塊——它產出的是一份**自信的錯誤清單**，而不是報錯。

### 1.5 macOS 沒有 `timeout`／`gtimeout`；`ps` 被擋但 `pgrep` 可用

要做「啟動 30 秒後自動收掉」的 smoke，用背景 PID ＋ 有界 sleep：

```sh
bash -c 'CGC_SERVER_PROFILE=prefill250 CGC_SERVER_UBATCH=4096 bash scripts/run_server.sh \
           >/tmp/p3.txt 2>&1 & P=$!; sleep 32; kill -TERM $P; sleep 5;
         kill -0 $P 2>/dev/null && echo STILL_ALIVE || echo EXITED'
pgrep -fl llama-server      # 必須空
```

判準是 log 裡同時有 `model loaded` 與 `listening on http://0.0.0.0:8080`，而且 `SIGTERM`／`SIGINT`
走優雅關閉（`[CGC] Received SIGINT — initiating graceful shutdown` ⇒ 不洩漏 Metal buffer；
`kill -9` 才會）。

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
正常收進去（也會出現在 `git status`）；只有**新增**的才需要 `git add -f`。實測的形狀是
`Backup/analyze_capture_nodes.py` 之類顯示 ` M`，而同一目錄的 `compare_slot_owner.py` 完全不顯示。
⇒ 判準：`git ls-files Backup/ | head` 看它是不是已追蹤。一個**已經存在的**驅動改了會被提交、
一個**新寫的**不會——所以「我在 Backup/ 修好了腳本」與「它進得了 commit」是兩件事。
（本 repo 的慣例是儀器驅動要進版控，所以新的那幾支就用 `git add -f` 收進來，並在 message 揭露。）

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

**`docs/` 的「新增」與「修改」差別很大**：新增一份 `docs/*.html`（D5 的白皮書附件）⇒ **不漂移**，
不必重生（實測新增後 `--check` 仍印 `manifest OK: 73 assets`）；但編輯 `CURATED` 裡**已列出的** docs 條目
（例如改某份白皮書的 note）⇒ 會紅，要重生。⇒ 省掉一輪「我加了檔案，先重跑索引吧」的無用重生。

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
3. **不要為了「看一眼」而跑不帶 flag 的 `index_assets.py`**——那就是重建，會覆寫 manifest。
4. 確認對方靜止（`sessions` 表裡它的 `status` 不再是 `working`、相關檔 mtime 不再動）再重生。

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
git add -A && RUN_REPLAY_BENCH=0 git commit -m 'docs(index): resync ...'
```

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
python3 agent_harness/engine_loop/traces/validate.py            # 無重複 id
python3 agent_harness/engine_loop/traces/selftest.py            # 必須 10/10
cd agent_harness/engine_loop && python3 index_assets.py --check 2>&1 | grep -E "OK:|error"
python3 agent_harness/engine_loop/memory/build_memory_index.py --check
git status --porcelain --untracked-files=all                    # 必須空（例外見 §1.7）
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
