---
name: cgc-commit-gate
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）提交 commit 時，正確通過 pre-commit 閘門並滿足 D5 的完整流程與陷阱。當要在該 repo commit、被 pre-commit hook 擋下、不確定要不要跑 replay benchmark / M1/M2/M3 oracle、或 `check_build_tracked.sh` 印 OK 卻全是 SKIP 時使用。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-commit-gate/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-16 手動複製進 repo，唯一目的是讓 `agent_harness/` 底下的內容
> 能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `Backup/import_harness_snapshot.py`。

# flashkv-devserver 提交閘門（流程與陷阱）

專案：`/Users/alexchuang/Documents/flashkv-devserver`
（**是 git worktree**，`.git` 是一個檔案 → `.../flashkv0516/.git/worktrees/flashkv-devserver`）

## 30 秒版：標準提交指令

```sh
cd /Users/alexchuang/Documents/flashkv-devserver
# 0) 引擎改動 → 先跑 D5 的數值閘門（~23 s，自己起 server）
python3 scripts/check/m123_oracle_gate.py --tag <標籤>
# 0b) 這一輪若做過 git stash 換臂的 A/B，或動過跨 dylib 邊界的 struct：
#     build tree 現在屬於「另一臂」——先重建，再跑任何測試（見陷阱五）
# 1) 若有動任何被索引的檔案 → 依序重生索引（不確定就先跑 --check，它會印出漂移）
python3 agent_harness/engine_loop/memory/build_memory_index.py   # 先：寫 INDEX.jsonl
cd agent_harness/engine_loop && python3 index_assets.py && cd -   # 後：MANIFEST 記錄 INDEX 的 bytes/mtime
# 2) 預演閘門（不要盲目提交）
BIN_DIR='src/llama.cpp/build/bin' RUN_REPLAY_BENCH=0 bash scripts/check_build_tracked.sh --repo "$PWD"
# 3) 提交（RUN_REPLAY_BENCH=0 必須在 commit 指令的環境裡 —— hook 繼承環境）
RUN_REPLAY_BENCH=0 git commit -F <message 檔>
```

## 陷阱一：`BIN_DIR` 沒指定 → 閘門全 SKIP 卻印 OK

`check_build_tracked.sh` 預設看 `build/bin`，**本 repo 的產物在 `src/llama.cpp/build/bin`**，
所以預設會讓追蹤／rpath／deadlock／原始碼↔binary 同步／replay 全部 SKIP，最後仍印一句
無條件的 `OK`。這是 B7（可被跳過的閘門不是閘門）的實例。
**`pre-commit` hook 已經幫你帶了 `BIN_DIR='src/llama.cpp/build/bin'`**，所以你手動跑要自己帶。

## 陷阱二：hook 落點在 worktree 之外，寫死路徑會偽 PASS

`git rev-parse --git-dir` → `/Users/alexchuang/Documents/flashkv0516/.git/worktrees/flashkv-devserver`，
而 hook 在 **common dir**：`/Users/alexchuang/Documents/flashkv0516/.git/hooks/pre-commit`。
它 `export BIN_DIR=...` 後 `exec "$REPO_ROOT/scripts/check_build_tracked.sh"`，其中
`REPO_ROOT="$(git rev-parse --show-toplevel)"` **在執行時解析**——刻意不寫死，否則從別的 worktree
commit 時會去驗錯的 repo（偽 PASS）。自己重裝 hook 時要保留這個特性。

## 陷阱三：`RUN_REPLAY_BENCH` 的預設有兩層，不要搞混

- 腳本自身預設：`check_build_tracked.sh:559` 是 `RUN_REPLAY_BENCH:-1` → **會跑**（然後因為沒有
  8080 上的活 server 而 FAIL）。
- 專案慣例 **D2**（`agent_harness/CONVENTIONS.md:447`）：**`RUN_REPLAY_BENCH=0` 一律預設**。
  D2 給的理由是 **`.replay_bench_baseline.json` 已 stale**（2026-09-08，commit `37305dbc6` 起沒動），
  跑它等於拿舊基線比新程式。

⇒ 提交時要 `RUN_REPLAY_BENCH=0 git commit ...`（hook 繼承環境）。
引用時要說是「依 D2」，不可說成「腳本預設」。

## 陷阱四（最貴的）：D5 的閘門很便宜，別用結構論證取代它

`D5` 要求每次 commit 前跑 **llama-bench + M1/M2/M3 + 最新 M2 oracle**。
`scripts/check/m123_oracle_gate.py` **只花約 23 秒**，而且會**自己**透過 `run_server.sh` 起 server
（所以 profile／env allowlist／load path 都是生產那套，不是會漂移的手寫 argv），
送同一個決定性 probe，與最新 M2 oracle 比對 logits：

```
M1 numeric identity 9/9   M2 decision agreement 9/9   M3 top-k set 9/9
(aux) full_fnv1a64 9/9    cross-tab 9/0/0/0
comparable=true  config_diffs=[]   ← 兩端配置由 run_server.sh CGC_DUMP_ENV=1 解析
```

它把「這個改動應該不影響數值」從推論變成測量。**判斷「這個 repo 的慣例要不要跑它，
要讀前幾個 commit 的 body，不是讀 CONVENTIONS.md。** 動到 engine 的前幾個 commit
（`53b7df693`、`25d61278f`）都在 message 裡報告了 M1/M2/M3 數字 ⇒ 跳過＝打破一個正在被遵守的慣例。
（教訓：`eng-gate-0016`。）

## 陷阱五：`git stash` 換臂之後，build tree 屬於**另一臂**；而 struct 加成員是連結器看不見的 ABI 破壞

做 A/B 的常見手法是 `git stash` 掉修法 → 重建 → 跑舊臂 → `git stash pop`。**pop 不會重建**，
所以 `src/llama.cpp/build/bin` 留著舊臂的產物。若這一輪又在跨 dylib 邊界的 struct 加了成員，
那就同時中了第二層：`llama_expert_cache` 這種以**指標**傳遞的 struct，新增 8 bytes 會把後續每個
成員的偏移整體推移，而**連結器只看得見符號、看不見成員偏移**——不會是連結錯誤，而是
「照新標頭編的 harness 連上照舊佈局編的庫」，症狀是**崩在函式庫裡**：

```
SIGSEGV / EXC_BAD_ACCESS (KERN_INVALID_ADDRESS at 0x0)
  libllama.0.0.239.dylib  llama_expert_cache_ensure_batch + 1068
```

2026-09-16 實例：新增 `n_hit_adopted_queued` 讓 `ever_loaded` 從 `0x608` 移到 `0x610`；
舊庫把 `0x608` 讀成 `n_slot_table_unchanged`（`size_t`，值 0）當成 data pointer 去索引。

**診斷四步，一分鐘內收斂（不要先去改測試——我的第一反應就是回頭審測試的初始化，
那條路會一路改到看不出問題，因為測試本身沒錯，錯的是它連到的庫）：**

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

## 誰在守著 commit：`check_build_tracked.sh`，不是 e2e gate

`precommit_e2e_gate.sh` **刻意沒有**接進 hook（`00b70bebc` 明載
「Deliberately NOT done here: whether the hook should invoke precommit_e2e_gate.sh …
It needs an explicit decision」）。所以缺少 e2e 是**既定決定**，不是遺漏——commit message 要這樣寫。

## 索引重生順序（同輪踩過兩次）

`INDEX.jsonl` 與 `MANIFEST.jsonl` 是 **byte 級快照，不是內容摘要**，改一個字就漂移。

1. **先** `memory/build_memory_index.py`（寫 `memory/INDEX.jsonl`）
2. **後** `agent_harness/engine_loop/index_assets.py`（`MANIFEST.jsonl` 會記錄 `INDEX.jsonl` 的
   bytes/mtime）

順序顛倒 → `index_assets --check` 報漂移；而且**只重生 manifest 不會修**，必須兩者都重跑。
推論：**memory 寫完要放在索引重生之前**，否則會在最後一刻把已提交的索引弄成 stale。
`index_assets.py` **不要加 `--out`**（相對路徑會寫出第二份 manifest）。

**這條要在「每一次」重生時按序跑，不是只有第一輪（2026-09-16 第四次踩）。** 最常見的犯法是在同一輪裡
**第二次**重生時把兩個指令的順序寫反（本輪：改完 `lessons.jsonl` 的路徑後，先跑 `index_assets.py`
才跑 `build_memory_index.py`）。漂移的**簽名很好認**——是 **mtime** 而不是 bytes：

```
agent_harness/engine_loop/memory/INDEX.jsonl: mtime: manifest '2026-09-16T11:15:00' vs disk '2026-09-16T11:15:15'
```

因為 `build_memory_index.py` **每次都會重寫 `INDEX.jsonl`，即使內容一字不變**（所以 bytes 相同、
只有 mtime 動）。看到 `INDEX.jsonl` 的 mtime 漂移，就是這一條，不是內容問題：
**重跑一次「先 `build_memory_index.py` 再 `index_assets.py`」即可**，不要去找內容差異。
檢查的收尾永遠是兩條 `--check` 都跑（`index_assets.py --check` 的錯誤訊息會被尾端那段
`--out` 說明包住，直接 `tail` 會只看到說明文字而看不到結論——要濾 `OK:` / `error` 這兩類行）。

**推論的推論（2026-09-16 第三輪實測）：commit 之後才寫的 memory，會需要第二個「resync」commit。**
commit 的結果（hash、推送成功與否、gate 的實際輸出）**只有 commit 之後才知道**，
所以「把本輪收尾寫進 `memory/YYYY-MM-DD.md`」這句話**結構上不可能**滿足上面那條順序。走過的實際序列：

```
RUN_REPLAY_BENCH=0 git commit ...      # fd246c756
# 此時把 commit hash / D5 / gate / push 結果 append 進 .workbuddy/memory/2026-09-16.md
python3 memory/build_memory_index.py && (cd .. && python3 index_assets.py)   # 順序照舊
git add -A && RUN_REPLAY_BENCH=0 git commit -m 'docs(index): resync ...'     # a22328adb
```

不要試圖把收尾 memory 塞進被提交的那個 commit（做不到），也不要把漂移留到下一輪
（下一個人會被 `--check` 的紅字誤導成「上一輪沒重生索引」）。**多一個 3 行的 resync commit 是正確答案。**
它只動 `MANIFEST.jsonl` + `INDEX.jsonl` 兩個檔，沒有 `src/`，所以 **D5 不必重跑**（`271538f2f`／
`e0152777d` 的前例），但 message 要寫明這一點。

**哪些檔案被索引（`--check` 會紅的）** —— 不只 `traces/*.jsonl` 與 `.workbuddy/memory/*.md`：
2026-09-16 那一輪改了 `docs/PREFILL250_THERMAL_TRANSIENT_20260916.html`、
`agent_harness/CONVENTIONS.md`、`scripts/check/powermetrics_gpu_freq.sh` 與 `..._parse.py`，
**以及 `index_assets.py` 自己**（改策展列也會改自己的 bytes），每一個都讓 `--check` 紅。
**不確定就跑 `index_assets.py --check`，它會直接印出漂移的 path 與 bytes 變化** —— 它就是為此存在的，
比猜便宜。

**新增 `scripts/check/` 底下的腳本時，要一併加進 `index_assets.py` 的 `CURATED` 列**，
否則它只會被自動索引、並讓 `index_assets.py` 一直印
`[info] N auto-indexed script(s) still need a role and a note`。
格式是 `(path, role, loop, replayable, produces_record, note)`；
`role` 用該檔的分類詞（`gate` / `probe` / `measure` / `evidence` / `log` / `compare` / `arms` /
`runner` / `conclusion` / `index`），`note` 寫成「它做什麼 + 讀者該拿它做什麼」。
改完 `CURATED` 要**再重生一次 manifest**（它改了 `index_assets.py` 自己的 bytes）。

## 檢查 8：原始碼 ↔ binary 同步

含 `src/` 原始碼的 commit，必須把對應的 build 產物一起 staged，且**binary 比 staged 原始碼新**。
例：改 `src/llama.cpp/ggml/src/ggml-backend.cpp` → 要 stage
`src/llama.cpp/build/bin/libggml-base.0.19.0.dylib`（`libggml-base` 是 backend registry／排程器
所在，`ggml-backend.cpp` 編在裡面）。
注意 `llama-server` 的 mtime **不必**更新：它動態連 `libggml-base.dylib`，
install name 不變就不需 relink。

已知對應（2026-09-16 實測）：

| 改到的原始碼 | 要一起 staged 的產物 |
|---|---|
| `src/llama.cpp/ggml/src/ggml-backend.cpp` | `src/llama.cpp/build/bin/libggml-base.0.19.0.dylib` |
| `src/llama.cpp/src/llama-expert-cache.cpp` / `.h` | `src/llama.cpp/build/bin/libllama.0.0.239.dylib` |

只改 `.h` 也要重建並 stage 同一個 dylib（成員偏移變了，見陷阱五）。
`git status --short` 有時因為 stat cache 沒把它列出來——用 `git diff --stat` 確認，
它會直接印 `Bin 3136592 -> 3136592 bytes`。

## 不會進 commit 的東西（別浪費時間找）

`.gitignore`：`.workbuddy/`（含 `memory/`）、`Backup/`（commit message 草稿與所有證據檔）、`bin/`
（`TurboFieldfareCLI-*`）、`__pycache__/`。所以 **commit message 草稿放 `Backup/` 是安全的**，
但它不會被提交——message 內容要真的寫進 commit。

## 提交後驗證收尾狀態

```sh
python3 agent_harness/engine_loop/traces/validate.py            # episodes/decisions/lessons，無重複 id
python3 agent_harness/engine_loop/traces/selftest.py            # 必須 10/10
python3 agent_harness/engine_loop/index_assets.py --check       # <N> assets
python3 agent_harness/engine_loop/memory/build_memory_index.py --check
git status --porcelain --untracked-files=all                    # 必須空
```

**最後那條會因為新增的 C++ 測試編譯產物而不空。** `scripts/check/*.cpp` 若會被編成同名無副檔名的
執行檔（目前只有 `expert_cache_ensure_batch_order.cpp`），要在 `.gitignore` 加**該檔的絕對路徑**
（`/scripts/check/<name>`），不要用 `scripts/check/*` 這種會吞掉 source 的樣式。
`.gitignore` 本身**不在**索引裡（改它不會讓 `--check` 紅）。

## 本 repo 的 commit 風格（從 body 讀出來的）

- subject 很長（常 90+ 字元，用 `--` 分兩段），型別前綴如 `engine(diag):`、`perf(prefill):`、`fix(ci):`。
- body 逐項列出「為什麼原本是錯的」與量到的數字，並**明寫未跑什麼、為何未跑**，
  不把 SKIP 折進「通過」。
- 新知識要落成 `traces/lessons.jsonl` 的 lesson，**並放進同一個 commit**（`a22ebe88e`、`2b284667f`
  都是這樣）。欄位固定為 `type/lesson_id/class/rule/because/counterexample_observed/applies_to/superseded_by`，
  `lesson_id` 取 `eng-gate-NNNN` 續號（`validate.py` 會擋重複 id，`selftest.py` 要有 10/10）。
  `class` 的既有詞表：`gate-integrity` / `measurement-hygiene` / `diagnosis` / `log-forensics` /
  `source-reading` / `honest-bounds` / `smoke`。

## 實測補充（2026-09-16 第二輪，`271538f2f`）

**D5 的實務判準是「有沒有動 `src/`」，不是「有沒有動 engine 邏輯」。** 逐個 commit 數過：
`a22ebe88e` / `6ceeb281b` / `57bd90801` / `e0152777d` 全部 **0 個 `src/` 檔案**，四者的 message
都**沒有**報告 M1/M2/M3；報告它的是動到 `src/` 的 `2b284667f` 與 `2162e8cbd`。
注意 `e0152777d`（subject 就叫 *the D5 gate is cheap -- run it*）本身 0 個 src 檔——它是
`2162e8cbd` 的補記 commit，src 改動在前一個。
⇒ 一個純 doc/腳本的 commit 技術上可以不跑，但**跑它只花 23 秒**，而「省下 23 秒然後在 message 裡
用一段話解釋為何不跑」是這個 repo 已經明確判為錯的做法（`eng-gate-0016`）。跑了就寫出來。

**`index_assets.py` 的 `CURATED` note 會腐爛，不是只有新腳本要加。** 本輪發現
`powermetrics_gpu_freq_parse.py` 的 note 寫著`--selftest: 6/6`（實際 9/9）而且**沒提它已經會讀
thermal sampler**——也就是索引在對讀者說「這支不會答散熱」。能力變動時要回頭改 note，
而且**改完要再重生一次 manifest**（`index_assets.py` 索引自己）。
另外 `[info] N auto-indexed script(s) still need a role and a note` 是**既有欠帳**（本輪 25 支），
不是你那一次的錯，別為了消掉它去亂填 role。

**`grep` 在這個 sandbox 會靜默失效——用內建 Grep 工具，不要用 bash 的 `grep`。**
本輪實測：`grep -n "SIGTERM\|SIGINT" <file>`（BSD BRE 把 `\|` 當字面）與 `grep -n "^## "` 都回空，
exit code 在 0/1 之間不一致，而檔案裡**明明有**那些字串（`sed -n` 與內建 Grep 都看得到）。
一個「查不到＝沒有」的假陰性會直接變成結論，這與 B7／B12 同族。要搜內容就呼叫 Grep 工具。

## 實測補充（2026-09-16 第三輪，`fd246c756` + `a22328adb`）

**D5 的兩個實務細節：**
- 它會**自己起 server**（`run_server.sh`），所以**先確認 8080 沒人**：
  `lsof -nP -iTCP:8080 -sTCP:LISTEN` 與 `pgrep -fl llama-server` 都要空。
- summary 落在 `Backup/m123_oracle_gate/summary_<tag>.json`，**tag 自己取**（本輪 `blockerstats`）。
  輸出是 `M1 9/9 · M2 9/9 · M3 9/9 · fnv1a64 9/9 · cross-tab 9/0/0/0 · comparable=true`，
  全程約 40–60 s（含載入模型 8–9 s，不是純 23 s）。

**驗證 fast-forward 要用 `ls-remote` 的**真實 SHA**，不要用 tracking ref。**
`git fetch <remote> <branch>` **只寫 `FETCH_HEAD`，不更新 `<remote>/<branch>`**，
所以拿 `<remote>/demo/...` 去算 ahead/behind 可能在拿舊快照下結論。正確做法：

```sh
git ls-remote <remote> 'refs/heads/demo/sweet-spot-windows-fix'   # 拿遠端的真實 SHA
git merge-base --is-ancestor <sha> HEAD && echo FF   # 是祖先 = 純 FF，不需要 force
git rev-list --count <sha>..HEAD                     # 會推上去幾個
# 推完再驗一次：三個 hash 應該完全相同
```

**zsh 的陷阱**：`for pair in "a b"; do set -- $pair; echo $2; done` 在 zsh 下 **`$2` 是空的**
（zsh 不對未加引號的參數做 word-split），會得到 `fatal: Not a valid object name` 而看起來像
「那個 SHA 不存在」——**是 shell 把參數吃掉了，不是 git 的問題**。直接寫成兩個獨立命令，別用迴圈拆字串。

**`cmake --build` 是驗證 check 8 的最省事方法**：`cd src/llama.cpp && cmake --build build --target llama -j8`，
若印 `Built target llama` 而**沒有任何編譯行**，就證明磁碟上的 dylib 確實由當前原始碼產生
（比對 mtime 更強，且訊息可貼進 commit body）。本輪 md5 `4c4d811ae66c69ea1d794b95a54c5ecc`。

## 實測補充（2026-09-16 第四輪，`bc9a184e2` + `8af8c99bd`）

**「改了註解要不要重跑 D5」有確定答案：看產物 md5，不要靠推理。**
註解改動不影響 codegen，但會**位移行號**，而 `__LINE__` 與 DWARF 行表都在二進位裡 ⇒ md5 會變、
D5 就必須重跑。反過來，只要重建後 md5 與 D5 當時那份相同，就可以合法地不重跑並在 message 寫明理由。
做法：**先把註解壓回原本的行數**再重建，然後比對 md5。本輪結果：

```
md5 before/after comment-restore rebuild = dd4d1bc4fbd8b1528eb73d813aae8dd4   (byte-identical)
```

⇒ message 寫「D5 未重跑，理由是產物逐位元相同」＋貼出 md5。這比「重跑一次然後說它 pass」
資訊量更高，因為它同時證明了**沒有東西改變**。

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

**判斷「兩臂之間到底換了哪幾個 image」要用 crash report 的 `usedImages` UUID。**
每份 `.ips` 都記錄了當次載入的每個 image 的路徑與 UUID（本地檔用 `dwarfdump --uuid` 對照）。
把兩份報告的 `usedImages` 做 diff，就能證明一次 A/B 是不是單變數——這是**最便宜**的單變數證明，
而且它在本輪推翻了我兩次直覺（一次把沒改的庫當成變了，一次把其實沒變的庫當成混淆項）。
注意 `libobjc.A.dylib` 會因為 faulting thread 停在 `objc_msgSend` 內而出現／消失，
那是清單呈現差異，不是載入差異。

**`imageOffset` 才是反組譯要用的偏移，而且前提是「當時那份二進位」還在。**
`.ips` 的 `symbolLocation` 是相對**符號起點**，`imageOffset` 才是相對 **image base**。
所以**重建前先備份舊產物**（本輪 `Backup/pre_mtp_rebuild_<date>/` ＋ `MANIFEST.txt` 記 md5/size/mtime），
否則事後無法證明根因。這是 B17 的實務前提。

**check 8 原本漏了 `.m`（本輪已修，但要知道它曾經長什麼樣）。**
`check_build_tracked.sh` 的 case 樣式列了 `*src/*.cpp|*.h|*.c|*.mm|*.metal`——**獨漏 `.m`**。
後果：改 `ggml/src/ggml-metal/ggml-metal-context.m` 會被判成
「8 無 llama 原始碼變更（僅 doc/腳本/產物）」＝假 PASS，並同時跳掉「產物有沒有 staged」與
「binary 比原始碼新」兩條。已補 `*src/*.m`。**新增任何會被編進 binary 的副檔名時要同步這個樣式**，
並用陽性對照驗它（餵一份含該副檔名的假 staged 清單，看 `staged_src` 是不是 0）。

**`traces/validate.py` 的 `class` 是封閉 enum。**
可用值：`measurement-hygiene` / `log-forensics` / `diagnosis` / `gate-integrity` /
`source-reading` / `honest-bounds` / `performance` / `smoke`。
自創 class（本輪 `error-path-integrity`）會被擋下並印出整份可選清單。挑最接近的既值
（「錯誤路徑的診斷判準」→ `diagnosis`），不要為了語意精確去擴 enum。

**收尾永遠是兩個 commit。** 第二個是 resync（只動 `MANIFEST.jsonl` + `INDEX.jsonl`），
因為「commit 之後才知道的事實」結構上不可能塞進被提交的那個 commit（`8af8c99bd` 是本輪實例）。

## 實測補充（2026-09-16 第五輪，`b1c3f75c1` + `efeade7e2`）

**D5 的 dump 沒有 binary 指紋，所以「這份 PASS 是哪個 build 跑的」事後無法證明。**
`Backup/m123_oracle_gate/cap_<tag>.json` 只記 `created` / `profile` / `resolved.{ARG,ENV,CGCENV}`
—— **沒有 md5、沒有 UUID**。因此若 gate 是在 `cmake --build` 收尾期間起的，那份 PASS
就不可歸屬（本輪 12:12 的 `flagalign` 與 `libllama-*-impl.dylib` 的 12:12 寫入重疊）。
處置：**重建完成後才跑 gate**，並把 `cap_<tag>.json` 的 `created` 與最新產物的 mtime 對一眼；
一旦重疊就**換一個新 tag 重跑，只引用新的那一份**（本輪改用 12:18 的 `flagalign2`，
舊的那份不引用也不刪，留在 `Backup/`）。成本 25 s，比事後解釋便宜。

**macOS 沒有 `timeout`，也沒有 `gtimeout`（實測 `command not found`）。**
要做「啟動 30 秒後自動收掉」的 smoke，用背景 PID + 有界 sleep：
```sh
bash -c 'CGC_SERVER_PROFILE=prefill250 CGC_SERVER_UBATCH=4096 bash scripts/run_server.sh \
           >/tmp/p3.txt 2>&1 & P=$!; sleep 32; kill -TERM $P; sleep 5;
         kill -0 $P 2>/dev/null && echo STILL_ALIVE || echo EXITED'
pgrep -fl llama-server      # 必須空
```
判準是 log 裡同時有 `model loaded` 與 `listening on http://0.0.0.0:8080`，而且
`SIGTERM`/`SIGINT` 走優雅關閉（`[CGC] Received SIGINT — initiating graceful shutdown`
⇒ 不洩漏 Metal buffer；`kill -9` 才會）。

**要知道「哪些行只印在真實啟動路徑」的話，先確認探針有沒有走到那裡。**
`run_server.sh` 的零成本探針有兩個，但它們**都在 `[fit]` 之前就退出**：
`CGC_SERVER_STRICT_BUDGET=1` 在 `[防護 2d]` 印完預算段就 `exit 1`；
`CGC_SERVER_LOAD_MODE=mmap` 在 `[防護 2e]` 就 `exit 1`。所以 `[fit]`／`[kv]` 這兩行
**只能用真實啟動驗**（上一個 recipe）。反過來說，只改 `[budget]` 那一段時，
這兩個探針就能在零載入成本下把四種情境（超額 OK/拒跑、mmap 拒跑）都印出來
——不要為了看一行 echo 去載 13 GB 的模型。

**`--check` 的收尾是「兩條都跑」且要濾行：**`index_assets.py --check` 的結論行被尾端的
`--out` 說明包住，直接 `tail` 只會看到說明文字；要濾 `OK:` / `error` 這類行才看得到結論。

## 實測補充（2026-09-16 第六輪，`89dd85b23` + `2a71b332f`）

**D5 的 oracle 旋鈕現在是「釘住的」，而且它會擋下一種你以為沒事的改動。**
`m123_oracle_gate.py` 新增了 `ORACLE_PINNED_ENV = ("CGC_SERVER_BATCH=6144", "CGC_SERVER_UBATCH=6144")`
（放在 `DEFAULT_REF` 旁邊），在 `resolve_launch` 之前併入，並把**有效集合印在啟動前**。
這件事是必要的，因為 gate 說明文字原本宣稱「預設會重現 oracle 配置（batch 6144）」，
而那句話**寄生在 `prefill250` 的生產預設值上**：把生產預設改成 5632 之後，gate 立刻對
`CGCENV.BATCH` / `CGCENV.UBATCH` / `ARG[27]` / `ARG[29]` 四列差異報 **INVALID COMPARISON**，
而同一份輸出裡 M1/M2/M3 都是 9/9（印著 *printed for information, NOT a verdict*）。
⇒ 實務規則：**改了 `prefill250` 的生產預設（或任何被 gate 預設 profile 帶上的 knob），
先看 gate 的 `comparable` 欄位再讀 M1/M2/M3**；`comparable=False` 時那三個 9/9 不是裁決。
要比另一個配置就得顯式覆蓋（`--env`）並**重新基線**（`--write-ref` ＋ `--no-pin-oracle-env`）——
覆蓋之後 INVALID 會再出現一次，那是訊號不是故障。
**殘留**：gate 認證的是 oracle 配置的數值；出廠預設 5632 沒有自己的參考檔，只量得到跨配置的 9/9。

**`cmake --build` 的輸出必須讀；`Building …` 那一行是「先前所有數字作廢」的通知。**
check 8 的證據推薦用 `cmake --build --target …`（印 `Built target` 且**沒有任何編譯行**＝產物
由當前原始碼產生），但反過來也一樣成立：**一旦它印出編譯行，就證明磁碟上的產物不是當前原始碼
產生的**。實例：12:44 全量重建後，13:00 又編輯了 `ggml-metal-context.m`（只改註解）而沒有重建，
於是 12:44–13:50 之間每一筆量測（含出廠驗收與一份 D5 PASS）都屬於舊產物；
`libggml-metal` md5 `968c36cf…` → `f2d1c961…`。**exit code 兩種情況都是 0。**
check 8 的 mtime 規則會擋這一格（binary < source），但它在 commit 前才跑，而量測更早發生。

**check 8 會因為「建置戳記」把 `libggml-base` / `libggml-cpu` 列成 modified，那不是程式碼差異。**
這兩個庫內嵌 build stamp（`strings` 差集恰好是 `8af8c99bd` → `efeade7e2-dirty`）。
`git diff --stat` 會印 `Bin 771480 -> 771480 bytes`（同大小不同內容），而
`cmp -l | wc -l` 可能報到 **47940** 個位元組不同——那是戳記字串變長造成的**整體位移**，
不是 47940 個位元組的程式碼改動。判準是 `strings` 的差集。
另一個同族的誤讀：`llama-server` 與 `libllama-server-impl.dylib` 被重寫（mtime 更新）
但可以與 HEAD **逐位元相同**（install name 沒動就不需 relink）⇒ `git diff --stat` 是空的，
不要因為「它被重建過」就預期它會出現在 staged 清單裡。

**`Backup/` 底下的東西進不了 commit（`.gitignore:396`），所以那裡修好的 harness 不算交付。**
`run_req2_retest.sh`、`run_mmap_ab.sh`、`commit_msg_*.txt` 全部在 `Backup/`。
commit message 草稿放那裡是安全的（它本來就不會被提交），但**修好的量測腳本不會跟著上線**——
要嘛把腳本搬進 `scripts/`，要嘛在 final reply 明說它只存在於本機。

**本輪的 commit 形狀（照抄）**：`git add -A` → 預演（0 FAIL、4 SKIP 逐列有理由、check 8 兩條 PASS）
→ `RUN_REPLAY_BENCH=0 git commit -F Backup/commit_msg_*.txt` → 推兩個遠端 →
`ls-remote` 驗三處相同 → 收尾驗證 → **第二個 resync commit**（只動 `INDEX.jsonl` + `MANIFEST.jsonl`）。
推送前務必先 `ls-remote <remote> 'refs/heads/<branch>'` 取**真實 SHA** 再
`merge-base --is-ancestor`，不要拿 `<remote>/<branch>`（那個 ref 可能沒被 fetch 更新）。
本輪兩次都是純 fast-forward：`efeade7e2..89dd85b23`、`89dd85b23..2a71b332f`。
