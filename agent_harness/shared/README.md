# `shared/` —— 兩個 loop 共用的東西

PLAN §3 替這個目錄安排三個檔案，這是它們的實況（＋ E4 之後的六個）。

| 檔案 | 是什麼 | 現況 |
|---|---|---|
| `trace_schema.md` | 指出三種 record 的權威在哪（**不重述** schema，那會製造第二個權威） | ✅ |
| `sanitize.py` | 去隱私：`$REPO`／`$HOME`／hostname／IP／金鑰／URL 憑證 | ✅ 自測 19 項全過（含 PLAN §8 指名的「連線卡」） |
| `pack_evidence.py` | `Backup/cgc_logs` → `evidence/<id>.txt.zst` ＋ sha256 索引 | ✅ 已建、已跑、**驗收結果取決於「總量」的定義**（見下） |
| `check_citations.py` | 把 `CONVENTIONS.md` **第 7 行**自己寫的驗收（每條都要指到證據）變成可機檢 | ✅ 61/61 條都有交代（59 可解析指針 ＋ 2 顯式標記）、0 懸空；自測 18 項含陽性對照 |
| `check_shell_cjk.py` | `$VAR` 緊接非 ASCII ⇒ bash 把中文字節吃進變數名（`bash -n` 抓不到） | ✅ 自測 9 項；全 repo 掃出 3 處既有（其中在 `scripts/` 的 2 處無 `set -u`，症狀是靜默的文字損壞） |
| `split_module.py` ＋ `knifeedge_split_map.json` | 把單體檔**逐位元組**拆成套件；拒絕歸屬不了的名字、import 環、以及「`global` 宣告的名字被拆散」 | ✅ E4 item 2 用它拆了 `scripts/check/knifeedge_matrix.py`（3310 行 → 11 個模組，最大 571 行） |
| `check_module_split.py` | 證明一次拆分沒改行為：AST／runtime 指紋／CLI／替換契約，四層各含陰性對照 | ✅ 自測 2/2；替換契約陰性對照移除轉發層 → 3/4 變紅、列出 101 個具體失敗 |
| `stale_registry.json` ＋ `check_stale.py` | 已 stale 的資產與「誰會把它講出來」的單一權威 | ✅ 3 個登記項；banner 跟著資料、消費點由 wrapper 印出、反向查未登記的標記 |
| `log_policy.json` ＋ `check_log_policy.py` | PLAN §8 的 log 政策（納管／不納管／上限／既有例外）從散文變成斷言 | ✅ 一啟動就抓到 12 個 `.log` 原文在版控裡（5.6 MB，登記為可見例外） |

```sh
python3 agent_harness/shared/sanitize.py --self-test
python3 agent_harness/shared/check_citations.py                  # 憲章的指針完整性（非零碼 = 沒過）
python3 agent_harness/shared/check_citations.py --self-test      # 含「注入假引用必須被抓到」
python3 agent_harness/shared/pack_evidence.py --dry-run          # 只報大小，不寫任何東西
python3 agent_harness/shared/pack_evidence.py                    # 真的產生 evidence/
python3 agent_harness/shared/check_shell_cjk.py                  # $VAR 緊接中文
python3 agent_harness/shared/check_stale.py --check              # stale 的雙向閘門
python3 agent_harness/shared/check_log_policy.py --check --privacy   # 政策 ＋ 去隱私（全掃 7.8 秒）
python3 agent_harness/shared/check_module_split.py --self-test   # 拆分等價證明的陰性對照
```

## `check_citations.py`：難的不是規則，是量具自己的假陽性

憲章第 7 行早就寫著「每一條都必須能指到今天的具體證據（檔名／log 行／**欄位值**）。
指不到的條文不是憲章，是感想」——**驗收句一直存在，只是沒有人把它變成可執行的檢查**。

而寫這個檢查的成本**不在規則，在量具**：第一版在「乾淨」的憲章上報了 **84 個問題**，全部是假的。
六類，每一類都留了對應處理（檔案 docstring 有完整清單）：只在 repo 根解析（80 個假懸空）、
行文斜線當路徑（`gpu_union/wait`）、decision id 精確比對（id 是 `dec-YYYYMMDD-HHMM-<slug>`，要前綴）、
副檔名本身（`.m`）、承前省略（`..._202725.log`）、以及最貴的一類——**同一格程式碼裡
路徑後面跟著參數**（`prefill_gputime_report.py --decprof-pair …`），它讓 A18 被報成「完全沒有指針」。

所以 `--self-test` 的**陽性對照**是這個檔案最重要的一段：把假路徑注入副本，檢查器必須從
「0 懸空」變成「1 懸空」。沒有那一格，「0 懸空」與「檢查器根本沒在跑」是同一個輸出。

## 為什麼 `sanitize.py` 是一個獨立檔案

PLAN §3 把它列為獨立檔案，理由和 `sft_common.py` 一樣：遮罩是一條**政策**，日後會有第二個
消費者（今天的 log 打包、明天的 evidence 匯出），而兩份內嵌的副本會在第一次修改時分叉 ——
安靜地分叉，因為兩份都還是會產生輸出、兩份輸出都看起來沒錯。

兩個性質在 `--self-test` 裡是被斷言的，不是被假設的：

- **冪等**：`f(f(x)) == f(x)`。否則輸出取決於「跑過幾次」，而第二次跑會把 `$REPO` 變成
  `$REPO/$REPO`。
- **不誤遮**：`0x12e80c000`（指標）、`441.02`（數字）、`sum=10153` 都不能被當成 IP 或金鑰。

## `pack_evidence.py` 的實測（2026-09-17）

```
source      : 2750 file(s) from Backup/cgc_logs
source size : 854.0 MB（原文，不進版控）
selected    : 95.5 MB 壓縮前（檔頭 40 行 ＋ 命中行 ±8 行，上限 256 KB/episode）
truncated   : 126 episode(s) 撞到上限
no hits     : 924 episode(s) 沒命中任何 pattern（仍產生只有檔頭的 pack）
→ evidence/  : 2446 個唯一 blob、12.79 MB（檔案大小總和）
```

### ★ 驗收「< 20 MB」的答案，以及為什麼它現在不需要爭

上一輪量到的是：**檔案大小總和 16.67 MB（PASS）／實佔磁碟 20.72 MB（FAIL）** ——
門檻正好夾在兩種都合理的量法之間。當時的結論是「§8 要指定其中一個」。

**2026-09-17 的處置不是挑一個定義，是讓兩種定義都過關。**
去重的槓桿比「無命中不產生 pack」大得多，而且它是**內容尋址**：

| | 去重前 | 去重後 |
|---|---|---|
| 唯一 blob 數 | 2646 個檔（195 組 sha256 重複、涉及 462 檔） | **2446** |
| 檔案大小總和 | 16.67 MB PASS | **12.79 MB PASS** |
| 實佔磁碟（`st_blocks`） | 20.72 MB **FAIL** | **16.33 MB PASS** |

重複的極端例子是一個 0.1 KB 的 `*.ips_before` 出現 **28 次**（位元組完全相同）。
**那些重複對版控不存在** —— git 的 blob 就是內容尋址的，28 個相同的檔它只存一份 ⇒
舊寫法印出的「總量」比版控實際增加的量多了 4.86 MB，**而那個差額比到門檻的餘裕還大**。

所以 `pack_evidence.py` 現在寫的是 `<sha256(pack text)[:16]>.txt.zst`，同一份內容只寫一次
（304 個 row 重用既有 blob，6.40 MB 的重複沒有被寫第二次），並在輸出**同時印兩種量法**，
以**較嚴格**的那個判 PASS/FAIL。定義之爭不需要贏。

**刻意不做的槓桿**：無命中的 924 檔不要產生 pack（可省 ~1 MB）。**不做**，因為
636／883 的來源 <= 4 KB ⇒ 整個檔就是內容，濾掉等於**丟資料**而不是省空間。

### packs 的去處（已決定）

`agent_harness/shared/evidence/`，**納管（tracked）**。理由：

- §8 的「< 20 MB」這個上界**只有在產物進版控時才有意義**（否則為什麼要限制它的大小）。
- lesson `eng-bound-0004`：**只發行指標的索引在「把內容送到另一台機器」這個用途下是無效的** ——
  要跨機器的內容必須進版控。這一整條線（sanitize → pack → INDEX + sha256）存在的目的
  就是取代 854 MB 的原文。
- §8 說「原文不進 repo」：pack 是**去隱私後的替代品**，不是第二份原文。

`.gitignore` 只忽略 `engine_loop/runs/` 與 `engine_loop/build.json`，
`shared/evidence/` **刻意不在忽略清單裡**（`.gitignore` 的註解寫明了這件事）。

---

## 拆分（E4 item 2）：難的也不是搬，是**搬動不會被檢查到的那幾件事**

`scripts/check/knifeedge_matrix.py` 是 3310 行 / 181 KB。手拆等於重寫，而重寫的風險
不是語法錯誤（那會被立刻發現）——是三類**沒有任何語法檢查會反應**的東西：

| 類別 | 實例（這次真的踩到） | 誰發現的 |
|---|---|---|
| 錨點 | `ROOT = dirname(dirname(dirname(abspath(__file__))))` 換一層目錄就落到 `scripts/` | AST 檢查的 `declared_delta`（值由 runtime 指紋斷言相同） |
| `__file__` 的指涉 | `record_provenance` 用 `open(abspath(__file__))` 記「跑的是哪支腳本」 | 逐字重建驗過，改成顯式的 `ENTRY_FILE` |
| `global` 宣告的名字 | `args_greedy_global` 只在 `main` 裡建立、只在 `print_matrix` 裡被讀 | `split_module.py` 的硬性檢查（把兩者放同一模組） |
| **entry 上的屬性寫入** | 離線自測用 `km.pool_geometry = 替身` 注入合成幾何 | **`check_module_split.py contracts`** —— AST 與指紋都比對不到它 |

最後一列是這次唯一真的弄壞東西的：拆完之後 `feasibility_cell` 讀的是
`feasibility.pool_geometry`，而 `km.pool_geometry` 只是 shim 的獨立綁定 ⇒ 注入**靜默失效**，
測試掉到真的 GGUF 路徑並以 `ModuleNotFoundError: numpy` 崩潰。處置兩層：
正常 import 的 entry 由 shim 換掉自己的類別攔 `__setattr__` 並轉發到定義它的模組；
**路徑載入**（`spec_from_file_location`，三支離線自測用的）拿不到 entry 那個物件 ——
Python 沒給任何回指的辦法 —— 所以那個情境的正解是 `km.<module>.<name> = x`。
兩支呼叫端據此各改 5 處與 1 處，改前後輸出**逐字元相同**。

## stale（E4 item 4）：banner 要跟著資料，也要在消費點被講出來

一個用 stale baseline 跑出來的 verdict，格式與用真 baseline 跑出來的**一模一樣**。
所以 `.gitignore` 的註解對「讀那個檔的人」有效，對「用那個檔的程式」無效。三層：

1. **單一權威** `stale_registry.json`（資產與消費者，各附 why／evidence／run_with／decision）；
2. **banner 跟著資料**：`.replay_bench_baseline.json` 的 `_stale` 區塊
   （先驗過讀者一律用 `.get(<key>)` 取值、不做鍵集合比對，所以加鍵是安全的）；
3. **消費點出口**：`wrappers/_common.sh` 的 `w_announce_stale` 在執行前問登記表 ——
   順帶一個設計約束：**查不到或 registry 壞掉一律當成「不是消費者」**，
   不讓閘門機制本身把命令弄失敗。

## log 政策（E4 item 1）：沒有閘門的政策就是一個願望

政策原文寫在 PLAN §8、`.gitignore` 的 dated 註解、以及幾支腳本的行為裡。
把它抽成 `log_policy.json` 並加上 `check_log_policy.py` 之後，**第一次執行就抓到**：
`Backup/cgc_logs` 底下有 **12 個 `.log` 原文在版控裡**（5.6 MB），而政策是「原文不進 repo」，
且 `docs/` 與 `*.md` 對它們的引用數是 **0**。

那 12 筆不是這條線的檔案，所以**不移出**（移出等於刪掉別人的證據），而是進 `grandfathered`：
每一筆印出來、附 owner 與大小，**多一筆就紅**。登記不是豁免 —— 它把一個沉默的違規
換成一個看得見的例外。另外 22 個衍生摘要（`arms.json`／`*.tsv`／`req2retest*.txt`，共 118 KB）
如實列為 `derived_but_tracked`：它們不是原文、不違反政策，但都是 `git add -f` 進來的，
而 `-f` 進來的東西沒有人覆核。
