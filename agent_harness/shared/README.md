# `shared/` —— 兩個 loop 共用的東西

PLAN §3 替這個目錄安排三個檔案，這是它們的實況（＋ 兩個後加的：`README.md` 與 `check_citations.py`）。

| 檔案 | 是什麼 | 現況 |
|---|---|---|
| `trace_schema.md` | 指出三種 record 的權威在哪（**不重述** schema，那會製造第二個權威） | ✅ |
| `sanitize.py` | 去隱私：`$REPO`／`$HOME`／hostname／IP／金鑰／URL 憑證 | ✅ 自測 19 項全過（含 PLAN §8 指名的「連線卡」） |
| `pack_evidence.py` | `Backup/cgc_logs` → `evidence/<id>.txt.zst` ＋ sha256 索引 | ✅ 已建、已跑、**驗收結果取決於「總量」的定義**（見下） |
| `check_citations.py` | 把 `CONVENTIONS.md` **第 7 行**自己寫的驗收（每條都要指到證據）變成可機檢 | ✅ 61/61 條都有交代（59 可解析指針 ＋ 2 顯式標記）、0 懸空；自測 18 項含陽性對照 |

```sh
python3 agent_harness/shared/sanitize.py --self-test
python3 agent_harness/shared/check_citations.py                  # 憲章的指針完整性（非零碼 = 沒過）
python3 agent_harness/shared/check_citations.py --self-test      # 含「注入假引用必須被抓到」
python3 agent_harness/shared/pack_evidence.py --dry-run          # 只報大小，不寫任何東西
python3 agent_harness/shared/pack_evidence.py                    # 真的產生 evidence/
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
