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

## `pack_evidence.py` 的實測（2026-09-17，2646 檔）

```
source      : 2646 file(s) from Backup/cgc_logs
source size : 756.9 MB（原文，不進版控）
selected    : 81.0 MB 壓縮前（檔頭 40 行 ＋ 命中行 ±8 行，上限 256 KB/episode）
truncated   : 72 episode(s) 撞到上限
no hits     : 883 episode(s) 沒命中任何 pattern（仍產生只有檔頭的 pack）
```

### ★ 驗收「< 20 MB」的答案取決於你怎麼量

| 量法 | 數字 | 對 20 MB |
|---|---|---|
| 檔案大小總和 | **16.67 MB** | PASS |
| 實佔磁碟（`st_blocks`） | **20.72 MB** | **FAIL** |
| `du -sh`（含 `INDEX.jsonl` 與目錄本身） | 23 MB | FAIL |

差距來自**區塊對齊**：2646 個檔每個至少佔一個 4 KB 區塊，而其中 883 個是只有檔頭的小檔。
**兩種量法都合理，而門檻正好夾在中間** ⇒ 這個驗收條件本身不足以決定它自己的結果。
`pack_evidence.py` 目前用「檔案大小總和」判 PASS/FAIL 並在輸出裡同時印出兩個數字；
真正的解法是讓 §8 指定其中一個（或兩者都寫），不是讓工具猜。

**若要把兩個數字都壓下去，槓桿在規格不在程式**：無命中的 883 檔不要產生 pack，可省
1.02 MB（檔案大小）＋ ~3.4 MB（區塊對齊浪費）——那是 §8 的一句話，不是這裡的程式碼。

### 這一版**沒有**被提交

packs 產生在 `/tmp/evidence_probe`（量測用），`agent_harness/shared/evidence/` 尚未建立。
原因不是大小，是**歸屬**：§8 說原文不進 repo、抽樣要進，但沒有說 packs 放在版控的哪裡、
要不要進版控。在沒有決定的情況下把 2646 個檔（＋16.7 MB 二進位）放進 commit，等於替一個
未決策定案。工具與實測數字都已就緒，決定權留給 §8 的 owner。
