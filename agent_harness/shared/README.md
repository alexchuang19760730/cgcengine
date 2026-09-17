# `shared/` —— 兩個 loop 共用的東西

PLAN §3 替這個目錄安排三個檔案，這是它們的實況。

| 檔案 | 是什麼 | 現況 |
|---|---|---|
| `trace_schema.md` | 指出三種 record 的權威在哪（**不重述** schema，那會製造第二個權威） | ✅ |
| `sanitize.py` | 去隱私：`$REPO`／`$HOME`／hostname／IP／金鑰／URL 憑證 | ✅ 自測 19 項全過（含 PLAN §8 指名的「連線卡」） |
| `pack_evidence.py` | `Backup/cgc_logs` → `evidence/<id>.txt.zst` ＋ sha256 索引 | ✅ 已建、已跑、**驗收結果取決於「總量」的定義**（見下） |

```sh
python3 agent_harness/shared/sanitize.py --self-test
python3 agent_harness/shared/pack_evidence.py --dry-run          # 只報大小，不寫任何東西
python3 agent_harness/shared/pack_evidence.py                    # 真的產生 evidence/
```

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
