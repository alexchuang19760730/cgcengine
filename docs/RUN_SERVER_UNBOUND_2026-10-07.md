# ⚠⚠⚠ `run_server.sh` 在任何池大小下都無法 `CGC_DUMP_ENV=1` —— 屬於**另一條線的未提交修改**（2026-10-07 22:3x）

## 0. 一句話

★ **`scripts/run_server.sh` 目前是 modified（未提交）狀態，且它讓**所有**走 `harness bench` 的實驗失敗。**
★ 這不是我的檔案，**我沒有動它**。以下是我用唯讀方式查到的根因與證據，交給 owner 決定。

## 1. 症狀（可獨立重現，不需要 harness）

```bash
CGC_SERVER_PROFILE=prod-new CGC_DUMP_ENV=1 \
  CGC_SERVER_EXPERT_CACHE_BYTES=8589934592 CGC_SERVER_MTP=0 \
  bash scripts/run_server.sh
# => scripts/run_server.sh: line 1085: SERVER_LOAD_MODE?: unbound variable
```

★ **與池大小無關**：4 GiB 與 8 GiB 都一樣炸（實測）。
★ 這條路徑（`CGC_DUMP_ENV=1`）是 `llama_bench_matrix.resolve()`（`:262`）的**前置步驟**
⇒ ★★ **任何走 harness 的 `bench` 都會死在這裡**，與 `--cell` 無關。

## 2. 根因（已用 /tmp 複本驗證）

`scripts/run_server.sh:1085`（**目前未提交版本**）：
```
echo "[budget] model 檔案 ${_b_model} MiB（load_mode=$SERVER_LOAD_MODE；檔案全量 ≠ 常駐）"
                                              ↑ 這裡是**全形分號 U+FF1B**
```
★ **`$VAR` 緊接全形分號時，在 `set -u` 下 shell 把它當成變數名的一部分**
⇒ 變數名變成 `SERVER_LOAD_MODE；檔案全量` 之類 ⇒ `unbound variable`。

**驗證方式**（不動原檔，只在 /tmp 複本上做一個字元替換）：
```
把「$SERVER_LOAD_MODE；檔案全量」改成「$SERVER_LOAD_MODE 檔案全量」
⇒ 那個 unbound 錯誤消失（剩下 /tmp 路徑問題，證明腳本本身能跑）
```
⇒ ★★ **根因確定。**（我第一個最小測試沒重現，是因為它沒帶 `set -u` 與那個完整上下文。）

## 3. 為什麼 HEAD 版沒有這個問題
`git diff HEAD -- scripts/run_server.sh` 顯示另一條線把那一段重寫了：
- 舊行：`echo "[budget] model ${_b_model} MiB（load_mode=$SERVER_LOAD_MODE => ${_MODEL_KIND}）"`
  （**用半形空格**分隔 ⇒ 變數名乾淨）
- 新行：`echo "[budget] model 檔案 ...（load_mode=$SERVER_LOAD_MODE；檔案全量 ≠ 常駐）"`
  （**用全形 `；`** ⇒ 變數名被污染）

★ 該 diff 同時把 `case "$SERVER_LOAD_MODE" in` 區塊往下移（1065 → 1082，中間新增一段註解），
★ **但它在 518 行仍設 `SERVER_LOAD_MODE="${CGC_SERVER_LOAD_MODE:-none}"`**
⇒ 所以變數**有值**，錯在**引用處**的全形標點。

## 4. 影響範圍
- ★ **所有** `harness.py bench` / `commit_bench` 呼叫（`resolve()` 是它的第一步）。
- ⇒ ★★ **`run_poolsweep.sh` 現在跑不起來，與我的 cell 設計無關** —— 死在更前面。

## 4a. ★★ 這是**第二個**獨立的 bug，不是 18:25 那次的同一個（我一度搞混，已更正）

時間線（三個獨立的失敗）：

| 時點 | 症狀 | 根因 | 誰的 |
|---|---|---|---|
| **18:25**（operator 跑） | p4–p7 無產物，log 裡有 `⛔ cell 口徑不一致` ＋ `expert_cache_bytes: 實際 4294967296 ≠ 權威 8589934592` | ★ **我的 `poolsweep` 一格設計**（格宣告 8 GiB、臂 env 帶 4 GiB） | ★ **我的** |
| **22:31**（我重跑） | 0 個 json，`p4_off.log` 尾巴是 Python traceback | ★ **`resolve()` 的 `UnicodeDecodeError`**（`run_server.sh` 輸出含非 UTF-8） | 症狀在 harness，**觸發源是下面的 #3** |
| **22:3x**（我深挖） | `run_server.sh:1085: SERVER_LOAD_MODE?: unbound variable`，**4 GiB 與 8 GiB 都一樣** | ★★ **另一條線對 `run_server.sh` 的未提交修改**（全形 `；`） | ★ **別條線的** |

⇒ ★★ **18:25 那次，我的 cell 設計確實是原因**（log 有明確的 `≠ 權威` 那一行）—— 我先懷疑它、
後來一度想推翻，**那是多餘的自我懷疑**；證據支持原本的判斷。
⇒ ★★ **22:31 之後的失敗是另一件事**，而且★ **`run_server.sh` 的 unbound 擋在 `resolve()`、
  連契約都沒走到** ⇒ ★ 我在 18:25 說「p8 成功是因為 (default) 宣告值剛好對上」——
  ★★ **那個解釋是對的**（8 GiB 臂的 env 確實等於格宣告值），**但不是唯一原因**：
  ★ 18:25 時 `run_server.sh` **還沒有那個 unbound**，所以 8 GiB 能走完全程。
  ⇒ ★★ 兩個原因**並存**：契約擋小池、unbound 擋一切。

## 4b. ★ 我 18:3x 的兩個「修」，一個對一個不相關
- ★ **對的**：`llama_bench_matrix.py` 的 `subprocess.run(..., text=True)` 補 `errors="replace"`
  ⇒ 症狀（decode 崩）確實消失了，但**根因（`run_server.sh` 的全形標點）沒被碰到**。
- ★ **不相關**（但本身正確）：測試卡從「一格 poolsweep」改成「每池一格」
  ⇒ `expert_cache_bytes` 確實是格級維度、臂 env 不能覆蓋它，★ **這個結論對**；
  ★ 但 18:25 的真實擋路者是契約，而 22:31 之後的真實擋路者是 unbound ⇒ ★ 兩者都要解。

## 5. 修法（**由 owner 動，因為是他的 modified 檔**）
把 `:1085` 的**全形分號**改成半形（或用 `${SERVER_LOAD_MODE}` 大括號 + 半形分隔）★：
```
echo "[budget] model 檔案 ${_b_model} MiB（load_mode=${SERVER_LOAD_MODE} 檔案全量 ≠ 常駐）"
```
★ 大括號 `${...}` 是最穩的寫法：它把變數名**顯式關起來**，後面無論接什麼標點都不會被吞。

## 6. ★★ 這也讓我今晚 18:3x 的兩個「修」都不必要
- 我把 `llama_bench_matrix.py` 的 `subprocess.run(..., text=True)` 補了 `errors="replace"`
  ⇒ ★ **那個修是對的但不是根因**（decode 錯只是症狀，且它讓 0 個產物變成「看起來跑了」）。
- 我把測試卡從「一格 poolsweep」改成「每池一格（pool4g/5g/6g/7g）」⇒ ★ 那個改動本身正確
  （`expert_cache_bytes` 確實是格級維度），★ **但 p8 能跑是因為 8 GiB 恰等於格宣告值，不是因為我改了什麼**。

## 7. 誠實邊界
- ★★ **我沒有修改 `run_server.sh`** —— 依分工，它是別條線的 modified 檔，我只讀不動。
- ★ 根因驗證在 **/tmp 複本**上做（`cp` 出去改一個字元再跑），原檔未動。
- ⚠ 「全形分號在 `set -u` 下會被當變數名一部分」這個機制，我第一次用最小例子**沒重現**，
  ★ 是在真實腳本上才確認 —— 所以機制的**邊界條件**（何種 shell／位置／上下文）我沒有完全釐清，
  但「改成半形就不炸」這個**事實**是確定的。