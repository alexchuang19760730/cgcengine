# 起 server 的腳本 × 窗口守門：帳與掛勾（2026-09-19）

## 為什麼這件事值得一個模組

這台機器上一個 launch 是 13 GB 模型 + 最多 8 GiB expert-cache pool，塞進 16 GB RAM。在別人的
server 還在跑時啟動，它會超額訂閱，量到的數字描述的是 **thrash regime**——同配置的實測離散度
17%，壞日子 2.5×。這不是小誤差，是「結論」與「巧合」的差別。

在此之前這件事有三種處理方式，或沒有處理：

| 做法 | 檔案 | 後果 |
|---|---|---|
| 按**執行檔**分類（正確） | `decode_window_harness.py` | 它自己的註解記了為什麼：`pgrep -f` 會抓到別的 session 的輪詢 shell，因為那條命令列「包含」llama 名字 |
| 呼叫 `_H.llama_pids()`（**harness 沒有這個函式**） | `plain_match_window.py` | `AttributeError` → 靜默退回 naive `pgrep -f`，也就是 harness 明文修掉的那個探針 |
| 兩者都沒做 | 其他 **21 支**會起 server 的腳本 | 沒有任何徵兆 |

## 共同的尺

`scripts/check/server_window.py`（新）

```python
import server_window as SW                       # scripts/check 已在 sys.path 上（慣例同 thermal_pressure）
SW.require_first(port=8080, need_mb=8000.0, where="arm on")
```

- `quiet(port, need_mb)` → `(bool, reason)`：**三個問題一次問完**（8080 有沒人聽、有沒別人的 llama、reclaimable 夠不夠），探針直接取自 harness，且**harness 沒有該函式時會出聲**（靜默替換是上一支的病因）。
- `require()`：不乾淨就 **raise `BusyBox`**（不是警告後照跑）。`CGC_WINDOW_OVERRIDE=1` 可放行，但**條件會被記錄在 run 的 provenance 裡**，不是被藏起來。
- `require_first()`：**只把關一個 process 的第一次 launch**。理由寫在模組裡——多臂 run 的第 2 臂會看到自己第 1 臂剛灌滿的 page cache，逐次檢查等於把要保護的 run 擋死。這是 over-block，不是安全。
- `samples()`：這次 process 做過的每一次盒況檢查，供 run 自己記錄出處。
- `check`（一次性判定，exit 0/3）、`audit`、`selftest`。

## 帳（`server_window.py audit`，text 判準，見下方邊界）

| 檔案 | launch 形狀 | 狀態 | 擁有 |
|---|---|---|---|
| `decode_window_harness.py` | 2 | **source of truth**（它定義探針） | — |
| `plain_match_ab.py` | 3 | **gated**（本輪接上，咽喉點） | 本線 |
| `plain_match_window.py` | 1 | gated（`decode_window_harness`） | 本線 |
| `server_window.py` | 1 | gated（它自己） | 本線 |
| `test_decode_window_harness.py` | 1 | gated（測試） | 本線 |
| `decode_step_profile.py` | 2 | **ungated** | 線 B/C（未提交） |
| `decode_sweep.py` | 3 | **ungated** | **線 B/C 持有** |
| `http_duo.py` | 1 | **ungated** | **線 B/C 持有** |
| `m123_oracle_gate.py` | 2 | **ungated** | **線 B/C 持有** |
| `mtp_accept_ab.py` | 2 | **ungated** | **線 B/C 持有** |
| `mtp_long_sequence_verify.py` | 1 | **ungated** | 已提交、未指派 |
| `phase_split_ab.py` | 3 | **ungated** | 已提交、未指派 |
| `pool_curve.py` | 2 | **ungated** | 已提交、未指派 |
| `slab_handoff_ab.py` | 1 | **ungated** | 未追蹤（本線新增） |

**14 支啟動器：4 gated、9 ungated、1 source。** 依 `docs/SHARED_SRC_OWNERSHIP_2026-09-19.md`，
`{decode_sweep, http_duo, m123_oracle_gate, mtp_accept_ab, prod_matrix}` 由線 B/C 持有 ⇒ 本輪
**沒有動**；它們的狀態現在是**可見的**，而不是隱形的。

註：`m123_oracle_gate.py` 的 ungated 指的是**它自己起 server 前不問盒況**——閘門被
`m123_gate_window.py` 包住時，外層已經把關（那條路徑是 gated 的）。直接跑閘門則不是。

## 邊界（誠實標註）

`audit` 是**文字比對**：它認得這棵樹裡出現的 launch 形狀（`subprocess.*` 呼叫 `run_server.sh`、
`build/libllama-server`、`LAUNCHER = ...`）與守門的接入方式。用**別的形狀**寫的啟動器會被漏掉，
所以「這支是 ungated」是**去看一眼的提示，不是缺陷的證明**。

反向的假陽性也有過，而且更有害：AST-only 版本把**明明有守門**的 `plain_match_window.py` 與
`m123_gate_window.py` 報成 ungated——儀器講與事實相反的話。現在改成文字判準，並把 harness 自己
分成 `source`。另外測試釘住「只在 docstring 提到 `run_server.sh`」不算啟動器。

## 證據

- `server_window.py selftest`：全過。其中三條是**它自己的缺陷被測試抓到**——(1) `_probe() is
  harness.fn` 為 False，因為 `_load_harness()` 每次重新執行模組（函式 identity 不穩，等於臂在
  自我欺騙）；(2) `audit` 不展開 glob，傳 `*.py` 會被當成字面路徑 ⇒ 什麼都沒檢查卻回報成功；
  (3) 上述的 source/gated 誤分類。
- 實地驗（真實繁忙盒況，別人的 server 佔著 8080）：
  ```
  QUIET/BUSY: port 8080 already listened on [61356]; other llama process(es): [(61356, 'llama-server')]; reclaimable=993MB<8000
  require_first: 拒絕 -> ... Refusing to launch into a busy box
  override 後: 放行但記錄為 not-quiet
  ```
- `Backup/phase_decomp/server_window_audit.json`（本次帳的機器可讀版）。

## 剩下要做的

1. 三支未指派的已提交檔（`mtp_long_sequence_verify`、`phase_split_ab`、`pool_curve`）與未追蹤的
   `slab_handoff_ab`：各插一行 `require_first`。
2. 線 B/C 持有的四支：等他們接手，或由他們自行掛勾（本輪不代改，只把狀態記進這份帳）。
3. 把 `audit` 掛進 pre-commit：`ungated` 數量只能下降。
