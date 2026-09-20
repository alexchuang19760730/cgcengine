# 统一窗口 harness（2026-09-20）

`scripts/check/harness.py`。一个前门，收斂四套各自實作的「等一個安靜的盒子，然後跑」。

## 為什麼要有它

同一件事在這個 repo 裡被寫了四遍：

| 檔案 | 它自己實作的 |
|---|---|
| `decode_window_harness.py` | 定義探針；兩條 lane + HTML 報告 |
| `m123_gate_window.py` | 把 M1/M2/M3 oracle gate 包進視窗 |
| `plain_match_window.py` | 把 plain_match_ab 包進視窗 |
| `server_window.py cmd_wait` | 共享探針自己，附一個等待迴圈 |

它們彼此不一致，而這個不一致不是風格問題：在忙碌的盒子上發射一次，量到的是鄰居
（同 config 實測離散 17%，壞的日子 2.5×）。這個失敗在數字本身看不出來，所以只能在發射前擋。
而今天擋不擋得到，取決於你剛好選到哪一個 wrapper。

`harness.py` **不取代**它們，它是前門：唯一一個地方同時知道

1. **哪些工具需要視窗**（registry）；
2. **盒子現在怎麼說**（一個共享探針，只問一次）；
3. **這個判決能不能被物理地相信** —— `--certify` 跑吞吐哨兵。2026-09-20 實測：thermal 全程
   NOMINAL、記憶體 54% usable，盒子卻慢 5 倍，而現有閘門查的兩個條件都成立；
4. **證據寫到哪裡** —— 每次 run 往 `Backup/phase_decomp/harness_runs.jsonl` 追加一筆。

## 用法

```sh
python3 scripts/check/harness.py show                  # 視窗看板（預設）
python3 scripts/check/harness.py show --json
python3 scripts/check/harness.py run prod_profile --need-mb 4000 -- --profile prod25
python3 scripts/check/harness.py run cb_headroom_probe --certify -- --tag x --band any
python3 scripts/check/harness.py list                  # registry
python3 scripts/check/harness.py audit                 # 未受閘的 launcher
python3 scripts/check/harness.py selftest
```

`run` 的底線：**沒有視窗就不跑，而且不會被記成數據**（exit 3，並明確印出
「nothing was run, so nothing was recorded as data」）。

## registry：為什麼需要一張表

「這個工具需不需要視窗」是一個沒有任何單一工具能替別的工具回答的問題。所以它是顯式的：

- `needs_window=True` —— 忙碌的盒子會污染它的輸出；
- `False` —— 它只解析檔案／做算術，鄰居的 server 不改它的答案（只改它跑多快）；
- 未標註 —— 由 `needs_window_for()` 視為 **True**。不對稱是有意的：把需要安靜的當成不需要，
  會產出一個「看起來正常、其實在描述鄰居」的數字；反過來只是讓人多等一下。

`launches_server` **不手寫**，它從 `server_window.audit()` 合併進來 —— 新增的 launcher 會自己出現，
不必有人記得改這張表。手寫的只有用途與歸屬；歸屬不確定的留空，讓 `show` 退回共享的
ownership 文件，因為寫錯 line 名字會把「請去上閘」的請求送給不擁有那個檔案的人。

## 修掉的真問題

**1. 一個真實的閘門回歸（我自己造成的）。** 上一輪 `6bc5e24b1` 新增的 `cb_headroom_probe.py`
是一個未受閘的 launcher，`window_gate.py check` 因此從 rc=0 變成 rc=1（未受閘 4 → 5）。
修法是按規則**接上閘**而不是 re-baseline：在 `launch` 之前呼叫
`sww.require_first()`（不是 `require` —— `--runs N` 的第 2 臂開始時，第 1 臂的頁快取還在，
每次發射都閘會拒掉一個正在正確執行的 run），並把 `sww.provenance()` 寫進每一次的結果。
現在：`ok: 4 ungated launcher(s), unchanged (gated: 11, total: 16)`，rc=0。

**2. `argparse.REMAINDER` 會吃掉自己的選項。** 初版用 `nargs=REMAINDER` 接子工具參數，結果
`--window-timeout-s 4` 被交給子工具，harness 用預設 900 s 去等（實測等了 76 秒才發射）。
改成在 `main()` 裡對第一個裸 `--` 手工切分：前面是我們的，後面原樣給子工具。

**3. 探針的 source 被算成未受閘。** `decode_window_harness.py` 定義探針，它是被檢查的對象而
不是客戶；把它算進 ungated 名單，等於要求 harness 給自己上閘。改用 audit 的 `source` 集合區分，
看板上顯示為 `src`。

## 誠實邊界

- **provenance 是本行程看到的。** 子行程在自己的位址空間裡跑。如果它是受閘的工具，它自己會
  問同一個探針並帶上自己的證據；如果不是，這筆紀錄是「run 前後盒子的狀態」的陳述，**不是**
  子行程內部的陳述。紀錄寫明是哪一種（`child_gated`）。
- **兩套定義的分歧被印出來，不取平均。** 預設 `--need-mb` 是 `server_window.NEED_MB`（8000），
  而 launcher 自己的口徑是 free% ≥ 40%。實測同一個盒子：harness 說 `reclaimable 7526MB < 8000`
  → REFUSED，launcher 說 81% → admits。看板標 `[DISAGREE: the launcher's own probe would admit]`
  並提示怎麼用 `--need-mb` 降到 launcher 的口徑 —— 但要求「刻意為之」，因為紀錄會寫下用的是哪一把尺。
- **等待需要連續 2 次安靜讀數**，不是 1 次。這裡單次安靜讀數已經多次是兩個鄰居發射之間的谷底。
- **未標註不等於不需要**：registry 目前覆蓋全部 16 個 launcher（`selftest` 有一條斷言守住這件事，
  補齊前它是紅的）。

## 沒做的

- **索引重生照舊延後**：`MANIFEST.jsonl` / `INDEX.jsonl` 目前帶著另一條線未提交的 bytes，
  重生會寫出一個既非前一版、也非他們完成版的狀態。
- **剩下 4 個未受閘 launcher 沒動**：`decode_step_profile.py`、`decode_sweep.py`、`http_duo.py`、
  `m123_oracle_gate.py`。其中 `m123_oracle_gate.py` 有另一條線未提交的改動；在別人未提交的工作上
  動手會造成衝突。正確的替代路徑是**從 harness 跑它們** —— `harness.py run` 會在發射前閘一次，
  不必改那些檔案。要讓它們在 `window_gate` 的靜態審計裡也變綠，得由各自的 owner 加 `require_first`。
