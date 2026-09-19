# 窗口閘門的驗屍報告：70 分鐘、60 次讀數、2 次「開啟」、0 次量測

**日期**：2026-09-20 01:31–02:14 · **零 GPU**（全程只讀 `vm_stat`／`sysctl`，不碰顯卡、不起 server）
**對象**：`scripts/check/server_window.py wait`（**今天新寫、尚未提交**的 `cmd_wait`，+100 行）
**證據**：`/tmp/joint_decprof.log`（waiter 自己的逐 45 秒記錄，60 筆）＋ 本線的 10 秒採樣器

---

## 0. 一句話

**兩次「窗口開啟」都是 `vm_free_mb()` 的暫態尖峰；它們在 poll 與 launch 之間那 ~10 秒裡就消失了。**
而且舊門檻 8000 在這個 box 上是**算術上不可達的**——這解釋了那 62 分鐘。

| 讀數 | 值 |
|---|---|
| 60 次讀數中 ≥ 8000 的 | **1**（就那一次尖峰） |
| 58 次 BUSY 讀數：中位 | **6515 MB** |
| p10 / p90 | 5372 / 7403 MB |
| 極值 | 5261 … 7701 MB |
| ≥ 6500 的比例 | 32/60 = **53%** |
| 結果 | 2 次 false open、2 次拒絕、**0 次量測** |

---

## 1. 兩次「開啟」的完整鄰域

waiter 印 `window open` 之後，被啟動的行程**自己又量一次**（`require_first` → `quiet()`，這是刻意的設計：provenance 不接受事後補蓋的章），然後拒絕。兩次的落差：

| attempt | poll | 前一次抽樣 | 後一次抽樣 | launch 自己的讀數 | poll→launch 落差 |
|---|---|---|---|---|---|
| 1（need 8000） | 01:39:18 **8778** | 6903（01:38:33） | 6820（01:39:28） | **6810** | **−1968 MB / 10 s** |
| 3（need 6500） | 02:07:56 **6753** | 5372（02:07:25） | 5269（02:08:06） | **5261** | **−1492 MB / 10 s** |

兩次開啟的讀數都**比它前後的鄰居高 1.4–1.9 GB**。它們不是「箱子變安靜了」，是一次持續不到 45 秒的暫態。

⇒ `[wait] window open` 現在是一句**預測**，而它預測的那件事在 10 秒後已經不成立。

## 2. 機制：在零 llama 行程下，這個讀數 10 秒動 1.3 GB

本線的 10 秒採樣器（`Backup/mem_gate_sampler_20260920.py`，02:12–02:14，量測當下 `pgrep 'llama'` 為空）：

```
    t   閘門   Δ10s  free  purg  inact    ext  swapF
   0s   5970     --    61    39   5870   2046   1087
  10s   5921    -50    72    16   5833   2047   1087
  20s   6069   +148   211    49   5809   2052   1087
  30s   4778  -1292    70     1   4707   3554   1087   ← 10 秒內 −1292 MB
  40s   5220   +442    66     6   5148   3396   1087
  ...
 140s   5730    +42    85    54   5590   2468   1087
```

t=20→30 那一步：**閘門 −1292 MB，而 `ext`（檔案頁）＋1502 MB、`inact` 被拿走 1102 MB。**
之後 `ext` 用 110 秒慢慢退掉（3554→2468），閘門也慢慢爬回 5730 —— **但它整段都低於 6500**。

⇒ 驅動源是**檔案快取的爭奪**：有東西在讀檔（快取漲 1.5 GB），而 RAM 是固定的，於是 OS 從
匿名 `inactive` 佇列裡拿頁。**這與 llama 無關，所以「確認沒有別的 llama 行程」擋不住它。**

**那 4 分鐘的統計**（n=24，02:12–02:16）：範圍 **4778–6069 MB**、中位 5686、
`|Δ10s|` 中位 **53 MB**、p90 **207 MB**、max **1291 MB**、**≥6500 的次數 0/24**。

⇒ 兩層結構要分開講，它們常常被混為一談：

| 尺度 | 行為 |
|---|---|
| **10 秒** | 尖峰／塌陷，可達 ±1.3 GB（poll→launch 的那個縫） |
| **數分鐘到數十分鐘** | 慢漂移，成批地在 ~5.3–6.0 與 ~6.9–7.7 兩個 regime 之間來回 |

70 分鐘的中位 6515 是**跨這兩個 regime** 算出來的；寫這份報告時箱子在低的那個
（我採樣 4 分鐘、他抽樣 11 次，全部落在 5.7–5.8 GB）⇒ **6500 現在還差約 800 MB**。

## 3. 為什麼門檻 8000 不可達 —— 這個閘門量的不是容量

`decode_window_harness.vm_free_mb() = Pages free + Pages purgeable + Pages inactive`。
實測三個成分：

| 成分 | 實測 |
|---|---|
| `Pages free` | 0.06 – 0.24 GB |
| `Pages purgeable` | 0.00 – 0.08 GB |
| **`Pages inactive`** | **4.7 – 5.9 GB** |

⇒ **這個閘門實際上就是「`Pages inactive` ＋ 約 0.1 GB」。** 而 `inactive` 是核心的 LRU
**老化佇列**——它是一個**速率**，不是一個**容量**。它會因為頁面從 `active` 老化而上升，也會因為
檔案快取來搶而立刻下降（§2 就是後者）。

進一步：`vm.page_pageable_internal_count` = **8.63 GB（匿名）** vs
`external_count` = **2.05–3.55 GB（檔案）**——**可放頁記憶體的 81% 是匿名頁**，而模型（磁碟上
12.72 GB）**根本不在 page cache 裡**。匿名頁只能靠壓縮（compressor 已用 2.70 GB）或 swap
（5.12 GB 中只剩 **1.09 GB** 可用）釋放。

所以更誠實的公式是 `free + purgeable + external` = **2.1–3.6 GB**，而它會拒絕每一次啟動。
那不是公式壞掉，那是**這個 box 真的沒有 6.5 GB 的硬餘裕**——它能跑是因為 macOS 願意**當場壓縮**
匿名頁。⇒ 現行閘門是一個**代理量**，它的意思隨 page cache 裝著什麼而變。

> ⚠️ **未確立**：我**沒有**測到「同一次 run 之後閘門是否系統性偏高」（page cache 裝著模型頁）。
> 那是這個模型最自然的預測，也是唯一能解釋「8000 是別的時候校準出來的」的說法——
> 但它**需要一次 run 才能驗**，在我寫這份報告時沒有。**不要引用成結論。**

## 4. 修法（會救回那兩次，且只讓判定變嚴、不變鬆）

**核心**：不要在一個**授權動作的前 10 秒**做單次抽樣的門檻測試。加一次**確認抽樣**。

`cmd_wait` 的內層迴圈，把 `if ok: break` 換成：

```python
            if ok:
                # A single quiet sample is not a window: measured 2026-09-20, this box's
                # vm_free_mb() moves up to 1.3 GB in 10 s with NO llama process running (a
                # file-cache burst raiding the anonymous inactive pool), and both windows this
                # mode declared open in its first 70 minutes were such spikes -- 8778 -> 6810
                # and 6753 -> 5261 between the poll and the launch's own re-read. Confirm before
                # declaring, so the "window open" line stops being a prediction.
                if args.confirm_s > 0:
                    time.sleep(args.confirm_s)
                    ok, why = quiet(args.port, args.need_mb)
                    n += 1
                    print(f"[wait] {time.strftime('%H:%M:%S')} attempt {attempt} poll {n} "
                          f"confirm: {'QUIET' if ok else 'BUSY'} ({why})", flush=True)
                if ok:
                    break
```

argparse 加一個（放在 `--retry-total-s` 之後）：

```python
    w.add_argument("--confirm-s", type=float, default=0.0, metavar="S",
                   help="re-check this many seconds after a quiet sample and require it to hold. "
                        "Default 0 keeps the single-sample behaviour. 10 is calibrated to the "
                        "measured spike lifetime on this box (see docs/WINDOW_GATE_POSTMORTEM).")
```

**對那兩次的回溯檢定**：`8778 → 6810`（< 8000 ⇒ 拒絕，不會誤開）；`6753 → 5261`
（< 6500 ⇒ 拒絕）。**兩次都救回。** 預設 0 ⇒ 現有行為與現有 selftest 全部不變。

**成本**：一次真開啟晚 10 秒到；一次假開啟省下 45 秒與一個 `--retries` 額度。

**不要用的捷徑**：`CGC_WINDOW_OVERRIDE=1`。它記錄條件而不是隱藏它，但它讓這一輪的數字
**描述鄰居**——只有當「已知偏高、且願意在 provenance 裡標成 busy-overridden」時才用。

## 5. 對隊列的影響（這份報告的實際用處）

1. **G4 不再需要窗口。** 佇列裡那一輪已經帶著 `CGC_DECODE_PROFILE=1`、
   `CGC_DECODE_PROFILE_ALL=1`、`CGC_GPU_TIMING=1`（launcher 自己印的 `extra=[...]`）⇒
   **它的 server log 會有 `union_sum`**。量測卡 §2 的儀器底因此可以**從它收割**，不需要另外
   起一台 server —— 那條待跑卡作廢，改由 `scripts/check/union_floor.py` 代勞。
2. 同一個工具也回答「哪一份既有 log 的精度夠」：目前最好的是 E2b
   `llama_server_20260919_210013.log`（ntok=4、n=179）⇒ 中位 132.13 ms、σ 37.69、
   **3SE 8.0%**（後半段 6.3%）⇒ **足以分辨 G4 的 12.8%**。
3. 佇列裡那一輪仍在等：本報告寫作時閘門落在 4.8–5.9 GB，門檻 6500 ⇒ 還在等。

## 6. 未確立的（免得被引用出去）

1. ❌「是別的 session 搶走了窗口」——waiter 的訊息 `window stolen between the poll and its
   launch` 是**誤導性的措辭**；兩次都不是被搶，是**讀數自己在 10 秒內塌掉**。
2. ❌「把 `--need-mb` 降夠低就能穩定開啟」——降到 p10（5372）確實會一直開，但那等於承認
   閘門沒有門檻；降到 6500 是丟硬幣（53%）。
3. ❌「檔案快取的暴衝是 X 造成的」——只知道 `ext` 漲、`inact` 被拿，**沒有識別出具體行程**。
4. ❌「G4 的 12.8% 量不出來」——已被本線推翻兩次；正確的說法是**「單一 run 的 3SE 決定一切」**，
   而一份 n≈180 的暖 run 給 8.0%。

---

## 7. 順手抓到的兩件事（都不在原本的問題裡）

### 7.1 這道閘門在**沒有 `ps` 權限**的環境裡是**直接崩潰**，不是回 BUSY

```
$ python3 scripts/check/server_window.py check --need-mb 6500
  File ".../decode_window_harness.py", line 193, in foreign_llama
    out = subprocess.run(["ps", "-Ao", "pid=,args="], ...).stdout
PermissionError: [Errno 1] Operation not permitted: 'ps'
```

`quiet()` → `_probe("foreign_llama", _fallback_llama)()`，而**兩者都用 `ps`**（`_fallback_llama`
自己也是 `subprocess.run(["ps", ...])`）。在 agent session 的 sandbox 裡 `ps` 被拒 ⇒
**閘門死掉，而不是說「我不確定」**。它失敗的方向是安全的（不會誤放行），但意思是：

* 從這種 session **沒辦法**用這道閘門守任何啟動 ⇒ 任何「等窗口再跑」的計畫在這種 session 裡
  都不成立（本線因此不去搶窗口，也就沒有 §EN-278 的風險）；
* 更普遍地：**一個守門員在缺少它依賴的工具時應該說 BUSY，不該拋例外**。`pgrep` 是可用的
  （本線全程用它），所以替代品就在手邊。

### 7.2 待跑卡 §2 叫我去跑的 `http_duo.py` **不受這道閘門管**

走 `server_window` 的共 13 支：`divergence_onset_sweep`、`m123_gate_window`、`mtp_accept_ab`、
`mtp_long_sequence_verify`、`phase_split_ab`、`plain_match_ab`、`plain_match_window`、`pool_curve`、
`slab_handoff_ab`、`sntok_curve`、`window_gate`、`gdn_split`、`test_decode_window_harness`。
**`http_duo.py`／`profile_duo.py`／`decode_sweep.py` 都不在其中。**
⇒ 照著那張卡跑，等於在別人排隊時**直接起 server**。卡已加註。

## 8. 修法已驗證（但**沒有**寫進那個檔）

`Backup/patch_cmd_wait_confirm_20260920.py`（預設 dry-run）把 §4 的兩處改動做成可套用的補丁。
**沒有對 `scripts/check/server_window.py` 動手**：該檔此刻是**別人未提交的 +100 行**，而且改的
正是同一個 `cmd_wait`；寫進去會讓他們的 diff 混進我的行列，並可能被他們下一次 `git add -A` 收走。
所以改在 **scratch 副本**（`/tmp/swscratch`，同層深度，`ROOT` 才解析得對）上證明它可用：

```
-- verify: --confirm-s is wired into the wait parser     parser accepts --confirm-s (rc=2)
-- verify: QUIET then BUSY (a spike) must NOT declare     no declaration, command not run, rc=3
-- verify: QUIET then QUIET must declare and run          declared, ran, confirm sample printed
-- verify: --confirm-s 0 keeps today's behaviour          declared on one sample, no confirm
ALL VERIFICATIONS PASSED
```

驗證刻意**在行程內注入探針**而不是跑 CLI —— 因為 §7.1：這道閘門在沒有 `ps` 的 session 裡根本
跑不起來。注入是完整的（一個 scripted 序列），所以被測到的正好是新增的那個分支。

**要套用**：`python3 Backup/patch_cmd_wait_confirm_20260920.py --apply --verify`
（跑之前先請那條線把他們在 `server_window.py` 的改動落地，否則就是兩條線在同一支檔上疊字）。
