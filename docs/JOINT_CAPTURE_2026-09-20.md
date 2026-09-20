# 聯合捕獲：decode 步的帳其實是閉合的

`docs/JOINT_CAPTURE_2026-09-20.md` — 線 A (ace)，2026-09-20 21:44–21:5x。

## 0. 這份文件回答什麼

`docs/CROSS_LINE_STATUS_AND_TARGET_MATH_2026-09-20.md` §4 ④ 留下兩筆「未閉合」：

> 線 A 的 `gap ⊆ cb+submit` 交叉檢查以 **10.09 ms** 失敗；線 I 的 cb 74.18 與線 A 的
> union 104.6 落在同一個 ~250 ms 步裡，相加**超過**步時 ⇒ **分解未閉合**。

兩筆都來自**跨 run 相加**。這份文件做該節要求的事：一次 run、把所有量讀進同一份 log、
逐 step 對帳。工具 `scripts/check/joint_reconcile.py`。

**結論先講：帳是閉合的。失敗的是那兩個檢查各自的前提，不是帳本身。**

---

## 1. 捕獲條件

| 項 | 值 |
|---|---|
| 途徑 | `scripts/run_server.sh --detach`（PID 63109/63186），8080 |
| 探針 | `CGC_DECODE_PROFILE=1` `CGC_DECODE_PROFILE_ALL=1` `CGC_GPU_TIMING=1` `CGC_HOOK_SPLIT=1` `LLAMA_EXPERT_CACHE_BATCH_DBG=1` |
| 請求 | 巴黎 prompt（`--probe-prompt` 那個，與 884 ref 同一題），`max_tokens=400`，`temperature=0` |
| 產出 | 251 completion tokens |
| log | `Backup/cgc_logs/llama_server_20260920_214450.log` |
| 對帳 JSON | `Backup/phase_decomp/joint_capture_20260920_2144.json` |
| 樣本 | 303 rows → **129 work rows**（`sum_gap>0`，與 `cb_miss_regression.py` 同一判準）；穩態取後 65 |

⚠ **途徑邊界**：這是 HTTP server 路徑，**不是交付口徑**（交付用 llama-bench）。
本文件只引用**閉合關係與佔比**，不引用絕對步時當交付數字。

---

## 2. 閉合結果（穩態，n=65）

| 量 | ms | 佔 total |
|---|---|---|
| `total` | **122.39** | 100% |
| `wait` | 105.64 | 86.3% |
| `cb` | 11.26 | 9.2% |
| `submit` | 3.36 | 2.7% |
| `union_sum` | **97.90** | 80.0% |
| `gap_sum` | **23.52** | 19.2% |
| `gpu_sum` | 124.70 | 101.9% |
| `span` (= max(en)−min(st)) | **122.44** | 100.04% |

### 閉合一：GPU 軸（有資訊量）

```
union_sum + gap_sum  =  span      逐 step 中位殘差 −0.06 ms（全體 −0.041）
```

段**嚴格串行**、無重疊、無亂序。逐段驗證過：`L0 union 7.38 = en−st 7.377`；
`L1 gap 6.77 = st₁−en₀ = 14.145−7.377`。
（`gpu_sum 124.70 > union_sum 97.90` 是預期的：gpu 逐 buffer 加總、union 是 span，段內 buffer 併發。）

### 閉合二：跨軸（有資訊量）

```
span 122.44  vs  total 122.39     殘差 +0.26 ms（0.2%）
```

GPU 時間軸幾乎**完整覆蓋**整個 CPU 步 —— 步時沒有落在 GPU 時間軸之外的第三塊。
這兩個數來自兩個獨立時鐘（CPU `ggml_time_us` vs GPU buffer timestamp），所以這個吻合不是構造出來的。

### 閉合三：CPU 軸 —— **恆等式，零資訊量**

```
ggml-backend.cpp:2691    const int64_t dp_tot = dp_w + dp_cb + dp_sb;
```

`total` 就是三個分量之和。所以 `wait+cb+submit == total` 精確重現 0.00 ms 是**定義**，
它不可能失敗，也不能拿來證明帳閉合。工具仍然印它，但輸出第一行就標了 tautology。

---

## 3. 「10.09 ms 失敗」到底是什麼

由閉合二與閉合三：

```
total        = wait + cb + submit
total        = union + gap          （span ≈ total）
⇒ gap − (cb + submit)  =  wait − union
```

**那個交叉檢查量的其實是 `wait − union`。** 兩個獨立算法在本輪吻合：

| 算法 | 值 |
|---|---|
| `gap_sum − (cb + submit)` | **+7.97 ms** |
| `wait − union_sum` | **+7.74 ms** |

差 0.23 ms（中位相加的誤差）。所以「失敗 10.09 ms」不是帳的漏洞，而是
**「CPU 等待時間」比「GPU 忙碌時間」多出約 8–10 ms** 這個事實被當成了一個恆等式來要求。

而 `wait` 與 `union` 本來就不需要相等：

* 兩個時鐘不同（`ggml_time_us` vs GPU timestamp）；
* 兩個窗口不同 —— `wait` 是 CPU 在 hook **之前**等的時間，`union` 是該層 GPU 段的跨度，
  兩者是**交錯**關係，不是包含關係。

⇒ **判決：`gap ⊆ cb+submit` 不是一個該成立的恆等式，它從構造上就是個 plausibility check。
它的「失敗」不構成帳未閉合的證據。** 這個檢查應該停止被當成閘門使用。

歷史上同一檢查的讀數：09-19 log **+3.94**（中位，`joint_reconcile` 重讀）、09-20 **+10.09**、
本次 **+7.97** —— 同號、同量級、系統性。它不是偶發污染，它就是這個量。

---

## 4. 「cb 74.18 + union 104.6 超過步時」

同一個 step 內讀（本次穩態）：

```
union_sum + cb − total  =  97.90 + 11.26 − 122.39  =  −11.53 ms
```

**不超，而且在步時之內。** 兩個理由使它不可能有意義地相加：

1. `union` 是 GPU 時鐘、`cb` 是 CPU 時鐘 —— 跨時鐘相加；
2. 兩者在時間上**重疊**（CPU 提交後 GPU 才跑，cb 與 union 併發）。

⇒ 「相加超過步時」是**跨 run 相加**造成的假象，不是真實衝突。兩個數字分屬
不同 run、不同記憶體水位、不同池熱度。

---

## 5. 本輪的實質發現（比閉合本身更值錢）

| | 09-19/09-20 記錄 | 本次穩態 |
|---|---|---|
| `cb` | **74.18 ms（29.9%）** | **11.26 ms（9.2%）** |
| `wait` | 159.77 ms（64%） | 105.64 ms（86.3%） |

**`cb` 差 6.6 倍。** 之前那次是「高 swap」regime（線 I 自己也記了 `CGC_HOOK_SPLIT` + 高 swap）；
本次啟動 `free=84%`。

含義：`cb` 是 host-side top-k hook，其成本是 `ensure` 的 fill，而 fill 走 `pread` ⇒
**`cb` 對記憶體水位極度敏感**。所以：

* 「cb 佔 30%，所以 cb 是最大槓桿」這個論證**依賴那個水位**，不能跨 run 引用；
* 線 I 的 victim rule 上界（Belady 31%）是在**那個** cb 上算的，水位變了上界也變；
* 任何「先優化 cb」的排序，都必須先寫明是在哪個水位讀的 cb。

這也順帶解釋了為什麼線 I 的 74.18 與線 A 的數字長期對不上 —— 不是儀器分歧，是**狀態分歧**。

---

## 6. 工具

```
python3 scripts/check/joint_reconcile.py --log <server.log> [--json out.json]
python3 scripts/check/joint_reconcile.py --selftest        # 12/12
```

設計上刻意做的三件事：

* **`cpu_resid` 標為 tautology**（第 2 節閉合三）。不標的話它會是表上看起來最漂亮的結果，
  而它攜帶的資訊是零。
* **沒有 `st=/en=` 就說「無法讀」而不是補 0**。舊 log（含 09-19 那批）沒有這兩個欄位 ——
  它們是後來才加進 binary 的 —— 所以時鐘安全閉合在舊 log 上不可用，工具明講。
  （這也是為什麼本次必須重跑：舊 log 沒有 `st=/en=`。）
* **work row 過濾**（`sum_gap>0`，與 `cb_miss_regression.py` 同判準）。`CGC_GPU_TIMING` 會讓
  每個 turn 產生 work + shadow 兩行，混池會得到一個描述不了任何一個族群的中位
  （未過濾時中位 total 1.93 ms、`median_sum_n=1`；過濾後 122.39 ms、`sum_n=40`）。

---

## 7. 誠實邊界

* **途徑是 HTTP 不是 llama-bench** ⇒ §2 的絕對值不可當交付數字；可引用的是閉合關係與佔比。
* `st=/en=` 只在本次新 binary 上讀到 ⇒ 「段嚴格串行」這個結論只對本次這顆 binary／這個 profile 成立。
* 樣本來自**單一請求**（251 tokens）的 129 個 work row；`ntok` 多為 4。沒有跨 prompt、跨水位重複。
* `union+gap = span` 精確 ⇒ 但這是**串行**的證據，不是「GPU 已滿載」的證據：
  `gap` 仍佔 19.2%。串行且空轉 19.2% 正是「有重疊可恢復」的形狀（G1 的那塊），與 G1 已判定
  不可達（下界 9.0% > 5%）不矛盾 —— 那是「能不能藏掉」，這是「有多少」。
