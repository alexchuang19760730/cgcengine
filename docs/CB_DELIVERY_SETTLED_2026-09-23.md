# `cb` 定讞：42 還是 74？（2026-09-23 18:0x）

> **一句話**：**42 和 74 不是兩個 regime，是同一支 run 的兩個窗口。**
> 交付錨點用 `--warm-skip 64`（時鐘從池預熱之後才開始）⇒ 交付 regime 的 `cb` 是
> **穩態那一格 ≈ 42~51 ms**，**不是 74**。
> ⇒ 真實收益是 **+14.5%~+19%（14.4~15.0 t/s）**，不是 +24%~+31%。

---

## 1. 為什麼這個數字值得單獨跑一趟

`cb` = 分段派送器裡 host 側的 expert-cache fill（`ensure_batch`）成本，整步加總。
它是 `window_joint_ev.py` 裡唯一決定「把 fill 藏進 GPU 陰影值多少」的輸入：

```
每層需求 F = cb / 40        單機制可藏 moved = min(lead, cov·F)
step_new  = 247.98 − (moved·40 − insert)
```

本專案量過 **11.26 / 21.23 / 42.04 / 74.18 ms（6.6×）**。42 與 74 差 1.76×，
定價結果就從 **+14.5%** 跳到 **+24~31%**。而 `joint_reconcile.py` 與
`cb_headroom_probe.py` 兩支工具存在的理由，正是**禁止跨 log 相加同一個句子的分量**
—— 所以這題不能靠挑一邊，只能去量。

## 2. 方法：在交付 cell 本體上量，讓 `t/s` 與 `cb` 出自同一支 log

不再從別的 run 借數字。開 `CGC_DECODE_PROFILE=1`（`ggml-backend.cpp:2010`，
`:2720` 印 `CGC-DECPROF`）跑**交付 cell 本來那一格**：

```
llama-bench -m ...gguf -ngl 99 --load-mode none -t 8 -expert-cache 8589934592 \
  -b 512 -ub 512 -p 0 -n 128 -d 512 -r 3 --warm-skip 64 --ctx-size 4096 --spec-type draft-mtp
```

（`prod_profile.py` 的 `DECODE_SHAPE`，一字未改。）
`llama_bench_matrix.py` 直接起 `llama-bench`，stderr 落 `{workdir}/*.stderr.log`。

跑了兩趟：

| 臂 | thermal | t/s | 用途 |
|---|---|---|---|
| **A** | **HEAVY**（前面 6 趟 GPU 熱浸） | 6.14 | 第一個樣本 |
| **B** | **NOMINAL 全程**（hist 125/125） | 9.18 | 乾淨樣本（單臂噪音 ±27%，9.18 在 12.57 的帶內） |

## 3. 三個會讓數字讀錯的坑（都已進 `cb_delivery_read.py` 的 selftest）

1. **prefill 圖也算一個 step row**：`-d 512` 的 prefill 是**一整個** `graph_compute`，
   印成 `ntok=512, cb=5791 ms`。它落在 rep1 頭上，一筆就把 rep1 的 mean
   從 61.8 拉到 134.4。⇒ `--max-ntok 16` 排除。
2. **MTP draft 步不是 verify 步**：`dp_layers` 在整支 log 裡只有兩個值 ——
   **1（draft，MTP 模組一層）與 40（verify，主模型 40 層可路由層）**。
   混著算 median，描述的是一個不存在的母體。⇒ `--layers 40` 只取 verify。
   （順帶確認了定價模型裡的 `N_LAYERS = 40` 是對的。）
3. **定價要用 mean 不是 median**：總時間 = Σcb ⇒ 能換多少 t/s 取決於 **mean**，
   而 `joint_reconcile.py` 報的是 median。兩個都報，並額外報
   **時間加權佔比 Σcb/Σtotal**。

## 4. 結果（兩趟，各自 3 rep）

### 4.1 視窗決定你讀到 42 還是 74

| 窗口 | A（HEAVY） | B（NOMINAL） |
|---|---|---|
| 全部（含池預熱） | cb mean 47.84，佔 22.3% | 49.62，29.5% |
| rep1 全部（池最冷） | **61.84，27.1%** | **70.59，35.2%** |
| **穩態（對齊 `--warm-skip 64`）** | **36.77，18.3%** | **38.72，25.4%** |

⇒ **74.18 只能出現在「含預熱」或「高 swap」那一格**；交付 regime 用不到它。

### 4.2 按 ntok 分層（`cb` 對 ntok 極不線性，不分層就是又一次跨口徑）

| 穩態 | A: cb mean / med / 佔比 | B: cb mean / med / 佔比 |
|---|---|---|
| ntok=2 | 14.40 / 10.87 / 10.9% | 13.16 / 9.15 / 14.4% |
| ntok=3 | 16.58 / 11.50 / 10.7% | 11.61 / 9.40 / 10.8% |
| **ntok=4**（交付飽和形狀） | **50.33 / 42.09 / 21.1%** | **51.50 / 42.24 / 28.9%** |

**★ 兩趟的 `cb` median 是 42.09 與 42.24** —— 與歷史上那一個 **42.04** 對上了，
而且是兩支獨立 log、兩種熱壓狀態下各自對上的。mean 則是 50.33 / 51.50。

⇒ 交付形狀下 **`cb = 42（median）~ 51（mean）ms`**。
⇒ 定價改用 **50.40**（Σcb 才是吞吐相關量），**42.04** 留下來當保守下界。

## 5. 代回定價（`window_joint_ev.py`，selftest **54/54**）

`cov=0.860`（ρ 實測）、`lead=1.31~2.27 ms`（EARLY 臂實測）：

| cb | 單做 ρ | 結構上限（cb 全藏） |
|---|---|---|
| **42.04**（median） | **14.39 t/s（+14.5%）** | 14.79（+17.7%） |
| **50.40**（mean） | **14.89 t/s（+18.4%）** | 15.41（+22.6%） |
| ~~74.18~~（含預熱／高 swap） | ~~15.53~16.50~~ | ~~17.46（+38.9%）~~ |

**⇒ 報對外的是 +14.5%~+19%，不是 +24~31%。**

### 5.1 而且「視窗」這一項現在被機械排除了

`lead` 在實測區間 1.31 → 3.00 ms 換任何值，**t/s 完全不變**（已做成 selftest 不變式
`lead-insensitive`）。因為 `cov·F = 0.860 × 50.40/40 = 1.084 ms < 1.31 ms`
⇒ **視窗在任何情形下都綁不住**，遮多少只由覆蓋率決定。
上一輪「下一個實驗去量視窗百分位」的必要性，到此收尾 —— 它已經不是主導項。

### 5.2 順帶：cb=74.18 那一格本身自相矛盾

`cb=74.18` 全藏 ⇒ step 178.56 ms（17.46 t/s）。但同批 log 的 `gpu_sum = 181.56 ms`
⇒ GPU 忙碌地板是 **17.17 t/s**，而且 step 不可能低於 GPU busy。
**17.46 > 17.17 ⇒ 那一格越過了地板。**（旁證，不是主證據：gpu_sum 出自另一支 log。）

## 6. 剩餘的不確定性（老實說）

1. **median 42 vs mean 50**：`cb` 在 ntok=4 下是長尾的（p10 ≈ 19.5、p90 ≈ 126）。
   吞吐該用 mean，但 mean 被少數「池重新收斂」的步拉高。⇒ 給區間，不給點。
2. **我這兩趟每 token 比錨點慢**：錨點 247.98/3.117 = 79.6 ms/token；
   B 是 178/1.36 = 131 ms/token（1.65×）。機器現在 swap 用了 4.99 GB。
   ⇒ 若錨點當天的水位更好，`cb` 可能比我現在量到的還小 ⇒ **偏 42 那一端**。
3. **accept 率對不上**：錨點 tok/step = 3.117，我這兩趟只有 1.36
   （`mean_ntok` 一樣是 3.4~3.5，所以是 accept rate 39% vs ~78%）。
   成因沒查（llama-bench 的隨機 token 填充／當天條件），
   **不影響 `cb`（已按 ntok 分層），但代表這兩趟的絕對 t/s 不等於錨點**。

## 7. 結論與下一步

- **`cb` 定讞為交付 cell 穩態 ntok=4 的 42（median）~ 51（mean）ms**；
  `74.18` 撤回（它是含池預熱／高 swap 那一格的數）。
- **真實收益 +14.5%~+19%（14.4~15.0 t/s）**。結構上限 +17.7%~+22.6%。
- **下一步不再需要量視窗**（§5.1 已機械排除）。剩下的槓桿是**覆蓋率**
  （ρ 已 0.860，越過 0.70 飽和點）與 **`insert = 4.76 ms/步` 這個估值的實測**。

## 附：重現

```sh
mkdir -p /tmp/cb_delivery_B
CGC_DECODE_PROFILE=1 python3 scripts/check/llama_bench_matrix.py \
  --arms prod25-stream --reps 3 --prompt 0 --gen 128 --depths 512 \
  --batch 512 --ctx-size 4096 --warm-skip 64 --spec-type draft-mtp \
  --workdir /tmp/cb_delivery_B
python3 scripts/check/cb_delivery_read.py \
  /tmp/cb_delivery_B/llama_bench_prod25-stream_p0_n128_d512_r3.stderr.log
python3 scripts/check/window_joint_ev.py --cb 50.40 --cov 0.860 --lead 1.31
```

新增：`scripts/check/cb_delivery_read.py`（selftest **16/16**）。
`window_joint_ev.py` 的 `MEASURED_CB` 已改寫出處並新增 `CB_DELIVERY` / `CB_DELIVERY_LO`。
