# k=2 vs k=3 認證協定 **v2：llama-bench 交付口徑**（2026-09-23 09:2x）

給【執行 Agent（freebuff）】的交付規格。**本文件取代 `docs/K3_PAIR_CERT_2026-09-23.md`（下稱 v1）**。
分工依 `docs/NEXT_ACTIONS_2026-09-23.md`：本文件只規定怎麼跑、跑多少、什麼時候才准看；
跑數是執行線的事。

---

## 1. 為什麼 v1 不能用，以及它那 5 對現在是什麼

v1 把認證跑在 **server / HTTP 口徑**（`mtp_accept_ab.py`，讀 `timings.draft_n_accepted`）上，
而它的結論要落到 **llama-bench 交付 cell**（12.57 t/s）。那是跨口徑外推：

- **同一個 k=3，兩把尺子差 20%**：server 10.43 t/s vs 交付 12.57 t/s。
- 09-17 的裁定就是因為 server 離散大才統一用 llama-bench（`decode_bench` 12.36／12.95／10.64
  vs llama-bench 三次 **1.1%**）；而 **09-20 起 llama-bench 已能表達 MTP**
  （`prod_profile.py` 的 `--spec-type draft-mtp`）⇒ **例外早就消失，v1 是照抄了過期的前提。**

⇒ **v1 那 5 對降級為 screen**：可以當方向先驗（k2 較快，4/5 對；且 k=3 雙峰），
**不得用來判決，也不得用來決定交付 cell 要不要換 k。**

### 1.1 ★ 更正（09-23 10:0x）：llama-bench **並不比 server 乾淨**，「1.1%」不能拿來推論

上面那句「llama-bench 三次 1.1%」是 09-17 那次場合的數字，**不是交付 cell 的性質**。
查 `Backup/prod_profile/prod_profile_20260920_1230.json` 的原始輸出：

```
prod25-stream  tg d=512  ->  12.57 ± 2.26 t/s   (b=512, reps=3, NOMINAL)
prefill250     tg d=512  ->   9.90 ± 1.25 t/s   (b=512, reps=3, NOMINAL)   ← 同形狀另一 arm
```

而 `llama-bench.cpp:2453` 的 `±` 印的是 `stdev_ts()` ＝ **reps 的樣本標準差**（不是 CI、不是 SE）
⇒ **交付 cell 的單次 rep sd = 2.26 t/s = 均值的 18.0%**，與 server 口徑的單臂噪音（配對後 17.5%）
**同一量級**。

⇒ 我上一輪寫的「bench 的配對 sd 應顯著更小、要的對數可能遠少於 8」**是錯的，撤回。**
正確的說法是：兩把尺子一樣吵，差別只在於 **bench 的結論能直接落到交付 cell**，server 的不行。

同一筆的精度也該看見（reps=3、sd=2.26、df=2）：
`95% CI = 12.57 ± 5.61 → [6.96, 18.18]`，半寬 **±45%**。
⇒ **12.57 只能當配對比較的一臂，不能拿去和另一次 anchor 的單點數字相比。**

## 2. 第 1 步已經打通：交付口徑現在能表達 k

`scripts/check/prod_profile.py` 新增 `--spec-draft-n-max N`（1..16）：

- 底層早就支援（`llama_bench_matrix.py:544`，且 `:435` 會寫進 json）；之前只是
  `prod_profile.py` 把 decode 形狀寫成字面列表、CLI 沒暴露。
- **不傳 ⇒ 生成的命令行與既有紀錄逐字相同**（已驗證：`--emit-spec` 的 shape 不含該旗標），
  所以 12.57 那筆不受影響。
- 傳值時 `declared["spec_draft_n_max"]` 與 `decode_cell.shape` 都會記，且 `rec` 會從
  子行程讀回（`a0.get("spec_draft_n_max")`）⇒ 旗標若中途掉了會在紀錄裡對不上，不會靜默。
- 越界（0 或 17）直接拒絕，不會把「無效 k」送進 llama-bench。

## 3. 事先登記（寫死，跑完不准改）

| 項目 | 登記值 |
|---|---|
| 口徑 | **`prod_profile.py`（llama-bench）**，不是 server |
| 一對 = | 一次 k=2 launch ＋ 一次 k=3 launch，ABBA 旋轉 |
| 主要統計量 | 一對的 **arm mean 差**（k2 − k3，t/s）|
| 檢定 | 配對 t，雙尾，α = 0.05 |
| **階段 1（標定）** | 先跑 **n₁ = 4 對**，用 `--sd-only`，**只估 sd，不判 t** |
| **階段 2（判決）** | 依階段 1 的 sd 事先登記 N，跑滿判 **一次** |
| N 的規則（寫死） | `N = min(12, max(8, planned_n_for(diff, sd_point)))` |
| 退出規則 | 若 `planned_n_for(diff, sd_point) > 12` ⇒ **放棄認證**，直接走 §6(B) 的機制路線 |
| 每一 arm 的條件 | prod25；`prod_profile.py` 自己會 `wait_nominal`（NOMINAL 才發射）|
| 順序 | r1 = k2 先，r2 = k3 先，r3 = k2 先，r4 = k3 先（後續同理交替）|

**為什麼是「點估 sd」而不是「sd 的 95% 上界」**：上界在 n=4 時極寬（見 §5），
照它規劃 N 會爆到幾十對 —— 而這個實驗的全部價值就是「比 25 臂便宜一個數量級」。
所以規則改成：**上界只印出來當警示，N 用點估，但把「N > 12 就放棄」當作事先寫死的退場條件。**

## 4. 指令

```sh
cd /Users/alexchuang/Documents/flashkv-devserver
mkdir -p /tmp/kb

# 0) 窗口閘門（兩個都必須空）
lsof -nP -iTCP:8080 -sTCP:LISTEN
pgrep -fl 'llama-bench|llama-server|decode_sweep|mtp_accept_ab'
python3 -c "import sys; sys.path.insert(0,'scripts/check'); import server_window as sw; \
print(sw.decision())"        # admits 要 True

# 1) 一對 = 兩個 launch。檔名必須 `rN_` 開頭且含 k2/k3 token（工具靠檔名配對）
#    k=3 臂刻意「不傳」 --spec-draft-n-max：這樣它的命令行與 12.57 那次逐字相同。
#    k=2 臂傳 2。
python3 scripts/check/prod_profile.py --axes decode --no-ref --reps 3 \
    --json /tmp/kb/r1_a_k2.json --spec-draft-n-max 2
python3 scripts/check/prod_profile.py --axes decode --no-ref --reps 3 \
    --json /tmp/kb/r1_b_k3.json
# ... r2_b_k3 / r2_a_k2 / r3_a_k2 / r3_b_k3 / r4_b_k3 / r4_a_k2  （共 8 個 launch = 4 對）

# 2) 階段 1：只看 sd，不看 t
python3 scripts/check/k_swing_decompose.py --dir /tmp/kb --groups k2,k3 --sd-only
```

工具在 `--sd-only` 下**故意不印 t 也不印判決**；那不是缺功能，是讓「用同一批對既估 sd 又檢定」
這件事做不出來。它會印：

```
  n=4 pairs   sd X.XX t/s   95% upper bound Y.YY t/s
  planning diff +Z.ZZ t/s -- OBSERVED, used only to size n.
  n needed at point sd       : N1
  n needed at sd upper bound : N2
```

⇒ 取 **N1** 套 §3 的規則得到 N（<8 用 8，>12 就放棄），**當場寫進本文件 §3 的表格**（或另開一份
append-only 的登記），然後才開始跑剩下的對。

```sh
# 3) 階段 2：跑滿 N 對之後，判一次
python3 scripts/check/k_swing_decompose.py --dir /tmp/kb --groups k2,k3 \
    --paired --planned-n N
```

n < N 時工具回 `NOT YET`，那是正確輸出。**它告訴你還欠幾對，不是錯誤。**

## 5. 階段 1 現在就能預演的一次警示（用 v1 那 5 對，僅供理解尺子）

`--sd-only` 跑在 v1 的 server 口徑 5 對上：

```
  n=5 pairs   sd 1.83 t/s   95% upper bound 4.35 t/s
  n needed at point sd       : 7
  n needed at sd upper bound : 28
```

**界是 4.35，不是 1.83。** 5 對估出來的 sd 本身極不穩定 ⇒ 「n=8 有餘裕」這句話只在點估下成立，
按 95% 上界要 28 對。這正是 §3 把「N > 12 就放棄」寫死的原因：**不要讓一個 sd 點估把你騙進
一場 28 對的實驗。**

（注意這是 **server 口徑**的數字。~~llama-bench 的離散小得多~~ —— **這句已撤回，見 §1.1**：
交付 cell 的 rep sd 是 18%，同量級。bench 上 sd 是多少是**待測**，不預設。）

### 5.1 ★ 加 reps 是偽槓桿：配對 sd 有地板，地板由 **launch 級項**決定

k=3 那個 **13.16% 的 launch 級項**（F=4.49, df 4/10）是「開 process 才被決定」的項
⇒ **同一個 launch 內的 reps 共享它** ⇒ 加 reps 只能平均掉 rep 級噪音，**降不下去這一項**。
所以配對 diff 的 sd 有一個地板 ≈ `√2 × σ_launch`（k=2 沒測到該項，視為 0）。

假設 bench 上 launch 級項同比例，基線 12.57 t/s，目標效應 16.4% ＝ **2.06 t/s**：

| launch 級 sd | σ_launch | 配對 sd 地板 `√2·σ` | 檢出 16.4% 需要 |
|---|---|---|---|
| 5% | 0.63 | 0.89 | **4 對** |
| 8% | 1.01 | 1.42 | **5 對** |
| 10% | 1.26 | 1.78 | **6 對** |
| **13.2%（k=3 實測）** | **1.65** | **2.34** | **8 對** |
| 16% | 2.01 | 2.84 | **10 對** |

把 rep 級（sd 2.26）與 launch 級（σ_l）合起來看「加 reps 有沒有用」（目標同為 2.06 t/s）：

| σ_launch | r=3 | r=6 | r=12 |
|---|---|---|---|
| 0.63 | 2.05 → 7 對 | 1.58 → 5 對 | 1.28 → 4 對 |
| 1.00 | 2.32 → 8 對 | 1.92 → 6 對 | 1.69 → 6 對 |
| **1.65** | 2.97 → **11 對** | 2.67 → 9 對 | 2.51 → **9 對** |

⇒ **reps 從 3 加到 12（wall time ×4）只把需要對數從 11 降到 9；卡住的是 launch 級項。**
所以協定**維持 `--reps 3`**（也才能和 12.57 那次可比），要檢出力就**加對數，不要加 reps**。

## 6. 跑完的三條路（事先寫好）

### (A) `CERTIFIED`（|t| ≥ crit，方向為 k2 較快）
1. **這次可以直接說「交付 cell 換 k=2」** —— 因為認證與交付在同一口徑上（v1 不行）。
   做法：llama-bench 加 `--spec-draft-n-max 2`（不用 rebuild）。
2. **仍然要重新 anchor**：換 k 後 12.57 這個數字失效，要重跑一次交付 cell。
   新的數字是多少**事先不知道**（k=2 走 plain `mul_mv`，k=3 的 ntok=4 跨 `ne11>=4`
   走 `mul_mv_ext`，是兩條 PSO）。
3. 天花板分析給的上界是 **13.5 t/s**（L3 誠實上界 ~+7%）；若 anchor 跑出明顯高於這個數，
   先懷疑口徑而不是慶祝。

### (B) `NOT SEPARATED`，或階段 1 就觸發「N > 12 放棄」
**不要再加對。** 改走 `pread_usec`（`llama-expert-cache.cpp:432`，免 rebuild，配方見 v1 §4）
直接看 k=3 的慢狀態是不是池 IO 服務時間 —— 那是**機制**問題，不是樣本數問題。
若 `pread_usec` 也不能分離 ⇒ 把 k 當已知噪音源：k=2 當便宜 screen，winner 回 k=3 確認一次。

### (C) `OVERRUN`（跑超過 N 對）
判決已經在 N 發生過。多跑的是另一個實驗，要另外事先登記。

## 7. 誠實欄

- 階段 2 的 N 是用**同一批對的 sd** 估出來的（internal pilot）⇒ 名目 α 略失真，無法完全補償。
  已做的補償：N 事先寫死、只判一次、且「>12 就放棄」也事先寫死。
- v1 的 5 對我看過了（t=2.08，p≈0.11）⇒ 那個方向先驗會影響我對 v2 結果的預期，**無法補償**。
- k=2 臂的命令行多一個 `--spec-draft-n-max 2`；k=3 臂刻意不傳以保住與 12.57 那次逐字相同
  ⇒ 兩臂在**命令行層面**不對稱，但在 llama-bench 層面等價（`llama-bench.cpp:436` 預設 3、
  `:1388` 只做範圍檢查）。若你寧可對稱，兩臂都傳（2 與 3），但那樣 k=3 臂就不再與 12.57 那次
  逐字相同 —— **兩種不對稱二選一，寫清楚就好，不要當成沒事。**
- 每次 `prod_profile.py` launch 會自帶一次最長 420 s 的 `wait_nominal`；4 對 = 8 launch
  的盒時要把它算進去。
- **「16.4% 會不會在 bench 上重現」是未知，而且有理由預期它更小**：那個效應的來源是
  「k=3 有些 launch 掉進慢狀態」，而 server 的 k=3 基線（10.43）比交付 cell（12.57）低 20%
  ⇒ bench 的 k=3 可能本來就較少掉慢狀態 ⇒ 可被 k=2 救回來的那一段比較短。
  另一個方向的先驗：bench 的 rep sd = 18% 顯示 **rep 之間就在切換狀態** ⇒ 機制在 bench 上
  確實存在。**兩邊都只是先驗，判決只能來自實跑。**
- §5.1 的地板表是**假設 bench 的 launch 級項與 server 同比例**（13.16%）推出來的；
  bench 的 launch 結構不同（單 process 多 rep、`--warm-skip 64`）⇒ 該比例可能不同。
  表給的是「對數怎麼隨這個比例走」，不是預測。
- 第 1 步的工具改動已過 `--emit-spec`／`--dry-run` 等零 GPU 驗證，**沒有實跑過 GPU**；
  selftest 47/47（含 chi² 的三個表值錨）。
