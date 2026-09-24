# h 已量到 ⇒ **方案 A（預指派 slot）判死**（2026-09-24）

> **性質**：實測（2 趟獨立交付 cell、GPU、`CGC_IDSEQ_DUMP` 真 ids 序列）　**0 重建**
> **一句話**：可用的預測源 `prev_union` 實測 **h = 0.032 / 0.022**（交付形狀 ntok=4），
> 對上門檻 0.65／0.75；而就算預測器完美，**「拓寬到 top-16」這個建議在數學上就不夠**
> （每次 hook 的真實 union 均值 **19.0~19.8** > 16 ⇒ top-16 的 h 結構上界只有 **18~27%**）
> ⇒ **A 判死，不是「還要優化」，是「這個方向的收益不存在」。**

---

## 1. 怎麼量的（可完整複算）

`CGC_IDSEQ_DUMP` 的儀器**早已編進 binary**，本輪**沒有 build**（也就不動共用產物）：

```
strings src/llama.cpp/build/bin/libllama.0.0.578.dylib | grep CGC_IDSEQ_DUMP   # ⇒ 1（命中）
```

產出 twice 獨立跑（非 reps，是兩個 process）：

```sh
# 閘門：8080 無 listener、無別線量測行程
export CGC_IDSEQ_DUMP=<repo>/Backup/idseq/idseq_delivery_$(date +%H%M%S).txt
python3 scripts/check/prod_profile.py --axes decode --reps 1 --no-ref
python3 scripts/check/idseq_h_curve.py Backup/idseq/idseq_delivery_<TAG>.txt
```

| 趟 | tag | t/s | rows（ctx=0, ntok=4） | 配對數 n（= 被定價的 comparison 數） |
|---|---|---:|---:|---:|
| 1 | 180228 | 7.24 | 1760 | 1804 |
| 2 | 180430 | 9.62 | 2080 | 2132 |

（`n` 含 ntok=4 rows 總數 1804/2132 —— 其中 ctx=1（MTP draft）各 44/52 行；腳本按 `(ctx, il)`
分層後對相鄰呼叫配對，且每層最後一行是部分步（ntok≠4），故 comparison 數恰等於 rows 數。）

全程 NOMINAL / memory gate ok。trace 格式 `<pass> <ctx_is_mtp> <il> <ntok> <k> <ids...>`，
prefill 的 512-token 呼叫已被上游 `n_tokens <= 8` 閘掉。

---

## 2. 結果：h 遠低於任何一個門檻

**可交付給 A 的預測源 = `prev_union`**（上一次 hook 呼叫自己的 ids —— A 在 submit 當下唯一真的能拿到的東西）：

| 趟 | cov | **h** | union_avg | uncovered_avg |
|---|---:|---:|---:|---:|
| 1 | 0.5601 | **0.0322** | 19.02 | 8.92 |
| 2 | 0.5134 | **0.0220** | 19.91 | 10.04 |

門檻：`GAP_FIX_WHITEPAPER` §6 寫的是 **0.65**；後來因為「A 要贏過 ρ 上界 14.39 才是淨賺」
而收緊到 **0.75**。兩個都 **FAIL，且 fail 的量級是 0.62~0.72**（不是擦邊）。

### ★ 這正是 `idseq_h_curve.py` 當初被寫出來要抓的陷阱，現在有真數據了
覆蓋率看起來「有一半」(0.51~0.56)，全中率卻是 **2~3%**。
⇒ **「coverage 好看」完全不能拿來定價 A**。白皮書 §6 當初的天真估計 `0.87⁸ = 33%`
比實測 **樂觀了 10 倍**。

---

## 3. 更強的一條：**與預測器無關的結構性上界**

每次 hook 呼叫的真實專家聯集大小（`ntok=4`）：

| 趟 | union 均值 | min | max |
|---|---:|---:|---:|
| 1 | 19.01 | 8 | 32 |
| 2 | 19.81 | 11 | 32 |

⇒ **候選集大小 K 若小於該次的 union，那次註定 uncovered > 0 ⇒ 必然 miss**。所以：

| 候選集大小 K | 可命中的呼叫佔比 ⇒ **h 的硬上界** |
|---:|---:|
| 8 | 1.6% / 0.0% |
| **16（原建議「拓寬到 top-16」）** | **26.8% / 18.0%** |
| 20 | 63.0% / 60.7% |
| 24 | 90.6% / 89.5% |
| 32 | 100% |

⇒ **`GAP_ELIMINATION_PLAN` §3「必須用比 top-8 寬的候選集（上一 token 的 top-16）」這句建議，
在本cell 下數學上達不成 0.75**：top-16 的天花板只有 **18~27%**。
要摸到 0.75 得把候選推到 **K ≥ 22~24**，而那時還要預測器完美命中全部 —— 而最便宜的預測器
在 K≈8 等級時實測只有 2~3%。

---

## 4. 加寬也救不回來

| 候選集 | 趟1 h | 趟2 h | 性質 |
|---|---:|---:|---|
| `prev_union`（A 真能用） | 0.0322 | 0.0220 | **最便宜、非 oracle** |
| `prev｜freq64` | 0.1663 | 0.2144 | 仍遠低於門檻 |
| `prev｜freq143` | 0.8476 | 0.9015 | ⚠ **oracle**：全 trace 頻率熱集，等價 PIN_PROFILE 的路線 |
| `freq143` | 0.8420 | 0.8917 | ⚠ 同上 |

⇒ **唯一能過 0.75 的組合全都依賴 oracle 靜態熱集**，而那條路線已經被撤回
（`RHO_MAXQ_SETTLED`／§EN-480：profile 由 `--prompt 0` 生 ⇒ 泄題；t/s **−23%**；
且 `pick_slot` 的 `pass<2` 契約未改）。⇒ **A 沒有第二條路。**

---

## 5. 判定

| 問題 | 答案 |
|---|---|
| h 有實測值了嗎 | ✅ **有**：0.032 / 0.022（ntok=4，n=1804/2132，兩趟一致） |
| A 值得做嗎 | ⛔ **判死**。低於 0.65 門檻 0.62、低於 0.75 門檻 0.72 |
| 拓寬候選集能救嗎 | ⛔ 原建議的 top-16 結構上界只有 18~27% |
| oracle 熱集能救嗎 | ⚠ 數字上能（0.84~0.90）但該路線已因泄題 + −23% 撤回 |
| 下一步 | 依使用者裁決：讓給 **ρ**（上界 14.39；但見下方 caveat）或 **「減少段數」**（41→1 段 ~9.9 ms/step，首次定價、**未驗**） |

### ⚠ 順手撿到的一個交叉核對（對 ρ 那條線有用）
本趟 2 號跑（**無任何 ρ env**、純交付 cell）讀到 **t/s 9.62**。
而另一條線今日註解主張「rho-batch **9.62** vs probe-only 12.04」。
⇒ **若今天 ρ-free 的基線本身就可能落在 ~9.6**，那「rho-batch = 9.62」到底是賺是賠，
要看它同場的對照臂，不能只看絕對值。（本輪不是那場實驗的配對，**不能替它下結論**，
只作為「該組數字需要重錨」的提示。）
另：本輪兩趟 7.24 / 9.62 也再次印證單臂不要直接比。

---

## 6. Caveat（誠實邊界）

- **單 prompt**（`--prompt 0`，交付 cell 的既有慣例）⇒ 換 prompt population 後 union 分佈可能變；
  但 union 均值 ~19~20 是**路由形狀**（ntok=4 × top-8）的直接結果，不是 prompt 特性，
  ⇒ 結構性上界那一條**對 population 不敏感**。
- trace 只含被選中的 top-8 ids，**不含完整 logit 排序** ⇒ 「標準意義的 prev-step top-16」
  無法用本 trace 精確定價；本文用 `prev｜freq16/32/64` 作代理，並另給了**不依賴此細節的硬上界**。
- 今天環境的絕對 t/s（7.24 / 9.62）低於交付錨點 12.57 ⇒ **t/s 不可引用，h 可以**（h 是路由序列屬性）。
- `prev_union` 的 cov 0.51~0.56 與 ρ 的 `cov_uni` 0.854 **不是同一個預測源**（ρ 用的是 pre-attn
  影子 router），兩者不矛盾：**「用上一步的 ids 猜這一步」就是比「用影子 router 算」弱得多。**

## 7. 產出檔

- `Backup/idseq/idseq_delivery_180228.txt`（2491 行，267 KB）
- `Backup/idseq/idseq_delivery_180430.txt`（3367 行，357 KB）
- 分析器：`scripts/check/idseq_h_curve.py`（`--self-test` 8/8）
- 儀器：`src/llama.cpp/src/llama-context.cpp:5528-5571`
