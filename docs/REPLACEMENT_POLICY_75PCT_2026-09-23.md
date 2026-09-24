# 換替換策略能到 75% hit 嗎 —— 有，而且已經在樹上，實測 89.4%

日期：2026-09-23　｜　方法：靜態推導 + **2 趟實跑**（同 cell／同 build，只差一個 env）
錨點：交付 cell 12.57 t/s / step 247.98 ms

---

## 0. 一句話

**75% 不是天花板，是路上的一個點。** 樹上現成的 `LLAMA_EXPERT_CACHE_PIN_PROFILE`（per-layer
靜態釘住 hot set）今晚實測把 count-cold 從 **34.2% 打到 10.6%**（hit ≈ 65.8% → **89.4%**），
而真正的天花板（counterfactual top-K）是 **98.8~98.9%**。

⚠ **但 t/s 收益未驗證** —— 見 §4，那裡有一條 09-21 的反向否證。

---

## 1. 前提：路由是極度重尾的（這是全部槓桿的來源）

`CGC_MASSCOV=1` 一趟就拿到了（路由是數學，不受 swap／記憶體壓力影響）：

```
uniform-routing baseline = K/n_expert = 143/256 = 55.9%
COUNTERFACTUAL top-K by mass:  K= 96 -> 95.7%
                               K=128 -> 98.2%
                               K=143 -> 98.9%   min=94.1%  max=100.0%
                               K=192 -> 99.8%
ROUTE-DUMP（按「次數」排，非 mass） top-142 coverage = 98.8%
```

**兩個獨立口徑（次數 / 質量）都落在 98.8~98.9%**，而均勻路由的基準只有 55.9%。
⇒ 程式碼註解自己的判據：「coverage far above baseline = heavy-tailed routing,
**placement lever live**」⇒ **這根槓桿是活的，而且很大。**

也就是說：**256 個專家裡，143 個就吃掉了 98.8% 的訪問，剩下 113 個只佔 1.2%。**

---

## 2. 現狀錯配：今天的池只吃到 65.9%

| 指標 | 均勻基準 | 今天實測（SpAc EMA） | 上界（counterfactual） | 錯配 |
|---|---:|---:|---:|---:|
| mass coverage | 55.9% | **65.9%** | 98.9% | **−33.0 pp** |
| SELECTED count-cold | — | **34.2%** | ~1.1% | **+33.1 pp** |

⇒ 池有 143 個槽，理論能裝下 98.8% 的路由，**但實際只裝到 65.9%**。
現行 `CGC_SPAC=1` 的 EMA-utility victim 在**動態地做錯這件事**：它把冷專家填進來、
把熱專家踢出去，形成一個負反饋——每次踢錯就要再付一次 1.0703 MiB 的 pread。

---

## 3. 策略清單：哪些能到 75%，哪些不能

| # | 策略 | 樹上狀態 | 可達 hit | 判決 |
|---|---|---|---:|---|
| **A** | **PIN_PROFILE 靜態釘住**（per-layer top-K hot set + load-time prefill） | ✅ **已完整實作**（`llama-expert-cache.cpp:2214` 解析、`:2417` load-time fill+static-pin） | **89.4%（今晚實測）** | ★ **直接答題，超標** |
| B | SpAc EMA-utility victim | ✅ prod 現行（`CGC_SPAC=1`, α=0.75） | 65.8%（今晚實測） | 現狀 |
| C | 純 LRU（SPAC off） | ✅ 臂 `prod25-stream-nospac` 現成 | 未測 | 應 ≤ B（B 是它的線上改進） |
| D | Belady（最優離線） | ❌ 未實作，只能 replay 評 | ~98.8%（= counterfactual 近似） | 理論天花板，不可上線 |
| E | `LLAMA_EXPERT_CACHE_WIN_PIN=K` window pin | ✅ 已實作，未掃 | 未測 | 次於 A（A 已證） |
| F | LAYER_CAPS 非均勻分配 | ✅ 已實作 | 未測 | 見 §5（逐層落差 24.4 pp） |
| G | LFU／機率式置換 | ❌ 未實作 | — | `KNOB_FULL_MAP_15` §3 明寫**別寫** |

**⇒ 答案：A 一個就夠了，而且不用寫任何新 code。**

---

## 4. ⚠ 收益：命中率拿到了，t/s 還沒

### 樂觀口徑（cb 全在關鍵路徑，`cb ∝ miss`）

| 策略 | count-cold | cb → | step | t/s |
|---|---:|---:|---:|---:|
| base | 34.2% | 42.0 ms | 247.98 | 12.57 |
| **PIN_PROFILE** | **10.6%** | **13.0 ms** | 219.0 | **14.24** |
| 上界 | 1.1% | 1.4 ms | 207.4 | 15.04 |

### 悲觀口徑（09-21 自然實驗：`IO_PATH_AB_2026-09-21.md` §③）

```
池 8 GiB -> 4 GiB：miss 1.91×、capacity miss 3.15×、cb 41.8 -> 90.0 ms（+48.2 ms）
而 t/s 10.28 -> 10.60（不動，還略升）
```

若 cb 真在關鍵路徑，+48.2 ms 應該把 t/s 打到 8.61；實測 10.60 ⇒ **反向壓力完全被吸收**
⇒ 「減 miss ⇒ 提速」在**那一個方向**上被否證。
`KNOB_FULL_MAP_15` §3 因此把 #10（置換策略）排在 pool-size 同場配對之後。

⇒ **真值區間 12.57 ~ 14.24。** 要定它，只有一條路：**同場配對 A/B**（見 §6）。

### 今晚那兩趟 t/s 不能直接引用

```
base  (swap 4411 MiB) : 3.11   ← 崩了（錨點 12.57）
pin   (swap 4615 MiB) : 11.98  ← 在 swap 4615 下仍接近錨點
base2                 : OOM（swap 10018/10240，free 221 MiB）
```

⚠ 環境在第三趟爆掉（pin 的 load-time prefill 讀了 ~5.94 GiB，把系統快取頂滿），
**三趟不是同一個 swap 狀態 ⇒ 3.11 vs 11.98 不能當成 A/B 結果。**
但它是一個值得追的訊號：**在 swap 4615 MiB 下 pin 仍跑出 11.98，而 base 在 4411 下只有 3.11。**

---

## 5. 剩餘落差：89.4% → 98.8%，還差 9.5 pp

pin 之後逐層 `cur` 從 **74.2% 到 99.3%**（上界 93.9%~100%），**落差 mean 9.5 pp、max 24.4 pp**
⇒ 少數層的 profile 嚴重不準。兩個可能成因，都還沒查：

1. **`pick_slot` 的 pass 2 會讓 static pin 讓位**（`:628` `if (pass < 2 && slot_pinned_static) continue;`
   ⇒ pass 2 允許踢 static pin）。冷專家的請求會擠掉熱專家 —— 這正是 static pinning 想擋的事。
   ⚠ 直接改成「pass 2 也跳過」會讓 `pick_slot` 返回 −1 ⇒ 觸發文件已警告的
   `table[e] == -1 → ggml_get_rows 讀 OOB pool row → NaN cascade`。**要連 caller 的契約一起改。**
2. **profile 是同一個 prompt 產生的**（`--prompt 0`）⇒ **oracle 泄題**。泛化性未驗證。

---

## 6. 下一步（按成本排序）

| 步 | 動作 | 成本 | 解什麼 |
|---|---|---|---|
| 0 | **重開機**（swap 現在 10018/10240） | — | 現在環境已污染，任何 t/s 都不可信 |
| 1 | **base vs pin 同場配對**（ABBA ×2，用 `prod25-stream` / `prod25-stream-nospac`） | ~15 min，0 重建 | 定 12.57 ~ 14.24 的真值 |
| 2 | 用**另一個 prompt** 產生 profile，去測 `--prompt 0` | ~10 min | 解 §5-2 的泛化性 |
| 3 | 逐層看落差最大的那層，確認是 pass 2 讓位還是 profile 不準 | 靜態 + 1 趟 | 解 §5-1 |
| 4 | 若 1 證實有收益 ⇒ 才去動 pass 2 的 static-pin 讓位邏輯 | 工程 | 吃 9.5 pp 的剩餘 |

**不要用 `--gen 128` 的單趟數字下結論**（今晚就是這樣失敗的）。

---

## 7. 產物

| 檔 | 內容 |
|---|---|
| `scripts/check/pin_profiles/route_top142_p0_2026-09-23.txt` | PIN_PROFILE 檔（40 行，每行 142 個 expert id） |
| `scripts/check/pin_profiles/masscov_base_2026-09-23.txt` | base 逐層 masscov |
| `scripts/check/pin_profiles/masscov_pin_2026-09-23.txt` | pin 逐層 masscov |

重現：

```sh
# 產生 profile
CGC_MASSCOV=1 LLAMA_EXPERT_CACHE_ROUTE_RECORD=1 \
LLAMA_EXPERT_CACHE_ROUTE_DUMP=/tmp/route_dump.txt \
  .../llama_bench_matrix.py --arms prod25-stream --reps 1 --prompt 0 \
  --gen 128 --depths 512 --batch 512 --ctx-size 4096 --warm-skip 64 --workdir /tmp/x

# 用它靜態釘住
CGC_MASSCOV=1 LLAMA_EXPERT_CACHE_PIN_PROFILE=/tmp/route_dump.txt \
  .../llama_bench_matrix.py --arms prod25-stream ... --workdir /tmp/y
```
