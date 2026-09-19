# M=4（MTP verify）定價：`--spec-type` 接入生產矩陣

日期：2026-09-18 15:32–16:05　build：`efeade7e2`（= HEAD `db6f843ae` 的引擎，無引擎改動）
模型：`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`

---

## 0. 結論（先看這個）

| 問題 | 答案 |
|---|---|
| `--spec-type` 該不該接進矩陣？ | **接了**，但做成**一格**（`decode-spec`），不是自動轉發 |
| M=4 verify 現在量得到嗎？ | **量得到**（`decode-spec` 首次實跑，rc=0） |
| M=4 定價成功了嗎？ | **沒有**：ABBA 兩臂交叉估計 1.248，但 **4 臂中 3 臂 launch=HEAVY**，且**接受率未分離** ⇒ **不可歸因** |
| 有沒有拿到新的生產數字？ | 有：**prod25 的 MTP-on decode = 9.88 t/s**（launch NOMINAL）。這是第一次從登記儀器拿到「開著 MTP」的生產 decode |
| 順便得到什麼？ | **閘門適用普查**：M=2..8 的 mul_mat 有 **37%** 真的會進 small-batch 家族，但 **60% 是 `iq4_xs`，不在型別白名單** |

---

## 1. 為什麼做成「一格」而不是自動轉發

`resolve()` 從 `run_server.sh` 拿到的 server argv 裡，七個 profile **全部**帶
`--spec-type draft-mtp`。所以「把 `--spec-*` 加進 `forward_argv()` 白名單」等於
**把每一個既有 cell 都變成 spec cell**，而 `decode` 那格從此不再等於它過去兩天記錄的數字——
正是 `prod_matrix.py` 存在的理由要防的「同標籤、不同量」。

所以做法是**加法**：

- `llama_bench_matrix.py` 新增 `--spec-type` / `--spec-draft-n-max`（**只有顯式給才會出現在 cmd**）；
- `prod_matrix.py` 新增一格 `decode-spec`，`--spec-draft-n-max` **從該 profile 自己解析出的 server
  argv 讀**（prod25 → 3），不是寫死在這裡——寫死會讓 cell 的 M 變成這支檔案的屬性，而不是 profile 的；
- profile 的 argv 沒有 `--spec-type` 時 `cell_command()` **直接拒絕**（「這格量不到這個 profile 的原樣」）；
- 既有五格逐字不動（拒絕訊息、batch、shape 全部相同）。

實跑指令（`--dry-run` 驗證過）：

```
python3 scripts/check/llama_bench_matrix.py --arms prod25 --prompt 0 --gen 128 --depths 512 \
  --reps 3 --batch 8 --spec-type draft-mtp --spec-draft-n-max 3
```

`decode-spec` 用 `batch="8"`：和 `prefill-m8` 同一個理由——這是 pool-only profile 上引擎**真的會
照做**的唯一寬度（`compat()` 不讓「標籤說 512、實效 8」這種格跑起來），而 verify batch（≤4）裝得進 8。

---

## 2. 第一次實跑：MTP-on 的生產 decode（launch NOMINAL）

`prod25 / decode-spec / -p 0 -n 128 -d 512 -b 8 -r 3`

| | 值 |
|---|---|
| samples_ts | 8.24 / 9.48 / 10.29 |
| avg_ts | 9.34 |
| **platform（丟 rep1）** | **9.88 t/s** |
| rep1 懲罰 | +5.88% |
| launch / worst | **NOMINAL** / HEAVY |

`avg_ts` 仍然是 `n_gen / wall`，所以單位是**輸出 token/s**，可以直接和 MTP-off 的 `decode` 並排。
但它**不是純 kernel 數字**：一輪 = 一次 MTP draft forward + 一次 target verify，
只作用在 verify matmul 上的閘門會被「那一輪裡不是這個 matmul 的部分」稀釋。

⚠️ 沒有「同形狀的 MTP-off 對照格」⇒ **9.88 不可歸因**（不能說 MTP 賺了或賠了多少）。
MTP 淨效應上一輪已用別的儀器量過：配對 ABBA×5 ⇒ **−4%~−6%，輪內漂移 ±12% ⇒ 不可區分**。

---

## 3. 閘門適用普查（`CGC_MM_DBG=1`，零額外 GPU 成本）

一次 `decode-spec` 帶 `CGC_MM_DBG=1` 的 stderr（55,239 行 MMDBG）：

```
family counts: {'mul_mv': 55239}      ← BITIDENT=1（預設）把所有 ne11<=8 都送到 mul_mv
ne11 in [2,8]: {2: 5213, 3: 4010, 4: 19248, 8: 24960}
```

**真正符合 small-batch 家族條件**（`src[1]=f32` **且** `ne00%128==0` **且** `src[0]` 在型別白名單）的：

| src0 | ne11=2 | 3 | 4 | 8 | 合計 |
|---|---|---|---|---|---|
| `f32` | 1,820 | 1,400 | 6,720 | 8,960 | **18,900** |
| `q8_0` | 104 | 80 | 384 | 448 | **1,016** |
| `bf16` | 26 | 20 | 96 | 0? | 142 |
| `q6_K` | 13 | 10 | 48 | — | 71 |
| **`iq4_xs`（不在白名單）** | 3,250 | 2,500 | 12,000 | 15,552 | **33,302** |

兩個結論：

1. **閘門不是結構性失效** —— M≤8 有約 **37%** 的 mul_mat 會被它搬到 small-batch 家族
   （主力是 `f32×f32`，很可能是 attention 側）。所以 A/B 是**有意義的**，不是量空氣。
2. **但 60% 是 `iq4_xs`，它不在白名單** ⇒ 生產主力型別**根本走不到**這條路。
   這正好是 §EN-159 記的那條「把融合的型別閘門擴到 IQ3_S／IQ2_S／IQ4_XS」——
   而本輪 M=4 的結果（下）是對那條**更不利**的證據。

---

## 4. M=4 的 ABBA：**量到了，但不可歸因**

`prod25 / decode-spec / -r 3`，臂序 `on, off, off, on`（`on` = `CGC_MM_BITIDENT=1`，**生產預設**；
`off` = 0，讓合格 op 走 small-batch 家族）。

| # | 臂 | launch | worst | samples | platform |
|---|---|---|---|---|---|
| — | on（可行性跑） | NOMINAL | HEAVY | 8.24 / 9.48 / 10.29 | **9.88** |
| 1 | on | NOMINAL | HEAVY | 7.82 / 8.85 / 8.26 | **8.55** |
| 2 | off | **HEAVY** | HEAVY | 6.59 / 7.52 / 6.92 | **7.22** |
| 3 | off | **HEAVY** | HEAVY | 15.38 / 7.32 / 7.41 | **7.37** |
| 4 | on | **HEAVY** | HEAVY | 11.55 / 6.56 / 12.82 | **9.69** |

- 交叉估計 `on/off` = **1.248**（開著 bit-identical 反而快 ~25%）
- 同臂兩跑：`on` 8.55 vs 9.69（**0.883×**）、`off` 7.22 vs 7.37（0.981×）
- 含可行性跑，同一配置的三次 `on` = 9.88 / 8.55 / 9.69 ⇒ **極差 1.156×**

**為什麼這組數字不能拿來定價**（兩條，各自都足以單獨否決）：

1. **熱態**：4 臂裡 3 臂 launch=HEAVY。本 repo 的規則（lesson `eng-mh-0054`）是 launch 非 NOMINAL
   的樣本可以報，但**不能當生產數字／不能拿來做 ≥門檻的主張**。而且 `on` 的兩次 NOMINAL 相差
   1.156×，效果（1.248）只比同配置自身的極差大一點點。
2. **接受率混淆（未分離）**：`CGC_MM_BITIDENT=0` **改變數值結果** ⇒ MTP draft 的**接受率可能下降**
   ⇒ 每輪接受的 token 變少 ⇒ t/s 掉。這跟「kernel 變慢」是**兩件事**，而 t/s 分不開它們。
   本來該用 `common_speculative_print_stats` 的接受率拆開，但它被
   `llama_log_set(llama_null_log_callback)`（`llama-bench.cpp:2726`）靜音了。

**而且方向和量級都不對**：M=8 那一輪（§EN-159）量到的同一個閘門效應是 **1.7–3.8%、方向與預期相反**；
機制（L2 的重複消費）隨 M 縮小，所以 M=4 **應該更小**，不是 25%。1.248 這麼大反而像
「不是 kernel 效應」的徵兆——也就是上面第 2 條。

⇒ **述句：`decode-spec` 已可達，M=4 仍未定價；「small-batch 家族在 M=4 比較快」這件事沒有被支持，
也沒有被否證，而是被兩個混淆壓住了。**

---

## 5. 生產級 prefill / decode（本次同步量測）

| profile | cell | launch | platform t/s | 可引用？ |
|---|---|---|---|---|
| prod25 | `prefill-m8`（-p 512 -b 8） | **NOMINAL** | **31.69** | ✅ |
| prod25 | `decode-spec`（MTP on, -b 8） | **NOMINAL** | **9.88** | ✅（但無同形狀 MTP-off 對照） |
| prefill250 | `decode`（-b 512） | HEAVY | 8.84 | ❌ |
| prefill250 | `decode-up`（-b 2048） | HEAVY | 7.83 | ❌ |
| prefill250 | `prefill-house`（pp2048） | HEAVY | 155.10 | ❌ |
| prefill250 | `prefill-up`（pp512） | HEAVY | 78.27 | ❌ |

- `prod25` 上 `decode` / `decode-up` / `prefill-house` / `prefill-up` **四格全部被 `compat()` 拒絕**
  （prod25 沒釘 BATCH 也沒開 PREFILL_STREAM ⇒ 引擎 pool-path clamp = 8，標籤說 512/2048 不是實效
  batch）。拒絕是**零 GPU**的，發生在載模型之前。
- ⚠️ **機器在這一輪全程停在 `thermalpressurelevel=2`**：跑了 25 分鐘後沒有冷下來，所以
  prefill250 那批四格 launch 全是 HEAVY。`prefill-house` 的 155.10 t/s 對比記錄中的
  **276.25 / 300.43（NOMINAL）**，差 **~45%** —— 這是熱態，不是退步（HEAD 以來沒有引擎改動），
  但**必須等冷機重跑才能當生產數字**。

---

## 6. 下一步（按「能不能真的裁決」排序）

1. **分離接受率**：`LLAMA_BENCH_SPEC_DBG=1` 是**純 env、不用重建**，會印
   `SPECDBG round: n_done=... draft=...`，可以算出每輪平均接受 token 數。
   一對跑（on/off）就能把「kernel 慢」和「接受率掉」拆開。**這是目前唯一能裁決的低成本動作。**
   （注意：它不在 `run_server.sh` 的 allowlist，要從 shell 環境傳，會一路進到 llama-bench。）
2. **冷機重跑 ABBA**：等 `thermalpressurelevel` 回到 0，四臂都要 NOMINAL launch。
3. **補 `decode-m8` 對照格**（`-p 0 -n 128 -d 512 -b 8`、**不開** spec）：讓 9.88 這個數字能歸因。
4. **不要做**：`iq4_xs` 進 small-batch 家族的擴充（§EN-159 的 (b)）——M=8 已示方向不對，M=4 又更不支持。
