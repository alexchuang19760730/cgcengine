# K5（新）—「每節點 ~45.6 µs 常數」假說與驗證計畫（2026-09-21 18:5x）

> 起因：K0–K4 全部走完（K1 禁區、K2 證偽、K3 實測 2.3%、K4 否證）之後，
> 「下一步是啥」的答案不再是「再找一個能少做點工作的槓桿」——那類槓桿已經全封。
> 本篇提出**唯一還沒被證偽的那個未知數**，並給出分辨它三個候選解釋的最便宜做法。
>
> **本篇 0 重建、0 起 server。** 全部數字來自今天 K3 那輪已留存的 raw stderr
> （`Backup/phase_decomp/K3/raw/count2.stderr`）與 K2 的圖 dump。

---

## 1. 一個之前沒人提過的規律：步時 ≈ 51 µs × 工作節點數

`CGC-GPUOPS` 按步印 `total`／`nodes_all`／`nodes_work`，而這一輪裡混著兩種步
（**20 步是 45-node 的 MTP draft 圖、17 步是 3800–3806 的真 decode 步**，混著取中位數全錯）：

| 步型 | n | nodes_work | total（中位） | µs / work_node |
|---|---:|---:|---:|---:|
| MTP draft | 20 | 30 | **0.96 ms** | 32.2 |
| 真 decode | 17 | 2365 | **119.63 ms** | 50.6 |

兩點回歸：**斜率 = 50.8 µs/node，截距 = −0.56 ms ≈ 0**。

⇒ **一步的 GPU 時間幾乎完全由「會產生工作的節點數」決定，截距為零。**
它不由 bytes 決定（K2 已證偽）、不由 dispatch 數決定（K3 只 2.3%）、不由 FLOPs 決定。

---

## 2. 而且那個斜率是一個**常數**，不是各自的工作時間

真 decode 步（n=17）的 per-op `wcntw`（實測時間）÷ `nd`（實測節點數）：

| OP | nd/step | work ms | work% | **µs/node** |
|---|---:|---:|---:|---:|
| `MUL_MAT` | 426 | 22.74 | 18.9% | 53.4 |
| `ADD` | 421 | 18.32 | 15.0% | 43.5 |
| `MUL` | 278 | 13.13 | 11.0% | 47.2 |
| `GET_ROWS` | 159 | 7.25 | 6.0% | **45.6** |
| `MUL_MAT_ID` | 117 | 5.33 | 4.3% | **45.6** |
| `RMS_NORM` | 130 | 4.96 | 4.2% | 38.2 |
| `GLU` | 78 | 3.56 | 2.9% | **45.6** |
| `SCALE` | 60 | 2.74 | 2.2% | **45.7** |
| `SUM_ROWS` | 39 | 1.78 | 1.4% | **45.6** |
| `DIV` | 39 | 1.78 | 1.4% | **45.6** |
| `CLAMP` | 39 | 1.78 | 1.4% | **45.6** |
| `SSM_CONV` | 30 | 1.37 | 1.1% | **45.7** |
| `CONCAT` | 30 | 1.37 | 1.1% | **45.7** |
| `GATED_DELTA_NET` | 30 | 4.80 | 4.0% | **160.0** |
| `FLASH_ATTN_EXT` | 10 | 2.66 | 2.4% | **266.0** |
| `MUL_MAT`(對照) | 426 | 22.74 | 18.9% | 53.4 |
| `SOFT_MAX` | 40 | 0.43 | 0.4% | 10.8 |
| `UNARY` | 169 | 5.86 | 5.0% | 34.7 |

**十一種 op —— 數據量差幾個量級、演算法完全不同 —— 全部落在 45.6 ± 0.1 µs。**

這不可能是「各自的工作時間」。**它是一個常數。**
（反證：`SOFT_MAX` 10.8、`UNARY` 34.7、`GATED_DELTA_NET` 160 —— 有 op 不在這條線上，
所以它不是單純的儀器量化步長。）

---

## 3. 扣掉 K3 已量過的那部分，剩下 **~28 µs 是融合省不掉的**

K3 實測（`docs/K3_PRICE_MEASURED_2026-09-21.md`）：
OFF 臂慢 11.64%、`gpu_union` 91.79 ms ⇒ 一個 dispatch 的全成本 = 91.79 × 0.0195% = **17.9 µs**。

而這裡 45.6 µs 是「一個 work node 的全部成本」。

```
45.6 µs  ──  17.9 µs（融合能省：dispatch + 中間結果落地）  ──  27.7 µs（融合省不掉）
```

**那 27.7 µs 才是現在唯一還沒被命名的東西。**
而它乘在 **2365 個 work node** 上 ⇒ 65.5 ms/步，佔交付步時的大頭。

---

## 4. 三個候選解釋（互斥，可分辨）

### (A) grid 太小 —— 小 op 只用了 GPU 的一小角

`ggml-metal-ops.cpp:915-921`（`ggml_metal_op_add`，其它 elementwise 同構）：

```c
const int nth_max = MIN(256, pipeline_max_threads_per_threadgroup);
int nth = 1;
while (2*nth < args.ne0 && nth < nth_max) { nth *= 2; }   // ne0=2048 => nth=256
ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);
//                                            ^^^^^^^^^^^^^^^^^ grid = ne11 x ne12 x ne13
```

⇒ **grid 完全由「行數」決定，`ne0`（行長）只決定每組幾個 thread。**
decode 時 `ne11 = T ≈ 1`，`ne12 = ne13 = 1` ⇒ **grid = 1 組 × 256 threads**。
M4 = **8-core GPU**（`system_profiler` 實讀），若每 core 駐留 128 threads ⇒ 全機 ~1024
⇒ **一個 ADD 只動用 GPU 的 ~25%**，而它要處理 2048 個 f32。

若成立 ⇒ 這是一條**不在 G2 禁區**的路：elementwise 沒有累加順序問題，
**沿 `ne0` 把 grid 撐開（按元素切、不按行切）是逐位元安全的**。
elementwise 群佔 work 時間 **~48%**；若 45.6 降到 ~10 µs ⇒ 上界 **~30%**，遠超 12.8% 門檻。

⚠️ **但現在還不能這麼說**：下面 §5 的反證。

### (B) Metal 單佇列的 per-kernel 固定成本（launch + ramp-up + drain）

與 kernel 大小無關，**不可從 ggml 側改** ⇒ 天花板就此鎖死，專案可正式收尾。

### (C) `wcntw` 儀器口徑問題

`wcntw` 未必是「kernel 執行時間」（可能是累計 counter、或含 command-buffer 邊界）。
若如此，45.6 這個數字本身就是假的。

---

## 5. ⚠️ 一個必須先清掉的反證（它讓 (A) 目前**不成立**）

用 K2 的圖 dump（`Backup/cgc_logs/llama_server_20260921_173218.log`，20671 節點）
配上面那個 grid 公式重算，**prefill 形狀**下的實際 threads：

```
全圖 threads 中位數 = 2048   （> M4 駐留 ~1024）
  < 128 threads 的節點佔 13%
  < 1024 threads 的節點佔 38%
```

⇒ **在 prefill 形狀下 (A) 不成立**（中位數已經超出駐留量）。

**但那份圖 dump 是 prefill**（`CGC_GRPH_DBG` 只在前 6 次 fire）。
decode 時 `ne11` 從 prefill 的幾百掉到 T≈1 ⇒ **grid 會小一兩個數量級**。
⇒ (A) 在 decode 下可能成立，但**必須拿 decode 形狀的 `ne` 重算才能說**。

---

## 6. 驗證計畫（0 重建，兩條路，先做便宜的）

### 6.1 便宜版：用既有的 M 掃描分辨 (A) vs (B)

同一 build、只改 `T`（= `ne11`）：
- **(A) grid 太小** ⇒ µs/node 應隨 `T` 增大而**下降**（更多 threadgroup ⇒ 更多並行度）
- **(B) 固定成本** ⇒ µs/node 與 `T` **無關**

既有的 `M_WIDTH_CURVE_2026-09-21.md` 就是這個掃描，**先看它有沒有留下 per-op 的 `wcntw`**。
若有 ⇒ 0 成本即可分辨。

### 6.2 貴一點但決定性：拿 decode 形狀的 `ne`

`CGC_GRPH_DBG` 現在只 fire 前 6 次（prefill）。要讓它在 decode 步也 dump
（改 `ggml-backend.cpp:2091-2099` 的計數條件，**這是本線的檔案但要重建**）
⇒ 拿到 decode `ne` ⇒ 配 grid 公式算 threads ⇒ 與 1024 比。

⚠️ 重建會蓋掉別條線正在 map 的 dylib ⇒ **必須等乾淨窗口**（既有約定）。

### 6.3 順手排除 (C)

讀 `ggml-backend.cpp` 裡 `wcntw` 的定義，確認它是「kernel 執行時間」還是別的。
純讀碼，0 代價，**應該最先做**。

---

## 7. 判決樹

```
先讀 wcntw 定義 ──(不是執行時間)──> (C) 成立，45.6 作廢，回到「K0–K4 全封」的收尾
      │
   是執行時間
      ↓
  M 掃描看 µs/node vs T
      ├── 隨 T 下降 ──> (A) 成立 ──> K5 上線：elementwise 撐 grid，上界 ~30%，bit-safe
      └── 與 T 無關 ──> (B) 成立 ──> 天花板鎖死，專案收尾
```

---

## 8. 弱點（這份不能當結論用）

- **n=2 的回歸**（draft 30 nd vs decode 2365 nd）。斜率 50.8 只有兩個點撐著，
  且兩種步的 **op 構成不同**（draft 圖沒有 MoE／attention）⇒ 斜率可能被構成污染。
- **絕對時間不可用**：這一輪 `total` 中位 119.63 ms，而交付形狀 `gpu_union` = 91.79 ms
  （開了 `CGC_GPU_TIMING` + `NODES` + `OPS` 全套儀器，慢 ~30%）⇒ **只能用百分比與比值**。
- **45.6 是眾數不是全部**：`SOFT_MAX` 10.8／`UNARY` 34.7／`GATED_DELTA_NET` 160／
  `FLASH_ATTN_EXT` 266 不在線上 ⇒ 常數說只對一部分 op 成立。
- **grid 公式只讀了 `ADD`**（ops.cpp:915-921），其它 elementwise 假設同構，**未逐個核對**。
- M4 駐留量 1024 是「8 core × 128」的**推算**，不是實讀。
