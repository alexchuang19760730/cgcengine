# G4：dispatch 級普查 —— 一步到底發了幾個 kernel、是哪幾個 op

2026-09-20 16:4x–17:0x。新儀器 `CGC_DISPATCH_CENSUS=1`（`ggml-metal-ops.cpp`，add-only）。

## 0. 為什麼要有這一份

`§EN-337` 證明了一件事：Metal 融合把「圖上的節點」和「真正發出去的 dispatch」分開了
（那條 9 連 ADD 是 9 個節點、2 個 dispatch）。而樹上所有既有的 op 表
（`CGC-GPUOPS`／`CGC-GPULAYK`／`CGC-GPUNODE`）剖分的都是 **buffer 時長 × 節點佔比** ⇒
**它們全部答不出「一步發幾個 kernel」**。這份普查就是為了補上那個量。

## 1. 儀器

在 `ggml_metal_op_encode_impl()` 的 `return n_fuse` 之前加計數：

- **dispatch**：一次 `encode_node` 呼叫 ＝ 一次 dispatch；
- **nodes**：同一次呼叫吃掉的圖節點數（`n_fuse`）；
- 兩欄的差 ＝ **融合當場省下來的 dispatch 數**；
- `idx == 0`（新的一張圖開始）時 flush 前一張，上限 48 張。

**ADD-ONLY**：讀 `node->op`／`n_fuse`、印表，不改編碼、不改任何值；未設 env 時完全靜默。

## 2. 讀數（一次解碼，前 48 張圖）

```
CGC-DISPATCH: graph=1  dispatches=29  nodes=32  (fusion saved 3)
CGC-DISPATCH: graph=2  dispatches=1   nodes=1   (fusion saved 0)     ← ARGSORT（top-k 段）
CGC-DISPATCH: graph=3  dispatches=31  nodes=39  (fusion saved 8)
...
CGC-DISPATCH: graph=48 dispatches=12  nodes=13  (fusion saved 1)
```

**合計：dispatches 1098 ／ nodes 1337 ／ 融合省下 239。**

| op | dispatches | nodes | nodes/dispatch | 佔全部 dispatch |
|---|---:|---:|---:|---:|
| **MUL_MAT** | **254** | 254 | 1.00 | **23.1%** |
| **UNARY** | **101** | 101 | 1.00 | **9.2%** |
| ADD | 88 | **249** | **2.83** | 8.0% |
| **MUL** | **88** | 88 | 1.00 | **8.0%** |
| RMS_NORM | 78 | **156** | **2.00** | 7.1% |
| MUL_MAT_ID | 69 | 69 | 1.00 | 6.3% |
| GET_ROWS | 59 | 59 | 1.00 | 5.4% |
| GLU | 46 | 46 | 1.00 | 4.2% |
| L2_NORM | 36 | 36 | 1.00 | 3.3% |
| SCALE | 36 | 36 | 1.00 | 3.3% |
| CPY | 36 | 36 | 1.00 | 3.3% |
| SOFT_MAX | 24 | 24 | 1.00 | 2.2% |
| ARGSORT | 24 | 24 | 1.00 | 2.2% |
| DIV | 23 | 23 | 1.00 | 2.1% |
| SUM_ROWS | 23 | 23 | 1.00 | 2.1% |
| CLAMP | 23 | 23 | 1.00 | 2.1% |
| CONCAT | 18 | 18 | 1.00 | 1.6% |
| GATED_DELTA_NET | 18 | 18 | 1.00 | 1.6% |

## 3. 從這張表讀出來的四件事

1. **最大的單一項是 `MUL_MAT`：254 次 dispatch（23.1%），而且比率 1.00。** 它是真實的矩陣乘
   （QKV／gate-up／shared expert／MTP head），**不是發派開銷** ⇒ 它不該被當成「融合的目標」。
2. **融合已經在做事，但只做了一半**：`ADD` 249 → 88（省 161）、`RMS_NORM` 156 → 78（省 78）。
   兩者合計 239，就是全部省下來的 dispatch。
3. **剩下的發派裡，elementwise 佔了一大塊**：
   `UNARY 101 ＋ MUL 88 ＋ GLU 46 ＋ SCALE 36 ＋ L2_NORM 36 ＋ DIV 23 ＋ CLAMP 23 ＋ SUM_ROWS 23`
   ＝ **376 次 dispatch（34.2%）**，比率全是 1.00。ggml-metal 的融合目前只處理
   `ADD/SUB/MUL/DIV` 的**鏈**（且要求 `f0 == f1->src[0]`），這些 op 大多不構成那種鏈 ⇒ 沒被併到。
   ⇒ **dispatch 級的剩餘空間主要就在這裡**，而不是在 ADD。
4. **`CPY` 36 次（3.3%）** 是純搬運；`GET_ROWS` 59 次是查表。兩者合計 8.7%。

## 4. 這份普查推翻了什麼

- `§EN-334` 的「移除 328 個 dispatch/步」：那 328 個**大部分不存在**（ADD 的 249 個節點現在只剩
  88 個 dispatch）。
- 「G4 的槓桿是 ADD 融合」：**不成立**，它已經被做完（§EN-337），而且做完之後 ADD 只剩 8.0%。
- ⇒ **找槓桿必須用 dispatch 數，不能用節點數。這份表是那個尺。**

## 5. 誠實邊界

- **這不是一次量測，是一次編碼期普查**：該輪 `llama-bench` 在 **command buffer 1 就
  `Insufficient Memory` 後 abort**（`CGC-METAL-FAIL`）。dispatch 數在編碼期就定了，所以普查有效，
  但**這一輪沒有產生任何 t/s**，而且前 48 張圖**含 warmup 圖**，不等於一個穩定步。
- 探針上限 48 張圖 ⇒ **第 48 張之後的圖沒有被計入**。
- 這輪是**直接跑 `llama-bench`**（把 `run_server.sh` 的 resolved env 手動帶上）。走
  `llama_bench_matrix.py` 時 arm-spec 的 env **不會被轉發到 llama-bench 行程**
  （實測：`prefill250:CGC_DISPATCH_CENSUS=1` 印 0 行，直接跑印 608 行）⇒
  **要讓這個普查進正式量測，得先讓 matrix 轉發 arm env。**
