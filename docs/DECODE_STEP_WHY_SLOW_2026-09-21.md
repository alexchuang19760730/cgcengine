# decode 步為什麼慢 ——「不是算力也不是頻寬」的第三個答案（2026-09-21 14:0x）

> 起因：使用者問「到底 verify 慢的原因是啥？又不是算力又不是頻寬，那 kernel 可以優化的話不就都解決了？」
> 本篇只做一件事：**把「第三種 bound（latency / occupancy）」講成可引用的數字。** 沒有跑新量測，沒有重建 binary。

---

## 0. 先拆掉一個不存在的問題：「verify 慢」

「MTP verify 的路徑是原本的好幾倍」是**分母錯置**（§EN-369/370）：

- 「42 ms/token」的墊底是**位置數 T=4**，「99.3 ms/token」的墊底是**實吐 token（3.197）**。
- 對齊分母後：同一鐵 M=4 的 plain 路徑 **457 ms**，而 verify 那 4 個位置只要 **317.4 ms**
  ⇒ **verify 反而便宜 1.44×**，它沒有缺陷。

而且從原理上也應該如此：verify 把 matmul 從 GEMV（M=1）變成 GEMM（M≈4），
**同樣的 bytes、多 4 倍 FLOPs ⇒ 算術強度改善 4×**，理論上 verify 比逐步 decode 更接近 roofline，不會更差。

⇒ **沒有「verify 特別慢」這個現象需要解釋。** 真正的問題是下面這個：「**一個 decode 步本身為什麼要 ~98 ms**」。

---

## 1. 前提校正：「又不是頻寬」這句話只有一半對

joint capture（`Backup/phase_decomp/joint_capture_20260920_2144.json`，n=65 穩態）：

```
每步 RAM 流量 1.34 GB（dense 827 MB ＋ 專家 ~509 MB）
GPU busy = 97.90 ms  ⇒  有效 13.6 GB/s  = 峰值(~120–124) 的 ~11%
```

所以它不是「頻寬已經用滿、無解」，也不是「FLOPs 打滿」——
**它是第三種：GPU 90% 的時間 busy，但那一 Pearson吞吐量只拿到峰值的約一成。**
⇒ 這正是「表面不像任何一種 bound」的原因：**你以為你在量близ頻寬，其實你在量 latency / occupancy。**

---

## 2. op 級歸因（既有 log，本次用工具重讀，不是新量測）

工具：`scripts/check/attn_moe_split.py ops Backup/cgc_logs/llama_server_20260919_191400.log`
（源係 `CGC-GPUOPS`；trunk 形狀 `nodes_all=4076`、`nodes_work=2455`、**40% 的節點不產生任何工作**）。

| OP | 佔 busy% | 節點數 | **µs/節點** @total 156.8 ms | @total 97.9 ms |
|---|---:|---:|---:|---:|
| `MUL_MAT`（稠密 matmul） | 15.40 | 426 | 56.7 | 35.4 |
| `ADD` | 14.20 | 421 | 52.9 | 33.0 |
| `MUL` | 11.10 | 278 | 62.6 | 39.1 |
| `RMS_NORM` | 7.60 | 130 | **91.7** | 57.3 |
| `UNARY` | 7.50 | 169 | 69.6 | 43.5 |
| `CPY` | 6.90 | 210 | 51.5 | 32.2 |
| `GET_ROWS` | 5.10 | 159 | 50.3 | 31.4 |
| `MUL_MAT_ID`（專家 gather） | 3.70 | 117 | 49.6 | 31.0 |
| `L2_NORM` | 3.30 | 60 | 86.2 | 53.8 |
| `GATED_DELTA_NET` | 2.80 | 30 | **146.4** | 91.4 |
| `GLU` 2.50 ／ `SCALE` 1.90 ／ `SUM_ROWS`·`CLAMP`·`DIV` 1.2 ／ `ROPE` 0.6 等 | | | | |
| `VIEW` 963 ／ `RESHAPE` 598 ／ `TRANSPOSE` 30 ／ `PERMUTE` 30 | **0.00** | 1621 | — | — |

（absolutems 有跨 run 差異：同一形狀在 09-19 那支 log 的中位 total 是 156.8 ms，
而 09-20 joint capture 的 GPU busy 是 97.9 ms ⇒ **只看百分比最穩，µs/節點按比例縮小 0.62 倍**。
下文推論全部只用兩個絕對值的帶。）

### 把 bytes 放進去 —— 就看出「誰真的差」

幾何（讀 GGUF，免費）：`hidden=2048`、`inter=512`、`experts=256`、`layers=41`，
down 是 **IQ1_S**（~1.56 bit）、gate/up 是 IQ4_NL。

| OP | 每節點 bytes（估） | 推算有效頻寬 | 峰值佔比 |
|---|---|---:|---:|
| `MUL_MAT_ID`（每層一味 kind，union ~26 experts） | ~3.6–5.4 MB | **~60–73 GB/s** | **50–61%** ✅ |
| `MUL_MAT`（827 MB/426 節點 ≈ 1.94 MB/node，含大量小 attention/delta-net 投影） | ~1.94 MB | ~34 GB/s | ~28% |
| `RMS_NORM`（hidden 2048 × T=4，f32：**讀+寫 ~64 KB**） | ~64 KB | **~0.7 GB/s** | **~0.6%** 🔴 |

**這是本篇的核心一句話：**

> **大塊的搬運其實不差：專家 gather 吃到峰值 50–61%。
> 而一個RMS_NORM 節點燒掉 57–92 µs 只搬 64 KB —— 那是峰值的 0.6%，比 matmul 差將近 100 倍。**

（`ADD`/`MUL`/`UNARY`/`CPY`/`GLU` 那 ~42% 也一樣：每一個都在 ~30–44 µs 裡搬幾十 KB。）

⇒ 所以「慢」的成分是：**2455 個工作節點，每一個都付了一次幾十 µs 的固定成本。**
它既不是 FLOPs 牆，也不是 DRAM 牆（DRAM 只用了 11%），**是第三類：latency / occupancy-bound。**

---

## 3. 那「kernel 可以優化」不就都解決了？——三層答案

### (1) 理論上：**是的，而且這是全專案最大嘅一塊**

若 1.34 GB 能用 ~100 GB/s 搬完 ⇒ 步時 ~13 ms ⇒ **~75 t/s**（今天 12.57 的 6×）。
這個不是幻想，它就是上面那張表裡被浪費掉的 89%。

### (2) 但最大的杠杆恰好被 **G2（逐位元相同）** 禁止

要讓 quant GEMV 提高 occupancy，標準手段是：**改 tile 形狀 / split-K / 多路並行 + tree reduce**。
而這三件**全部改變累加順序** ⇒ 改變四捨五入 ⇒ G2 直接紅。
（這不是猜的：本線早已把 G2 寫死過一次 —— 「融合必須串序累加（同原順序）才逐位元相同，
樹狀 reduce 過不了 G2」。把它套到 matmul 上，就是同一句話。）

外加 `ffn_down_exps` 是 **IQ1_S**（~1.56 bit）：shader 裡要查表 dequant，
吃 ALU 又吃暫存器 ⇒ occupancy 上限本來就被壓著，這也是 matmul 只到 28% 的一部分原因。

### (3) 「融合」這帖藥的天花板已經被量過了

- 實測**一個 dispatch 的「上界」= 0.0736%**（配對 ON 10.434 / OFF 9.620、Δ115 dispatch/步）。
- 最有肉的一條：**41 層各一條 9 連 ADD 鏈**，全熔 → 每步可移除 328 kernel，
  **上界 11.5%**，低於 G4 門檻 12.8%。
- ⇒ **「kernel 數太多」不是主詞**（真要只是 kernel 數，2455 個 × 0.0736% 早就超過 100% 了，
  那個數字含各 kernel 自己做的工作）。主詞是**每一個 kernel 自己的 load pattern / occupancy**。
  ⇒ **「把它們黏起來」跟「把它們寫快」是兩件事，只有後者有 ~6×。**

### (4) 工程面（今天真實存在的成本）

- 重建會蓋掉別條線正在 map 的 `.dylib`（tree 上有 872 行未提交引擎改動）。
- 每條 kernel 要過數值閘門 117/117。
- Metal 單佇列：相鄰 kernel 無法互相隱藏延遲，小 kernel 的 ramp-up 是白付的。

---

## 4. 結論一句話

**「又不是算力又不是頻寬」是真的，但它們不是同一件事的兩個說法 ——**
整件事是這樣：**GPU busy 90%，而搬那些位元組的效率只有峰值一成**
（ matmul 已到 28–61%，而_norm/elementwise 那 55%+ 只有 ~0.6%）。
**=> 它是 latency/occupancy-bound，而唯一能碰它的手段（改 tile / split-K / tree reduce）
恰好就是 G2 逐位元相同禁止的那一組。**

⇒ 「kernel 優化能解決嗎」：**能，但它不是"順手改一下"，它是「在維持逐位元相同的前提下，
把 Metal 的 quant GEMV 與整條 elementwise 重寫到接近峰值頻寬」 —— 一個專案級打工。**
今天已量測且不用碰 G2 的手段（融合 ≤11.5%、prefetch 輸在 bytes、IO 形狀 ≤1%、超訂 ~6%）合計只有 ~19%。

---

## 5. 一個可驗的下一步（若要動手，先做這個再寫碼）

上面 op 級 bytes 是**含假設的估計**（激活假設 f32、verify T≈4）。要把它變成不含假設的事實，
只差一台量測：同一 build、np1、交付形狀，開

```sh
CGC_GPU_NODES=1 CGC_GPU_OPS=1 CGC_GPU_NODES_TRACE=1 CGC_GPU_TIMING=1
```

拿 `CGC-GPUOPS`（per-op 時間）＋ `CGC-NSM`（per-command-buffer）＋ output 形狀
⇒ **直接算 per-op 的有效頻寬**，把「0.6%」那個數字做成 undisputed 的。
⚠️ 跑之前要先確認 8080 空、無他線量測（本日 13:35 已因本線起 server 害另一條線補跑被閘門擋下一次）。
