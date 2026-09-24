# K3 實測：一次 dispatch 值多少、cluster-1 值多少

2026-09-21 18:2x–18:4x。0 重建（兩個 census 都在現有 binary 裡）。
回答 `K3_CEILING_REAUDIT_2026-09-21.md` 留下的「上界是 5%～16%，中央 ~10%」這個待測區間。

## 0. 一句話結論

**cluster-1（MoE 權重歸一化 sum_rows→clamp→div 融合）實測 ≈ 2.3%，不是 5.3%，也不是 10%。**
低於 3% 可見門檻、遠低於 12.8% ⇒ **不要寫那個 300 行融合 kernel**。

G4 原本的結論（別寫）**存活**，但它當年的理由是「5.3% 連 12.8% 的一半都不到」——
那個數字是錯的，真正的數字更小，理由也更強。

---

## 1. 修好儀器（三件事）

| 缺陷 | v1 | v2 |
|---|---|---|
| regex 雙算 | `CGC-DISPATCH:\s+(\S+)\s+dispatches=` 同時吃 `graph=1 dispatches=1098` 與 `ADD dispatches=27`，dict 以 group(1) 為 key ⇒ `disp_total ≈ 2×` | 分成 `RE_GRAPH`／`RE_OP`，總量只取 summary 行 |
| raw stderr | 沒留 ⇒ 事後無法複查 | 每臂存 `Backup/phase_decomp/K3/raw/{on,off}.stderr` |
| build 指紋 | 沒記 | 跑前跑後各記一次，**本次未變**（`build_changed: False`） |

另外加開 `CGC_GPU_TIMING=1` —— 這是整份裡最關鍵的決定，見 §2。

## 2. 兩臂讀數

| | ON（融合開） | OFF（`GGML_METAL_FUSION_DISABLE=1`） |
|---|---|---|
| **t/s** | **8.864789** | **7.940683** |
| thermal | NOMINAL → NOMINAL | NOMINAL → NOMINAL |
| sentinel | 286.76（臂前，HEALTHY） | 270.25（臂**後**，HEALTHY） |
| census cmd_bufs | 48（截斷） | 48（截斷） |
| dispatches | 732 | 875 |
| nodes | 911 | 875（**融合省下 0 ✓**） |
| `gpu_union`/步 | 91.79 ms | 102.12 ms |
| `wait`/步 | 101.19 ms | 111.50 ms |

**Δ% 由兩個獨立儀器給出，彼此吻合：**

- t/s：OFF 慢 **+11.64%**
- `gpu_union`：OFF 慢 **+11.25%**

Δ dispatch（同一個 48-cmd_buf 窗口）= 875 − 732 = **143**。

> ⚠️ sentinel 的坑：v2 把 sentinel 放在臂**之前**，結果 `off` 臂被 gate 以 `thermal HEAVY` 拒絕 ——
> sentinel 本身就是一支 `-p 2048` prefill，是**它**把機器推進 HEAVY（讀 HEALTHY 270 → 隨即 gate 讀 HEAVY）。
> 改成臂後再驗（`off_arm_only.py`），視窗就拿到了。**「用掉視窗來認證視窗」是自我打敗的。**

## 3. 分母：每一步到底有幾個 dispatch（這是整份最大的修正）

`CGC-GPUTIME`（`ggml-backend.cpp:3018`，需 `CGC_GPU_TIMING=1`）**按步**印：

```
CGC-GPUTIME: step=4 segs=40 bufs=353 ... wait=101.19 gpu_union=91.79 ...
```

- **`segs=40`**（34 個樣本，**恆定 40**）、**`bufs≈353`**（345–360）
- census 的 `graph=` 單位是 **command buffer**（`idx==0` 在 `ggml-metal-context.m:1476` 的 per-cmd_buf 迴圈裡），
  和 `bufs` 同單位 ⇒ **一步有 ~353 個 command buffer，而 census 上限只有 48**。

⇒ **census 的 48 個 cmd_buf 只覆蓋一步的一部分，不是「48 步」。**
`G4_DISPATCH_CENSUS` 那個「1098 dispatches/step」來自一次 **abort 掉的 run 的前 48 個 cmd_buf**
（該文件 §5 自己承認），**數字作廢**。

真值（用 `CGC-GPUOPS` 的 `nodes_all` 定錨，見 §4）：
**nodes/step = 3804**，窗口內 911 nodes ⇒ 窗口 = **23.9% 的一步** ⇒ `k = 3804/911 = 4.176`
⇒ **dispatches/step ≈ 3056**（ON）／3654（OFF），**Δ/step = 143 × 4.176 = 597**。

> 注意：不要用 `bufs/step ÷ 48 = 7.354` 當外推因子 —— 前 48 個 cmd_buf 比平均大
> （15.25 vs 8.7 dispatch/buffer），那個因子會把 Δ 高估 1.76×。

## 4. cluster-1 的真實數量：**117，不是 72**

`CGC_GPU_NODES=1 + CGC_GPU_OPS=1 + CGC_GPU_TIMING=1`（`ns_total > 0` 是印表條件）按步印 op 表。
37 個 step 裡有**兩種異構步**，混在一起取中位數會全錯：

| 步型 | 個數 | `nodes_all` |
|---|---:|---:|
| MTP draft 圖 | 20 | 45 |
| **真 decode 步** | **17** | **3800–3806** |

只取 17 個真 decode 步（min = max，完全穩定）：

| op | nd/step |
|---|---:|
| SUM_ROWS | **39** |
| CLAMP | **39** |
| DIV | **39** |
| GET_ROWS | 159 |
| CPY | 120 |
| MUL_MAT_ID | 117 |
| GLU | 78 |
| SSM_CONV / GATED_DELTA_NET | 30 / 30 |

⇒ **cluster-1 = 117 nodes/step**（三者 ratio 皆 1.00，無融合 ⇒ nodes = dispatches）。
**「72 = 24 層 × 3」是錯的：這模型是 39 層。**

## 5. 結帳

```
單位價（每 dispatch、每步） = 11.64% / 597  = 0.0195 %
cluster-1 全部移除          = 0.0195% × 117 = 2.28 %      (t/s)
                            = 11.25% × 117/597 = 2.21 %   (gpu_union)
```

**⇒ cluster-1 ≈ 2.3%（兩個儀器：2.21 / 2.28）。**

### v1 為什麼算出 5.3%

v1 是 `0.0736%/dispatch × 72 = 5.3%`。兩個錯，方向相反、沒有完全抵消：

1. **單位錯配（主因，×4.2）**：`0.0736%` 是「每 dispatch、**每窗口**」的價，窗口只有一步的 23.9%
   ⇒ 真正的「每 dispatch、每步」是 `0.0736%/4.176 = 0.0176%`。**拿每窗口的單價去乘每步的數量。**
2. **數量錯（×0.615）**：乘的是 72，真值是 117。

反過來的交叉驗證：用窗口內比值（外推因子 k 自動抵掉）
`cluster-1/Δ = 19/143 = 0.133` ⇒ 1.55%。
但 census 的 per-op 表被 top-14 截斷，**cluster-1 只捕到 28 個中的 19 個**（68%）
⇒ 修正後 `28/143 = 0.196` ⇒ 2.28%，**與 §5 的 2.28% 一致**。✓

## 6. 對階梯的處置

| 項目 | 判定 |
|---|---|
| **K3 cluster-1** | **≈2.3%** ⇒ 低於 3% 可見門檻 ❌ **不寫** |
| K3 整體 | 要摸到 12.8% 需移除 `597 × 12.8/11.64 = 656` dispatch/步（全部 dispatch 的 ~21%），cluster-1 只佔 3.8% ⇒ **K3 單獨不可能達標** |
| 「上界 5%～16%」 | ❌ 作廢，實測 2.3% |
| 「72 個 dispatch/步」 | ❌ 作廢，真值 117 |
| 「1098 dispatch/步」 | ❌ 作廢，真值 ~3056 |

## 7. 誠實邊界

1. **n=1／臂**。t/s 的單臂噪音是 ±27%，但兩個獨立儀器（t/s 與 `gpu_union`）給出 11.64% 與 11.25%，
   且兩臂 thermal NOMINAL→NOMINAL、sentinel 皆 HEALTHY ⇒ 這個 Δ 可信度高於一般 n=1。
   `gpu_union` 的臂內 p90/p10 = 2.27×（50–193 ms），**並不比 t/s 緊** —— §EN-393 說的「0.2% 靈敏度」
   指的是 prefill 配對，不是 decode 逐步。
2. **yardstick 的轉移假設**：OFF 臂省下的不只是 dispatch，還省了融合避免的**中間結果落地**。
   cluster-1 的中間張量很小（sum_rows 輸出是 per-layer 小向量），所以它的融合**多半只有 dispatch 那一半**
   ⇒ **2.3% 是上界，真值可能 <1%。**
3. **census 的 per-op 行會交錯**：多執行緒 encode 同時寫 stderr，檔案裡 `graph=` 編號**不是順序的**
   （實測 1,2,3,5,6,7,4,8…）⇒ **逐圖歸因不可用**，只有全域求和有效。
4. **top-14 截斷**：per-op 表只解釋了 44–47% 的 dispatch，cluster-1 捕獲率 68%。
   凡用 census per-op 數做絕對量的，都必須先做這個修正。
5. 外推因子 `k=4.176` 假設窗口內外的 dispatch/node 比一致（窗口是每步最前面的 24%）。
