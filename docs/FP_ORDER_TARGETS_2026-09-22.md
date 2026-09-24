# 「拆分累加 + 固定顺序合并」方法论：在本项目 decode 上还有没有目标？

日期：2026-09-22　線：線A (ace)　HEAD：`demo/sweet-spot-windows-fix`
結論：**方法論成立且本線 09-07 就在用；但在 decode 的三個候選目標上逐一落空 ⇒
decode 端淨空間 = 0。它是「通行證」（解鎖 bit-exact 前提下的重組自由），不是「加速器」。**

---

## 0. 方法論的精確陳述（只接受這個版本）

```
① 多個獨立累加器（拿 ILP）
② 固定順序合併樹（綁住 FP 順序）
③ 關掉 fast-math（讓 ② 真的約束 compiler）
```

三者**缺一不可**：

- 只做 ① ⇒ 拿到 ILP，但 fast-math 下有權重組 `((p0+p1)+p2)+p3` ⇒ bit-exact 不成立
- 只做 ②③ ⇒ bit-exact，但 ILP 歸零
- ③ 在本線是**一行現成的**：`ggml-metal-device.m:232` 的
  `//[options setFastMathEnabled:false];` 取消註解（`GGML_METAL_EMBED_LIBRARY=ON`
  ⇒ `.metal` 原始碼 runtime 才編譯，`fastMathEnabled` 保持默認 YES）

⚠ 開 ③ 的代價：**結果與現在這顆 build 不再 bit-identical ⇒ M1 一次歸零，必須重新 baseline**。

---

## 1. 這形狀本線 09-07 就在用（不是新東西）

`b8a564d45`（P2-B，09-07，是 12.57 基線 `c4e1c1778` 的祖先）：

```c
float2 sum = {0};                        ← 2 個累加器
#pragma unroll
for (short l = 0; l < 4; ++l) {
    #pragma unroll
    for (short j = 0; j < 4; ++j) {
        sum[0] += ...;  sum[1] += ...;
    }
}
sumf_g[row] += d * (sum[0] + sum[1]);    ← 固定順序合併
```

p2 那 24 行做的是：**2 累加器 → 8 累加器**（`sum_parts[l][0/1]`，l 彼此獨立），
依賴鏈 16 → 4。實測結果（09-22 13:51 gate）：**M1 = 9/9（數值 no-op）**，性能增量落
在 baseline 離散內（12.57±2.26；控制臂中位 12.77 區間 11.83–13.46）。

---

## 2. 三個候選目標逐一排查

| # | 目標 kernel | 現在的歸約形狀 | 方法論適用？ | 佔步時 | 端到端空間 |
|---|---|---|---|---|---|
| 1 | `moe_gemv`（IQ3_XXS GEMV，p2 改的） | `float2 sum` 2 累加器 ＋ `#pragma unroll` | ❌ 已在用；且鏈被訪存掩蓋（§3） | **3.2 / 6.3%** | ~0（實測噪音內） |
| 2 | `kernel_gated_delta_net_impl` | `simd_sum()`（硬體歸約，已是「多 lane ＋ 固定歸約」） | ❌ 已是同形狀；且核心是**遞歸**，拆不了（§4） | 支配區塊 **35–45%** | ❌ 方法論不適用 |
| 3 | `kernel_ssm_conv_f32_f32` | **`float sumf = 0; sumf += ...` 單累加器鏈** | ✅ **完美目標** | 小 op 群（5-node range） | ❌ 被 18 µs dispatch 固定開銷吃掉（§5） |

---

## 3. 為什麼 #1 拿不到：鏈不是關鍵路徑

`kernel_mul_mv_id_glu_iq3_xxs_impl` 內層：

```c
for (short l = 0; l < 4; ++l) {
    const threadgroup uint8_t * grid1 =
        (const threadgroup uint8_t *)(svalues + q3[2*l+0]);   ← 間接查表
    const threadgroup uint8_t * grid2 =
        (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
    const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
    for (short j = 0; j < 4; ++j) {
        sum[0] += yl[8*l+j+0] * grid1[j] * (signs & ... ? -1.f : 1.f);
        sum[1] += yl[8*l+j+4] * grid2[j] * (signs & ... ? -1.f : 1.f);
    }
}
```

- 每次 `sum[0] +=` 之間夾的是 **threadgroup 間接查表**（`svalues + q3[2*l+0]`，
  **地址本身依賴資料 `q3`** ⇒ 無法預取、無法提前發出）
- threadgroup 載入延遲 >> FP 加法延遲 ⇒ **串列鏈早被訪存延遲掩蓋**
- ⇒ 拆成 8 個累加器只是把一條「不是關鍵路徑」的鏈縮短

**這是 p2 那 24 行既 M1=9/9（數值 no-op）又在噪音內的機制解釋**——
不是「FP 重組無效」，是「這裡的 FP 鏈不在關鍵路徑上」。

⚠ **這是從程式碼結構推出的推論，不是實測。** 可證偽的檢驗：microbench（§6）上量
`moe_gemv` 拆 8 累加器前後的 µs/dispatch；若有顯著變化 ⇒ 本節錯。

---

## 4. 為什麼 #2 不適用：核心是遞歸，不是歸約

`kernel_gated_delta_net_impl`（`ggml-metal.metal:2649`）：

```c
for (short t = 0; t < args.ne22; t++) {
    ...
    s_k += ls[j]*k_ptr[is];
    s_k = simd_sum(s_k);              ← 已經是「多 lane 並行 ＋ 固定歸約」
    ...
    ls[j] += k_ptr[is]*d;             ← 狀態更新
    y += ls[j]*q_ptr[is];             ← 依賴上一行的 ls[j]
    y = simd_sum(y);                   ← 同樣已並行歸約
}
```

兩件事：

1. **它已經在用這個形狀**（`simd_sum` 就是硬體的固定順序歸約）
2. **跨 `t` 是遞歸**：`ls[j]` 在迭代間傳遞，`y` 依賴它 ⇒ **演算法本質串列**，
   拆累加器無從下手。要改只能做 **chunked / parallel scan（演算法級）**，
   那會改變數值 ⇒ M1 歸零，且與 FP 歸約重組是兩回事

⇒ 支配區塊（35–45%）**不在方法論射程內**。

---

## 5. 為什麼 #3 是「完美目標但沒用」

```c
kernel void kernel_ssm_conv_f32_f32(...) {
    float sumf = 0.0f;                        ← 單累加器
    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];                ← 純鏈，長度 nc
    }
    x[0] = sumf;
}
```

這是方法論教科書級的目標：拆 4/8 個 `sum_parts`、最後固定順序合併 ⇒ 依賴鏈 nc → nc/8。
（旁證：`_4` 變體 `:2241` 已用 `dot(s[i0], c[i0])` 做 SIMD 化，但累加器仍是 1 個。）

**但它落在 5-node range 的小 op 群**（q_conv / k_conv / gate 各 6.0%，合計 36.5%），
而 **K5 已判：一個 dispatch ≈ 18–19.5 µs 且幾乎全是固定開銷**，撐 grid／撐 ILP 都動不了它
（`CGC_GPU_NODES_MATRIX` 實測 µs/node 6× 跨度、5-node 中位 58.2 vs 均值 165.1）。

⇒ **kernel 內省下的時間 < dispatch 固定開銷** ⇒ 端到端拿不到。

---

## 6. 若仍要判定：正確儀器不是 t/s

- **t/s 上測 3% 效應需 ~50 臂/側**（由「1.7% 需 165 臂」換算），單臂噪音底 ±27% ⇒ 不現實
- `gpu_union` 只在 **prefill 配對**成立，decode 逐步臂內 p90/p10 = 2.27× ⇒ 不可用
- **正解**：`m1_harness.py`（`autotuner/m1-shape-inventory` 分支，跑真 server、真 kernel、
  量 **µs/dispatch**、**自帶 M1 bit-exact gate**）⇒ 先量 kernel 本身提速 Y%，
  再量它佔步時 X% ⇒ 端到端 = X × Y

⚠ 該 harness 目前**不在** `demo/sweet-spot-windows-fix` 上（在 `autotuner/m1-shape-inventory`
與 p2 的 `505b3a851`），要用得先搬過來。

---

## 7. 結論

**有了這個方法論，decode 還是提升不了。** 三個理由彼此獨立：

1. 支配區塊（GatedDeltaNet 35–45%）是**遞歸串列**，歸約重組幫不上，要動得走演算法級
2. 方法論的完美目標（ssm_conv 單累加器鏈）**被 18 µs dispatch 固定開銷吃掉**
3. p2 改的 MoE GEMV 只佔 **3.2–6.3%**，且它的鏈被 threadgroup 間接查表掩蓋，
   何況 2 累加器形狀 09-07 就在用

⇒ **方法論的價值是「通行證」不是「加速器」**：它讓未來 autotuner／自動 kernel 生成
（跨硬體 AI compiler 那條路）在 bit-exact 硬約束下**有權重組 FP**。那條路的 upsides
記憶已估為 **kernel 效率層 10–20%**，且不與本節矛盾——因為它射的是「自動搜尋大量 kernel」，
不是「手工拆這一個」。

## 8. 方法論射程外、但更大的魚（供對照）

| 標的 | 佔比 | 性質 |
|---|---|---|
| `node`（無名真運算） | **24.1 / 20.3%** | 支配區塊第一名，**身分未查明** ⇒ 最該先問「這是什麼」 |
| `cache`（遞歸／KV 狀態管線） | 10.6% | 真工作 290 裡 **CPY 210（72%）** ⇒ 純資料搬移，可能是最大單點浪費 |

⇒ 若問「decode 還能從哪裡要時間」，這兩條比 FP 歸約重組更值得先查。
