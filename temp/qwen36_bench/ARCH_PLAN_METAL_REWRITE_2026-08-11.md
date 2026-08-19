# Qwen3.6-r3 Metal Kernel 深度重寫 —— 可實操方案（2026-08-11）

**目標**：GPU 176ms/step → 95-115ms（wall 280 → 160-190ms，即 3.7-4.1 → **5.2-6.2 tok/s**）
**邊界**：不改任何訓練好的權重；與 expert-streaming 調度完全解耦（streaming 繼續管記憶體/fill/hit）。
**誠實上限**：即使全部做完也到不了 22 tok/s——那是 Gemma4 的架構紅利。22 屬於「模型架構改動」戰線（減層/輕量 DeltaNet/蒸餾），本方案不做。

---

## 0. 前置：一次性準確測量（1-2 天，先做，避免憑空優化）

目前只有每 step 聚合數字（wall 280 = gpu 176 + fill 46 + idle 58）。重寫前必須有 **per-layer、per-kernel 的分解**，否則不知道 fusion 該打哪。

**要做**：
1. 給 `Qwen36ForwardRunner.produce()` 加 **per-kernel GPU timing**（`MTLCommandBuffer.addCompletedHandler` + kernel label，或 `MTLCaptureManager` 一次 GPU trace 分析）
2. 量出：40 層中哪幾層最貴（DeltaNet 30 層 vs GatedAttn 10 層）、每層內 router/project/conv/prepare/delta_rule/output/experts/residual 各佔多少
3. 量出：idle 58ms 裡有多少是真正 CPU 空等（可以收），有多少是 GPU 執行本身（收不了）

**gate**：產出一張「每層每 kernel ms」表。如果 DeltaNet 30 層只佔 GPU 的 <20%，fusion 收益上限直接下調，改打別處（見 §4 決策樹）。

---

## 1. Phase-1：DeltaNet 5-kernel 鏈內部融合（最高槓桿，2-3 週）

**現況**：`deltanet_project`(4 次 dispatch: qkv/z/b/a) → `conv1d` → `prepare` → `delta_rule` → `output`，5 個 kernel 各自寫 UMA、下一個再讀回。

**做法（逐層融合，每步都 byte-identity 驗證）**：
1. **Step 1a：project+conv1d 融合**——qkv 投影的 4 次 dispatch（8192/4096/256/256 threads）合併成 1 個 kernel，用 threadgroup 內共享 qkv 的 conv1d 滑窗；中間 qkvBuf 從 UMA 改 threadgroup memory。**預期省 3 次 kernel launch + 8192×128×2B×2 次 UMA 往返**。
2. **Step 1b：prepare+delta_rule 融合**——prepare（32 threads，極小）併入 delta_rule 的 prologue，共享 beta/decay 的算術。
3. **Step 1c：delta_rule+output 融合**——output 的 256 dim 投影併入 delta_rule 尾部，hState 更新與 y 投影在一個 kernel 內完成。

**驗證**：每步後跑 `Qwen36MetalTests` + 對 gemma4 的 golden（現有 `moeBatchAmortizedMatchesGolden` 同款模式），必須 byte-identity。GPU_TIMING 對比融合前後每層 deltanet 時間。

**注意（M4 約束）**：
- threadgroup memory 有限（~32KB），conv 滑窗 + hState 快取不能全塞，需掃 threadgroup size（64/128/256）
- 不能照搬 Gemma4 的 fusion 參數——Qwen3.6 每層 256 專家、DeltaNet 通道 8192，workload 完全不同

**預期**：DeltaNet 相關 GPU 時間 −35-50%。若 DeltaNet 佔 GPU 60%（待 §0 確認），整體 GPU −21-30%。

---

## 2. Phase-2：3-bit dequant 融合進 expert GEMM（2 週）

**現況**：expert 權重 3-bit 打包（每 expert 1.27MB），`encodeExpertSlot8` 內 dequant 與 GEMM 的融合程度待確認（若已部分融合則跳過此項）。

**做法**：
1. 確認 `moe_qwen36.metal` 的 slot8 kernel 是否在讀 packed bytes 時即時 dequant（`q*scale+bias`），還是先解到完整 fp16 buffer
2. 若是後者：把 dequant 併入 GEMM 的 load 階段（threadgroup 載入 packed 後就地展開到 smem，不做整層解壓副本）
3. 目標：消除「解壓權重副本」的 UMA 寫入（40 層 × 256 expert 的攤薄）

**驗證**：r3 vs 現行 kernel 的 byte-identity（3-bit dequant 數學不能變）+ GPU 時間對比。

---

## 2.5 Phase-2.5：Shared Expert 網路融合（dense 固定成本，1 週）

**現況**：每層的 shared expert（40 層全跑、與 routing 無關的固定 dense 成本）：
- `q36_moe_shared` kernel：gate/up 512×2048 GEMM + silu + down 2048×512 + scalar sigmoid（`shG`）→ 中間 `sharedOut` 寫 UMA
- `q36_moe_merge` kernel：`acc + shg × sharedOut`（讀回 sharedOut）

**做法**：
1. **shared + merge 融合**：`sharedOut`/`shgOut` 中間 buffer 消除——merge 直接吃 shared kernel 的 threadgroup 內結果；scalar sigmoid 併入 merge 的 prologue
2. **down-proj 的 act smem 化**：`actTg` 已在 threadgroup，down 迴圈直接讀 smem（現況已是如此）——主要收益是消掉 sharedOut 的 UMA 往返（2048×4B×2）
3. **router 與 shared 並行**：router（選 8 個 id）與 shared（dense 計算）無資料依賴——同一個 CB 內先 dispatch shared 再 router，或共用同一個 encoder 減少 launch 次數

**驗證**：byte-identity（merge 數學 `acc + shg*shared` 不能變）+ GPU_TIMING。

**預期**：每層省 1 次 kernel launch + 2048×4B×2 UMA 往返；40 層合計可觀（shared 是每層固定成本，不受 hit 率影響）。

---

## 3. Phase-3：GatedAttn + Router + 尾處理 fusion（2 週）

**做法**：
1. **GatedAttn 鏈**（10 層）：project → attn → o_proj 的 yOut blit（現況有一筆 `blit copy` 從 private yOut 到 attnOut）——查能否直接改 kernel 輸出目標消除 blit
2. **Router top-k 下沉 GPU**：router 目前 GPU 算 logits → CPU readback → CPU 做 top-8 選擇。把 top-8 argmax 下沉進 kernel（GPU 直接輸出 expert ids），CPU 只收 8 個 id 而非 256 個 logits——**減少 GPU→CPU readback 資料量，且省 CPU 側 top-k 排序**。完整的 slot 分配仍留 CPU。
3. **expert GEMM 尾處理融合**：gate/silu → residual add → norm 併入 GEMM 尾（或至少 residual+norm），減少中間 buffer 往返

**驗證**：同 Phase-1 模式，逐項 byte-identity + GPU_TIMING。

---

## 4. 決策樹（每 phase 結束後跑）

| 實測（§0） | 結論 |
|---|---|
| DeltaNet 佔 GPU <20% | fusion 上限低，跳 Phase-1，先做 Phase-3（router readback + blit 消除） |
| idle 58ms 多為 CPU 空等 | 先做「層內雙緩衝 encode」——encode 下一層 kernel 與當前層 GPU 執行重疊（不跨層、不違反 §13.64） |
| dequant 已融合 | 跳 Phase-2，時間投回 Phase-1 |

**每 phase 的 A/B gate**：`GPU_TIMING + Q36STEP` 對比，tok/s 提升 <3% 則停（時間投別處）。

---

## 5. 不做的事（明確排除）

- ❌ 跨層 CB 合併 42→6（§13.64 已證偽：層鏈強依賴）
- ❌ top-k 8→4、256 expert→128（品質退化，跑分向，非主線）
- ❌ 任何改權重的動作（本方案戰線內不碰）
- ❌ 追求 22 tok/s（需要減層/蒸餾，見下方獨立戰線）

---

## 6. 時間線與里程碑

| 里程碑 | 內容 | 預估 | 預期 tok/s |
|---|---|---|---|
| M0 | per-kernel 分解測量 + 決策樹 | 1-2 天 | —（定方向） |
| M1 | DeltaNet 融合（1a/1b/1c） | 2-3 週 | 4.3-4.8 |
| M2 | dequant 融合（若需要） | 2 週 | 4.5-5.0 |
| M3 | Shared Expert 融合（Phase-2.5）| 1 週 | 4.8-5.3 |
| M4 | GatedAttn blit + router 下沉 + 尾融合 | 2 週 | 5.4-5.9 |

**總投入**：~2 個月（一人）。**上限 6.2 tok/s，現實落點 5-5.5**。

---

## 7. 若要 22 tok/s（獨立戰線，另立項目）

唯一路徑 = **改模型本身**（需訓練算力，產出衍生模型）：
1. 40→30 層蒸餾（對齊 Gemma4 深度）
2. DeltaNet 輕量化/移除（蒸餾進 MoE）
3. 專家蒸餾 + QAT（感知量化訓練）

做完才有理論機會摸 14-18 tok/s；35B 總參仍大於 Gemma4 26B，到不了 22。**這不是 runtime 工程，是訓練項目，本方案不包含。**

---

## 8. 立即動作（review 後第一天）

1. 讀 `deltanet.metal` 全文（確認 5 kernel 的 buffer 依賴與 threadgroup 使用）→ 決定 1a 融合的 tiling
2. 確認 `moe_qwen36.metal` slot8 的 dequant 位置（§2 現況）→ 決定 Phase-2 是否成立
3. 加 per-kernel GPU 測量（§0）→ 跑一次 128 tok 拿分解表
4. 產出分解表後按 §4 決策樹定 Phase 順序
