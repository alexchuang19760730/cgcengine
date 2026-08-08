# TurboFieldfare routedExpert weightBits=2/3 A/B 實驗報告

**日期**: 2026-08-07
**範圍**: 在 TurboFieldfareRebits 增加 `weightBits=2/3` 的 repack 路徑，對 routed expert 降位寬做 A/B
**對照組**: 4-bit affine (group 64, BF16 scale+bias) 原模型
**硬體**: Apple Silicon (M 系列, macOS) + 統一記憶體

---

## 一、結論摘要

| 指標 | 4-bit (baseline) | 3-bit | 2-bit |
|---|---|---|---|
| routed expert 大小 | 12.90 GB | 10.00 GB (−22.4%) | 7.17 GB (−44.4%) |
| expertStride | 3,358,720 B | 2,605,056 B (−22.4%) | 1,867,776 B (−44.4%) |
| decode tok/s (中位數) | ~11.5 | ~13.6 (**+18%**) | ~20.5 (**+90%**) |
| TTFT (中位數) | ~3.3 s | ~2.9 s (−13%) | ~2.5 s (−24%) |
| GPU busy (中位數) | ~7.3 s | ~6.2 s (−16%) | ~4.2 s (−39%) |
| routedMoE GPU (中位數) | ~2.4 s | ~2.0 s (−16%) | ~1.2 s (−44%) |
| 暴露 IO / decode 比例 | 56.9% | 47.7% | 42.7% |
| 權重空間 RMSE (gate/up/down) | — | 0.0053 / 0.0054 / 0.0081 | 0.0125 / 0.0127 / 0.0190 |
| 固定 prompt 生成品質 | 基準 | 可讀、輕微漂移 | 明顯退化（長文本易崩） |

**一句話結論**: 3-bit 是「接近免費的加速」——權重誤差 ~0.5%，decode 吞吐 +18%，TTFT −13%，固定 prompt 仍能產出結構正確的程式碼。2-bit 把吞吐再翻倍 (+90%) 且 IO 暴露降一半，但權重誤差 ~1.3–1.9% 且生成品質明顯下降，**不建議作為預設**，只適合 IO-bound 場景的「效能優先」模式。

---

## 二、實作內容（weightBits=2/3 repack 路徑）

### 2.1 新工具：`TurboFieldfareRebits`
- 讀取既有 4-bit `.gturbo`，把 routed expert（gate/up/down）dequant 後以 `bits=2|3` 重新量化
- 維持相同 affine 語義：group 64、BF16 scale+bias、低 index = LSB 的 bit-packing
- 重新寫出 `packed_experts/layer_*.bin` + `layout.json` + `manifest.json` + `verified-install.json`（含新 sha256），attention/embedding/router/shared experts 原樣保留

### 2.2 Runtime / Metal 支援
- `Quantization.swift`: `quantizeAffine` / `packCode` / `unpackCode` / `packedRowBytes` 支援 2/3-bit
  - 2-bit: 4 codes/byte（low index = LSB）
  - 3-bit: 8 codes little-endian 跨 3 bytes（24 bits）
- `moe.metal`: decode 路新增 `moe_phase1_gate_up_act_b2/b3`、`moe_phase1_gate_up_act_subset_b2/b3`、`moe_phase2_down_reduce_b2/b3`（SIMD 32-lane 快速路 + tail 路）
- `prefill.metal`: prefill 路新增 `prefill_grouped_routed_moe_batched_phase1_b2/b3`、`_down_b2/b3`
- `MoE.swift` / `PrefillGroupedRoutedMoE.swift`: 依 `weightBits` 選擇對應 PSO；`RealForwardRunner` 全 dispatch site 穿透 `model.routedExpertWeightBits`
- manifest validator 放寬 routed expert 接受 `weightBits ∈ {2,3,4}`

### 2.3 A/B 啟用過程中修掉的 bug（這些 bug 曾讓 2/3-bit 輸出全錯）
1. **Repack 工具把 blob-relative tensor offset 當成 file-absolute offset** → 8 個 expert 全部寫成 expert 0 的內容（模型等於退化到單一 expert）。修法: 讀取時加上 expert blob 的絕對 file offset。
2. **工具寫了新 bits 的 code 卻複製原 4-bit scale/bias** → runtime 用 3-bit code 配 4-bit scale 反量化 → garbage。修法: 重新計算並寫入新 scale/bias。
3. **moe.metal 3-bit tail decode 雙重加 group offset**（`base = g*24 + b0i`，而 `b0i` 已含 `g*24`）→ down (N=704, 3 tail groups) 全部讀錯位置。修法: `base = b0i`。
4. **工具 padding 到 4096，runtime 要求 `getpagesize()` (16384)** → 2-bit stride 非 16KB 倍數時 streamer 讀越界。修法: 工具改用 `getpagesize()`。
5. **bundle 中 dequant_int4.metal 殘留引用未定義的 `kAttnMaxHeadDim`** → 版本已修正為自包含 `kMaxGemvN`（重建即可）。

> 驗證方法: Python 以「與 Metal kernel 完全相同的位元運算」解碼 live row，與 4-bit 對照 cos≈0.98；byte-level 比對 8 個 expert 前綴確保已不再相同。

---

## 三、大小 / 佈局

| variant | stride (B) | 頁對齊 (16KB) | packed_experts | 目錄大小 |
|---|---|---|---|---|
| 4-bit | 3,358,720 | ✓ (205×16384) | 12,897,484,800 B | 27 GB |
| 3-bit | 2,605,056 | ✓ (159×16384) | 10,003,415,040 B | 11 GB |
| 2-bit | 1,867,776 | ✓ (114×16384) | 7,172,259,840 B | 8.0 GB |

---

## 四、權重空間品質（工具內建對照: 4-bit dequant vs N-bit dequant）

| variant | gate RMSE / maxAbs | up RMSE / maxAbs | down RMSE / maxAbs | n |
|---|---|---|---|---|
| 3-bit | 0.0053 / 0.0889 | 0.0054 / 0.0444 | 0.0081 / 0.1504 | 7.6e9 |
| 2-bit | 0.0125 / 0.1724 | 0.0127 / 0.1042 | 0.0190 / 0.3027 | 7.6e9 |

- 3-bit 誤差 ~0.5–0.8%（down 略高，因 N=704 組數少、量化粒度粗）
- 2-bit 誤差 ~1.3–1.9%，maxAbs 達 0.30（down）

---

## 五、效能 A/B（交錯執行，每組 5 輪取中位數）

**設定**: B-tree prompt、max-new 128、64 slots/layer + expert prefetch + lookahead、`--trust-receipt`（與生產 CLI 一致）

### 5.1 4-bit vs 3-bit

| round | 4-bit tok/s | 3-bit tok/s | 4-bit TTFT | 3-bit TTFT | 4-bit busy | 3-bit busy | 4-bit routedMoE | 3-bit routedMoE | 4-bit IO GiB | 3-bit IO GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.51 | 12.02 | 3.35 | 2.74 | 7.34 | 6.92 | 2.45 | 2.35 | 8.47 | 8.60 |
| 2 | 9.31 | 13.64 | 3.23 | 2.92 | 8.68 | 6.15 | 2.71 | 2.02 | 15.25 | 8.38 |
| 3 | 10.56 | 15.48 | 3.30 | 3.00 | 7.57 | 5.35 | 2.44 | 1.73 | 11.14 | 8.39 |
| 4 | 15.29 | 13.81 | 3.38 | 2.86 | 5.35 | 6.12 | 1.71 | 2.05 | 8.48 | 8.31 |
| 5 | 13.99 | 12.56 | 3.29 | 2.87 | 5.86 | 6.65 | 1.87 | 2.18 | 8.19 | 10.23 |
| **中位數** | **11.51** | **13.64** | **3.30** | **2.87** | **7.34** | **6.15** | **2.44** | **2.05** | **8.48** | **8.39** |

→ decode **+18.5%**、TTFT **−13%**、GPU busy **−16%**、routedMoE **−16%**

### 5.2 4-bit vs 2-bit

| round | 4-bit tok/s | 2-bit tok/s | 4-bit TTFT | 2-bit TTFT | 4-bit busy | 2-bit busy | 4-bit routedMoE | 2-bit routedMoE | 4-bit IO GiB | 2-bit IO GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10.13 | 20.50 | 3.34 | 2.51 | 7.58 | 4.23 | 2.34 | 1.18 | 11.22 | 8.38 |
| 2 | 16.57 | 21.93 | 3.28 | 2.44 | 4.75 | 3.97 | 1.46 | 1.16 | 8.20 | 5.67 |
| 3 | 16.08 | 21.75 | 4.05 | 2.28 | 4.68 | 4.00 | 1.44 | 1.14 | 8.48 | 5.94 |
| 4 | 9.24 | 14.83 | 3.32 | 3.17 | 8.34 | 5.59 | 2.58 | 1.65 | 14.57 | 7.44 |
| 5 | 10.81 | 17.96 | 3.24 | 2.60 | 6.97 | 4.96 | 2.11 | 1.53 | 12.21 | 5.54 |
| **中位數** | **10.81** | **20.50** | **3.32** | **2.51** | **6.97** | **4.23** | **2.11** | **1.18** | **11.22** | **5.94** |

→ decode **+90%**、TTFT **−24%**、GPU busy **−39%**、routedMoE **−44%**、暴露 IO **−47%**

### 5.3 暴露 IO（expert cache 統計，單跑對照）

| variant | req / hit / miss | hit rate | read bytes | readWall | IO 佔 decode 比例 |
|---|---|---|---|---|---|
| 4-bit | 61,464 / 58,843 / 2,621 | 95.7% | 8.20 GiB | 4.81 s | **56.9%** |
| 3-bit | 61,466 / 57,214 / 4,252 | 93.1% | 10.32 GiB | 5.59 s | **47.7%** |
| 2-bit | 61,458 / 57,188 / 4,270 | 93.1% | 7.43 GiB | 4.49 s | **42.7%** |

> 注意: 2/3-bit 產生的 token stream 與 4-bit 不同 → routing 更分散 → miss 次數較多（4252/4270 vs 2621），但每次 miss 的 stride 小 22–44%，因此「IO 佔 decode 比例」仍單調下降。公平的每-fetch 成本 = stride: 3.36 → 2.61 → 1.87 MB (−22% / −44%)。

### 5.4 單跑 sanity（fibonacci prompt、無 bench env）

| variant | TTFT | decode | tok/s |
|---|---|---|---|
| 4-bit | 4.02 s | 9.20 s | 6.52 |
| 3-bit | 3.02 s | 6.39 s | 9.40 |
| 2-bit | 2.81 s | 3.07 s | 13.04 |

---

## 六、輸出品質（固定 prompt、greedy temp=0、固定 seed）

### 6.1 `def fibonacci(n):`（baseline 在此 prompt 乾淨）

- **4-bit**: `return fibonacci(n - 1) + fibonacci(n - 2)` → 接著流暢解釋 O(2^n) 複雜度 ✓
- **3-bit**: `return fibonacci(n - 1) + fibonacci(n - 2)` → 接著 main block 測試（輕微重複，可讀）✓
- **2-bit**: 退化成 `(1/n)*(1-2*(n/10))*(1-(1/n)*...` 無限遞迴算式 ✗（語意崩潰）

### 6.2 `def fizzbuzz(n):`（baseline 略漂移）

- **4-bit**: `out.append(i * 3)`（結構正確、邏輯誤植，baseline 本身有此漂移）
- **3-bit**: `out.append(str(i))` + 縮排小混亂（結構可讀、輕微漂移）
- **2-bit**: `if i % 1:` + 重複註解（明顯退化）

### 6.3 散文 prompt（B-tree 說明）

- 4-bit baseline 本身在此 prompt（temp 0 / 預設 sampling 皆）即陷入重複 loop（先前 MTP 報告已記錄 4-bit 的多數 prompt 退化）→ 不適合作為 discriminator。

**品質判讀**: 3-bit 維持結構完整、僅輕微語意漂移；2-bit 在需精確邏輯的任務上明顯崩潰。perplexity 目前 CLI 未支援（已確認無此 flag），故以「權重空間 RMSE + 固定 prompt 對照」作為品質門檻。

---

## 七、建議

1. **3-bit 適合預設下放**: 誤差 <1%，decode +18%、TTFT −13%，生成品質可接受 → 建議做為端側 IO-bound 情境的預設 routed expert 位寬。
2. **2-bit 作為「效能模式」**: 吞吐翻倍、IO 半減，但品質退化明確 → 不設為預設，可在低記憶體/高吞吐場景以 manifest 切換。
3. **重跑正式驗收**: 3-bit 若要走正式 gate（如 UPKG 體系），需補 perplexity/benchmark 對照與固定模型 snapshot 的 formal rerun artifact。
4. **後續**: 可再試 group 128 / 對 down 保持 3-bit 而 gate/up 降到 2-bit 的混合位寬，平衡品質與 IO。

---

## 八、附錄: 本次修改檔案

- `Sources/TurboFieldfareRebits/main.swift` — 新工具（weightBits 2/3 repack、expert offset 修正、scale/bias 重寫、頁對齊）
- `Sources/TurboFieldfare/Infrastructure/ModelIO/Quantization.swift` — 2/3-bit affine quantize/dequant/pack
- `Sources/TurboFieldfare/Metal/MoE/moe.metal` — b2/b3 decode kernels（含 tail 修正）
- `Sources/TurboFieldfare/Metal/Prefill/prefill.metal` — b2/b3 prefill kernels
- `Sources/TurboFieldfare/Kernels/MoE/MoE.swift` — bits→PSO 選擇 + dispatch
- `Sources/TurboFieldfare/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift` — prefill bits PSO
- `Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift` — bits 穿透
- manifest validator — 接受 2/3-bit routed experts

**產物**: `models/gemma4-r3.gturbo`（3-bit）、`models/gemma4-r2.gturbo`（2-bit）

### 8.1 CLI 選 bit 用法（新增 `--routed-bits`）

```bash
# 給 base 路徑 + bit 寬，自動解析到對應 variant（找不到會提示 Rebits 指令）
TurboFieldfareCLI --model models/gemma4.gturbo --routed-bits 3 --prompt "..."   # -> gemma4-r3.gturbo
TurboFieldfareCLI --model models/gemma4.gturbo --routed-bits 2 --prompt "..."   # -> gemma4-r2.gturbo
TurboFieldfareCLI --model models/gemma4.gturbo --routed-bits 4 --prompt "..."   # 使用 base 原樣

# 直接指定 variant 目錄亦等效（idempotent）
TurboFieldfareCLI --model models/gemma4-r3.gturbo --prompt "..."
```

- `--routed-bits 2|3|4`（新增於 `Args.swift`）: 2/3 時解析 `--model` 的 sibling `-r2`/`-r3` 目錄；已宣告相同 bit 的目錄直接使用；variant 不存在時印出
  `TurboFieldfareRebits --input <base> --output <variant> --routed-bits N` 提示。
