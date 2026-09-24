# `p2/bit-identical` 審查報告：兩個數字，零個可歸因的證據

**日期** 2026-09-22 · **受審分支** `flashkv-p2bitident` @ `16f40a8ff`（同一個 repo 的 worktree）
**審查者** 本線（devserver / M1–M5 線）· **未改動該分支任何檔案**

---

## 0. 一句話

**那個「prefill 285 / decode 31.5」沒有問題意識上的錯誤，但它不是這個分支的成就，也不是那個 kernel 改動的成就。**
分支相對本線的全部程式碼 delta 是 **8 行 Metal**；而它宣稱達成的「prefill250 profile」有一半的 env 在這顆引擎裡**根本不存在**，
唯一留下的那支 server log 列印的是**別的 binary** 才有的字串 —— 所以那兩個數字**無法歸因到任何被 commit 的東西**。

真正值得拿走的是一個候選（`sum_parts` 重寫），而它的價值是它自己 commit 說的 **+3%**，不是 2.4×。

---

## 1. 分支的事實狀態

```
merge-base(p2, devserver) = b8a564d45  (P2 GEMV coalescing, 09-07)
p2 獨有 commits = 7     devserver 獨有 = 393
```

七個獨有 commit 裡，**只有一個**動到引擎：

| commit | 時間 | 內容 |
|---|---|---|
| `da6903739` | 02:56 | **唯一**的引擎改動：`ggml-metal.metal` 8 行（P2-C revised） |
| `193adba66` | 03:46 | commit **二進位 dylib**（含 `build/bin/*`），宣稱 prefill 285 / decode 31.5 |
| `813d72d35` | 03:47 | 白皮書 |
| `512920c22` | **10:20** | 從 demo branch **複製 harness**（decode_bench / llama_bench_matrix / thermal_pressure / run_server.sh） |
| `1e398e3f8` | 10:25 | commit 12 顆重編的 test binary，**0 行原始碼變更** |
| `e14d9ca89` / `16f40a8ff` | 10:26 / 10:34 | 改白皮書數字 |

注意時序：**數字寫在 03:47，能重現它的 harness 在 10:20 才進來**（晚了 6.5 小時）。

---

## 2. 引擎 delta：「P2-C revised」就是這 8 行

本線（devserver）**已經有** P2-B（float4 向量載入）與原本的 `#pragma unroll`；p2 唯一的源碼差異是把 l-loop 的 unroll
換成 `float2 sum_parts[4]` + 固定順序合併：

```metal
// devserver 現況（b8a564d45 帶進來的）        // p2/bit-identical
float2 sum = {0};                              float2 sum_parts[4] = {{0},{0},{0},{0}};
#pragma unroll                                 for (l...) {
for (l = 0; l < 4; ++l)                            #pragma unroll
  #pragma unroll                                   for (j...) sum_parts[l][.] += ...;
  for (j = 0; j < 4; ++j)                      // 固定順序合併
    sum[0] += ...; sum[1] += ...;              float2 sum = { sum_parts[0][0]+sum_parts[1][0]+... };
```

驗證方式（可重跑）：`git grep -c 'sum_parts' <metal>` → p2=10、devserver=0。

**這個 pattern 本身是對的**：保 ILP、鎖 FP 順序，正是 bit-identity 要的形狀。問題不在這 8 行，問題在它被拿去解釋 2.4×。

---

## 3. 決定性證據：數字不是這顆引擎產生的

### 3.1 白皮書的 config 有一半在這顆引擎裡不存在

`run_server.sh`（10:20 從 demo 複製）allowlist 了，但引擎原始碼**零命中**：

| env | 白皮書 §2.1 用途 | `run_server.sh` | 引擎原始碼 |
|---|---|---|---|
| `CGC_PREFILL_STREAM` | whole-layer slab streaming | 7 | **0** |
| `CGC_GATHER_SLAB_CAP=256` | 256 experts in slab | 6 | **0** |

（`CGC_MM_BITIDENT` / `CGC_GLU_FUSED_DOWN` / `CGC_SPAC` / `CGC_DBUF` / `CGC_VERIFY_DECODE` / `CGC_DRAFT_DECODE` 是**有**實作的。
我第一次用 `git grep <tree-ish> -- <pathspec>` 得到「0」，是那個形式的誤報，改用 worktree grep 後修正 —— 這一格必須是這樣查的。）

整個 M1 家族在 p2 引擎裡是 0 命中，本線是齊的：

| 字串 | p2 引擎 | devserver 引擎 |
|---|---|---|
| `CGC-PHASE-SPLIT` | 0 | 1 |
| `CGC-PREFILL-STREAM` | 0 | 3 |
| `CGC-GATHER-SLAB` | 0 | 1 |
| `CGC-HOOKSPLIT` / `CGC-DECPROF` / `CGC-MMID-ASSERT` | 0 / 0 / 0 | 1 / 3 / 5 |

⇒ 「**完整套用 prefill250 profile（PREFILL_STREAM + SLAB_CAP + batch 6144）**」這句話，在這顆引擎上做不到。
285 只可能來自**另一條線的 binary**。

### 3.2 唯一留下的 log，證明跑的是別的 binary

`Backup/cgc_logs/llama_server_20260922_023349.log`（02:33，就在 02:56 的 kernel commit **之前**）印著：

```
CGC-PHASE-SPLIT: cap=8 routable=142 slots ... prefill slab NOT armed
CGC-MMID-ASSERT name=ffn_moe_gate-1 ne02=143 ... zero_row=1
CGC-GATHER-SLAB ...
```

這三族字串在 p2 的原始碼裡是 **0**，在 devserver 是 1–5。⇒ 這支 log 由 **M1 線的 binary** 產生。
而且它是 **launch log**：`wc -l = 134`，`grep -c 't/s' = 0` —— 它**不含任何效能數字**。

同一目錄的 `llama_server_latest.log` 是一個**斷掉的 symlink**，指向 10:19 那個不存在的檔。
`Backup/` 被 `.gitignore:391` 忽略 ⇒ 這兩份證據都不會隨分支走。

### 3.3 但不能反過來說「二進位是偷來的」

我對 commit 進去的 dylib 做了字串比對：`libllama.0.0.182.dylib` 只有 `CGC_SPAC`/`CGC_DBUF`，
**沒有** `CGC-PHASE-SPLIT`/`GATHER-SLAB`/`HOOKSPLIT` —— 與 p2 自己的原始碼**一致**。
⇒ 03:46 commit 的那顆 binary **是**從 p2 源碼編的（可信），但**量測當時**（02:xx–03:4x）在跑的不是它（見 §3.2）。

**這是本報告最嚴重的一項**：產生頭條數字的載體已經不可回收；而被 commit 的載體從未被量過。
`libggml-metal` 內的 metallib 是編譯產物，**光靠檢查無法回答**commit 的 binary 是否含 `sum_parts` —— 那需要一輪實測（§7）。

---

## 4. 同一個配置，四個互斥的數字

| 來源 | 基線 | 該項改動後 | bit-exact |
|---|---:|---:|---|
| `b8a564d45` commit msg | **23.88** median | 25.17 (+5.4%) | 未提 |
| `da6903739` commit msg | **12.57** | **12.93** (+3%) | 「safe」；並說 25.17 那份是 broken |
| `193adba66` commit msg | — | prefill 285 / **decode 31.5** | bit-exact revised |
| 白皮書 §1 | — | prefill 285 / **decode 31.5** | PASS |
| 白皮書 §3.2 表 | demo baseline 12.57 ✅ | **P2-C revised 31.5 ✅**；P2-B only 9.59 ✅；P2-B+P2-C original 25.17 ❌ | |
| 白皮書 §4.1（10:34 更新） | — | 短 ctx **25.3–28.9** @100% accept；長 ctx **4.6–5.3** @49–57% | |

- 同一個「P2-C revised」，分支自己的產物給出 **12.93 / 25.3–28.9 / 31.5** —— 差距 **2.4×**。
- 基線在兩份 commit message 裡是 **23.88 與 12.57**（同一台機器標籤、同一配置家族）。
- **31.5 只存在於白皮書與 commit message**，在全 repo 的 json/log/md/txt/html 裡沒有任何量測產物支持它。
- 白皮書 §4.1 說「短 context 25.3–28.9」，§1 卻仍寫 31.5 —— 同一份文件內的自我矛盾。

---

## 5. 31.5 是 accept rate 的效果，不是 kernel 的效果

用本線自己的交付數字算一次：plain step ≈ 100–204 ms，**accept 100% 且 k=3 ⇒ 每步 4 個 token**：

```
step 135–175 ms ÷ 4 token  =  23–30 t/s
```

⇒ 31.5 落在「**accept = 100% 的短 context**」算得出來的區間內，**不需要任何 kernel 改動**。
而白皮書 §4.1 自己說交付 regime 是 **4.6–5.3 t/s @ 49–57% accept** —— 這比本線現況還差。

**「長 context 慢 = MTP accept 掉到 ~50%」不是新發現，是這條線量了兩週的同一個牆**（我們的記錄：accept 46.5%–58%，
mean_len 2.57–3.72，25 = mean_len / step）。他們把它寫成「已確認原因」，但在該分支上沒有任何 accept 量測產物。

**結論**：頭條數字描述的是**工作負載性質**（短、好猜），不是**引擎改進**。

---

## 6. 方法論上缺失的東西（本線已經有，他們沒有）

| 要求 | p2 | 本線 |
|---|---|---|
| oracle 閘門（M1/M2/M3） | **不在 `scripts/check`**（只有 `cgc_logits_oracle_compare.py` 一支無閘門腳本） | `m123_oracle_gate.py`，9/9/9 |
| engine digest 蓋在產物上 | 無 | `engine_digest` 進 summary |
| 窗口條件（reclaimable/swap/thermal/foreign procs） | 無 | `server_window` + thermal gate |
| 交錯 reps、丟第 1 rep、報中位 | 無 | 標準協定 |
| 硬體身分 | **「MacBook Pro M4 Max 16GB」** | `sysctl hw.model` = **Mac16,12**（MacBook Air M4, 10 核, 16 GB） |
| 熱衰減記錄 | 有寫（§4.3 271→104 t/s）但**沒進任何數字的條件欄** | 進條件欄並拒跑 |

硬體那一格是硬錯誤：**16 GB 的 M4 Max 不存在**（M4 Max 從 36 GB 起）。這一台是 Mac16,12。
在我們的規則下，這一句就足以讓 285 / 31.5 **不可與任何其他量測並排**。

repo 衛生另外兩項：`1e398e3f8` 用 **0 行原始碼變更** commit 12 顆重編 test binary；
`src/llama.cpp/build/bin/*` 與 `deploy-harmonyos/*/lib*.dylib` 都在版控內 ——
「引擎身分」實際上是一團沒有 digest 綁定源碼的 binary blob。

---

## 7. 「P2-C original 破壞 bit-exact」這句，本線的證據不支持

分支自己的表把 `P2-B+P2-C original = 25.17` 標成 ❌ bit-exact broken，並用這個 ❌ 來論證 revised 版本的必要性。

但本線**跑的就是那段程式**：`ggml-metal.metal` 在 devserver 有 P2-B 與 l-loop 的 `#pragma unroll`（`sum_parts` = 0 命中），
而且 `run_server.sh:107` 預設 `CGC_SERVER_GLU_FUSED_DOWN=1` ⇒ `kernel_mul_mv_id_glu_iq3_xxs` 這條路**預設就在**，
閘門在這種配置下 **M1/M2/M3 = 9/9（含跨池不變性）**。

兩個可能，都指向同一個結論：要嘛 unroll 沒有被 Metal 重排（那 ❌ 是別的原因），
要嘛被重排的效果在本線的 workload 上不顯現（那 ❌ 只在它們的 workload 成立）。
**要裁決必須跑一次「同一顆 build、只換這 8 行」的閘門對照 —— 它們從未跑過閘門，所以這句在該分支上從未被檢驗。**

---

## 8. 結論與建議

### 該拿走的
- **候選：`sum_parts` 固定順序合併**（8 行）。形狀正確、與 bit-identity 相容、成本低。
  但它自己的 commit 說 **+3%**，本線應該用**自己的閘門**重新量，預期收益以「個位數 %」計，不是 2.4×。

### 不該引用的
- `285` / `31.5` **不可引用**：載體不可歸因（§3.2）、配置在這顆引擎不存在（§3.1）、
  同一配置四個數字（§4）、硬體身分錯誤（§6）。
- `25.17`（❌）/`9.59` 同樣不可引用。
- 「P2-C original 破壞 bit-exact」（§7）在本線證據下不成立，**不要**拿它當重寫 kernel 的理由。

### 可以要求對方做的兩件事（各一輪即可自證）
1. **把 binary 綁到 source**：對 commit 進去的 `libggml-metal` 跑一次 `strings`/md5 並寫進白皮書，並說明量測當時用的是哪一顆。
2. **跑一次 oracle 閘門**（`m123_oracle_gate.py`，或至少 `row_fnv1a64` + argmax 兩欄），
   在同一顆 build 上比 `unroll` 與 `sum_parts` 兩版 —— 這是唯一能裁決 §7 的實驗，成本一輪。

### 本線的下一步（若要把候選變成成果）
把 `sum_parts` 移植到 devserver，**先過閘門後量速度**：M1/M2/M3 9/9 → 同一窗口 MTP-on/off 交錯 3 rep，
報 step / mean_len / accept，並與現行 `unroll` 版配對比較。預期 **+0–5%**，可能為零；若過閘門且 ≥3% 才值得留。

---

## 12. 複查 `p2/bit-identical-rebased`（2026-09-22 14:0x）——兩件事做到了，但那 24 行打不到我們的路徑

他們把 §8 的兩件事都動了：rebase 到 `demo/sweet-spot-windows-fix`、加了 engine digest、跑了一次 oracle 閘門。
以下每一項都核對過檔案，不是讀 commit message。

### 12.1 做到了（可引用）

| 項 | 讀數 | 出處 |
|---|---|---|
| 分支幾何 | `demo/sweet-spot-windows-fix...HEAD` = **0 / 1**（沒有漏掉任何 demo 的 commit） | `git rev-list --left-right --count` |
| delta | **1 檔 24 行**，2 個 hunk，都在 `kernel_mul_mv_id_glu_iq3_xxs_impl` | `git show HEAD --stat` |
| commit 訊息聲稱的 391 | `git rev-list --count b8a564d45..b54b83e80` = **391** ✔ | |
| engine digest | 4 個 artifact 的 md5 + mtime，**外加 `tree.head` 與 `dirty_paths`** | `Backup/m123_oracle_gate/summary_p2_rebased.json` |
| 閘門 | M1/M2/M3 **9/9/9**、`comparable=true`、`config_diffs=[]`、`ref_pinned=true` | 同上 |
| 參考沒被重生 | `ref_md5=72d82a33ad79e0e69bc935acd24228f2`，**與 09-20 起所有 summary 記錄的同一顆** ⇒ 不是拿新 build 的產物跟自己比 | 43 份 summary 的時間序列 |
| 量到的那顆 = 現在的原始碼 | summary 記的 metal md5 `430f9d315cbcf6c4` = 現在磁碟上那顆；14:01 重編**逐位元組相同**（md5 不變）⇒ build 可重現 | `md5 -q` 對照 |

⇒ **§8 要求的「兩個自證」裡，第①項（digest 綁定 + 說明量的是哪顆）完成了**，而且做得比要求的細（連 dirty
狀態都記）。第②項（同 build 只換 8 行的裁決）只在「有改動的那一側」跑了；baseline 臂仍是別的 session、別的 digest
（`head c2690b67c` / metal `6aa7981e159146cc`），所以嚴格說**沒有一個同窗口的配對**。

### 12.2 但這個閘門對這 24 行**零覆蓋**——它測不到

`kernel_mul_mv_id_glu_iq3_xxs_f32` 在我們的交付載體上不可能被編進圖，證據鏈是靜態且逐環可查的：

1. **唯一入口**：這族 kernel 只由 `ggml_metal_library_get_pipeline_mul_mv_id_glu()` 選擇，而它全庫只有**一個**
   呼叫點（`ggml-metal-ops.cpp:3248`）。
2. **名字由 gate/up 的型別決定**（不是 down）：`device.cpp:1397`
   `snprintf(base, "kernel_mul_mv_id_glu_%s_%s", ggml_type_name(tsrc0), ...)`，`tsrc0 = w->type`；
   而 `switch (tsrc0)` **只接受 `IQ3_XXS` 與 `IQ2_S`，其餘 `GGML_ABORT`**。
3. **我們的 gate/up 是 IQ2_S**：`blk.1.ffn_gate_exps.weight` / `ffn_up_exps.weight` = **IQ2_S**（82 顆裡 78 顆），
   `ffn_down_exps` = IQ3_S。
4. **整個檔案沒有一顆 IQ3_XXS**：753 顆的型別分佈 = F32 368 / IQ4_XS 253 / **IQ2_S 78** / IQ3_S 39 / Q8_0 8 /
   Q6_K 2 / Q2_K 2 / BF16 2 / Q3_K 1。⇒ 就算把手寫的 `_iq3_xxs` 路徑想成「patch 到錯的變體」，它連被選中的資格都沒有。
   （另一顆 35B 載體 Ornith 的 expert 是 Q4_K/Q3_K/Q8_0，同樣 0 顆 IQ3_XXS。）
5. **而且這族是 opt-in**：`ggml-metal-ops.cpp:2800` 的註解自己寫「Enabled via env **CGC_MMV_FUSE=1**（default off）」；
   `ggml_metal_op_can_fuse_mmv_glu` 要求 `getenv("CGC_MMV_FUSE")[0]=='1'`；本線啟動器的 banner 逐字印
   `glu_fused_down=1(mm_fuse=off)` ⇒ **連 `_iq2_s` 那個變體在交付配方裡都不存在**。
6. **連移植都不成立**：`kernel_mul_mv_id_glu_iq2_s_impl` 的 l-loop 只有 **2 次迭代且沒有 `#pragma unroll`**，
   收尾是 `d1*sum[0] + d2*sum[1]`——24 行的形狀（4 條鏈）不適用。

⇒ 所以 **9/9/9 不是「這 8 行 bit-safe 的證據」，是「這 8 行從未被執行」的證據**。
「bit-exact safe」在這個分支上目前是**未受測的主張（空覆蓋）**，而不是已驗證的性質。同樣地：這 24 行**不可能**在我們的
載體上產生任何 prefill/decode 增益——不論方向、不論幅度。

### 12.3 這一輪完全沒有速度產物

rebase 之後的 server log（13:53 / 13:57 / 14:01 / 14:05）**`t/s` 命中 = 0**，內容是 acceptance/diagnostic。
⇒ 「prefill 250 / decode 30」在這條分支上**沒有拿到任何新支持**，狀態與 §4 相同（四個互斥數字、無落盤產物）。

### 12.4 分支本身還不是自洽的（誠實但危險）

`git show HEAD:.../libggml-metal.0.19.0.dylib` md5 = **544b6c94bc7a372d**（baseline），working tree 的 = **430f9d315cbcf6c4**。
那顆 dylib 是 **tracked 但沒進 commit** ⇒ **直接 checkout 這條分支，拿到的是 baseline 的執行檔 + 已改的原始碼**——這正是
先前讓「量測對象 ≠ commit 內容」的那個 hazard（他們用 `dirty_paths` 記下来了，這點是做對的；但分支自身仍不自洽）。

### 12.5 下一步（順序不能顛倒）

1. **先讓 kernel 可達**：把同樣的意圖搬到 `_iq2_s` 變體（2 條鏈）**或**開 `CGC_MMV_FUSE=1` 並用一顆真的有 IQ3_XXS gate/up
   的載體。不可達的情況下，任何閘門與任何 t/s 都只是空白。
2. **用 witness 證可達**：`CGC_MMV_FUSE_DBG=2` 會逐次印 `dispatching fused gate+up+glu (type %s ...)`；
   那一行必須出現 `IQ3_XXS` 才能開始談這 8 行。
3. **然後才是配對**：同 build、同窗口，只換這 8 行，跑 oracle 閘門（M1 必須是 9/9 **且** witness 有命中）+ 一輪速度。

預期值仍需說清楚：`m` 這一側本線已量到 +1.7%（M-W）、他們自己量到 +2.9%（12.57→12.93）⇒ 這條路的天花板是 **單一
數字百分比**，不是 2.4×。

---

## 13. 「把 09-07 的優化做到 bit-identical 就能到 25 t/s」——方法對，但 25 不是它買到的

這是那條分支上唯一還站著的論述，所以值得逐項拆。三個術語裡有兩個是對的，第三個（25）掛在別的東西上。

### 13.1 它自己的兩臂說的是 +5.4%，不是 2×

`b8a564d45`（09-07，P2 的 merge-base）訊息原文：

```
Benchmark results (16GB M4 Max, 8GB pool, DBUF+SPAC, coding profile, 3 runs+warmup):
- Baseline:        23.88 t/s median / 26.22 best
- P2-B+P2-C:       25.17 t/s median / 25.87 best (+5.4% median)
- draft_accept: 98.2% (stable)
```

同一顆 commit 裡，baseline 與 P2 兩臂是**同一配置、同一窗口**，差 **+5.4%**。那是這 8 行（或 24 行）能主張的全部。25.17 是「23.88 已經在那裡」之後的乘積。

### 13.2 那個 23.88 是站在 98.2% accept 上的

`9dde528a5`（09-07，`release: v1.0.0 production`，分支 `production`）原文：

```
- 性能：coding profile 25.17 t/s 中位数 / 25.87 最佳
- 质量：3/3 profiles 达标（qa-zh 1.0 / longform-zh 0.882 / coding 1.0）
- draft_accept：98.2%
```

所以那個 25.17 的完整描述是「**coding profile @ 98.2% accept**」。`coding` profile 的定義（`git show b8a564d45:scripts/run_server.sh`，:184-199）不是一般問答：它用 assistant prefill 強制 ` ```python\n# ` 起手、`stop=``` `、`max_tokens=512` —— 也就是把模型鎖在**逐字接續一段程式碼/註解**，那正是 MTP 最好中的 regime。

### 13.3 同一顆 09-07 的 `run_server.sh` 自己記了這個 accept 家族的代價

同檔 :254-262（就在 pool budget 那段）：

```
# [CGC 2026-09-06 pool budget 4GiB -> 8GiB] A/B on this 16GB Mac (MTP+denseIQ4X, ngl=99):
#   - 8GiB default (no renorm): quality = baseline exactly (qa-zh 1.0 / longform 0.882 /
#     coding 1.0 over 5 runs), decode ~11-22 t/s vs 4GiB's 7-14
#   - 8GiB + CGC_RN_ROUTING=1 + CGC_WCOLD_EN=1: decode 25.9 t/s / 98.7% accept but quality 0.3
#     (renorm mask mechanism bug, invariant to pool size)
```

三個事實寫在同一段裡：**default 是 11–22 t/s**；**25.9 t/s 是 98.7% accept 的產物**；而**那個 accept 水準是與 quality 0.3 同時出現的**。98.2% 與 98.7% 是同一族數字。

我**不能**證明 v1.0.0 那次 98.2% 的輸出是退化的（它宣稱三 profile 品質達標，而強制程式碼續寫本來就真的比較可預測）。但兩種讀法導向**同一個結論**：25 是 accept 的函數，不是 kernel 的函數——所以「把 kernel 做到 bit-identical」不會把它帶過來。

### 13.4 2.00× 的缺口怎麼分

| 因子 | 09-07 的編織 | 我們的交付 | 比值 |
|---|---|---:|---:|
| 每步 token 數（=1+3·accept） | 3.95（98.2%） | 2.77（58.9%） | **1.43×** |
| 步時（推導） | 157 ms | ~220 ms | **1.40×** |
| **乘積** | **25.17 t/s** | **12.57 t/s** | **2.00×** |
| 那 24 行 | 23.88→25.17 | — | **1.054×** |

（兩臂的 12.57 t/s / 58.9% accept 取自同一顆 build 的閘門 run；步時是從 t/s 與 mean_len 反推，不是獨立儀器，所以只當比例用。）

**kernel 是 2.00× 裡的 1.054×。** 就算這 24 行完美移植、且可達，落地值是 12.57 × 1.054 = **13.25 t/s**。那不是「衝到 25 的路」，那是線 I 上又一格 +0–5% —— 與它自己另一顆 commit 量的 **+3%**（12.57→12.93）同量級。

而那 1.40× 的步時項，本線自己的紀錄已經定性過（`4396874e8`，`io_granularity_curve.py`）：

> joint_capture puts expert fill at 9.2% of the step and GPU busy at 80.0%. So even FREE expert IO is +10.1% t/s. 25 needs the step 1.80x faster, which means GPU busy 1.8-2.2x faster -- **25 is a GPU-efficiency problem, not an IO problem.**

### 13.5 那 24 行在本線上的可執行性

| 前提 | 狀態 |
|---|---|
| 改的變體 | `kernel_mul_mv_id_glu_iq3_xxs_impl`（2 hunk，16+/8−，1 檔） |
| 誰決定用哪個變體 | `ggml-metal-device.cpp:1376-1397`，由 `tsrc0 = w->type` 拼名 |
| 我們載體的 gate/up | **IQ2_S**（753 顆張量裡 IQ3_XXS = **0**） |
| `_iq2_s` 變體存不存在 | 存在（`ggml-metal.metal:11416`），但**沒有 unrolled l-loop** ⇒ 這 24 行的形狀不適用，要**新寫**不是 cherry-pick |
| 這族怎麼才會被走 | `CGC_MMV_FUSE=1`，**預設 off**（`ggml-metal-ops.cpp:2800`、:2812） |
| 它與 bit-identity 相容嗎 | 設計上相容：同檔 :2800-2803 寫「Bit-identical to the unfused pair + swiglu」，:2947-2956 說它是 pure perf、並用 galloc 重疊 guard 拒用會壞的那一格 |

**可執行，但形狀要改成**：把同樣的「分離部分和 + 固定順序合併」寫進 `_iq2_s` 變體，開 `CGC_MMV_FUSE=1`，用 `CGC_MMV_FUSE_DBG=2` 見證它真的印出 `IQ2_S`（而不是沒印），再過我們的 M1/M2/M3。預期 **+0–5%**。

### 13.6 一句話

**他的方法對（鎖 FP 順序而不是關掉 ILP，是唯一能同時要 ILP 與 bit-identity 的形狀），他的證據也誠實（同一 commit 兩臂 +5.4%）。錯的只有把 09-07 那個 98.2% accept regime 的頭條數字，當成這次 kernel 改動的成果，再搬來我們分支當 25 t/s 的路。** 那條數字 09-13 就已經被本線歸檔降級（`3b70390f3` → `docs/archive/pre-consistency-metrics-2026-09-11/`），理由正是「ungated numbers 不再被引用」——含那個 25 t/s。

---

## 附錄：本報告每一格的查法（都可重跑）

```bash
# 分支幾何
git -C flashkv-p2bitident merge-base p2/bit-identical autotuner/m1-shape-inventory   # b8a564d45
git -C flashkv-p2bitident log --oneline autotuner/m1-shape-inventory..p2/bit-identical

# kernel delta（唯一源碼改動）
git -C flashkv-p2bitident show da6903739 -- src/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal
grep -c sum_parts <p2-metal>        # 10
grep -c sum_parts <devserver-metal> # 0

# 引擎是否有該機制（注意：用 worktree grep，不要用 <tree-ish> + pathspec 形式）
git -C flashkv-p2bitident grep -l -- CGC-PREFILL-STREAM -- 'src/llama.cpp/src/*' 'src/llama.cpp/ggml/src/*'   # 空

# log 的身分與內容
grep -c 'CGC-PHASE-SPLIT' Backup/cgc_logs/llama_server_20260922_023349.log   # 3
wc -l ...  &&  grep -c 't/s' ...                                            # 134 / 0
git -C flashkv-p2bitident check-ignore -v Backup/cgc_logs/...                # .gitignore:391

# 硬體
sysctl -n hw.model    # Mac16,12

# §12 kernel 可達性（靜態，不需要跑）
ls -lh src/llama.cpp/build/bin/libggml-metal.0.19.0.dylib      # 對 summary 的 engine_digest
python3 -c "
import sys,collections; sys.path.insert(0,'src/llama.cpp/gguf-py')
from gguf import GGUFReader
r=GGUFReader('models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf')
print(collections.Counter(str(t.tensor_type).split('.')[-1] for t in r.tensors))"
sed -n '1397p' src/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp   # 名字由 w->type 決定
sed -n '2800,2815p' src/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp # CGC_MMV_FUSE default off
```

**未 commit 這份報告以外的任何東西；未改動 p2 分支的任何檔案；沒有殘留行程。**
