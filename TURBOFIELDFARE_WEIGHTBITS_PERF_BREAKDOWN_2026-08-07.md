# TurboFieldfare 4-bit vs 3-bit 效能分解報告（TTFT / decode / 時間去向）

**日期**: 2026-08-07
**配置**: B-tree prompt、max-new 128、64 slots/layer + cross-layer prefetch + lookahead、`--trust-receipt`（全部優化已落地）
**測量**: 同一環境背靠背各跑一次完整 telemetry；吞吐另有 5 輪交錯中位數

---

## 一、總覽（3-bit vs 4-bit）

| 指標 | 4-bit | 3-bit | 變化 |
|---|---|---|---|
| 首字延遲 (TTFT) | 3.73 s | 2.89 s | **↓22%** |
| 端到端牆鐘 | 16.24 s | 12.57 s | **↓23%** |
| 解碼吞吐 | 10.24 tok/s | 13.22 tok/s | **↑29%** |
| 解碼時長 (decode) | 12.51 s | 9.68 s | **↓23%** |
| 實際讀盤量 (此輪) | 10.05 GiB | 9.72 GiB | −3%（routing 差異） |
| 每次 fetch 成本 (stride) | 3.36 MB | 2.61 MB | **↓22%** |
| expert 命中率 | 94.8% | 93.5% | −1.3pp |

5 輪交錯中位數（同一 harness）：**4-bit ≈ 11.5 tok/s，3-bit ≈ 13.6 tok/s（+18%）**；TTFT 3.30 → 2.87s。

---

## 二、時間去向分解

### 4-bit（牆鐘 16.24 s）

```
GPU 計算      ████████████░░░░░░░░░░░░░░░░░░░░  5.22 s  (32%)
  ├ 注意力              ████████████  1.97 s   ← 最大單項
  ├ routedMoE           ██████████   1.57 s
  ├ sharedFFN           █████        0.73 s
  ├ phase1Hit           █           0.15 s
  └ head                █████        0.80 s
GPU 空轉      ████████████████░░░░░░░░░░░░░░░░  7.27 s  ← 119 個 >5ms 長停 = 4.05 s
  │                                                   max = 1033 ms (磁碟抖動)
調度空轉      ███████████████░░░░░░░░░░░░░░░░░  6.75 s  (sched, CPU submit gap)
CB 驅動開銷   ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.13 s  (12,204 個 × 93 μs)
LM head      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.84 s  (CPU) + 2.56 s 跟在 head 後的空轉
暴露 IO      ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2.00 s  (CPU 阻塞等待 fetch；readWall 6.94s)
```

### 3-bit（牆鐘 12.57 s）

```
GPU 計算      ████████████████░░░░░░░░░░░░░░░░  6.01 s  (48%)
  ├ 注意力              ██████████████  2.23 s   ← 最大單項
  ├ routedMoE           ████████████   1.89 s
  ├ sharedFFN           █████          0.81 s
  ├ phase1Hit           █             0.22 s
  └ head                ██████         0.86 s
GPU 空轉      ████████░░░░░░░░░░░░░░░░░░░░░░░░  3.68 s  ← 只剩 28 個 >5ms = 0.17 s
  │                                                   max = 8.7 ms
調度空轉      ██████████████████░░░░░░░░░░░░░░  7.96 s  (sched, CPU submit gap)
CB 驅動開銷   ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.95 s  (12,416 個 × 77 μs)
LM head      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.91 s  (CPU) + 0.41 s 跟在 head 後的空轉
暴露 IO      ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.21 s  (CPU 阻塞等待 fetch；readWall 4.94s)
```

> 注：各欄是不同視窗的計數器（CPU 階段、GPU timeline、sync 開銷），不嚴格相加；用途是看出**瓶頸結構**。

---

## 三、關鍵差異：長停（>5ms GPU gap）

| | 4-bit | 3-bit |
|---|---|---|
| >5ms 停頓數 | **119 個** | 28 個 |
| >5ms 累計 | **4.05 s** | 0.17 s |
| 最長單一停頓 | **1033 ms** | 8.7 ms |
| head 後空轉 | 2.56 s (10.1ms 平均) | 0.41 s (1.6ms 平均) |

**這是 3-bit 最主要的勝利**：46 GB 模型全在盤上、可用記憶體吃緊時，4-bit 每層專家讀取常被打出 page cache → 每次 decode 卡 0.1–1 秒。3-bit 的較小 stride 把這類抖動幾乎消除（48% GPU busy vs 32%），IO 從「主導瓶頸」退場。

---

## 四、剩餘時間去哪了（3-bit，牆鐘 12.57 s）

```
GPU 真實計算   6.01 s   ← 注意力 2.23s 是最大單項（已從「IO-bound」轉為「compute-bound」）
調度空轉       7.96 s   ← CPU encode/submit 趕不上 GPU；需要減少 CB 數與加深流水線
GPU 空轉       3.68 s   ← 已無長停，剩細碎 gap
CB 驅動開銷    0.95 s   ← 12,416 個 command buffer × 77 μs
LM head        0.91 s   ← CPU 取樣 + 下一 token 準備
暴露專家 IO    1.21 s   ← 已很小，不再卡 decode
```

---

## 五、後續優化建議（按 ROI 排序）

### A. TTFT（現在 2.89 s）

| # | 手段 | 預期 | 現況 |
|---|---|---|---|
| A1 | 切 prefill read mode：`TURBO_FIELDFARE_PREFILL_EXPERT_READ=coalesced` / `layerLocalReadahead`（代碼已支援，未測） | TTFT −5~15% | 預設 baseline |
| A2 | prefill 多 worker：`TURBO_FIELDFARE_EXPERT_READ_WORKERS` 提高 | TTFT −5~10% | 未調 |
| A3 | TTFT 敏感場景直接用 2-bit（TTFT ≈ 2.5 s） | −13% | 質量可接受再上 |
| A4 | prefill 層級局部預讀（layerLocalReadahead）提前把下一層專家讀進 cache | −10% | 未測 |
| A5 | 啟動前用 `fadvise`/檔案暖 cache（`verified-install` 已省 3.9s 重 hash） | 穩定 TTFT 抖動 | 未做 |

### B. Decode（現在 9.68 s / 13.2 tok/s）

| # | 手段 | 預期 | 現況 |
|---|---|---|---|
| B1 | **Command-buffer 融合**：12,416 個 CB → 每層 1–2 個（attention+MoE+head 合併 encode）。直接消掉 0.95s overhead 與部分 sched | **tok/s +15~25%** | 最大的結構性開銷 |
| B2 | **head/取樣與下一 token 重疊**：head 後空轉 0.41s（253×1.6ms）→ 把取樣挪進 GPU kernel 或與下一層 encode 並行 | +3~5% | 次大單點 |
| B3 | **注意力優化**：attn 已是最大 GPU 單項 (2.23s) → 確認 decode 是否走 tensor-ops 路徑 / SWA window 是否可縮 | +5~10% | 值得單獨 A/B |
| B4 | **調度空轉 (sched 8s) 治理**：encode 與 GPU 執行 double-buffer、加深 pipeline，讓 CPU 提前 1 層 | 吃回大部分 sched | 需要 pipeline 改造 |
| B5 | **MTP 重新評估**：先前「MTP 無法提速」的結論是在 4-bit IO-bound（GPU busy 32%）下得出；3-bit 已轉為 compute-bound (48%)，投機解碼的收益條件已不同，值得用 r3 重新 bench | 待測 | 舊結論已不適用於 r3 |
| B6 | 極致吞吐場景：2-bit ≈ 20.5 tok/s（質量退化的效能模式） | +55% vs r3 | 需質量門檻 |

### C. 環境

- 釋放磁碟（現在 46 GB 模型 / 剩餘 <15 GB）：長停與 cache 抖動與此高度相關；同一時間只保留 4-bit + 一個變體可再降抖動。
- 若 16 GB 統一記憶體允許，把 2–3 個層的 experts 長駐（`sharedResident` 已有機制）可把 decode IO 完全歸零。

---

## 六、一句話結論

3-bit 已把「暴露 IO」從主矛盾打成次要項（長停 4.05s→0.17s、GPU busy 32%→48%），decode 結構上仍被「12k 個 command buffer + CPU 調度空轉」壓著；下一步 ROI 最高的是 **B1 CB 融合** 與 **B5 在 r3 上重測 MTP**，TTFT 則先試 **A1/A2 prefill read mode**。

（原始 telemetry 存於 `/tmp/tf_tele_gemma4.gturbo.txt`、`/tmp/tf_tele_gemma4-r3.gturbo.txt`）


---

## 七、Phase 1/2 实测更新（2026-08-07 午后）

### 7.1 Phase 1 完成：newkernel 解决 attention + GEMM 正确性（并带来性能收益）

`bin/` 内两个二进制（源码切换点 `Attention.swift` partialPipeline / `RealForwardRunner.swift` 的 `encodeTgx`）：

| 项目 | baseline（旧 generic attention + encode） | newkernel（GQA-full + encodeTgx） |
|---|---|---|
| 正确性 | **全部 prompt 乱码/空白**（attention 与 GEMM 旧 bug） | 全部干净（`3+5=8`、haiku、`def greet`、fibonacci） |
| 4-bit tok/s（交错） | ~13.3 | ~17.2（**+29%**） |
| 3-bit tok/s（交错） | 乱码（不可比） | ~15.8（干净） |

- 结论：**newkernel 是唯一可用内核**，且 4-bit 上明显更快；`.build/release` 之前被增量缓存卡在 baseline 产物，已重建对齐。

### 7.2 B1：shared FFN 融合进 cb1 — 三窗口证据冲突，默认保持 split（opt-in 保留）

实现：`TURBO_FIELDFARE_FUSE_SHARED=1` 把 sharedFFN+sharedNorm 编进 cb1（每层少 1 个 CB：12,416 → 8,569）；默认走原 early-split 路径。

| A/B 窗口 | 条件 | split | fused | 结论 |
|---|---|---|---|---|
| 窗口 1（16 slots，msg_code prompt） | IO-bound | ~8.1–9.9 | ~7.1–7.8 | fused −12~21% |
| 窗口 2（64 slots，交错 ×4） | 磁盘 88% 满、页缓存抖动 | ~8.4–9.3 | ~12.3–14.2 | 假阳性（fused 在 split 之后跑、缓存较热） |
| 窗口 3（64 slots，交错 ×6） | 同上 | ~8.4–8.6 | ~7.9–8.4 | 打平 |
| **窗口 4（定案）** | **清理磁盘后（42Gi 空闲）+ 64 slots 交错 ×6 取中位** | **r3 13.0 / r4 15.9** | **r3 8.3 / r4 8.5** | **split 每轮全胜，r3 +57% / r4 +88%** |

- **最终结论：fused 是明确回归，split（原生产路径）保持默认**。窗口 2 的「+25~51%」是页缓存污染假象——split 在每轮先跑、冷读专家文件，fused 后跑时缓存已热。清理磁盘后同配置下 split 稳定 11–16 tok/s、fused 恒 8.3~8.5。
- 机制：early-split 把 shared GPU 时间藏在 CPU 的 router-readback/planning 窗口下；融合把它串回 cb1 的同步等待关键路径上。
- 保留 `TURBO_FIELDFARE_FUSE_SHARED=1` opt-in 供未来硬件复测；两个路径输出逐 token 一致（greedy temp 0）。
- **附带收益：磁盘清理（42Gi 空闲）本身让 r4 从 ~9 提升到 ~16 tok/s**——27G base 模型终于能整进页缓存，这比 B1 融合的收益大一个数量级。

### 7.3 B2：head 是带宽瓶颈，接近硬件地板（不优先）

- fused greedy head 与 encodeTgx logits 路径的 head GPU 均为 ~9.9ms/token（96 tok → 0.95s）。
- 原因：LM head = 262144×2816 int4 ≈ **415MB/token 读取**，实测 ~41GB/s（~10ms）。这是 GEMV 的固有读取量，任何 kernel 都无法避免；优化空间仅 ~2–4ms（把效率从 65% 提到 85%），不是第一杠杆。
- 融合 head+embed 的 GPU 侧 token 链也救不了——head 在 GPU 关键路径上，CPU 往返不是瓶颈。

### 7.4 真正第一杠杆仍是 expert cache slot 数（已在生产配置中启用）

| slots/layer | decode 命中率 | 实际读盘 | 表现 |
|---|---|---|---|
| 16（默认） | 62.7% | 18.03GiB / 64 tok | IO-bound |
| 64（生产，`TURBO_FIELDFARE_EXPERT_SLOTS=64`） | 95.7% | 9.91GiB / 128 tok | io=1.35s(8%) |

- 生产配置（64 slots + prefetch + lookahead）下：GPU busy 54%、cb1Wait 9.35s（含 ~6s 非 GPU 的 submit/schedule/wake 往返）、CB overhead 1.83s（147μs/cb × 12,416）。
- 提示：`--messages-file`/`--prompt` 不同会显著改变 routing 与命中率，A/B 必须固定同一 prompt。

### 7.5 下一步建议（按 ROI，需空闲机器复测）

1. 在空闲机器上重跑 B1 fused vs split（窗口 2 显示 +25~51%，值得定案）。
2. B5 MTP 重测（r3 compute-bound 条件已具备；MTP draft 命中当前 ~3–14%，文档 68% 不可复现，需先查 MTP 接受率回归）。
3. B4 pipeline 加深（CPU encode 提前 1 层），吃 cb1Wait 里的调度往返。
