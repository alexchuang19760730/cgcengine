# 第 3 步开工闸门：稳态 miss 率到底是多少（2026-09-24 22:3x）

上游：`docs/NEXT_STEP_MISS_HANDLER_2026-09-24.md`（§0 定价）、`docs/MISS_MASK_STEP2_2026-09-24.md`（第 2 步出口）。
工具：`scripts/check/miss_rate_series.py`（`--self-test` 15/15）。

---

## 0. 结论先给：**闸门 OPEN，而且不必等 fill 后台化**

上一轮我把闸门写成「量『单段提交 + 背景 fill』臂的稳态 miss 率，<10% 才开工」。
**这个闸门本身是不可执行的** —— 那一臂不存在：单段提交把整个 hook 拿掉了，而
fill（含 `ensure_batch` 的同步填、以及 `CGC_LAYER_AHEAD_PREFETCH` 的后台预取）
**全部住在 hook 里**。要跑那一臂，就得先把「背景 fill」做出来，而那正是闸门要决定是否先做的东西。
⇒ 循环依赖。

**换成可执行的判据：算损益平衡点，看实测区间落在哪一侧。**

| | miss 率 | 来源 |
|---|---:|---|
| 平衡点（悲观：34.5 ms 省下 / 0.169 ms 每次補算） | **66.1%** | 计算，§3 |
| 平衡点（乐观：44.5 ms / 0.126 ms） | **114.3%** | 计算，§3 |
| **实测上界**（零 fill，最坏） | **43.0%** | 本轮 B 臂 256 步 |
| **实测下界**（同步 fill 稳态） | **4.7%** | 本轮 A 臂 SHAPE final |

⇒ 整个可能区间 **4.7% ~ 43.0% 全部在平衡点 66% 以下**，最坏情形仍有 **1.54×** 边际。
**per-expert 重算在 0 ~ 66% 的任一 miss 率下都是净正的**，所以：

- ✅ **第 3 步可以开工**，不必先做 fill 后台化。
- ⚠ 但 **fill 是「正确性」前提，不是「速度」前提**：没有 fill，缺席 expert 的权重永远不出现，
  输出永远是 garbage。所以「背景 fill」不是可选项，只是**它决定拿 14.1 还是 19.9，不决定正负**。
  修正 `NEXT_STEP_MISS_HANDLER` §4 里「fill 后台化是锦上添花」与 §6 里「必须先做」两种说法：
  **正确性上必须做，速度上是第二顺位。**

---

## 1. 实测一：单段提交 + 零 fill 的稳态 = **43.0%，平的**

臂 = `--mode identity` 的 B 臂（`CGC_SEG_BATCH=1 CGC_B_SCHEME=1 CGC_SLOT_TABLE_GPU=1`），
`--gen 256 --reps 1 --ctx-size 0`，带 `CGC_MISS_MASK=1 CGC_MISS_MASK_DBG=1`。

```
257 步，warmup 1，切 5 段：
  bin0 51 步  miss 133.65/步  rate 0.4320
  bin1 51 步  miss 133.71/步  rate 0.4339
  bin2 51 步  miss 133.04/步  rate 0.4318
  bin3 51 步  miss 131.59/步  rate 0.4276
  bin4 52 步  miss 132.75/步  rate 0.4298
  decay_rel = 0.005  ->  FLAT
```

**两个视图互证「一步都没填」**：

| 视角 | 计数器 | 值 |
|---|---|---|
| 设备端（占位） | `MISSMASK` | 132.75 次/步，占 308.9 请求的 **43.0%** |
| host 端（fill） | `CGC-SHAPE phase=final` | `misses=0 read_mib=0.0 hit_pct=100.00` |

SHAPE 的 `misses` 是 **fill 视角**（只统计 `ensure_batch` 真正去填的次数），mask 是**占位视角**。
两者不矛盾：B 臂一次都没填（0 MiB），所以占位永远补不掉。

⚠ 修正上一轮的口播值：我在第 2 步结尾说「≈62%，39 层 × 5 ≈ 195 次/步」，那是 **gen 8 的前几步**。
256 步的稳态是 **43.0% / 132.75 次/步**。62% 那个数不要再引用。

---

## 2. 实测二：同步 fill 的稳态 = **4.7%**

A 臂同 cell 的 `CGC-SHAPE v=1 phase=final`：

```
req=87960 hits=83837 misses=4123 hit_pct=95.31 compulsory=3715 capacity=408 evict=4123
read_mib=1674.6  pread_us=1500567793  fill_wait_us=3398701
n_layer=40  n_expert=256  pool_cap_slots=143  slots_layer=143
```

- 4123 / ~275 步 = **15.0 次/步**（4.7%）。与既有干净臂（gen 128 / warm-skip 64）的
  `hit 93.1%`（21.9 次/步）同一量级。
- `capacity=408` 只占 miss 的 9.9% ⇒ **池容量不是瓶颈**（143/256 = 55.9% 的 expert 可常驻，够用）。
- 每次 fill 搬 **1674.6 MiB / 4123 = 0.406 MiB**。

---

## 3. 损益平衡（第 3 步能不能赚钱的唯一判据）

输入全部是实测值：

- `per-expert 补算` = **0.126 ~ 0.169 ms**（`miss_reprice.py`：step 级 `gpu_union` 67.42 ms ÷ 40 层
  = 1.685 ms/层 × MoE 占比 0.6~0.8 ÷ k=8）。
- `省下` = A→B 的 **34.5 ms/step**（12.05 → 20.66 t/s 口径）或 **44.5 ms/step**（本轮 11.54 → 23.68）。

| 省下 / 单次補算 | 平衡点（次/步） | 平衡点（miss 率） |
|---|---:|---:|
| 34.5 ms / 0.169 ms | 204.1 | **66.1%** |
| 34.5 ms / 0.126 ms | 273.8 | 88.6% |
| 44.5 ms / 0.169 ms | 263.3 | 85.2% |
| 44.5 ms / 0.126 ms | 353.2 | 114.3% |

两端点的净收益：

| 情境 | 補算成本 | 净收益 | 落点 t/s |
|---|---:|---:|---:|
| 零 fill（132.75 次/步，0.169） | 22.43 ms | **+12.1** | **14.1** |
| 零 fill（132.75 次/步，0.126） | 16.73 ms | +17.8 | 15.4 |
| 稳态 fill（15 次/步，0.169） | 2.54 ms | **+32.0** | **19.6** |
| 稳态 fill（15 次/步，0.126） | 1.89 ms | +32.6 | 19.9 |

⇒ **14.1 t/s（最坏）到 19.9 t/s（最好）**，全部高于交付 anchor 12.57 与干净臂 12.05。
⚠ 仍是推算：`省下 34.5/44.5 ms` 的**内部分解没有独立量测**（见 §4），只当指引不要当交付值。

---

## 4. 供给侧为什么量不出来（别拿 `fill_wait_us` 去算）

我很想给「背景 fill 供不供得上」一个数，但三个候选统计都不合格：

| 统计 | 为什么不能用 |
|---|---|
| `fill_wait_us` | 有**两个**累加点（`llama-expert-cache.cpp:998` 的 prefetch 等待、`:1102` 的**单 expert** `fill_pool_direct`）。decode 走的是 `ensure_batch` → `fill_segments_pool`，**这一条不累加**。3.40 s 基本来自 prefill，不能除以 decode 步数。 |
| `pread_usec` | 跨 worker **累加**，可以超过 wall clock（代码注释自己写了「553 s inside a 92.6 s wall」）。本轮 1500.6 s / 56 s wall = 26.8× > 8 workers ⇒ 归因不明，不能用。 |
| `read_mib` | 只给总量，不给时间。 |

⇒ 想要 decode 段的填池吞吐，得**新加一个只包住 `fill_segments_pool` 的计时器**（挂在
`ensure_batch` 里、按 step 累计）。这是本轮唯一没闭的口子，**不阻塞第 3 步开工**（§0 已判），
但它决定 §3 表格里落在 14.1 还是 19.9，值得在第 3 步做完之前补上。

---

## 5. 用到的开关与参数（复现用）

```sh
export LLAMA_EXPERT_CACHE_ALLOW_NGL=1 CGC_EXPERT_SKIP_READRAW=1 \
       CGC_MISS_MASK=1 CGC_MISS_MASK_DBG=1 LLAMA_EXPERT_CACHE_BATCH_DBG=1
python3 scripts/check/seg_batch_abba.py --mode identity --gen 256 --reps 1 --ctx-size 0 \
       --workdir Backup/mm_gate_identity
python3 scripts/check/miss_rate_series.py --log <B臂stderr.log> --warmup 1 --bins 5
```

⚠ `--ctx-size 8192` 在 llama-bench 载体上必 OOM（本轮与 `miss_axis.py` 同日定谳），必须给 0。

---

## 6. 下一步

1. **第 3 步开工**：per-expert 重算 kernel。保留 `x` 的层数按 step 1 定死的 **18.1 层**
   （不是 8；8 在 256 步数据里一步都没出现，min = 9）。
2. **并行补一个 `fill_segments_pool` 计时器**（§4），把 14.1 vs 19.9 这个区间收窄。
3. 判据仍照旧：greedy 逐 rep 相同 ＋ 落在预测区间内；**低于 13.0 就退回**。
