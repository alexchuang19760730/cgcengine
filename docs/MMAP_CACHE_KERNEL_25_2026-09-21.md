# 「mmap ＋ expert cache ＋ revised kernel」到不到 25 t/s？（2026-09-21）

> **结论：有机会，但机会不在你以为的地方，而且「revised kernel」这一项已经写过、量过、结果是 ≈0 或更慢。**
> 25 的成败最后压在**一个还没钉死的数字**上（真实 GPU busy，见 §5）。
> 我给的机率：**P ≈ 10–20%**，而且**必须先花 0 重建把那个数字钉死**，否则写 kernel 又是盲试。

判定依据全部来自既有文档与读码，**本轮 0 重建、0 写 C++**。

---

## 1. 先把 25 换成硬条件

在一个自洽的 cell 里（`MEMORY_PERF` §L3 定锚）：

| 量 | 值 |
|---|---|
| 现在 step | **247.98 ms**（13.61 t/s @ mean_len 3.375） |
| 其中 **GPU busy** | **181.56 ms**（73.2%） |
| 其中 主机侧（IO/gap/submit/wait） | **66.42 ms**（26.8%） |
| 主机侧全归零 ⇒ | **18.59 t/s** |
| **25 t/s @ ml 3.375 需要** | **step ≤ 135 ms** |

⇒ **即使主机侧全部归零（18.59），离 25 还差 25.6%。**
⇒ **25 必须压 GPU busy 本身。** 这是唯一一条路，也立刻判定：

- **mmap 与 expert cache 都只作用在主机侧的 66.42 ms** ⇒ **两个加起来封顶 18.59 < 25**。
- **只有 revised kernel 碰得到 GPU busy。**

---

## 2. mmap：封顶不够，而且在现在的代码上是坏的

- **硬障碍（已踩过）**：`--load-mode mmap` **与 expert cache 不相容** —— L4 pool 是
  read-only file-backed 映射，zeroing 写只读页 ⇒ **`SIGBUS` in `__bzero`**。缺陷已修，
  但**任何量过的宽度仍是 0/5** ⇒ `MEMORY_FACTS` 明写：**不是 OOM 的杠杆，不要再试。**
- **就算修好，它也不够**：mmap 只是把 `pread` 换成 page fault，**不减少字节**。
  它最多把主机侧的 fill 时间压下去 ⇒ 仍是 18.59 封顶。
- **额外风险**：12.72 GiB 档 vs 16 GB RAM，池实配 ≈2.5 GiB ⇒ mmap 常驻会挤压内存、
  换来 swap 风险；而且 16 KiB 页粒度下，一个 expert（1.0703 MiB）= **67 次 fault**，
  比现在的 ~220 KiB/次读**碎得多**（要 `MAP_PREFETCH`／大页才不亏）。

**判决：不值得做。** 它给不了 25，而成本（SIGBUS + 内存压力）是实的。

---

## 3. expert cache：必要但不充分，而且已调到头

- 已存在、已调过：唯一零风险旋钮 `WORKERS=8→2` 机制确证（µs/miss **0.902，5/5 同向，p=0.031**），
  但 `pool_wait` 只占步时 16.7% ⇒ **端到端 ≈ +1.7%**，低于 3% 门槛。
- 把池加大 ⇒ miss 变少 ⇒ 少的是**主机侧**的 `cb` ⇒ 同样封顶 18.59。

**判决：留着，但它送不到 25。**

---

## 4. ★ revised kernel：唯一够得着 25 的一项 —— 但已经做过，结果是 0

这条最该知道：fork 里**已经有两个自制的融合 MoE kernel，都默认关闭**：

| 开关 | 做了什么 | 量过的结果 |
|---|---|---|
| `CGC_MMV_FUSE=1` | gate+up 的 `MUL_MAT_ID` + SwiGLU 融合 | 干净窗口 p50 Δ = **−0.59 ms**（≈ −0.2%，等于没有） |
| `CGC_DOWN_COMBINE=1` | down projection combine | **23.29 vs baseline 25.87 t/s ⇒ 更慢**；且硬绑 `GGML_TYPE_Q3_K` 的 down，**本模型不适用** |

- 09-17 的 M3 裁决写得很直白：**M3「仍然没有」**，两个探针都不可引用；
  而且回头解释了原因：**「`CGC_MMV_FUSE` 为什么失败：它在没有判准的情况下实作的（盲试）。」**
- 旁证：`CGC_MMV_FUSE` **不在 `run_server.sh` 的 env allowlist** 里（后来才补 pass-through），
  默认 off ⇒ 从启动器长期无法开启。

**而且「少一遍 weights 往返」那 2–3× 已经拿过了**：MoE 用的是 **`GGML_OP_MUL_MAT_ID`**
（融合的 gather + matmul），不是「先 `get_rows` 再 `mul_mat`」。⇒ 常见的「融合 MoE 省一次
weights 读写」这条捷径**在这份代码里已经用掉了**。

---

## 5. GPU busy 里还剩多少？—— 这是唯一没量过的数

带宽地板（`CEILING_STACK` L4）：**1.34 GB/token @ ~100 GB/s = 13.4 ms/token ⇒ ~75 t/s**。

换算今天的位置：

```
GPU busy        = 181.56 ms / 3.375 token = 53.80 ms/token
有效带宽         = 1.34 GB / 53.80 ms     = ~24.9 GB/s  ⇒ 峰值的 25%
25 t/s 需要      = 40 ms/token            ⇒ 1.34 GB / 40 ms = 33.5 GB/s ⇒ 峰值的 34%
⇒ 需要把权重流效率 ×1.345（GPU busy −25.6%）
```

**25% → 34% 听起来不多，但要看清对手是谁**：

- GPU busy 里最大的一块是**真矩阵乘**：MUL_MAT **254 dispatch/步、占步时 23.1%**（G4 §4.4）。
- **不是 launch 开销**：一个 dispatch 固定成本 18.6–19.5 µs × 102 elementwise = **1.9 ms**，
  只占 GPU busy 的 **~1%**。⇒ 再怎么减 dispatch 也拿不到 25。
- **也不是「elementwise 太小」**：42.2% 的 elementwise dispatch ≤1024 元素是事实（M_K5_SHAPE），
  但撑 grid 动不了那 18 µs 的固定开销。

⇒ **要拿到那 1.345×，得动的是 MoE matmul 的权重流效率本身**（tile／gather 局部性／dequant 开销），
**而这一项今天没有量过**。

### ⚠️ 一个会改变结论的口径问题（必须先钉死）

`MEMORY_PERF` §L3 有一条注记：**`gpu_busy_sum 181.56` 段内 buffer 并发会被算两次（比值 1.22）**。

| 若 | 真实 GPU busy | 主机侧归零的天花板 | 25 还需压 |
|---|---:|---:|---:|
| 181.56 正确 | 181.56 ms | **18.59 t/s** | **−25.6%** |
| 按 1.22 修正 | ≈148.8 ms | **22.7 t/s** | **−9.2%** |

**这是同数量级地改变结论。** −25.6% 是硬仗；−9.2% 就相当 plausible。
⇒ **在钉死它之前，不要写 kernel。**

---

## 6. 判决

| 项 | 能不能碰 GPU busy | 封顶 | 判决 |
|---|---|---|---|
| mmap | 否（主机侧） | 18.59 | ❌ 不做；且当前代码上 SIGBUS、0/5 |
| expert cache | 否（主机侧） | 18.59 | ⚠ 留着，+1.7%，送不到 25 |
| **revised kernel** | **是** | 带宽地板 75 | ⚠ **唯一够得着的，但已盲试一轮＝0／更慢** |

**给这个组合的机率：P ≈ 10–20%。**

- 支撑给分：25% → 34% 只差 1.345×，物理上不荒谬；且 1.22 修正若成立，需求降到 −9.2%。
- 压分：① G2 逐位元等价会挡掉大部分「重排／K-split」手段；② 上一轮盲试已失败一次
  （MMV_FUSE −0.59 ms、DOWN_COMBINE 更慢）；③ MoE 已经是融合的 `MUL_MAT_ID`，
  最诱人的那条捷径已用掉。

---

## 7. 若真要做，正确顺序（便宜到贵）

1. **钉死真实 GPU busy**（0 重建）：用现有 `CGC_GPU_TIMING`／`CGC-NSM` 判明
   「段内 buffer 并发算两次」的 1.22 到底成不成立 ⇒ 决定需求是 −25.6% 还是 −9.2%。
2. **量有效带宽**（0 重建）：GPU busy 期间 MUL_MAT 的实测 GB/s。若确实只有 ~25 GB/s，
   再定位是 tile 太小／gather 分散／dequant 开销。
3. **先写判准，再写 kernel**（09-17 的血泪：MMV_FUSE 是没判准盲试的）。
   判准建议：目标「有效带宽 25 → 34 GB/s」，过不了就停。
4. 主机侧那 66.42 ms 顺手拿（`WORKERS=2`），但它不是通往 25 的路。

**别再试**：`--load-mode mmap`（SIGBUS 0/5）／减 dispatch 数（只占 1%）／
再开 `CGC_MMV_FUSE`／`CGC_DOWN_COMBINE`（已量：0／更慢，且后者绑 Q3_K 不适用）。

---

## 8. 引用

- `docs/CEILING_STACK_2026-09-21.md`（L0–L4 分层、gpu_sum 181.56、L4 带宽地板）
- `docs/G4_FUSION_ANSWER_2026-09-20.md`（MUL_MAT 254 dispatch/步 23.1%；「杠杆在单次 dispatch 时长」）
- `docs/M_K5_SHAPE_2026-09-21.md`（dispatch 固定成本 18.6 vs 19.5 µs；42.2% ≤1024 元素）
- `.workbuddy/memory/MEMORY_FACTS.md`（mmap 不相容、0/5）
- `.workbuddy/memory/2026-09-15.md`（MMV_FUSE −0.59 ms；DOWN_COMBINE 23.29 vs 25.87）
- `docs/M3_VERDICT_2026-09-17.md`、`.workbuddy/memory/2026-09-17.md`（M3 判死、「盲试」教训）
- 读码：`src/llama.cpp/src/models/qwen35moe.cpp`、`ggml/src/ggml-metal/ggml-metal-ops.cpp:420,2800,4904`
