# oMLX verify 算子对我们的 MTP verify：**没用，而且是负的**

**Date** 2026-09-24 16:1x · **零 GPU**（静态核对既有代码 ＋ 重算既有 log ＋ 读 oMLX 原档）·
对象：`/Users/alexchuang/WorkBuddy/2026-09-16-16-48-23/Verify-Shape_技術白皮書.html`
＋ 已在本工作树的 `ggml-metal-context.h/.m` 新 API
＋ **原始版源码** `/Users/alexchuang/WorkBuddy/2026-09-16-16-48-23/reference/omlx-bonsai/{spec_decode.metal, bonsai_kernels.cpp}`
· **本线＝線A（静态分析），不建置、不改 code**

## 0. One line

白皮书说的那个「瓶颈」不是一个瓶颈，`spec_decode_verify` 能替换掉的那一行是**一次 int32 比较**；
而把它搬上 GPU 要付**一个阻塞式的 Metal command buffer 往返**，比它省下的多一到两个数量级。

读完原档之后再加一条更硬的：**原始版那个 kernel 只支持 greedy**（`bonsai_kernels.cpp:805-809`
要求「argmax target logits before calling」），我们跑的是 temp 0.8 采样 ⇒ **连语义上也不适用**。

> **⚠ 建议：不要把这个 API 接进 MTP 流程。它已经写好了（+106 行、编译通过），
> 但建议 revert 而不是保留，理由见 §6。**

---

## 1. 这个算子实际做的事

`ggml_metal_spec_decode_verify()` 的主体是 `thread_position_in_grid = b` 的一个 thread，
从 `draft[b*K+j] != target[b*(K+1)+j]` 里找第一个不匹配位置，写回 `n_accepted[b]`、
`committed[b*(K+1)+j]`（原档见 `reference/omlx-bonsai/spec_decode.metal:26-41`，我们移植的版本逐行相同）。

**它的全部工作量 = B×K 次 int32 比较。**

我们的交付形状：llama-bench 单序列、`ntok=4`（K=3）⇒ **B=1、K=3 ⇒ 整个 kernel 做 3 次整数比较**，
dispatch 出去 **1 个 thread**（`dispatchThreads:MTLSizeMake(B,1,1)`）。

即便在 server 上放大到 `n_parallel=64`，也就是 192 次整数比较 ——
**永远比一次 kernel launch 便宜**（这是结构性事实，不是口语上的「有限」）。

---

## 2. 逐条核对白皮书的四个论断

| # | 白皮书论断 | 核对结果 | 证据 |
|---|---|---|---|
| 1 | 「llama.cpp 的 MTP verify 在 CPU 侧逐個比較 draft vs target」 | **半对，且错在关键处** | 比较确实在 CPU（见下），但**它在一个 loop 里，loop 的代价不是比较** |
| 2 | 「O(B×K) 的 int32 比较成为 CPU↔GPU 同步瓶颈」 | ❌ **错** | 见 §3、§4 |
| 3 | 接入点是 `sampling.cpp:803-846` | ✅ **对** | `common_sampler_sample_and_accept_n`，正是 accept 判定所在 |
| 4 | 「真正的机会窗口在 QMV batch 化，需要从量化权重布局开始设计」 | ❌ **错：llama.cpp 本来就有** | 见 §5 |
| 5 | 「`spec_decode.metal` 无 MLX 依赖，纯 Metal Shading Language」 | ❌ **错** | 原档第 5 行 `#include "mlx/backend/metal/kernels/utils.h"` |
| 6 | 「`bonsai_kernels.cpp` 裡硬編碼 `if (B != 1) throw`」 | ✅ **对，且我核到了** | `bonsai_kernels.cpp:257-262`（fast）与 `:315-320`（wide） |

### 2.1 论断 1 的错处（`common/sampling.cpp:803-846`）

```cpp
815: for (; i < draft.size(); i++) {
816:     llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);   // ← 真正的开销
823:     accepted = (draft[i] == id);                                                 // ← 只有这行是 kernel 能替的
828:     common_sampler_accept(gsmpl, id, true);                                      // ← 改动 sampler 状态
832:     if (!accepted) break;
835: }
```

**这个 loop 里贵的是 `common_sampler_sample()`（一次完整采样：penalty／grammar／temp／top-k／top-p
／min-p 跑过 V≈151k 的词表），不是 `draft[i] == id`。** 而那次采样会被接下来的
`common_sampler_accept()` 写进 sampler 状态（grammar stack、penalty 历史、RNG），
**下一次迭代读的就是这个状态** —— 所以就算想把比较抽出来，采样本身仍然必须一次一次跑。

---

## 3. 量到的成本（这才是决定性的一段）

`CGC-MTP-PERF`（`speculative.cpp:3005-3021`）在 2026-09-18 的生产 cell 上实测
（`docs/MTP_IN_LLAMA_BENCH_2026-09-18.md:84-88`）：

```
calls_begin=3 calls_draft=118 calls_accept=118
t_begin_ms=0.6  t_draft_ms=3482.9  t_accept_ms=5.5
ms_per_round=29.516   emit_tok_per_round=3.314
```

⚠ **口径（重要，别读错）**：`t_accept_ms` 是**全程累计**（`t_accept_us/1000.0`，
`speculative.cpp:3020`），不是 per-round —— 只有 `ms_per_round` 除了 `calls_draft`。

⇒ **per-round accept bookkeeping = 5.5 ms / 118 = 46.6 µs/round**。

对齐分母（`docs/ROUND_REMAINDER_IDENTIFIED_2026-09-23.md` §2 的 step 表，k3 三 rep）：

| 量 | 值 | 来源 |
|---|---|---|
| 单 verify step 中位 | **178.0 / 187.1 / 180.5 ms** | 实测（同上 §2） |
| steps/round（k3） | **2.25 / 2.28 / 2.30** | 实测（同上 §2） |
| draft head | 29.5 ms/round | 实测（`ms_per_round`） |
| **accept bookkeeping** | **0.0466 ms/round** | 实测 ÷118 |
| ⇒ 占一个 round | **≈ 0.012%** | 派生 |

即便把 `sampling.cpp` 那个 loop 的**全部**（4 次完整采样＋比较）都算进来 —— 这个 loop **没有被任何
仪器量过**，我没有实测数，不敢报 —— 粗算 ≤4×~10⁻⁴ s ≈ ≤0.4 ms/round ⇒ **≤0.1%**。
而 3% 是本线的改动门槛（`.workbuddy/memory/MEMORY.md` 交付 decode 节）。

⇒ **就算把它砍成 0，也在噪声底下两个数量级。**

---

## 4. 为什么搬上 GPU 是**负**的（不是「收益有限」，是净亏）

白皮书自己给的 `#6/#7/#8` 步就是答案：`[queue commandBuffer]` → `endEncoding` → `commit` →
**`waitUntilCompleted`**（`ggml-metal-context.m` 实现，diff +92 行）。这是一次**阻塞式** GPU 往返。

而本线在 ρ 路线上已经量过「**每步插 GPU 节点 ≈ 4.76 ms**」（`docs/RHO_STAGE0_RESULT_2026-09-23.md`
／`MEMORY.md` ★prebind／ρ 节）—— **那是这笔交易的价钱：
用它去替换 0.0466 ms 的 CPU 工作，是 ~100× 的净亏。**

还有个额外的坑：`draft` / `target` 的 token id 现在住在 CPU 的 `std::vector` 里
（`llama-bench.cpp:3041` 的 `draft`、`sample_and_accept_n` 的 `result`）。
要喂给这个 kernel 得**先上传一次**（又一次 CPU→GPU ＋ 同步），
然后再把 `n_accepted`/`committed` 读回来（又一次同步）。三次阻塞往返抢 46.6 µs。**不可能回本。**

### 4.1 「消除 CPU↔GPU 同步开销」这个收益项本身不成立

白皮书列出的收益只有这一条，而它在 llama.cpp 里是不存在的钱：

- 同步点在 accept loop **之前**就已经付过了 —— sampling 要读 `llama_get_logits_ith()`，
  这一步就把该 step 的 GPU 工作逼到完成。
- 于是 accept loop 跑起来时，**该 step 的 GPU 工作已经完成**，`draft` 是 CPU 上的小 vector。
  `draft[i] == id` 这一次比较**不引入任何同步** —— 没有同步可以消除。

⇒ 这不是「收益很小」，是**收益项为零，成本项为正**。

---

## 5. 白皮书最大的误判：「QMV batch 化」不是机会窗口，它本来就是这个形状

白皮书 §4 说：

> 「oMLX 的 QMV 部分也是 B=1 逐個循環的，和 llama.cpp 當前實作完全一樣」
> 「真正的機會窗口在 QMV batch 化……需要從量化權重布局開始設計」

**前半句的后半段是错的。** llama.cpp 的 verify **一次 `llama_decode()` 就把 1+K 个位置全部带上**，
`[n_draft]` 维度是**展开进 token 维度**并行算掉的：

- `llama-bench.cpp:3003` `llama_decode(ctx, batch)` —— batch 已含全部 draft token，**只调一次**；
  accept loop 在它**之后**（`:3041`）。
- `common/speculative.cpp:2596-2598` `per_seq = 1 + n_draft`（形状就是这样算的）。
- 实测：K=3 ⇒ `CGC-DECPROF` 的 ntok 出现 **4**（=1+3，`ROUND_REMAINDER_2026-09-23.md` §2）；
  K=7 ⇒ Metal kernel 看到 `ne11=8`（`MTP_Verify_Analysis.md` §2.2）。

⇒ **白皮书建议「research from scratch」的那件事，graph builder 已经在做了。**
（它的 `[B,K]` 里的 `B` 在我们这儿是 1，`K` 早就被摊进 token 维度了。）
而这一侧我们还有 oMLX 没有的东西：**CGC 的 fused `MUL_MAT_ID_GLU` MoE kernel**。

顺带，`ROUND_REMAINDER` 那个 2.25 steps/round 不是 per-token forward ——
那是 **partial acceptance 之后的重验**（`llama-bench.cpp:3052-3066` 的 `continue` 恢复 checkpoint
再验一次），不是「逐 token 循环」。那里确实可能有真实的空间可挖，
但（a）它 mirror 了 `speculative-simple.cpp:570-593` 与 `server-context.cpp:4186-4221`，是刻意的；
（b）**这个 kernel 一样帮不上**：`n_accepted` 要等 target forward 出来才知道，不存在「提前知道」。

---

## 6. 建议：revert，别保留

已经落地的 `+106 行`（`ggml-metal-context.h` +13、`.m` +92、`ggml-metal.metal` +1）目前**没有呼点**，
白白的风险：

1. **一个共用面上的污染**：`ggml-metal-context.*` 是 `libggml-base` 的**共用面**
   （`MEMORY.md` 分工節的碰撞面 ②）。为一个不该接的路由制造 diff ⇒ 平白生出 merge 冲突面。
2. **`static id<MTLComputePipelineState> cached_pipeline` 没有锁、没有 dev 粒度**：
   多个 backend context／多 device 下会串，且 autoreleasepool 出来后 `cached_pipeline` 是悬空引用
   （它是在 pool 里 assign 的，static 不会延长寿命）。
3. 它走了一条「绕过 `.metal` 模板系统、run-time `newLibraryWithSource` 编译」的
   **shadow path** —— 这条路本身没错，但不该由一支注定要删的 kernel 来开先例。

⇒ **revert 这三个档。**
（如果想留着「内联编译通路」当以后的参考 ⇒ 请另开一格、单独的一次性 commit 加 why 注释，
不要挂在 `spec_decode_verify` 名下，否则以后的人会以为这条路径是给 spec verify 用的。）

---

## 7. 那 GPU 在这里到底有没有戏？——有，但对象不是 verifier，是 **sampler**

把结论翻正讲一句：

> 唯一在这个位置上量级对得起一次 GPU 往返的，是 `common_sampler_sample()` 的线性词表部分
> （V≈151k 的 penalty／softmax／top-k），不是 accept/reject 的比较。

- 现在它每 round 跑 ≤4 次、串行、在 CPU 上，**且带状态依赖**（要先把本次的 token 喂进 sampler
  状态，才能采下一个位置）。
- 想「GPU batch 采样」就必须先把那个状态依赖拆掉（改成每个 position 可独立求值），
  那**不再是移植 oMLX 的算子，是自己设计一条新的路径**，
  并且**会改变输出分布** ⇒ 需要 bit-identical 判据；而我们的 accept 在 rejection 模式下
  还会改写 `id`（`sampling.cpp:821` `cgc_rejection_accept`）
  ⇒ **连「一次采完再比」都不是 trivially 可 GPU 化的**。

结论：**这件事值得开一格去判断题的量级，但它不是一个「搬算子」的工程；它起始于「sampler 该不该上 GPU」这个决策。**
在那之前，这个 kernel 不该接。

---

## 8. 原始版 oMLX 对我们有加分吗？（读了原档之后）

把 `reference/omlx-bonsai/` 两支原档读完，三个组件逐一定：

### 8.1 `spec_decode.metal`（1290 B，就是那个 L2）

**没有加分，而且比我们的移植版更不适用。** 三条硬缺陷，全部在原档里：

1. **只支持 greedy。** 原档第 9 行注释「Greedy speculative-decoding verify」，
   且 `bonsai_kernels.cpp:805-809` 明确要求 target 必须是 argmax 出来的 token id ——
   原文 "draft and target must be int32 token ids (**argmax target logits before calling**)"。
   **我们的 production 跑 temp 0.8／top-k 40／top-p 0.95**
   （`docs/MTP_IN_LLAMA_BENCH_2026-09-18.md:100-102`）⇒ 这个 kernel 表达的 accept rule 和我们用的**不是同一个**。
2. **不是 oMLX 自研的。** 抬头是 `Copyright © 2024 Apple Inc.`、
   `Originally from the Bonsai MLX fork (github.com/PrismML-Eng/Bonsai-demo)`（`:1-3`）；
   上游 oMLX README 也明写 verify-shape Metal kernels 来自第三方 **MTPLX**。
   ⇒ 它是一份 demo fork 的 greedy 工具函数，不是「Lightning MTP 的核心技术」。
3. **依赖 MLX。** `:5` `#include "mlx/backend/metal/kernels/utils.h"` ——
   白皮书说的「无 MLX 依赖，纯 MSL」是错的。

### 8.2 `bonsai_kernels.cpp` 的 QMV（837 行）

**没有加分，且它明确拒绝 batch：**

```cpp
// bonsai_kernels.cpp:257-262（fast）／:315-320（wide），两处逐字相同
int B = static_cast<int>(out.size()) / M / N;
if (B != 1) {
    throw std::runtime_error("[bonsai] batched qmv dispatch is not supported (B=" + ...);
}
bool batched = false;
```

`batched` 在这两支 dispatch 里是**写死的 false**，`:263`/`:321` 之后直接进 kernel 选择。
⇒ 白皮书说的「B=1 only，硬编码 throw」**完全属实**。

至于「那 QMV 本身的 tuning 值不值得学」—— 它们是绑在自己那套 packed 量化格式上的
（affine/symmetric × group_size × bits × fast/wide 四种组合 + 5-way row split）。

我们这边同位置上是：量化 MV/ACCESS 全套 ＋ small-batch MM路径（`ne11∈[2,8]`，
见 `MTP_Verify_Analysis.md` §3.2.1）＋ **fused `MUL_MAT_ID_GLU` MoE kernel**
（`ggml-metal.metal:11155+`，oMLX 没有）。**我们在这一区是领先的那一方，不是追赶的那一方。**

### 8.3 小结

| 组件 | 原始版给不给得起 | 原因 |
|---|---|---|
| `spec_decode.metal`（L2 accept/reject） | ❌ | greedy-only；OG来源 Apple demo fork；MLX 依赖；工作量 ≤192 次整数比较 |
| QMV forward（`bonsai_kernels.cpp`） | ❌ | `if (B != 1) throw` 写死；绑私有量化格式 |
| QMV 的 batch 化（白皮书的「机会窗口」） | ❌ **我们本来就有** | verify 一次 `llama_decode` 带 1+K 位置（§5） |

⇒ **原始版没有加分。** 唯一「加分」的是**负的知识**：它证明了「这条路走不通」，
而且这条负结果现在有了源码级证据（`if (B != 1) throw` ＋ argmax-only），以后不必再问一次。

⚠ **诚实边界**：我只拿到这两支档（正是白皮书说重要的那两支），没有整棵 oMLX tree
（没有 `bonsai_quantized.metal`、没有 Python 侧怎么驱动）。上面的每条结论都引到了行号，
可以逐条复核；但如果有人要主张「整包 oMLX 有价值」，那需要另外评 Python 侧与
`bonsai_quantized.metal` 那支 —— 本轮**没有做**那个评估。

---

## 9. 证据链（我怎么知道的）

- **`git diff` / `grep`（本 repo）**：`common/speculative.cpp:2879-2905, 2960-3022`、
  `common/sampling.cpp:799-846`、`tools/llama-bench/llama-bench.cpp:3000-3092`、
  `ggml/src/ggml-metal/ggml-metal-context.h/.m`，全部读过。
- **读 oMLX 原档**：`reference/omlx-bonsai/spec_decode.metal`（整支 43 行）、
  `bonsai_kernels.cpp` 的 `:241-354`（两支 dispatch）与 `:797-835`（verify 封装）。
- **上游交叉验证**：oMLX README 把 verify-shape kernels 归给 MTPLX（第三方）。
- **没有建置、没有起 GPU、没有跑任何量测**；所有触发用的量都来自既有 log／既有档／原档，**zero GPU**。

## 10. 数字口径

- **MEASURED**：`t_accept_ms=5.5 / 118 rounds`、`step 中位 178-187 ms`、`ntok=4 for k3`、`ne11=8 for K=7`；
  原档 `if (B != 1) throw`、`argmax target logits before calling`。
- **DERIVED**：`0.012%`、`≤0.4 ms/round 上界`、`~100× 净亏`。
- **未闭合**：sampling loop 的实测（需要一个 round 级仪器把它挤出黑箱）。
- **不接受**把我的 ≤0.4 ms 粗算当成实测数字引用。
