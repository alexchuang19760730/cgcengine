# Expert cache → Metal 的「shape」第一性原理审计

日期：2026-09-23 22:1x
方法：**不跑 benchmark**。只从 GGUF 几何 ＋ 设备常数 ＋ 已定谳的同类 cell 实测推。
（`docs/METHODOLOGY_RETROSPECTIVE_2026-09-23.md` §3：先第一性原理，再量测。）

问题不是「张量 `ne/nb` 自不自洽」（那是后话），而是**时间形状**：
expert cache 喂给 Metal 的这条数据通道，有没有做到
**CPU(IO) 与 GPU(计算) 完全重叠 ＋ 带宽打满 ＋ 并行**。

---

## 0. 那个唯一的数：1 个 expert slot = 1.0703 MiB

全部结论都从这一个数长出来，所以先把它钉死 —— **它可以从 GGUF 直接读出，不需要任何仪器**。

`/tmp/expert_shape_probe.py` 读 `models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`：

```
blk.0.ffn_down_exps.weight   ne=[512, 2048, 256]  type=21
blk.0.ffn_gate_exps.weight   ne=[2048, 512, 256]  type=22
blk.0.ffn_up_exps.weight     ne=[2048, 512, 256]  type=22
```

每专家的元素数 = 2048×512 = 512×2048 = **1,048,576** ⇒ 4096 个 256-block。

| kind | 类型 | bpw | bytes/block | bytes/expert |
|---|---|---|---|---|
| gate | IQ2_S | 2.5625 | 82 | **335,872** |
| up | IQ2_S | 2.5625 | 82 | **335,872** |
| down | IQ3_S | 3.4375 | 110 | **450,560** |
| **合计（1 slot）** | | | | **1,122,304 B = 1.0703 MiB** |

**独立校验（这一条让上面的推导不是自说自话）：**

```
实测 base_r1:  misses=8446,  读取 = 8.80 GiB
8.80 GiB / 8446 = 1.0669 MiB / miss
--------------------------------------------
推导          1.0703 MiB
闭合到         0.3%
```

这与 `CGC-GATHER-SLAB` 日志里的 `requested=335872 / 450560` 也是同一组数字，三条独立来源互相印证。

**⇒ 1.0703 MiB 是每搬运一个专家的不可压缩成本。它不是策略，是物理。**

---

## 1. 三个轴的判决

### ① 带宽打满 —— ❌ 没达成，而且**它是个伪目标**

```
设备在一个 step 能供：  1638 MiB/s × 247.98 ms = 406 MiB
一个 step 实际需要：    128 MiB（乐观） ~ 320 MiB（悲观，见 §2）
------------------------------------------------------------------
占用率                  32%  ~  79%（不是 "打满"）
```

「带宽打满」在这个 workload 下的字面意思是：**每步硬搬 406 MiB**，而其中大部分没有专家要用。
那是把 SSD 当暖炉烧。**这不是目标，这是 anti-goal。**

顺带纠正一个常见误读：`cb = 42 ms/step` 不是说设备在 42 ms 里闲 42 ms，
是说**这 42 ms 设备是全速的**（42 ms × 1638 MiB/s = 68.8 MiB）。
Duty cycle 是 17%，不是速率不满。两者别混。

### ② CPU / GPU 时间重叠 —— ❌ 没达成，**这才是真目标，而且余量已经在见底**

现在的形状不是「IO 与 GPU 并行」，而是 **41 段 ping-pong**
（既有结论：`docs/STEP_SERIALIZATION_2026-09-23.md` §4）：

```
每层的 ids 在该层开始时才知道
   → CPU 无法提前发起该层的 IO
   → 每层一个同步点
   → GPU 每层等一段（既有量测：gap 44.4 ms/step）
```

而 §2 会给出这件最要紧的事：**即使完全消除这个 ping-pong，物理余量也只有 21%。**

### ③ 并行 —— ❌ 现在是**伪并行**，粒度错了

从今天的 ABBA（`rho_ab_192500`）实跑：

| arm | 搬运字节 | IO 次数 | 平均每次 | `fill_wait_us` |
|---|---|---|---|---|
| base | 8.80 GiB | 39,219 | **0.229 MiB** | 56~75 ms（全程） |
| rho | 3.18 GiB | 79,837 | **0.041 MiB**（41 KiB） | **15~20 s** |
| rholeg | 3.12 GiB | 80,204 | 0.040 MiB | 14.6~15.3 s |

理论最优粒度 = **一个完整专家的三个 kind 一次搬完 = 1.0703 MiB**。

- base 已经碎片化了 **4.7×**（1.0703 / 0.229）
- ρ 把它推到 **26×**（1.0703 / 0.041）

ρ 把字节砍了 64%（这是对的！），但用 **双倍的请求数 + 1/5 的粒度**去买
⇒ 每次 pread 的固定开销（syscall + NVMe command + 完成面）摊不回来
⇒ `fill_wait_us` 从 75 ms 爆到 15~20 s。**在一个只剩 21% 余量的通道里加碎片，必然是这个结果。**

---

## 2. 关键判决：为什么「提前一步预测」被物理否掉

这是本次审计最硬的一条，**不需要任何量测**：

```
一步所需 union（完美预测、零命中的情形）
  = 40 层 × U=19.36 experts/layer  = 774 slots
  = 774 × 1.0703 MiB               = 829 MiB

设备搬完 829 MiB 需要
  = 829 / 1638 MiB/s               = 506 ms
  = 506 / 247.98 ms/step           = 2.04 个 step
```

> **一个 step 里绝对搬不完它自己需要的全部专家。物理上不行。**
>
> `prebind` / `ρ` 的天花板因此**不是预测准不准**，而是这一个比值：
> **需求 829 MiB vs 设备一步供 406 MiB = 2.04×**。
> 完美预测也救不了 —— 它只会把 miss 归零，同时把 union 拉到 829 MiB。

**所以 prebind 类方向的上限，从一开始就被这个 2.04× 锁住了。**
这与先前 `MEMORY.md` 里「prebind 上界 14.33~16.06 t/s」是同一件事的两种看法，
但这一次它是从文件几何推出来的，**不依赖任何覆盖率/接受率模型**。

### 那真正还剩多少余量？

按 **miss 才是真搬运**重算（hit% 61.4% ⇒ miss 38.6%）：

| 情形 | req/step | miss slots | 需搬 | 设备时间 | 占 step |
|---|---|---|---|---|---|
| 乐观（draft-like，每层 top8 去重） | 311 | 120 | 128 MiB | **78 ms** | 32% |
| 悲观（verify step，ntok=4 union） | 774 | 299 | 320 MiB | **195 ms** | **79%** |

⇒ **verify step 已经吃掉 step 的 79%。重叠的余量只有 21%。**
⇒ 这就是为什么「再往这条通道里塞任何东西」都会立刻变现成延迟（ρ 就是证据）。

---

## 3. ★ 命中率也不是被预测决定的 —— 是被「内存能放几个专家」决定的

又一个第一性原理闭合：

```
8 GiB budget / 1.0703 MiB        = 7654 slots  ⇒ 191 slots/layer ⇒ 覆盖率 74.7%
resident 实测 6430.62 MiB        = 6008 slots  ⇒ 150 slots/layer ⇒ 覆盖率 58.7%

实测 hit rate = 61.4%
```

**覆盖率 58.7% vs 命中率 61.4% —— 差 2.7pp。**

⇒ **命中率 ≈ 覆盖率**。它不是预测能力的函数，是「单位内存能放几个专家」的函数。
⇒ 要提高命中率，杠杆在 **`per-expert bytes`** 和 **budget**，不在预测器。

---

## 4. 顺手查清的一个布局问题（**结论：不是 decode 的收益**）

`llama-context.cpp:721` 的注释给出了 mixed-quant 的真相：

> these GGUFs are MIXED QUANT, so one kind carries several per-expert sizes across layers
> (Nail: kind 0/1 are iq2_s|iq3_s|q2_K at **335,872 / 450,560 / 344,064** B,
> kind 2 is iq3_s|iq4_xs at **450,560 / 557,056** B) — 8 distinct (kind,stride) combinations
> inside one 41-layer model.

实测日志：`CGC-GATHER-SLAB: kind=0 cap=256 stride=450560 (requested=335872) size=110.00 MiB`

⇒ the **gather slab**（prefill wide-union 路径，356 MiB）把 per-slot stride 取**全模型跨层最大值**，
对 iq2_s 的层来说 padding **34.1%**，down 的 iq3_s 层 padding **23.6%**。

但两点让它**不构成 decode 收益**：

1. `fill_pool_direct_collect` 里 `seg.bytes = e.bytes`（专家真实大小），**不是** stride
   ⇒ **pread 没有多读字节 ⇒ 带宽没有浪费，浪费的只是这块临时 slab 的容量。**
2. decode 走的 8 GiB expert pool 是 **per-layer stride**（`llama.cpp:522` `total += r.slots * r.stride`），
   **不受这个 round-up 影响**。

⇒ 记账要说清楚，但它不是杠杆。**别把它当成发现的问题去修。**

---

## 5. 所以「最佳的 shape」到底长什么样

按上面的三个闭合重新定义（替换掉「更早更多地预取」这个被判死的方向）：

| 维度 | 现在 | 应该 |
|---|---|---|
| IO 粒度 | 0.229 MiB（碎片 4.7×），ρ 后掉到 0.041 | **1.0703 MiB = 一个专家的 3 个 kind 一次搬完** |
| 并行 | 39,219 次小请求 | 请求数降到experts 数：粒度上去，次数自然下来 |
| 提前量 | 每层等 ids ⇒ 41 段 ping-pong | 以**层块**为单位提前（不是一步全预测 —— 那被 2.04× 判死） |
| 命中率 | 靠覆盖率 58.7% | 靠 **`per-expert bytes`↓ / budget↑**，不是靠预测器 |
| 带宽 | 目标 32~79% 占用 | **别追求打满**；406 MiB/step 是天花板，829 MiB 是单一需求梦 |

**一句话：最优 shape 是「少而大、按层块提前、粒度等于一个专家」，
而不是「多而早、逐 segment、指望预测」。**

---

## 6. 待闭合（诚实交代）

上面标为「推导」的都需要一次同 cell 的量测来钉死，目前它们是**一致的但不是已证的**：

1. `req/step` 的两种口径（311 vs 774）—— 乐观/悲观的分歧点。需要 `CGC-DECPROF` 的 ntok 分层 dikoved。
2. `miss率 = 1 − hit%` 假设全局均匀 —— births每层重要度不同，覆盖率缀。
3. `5th.` §3 用 resident 反推 slots，假定了当前 stride 是紧凑的（§4 只证明了 gather slab 有 padding，
   expert pool 的 `r.stride` 尚未逐层打印核对）。

**不需要量测的（本报告的主体）：** per-expert 1.0703 MiB、它与 8.80 GiB/8446 miss 的 0.3% 闭合、
设备一步供 406 MiB、整步 union 829 MiB = 2.04× 超限、覆盖率 58.7% ≈ 命中率 61.4%。
