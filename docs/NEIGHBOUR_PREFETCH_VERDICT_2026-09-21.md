# File-neighbour prefetch — 到底能优化 prefill 还是 decode？

日期 2026-09-21 · 裁决：**两个都帮不上，但理由完全相反。**
方法：不写 C++、不重建（tree 上还有别条线的引擎改动）。在**引擎自己的需求流**上 replay。

---

## 0. 结论表

| | neighbour prefetch 有用吗 | 为什么 |
|---|---|---|
| **prefill** | ❌ 无处可用 | prompt 对每专家 LRU 的需求流贡献**恒为零**（252 / 1005 / 2009 token 实测），它整层读（slab 路径，256/256 expert）。**已读 100% 的东西没有可预取的。** |
| **decode** | ❌ 有效但不划算 | 确实能提命中率：R=1 省 **22.1%** miss。但代价是**读入专家数 1.98×**，而device 吞吐「与读取形状无关」⇒ 时间 ∝ 字节 ⇒ 净亏。利用率只有 **28.4%**。 |

一句话：**decode 那一半是「有效但太贵」，prefill 那一半是「根本没有可以优化的对象」。**

---

## 1. 方法：为什么这次可以直接 replay

`LLAMA_EXPERT_CACHE_DEMAND_DUMP`（格式 `cgc-demand v2`）记录**每一个**改变某层 residency 或 LRU 顺序的操作，
顺序即 `cache->m` 串行化的顺序。用引擎自己的策略 replay 它，比较就是一个**恒等式检验**，不是拟合。

**闸门（先写死再跑）**：baseline replay 必须同时复现引擎自己给出的两个 miss 数。

```
MISS_DUMP 行数 = 3175   ← ensure_batch pass-2 写出的 miss（对比用的 multiset）
n_misses      = 3328   ← final stats（含 ensure_slot 的 miss，MISS_DUMP 不写）
```

实测：

```
batch misses   replay=  3175   engine MISS_DUMP=  3175   err=+0.00%
all misses     replay=  3328   engine n_misses  =  3328   err=+0.00%
hits           replay=  9679
```

**两个都 0.00%。** 于是下面每一个 swept 数字都是同一份已验证模型的派生，不是新模型。

> 这个方法论的代价在这里已经付过一次：我第一次拿 **36312** 当参考值，闸门直接拒了 ——
> 那是 `llama_server_20260920_131425.log` 的数字，那份 log 自己的 header 写 `n_slots=71`，
> 而今天这一格是 **142 槽**（resident 6430.62 MiB 对那边 3076.85 MiB，约 2×）。
> **拿 A run 的 trace 去对 B run 的 miss 数**，正是这 repo 反复要撤回的错误。

Capture 同时验证了它抓的正是被测的那一格：`resident=6430.62 MiB`，与 np1 量测那一格**逐位元相同**。

工具：
- `Backup/phase_decomp/capture_demand_trace.py`（一份自洽 capture：trace + MISS_DUMP + final stats 同进程）
- `Backup/phase_decomp/neighbour_gate.py`（replay + 闸门 + radius sweep；parser 复用 `scripts/check/reuse_distance.py`，只读）

---

## 2. prefill 那一半：没有可优化的对象

union 直方图是**完美双峰**（1–8 与 14–32，中间无重叠），而 union **恰好等于 8** = 单 token 的 top-8，
15–16 = 2 token。831 个 `B` 事件 / 41 层 ≈ 20 步，正好对上 64 token ÷ mean_len 3.2。
⇒ **`B` 事件全部是生成步。**

那 prompt 在哪？做 prompt 长度缩放：

| reps | prompt tokens | 生成 tokens | demand 行数 | requests | misses |
|---|---|---|---|---|---|
| 1 | 252 | 32 | 7,259 | 10,689 | 2,319 |
| 4 | 1,005 | 64 | 7,547 | 13,007 | 3,328 |
| 8 | 2,009 | **1** | 6,099 | 6,605 | 433 |

关键点：`6,099 − 5,976(S) − 41(X) − 41(Z) = 41` = 恰好一趟 41 层的 batch（1 个 token）。
而 `S` 事件数在三次运行里**恒为 5,976**，且与 prompt 长度（252→2009）和生成长度都无关。

⇒ **prompt evaluation 对每专家 LRU 的需求流贡献恒为零。**
它与 `CGC_PREFILL_STREAM=1` + `CGC_GATHER_SLAB_CAP=256` 一致：prefill 走 slab 路径**整层读**
（256/256 expert，先前实测 M≥16 时 step≈4.28 s ≈ 填料全部专家）。

**所以 prefill 上根本无处下手**：一个每次都读 100% 专家的路径，没有任何「还没读、但快要用」的邻居可以预测。
（若真要在 prefill 省，唯一对症的是**把相邻读合并成一个 span** —— 那属于 request coalescing，
不属于预取；而且它是 Stage 2-a 的另一半，与「写一个新的 prefetcher」不是同一件事。）

---

## 3. decode 那一半：有效，但败在字节上

### 3.1 在哪里 miss

把 trace 切开分别 replay：

| 段 | demands | miss (R=0) | 占比 |
|---|---|---|---|
| `S` 相（id 递增扫描，与请求无关） | 5,976 | 153 | 4.6% |
| **`B`+`T` 相（生成）** | 7,031 | **3,215** | **96.6%** |

**96.6% 的 miss 在 decode。** 每层在整个 capture 里触及 **190–256 个不同 expert（中位 198）**，而槽只有 **142**
⇒ miss 是**稳态 capacity miss**，不是预热。老朋友 `reuse_distance` 也这么说：25.6% compulsory / 74.4% capacity。

### 3.2 预取确实有效（我的先验错了）

每次 miss 顺手把文件相邻 id（e±1..±R）也放进来，用同一个 victim 规则：

| R | miss | 省下 | prefetch 放入 | 有用 | 利用率 | **读入专家数** |
|---|---|---|---|---|---|---|
| 0 | 3,215 | — | 0 | — | — | 3,215 (1.00×) |
| **1** | 2,506 | **22.1%** | 3,850 | 1,093 | **28.4%** | **6,356 (1.98×)** |
| 2 | 2,300 | 28.5% | 6,708 | 1,557 | 23.2% | 9,008 (2.80×) |
| 4 | 2,171 | 32.5% | 12,557 | 2,079 | 16.6% | 14,728 (4.58×) |
| 8 | 1,922 | 40.2% | 21,905 | 2,686 | 12.3% | 23,827 (7.41×) |

命中率**真的上去了**。原因不是「相邻 expert 有特殊相关性」——file adjacency 与 routing 统计独立；
而是 ∵ 每层每步约 8/256 的需求率很高，一个随机 expert 在 ~18 步的驻留窗里有约 30% 机会被再次路由到
（实测 28.4%，同量级）。**这是「随便拉一组 random expert 进来」也会有的效果**，不是预测能力。

### 3.3 但currency 是字节，不是 miss 数

`docs/CB_IS_THE_DEVICE_RESULT_2026-09-20.md` 的实测结论：**device 吞吐与读取形状无关**（read shape flat）
⇒ **时间 ∝ 字节**。那么：

- R=1 抹掉 22.1% 的 miss，代价是**读入专家数 1.98×**。
- 要打平，「顺手带的字节」每字节必须便宜到 `3215/6356 = 0.52`，即**便宜 ≥ 48%**。
- 我今天用 `coalesce_race.py` 量到的**最好情况**是 35%（gap=1、零 junk：1447 → 943 µs）。
- 35% < 48% ⇒ **R=1 顶多也就是略亏；而那 35% 还是有零 junk 且两边都用得上才拿到的**，
  本 Case 里 71.6% 的偷渡字节永远没被读过。

⇒ **decode 上不要写这个 C++。**

### 3.4 真正对症的东西，以及为什么今天做不到

池缺的是**容量**：每层触及 198 个不同 expert，而槽只有 142。
```
slots 142 → 284 : miss 13,374 → 7,921  （−41%，普通 LRU）
slots 71  → 142 : miss 36,312 → 13,374 （−63%，普通 LRU）
Belady OPT @142 : 9,222              ← 任何策略（含任何预取）的上界
```
注意这个排序：**把槽加到 284 的普通 LRU（7,921）比 142 上的完美策略（9,222）还低。**
⇒ 有效期最长的杠杆是容量，而它被 16 GB 卡住（284 槽需要约 12.3 GB，今天已经 6.4 GB）。

---

## 4. 与历史证据的一致性

这不是第一次有人想「预取 decode 会用到的 expert」。repo 里已经建过一次，并已判死：

`run_server.sh:1501` 的 M5 prerouter（`CGC_PREROUTER`），2026-09-17 实测：
```
CGC-PREROUTER: calls=80 queued=0 scored=40 pred_total=320 hit=41 (precision 12.8%)
```
**prefetch_slot 拒绝了全部 320 次预测**（`prewarm_hot` 已按同一个 freq top-K 填满，预测的早就 resident 了），
自评天花板 **0.2pp**，默认 OFF。

本轮的 neighbour prefetch 与它不同（不预测，只是搭 file 邻居的便车），所以在 LSTM 意义上避开了
「预测失败」这个坑 —— 但换来的是「防不胜防的字节代价」。两条路落到同一个结论。

---

## 5. 建议

1. **不写这个 C++。** 两种 regime 都不成立，且有可证的上界。
2. 若还想继续扣 IO，唯一还活着的目标是 **request coalescing 在 prefill/slab 那一侧**
   （那里相邻读真的连着，条件最有利），而不是新 prefetcher。
   注意既有 merge 在 np1/np4 已被证明是死的（`NO_MERGE` 只让 job 数变 0.83% / 0.018%）——
   「 contribute 合并」和「分割为形状」是两件事，别混为一谈。
3. 本轮的**方法论**比结论更值钱，已落 Skill：`cgc-io-request-shape`
   （三支柱：几何免费直读 GGUF / 计数器 vs 时间读数 / trace replay 必须自带同 run 闸门）。

---

## 附：本轮踩到的两个坑（留给下次）

- **`pread_usec` 不能用于归因。** 它在 `fill_job` 内（`:41`）与 worker 循环（`:3424`）**各累加一次**且按 segment 计，
  推出来 17.9 ms/job，而 `fill_batch_usec/file_reads` 是 184.8 µs/job —— 差 97×。
- **跨 run 校验的陷阱** 已被闸门挡下（见 §1）；正确做法是「同进程同时产出 trace 与 counter」。
