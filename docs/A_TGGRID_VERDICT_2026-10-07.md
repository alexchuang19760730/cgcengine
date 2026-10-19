# A 方向判别判决：查表访存重排判死，多字段 gather 才是真因

日期：2026-10-07
任务：#20（A 第一步判别：IQ2_S 查表访存可否重排）
关联：`EXPERT_KERNEL_BANDWIDTH_ROOTCAUSE_2026-10-07.md`（任务 #19 根因定位）
量具：`cgc_shape_probe --op mul_mat_id`，形状 k=2048 / out=512 / experts=256 / used=8 / tokens=1 / mode=batch，bank 82.00 MiB，ids cycle bank（无 expert 重复读）。

## 1. 要回答的问题

§19 把 expert GEMV 只有峰值 30–45% 的根因候选定为「IQ2_S/IQ3_S 的查表式访存组织」，其中含两个可分离的部件：

- **(a) constant grid 查表**：用 qs/qh 当索引查 constant memory 的 `iq2s_grid`（数据依赖、随机 gather）；
- **(b) block 多字段 gather**：每 32 个 K 对 qs(4×ib)/qh/scales/signs/d 做多笔非连续小 load。

本判别用两个受控变体把 (a) 单独定价，回答「把查表从 constant 重排到 threadgroup / 直接去掉，能否让带宽连续」。

## 2. 判别矩阵（同一 id kernel，仅改查表来源）

| 变体 | 查表来源 | 带宽 GB/s | % 峰值(108.8) | vs baseline |
|---|---|---|---|---|
| baseline | constant `iq2s_grid` + 多字段 gather | **36.30** | 33% | — |
| TG (`CGC_IQ2_TGGRID=1`) | threadgroup 查表（每 expert 复制 8 KB + barrier） | **29.34** | 27% | **−19.2%** |
| NOGRID (`CGC_IQ2_NOGRID=1`) | 连续 device 读、无查表、无复制、保留多字段 gather | **40.53** | 37% | **+11.7%** |
| Q8_0（对照） | 连续 block、线性反量化 | 86.05 | 79% | +137% |
| F16（对照） | 无反量化、连续 | 91.16 | 84% | +151% |

## 3. 结论（逐项，皆有读数）

1. **查表来源不是杠杆，A 方向（查表重排到 threadgroup）判死。**
   - 朴素搬到 threadgroup 反而 **−19%**：id 框架里每个 threadgroup 处理一个 expert，每个 expert 都要独立复制全表 8 KB + barrier；decode batch=1 下一个 expert 只做一次 GEMV（读 0.32 MB），8 KB 复制（2.5% 字节）+ barrier 的启动延迟摊不薄，净亏。Metal threadgroup memory 不跨 threadgroup 共享，无法「全表只复制一次」。
   - 即使完全去掉查表（NOGRID，连续读、零复制零 barrier），带宽也只 **+11.7%**（36.3 → 40.5）。

2. **真正的 ~46 GB/s 缺口（40.5 → 86）来自多字段 gather（部件 b），不是查表（部件 a）。**
   - NOGRID 已消除全部查表数据依赖，却只到 Q8_0 水平的 **47%**。
   - 即 IQ2_S block 每 32 个 K 读 qs/qh/scales/signs/d 的多笔非连续小 load，这种散射访存才是喂不饱 load 管道的主因。

3. **多字段 gather 是 IQ2_S 编码格式的固有结构，不是一个开关能重排。** 要动它只有三条，均非无痛：
   - 重写 kernel 的线程↔block 映射与 inner load 编排（让多字段 load 合并/更连续）——大型 kernel 工程、高风险，且字段在 block 内布局固定，可重排空间未知；
   - 换连续 block 编码（Q8_0 等）——字节大增、active 变大，第一性上净亏；
   - 换更现代、访存布局更连续的量化格式——属模型/量化侧改动。

## 4. 对 t/s 与方向的含义

- A（查表重排）实测收益上界 ≈ **+12% 的 expert kernel 带宽**，且朴素实现为负；不构成可落地杠杆。
- expert bank 在 T=4 为 31.81 ms。即使乐观假设把多字段 gather 也重排到 Q8 水平（~2.6×），省 ~19 ms 是**不可达的理论上界**——NOGRID 证明查表只占其中 ~4 ms，其余 ~15 ms 锁在格式固有的 gather 里。
- 这与 roofline 一致并进一步收敛：**在不改量化格式 / 模型的前提下，decode 的 bit-exact、单流可落地杠杆（调度、同步、查表重排）已基本穷尽。**

## 5. 边界（诚实声明）

- 本判别只在 **IQ2_S**（gate/up，82 MiB/bank）上实测。IQ3_S（down，31.36 GB/s）同为 grid 查表 + 多字段 gather、机制相同，推断同理，但**未单独跑 TG/NOGRID**。
- NOGRID 用 activation `y` 作连续源，与权重的 cache 路径可能不同；但「去掉查表仅 +12%、且只到 Q8 的 47%」这一结论对该偏差不敏感（不可能因此从 40 跳到 86）。
- 所有变体默认 off，仅经 env 启用，不影响生产路径。

## 6. 复现命令

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
P=scripts/check/shape_probe/cgc_shape_probe
A="--op mul_mat_id --type iq2_s --k 2048 --out 512 --experts 256 --used 8 --tokens 1 --mode batch --iters 60"
$P $A                       # baseline 36.30
CGC_IQ2_TGGRID=1  $P $A     # TG 29.34
CGC_IQ2_NOGRID=1  $P $A     # NOGRID 40.53
```

改动落点（未 commit）：`ggml-metal.metal`（:9263 模板加 TG_GRID/NO_GRID、grid 复制、三分支查表、id _tg/_nogrid 实例）、`ggml-metal-device.cpp`（tg_grid/no_grid 标志、env CGC_IQ2_TGGRID/CGC_IQ2_NOGRID、smem、base 切换）。
