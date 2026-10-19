# S 序列化/段数消减线 — 终局判决（bit-exact 前提下）

**日期**：2026-10-07　**owner**：lineI　**硬件/模型**：M4 MacBook Air 16GB、Qwen3.6-35B-A3B（MTP off）、pool 8GB
**方法**：先用现有 witness 量 per-step cold 与 cold 成因（0 新代码→加一个 0 额外 GPU 的成因探针），再下结论；不先信旧 S1 ×1.83。

---

## 1. 要回答的问题

41 段提交（每 MoE 层一个 argsort 边界、段间 `wait → topk hook(fill+写 remap leaf) → submit 下一段`）的
**段间同步累计 ~36.3 ms/step**（S1_LINE_VERDICT：fill 同步仅 3.955 ms、其余 ~32.3 ms 为段提交/调度序列化）。
能否通过**单段**或**少段**提交，在 **bit-identical** 前提下省掉这笔序列化？

---

## 2. 实测一：单段（S1_SINGLE_SUBMIT，GPU 设备查表）per-step cold 序列

开关：`CGC_S1_SINGLE_SUBMIT=1;CGC_SLOT_TABLE_GPU=1;CGC_S1_MIN_IL=0;CGC_S1_SS_VERIFY=1`
（前两个开关缺失会分别导致 **arm no-op** 与 **host-leaf → VOID for bit-identity**，由引擎 fail-loud 拦下。）
cold ＝ 该步被消费的 routed expert 在表发布时**没有 real slot**（读到保留 ZERO slot、贡献缺失）。

| vstep | cold / 320（40层×8） | 占比 |
|---|---|---|
| 0 | 88 | 27.5% |
| 1 | 87 | 27.2% |
| 2 | 27 | 8.4% |
| 3 | 11 | 3.4% |
| 4 | 33 | 10.3% |
| 5 | 7 | 2.2% |
| 6 | 20 | 6.3% |
| 7 | 2 | 0.6% |
| 8 | 14 | 4.4% |
| 9 | 16 | 5.0% |
| 10 | 0 | 0% |
| 11 | 1 | 0.3% |
| 199 | 3 | 0.9% |
| 399 | 9 | 2.8% |
| 599 | 0 | 0% |

**读数**：冷启动前两步高（27%，池未热）；vstep2 起快速衰减；稳态（vstep≥10）每步总 cold 0–9 个（0–2.8%），
池基本热了——但**稳态仍有零星 cold、并非每步 cold==0**。诊断臂 thermal 全程 NOMINAL、swap growth=0
（witness 每步付 full-graph drain，**速度数字不可引用**）。

---

## 3. 实测二（决定性）：cold 的成因 —— 96% 是跨步全新专家

在 witness 里把每个 cold 专家分类（不额外跑 GPU，ids 本就在 readback）：
- **cold_in_prev**：上一步**同层也选中**它 → 只是 fill 晚了，"提前一步 fill"可救；
- **cold_new**：上一步同层**没选中** → 跨步新专家，任何"提前一步"都无法预知。

**累计 640 decode step：cold_in_prev = 131（3.7%）、cold_new = 3414（96.3%）。**

稳态抽样（vstep199/399/599）cold_new = 8/9/3、cold_in_prev 全为 0。
⇒ **"用上一步真实 ids 提前 fill"救不了单段 residency**：绝大多数缺口是上一步根本不存在的专家。
（第一性可预见：每层每步只选 8/256，专家集合跨步大幅轮换；相邻 token 位置重合 ~87% 不等于逐层专家集合稳定。）

---

## 4. 第一性根因：ids 与 fill 的循环依赖，Metal 提交后不可插入

- 每层 routed ids 是**图内** `gate matmul → argsort/topk` 的产物，**图算到该层才存在**；
- fill 必须在该层 consumer（mul_mat_id）**之前**完成，而 fill 需要 ids；
- **单段**把 40 层一次 async 提交，提交后 CPU 无回头点、Metal 单队列也无法在命令缓冲中途插入 fill
  （没有"图内同步缺页读 SSD"机制）⇒ submit 时专家必须**已在池**；
- 唯一的"ids 已知 → fill → 再继续"时机，就是**段边界的同步点**。

**因此每一个"引入当场不在池的专家"的层，都强制需要一个同步点**；消同步点 = 赌该层专家已在池。

---

## 5. 少段折中（每 G 层一段）同样不成立为"简单 bit-exact 收益"

- 段内 G 层一次提交：第 1 层可由段间 hook 用已知 ids fill；**第 2..G 层 ids 在段内、提交前未知**，
  只能靠"上一步 ids 提前 fill + 池已热"；
- 实测每层稳态仍约 0.2 个跨步新专家（§3）⇒ 段内第 2..G 层残留 cold，必须**段内错重算/整段回退**才 bit-exact；
- 新专家分散（每步 6–9 个 over 40 层、约 16% 的层当步有新专家）：
  - G=4 段"恰好全无新专家"概率 ≈ 0.84⁴ ≈ 0.50，其余段整段回退（付"段+41段"双倍）；
  - 回退整段的成本远大于省下的几个同步点 ⇒ 算术上难有正收益，与 FAST 臂 **position 命中 0/200、总回退无收益**一致。
- 即旧外推"+5%（~14.3 t/s）"也自带"段内错重算 ~4 ms"前提，**不是纯 bit-exact 的小改**，工程并不小。

---

## 6. 判决

1. **单段 S1（含"cold>0 整步回退"）判死**：96.3% cold 是跨步新专家，几乎每步回退、双倍净亏。
2. **少段无简单 bit-exact 正收益**：残留新专家分散、乐观分组赌错即整段重跑；要做必须先实现段内局部重算（中大型工程），现实量级 ~+5%。
3. **根因是 Metal 执行模型**（单队列、提交后不可插入、无图内缺页），不是量具或调参问题。
4. 真正破墙只有 kernel/模型侧 —— GPU 图内异步缺页 fill（Metal 不支持、工程巨大）、或让路由跨步更稳定（模型侧、做不到）；**这些都在"我们改不了"的边界内**。
5. 本判决与 **ρ layer-batch NO_EFFECT（字节 −64% 但 t/s −3.6%）**、**IO_PATH_AB（miss×1.9 但 t/s 不动）**、
   **DECODE_25TPS_VERDICT（shape 上限 16.3–20.5）** 互相收敛：**host 侧（fill/段序列化）账面有钱，但被 Metal 同步结构锁死；可落地瓶颈在 device compute / 执行模型。**

## 7. 残余可选项（如实、不夸大，供决策）

- **D-少段 + 段内局部重算**：唯一仍 bit-exact 的 S 线变体，量级 ~+5%、工程中大型、成功率中等；
- 资源应优先投在**device compute span（占边际 token 56–61%）**可落地的 kernel/形状项，而非继续 host 段序列化；
- 若无可落地 device 项，则以现状交付锚点收尾，S 线登记判死、避免重试。

---

### 附：产物与复现

- cold 序列/cold-origin 原始 log：`/tmp/s1_cold_diag2.json.logs/live/prod-new_CGC_S1_SINGLE_SUBMIT_1_CGC_SLOT_TABLE_GPU_1_CGC_S1_MIN_IL_0_CGC_S1_SS_V.p512_n128_d0_r5.stderr.log`、`/tmp/s1_cold_origin.stderr.log`
- 成因探针源码：`llama-context.cpp` witness 块（`CGC-S1-SS-ORIGIN`，随 CGC_S1_SS_VERIFY 输出）
- 立项卡：`scripts/check/charters/e-s-single-submit-correctness.yaml`
