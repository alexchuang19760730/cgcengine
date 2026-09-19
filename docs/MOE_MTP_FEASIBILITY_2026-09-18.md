# MoE-MTP 可行性评估 —— 对着 `powerauto-secrets/MOE_MTP_IMPLEMENTATION_PLAN.md` v1.0

日期：2026-09-18
对象：`powerauto-secrets` 私有仓库 `MOE_MTP_IMPLEMENTATION_PLAN.md`（329 行，2026-09-18 v1.0）
取得方式：HTTPS/gh 两条路都无凭证（`gh auth status` 报 token invalid），**SSH 通道可用** ⇒
`git clone --depth 1 git@github.com:alexchuang19760730/powerauto-secrets.git`（工作副本在 `/tmp/powerauto`）。
本文件不复制计划书内容，只做评估；所有判准都引本 repo 的实测。

**一句话结论**：**2× 可达、3× 需要额外条件；但计划书把因果链写反了** —— 它把 2-3× 归因于
accept rate，而本 repo 自己的数据显示**真正的杠杆是「verify 批次有没有摊薄」**：摊薄不存在时
MTP 是净亏 1.8×（09-14），摊薄修好之后同一个 accept rate 变成 +31%（09-17）。
**accept rate 只有在摊薄存在之后才开始值钱。**

---

## 1. 先看基线：计划书说的 27 tok/s 不是我们这条线的数字

| 来源 | decode t/s | 备注 |
|---|---|---|
| 计划书 §1.1「llama-speculative-simple (MTP)，Mac M4 常驻」 | 27 | 属 **n30cache / edge_server / LoopMoE** 那条线 |
| 本 repo `docs/LATEST_COMMIT_GAP_ANALYSIS_2026-09-15.md:74` 09-14 实测 MTP-off | 8.87 | 本 repo（CGC fork，llama.cpp） |
| 本 repo `docs/M1_WORKITEM4_CANON_ORDER_STATUS_2026-09-17.md:139` `nail_nomtp` | 9.59 | 同 build、同 8GiB 池、交错、每臂 3 请求 |
| 本 repo 可引用 decode 区间（09-18 快照，llama-bench 两轴） | 7.7 – 10.8 | 见 `MEMORY_PERF.md` |

⇒ 两条线的基线差 **2.5–3.5×**。计划书 §6 的「50–80 tok/s」是在它的 27 上乘 2-3× 得来的，
**这个绝对数字不能搬过来**。要在本 repo 立目标，必须先按既有约定（llama-bench、prefill＋decode
两轴并列）重定基线。混用基线会让「2-3×」这个比率看起来达成、绝对值却差三倍。

模型也对不上：计划书 §1.1 写 IQ4_XS；本 repo 实际是
`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（13.66 GB，**256 experts/layer**，
`docs/LATEST_COMMIT_GAP_ANALYSIS_2026-09-15.md:4`）。

---

## 2. 决定性证据：MTP 曾经是净亏，修好之后才转正（都不是 accept 造成的）

| 时点 | accept | MTP-on | MTP-off | 结论 |
|---|---|---|---|---|
| 2026-09-14 | 70.39%（**事后证明被 bug 抬高**） | 6.74 | 8.87 | **净损失 1.8×**（`LATEST_COMMIT_GAP_ANALYSIS_2026-09-15.md:75`） |
| 2026-09-17（canon-order bug 修后） | **58.25%**（180/309） | **12.57** | **9.59** | **+31%**（`M1_WORKITEM4_CANON_ORDER_STATUS_2026-09-17.md:136-141`） |

**注意方向**：accept 从 70% 掉到 58%（数字变难看），MTP 却从亏 1.8× 变成赚 31%。
变的那一项不是 accept，是 **verify 批次终于吃到摊薄** —— canon-order bug 修的是
「verify 批次里 token ≥ 1 是用别的 token 的专家算出来的」（同文档 §M4）。

同一份文档给出的成本读数（`LATEST_COMMIT_GAP_ANALYSIS_2026-09-15.md:80-83`）：

- **W2**：`T=1` 是 92 ms/token、`T=2.6` 是 104 ms/token ⇒ **verify 完全没吃到批量红利**；
  MTP 额外成本的 **84% 在这里**。
- **W3**：pool 143 → 179 slots 让 I/O 少 **3.7×**，decode 只 **+9%** ⇒ **decode 已不是 I/O bound**，
  只能从 compute 侧砍。

### ⚠ 这里推翻了我自己的第一反应（记下来免得下次再犯）

看到「流式 MoE + SSD 专家池」我第一反应是「decode 是 IO bound，所以 verify 批次的专家并集
（`union = n_tokens · top_k`，见 `docs/CAP_INVARIANCE_LOCALIZATION_2026-09-14.md:101`）会随 k 膨胀、
把 dense 上的摊薄红利吃光」。**本 repo 的 W3 直接否证了它**：把 I/O 砍 3.7× 只换到 +9% decode。
⇒ 并集／换入的字节数**不是**主要项；主要项是**每 token 的 compute 与同步**。
下面第 4 节的提案因此从「省字节」改成「让批次的每 token 成本真的摊薄」。

---

## 3. 用本 repo 的成本模型外推：85% accept 能换到多少

设 verify 步成本 `cost(k) = 1 + m·k`（单位＝单 token 成本），`k` = 每步 draft 数，
期望产出 `E(k,a) = (1 − a^{k+1})/(1 − a)`，加速比 `= E/cost`。

用 09-17 那一组反推 `m`（a=0.5825，取 k=3）：
`E = 1+0.583+0.340+0.198 = 2.12`，实测加速 1.31 ⇒ `cost(3) = 2.12/1.31 = 1.62` ⇒ **`m ≈ 0.21`**。

| accept a | k=3 的 E | cost(3)=1.62 | **加速比** | k=5（cost=2.05） |
|---|---|---|---|---|
| 0.58（现状，实测） | 2.12 | 1.62 | **1.31×**（实测 1.31 ✓） | 1.35× |
| 0.70 | 2.53 | 1.62 | 1.56× | 1.63× |
| **0.85（计划书目标）** | 3.19 | 1.62 | **1.97×** | 2.02× |
| 0.92（计划书上限） | 3.55 | 1.62 | **2.19×** | 2.17× |

**读数**：

- **2× 是可达的**：accept 58% → 85% 大约给 1.31× → **1.97×**。计划书的方向在这个意义上是成立的。
- **3× 拿不到**：即使 accept 拉到 92%，k 加到 5–7 也只是 2.2× 左右，因为 `E` 有上界 `k+1`
  而 `cost` 随 k 线性涨。要到 3× 必须**同时**把 `m` 从 0.21 压到 ≈0.10（更好的摊薄）。
- 换算成本 repo：MTP-off 9.59 ⇒ 2× ≈ **19 t/s**，不是 50–80。
- ⚠ `m ≈ 0.21` 是从**单点**反推的（只有 k=3 一组），**不是拟合**。第 5 节第 (i) 项就是去测它。

### 定理式的边界（为什么 accept alone 永远不够）

`E ≤ k+1`（等号只在 a=1 取到），所以：

- 若 `m = 1`（**零摊薄**，09-14 的状态）：`加速比 ≤ (k+1)/(k+1) = 1` ⇒ **任何 accept 都救不了，
  MTP 只会亏**（与 09-14 实测净亏 1.8× 吻合：a=0.70、k=3 时 E=2.53、cost≈4 ⇒ 0.6×）。
- 若 `m = 0`（理想摊薄）：`加速比 = E`，a=0.58 就有 2.12×。

⇒ **摊薄是乘法项，accept 是加法项。先修乘法项。**

---

## 4. 那「修正后的 MTP 架构」应该长什么样

计划书的三层递进（Router-Guided → Iteration-Aware → Expert-Hashed）全部在**改 draft 侧的输入**
（`h_t + router_logits + topk_indices → head → h_{t+1}`）。而本 repo 的证据指向另一侧：
**限制并强制摊薄**。我提议的修正版叫 **RSL-MTP（Resident-Set-Locked MTP）**：

1. **draft 照旧**：沿用现有 nextn MTP head，**先不训练新 head**（原型阶段零训练成本）。
2. **锁定专家集**：对第 i 个 draft token，先算它的 top-8；若 `top8_i ⊄ (本步已付费的专家并集)`，
   **在 i 处截断**，不再验证后面的 draft token。
3. 效果：`union` 恒等于 anchor 的 8/layer，**不随 k 增长** ⇒ verify 步的边际成本只剩 compute +
   gather 的宽度，摊薄的上限被抬高。
4. 代价：accept 从 `a` 降到 `a · p_route`（`p_route = P(top8_{t+i} ⊆ 已驻留集)`）。
5. 判别式：`加速 = (1 + p + p² + …)/cost(k)`，其中 `p = a · p_route`。
   若 `p = 0.7`、`k = 3`、`cost = 1.2`（锁定后摊薄更好）⇒ `2.53/1.2 = 2.1×`。
   ⇒ **用 accept 的下降换摊薄的上升，在流式 MoE 上通常是划算的** —— 这正是它与 dense 版 MTP 的分歧点。

**为什么它比「训一个 Router-Guided head」更贴合我们的架构**：dense 模型里 draft 的边际成本≈0，
所以只能靠 accept；流式 MoE 里 draft 的边际成本**不为零**（并集 remap、gather 宽度），
所以「少 draft 但每一步更便宜」可以赢。

⚠ 它是**假设**，不是结论：W3 说 I/O 只占 ~9%，所以 RSL 的收益必须来自 **gather/compute 宽度**，
不是省下来的字节。这正是第 5 节第 (i) 项要量的东西 —— 量完再决定要不要实现。

---

## 5. 建议顺序：先做三件便宜的，再决定要不要训 head

| # | 做什么 | 成本 | 为什么先做 |
|---|---|---|---|
| (i) | **量 `cost(k)` 曲线**：同一 build 跑 `k = 1, 3, 5` 的 verify 步耗时，拟合 `cost(k)=1+m·k` | 半天，零训练 | `m` 直接决定天花板：`m≈1` ⇒ 训练 head 是白花钱；`m≈0.2` ⇒ accept 58→80% 就给 ~1.9× |
| (ii) | **先验 MTP-on 的 union remap 是对的** | 半天 | `scripts/run_server.sh:516-521` 明写：MTP verify 批次 `n_tokens = n_draft+1 > 1` 是**唯一**需要「一次写满所有 draft token 的 expert union」的路径，且**所有已完成的 probe 都没开 MTP，从没被检查过**。上一个同类 bug（canon-order）正是靠它造出假的 70% accept ⇒ **在没验过这条路径之前，12.57 这个数字带同样的嫌疑** |
| (iii) | **离线量 `p_route`**：用既有的 `LLAMA_EXPERT_CACHE_ROUTE_RECORD` + `_ROUTE_DUMP`（`run_server.sh:411`）dump 路由，算相邻 token 的 `top8` 包含率 | 半天，零训练 | 直接给 RSL-MTP 的 `p_route`，可证伪：若包含率 < 0.5，RSL 不值得做 |

三件都做完（约 1.5–2 天）再决定 Phase 1/2 要不要投。**这比计划书 Phase 0（1-2 天）+ Phase 1（2-3 天）
先训 head 的顺序安全得多** —— 因为 (i) 有可能直接给出「此路不通」。

---

## 6. 计划书里三处需要更正的地方

1. **根因①②在目标模型上不存在**。§1.2 列的四大根因里，「ACT 动态深度」与「DeltaNet 状态依赖」
   属于 **LoopMoE（Gated DeltaNet + RecurrentBlock）**；而 §1.1 自己写 LoopMoE 是 **Phase 0 骨架**，
   已部署的是 Qwen3.6 标准 MoE（本 repo 的 gguf 就是它，带 nextn MTP 头）。
   ⇒ 4 条根因里有 2 条不适用于受试对象。（**推断**：我没读到 `loopmoe/` 的代码，
   依据是 §1.1 的状态栏 + 本 repo 的模型几何。）
2. **「Expert-Hashed」在推理时不是 hash**。§2.4 的 forward 是
   `torch.stack([proj(h) for proj in self.bucket_projections])` —— **64 个 bucket 全做一遍
   `2048×2048` 投影再加权求和**，是 64 倍 dense，不是查表。真要 hash 应该 gather 一个 bucket
   （4.2M 而非 268M MAC/token）。按写的那样实现，draft 侧会先变慢。
3. **参数量自相矛盾**。§2.4 括号写 `64×2048×2048 + 256×64 ≈ 270M`（占 35B 的 **0.77%**），
   却标「+0.15-0.3%」；§3.3 又说训练参数「~50-100M」。另外 §2.2 的 `256×2048 = 524K` 是 0.0015%
   而非「+0.1%」。这些不影响结论，但会让排期估算失真。

（§3.3 的算力估算本身**是对的**：`2 × 100M × 512K tokens = 1.02e14 FLOP = 102 TFLOP`，
除以 5 TFLOP/s ≈ 20 s。我一开始以为差 1000×，重算后确认没错，此处更正我自己。）

---

## 7. 判准（可证伪）

- 若 (i) 量出 `m ≳ 0.6` ⇒ **MTP 路线在本架构上不成立**，应把资源挪到 compute 侧
  （per-layer `eval()` 同步、MoE gather 融合、batched-union gather —— W3 已经点名）。
- 若 (ii) 发现 union remap 有错 ⇒ **12.57 / 58.25% 整组作废**，先修再谈。
- 若 (iii) 量出 `p_route < 0.5` ⇒ RSL-MTP 不值得实现，回到「提 accept」的路线。
- 任何 accept 数字都必须标注「修后语义」（`M1_WORKITEM4…md:146`）：修前记录的 70.4%、19.9%
  属于被抬高的家族，**不可作为 before/after 的一侧**。
