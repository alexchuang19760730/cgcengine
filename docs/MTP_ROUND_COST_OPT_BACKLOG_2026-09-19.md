# 轮成本还能怎么优化 —— 从「喂什么」到「六个可测候选」（2026-09-19 00:05）

## §0 一句话

上一轮把 20% 差异归因到「喂进去什么」之后，量测顺带把**引擎侧真正的瓶颈**也照出来了：
**轮时 ≈ 每轮 expert cache 请求数 × 1.59 ms/请求，固定项 ≈ 0**。
所以优化只有三个杠杆：**降请求数、降单价、提高每轮 emit**。
下面六个候选按「能不能现在跑」排序，每个都带可证伪判准。

---

## §1 实测基线（同一窗口，可引用）

### HTTP / server 侧（V4b，MTP on，真 token）

| 指标 | 值 |
|---|---|
| decode | **12.99 t/s**（[12.96, 13.07, 12.99]，spread 1.009） |
| draft acceptance | 0.46541，mean len 2.40 |
| cache requests | 24705，hits 14780，misses 9925（hit **59.8%**） |
| miss 归因 | **capacity 6180 (62.3%)** / compulsory 3745 (37.7%) |
| evictions | 9642（≈ misses 9925，即几乎每次 miss 都驱逐一个活槽） |
| worst layer | layer 1 distinct=**217** vs slots=**143**；layers_distinct_over_slots=5 |
| MTP fast path | calls=9964 union=179719（**18.04/call**）；verify 18.81，draft 8.00 |
| read shape | jobs=309810，bytes=11.0 GB，**0.03 MiB/job** |
| prefetch | 518/171（丢弃 171，**100% drain_cleared**） |
| prewarm | **req=0**（从未启用） |
| pool resident | 6430.62 MiB；LAYER_CAPS total 5976 slots，min 143/layer |

### llama-bench 侧（V1，MTP on，`rand()%n_vocab` token）

| 指标 | 值 |
|---|---|
| decode | 10.46 t/s |
| cache requests | 27274，misses 12022（hit 55.9%） |
| miss 归因 | compulsory 6959 (**57.9%**) / capacity 5063 (42.1%) |
| worst layer | layer 0 distinct=**252** slots=143；layers_distinct_over_slots=**28** |
| MTP fast path | calls=12219 union=216565（17.72/call）；verify 18.14，draft 8.00 |
| read shape | jobs=90930，**0.14 MiB/job** |

**两边最刺眼的对比**：server 只有 5 层 distinct>slots，bench 有 28 层；
server 的 miss 以 capacity 为主（62.3%），bench 以 compulsory 为主（57.9%）。
⇒ 乱 token 把每层工作集铺到 252/256，真 token 只铺到 217/256 —— 但 **143 槽连 217 都装不下**。

---

## §2 六个候选（按可跑性排序）

### O1. `CGC_PREFILL_PROTECT` —— 现成开关，默认 OFF，可同 server 交错 A/B ★★★

- **代码**：`src/llama.cpp/src/llama-expert-cache.cpp:381-401`
- **机制**：prefill 的 per-chunk union（≤ n_batch 8 × topk 8 = 64 槽/层）会冲掉 decode 收敛的 ~45 槽/层，
  于是 prefill 之后要重读 ~1850 experts（≈2 GB / ≈2.4 s）才重新收敛。
- **已有量测**（同型号、同量化、8 GiB pool，注释内）：
  steady-state **22.2 t/s** → prefill 后首个 generation 掉到 **8.1–8.9 t/s**。
- **为什么对本场景特别相关**：我们的形状正是 **depth=2025 prefill + gen=128**，
  128 个 token 里前 ~32 个直接踩这个坑。
- **成本**：只改 victim 选择，不改 fills / remap / maths。
- **关键便利**：有 `CGC_PREFILL_PROTECT_FILE`（每次 batch 重读）⇒ **一个 server 跑两臂，无 reload drift**。
- ⚠️ **陷阱**：`static const bool env_on = getenv("CGC_PREFILL_PROTECT") != nullptr;`
  —— 设成 `CGC_PREFILL_PROTECT=0` **也会开**。A/B 必须用 FILE 开关，不能用 env。
- **判准**：V4b 12.99 → 若 prefill 后首段恢复，预期 **+10~25%**；
  同时 requests/round 应下降（少一次 catch-up 重读）。

### O2. 提高 slots/layer 143 → ≥217 —— 结构性，但有过订阅风险 ★★☆

- 算术：槽大小 = 110 MiB / 256 experts = **0.4297 MiB**；
  217 槽/层 × 40 层 = 8680 槽 ≈ **3.64 GiB**（现在 5976 槽 ≈ 2.5 GiB）⇒ **+1.14 GiB**。
- 旋钮：`LLAMA_EXPERT_CACHE_LAYER_CAPS`（`start-end:cap,...`）、`CGC_SERVER_EXPERT_CACHE_BYTES`。
- ❌ **风险（必须先算）**：这台机器已经 **OVERSUBSCRIBED by 4838 MiB**
  （model resident 13030 + pool 8192 = 21222 vs 物理 16384），靠 macOS 压缩 + swap 才活着。
  历史：ub=6144 pool 8GiB 存活 **0/5**（status 5 OOM）。
- ⇒ **不能直接加池**，除非先腾空间（降 ub / 降 model resident / 换更小 pool + 更准的 victim）。
  这是要单独排窗口、且要 OOM 复测的活。

### O3. 不改容量改命中率：整轮 pin union 的 18 个槽 ★★☆

- 证据：evictions 9642 ≈ misses 9925 ⇒ 每次 miss 都在驱逐一个**马上还要用**的槽（经典 thrash 指纹）。
- 思路：一轮 verify 的 union（18.04 槽）已经算出来了，把它整轮 pin、轮末释放，
  而不是只在 batch 级保护。
- **判准**：evictions/misses 从 0.97 降下来，hit rate 59.8% → 目标 **≥75%**；
  requests/round 116.5 不变的话收益有限，要看 ms/round。

### O4. 降 read 粒度碎片：0.03 MiB/job → 一次读满一槽 ★★☆（待验证）

- server 每槽 0.43 MiB，却切成 **~14 个 30 KB job**；bench 是 0.14 MiB/job（~3 个/槽）。
- 30 KB 的 pread 在 NVMe 上远离最优请求大小（这台 SSD 顺序 ~7 GB/s，小随机读要掉很多）。
- 若这 14 个 job 是并发发出，收益会被重叠掉 ⇒ **必须先确认是不是并发**。
- **判准**：合并后 MiB/job 0.03 → ≥0.3，且 `pread_usec` 明显下降；否则证伪。

### O5. 提高 MTP 每轮 emit（n_max 3 → 5） ★☆☆（先测接受率）

- 现在 acceptance 0.46541 / mean len 2.40（n_max=3，理论上限 4）。
- 但因为 **轮时 ∝ 请求数**（不是 ∝ 轮数），emit 提高会同时推高 union 请求数，
  收益不是免费的。
- **判准**：n_max=5 时 emit/轮 是否 > 2.40×(5/3)=4.0，且 ms/round 增幅 < emit 增幅。

### O6. prewarm / prefetch 全链路失效 —— 修好是净赚，但要查为什么 ★★☆

- `prewarm req=0 hit=0 miss=0`：prewarm **从未启用**。
- `prefetch=518/171`：丢 171（33%），丢弃原因 **100% 是 drain_cleared**（预取还没用就被清掉）。
- 两者都是「已经写好但没生效」的部件 ⇒ 修好是净收益，但先要查清 drain 是谁触发的。

---

## §3 工具侧：唯一能裁决「喂什么」假设的实验（不是优化，是判决）

**T1. 让 llama-bench 喂真 token** —— **这不是附属实验，它就是任务 ② 本身**

- 修一处，两个偏差**同时**消失：
  - 轮时 325.4 → ~185（requests/轮 200.3 → ~117）
  - emit/轮 3.404 → ~2.40（acceptance 0.8012 → ~0.4654）
- ⇒ 预测 **2.40 / 0.185 = 12.97 t/s**，正对上 HTTP 12.99（差 0.2%）。

> **更正（2026-09-19 00:20）**：本節初稿寫的是 18.4 t/s，那是錯的。
> 它一邊說「差異全在餵什麼」，一邊把 emit/輪的 +42% 當成 bench 的結構優勢保留下來，
> 只修了一半。而 emit/輪的 42% **正是** acceptance 0.8012 vs 0.46541 的產物
> （mean len = 1 + 3p：1+3(0.8012)=**3.404** ✓，1+3(0.4654)=**2.396** ✓），
> 餵真 token 後它會跟著消失。所以 T1 是「重建」，不是「跑贏 42%」。

- **判准**：喂真 token 后 bench 与 HTTP 差 **<5%** 算重建成功。
- 乱码上接受率反而高（0.8012 > 0.4654）的机理解释：乱码 prompt 上模型分布极尖，
  draft 与 target 的 top-1 高度重合 ⇒ 接受率虚高；真文本分布平坦，接受率回落到 0.465。

### T1 的实现方案（已定位到行）

HTTP 侧用的 prompt 是 `scripts/check/http_duo.py:77` 的 `_PREFILL_UNIT`（一段 ~75 token 的
中文散文，**重复**到 2025 token）。bench 侧对应位置是
`src/llama.cpp/tools/llama-bench/llama-bench.cpp`：

- `test_prompt()` **:2308**，填充循环在 **:2321-2323**（`std::rand() % n_vocab`）
- depth 与 prompt **都**走它：`:3023 test_prompt(ctx, t.n_depth, ...)`、`:3050 ... t.n_prompt`

⇒ 最小改动：加 `--prompt-file <path>`，tokenize 后**循环取模**填进 `:2321-2323`，
同一份散文、同一个长度，两边就吃到同一串 token。核心约 10 行 + 参数管道
（`cmd_params` :347 / 默认值 :399 / help :474 / 解析 :641）。

---

## §4 建议的下一步顺序

1. **T1** —— 任务 ②（让 bench 重建 HTTP）本身，未完成的主线，排第一。
2. **O1**（零风险、现成开关、可同 server 交错 A/B、有历史量测支撑）—— 引擎侧最优先。
3. **O6**（prewarm/prefetch 失效诊断，先查 drain_cleared 来源）—— 诊断，不改行为。
4. **O3 / O4**（要动代码、要排 GPU 窗口）。
5. **O2**（动内存预算，必须先过 OOM 复测，单独排窗口）。

每一项都必须走 `--reps 4` + MIN_KEPT=3 + SPREAD_LIMIT=1.10，并确认窗口里没有别条线的
量测进程（`lsof -nP -iTCP:8080 -sTCP:LISTEN` + `pgrep -fl 'llama-bench|llama-server|spec_cost_curve'`）。

---

## §5 未结（承前）

- W3（nospec，同 binary）8.44 vs V2 10.62，**−21% 未解释**（W3 只等 45 s 就接在 CPU-SPLIT 2686 的 W2 后，疑环境残留）。
- bench 接受率 **0.8012** vs server **0.46541**：乱 token 上接受率反而高 72%，疑仪器假象，未查。
- X 臂（X1/X2/X1b/X2b）因 `Insufficient Memory` 与 `XPC_ERROR_CONNECTION_INVALID` **未拿到数**。
- 本轮所有改动**未 commit**（树上有别条线 `llama_bench_matrix.py` / `prod_matrix.py` 的改动）。


---

## §10 T1 实测（2026-09-19 01:09–01:16）：预测 12.97 未兑现，且**机制预测被证伪**

### 10.1 发生了什么

`--prompt-file` 已实现（llama-bench.cpp，6 处补丁）并 build 成功。真 prompt 取自
`scripts/check/http_duo.py:77` 的 `_PREFILL_UNIT`，一字未改、重复 30 次供循环取模。
Metal 编译服务在 00:00–00:51 期间不可用（`XPC_ERROR_CONNECTION_INVALID`，连跑 5 轮全死），
用户关掉多余 session 后（compressor 9.8 GB → 1.07 GB、swap 5.14 GB → 0）于 **01:09 恢复**。

### 10.2 结果

| 臂 | 时间 | prompt | t/s | stddev | thermal(worst) |
|---|---|---|---|---|---|
| T1 | 01:09 | 真 | **8.53** | 26.7% | MODERATE(15) |
| 对照（同 binary, 无 --prompt-file） | 01:13 | 乱 `rand()%n_vocab` | **7.18** | 2.6% | **HEAVY(100)** |
| V1（历史，旧窗口） | 9/18 21:59 | 乱 | 10.46 | — | NOMINAL |

预测是 **12.97**（对上 HTTP 12.99），实测 8.53 —— **预测未兑现**。

### 10.3 但更重要的：机制预测被证伪

我预测「喂真 token ⇒ expert 路由集中 ⇒ cache 请求数/轮 200.3 → ~117（−42%）」。
同窗口对照实测：

| 指标 | T1（真） | 对照（乱） | 变化 | 预测 |
|---|---|---|---|---|
| runtime requests | 28626 | 26444 | **+8.2%** | −42% ✗ |
| compulsory miss | 7785 (60.5%) | 6681 (58.0%) | +16.5% | 下降 ✗ |
| worst layer distinct | **256** | 250 | +6 | 下降 ✗ |
| layers distinct over slots | **40** | 28 | +43% | 下降 ✗ |
| us/job（读单价） | 24070 | 18941 | +27% | — |

**真 prompt 的 expert 工作集并没有变小 —— worst layer distinct = 256，即全部 expert。**
「真文本路由更集中」这个前提是错的：256 个 expert 上，真散文的路由照样铺满。

⇒ **「差异在喂什么（经由 cache 请求数）」这条机制不成立。**

### 10.4 绝对数暂不可引用

- T1 stddev 26.7%（门槛 10%）
- 对照 thermal worst **HEAVY**（100 次），T1 MODERATE（15 次）—— 而所有历史可引用数
  （V1 10.46、HTTP V3b/V4b）都是 launch **NOMINAL** 下取的
- 两臂都比 V1 低 18–31%，量级与热节流相符

⇒ 正在跑**交错 A/B**（`t1_interleaved_ab.sh`）：T1/对照交替 ×2 轮，每轮前等 thermal 回
NOMINAL，每臂 `--reps 4`（丢 rep1 留 3），共 6 个保留样本/条件。用它才能把 prompt 效应
与热漂移分开。

### 10.5 待答

1. 冷却后 bench 的**绝对**水位是多少（回到 ~10.5？还是新常态更低？）
2. prompt 效应的**真实符号与大小**（单次对照提示真 prompt 反而快 18.8%，与
   「乱码让 bench 虚高」的原有说法相反，需交错数据裁决）
3. 若 prompt 效应≈0，则 bench 与 HTTP 的 20% 差距**另有机制**，需回到 O3/O4（pin union、
   read 粒度）而非「喂什么」
