# CGC UPKG 2.1 DFlash 与 SGLang Spec V2 路由比较说明

**版本**: v0.2  
**状态**: 工作稿（历史比较稿，现状已由当前代码验证补强）  
**定位**: 汇总 `UPKG 2.1` 在 `M7.4` 纳入后，对 `DFlash`、`SGLang` 与 `Spec V2` 的开源地址、文档入口、实现边界、official upstream vs vendored benchmark 指标口径与验收选型结论；并补充当前 repo 中 `upkg21` 已实跑通过后的最新理解。

---

## 一、目的

本说明只回答四个问题：

- `DFlash` 的上游正式开源地址是什么
- `SGLang` 的上游正式开源地址与官方文档入口是什么
- 项目内 vendored `cloud_sglang` 与上游官方 `SGLang` 的关系是什么
- `UPKG 2.1` 应该把哪条 `SGLang` 路径作为正式 gate 验收基线

### 1.1 当前验证快照

在当前代码快照下，以下结论已经从“选型建议”推进为“实跑验证后可采纳的工程口径”：

- `m74 = PASS`
- `m75 = PASS`
- `m76 = PASS`
- `upkg21 = PASS`

因此，本文仍保留为 archive 下的历史比较稿，但不应再被理解为“`UPKG 2.1` 尚未跑通前的待定路线讨论”。

### 1.2 与 `CGC_Gate_2.0` 的交叉位置

在当前正式口径中，本文与 `CGC_Gate_2.0` 的关系应理解为：

- 本文仍是 `UPKG 2.1` 下 `DFlash / SGLang / Spec V2` 路由比较与验收边界说明
- 它不是 `CGC_Gate_2.0` 的主白皮书，也不是当前 `Gate 2.0` 的最新主入口

更准确地说：

- `UPKG 2.1`
  - 负责 `DFlash + vendored SGLang + DeepEP` 的当前正式可验证组合路径
- `CGC_Gate_2.0`
  - 负责 layer-adaptive edge-cloud PD disaggregation 的总体边界

因此，从 `CGC_Gate_2.0` 的 `done / proof / target` 口径看，本文更接近：

- `proof` 的支撑材料

也就是：

- 它为 `Gate 2.0` 的云侧执行基座、异构集成与 `DFlash` 路由提供历史比较与选型证据
- 但它不等于 `Gate 2.0` 的 `max_local_layer / finished_layer + 1 / hidden_states + partial_kv` 等核心 `target` 能力已经完成

若需要查看当前最新的 `Gate 2.0` 正式入口，应优先读取：

- `docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/`

---

## 二、上游地址

### 2.1 DFlash

- 官方源码仓库：`https://github.com/z-lab/dflash`
- 项目页：`https://z-lab.ai/projects/dflash/`
- 主要角色：提供 `DFlash` drafter、基于 block diffusion 的 speculative decoding 路径、示例与 benchmark 入口
- 开源协议：`MIT`
- 官方仓库当前对外声明的重点内容包括：
  - `Block Diffusion` 训练 pipeline，可训练与基座模型绑定的 `DFlash draft model`
  - `Transformers / SGLang / vLLM` 多后端适配代码
  - `HumanEval / GSM8K / MT-Bench` 评测复现脚本
  - `Modal` 云侧一键部署模板

### 2.2 SGLang

- 官方源码仓库：`https://github.com/sgl-project/sglang`
- 官方 speculative decoding 文档：`https://docs.sglang.io/docs/advanced_features/speculative_decoding`
- DFlash 集成参考入口：`https://github.com/sgl-project/sglang/pull/16818`
- 官方文档当前可直接作为 `UPKG 2.1` 外部权威参考的内容包括：
  - `DFLASH` 参数入口
  - `Speculative Decoding V2 (Overlap Scheduler)` 说明
  - `EP / TP` 拓扑兼容与限制说明
  - speculative algorithm 与 scheduler capability 的边界说明

### 2.3 项目内 vendored SGLang

- 本地路径：`ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/`
- 定位：项目当前实际消费的 `SGLang` 运行时快照/集成版

---

## 三、已核对的实现信号

### 3.1 DFlash 上游

当前可直接核对到的上游信号包括：

- `README.md`
- `requirements.txt`
- `benchmark.py`
- `run_benchmark.sh`
- `model/`
- `LICENSE = MIT`

这说明 `DFlash` 至少已经公开了：

- 基本安装入口
- Python 侧使用方法
- benchmark 入口
- draft model 相关代码目录
- 可复现 benchmark 与训练脚本入口

### 3.1.1 官方基线应如何理解

对 `UPKG 2.1` 来说，上述 `DFlash` 官方仓库不是“仅供引用的链接集合”，而是：

- `DFlash` 算法、训练与 benchmark 的原始上游定义
- `draft model` 训练与复现实验的权威来源
- `Transformers / SGLang / vLLM` 适配存在性的官方证明
- 后续 `official upstream vs vendored runtime` 比较时的外部基线来源

### 3.2 官方 SGLang 文档

官方 speculative decoding 文档已经明确列出：

- `DFLASH`
- `EAGLE / EAGLE-2 / EAGLE-3`
- `Standalone speculative decoding`
- `Speculative Decoding V2 (Overlap Scheduler)`

文档层面的关键结论是：

- `DFLASH` 是正式支持的 speculative algorithm
- `Spec V2` 是 overlap scheduler 路径
- 当前文档把 `DFLASH` 与 `Spec V2` 分开描述，而不是作为同一执行模式

### 3.2.1 官方 SGLang 基线的验收含义

对 `UPKG 2.1` 来说，`SGLang` 官方仓库与文档的职责不是替代当前工程内 runtime，而是：

- 提供最权威的参数语义与功能边界定义
- 作为 `official upstream SGLang + DFlash` 对比基线
- 用于验证 `vendored cloud_sglang` 与上游文档口径是否仍然对齐
- 为 `Spec V2` 提供正式的比较路径说明，而不是把它和 `DFLASH` 强绑定

### 3.3 本地 vendored cloud_sglang

当前项目内 `cloud_sglang` 已包含以下 `DFlash` 相关实现：

- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/speculative/dflash_utils.py`
- `python/sglang/srt/speculative/spec_info.py`
- `python/sglang/srt/arg_groups/speculative_hook.py`
- `python/sglang/srt/environ.py`

本地代码还能直接确认两个关键事实：

- 存在 `SGLANG_ENABLE_SPEC_V2`
- 存在 `DFLASH does not support overlap scheduling (spec v2).`

这表示项目内实际运行时与官方文档在一个关键点上是对齐的：

- `DFlash` 目前不是通过 `Spec V2 overlap scheduler` 来作为正式执行模式

---

## 四、两条 SGLang 路径的比较

### 4.1 候选 A：官方上游 SGLang

优点：

- 官方文档最完整
- 参数口径最权威
- 与上游 PR / release 节奏同步

限制：

- 对 `UPKG 2.1` 来说，它更像“外部参考基线”
- 若直接以远端 PR 作为唯一验收运行时，会让本地工程验收依赖外部仓库状态

### 4.2 候选 B：项目内 vendored cloud_sglang

优点：

- 已在当前 repo 内实体存在
- 已含 `DFlash` 相关 worker / utils / scheduler 接线
- 可直接被本地 gate 做静态与结构化验证

限制：

- 需要持续与官方文档口径对齐
- 文档说明要显式记录它与上游官方仓库的映射关系

---

## 五、UPKG 2.1 的选型结论

`UPKG 2.1` 当前正式采用：

- `selected_sglang_runtime = project_vendored_cloud_sglang`

原因不是“它名字更新”，而是：

- 它是当前工程内真实可验证、可被 gate 消费的运行时
- 它已经内置 `DFlash` 所需的关键代码路径
- 它与官方文档在 `DFLASH != Spec V2 overlap scheduler` 这一点上保持一致

因此，`UPKG 2.1` 的验收口径应写成：

- 验 `DFlash on vendored SGLang`
- 比较 `Spec V2` 能力边界
- 不把 `DFlash` 强行写成 “必须开启 Spec V2”

当前代码验证结果进一步说明：

- 上述口径不是纸面选型，而是已经与当前 `upkg21` 的实跑通过结果对齐
- `DFlash on vendored SGLang + DeepEP` 已形成当前 repo 的正式可验证组合路径

---

## 六、正式验收口径

自本稿起，`UPKG 2.1` 的最小 `DFlash / SGLang` 验收条件为：

- `M5 = PASS`
- `M7.4 = PASS`
- 已产出 `selected_sglang_runtime`
- 已明确记录 `DFlash` 当前执行模式
- 已明确记录 `DeepEP` 在当前组合路径中的 dispatch backend 与 parallel profile
- 已明确记录 `Spec V2` 在当前选择中的角色是“比较路径”而不是“强制执行模式”
- 已保留 `official upstream SGLang + DFlash` 的开源地址、文档入口与基线说明

### 6.1 官方基线 benchmark 要求

若 `UPKG 2.1` 要把 `DFlash + SGLang` 正式收口为产品级比较结论，除了结构化路由选择外，仍建议持续保留一份基于官方上游的量化对比证据。

最小比较对象：

- baseline：`official upstream SGLang + DFLASH`
- optimized：`project vendored cloud_sglang + DFLASH`

最小固定条件：

- 同一台 GPU 主机
- 同一主模型
- 同一 `draft model`
- 同一 `max_new_tokens`
- 同一采样参数与 batch 规模

最小比较指标：

- 不同 context 下的 `prefill_tps`
- 不同 context 下的 `decode_tps`
- 不同 context 下的 `peak_memory_gb`

建议最小 context 集合：

- `1024`
- `4096`
- `8192`
- `16384`

这份 benchmark 不替代 `M7.4 / M7.5 / M7.6` 的功能验证；当前代码下这些功能验证与 `upkg21` composite 已经跑通，而 benchmark 更适合作为对外说明“为何选择 vendored runtime，同时仍与官方上游保持可比较性”的补充支撑材料。

当 `UPKG 2.1` 进入 `SGLang + DFlash + DeepEP` 的组合路线时，当前正式收口口径进一步固定为：

- `selected_sglang_runtime = project_vendored_cloud_sglang`
- `speculative_algorithm = DFLASH`
- `requested_dispatch_backend = deepep`
- `deepep_parallel_profile = ep16_tp1`
- `Spec V2` 继续保留为比较基线，不与 `DFLASH` 强制绑定
- `official upstream SGLang + DFLASH` benchmark 继续保留为外部比较基线

推荐对外入口：

- `cgc gate upkg21`

---

### 6.2 DFLASH “最优” 的正式定义

对 `UPKG 2.1` 来说，`DFLASH 最优` 不能被简化为“某个配置的 `decode_tps` 最高”，而必须同时满足以下三类条件：

- 精度不发生产品级退化
- 速度相对 non-spec 路径和 official upstream 基线有明确收益
- 资源占用与稳定性在可重复运行的范围内收敛

因此，`DFLASH 最优` 的正式定义应写成：

- 在相同主模型、相同 `draft model`、相同采样参数和相同硬件条件下
- 最终输出质量与 target-only 或 non-spec baseline 保持等价或近似等价
- 同时取得更优的 `prefill / decode / latency / memory / stability` 综合结果

换句话说，`最优` 不是单指标最优，而是：

- `同精度下更快`
- `同速度下更稳`
- `同精度同速度下更省`

### 6.3 精度指标口径

若 `UPKG 2.1` 要把 `DFLASH` 作为正式 runtime 选择的一部分，必须补齐精度 guardrail。推荐最小精度指标如下：

- `exact_match_rate`
  - 固定 prompt、固定 seed、`temperature = 0`
  - 比较 `DFLASH on` 与 `DFLASH off` 或 target-only baseline 的最终输出是否一致
- `token_match_rate`
  - 用于补充 exact match 过于严格的问题
  - 长回答、模板差异或空白字符差异时更有辨识力
- `task_score_delta`
  - 在固定任务集上比较 `DFLASH on` 相对 non-spec baseline 的得分变化
  - 可用于摘要、问答、长上下文抽取、数学推理等样本集
- `draft_acceptance_efficiency`
  - 记录 draft token 被 verify 接受的效率
  - 它不是最终精度指标，但能反映 `draft model` 与 target model 的对齐质量

推荐最小精度验收门槛：

- `exact_match_rate >= 99%`
- `token_match_rate >= 99.9%`
- `task_score_delta >= -1%`
- 不出现系统性错误模式：
  - 长文漏答
  - 重复生成
  - verify 截断异常
  - speculative accept/reject 行为明显异常

### 6.4 速度与资源指标口径

对 `official upstream SGLang + DFLASH` 与 `project vendored cloud_sglang + DFLASH` 的正式比较，不应只保留 `decode_tps`，而应至少覆盖以下指标：

- `prefill_tps`
  - 衡量长 context 输入阶段的吞吐
- `decode_tps`
  - 衡量 speculative decode 阶段的吞吐
- `ttft_ms`
  - `time to first token`
  - 用于衡量线上体感与首 token 响应
- `e2e_latency_ms_p50 / p95`
  - 用于衡量整体请求耗时和尾延迟
- `peak_memory_gb`
  - 当前 runner 已经量测的 GPU 峰值显存指标
- `host_ram_peak_gb`
  - 用于识别 `cpu_offload_gb` 带来的 host OOM 风险
- `startup_success_rate`
  - 用于识别是否存在必须靠“碰运气”才能启动的配置

其中，当前 `UPKG 2.1` 第一轮 benchmark artifact 已实现和必须保留的核心指标为：

- `prefill_tps`
- `decode_tps`
- `peak_memory_gb`

后续建议在同一 artifact schema 中逐步补齐：

- `ttft_ms`
- `e2e_latency_ms`
- `host_ram_peak_gb`
- `startup_success_rate`

### 6.5 official upstream vs vendored 的正式比较口径

对外部可复核的 benchmark，必须固定以下条件，避免把 runtime 差异与 launch shape 差异混在一起：

- 同一台 GPU 主机
- 同一主模型：`DeepSeek-V4-Flash`
- 同一 `draft model`：`Qwen3.5-4B-DFlash`
- 同一 speculative algorithm：`DFLASH`
- 同一 `tp / ep / pp`
- 同一 `context_length`
- 同一 `max_new_tokens`
- 同一采样参数
- 同一 warmup / runs 次数

正式比较对象定义如下：

- `baseline = official upstream SGLang + DFLASH`
- `optimized = project vendored cloud_sglang + DFLASH`
- `reference = target-only / non-spec baseline`

三者职责不同：

- `baseline`
  - 回答“相对官方上游，我们是否更快、更省或至少不退步”
- `optimized`
  - 回答“当前工程内实际验收 runtime 的真实表现”
- `reference`
  - 回答“引入 speculative decoding 后，最终质量有没有退化”

### 6.6 当前 host2 第一轮收敛配置

在 `host2` 的 `DeepSeek-V4-Flash + DFLASH + Qwen3.5-4B-DFlash` 路线下，当前第一轮 benchmark 的安全收敛配置应显式写入方案，作为“优先跑通”口径：

- `tp_size = 2`
- `pp_size = 1`
- `moe_a2a_backend = deepep`
- `deepep_mode = normal`
- `mem_fraction_static = 0.6`
- `cpu_offload_gb = 4`
- `context_length = 8192`
- `contexts = [1024, 4096, 8192]`

这组参数不是“最终最快配置”，而是：

- 第一轮真实 benchmark 数据的可运行配置
- 用于先拿到 `prefill / decode / memory` 正式证据
- 在 `host RAM OOM`、`GPU OOM` 与残留进程污染之间做出的最低风险折中

### 6.7 DFLASH 最优的正式 PASS / FAIL 门槛

若要把 `DFLASH` 路线作为 `UPKG 2.1` 的正式验收项，建议采用以下门槛：

精度门槛：

- `exact_match_rate >= 99%`
- `token_match_rate >= 99.9%`
- `task_score_delta >= -1%`

性能门槛：

- `decode_tps > target-only baseline`
- `prefill_tps` 不低于 official upstream 的可接受范围
- `vendored decode_tps >= official upstream decode_tps`
- 若 vendored 目标是“优化版”，则建议额外要求：
  - `vendored decode_tps >= official upstream decode_tps * 1.05`

资源与稳定性门槛：

- `peak_memory_gb <= official upstream`
- 不出现 host OOM kill
- 不出现连续残留进程导致的第二个 runtime 污染
- benchmark 在同一配置下可重复执行

若上述任一条件不满足，则不能称该配置为 `DFLASH 最优`，只能称为：

- `可运行配置`
- `临时收敛配置`
- 或 `局部更快但未完成产品级收口的实验配置`

### 6.8 三层正式配置归属

结合当前 `host2` 上 `DeepSeek-V4-Flash + DFLASH + Qwen3.5-4B-DFlash` 的实际收敛过程，`UPKG 2.1` 不应继续把启动参数、环境兼容策略和系统主配置混在 benchmark runner 中临时试错，而应按统一 kernel 白皮书的三层口径拆开：

- `profile setting + binding`
  - 解决“这次怎么跑”
  - 负责本轮 benchmark 的任务级运行形态，例如：
    - `contexts = [1024]` 或 `[1024, 4096, 8192]`
    - `gen_tokens = 32 / 128`
    - `warmup_runs / runs`
    - `baseline -> cleanup -> optimized` 的比较顺序
    - benchmark artifact 的 binding key 与 profile 引用
- `environment bootstrap`
  - 解决“这台机器 / 这个 runtime 有什么能力”
  - 负责硬件、runtime、分布式和兼容性探测，例如：
    - `SM120` 上 `deep_gemm` 不可用
    - `deepep_mode = normal`
    - `DFLASH` 当前要求 `pp_size = 1`
    - `cpu_offload_gb > 0` 时禁用 `fp8 CPU pin-memory staging`
    - runtime 启动失败后的进程组清理、显存清理、残留实例回收
- `model setting / system profile`
  - 解决“整个系统到底是谁在跑、怎么组、哪些变因是正式配置”
  - 负责系统主模型、draft、runtime 家族、拓扑和 required component，例如：
    - 主模型：`DeepSeek-V4-Flash`
    - `draft model`：`Qwen3.5-4B-DFlash`
    - runtime family：`official upstream SGLang` / `project vendored cloud_sglang`
    - `speculative_algorithm = DFLASH`
    - `requested_dispatch_backend = deepep`
    - `tp_size = 2`
    - `mem_fraction_static = 0.6`
    - `cpu_offload_gb = 4`
    - `DeepSeek-V4-Flash` 的 `wqkv_a / legacy_kv / legacy_o_proj` 权重契约

因此，当前 benchmark runner 中为收敛 `OOM` 和启动失败而引入的临时逻辑，后续应分别上收为：

- 进入 `profile setting + binding` 的：
  - benchmark shape
  - contexts / token 数
  - official/vendored 顺序比较模式
- 进入 `environment bootstrap` 的：
  - `SM120 -> deepep_mode=normal`
  - `disable fp8 CPU pin-memory staging`
  - 启动清理与进程残留回收
- 进入 `model setting / system profile` 的：
  - `DeepSeek-V4-Flash + Qwen3.5-4B-DFlash + DFLASH + DeepEP`
  - runtime family 选择
  - 正式的模型权重映射与 attention 兼容契约

也就是说，`UPKG 2.1` 这次遇到的问题不应再被理解为“benchmark 参数还没猜到”，而应被理解为：

- 哪些属于任务级 profile
- 哪些属于环境能力 bootstrap
- 哪些属于系统与模型的正式 contract

只有这样，后续 `M7.4 / M7.5 / M7.6 / UPKG 2.1` 重跑时，才不会继续在 runner 中重复手工试参。

---

## 七、技术方案结论

`UPKG 2.1` 对 `DFLASH` 的正式技术方案，不应只回答“是否支持”，而应完整回答：

- `官方上游的定义是什么`
- `工程内实际消费的 runtime 是什么`
- `official upstream vs vendored` 如何做同口径 benchmark
- `DFLASH 最优` 的精度、速度、资源和稳定性门槛是什么

因此，本方案正式收口为：

- 用 `project_vendored_cloud_sglang` 作为当前工程内正式验收 runtime
- 保留 `official upstream SGLang + DFLASH` 作为外部对照基线
- 用 `target-only / non-spec baseline` 作为精度 reference
- 用 `prefill_tps / decode_tps / peak_memory_gb` 作为第一轮强制 benchmark 指标
- 用 `exact_match_rate / token_match_rate / task_score_delta` 作为精度 guardrail
- 用 `ttft / e2e latency / host_ram_peak / startup_success_rate` 作为后续增强指标
- 先以可稳定启动的 `host2` 安全配置产出第一轮真实 benchmark 数据，再在此基础上继续调优

---

## 八、一句话结论

如果目标是把 `M7.4` 正式放入 `UPKG 2.1 gate`，当前最稳妥的做法不是把 `DFlash` 与 `Spec V2` 强行合并，而是：

- 以项目内 `vendored cloud_sglang` 作为 `DFlash` 的正式验收运行时
- 以 `DeepEP` 作为 `SGLang MoE` dispatch backend，并显式收口 `ep16/tp1`
- 以官方 `SGLang speculative decoding` 文档作为参数与边界参考
- 把 `Spec V2` 保留为能力比较与路线说明，而不是 `DFlash` 的硬性开关
