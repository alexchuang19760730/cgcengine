# CGC Unified Pipeline Kernel Design v1.0

## 1. 文档目标

本文档用于定义 `CGC engine pipeline` 与 `Megatrain pipeline` 的统一设计方向。

核心结论不是“让两条 pipeline 互相调用”，而是：

- 它们本质上应下沉为同一个 `Unified Pipeline Kernel`
- 差异只来自不同的执行上下文与策略选择
- `Megatrain` 不应继续作为另一套平行 engine，而应成为 `CGC Engine` 的 train profile

一句话说：

```text
CGC Engine 是平台内核，Megatrain 是该内核在 train 场景下的一组 profile。
```

进一步说，`CGC Unified Pipeline Kernel` 的正式产品定位，不应只是单一训练框架、单一推理框架或单一 MLOps 编排层，而应定义为一套面向企业 `agent / model / embodied` 三类任务实体的统一采集、训练、推理与端云交付内核。它通过统一的 `ExecutionContext`、`State ABI`、`publish_manifest`、`runtime_contract` 与 `matrix_axes`，把多框架训练、多框架权重交付、多环境推理与多硬件部署收敛到同一个可编排、可比较、可审计、可交付的执行主干之下。

---

## 2. 统一设计动机

当前系统已经同时出现以下能力：

- `runtime_mode`
  - `local_infer`
  - `local_train`
  - `edge_cloud_infer`
  - `edge_cloud_train`
- `State ABI`
  - `legacy_o_proj`
  - `legacy_kv`
  - `runtime_branch_required`
- `compatible_state_dict` cache
- local / distributed 两类执行路径
- train / infer 两类 wrapper

这说明系统实际上已经拥有统一 pipeline 的必要部件，只是目前仍以不同代码路径表达。

若继续保持：

- 一套 `CGC engine pipeline`
- 一套 `Megatrain pipeline`

则会持续出现以下问题：

- 配置项重复定义
- runtime_mode 与 environment 语义分散
- cache / distributed / ABI decision 逻辑重复
- summary / stage trace / smoke runner 难以统一
- train 与 infer 的公共主干无法被平台化

因此，系统下一阶段应从“多条功能线并行演进”，升级为“统一 kernel + profile + plugin”。

---

## 3. 统一核心判断

`CGC engine pipeline` 与 `Megatrain pipeline` 的差异，不应被视为两套不同系统，而应被视为：

- 相同内核
- 不同执行上下文
- 不同策略选择

影响策略的主要维度不是“这是 CGC 还是 Megatrain”，而是：

- 任务实体不同
- 硬件拓扑不同
- 模型装配层级不同
- 部署环境不同
- 运行模式不同

因此，真正需要统一的是：

- 统一上下文描述
- 统一 ABI 描述
- 统一策略解析器
- 统一八步主干
- 统一可插拔 train / infer / distributed / edge-cloud 插件

---

## 4. 统一后的系统形态

推荐统一为以下五层：

### 4.1 Context Layer

负责感知当前任务所处的执行上下文。

### 4.2 ABI Layer

负责描述模型状态语义与兼容性边界。

### 4.3 Strategy Layer

负责根据上下文与 ABI 选择执行策略。

### 4.4 Kernel Layer

负责执行统一八步主干。

### 4.5 Evidence Layer

负责统一输出：

- `summary.json`
- `stage_trace.jsonl`
- `compatibility report`
- `loss / latency / throughput`
- `cache source`
- `distributed collective evidence`

---

## 5. ExecutionContext 定义

统一 kernel 的入口不应再直接接收零散 config，而应优先接收 `ExecutionContext`。

推荐最小字段如下：

- `task_entity`
- `task_type`
- `hardware_platform`
- `hardware_topology`
- `model_scope`
- `model_assembly`
- `environment`
- `runtime_mode`
- `model_family`
- `abi_descriptor`
- `distributed_topology`
- `weight_source_policy`

当前代码里第一版 `ExecutionContext` 已经实际落地，并至少承载以下字段：

- `task_entity`
- `task_domain`
- `task_type`
- `hardware_scope`
- `hardware_platform`
- `hardware_topology`
- `model_scope`
- `model_assembly`
- `environment`
- `runtime_mode`
- `backend`
- `model_name`
- `model_family`
- `runtime_profile`
- `distributed_backend`
- `parallel_tp_size / parallel_pp_size / parallel_ep_size`
- `abi_descriptor`

推荐语义如下：

### 5.1 task_entity

标识上层任务实体：

- `agent`
- `embodied`
- `model`

### 5.2 task_type

标识当前任务行为：

- `train`
- `inference`
- `compile`
- `collective_smoke`
- `cache_restore`

### 5.3 hardware_platform

标识硬件平台：

- `l20n`
- `mac`
- `ascend`
- `nvidia5090`
- `windows`

### 5.4 hardware_topology

标识硬件执行拓扑：

- `single_node_1gpu`
- `single_node_8gpu`
- `dual_node_1gpu`
- `dual_node_8gpu`
- `multi_node_tp_pp_ep`
- `edge_single_accelerator`

其中历史字段 `hardware_scope` 在当前实现里保留为 `hardware_topology` 的兼容别名。

### 5.5 model_scope

标识模型部署侧：

- `edge_model`
- `cloud_model`

### 5.6 model_assembly

标识模型装配层级：

- `none_minimal_collective`
- `tiny`
- `real_weights`

### 5.7 environment

标识部署环境：

- `cloud_single`
- `cloud_cluster`
- `edge_cloud`

### 5.8 runtime_mode

沿用现有四态：

- `local_infer`
- `local_train`
- `edge_cloud_infer`
- `edge_cloud_train`

---

## 6. ABI 与统一矩阵的关系

统一 kernel 需要同时读取两类输入：

- `State ABI`
- `统一执行矩阵`

二者职责不同：

- `State ABI`
  - 回答“模型状态语义是否兼容”
- `统一执行矩阵`
  - 回答“当前应以什么执行策略运行”

推荐统一理解为：

```text
ABI 决定兼容边界
统一矩阵决定执行策略
```

其中矩阵字段在工程上建议至少包含：

- `task_entity`
- `runtime_mode`
- `environment`
- `model_scope`
- `hardware_platform`
- `hardware_topology`
- `model_assembly`
- `model_name`

尽管历史上常口语化称为 `4D 感知矩阵`，当前落地已经收敛为一套 `5 轴 + 扩展字段` 的正式矩阵：

- 轴 1：`task_entity`
- 轴 2：`runtime_mode`
- 轴 3：`environment`
- 轴 4：`model_scope`
- 轴 5：`hardware_platform`

同时补充：

- `hardware_topology`
- `model_assembly`
- `model_name`

---

## 7. Unified Pipeline Kernel

统一后的 `PipelineKernel` 负责运行同一套八步骨架。

推荐骨架如下：

### Step0 Context Detect

输入：

- `ExecutionContext`
- 硬件信息
- ABI metadata
- 模型 config

输出：

- 已归一化的执行上下文
- 初始策略选择结果

### Step1 Materialize

负责：

- 构建 static model / wrapper host
- 载入 checkpoint
- 应用 ABI decision
- 恢复 compatible cache

### Step2 Staticize / Graph Build

负责：

- 图结构捕获
- 编译前约束检查
- runtime branch host 准备

### Step3 Layout / Partition Compile

负责：

- TP / PP / EP 切分
- quant layout 选择
- cache schema / tensor layout 绑定

### Step4 Runtime Branch Resolve

负责：

- `legacy_o_proj`
- `legacy_kv`
- future vision / router / moe branch

### Step5 Wrapper Build

负责：

- infer wrapper
- train wrapper
- edge-cloud wrapper

### Step6 Execution Strategy Resolve

负责：

- local / distributed
- collective backend
- cache read/write policy
- zero-copy / resume policy

### Step7 Run / Measure / Export

负责：

- run step
- loss / latency / throughput
- `summary report / matrix_axes / execution_context`
- `publish_manifest`
- `runtime_contract`
- `dualtrack_summary`
- trace / compatibility report

### Step8 Bridge / Edge Delivery

负责：

- 将 train profile 产出的模型、adapter、Q2RL payload 导出为 bridge artifact
- 将统一矩阵写入 `publish_manifest / runtime_contract / bridge_info`
- 将 edge ingest / runner handoff 统一为同一套 `matrix_axes`
- 为 `official track / cgc track / edge ingest` 生成可直接比较的 summary

---

## 8. StrategyResolver

统一 kernel 不应在主流程中写死大量 if/else，而应通过 `StrategyResolver` 选择策略。

设计上的推荐最小策略组如下：

- `LoaderStrategy`
- `AbiCompatStrategy`
- `RuntimeBranchStrategy`
- `PartitionStrategy`
- `CollectiveStrategy`
- `CacheStrategy`
- `TrainExecutionStrategy`
- `InferExecutionStrategy`

策略选择的输入应为：

- `ExecutionContext`
- `AbiDescriptor`
- runtime device inventory
- distributed topology

策略选择的输出应为：

- loader 插件
- runtime branch 插件
- train / infer execution profile
- distributed collective profile

### 8.1 当前代码已落地的第一版 StrategyPlan

当前代码中，`StrategyResolver` 已经先以“纯判定层”的方式落地为 `StrategyPlan`，并实际输出以下 9 类策略：

- `model_branch_strategy`
- `runtime_branch_strategy`
- `distributed_strategy`
- `collective_strategy`
- `cache_strategy`
- `weight_loading_strategy`
- `edge_cloud_transport_strategy`
- `runtime_plugin_strategy`
- `weight_mapping_strategy`

这一阶段的目标不是立即重写执行逻辑，而是先把分散在主流程中的语义判断，收敛为统一策略输出对象。

### 8.2 当前 9 类策略的工程角色

- `model_branch_strategy`
  - 决定当前模型族应进入哪类主分支
- `runtime_branch_strategy`
  - 决定当前 runtime 是本地原生、ABI bridge，还是 edge-cloud
- `distributed_strategy`
  - 决定单进程、单机分布式、多机分布式的大类
- `collective_strategy`
  - 决定 collective 使用最小化、单机 CUDA、多机 CUDA 等形态
- `cache_strategy`
  - 决定当前更偏向 prefix-state、compatible cache、node-local cache 或临时态
- `weight_loading_strategy`
  - 决定当前是 tiny synthetic、HF direct、HF + compatible cache、还是 deferred
- `edge_cloud_transport_strategy`
  - 决定 edge-cloud prefill 更偏向 `PD / LLM1(OpenAI) / bundle / deferred`
- `runtime_plugin_strategy`
  - 决定应挂接哪类 runtime plugin host
- `weight_mapping_strategy`
  - 决定 checkpoint 进入 static model 时更应走哪类映射语义

### 8.3 判定到执行入口的第一条安全接线

为了避免一开始就把策略层直接接进最脆弱的训练/权重路径，推荐先从最安全的 selector 开始接线：

- `edge_cloud_transport_strategy`
  - 先只接到 `edge-cloud prefill` 的入口选择
  - 不改 `_cloud_prefill_pd()`、`_cloud_prefill_openai()` 等子实现
  - 只让上层 selector 从“看 config 条件”升级为“优先看 strategy”

当前代码已按这个原则开始第一条接线：

- `pd_prefix_kv`
  - 对应 `_cloud_prefill_pd()`
- `llm1_openai`
  - 对应 `_cloud_prefill_openai()`
- `bundle_import / edge_cloud_deferred`
  - 当前仍保留兼容回退，不强制改变原始行为

这条接线的价值在于：

- 它首次证明 `StrategyResolver` 不只是报告对象，而能开始驱动执行入口
- 它不碰训练主路径，不碰 weight remap，不碰 distributed collectives
- 它为后续“策略 -> 插件/执行入口”提供最低风险模板

### 8.4 当前已落地的矩阵传播原则

当前主线不只是把矩阵字段收进 `ExecutionContext`，而是要求它们穿透到所有关键 artifact：

- `pipeline report`
- `bridge_info.json`
- `publish_manifest.json`
- `runtime_contract.json`
- `official_track_summary.json`
- `track_summary.json`
- `dualtrack_summary.json`

统一写法为：

- 顶层保留关键字段
- 同时附带 `matrix_axes`

推荐最小 `matrix_axes` 为：

- `task_entity`
- `task_domain`
- `runtime_mode`
- `environment`
- `model_scope`
- `hardware_platform`
- `hardware_topology`
- `model_assembly`
- `model_name`

这样同一条 train -> bridge -> edge -> compare 链路中的所有产物，都能直接在工具层做并排比较，而不需要再靠文件路径或人工推断上下文。

---

## 9. Plugin 化原则

统一 kernel 不应把模型族、任务族、设备族硬编码在主流程中。

推荐至少把以下部分插件化：

### 9.1 WeightMappingPlugin

负责：

- checkpoint key remap
- post-load adaptor
- forbidden mapping 判定

### 9.2 RuntimeBranchPlugin

负责：

- `runtime_branch_required`
- forward branch install
- branch activation evidence

### 9.3 TrainExecutionPlugin

负责：

- optimizer
- scaler
- gradient clip
- scheduler
- loss reduction

### 9.4 InferExecutionPlugin

负责：

- inference wrapper
- decode path
- edge-cloud infer adapter

### 9.5 DistributedPlugin

负责：

- process group init
- collective backend selection
- barrier / all_reduce / monitored collectives
- failure evidence export

---

## 10. Megatrain 的新定位

统一后，`Megatrain` 不应再被视为一套独立 engine，而应定义为：

```text
CGC Unified Pipeline Kernel 在 train 任务下的一组 profile。
```

具体来说：

- `Megatrain = Train Profile`
- `CGC local infer = Infer Profile`
- `CGC edge-cloud infer = EdgeCloud Infer Profile`
- `bridge export = Train Publish / Delivery Profile`

也就是说：

- `Megatrain` 保留业务名称
- 但架构身份从“独立 pipeline”下沉为“统一 kernel 的 train profile”
- `bridge / edge publish` 不再是独立脚本岛，而是统一 kernel 的交付剖面

这会带来两个直接收益：

- train / infer 可以复用同一套 ABI / cache / summary / trace 主干
- local / distributed / edge-cloud 差异不再演变成多条分裂主流程
- publish / runtime_contract / edge ingest 也能复用同一套矩阵上下文

---

## 11. 统一后与当前系统的映射关系

### 当前概念

- `CGC engine pipeline`
- `Megatrain pipeline`
- `runtime_mode`
- `State ABI`
- `compatible_state_dict`

### 统一后概念

- `CGC Unified Pipeline Kernel`
- `ExecutionContext`
- `AbiDescriptor`
- `StrategyResolver`
- `Profile`
  - `Train Profile`
  - `Infer Profile`
  - `EdgeCloud Train Profile`
  - `EdgeCloud Infer Profile`

### 映射关系

- `MegatrainPipelineConfig`
  -> `ExecutionContext + Train Profile Config`
- `runtime_mode`
  -> `ExecutionContext.runtime_mode`
- `State ABI`
  -> `AbiDescriptor + CompatRuleEngine`
- `legacy_o_proj / legacy_kv`
  -> `RuntimeBranchPlugin`
- distributed sync / collective smoke
  -> `DistributedPlugin`
- `publish_manifest / runtime_contract`
  -> `DeliveryProfile + matrix_axes`
- `dualtrack_summary / official_track_summary / cgc_track_summary`
  -> `ComparativeEvidenceProfile`

---

## 12. 最小重构路线

不建议一步重写全部系统，建议按以下顺序推进：

### Phase 1

抽出：

- `ExecutionContext`
- `AbiDescriptor`
- `StrategyResolver` 空壳接口

目标：

- 不改变现有行为
- 先统一概念边界

### Phase 2

让以下字段统一进入 context selector：

- `task_type`
- `runtime_mode`
- `environment`
- `hardware_platform`
- `hardware_topology`
- `model_scope`
- `model_assembly`

目标：

- train / infer / edge-cloud 不再各自持有分裂的入口语义

### Phase 3

把现有：

- `legacy_o_proj`
- `legacy_kv`
- `_map_static_weight_key()`

抽成：

- `RuntimeBranchPlugin`
- `WeightMappingPlugin`

并继续遵循：

- 先让 `StrategyResolver` 负责统一判定
- 再把某一类最安全的 selector 挂到执行入口
- 最后才让具体实现完全插件化

### Phase 4

把 `Megatrain` 下沉成：

- `TrainExecutionPlugin`

并让：

- `local_train`
- `edge_cloud_train`

成为同一 kernel 的 profile 分支。

### Phase 5

统一 evidence 输出：

- `summary report`
- `stage_trace.jsonl`
- compatibility report
- collective smoke report
- `publish_manifest.json`
- `runtime_contract.json`
- `dualtrack_summary.json`

目标：

- train / bridge / edge / compare 不再各自定义一套上下文字段
- 所有输出统一携带 `matrix_axes`
- official / cgc / edge 三条轨都能被统一比较器直接消费

### 当前阶段补充说明

当前代码状态已经处于 `Phase 1 -> Phase 3` 的过渡带：

- `ExecutionContext` 已落地
- `StrategyResolver` 已落地为 9 类纯判定层
- `edge_cloud_transport_strategy -> _cloud_prefill()` 已开始第一条执行入口接线
- `pipeline report / publish_manifest / runtime_contract / dualtrack summary` 已开始统一矩阵字段收口

因此，接下来的重构不应再从“是否需要统一”出发，而应从“哪一类策略最适合下一条安全接线”出发。

---

## 13. Unified Pipeline Kernel Gate 1.0

前述 `ExecutionContext / StrategyResolver / matrix_axes / publish_manifest / runtime_contract` 的统一，不只是为了整理概念，也必须最终落到同一套正式 gate 验收口径。

结合现有 `runtime_mode` 四模式 fresh smoke 结果与云侧已有 `M7.6 FusionRoute Formal Gate` 要求，`Unified Pipeline Kernel Gate 1.0` 应被定义为：

- 统一 kernel 的正式技术 gate
- 面向 `agent / model / embodied` 三类任务实体的统一验收口径
- 吸收 `M7.6` 中所有可落到内核、运行时、协议与证据层的正式要求
- 明确区分：
  - `kernel smoke PASS`
  - `formal gate PASS`

一句话说：

```text
Unified Pipeline Kernel Gate 1.0
= 四模式 kernel 主干跑通
+ 统一矩阵字段穿透
+ bridge / edge delivery 契约成立
+ M7.6 formal evidence 需求被收口到同一套验收框架
```

### 13.1 Gate 1.0 的边界

`Gate 1.0` 不是产品化 `M8`，也不等于任意单条 demo path 可运行。

它要求：

- `local_infer / local_train / edge_cloud_infer / edge_cloud_train` 四模式都进入同一 kernel 主线
- `ExecutionContext` 与 `matrix_axes` 成为唯一上下文语义
- `publish_manifest / runtime_contract / bridge_info / summary report` 统一携带同一套矩阵字段
- `M7.6` 中涉及 router、多实例、runtime foundation、speculative、DeepEP 与 formal evidence 的要求，必须进入同一份 gate 结构

它不要求：

- 最终对外 CLI / DX / 商业化入口完整收口
- 安装体验与跨平台发行问题全部解决
- 所有外部服务都已产品化

### 13.2 Gate 1.0 的两层通过标准

推荐把 `Gate 1.0` 明确分成两层：

- `Layer A: kernel smoke`
  - 验证四模式是否在统一 kernel 内 fresh run 跑通
  - 验证 `ExecutionContext / StrategyResolver / runtime_mode` 是否已成为真实入口
- `Layer B: formal evidence`
  - 验证 router、多实例、bridge、端云 runtime、speculative、DeepEP 与批量评测证据是否达到正式口径

当前主线已经能回答 `Layer A`：

- `local_infer = PASS`
- `local_train = PASS`
- `edge_cloud_infer = PASS`
- `edge_cloud_train = PASS`

但 `Layer B` 必须继续按正式证据收口，不能因为 kernel smoke 已通过就直接宣称 `M7.6 formal PASS`。

### 13.3 从 M7.6 并入 Gate 1.0 的正式验收对象

来自云侧 `CGC_M76_FUSIONROUTE_FORMAL_GATE_WHITEPAPER_v1.0_zh_CN.md` 的要求，应全部并入 `Gate 1.0`，并映射为以下统一验收对象：

1. `router_runtime`
   - `MiniCPM5` 必须真实参与 route decision
2. `four_instance_topology`
   - cloud 侧必须是 `4` 个独立 `DeepSeek-V4-Flash` instances
3. `fusionroute_hit_evidence`
   - 单题 trace 必须能证明候选实例、命中实例与融合策略
4. `trueorthokda_foundation`
   - `M7.5` active runtime foundation 不得退化
5. `dflash_runtime`
   - speculative / DFlash 必须有真实 runtime evidence
6. `deepep_real_chain`
   - 若宣称启用 `DeepEP`，不得以 fallback 到 native routing 作为 PASS
7. `request_trace_observability`
   - gateway / instance / runtime / fallback 链必须可追踪
8. `multi_instance_resilience`
   - 单实例故障时，系统必须可归因且其余实例可继续服务
9. `swe_verified_formal_evidence`
   - 必须具备面向 `SWE-bench Verified 500` 的批量 formal evidence 结构

### 13.4 Gate 1.0 在统一内核中的映射

上述 9 类检查，不应再散落为多份 gate 白皮书各自定义，而应映射进统一内核五层：

- `Context Layer`
  - `task_entity`
  - `runtime_mode`
  - `environment`
  - `model_scope`
  - `hardware_platform`
  - `hardware_topology`
- `ABI Layer`
  - `State ABI`
  - `TrueOrthoKDA active runtime foundation`
- `Strategy Layer`
  - `router_runtime`
  - `fusionroute_hit_evidence`
  - `four_instance_topology`
  - `multi_instance_resilience`
- `Kernel Layer`
  - `dflash_runtime`
  - `deepep_real_chain`
  - edge-cloud prefill / decode / resume branch
- `Evidence Layer`
  - `request_trace_observability`
  - `swe_verified_formal_evidence`
  - `summary report / publish_manifest / runtime_contract / dualtrack_summary`

### 13.5 Gate 1.0 的正式 PASS 条件

`Unified Pipeline Kernel Gate 1.0 = PASS` 至少应同时满足以下两组条件：

第一组，kernel 主干成立：

- 四个 `runtime_mode` fresh run 全部通过
- `ExecutionContext` 成为四模式统一入口
- `StrategyResolver` 成为执行入口的统一判定层
- `matrix_axes` 已穿透 `pipeline report / publish_manifest / runtime_contract / edge ingest / summary`

第二组，formal evidence 成立：

- `MiniCPM5` router 有真实 route evidence
- `4` 个独立 cloud instances 可被独立 probe、独立 trace、独立失败归因
- `FusionRoute` 可证明多实例命中与融合，而不是单实例伪装
- `TrueOrthoKDA` foundation 继续保持：
  - `true_state_transport = PASS`
  - `edge_state_resume_decode = PASS`
  - `compression_effective = PASS`
  - `zero_copy_vram_real = PASS`
- `DFlash` / speculative path 若被宣称开启，必须有真实 runtime hit evidence
- `DeepEP` 若被宣称 active，必须是 real-chain evidence，而不是 fallback log
- `SWE-bench Verified 500` 必须存在可聚合、可回放、可失败归因的 formal evidence

### 13.6 Gate 1.0 的标准证据输出

为了让 `M7.6` 需求真正收口到统一内核，`Gate 1.0` 的标准证据应至少输出：

- `gate_matrix_summary.json`
  - 四模式 fresh run 结果
- `router_evidence.json`
  - `trace_id`
  - `router_model`
  - `selected_route`
  - `route_reason`
- `instance_evidence.json`
  - `instance_id`
  - `gateway_port`
  - `backend_port`
  - `ready_status`
  - `failure_reason`
- `fusion_evidence.json`
  - `candidate_instances`
  - `selected_instances`
  - `fusion_strategy`
  - `fallback_used`
- `runtime_evidence.json`
  - `state_kind`
  - `state_codec`
  - `resume_decode_executed`
  - `compression_ratio`
  - `cpu_copy_count`
  - `device_resume_consumed`
  - `speculative_algorithm`
  - `deepep_backend`
- `swe_verified_formal_summary.json`
  - `suite_name`
  - `suite_version`
  - `total_tasks`
  - `passed_tasks`
  - `failed_tasks`
  - `timeout_tasks`
  - `cancelled_tasks`
  - `fallback_tasks`
  - `per_task_evidence_paths`

### 13.7 双机 L20N + eRDMA 的 M7.6 Formal Gate 强约束口径

对于当前 `host1/host2 + L20N 72GB + eRDMA` 的双机环境，`M7.6 formal gate` 必须采用强约束口径，不接受“底层通信已通”被表述为“分布式 runtime 已正式成立”，也不接受“contract 已声明”被表述为“real-chain evidence 已成立”。

本节的唯一目的，是把当前双机 `DeepEP / NCCL / runtime evidence` 的正式边界固定为不可弱化、不可口头补齐、不可用 fallback 冒充 real-chain 的统一 gate 标准。

#### 13.7.1 强约束基本原则

对当前双机环境，必须采用以下不可放宽的解释规则：

- `eRDMA / RoCE / IB / GPUDirect RDMA` 已打通
  - 只代表跨机传输底座成立
  - 不代表 `distributed runtime = PASS`
  - 不代表 `DeepEP real-chain = PASS`
  - 不代表 `M7.6 formal gate = PASS`
- `NCCL` 可初始化
  - 只代表分布式 runtime 具备启动条件
  - 不代表 `NCCL real-chain` 已成立
- `DeepEP` 已写入配置、环境变量或 contract
  - 只代表目标路径被声明
  - 不代表 `DeepEP` 已真实 active
- `CUDA Graph` 已开启或未开启
  - 只属于性能/运行时观测项
  - 不构成 `M7.6 formal gate` 成败的直接判定条件
- `NCCL P2P`
  - 只属于单机内 peer communication 优化能力
  - 不得被拿来替代双机 `NCCL distributed runtime real-chain`
  - 更不得被拿来替代 `DeepEP real-chain`

一句话说：

```text
M7.6 formal gate 只认 real-chain runtime evidence，
不认“底层已通”、不认“配置已写”、不认“fallback 也能跑”。
```

#### 13.7.2 当前双机环境下，不可被宣称为 PASS 的情况

只要出现以下任一情况，就必须明确判为“未达到 `M7.6 formal gate`”：

1. 只有 `eRDMA / RDMA` 连通性证据，没有上层 distributed runtime 消费证据
2. 只有 `NCCL init` 或 bootstrap 成功，没有 request 级别或 runtime 级别 collective evidence
3. 只有 `requested_dispatch_backend = deepep` 的静态声明，没有 `deepep` active runtime evidence
4. 出现 `DeepEP runtime not found. Falling back to native SGLang routing.`
5. 存在 fallback，但未在正式 artifact 中显式披露
6. 只有 launch args / environment variable，没有 `effective_dispatch_backend`
7. 只有 contract patch，没有 request trace / dispatch trace / combine evidence
8. 只有 benchmark 数值，没有可回放的 runtime artifact
9. 只有单机内优化结果，没有双机 real-chain 证据
10. 只有 smoke 通过，没有 formal evidence 聚合结果

在上述任一情况下，允许的最强表述只能是：

- 底层互联已打通
- bootstrap 条件已部分满足
- contract 已具备目标路径声明
- 但尚未达到 `M7.6 formal gate` 正式通过标准

#### 13.7.3 当前双机环境下，M7.6 Formal Gate 的硬性必须项

若要对当前 `host1/host2 + L20N + eRDMA` 环境宣称进入 `M7.6 formal gate` 正式口径，则至少必须同时满足以下全部条件，缺一不可：

1. `hardware_topology` 已明确为双机多卡正式拓扑
   - 例如 `dual_node_8gpu`、`multi_node_tp_pp_ep` 或等价正式语义
2. `requested_distributed_runtime = nccl` 已显式进入正式 artifact
3. `requested_dispatch_backend = deepep` 已显式进入正式 artifact
4. `enable_nccl = true` 已显式进入正式 artifact
5. `NCCL` 已被真实消费，而不是仅可初始化
6. `effective_collective_backend` 已落盘且与实际运行一致
7. `effective_distributed_runtime` 已落盘且与实际运行一致
8. `DeepEP` backend 已真实 active，而不是 fallback
9. `effective_dispatch_backend` 已落盘且可证明为 `deepep`
10. request trace 中可区分 active backend、dispatch path 与 combine path
11. `report.json / summary.json / runtime_contract.json / publish_manifest.json / runtime_evidence.json` 已统一存在
12. 失败路径、fallback 路径与未命中路径均可归因，而不是只保留成功样例

只要上述任一项缺失，便不得宣称：

- `NCCL distributed runtime real-chain = PASS`
- `DeepEP real-chain = PASS`
- `M7.6 formal gate = PASS`

#### 13.7.4 双机环境下，哪些项是加分项，哪些项不是 gate 本体

对于当前双机 `L20N + eRDMA` 环境，以下能力可以增强性能、增强稳定性、增强观测完整性，但不应混淆为 `M7.6 formal gate` 的本体条件：

- `effective_cuda_graph`
- host 内 `NCCL P2P`
- host 内 `peer access`
- `compression_effective`
- 单机内 `p2pBandwidthLatencyTest`
- 独立 `nccl-tests` benchmark 分数

这些项目的正确定位是：

- 可以作为性能证据
- 可以作为优化证据
- 可以作为回归证据
- 但不能替代：
  - `NCCL distributed runtime real-chain`
  - `DeepEP dispatch real-chain`
  - `request trace observability`
  - `formal evidence aggregation`

#### 13.7.5 M7.6 Formal Gate 的最小证据闭环

对当前双机环境，`M7.6 formal gate` 至少必须形成如下证据闭环：

- `runtime_contract.json`
  - 明确声明：
    - `requested_distributed_runtime`
    - `requested_dispatch_backend`
    - `state_kind`
    - `state_codec`
    - `storage_backend`
- `runtime_evidence.json`
  - 明确落盘：
    - `effective_collective_backend`
    - `effective_dispatch_backend`
    - `effective_distributed_runtime`
    - `effective_cuda_graph`
    - `speculative_algorithm`
    - `deepep_backend`
    - `fallback_used`
- `report.json`
  - 作为单一真源汇总本轮结论
- `summary.json`
  - 给出最终 gate 判定与失败归因
- `publish_manifest.json`
  - 可回指本轮 artifact、runtime、contract 与证据路径
- `instance_evidence.json`
  - 若涉及多实例 / 多 runtime / 多 backend port，必须可区分实例身份与 ready 状态
- `fusion_evidence.json`
  - 若涉及 `FusionRoute`，必须能证明 candidate / selected / fallback 关系
- `router_evidence.json`
  - 若涉及 `MiniCPM5` route decision，必须可回放 route 证据链
- `swe_verified_formal_summary.json`
  - 若要正式宣称 `M7.6 formal PASS`，则必须存在批量 formal evaluation 聚合结果

缺失上述任一关键证据时，不得以日志片段、截图、命令行输出或口头说明替代。

#### 13.7.6 当前双机环境的唯一正确表述

因此，对当前 `host1/host2 + L20N 72GB + eRDMA` 环境，唯一允许的正式表述应严格分层如下：

- 若仅证明 `eRDMA / RDMA` 传输可用
  - 只能表述为：`transport foundation = PASS`
- 若进一步证明 `NCCL` 被真实消费
  - 才能表述为：`distributed runtime real-chain = PASS`
- 若进一步证明 `DeepEP` 真正 active 且无 fallback
  - 才能表述为：`DeepEP dispatch real-chain = PASS`
- 若进一步补齐 `router / instance / fusion / runtime / formal evaluation` 全部证据
  - 才能表述为：`M7.6 formal gate = PASS`

禁止继续使用以下模糊或错误说法：

- `eRDMA 通了，所以 M7.6 已过`
- `NCCL 能拉起来，所以 distributed runtime 已正式成立`
- `DeepEP 配上了，所以 real-chain 已成立`
- `CUDA Graph 没开，所以 formal gate 不成立`
- `NCCL P2P 没测，所以双机链路不成立`

#### 13.7.7 最终定义

对当前双机 `L20N + eRDMA` 路径，`M7.6 formal gate` 的最终定义必须固定为：

> `M7.6 dual-node formal ready = eRDMA transport foundation + NCCL distributed runtime real-chain + DeepEP dispatch real-chain + request-trace-visible runtime evidence + unified formal artifacts`

只有当上述五部分同时成立时，才允许对内或对外宣称：

> `当前 host1/host2 + L20N 72GB 环境已满足 M7.6 formal gate 的双机 runtime 基线`

否则，必须明确表述为：

> `当前仅达到底座可用、bootstrap 可用或 contract 已声明阶段，尚未进入 M7.6 formal PASS`

#### 13.7.8 本次 host1/host2 Formal PASS 回填

针对当前 `host1/host2 + L20N 72GB + eRDMA` 路径，已完成一次按正式入口执行的 `M7.5 active runtime -> M7.6 formal gate` 双机重跑，并形成可审计的正式 artifact。

本次双机正式产物路径如下：

- `host1`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m75_formal_20260624_host1/runtime_evidence/m75_trueorthokda_active_runtime.json`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m76_formal_20260624_host1/runtime_evidence/nvidia_runtime.json`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m76_formal_20260624_host1/m76_heterogeneous/m76_report.json`
- `host2`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m75_formal_20260624_host2/runtime_evidence/m75_trueorthokda_active_runtime.json`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m76_formal_20260624_host2/runtime_evidence/nvidia_runtime.json`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m76_formal_20260624_host2/m76_heterogeneous/m76_report.json`

为便于后续审计与本地复核，本轮还同步回收了本地镜像快照：

- `temp/misc/m76_formal_fetch_20260624/host1_m75_runtime.json`
- `temp/misc/m76_formal_fetch_20260624/host1_nvidia_runtime.json`
- `temp/misc/m76_formal_fetch_20260624/host1_m76_report.json`
- `temp/misc/m76_formal_fetch_20260624/host2_m75_runtime.json`
- `temp/misc/m76_formal_fetch_20260624/host2_nvidia_runtime.json`
- `temp/misc/m76_formal_fetch_20260624/host2_m76_report.json`

上述 artifact 已共同证明以下 formal 条件同时成立：

- `transport foundation = PASS`
  - `rdma_contract.status = PASS`
  - `rdma_available = true`
  - `send_ok = true`
- `NCCL distributed runtime real-chain = PASS`
  - `requested_distributed_runtime = nccl`
  - `enable_nccl = true`
  - `effective_collective_backend.status = PASS`
  - `effective_distributed_runtime.backend = nccl`
- `DeepEP dispatch real-chain = PASS`
  - `requested_dispatch_backend = deepep`
  - `effective_dispatch_backend.backend = deepep`
  - `deepep_real_chain_gate.status = PASS`
- `mandatory protocol gate = PASS`
  - `protocol_family = trueorthokda`
  - `state_codec = cq4`
  - `zero_copy_vram_real.status = PASS`
- `M7.6 formal gate = PASS`
  - `host1 m76_report.json: ok = true`
  - `host1 gate_result.m76.status = PASS`
  - `host2 m76_report.json: ok = true`
  - `host2 gate_result.m76.status = PASS`

因此，对当前已验收的双机路径，唯一允许的正式表述应更新为：

> `当前 host1/host2 + L20N 72GB + eRDMA 路径已具备 transport foundation、NCCL distributed runtime real-chain、DeepEP dispatch real-chain 与统一 formal artifacts，因而已满足 M7.6 formal gate = PASS`

#### 13.7.9 ColossalAI distributed runtime 在 M7.6 中的正式收口方式

对于 `ColossalAI`，当前正确的工程口径不应是“另起一条独立 gate”，而应是：将其作为 `requested_distributed_runtime` 的一个正式候选后端，统一纳入 `M7.6` 的 distributed runtime 收口体系。也就是说，`ColossalAI` 的正式定义、formal PASS 判定、enable/disable benchmark 与最终部署结论，均应以 `M7.6` 为唯一收口点；`M7.5 active runtime` 仍仅负责产出前置 runtime evidence，不单独承载最终部署结论。

对 `requested_distributed_runtime = colossalai` 的路径，若要宣称已进入 `M7.6 formal gate` 正式口径，则至少必须同时满足以下条件：

1. `M7.5 active runtime` 已按正式入口重跑，并在 artifact 中显式落出：
   - `requested_distributed_runtime = colossalai`
   - `use_colossalai = true`
   - `colossalai_plugin` 已明确记录
   - `colossalai_effective.status = PASS`
   - `effective_distributed_runtime.backend = colossalai`
2. `M7.6 runtime evidence` 已正式回收上述字段，并保持与 `M7.5` active runtime 一致：
   - `effective_distributed_runtime.status = PASS`
   - `effective_distributed_runtime.backend = colossalai`
   - `colossalai_effective.status = PASS`
3. 不得仅以 `import colossalai` 成功、bootstrap 初始化成功、或 contract 中声明了 `use_colossalai = true`，就表述为 `ColossalAI distributed runtime = PASS`

在此基础上，`ColossalAI enable/disable` 的性能比较也应统一纳入 `M7.6`，但这部分不应被混入 formal PASS 硬门槛。更准确的收口方式应为：

- `formal PASS` 只回答 `ColossalAI` 路径是否真正生效、是否具备完整 artifact 闭环
- `A/B benchmark` 再回答 `ColossalAI` 相对 `single_process` 或 `nccl` 是否带来可复现的性能/显存/稳定性收益
- `deployment positioning` 最后才回答该路径应被标记为 `supported but optional`，还是 `recommended default`

因此，对 `ColossalAI` 路径的 benchmark 设计，至少应满足以下约束：

- 必须在同一硬件、同一模型、同一实例数、同一权重路径、同一 `GDS` 状态下，执行 `enable/disable ColossalAI` 对照
- 至少应覆盖：
  - 单实例 cold start
  - 多实例启动时间
  - prefill / decode 吞吐
  - `P50 / P95` latency
  - GPU memory footprint
  - `FusionRoute 4 instance` 下的稳定性与尾延迟
- benchmark 的作用是回答“值不值得开”，不能反向替代 formal PASS 本身

基于当前仓库内已回收的 artifact，现阶段仍只能得出以下保守结论：

- `ColossalAI` 已被正式定义为 `M7.6` distributed runtime 的候选后端之一
- 当前已审计 artifact 仍以 `requested_distributed_runtime = single_process` 或 `nccl` 为主，尚未形成 `requested_distributed_runtime = colossalai` 且 `colossalai_effective.status = PASS` 的正式证据闭环
- 因此，当前尚不得把 `ColossalAI` 表述为已在 `M7.6` 上完成 formal PASS，更不得直接表述为默认推荐 runtime

对外可接受的唯一正确表述应为：

> `ColossalAI` 已在 `M7.6` 中被定义为正式候选 distributed runtime backend；其是否构成 formal PASS，取决于 `M7.5 active runtime -> M7.6 runtime evidence` 的完整闭环，而其是否值得默认开启，则必须以后续 `enable/disable` benchmark 结果为准。

### 13.8 当前状态的正确表述

当前最准确的结论不是“所有环境、所有拓扑、所有候选路由都已整体 formal PASS”，而是：

- 对已验收的 `host1/host2 + L20N 72GB + eRDMA` 这一路径，`Gate 1.0 Layer A / kernel smoke` 与 `Gate 1.0 Layer B / formal evidence` 已同时满足
- 对未执行同等级 rerun、未补齐同等级 artifact 的其他环境，仍不得直接套用本节 `PASS` 结论

因此，对内对外的正确措辞应为：

- 已完成统一内核四模式打通
- 已把 `M7.6` 的正式验收需求全部收口到 `Unified Pipeline Kernel Gate 1.0`
- 已在 `host1/host2 + L20N 72GB + eRDMA` 路径上形成：
  - `transport foundation PASS`
  - `NCCL real-chain PASS`
  - `DeepEP real-chain PASS`
  - `M7.6 formal gate PASS`
- 但仍不得把该结论无条件外推为：
  - 任意硬件拓扑均已 `formal PASS`
  - 任意 `FusionRoute` 候选路径均已 `formal PASS`
  - 任意 formal evaluation 套件均已自动 `PASS`

---

## 15. 与当前 NCCL 问题的关系

本次 `dual_node_8gpu` NCCL 二分结果，反而进一步证明统一 kernel 的必要性。

因为同一个最小 collective smoke，在不同 context 下表现如下：

- `single_node_8gpu + nccl + cuda`：通过
- `dual_node_1gpu + nccl + cuda`：通过
- `dual_node_8gpu + nccl + cuda`：失败
- `dual_node_8gpu + gloo + cpu`：通过

这说明分歧真正来自：

- `hardware_platform`
- `hardware_topology`
- `environment`
- collective strategy

而不是来自“这是 CGC 还是 Megatrain”。

因此：

- `DistributedPlugin`
- `ExecutionContext.hardware_platform`
- `ExecutionContext.hardware_topology`
- `ExecutionContext.environment`

必须成为统一 kernel 的一等输入。

---

## 15. 一句话结论

`CGC engine pipeline` 与 `Megatrain pipeline` 应整合为同一个 `Unified Pipeline Kernel`。

二者差异不应再表现为两套主流程，而应表现为：

- 相同 kernel
- 不同 `ExecutionContext`
- 不同 `ABI decision`
- 不同 `Strategy/Profile`

其中：

- `CGC Engine` 是平台名
- `Megatrain` 是 train profile

这将把当前系统从“多条能跑的功能线”，升级为“可扩展、可平台化、可统一验证的执行底座”。
