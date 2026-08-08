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

#### 4.1.1 正式三层配置分工

为了避免 `profile`、`bootstrap` 与系统拓扑变量继续散落在不同 gate 与不同 helper 中，统一 kernel 现在应明确采用三层正式配置分工：

- `Environment Bootstrap`
  - 回答“当前 runtime 有什么能力”
- `Profile Setting + Binding`
  - 回答“当前任务应该怎么跑”
- `Model Setting / System Profile`
  - 回答“当前系统到底由谁组成、如何编组、哪些组件是正式 required variable”

三层职责必须严格分离：

- `Environment Bootstrap`
  - 负责 `backend / protocol_family / state_kind / state_codec / requested_dispatch_backend / requested_distributed_runtime / requested_storage_backend / enable_pd / enable_nccl / enable_cuda_graph`
- `Profile Setting + Binding`
  - 负责 `profile_settings.json` 与 `execution_profile_binding / delivery_profile_binding / bootstrap_contract_binding / flow_parameter_contract_binding`
- `System Profile`
  - 负责 `DeepSeek-V4-Flash / MiniCPM5 / FusionRoute` 等系统主变量、实例数、角色矩阵与路由拓扑

换句话说：

- `bootstrap` 不是系统拓扑真源
- `binding` 不是系统拓扑真源
- `system_profile` 才是所有 gate 一致消费的正式系统变量

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
- `system_profile_ref`
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
- `component_id / component_role`
- `runtime_profile`
- `distributed_backend`
- `parallel_tp_size / parallel_pp_size / parallel_ep_size`
- `system_profile_ref`
- `abi_descriptor`

这里需要特别强调：

- `ExecutionContext` 可以携带 `system_profile_ref`
- `ExecutionContext` 不应完整复制 `system_profile`
- 完整 `system_profile` 应以 `system_execution_manifest.json` 为系统级单一真源

推荐最小引用结构如下：

- `system_profile_ref.profile_id`
- `system_profile_ref.profile_version`
- `system_profile_ref.source`
- `system_profile_ref.source_path`
- `system_profile_ref.llm_component_family`
- `system_profile_ref.router_component_family`
- `system_profile_ref.gateway_component_family`

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

在工程落地中，还必须再引入一个高于“执行矩阵”的正式系统层输入：

- `System Profile`

二者职责不同：

- `State ABI`
  - 回答“模型状态语义是否兼容”
- `统一执行矩阵`
  - 回答“当前应以什么执行策略运行”
- `System Profile`
  - 回答“当前系统由哪些正式组件组成，以及这些组件如何形成稳定的系统主变量”

推荐统一理解为：

```text
ABI 决定兼容边界
统一矩阵决定执行策略
System Profile 决定系统拓扑真源
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

### 7.1 主契约 vs 实作层

为了避免把 runtime contract 和阶段性优化实现混为一谈，统一 kernel 需要明确区分“主契约层”与“实作层”：

| 层级 | 当前正式对象 | 角色 | 当前代码/文档落点 |
|---|---|---|---|
| 主契约层 | `State ABI` | 负责语义兼容、runtime branch、合法 payload 判定 | `docs/deepseek-v2-to-v4-state-abi-whitepaper-v1.md` |
| 主契约层 | `bridge artifact + edge-cloud protocol` | 负责 train profile 到 edge delivery 的正式交付载体，并向下拆分为 `publish_manifest / runtime_contract / bridge_info / deploy_contract / consume_contract` | `publish_manifest / runtime_contract / bridge_info` |
| 主契约层 | `TrueOrthoKDA active runtime` | 负责真 state transport、edge resume decode、compression effective | `app/servers/cloud_socket_server.py`、`app/edge_engine/local_infer.py`、`app/edge_engine/kda_state_runtime.py` |
| 主契约层 | `zero-copy / VRAM handoff evidence` | 负责证明 edge 端不是 fake resume，而是 runtime 真正消费 state | `m75_trueorthokda_active_runtime.py` 与 `M7.5`/`M7.6` gate evidence |
| 实作层 | `CQ4 / FusedCQ4BitPass` | 压缩与带宽优化实现之一，可替换，不应上升为唯一契约名词 | `cgc_engine/cgc/fused_compression_pass.py`、`Backend/CGC/kv_compressor.py` |
| 实作层 | `DeltaMem / UMA hook / RDMA passthrough` | 特定平台上的 0-copy / handoff 实现手段，不等于正式协议语义 | `cgc_metal_vram_hook.mm`、`network/rdma_passthrough.py` |

这张表的核心目的是：

- 主契约层负责定义“系统必须成立什么”
- 实作层负责回答“当前版本用什么方法把它做出来”
- 未来可以替换 `CQ4`、`DeltaMem`、特定 codec 或 transport backend，但不能绕过 `State ABI`、`bridge artifact`、`TrueOrthoKDA active runtime` 与 `zero-copy evidence`

在 `UPKG 4.0 embodied` 当前收口实现中，这个“主契约层”已经进一步细化为统一 descriptor 平面，并直接落到正式 artifact：

- `execution_profile_descriptor`：负责声明当前链路属于 `local_infer / local_train / edge_cloud_infer / edge_cloud_train` 中哪一种 canonical execution profile。
- `delivery_profile_descriptor`：负责声明 cloud 到 edge、或 local runtime 内部 handoff 的正式交付关系。
- `bootstrap_contract_descriptors`：负责声明 runtime 启动、distributed bootstrap、bridge activate、local runtime activate 等前置契约步骤与参数。
- `flow_parameter_contract_descriptors`：负责声明 train / deploy / consume / infer 各阶段必需输入、必需输出与流程参数约束。

这意味着 `bootstrap*` 以及不同 `local infer/train`、`edge_cloud infer/train` 所使用的定制化流程参数，不应再停留在脚本参数或 helper 约定，而应统一成为 ABI/CGC 输入输出边界的一部分。

当前 `UPKG 4.0` 已验证落地的正式 contract artifact 包括：

- `psi0_runtime_contract`
- `psi0_deploy_contract`
- `realtime_vla_consume_contract`
- `realtime_vla_edge_inference_contract`
- `cloud_ingest_manifest`

这些 artifact 已经实际带出：

- `distributed_runtime_bootstrap_path`
- `training_stage_scope`
- `distributed_backend`
- `delivery_channel`
- `transport_strategy`
- `selected_route`
- `selected_backend_family`
- `edge_model`

但需要明确区分：

- 这说明“描述性产出物与契约层”已经收口。
- 这不等于 `Stage1-3` 真 full training、真 full-weight publish、真 edge deploy 已全部完成。

### 7.2 Unified Pipeline Kernel Gate 1.0（UPKG_1.1）

为了避免 `M1-M7.5` 的 gate 重跑结果继续受“模型资产缺失”“fingerprint lock 未落地”“平台适配版本漂移”影响，统一 kernel 需要把 `UPKG_1.1` 明确定义为正式可验收要求，而不是临时操作经验。

`UPKG_1.1` 的目标不是新增另一套 gate，而是把 `Unified Pipeline Kernel` 在 `M1-M7.5` 范围内的最小可复现前提、验收边界与失败归因口径写成统一规范。

#### 7.2.1 最小前置条件

- 必须存在至少一个可用的本地 `GGUF` 模型，且放在 `ComputeGraphCompiler-main/Output/Models/` 下，不能只保留历史 report 中的旧路径引用。
- 必须先生成并显式提供 `backend fingerprint lock`，不能依赖未落地的默认 `backend_fingerprint.lock.json`。
- `M1/M2/M3/M5` 的重跑必须使用“先生成建议 lock，再用该 lock 复跑”的严格流程，避免因为 lock 缺失或陈旧而把环境问题误判为 kernel 失败。
- `M5` 所依赖的 `oMLX / FlashMoE / mlx_lm` 运行时版本必须与当前 fallback 接口兼容；若 `GenerationBatch` 等符号不兼容，应归类为平台适配失败，而不是 `Unified Kernel` 语义失败。
- `M4` 必须在满足训练性能证据、编译产物证据、以及分布式证据的前提下验收；单机 `world_size=1` 只能作为 smoke，不得视为 `M4 PASS`。
- `M7 / M7.1` 必须能同时给出 `dynamic trace / state compression / replay / audit` 四类核心证据。
- `M7.2 / M7.3` 必须能从 `M7` core artifact 延伸出场景化 report，不能只保留白皮书口径。
- `M7.4 / M7.5` 必须同时保留 contract check 与 runtime evidence；若只有 contract marker 而无 evidence，不得视为正式 `PASS`。

#### 7.2.2 M1-M7.5 验收矩阵

| Milestone | Unified Kernel 关注点 | `UPKG_1.1` 必须成立的 gate 条件 |
|---|---|---|
| `M1` | 八步主干在本地 `native` 路径可跑通 | `GGUF` 可用、fingerprint lock 生效、`step1-step8` 可完整落地，不允许因人工中断或缺资产导致假失败 |
| `M2` | inject/compile 前的策略收敛与 gate 包装成立 | 允许 `pipeline_gate_result` 为 `SKIP(use_gate=false)`，但 `pipeline_ok` 必须为 `true`，且 fingerprint lock 必须来自本次重跑 |
| `M3` | bundle/export 能从统一 kernel 主干导出 | 除 `M2` 条件外，必须生成 bundle/export 产物，且 report 需要可回溯到本次重跑输出目录 |
| `M4` | train/infer 双路在统一 kernel 下被同一 gate 聚合 | training 至少同时满足 `performance_gate`、`compile_gate`、`distributed_gate`；缺任一项都应判定为 `FAIL`，不得降级为通过 |
| `M5` | fullgraph/AOT 与 runtime deploy 能接到统一 kernel 交付面 | 允许低内存 Mac 走经验证的 `oMLX fallback`，但前提是运行时依赖版本兼容，且 AOT/bench/deploy 证据必须完整 |
| `M7` | 工业级 unified kernel 验证总入口 | `dynamic_trace`、`state_compression`、`replay`、`audit` 必须同时 `PASS` |
| `M7.1` | `M7` 的核心内核层 | 当前版本与 `M7` 共用 core artifact，但必须明确包含 `compile_success_rate`、`cache_hit_rate`、`compression_ratio`、`hash chain` |
| `M7.2` | 数字世界 GUI Agent 工业验收 | `dynamic_trace_l1`、`soft_rt_replay`、`state_compression`、`industrial_audit` 必须全部 `PASS` |
| `M7.3` | 物理具身智能端云桥接验收 | `cloud_training_psi0`、`edge_inference_bridge`、`state_compression`、`industrial_audit` 必须全部 `PASS` |
| `M7.4` | `dflash + TrueOrthoKDA` 合同与 runtime evidence 验收 | `dflash_contract`、`trueorthkda_contract`、`trueorthkda_runtime`、`edge_runtime_evidence` 必须全部 `PASS` |
| `M7.5 API compat` | 协议相容、分布式交付与 edge router 交付面 | `api_surface`、`tool_call_hotfix`、`local_loopback`、`client_entrypoints`、`distributed_runtime_evidence`、`edge_router_runtime_evidence`、`edge_router_cluster_nfs_evidence`、`extreme_scale_runtime_evidence` 必须全部 `PASS` |
| `M7.5 active runtime` | 真 state transport / edge resume / zero-copy 证据 | `report_schema`、`true_state_transport`、`edge_state_resume_decode`、`runtime_evidence` 必须全部 `PASS` |

#### 7.2.3 统一失败归因规则

- 缺少 `Output/Models/*.gguf`：归因为“验收资产未就绪”，不是 kernel 设计失败。
- 缺少 `backend fingerprint lock` 或 lock 与本次产物不匹配：归因为“环境锁定未闭环”，不是 ABI/strategy 失败。
- `M4` 中 `world_size<=1`、`mfu` 缺失、训练速度未达阈值：归因为“分布式与性能证据未成立”，必须阻断 `UPKG_1.1` 通过。
- `M5` 中 `mlx_lm` API 漂移或 fallback 运行时导入失败：归因为“平台适配层版本不兼容”，必须在 gate 中单独暴露。
- `M7 / M7.1` 中 `dynamic trace / replay / audit` 任一项缺失：归因为“工业级 kernel 证据未闭环”，不得降级为 PASS。
- `M7.2 / M7.3` 缺少场景化 report 或只保留 bootstrap 文本说明：归因为“场景化验收未成立”。
- `M7.4 / M7.5` 仅有 contract marker、没有 runtime evidence：归因为“契约存在但 runtime 证据未成立”。

#### 7.2.4 文档与产物要求

- `UPKG_1.1` 的正式口径放在本文件，不再分散写在零散 gate 说明或临时执行脚本里。
- 后续每次重跑 `M1-M7.5` 时，必须同时保存：
  - 本次使用的 `GGUF` 路径
  - 本次生效的 `backend fingerprint lock` 路径
  - 每个 milestone 的 `report.json`
  - 对 `PASS / FAIL / SKIP` 的统一归因
- 若包含 `M7 / M7.1`，必须额外保存：
  - `dynamic_trace_l1`
  - `state_compression_summary`
  - `soft_rt_replay`
  - `industrial_audit`
- 若包含 `M7.5 active runtime`，必须额外保存：
  - `runtime_evidence_path`
  - `state_kind`
  - `state_codec`
  - `compression_ratio`
  - `cpu_copy_count`
- `State ABI`、`bridge artifact + edge-cloud protocol`、`TrueOrthoKDA active runtime` 仍是上位主契约；`UPKG_1.1` 负责的是统一 kernel 在 `M1-M7.5` 的工程验收，不替代主契约层。

#### 7.2.5 当前正式验收证据（2026-06-19）

`UPKG_1.1` 在当前版本的正式收口，不再只依据抽象规则，而是明确绑定下列可回放 artifact：

- `M4 distributed training route`
  - 证据 report：`temp/misc/host1_m4_training_route/rank0_report_20260619T113816Z.json`
  - 已成立证据：
    - `distributed_init.status = PASS`
    - `world_size = 2`
    - `backend = nccl`
    - `step2_graph_capture.compile_wrapper = ddp_unwrapped`
    - `step6_dispatch.status = PASS`
    - `step7_compare.performance_gate.status = PASS`
    - `speedup = 2.2217131304915823`
- `M5 canonical fullgraph/AOT route`
  - 证据 report：`/tmp/cgc_upkg_fix_20260619_m5/pass2/report.json`
  - 已成立证据：
    - `backend_fingerprint_gate.status = PASS`
    - `step2_fullgraph_capture.status = PASS`
    - `step6_fullgraph_compile.status = PASS`
    - `step7_fullgraph_bench.status = PASS`
    - `step8_fullgraph_deploy.status = PASS`
    - `m5.aot_precompile_gate.status = PASS`
    - `provider = omlx_dflash`
    - `omlx_fallback.engine = dflash`
    - `omlx_fallback.compile_mode = omlx_dflash`
- `M7 / M7.1 core`
  - 证据 report：`ComputeGraphCompiler-main/Output/cli_gate_m7_series_20260619/m7_industrial/m7_report.json`
  - 已成立证据：
    - `dynamic_trace.status = PASS`
    - `compile_success_rate = 1.0`
    - `cache_hit_rate = 1.0`
    - `state_compression.status = PASS`
    - `compression_ratio = 0.012370768213415912`
    - `replay.status = PASS`
    - `audit.status = PASS`
- `M7.2 GUI agent route`
  - 证据 report：`ComputeGraphCompiler-main/Output/cli_gate_m7_series_20260619/m72_gui_agent/report.json`
  - 已成立证据：
    - `dynamic_trace_l1 = PASS`
    - `soft_rt_replay = PASS`
    - `state_compression = PASS`
    - `industrial_audit = PASS`
- `M7.3 physical bridge route`
  - 证据 report：`ComputeGraphCompiler-main/Output/cli_gate_m7_series_20260619/m73_physical/m73_report.json`
  - 已成立证据：
    - `cloud_training_psi0.status = PASS`
    - `edge_inference_bridge.status = PASS`
    - `edge_inference_bridge.edge_latency_ms = 4.151859375269851`
    - `state_compression.status = PASS`
    - `industrial_audit.status = PASS`
- `M7.4 dflash + TrueOrthoKDA verification route`
  - 证据 report：`ComputeGraphCompiler-main/Output/cli_gate_m74/m74_dflash_kda/m74_report.json`
  - 已成立证据：
    - `dflash_contract = PASS`
    - `trueorthkda_contract = PASS`
    - `trueorthkda_runtime = PASS`
    - `edge_runtime_evidence = PASS`
- `M7.5 API compatibility route`
  - 证据 report：`ComputeGraphCompiler-main/Output/cli_gate_m75/m75_api_compat/m75_report.json`
  - 已成立证据：
    - `api_surface = PASS`
    - `tool_call_hotfix = PASS`
    - `local_loopback = PASS`
    - `client_entrypoints = PASS`
    - `distributed_runtime_evidence = PASS`
    - `edge_router_runtime_evidence = PASS`
    - `edge_router_cluster_nfs_evidence = PASS`
    - `extreme_scale_runtime_evidence = PASS`
- `M7.5 TrueOrthoKDA active runtime route`
  - 证据 report：`ComputeGraphCompiler-main/Output/cli_gate_m75_trueorthokda_active/m75_trueorthokda_active/m75_trueorthokda_active_report.json`
  - 已成立证据：
    - `report_schema = PASS`
    - `true_state_transport = PASS`
    - `state_kind = kda_state_v1`
    - `state_codec = zlib_torch_save_bytes`
    - `edge_state_resume_decode = PASS`
    - `runtime_evidence = PASS`
    - `compression_ratio = 0.9254533979812027`
    - `cpu_copy_count = 0`
    - `uma_buffer_used = true`
    - `device_resume_consumed = true`

#### 7.2.6 当前版本收口说明

- `M4` 的正式口径以 `host1` 真分布式训练证据为准，不再接受 `world_size=1` 的本地 smoke 代替正式 distributed gate。
- `M5` 的正式口径以 `pass2` canonical report 为准；先前同路径出现过 `KeyboardInterrupt`，但该次中断不再作为失败依据，最新 rerun 已补成正式 `PASS`。
- 对低内存 Mac，`M5` 允许接受经验证的 `oMLX + dflash` fallback，但必须同时满足：
  - `bench_smoke PASS`
  - `fullgraph compile PASS`
  - `deploy PASS`
  - `aot_precompile_gate PASS`
- 若 `oMLX` runtime 内部存在调试探针、观测回报或外部 debug server 依赖，这些机制必须 `fail-open`，不得反向影响 `UPKG_1.1` 的正式 gate 结果。
- `M7 / M7.1` 当前以单份 core artifact 同时承接“总入口 + 核心层”语义，后续若拆分 report，不得弱化现有 PASS 条件。
- `M7.2 / M7.3` 当前都已有 fresh rerun artifact，因此 `UPKG_1.1` 不再把它们视为“未来工作项”，而是当前版本正式交付面的一部分。
- `M7.4 / M7.5` 当前都要求 contract 与 runtime evidence 双闭环；只靠 marker/说明文字不得判定通过。

### 7.3 Unified Pipeline Kernel Gate 2.0（UPKG 2.0）

在 `UPKG_1.1` 已经把 `M1-M7.5` 的工程验收口径收口之后，下一阶段首先要单独定义：

```text
Unified Pipeline Kernel Gate 2.0（UPKG 2.0）
```

`UPKG 2.0` 的定位明确为：

- `通用模型产品化 gate`
- 把统一 kernel 从“工程可重跑”提升到“模型可交付、可发布、可治理”
- 把模型产物、模型 contract、ABI 判定、runtime branch 选择、交付归因写成正式产品化口径

`UPKG 2.0` 关注的核心不再只是 milestone 是否 `PASS`，而是模型层是否具备正式交付条件，至少包括：

- 模型产物是否可稳定导出、可稳定发布、可稳定追溯
- `State ABI` 与 `runtime_branch_required` 是否已经进入正式模型交付契约
- `publish_manifest / runtime_contract / model summary` 是否已具备统一模型口径
- 失败是否能归因为模型资产、模型映射、ABI、runtime branch、provider contract，而不是停留在脚本层解释

`UPKG 2.0` 明确不负责：

- 通用 agent 产品化 gate
- 六元一体通用审计与归因框架
- `realtime-vla`、官方 `psi0 comparative`、具身 benchmark

这意味着版本边界应明确区分为：

- `UPKG_1.1`：`M1-M7.5` 的工程验收 gate
- `UPKG 2.0`：通用模型产品化 gate
- `UPKG 3.0`：通用 Unified Pipeline Kernel Agent 产品化 gate
- `UPKG 4.0`：具身 runtime / comparative / benchmark gate

### 7.4 Unified Pipeline Kernel Gate 3.0（UPKG 3.0）

在 `UPKG_1.1` 已经把 `M1-M7.5` 的最小正式 gate 口径收口之后，下一阶段不应继续把所有新增诉求都塞回 `UPKG_1.1`，而应把通用产品化要求单独提升为：

```text
Unified Pipeline Kernel Gate 3.0（UPKG 3.0）
```

`UPKG 3.0` 的定位明确为：

- `通用 Unified Pipeline Kernel Agent 产品化 gate`
- 把 `M7 / M7.1 / M7.2 / M7.3` 既有的 `audit / replay / trace / bridge` 升级为产品级验收
- 把 `agent + edge + runtime` 的统一 artifact、summary、failure attribution 写成正式 contract
- 把六元一体架构先收敛为通用 agent 场景下的产品化审计与归因框架

`UPKG 3.0` 的独立 gate 白皮书为：

- `docs/gate_whitepapers/CGC_UPKG_3_0_AGENT_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md`
  - 负责 `3.1-3.7` 的详细 gate 定义、最小产物、最小 PASS 条件、失败归因与第一批实施接线

`UPKG 3.0` 明确不承接以下具身专属要求，这些内容统一后移到 `UPKG 4.0`：

- `realtime-vla` 作为具身 runtime host 的正式验收
- 官方 `psi0` 训练+推理对照基线
- `无 realtime-vla` 与 `有 realtime-vla` 的正式 comparative
- `>5x` 的明确 benchmark threshold
- `psi0 feedback`、`view-invariance`、`one-shot`、`structured conditioning smoke test`、真实 `atom library` 挂载等具身深化条目

#### 7.4.1 Kernel Core Product Gate

`3.1 Kernel Core Product Gate` 负责把 `M7 / M7.1` 的核心 kernel 证据升级为产品级 gate，至少要求：

- `dynamic_trace`
- `soft_rt_replay`
- `state_compression`
- `industrial_audit`

这一层回答的是：

- kernel 是否稳定可重放
- trace / replay / compression / audit 是否能作为对外产品验收项
- 失败时是否能统一归因为 `compile`、`cache`、`replay`、`audit chain` 或 `state codec`

建议正式要求进一步细化为：

- **对应来源**
  - 继承 `M7 / M7.1` 的 core artifact
  - 以 `m7_report.json` 及等价 core report 为最小正式来源
- **必须产物**
  - `report.json`
  - `dynamic_trace_l1`
  - `state_compression_summary`
  - `soft_rt_replay`
  - `industrial_audit`
  - `events.jsonl / chain_head.json`
- **PASS 条件**
  - `compile_success_rate = 1.0`
  - `cache_hit_rate` 必须达到可接受产品阈值，当前建议不低于 `2/3`
  - `soft_rt_replay.status = PASS`
  - `industrial_audit.event_integrity = 1.0`
  - `industrial_audit.hash_chain_valid = 1.0`
- **失败归因**
  - `compile_failure`
  - `cache_instability`
  - `state_codec_failure`
  - `replay_deadline_failure`
  - `audit_chain_break`

#### 7.4.2 Agent Runtime Gate

`3.2 Agent Runtime Gate` 负责把 `M7.2` 的 GUI/agent 路线，从“场景化 gate PASS”提升为通用 agent 运行时产品化要求，至少要求：

- `agent workflow` 可进入统一 kernel
- tool / GUI / runtime execution 能产出连续 trace
- runtime evidence 与 stage trace 可统一回放
- 失败能归因为 `workflow`、`runtime host`、`tool execution`、`environment`、`state handoff`

这一层的目标不是只证明 agent 能跑，而是证明 agent route 已经具备正式产品运行时语义。

建议正式要求进一步细化为：

- **对应来源**
  - 继承 `M7.2` 的 GUI/agent route
  - 允许扩展到非 GUI 的通用 agent route，但必须共享同一套 `ExecutionContext`
- **必须产物**
  - `report.json`
  - `summary`
  - `stage_trace.jsonl`
  - `runtime evidence`
  - tool / workflow / runtime host 的结构化事件
- **PASS 条件**
  - `dynamic_trace_l1 = PASS`
  - `soft_rt_replay = PASS`
  - `industrial_audit = PASS`
  - agent route 至少完成一次真实 workflow dispatch，而不是只停在 smoke 初始化
  - tool 调用、GUI 事件、runtime host 事件三类 trace 必须可串接
- **失败归因**
  - `workflow_plan_failure`
  - `tool_execution_failure`
  - `runtime_host_failure`
  - `environment_not_ready`
  - `state_handoff_failure`

#### 7.4.3 Edge Bridge Product Gate

`3.3 Edge Bridge Product Gate` 负责把 `M7.3` 中可通用化的 bridge 与交付 contract 收进产品级 gate，至少要求：

- `publish_manifest`
- `runtime_contract`
- `bridge_info`
- `edge delivery evidence`
- `bridge publish evidence`

本节只承接：

- 通用 `bridge contract`
- 通用 `edge delivery`
- 通用 `artifact publish`
- 通用 `runtime evidence`

本节明确不承接：

- 具身专属 runtime host 比较
- 官方 `psi0` comparative
- `realtime-vla` benchmark

这样才能保证 `UPKG 3.0` 与 `UPKG 4.0` 的边界不重叠。

建议正式要求进一步细化为：

- **对应来源**
  - 继承 `M7.3` 中已经成立的 `bridge artifact + edge-cloud protocol` 主干
  - 兼容通用 edge delivery，不限定为具身 publish path
- **必须产物**
  - `publish_manifest.json`
  - `runtime_contract.json`
  - `bridge_info.json`
  - `edge delivery evidence`
  - `bridge publish evidence`
  - 至少一份可回放的 `summary`
- **PASS 条件**
  - `publish_manifest / runtime_contract / bridge_info` 三者必须共享同一套 `matrix_axes`
  - `edge delivery` 必须对应到明确 runtime target，而不是只存在导出包
  - bridge 交付失败不得被伪装成 runtime 成功
  - 所有交付产物必须能从 `report.json` 或 `summary` 反向索引
- **失败归因**
  - `publish_manifest_incomplete`
  - `runtime_contract_incomplete`
  - `bridge_info_incomplete`
  - `edge_delivery_failure`
  - `matrix_axes_mismatch`

#### 7.4.4 Unified Artifact And Summary Gate

`3.4 Unified Artifact And Summary Gate` 负责把 `agent + edge + runtime` 的所有关键产物统一为可比较、可检索、可归因的正式单一真源，至少要求：

- `report.json`
- `summary`
- `stage_trace.jsonl`
- `matrix_axes`
- `failure attribution`
- `edge_inference_result.json`
- `replay_anchor.json`
- `reward_trace.json`
- `cloud_ingest_manifest.json`
- `cloud_summary.json`

这一层的核心目标是：

- 所有产品级 route 都必须能回写统一 artifact
- `pipeline report / publish_manifest / runtime_contract / summary` 必须共享同一套 `matrix_axes`
- `PASS / FAIL / SKIP` 的失败归因必须是结构化字段，而不是散落在脚本输出或人工备注里
- `cloud_summary.json` 必须作为云端聚合单一真源，反向索引 edge 侧 `report / summary / runtime_evidence / replay_anchor / reward_trace`

建议正式要求进一步细化为：

- **对应来源**
  - 继承当前 `ExecutionContext / matrix_axes / summary report / stage_trace` 主干
  - 面向所有 agent + edge + runtime route，而不是单一 milestone
- **必须产物**
  - `report.json`
  - `summary.json`
  - `stage_trace.jsonl`
  - `failure_attribution`
  - `matrix_axes`
  - artifact path index
  - `edge_inference_result.json`
  - `replay_anchor.json`
  - `reward_trace.json`
  - `cloud_ingest_manifest.json`
  - `cloud_summary.json`
- **PASS 条件**
  - `cloud_summary.json` 必须是云端聚合单一真源
  - 本地 `report.json / summary` 必须作为 edge source 被 `cloud_ingest_manifest.json` 反向索引
  - `stage_trace` 必须覆盖从 dispatch 到 combine 的关键阶段
  - `failure attribution` 必须结构化到可机读字段，而不是自由文本
  - artifact path index 必须可反向索引 publish / runtime / audit / edge-to-cloud return 产物
- **失败归因**
  - `missing_single_source_of_truth`
  - `stage_trace_incomplete`
  - `matrix_axes_missing`
  - `artifact_index_missing`
  - `failure_attribution_missing`

#### 7.4.5 Six-Element Audit And Attribution Gate

`3.5 Six-Element Audit And Attribution Gate` 负责把六元一体架构先落成通用 agent 场景下的产品化审计框架，而不是直接跳到具身特化 benchmark。

在 `UPKG 3.0` 中，六元建议先统一抽象为：

- `模型元`
- `工作流元`
- `运行环境元`
- `感知 / 界面元`
- `执行元`
- `全局记忆元`

这一层至少要求：

- 六元事件可统一入链
- 六元状态可统一回放
- 六元失败可统一归因
- hash / audit continuity 不因跨元件流转而断裂

也就是说，`UPKG 3.0` 里的六元一体，先解决：

- 是否能进入同一套 `ExecutionContext`
- 是否能进入同一套 artifact / summary
- 是否能进入同一套 audit / replay / attribution

而不是提前把所有具身专属 runtime / controller / feedback benchmark 一次性塞入同一版 gate。

建议正式要求进一步细化为：

- **六元最小映射**
  - `模型元`：模型调用、状态读写、模型分支切换
  - `工作流元`：规划、节点跳转、工具编排
  - `运行环境元`：硬件、镜像、依赖、fingerprint
  - `感知 / 界面元`：GUI、screen、输入采集或等价事件
  - `执行元`：tool call、动作 dispatch、runtime host 实际执行
  - `全局记忆元`：state snapshot、cache、long-lived memory handoff
- **必须产物**
  - 六元事件分类后的 `events.jsonl`
  - audit hash chain
  - attribution summary
  - replay anchor
  - `gui_source_registry.json`
  - `gui_stage_bindings.json`
  - `gui_operator_graph.json`
  - `gui_execution_context.json`
- **PASS 条件**
  - 六元中不能只记录部分事件而把剩余路径留在黑盒日志
  - 任一失败必须能定位到至少一个元与其上游/下游依赖
  - replay 必须能重建最小业务路径，而不是只回放单点 kernel 事件
  - audit chain 不能因跨元流转而中断
- **失败归因**
  - `model_element_missing`
  - `workflow_element_missing`
  - `environment_element_missing`
  - `perception_element_missing`
  - `execution_element_missing`
  - `memory_element_missing`
  - `cross_element_chain_break`

当前工程收口可进一步分成两层：

- `gate-native`
  - 已进入可运行阶段，`GUI route` 已可进入 `pipeline report`、`m72 gate` 与 `cgc run` artifact
- `graph-native`
  - 已完成 `source registry / stage bindings / operator graph / execution context`
  - 但仍属于 `partial` 状态，因为 `native operator execution` 尚未在每个 stage 全面打开

#### 7.4.6 Missing Capability Closure Gate

`3.6 Missing Capability Closure Gate` 负责把当前 `UPKG 3.0` 尚未落地、但已明确识别出的关键缺口统一收口为一个可追踪、可分批实现、可独立验收的 closure gate。

这一层至少要求：

- 所有已识别缺口必须进入统一 `gap register`
- `workflow DAG -> trajectory synthesis -> fine-tune -> dual-mode governance -> audit alignment` 必须给出一版完整 spec
- `3.1-3.5` 的未完成项必须可映射到统一 `closure plan`
- `GUI source` 从 `graph-bound route` 走向 `graph-native operator execution` 的缺口也必须进入统一 closure plan

建议统一失败归因为：

- `gap_register_missing`
- `closure_plan_missing`
- `workflow_dag_schema_missing`
- `trajectory_synthesis_spec_missing`
- `fine_tune_profile_missing`
- `dual_mode_governance_missing`
- `audit_alignment_spec_missing`
- `unmapped_gap_item`

#### 7.4.7 Cloud-Edge Training And Inference Q2RL Gate

`3.7 Cloud-Edge Training And Inference Q2RL Gate` 负责把 `GUI agent` 训练后模型正式纳入 `云侧训练 / 端侧推理` 产品模式，并要求 `CGC Engine` 以 `CGC Unified Pipeline Kernel Design v1.0` 为约束底座，通过 `Q2RL` 承接云侧后训练，同时允许端侧通过 `CLI / cgc run / 其他命令入口` 发起执行与回传结果。

这一层至少要求：

- 必须明确定义 `cloud_train -> publish -> edge_delivery -> edge_infer` 的正式模式
- `GUI agent` 训练后的模型必须可经 `publish_manifest / runtime_contract / state_abi` 下推到端侧，由 `CGC Engine` 承载推理
- `Q2RL` 必须明确属于云侧训练/后训练，并对 `workflow / tool_call / runtime_host / screenshot / replay` 建立 reward source 绑定
- 端侧必须可通过 `CLI / cgc run / 其他命令入口` 发起推理、生成证据并回传云端聚合层
- 端侧部署 bundle 必须至少包含：
  - `trained_weights`
  - `state_abi_contract`
  - `runtime_contract`
  - `publish_manifest`

建议正式产物至少包括：

- `cloud_edge_training_inference_mode.json`
- `gui_agent_edge_inference_contract.json`
- `q2rl_post_training_profile.json`
- `edge_deployment_bundle_manifest.json`
- `cloud_edge_q2rl_evaluation_plan.json`

建议正式执行入口：

- `cgc gate m77`
- `cgc gate upkg37`
- `cgc_engine/agent/cli.py pipeline --milestone m77`
- `cgc_engine/agent/cli.py pipeline --milestone upkg37`

#### 7.4.8 Teaching Mode And Pure LLM Six-Element Inference Gate

`3.8 Teaching Mode And Pure LLM Six-Element Inference Gate` 负责把 `GUI agent` 示教、示教数据训练模型、端云下推与端侧 `pure_llm_six_element_inference` 收成单一产品 gate，并要求比较计算图、错误图与云端聚合真源同时成立。

这一层至少要求：

- 必须存在 `GUI agent demonstration -> teaching_dataset -> cloud_supervised_plus_q2rl -> edge_delivery -> pure_llm_six_element_inference` 的正式链路
- 示教链必须区分 `development` 与 `customer` 两种模式；其中客户实战模式必须具备真实录屏与键盘/鼠标事件
- 示教数据必须能够整理成正式 `teaching_dataset_manifest`，并产出可部署到端侧的 `teaching_trained_model_manifest`
- 云侧训练后的模型必须可通过 `edge_inference_push_contract` 下推到端侧，由 `CGC Engine` 承载 `pure_llm_six_element_inference`
- 必须提供 `teaching_alignment_report`，用于度量示教结果与纯大模型六元推理结果的贴近程度
- 必须同时产出 `teaching_vs_inference_graph` 与 `graph_error_visualization`，让比较计算图与错误可视化成为正式 artifact，而不是临时调试输出
- `cloud_summary.json` 必须继续作为云端聚合单一真源，反向索引示教、训练、下推、推理、比较与错误证据
- 产品级入口必须同时覆盖 release CLI 与 engine CLI；对于 verification-only milestone，若 gate PASS，则最外层 pipeline `ok` 也必须为 `true`

建议正式产物至少包括：

- `teaching_mode_contract.json`
- `teaching_dataset_manifest.json`
- `teaching_trained_model_manifest.json`
- `screen_recording.mp4` 或等价完整录屏文件索引
- `keyboard_mouse_events.jsonl` 或等价键盘/鼠标事件文件索引
- `edge_inference_push_contract.json`
- `llm_six_element_inference_mode.json`
- `teaching_alignment_report.json`
- `teaching_vs_inference_graph.json`
- `graph_error_visualization.json`
- `graph_error_visualization.mmd`
- `cloud_summary.json`

建议正式执行入口：

- `cgc agent teach --teaching-mode development --dag-file <workflow.json> --gui-duration-s <seconds>`
- `cgc agent teach --teaching-mode customer --dag-file <workflow.json> --screen-recording-path <screen_recording.mp4> --keyboard-mouse-events-path <keyboard_mouse_events.jsonl> --gui-evidence-path <gui_agent_runtime_evidence.json>`
- `cgc agent train --teach-session <agent_teach_session.json> --teaching-mode customer --screen-recording-path <screen_recording.mp4> --keyboard-mouse-events-path <keyboard_mouse_events.jsonl>`
- `cgc gate m78`
- `cgc gate upkg38`
- `cgc_engine/agent/cli.py agent teach --teaching-mode customer --dag-file <workflow.json> --screen-recording-path <screen_recording.mp4> --keyboard-mouse-events-path <keyboard_mouse_events.jsonl> --gui-evidence-path <gui_agent_runtime_evidence.json>`
- `cgc_engine/agent/cli.py pipeline --milestone m78`
- `cgc_engine/agent/cli.py pipeline --milestone upkg38`

`2026-06-20` 已形成正式 PASS artifact，并补充了 release CLI 全量 gate 重跑索引：

- release CLI：`/private/tmp/upkg38_formal_20260620/release/m78_v6/report.json`
- release gate report：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m78_teaching_pure_llm/m78_report.json`
- engine CLI：`/private/tmp/upkg38_formal_20260620/engine/m78_v2/report.json`
- engine gate report：`/private/tmp/upkg38_formal_20260620/engine/m78_v2/m78_teaching_pure_llm/m78_report.json`
- release full sweep index：`/private/tmp/full_gate_rerun_20260620/release/release_gate_status_index.json`
- 关键 artifact：
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_mode_contract.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_dataset_manifest.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_trained_model_manifest.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/q2rl_training_report.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/edge_inference_push_contract.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/llm_six_element_inference_mode.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_alignment_report.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_optimization_triplet_comparison.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/triplet_comparison.html`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/before_vs_after_vs_teaching_chart.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/graph_error_visualization.json`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/graph_error_visualization.mmd`
  - `/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/cloud_summary.json`

#### 7.4.9 2026-06-20 `UPKG 3.x` pipeline-first 封板补充

`2026-06-20` 这轮 `UPKG 3.x` 收口，已经把“统一 kernel 先重生 contract artifact”进一步推进到“gate 内部也优先直接消费新版 pipeline artifact”，因此 `UPKG 3.x` 不再只是语义层对齐 `pipeline.py`，而是开始把它作为 gate 内部 contract source-of-truth。

同时，这组 helper 的定位也不再限定为 `agent gate` 私有逻辑，而是提升为统一 pipeline contract 定义层：

- `agent` 任务
  - 可用同一组 helper 解析 `execution_context / state_abi / contract_manifest / system_execution_manifest`
- `embodied` 任务
  - 可用同一组 helper 把 train / deploy / consume 路径绑定到统一 `pipeline_kernel_contract_artifacts`
- `model` 任务
  - 可用同一组 helper 在 runtime bootstrap、system manifest materialize、contract smoke test 中复用相同 readiness 规则

当前通用实现已抽到 `cgc_engine/pipeline_contract_common.py`，其中包含：

- `candidate_output_roots(...)`
- `pipeline_kernel_contract_artifacts_from_report(...)`
- `pipeline_contract_descriptor_from_artifacts(...)`
- `pipeline_contract_descriptor_from_report(...)`

因此后续若要把 `UPKG 4.x embodied`、模型 bootstrap、runtime server 或其他 gate 接到统一 kernel 契约层，不需要再复制 `UPKG 3.x agent` 的专用 helper，只需要复用同一套通用 API 与 readiness 语义。

当前正式做法分为两层：

- rerun 层
  - `temp/misc/run_upkg3x_rerun.py` 会先调用 `MegatrainEightStepPipeline.materialize_contract_artifacts_only(...)`
  - 再统一产出 `execution_context / state_abi / strategy_decision / compatibility_report / distributed_runtime_bootstrap / contract_manifest / system_execution_manifest`
  - 最后才执行 `cgc gate upkg30 ~ upkg38`
- gate consume 层
  - `m7 / m72 / m73 / m77 / m78` 已优先读取 `pipeline_kernel_contract_artifacts`
  - 若 gate 运行在子目录下，则允许从 `output_dir -> parent -> grandparent` 回溯寻找 root `report.json` 与 contract artifact
  - `incoming cgc_report.gate_result` 会与 root pipeline report 合并，而不是覆盖 root report

这轮收口后，以下正式 contract 关系已经落到 gate 内部而不只是停留在 rerun wrapper：

- `m7`
  - gate/report 顶层直接暴露 `pipeline_kernel_contract_artifacts`
  - `matrix_axes.extra.state_abi_contract` 直接指向 root `state_abi.json`
- `m72`
  - `state_abi_contract` 已升级为优先引用 `state_abi_path / execution_context_path / contract_manifest_path / system_execution_manifest_path`
  - `runtime_evidence.json` 与 `artifact_index.json` 已纳入完整 pipeline kernel contract artifact
- `m73`
  - `publish_manifest / runtime_contract / bridge_info` 已直接挂入 `pipeline_contract_descriptor`
  - `runtime_contract` 已直接暴露 `contract_manifest_path / system_execution_manifest_path / execution_context_path / state_abi_path / distributed_runtime_bootstrap_path`
- `m77 / m78`
  - `matrix_axes.extra`、`single_source_of_truth`、`artifact_index` 与 gate/report 顶层，已正式纳入 `contract_manifest_path / system_execution_manifest_path`

对应的本地封板索引为：

- `temp/test/upkg3x_rerun_20260620/upkg3x_rerun_index.json`

该索引已经同时保存：

- `pipeline_regenerated`
- `results`

并验证：

- `upkg30 = PASS`
- `upkg31 = PASS`
- `upkg32 = PASS`
- `upkg33 = PASS`
- `upkg34 = PASS`
- `upkg35 = PASS`
- `upkg36 = PASS`
- `upkg37 = PASS`
- `upkg38 = PASS`

### 7.5 Unified Pipeline Kernel Gate 4.0（UPKG 4.0）

在 `UPKG 3.0` 完成通用产品化收口之后，具身深化路线应单独定义为：

```text
Unified Pipeline Kernel Gate 4.0（UPKG 4.0）
```

`UPKG 4.0` 的定位是：

- 具身 runtime / comparative / benchmark gate
- `realtime-vla`、官方 `psi0`、具身 task gain、control quality、advanced replay/audit 的正式验收层

`UPKG 4.0` 承接的内容至少包括：

- `realtime-vla` 作为具身 runtime host 的正式验收
- 官方 `psi0` 训练+推理对照基线
- `无 realtime-vla` 与 `有 realtime-vla` 的正式 comparative
- `>5x` 的明确 benchmark threshold
- `psi0 feedback` 回写
- `view-invariance`
- `one-shot`
- `structured conditioning smoke test`
- 真实 `atom library` 挂载

这意味着：

- `UPKG 3.0` 负责通用 agent 产品化 gate
- `UPKG 4.0` 负责具身能力化 gate

这样的拆分可以避免 appendix 中尚未工程闭环的具身条目，反向拖住 `UPKG 3.0` 的正式定义与产品收口。

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
  -> `ExecutionContext.runtime_mode + CanonicalExecutionProfile`
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

补充约束：

- `runtime_mode` 不应只作为分支字符串存在，而应稳定映射到四种 canonical execution profile：
  - `local_infer`
  - `local_train`
  - `edge_cloud_infer`
  - `edge_cloud_train`
- `publish_manifest / runtime_contract / bridge_info` 应共享同一份 `DeliveryProfile + matrix_axes`，避免 train / infer / edge handoff 各自重造 custom 字段。
- audit / replay / trace artifact 应保留 `canonical_execution_profiles_supported` 或等价 catalog 引用，使同一套 contract 定义可以覆盖 local 与 edge-cloud 两类入口。

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

## 13. 与当前 NCCL 问题的关系

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

## 14. 一句话结论

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
