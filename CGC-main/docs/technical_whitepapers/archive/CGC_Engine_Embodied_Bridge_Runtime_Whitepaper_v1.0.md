# CGC Engine Embodied Bridge Runtime Whitepaper v1.0

**版本**：v1.0  
**更新时间**：2026-06-19  
**定位**：CGC Engine 具身主线统一白皮书。本文将 `CGC_Engine_Embodied_Technical_Whitepaper_v1.0.md`、`CGC_Engine_Embodied_Executive_Summary_v1.0.md`、`CGC_Engine_Embodied_Integration_Technical_Whitepaper_v1.0.md` 与 `CGC_M7.3.1_M7.3.2_M7.3.3_Gate_Proposal_v1.0.md` 收敛为一份统一主文档，并显式移除历史训练 helper 与旧 runtime 命名。

---

## 1. 一句话结论

CGC Engine Embodied 的正式主线应定义为：

```text
GeoMirror / Canonical BEV / EE Pose
        ->
CGC Unified Pipeline Kernel
        ->
local_train / edge_cloud_train train profile
        ->
bridge artifact + edge-cloud protocol
        ->
edge runner / CGC edge engine
        ->
runtime host (realtime-vla-v2 / psi0-bridge)
        ->
端侧推理 / 控制 / Q2RL
```

核心变化有三点：

- 不再把历史训练 helper 作为具身训练主线的一等组件
- 不再把旧 backend family 作为 runtime host
- 明确把“训练得到的模型能力经 `bridge + 端云协议` 发布到端侧 CGC engine 执行”定义为正式交付路径

在产品定位上，这条具身主线不是一个孤立的机器人训练工程，也不是一个独立的端侧推理工程，而是企业级 `agent / model / embodied` 统一内核在具身场景下的正式落地 profile。它的目标是让具身数据采集、训练产物生成、跨框架权重交付、bridge 协议发布、端侧 runtime host 执行与双轨 comparative evidence，全部复用同一套 `ExecutionContext`、`State ABI`、`publish_manifest`、`runtime_contract` 与 `matrix_axes` 语义。

---

## 2. 文档目标

本文解决以下四个问题：

1. 统一当前 CGC Engine Embodied 的主叙事，避免技术白皮书、执行摘要、整合文档、M7.3 gate proposal 各说各话。
2. 明确训练主线不再以任何历史 helper 为核心，而以 `CGC Unified Pipeline Kernel` 的 `train profile` 为核心。
3. 明确 `bridge artifact + edge-cloud protocol` 不是导出辅助脚本，而是正式交付契约。
4. 明确端侧运行位置在 `CGC edge engine / edge runner / runtime host`，而不是历史上的 backend 包装层。

---

## 3. 设计约束

本文遵循两条已落地的上位约束。

### 3.1 Unified Pipeline Kernel 约束

来自 `CGC Unified Pipeline Kernel Design v1.0` 的核心约束：

- `CGC engine pipeline` 与 `Megatrain pipeline` 必须统一到同一个 kernel
- `task_entity / runtime_mode / environment / model_scope / hardware_platform` 必须收口到 `ExecutionContext`
- `hardware_topology / model_assembly / model_name` 必须作为扩展矩阵字段统一传播
- `runtime_plugin_strategy` 决定 runtime host
- `edge_cloud_transport_strategy` 决定 bridge / transport 入口

也就是说：

- 训练不是另一套系统
- 推理不是另一套系统
- 端云发布也不是另一套系统

它们都是同一个 `Unified Pipeline Kernel` 在不同 `ExecutionContext` 下的 profile 与 strategy 分支。

### 3.2 State ABI 约束

来自 `DeepSeek V2 -> V4 最小 State ABI 技术白皮书 v1.2` 的核心约束：

- 物理层差异可以由 loader adaptor 处理
- 语义层差异必须进入 `runtime_branch_required`
- `bridge / edge-cloud protocol` 必须建立在 ABI 语义契约之上

因此：

- 端云协议不能偷渡伪兼容权重
- bridge 不能绕过 ABI 判定
- 端侧执行必须消费已经过 ABI 判定的合法产物

---

## 4. 顶层系统形态

CGC Engine Embodied 的统一系统形态应分为五层：

### 4.1 数据与语义层

负责：

- GeoMirror 数据目录
- `source frame + canonical_bev frame`
- `EE pose`、`contact/object/support semantics`
- `07_action_atom_lib / 08_bev_action_atom_lib`

### 4.2 训练与编译层

负责：

- `CGC Unified Pipeline Kernel`
- `local_train / edge_cloud_train`
- staticize / graph build / layout compile
- ABI decision / runtime branch install
- 权重映射、cache、distributed 选择

### 4.3 发布与协议层

负责：

- `bridge artifact`
- `runtime_contract`
- `publish_manifest`
- `bridge_info`
- `edge-cloud protocol`
- 训练产物到端侧执行产物的标准化交付
- 将统一矩阵写入所有交付 artifact

### 4.4 端侧执行层

负责：

- `CGC edge engine`
- `edge runner`
- runtime host 选择
- provider 启动、回收 metrics、记录 evidence

### 4.5 验收与证据层

负责：

- M7.3 gate
- `summary report`
- `stage_trace.jsonl`
- bridge publish evidence
- edge deploy evidence
- `official_track_summary`
- `cgc_track_summary`
- `dualtrack_summary`
- task gain / latency / throughput / control quality evidence

---

## 5. GeoMirror 与 Canonical BEV 的角色

GeoMirror 在这条主线中的角色不变，但其定位更明确：

- 它不是训练框架
- 它不是推理框架
- 它是高质量结构化中间表示与具身数据语义的来源

其正式输出包括：

- `canonical_bev`
- `left/right end-effector pose`
- `object/contact/support semantics`
- `transform chain`
- `segmentation_summary`
- `08_bev_action_atom_lib`

这保证训练侧和端侧不再直接围绕机器人私有 joint layout 建主语义，而是围绕统一具身中间语言建主语义。

---

## 6. CGC Unified Pipeline Kernel：训练主线

在具身主线里，训练应被明确视为：

```text
CGC Unified Pipeline Kernel 的 train profile
```

而不是：

- 独立的第三方训练岛
- 只产生 checkpoint 的离线黑盒

### 6.1 训练主入口

训练入口由以下字段共同决定：

- `ExecutionContext.task_entity = embodied`
- `ExecutionContext.runtime_mode = local_train / edge_cloud_train`
- `ExecutionContext.environment`
- `ExecutionContext.model_scope = cloud_model`
- `ExecutionContext.hardware_platform`
- `ExecutionContext.hardware_topology`
- `ExecutionContext.model_assembly`

其中：

- `model_scope` 用于回答“当前主模型能力属于云侧还是端侧”
- `model_assembly` 用于回答“当前实际装配的是 tiny 还是 real_weights”
- `hardware_platform` 用于回答“当前落在 l20n / mac / ascend / nvidia5090 / windows 哪个平台”
- `hardware_topology` 用于回答“当前是单卡、单机 8 卡、双机还是更复杂的分布式拓扑”

### 6.2 训练侧策略

训练时至少要受以下策略控制：

- `runtime_plugin_strategy`
- `weight_loading_strategy`
- `weight_mapping_strategy`
- `cache_strategy`
- `distributed_strategy`

### 6.3 训练产物不再只等于 checkpoint

训练的正式输出不应只是一份 checkpoint，而应至少包括：

- model weight / adapter / LoRA / VLA head
- ABI decision summary
- runtime branch evidence
- `Q2RL` 相关策略产物
- bridge publish 所需 metadata
- 统一的 `matrix_axes`

---

## 7. bridge artifact 与端云协议：正式交付层

`bridge artifact` 的角色应被正式提升为端云协议的一部分。

### 7.1 bridge 不只是导出脚本

它不再只是：

- “训练后顺手导一个包”
- “供 demo 使用的 bundle”

而是：

- 训练产物对端侧的正式交付契约
- ABI 合法产物的发布形式
- 端云协商的结构化载体

### 7.2 bridge artifact 至少应承载

- `publish_manifest.json`
- `runtime_contract`
- `bridge_info.json`
- 模型权重引用或 adapter 引用
- ABI / runtime branch metadata
- `Q2RL` 策略向量或策略头引用
- tokenizer / norm stats / control config
- edge target metadata
- 统一的 `matrix_axes`

### 7.3 edge-cloud protocol 的职责

端云协议负责：

- 发布什么
- 版本如何握手
- ABI 如何校验
- 端侧如何知道该启动哪个 runtime host
- 端侧如何知道该恢复哪些状态与策略载体
- 端侧如何直接读取并复用同一套 `matrix_axes`

换言之，协议层回答的是：

```text
训练产物如何成为端侧可执行产物
```

当前统一矩阵最小集合应至少包括：

- `task_entity`
- `task_domain`
- `runtime_mode`
- `environment`
- `model_scope`
- `hardware_platform`
- `hardware_topology`
- `model_assembly`
- `model_name`

---

## 8. 端侧 CGC engine：执行宿主

训练产物到端侧之后，正式执行位置应是：

- `CGC edge engine`
- `edge runner`
- runtime host

而不是历史上的 backend 包装层。

### 8.1 edge runner 的职责

edge runner 负责：

- 读取 bridge artifact
- 校验 `publish_manifest` 与 `runtime_contract`
- 校验 `matrix_axes`
- 解析 edge target
- 选择并启动 runtime host
- 汇总 edge evidence 与 runtime metrics

### 8.2 runtime host 的职责

runtime host 负责：

- 真正执行 inference / control
- 消费 `Q2RL` payload
- 消费 bridge artifact 中的模型引用与配置

当前可落地的 runtime host 路径包括：

- `realtime-vla-v2`
- `psi0-bridge`

### 8.3 为什么端侧执行不能再放在 backend family

因为 backend family 混合了以下多种职责：

- artifact 解析
- runtime 选择
- provider 执行
- 指标汇总

这会导致：

- 端云协议层与执行层耦合
- runtime host 无法独立演进
- Unified Kernel 的 `runtime_plugin_strategy` 无法成为真正入口

因此，端侧执行必须上移到 `edge runner / runtime host`。

---

## 9. Q2RL：训练侧产物与端侧消费

`Q2RL` 在这条主线中不是一个独立 demo，而是正式能力的一部分。

### 9.1 训练侧角色

在训练侧，`Q2RL` 可以作为：

- trainer hook
- auxiliary loss
- strategy head
- policy vector producer

### 9.2 发布侧角色

在发布侧，`Q2RL` 的正式产物应通过 bridge artifact 进入端云协议。

可以承载为：

- `q2rl_strategy_vector`
- `q2rl_policy_head`
- `q2rl_config`
- `q2rl_runtime_metadata`

### 9.3 端侧角色

在端侧，`Q2RL` 应被 runtime host 真正消费，而不是停留在 fake op 或离线脚本层。

也就是说：

- 训练侧负责产出
- bridge 负责交付
- edge runner 负责装载
- runtime host 负责执行

---

## 10. M7.3 gate：训练主线重定义

原 `M7.3.1 / 7.3.2 / 7.3.3` 的 gate 设计仍然保留，但不再围绕任何历史 helper 组织。

### 10.1 M7.3.1：Train Publish Gate

目标：

- 验证 `embodied local_train / edge_cloud_train` 是否稳定
- 验证训练产物是否能形成合法 bridge artifact

关注指标：

- train throughput
- step time
- loss
- adapter/head export success
- bridge publish success
- `matrix_axes` completeness

### 10.2 M7.3.2：Bridge Contract Gate

目标：

- 验证 bridge artifact 与 `runtime_contract` 是否完整
- 验证 ABI / structured conditioning / Q2RL payload 是否可被端侧识别

关注指标：

- runtime contract completeness
- ABI decision completeness
- structured conditioning completeness
- Q2RL payload completeness
- `publish_manifest / runtime_contract / bridge_info` 的矩阵一致性

### 10.3 M7.3.3：Edge Delivery And Task Gain Gate

目标：

- 验证训练产物是否真正抵达端侧 CGC engine
- 验证 runtime host 是否正确执行
- 验证任务收益是否为真

关注指标：

- edge deploy success
- edge inference latency
- control loop stability
- task success / quality gain
- comparative evidence
- `official / cgc / edge` 三条 summary 的矩阵可比性

---

## 11. 不再保留旧训练岛叙事

本白皮书明确去掉旧训练岛叙事的原因，不是因为 RL / post-training 不重要，而是因为：

- 它不应作为具身主线的系统身份
- 它不能替代 `Unified Pipeline Kernel`
- 它不能替代 bridge / edge-cloud protocol
- 它不能替代 edge runner / runtime host

从系统分层上看：

- 训练仍然存在
- 后训练仍然存在
- RL 仍然存在

但它们都应被吸收到：

```text
CGC Unified Pipeline Kernel 的 train profile
```

而不是继续占据一条独立主叙事。

---

## 12. 正式主链

本文最终确认的正式主链如下：

```text
GeoMirror / canonical_bev / ee_pose / semantics
        ->
CGC Unified Pipeline Kernel
        ->
embodied local_train / edge_cloud_train
        ->
checkpoint / adapter / Q2RL policy payload
        ->
bridge artifact + publish manifest + runtime contract + matrix_axes
        ->
edge-cloud protocol
        ->
CGC edge engine / edge runner
        ->
runtime host (realtime-vla-v2 / psi0-bridge)
        ->
端侧 inference / control / Q2RL
        ->
official/cgc/edge summary + M7.3 evidence / audit / comparative
```

---

## 13. 当前工程边界

截至 v1.0，本文承认以下诚实边界：

- bridge 已有雏形，但还需继续向正式 publish contract 收敛
- edge runner/runtime host 的命名与注册体系还需继续统一
- `Q2RL` 已进入训练主线讨论，但端侧真实消费还需继续接线
- `realtime-vla` 已是现实可用的端侧执行宿主，但仍需与 CGC edge engine 侧更正式收敛
- edge ingest 实作本体仍需继续对齐到同一套矩阵 handoff payload

---

## 14. 对代码重构的直接要求

本文对后续代码重构给出以下直接要求：

1. 继续让 `runtime_plugin_strategy` 成为 runtime host 正式入口
2. 继续让 `edge_cloud_transport_strategy` 成为 bridge / transport 正式入口
3. bridge artifact 必须继续收敛为端云协议正式载体
4. `publish_manifest / runtime_contract / bridge_info / dualtrack summary` 必须统一携带 `matrix_axes`
5. 不再恢复旧 backend 包装层作为 runtime host
6. 不再恢复旧训练 helper 作为具身主线叙事

---

## 15. 结语

CGC Engine Embodied 的真正价值，不在于“又接了一个训练框架”或“又导出了一份模型包”，而在于：

```text
把具身数据语义、训练内核、ABI 判定、bridge 协议、端侧执行与工程验收，收敛成同一条可交付主线。
```

这条主线的正式表达就是：

- 数据由 `GeoMirror` 提供
- 训练由 `CGC Unified Pipeline Kernel` 负责
- 交付由 `bridge artifact + edge-cloud protocol` 负责
- 执行由 `CGC edge engine / edge runner / runtime host` 负责
- 验收由 `M7.3 gate` 负责

这就是 CGC Engine Embodied 在 v1.0 应该对内对外使用的统一系统叙事。
