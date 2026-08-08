# DeepSeek V2 -> V4 最小 State ABI 技术白皮书 v1.2

适用场景：模型权重断点兼容、端云 KV 复用、零拷贝 VRAM resume、CGC Engine 八步流水线统一调度

核心约束：只聚焦 `KV cache` 与 `Output Projection` 两条核心不兼容链路

判定原则：语义不变 -> `loader adaptor` 可适配；语义改变 -> 必须 `runtime branch`

## 1. 概述

本文定义一套最小、可自动化校验、无歧义的模型状态 ABI 规范，用于严格判定 `DeepSeek-V2` 权重断点在 `DeepSeek-V4` 运行时上的兼容性，并为 `CGC Engine` 计算图优化、端云协议、零拷贝 KV 迁移提供统一语义契约。

本次不兼容的本质不是命名、shard、量化、layout 等物理层差异，而是 `KV` 语义结构与输出投影数学范式的两代架构级不匹配。该类不兼容无法仅通过加载器适配修复，必须启用独立的 runtime 分支。

## 2. 设计目标

本 ABI 仅解决以下问题：

- 如何用结构化方式描述 `KV` 与 `Output Projection` 的语义契约
- 如何区分“可在加载期修复”的物理层差异与“必须前向分支处理”的语义层差异
- 如何为 `CGC Engine`、端云 KV、zero-copy VRAM resume 提供统一判定入口

本 ABI 明确不做以下事情：

- 不尝试通过张量名或 shape 猜测数学语义
- 不把 runtime 结构不兼容伪装成 loader remap
- 不让 `CGC Engine` 自动推导模型语义等价性

## 3. 最小 State 类型定义

所有兼容判定仅基于以下四类结构化描述，禁止仅凭张量名或 shape 主观判断。

### 3.1 KVState

`KVState` 定义注意力缓存的数学语义与计算链路，包含：

- KV 生成范式：原生 KV、latent MLA、grouped WKV
- 中间结构：是否含数据依赖 Norm、是否为两段式投影
- 隐域空间：是否存在独立中间 latent domain
- consumer 端期望的 forward contract

### 3.2 OutputProjectionState

`OutputProjectionState` 定义 attention output 映射回 hidden space 的投影结构，包含：

- 单段线性投影或两段低秩投影
- 是否带分组边界
- 是否存在 LoRA 压缩域
- 输入 tensor contract 与 reshape 规则

### 3.3 TensorPhysicalSpec

`TensorPhysicalSpec` 仅描述物理层信息，不决定数学等价性，包含：

- `dtype`
- `block quant` 格式
- `scale layout`、scale 数量与位置
- `TP/PP` 分片方式
- 内存排布、stride、连续性

### 3.4 CompatDecision

`CompatDecision` 是唯一合法的兼容判定结果：

- `loader_adaptor_ok`：仅物理层差异，加载期可修复
- `postload_adaptor_ok`：需后处理转换，但无需改 forward
- `runtime_branch_required`：语义不兼容，必须独立前向分支
- `incompatible`：无法兼容，需重训或结构对齐

## 4. 核心判定原则

### 4.1 允许 loader adaptor 的前提

仅当以下条件同时满足时，才允许判定为 `loader_adaptor_ok` 或 `postload_adaptor_ok`：

- 数学语义不变
- 中间张量域不变
- forward 计算链路不变
- 差异仅存在于命名、分片、量化、scale layout、物理排布

### 4.2 必须 runtime branch 的前提

满足任一条件即必须判定为 `runtime_branch_required`：

- `KVState` 语义发生变化
- `OutputProjectionState` 语义发生变化
- 引入或删除中间 latent domain
- 引入或删除数据依赖 Norm
- 单段投影变为两段低秩投影
- grouped boundary 或 reshape contract 改变
- checkpoint 缺失 runtime 必需的完整子模块权重

## 5. DeepSeek-V2 / V4 语义对照与判定

### 5.1 KV 路径对比

#### V2 Checkpoint 语义

- KV 范式：`latent_kv_two_stage`
- 计算链路：`kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj`
- 关键特征：含数据依赖 `RMSNorm`，存在独立中间隐域

#### V4 Runtime 语义

- KV 范式：`native_grouped_wkv`
- 计算链路：`wkv + kv_norm`
- 结构特征：原生分组 KV，一体化 consumer contract

#### 判定

- 该差异属于 `semantic mismatch`
- 加载期最多只能通过 `legacy_kv_*` 键名映射实现“可加载”
- 无法实现“可正确执行”

#### 结论

- `CompatDecision = runtime_branch_required`

### 5.2 Output Projection 路径对比

#### V2 Checkpoint 语义

- 投影范式：`single_o_proj`
- 计算链路：`attn_output -> o_proj`
- 权重构成：仅 `o_proj.weight` 与 `o_proj.weight_scale_inv`
- 结构特征：无分组、无低秩、无两段结构

#### V4 Runtime 语义

- 投影范式：`two_stage_wo_a_wo_b`
- 计算链路：`grouped attn output -> wo_a -> wo_b`
- 结构特征：分组低秩输出路径，依赖完整 `wo_a` 权重与 scale

#### 判定

- `V2 checkpoint` 完全缺失 `wo_a` 结构
- 当前 `.o_proj. -> .wo_b.` 属于错误降格兼容，语义不等价
- front-most 的 `wo_a.scale` 异常不是单纯 scale recipe 问题，而是结构缺失的表象

#### 结论

- `CompatDecision = runtime_branch_required`

## 6. 严格兼容规则

### 6.1 Loader Adaptor 允许范围

以下类型属于“语义不变”的纯物理层转换，允许由 loader 处理：

- 键名重映射：`q_a_proj -> wq_a`
- 键名重映射：`q_a_layernorm -> q_norm`
- 键名重映射：`q_b_proj -> wq_b`
- `TP` 分片对齐
- 量化与反量化转换
- `scale layout` 转换
- `legacy_kv_*` 前缀映射，仅用于加载通关

其共同前提是：不改变中间张量域、不改变 forward 链路、不改变数学含义。

### 6.2 Loader Adaptor 严格禁止

以下映射一律不允许，属于语义造假：

- `V2 kv_a_proj_with_mqa + kv_a_layernorm + kv_b_proj -> V4 wkv/kv_norm`
- `V2 o_proj -> V4 wo_b`
- 任何跨越 `KVState` 或 `OutputProjectionState` 语义边界的映射

### 6.3 必须 Runtime Branch 的条件

满足任一条件即触发独立 forward 分支：

- KV 范式从两段 latent MLA 变为 grouped WKV
- 输出投影从单段全秩结构变为两段低秩结构
- 存在中间隐域差异
- 存在数据依赖 Norm 差异
- 存在分组边界差异
- checkpoint 缺失 runtime 必需的完整子模块权重

## 7. 与 CGC Engine 和端云 KV 的关系

### 7.1 CGC Engine 能做什么

`CGC Engine` 负责物理层统一与运行时调度，能力包括：

- 统一 KV 传输与 page 管理
- VRAM 零拷贝搬运
- 计算图优化与算子融合
- 通信调度与 backpressure 控制
- resume 解码调度
- 物理层 tensor layout 统一

### 7.2 CGC Engine 不能做什么

`CGC Engine` 无法自动推导模型语义等价性，具体包括：

- 无法判断 `V2 latent KV` 与 `V4 grouped WKV` 是否数学可替换
- 无法补全 checkpoint 中缺失的 `wo_a` 结构
- 无法自动修正 attention output contract 不匹配

### 7.3 核心结论

- `CGC Engine` 负责物理层统一
- `State ABI` 负责语义层契约
- 缺少 ABI 约束时，zero-copy KV 只会高速搬运错误状态
- 必须先完成 ABI 语义握手，再进入 `CGC Engine` 八步流水线

## 8. 最终兼容决策

### 8.1 KV 兼容

- 类型：`semantic mismatch`
- 方案：新增 `legacy_kv` runtime branch
- 禁止：loader 直接映射到原生 `wkv` 路径

### 8.2 Output Projection 兼容

- 类型：`semantic mismatch`
- 方案：新增 `legacy_o_proj` runtime branch
- 要求：必须同步对齐 attention output 的 reshape 与 contract
- 禁止：`o_proj -> wo_b` 伪兼容

## 9. 可代码化决策矩阵

### 9.1 Allowed

- `q_a_proj -> wq_a`
- `q_a_layernorm -> q_norm`
- `q_b_proj -> wq_b`
- `legacy_kv_*` 的 key remap 到新增载入目标

### 9.2 Forbidden

- `.kv_a_proj_with_mqa/.kv_a_layernorm/.kv_b_proj -> wkv/kv_norm`
- `.o_proj. -> .wo_b.`

### 9.3 Runtime Required

- `legacy_kv` forward path
- `legacy_o_proj` forward path

## 10. 对应当前代码实现

本节将白皮书中的 ABI 结论直接映射到当前代码实现，便于后续 patch 设计、代码评审与 compat 策略落地。

### 10.1 已存在的 KV Stage-1 载入兼容

当前 `DeepSeek-V4` 代码中，`MQALayer.__init__()` 已经新增以下模块，用于承接 `DeepSeek-V2` 风格的 KV 权重：

- `legacy_kv_a_proj_with_mqa`
- `legacy_kv_a_layernorm`
- `legacy_kv_b_proj`

对应位置：

- `ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python/sglang/srt/models/deepseek_v4.py`
- 当前实现位于 `MQALayer.__init__()` 中 `legacy_kv_*` 定义区域

该实现说明：

- 当前代码已经接受“V2 KV 不能直接落到 V4 原生 `wkv/kv_norm`”这一事实
- 现阶段只解决“checkpoint 可加载”
- 尚未解决“forward 可正确执行”

因此，该部分严格符合本白皮书对 `KVState` 的判定：

- 允许 Stage-1 loader adaptor
- 但最终仍然需要 `legacy_kv runtime branch`

### 10.2 当前 remap 中已经存在的正确与错误边界

当前 `remap_weight_name_to_dpsk_hf_format()` 中，以下映射属于允许范围：

- `.q_a_proj. -> .wq_a.`
- `.q_a_layernorm. -> .q_norm.`
- `.q_b_proj. -> .wq_b.`
- `.kv_a_layernorm. -> .legacy_kv_a_layernorm.`
- `.kv_a_proj_with_mqa. -> .legacy_kv_a_proj_with_mqa.`
- `.kv_b_proj. -> .legacy_kv_b_proj.`

这些映射的性质是：

- 前三项属于语义不变的命名适配
- 后三项属于“仅为载入打通”的 Stage-1 兼容目标映射

但当前仍存在一条违反 ABI 规则的映射：

- `.o_proj. -> .wo_b.`

该映射的问题不是名字选错，而是跨越了 `OutputProjectionState` 的语义边界：

- `V2` 的 `o_proj` 是单段全秩输出投影
- `V4` 的 `wo_b` 只是两段投影中的第二段

因此，这条 remap 在 ABI 上应被明确定义为：

- `forbidden loader adaptor`

### 10.3 当前 Output Projection 的真实运行路径

当前 `DeepSeek-V4` 的 output projection 路径已经明确写死为：

- `grouped attn output -> wo_a -> wo_b`

具体特征包括：

- attention backend 输出先被 `view(..., n_local_groups, -1)` 重解释为 grouped domain
- 若开启 `_FP8_WO_A_GEMM`，先经 `wo_a.weight + wo_a.weight_scale_inv`
- 否则通过 `einsum` 执行 `wo_a`
- 最后再进入 `wo_b`

这说明当前 runtime 的 `OutputProjectionState` 已经是：

- `two_stage_wo_a_wo_b`

而非：

- `single_o_proj`

这也是为什么 `V2 checkpoint` 即使能提供 `o_proj.weight`，也无法仅靠 remap 填补当前 runtime 所需的完整 `wo_a` 结构。

### 10.4 当前 post-load 行为为何会暴露 wo_a 问题

当前 `post_load_weights()` 会检查：

- 是否出现 `wo_a`
- 是否出现 `o_proj`
- 是否出现 `wo_b`

随后，在 `_FP8_WO_A_GEMM` 打开时，会无条件进入：

- `_setup_fp8_wo_a_scales()`

而 `_setup_fp8_wo_a_scales()` 的假设前提是：

- runtime 已经拥有语义正确且结构完整的 `wo_a.weight_scale_inv`

但 `V2 checkpoint` 实际上只提供：

- `o_proj.weight`
- `o_proj.weight_scale_inv`

这意味着当前 front-most 的 `wo_a.scale` 异常并不是一个孤立的 quant recipe 问题，而是以下结构性问题的直接外显：

- checkpoint 缺失 `wo_a`
- 当前 remap 又把 `o_proj` 错投到 `wo_b`
- post-load 仍按原生 V4 路径继续处理 `wo_a`

### 10.5 future legacy_o_proj branch 的最小落点

根据当前代码结构，未来 `legacy_o_proj` runtime branch 至少需要在以下位置落地：

#### A. `MQALayer.__init__()`

新增独立模块，例如：

- `legacy_o_proj`

其职责是承接 `DeepSeek-V2` checkpoint 中真实存在的：

- `o_proj.weight`
- `o_proj.weight_scale_inv`

#### B. `remap_weight_name_to_dpsk_hf_format()`

撤销当前错误映射：

- `.o_proj. -> .wo_b.`

替换为：

- `.o_proj. -> .legacy_o_proj.`

这一步的目的不是完成语义兼容，而是首先恢复“权重落点与 checkpoint 结构一致”。

#### C. `post_load_weights()`

增加 ABI 感知分支：

- 如果本次载入的是 `legacy_o_proj` 路径，则不得继续按原生 `wo_a` 路径执行 `_setup_fp8_wo_a_scales()`

否则会继续把一个本质上缺失 `wo_a` 的 checkpoint 强行送入 `V4 native output path`。

#### D. output forward path

当前 output forward 位于 `MQALayer.forward()` 中 `o = o.view(...); wo_a; wo_b` 这一段。

未来的 `legacy_o_proj` branch 应当在这里显式分支：

- `native V4 path`: `grouped output -> wo_a -> wo_b`
- `legacy V2 compat path`: `legacy-compatible attention output -> legacy_o_proj`

这一分支的关键不是只替换最后一层 linear，而是必须同时确认：

- `legacy_o_proj` 所吃到的输入 tensor contract 是否与 `V2 o_proj` 语义一致
- 当前 grouped `o` 是否需要额外 reshape / merge / contract 对齐

换言之，`legacy_o_proj` 不能被实现成“只多加一颗线性层”的伪兼容，而必须是 ABI 驱动的显式 output projection runtime branch。

### 10.6 对当前 patch 规划的直接约束

基于现有代码状态，后续 patch 应遵循以下约束：

- `legacy_kv_*` 可以继续保留为 Stage-1 load-time 兼容入口
- `.o_proj. -> .wo_b.` 必须被视为待移除错误兼容
- `wo_a` 相关 post-load 逻辑必须以 ABI 分支为前提执行
- `legacy_o_proj` 必须同时覆盖 load-time 与 forward-time，而不能只做其中一半

## 11. 对应 Patch Plan

本节将 `legacy_o_proj` 的兼容方案整理为可直接实施的工程清单。目标不是一次性完成所有兼容，而是以最小风险顺序把错误 remap 替换为可验证的 runtime branch。

### 11.1 Patch 目标

本 patch plan 仅解决以下问题：

- 为 `V2 checkpoint` 提供语义正确的 output projection 落点
- 终止当前 `.o_proj. -> .wo_b.` 的伪兼容路径
- 让 `post_load_weights()` 不再把缺失 `wo_a` 的 checkpoint 强行送入原生 `V4` output path
- 为后续 `legacy_o_proj` forward 分支预留清晰、独立、可测试的落点

本 patch plan 不在本阶段解决：

- `legacy_kv` forward branch 的完整实现
- attention backend 输出 contract 的最终数学对齐证明
- zero-copy KV / CGC runtime 调度联动

### 11.2 变更顺序

推荐按以下顺序实施，避免多变量同时变化：

1. 新增 `legacy_o_proj` module
2. 修改 `o_proj` 的 remap 落点
3. 为 `post_load_weights()` 增加 ABI 感知分支
4. 在 output forward path 增加显式 branch
5. 增加最小可用验证点

### 11.3 Step A: 新增 legacy_o_proj module

目标位置：

- `ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python/sglang/srt/models/deepseek_v4.py`
- `MQALayer.__init__()`

建议新增内容：

- 新增 `self.legacy_o_proj`
- 模块类型应优先与 `V2 o_proj` 保持同构，即优先使用与 `V2` 相同语义的 `RowParallelLinear`

设计约束：

- `legacy_o_proj` 的输入 contract 必须服务于 `V2 single_o_proj` 语义
- 不应复用 `wo_b` 参数名，也不应与 `wo_a/wo_b` 共用权重缓冲
- 若 `quant_config` 会隐式为 `wo_a` 型路径创建 scale buffer，则 `legacy_o_proj` 应避免继承这类假设

完成判据：

- `params_dict` 中可见 `legacy_o_proj.*`
- `V2 checkpoint` 的 `o_proj.weight` 与 `o_proj.weight_scale_inv` 有明确落点

### 11.4 Step B: 修改 remap 规则

目标位置：

- `remap_weight_name_to_dpsk_hf_format()`

必须执行的变更：

- 删除当前 `.o_proj. -> .wo_b.`
- 替换为 `.o_proj. -> .legacy_o_proj.`

禁止做法：

- 保留 `.o_proj. -> .wo_b.` 再增加额外 fallback
- 同时把 `o_proj` 分流到 `wo_b` 与 `legacy_o_proj`

原因：

- 这会继续污染 `weight_names`
- 会让 post-load 逻辑误判当前 checkpoint 具备原生 `V4` output projection 结构

完成判据：

- `weight_names` 中 `o_proj` 已只落到 `legacy_o_proj`
- 加载日志不再把 `o_proj` 解释成 `wo_b`

### 11.5 Step C: 调整 post_load_weights 分支

目标位置：

- `post_load_weights()`
- `_setup_fp8_wo_a_scales()`

必须执行的变更：

- 增加 ABI 感知布尔条件，例如：
  - 当前 layer 是否命中 `legacy_o_proj`
  - 当前权重集合是否缺失 `wo_a.*`
  - 当前载入是否属于 `V2 compat output branch`
- 若命中 `legacy_o_proj` 路径，则不得进入 `_setup_fp8_wo_a_scales()`

推荐原则：

- `wo_a` 相关 post-load 初始化只对 `native V4 output projection` 生效
- 不要在 `legacy_o_proj` 路径上做“空 wo_a 占位”或“假 scale bypass”

完成判据：

- `V2 checkpoint` 加载时不再因 `wo_a.weight_scale_inv` 初始化而失败
- `native V4 checkpoint` 仍保持原有 `wo_a` post-load 逻辑

### 11.6 Step D: 增加 output forward branch

目标位置：

- `MQALayer.forward()`
- 当前 `o = o.view(...); wo_a; wo_b` 所在路径

必须执行的变更：

- 新增显式 ABI 分支：
  - `native_v4_output_branch`
  - `legacy_o_proj_output_branch`

原生分支保留：

- `grouped output -> wo_a -> wo_b`

兼容分支要求：

- `legacy-compatible attention output -> legacy_o_proj`

这里的关键不是“多一颗 linear”，而是先把输入 contract 对齐清楚：

- 当前 grouped `o` 是否需要 merge groups
- 当前 `o` 的最后一维是否已经等价于 `V2 o_proj` 期望的输入域
- 是否需要单独的 reshape / flatten / reorder

建议实施方式：

- 第一版 forward branch 先把 contract 断言与日志打全
- 在确认 `legacy_o_proj` 输入张量语义后，再决定最终 reshape 方案

完成判据：

- forward 不再强制经过 `wo_a -> wo_b`
- `legacy_o_proj` 分支的输入 shape 与 `V2 o_proj` contract 可被打印并验证

### 11.7 Step E: 最小验证清单

建议最小验证只做以下几项：

- 加载期验证
  - `o_proj.weight` 成功加载到 `legacy_o_proj.weight`
  - `o_proj.weight_scale_inv` 成功加载到 `legacy_o_proj.weight_scale_inv`
- post-load 验证
  - `V2 checkpoint` 路径不再进入 `wo_a` scale setup
  - `V4 native checkpoint` 仍会进入 `wo_a` scale setup
- forward 验证
  - `legacy_o_proj` 分支是否被显式命中
  - 分支输入 tensor 的 shape、group 维、flatten 行为是否符合预期

不建议在这一阶段做的事情：

- 为了让服务先起来而临时把 `wo_a` 逻辑全部关闭
- 直接把 grouped `o` 强行 reshape 成看起来像 `o_proj` 可吃的形状
- 在没有 ABI 断言的情况下用 silent fallback 吞掉不匹配

### 11.8 实施清单

可直接执行的工程清单如下：

- 在 `MQALayer.__init__()` 新增 `legacy_o_proj`
- 在 `remap_weight_name_to_dpsk_hf_format()` 将 `.o_proj.` 改映射到 `.legacy_o_proj.`
- 在 `post_load_weights()` 增加 `legacy_o_proj` 路径判断，阻止 `wo_a` 初始化误入
- 在 `MQALayer.forward()` 为 output projection 增加 `native` 与 `legacy` 显式分支
- 为 `legacy_o_proj` 分支增加输入 contract 日志与断言
- 用 `V2 checkpoint` 做一次最小加载与 forward 命中验证

### 11.9 风险与边界

本 patch plan 的最大风险不在 loader，而在 forward contract：

- 如果当前 `attn_backend.forward()` 返回的 grouped `o` 不可逆地偏离 `V2 o_proj` 输入域，则 `legacy_o_proj` branch 仍需在 attention output 之前或之中补额外 compat 逻辑

因此，本计划的真实目标是：

- 先把“权重落点正确性”与“post-load 误路径”修正掉
- 再用显式 branch 逼出 output contract 的真实差异

这也是为什么 `legacy_o_proj` 应被视为 ABI 驱动的 runtime branch，而不是 rename 层面的兼容补丁。

## 12. 一句话总结

本 `State ABI v1.2` 在语义判定与代码映射之外，进一步给出了 `legacy_o_proj` 的最小 patch plan，明确 `DeepSeek-V2` 与 `DeepSeek-V4` 的 output projection 兼容必须以显式 `runtime branch` 落地，而不能继续依赖 `.o_proj. -> .wo_b.` 这类伪兼容。

`CGC Engine` 与端云零拷贝 KV 只能在语义一致的前提下进行物理层优化，无法替代模型结构层面的显式兼容分支。
