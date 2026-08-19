# CGC Unified Pipeline Kernel Gate 3.0（UPKG 3.0）Agent 产品化技术白皮书

**版本**: v1.0  
**状态**: 草案实现版  
**定位**: 定义 `UPKG 3.0` 作为 `通用 Unified Pipeline Kernel Agent 产品化 gate` 的正式边界、七个子 gate、最小产物、最小通过条件、失败归因与第一批实施接线。

--- 

## 一、文件定位

`UPKG 3.0` 不再重复 `UPKG 1.0` 对统一内核的验收定义，也不取代 `UPKG 2.0` 对 `model` 产品化的定义，而是回答下面四个问题：

- 如何把 `M7 / M7.1 / M7.2 / M7.3` 已成立的 `trace / replay / audit / bridge` 升级为产品级 gate
- 如何把 `agent + edge + runtime` 收敛成统一 artifact、summary 与 failure attribution
- 如何在不引入具身专属 benchmark 的前提下，把六元一体架构落成通用 agent 审计框架
- 如何给后续实现提供最小、可执行的接线路线

本文件与上位文档的关系如下：

- `CGC_Unified_Pipeline_Kernel_Design_v1.0_zh_CN.md`
  - 负责上位设计、版本分界与 `UPKG 3.0` 原则定义
- `CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md`
  - 负责 `UPKG 1.0 kernel` 当前工程验收口径与 accepted artifacts
- `CGC_UPKG_2_0_MODEL_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md`
  - 负责 `UPKG 2.0 model` 产品化 gate 定义
- `CGC_UPKG_3_X_UNIFIED_PRODUCT_CHAIN_TECHNICAL_SPEC_v0.2_zh_CN.md`
  - 负责 `UPKG 3.x` 从 `DAG import / GUI teaching / train / infer / compare / audit / replay / trace` 进入统一产品链的技术补充规范、schema、字段字典、验收公式与最小合法样例
- `docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/`
  - 提供 `CGC_Gate_2.0` 的 `done / proof / target` 正式口径；当前 `UPKG 3.x` 在该 Gate 中被明确定义为 `proof` 承接层，而不是 2.0 核心 layer-adaptive 能力本体
- 本文件
  - 负责 `UPKG 3.0` 的 agent 产品化 gate 正式定义与第一批实施要求

---

补充说明：如果读者需要继续查看 `UPKG 3.0` gate 之外的实现 contract，例如 `artifact schema`、`DAG Node / GUI Binding / six-element inference / evidence bundle` 字段字典、错误码、验收公式与最小合法样例，应继续阅读 `docs/technical_whitepapers/CGC_UPKG_3_X_UNIFIED_PRODUCT_CHAIN_TECHNICAL_SPEC_v0.2_zh_CN.md`。

---

## 二、范围与非范围

### 2.1 范围

`UPKG 3.0` 只处理通用 agent 产品化要求，覆盖：

- `Kernel core`
- `Agent runtime`
- `Edge bridge product contract`
- `Unified artifact / summary / failure attribution`
- `Six-element audit and attribution`

### 2.2 非范围

以下内容不属于 `UPKG 3.0`，统一后移到 `UPKG 4.0`：

- `realtime-vla`
- 官方 `psi0` 训练+推理对照基线
- `无 realtime-vla` 与 `有 realtime-vla` 的正式 comparative
- `>5x` benchmark threshold
- `psi0 feedback`
- `view-invariance`
- `one-shot`
- `structured conditioning smoke test`
- 真实 `atom library` 挂载

### 2.3 与 `CGC_Gate_2.0` 的关系

`UPKG 3.0 / 3.x` 与 `CGC_Gate_2.0` 的关系，当前应统一理解为：

- `UPKG 3.x`
  - 负责 `agent product chain`
- `CGC_Gate_2.0`
  - 负责 layer-adaptive edge-cloud runtime 与 continuation 目标边界

因此两者是上层消费与下层承接关系：

- `UPKG 3.x` 为 `Gate 2.0` 提供 `artifact / summary / attribution / replay` 的产品链语义
- `Gate 2.0` 为 `UPKG 3.x` 提供可被消费的端云自治与 layer-adaptive runtime 底座

在 `CGC_Gate_2.0` 的当前正式口径里：

- `UPKG 3.x = proof`

这表示：

- `UPKG 3.x` 已经为 `Gate 2.0` 提供正式 agent product chain 承接关系
- 但不等于 `Gate 2.0` 自身的 `target` 能力已经全部完成

---

## 三、总体目标

`UPKG 3.0` 要证明：

- `Unified Pipeline Kernel` 已不只是工程内核，而是通用 agent 产品运行底座
- `agent + edge + runtime` 不再靠零散脚本或说明文字拼接成功
- 核心产物、交付契约、trace、audit、replay、summary 已经进入同一套正式 gate
- 六元一体架构在通用 agent 场景下已经具有统一入链、统一回放、统一归因能力

### 3.1 2026-06-20 全量 Gate 重跑补充

本次已按当前 `cgc gate list` 可执行范围，对 release CLI 的可用 gate 做了一轮全量重跑，统一输出目录为：

- `/private/tmp/full_gate_rerun_20260620/release`

同日也在当前工作区通过正式 release CLI 对 `UPKG 3.x` 全部产品里程碑做了一轮本地复跑，输出目录为：

- `/Users/alexchuang/Documents/flashkv0516/temp/test/upkg3x_rerun_20260620`

本地复跑使用的正式入口为：

- `python3 app/cli/cgc.py gate upkg30 --print-json`
- `python3 app/cli/cgc.py gate upkg31 --print-json`
- `python3 app/cli/cgc.py gate upkg32 --print-json`
- `python3 app/cli/cgc.py gate upkg33 --print-json`
- `python3 app/cli/cgc.py gate upkg34 --print-json`
- `python3 app/cli/cgc.py gate upkg35 --print-json`
- `python3 app/cli/cgc.py gate upkg36 --print-json`
- `python3 app/cli/cgc.py gate upkg37 --print-json`
- `python3 app/cli/cgc.py gate upkg38 --print-json`

本次全量重跑的总索引为：

- `/private/tmp/full_gate_rerun_20260620/release/release_gate_status_index.json`

对 `UPKG 3.0` 主产品链，当前正式结论为：

- `m72 = PASS`
- `m73 = PASS`
- `m77 = PASS`
- `m78 = PASS`
- `upkg30 = PASS`
- `upkg31 = PASS`
- `upkg32 = PASS`
- `upkg33 = PASS`
- `upkg34 = PASS`
- `upkg35 = PASS`
- `upkg36 = PASS`
- `upkg37 = PASS`
- `upkg38 = PASS`

本地复跑对应的主要 stdout / stderr 证据保存在：

- `temp/test/upkg3x_rerun_20260620/upkg30.stdout.txt`
- `temp/test/upkg3x_rerun_20260620/upkg31.stdout.txt`
- `temp/test/upkg3x_rerun_20260620/upkg32.stdout.txt`
- `temp/test/upkg3x_rerun_20260620/upkg33.stdout.txt`
- `temp/test/upkg3x_rerun_20260620/upkg34.stdout.txt`
- `temp/test/upkg3x_rerun_20260620/upkg35.stdout.txt`
- `temp/test/upkg3x_rerun_20260620/upkg36.stdout.txt`
- `temp/test/upkg3x_rerun_20260620/upkg37.stdout.txt`
- `temp/test/upkg3x_rerun_20260620/upkg38.stdout.txt`

这轮 `UPKG 3.x` 本地复跑没有出现新的 `FAIL / NOT_IMPLEMENTED` 里程碑，因此本次不需要追加 gate 侧补丁，主要是把复跑证据与入口路径补回文档。

### 3.2 2026-06-20 pipeline-first 与 gate-internal consume 收口

`2026-06-20` 这轮本地复跑，已经不再是“只重跑 `product/*.py` gate”的旧模式，而是先通过统一 pipeline 重生主契约，再让 `UPKG 3.x` gate 在内部优先消费这些新版 artifact。

虽然这轮改造首先落在 `UPKG 3.x agent gate`，但 helper 的最终定位不是 `agent only`。当前已经把 pipeline contract 的公共语义抽成更上层的通用定义，后续 `embodied` 与 `model` 任务可以直接复用同一套 artifact resolve 与 readiness 判定。

- 通用实现入口：`cgc_engine/pipeline_contract_common.py`
- 通用读取能力：从 root `report.json` 抽取 `pipeline_kernel_contract_artifacts`
- 通用描述能力：统一产出 `pipeline_contract_descriptor`
- 通用就绪判定：要求 `execution_context / state_abi / contract_manifest / system_execution_manifest` 同时存在且路径有效

因此这里的 `UPKG 3.0 agent` 改造，应该视为统一 pipeline contract source-of-truth 的第一批落地点，而不是一套只能在 agent 内部复用的临时 helper。

- 统一重生入口：`temp/misc/run_upkg3x_rerun.py`
- 统一先产物化：`execution_context.json`
- 统一先产物化：`state_abi.json`
- 统一先产物化：`strategy_decision.json`
- 统一先产物化：`compatibility_report.json`
- 统一先产物化：`contract_manifest.json`
- 统一先产物化：`system_execution_manifest.json`
- 统一先产物化：`distributed_runtime_bootstrap.json`

本轮总索引 `temp/test/upkg3x_rerun_20260620/upkg3x_rerun_index.json` 已额外保存 `pipeline_regenerated`，用于回放每个里程碑在 gate 执行前实际绑定到哪一组 pipeline kernel contract artifact。

同时，以下 gate 已改为在内部优先读取 `pipeline_kernel_contract_artifacts`，而不是继续把旧的 gate-local 派生产物当成唯一真源：

- `m7`
  - `matrix_axes.extra.state_abi_contract`、`artifact_index`、gate/report 顶层都直接暴露 root `state_abi.json / contract_manifest.json / system_execution_manifest.json`
- `m72`
  - `state_abi_contract` 已从旧的字符串摘要，升级为优先引用 root `state_abi_path / execution_context_path / contract_manifest_path / system_execution_manifest_path`
  - `runtime_evidence.json` 与 `artifact_index.json` 已带出完整 `pipeline_kernel_contract_artifacts`
- `m73`
  - `publish_manifest.json`、`runtime_contract.json`、`bridge_info.json` 已显式挂入 `pipeline_contract_descriptor`
  - `runtime_contract.json` 已直接暴露 `execution_context_path / state_abi_path / contract_manifest_path / system_execution_manifest_path / distributed_runtime_bootstrap_path`
- `m77`
  - `matrix_axes.extra`、`single_source_of_truth`、`artifact_index` 已把 `contract_manifest_path / system_execution_manifest_path` 纳入正式字段
- `m78`
  - `matrix_axes.extra`、`single_source_of_truth`、`artifact_index` 已把 `contract_manifest_path / system_execution_manifest_path` 纳入正式字段

为了解决 `m72 / m77 / m78` 等子目录 gate 与 root pipeline artifact 不在同一路径的问题，当前共用 helper 也已支持从 child output dir 回溯 `parent / grandparent` 读取 `report.json` 与 kernel contract artifact，并把 `incoming cgc_report.gate_result` 合并回最终 report。

因此当前 `UPKG 3.x` 的正式口径，已经从：

- `pipeline 先重生，再由 gate 继续自产旧 artifact`

推进到：

- `pipeline 先重生，gate 内部也优先直接消费新版 pipeline kernel contract artifact`

对历史 / 基线 / 长流程 gate，当前应单独解释：

- `m1 / m2 / m3 / m5 / m6` 的旧顶层 report 仍可能受到 `backend fingerprint strict gate` 影响，不能直接按顶层 `ok` 推导为产品链失败
- `m4` 在本轮仍体现真实 training / distributed 条件未满足
- `m76` 仍要求 `Nvidia real-chain runtime evidence`
- `m8` 为更长时间的 productization 聚合流程，不与 `UPKG 3.0` 主产品链共用同一解释方式

---

## 四、六个正式 Gate

### 4.1 `3.1 Kernel Core Product Gate`

**目标**

- 把 `M7 / M7.1` 的 core artifact 升级为产品级 kernel 验收入口

**对应来源**

- `M7`
- `M7.1`
- 等价 core runtime route

**最小正式产物**

- `report.json`
- `dynamic_trace_l1`
- `state_compression_summary`
- `soft_rt_replay`
- `industrial_audit`
- `events.jsonl`
- `chain_head.json`

**2026-06-19 正式 rerun artifact**

- 顶层矩阵：`/private/tmp/upkg30_formal_pass_20260619/upkg30_completion_matrix.json`
- 顶层 manifest：`/private/tmp/upkg30_formal_pass_20260619/upkg30_formal_pass_manifest.json`
- `3.1` 主 report：`/private/tmp/upkg30_formal_pass_20260619/m7/m7_industrial/m7_report.json`
- `3.1` 主 summary：`/private/tmp/upkg30_formal_pass_20260619/m7/m7_industrial/summary.json`
- `3.1` artifact index：`/private/tmp/upkg30_formal_pass_20260619/m7/m7_industrial/artifact_index.json`

**正式执行入口**

- 推荐 release CLI：`cgc gate upkg31`
- 底层承载 gate：`cgc gate m7`
- 顶层聚合入口：`cgc gate upkg30`

**最小 PASS 条件**

- `compile_success_rate = 1.0`
- `cache_hit_rate >= 2/3`
- `soft_rt_replay.status = PASS`
- `industrial_audit.event_integrity = 1.0`
- `industrial_audit.hash_chain_valid = 1.0`

**统一失败归因**

- `compile_failure`
- `cache_instability`
- `state_codec_failure`
- `replay_deadline_failure`
- `audit_chain_break`

### 4.2 `3.2 Agent Runtime Gate`

**目标**

- 把 `M7.2` 从场景化 GUI agent 验收，提升为通用 agent runtime 产品化 gate

**对应来源**

- `M7.2`
- 等价非 GUI agent route

**最小正式产物**

- `report.json`
- `summary.json`
- `stage_trace.jsonl`
- `runtime_evidence.json`
- tool / workflow / runtime host 结构化事件

**2026-06-19 正式 rerun artifact**

- 顶层矩阵：`/private/tmp/upkg30_formal_pass_20260619/upkg30_completion_matrix.json`
- `3.2` 主 report：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/report.json`
- `3.2` 主 summary：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/summary.json`
- `3.2` artifact index：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/artifact_index.json`
- `3.2` runtime evidence：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/runtime_evidence.json`
- `3.2` GUI evidence：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/gui_agent_runtime/gui_agent_runtime_evidence.json`
- 本次 rerun 结论：`status = PASS`

**正式执行入口**

- 推荐 release CLI：`cgc gate upkg32`
- 底层承载 gate：`cgc gate m72`
- 顶层聚合入口：`cgc gate upkg30`

**最小 PASS 条件**

- `dynamic_trace_l1 = PASS`
- `soft_rt_replay = PASS`
- `industrial_audit = PASS`
- 至少完成一次真实 workflow dispatch
- tool 事件、workflow 事件、runtime host 事件可串接

**正式 route 口径**

- `agent domain` 应视为通用 agent 的正式产品化主路由
  - 默认走 `主 pipeline + GUI-native route`
  - `workflow / runtime_host / screenshot / tool_call` 应直接进入 `pipeline report`、`summary`、`m72 gate` 与 `cgc run` artifact
- `harness domain` 仅保留为旧测试 / 验证专用 route
  - 只在明确指定 `task_domain = harness / moe` 或 `model_name = moe_harness` 时进入
  - 不再作为通用 `agent domain` 的默认承载路径
- 因此 `3.2 Agent Runtime Gate` 的正式判定对象，应优先以 `agent domain` 主路由为准，而不是以 `harness route` 的测试便利性替代产品 runtime

**统一失败归因**

- `workflow_plan_failure`
- `tool_execution_failure`
- `runtime_host_failure`
- `environment_not_ready`
- `state_handoff_failure`

### 4.3 `3.3 Edge Bridge Product Gate`

**目标**

- 把 `M7.3` 中已经成立的 bridge / publish / edge delivery 能力收成通用产品交付 gate

**对应来源**

- `M7.3`
- 等价 edge delivery route

**最小正式产物**

- `publish_manifest.json`
- `runtime_contract.json`
- `bridge_info.json`
- `edge_delivery_evidence.json`
- `bridge_publish_evidence.json`
- `summary.json`

**2026-06-19 正式 rerun artifact**

- 顶层矩阵：`/private/tmp/upkg30_formal_pass_20260619/upkg30_completion_matrix.json`
- `3.3` 主 report：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/m73_report.json`
- `3.3` 主 summary：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/summary.json`
- `3.3` artifact index：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/artifact_index.json`
- `3.3` publish manifest：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/publish_manifest.json`
- `3.3` runtime contract：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/runtime_contract.json`
- `3.3` bridge info：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/bridge_info.json`
- 本次 rerun 结论：`status = PASS`

**正式执行入口**

- 推荐 release CLI：`cgc gate upkg33`
- 底层承载 gate：`cgc gate m73`
- 顶层聚合入口：`cgc gate upkg30`

**最小 PASS 条件**

- `publish_manifest / runtime_contract / bridge_info` 共享同一套 `matrix_axes`
- `edge delivery` 对应明确 runtime target
- bridge 交付失败不得伪装成 runtime 成功
- 所有交付产物可由 `report.json` 或 `summary.json` 反向索引

**统一失败归因**

- `publish_manifest_incomplete`
- `runtime_contract_incomplete`
- `bridge_info_incomplete`
- `edge_delivery_failure`
- `matrix_axes_mismatch`

### 4.4 `3.4 Unified Artifact And Summary Gate`

**目标**

- 把所有 agent + edge + runtime 产物统一成单一真源与统一摘要体系

**对应来源**

- 所有进入 `UPKG 3.0` 的 route

**最小正式产物**

- `report.json`
- `summary.json`
- `stage_trace.jsonl`
- `failure_attribution.json`
- `matrix_axes`
- artifact path index
- `edge_inference_result.json`
- `replay_anchor.json`
- `reward_trace.json`
- `cloud_ingest_manifest.json`
- `cloud_summary.json`

**2026-06-20 正式 rerun artifact**

- `3.4` 云端聚合真源：`/private/tmp/upkg37_cloud_aggregate_20260619/m77/m72_industrial/cloud_summary.json`
- `3.4` 云端 ingest manifest：`/private/tmp/upkg37_cloud_aggregate_20260619/m77/m72_industrial/cloud_ingest_manifest.json`
- `3.4` edge inference result：`/private/tmp/upkg37_cloud_aggregate_20260619/m77/m72_industrial/edge_inference_result.json`
- `3.4` replay anchor：`/private/tmp/upkg37_cloud_aggregate_20260619/m77/m72_industrial/replay_anchor.json`
- `3.4` reward trace：`/private/tmp/upkg37_cloud_aggregate_20260619/m77/m72_industrial/reward_trace.json`
- `3.4` 统一主索引：`/private/tmp/upkg37_cloud_aggregate_20260619/m77/m72_industrial/artifact_index.json`
- `3.4` 本地 edge report：`/private/tmp/upkg37_cloud_aggregate_20260619/m77/m72_industrial/report.json`
- `3.4` 本地 edge summary：`/private/tmp/upkg37_cloud_aggregate_20260619/m77/m72_industrial/summary.json`
- supporting source index：
- `/private/tmp/upkg30_formal_pass_20260619/m7/m7_industrial/artifact_index.json`
- `/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/artifact_index.json`
- `/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/artifact_index.json`
- 本次 rerun 结论：`status = PASS`

**正式执行入口**

- 推荐 release CLI：`cgc gate upkg34`
- 底层承载 gate：`cgc gate m72`
- 顶层聚合入口：`cgc gate upkg30`

**最小 PASS 条件**

- `cloud_summary.json` 是云端聚合单一真源
- 本地 `report.json` / `summary.json` 必须作为 edge source 被 `cloud_ingest_manifest.json` 反向索引
- `stage_trace.jsonl` 覆盖关键阶段
- `failure attribution` 必须结构化、可机读
- artifact path index 可用于反向索引 publish / runtime / audit / edge-to-cloud return 产物

**统一失败归因**

- `missing_single_source_of_truth`
- `stage_trace_incomplete`
- `matrix_axes_missing`
- `artifact_index_missing`
- `failure_attribution_missing`

### 4.5 `3.5 Six-Element Audit And Attribution Gate`

**目标**

- 把六元一体架构先落实为通用 agent 场景下的产品化审计与归因框架

**六元最小映射**

- `模型元`
- `工作流元`
- `运行环境元`
- `感知 / 界面元`
- `执行元`
- `全局记忆元`

**最小正式产物**

- 六元分类 `events.jsonl`
- audit hash chain
- attribution summary
- replay anchor
- `gui_source_registry.json`
- `gui_stage_bindings.json`
- `gui_operator_graph.json`
- `gui_execution_context.json`

**2026-06-19 正式 rerun artifact**

- `3.5` 主 report：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/report.json`
- `3.5` 主 summary：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/summary.json`
- `3.5` 六元事件链：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/six_element_events.jsonl`
- `3.5` graph-native 结论：
- `graph_native_status = PASS`
- `graph_native_integration_level = graph_native_stage_execution`
- 本次 rerun 结论：`status = PASS`

**正式执行入口**

- 推荐 release CLI：`cgc gate upkg35`
- 底层承载 gate：`cgc gate m72`
- 顶层聚合入口：`cgc gate upkg30`

**最小 PASS 条件**

- 六元事件统一入链
- 任一失败都能定位到至少一个元与其上下游依赖
- replay 可重建最小业务路径
- audit chain 不因跨元流转而中断

**统一失败归因**

- `model_element_missing`
- `workflow_element_missing`
- `environment_element_missing`
- `perception_element_missing`
- `execution_element_missing`
- `memory_element_missing`
- `cross_element_chain_break`

**当前实现收口**

- `gate-native` 已进入可运行阶段
  - `GUI route` 已可进入 `pipeline report`、`summary`、`m72 gate`、`cgc run` artifact
- `graph-native` 已进入 stage-native execution 阶段
  - 已有 `source registry`
  - 已有 `stage binding`
  - 已有 `stage operator execution`
  - 已有 `operator graph`
  - 已有 `execution context`
- 当前仍未完成的部分
  - `per-stage tensorized GUI source` 尚未打开
  - 尚未把 GUI source 深入到底层张量化 / kernel 级 operator 输入
  - 因此当前阶段应定性为 `graph-native stage execution`，但仍未到 `fully tensorized native execution`

### 4.6 `3.6 Missing Capability Closure Gate`

**目标**

- 把当前 `UPKG 3.0` 尚未落地、但已明确识别出的关键缺口统一收敛成一个可追踪、可分批实现、可独立验收的 closure gate

**本 Gate 负责收口的缺失项**

- `workflow DAG` 标准化导出
- DAG 遍历后的轨迹 / 样本合成
- 面向业务流程内化的 `LoRA / full fine-tune / expert fine-tune`
- 解释模式与编译模式的双模式治理
- 内化后决策路径与外部审计事件的结构化对位
- `failure_attribution`、`matrix_axes`、`summary`、`runtime_contract` 在全链路上的一致化残缺项
- `GUI source` 从 `graph-bound route` 进一步走向 `graph-native operator execution` 的残缺项

**对应来源**

- `Subterranean Agent` 兼容性与 roadmap 缺口
- `UPKG 3.0` 当前 `3.1-3.5` 尚未闭合的实现面
- 后续通用 agent 训练侧产品链

**最小正式产物**

- `gap_register.json`
- `closure_plan.json`
- `workflow_dag_schema.json`
- `trajectory_synthesis_spec.json`
- `fine_tune_profile.json`
- `dual_mode_governance.json`
- `audit_alignment_spec.json`

**2026-06-19 正式 rerun artifact**

- `3.6` 主 report：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/report.json`
- `3.6` 主 summary：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/summary.json`
- `3.6` gap register：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/gap_register.json`
- `3.6` closure plan：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/closure_plan.json`
- `3.6` workflow schema：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/workflow_dag_schema.json`
- `3.6` trajectory spec：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/trajectory_synthesis_spec.json`
- `3.6` fine-tune profile：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/fine_tune_profile.json`
- `3.6` dual-mode governance：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/dual_mode_governance.json`
- `3.6` audit alignment：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/audit_alignment_spec.json`
- `3.6` graph-native GUI source integration：`status = PASS`

**正式执行入口**

- 推荐 release CLI：`cgc gate upkg36`
- 底层承载 gate：`cgc gate m72`
- 顶层聚合入口：`cgc gate upkg30`

**最小 PASS 条件**

- 所有已声明缺口必须进入统一 `gap register`，不得继续只存在于白皮书叙事或会议口头结论
- 每个缺口必须绑定明确：
  - owner
  - input artifact
  - output artifact
  - gate dependency
  - failure attribution
- `workflow -> trajectory -> fine-tune -> governance -> audit alignment` 这条训练侧产品链必须至少给出一版完整 spec
- `3.1-3.5` 的未完成项必须可映射到 `3.6` 的 closure plan，而不是散落为独立 TODO

**与 route 边界的关系**

- `3.6` 不再负责修正 `agent domain` 与 `harness domain` 的定义混线
- 该边界现已明确：
  - `agent domain = 主 pipeline + GUI-native route`
  - `harness domain = 旧测试 / 验证专用 route`
- `3.6` 当前真正剩余的 GUI graph-native 缺口，应聚焦在：
  - `per-stage tensorized GUI source`
  - 更深层的 native operator / kernel input 融合
  - 与训练侧 `workflow -> trajectory -> fine-tune -> governance -> audit alignment` 的主链收口
- `3.6` 不单独承载正式 `cloud-train / edge-infer / Q2RL post-train` 产品模式
  - 该部分自本版起提升为 `3.7` 独立 gate

**统一失败归因**

- `gap_register_missing`
- `closure_plan_missing`
- `workflow_dag_schema_missing`
- `trajectory_synthesis_spec_missing`
- `fine_tune_profile_missing`
- `dual_mode_governance_missing`
- `audit_alignment_spec_missing`
- `unmapped_gap_item`

### 4.7 `3.7 Cloud-Edge Training And Inference Q2RL Gate`

**目标**

- 为 `UPKG 3.0` 提供正式的 `端云训练 / 端侧推理` 产品模式
- 让 `GUI agent` 可把训练后的模型经 `publish / delivery / runtime contract` 下推到端侧，由 `CGC Engine` 承载推理
- 把 `CGC Unified Pipeline Kernel Design v1.0` 作为训练后产品约束底座，并引入 `Q2RL` 作为后训练方法

**本 Gate 负责收口的能力**

- `cloud_train -> publish -> edge_delivery -> edge_infer` 的单一正式模式定义
- `GUI agent` 训练后模型在端侧 `CGC Engine` 的推理契约
- `Q2RL` 在云侧训练，对 `workflow / tool_call / runtime_host / screenshot / replay` 的 reward 绑定
- 端侧可通过 `CLI / cgc run / 其他命令入口` 发起推理与结果回传
- `trained_weights / state_abi / runtime_contract / publish_manifest` 的端侧部署 bundle 规范

**对应来源**

- `M7.2` 的 `GUI agent` runtime evidence
- `M7.3` 的 bridge / publish / edge delivery contract
- `CGC_Unified_Pipeline_Kernel_Design_v1.0_zh_CN.md` 的统一 kernel 设计与运行时边界

**最小正式产物**

- `cloud_edge_training_inference_mode.json`
- `gui_agent_edge_inference_contract.json`
- `q2rl_post_training_profile.json`
- `edge_deployment_bundle_manifest.json`
- `cloud_edge_q2rl_evaluation_plan.json`
- `cloud_edge_q2rl_register.json`

**正式执行入口**

- 推荐主入口：`cgc gate upkg37`
- 底层承载 gate：`cgc gate m77`
- engine CLI 入口：`cgc_engine/agent/cli.py pipeline --milestone m77`
- engine CLI alias：`cgc_engine/agent/cli.py pipeline --milestone upkg37`
- 其中：
  - `upkg37` 作为与 `UPKG 3.7` 口径对齐的 release-facing主入口
  - `m77` 作为 `3.7` 的底层独立 gate 名称

**2026-06-19 正式 rerun artifact**

- `3.7` 主 report：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/report.json`
- `3.7` 主 summary：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/summary.json`
- `3.7` cloud-edge mode：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/cloud_edge_training_inference_mode.json`
- `3.7` GUI edge inference contract：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/gui_agent_edge_inference_contract.json`
- `3.7` Q2RL profile：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/q2rl_post_training_profile.json`
- `3.7` edge deployment bundle：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/edge_deployment_bundle_manifest.json`
- `3.7` evaluation plan：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/cloud_edge_q2rl_evaluation_plan.json`
- 本次 rerun 结论：`status = PASS`
- 独立 gate rerun：
  - `cgc gate m77`：`/private/tmp/upkg37_standalone_20260619/m77/report.json`
  - `cgc gate upkg37`：`/private/tmp/upkg37_standalone_20260619/upkg37/report.json`

**最小 PASS 条件**

- 必须明确定义 `cloud_train_edge_infer` 模式的云侧角色、端侧角色与必要 handoff
- `GUI agent` 必须有训练后模型下推端侧执行推理的正式 contract，而不是只停留在本地 runtime
- `Q2RL` 必须明确归属于云侧训练/后训练，而端侧负责触发、推理与结果回传
- 产品级入口必须至少同时覆盖 release CLI 与 engine CLI，而不能只停留在单一路径
- `Q2RL` 必须绑定至少以下 reward source：
  - `workflow_completion`
  - `tool_call_validity`
  - `runtime_host_stability`
  - `screenshot_state_transition`
  - `edge_replay_consistency`
- 端侧 bundle 必须显式包含：
  - `trained_weights`
  - `state_abi_contract`
  - `runtime_contract`
  - `publish_manifest`

**统一失败归因**

- `cloud_edge_training_inference_mode_missing`
- `gui_agent_edge_inference_contract_missing`
- `q2rl_post_training_profile_missing`
- `edge_deployment_bundle_manifest_missing`
- `cloud_edge_q2rl_evaluation_plan_missing`

### 4.8 `3.8 Teaching Mode And Pure LLM Six-Element Inference Gate`

**目标**

- 为 `UPKG 3.0` 提供正式的 `GUI agent 示教 -> 云侧训练 -> 端云下推 -> 端侧纯大模型六元一体推理` 产品模式
- 让 `GUI agent` 示教数据能被整理为正式训练数据集，并产出可部署到端侧 `CGC Engine` 的模型 bundle
- 让端侧在 `pure_llm_six_element_inference` 模式下接近示教结果，并提供比较计算图与错误可视化

**本 Gate 负责收口的能力**

- `GUI agent demonstration -> teaching_dataset -> cloud_supervised_plus_q2rl -> edge_delivery -> pure_llm_six_element_inference` 的单一正式模式定义
- `teaching_mode_contract / teaching_dataset_manifest / teaching_trained_model_manifest` 的训练前后契约
- `edge_inference_push_contract` 对 `CLI / cgc run / 其他命令入口` 的统一下推控制入口
- `teaching_vs_inference_graph` 与 `graph_error_visualization` 的正式比较与错误可视化
- `cloud_summary.json` 继续作为云端聚合单一真源，统一索引示教、推理、比较与错误证据

**对应来源**

- `M7.2` 的 `GUI agent` runtime evidence 与 teaching source
- `M7.3` 的 bridge / publish / edge delivery contract
- `3.7` 的 `cloud_train / edge_infer / cloud_aggregate` 主干
- `CGC_Unified_Pipeline_Kernel_Design_v1.0_zh_CN.md` 的统一 kernel 设计与运行时边界

**最小正式产物**

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

**正式执行入口**

- release CLI 面向使用者的产品工作流入口：
  - `cgc agent import-dag --dag-file <workflow.json>`
  - 开发模式：`cgc agent teach --teaching-mode development --dag-file <workflow.json> --gui-duration-s <seconds>`
  - 客户模式：`cgc agent teach --teaching-mode customer --dag-file <workflow.json> --screen-recording-path <screen_recording.mp4> --keyboard-mouse-events-path <keyboard_mouse_events.jsonl> --gui-evidence-path <gui_agent_runtime_evidence.json>`
  - 开发模式：`cgc agent train --teach-session <agent_teach_session.json> --teaching-mode development`
  - 客户模式：`cgc agent train --teach-session <agent_teach_session.json> --teaching-mode customer --screen-recording-path <screen_recording.mp4> --keyboard-mouse-events-path <keyboard_mouse_events.jsonl>`
  - `cgc agent infer --train-session <agent_train_session.json>`
  - `cgc agent visualize --train-session <agent_train_session.json>`
  - `cgc agent compare --train-session <agent_train_session.json>`
  - `cgc agent audit --train-session <agent_train_session.json>`
  - `cgc agent replay --train-session <agent_train_session.json>`
  - `cgc agent trace --train-session <agent_train_session.json>`
- engine CLI 对等产品工作流入口：
  - `python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent import-dag --dag-file <workflow.json>`
  - 开发模式：`python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent teach --teaching-mode development --dag-file <workflow.json> --gui-duration-s <seconds>`
  - 客户模式：`python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent teach --teaching-mode customer --dag-file <workflow.json> --screen-recording-path <screen_recording.mp4> --keyboard-mouse-events-path <keyboard_mouse_events.jsonl> --gui-evidence-path <gui_agent_runtime_evidence.json>`
  - 开发模式：`python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent train --teach-session <agent_teach_session.json> --teaching-mode development`
  - 客户模式：`python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent train --teach-session <agent_teach_session.json> --teaching-mode customer --screen-recording-path <screen_recording.mp4> --keyboard-mouse-events-path <keyboard_mouse_events.jsonl>`
  - `python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent infer --train-session <agent_train_session.json>`
  - `python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent visualize --train-session <agent_train_session.json>`
  - `python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent compare --train-session <agent_train_session.json>`
  - `python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent audit --train-session <agent_train_session.json>`
  - `python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent replay --train-session <agent_train_session.json>`
  - `python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py agent trace --train-session <agent_train_session.json>`
- 推荐主入口：`cgc gate upkg38`
- 底层承载 gate：`cgc gate m78`
- engine CLI 入口：`cgc_engine/agent/cli.py pipeline --milestone m78`
- engine CLI alias：`cgc_engine/agent/cli.py pipeline --milestone upkg38`
- 其中：
  - `m78` 作为 `3.8` 的独立 gate 名称
  - `upkg38` 作为与 `UPKG 3.8` 口径对齐的 release-facing alias
  - `cgc agent ...` 与 `cgc_engine/agent/cli.py agent ...` 共同作为 `3.8` 的双 CLI 产品级用户操作入口

**Subterranean Agent 兼容链**

- `upkg38` 现已具备最小可用的 `Subterranean Agent` 兼容能力：
  - `workflow DAG` 可由 `cgc agent import-dag` 或 `cgc_engine/agent/cli.py agent import-dag` 导入
  - 导入后的 DAG 可通过 `agent_graph_insertion_contract.json` 作为计算图子图或模型节点插入
  - GUI 示教结果、训练结果、推理结果可通过 release CLI 或 engine CLI 的 `compare/audit/replay/trace` 命令进行比较、审计、回放与回朔
- 这一能力当前以 `UI-TARS + cloud_supervised_plus_q2rl + edge pure_llm_six_element_inference` 为正式产品链落地
- 这一能力的当前定位是 `Subterranean Agent compatible product chain`，不是通用任意 DAG 到权重编译器

**2026-06-20 正式 rerun artifact**

- `3.8` release gate report：`/private/tmp/upkg38_formal_20260620/release/m78_v6/report.json`
- `3.8` release m78 report：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m78_teaching_pure_llm/m78_report.json`
- `3.8` release teaching mode contract：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_mode_contract.json`
- `3.8` release teaching dataset manifest：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_dataset_manifest.json`
- `3.8` release trained model manifest：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_trained_model_manifest.json`
- `3.8` release Q2RL training report：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/q2rl_training_report.json`
- `3.8` release edge inference push contract：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/edge_inference_push_contract.json`
- `3.8` release pure LLM six-element mode：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/llm_six_element_inference_mode.json`
- `3.8` release teaching alignment report：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_alignment_report.json`
- `3.8` release comparison graph：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_vs_inference_graph.json`
- `3.8` release triplet comparison：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_optimization_triplet_comparison.json`
- `3.8` release triplet Mermaid：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/triplet_comparison.mmd`
- `3.8` release triplet HTML：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/triplet_comparison.html`
- `3.8` release metric chart：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/before_vs_after_vs_teaching_chart.json`
- `3.8` release audit replay bundle：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/teaching_optimization_audit_replay_bundle.json`
- `3.8` release graph error visualization：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/graph_error_visualization.json`
- `3.8` release Mermaid error graph：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/graph_error_visualization.mmd`
- `3.8` release cloud summary：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/cloud_summary.json`
- `3.8` release GUI runtime evidence：`/private/tmp/upkg38_formal_20260620/release/m78_v6/m72_industrial/gui_agent_runtime/gui_agent_runtime_evidence.json`
- `3.8` engine pipeline report：`/private/tmp/upkg38_formal_20260620/engine/m78_v2/report.json`
- `3.8` engine m78 report：`/private/tmp/upkg38_formal_20260620/engine/m78_v2/m78_teaching_pure_llm/m78_report.json`
- 本次 rerun 结论：`release CLI = PASS`、`m78 = PASS`、`upkg38 = PASS`、`engine CLI gate = PASS`、`engine CLI outer ok = true`
- 本次正式索引要点：`target_model_id = bytedance-research/UI-TARS-2B-SFT`、`alignment_score = 1.0`、`overlay.status = PASS`

**最小 PASS 条件**

- 必须明确定义 `GUI agent` 示教数据到云侧训练模型的正式 handoff，而不是只保留本地演示证据
- 客户实战模式下，必须同时具备真实录屏与键盘/鼠标事件，而不能只依赖开发期 `--gui-duration-s` 采样
- 必须形成可下推到端侧 `CGC Engine` 的训练后模型 manifest 与 edge push contract
- 端侧必须支持 `pure_llm_six_element_inference`，并保留与示教结果的正式对齐报告
- `m78` gate 当前临时采用 `alignment_score >= 0.5` 作为独立放宽门槛；此放宽仅作用于 `m78` 本身，不改变上游通用 `teaching_alignment_report.json` 的默认生成口径
- 必须同时产出比较计算图与错误可视化，而不能只给单边 summary
- 产品级入口必须至少同时覆盖 release CLI 与 engine CLI，而不能只停留在单一路径
- 云端聚合单一真源必须继续落在 `cloud_summary.json`

**2026-06-20 临时门槛说明**

- 当前 `m78` 允许通过环境变量 `CGC_M78_ALIGNMENT_THRESHOLD` 覆盖独立对齐门槛，默认值为 `0.5`
- 共用 teaching/alignment 生成链仍保持 `CGC_TEACHING_ALIGNMENT_THRESHOLD` 默认 `0.8` 的正式口径
- 待 `UI-TARS-2B-SFT` 在现行 teaching/Q2RL 产物链上稳定达到 `0.8` 后，应移除该临时放宽并恢复 `m78` 与上游报告同门槛

**统一失败归因**

- `teaching_mode_contract_missing`
- `teaching_dataset_manifest_missing`
- `teaching_trained_model_manifest_missing`
- `edge_inference_push_contract_missing`
- `llm_six_element_inference_mode_missing`
- `teaching_alignment_report_missing`
- `teaching_vs_inference_graph_missing`
- `graph_error_visualization_missing`
- `cloud_summary_missing`

### 4.9 `3.9 Strict Closure And Schema-Validated Gate`

**目标**

- 基于 `v0.2` 技术补充规范，把 `3.8` 之后仍属“文档已定义但尚未严格机验”的部分正式收口
- 将 `m78` 的临时 `alignment_score >= 0.5` 放宽门槛恢复为 `0.8` 严格验收口径
- 把 `schema / 字段字典 / 合法样例 / 非法样例 / validator execution` materialize 为正式 gate artifact
- 把 `graph-native stage execution` 推进到 `per-stage tensorized GUI source` 并形成 strict closure report
- 形成可独立验收的 `end_to_end_executor_closure`

**本 Gate 负责收口的能力**

- `strict_alignment_acceptance`：要求 `alignment_score >= 0.8`、`report_target_threshold >= 0.8`、`missing_six_elements = []`
- `q2rl_strict_acceptance`：要求训练后 reward 改善成立，且 post-Q2RL alignment 不低于 `0.8`
- `schema_bundle_materialization`：要求 `artifact envelope / DAG node / GUI binding / six-element inference / evidence bundle` 全部落盘
- `validator_execution_report`：要求所有合法样例通过，所有非法样例被拒绝
- `graph_native_tensorized_execution`：要求 `native_operator_execution = true`、`tensorized_gui_source_enabled = true`、`remaining_gaps = []`
- `end_to_end_executor_closure`：要求 `workflow -> teaching -> training -> edge infer -> compare/audit/replay/trace` 所需 closure chain 齐全

**最小正式产物**

- `schema_bundle_manifest.json`
- `field_dictionary_manifest.json`
- `validator_execution_report.json`
- `strict_alignment_acceptance.json`
- `q2rl_strict_acceptance.json`
- `graph_native_tensorized_execution_report.json`
- `end_to_end_executor_closure.json`
- `upkg39_completion_manifest.json`
- `artifact_index.json`
- `stage_trace.jsonl`
- `upkg39_report.json`
- `summary.json`

**正式执行入口**

- release CLI 主入口：`cgc gate upkg39`
- engine CLI milestone：`python3 ComputeGraphCompiler-main/cgc_engine/agent/cli.py pipeline --milestone upkg39`
- product export：`cgc_engine.product.run_upkg39_gate`

**与 `3.8 / 4.0` 的边界**

- `3.9` 不替代 `3.8` 的示教/训练/比较主链，而是在其之上做 strict closure
- `3.9` 不等于 `4.0`，也不承载 `psi0 / realtime-vla / embodied benchmark`
- 现有 `m79` 继续保留为 `UPKG 4.0`，因此 `3.9` 以独立 `upkg39` gate 形式存在

**统一失败归因**

- `strict_alignment_threshold_not_restored`
- `validator_execution_failed`
- `invalid_sample_not_rejected`
- `graph_native_tensorization_incomplete`
- `closure_chain_incomplete`
- `missing_or_failed_upstream_m78`

**2026-06-22 正式验收结果**

- `cgc gate upkg39 = PASS`
- `strict_alignment_acceptance = PASS`
- `q2rl_strict_acceptance = PASS`
- `validator_execution_report = PASS`
- `graph_native_tensorized_execution = PASS`
- `end_to_end_executor_closure = PASS`
- `upkg39_completion_manifest = PASS`

**本轮正式修正说明**

- 本次收口的主阻塞点不是 `UPKG 3.9 strict closure` contract 本身失败
- 真正问题是 `upkg38` 聚合阶段未将 `pipeline contract` 正式向上游 gate 传递
- 该缺口会导致 `m72 / m77 / m78 / upkg39` 在聚合验收时把 `pipeline_kernel_contract_artifacts` 误判为未就绪
- 现已补齐上游 `fallback` 机制：当本地 root report 未携带所需 contract 时，后续 gate 允许回退读取上游 gate 已存在的 `pipeline_kernel_contract_artifacts / pipeline_contract_descriptor`
- 因此 `UPKG 3.9` 已可在不破坏现有 `3.8` 与 `4.0` 边界的前提下正式收口并通过验收

---

`UPKG 3.0` 不是从零重新发明 gate，而是建立在现有 `M7.x` 之上：

| `UPKG 3.0` Gate | 主要继承来源 | 当前已具备基础 |
|---|---|---|
| `3.1` | `M7 / M7.1` | `dynamic_trace`、`soft_rt_replay`、`industrial_audit`、`state_compression` |
| `3.2` | `M7.2` | GUI/agent route、runtime trace、industrial audit |
| `3.3` | `M7.3` | `bridge artifact`、`publish`、`edge delivery evidence` |
| `3.4` | `ExecutionContext / matrix_axes / stage_trace` 主干 | 单一真源、summary、matrix 传播 |
| `3.5` | `industrial_audit` 与六元一体架构概念 | hash chain、事件链、归因框架雏形 |
| `3.6` | `Subterranean Agent` 兼容性缺口、graph-native integration 缺口与 `UPKG 3.0` 实现残缺项 | roadmap 缺口清单、closure plan、训练侧产品链规范、GUI graph-native 收口要求 |
| `3.7` | `M7.2 + M7.3 + Unified Pipeline Kernel Design v1.0` | 端云训练推理模式、GUI agent 训练后模型端侧推理、Q2RL 后训练约束 |
| `3.8` | `M7.2 + M7.3 + 3.7 Cloud-Edge 主干` | GUI agent 示教模式、示教训练模型、端云下推、纯大模型六元推理、比较图、错误图 |
| `3.9` | `3.8 + v0.2 正式 contract` | strict alignment、schema materialization、validator execution、tensorized graph-native closure、end-to-end executor closure |

---

## 六、2026-06-20 正式 Rerun Artifact 索引

本次 `UPKG 3.0` 已完成一轮正式 rerun，并补充了一轮 release CLI 全量 gate 重跑索引，当前顶层汇总 artifact 为：

- 完成矩阵：`/private/tmp/upkg30_formal_pass_20260619/upkg30_completion_matrix.json`
- formal manifest：`/private/tmp/upkg30_formal_pass_20260619/upkg30_formal_pass_manifest.json`
- release CLI 聚合 report：`/private/tmp/upkg30_cli_alias_20260620/report.json`
- release CLI 全量 gate 重跑索引：`/private/tmp/full_gate_rerun_20260620/release/release_gate_status_index.json`
- release CLI 主入口：`cgc gate upkg30`

本轮正式结论为：

- `3.1 = PASS`
- `3.2 = PASS`
- `3.3 = PASS`
- `3.4 = PASS`
- `3.5 = PASS`
- `3.6 = PASS`
- `3.7 = PASS`

补充正式结论（`2026-06-20`）：

- `3.8 = PASS`
- `m78 / upkg38` 已成为 `3.8` 独立 gate
- `release CLI` 与 `engine CLI` 均已有正式 PASS artifact
- `engine CLI` 在 `pipeline --milestone m78` 下最外层 `ok = true`

其中 gate 对应关系为：

- `3.1` 对外入口为 `upkg31`，由 `m7` 正式 PASS artifact 承载
- `3.2` 对外入口为 `upkg32`，由 `m72` 正式 PASS artifact 承载
- `3.3` 对外入口为 `upkg33`，由 `m73` 正式 PASS artifact 承载
- `3.4` 对外入口为 `upkg34`，由 `m72` 正式 PASS artifact 承载
- `3.5` 对外入口为 `upkg35`，由 `m72` 正式 PASS artifact 承载
- `3.6` 对外入口为 `upkg36`，由 `m72` 正式 PASS artifact 承载
- `3.7` 对外入口为 `upkg37`，由 `m77` 独立 gate 与 `m72` supporting artifact 共同承载

推荐对外 `cgc` 指令为：

- `cgc gate upkg30`
- `cgc gate upkg31`
- `cgc gate upkg32`
- `cgc gate upkg33`
- `cgc gate upkg34`
- `cgc gate upkg35`
- `cgc gate upkg36`
- `cgc gate upkg37`
- `3.8` 由 `m72` 的示教/推理 artifact 与独立 `m78` gate 共同承载
- `cgc gate upkg39`

注：`3.6 = PASS` 表示 closure artifact、graph-native GUI source integration artifact 与统一 closure plan 已正式存在并通过；不等价于训练侧 `workflow -> trajectory -> fine-tune -> governance` 已成为完整 end-to-end executor。`3.7 = PASS` 表示 `cloud-train / edge-infer / GUI agent edge inference / Q2RL post-train` 的产品模式 contract 与正式 artifact 已落地；不等价于具身 `realtime-vla` comparative 或 `UPKG 4.0` benchmark 已成立。`3.8 = PASS` 表示 `GUI agent` 示教、云侧训练模型、端云下推、端侧 `pure_llm_six_element_inference`、比较图与错误图均已有正式 gate artifact；不等价于已正式采用某个 GUI foundation model 作为当前主决策引擎。

---

## 七、Subterranean Agent 兼容性与 Roadmap 判断

`Compiling Agentic Workflows into LLM Weights: Near-Frontier Quality at Two Orders of Magnitude Less Cost` 这类 `Subterranean Agent` 路线，本质上属于：

```text
业务 DAG / Agent workflow
    ->
轨迹合成与样本生成
    ->
权重内化微调
    ->
无外部编排器或弱编排器执行
```

它与 `CGC` 当前主线的关系，应明确界定为：

- 不是重复能力
- 已在 `UPKG 3.8` 中落地最小兼容产品链
- 而是“训练侧流程内化范式”与“运行时全栈底座”的高度互补关系

### 7.1 当前已经具备的兼容底座

`CGC` 当前已经具备、且可直接承接这一路线的底座至少包括：

- `工作流 / agent route`
  - 已有 `M7.2` 这类 agent route，可作为后续 `workflow -> trajectory` 的统一入口
- `State ABI`
  - 可作为流程内化后模型产物是否合法、是否需要 runtime branch 的上位语义约束
- `bridge artifact + edge-cloud protocol`
  - 可承接内化后权重、adapter、contract 的发布与交付
- `industrial audit / replay`
  - 可补齐“流程进入权重后可观测性下降”的工业化短板
- `TrueOrthoKDA active runtime`
  - 可承接内化后模型在 edge / cloud / state transport 场景下的正式运行时证据

### 7.2 当前缺失的核心模块

`CGC` 当前已经内建的最小训练侧产品链包括：

- `workflow DAG` 标准化导入与 `compute graph` 插入契约
- GUI 示教证据收集与 teaching/replay bundle
- `cloud_supervised_plus_q2rl` 训练期模型优化链
- 训练后模型下推到 edge 的 `pure_llm_six_element_inference` 契约
- `compare / audit / replay / trace` 的正式 CLI 收口

`CGC` 当前仍未完全通用化的部分，是下面这些更强的编译式能力：

- DAG 遍历后的大规模轨迹 / 样本自动合成编排器
- 面向任意业务流程内化的通用 `LoRA / full fine-tune / expert fine-tune` 编译器
- 内化模式与解释模式的双模式治理自动切换器
- 内化后决策路径与外部审计事件的一键对位治理平面

因此，更准确的判断不是“CGC 仍完全没有 Subterranean Agent”，而是：

- `CGC` 已在 `UPKG 3.8` 中具备最小兼容的 `workflow -> teaching -> Q2RL -> edge inference -> compare/audit/replay/trace` 主链
- 仍缺的是面向任意 DAG 的通用高阶“流程 -> 权重”编译器

上述缺失项在 `UPKG 3.0` 中，统一收口到：

- `3.6 Missing Capability Closure Gate`

### 7.3 在 `UPKG 3.0` 中应如何表述

这一路线在 `UPKG 3.0` 中应被表述为：

- `已落地的最小兼容产品链 + 持续扩展中的高价值 roadmap 能力`
- `与现有 runtime / bridge / audit 主线兼容`
- `其通用化程度不作为当前版本正式 PASS 前提`

也就是说，`UPKG 3.0` 当前可以承认：

- `Subterranean Agent` 式训练期权重编译，是 `CGC` 已开始吸收并在 `3.8` 中落地的训练侧范式
- 当前已正式验收成立的是最小兼容产品链，而不是通用化完整编译器

### 7.4 建议的正式口径

建议在后续技术文稿中统一使用如下口径：

> `CGC` 已在 `UPKG 3.8` 中原生提供 `Subterranean Agent compatible` 的最小产品链：`workflow DAG -> GUI teaching -> cloud_supervised_plus_q2rl -> edge pure_llm_six_element_inference -> compare/audit/replay/trace`。  
> 这条链通过 `cgc agent import-dag / teach / train / infer / visualize / compare / audit / replay / trace` 对使用者开放，并通过 `agent_graph_insertion_contract.json` 支持将 DAG 或训练后模型作为计算图子图/模型节点插入。  
> 但 `CGC` 当前仍未把这一路线完全泛化为“任意业务 DAG -> 大规模轨迹合成 -> 通用权重编译器”的完整训练编译系统。  
> 因此更准确的表述不是“没有能力”，而是“已具备正式可验收的兼容产品链，并保留更通用编译器能力作为持续增强方向”。

---

## 八、第一批实施接线

为了让 `UPKG 3.0` 从白皮书进入实际实现，第一批接线建议只做最小闭环，不一次性改全：

### 8.1 第一批必须落地

- 给 `M7 / M7.1 / M7.2 / M7.3` 补统一的 `failure_attribution` 字段
- 给 `summary` 与 `runtime_contract / publish_manifest / bridge_info` 补统一 `matrix_axes`
- 给 `agent route` 补 tool / workflow / runtime host 三类结构化事件
- 给 audit 事件补六元分类字段
- 给 `3.6` 补第一版 `gap_register / closure_plan / workflow_dag_schema / fine_tune_profile`
- 给 `GUI route` 补 `source registry / stage bindings / operator graph / execution context`

### 8.2 第一批不做

- 不引入 `realtime-vla`
- 不做官方 `psi0` comparative
- 不做具身 benchmark
- 不做 `>5x` threshold
- 不做 appendix 中的具身深化实验项

### 8.3 第一批代码落点

- `cgc_engine/product/m7_gate.py`
- `cgc_engine/product/m72_gate.py`
- `cgc_engine/product/m73_gate.py`
- `cgc_engine/pipeline.py`
- `cgc_engine/agent/gui_graph_native.py`
- `cgc_engine/agent/*` 中负责 summary / runtime evidence 的路径
- `cgc_engine/train/*` 中后续承接 `workflow -> trajectory -> fine-tune` 的路径

### 8.4 当前完成度口径

- `gate-native`
  - 当前已进入 `80%+` 的可运行状态
  - `m7 / m72 / m73 / pipeline / cgc run` 已共享同一套 `summary / matrix_axes / failure_attribution / six-element` 主线
- `graph-native`
  - 当前约处于 `50%` 左右的结构化收口阶段
  - 已完成 `source registry / stage bindings / operator graph / execution context`
  - 但尚未完成每个 stage 的 `native operator execution`

---

## 八、建议归档

每次 `UPKG 3.0` rerun，建议至少保留：

- `report.json`
- `summary.json`
- `stage_trace.jsonl`
- `failure_attribution.json`
- `events.jsonl`
- `chain_head.json`
- `publish_manifest.json`
- `runtime_contract.json`
- `bridge_info.json`
- `runtime_evidence.json`
- `gap_register.json`
- `closure_plan.json`

---

## 九、一句话结论

`UPKG 3.0` 的本质不是再新增一组零散 gate，而是把现有 `M7 / M7.1 / M7.2 / M7.3` 已成立的能力收敛成一套正式、统一、可归因、可发布、可审计的通用 agent 产品化 gate。
