# CGC Unified Pipeline Kernel Gate 2.0（UPKG 2.0）Model 产品化技术白皮书

**版本**: v1.0  
**状态**: 正式收口草案  
**定位**: 定义 `UPKG 2.0` 作为 `Unified Pipeline Kernel` 在 `model` 领域的正式产品化 gate，统一承接 `M6`、`M8`、`UPKG_1.0/1.1` 已成立能力，并吸收 `UPKG 3.x` 已验证的可审计、可回溯、可回放与失败归因要求。

---

## 一、文件定位

从当前版本开始，`UPKG` 的产品分层明确收口为四层：

- `UPKG 1.0`
  - `kernel`
- `UPKG 2.0`
  - `model`
- `UPKG 3.0`
  - `agent`
- `UPKG 4.0`
  - `embodied`

`UPKG 2.0` 不再重复 `UPKG 1.x` 对统一内核的底层验收，也不提前吞并 `UPKG 3.0/4.0` 的 agent 与 embodied 范围，而是回答下面五个问题：

- 如何把 `M6` 的 bundle / build / run / verify 产品化闭环升级为正式 `model gate`
- 如何把 `M8` 的 `cgc list / run / serve / claude / build` 等产品入口收敛成统一模型交付口径
- 如何把 `UPKG_1.0/1.1` 已成立的 kernel、runtime、contract artifact 变成 `model` 层的单一真源
- 如何把 `UPKG 3.x` 已成立的 `auditability / traceability / replayability / failure attribution` 下沉到模型产品链，而不是只留在 agent
- 如何给未来的 `cgc model` 正式 CLI 提供一套与 `cgc agent` 对等的命令设计与最小产物规范

本文件与上位文档的关系如下：

- `docs/technical_whitepapers/CGC_Unified_Pipeline_Kernel_Design_v1.0_zh_CN.md`
  - 负责四层 `UPKG` 总体分界与统一主干
- `CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md`
  - 负责 `kernel` 范围内 `M1-M7.5` 的工程验收与 accepted artifacts
- `CGC_Edge_Engine_Whitepaper_v1.0.md`
  - 负责 `M6` 的端侧运行与产品交付边界
- `CGC_UPKG_3_0_AGENT_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md`
  - 提供 `audit / replay / trace / failure attribution` 的产品化写法参考
- `docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/`
  - 提供 `CGC_Gate_2.0` 的 `done / proof / target` 正式口径；当前 `UPKG 2.0` 在该 Gate 中被明确定义为 `proof` 承接层，而不是 2.0 核心 layer-adaptive 能力本体
- 本文件
  - 负责 `UPKG 2.0 model` 的正式 gate 定义、产物要求、CLI 设计原则与当前已落地/待补缺口

---

## 二、范围与非范围

### 2.1 范围

`UPKG 2.0` 只处理 `model` 产品化要求，覆盖：

- `model discovery`
- `model packaging`
- `model run / serve / verify`
- `edge/cloud route decision`
- `model runtime contract consumption`
- `audit / replay / trace / compare` 在模型产品链上的最小落地
- `release build / dist manifest / installable artifact`

### 2.2 非范围

以下内容不属于 `UPKG 2.0`：

- 通用 agent workflow 编排、GUI teaching、tool execution、runtime host orchestration
- `realtime-vla`、`psi0` 官方训练与具身 comparative
- 具身场景中的 feedback、view-invariance、one-shot、structured conditioning
- 把任何单一 benchmark 分数直接当作 `model product gate` 的唯一通过标准

换句话说：

- `UPKG 1.0` 负责证明内核成立
- `UPKG 2.0` 负责证明模型能被发现、被打包、被运行、被验证、被审计
- `UPKG 3.0` 负责证明 agent 链路能被产品化
- `UPKG 4.0` 负责证明 embodied 链路能被产品化

### 2.3 与 `CGC_Gate_2.0` 的关系

`UPKG 2.0` 与 `CGC_Gate_2.0` 的关系，当前应统一写成：

- `UPKG 2.0`
  - 负责 `model productization`
- `CGC_Gate_2.0`
  - 负责 layer-adaptive edge-cloud PD disaggregation

两者不是互相替代，而是上下层关系：

- `UPKG 2.0` 证明 `Gate 2.0` 最终可以进入正式模型产品链
- `CGC_Gate_2.0` 证明端云 layer-adaptive runtime 与治理边界

因此，在 `CGC_Gate_2.0` 的当前正式口径里：

- `UPKG 2.0 = proof`

这表示：

- `UPKG 2.0` 已经为 `Gate 2.0` 提供强承接关系
- 但不等于 `max_local_layer / finished_layer + 1 / hidden_states + partial_kv` 已经闭环

---

## 三、来源收敛

`UPKG 2.0` 的正式边界来自四条现有来源线：

### 3.1 来自 `UPKG_1.0/1.1`

继承以下已成立能力：

- `Unified Pipeline Kernel`
- `8-step` 单一真源 `report.json`
- `execution_context.json`
- `state_abi.json`
- `strategy_decision.json`
- `compatibility_report.json`
- `contract_manifest.json`
- `system_execution_manifest.json`
- `distributed_runtime_bootstrap.json`

`UPKG 2.0` 不重新定义这些 artifact，而是要求 `model gate` 在内部优先消费它们。

### 3.2 来自 `M6`

继承以下正式产品化闭环：

- `cgc build`
- `cgc run`
- `cgc product`
- `cgc verify`
- bundle 描述与双侧 `report.json` 聚合验收
- `build_bundle_gate`
- `run_bundle_gate`
- `second.cache_hit = true`

`M6` 提供的是最小产品交付语义，也就是“一个模型产品必须能被构建、运行、二次命中并被验证”。

### 3.3 来自 `M8`

继承以下正式模型产品入口：

- `cgc list`
- `cgc run`
- `cgc serve`
- `cgc claude`
- `cgc build`
- `CGC_Release/dist`、build matrix、platform package format、size budget
- local success 与 edge-cloud takeover 双验收
- route decision、response contract、streaming takeover、release build contract

`M8` 提供的是“模型产品如何被终端用户发现、调用、接管与发行”的正式 DX 与交付边界。

### 3.4 来自 `UPKG 3.x`

继承以下产品化质量属性，但收口到 `model` 场景：

- `auditability`
- `traceability`
- `replayability`
- `comparability`
- 结构化 `summary.json`
- 结构化 `artifact_index.json`
- `events.jsonl` / `stage_trace.jsonl`
- `failure attribution`

这里的意思不是把 `agent` 功能搬进 `model`，而是把它已经证明可行的审计与回放方法，下沉为 `model gate` 的正式要求。

---

## 四、总体目标

`UPKG 2.0` 要证明：

- 模型产品链已经不再只是“能跑一次”的 demo，而是可发现、可构建、可接管、可验证、可审计、可回放的正式交付链
- `M6` 的 bundle/verify 与 `M8` 的产品入口不再是两套平行语义，而是统一收口到同一套 `model gate`
- `UPKG_1.0/1.1` 生成的 contract artifacts 已经成为模型产品链的 source-of-truth
- 模型侧也开始采用 `UPKG 3.x` 风格的统一摘要、统一索引与统一失败归因
- 后续 `cgc model` CLI 可以沿用 `cgc agent` 的命令设计模式，而不是继续散落为 `run/list/build/serve` 的松散别名集合

---

## 五、六个正式 Gate

`UPKG 2.0` 建议收敛为六个正式子 gate。

### 5.1 `2.1 Model Contract Gate`

**目标**

- 确保模型产品链不是绕开统一 kernel 自行拼装，而是正式消费 `pipeline_kernel_contract_artifacts`

**最小正式产物**

- `report.json`
- `execution_context.json`
- `state_abi.json`
- `contract_manifest.json`
- `system_execution_manifest.json`
- `pipeline_contract_descriptor`

**最小 PASS 条件**

- `execution_context / state_abi / contract_manifest / system_execution_manifest` 路径全部存在
- `report.json` 可直接回指上述 artifact
- `selected_model`
- `resolved_model_path`
- `selected_backend`
- `selected_route`
- `decision_reason.code`

**统一失败归因**

- `pipeline_contract_missing`
- `state_abi_not_ready`
- `manifest_not_ready`
- `route_context_missing`

### 5.2 `2.2 Model Package Gate`

**目标**

- 把 `M6 build_bundle_gate` 与 `M8 build release contract` 统一成模型打包与发行 gate

**最小正式产物**

- `bundle_manifest.json`
- `build_report.json`
- `dist_manifest.json`
- platform build matrix
- `artifact_sha256`
- `executable_sha256`

**最小 PASS 条件**

- bundle 可生成
- 输出物存在且大小大于 `0`
- 三平台 package format 可被结构化描述
- `supported_platforms` 明确可见
- size budget 有 `PASS/WARN/FAIL` 明确结果

**统一失败归因**

- `bundle_generation_failure`
- `dist_manifest_missing`
- `build_matrix_incomplete`
- `artifact_size_over_budget`

### 5.3 `2.3 Model Runtime Gate`

**目标**

- 把 `cgc run / cgc serve` 的 local success 与 edge-cloud takeover 统一为模型运行 gate

**最小正式产物**

- `run_report.json`
- `route_decision.json`
- `edge_inference_bridge.json`
- streaming response contract
- runtime summary

**最小 PASS 条件**

- 本地路由可成功完成一次 `local success`
- 接管路由可成功完成一次 `takeover success`
- `selected_route / selected_backend / decision_reason.code` 可被结构化记录
- `local_execution` 与 `cloud_bridge_used` 不得靠口头解释推断

**统一失败归因**

- `local_runtime_failure`
- `route_decision_invalid`
- `takeover_contract_failure`
- `streaming_contract_failure`

### 5.4 `2.4 Model Verify Gate`

**目标**

- 把 `M6 verify`、`M8 acceptance contract` 与 `UPKG 1.x report` 汇总成模型产品验证 gate

**最小正式产物**

- unified `report.json`
- `summary.json`
- `artifact_index.json`
- edge/cloud verify result

**最小 PASS 条件**

- `report.json` 为单一真源
- 子报告路径都能被反向索引
- 至少存在一次正式 `compare` 或 `verify` 结果
- 二次运行时 cache / repeatability 结果可见

**统一失败归因**

- `single_source_of_truth_missing`
- `verify_report_incomplete`
- `cache_repeatability_failure`
- `compare_artifact_missing`

### 5.5 `2.5 Model Audit-Replay Gate`

**目标**

- 把 `UPKG 3.x` 的审计、回放、追溯能力下沉到模型产品链

**最小正式产物**

- `events.jsonl` 或 `stage_trace.jsonl`
- `replay_anchor.json`
- `audit_summary.json`
- `failure_attribution.json`

**最小 PASS 条件**

- 至少一次正式运行具备 trace 事件链
- 至少一份 replay anchor 可回指关键 artifact
- 失败项有结构化 attribution，而不是 stderr 文本堆叠
- `auditability / traceability / replayability` 至少有显式字段输出

**统一失败归因**

- `event_chain_missing`
- `replay_anchor_missing`
- `audit_summary_missing`
- `failure_attribution_missing`

### 5.6 `2.6 Model CLI Product Gate`

**目标**

- 把当前散落的模型指令收敛成未来正式的 `cgc model` 产品入口

**正式设计原则**

- `cgc model` 的命令形状必须参考 `cgc agent`
- 不再只暴露松散的顶层 `run / serve / build / verify`
- 要有与 `cgc agent` 对等的子命令组织、统一 `--output-dir`、统一 `--json`、统一 artifact summary 与统一 replay/audit 入口

**建议最小子命令**

- `cgc model list`
- `cgc model package`
- `cgc model run`
- `cgc model serve`
- `cgc model verify`
- `cgc model compare`
- `cgc model audit`
- `cgc model replay`
- `cgc model trace`

**最小 PASS 条件**

- CLI 入口具备统一 schema
- 每个子命令都能输出结构化产物
- 至少 `run / serve / verify / audit / replay / trace` 具备正式 artifact 路径

**统一失败归因**

- `cli_shape_not_aligned`
- `structured_output_missing`
- `audit_route_missing`
- `replay_route_missing`

---

## 六、当前已落地与下一刀要补

### 6.1 当前已落地

以下能力已经有明确来源可承接进 `UPKG 2.0`：

- `UPKG_1.0/1.1` 的统一 kernel contract artifacts
- `M6` 的 `build / run / product / verify` 最小闭环
- `M8` 的 `list / run / serve / claude / build` 与 release build contract
- `M8` 的 route decision、takeover contract、response contract、streaming contract
- `UPKG 3.x` 的 `summary / artifact_index / audit / replay / trace / failure attribution` 写法

### 6.2 当前未完全落地

以下能力应明确标记为 `UPKG 2.0` 的后续补齐项，而不是伪装成现状：

- 当前还没有正式的 `cgc model` 子命令树
- 当前 `cgc model` 还没有像 `cgc agent` 那样的 `audit / replay / trace / compare` 对等入口
- 当前模型产品链的 `events.jsonl`、`replay_anchor.json`、`failure_attribution.json` 还没有全量标准化
- 当前 `M6` 与 `M8` 的 accepted artifact 仍分散在不同脚本与 report，尚未完全统一到 `UPKG_2.0` 单一聚合入口

### 6.3 正式口径

因此本文件采用以下正式口径：

- 已有 `M6/M8/UPKG_1.x/UPKG 3.x` 能力可以作为 `UPKG 2.0` 的来源与 accepted baseline
- `UPKG 2.0` 的白皮书可以先正式定义
- 但 `cgc model` CLI 与模型侧审计回放产物，仍属于下一阶段产品化补齐工作

---

## 七、推荐执行入口

在 `UPKG 2.0` 完整落地前，当前推荐把现有 CLI 理解为未来 `cgc model` 的过渡入口：

- `cgc list`
  - 未来对应 `cgc model list`
- `cgc run`
  - 未来对应 `cgc model run`
- `cgc serve`
  - 未来对应 `cgc model serve`
- `cgc build`
  - 未来对应 `cgc model package`
- `cgc verify`
  - 未来对应 `cgc model verify`
- `cgc claude`
  - 作为模型侧客户端接入兼容入口保留，但不替代 `model audit/replay/trace`

推荐后续新增的正式命令组为：

```bash
cgc model list
cgc model package
cgc model run
cgc model serve
cgc model verify
cgc model compare
cgc model audit
cgc model replay
cgc model trace
```

这些命令的参数风格应直接参考现有 `cgc agent`：

- 统一 `--output-dir`
- 统一 `--json`
- 统一 artifact root
- 统一 session/index 概念
- 统一 `audit / replay / trace / compare` 的输出形状

---

## 八、当前收口结论

截至当前版本，`UPKG` 的产品分层正式收口为：

- `UPKG 1.0`
  - `kernel`
- `UPKG 2.0`
  - `model`
- `UPKG 3.0`
  - `agent`
- `UPKG 4.0`
  - `embodied`

其中：

- `UPKG 1.0` 解决“统一内核是否成立”
- `UPKG 2.0` 解决“模型产品是否可发现、可打包、可运行、可验证、可审计、可回放”
- `UPKG 3.0` 解决“agent 产品链是否成立”
- `UPKG 4.0` 解决“embodied 产品链是否成立”

因此，`UPKG 2.0` 的正式定义不是新造一套平行体系，而是把：

- `M6`
- `M8`
- `UPKG_1.0/1.1`
- `UPKG 3.x` 的审计回放方法

整理为一套面向 `model productization` 的统一 gate。
