# CGC UPKG 3.x 统一产品链路技术方案补充规范 v0.1

**版本**: v0.1  
**状态**: 第一版补充草案  
**定位**: 作为 `CGC_UPKG_3_0_AGENT_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md` 的技术补充件，面向 `DAG 导入 -> GUI 示教 -> 训练 -> 推理 -> compare -> audit -> replay -> trace` 的统一产品链路，补齐对象模型、artifact schema、训练闭环、推理 contract、证据链与验收指标。

---

## 1. 文档定位

当前 `UPKG 3.0` gate 白皮书已经回答了下面两个问题：

- `UPKG 3.x` 当前要验什么
- `agent + edge + runtime` 的正式 gate 与最小产物是什么

但它仍然偏向：

- gate 口径
- CLI 入口
- rerun 证据
- 当前工程产物

如果要把 `UPKG 3.x` 真正扩展成产品级统一链路，仅有 gate 白皮书还不够。系统还需要一份面向研发实施、联调、评审、交付与审计的技术方案补充件，明确：

- 统一对象模型
- 统一 schema 与字段语义
- `DAG` 到 `GUI` 的绑定逻辑
- 训练闭环的真实有效性定义
- `pure_llm_six_element_inference` 的正式 contract
- `compare / audit / replay / trace` 的统一验收标准

因此，本文件不替代 `UPKG 3.0 gate` 白皮书，而是补齐其下位实现规范。

### 1.1 与现有文档关系

- `CGC_Unified_Pipeline_Kernel_Design_v1.0_zh_CN.md`
  - 负责上位产品分层与统一内核定位
- `CGC_UPKG_3_0_AGENT_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md`
  - 负责 `UPKG 3.0` 当前 gate 边界、子 gate、最小 PASS 条件
- `CGC_UPKG40_EMBODIED_PSI0_REALTIMEVLA_AUDIT_REPLAY_TRACE_zh_TW.md`
  - 负责 `UPKG 4.0 embodied` 的链路型 contract 组织参考
- 本文件
  - 负责把 `UPKG 3.x` 从“可跑 gate”升级为“可实现、可验证、可审计、可交付”的统一产品链路技术方案

### 1.2 适用范围

本文件适用于：

- `UPKG 3.0 ~ 3.8`
- 以 `agent` 为中心的 GUI / workflow / edge runtime 产品链
- 面向企业客户的示教、训练、推理、审计与回放交付场景

本文件不覆盖：

- `UPKG 4.0 embodied realtime-vla` 的具身专项协议细节
- 底层操作系统驱动或浏览器内核实现
- 具体 reward 算法论文推导

---

## 2. 设计目标

`UPKG 3.x` 统一产品链路要解决的不是“单个命令是否能跑通”，而是下面六个问题：

1. 如何把业务流程导入为统一 `DAG`
2. 如何把 `DAG` 节点映射成可执行 `GUI` 意图与动作
3. 如何把示教数据转化为可训练、可评估的训练单元
4. 如何把端侧推理收敛为可结构化、可校验的 `six-element inference`
5. 如何把客户现场执行转成可复现、可审计的证据链
6. 如何用统一指标评估 `compare / audit / replay / trace`

一句话目标：

```text
把 “一组 CLI 命令 + 一批 JSON 产物” 升级为 “单一对象模型驱动的统一产品链路”。
```

### 2.1 成功标准

本技术方案成立的最小标志是：

- 所有主要 artifact 都有正式 schema 与 validator
- `DAG -> Execution Intent -> GUI Binding` 的映射清晰可实现
- `train` 能产出可审计的训练有效性证据
- `infer` 的输入输出不再是黑盒 prompt，而是结构化 six-element contract
- `replay` 不再只是视频回放，而是可重复执行的 evidence-driven replay
- `compare / audit / replay / trace` 有统一的成功阈值与失败归因口径

---

## 3. 总体架构

### 3.1 统一链路

`UPKG 3.x` 统一产品链路定义为：

```text
DAG Import
  -> GUI Teaching
  -> Data Curation
  -> Cloud Supervised / Q2RL Training
  -> Edge Inference
  -> Compare
  -> Audit
  -> Replay
  -> Trace / Attribution
```

### 3.2 模块边界

建议系统收敛为以下模块：

- `workflow import layer`
  - 负责导入业务流程、生成 `workflow / dag artifact`
- `execution planning layer`
  - 负责把 `DAG` 节点实例化为 `Execution Intent`
- `gui binding layer`
  - 负责把 `Execution Intent` 绑定到 selector / OCR / image anchor / shortcut
- `teaching & evidence layer`
  - 负责采集截图、事件、DOM、OCR、失败上下文
- `training layer`
  - 负责 `cloud supervised / Q2RL / reward / eval`
- `inference layer`
  - 负责 `pure_llm_six_element_inference`
- `audit & replay layer`
  - 负责 `compare / audit / replay / trace` 统一输出

### 3.3 云边分工

- 云侧负责：
  - dataset 构造
  - 模型训练
  - reward/eval
  - 策略版本管理
- 端侧负责：
  - 感知
  - 执行
  - six-element inference
  - 证据采集
  - replay / audit 输入落地

### 3.4 控制面与数据面

- 控制面：
  - workflow version
  - policy version
  - binding version
  - execution profile
  - acceptance profile
- 数据面：
  - state snapshot
  - action trace
  - screenshots
  - DOM/OCR evidence
  - training unit
  - replay report

---

## 4. 核心对象模型

### 4.1 总原则

`UPKG 3.x` 必须先统一对象模型，再统一命令与产物。

所有 CLI 入口、artifact、summary、report 最终都应围绕同一组核心对象组织：

- `Task`
- `Workflow`
- `DAG Node`
- `Execution Intent`
- `GUI Binding`
- `Execution Step`
- `Evidence Unit`
- `Training Unit`
- `Inference Unit`
- `Audit Finding`

### 4.2 Workflow

`Workflow` 是任务级业务流程描述，不直接等于 GUI 事件序列。

最小字段：

- `workflow_id`
- `workflow_name`
- `workflow_version`
- `entry_nodes`
- `nodes`
- `edges`
- `owner_domain`
- `schema_version`

### 4.3 DAG Node

`DAG Node` 表示业务语义节点，而不是具体点击动作。

最小字段：

- `node_id`
- `node_type`
- `intent_type`
- `preconditions`
- `postconditions`
- `failure_policy`
- `binding_ref`

示例：

```json
{
  "node_id": "approve_order",
  "node_type": "business_task",
  "intent_type": "gui_action_group",
  "preconditions": [
    "page=order_detail",
    "role=approver"
  ],
  "postconditions": [
    "status=approved",
    "toast=success"
  ],
  "failure_policy": {
    "retry": 2,
    "fallback": "human_takeover",
    "on_error_node": "capture_failure_evidence"
  },
  "binding_ref": "binding_approve_order_v3"
}
```

### 4.4 Execution Intent

`Execution Intent` 是 `DAG Node` 与实际 GUI 执行之间的中间层。

它回答的问题是：

```text
系统此时想做什么，而不是它打算按哪个坐标去点。
```

典型类型：

- `click`
- `input`
- `select`
- `wait`
- `assert`
- `navigate`
- `recover`

### 4.5 GUI Binding

`GUI Binding` 是将 `Execution Intent` 落到具体环境的绑定对象。

最小字段：

- `binding_id`
- `action_type`
- `target_locator`
- `locator_priority`
- `timeout_ms`
- `retry_policy`
- `fallback_policy`
- `environment_constraints`

### 4.6 Evidence Unit

`Evidence Unit` 是 replay 与 audit 的最小证据单元。

一个最小 `Evidence Unit` 至少包含：

- `env_snapshot`
- `perception_snapshot`
- `action_record`
- `result_record`

### 4.7 Training Unit

`Training Unit` 是训练闭环的最小可训练对象。

它应绑定：

- `task_id`
- `episode_id`
- `state_trace_ref`
- `action_trace_ref`
- `reward_trace_ref`
- `outcome`
- `source_type`

### 4.8 Inference Unit

`Inference Unit` 是一次 `pure_llm_six_element_inference` 的结构化输入输出对象。

它应具备：

- `input.goal`
- `input.state`
- `input.constraints`
- `input.candidates`
- `input.policy_context`
- `input.success_criteria`
- `output.selected_action`
- `output.expected_post_state`
- `output.fallback_plan`

---

## 5. 统一 Artifact Contract

### 5.1 问题定义

当前 `UPKG 3.x` 已有大量 JSON 产物，但仍缺少：

- schema 版本
- 字段字典
- required/optional 约束
- validator
- semantic check

因此，本方案规定：所有 `UPKG 3.x` 正式 artifact 必须使用统一包络结构。

### 5.2 统一包络

```json
{
  "schema_version": "3.0.0",
  "artifact_type": "gui_trace",
  "artifact_id": "trace_001",
  "producer": {
    "module": "cgc.upkg3.trace",
    "version": "3.0.0"
  },
  "created_at": "2026-06-21T12:00:00Z",
  "input_refs": [
    "workflow_xxx",
    "binding_xxx"
  ],
  "content": {},
  "validation_rules": {
    "required_fields": [],
    "semantic_checks": []
  }
}
```

### 5.3 最小强制字段

所有正式 artifact 必须包含：

- `schema_version`
- `artifact_type`
- `artifact_id`
- `producer`
- `created_at`
- `input_refs`
- `content`

### 5.4 Artifact 类型

建议统一为以下大类：

- `workflow_artifact`
- `dag_artifact`
- `gui_binding_artifact`
- `teaching_trace_artifact`
- `training_unit_artifact`
- `eval_report_artifact`
- `inference_artifact`
- `compare_report_artifact`
- `audit_report_artifact`
- `replay_report_artifact`
- `trace_report_artifact`

### 5.5 校验层级

artifact validator 至少分四层：

1. `schema check`
2. `enum/value range check`
3. `reference integrity check`
4. `semantic check`

### 5.6 校验失败口径

建议统一失败分类：

- `SCHEMA_INVALID`
- `FIELD_MISSING`
- `REFERENCE_BROKEN`
- `SEMANTIC_CONFLICT`
- `TIMELINE_INCONSISTENT`

---

## 6. DAG 导入与 GUI 示教映射规范

### 6.1 映射总原则

`DAG` 和 `GUI 示教` 之间不能直接一一硬绑定，而必须通过三层映射：

```text
Business DAG Node
  -> Execution Intent
  -> GUI Binding
```

### 6.2 为什么不能直接绑定

如果直接把 `DAG Node` 等价为 GUI 动作，会带来：

- 业务语义丢失
- 环境切换困难
- replay 只能重播旧坐标
- compare 无法区分“动作成功”与“业务成功”

### 6.3 Preconditions / Postconditions

每个 `DAG Node` 必须显式定义：

- `preconditions`
- `postconditions`

推荐前置条件：

- 页面状态
- 角色权限
- 焦点窗口
- 必须元素可见
- 输入数据可用

推荐后置条件：

- 业务状态变化
- 关键字段变化
- 成功提示出现
- 页面跳转完成

### 6.4 Failure Policy

每个节点必须定义失败处理策略：

- 最大重试次数
- locator fallback 顺序
- OCR / selector 切换
- 失败截图采集
- 人工接管策略

### 6.5 GUI Binding 最小结构

```json
{
  "binding_id": "binding_approve_order_v3",
  "action_type": "click",
  "target_locator": {
    "selector": "#approve-btn",
    "ocr_text": "审批通过",
    "image_anchor": null
  },
  "locator_priority": [
    "selector",
    "ocr_text",
    "image_anchor"
  ],
  "timeout_ms": 8000,
  "retry_policy": {
    "max_retry": 2,
    "retry_interval_ms": 1000
  },
  "fallback_policy": {
    "fallback_to_ocr": true,
    "fallback_to_human_takeover": true
  }
}
```

### 6.6 业务成功与 GUI 成功的区别

规范要求：

- GUI 成功不等于任务成功
- 只有满足 `postconditions` 才能记为节点成功
- replay / compare / audit 都必须以后置业务条件为准

---

## 7. GUI 示教与采集规范

### 7.1 输入

示教阶段的正式输入建议包括：

- 录屏
- 鼠标事件
- 键盘事件
- 截图
- DOM 快照
- OCR 快照
- 窗口与焦点信息
- 环境元数据

### 7.2 输出

示教阶段至少要输出：

- `teaching_trace`
- `selector_candidates`
- `binding_candidates`
- `failure_examples`
- `evidence_bundle`

### 7.3 环境元数据

建议作为 `env_snapshot` 正式保留：

- `os`
- `resolution`
- `scale_factor`
- `browser`
- `browser_version`
- `locale`
- `timezone`
- `window_size`
- `network_state`
- `permission_state`

### 7.4 感知元数据

建议正式保留：

- `selector`
- `ocr_text`
- `bbox`
- `window_handle`
- `focus_element`
- `confidence`

### 7.5 脱敏与安全

示教与 evidence 落地必须支持：

- 输入框脱敏
- 敏感区域裁剪
- 账号字段遮罩
- 客户数据最小留存

---

## 8. 训练闭环规范

### 8.1 当前缺口

如果 `train` 只输出：

- checkpoint
- report
- dataset manifest

仍然无法证明“训练真的有效”。

因此，本方案要求训练闭环明确回答：

- 数据从哪里来
- 目标是什么
- reward 如何定义
- 训练后是否真的改善执行

### 8.2 训练模式

建议分为三层：

- `cloud supervised`
  - 学习示教与标准操作
- `Q2RL`
  - 学习长流程策略与失败恢复
- `reward shaping`
  - 学习更优的执行代价与风险平衡

### 8.3 Training Unit

```json
{
  "training_unit": {
    "task_id": "task_001",
    "episode_id": "ep_001",
    "state_trace_ref": "trace_state_001",
    "action_trace_ref": "trace_action_001",
    "reward_trace_ref": "trace_reward_001",
    "outcome": "success",
    "source_type": "human_teaching"
  }
}
```

### 8.4 Reward 设计

reward 不能只在任务完成时给一个总分，建议至少拆为：

- `terminal_reward`
- `step_efficiency_reward`
- `recovery_reward`
- `risk_penalty`
- `human_takeover_penalty`

### 8.5 训练输出

训练阶段正式输出建议包括：

- `training_dataset_manifest`
- `training_run_report`
- `reward_summary`
- `checkpoint_manifest`
- `eval_report`
- `ab_compare_report`

### 8.6 训练有效性证明

`train` 必须能给出训练前后对比，而不是只给 loss。

建议最少包含：

- `task_success_rate_before/after`
- `average_step_count_before/after`
- `failure_recovery_rate_before/after`
- `human_takeover_rate_before/after`

---

## 9. pure_llm_six_element_inference 规范

### 9.1 定义目标

`pure_llm_six_element_inference` 是端侧推理链路的正式核心 contract。

其目标不是把一段 prompt 发给模型，而是把端侧决策收敛为可结构化、可回放、可审计的统一推理对象。

### 9.2 六元素正式定义

建议定义为：

- `goal`
  - 当前任务目标
- `state`
  - 当前环境状态摘要
- `constraints`
  - 风险边界、权限边界、时间边界、动作禁区
- `candidates`
  - 当前可选动作集合
- `policy_context`
  - 上下文、历史执行摘要、局部策略提示
- `success_criteria`
  - 本步或本阶段成功判据

### 9.3 输入结构

```json
{
  "input": {
    "goal": {},
    "state": {},
    "constraints": {},
    "candidates": [],
    "policy_context": {},
    "success_criteria": {}
  }
}
```

### 9.4 输出结构

```json
{
  "output": {
    "selected_action": {},
    "reasoning_summary": "",
    "expected_post_state": {},
    "risk_flags": [],
    "fallback_plan": {},
    "confidence": 0.0
  }
}
```

### 9.5 输出约束

`selected_action` 必须可映射回：

- `dag_node_id`
- `execution_intent_id`
- `gui_binding_id`

否则不能进入 replay 与 trace 主链。

### 9.6 低置信度处理

当 `confidence` 低于阈值时，系统建议支持：

- 请求更多观测
- 回退到规则策略
- 请求人工确认
- 进入安全模式

---

## 10. 客户模式证据链规范

### 10.1 目标

客户模式证据链的目标是：

```text
从 “看起来做过” 升级为 “可以重放、可以审计、可以归因”。
```

### 10.2 四层证据结构

建议统一为四层：

- 环境层
- 感知层
- 动作层
- 结果层

### 10.3 最小 Evidence Unit

```json
{
  "env_snapshot": {
    "os": "windows",
    "resolution": "1920x1080",
    "scale_factor": 1.0,
    "browser": "Chrome 137",
    "locale": "zh-CN"
  },
  "perception_snapshot": {
    "selector": "#approve-btn",
    "ocr_text": "审批通过",
    "bbox": [100, 120, 180, 150],
    "confidence": 0.94
  },
  "action_record": {
    "type": "click",
    "target": "approve_button",
    "parameters": {}
  },
  "result_record": {
    "postcondition_passed": true,
    "before_screenshot_ref": "shot_before_001",
    "after_screenshot_ref": "shot_after_001",
    "error": null
  }
}
```

### 10.4 失败证据要求

每个失败步骤必须至少留存：

- 失败前截图
- 失败后截图
- error code
- 当前 selector / OCR 快照
- fallback 选择记录
- 是否人工接管

### 10.5 人工接管证据

人工接管必须有正式边界点：

- `takeover_trigger_step`
- `takeover_reason`
- `operator_id`
- `pre_takeover_context`
- `post_takeover_outcome`

### 10.6 可回放要求

证据链必须足以支撑：

- 视频回放
- 逐事件重放
- 基于意图重执行

其中，正式产品验收应以后两者优先。

---

## 11. Compare / Audit / Replay / Trace 规范

### 11.1 Compare

`compare` 用于回答：

- 当前执行与基线相比差了什么
- 是否产生回归
- 差异是数据问题、感知问题还是策略问题

最小输出建议包括：

- `delta_step_count`
- `delta_execution_time`
- `delta_success_rate`
- `delta_postcondition_pass_rate`
- `regression_flags`

### 11.2 Audit

`audit` 用于回答：

- 证据是否完整
- 动作是否合理
- 风险控制是否符合要求
- 失败是否可解释

最小输出建议包括：

- `finding_id`
- `severity`
- `category`
- `evidence_ref`
- `root_cause`
- `fix_hint`

### 11.3 Replay

`replay` 建议支持两种模式：

- `event_replay`
  - 按原始事件重放
- `intent_reexecute`
  - 按意图重新执行

前者用于调试，后者用于证明产品可复现。

### 11.4 Trace

`trace` 要求完整记录：

- 本步目标
- 输入状态
- 推理输出
- 选择动作
- 执行结果
- 后置条件判定
- 失败恢复

---

## 12. 验收指标体系

### 12.1 统一指标

建议 `UPKG 3.x` 的 `compare / audit / replay / trace` 统一采用以下指标：

- `task_success_rate`
- `step_success_rate`
- `average_execution_time_ms`
- `p95_execution_time_ms`
- `average_step_count`
- `retry_count`
- `failure_recovery_rate`
- `human_takeover_rate`
- `postcondition_pass_rate`
- `evidence_completeness_rate`
- `replay_determinism_rate`
- `compare_delta_score`
- `audit_issue_density`

### 12.2 建议阈值

- `task_success_rate >= 95%`
- `postcondition_pass_rate >= 98%`
- `failure_recovery_rate >= 70%`
- `human_takeover_rate <= 5%`
- `evidence_completeness_rate >= 99%`
- `replay_determinism_rate >= 90%`

### 12.3 指标解释原则

规范要求：

- 不以单次 run 是否返回成功作为唯一标准
- 不以 GUI 事件是否执行作为任务成功标准
- 不以视频是否能播放作为 replay 成功标准

最终应以：

- 业务后置条件
- 证据完整性
- 可重执行性
- 失败归因清晰度

共同判定。

---

## 13. 命令与产物映射建议

### 13.1 import

输入：

- business flow DSL
- BPMN / workflow file
- external DAG

输出：

- `workflow_artifact`
- `dag_artifact`

### 13.2 teach

输入：

- GUI teaching session

输出：

- `teaching_trace_artifact`
- `binding_candidates`
- `evidence_bundle`

### 13.3 train

输入：

- curated training units

输出：

- `training_dataset_manifest`
- `checkpoint_manifest`
- `training_run_report`
- `eval_report`

### 13.4 infer

输入：

- `pure_llm_six_element_inference` payload

输出：

- `inference_artifact`

### 13.5 compare / audit / replay / trace

统一要求输出：

- `compare_report_artifact`
- `audit_report_artifact`
- `replay_report_artifact`
- `trace_report_artifact`

---

## 14. 实施优先级

### 14.1 第一优先级

先补下面三件事：

- 正式 artifact schema + validator
- `DAG Node -> Execution Intent -> GUI Binding` 映射模型
- `pure_llm_six_element_inference` 正式字段定义

### 14.2 第二优先级

- 训练闭环与 eval 指标
- evidence bundle 统一结构
- replay / trace 最小 contract

### 14.3 第三优先级

- compare/audit 统一阈值
- 客户交付格式
- schema migration 与版本兼容工具

---

## 15. 风险与待决问题

### 15.1 当前主要风险

- GUI locator 在多环境下稳定性不足
- OCR 与 DOM 的不一致可能导致错误动作
- reward 设计不合理会造成策略偏移
- 客户环境权限不足会破坏证据完整性
- 回放如果只依赖视频会退化为演示系统而不是可复现系统

### 15.2 当前待决项

- `six-element` 是否需要增加 `risk_profile` 作为第七补充字段
- `Q2RL` 的状态切片粒度是否按 step 还是按 subgoal
- replay 正式验收是否要求同时满足 `event_replay` 与 `intent_reexecute`
- 客户交付包是否强制包含脱敏后的 DOM snapshot

---

## 16. 附录建议

建议后续继续补齐以下附录：

- 附录 A：Artifact JSON Schema 清单
- 附录 B：DAG Node 字段字典
- 附录 C：GUI Binding 字段字典
- 附录 D：Six Element Inference 字段字典
- 附录 E：Evidence Bundle 字段字典
- 附录 F：验收指标计算公式
- 附录 G：错误码与失败分类表
- 附录 H：最小合法样例集

---

## 17. 结论

`UPKG 3.x` 当前方向是成立的，但如果要把它从“agent gate + CLI 产物链”升级为“统一产品链路”，系统必须补齐：

- 统一对象模型
- 统一 artifact contract
- `DAG` 到 `GUI` 的正式映射
- 训练闭环有效性定义
- `pure_llm_six_element_inference` 的正式 contract
- 可复现证据链
- compare / audit / replay / trace 的统一验收指标

本文件的定位，就是把这些补件正式写入技术方案，使 `UPKG 3.x` 可以进入下一阶段的研发实施与产品化交付。
