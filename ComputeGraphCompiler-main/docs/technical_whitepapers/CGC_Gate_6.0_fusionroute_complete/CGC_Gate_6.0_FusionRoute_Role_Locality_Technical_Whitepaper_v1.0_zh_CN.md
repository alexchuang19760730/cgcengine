# CGC Gate 6.0 FusionRoute Role Locality / Placement 技术白皮书 v1.0

**版本**: v1.0  
**状态**: `draft_projection`  
**归属 Gate**: `CGC_Gate_6.0_fusionroute_complete`  
**定位**: 定义 FusionRoute 各角色模型在 `cloud / edge / hybrid` 之间的部署位置决策、命名方案、CLI 投影与 verifier 草案。  
**治理边界**: 本文定义的是 Gate 6.0 的新增能力草案，不直接变更 Gate 6.0 当前 `22 done / 1 integrated` 的正式统计。

---

## 1. 为什么挂到 Gate 6.0

`role locality / placement` 不属于单一 agent 工作流，也不只属于 embodied 能力，因此不宜放到：

- `Gate 5.0`
  - 因为 Gate 5.0 应专注 agent runtime、治理、trace、replay、审计
- `Gate 4.0`
  - 因为 Gate 4.0 应专注 embodied substrate，而不是跨角色的部署控制面

该能力最适合挂到 `Gate 6.0`，原因是：

- `FusionRoute` 已归属于 Gate 6.0 的控制面
- `role locality` 是跨 `agent / coding / embodied` 的统一路由策略
- `cloud / edge / hybrid` placement 本质上属于 runtime topology 与 policy decision

---

## 2. 能力定义

### 2.1 目标

让以下角色都可以由 FusionRoute 决定部署位置：

- `Hermes`
- `TMAX`
- `UI-TARS`
- `Coding Executor`
- `CLI-Universe`

### 2.2 可选位置

| Locality | 说明 |
|----------|------|
| `cloud` | 角色常驻云端 GPU / server runtime |
| `edge` | 角色常驻端侧设备或本地节点 |
| `hybrid` | 角色主逻辑在云端，但可把执行或状态同步到端侧 |
| `auto` | FusionRoute 根据 policy 自动决定 |

---

## 3. 决策输入

FusionRoute 做 locality decision 时，建议至少读取以下输入：

| 输入 | 说明 |
|------|------|
| `task_type` | 当前任务属于 agent / coding / embodied / synthesis |
| `latency_budget_ms` | 延迟预算 |
| `privacy_level` | 数据是否必须留在端侧 |
| `device_availability` | 端侧设备是否可用 |
| `gpu_memory_budget_gb` | 云侧/端侧显存预算 |
| `network_condition` | 带宽、抖动、是否允许 state handoff |
| `runtime_identity` | 当前角色实际绑定的 runtime endpoint |
| `policy_override` | 人工指定的强制 locality |

---

## 4. 角色与 locality 的设计建议

| 角色 | 默认 locality | 可切换 locality | 备注 |
|------|----------------|------------------|------|
| `Hermes` | `cloud` | `edge`, `auto` | 默认做治理与编排，通常放云端 |
| `TMAX` | `cloud` | `edge`, `auto` | 长程规划通常需要更大模型与更高显存 |
| `UI-TARS` | `hybrid` | `edge`, `cloud`, `auto` | 执行器可能在端侧执行、在云端做重推理 |
| `Coding Executor` | `cloud` | `edge`, `auto` | coding 主链默认云端，必要时可下沉本地 |
| `CLI-Universe` | `cloud` | `edge`, `hybrid`, `auto` | 数据合成默认离线云端执行 |

---

## 5. Gate 6.0 capability 命名方案

本文中的核心 locality / placement artifact 已正式并入 Gate 6.0 capability closure；以下保留的是 capability 语义分层：

| Capability ID | 建议状态 | 说明 |
|---------------|----------|------|
| `fusionroute_role_locality_projection` | `planned_projection` | 角色 locality 决策总能力 |
| `role_placement_policy` | `planned_projection` | placement policy 输入与决策逻辑 |
| `role_runtime_binding_contract` | `planned_projection` | 角色到 runtime endpoint 的静态契约 |
| `role_locality_contract` | `done` | `role -> locality -> endpoint` 的 machine-readable contract |
| `role_edge_cloud_handoff` | `planned_projection` | 当 locality 切换时的 state handoff 规则 |
| `placement_decision_report` | `done` | 每次决策输出 JSON 证据报告 |

---

## 6. CLI 草案

建议为 Gate 6.0 增加以下 CLI 投影：

```bash
python3 cgc_engine/cli.py fusionroute plan --task-type CODEGEN --print-json
python3 cgc_engine/cli.py fusionroute placement show --role UI-TARS --print-json
python3 cgc_engine/cli.py fusionroute placement verify --task-type EXECUTION --role UI-TARS --print-json
python3 cgc_engine/cli.py fusionroute verify --capability fusionroute_placement_decision_report --print-json
```

其中 `fusionroute placement verify` 的目标是输出：

- 角色
- locality
- runtime endpoint
- 触发条件
- policy override
- evidence path

---

## 7. Verifier 草案

建议配套以下 verifier 名称：

| Verifier | 说明 |
|----------|------|
| `validate_fusionroute_role_locality_projection` | 校验总 locality 决策链 |
| `validate_role_placement_policy` | 校验策略输入与决策输出 |
| `validate_role_runtime_binding_contract` | 校验 endpoint 绑定是否与 contract 一致 |
| `validate_role_edge_cloud_handoff` | 校验 locality 变化时的 ABI / transport handoff |
| `validate_placement_decision_report` | 校验 JSON 证据报告结构与字段完整性 |

建议输出正式 JSON：

```json
{
  "role": "UI-TARS",
  "locality": "hybrid",
  "runtime_endpoint": "edge://robot-gateway-01",
  "policy_source": "fusionroute_auto_policy",
  "decision_reason": [
    "task_type=embodied_action",
    "latency_budget_ms=80",
    "device_availability=true"
  ],
  "handoff_contract": "EdgeCloudLayerHandoff",
  "status": "PASS"
}
```

---

## 8. 与 Gate 4.0 / 5.0 的关系

- `Gate 5.0`
  - 消费 role locality 决策
  - 但不定义 locality 控制面本身
- `Gate 4.0`
  - 提供 embodied capability substrate
  - 但不负责决定角色放在云端还是端侧

因此最合理的边界是：

- `Gate 5.0`：运行 agent 角色
- `Gate 4.0`：提供 embodied substrate
- `Gate 6.0`：决定角色 placement policy

---

## 9. 正式挂接方式

本白皮书将通过以下方式挂接到真源：

- `CGC_Gate_6.0_fusionroute_complete_summary.example.json`
- `CGC_Gate_6.0_fusionroute_complete_checkin.example.json`
- `CGC_Gate_6.0_fusionroute_complete/gate_map.json`
- `CGC_Gate_4.0_embodied_*`
- `CGC_Gate_5.0_audit_trace_replay_visualization_*`
- `role_locality_contract.schema.json`
- `placement_decision_report.schema.json`
- `gate6_fusionroute_v2_draft_contract.json`
- `gate6_fusionroute_v2_formal_contract.json`
- `CGC_Perception_Matrix_LLM_Technical_Whitepaper_v1.0_zh_CN.md`
- `policy_suggestion_report.schema.json`
- `contract_projection_report.schema.json`
- `fusionroute_v2_formal_contract_report.json`

挂接方式采用：

- `architecture_projection_refs`
- `gate_map.capabilities`
- `gate6_capability_cli_self_harness_contract.json`

并已直接改写正式 capability 状态。

---

## 10. 下一步

1. 在 `FusionRoute` runtime 中补 `placement policy` 实现
2. 在 `self_harness_validation_framework.py` 中补 Gate 6.0 locality verifier
3. 为 `role_locality_contract` 与 `placement_decision_report` 定义正式 JSON schema
4. 通过 `fusionroute verify --capability all` 持续生成 formal contract report
5. 在双机环境补齐 `29/29` 主链远端证据
