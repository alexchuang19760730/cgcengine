# CGC FusionRoute 最终拓扑矩阵技术白皮书 v1.0

**版本**: v1.0  
**状态**: `draft_projection`  
**定位**: 定义 FusionRoute 在 `Gate 4.0 / 5.0 / 6.0` 之间的最终拓扑矩阵、角色归属、CLI 入口与 verifier 映射。  
**治理边界**: 本文是跨 Gate 架构真源，不直接改写任何单一 Gate 已验证 capability 的 `done / integrated` 计数。

---

## 1. 文档目标

本文用于回答四个问题：

1. `Gate 4.0 / 5.0 / 6.0` 在最终产品拓扑中分别负责什么 plane
2. `Hermes / TMAX / UI-TARS / CLI-Universe / Coding Model` 分别归属于哪个主平面
3. `FusionRoute` 如何执行 `Domain Router -> Role Router` 双层分流
4. 哪些 CLI 与 verifier 应作为各角色的正式治理入口

---

## 2. 总体结论

FusionRoute 最终拓扑采用以下三层分工：

- `Gate 5.0 = Agent Runtime Plane`
- `Gate 6.0 = Coding Model Plane + Role Locality Control Plane`
- `Gate 4.0 = Embodied Capability Plane`

角色归属采用以下口径：

- `Hermes`：归属 `Gate 5.0`，负责编排、治理、审计、fallback 决策
- `TMAX`：归属 `Gate 5.0`，负责长程规划、RL 决策、任务拆解
- `UI-TARS Executor`：归属 `Gate 5.0`，负责前线执行与环境交互
- `Coding Model / Coding Executor`：归属 `Gate 6.0`，负责代码生成、repo 修改、tool-use coding、verify
- `CLI-Universe`：归属 `Offline Synthesis Plane`，负责数据合成、成功轨迹、rubric 与训练样本生产

其中 `Gate 4.0` 不再承担 `UI-TARS` 主运行时角色，而是为 `UI-TARS` 提供可调用的具身能力底座：

- robot/device control
- sensor fusion
- scene/environment state
- edge-cloud embodied protocol

---

## 3. FusionRoute 双层路由

### 3.1 第一层：Domain Router

FusionRoute 首先判定请求属于哪个 domain：

| Domain | 目标 Gate | 说明 |
|--------|------------|------|
| `agent_runtime` | `Gate 5.0` | 任务编排、审计、计划、执行 |
| `coding_runtime` | `Gate 6.0` | repo 修改、代码验证、coding tool use |
| `embodied_capability` | `Gate 4.0` | robot/device/sensor/action substrate |
| `offline_synthesis` | `CLI-Universe` | 训练数据、成功轨迹、rubric |

### 3.2 第二层：Role Router

在确定 domain 后，FusionRoute 再决定主角色与辅角色：

- `agent_runtime`:
  - primary: `Hermes` / `UI-TARS`
  - secondary: `TMAX`
- `coding_runtime`:
  - primary: `Coding Executor`
  - secondary: `Hermes` / `TMAX`
- `embodied_capability`:
  - primary: `Gate 4.0 embodied substrate`
  - consumer: `UI-TARS`
- `offline_synthesis`:
  - primary: `CLI-Universe`

---

## 4. 最终拓扑图

```text
User Request
  -> FusionRoute
    -> Domain Router
      -> Gate 5.0 Agent Runtime Plane
      -> Gate 6.0 Coding Model Plane
      -> Gate 4.0 Embodied Capability Plane
      -> Offline Synthesis Plane

Gate 5.0 Agent Runtime Plane
  -> Hermes Orchestrator
  -> TMAX Planner
  -> UI-TARS Executor
      -> calls Gate 4.0 embodied capabilities when needed

Gate 6.0 Coding Model Plane
  -> Coding Executor / Coding Model
  -> Hermes Governance Wrapper
  -> TMAX Planning Support

Gate 4.0 Embodied Capability Plane
  -> robot control
  -> device control
  -> sensor fusion
  -> environment state
  -> edge-cloud embodied protocol

Offline Synthesis Plane
  -> CLI-Universe
```

---

## 5. Gate 到 Plane 的正式矩阵

| Gate | Plane | 主职责 | 当前治理边界 |
|------|-------|--------|--------------|
| `Gate 4.0` | `Embodied Capability Plane` | 具身能力底座、设备与环境接口 | 不直接承担主 agent runtime 角色 |
| `Gate 5.0` | `Agent Runtime Plane` | agent 编排、计划、执行、审计 | `Hermes / TMAX / UI-TARS` 主归属 |
| `Gate 6.0` | `Coding Model Plane` | coding model、代码验证、CLI/tool-use coding | 同时承接 role locality / placement 控制面 |

---

## 6. Gate -> Plane -> Role -> CLI -> Verifier 矩阵

| Gate | Plane | Primary Role | Secondary Role | CLI | Verifier |
|------|-------|--------------|----------------|-----|----------|
| `Gate 5.0` | `Agent Runtime Plane` | `Hermes` | `TMAX` | `cgc gate5 audit list`, `cgc gate5 trace get`, `cgc gate5 task replay` | `fusionroute_role_runtime_binding`, `gate_test_framework --gate CGC_Gate_5.0_audit_trace_replay_visualization` |
| `Gate 5.0` | `Agent Runtime Plane` | `UI-TARS Executor` | `Hermes` | `cgc embodied infer`, `cgc embodied monitor`, `cgc embodied validate` | `fusionroute_role_runtime_binding`, `self_harness_validation_framework.py --gate 5.0` |
| `Gate 5.0` | `Agent Runtime Plane` | `TMAX Planner` | `Hermes` | `cli-universe fusionroute-run`, `cli-universe fusionroute-train` | `fusionroute_role_runtime_binding`, `gate5 runtime binding proof chain` |
| `Gate 6.0` | `Coding Model Plane` | `Coding Executor / Coding Model` | `Hermes`, `TMAX` | `python3 cgc_engine/cli.py validate --all --print-json`, `python3 cgc_engine/cli.py model verify --gate 6.0` | `self_harness_validation_framework.py --gate 6.0`, `gate6 capability -> CLI -> self-harness contract` |
| `Gate 6.0` | `Role Locality Control Plane` | `FusionRoute Router` | `MiniCPM5 Router`, policy layer | `python3 cgc_engine/cli.py model --fusion-config ...`, `python3 cgc_engine/cli.py validate --capability fusionroute` | `role_locality_contract`, `role_runtime_binding`, `placement_decision_report` |
| `Gate 4.0` | `Embodied Capability Plane` | `Embodied Substrate` | `UI-TARS` | `cgc embodied train`, `cgc embodied infer`, `cgc embodied validate`, `cgc embodied monitor` | `gate_test_framework --gate CGC_Gate_4.0_embodied`, `cgc model verify --gate 4.0 --embodied` |
| `Offline` | `Synthesis Plane` | `CLI-Universe` | `TMAX`, `Hermes` | `cli-universe fusionroute-train`, synthesis dataset builders | rubric verifier, trajectory quality checks, dataset audit |

> 说明：表中的 `CLI` 与 `Verifier` 是正式草案口径，最终 claim 仍以各 Gate 真源中的已验证条目为准。

---

## 7. TaskType 到 GateDomain 的映射建议

| TaskType | GateDomain | Primary Role | 备注 |
|----------|------------|--------------|------|
| `ORCHESTRATION` | `Gate 5.0` | `Hermes` | 编排、治理、审计 |
| `PLANNING` | `Gate 5.0` | `TMAX` | 长程规划、RL 决策 |
| `EXECUTION` | `Gate 5.0` | `UI-TARS` | 前线执行；需要具身能力时向 `Gate 4.0` 下沉 |
| `CODEGEN / PATCH / VERIFY` | `Gate 6.0` | `Coding Executor` | coding model 主链 |
| `EMBODIED_ACTION / ROBOT / DEVICE` | `Gate 4.0` | `Embodied Substrate` | 由 `UI-TARS` 或 agent 调用 |
| `DATA_SYNTHESIS` | `Offline Synthesis` | `CLI-Universe` | 默认不进入在线主链 |

---

## 8. 与单一 Gate 真源的关系

本矩阵不直接取代单一 Gate 白皮书，而是作为跨 Gate 架构总入口，被回挂到：

- `CGC_Gate_4.0_embodied_summary.example.json`
- `CGC_Gate_4.0_embodied_checkin.example.json`
- `CGC_Gate_4.0_embodied_gate_map.json`
- `CGC_Gate_5.0_audit_trace_replay_visualization_summary.example.json`
- `CGC_Gate_5.0_audit_trace_replay_visualization_checkin.example.json`
- `CGC_Gate_5.0_audit_trace_replay_visualization_gate_map.json`
- `CGC_Gate_6.0_fusionroute_complete_summary.example.json`
- `CGC_Gate_6.0_fusionroute_complete_checkin.example.json`
- `CGC_Gate_6.0_fusionroute_complete/gate_map.json`

---

## 9. Claim Boundary

### 9.1 可以正式写成

- `Gate 5.0` 是 Agent Runtime Plane
- `Gate 6.0` 是 Coding Model Plane
- `Gate 4.0` 作为 Embodied Capability Plane 被 `UI-TARS` 调用
- `FusionRoute` 采用 `Domain Router -> Role Router` 双层路由

### 9.2 不能直接写成

- 矩阵中所有 `CLI -> Verifier` 草案都已经正式验证通过
- `role locality / placement` 已作为 Gate 6.0 新 capability 完成正式 gate closure

上述两点仍需以 Gate 6.0 后续正式验证与 JSON evidence 为准。

---

## 10. 下一步

1. 将 `role locality / placement` 作为 Gate 6.0 独立子白皮书维护
2. 在 Gate 6.0 中以 `planned_capability_naming` 方式声明命名方案
3. 若后续实现正式 verifier，再升级为 Gate 6.0 capability 条目与 contract row
