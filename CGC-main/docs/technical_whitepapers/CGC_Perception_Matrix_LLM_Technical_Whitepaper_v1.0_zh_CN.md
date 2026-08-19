# CGC Perception Matrix + LLM 技术白皮书 v1.0

**版本**: v1.0  
**状态**: `formal_closure_component`  
**定位**: 定义 `Perception Matrix` 作为 `FusionRoute v2` 的上游动态决策层，允许嵌入大模型做高阶策略推理，但所有输出必须投影到受契约约束的合法 profile。  
**治理边界**: 本文中的 `policy_suggestion_report` 与 `contract_projection_report` 已正式并入 `Gate 6.0` capability closure；上层推理 authority 仍不得绕开 profile / ABI / bootstrap / topology contracts。

---

## 1. 为什么需要 Perception Matrix

仅靠静态 `TaskType -> GateDomain -> PrimaryRole -> SecondaryRole` 矩阵仍然不足以处理以下动态因素：

- 环境类型是否为 `web / desktop / embodied / code repo / terminal`
- 当前任务是 `agent / coding / embodied / synthesis`
- 可用模型池是否支持长上下文、多模态或大规模 reasoning
- 硬件条件是否允许 `cloud / edge / hybrid`
- 隐私、成本、延迟预算是否要求不同的 runtime 选择

因此 `Perception Matrix` 应作为 `FusionRoute` 之前的一层动态决策器：

- 先读取环境与约束
- 由 LLM 给出高阶策略建议
- 再由结构化契约层把建议投影为合法 profile

---

## 2. 设计原则

### 2.1 LLM 可以参与，但不能裸控

`Perception Matrix` 可以嵌入大模型作为：

- 环境理解器
- 策略建议器
- 多因素权衡器
- fallback / handoff 推理器

但 LLM 不能直接成为最终 authority。最终 authority 必须是：

- `state ABI`
- `bootstrap`
- `system profile`
- `profile binding`
- `topology profile`
- `unified pipeline kernel`

### 2.2 动态决策必须受静态契约约束

Perception Matrix 的输出不是任意 runtime 参数，而应是：

- `gate_domain`
- `primary_role`
- `secondary_roles`
- `selected_model_profile`
- `selected_locality`
- `topology_profile`
- `bootstrap_profile`
- `state_abi_mode`

这些输出必须能够被既有契约层验证。

---

## 3. 分层关系

```text
Environment / Task / Model / Hardware / Governance Inputs
  -> Perception Matrix + LLM
  -> Contract Projection Layer
  -> FusionRoute v2
  -> Runtime / TP4EP4 / Edge Runtime / Unified Pipeline Kernel
```

### 3.1 Perception Matrix + LLM

负责：

- 读取高维输入
- 生成 `policy suggestion`
- 对不确定场景做高阶判断

### 3.2 Contract Projection Layer

负责把策略建议投影到合法集合：

- `system_profile_id`
- `profile_binding_id`
- `topology_profile`
- `bootstrap_profile`
- `state_abi_mode`
- `runtime_endpoint`

### 3.3 FusionRoute v2

负责根据投影结果做：

- `GateDomain` 路由
- `Role` 路由
- `Locality / Placement`

### 3.4 Runtime Layer

负责执行：

- cloud 端正式底座仍以 `TP4/EP4` 为固定 topology substrate
- edge 端使用已声明的 edge runtime profile
- `unified pipeline kernel` 承接最终数据面与执行面

---

## 4. 输入维度

建议 `Perception Matrix` 至少读取以下输入：

| 维度 | 说明 |
|------|------|
| `environment_type` | web / desktop / embodied / terminal / repo |
| `task_type` | ORCHESTRATION / EXECUTION / CODEGEN / PATCH_VERIFY / EMBODIED_ACTION |
| `model_inventory` | 可用模型列表与 profile |
| `hardware_inventory` | GPU、内存、device endpoint、带宽、延迟 |
| `privacy_level` | low / standard / high |
| `cost_budget` | 本次请求可接受成本 |
| `latency_budget_ms` | 延迟预算 |
| `system_profile_id` | 当前系统 profile |
| `profile_binding_id` | 当前绑定 profile |
| `state_abi_mode` | 允许的 state ABI 模式 |
| `topology_profile` | 允许的 topology profile |
| `pipeline_kernel_mode` | unified pipeline kernel 模式 |

---

## 5. 输出结构

Perception Matrix 需要输出两层结果：

### 5.1 policy suggestion

这是由 LLM 或策略模块给出的高阶建议：

- 推荐 `gate_domain`
- 推荐 `primary_role`
- 推荐 `selected_locality`
- 推荐 `topology_profile`
- 推荐 `bootstrap_profile`
- 推荐 `selected_model_profile`

### 5.2 contract projection

这是经过硬约束投影后的正式结构：

- `system_profile_id`
- `profile_binding_id`
- `selected_runtime_endpoint`
- `topology_profile`
- `state_abi_mode`
- `bootstrap_profile`
- `projection_status`

---

## 6. 与现有契约层的关系

### 6.1 state ABI

Perception Matrix 不能选择 ABI 不支持的 handoff 形式。

### 6.2 bootstrap

Perception Matrix 不能选择 bootstrap 未注册的 runtime / topology。

### 6.3 system profile & profile binding

Perception Matrix 的动态决策必须先落到 profile，再交给 `FusionRoute`。

### 6.4 unified pipeline kernel

Perception Matrix 是控制面，而 `unified pipeline kernel` 是执行承载底座。

### 6.5 TP4/EP4

云端正式 runtime substrate 仍以 `TP4/EP4` 为 anchor；Perception Matrix 只能决定是否使用该 substrate，而不是任意改写它。

---

## 7. 机器可读 artifact

本文建议增加以下 artifact：

| Artifact | 路径 | 作用 |
|----------|------|------|
| `policy_suggestion_report` schema | `CGC_Gate_6.0_fusionroute_complete/policy_suggestion_report.schema.json` | LLM/策略建议层的结构化输出 |
| `contract_projection_report` schema | `CGC_Gate_6.0_fusionroute_complete/contract_projection_report.schema.json` | 契约投影结果的结构化输出 |

---

## 8. Gate 6.0 capability 命名建议

本文中的两条 machine-checkable JSON evidence 已正式并入 `Gate 6.0`；以下保留的是更高层能力语义：

| Capability ID | 建议状态 | 说明 |
|---------------|----------|------|
| `perception_matrix_policy` | `planned_projection` | 感知矩阵总能力 |
| `llm_policy_suggestion` | `planned_projection` | LLM 参与策略建议 |
| `environment_task_model_hardware_projection` | `planned_projection` | 环境/任务/模型/硬件联合投影 |
| `policy_suggestion_report` | `done` | 策略建议 JSON 报告 |
| `contract_projection_report` | `done` | 契约投影 JSON 报告 |
| `perception_contract_projection` | `planned_projection` | 从建议到合法 profile 的投影能力 |

---

## 9. Claim Boundary

### 9.1 现在可以正式写成

- Perception Matrix 已被定义为 `FusionRoute v2` 的上游动态决策层
- LLM 只作为建议器，不直接成为 runtime authority
- `policy_suggestion_report` 与 `contract_projection_report` 已有独立 schema、example 与 formal report

### 9.2 现在不能直接写成

- Perception Matrix 可以绕开 profile / ABI / bootstrap / topology 硬约束
- LLM 建议链已经自动获得外部 benchmark claimable 资格
- 动态决策已可正式宣称替代 `TP4/EP4` 或 bootstrap/static contract

---

## 10. 挂接方式

本文通过以下方式挂接到 `Gate 6.0 / FusionRoute v2`：

- `CGC_Gate_6.0_fusionroute_complete_summary.example.json`
- `CGC_Gate_6.0_fusionroute_complete_checkin.example.json`
- `CGC_Gate_6.0_fusionroute_complete/gate_map.json`
- `CGC_FusionRoute_v2_Static_Contract_Technical_Whitepaper_v1.0_zh_CN.md`
- `CGC_Gate_6.0_FusionRoute_Role_Locality_Technical_Whitepaper_v1.0_zh_CN.md`

挂接方式采用：

- `architecture_projection_refs`
- `planned_capability_naming`
- `formal_contract_refs`

其中 `policy_suggestion_report` 与 `contract_projection_report` 已直接改写正式 capability 计数。
