# CGC FusionRoute v2 静态契约白皮书 v1.0

**版本**: v1.0  
**状态**: `formal_closure`  
**定位**: 以静态契约形式定义 `TaskType -> GateDomain -> PrimaryRole -> SecondaryRole`，并把 `role locality / placement / Perception Matrix` 作为 Gate 6.0 的正式控制面链路。  
**治理边界**: 本文用于定义 Gate 6.0 的正式静态契约闭环；其 machine-checkable JSON evidence 已并入当前正式 capability 主链。

---

## 1. 目标

FusionRoute v2 静态契约回答以下问题：

1. `TaskType` 应该投影到哪个 `GateDomain`
2. 每个 `GateDomain` 的 `PrimaryRole / SecondaryRole` 是什么
3. 角色的 `role locality` 与 `placement_decision_report` 如何形成 machine-readable 证据
4. `Perception Matrix + LLM` 如何作为上游建议层，并被 contract projection 收敛为合法 profile

---

## 2. 总体分层

```text
Perception Matrix + LLM
  -> Contract Projection Layer
  -> FusionRoute v2 Static Contract
  -> Role Locality / Placement
  -> Runtime / TP4EP4 / Unified Pipeline Kernel
```

分层职责如下：

- `Perception Matrix + LLM`
  - 负责环境、任务、模型、硬件的高阶建议
- `Contract Projection Layer`
  - 负责把建议映射到合法的 `system profile / profile binding / bootstrap / state ABI / topology profile`
- `FusionRoute v2 Static Contract`
  - 负责 `TaskType -> GateDomain -> PrimaryRole -> SecondaryRole`
- `Role Locality / Placement`
  - 负责 `cloud / edge / hybrid / auto`
- `Runtime Layer`
  - cloud 正式底座仍以 `TP4/EP4` 为 anchor

---

## 3. TaskType -> GateDomain -> PrimaryRole -> SecondaryRole

下表是本白皮书的核心静态矩阵：

| TaskType | GateDomain | PrimaryRole | SecondaryRole | 说明 |
|----------|------------|-------------|---------------|------|
| `ORCHESTRATION` | `agent_runtime` | `Hermes` | `TMAX` | 编排、审计、治理 |
| `EXECUTION` | `agent_runtime` | `UI-TARS` | `Hermes,TMAX` | 前线执行与环境交互 |
| `CODEGEN` | `coding_runtime` | `Coding Executor` | `Hermes,TMAX` | coding model 主链 |
| `PATCH_VERIFY` | `coding_runtime` | `Coding Executor` | `Hermes,TMAX` | 补丁验证与 repo 操作 |
| `EMBODIED_ACTION` | `embodied_capability` | `Embodied Substrate` | `UI-TARS` | 具身底座调用 |
| `DATA_SYNTHESIS` | `offline_synthesis` | `CLI-Universe` | `TMAX,Hermes` | 离线数据合成 |

该矩阵就是 `TaskType -> GateDomain -> PrimaryRole -> SecondaryRole` 的正式 truth source。

---

## 4. Role Locality 与 Placement

FusionRoute v2 在静态矩阵之上继续定义：

- `role_locality_contract`
- `placement_decision_report`

其中：

- `role_locality_contract`
  - 定义角色偏好的 `preferred_locality / allowed_localities / runtime_endpoint / fallback_endpoint`
- `placement_decision_report`
  - 定义一次请求最终选择了什么 `locality / runtime_endpoint / decision_reason / policy_source`

这两份 artifact 共同约束 `role locality / placement` 决策。

---

## 5. Perception Matrix 作为上游建议层

Perception Matrix 并不是最终 authority，而是：

- 读取 `environment + task + model + hardware + governance`
- 由 LLM 给出 `policy_suggestion_report`
- 再由 contract projection 生成 `contract_projection_report`
- 最后把合法结果交给 FusionRoute v2 Static Contract

因此正式边界是：

- `Perception Matrix` 可以更智能
- `FusionRoute v2` 必须更结构化
- `Contract Projection` 必须更保守

---

## 6. 机器可读 Artifact

本白皮书与以下 artifact 成对挂接：

| Artifact | 作用 |
|----------|------|
| `role_locality_contract.schema.json` | role locality 契约 |
| `placement_decision_report.schema.json` | placement 决策报告 |
| `policy_suggestion_report.schema.json` | LLM / policy 建议报告 |
| `contract_projection_report.schema.json` | contract projection 报告 |
| `gate6_fusionroute_v2_draft_contract.json` | historical draft contract manifest |
| `gate6_fusionroute_v2_formal_contract.json` | formal contract manifest |
| `fusionroute_v2_formal_contract_report.json` | 正式聚合验证报告 |

---

## 7. CLI 投影

本静态契约对应以下 CLI：

```bash
python3 cgc_engine/cli.py fusionroute plan --task-type CODEGEN --print-json
python3 cgc_engine/cli.py fusionroute contract show --kind role-locality --role UI-TARS --print-json
python3 cgc_engine/cli.py fusionroute placement verify --task-type EXECUTION --role UI-TARS --print-json
python3 cgc_engine/cli.py fusionroute perception plan --task-type CODEGEN --environment-type repo --print-json
python3 cgc_engine/cli.py fusionroute perception project --task-type CODEGEN --environment-type repo --print-json
python3 cgc_engine/cli.py fusionroute verify --capability all --print-json
```

---

## 8. Verifier 命名

当前正式链对应 verifier：

- `validate_fusionroute_v2_tasktype_gate_domain_contract`
- `validate_fusionroute_role_locality_contract`
- `validate_fusionroute_placement_decision_report`
- `validate_fusionroute_policy_suggestion_report`
- `validate_fusionroute_contract_projection_report`
- `validate_fusionroute_v2_contract_chain`

---

## 9. Claim Boundary

### 9.1 可以正式写成

- `TaskType -> GateDomain -> PrimaryRole -> SecondaryRole` 已有静态矩阵白皮书
- `role locality / placement / Perception Matrix` 已有 schema、example JSON、formal manifest 与 formal report
- `fusionroute verify --capability all` 可生成独立正式聚合报告

### 9.2 不能直接写成

- FusionRoute v2 正式链已经完成远端 `host1/host2 29/29` 主链复核
- `Perception Matrix + LLM` 已可替代所有 bootstrap / ABI / topology 硬约束
- `swe_verified_500` 已可提升为 release-facing external score claim

---

## 10. 真源挂接

本文挂接到以下真源：

- `CGC_Gate_6.0_fusionroute_complete_summary.example.json`
- `CGC_Gate_6.0_fusionroute_complete_checkin.example.json`
- `CGC_Gate_6.0_fusionroute_complete/gate_map.json`
- `CGC_Gate_6.0_fusionroute_complete/README.md`
- `CGC_Gate_6.0_FusionRoute_Role_Locality_Technical_Whitepaper_v1.0_zh_CN.md`
- `CGC_Perception_Matrix_LLM_Technical_Whitepaper_v1.0_zh_CN.md`

挂接方式使用：

- `architecture_projection_refs`
- `gate_map.capabilities`
- `formal_contract_refs`

并已直接提升为正式 capability 状态。
