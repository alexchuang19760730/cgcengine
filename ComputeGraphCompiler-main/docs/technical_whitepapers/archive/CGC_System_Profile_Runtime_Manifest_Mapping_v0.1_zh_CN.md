# CGC System Profile 与 Runtime Contract / System Execution Manifest 字段映射草案 v0.1

## 1. 文档目标

本文定义三件事：

- `system_profile` 的正式职责
- `system_profile` 与 `runtime_contract_bootstrap.py` 的字段映射
- `system_profile` 与 `system_execution_manifest.json` 的字段映射

本文不替代：

- `environment bootstrap`
- `profile setting + binding`
- `State ABI`

本文的目标是把它们上面缺失的“系统拓扑主变量”补成单一真源。

---

## 2. 三层正式配置分工

### 2.1 Environment Bootstrap

负责回答：

- 当前 runtime 具备什么能力
- 当前协议与分布式路径如何配置

典型字段：

- `backend`
- `environment`
- `runtime_profile`
- `protocol_family`
- `state_kind`
- `state_codec`
- `requested_dispatch_backend`
- `requested_distributed_runtime`
- `requested_storage_backend`
- `enable_pd`
- `pd_mode`
- `enable_nccl`
- `enable_cuda_graph`

### 2.2 Profile Setting + Binding

负责回答：

- 当前任务应该绑定到哪套 profile 与哪套 artifact family

典型字段：

- `profile_settings_path`
- `execution_profile_binding_key`
- `delivery_profile_binding_key`
- `bootstrap_contract_binding_key`
- `flow_parameter_contract_binding_key`

### 2.3 Model Setting / System Profile

负责回答：

- 当前系统由哪些正式组件组成
- 这些组件的家族、实例数与路由拓扑是什么
- 哪些是 required component

典型字段：

- `profile_id`
- `component_families`
- `component_matrix`
- `required_components`
- `routing_topology_profile`

一句话分工：

- `environment bootstrap` 解决能力层
- `profile setting + binding` 解决流程层
- `system_profile` 解决系统拓扑层

---

## 3. system_profile 应放在哪里

### 3.1 system_execution_manifest.json

`system_profile` 的完整真源应放在：

- `system_execution_manifest.json`

理由：

- 它是系统级文件，而不是单 component 文件
- 它天然负责表达 component 列表与 routing edges
- 它最适合成为所有 gate 的统一读取入口

### 3.2 runtime_contract_bootstrap.py 产物

单个 runtime artifact 不应完整复制整个 `system_profile`，只应带：

- `system_profile_ref`
- `system_profile_summary`

理由：

- 避免每个 runtime artifact 过重
- 避免多份副本漂移
- 保持“系统级真源唯一，组件级产物只保留引用”

---

## 4. 与 runtime_contract_bootstrap.py 的字段映射

### 4.1 当前已有字段

`runtime_contract_bootstrap.py` 当前已经稳定输出：

- `component_id`
- `component_role`
- `backend`
- `environment`
- `model_name`
- `task_domain`
- `runtime_profile`
- `protocol_family`
- `state_kind`
- `state_codec`
- `requested_dispatch_backend`
- `requested_distributed_runtime`
- `requested_storage_backend`
- `enable_pd`
- `pd_mode`
- `system_id`
- `system_manifest_components`
- `system_manifest_routing_edges`

这些字段足以承接 `system_profile` 的引用摘要，但还不足以成为系统拓扑真源。

### 4.2 建议新增输出字段

建议在 `materialize_runtime_contract_artifacts()` 的返回值以及写出的 runtime contract artifact 中新增：

- `system_profile_ref`
- `system_profile_summary`

推荐结构如下：

```json
{
  "system_profile_ref": {
    "profile_id": "deepseek_v4_flash_x4_minicpm5_fusionroute",
    "profile_version": "v1",
    "source": "system_execution_manifest",
    "source_path": "<export_dir>/system_execution_manifest.json"
  },
  "system_profile_summary": {
    "llm_component_family": "deepseek_v4_flash",
    "llm_instance_count": 4,
    "router_component_family": "minicpm5",
    "gateway_component_family": "fusionroute",
    "route_policy_family": "fusionroute",
    "deployment_mode": "cloud_prefill_edge_decode"
  }
}
```

### 4.3 字段映射表

| system_profile 字段 | runtime_contract_bootstrap.py 当前来源 | 建议在 runtime artifact 中如何表达 |
| --- | --- | --- |
| `profile_id` | 当前无单一字段 | 新增 `system_profile_ref.profile_id` |
| `profile_version` | 当前无单一字段 | 新增 `system_profile_ref.profile_version` |
| `deployment_mode` | 可由 `environment + pd_mode + runtime_profile` 近似推导 | 新增 `system_profile_summary.deployment_mode` |
| `component_families.llm_component_family` | 当前无正式字段；仅散落在 `model_name` / manifest component 中 | 新增 `system_profile_summary.llm_component_family` |
| `component_families.llm_instance_count` | 可由 manifest components 中 `llm_runtime` 数量统计 | 新增 `system_profile_summary.llm_instance_count` |
| `component_families.router_component_family` | 当前可从 router component 的 `model_name` 间接得到 | 新增 `system_profile_summary.router_component_family` |
| `component_families.gateway_component_family` | 当前可从 gateway component 的 `model_name` 间接得到 | 新增 `system_profile_summary.gateway_component_family` |
| `component_families.route_policy_family` | 当前无正式字段；散落在 routing logic / gateway 名称 | 新增 `system_profile_summary.route_policy_family` |
| `environment_bootstrap_ref.environment` | `environment` | 可直接镜像到 `environment_bootstrap_ref` |
| `environment_bootstrap_ref.runtime_profile` | `runtime_profile` | 可直接镜像到 `environment_bootstrap_ref` |
| `environment_bootstrap_ref.protocol_family` | `protocol_family` | 可直接镜像到 `environment_bootstrap_ref` |
| `environment_bootstrap_ref.state_kind` | `state_kind` | 可直接镜像到 `environment_bootstrap_ref` |
| `environment_bootstrap_ref.state_codec` | `state_codec` | 可直接镜像到 `environment_bootstrap_ref` |
| `environment_bootstrap_ref.requested_dispatch_backend` | `requested_dispatch_backend` | 可直接镜像到 `environment_bootstrap_ref` |
| `environment_bootstrap_ref.requested_distributed_runtime` | `requested_distributed_runtime` | 可直接镜像到 `environment_bootstrap_ref` |
| `environment_bootstrap_ref.requested_storage_backend` | `requested_storage_backend` | 可直接镜像到 `environment_bootstrap_ref` |

### 4.4 runtime_contract_bootstrap.py 的最小改动建议

建议仅增加三步：

1. 从 `system_execution_manifest.json` 读取顶层 `system_profile`
2. 组装 `system_profile_ref`
3. 组装 `system_profile_summary`

不建议在 bootstrap 阶段：

- 重新推导整份 `system_profile`
- 复制全部 `component_matrix`
- 复制全部 `routing_edges`

---

## 5. 与 system_execution_manifest.json 的字段映射

### 5.1 建议的放置位置

建议把 `system_profile` 放在顶层：

```json
{
  "system_id": "fusionroute_ds4x4_minicpm5",
  "system_role": "multi_component_runtime",
  "system_profile": {
    "...": "..."
  },
  "components": [],
  "routing_edges": []
}
```

### 5.2 与现有 manifest 字段的关系

| system_profile 字段 | system_execution_manifest.json 对应来源 | 说明 |
| --- | --- | --- |
| `profile_id` | 顶层新增 | 不建议继续从 `system_id` 间接推导 |
| `profile_version` | 顶层新增 | 系统拓扑版本应独立 versioned |
| `deployment_mode` | 顶层新增，允许参考 `system_role` / pd mode | 不能只靠 `environment` 代替 |
| `component_families.llm_component_family` | 由 `components[*].component_role=llm_runtime` 的 `model_name/model_family` 归并得到 | 建议固化后不再每次临时归并 |
| `component_families.llm_instance_count` | 由 `components` 中 `llm_runtime` 数量得到 | 建议固化到 system_profile |
| `component_families.router_component_family` | 由 `router_runtime` component 归并得到 | 建议固化 |
| `component_families.gateway_component_family` | 由 `gateway_orchestrator` component 归并得到 | 建议固化 |
| `route_policy_family` | 当前 manifest 无正式顶层字段 | 建议新增到 system_profile |
| `component_matrix` | 由 `components[*].component_role` 分组得到 | 建议作为 system_profile 的正式摘要 |
| `required_components` | 由 `components[*].required=true` 提取 | 建议固化，供 gate 直接消费 |
| `routing_topology_profile` | 当前无正式字段 | 建议新增，例如 `fusionroute_router_sidecar` |
| `routing_edges` | 已有顶层 `routing_edges` | system_profile 可引用或保留摘要，不必重复完整拓扑 |

### 5.3 推荐最小结构

```json
{
  "system_id": "fusionroute_ds4x4_minicpm5",
  "system_role": "multi_component_runtime",
  "system_profile": {
    "profile_id": "deepseek_v4_flash_x4_minicpm5_fusionroute",
    "profile_version": "v1",
    "deployment_mode": "cloud_prefill_edge_decode",
    "component_families": {
      "llm_component_family": "deepseek_v4_flash",
      "llm_instance_count": 4,
      "router_component_family": "minicpm5",
      "gateway_component_family": "fusionroute",
      "route_policy_family": "fusionroute"
    },
    "component_matrix": {
      "llm_runtime": [
        "deepseek_inst1",
        "deepseek_inst2",
        "deepseek_inst3",
        "deepseek_inst4"
      ],
      "router_runtime": [
        "minicpm5_router"
      ],
      "gateway_orchestrator": [
        "fusionroute_gateway"
      ]
    },
    "required_components": [
      "fusionroute_gateway",
      "minicpm5_router",
      "deepseek_inst1",
      "deepseek_inst2"
    ],
    "routing_topology_profile": "fusionroute_ds4x4_router_sidecar"
  },
  "components": [],
  "routing_edges": []
}
```

---

## 6. 所有 Gate 的统一消费方式

建议后续 gate 统一按以下顺序消费：

1. `system_execution_manifest.system_profile`
2. `runtime_contract.system_profile_ref / system_profile_summary`
3. `profile_settings + binding`
4. `environment bootstrap / runtime_protocol_contract`
5. `State ABI`

一句话理解：

- `system_profile`
  - 决定系统是谁、怎么组
- `profile binding`
  - 决定这次怎么跑
- `environment bootstrap`
  - 决定当前能跑什么
- `State ABI`
  - 决定语义能不能兼容

---

## 7. 正式结论

当前工程已经具备：

- `profile setting + binding`
- `environment bootstrap`
- `component_id / component_role / system manifest`

但尚缺一层被所有 gate 统一消费的系统拓扑真源：

- `system_profile`

因此下一步最小正确收敛不是继续给各 gate 增加临时模型名判断，而是：

- 把 `MiniCPM / DeepSeek-V4-Flash / FusionRoute` 及其实例数、角色矩阵、路由拓扑正式收进 `system_profile`
- 并让 `runtime_contract_bootstrap.py` 只携带引用摘要，而由 `system_execution_manifest.json` 作为完整真源

这才是三层正式配置在工程上的稳定落点。
