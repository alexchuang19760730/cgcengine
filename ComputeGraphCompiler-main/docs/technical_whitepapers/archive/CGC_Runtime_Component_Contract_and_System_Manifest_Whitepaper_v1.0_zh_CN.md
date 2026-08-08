# CGC Runtime Component Contract 与 System Execution Manifest 白皮书 v1.0

## 1. 文档目标

本文定义一套与 `pipeline.py` 当前实现对齐的最小运行时契约，用于统一描述以下两类对象：

- 单个 runtime component 的执行契约
- 多个 runtime component 组成的系统级执行清单

本文不再只围绕 `psi0`。它面向更通用的组合系统，特别是：

- `DeepSeek-V4-Flash x4`
- `MiniCPM5 x1`
- `FusionRoute x1`

一句话定义：

```text
当前 pipeline 已经能稳定输出 runtime-scoped contract；
下一步最小扩展不是重写 helper，而是在其上增加 system-scoped manifest。
```

---

## 2. 当前实现已落地的最小契约

`pipeline.py` 当前已落地三个关键 helper：

- `_materialize_execution_context()`
- `_materialize_state_abi()`
- `_resolve_strategy_decision()`

它们共同输出四份 runtime artifact：

- `execution_context.json`
- `state_abi.json`
- `strategy_decision.json`
- `compatibility_report.json`

并且当前版本已可以额外写出：

- `system_execution_manifest.json`

这些 artifact 已经足够表达：

- 当前 runtime 在什么上下文中运行
- 当前 model/state 是否兼容某条执行路径
- 当前策略选择了哪条 runtime branch
- 当前 gate 最终给出 `PASS / BLOCKED / FAIL` 的哪一种结果

需要强调的是：

- 当前实现稳定输出的是 `runtime-scoped contract`
- 当前也已能写出最小 `system-scoped manifest`
- 但多 component 信息目前仍主要来自 config 注入，而不是自动拓扑发现

也就是说，当前代码更适合描述：

- `deepseek_inst1`
- `deepseek_inst2`
- `deepseek_inst3`
- `deepseek_inst4`
- `minicpm5_router`
- `fusionroute_gateway`

这些对象各自的契约，而不是它们整体的系统拓扑。

---

## 3. 与 pipeline.py 对齐后的新增字段

为了让单 runtime artifact 可以被系统级 manifest 安全引用，最小新增字段是：

- `component_id`
- `component_role`

这两个字段已经适合直接进入：

- `execution_context.json`
- `state_abi.json`
- `strategy_decision.json`
- `compatibility_report.json`

设计原则：

- 不改变现有 helper 的主流程顺序
- 不改变现有 artifact 的主语义
- 只增加可索引、可组合、可聚合的 component 维度标识

### 3.1 component_id

`component_id` 是系统中某个 runtime component 的稳定身份标识。

它的用途是：

- 被 `system_execution_manifest.json` 引用
- 作为跨 artifact 关联键
- 作为 runtime health、evidence、routing edge 的外键

约束：

- 必须在一个系统 manifest 内唯一
- 应优先由调用方显式指定
- 未显式指定时，可由 `model_name + component_role` 推导

推荐示例：

- `deepseek_inst1`
- `deepseek_inst2`
- `deepseek_inst3`
- `deepseek_inst4`
- `minicpm5_router`
- `fusionroute_gateway`

### 3.2 component_role

`component_role` 是该 component 在系统中的职责类型，而不是模型家族名。

它回答的问题是：

```text
这个 component 在系统里负责干什么？
```

推荐角色：

- `llm_runtime`
- `router_runtime`
- `gateway_orchestrator`
- `embodied_runtime`
- `agent_runtime`
- `harness_runtime`
- `model_runtime`

约束：

- `component_role` 不等于 `model_family`
- `component_role` 描述职责，不描述权重谱系
- 同一 `model_family` 可以对应不同 `component_role`

例如：

- `DeepSeek-V4-Flash` 通常是 `llm_runtime`
- `MiniCPM5` 在 router 场景可归为 `router_runtime`
- `FusionRoute` 通常是 `gateway_orchestrator`

---

## 4. 单 Runtime Contract 最小结构

以下结构是对 `pipeline.py` 当前 helper 形状的最小正式化，而不是未来系统图的提前假设。

### 4.1 execution_context.json

```json
{
  "schema_version": "execution_context_v1",
  "component_id": "deepseek_inst1",
  "component_role": "llm_runtime",
  "task_entity": "model",
  "task_domain": "models",
  "task_type": "inference",
  "hardware_scope": "single_host",
  "hardware_platform": "cuda",
  "hardware_topology": "single_node_4gpu",
  "model_scope": "ds4_flash_pro",
  "model_assembly": "static_model_graph_host",
  "environment": "cloud_cluster",
  "runtime_mode": "local_infer",
  "backend": "cuda",
  "model_name": "deepseek_v4_flash_pro",
  "model_family": "ds4_flash_pro",
  "runtime_profile": "cloud_cluster",
  "distributed_backend": "nccl",
  "distributed_topology": {
    "world_size": 4,
    "parallel_tp_size": 4,
    "parallel_pp_size": 1,
    "parallel_ep_size": 1
  },
  "weight_source_policy": {
    "base_model_source_path": "/data/models/DeepSeek-V4-Flash",
    "load_weights": true,
    "dtype": "torch.bfloat16"
  },
  "abi_descriptor": {
    "schema_version": "generic_state_abi_v1",
    "state_abi_path": "<export_dir>/state_abi.json",
    "compatibility_report_path": "<export_dir>/compatibility_report.json"
  },
  "artifact_path": "<export_dir>/execution_context.json"
}
```

### 4.2 state_abi.json

```json
{
  "schema_version": "generic_state_abi_v1",
  "status": "PASS",
  "component_id": "deepseek_inst1",
  "component_role": "llm_runtime",
  "model_family": "ds4_flash_pro",
  "task_profile": "inference",
  "model_assembly": "static_model_graph_host",
  "checkpoint_fingerprint": {
    "base_model_source_path": "/data/models/DeepSeek-V4-Flash",
    "load_weights": true,
    "weights_loaded": true
  },
  "trainability_contract": {
    "frozen_module_prefixes": [],
    "trainable_module_prefixes": [
      "model",
      "lm_head"
    ],
    "trainable_param_count": 68543210912,
    "frozen_param_count": 0
  },
  "device_layout": {
    "param_device_policy": "all_params_on_runtime_device",
    "buffer_device_policy": "all_buffers_on_runtime_device",
    "observed_param_device_histogram": {
      "cuda:0": 412,
      "cuda:1": 408,
      "cuda:2": 406,
      "cuda:3": 410
    },
    "observed_buffer_device_histogram": {
      "cuda:0": 1
    }
  },
  "dtype_layout": {
    "global_param_dtype_histogram": {
      "torch.float32": 12,
      "torch.bfloat16": 1632
    },
    "global_buffer_dtype_histogram": {
      "torch.float32": 1
    },
    "module_dtype_layout": [
      {
        "module_prefix": "model.embed_tokens",
        "param_dtypes": {
          "torch.bfloat16": 262144000
        }
      },
      {
        "module_prefix": "model.layers",
        "param_dtypes": {
          "torch.bfloat16": 68275015680
        }
      }
    ]
  },
  "distributed_compatibility": {
    "single_gpu": {
      "compatible": true,
      "reason": ""
    },
    "ddp_constructor": {
      "compatible": "unknown",
      "reason": "runtime_collective_evidence_required"
    }
  },
  "artifact_path": "<export_dir>/state_abi.json"
}
```

### 4.3 strategy_decision.json

```json
{
  "schema_version": "strategy_decision_v1",
  "component_id": "deepseek_inst1",
  "component_role": "llm_runtime",
  "model_family": "ds4_flash_pro",
  "task_type": "inference",
  "decision_id": "ab12cd34ef56gh78",
  "inputs": {
    "execution_context_path": "<export_dir>/execution_context.json",
    "state_abi_path": "<export_dir>/state_abi.json",
    "compatibility_report_path": "<export_dir>/compatibility_report.json"
  },
  "decision": {
    "selected_runtime_branch": "ddp_multi_gpu",
    "selected_pipeline_profile": "cloud_cluster",
    "selected_collective_mode": "ddp",
    "selected_weight_policy": "load_checkpoint_directly"
  },
  "gates": {
    "abi_compatible_for_single_gpu": true,
    "abi_compatible_for_ddp_constructor": true,
    "distributed_collective_evidence_required": true
  },
  "decision_reason": {
    "root_code": "",
    "summary": ""
  },
  "fallback_plan": {
    "fallback_runtime_branch": "single_gpu",
    "fallback_allowed": true
  },
  "distributed_init_status": {
    "status": "PASS",
    "backend": "nccl",
    "world_size": 4
  },
  "artifact_path": "<export_dir>/strategy_decision.json"
}
```

### 4.4 compatibility_report.json

```json
{
  "schema_version": "compatibility_report_v1",
  "component_id": "deepseek_inst1",
  "component_role": "llm_runtime",
  "model_family": "ds4_flash_pro",
  "state_abi_path": "<export_dir>/state_abi.json",
  "strategy_decision_path": "<export_dir>/strategy_decision.json",
  "check_items": [
    {
      "name": "execution_context_materialized",
      "status": "PASS",
      "reason": ""
    },
    {
      "name": "state_abi_materialized",
      "status": "PASS",
      "reason": ""
    },
    {
      "name": "ddp_constructor_compatibility",
      "status": "PASS",
      "reason": ""
    }
  ],
  "overall_status": "PASS",
  "overall_reason": "",
  "artifact_path": "<export_dir>/compatibility_report.json"
}
```

---

## 5. 当前未落地但必须正式化的系统层对象

当系统变成：

- `DeepSeek-V4-Flash x4`
- `MiniCPM5 x1`
- `FusionRoute x1`

仅靠单个 runtime contract 已不足以描述：

- 哪些 component 共同组成一个系统
- 哪些 routing edge 在谁和谁之间建立
- 系统 readiness 如何由多个 component 汇总
- 哪些 component 是必须 READY，哪些是可选 READY

因此需要新增一层最薄的系统包装对象：

- `system_execution_manifest.json`

这不是替换单 runtime contract，而是引用它们。

当前实现状态是：

- 已能为当前 component 自动写出一份 system manifest
- 已支持在同一 discovery root 下自动发现 sibling `contract_manifest.json` 并聚合多个 runtime
- 也可通过 config 注入额外 components、routing edges、required/optional policy
- 但还没有做到跨 host、跨独立 launcher 的自动注册与自动组装

---

## 6. system_execution_manifest.json 最小 Schema 草稿

### 6.1 设计原则

- 不替换现有四份 runtime artifact
- 只负责表达系统拓扑与引用关系
- 不承担 model state ABI 细节
- 不让 `FusionRoute` 伪装成模型权重 contract

### 6.2 最小结构

```json
{
  "schema_version": "system_execution_manifest_v1",
  "system_id": "fusionroute_ds4x4_minicpm5",
  "system_role": "multi_component_runtime",
  "environment": "cloud_cluster",
  "runtime_mode": "local_infer",
  "components": [
    {
      "component_id": "fusionroute_gateway",
      "component_role": "gateway_orchestrator",
      "required": true,
      "health_endpoint": "http://host:8080/health",
      "artifact_paths": {
        "execution_context": "<export_dir>/fusionroute_gateway/execution_context.json",
        "strategy_decision": "<export_dir>/fusionroute_gateway/strategy_decision.json",
        "compatibility_report": "<export_dir>/fusionroute_gateway/compatibility_report.json"
      }
    },
    {
      "component_id": "deepseek_inst1",
      "component_role": "llm_runtime",
      "required": true,
      "health_endpoint": "http://host:50053/health",
      "artifact_paths": {
        "execution_context": "<export_dir>/deepseek_inst1/execution_context.json",
        "state_abi": "<export_dir>/deepseek_inst1/state_abi.json",
        "strategy_decision": "<export_dir>/deepseek_inst1/strategy_decision.json",
        "compatibility_report": "<export_dir>/deepseek_inst1/compatibility_report.json"
      }
    },
    {
      "component_id": "deepseek_inst2",
      "component_role": "llm_runtime",
      "required": true,
      "health_endpoint": "http://host:50063/health",
      "artifact_paths": {
        "execution_context": "<export_dir>/deepseek_inst2/execution_context.json",
        "state_abi": "<export_dir>/deepseek_inst2/state_abi.json",
        "strategy_decision": "<export_dir>/deepseek_inst2/strategy_decision.json",
        "compatibility_report": "<export_dir>/deepseek_inst2/compatibility_report.json"
      }
    },
    {
      "component_id": "deepseek_inst3",
      "component_role": "llm_runtime",
      "required": false,
      "health_endpoint": "http://host:50073/health",
      "artifact_paths": {
        "execution_context": "<export_dir>/deepseek_inst3/execution_context.json",
        "state_abi": "<export_dir>/deepseek_inst3/state_abi.json",
        "strategy_decision": "<export_dir>/deepseek_inst3/strategy_decision.json",
        "compatibility_report": "<export_dir>/deepseek_inst3/compatibility_report.json"
      }
    },
    {
      "component_id": "deepseek_inst4",
      "component_role": "llm_runtime",
      "required": false,
      "health_endpoint": "http://host:50083/health",
      "artifact_paths": {
        "execution_context": "<export_dir>/deepseek_inst4/execution_context.json",
        "state_abi": "<export_dir>/deepseek_inst4/state_abi.json",
        "strategy_decision": "<export_dir>/deepseek_inst4/strategy_decision.json",
        "compatibility_report": "<export_dir>/deepseek_inst4/compatibility_report.json"
      }
    },
    {
      "component_id": "minicpm5_router",
      "component_role": "router_runtime",
      "required": true,
      "health_endpoint": "http://host:19090/health",
      "artifact_paths": {
        "execution_context": "<export_dir>/minicpm5_router/execution_context.json",
        "strategy_decision": "<export_dir>/minicpm5_router/strategy_decision.json",
        "compatibility_report": "<export_dir>/minicpm5_router/compatibility_report.json"
      }
    }
  ],
  "routing_edges": [
    {
      "from_component_id": "fusionroute_gateway",
      "to_component_id": "deepseek_inst1",
      "edge_role": "primary_llm_route"
    },
    {
      "from_component_id": "fusionroute_gateway",
      "to_component_id": "deepseek_inst2",
      "edge_role": "secondary_llm_route"
    },
    {
      "from_component_id": "fusionroute_gateway",
      "to_component_id": "minicpm5_router",
      "edge_role": "router_sidecar_route"
    }
  ],
  "readiness_policy": {
    "required_components": [
      "fusionroute_gateway",
      "deepseek_inst1",
      "deepseek_inst2",
      "minicpm5_router"
    ],
    "optional_components": [
      "deepseek_inst3",
      "deepseek_inst4"
    ],
    "overall_ready_rule": "all_required_ready"
  }
}
```

---

## 7. 为什么 system manifest 必须独立存在

如果没有 `system_execution_manifest.json`，系统只能得到很多孤立 artifact：

```text
deepseek_inst1/execution_context.json
deepseek_inst2/execution_context.json
minicpm5_router/strategy_decision.json
fusionroute_gateway/compatibility_report.json
```

但系统仍然回答不了：

- 哪些 component 同属一个 deployment
- 哪些 component 由谁调度
- 哪些 health 是系统 gating 的必需项
- 哪些 runtime contract 彼此存在路由关系

因此：

- `ExecutionContext / State ABI / StrategyDecision / CompatibilityReport`
  解决的是单 runtime 合法性
- `System Execution Manifest`
  解决的是多 runtime 组装合法性

---

## 8. 当前已落地 vs 下一刀该补

### 8.1 已落地

当前已经适合进入正式 contract 的内容：

- `execution_context.json`
- `state_abi.json`
- `strategy_decision.json`
- `compatibility_report.json`
- `component_id`
- `component_role`
- `system_execution_manifest.json`

### 8.2 下一刀该补

最小、最稳、不破坏现有 helper 的下一刀是：

1. 让运行器自动收集多个 component 的 artifact path，而不是只依赖 config 注入
2. 用 `required_components + overall_ready_rule` 统一系统级 ready 判定
3. 让 `FusionRoute` 不再只是“外部网关描述”，而是系统 manifest 中的正式 component
4. 补跨 host / 跨进程的 manifest 聚合入口

不建议当前就做的事情：

- 不建议把多 component graph 直接塞回单个 `execution_context.json`
- 不建议让 `state_abi.json` 承担 routing topology
- 不建议把系统清单逻辑硬塞进 `strategy_decision.json`

---

## 10. 示例配置与生成脚本

当前仓库已提供一套最小可执行示例：

- 示例配置：
  `docs/technical_whitepapers/examples/deepseek_x4_minicpm5_fusionroute_system_manifest.example.json`
- 生成脚本：
  `cgc_engine/tools/scripts/demo/generate_system_execution_manifest_example.py`

这套示例会：

- 为 `FusionRoute + DeepSeek x4 + MiniCPM5` 逐个生成 runtime contract artifact
- 自动聚合同一 export root 下的 sibling `contract_manifest.json`
- 最终输出完整 `system_execution_manifest.json`

---

## 9. 结论

本白皮书给出的不是一套替换现有 pipeline 的大重构方案，而是：

- 先正式化当前已经存在的 runtime contract
- 再在其上增加最薄的 system manifest

这条路径的优点是：

- 与 `pipeline.py` 当前 helper 直接对齐
- 不要求先完成统一大内核重写
- 可以先服务 `DeepSeek-V4-Flash x4 + MiniCPM5 + FusionRoute`
- 同时仍兼容后续 `psi0`、agent、edge-cloud 等执行模式

最终工程结论：

```text
单 runtime 用 component contract；
多 runtime 组合用 system execution manifest。
```
