# CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation

本目录存放 `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation` 的正式白皮书与配套 artifact。

当前口径已经同步到最新代码审计结果：

- 当前正式真源以同目录 `*_gate_map.json` 为准
- 最新正式分级为：`52 done / 0 proof / 0 target / 0 stub`
- `summary / checkin / README` 已同步跟随该分级
- Host1 / Host2 已实跑收敛的近期闭环能力为：`sglang_deepep_tp4ep4_prefill_foundation`、`deepseek_v4_flash_resume_decode_path`、`ray_engine_dual_host_service_topology`、`colossalai_distributed_runtime_candidate`、`g21_eight_step_pipeline_governance_integration`、`g21_upk_binding_for_fusion_variants`、`g21_state_abi_extension_hook`、`g22_kv_kv_cache_management`、`g22_kv_cache_reuse`、`g22_kv_dynamic_cache_sizing`、`g22_kv_cache_prefetching`、`g22_deepep_l20n_dualnode_16gpus`、`g22_deepep_l20n_megatrain_8step`、`g22_deepep_l20n_inference_8step`、`g22_deepep_bootstrap_deepep_compat`、`g22_deepep_system_profile_l20n`、`g22_deepep_upk_l20n_optimization`、`g22_deepep_state_abi_l20n`、`g23_rswa_double_layer_kv`、`g23_gds_nfsordma_direct_io`、`g23_trueorthokda_adapter`、`g23_cloud_l20n_tp4_adaptation`、`g23_unified_ir_inject_sglang_compute_graph`

## 目录内容

- `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_Technical_Whitepaper_v1.0_zh_CN.md`
  - Gate 2.0 的正式技术白皮书
  - 用于定义“层粒度自适应端云 PD 分离”边界、当前已落地基座与 2.0 目标能力
- `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_gate_map.json`
  - 结构化 gate map
  - 适合 `CLI summary`、`bundle audit`、`release checkin`、`dashboard/report`
- `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_checkin.example.json`
  - release checkin 实例格式
  - 保持与现有 checkin 投影兼容
- `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_summary.example.json`
  - 面向 UI / dashboard 的扁平 summary 输出样板

## Gate 2.0 的核心语义

`CGC_Gate_2.0` 不是重新讲一遍端云自治，而是在 Gate 1.0 已经建立的基座上，继续定义：

- 端侧 `by-layer` 自适应执行
- `max_local_layer` 驱动的安全层数选择
- `finished_layer + 1` 驱动的云侧继续 Prefill
- `hidden_states + partial_kv` 的正式中间态 ABI
- layer-wise KV 向独立 Decode 集群同步
- `UD-IQ2 + KDA` 联合传输档位
- MoE 专家路由跨端云一致性

因此，Gate 2.0 的重点是：

- **层粒度切分**
- **层粒度接续**
- **层粒度状态传输**

而不是单纯证明 handoff 已经存在。

## 当前文档的使用口径

Gate 2.0 目录中的文档采用与正式白皮书和 gate_map 一致的 evidence-aware 口径；当前 52 项能力均已被正式收敛为 `done`。

当前状态分为：

- `done`

其中：

- `done`
  - 表示已具备正式代码证据、契约面与当前文档认可的 runtime / 产品化闭环

## 新增承接关系

本轮已明确把以下三类关系纳入 Gate 2.0：

- `UPKG 2.0`
  - 作为 `model product gate` 承接 `Gate 2.0` 的模型产品化、治理与交付面
- `UPKG 3.x`
  - 作为 `agent product chain` 承接 `Gate 2.0` 的 artifact、summary、attribution 与 replay 消费面
- `m7.6 dev gate`
  - 作为 `m76` 的开发侧解释，承接 heterogeneous integration bring-up、marker 对齐与 DFlash runtime contract 补齐

其中 `m7.6 dev gate` 是本文中的解释性术语，不代表 repo 新增了一个独立 CLI gate；当前对外正式 gate 仍是 `m76`。

## Legacy ID 映射

为避免 `CLI / verifier` 与当前 `gate_map.json` 看起来像两套能力模型，本目录采用以下映射口径：

| legacy id | current gate_map bucket | audited status |
| --- | --- | --- |
| `g23_rswa_double_layer_kv` | `g23_rswa_double_layer_kv` | `done` |
| `g23_prefill_pool_dynamic_management` | `g23_prefill_pool_dynamic_management` | `done` |
| `g23_gds_nfsordma_direct_io` | `g23_gds_nfsordma_direct_io` | `done` |
| `g23_trueorthokda_adapter` | `g23_trueorthokda_adapter` | `done` |
| `g23_cloud_l20n_tp4_adaptation` | `g23_cloud_l20n_tp4_adaptation` | `done` |
| `g23_unified_ir_inject_sglang_compute_graph` | `g23_sglang_backend_integration` + `moe_route_consistency_across_edge_cloud` | `done` |
| `g23_endtoend_moe_tensor_transport` | split across `g23_gds_nfsordma_direct_io` + `g23_trueorthokda_adapter` + `g23_rswa_double_layer_kv` + `g23_prefill_pool_dynamic_management` + `g23_sglang_backend_integration` + `moe_route_consistency_across_edge_cloud` | `done` |

说明：

- `same capability id` 的条目表示 legacy id 在当前 `gate_map` 中仍保留为一等能力项。
- `folded` / `split` 的条目表示旧版 CLI/verifier 里的单个 capability，已被审计收口为更大的 current bucket，或被拆分成多个更细的 current bucket。
- 机器消费请以 `gate_map.json` 中的 `legacy_capability_mapping` 字段为准；本 README 仅提供人读入口。

## 与 Gate 1.0 的关系

`CGC_Gate_2.0` 依赖：

- [CGC_Gate_1.0 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy/CGC_Gate_1.0_edge_cloud_autonomy_Technical_Whitepaper_v1.0_zh_CN.md)

Gate 1.0 负责收口：

- 端侧自治入口
- 状态传输与恢复
- `PD -> DOPD` handoff 主链
- gateway auto-publish
- contract 与 bundle governance

其中 Gate 1.0 已实证闭环并因此被 Gate 2.0 继承为 `done` 的基础项包括：

- `edge_omlx_flashmoe_autonomous_entry`
- `cq4_transport_plane`
- `trueorthokda_zero_copy_state_runtime`
- `dopd_handoff_control_plane`
- `real_prefill_producer_and_auto_publish`
- `task_type_contract_and_bundle_governance`

Gate 2.0 则在这个基座上，正式收口层粒度 continuation、中间态 ABI、端云分层传输与 merged 2.1/2.2/2.3 能力层。

## 何时看这个目录

优先查看本目录的场景：

- 需要理解 2.0 的正式设计边界
- 需要查看 Gate 2.0 当前的正式 `done` 能力矩阵与消费边界
- 需要消费 2.0 的 `gate_map / checkin / summary`
- 需要为 `max_local_layer / finished_layer / hidden_states / partial_kv / route consistency` 等能力消费正式真源
