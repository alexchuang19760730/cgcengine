# CGC_Gate_1.0_edge_cloud_autonomy

本目录存放 `CGC_Gate_1.0_edge_cloud_autonomy` 的正式白皮书与配套 artifact。

当前口径已经同步到最新审计结果：

- `gate_map / summary / checkin` 统一使用 `done / proof / target / stub`
- 当前 Host1 / Host2 的实证结果已经把 Gate 1.0 收口到 `8 done + 0 proof`
- 当前正式真源以同目录下的 `*_gate_map.json` 为准

## 目录内容

- `CGC_Gate_1.0_edge_cloud_autonomy_Technical_Whitepaper_v1.0_zh_CN.md`
  - Gate 1.0 的正式技术白皮书
  - 用于定义端云自治能力、状态传输、DOPD handoff 与治理链的正式边界
- `CGC_Gate_1.0_edge_cloud_autonomy_gate_map.json`
  - 面向机器消费的 gate map
  - 用于 `CLI summary`、`bundle audit`、`release checkin`、`dashboard/report`
- `CGC_Gate_1.0_edge_cloud_autonomy_checkin.example.json`
  - release checkin 实例格式
  - 保持与现有 checkin 风格兼容
- `CGC_Gate_1.0_edge_cloud_autonomy_summary.example.json`
  - 面向 UI / dashboard 的扁平 summary 输出样板

## Gate 1.0 的核心语义

`CGC_Gate_1.0_edge_cloud_autonomy` 解决的是“端云自治能力是否已经存在并可被正式治理”的问题，重点包括：

- 端侧自治入口是否存在
- 状态传输与恢复链是否存在
- `PD -> DOPD` handoff 主链是否存在
- gateway auto-publish 是否已接入真实请求完成点
- `task_type` contract 与 profile bundle governance 是否已经正式化

换句话说，Gate 1.0 的重点是：

- **自治存在**
- **handoff 存在**
- **治理存在**

而不是层粒度 continuation 本身。

## 与 Gate 2.0 的关系

`CGC_Gate_1.0` 是 `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation` 的基座。

如果没有 Gate 1.0 已经落地的这些能力：

- `OMLX + FlashMoE` 端侧自治入口
- `CQ4`
- `TrueOrthoKDA`
- `Zero-Copy VRAM`
- `DOPD` handoff 控制面
- gateway auto-publish
- bundle governance

那么 Gate 2.0 的层粒度切分、层粒度接续与中间态 ABI 就没有正式工程前提。

## Legacy ID 映射

Gate 1.0 当前正式真源只保留 8 个 audited capability bucket，但历史 CLI / verifier / 白皮书矩阵里仍保留 12 个旧 capability id。当前统一以同目录 `*_gate_map.json` 的 `legacy_capability_mapping` 为机器真源。

旧 id 的主要来源有三类：

- `gate_test_framework.py` 里的 Gate 1.0 旧 capability 注册
- Gate 1.0 技术白皮书 11.1 节的 12 个 CLI flag
- `UPKG 1.1 / M7.5 active runtime / upkg12_dopd_runtime_closure / upkg120_edge_cloud_autonomy` 这些历史命名面

| Legacy capability id | Current gate_map bucket | Mapping kind | Audited status |
| --- | --- | --- | --- |
| `dopd_handoff` | `dopd_prefill_decode_decoupling` | `folded_into_broader_bucket` | `done` |
| `cq4_transport` | `cq4_edge_cloud_protocol` | `folded_into_broader_bucket` | `done` |
| `trueorthokda` | `trueorthokda_kv_cq4_compression` | `folded_into_broader_bucket` | `done` |
| `zero_copy` | `zero_copy_vram_handoff` | `folded_into_broader_bucket` | `done` |
| `prefill_producer` | `dopd_prefill_decode_decoupling`, `deepseek_v4_flash_cloud_runtime_resume_decode` | `split_across_multiple_buckets` | `proof` |
| `task_type_contract` | `task_type_contract_bundle_governance` | `folded_into_broader_bucket` | `done` |
| `ray_dual_host` | `sglang_deepep_tp4ep4_cloud_foundation` | `folded_into_broader_bucket` | `done` |
| `moe_route_consistency` | `edge_omlx_flashmoe_memory_threshold_decision` | `folded_into_broader_bucket` | `done` |
| `upkg_manager` | `task_type_contract_bundle_governance` | `folded_into_broader_bucket` | `done` |
| `system_profile` | `task_type_contract_bundle_governance` | `folded_into_broader_bucket` | `done` |
| `state_abi` | `cq4_edge_cloud_protocol`, `trueorthokda_kv_cq4_compression`, `dopd_prefill_decode_decoupling` | `split_across_multiple_buckets` | `done` |
| `bootstrap` | `deepseek_v4_flash_cloud_runtime_resume_decode`, `task_type_contract_bundle_governance` | `split_across_multiple_buckets` | `done` |

## 何时看这个目录

优先查看本目录的场景：

- 需要理解 Gate 1.0 的正式边界
- 需要查看当前已可正式宣称的端云自治能力
- 需要消费 1.0 的 `gate_map / checkin / summary`
- 需要给 `bundle review / verify / audit` 体系对齐治理语义
