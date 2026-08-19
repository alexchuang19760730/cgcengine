# CGC Gate 6.0 FusionRoute Complete

CGC Engine 产品化完整版本，整合 FusionRoute 4-instance、DeepEP、端云协同三大核心能力，以及 M8.x 系列全部产品化 CGC 指令集。

## 📋 快速概览

| 项目 | 说明 |
|------|------|
| **Gate ID** | `CGC_Gate_6.0_fusionroute_complete` |
| **状态** | ✅ validated |
| **版本** | v1.0 |
| **日期** | 2026-07 |

## 🎯 核心能力

### FusionRoute
- **四实例路由**：4 个 DeepSeek-V4-Flash 实例
- **MiniCPM5 Router**：智能路由决策引擎
- **Perception Matrix**：环境/任务/模型/硬件感知后给出受契约约束的策略建议

### DeepEP MoE
- **LPLB**：线性规划负载均衡
- **Waterfill**：带宽感知注水算法
- **EPLB**：静态专家副本调度

### 端云协同
- **四种执行模式**：cloud_only、edge_only、hybrid、auto
- **Self-Harness**：三阶段闭环架构
- **Guardian**：防退化机制

### M8.x CGC 指令集

| 指令 | 功能 | 端云协同映射 |
|------|------|--------------|
| `cgc bundle` | Bundle 治理与审计 | task_type contract 四段一致性 |
| `cgc model` | 模型验证、审计与部署 | profile_bundle_validator |
| `cgc gateway` | 网关管理 | auto-publish streaming/non-streaming |
| `cgc edge` | 端侧状态与恢复 | DOPD resume_from_kda_state |
| `cgc train` | 训练任务管理 | 云侧训练调度 |
| `cgc infer` | 推理服务管理 | 端云推理协同 |
| `cgc validate` | Gate 验证 | m1-m7.6 + upkg21 实跑验证 |
| `cgc bench` | 性能基准测试 | 端云延迟对比 |
| `cgc monitor` | 监控管理 | 显存/内存监控 |
| `cgc audit` | 审计日志 | 治理链审计 |
| `cgc ops` | 运维操作 | 系统运维 |

## 📁 目录结构

```
CGC_Gate_6.0_fusionroute_complete/
├── CGC_Gate_6.0_fusionroute_complete_Technical_Whitepaper_v1.0_zh_CN.md
├── CGC_Gate_6.0_fusionroute_complete_summary.example.json
├── CGC_Gate_6.0_fusionroute_complete_checkin.example.json
├── gate6_capability_cli_self_harness_contract.json
├── Gate6_ModelVerify_to_M76_ManifestFirst_Mapping.md
├── role_locality_contract.schema.json
├── placement_decision_report.schema.json
├── policy_suggestion_report.schema.json
├── contract_projection_report.schema.json
├── policy_suggestion_report.example.json
├── contract_projection_report.example.json
├── gate6_fusionroute_v2_draft_contract.json
├── gate6_fusionroute_v2_formal_contract.json
├── fusionroute_v2_formal_contract_report.json
├── swe_verified_formal_summary.json
├── swe_verified_500_detailed_evidence.json
├── gate_map.json
└── README.md
```

## 🔗 依赖关系

```
Gate 6.0
├── Gate 1.0 (端云自治)
├── Gate 2.2 (DeepEP MoE)
├── Gate 3.1 (Self-Harness)
├── UPKG M7.6 (FusionRoute)
└── UPKG M8.x (CLI 指令集)
```

## 🧭 治理边界

- 所有正式优化必须以 `Gate 6.0` 已定义能力与可执行 CLI 为边界。
- 超出 `Gate 6.0 + CLI` 边界的工作只能记为探索证据，不得直接提升为正式 release claim、summary 或 gate 结论。

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| FusionRoute 延迟 | < 50ms |
| 推理吞吐量 | 145 req/s |
| 负载均衡效率 | 98% |
| 端云切换延迟 | < 10ms |
| Guardian 验证率 | 100% |

## 🚀 快速开始

```bash
# 验证 Gate 6.0
python cgc_engine/tools/scripts/run/gate_test_framework.py --gate CGC_Gate_6.0_fusionroute_complete

# Self-Harness 三阶段验证
python cgc_engine/tools/scripts/run/gate_test_framework.py --self-harness --gate CGC_Gate_6.0_fusionroute_complete

# 验证 SWE Verified 500 能力
python cgc_engine/cli.py validate --capability swe_verified_500 --print-json

# 查看 validate 侧完整能力矩阵
python cgc_engine/cli.py validate --all --print-json

# 启用 SWE Verified 500 任务类型 alias
python cgc_engine/cli.py model --task-type swe --model swe_verified --print-json

# 启用 SWE Verified 500 FusionRoute alias
python cgc_engine/cli.py model --fusion-config swe --model swe_verified --print-json

# 在 model 侧直接验证 SWE Verified 500 能力
python cgc_engine/cli.py model --validate-capability swe_verified_500 --print-json

# 启动推理服务
cgc infer start --backend fusionroute

# 查看监控
cgc monitor dashboard

# FusionRoute v2 静态矩阵
python cgc_engine/cli.py fusionroute plan --task-type CODEGEN --print-json

# 角色 locality 契约
python cgc_engine/cli.py fusionroute contract show --kind role-locality --role UI-TARS --print-json

# Perception Matrix 策略建议
python cgc_engine/cli.py fusionroute perception plan --task-type CODEGEN --environment-type repo --print-json

# Perception Matrix 契约投影
python cgc_engine/cli.py fusionroute perception project --task-type CODEGEN --environment-type repo --print-json

# FusionRoute v2 正式契约聚合验证
python cgc_engine/cli.py fusionroute verify --capability all --print-json
```

`validate --all` 当前会显式汇总四条能力，并按证据强度区分状态语义：

- `swe_verified_500`: `PARTIAL`
- `dflash`: `PASS`
- `jetspec`: `CONFIGURED`
- `fusionroute`: `PASS`

其中目前的收敛口径是：

- `fusionroute=PASS`：已能从 `upkg21 + m76 runtime/topology` 证据链读到 route/topo/backend。
- `swe_verified_500=PARTIAL`：当前必须按双层口径解读：
  - `formal_chain_status=PASS`
  - `official_eval_status=SUBMITTED`
  - `claimable=false`
  - `swe_verified_passed_tasks=0`
- 这表示 `swe_verified_500` 已正式挂接到 `upkg21` 通过产物链，但官方评测结果仍不可 claimable，不能把 formal chain `PASS` 误写成 capability `PASS`。
- 因此 `gate_map.json` 中该条能力记为 `status=integrated`、`proof=m76_swe_verified_formal_chain`，表示正式链路已接入，但 release-facing claim 仍保持保守口径。
- `jetspec=CONFIGURED`：bridge/manifest 已记录 `fusion + jetspec_branches=8`，但当前文档约束仍将 JetSpec 保持在探索语义层，尚未升级成稳定 release-facing runtime contract。

## ✅ 当前验证快照

- `formal preflight`：本地 `8/8 PASS`
  - `/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/tools/scripts/run/gate_test_report_20260708_004019.json`
- `self-harness`：本地 `11/11 PASS`
  - `/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/validation_report_gate60_20260708_formal_fusionroute_closure.json`
- `host1 formal preflight`：`8/8 PASS`
  - `/Users/alexchuang/Documents/flashkv0516/temp/gate60_truthsource_closure_20260708/host1/gate_test_report_20260708_005349.json`
- `host2 formal preflight`：`8/8 PASS`
  - `/Users/alexchuang/Documents/flashkv0516/temp/gate60_truthsource_closure_20260708/host2/gate_test_report_20260708_005359.json`
- `host1 self-harness`：`11/11 PASS`
  - `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate60_20260708_formal_fusionroute_closure_remote_host1.json`
- `host2 self-harness`：`11/11 PASS`
  - `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate60_20260708_formal_fusionroute_closure_remote_host2.json`
- `Gate 6.0 capability -> CLI -> self-harness 静态契约`：`29/29 PASS`
  - contract manifest:
    `/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers/CGC_Gate_6.0_fusionroute_complete/gate6_capability_cli_self_harness_contract.json`
  - local generated report:
    `/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/validation_report_gate60_20260708_formal_fusionroute_closure_capability_cli_contract.json`
  - host1 generated report:
    `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate60_20260708_formal_fusionroute_closure_remote_host1_capability_cli_contract.json`
  - host2 generated report:
    `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate60_20260708_formal_fusionroute_closure_remote_host2_capability_cli_contract.json`

上述快照已经投影到：

- `CGC_Gate_6.0_fusionroute_complete_summary.example.json`
- `CGC_Gate_6.0_fusionroute_complete_checkin.example.json`

## 🗣️ 正式宣称摘录

- 可以正式写成：
  - `Gate 6.0 formal preflight` 已在 local、host1、host2 达成 `8/8 PASS`，`self-harness` 已在 local、host1、host2 达成 `11/11 PASS`。
  - `Gate 6.0 capability -> CLI -> self-harness` 静态契约已在 local、host1、host2 达成 `29/29 PASS`，并把 FusionRoute v2 / Role Locality / Perception Matrix 六条证据链正式并入主链。
  - 当前 repo snapshot 下共有 `28` 条 capability 处于 `done / formally claimable`。
- 必须保留的治理边界：
  - `swe_verified_500` 当前仍为 `integrated / PARTIAL / non-claimable`。
  - 原因是 `formal_chain_status=PASS` 仅表示正式链路接入完成；`official_eval_status=SUBMITTED`、`claimable=false` 代表外部 benchmark 分数仍不可对外认领。

## 📄 文档

- [技术白皮书](CGC_Gate_6.0_fusionroute_complete_Technical_Whitepaper_v1.0_zh_CN.md)
- [能力映射](gate_map.json)
- [Summary Example](CGC_Gate_6.0_fusionroute_complete_summary.example.json)
- [Checkin Example](CGC_Gate_6.0_fusionroute_complete_checkin.example.json)
- [Capability CLI Contract](gate6_capability_cli_self_harness_contract.json)
- [FusionRoute Role Locality 子白皮书](CGC_Gate_6.0_FusionRoute_Role_Locality_Technical_Whitepaper_v1.0_zh_CN.md)
- [跨 Gate FusionRoute 最终拓扑矩阵](../CGC_FusionRoute_Final_Topology_Matrix_Technical_Whitepaper_v1.0_zh_CN.md)
- [FusionRoute v2 静态契约白皮书](../CGC_FusionRoute_v2_Static_Contract_Technical_Whitepaper_v1.0_zh_CN.md)
- [Perception Matrix + LLM 白皮书](../CGC_Perception_Matrix_LLM_Technical_Whitepaper_v1.0_zh_CN.md)
- [Role Locality Schema](role_locality_contract.schema.json)
- [Placement Decision Schema](placement_decision_report.schema.json)
- [Policy Suggestion Schema](policy_suggestion_report.schema.json)
- [Contract Projection Schema](contract_projection_report.schema.json)
- [FusionRoute v2 Formal Contract](gate6_fusionroute_v2_formal_contract.json)
- [FusionRoute v2 Draft Contract](gate6_fusionroute_v2_draft_contract.json)
- [探索命令到正式链映射](Gate6_ModelVerify_to_M76_ManifestFirst_Mapping.md)
