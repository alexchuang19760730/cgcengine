# CGC Gate 3.1 Self-Harness

## 概述

CGC Gate 3.1 Self-Harness 是弱模型自优化执行框架；当前本地真源已收敛到 `7 done / 0 proof / 0 stub`。当前能力表以同目录 `gate_map / summary / checkin` 为准：

- **Self-Harness 三阶段闭环**：推理-分析-优化三阶段执行闭环
- **RHO（Runtime Health Observer）**：实时运行时健康监测系统
- **端云桥接**：支持端侧（Apple Silicon）和云端（GPU）的无缝切换
- **Guardian 防退化机制**：三重防护维度确保系统稳定性

## 目录结构

```
CGC_Gate_3.1_self_harness/
├── CGC_Gate_3.1_self_harness_Technical_Whitepaper_v1.0_zh_CN.md  # 技术白皮书
├── CGC_Gate_3.1_self_harness_gate_map.json                        # 能力矩阵配置
├── CGC_Gate_3.1_self_harness_checkin.example.json                 # 签入配置示例
├── CGC_Gate_3.1_self_harness_summary.example.json                 # 摘要配置示例
├── gate31_capability_cli_self_harness_contract.json               # 能力->CLI->verifier 静态契约
└── README.md                                                       # 本文件
```

## 核心能力

| 能力 ID | 名称 | 状态 |
|---------|------|------|
| `self_harness_three_stage_loop` | Self-Harness 三阶段闭环 | done |
| `rho_runtime_health_observer` | RHO 运行时健康监测 | done |
| `edge_cloud_bridge_adaptive` | 端云自适应桥接 | done |
| `guardian_degeneration_prevention` | Guardian 防退化 | done |
| `fixed_weight_execution` | 固定权重执行 | done |
| `local_optimization_engine` | 本地优化引擎 | done |
| `self_harness_cli` | Self-Harness CLI 工具 | done |

## 与其他 Gate 的关系

- **基于 Gate 3.0 构建**：继承训推一体能力
- **为 Gate 5.0 提供核心能力**：作为审计追踪重放可视化的基础

## 状态

- 当前本地审计结果为 `7 done / 0 proof / 0 stub`
- `python3 cgc_engine/tools/scripts/run/self_harness_validation_framework.py --gate 3.1 --output validation_report_gate31_contract_extension.json` 当前结果为 `7/7 PASS`
- companion contract report `validation_report_gate31_contract_extension_capability_cli_contract.json` 当前结果为 `7/7 PASS`
- Gate 3.1 formal preflight 已改为消费同一份 capability contract report，不再仅依赖 artifact 存在性
- Host1 远端已完成 truth-source 同步与补跑，`validation_report_gate31_host1_truthsource_sync.json`、`validation_report_gate31_host1_truthsource_sync_capability_cli_contract.json`、`validation_report_gate31_host1_formal_preflight.json` 均为 `7/7 PASS`
- Host2 远端已完成 truth-source 同步与补跑，`validation_report_gate31_host2_truthsource_sync.json`、`validation_report_gate31_host2_truthsource_sync_capability_cli_contract.json`、`validation_report_gate31_host2_formal_preflight.json` 均为 `7/7 PASS`

## 双机证据路径

- Host1 self-harness: `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host1_truthsource_sync.json`
- Host1 capability contract: `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host1_truthsource_sync_capability_cli_contract.json`
- Host1 formal preflight: `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host1_formal_preflight.json`
- Host2 self-harness: `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host2_truthsource_sync.json`
- Host2 capability contract: `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host2_truthsource_sync_capability_cli_contract.json`
- Host2 formal preflight: `/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host2_formal_preflight.json`

## 快速开始

```bash
# 运行 Gate 3.1 self-harness 验证
python3 cgc_engine/tools/scripts/run/self_harness_validation_framework.py --gate 3.1 --output validation_report_gate31_contract_extension.json

# 运行 Gate 3.1 formal / preflight
python3 cgc_engine/tools/scripts/run/gate_test_framework.py --self-harness --gate CGC_Gate_3.1_self_harness

# 检查 shared CLI surface
python3 cgc_engine/cli.py model verify --help
```

## 文档

详细技术文档请参考：[技术白皮书](CGC_Gate_3.1_self_harness_Technical_Whitepaper_v1.0_zh_CN.md)
