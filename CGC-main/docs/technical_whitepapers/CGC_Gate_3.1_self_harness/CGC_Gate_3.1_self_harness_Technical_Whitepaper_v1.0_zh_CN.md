
# CGC Gate 3.1 Self-Harness 技术白皮书

## 1. 文档信息

| 项目 | 内容 |
|------|------|
| **版本** | V1.0 |
| **日期** | 2026年6月 |
| **适用范围** | CGC Engine Gate 3.1 |
| **能力来源** | Self-Harness + RHO + 端云桥接 + Guardian |
| **Gate 状态** | VALIDATED |
| **功能状态** | 7 done / 0 proof / 0 stub |

> 审核说明：当前本地、host1、host2 真源均已为 Gate 3.1 建立 `capability -> CLI -> self-harness verifier -> JSON evidence` 静态契约；本地 `validation_report_gate31_contract_extension_capability_cli_contract.json`、host1 `validation_report_gate31_host1_truthsource_sync_capability_cli_contract.json`、host2 `validation_report_gate31_host2_truthsource_sync_capability_cli_contract.json` 均为 7/7 PASS。权威能力状态以同目录 `CGC_Gate_3.1_self_harness_gate_map.json` 为准。

## 2. 目录

1. [文档信息](#1-文档信息)
2. [核心架构](#2-核心架构)
3. [四大核心能力](#3-四大核心能力)
4. [Self-Harness 三阶段闭环](#4-self-harness-三阶段闭环)
5. [能力矩阵](#5-能力矩阵)

## 3. 核心架构

CGC Gate 3.1 Self-Harness 是弱模型自优化执行框架，具备以下特性：

- **固定权重**：模型权重在执行过程中保持不变
- **本地闭环**：推理和优化在本地完成，无需外部依赖
- **防退化**：内置 Guardian 机制防止性能退化
- **端云桥接**：支持端侧和云端的无缝切换

## 4. 四大核心能力

### 4.1 Self-Harness 三阶段闭环

| 阶段 | 名称 | 说明 |
|------|------|------|
| 阶段一 | 推理执行 | 标准推理流程，收集执行数据 |
| 阶段二 | 性能分析 | RHO（Runtime Health Observer）实时监控 |
| 阶段三 | 自适应优化 | 基于分析结果进行本地优化 |

### 4.2 RHO（Runtime Health Observer）

实时运行时健康监测系统：
- 推理延迟监控
- 内存使用追踪
- 异常检测与预警

### 4.3 端云桥接

支持端侧（Apple Silicon）和云端（GPU）的无缝切换：
- 模型格式自动转换
- 推理结果一致性保证
- 负载智能调度

### 4.4 Guardian 防退化机制

| 防护维度 | 机制 |
|----------|------|
| 性能退化 | 检测到性能下降时自动回退 |
| 结果异常 | 输出质量监控与修正 |
| 安全边界 | 输入输出范围约束 |

## 5. 能力矩阵

| capability_id | 名称 | 状态 | 说明 |
|---------------|------|------|------|
| `self_harness_three_stage_loop` | Self-Harness 三阶段闭环 | done | Gate 3.1 preflight、shared CLI 与 legacy closure verifier 已形成行级证据 |
| `rho_runtime_health_observer` | RHO 运行时健康监测 | done | Gate 3.1 verifier 与 `embodied monitor --help` surface 已对齐 |
| `edge_cloud_bridge_adaptive` | 端云自适应桥接 | done | `transport_contract / cq4_session / deepseek_v4 resume hook` 已被 verifier 逐项验真 |
| `guardian_degeneration_prevention` | Guardian 防退化 | done | Guardian legacy metrics 与 Gate 3.1 verifier 已汇入 companion contract report |
| `fixed_weight_execution` | 固定权重执行 | done | `@torch.no_grad / torch.inference_mode / model.eval()` runtime markers 已收口 |
| `local_optimization_engine` | 本地优化引擎 | done | `validate_dynamic_policy` 与 Gate 3.1 本地优化 verifier 已形成直接证据 |
| `self_harness_cli` | Self-Harness CLI 工具 | done | `cgc model verify --self-harness --gate 3.1` parser surface 与 help 输出已对齐 |

## 6. 与其他 Gate 的关系

| Gate 版本 | 关系 | 说明 |
|-----------|------|------|
| Gate 3.0 | 基础 | Gate 3.1 基于 Gate 3.0 构建 |
| Gate 5.0 | 集成 | Gate 3.1 是 Gate 5.0 的核心能力来源之一 |

## 7. 部署状态

- **本地**：`validation_report_gate31_contract_extension.json` 为 7/7 PASS；companion contract report `validation_report_gate31_contract_extension_capability_cli_contract.json` 为 7/7 PASS
- **Host1**：`/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host1_truthsource_sync.json`、`/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host1_truthsource_sync_capability_cli_contract.json`、`/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host1_formal_preflight.json` 均为 7/7 PASS
- **Host2**：`/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host2_truthsource_sync.json`、`/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host2_truthsource_sync_capability_cli_contract.json`、`/root/flashkv0516/ComputeGraphCompiler-main/validation_report_gate31_host2_formal_preflight.json` 均为 7/7 PASS

---

## 8. CLI 参数与测试框架

### 8.1 CLI 参数总览

`CGC_Gate_3.1_self_harness` 对应的 CLI 参数如下：

| 能力 | CLI 参数 | 说明 |
|------|----------|------|
| Gate 3.1 shared CLI | `cgc model verify --gate 3.1 --self-harness` | 真实 Gate 3.1 self-harness 入口 |
| RHO 监控 surface | `cgc embodied monitor --metrics` | RHO / metrics / alerts 共享监控表面 |
| Formal preflight | `python3 cgc_engine/tools/scripts/run/gate_test_framework.py --self-harness --gate CGC_Gate_3.1_self_harness` | 消费 Gate 3.1 capability contract report 的 formal/preflight 入口 |

### 8.2 测试框架集成

本 gate 的能力通过 CGC Gate Test Framework 进行验证：

```bash
# 运行 Gate 3.1 全量测试
python cgc_engine/tools/scripts/run/gate_test_framework.py --gate CGC_Gate_3.1_self_harness

# Self-Harness 三阶段验证
python cgc_engine/tools/scripts/run/gate_test_framework.py --self-harness --gate CGC_Gate_3.1_self_harness

# 生成 Gate 3.1 companion contract report
python3 cgc_engine/tools/scripts/run/self_harness_validation_framework.py --gate 3.1 --output validation_report_gate31_contract_extension.json
```

### 8.3 测试覆盖范围

| 测试维度 | 覆盖内容 |
|----------|----------|
| 三阶段闭环 | 策略决策 → 图捕获 → 执行校验 |
| RHO 监控 | 实时性能监控、异常检测 |
| Guardian | 性能退化保护、质量保障 |
| 本地优化 | 无需外部依赖的优化能力 |
