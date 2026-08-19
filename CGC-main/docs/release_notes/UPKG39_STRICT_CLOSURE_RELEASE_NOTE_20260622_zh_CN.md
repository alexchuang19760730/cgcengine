# UPKG 3.9 Strict Closure Release Note

**发布日期**: 2026-06-22  
**状态**: Formal PASS  
**适用范围**: `UPKG 3.x agent product gate`  
**正式入口**: `cgc gate upkg39`

## 1. 摘要

`UPKG 3.9` 已完成正式收口并通过验收。该版本不是新的训练产品链，也不是 `UPKG 4.0 embodied` 的别名，而是基于 `UPKG 3.8` 运行产物链与 `v0.2` 正式 contract 的 strict closure gate。

本次 release 的核心价值在于：

- 恢复 `alignment_score >= 0.8` 的严格验收门槛
- 将 `schema / 字段字典 / 合法样例 / 非法样例 / validator execution` materialize 为正式 artifact
- 将 `graph-native stage execution` 收口到 `per-stage tensorized GUI source`
- 输出可独立验收的 `end-to-end executor closure`

## 2. 正式结果

本次正式验收结果为：

- `cgc gate upkg39 = PASS`
- `strict_alignment_acceptance = PASS`
- `q2rl_strict_acceptance = PASS`
- `validator_execution_report = PASS`
- `graph_native_tensorized_execution = PASS`
- `end_to_end_executor_closure = PASS`
- `upkg39_completion_manifest = PASS`

## 3. 关键修正

本次正式收口中，真正需要修正的问题并非 `UPKG 3.9 strict closure` contract 本体，而是聚合实现缺口：

- `upkg38` 聚合阶段此前未将 `pipeline contract` 正式向上游 gate 传递
- 在这一情况下，`m72 / m77 / m78 / upkg39` 于聚合验收时会把 `pipeline_kernel_contract_artifacts` 误判为未就绪
- 该问题会导致 strict closure 所需的 upstream contract 被错误标记为失败

为此，本次版本补入了统一的 upstream fallback 机制：

- 当本地 root report 未携带完整 `pipeline contract` 时
- 后续 gate 允许回退读取上游 gate 已存在的 `pipeline_kernel_contract_artifacts`
- 同时回退读取上游 gate 已存在的 `pipeline_contract_descriptor`

该修正使 `UPKG 3.9` 可以在不破坏现有 `3.8` 产物链与 `4.0` 边界的前提下正式收口。

## 4. 边界说明

- `UPKG 3.9` 属于 `UPKG 3.x`
- `UPKG 3.9` 不等于 `UPKG 4.0`
- `UPKG 3.9` 不承载 `psi0 / realtime-vla / embodied benchmark`
- `UPKG 3.9` 的职责是将 `3.8 + v0.2` 中已定义但尚未严格机验的部分收口为独立 gate

## 5. 主要正式 artifact

- `schema_bundle_manifest.json`
- `field_dictionary_manifest.json`
- `validator_execution_report.json`
- `strict_alignment_acceptance.json`
- `q2rl_strict_acceptance.json`
- `graph_native_tensorized_execution_report.json`
- `end_to_end_executor_closure.json`
- `upkg39_completion_manifest.json`
- `upkg39_report.json`
- `summary.json`

## 6. 关联文档

- `docs/gate_whitepapers/CGC_UPKG_3_0_AGENT_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md`
- `docs/technical_whitepapers/CGC_UPKG_3_X_UNIFIED_PRODUCT_CHAIN_TECHNICAL_SPEC_v0.2_zh_CN.md`

## 7. 验收产物位置

- aggregate report: `/private/tmp/upkg39_formal_20260622/release/report.json`
- gate report: `/private/tmp/upkg39_formal_20260622/release/upkg39_strict_closure/upkg39_report.json`
- summary: `/private/tmp/upkg39_formal_20260622/release/upkg39_strict_closure/summary.json`
- completion manifest: `/private/tmp/upkg39_formal_20260622/release/upkg39_strict_closure/upkg39_completion_manifest.json`
