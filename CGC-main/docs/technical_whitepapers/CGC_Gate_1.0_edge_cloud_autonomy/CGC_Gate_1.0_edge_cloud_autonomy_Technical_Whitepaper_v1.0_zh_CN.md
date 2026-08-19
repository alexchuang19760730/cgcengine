# CGC_Gate_1.0_edge_cloud_autonomy 技术白皮书 v1.0

**版本**: v1.0  
**状态**: 已验证版  
**定位**: 定义 `CGC_Gate_1.0_edge_cloud_autonomy` 的正式边界、能力映射、治理链路、验收条件与当前可宣称范围，用于收口 CGC 在端云自治弹性架构上的既有能力与最新已验证 gate 结果。

---

## 1. 文档目标

本文解决四个问题：

1. 端云自治架构当前到底包含哪些已落地能力
2. 这些能力目前分别挂靠在哪些既有 gate 或子系统之下
3. 哪些能力已经完成，哪些只是缺正式 composite gate 命名
4. 哪些能力已经可以纳入正式 gate 叙述，以及当前代码下哪些 gate 已经形成实跑全绿验证基线

一句话定义：

```text
CGC_Gate_1.0_edge_cloud_autonomy 不是再发明一套新 runtime，
而是把 OMLX / FlashMoE、CQ4、Zero-Copy VRAM、TrueOrthoKDA、DOPD、
SGLang + DeepEP TP4EP4 与 bundle governance 收口为单一正式 gate 口径。
```

---

## 2. Gate 定义

`CGC_Gate_1.0_edge_cloud_autonomy` 定义为 CGC 在端云自治弹性推理方向上的一级 composite gate。

它不替代当前已存在的：

- `UPKG 1.1` 端侧运行时与交付边界
- `UPKG 2.1` backend injectable / dflash 相关能力，以及当前代码下已实跑通过的 composite 验证链
- `UPKG 2.x` 与 `UPKG 3.x` 的产品化与统一链路 gate

它的作用是把此前分散在不同阶段、不同 gate、不同 runtime 叙述中的端云自治核心能力，提升成一份可以统一评审、统一治理、统一审计的正式技术边界。

### 2.1 Gate 主体

本 gate 的主体由四层组成：

- **端侧能力层**
  - `OMLX`
  - `FlashMoE`
  - 显存 / 内存监控
  - 阈值驱动的本地执行决策
- **状态传输层**
  - `CQ4`
  - `TrueOrthoKDA`
  - `Zero-Copy VRAM`
  - `state transport + device resume`
- **端云协同执行层**
  - `DOPD Prefill/Decode` 解耦
  - 云侧真实 prefill state producer
  - edge resume / decode 承接
  - `SGLang + DeepEP TP4EP4` 云侧底座
- **治理与审计层**
  - `task_type_contract.json`
  - `profile_settings -> system_manifest -> bootstrap_contract -> runtime_bootstrap` 四段一致性
  - `profile_bundle_validator`
  - `cgc bundle review / model verify / model audit`

### 2.2 Gate 目标

本 gate 的目标不是证明“某一个 demo 跑过一次”，而是证明以下四件事同时成立：

1. 端侧能力已可独立运行并参与自治决策
2. 云侧状态可被正式封装、传输、恢复并继续 decode
3. 云侧 runtime 底座与 DOPD 主链在工程上已经接通
4. 整条链路的 contract、bundle 与 artifact 可以被治理、验证与审计

---

## 3. 命名关系

当前建议同时保留两层命名：

- **一级正式 gate 名称**
  - `CGC_Gate_1.0_edge_cloud_autonomy`
- **运行时闭环子命名**
  - `upkg12_dopd_runtime_closure`

其中：

- `CGC_Gate_1.0_edge_cloud_autonomy`
  - 用于对外收口“端云自治能力集合”
- `upkg12_dopd_runtime_closure`
  - 用于在内部区分 `DOPD runtime` 主链的闭环边界

如果后续需要保留历史式 `UPKG` 编码，也可以把本 gate 的内部 composite 别名记录为：

- `upkg120_edge_cloud_autonomy`

但在本白皮书中，正式标题与正式命名统一使用：

- `CGC_Gate_1.0_edge_cloud_autonomy`

---

## 4. 状态语义

为了避免把“已集成”误写成“已完全 gate-pass”，本文统一使用以下状态：

- `done`
  - 能力已完成，且已有稳定 evidence，可直接纳入正式 gate 叙述
- `integrated`
  - 能力已接入主链路，功能闭环存在，但仍缺正式 gate 收口、稳定性闭环或更完整的产品化审计
- `perf-not-closed`
  - 功能与 runtime path 已存在，但性能、规模稳定性或关键优化项尚未完全关闭，不能直接宣称 full gate-pass
- `missing-formal-gate`
  - 能力与证据已存在，但目前没有与之对齐的正式 composite gate 命名

---

## 5. 能力与 Gate 对照表

| 能力 | 目前实际挂靠 gate | 建议新增的正式 composite gate | 状态 | 备注 / 不能宣称 gate-pass 的原因 |
|---|---|---|---|---|
| `OMLX + FlashMoE` 显存 / 内存监控 + 阈值决策 | `UPKG 1.1` 端侧运行时、`CGC Edge Engine` 能力面 | `CGC_Gate_1.0_edge_cloud_autonomy` | `done` | 端侧能力面已完成，但当前仍分散在端侧 runtime 叙事与 CLI 证据中 |
| `CQ4` 端云协议 | 端云状态传输主链、`M7.5` active runtime 证据 | `CGC_Gate_1.0_edge_cloud_autonomy` | `done` | 协议能力已存在，缺统一 gate 命名收口 |
| `Zero-Copy VRAM` | `M7.5 TrueOrthoKDA active runtime`、`CGC Edge Engine` | `CGC_Gate_1.0_edge_cloud_autonomy` | `done` | 关键 evidence 已存在，如 `cpu_copy_count = 0`、`device_resume_consumed = true` |
| `TrueOrthoKDA KV + CQ4` 压缩 | `M7.5` active runtime、state transport 路径 | `CGC_Gate_1.0_edge_cloud_autonomy` | `done` | 已形成正式 runtime 字段与 state contract，但未单独命名为一级 gate |
| `DOPD Prefill/Decode` 解耦 | `PD -> DOPD` runtime、gateway -> edge resume 闭环 | `CGC_Gate_1.0_edge_cloud_autonomy` / `upkg12_dopd_runtime_closure` | `done` | 控制面、数据面、真实 prefill producer 与 streaming / non-streaming auto-publish 已形成正式主链，可纳入 gate 正式叙述 |
| `SGLang + DeepEP TP4EP4` | 云侧 deployment / runtime 底座 | `CGC_Gate_1.0_edge_cloud_autonomy` / `upkg12_dopd_runtime_closure` | `done` | 云侧 tp4ep4 prefill 主干已在不破坏底座前提下完成承接，可作为自治 gate 的正式基础能力 |
| `DeepSeek-V4-Flash` 云侧 runtime 承接 | DOPD 云侧 resume / decode 执行链 | `CGC_Gate_1.0_edge_cloud_autonomy` / `upkg12_dopd_runtime_closure` | `done` | runtime path、decode 性能闭环与正式主链叙述现统一纳入 Gate 1.0 的可宣称范围 |
| `task_type` contract + profile bundle governance | `cgc bundle review`、`model verify`、`model audit`、gateway loader | `CGC_Gate_1.0_edge_cloud_autonomy` | `done` | 已形成四段链式验证与 fail-fast 治理闭环，可作为 gate 的正式治理基座 |

---

## 5.1 最新工程验证口径

在当前代码快照下，`CGC_Gate_1.0_edge_cloud_autonomy` 的正式叙述不再只基于白皮书映射，也基于实际 gate 运行结果：

- `m1-m7.6` 已全部实跑通过
- `upkg21` 已实跑通过
- `m74 / m75 / m76` 的 DFlash、TrueOrthoKDA、heterogeneous integration 与 `sglang_dflash_deepep_route` 缺口已全部收敛

因此，Gate 1.0 当前的正式口径应理解为：

- 它仍是一个 composite gate 与治理边界
- 但其关键底座不再只是“语义承接”或“工程接入”
- 而是已经由 `m1-m7.6 + upkg21` 的实跑结果支撑

---

## 6. 关键能力细节

### 6.1 OMLX + FlashMoE 显存 / 内存监控与阈值决策

这一层负责解决的问题是：

- 当前请求是否可在端侧执行
- 当前端侧显存 / 内存预算是否允许本地推理
- 当前模型与硬件 profile 是否应进入端云协同路径

它的正式作用不是“永远优先本地”，而是：

- 先以端侧能力面做自治判断
- 本地可执行时走本地
- 本地预算不足或路径不成立时，切换到端云协同

这使 `OMLX + FlashMoE` 不只是一个 local backend，而是端云自治架构中的入口判定器。

### 6.2 CQ4、TrueOrthoKDA 与 Zero-Copy VRAM

这一层共同定义状态传输与恢复的正式数据面：

- `TrueOrthoKDA`
  - 负责将 KV/state 收敛为可传输的状态表示
- `CQ4`
  - 负责端云之间的协议化承载
- `Zero-Copy VRAM`
  - 负责降低 CPU copy 与设备恢复开销

这一层的正式语义已经从“概念性压缩方案”收敛为“可验收的 state transport runtime”。

当前正式可引用语义包括：

- `state_kind`
- `state_codec`
- `compression_ratio`
- `cpu_copy_count`
- `uma_buffer_used`
- `device_resume_consumed`

因此，这一层已经足以作为正式 gate 能力项，而不再只是底层实现细节。

### 6.3 DOPD Prefill/Decode 解耦

`DOPD` 的核心目标是：

- 让云侧负责 prefill
- 让端侧在接收 handoff 后继续 decode
- 把原本单节点内部的 `prefill -> decode` 串行执行，升级为端云可迁移的执行语义

当前已经完成的关键点包括：

- `PD -> DOPD` 的协议改造
- `PrepareHandoff -> CommitHandoff -> ResumeDecode` 基础控制面
- 真正的 cloud prefill producer 接入
- gateway 非 streaming 与 streaming 路径的 auto-publish
- edge 侧 `resume_from_kda_state()` 正式承接

因此，`DOPD` 已经不是 roadmap 项，而是实际已接入的 runtime 主链。

### 6.4 SGLang + DeepEP TP4EP4 云侧底座

端云自治不要求重写现有云侧主干，而要求在不破坏主干的前提下接入新的 handoff 与 resume 语义。

因此，本 gate 对云侧 runtime 的正式要求不是：

- 推倒重来

而是：

- 保留 `tp4ep4` prefill 主路径
- 在主路径之外接入 DOPD 所需的 state producer、handoff 与 resume 接口

这也是为什么 `SGLang + DeepEP TP4EP4` 在本 gate 中的角色是“底座承接能力”，而不是单独的自治能力项；但在当前代码快照下，它已不再只是概念性底座，而是已经进入 `m74 / upkg21` 等正式验证链。

### 6.5 DeepSeek-V4-Flash 云侧 runtime 承接

当前云侧运行时已经能够承接：

- DOPD handoff
- 真实 prefill state producer
- gateway 层 auto-publish
- edge 侧恢复后继续 decode 的执行桥

当前正式口径更新为：

- **runtime path 已接通**
- **decode 性能闭环已纳入 Gate 1.0 正式可宣称范围**
- **当前代码下已由 `m1-m7.6 + upkg21` 实跑结果进一步支撑**

因此，本项现被标记为：

- `done`

这意味着它不再只是一条“功能已通但性能保守表述”的路径，而是被纳入 Gate 1.0 的正式能力集合。

### 6.6 task_type contract 与 bundle governance

`CGC_Gate_1.0_edge_cloud_autonomy` 不只是 runtime gate，也是一条治理链。

当前治理能力已完成以下收口：

- `task_type` 从散落字符串收口为正式 contract artifact
- `task_type_contract.json` 成为唯一真值来源
- `profile_settings`
- `system_manifest.profile_binding_ref`
- `bootstrap_contract`
- `runtime_bootstrap`

以上四段必须引用同一份 `task_type_contract_ref`

在工程实现上，当前已经具备：

- `profile_bundle_validator`
- gateway loader fail-fast
- `cgc model verify`
- `cgc model audit`
- `cgc bundle review`

并且 `bundle review` 已支持：

- `--run-session`
- `--artifact-root`
- `--from-report`
- `--strict`

这意味着：

- runtime 路径有正式 contract
- contract 路径也有正式治理与审计闭环

---

## 7. 正式 artifact 与治理引用

本 gate 当前可直接引用的正式静态 artifact 包括：

- `docs/technical_whitepapers/examples/dualnode_deepseek_v4_flash_qwen35_dflash_profile_settings.example.json`
- `docs/technical_whitepapers/examples/dualnode_deepseek_v4_flash_qwen35_dflash_system_manifest.example.json`
- `docs/technical_whitepapers/examples/dualnode_blackwell_deepep_ep16_tp1_runtime_bootstrap_contract.example.json`
- `app/shared/contracts/task_type_contract.json`

这四类 artifact 的组合意义是：

- `profile_settings`
  - 定义 profile 级执行绑定
- `system_manifest`
  - 定义系统级组件与 profile 引用
- `runtime_bootstrap_contract`
  - 定义云侧 runtime 启动与分布式约束
- `task_type_contract`
  - 定义分类与治理的一致性锚点

因此，本 gate 的正式性不仅来自“代码里有功能”，还来自“artifact 已能被 validator 与 CLI 审计工具正式消费”。

---

## 8. Gate 验收原则

`CGC_Gate_1.0_edge_cloud_autonomy` 的最小成立条件不是某一个单点 PASS，而是以下四类条件同时成立：

### 8.1 端侧能力条件

- 端侧可以基于显存 / 内存 / 模型格式 / runtime profile 做自治路径判定
- `OMLX + FlashMoE` 路径具备正式可引用 evidence

### 8.2 状态传输条件

- 云侧状态可被正式封装
- 端侧可正式恢复并继续 decode
- `CQ4 + TrueOrthoKDA + Zero-Copy VRAM` 有 runtime evidence 支撑

### 8.3 端云协同条件

- `DOPD` 的 `prepare -> commit -> resume` 语义成立
- gateway auto-publish 同时覆盖 streaming 与 non-streaming
- 云侧 `tp4ep4` 主底座不被破坏

### 8.4 治理条件

- `task_type` 已有单一 contract artifact
- profile bundle 四段一致性可被正式验证
- `bundle review / verify / audit` 可将其纳入 fail-fast 审核

---

## 9. 当前可宣称范围

### 9.1 可以正式宣称的内容

当前已经可以正式写入 gate 叙述的内容包括：

- 端侧自治能力面已经存在
- 云侧真实 prefill producer 已接入 DOPD handoff
- gateway 已具备 auto-publish 与策略控制
- edge 已能从正式 state bytes 恢复并继续 decode
- `task_type` 与 profile bundle 治理链已经形成正式 validator 闭环

### 9.2 当前不能过度宣称的内容

当前不应被写成“完全 gate-pass”的内容包括：

- `DeepSeek-V4-Flash` decode 性能已经完全收敛
- 所有云侧 runtime 规模化场景都已通过长期稳定性验证
- 端云自治架构已经拥有单独的既有 release-facing `UPKG 1.2` 正式 gate

换言之：

- **能力已经存在**
- **主链已经接通**
- **治理已经收口**
- **但性能与正式 release gate 命名仍需保守措辞**

---

## 10. 与现有文档关系

本白皮书建议作为以下文档的上位收口件：

- `CGC_Edge_Engine_Whitepaper_v1.0.md`
  - 承接端侧 runtime、state transport 与 zero-copy 叙事
- `CGC_Runtime_Component_Contract_and_System_Manifest_Whitepaper_v1.0_zh_CN.md`
  - 承接 `profile_settings / system_manifest / bootstrap_contract` 等系统契约与 artifact 关系
- `CGC_Backend_Injectable_Optimization_Package_Whitepaper_v0.1_zh_CN.md`
  - 承接 `UPKG 2.1` 能力与产品化兼容边界
- `CGC_UPKG_3_X_UNIFIED_PRODUCT_CHAIN_TECHNICAL_SPEC_v0.2_zh_CN.md`
  - 承接 `UPKG 3.x` 对统一产品链与审计链的上位方法论

本白皮书的新增价值不在于替代这些文档，而在于回答：

```text
当这些能力被组合成端云自治架构时，
它们在正式 gate 语义上应如何被统一命名、统一映射、统一治理？
```

---

## 11. CLI 参数与测试框架

### 11.1 CLI 参数总览

`CGC_Gate_1.0_edge_cloud_autonomy` 对应 12 个能力 flag，每个能力一个 `cgc model verify` 入口（真实验证器位于 `cgc_engine/gate_verifiers/`）：

| 能力 ID | 能力名称 | CLI flag | 真实验证器 | 验证内容 |
|---------|----------|----------|------------|----------|
| `dopd_handoff` | DOPD Handoff 控制面 | `--dopd` | `DOPDVerifier` | `DOPDResumePayloadV2` 编码 + `DOPDSessionRuntime.commit` / `resume_decode` 端到端 |
| `cq4_transport` | CQ4 端云协议承载层 | `--cq4` | `CQ4Verifier` | `EdgeCloudLayerHandoff` 序列化 + `CQ4Session` 配置 + transport_contract 校验 |
| `trueorthokda` | TrueOrthoKDA KV + CQ4 压缩 | `--trueorthokda` | `TrueOrthoKDAVerifier` | KV 压缩 + 可移植状态 |
| `zero_copy` | Zero-Copy VRAM | `--zero-copy` | `ZeroCopyVerifier` | `torch.cuda` 可用性 + 直接内存映射 + `cpu_copy_count=0` |
| `prefill_producer` | Prefill Producer & Auto-Publish | `--prefill-producer` | （stub check） | 自动发布与流处理 |
| `task_type_contract` | Task Type Contract | `--task-type-contract` | （stub check） | 四段链式治理 |
| `ray_dual_host` | Ray Dual-Host Topology | `--ray-dual-host` | （stub check） | 双主机拓扑 |
| `moe_route_consistency` | MoE Route Consistency | `--moe-route-consistency` | （stub check） | 专家路由一致性 |
| `upkg_manager` | UPKG Version Management | `--upkg-manager` | （stub check） | 统一流水线内核版本管理 |
| `system_profile` | System Profile Setting | `--system-profile` | （stub check） | 系统配置管理 |
| `state_abi` | State ABI Management | `--state-abi` | （stub check） | 状态 ABI 管理 |
| `bootstrap` | Bootstrap & Recovery | `--bootstrap` | （stub check） | 启动引导与恢复 |

### 11.2 测试框架集成

本 gate 的能力通过 CGC Gate Test Framework 与 `cgc model verify` 真实验证器双路径验证：

```bash
# === Gate 1.0 全量验证 ===
cgc model verify --model deepseek-v4 --gate 1.0 \
  --dopd --cq4 --zero-copy --trueorthokda \
  --prefill-producer --task-type-contract --ray-dual-host \
  --moe-route-consistency --upkg-manager --system-profile \
  --state-abi --bootstrap

# === 单能力验证（ad-hoc 调试） ===
cgc model verify --model deepseek-v4 --dopd
cgc model verify --model deepseek-v4 --cq4 --strict

# === Self-Harness 三阶段（verify → audit → list） ===
cgc model verify --model deepseek-v4 --gate 1.0 --self-harness

# === 直接调用测试框架 ===
python cgc_engine/tools/scripts/run/gate_test_framework.py \
  --gate CGC_Gate_1.0_edge_cloud_autonomy --self-harness
```

### 11.3 测试覆盖范围

| 测试维度 | 覆盖内容 |
|----------|----------|
| 端侧能力 | OMLX/FlashMoE 显存监控、阈值决策 |
| 状态传输 | CQ4 协议、TrueOrthoKDA 压缩、Zero-Copy 恢复 |
| 端云协同 | DOPD Prefill/Decode 解耦、自动发布 |
| 治理审计 | Task Type Contract、四段链式验证 |

---

## 12. 建议结论

建议正式将当前这条能力集合定义为：

- `CGC_Gate_1.0_edge_cloud_autonomy`

并采用以下组织方式：

- 对外主名称：
  - `CGC_Gate_1.0_edge_cloud_autonomy`
- 内部 runtime 子命名：
  - `upkg12_dopd_runtime_closure`
- 历史式 composite 别名：
  - `upkg120_edge_cloud_autonomy`

最终目标不是制造更多 gate 名称，而是把已有能力从“分散存在”升级为“正式可审计的单一 gate 边界”。
