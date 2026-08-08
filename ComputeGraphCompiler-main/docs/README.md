# Docs Index

- 跨根项目、`ComputeGraphCompiler-main/` 与 `CGC_Release/` 的最终目录规范，请同时参考根目录 `DIRECTORY_POLICY.md`

## technical_whitepapers
- 目录名称：`ComputeGraphCompiler-main/docs/technical_whitepapers/`
- 收录内容：当前正式技术主文档、M1/CLI 基座、Edge runtime、训练与 Psi-Zero 技术白皮书，以及少量历史术语迁移说明
- 代表文件：`CGC_M1_8STEP_CLI_TECH_WHITEPAPER_v1.0_zh_CN.md`、`CGC_Edge_Engine_Whitepaper_v1.0.md`、`CGC_Unified_Pipeline_Kernel_Design_v1.0_zh_CN.md`
- `UPKG 3.x` 统一产品链路技术补充规范：`CGC_UPKG_3_X_UNIFIED_PRODUCT_CHAIN_TECHNICAL_SPEC_v0.2_zh_CN.md`
- `UPKG 2.1` 的 `DFlash / SGLang / Spec V2` 路由比较说明：`CGC_UPKG_2_1_DFLASH_SGLANG_SPEC_V2_COMPARISON_zh_CN.md`
- 说明：该技术补充规范现已正式补入 `UPKG 3.9 strict closure` 的定义、artifact materialization、schema validation 与验收口径
- 说明：`CGC_Edge_Engine_Whitepaper_v1.0.md` 已吸收仍有效的 `DeltaMem` / state transport / zero-copy handoff 内容；两份 `CGC_DeltaMem_Architecture_Whitepaper_v1.0*.md` 现在仅保留历史术语与迁移说明，不再作为现行主契约引用
- 说明：`CGC_Gate_2.0` 现已统一采用 `done / proof / target` 口径；其中 `UPKG 2.0`、`UPKG 3.x` 与 `m7.6 dev gate` 在当前正式说法中属于 `proof` 承接层，而不是 `Gate 2.0` 核心 layer-adaptive 目标已完成
- 说明：读取 `technical_whitepapers/` 时可直接采用对称语义: 主目录 = 正式主入口，`archive/` = 历史稿 / proof 支撑 / 非主入口

## gate_whitepapers
- 目录名称：`ComputeGraphCompiler-main/docs/gate_whitepapers/`
- 收录内容：当前有效的 gate 白皮书、runtime evidence 白皮书与硬件验收说明
- 代表文件：`CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md`、`CGC_UPKG_2_0_MODEL_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md`、`CGC_UPKG_3_0_AGENT_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md`、`CGC_M7_INDUSTRIAL_BASE_WHITEPAPER_v1.0_zh_CN.md`、`CGC_EDGE_CLOUD_PROTOCOL_MANDATORY_GATE_WHITEPAPER_v1.0_zh_CN.md`
- 说明：当前 `UPKG` 正式分层已收口为 `UPKG 1.0 = kernel`、`UPKG 2.0 = model`、`UPKG 3.0 = agent`、`UPKG 4.0 = embodied`；其中现有 `CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md` 继续承接 `kernel` 范围内的 `M1-M7.5` gate 口径，`CGC_UPKG_2_0_MODEL_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md` 负责承接 `M6 + M8 + UPKG_1.x + UPKG 3.x audit/replay/trace` 的 `model productization gate`，`CGC_UPKG_3_0_AGENT_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md` 负责 `agent product gate`；新增 `CGC_EDGE_CLOUD_PROTOCOL_MANDATORY_GATE_WHITEPAPER_v1.0_zh_CN.md` 统一定义 `trueorthokda + cq4 + zero-copy VRAM` 的强制协议 gate 口径；当前 `UPKG 3.8 / m78` 仍保留独立 `alignment_score >= 0.5` 临时放宽门槛，而 `UPKG 3.9 / upkg39` 已作为 strict closure gate 恢复 `0.8` 对齐验收并 materialize `schema bundle / validator execution / tensorized graph-native closure`；可执行的 gate YAML 设定继续与程序同放于 `ComputeGraphCompiler-main/cgc_engine/agent/eval/`
- 说明：从 `CGC_Gate_2.0` 视角看，`UPKG 2.0` 与 `UPKG 3.x` 现在都已被正式纳入 `proof` 层，分别对应 `model product gate` 与 `agent product chain` 的上层承接关系；它们证明 `Gate 2.0` 可被产品链消费，但不等于 `max_local_layer / finished_layer + 1 / hidden_states + partial_kv` 这些 2.0 核心目标已经闭环

### UPKG 分层对照表

| 分层 | 当前定位 | 主要对应要求 | 当前主文档 |
| --- | --- | --- | --- |
| `UPKG 1.0` | `kernel` 分层定义 | 证明统一内核成立，回答 “统一 kernel 是否成立、边界是什么” | `CGC_Unified_Pipeline_Kernel_Design_v1.0_zh_CN.md` |
| `UPKG 1.1` | `kernel` 当前正式工程验收口径 | 覆盖 `M1-M7.5`，要求统一 `8-step`、单一真源 `report.json`、distributed/runtime/bridge/runtime evidence 全部可归因 | `CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md` |
| `UPKG 2.0` | `model` 产品化 gate | 要求模型可被发现、打包、运行、验证、审计、回放，并正式消费 `UPKG 1.x` contract artifacts | `CGC_UPKG_2_0_MODEL_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md` |
| `UPKG 3.0` | `agent` 产品化 gate | 要求 `agent + edge + runtime` 形成统一 artifact、summary、failure attribution，并完成可审计 agent 产品链 | `CGC_UPKG_3_0_AGENT_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md` |
| `UPKG 4.0` | `embodied` 产品化 gate | 要求 `psi0 / realtime-vla / embodied audit / replay / trace` 进入正式具身链路与交付口径 | `CGC_UPKG40_EMBODIED_PSI0_REALTIMEVLA_AUDIT_REPLAY_TRACE_zh_TW.md` |

### CGC_Gate 2.0 交叉读取

- `CGC_Gate_2.0`
  - 当前正式状态模型为 `done / proof / target`
- `done`
  - 表示已验证 foundation，可直接正式宣称
- `proof`
  - 表示已有正式 gate / 白皮书 / 实跑证据可承接 `Gate 2.0`
- `target`
  - 表示仍属于 `Gate 2.0` 核心 layer-adaptive 目标，尚未完成

当前与 `UPKG` 的交叉关系是：

- `UPKG 2.0`
  - 在 `CGC_Gate_2.0` 中属于 `proof`
  - 负责承接 `model productization`
- `UPKG 3.x`
  - 在 `CGC_Gate_2.0` 中属于 `proof`
  - 负责承接 `agent product chain / summary / attribution / replay`
- `m7.6 dev gate`
  - 在 `CGC_Gate_2.0` 中也属于 `proof`
  - 但它只是 `m76` 的 developer-facing 解释，不是新的独立 CLI gate

### 读取建议

- 想看上位版本边界与四层分工：先读 `CGC_Unified_Pipeline_Kernel_Design_v1.0_zh_CN.md`
- 想看 `kernel` 当前到底验什么：读 `CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md`
- 想看 `model / agent / embodied` 各自产品化要求：分别读 `UPKG 2.0 / 3.0 / 4.0` 对应文档
- 想看 `UPKG 2.0 / 3.x` 为什么会在 `CGC_Gate_2.0` 中被归到 `proof`，以及哪些仍是 2.0 的 `target`：补读 `technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/`
- 想看 `UPKG 2.1` 如何把 `M7.4` 纳入 `DFlash / SGLang` 路由验收：补读 `CGC_UPKG_2_1_DFLASH_SGLANG_SPEC_V2_COMPARISON_zh_CN.md`
- 想看 `UPKG 3.x` 如何从 `DAG import / GUI teaching / train / infer / audit / replay / trace` 形成统一技术方案，以及 `UPKG 3.9 strict closure` 如何正式收口：补读 `CGC_UPKG_3_X_UNIFIED_PRODUCT_CHAIN_TECHNICAL_SPEC_v0.2_zh_CN.md`
- 想看端云协议强制口径：补读 `CGC_EDGE_CLOUD_PROTOCOL_MANDATORY_GATE_WHITEPAPER_v1.0_zh_CN.md`

## release_notes
- 目录名称：`ComputeGraphCompiler-main/docs/release_notes/`
- 收录内容：重要里程碑或正式验收通过后的版本发布说明
- 当前文件：`UPKG39_STRICT_CLOSURE_RELEASE_NOTE_20260622_zh_CN.md`
- 说明：本目录用于沉淀正式 PASS 后的变更摘要、根因说明、修正口径与 artifact 入口，避免白皮书承担过多版本公告职责

## product_manuals
- 目录名称：`ComputeGraphCompiler-main/docs/product_manuals/`
- 收录内容：偏操作手册、runbook、CLI/产品链路的执行说明
- 代表文件：`CGC_AGENT_PRODUCT_USER_MANUAL_zh_CN.md`、`CGC_AGENT_PRODUCT_USER_MANUAL_zh_TW.md`、`CGC_PSI0_CLOUD_TO_EDGE_REALTIMEVLA_RUNBOOK_zh_TW.md`、`CGC_UPKG40_EMBODIED_PSI0_REALTIMEVLA_AUDIT_REPLAY_TRACE_zh_TW.md`
- 说明：当需求是“如何实际执行一条链路”，优先查看本目录；其中 `CGC_PSI0_CLOUD_TO_EDGE_REALTIMEVLA_RUNBOOK_zh_TW.md` 固化了当前 `hostb psi0 -> edge realtimevla` 的可重跑流程，`CGC_UPKG40_EMBODIED_PSI0_REALTIMEVLA_AUDIT_REPLAY_TRACE_zh_TW.md` 進一步定義了 `UPKG 4.0 embodied` 的 contract / session / audit / replay / trace artifact 集

## archive
- 目录名称：`ComputeGraphCompiler-main/docs/archive/`
- 收录内容：与当前目录结构或现况偏差较大的旧稿、重复稿与营销型文档
- 代表文件：`CGC Universe 端云协同AI开源社区｜完整生态方案+官方文档套装0602.md`
- 说明：若文件名仍保留在 `technical_whitepapers/` 目录下但正文已改为“归档说明”，其用途也视同 archive，不应再作为当前正式契约引用
