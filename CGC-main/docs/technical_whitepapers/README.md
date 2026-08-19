# Technical Whitepapers 目录说明

本目录用于存放 ComputeGraphCompiler 的技术白皮书、正式 gate 文档与可直接引用的示例 artifact。

当前主目录中的 `CGC_Gate_1.0` / `CGC_Gate_2.0` 资产，已同步到最新验证口径：

- `CGC_Gate_1.0` 对应的 `m1-m7.6 + upkg21` 当前代码实跑结果已全绿
- `CGC_Gate_2.0` 则据此更新为 `done / proof / target` 三段口径
- 其中 `UPKG 2.0`、`UPKG 3.x` 与 `m7.6 dev gate` 已明确写入 `proof` 承接层
- `max_local_layer / finished_layer + 1 / hidden_states + partial_kv` 等 layer-adaptive continuation 核心能力仍保持 `target`

当前目录采用三层语义划分：

- 主目录：当前有效的正式 gate 版本文件夹与少量仍需直接引用的核心 JSON artifact
- `archive/`：历史白皮书、旧版本说明、补充附录与不再作为当前主入口的 Markdown 文档
- `examples/`：可被白皮书、bundle review、runtime bootstrap、审计流程直接引用的 JSON 示例

## 对称读取语义

为避免主目录与 `archive/` 被混读，当前推荐直接采用以下对称语义：

- 主目录
  - `正式主入口`
  - 当前有效的白皮书、Gate 目录与 machine-consumable artifact，应优先从这里读取
- `archive/`
  - `历史稿 / proof 支撑 / 非主入口`
  - 仍可提供历史背景、比较材料与证据支撑，但不应替代主目录中的正式入口

换句话说：

- 想看当前生效的正式定义、状态口径与可消费 artifact：
  - 先看主目录
- 想看历史分叉、选型过程、补充比较与 proof 背景：
  - 再看 `archive/`

这也对应当前 `CGC_Gate_2.0 = done / proof / target` 的读取方式：

- `done / proof / target` 的正式定义、能力矩阵与 claim boundary
  - 以主目录中的 `CGC_Gate_2.0` 资产为准
- 与 `UPKG 2.1 / DFlash / SGLang / DeepEP` 相关的历史比较稿
  - 可作为 `proof` 支撑材料读取
  - 但不应被误读成 `Gate 2.0` 最新主白皮书或 `target` 已完成的正式宣称

## 主目录放什么

主目录保留“当前有效、需要被直接消费”的正式资产，尤其是 `CGC_Gate*` 系列版本目录：

- 当前正式 gate 的版本文件夹
- 每个版本文件夹下的主文档
- `gate_map / checkin / summary` 等配套 JSON artifact
- 少量仍作为 schema 或 contract 入口的 JSON

当前主目录的典型结构包括：

- `CGC_Gate_1.0_edge_cloud_autonomy/`
- `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/`
- `CGC_FusionRoute_Final_Topology_Matrix_Technical_Whitepaper_v1.0_zh_CN.md`
- `CGC_FusionRoute_v2_Static_Contract_Technical_Whitepaper_v1.0_zh_CN.md`
- `CGC_Perception_Matrix_LLM_Technical_Whitepaper_v1.0_zh_CN.md`
- `CGC_System_Profile_Schema_v0.1.json`

保留这些版本目录在主目录的原因是：

- 它们属于当前有效的正式入口，而不是历史沉淀
- CLI、release checkin、dashboard/report 等流程可能直接引用这些目录中的文档与 JSON artifact
- 让“当前正式资产”在目录层面保持最高可见性
- 让同一代 gate 的白皮书与配套 artifact 保持同目录聚合，避免顶层平铺
- 对于跨 `Gate 4.0 / 5.0 / 6.0` 的独立架构真源，可在主目录保留单份总白皮书，避免强行归入单一 gate 目录

### 当前 Gate 目录

- `CGC_Gate_1.0_edge_cloud_autonomy/`
  - `CGC_Gate_1.0_edge_cloud_autonomy_Technical_Whitepaper_v1.0_zh_CN.md`
  - `CGC_Gate_1.0_edge_cloud_autonomy_gate_map.json`
  - `CGC_Gate_1.0_edge_cloud_autonomy_checkin.example.json`
  - `CGC_Gate_1.0_edge_cloud_autonomy_summary.example.json`
- `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/`
  - `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_Technical_Whitepaper_v1.0_zh_CN.md`
  - `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_gate_map.json`
  - `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_checkin.example.json`
  - `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_summary.example.json`
- `CGC_Gate_3.0_train_inference_unification/`
  - `CGC_Gate_3.0_train_inference_unification_Technical_Whitepaper_v1.0_zh_CN.md`
  - `CGC_Gate_3.0_train_inference_unification_gate_map.json`
  - `CGC_Gate_3.0_train_inference_unification_checkin.example.json`
  - `CGC_Gate_3.0_train_inference_unification_summary.example.json`
  - 状态：`draft_skeleton`（Megatrain/MLX-Tune 代码已落地，待 gate run 升级为 `validated`）
- `CGC_FusionRoute_Final_Topology_Matrix_Technical_Whitepaper_v1.0_zh_CN.md`
  - 跨 `Gate 4.0 / 5.0 / 6.0` 的 FusionRoute 最终拓扑矩阵总白皮书
  - 定义 `Gate -> Plane -> Primary Role -> Secondary Role -> CLI -> Verifier` 草案
- `CGC_FusionRoute_v2_Static_Contract_Technical_Whitepaper_v1.0_zh_CN.md`
  - FusionRoute v2 静态契约总白皮书
  - 定义 `TaskType -> GateDomain -> PrimaryRole -> SecondaryRole` 与相关 schema / draft contract
- `CGC_Perception_Matrix_LLM_Technical_Whitepaper_v1.0_zh_CN.md`
  - Perception Matrix + LLM 总白皮书
  - 定义感知矩阵如何作为 FusionRoute v2 上游决策层，并受 system profile / profile binding / bootstrap / state ABI / topology contract 约束

## `archive/` 放什么

`archive/` 用于存放历史技术白皮书与补充材料，包括但不限于：

- 旧版技术白皮书
- ABI / Runtime / Manifest 说明
- UPKG 技术规格与逻辑说明
- 各类附录、定位文档、历史设计说明

如果一份 Markdown 文档不再是当前正式 gate 主入口，但仍有保留价值，建议放入 `archive/`。

更具体的归档规则见：

- [archive/README.md](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers/archive/README.md)

## `examples/` 放什么

`examples/` 用于存放可直接被系统或审计流程消费的示例 JSON，主要服务于：

- profile settings
- system manifest
- runtime bootstrap contract
- model contract
- benchmark / deployment 示例

这些文件的目标不是叙述性说明，而是“可被引用、可被校验、可进入治理链”的示例 artifact，因此不放入 `archive/`。

## 目录治理约定

后续新增文档时，建议遵循以下规则：

- 如果是当前正式 gate 主文档或其直接配套 artifact：
  - 在主目录下创建对应的 `CGC_Gate_*` 版本文件夹
  - 将白皮书与 `gate_map / checkin / summary` 一起放入该文件夹
  - 优先采用 `CGC_Gate_*` 一致命名
- 如果是历史版本、旧方案、附录、说明性材料：
  - 放入 `archive/`
- 如果是可被 runtime / bundle governance / whitepaper 直接引用的示例 JSON：
  - 放入 `examples/`

## 一句话区分

- 主目录：当前正式 gate 入口与版本目录
- `archive/`：历史与沉淀
- `examples/`：可直接消费的示例 artifact
