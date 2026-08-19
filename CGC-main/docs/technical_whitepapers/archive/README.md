# Archive 说明

本目录用于存放 `docs/technical_whitepapers` 下已归档的历史技术白皮书与补充说明文档。

## 归档范围

当前归档规则采用保守口径，仅移动 top-level 的非 `CGC_Gate*` Markdown 白皮书：

- 归档对象：`docs/technical_whitepapers/*.md`
- 保留对象：文件名以 `CGC_Gate` 开头的主文档
- 不移动对象：
  - `examples/` 目录下的 JSON 示例
  - `CGC_Gate*.json` 配套 artifact
  - 非白皮书用途的 schema / contract JSON

## 为什么保留 `CGC_Gate` 在主目录

`CGC_Gate` 系列文件属于当前有效的正式 gate 资产，而不是纯历史资料，因此保留在 `docs/technical_whitepapers` 主目录：

- `CGC_Gate_1.0_edge_cloud_autonomy_Technical_Whitepaper_v1.0_zh_CN.md`
- `CGC_Gate_1.0_edge_cloud_autonomy_gate_map.json`
- `CGC_Gate_1.0_edge_cloud_autonomy_checkin.example.json`
- `CGC_Gate_1.0_edge_cloud_autonomy_summary.example.json`

这样做的目的有三点：

- 让当前生效的 gate 主文档与配套 artifact 保持同层可见
- 避免 release checkin、CLI summary、dashboard/report 等消费路径再去 archive 中查找当前资产
- 将“当前正式 gate”与“历史白皮书沉淀”在目录语义上明确分离

## 当前归档内容

本目录目前承接的是先前放在主目录、但不属于 `CGC_Gate` 主线资产的 Markdown 文档，例如：

- ABI / Runtime / Manifest 相关白皮书
- UPKG 2.1 / 3.x 技术规格与逻辑说明
- Edge Engine / DeltaMem / Embodied Runtime 相关白皮书
- Unified Pipeline Kernel 设计与定位说明
- 其他历史说明性白皮书与补充材料

## 统一读取语义

读取本目录下文档时，默认应采用以下统一语义：

- `历史稿`
  - 表示该文档保留历史背景、设计分叉、比较过程或阶段性结论
- `proof 支撑`
  - 表示其中部分内容仍可作为当前正式 Gate 或 UPKG 口径的证据、比较材料或工程背景说明
- `非主入口`
  - 表示它不是当前 repo 下最新的正式主白皮书、主 gate 入口或 machine-consumable artifact 的首选读取路径

换句话说，archive 文档可以继续被引用，但引用方式应更保守：

- 可以作为历史上下文、选型比较、proof 支撑材料
- 不应直接替代当前主目录中的正式 Gate / UPKG 主入口

以当前 `CGC_Gate_2.0 = done / proof / target` 口径为例：

- archive 中与 `UPKG 2.1 / DFlash / SGLang / DeepEP` 相关的比较稿
  - 可以作为 `proof` 支撑材料
- 但不应被误读为：
  - `CGC_Gate_2.0` 的最新主白皮书
  - `Gate 2.0 target` 能力已经完成的正式宣称

若需要查阅当前生效的正式入口，应优先回到：

- `docs/technical_whitepapers/` 主目录中的 `CGC_Gate_*`
- `docs/gate_whitepapers/` 下当前仍作为正式入口消费的 `UPKG` 文档
- 对应的 `gate_map.json`、`checkin.example.json`、`summary.example.json`

## 目录治理约定

后续如新增新的正式 gate 主文档，建议继续保留在 `docs/technical_whitepapers` 主目录，并采用与 `CGC_Gate_*` 一致的命名方式。

若是历史版本、旧方案说明、补充附录或不再作为当前 gate 主入口的技术文档，建议放入本 `archive/` 目录。
