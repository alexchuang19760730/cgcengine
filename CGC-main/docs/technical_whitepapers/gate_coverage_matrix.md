# 既有 M/UPKG 体系 vs CGC_Gate_1.0/2.0 覆盖矩阵

**版本**: v1.0  
**状态**: 工作整理版  
**目的**: 将当前 repo 中 `M gate`、`UPKG gate`、`CGC_Gate_1.0`、`CGC_Gate_2.0` 的定义、映射与覆盖关系整理成一份正式 Markdown，作为后续白皮书、gate review、CLI 汇总与 release checkin 的统一参考。

---

## 1. 适用范围

本文覆盖以下内容：

- `M1-M6`
- `M7-M7.9`
- `M8 / M81-M84`
- `UPKG 1.1`
- `UPKG 2.1`
- `UPKG 3.x`
- `UPKG 4.0`
- `CGC_Gate_1.0_edge_cloud_autonomy`
- `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation`

本文中：

- `m8x` 按 `m8 + m81 + m82 + m83 + m84` 解释
- `m71 = M7.1`
- `m72 = M7.2`
- `m73 = M7.3`
- 以此类推

---

## 2. 整理口径

本次整理采用以下口径：

1. 优先接受当前 repo 中**已注册到 CLI registry** 的 gate 定义
2. 若某 gate 未直接出现在 registry，但已有**正式白皮书定义**
   - 则按白皮书口径纳入
3. 若某 gate 的定义主要来自 gate 配置文件
   - 则按 `M8 YAML acceptance contract` 的正式字段纳入
4. 不把零散临时脚本、旧实验目录中的名字直接提升为正式 gate 定义

因此，本文中的 gate 定义主要来自：

- [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py)
- [m1_m6_pipeline_gates.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m1_m6_pipeline_gates.py)
- [m7_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m7_gate.py)
- [m72_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m72_gate.py)
- [m73_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m73_gate.py)
- [m74_dflash_kda_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m74_dflash_kda_gate.py)
- [m75_api_compat_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m75_api_compat_gate.py)
- [m76_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m76_gate.py)
- [m77_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m77_gate.py)
- [m78_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m78_gate.py)
- [m79_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m79_gate.py)
- [m8_gate.yaml](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.yaml)
- [m8_gate.py](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.py)
- [CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md)
- [CGC_Gate_1.0](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy/CGC_Gate_1.0_edge_cloud_autonomy_Technical_Whitepaper_v1.0_zh_CN.md)
- [CGC_Gate_2.0](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_Technical_Whitepaper_v1.0_zh_CN.md)

---

## 3. 工作记录

为避免遗漏 `alias`、`aggregate gate` 与 `非 registry 但有正式语义` 的情况，本次整理包含以下检索动作：

- 在 [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py) 中检查 gate registry、`load_*_gate_runner()`、`alias mapping`
- 在工作区搜索：
  - `upkg11|upkg21|upkg22|upkg30|upkg31|upkg32|upkg33|upkg34|upkg35|upkg36|upkg37|upkg38|upkg39|upkg40`
  - `UPKG 1\.1|upkg11|UPKG 2\.2|upkg22|UPKG 3\.9|upkg39`
  - `def run_upkg11|def run_upkg22|def run_upkg39|upkg11|upkg22|upkg39`
- 在 [checkins](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/checkins) 中核对当前是否存在 `*_latest.json`

本次整理特别采用了两段法：

- 先列 `M gate`
- 再把所有 `UPKG gate` append 在后面

同时，`UPKG gate` 再细分为两类：

- **当前 registry / CLI 可调用**
- **repo 中已有正式语义，但不一定在当前 registry 直接暴露**

这样可以避免把 `upkg11 / upkg22 / upkg39` 混成同一种状态。

---

## 4. M Gate 定义表

| Gate | 定义 | 当前定位 / 备注 | 主要来源 |
|---|---|---|---|
| `m1` | 本地 native baseline executable gate | `M1` 基线执行入口 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3117-L3121), [m1_m6_pipeline_gates.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m1_m6_pipeline_gates.py#L1038-L1043) |
| `m2` | inference kernel and safety gate | 编译前策略与 gate 包装成立 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3122-L3126), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L42-L45) |
| `m3` | model solidification and edge packaging gate | bundle / export 产物落地 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3127-L3131), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L42-L45) |
| `m4` | training and distributed scale-out gate | training + inference 双路聚合；training 路必须有 distributed 证据 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3132-L3136), [m1_m6_pipeline_gates.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m1_m6_pipeline_gates.py#L1050-L1135), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L45-L46) |
| `m5` | terminal-state compile and runtime closure gate | fullgraph / AOT / bench / deploy 收口；低内存 Mac 可接受 `oMLX + dflash` fallback | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3137-L3141), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L46-L47), [PASS 条件](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L117-L133) |
| `m6` | product bundle build-and-run gate | 产品 bundle 构建与运行 gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3142-L3146), [m1_m6_pipeline_gates.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m1_m6_pipeline_gates.py#L1138-L1143) |
| `m7` | industrial baseline verification-only gate | 工业级 unified kernel 验证总入口 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3147-L3151), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L47-L53), [m7_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m7_gate.py#L78-L220) |
| `m71` | `M7.1` 核心内核层 | 负责 `dynamic trace / state compression / soft-RT replay / industrial audit`；与 `m7` 共用 core artifact | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3152-L3156), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L48-L49), [M7/M7.1 条件](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L135-L147) |
| `m72` | `M7.2` 数字世界 GUI Agent 验收层 | 在 `M7.1` 基础上增加 GUI / 桌面场景验证 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3157-L3161), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L48-L50), [m72_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m72_gate.py#L154-L220) |
| `m73` | `M7.3` 物理执行 / 具身端云桥接验收层 | 要求 `cloud training + edge bridge + state compression + audit` 同时成立 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3162-L3166), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L49-L50), [m73_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m73_gate.py#L142-L220) |
| `m74` | `M7.4` DFlash + TrueOrthoKDA 验收层 | 检验 `dflash` 合同、`TrueOrthoKDA` 合同与 runtime evidence | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3167-L3171), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L50-L52), [m74_dflash_kda_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m74_dflash_kda_gate.py#L159-L194) |
| `m75` | `M7.5` API compatibility gate | 当前 registry 暴露的是 API compatibility 线；白皮书同时说明还有 `TrueOrthoKDA active runtime` 验证线 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3182-L3186), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md#L51-L53), [m75_api_compat_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m75_api_compat_gate.py#L41-L155) |
| `m76` | `M7.6` heterogeneous acceleration integration gate | 异构加速集成验证层 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3187-L3191), [m76_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m76_gate.py) |
| `m77` | `M7.7` cloud-edge training / edge inference / Q2RL gate | 云训边推 + Q2RL 独立 gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3192-L3196), [m77_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m77_gate.py#L26-L188) |
| `m78` | `M7.8` GUI teaching / trained-model edge inference / pure LLM six-element comparison gate | 教学模式与纯 LLM 六元素推理比较 gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3242-L3250), [m78_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m78_gate.py#L36-L220) |
| `m79` | `UPKG 4.0 psi0 cloud training + realtime-vla edge inference + comparative benchmark gate` | 虽编号是 `m79`，定义上承接 `UPKG 4.0 embodied benchmark` | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3257-L3265), [m79_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m79_gate.py#L26-L190) |
| `m8` | `M8.0` productization / DX gate | 聚合 `M8.1-M8.4` 的产品化与开发者体验 gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3267-L3270), [m8_gate.py](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.py#L1598-L1756), [m8_gate.yaml](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.yaml#L1-L36) |
| `m81` | `M8.1` productization | `M7.5 API 相容性 + Claude Code` 双验收，并覆盖用户模型接入与 NFS 模型发现 | [m8_gate.yaml](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.yaml#L38-L72), [m8_gate.py](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.py#L1647-L1660) |
| `m82` | `M8.2` CLI build / run-route acceptance | `cgc run response contract + route decision takeover` 双验收，覆盖 `M4 local` 与 `M4->M7.3 takeover` | [m8_gate.yaml](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.yaml#L73-L156), [m8_gate.py](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.py#L1661-L1678) |
| `m83` | `M8.3` cloud-native / serve acceptance | `cgc serve` streaming local success + `M7.3 takeover streaming` 双验收 | [m8_gate.yaml](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.yaml#L157-L208), [m8_gate.py](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.py#L1679-L1690) |
| `m84` | `M8.4` release build acceptance | `cgc build`、build matrix、dist manifest、per-platform size budget 多重验收 | [m8_gate.yaml](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.yaml#L209-L220), [m8_gate.py](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.py#L1691-L1726) |

### 4.1 补充说明

- 当前**没有看到正式 `m80`** 的 registry、runner 或 checkin
- 当前 checkin 可见到 `m1~m8`、`m71~m79`
- `m81~m84` 当前是通过 `m8` 聚合结果暴露，不是独立 `*_latest.json` checkin

---

## 5. UPKG Gate 定义表

以下分成两类：

- 当前 `registry / CLI` 可调用
- repo 中已有正式语义，但不一定在当前 registry 直接暴露

| UPKG Gate | 定义 | 当前状态 / 备注 | 主要来源 |
|---|---|---|---|
| `upkg11` | `UPKG 1.1` kernel 当前正式工程验收口径 | 覆盖 `M1-M7.5`；更像正式白皮书 / 口径层，不在当前 `get_gate_registry()` 直接暴露 | [docs/README.md](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/README.md#L18-L25), [UPKG 1.1 白皮书](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/gate_whitepapers/CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md) |
| `upkg21` | backend-injectable optimization gate | 聚合 `M5 + M7.4 + 选定的 SGLang DFlash route` | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3172-L3176) |
| `upkg21-rerun` | composite rerun wrapper | 负责 `m75 -> m76 -> upkg21` sibling evidence wiring | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3177-L3181) |
| `upkg22` | 未找到正式独立 gate 定义 | 当前只看到它作为 `m84 build / optimization package` 语境中的组件影子，不是独立 registry gate | [m8_gate.yaml](file:///Users/alexchuang/Documents/flashkv0516/CGC_Release/m8_gate.yaml#L209-L220), [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py) |
| `upkg3` | `UPKG 3.0` 的 alias | 指向 `upkg30` | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3202-L3206) |
| `upkg30` | `UPKG 3.0` aggregate product gate | 顶层聚合 `3.1-3.7`，当前实现里也映射到 `3.8` 产物关系 | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3197-L3201), [mapping](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L1894-L1902) |
| `upkg31` | `UPKG 3.1` alias for `m7` | Kernel core product gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3207-L3211), [mapping](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3352-L3356) |
| `upkg32` | `UPKG 3.2` alias for `m72` | agent runtime gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3212-L3216), [mapping](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3352-L3358) |
| `upkg33` | `UPKG 3.3` alias for `m73` | physical execution / edge bridge gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3217-L3221), [mapping](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3354-L3355) |
| `upkg34` | `UPKG 3.4` alias | unified artifact and cloud-summary gate path | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3222-L3226) |
| `upkg35` | `UPKG 3.5` alias | six-element audit and attribution gate path | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3227-L3231) |
| `upkg36` | `UPKG 3.6` alias | closure and graph-native integration gate path | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3232-L3236) |
| `upkg37` | `UPKG 3.7` alias for `m77` | standalone cloud-edge training / edge inference / Q2RL gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3237-L3241), [m77_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m77_gate.py#L119-L126) |
| `upkg38` | `UPKG 3.8` alias for `m78` | GUI teaching / pure LLM six-element comparison gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3247-L3250), [m78_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m78_gate.py#L168-L180) |
| `upkg39` | `UPKG 3.9` strict closure gate | 恢复 `0.8` 对齐、schema validation、tensorized graph-native closure 的 strict closure | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3252-L3256), [UPKG39 release note](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/docs/release_notes/UPKG39_STRICT_CLOSURE_RELEASE_NOTE_20260622_zh_CN.md#L1-L10) |
| `upkg40` | `UPKG 4.0` alias for `m79` | `psi0 cloud training + realtime-vla edge inference + comparative benchmark` gate | [cgc.py](file:///Users/alexchuang/Documents/flashkv0516/app/cli/cgc.py#L3262-L3265), [m79_gate.py](file:///Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/product/m79_gate.py#L131-L138) |

---

## 6. 当前 gate 体系的简化理解

从当前代码与白皮书口径看，可以先把体系分成以下层次：

- `M1-M6`
  - 基础 pipeline / compile / training / packaging / runtime closure
- `M7-M7.6`
  - kernel、GUI、edge bridge、protocol、API、heterogeneous integration
- `M7.7-M7.9`
  - 产品链、教学链、embodied benchmark
- `M8 / M81-M84`
  - productization / DX / serve / release build
- `UPKG 1.1`
  - 覆盖 `M1-M7.5` 的 kernel 正式口径
- `UPKG 2.1`
  - model / backend injectable 优化聚合
- `UPKG 3.x`
  - agent / product chain 聚合
- `UPKG 4.0`
  - embodied runtime comparative benchmark

---

## 7. CGC_Gate 1.0 / 2.0 覆盖强度定义

为避免把 `CGC_Gate_1.0` / `CGC_Gate_2.0` 错看成对整个 `M/UPKG` 体系重新分层，本文用以下覆盖强度来表达关系：

- `主覆盖`
  - 该 Gate 的主要叙事和核心价值落在这里
- `中覆盖`
  - 明显依赖或深度相连，但不是主主体
- `弱覆盖`
  - 只是基座、下游消费者或旁路关联
- `目标覆盖`
  - `CGC_Gate_2.0` 明确想进入，但当前还未正式完成
- `done-support`
  - 当前代码已实跑验证，可作为 `CGC_Gate_2.0` 的已验证承接层或消费锚点
- `boundary_only`
  - 只保留边界说明，不并入当前 `CGC_Gate_2.0 done` 能力集

---

## 8. 既有 M/UPKG 体系 vs CGC_Gate_1.0 / 2.0 覆盖矩阵

| 既有体系 | `CGC_Gate_1.0` 覆盖强度 | `CGC_Gate_2.0` 覆盖强度 | 说明 |
|---|---|---|---|
| `M1-M6` | `弱覆盖` | `弱覆盖` | 两者都把 `M4/M5/M6` 当作上游基础，不是主叙事 |
| `M7-M7.6` | `主覆盖` | `主覆盖` | 这是两代 Gate 最核心的重叠区；1.0 偏自治、handoff、governance，2.0 偏 layer-adaptive continuation；其中 `m76` 现也被 Gate 2.0 显式解释为 `m7.6 dev gate` 的 proof 锚点，同时 2.0 已把云侧 runtime 进一步拆成 `SGLang TP4EP4 = done`、`DeepEP = done`、`Ray engine = done`、`ColossalAI = done` 的独立 capability |
| `M7.7-M7.8` | `弱覆盖` | `done-support` | `m77` 与 `m78` 当前代码已分别实跑 `PASS`，可作为 `CGC_Gate_2.0` 的已验证上游消费锚点，但仍不改变 2.0 的核心主体仍在 `M7-M7.6 + layer-adaptive continuation target` |
| `M7.9` | `几乎不覆盖` | `boundary_only` | `m79` 当前虽已实跑 `PASS`，但语义上承接的是 `UPKG 4.0 embodied benchmark`，本轮仅保留边界说明，不并入 `CGC_Gate_2.0 done` |
| `M8 / M81-M84` | `弱覆盖` | `弱覆盖` | `M8` 是 productization / DX / release build，更多是消费 1.0/2.0 artifact，不是它们的主体 |
| `UPKG 1.1` | `中覆盖` 到 `主覆盖` | `中覆盖` 到 `主覆盖` | 两代 Gate 都强依赖 kernel/runtime substrate；尤其 2.0 的 continuation ABI 本质上踩在这层上 |
| `UPKG 2.0` | `弱覆盖` | `中覆盖` | 2.0 现已显式吸收 `UPKG 2.0 model product gate` 作为 `proof` 承接层，用来证明 layer-adaptive runtime 最终可进入正式 model productization 体系 |
| `UPKG 2.1` | `中覆盖` | `中覆盖` 到 `主覆盖` | 1.0 已明显进入 `DFlash / TrueOrthoKDA / backend-injectable` 语义；2.0 如果继续做层粒度 continuation，会更深踩这层；当前 2.0 也借此把 `DeepEP route contract`、`Ray engine dual-host topology` 与 `ColossalAI candidate runtime` 明确拆成独立 capability，并纳入独立 `done` runtime capability 集 |
| `UPKG 3.x` | `弱覆盖` | `done-support` | 当前代码已完成 `upkg30-upkg39 = PASS`，且 `m77/m78` 已单独跑通，因此 `UPKG 3.x` 对 `CGC_Gate_2.0` 已从 `proof` 承接层提升为已验证产品链支撑层；但这不等于 2.0 直接取代 agent gate 本体 |
| `UPKG 4.0` | `几乎不覆盖` | `boundary_only` | `UPKG 4.0` 当前对应 `m79/upkg40` embodied benchmark 语义；本轮只保留其与 2.0 的边界关系，不将其并入 `CGC_Gate_2.0 done` |

---

## 9. 结论

### 9.1 直接结论

`CGC_Gate_1.0` 和 `CGC_Gate_2.0` **不是按 `M1-M8x` 或 `UPKG1.1-4.0` 重新切一遍**，而是叠加在既有 gate 体系上的“端云自治 / 层粒度 PD 分离”横切面 gate。

因此，“覆盖多少”更合理的看法不是算术加总，而是看**覆盖强度**。

需要补充的是，在当前代码验证快照下：

- `m1-m7.6` 已实跑通过
- `upkg21` 已实跑通过
- `m77` 已实跑通过
- `m78` 已实跑通过
- `upkg30-upkg39` 已实跑通过

这意味着 `CGC_Gate_1.0` 与 `UPKG 2.1` 的关系，已经从“明显进入相关语义”进一步推进为“当前代码下具备正式实跑闭环”。

### 9.2 对两代 Gate 的简化判断

- `CGC_Gate_1.0`
  - 主覆盖：`M7-M7.6`, `UPKG 1.1`, `UPKG 2.1`
  - 次覆盖：`M1-M6`
  - 弱覆盖：`M7.7-M8x`, `UPKG 3.x`
- `CGC_Gate_2.0`
  - 主覆盖：`M7-M7.6`, `UPKG 1.1`, `UPKG 2.1`
  - 次覆盖：`UPKG 2.0`
  - `done-support`：`UPKG 3.x`, `M7.7-M7.8`
  - `boundary_only`：`M7.9`, `UPKG 4.0`
  - 弱覆盖：`M8x`

进一步说：

- `CGC_Gate_1.0` 当前已不只是覆盖 `UPKG 2.1` 语义，而是已被 `upkg21` 实跑结果反向验证
- `CGC_Gate_2.0` 当前仍不是完整 gate-pass，但其云側执行基座可以直接继承这一已验证底座；其中 `UPKG 3.x` 与 `M7.7/M7.8` 已进入 `done-support`，`UPKG 2.0` 仍为 `proof`，`M7.9 / UPKG 4.0` 则保持 `boundary_only`；同时云侧 runtime 能力已进一步拆成 `SGLang TP4EP4 / DeepEP / Ray engine / ColossalAI = done`

### 9.3 一句话理解

- `CGC_Gate_1.0`
  - 主体是**端云自治闭环**
- `CGC_Gate_2.0`
  - 主体是**端云自治之上的层粒度 PD 分离与 continuation 闭环**

它们都**不直接等于** `M8` 或 `UPKG 3.x / 4.0`，而是更像这些产品级 gate 的底座能力层；只是当前 `CGC_Gate_2.0` 已能对 `UPKG 3.x` 与 `M7.7/M7.8` 给出 `done-support` 级别的正式支撑。
