# CGC M5 / UPKG 2.1 Backend Injectable Optimization Gate 规范草案

**版本**: v0.2  
**状态**: 草案（现状已由当前代码验证补强）  
**定位**: 定义 `M5` 从“本机 fullgraph AOT 编译 gate”升级为“可插入 backend 的优化包 gate”后的正式边界，并给出 `UPKG 2.1` 的最小产物、最小通过条件、失败归因、`M7.4` 聚合方式与 `DFlash / SGLang` 路由选择口径；同时补充当前代码下 `upkg21` 已实跑通过的现状说明。

---

## 一、文件定位

当前 `M5` 已能证明：

- `fullgraph AOT compile` 可执行
- 目标模型可生成编译产物
- 编译结果可以进入统一 pipeline 报告

但当前 `M5` 仍主要停留在“编译产物存在”这一层，而没有正式回答下面四个问题：

- 这些优化产物是否可以在**不改用户源码**的前提下插入既有 `mlx` / `llama.cpp` 安装
- 这些优化产物是否可以在**不同 backend 版本族**下进行兼容判定与可回退部署
- 用户是否可以保留自己的**模型版本、量化版本、硬件优化路径**
- 编译结果是否已经从“工程内产物”升级为“可交付优化包”

因此，本草案引入：

- `M5`
  - 负责生成与验证 `backend-injectable optimization package`
- `UPKG 2.1`
  - 负责把该能力提升为 `model product gate` 的正式交付要求
- `M7.4`
  - 负责 `DFlash + TrueOrthoKDA` 的独立 verification gate，并作为 `UPKG 2.1` 的必经补充验收段

本文件与现有文档的关系如下：

- `CGC_UPKG_2_0_MODEL_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md`
  - 负责 `UPKG 2.0` 的 model 产品化主边界
- `CGC_Edge_Engine_Whitepaper_v1.0.md`
  - 负责 build / package / runtime 交付方式参考
- `CGC_Runtime_Component_Contract_and_System_Manifest_Whitepaper_v1.0_zh_CN.md`
  - 提供 contract / manifest 的结构化收敛方式
- 本文件
  - 负责 `M5 / UPKG 2.1` 的新 gate 规范草案

补充说明：

- 自 `v0.2` 起，`UPKG 2.1` 不再只看 `M5` 编译/注入产物本身，还必须聚合 `M7.4`
- `M7.4` 在 `UPKG 2.1` 中的职责不是替代 `M5`，而是证明 `DFlash + TrueOrthoKDA` 路由已被正式 gate 化
- `DFlash` 与 `SGLang Spec V2 overlap scheduler` 在当前口径下应视为**比较关系**而不是**同一路径的强绑定要求**
- 当前代码验证快照下，`upkg21` 已实跑通过，因此本文的“最小通过条件”已不只是设计要求，也已被当前 repo 结果支撑

### 1.1 当前验证快照

为避免把本文误读成“仍未被代码验证的纯草案”，这里补充当前代码下已经成立的事实：

- `m74 = PASS`
- `m75 = PASS`
- `m76 = PASS`
- `upkg21 = PASS`

这意味着本文中关于 `DFlash / SGLang / DeepEP` 路由、`M7.4` 聚合要求、以及 `UPKG 2.1 = PASS` 的组合条件，当前已不只是目标规范，而是有现行代码与正式 report 支撑的工程口径。

---

## 二、问题定义

### 2.1 当前问题

当前工程中，优化能力大体分成三类：

- 源码级 patch
- 编译级产物
- runtime 侧 contract / evidence

问题在于，源码级 patch 无法成为用户侧正式交付方式，因为它：

- 要求用户接受源码改动
- 难以与用户已有 backend 安装并存
- 难以表达版本族兼容性
- 难以做安全回退

### 2.2 新目标

`M5 / UPKG 2.1` 要证明的不是“生成了一个 `.so/.dylib`”，而是：

- 生成了一套**可插入的优化包**
- 该优化包具有**明确 ABI / 版本族边界**
- 可以在受支持 backend 上**完成最小注入与最小推理 smoke**
- 注入失败时可以**自动回退**
- 可以在不改用户源码的前提下，让用户继续使用：
  - 个性化模型
  - 个性化量化版本
  - 个性化硬件优化路径

---

## 三、范围与非范围

### 3.1 范围

`M5 / UPKG 2.1` 当前只覆盖：

- `macOS arm64`
- `Apple Silicon`
- `mlx` 版本族的受控注入
- `llama.cpp` 版本族的受控注入
- 本地模型推理链上的注入式优化包
- `probe -> inject -> smoke -> rollback` 的最小闭环

### 3.2 非范围

以下内容不在本草案首批范围内：

- 任意版本 `mlx` 的完全通吃
- 任意 commit `llama.cpp` 的完全通吃
- 跨平台统一 ABI 一次性全覆盖
- 远端云侧 runtime 注入
- 端云桥接 takeover 的正式 comparative
- 以 benchmark 分数替代兼容性验证

---

## 四、核心术语

### 4.1 Backend Injectable Optimization Package

指一套可以挂接到现有 backend 安装上的优化包，至少包含：

- 优化 binary
- shim / adapter
- compat manifest
- install / inject recipe
- probe / rollback 说明

### 4.2 Stable Plugin ABI

指 CGC 对外发布的稳定插件契约，不要求 backend 内部实现稳定，但要求交付包的兼容性声明可被自动验证。

最小字段包括：

- `abi_version`
- `package_id`
- `backend_family`
- `backend_version_range`
- `platform`
- `arch`
- `device_family`
- `model_format_family`
- `required_symbols`
- `entrypoints`
- `fallback_mode`

### 4.3 Version Family Compatibility

不再宣称“任意版本可插”，而是明确声明：

- 支持哪个 `major.minor` 版本族
- 是否需要特定 commit family
- 是否要求特定 Python / dylib / loader 条件

---

## 五、产物模型

`M5` 新版通过后，最小输出目录应为：

```text
output/
  report.json
  optimization_package/
    package_manifest.json
    abi_manifest.json
    compatibility_matrix.json
    install_recipe.json
    rollback_recipe.json
    binaries/
      lib/
      hooks/
      adapters/
    smoke/
      probe_report.json
      inject_report.json
      inference_smoke_report.json
      rollback_report.json
```

### 5.1 package_manifest.json

必须回答：

- 这是什么包
- 面向哪个 backend family
- 面向哪个模型格式族
- 由哪个 `M5` 编译链生成

### 5.2 abi_manifest.json

必须回答：

- 该包对外暴露什么 entrypoint
- 需要什么 required symbols
- 兼容哪个 backend version range
- 需要什么 platform / arch / device family

### 5.3 compatibility_matrix.json

必须表达：

- `backend_family`
- `supported_version_ranges`
- `tested_version_ranges`
- `blocked_version_ranges`
- `reason`

### 5.4 install_recipe.json

必须表达：

- `install_mode`
- `inject_mode`
- `target_layout`
- `preflight_checks`
- `post_install_checks`

支持但不限于：

- `bundle_patch`
- `loader_relink`
- `plugin_dropin`
- `dyld_preload`
- `sidecar_loader`

### 5.5 rollback_recipe.json

必须表达：

- 注入失败时如何恢复
- 如何恢复原始 loader path / binary link
- 如何判定回退成功

---

## 六、Gate 结构

`M5 / UPKG 2.1` 的新 gate 拆成五个强制子 gate。

### 6.1 m5_artifact_package_gate

验证优化包是否已按正式结构产出。

**通过条件**：

- `package_manifest.json` 存在
- `abi_manifest.json` 存在
- `install_recipe.json` 存在
- `rollback_recipe.json` 存在
- 至少一个优化 binary 存在

### 6.2 m5_abi_compatibility_gate

验证优化包是否明确声明兼容边界，而不是仅输出裸二进制。

**通过条件**：

- `abi_version` 非空
- `backend_family` 非空
- `backend_version_range` 非空
- `required_symbols` 非空
- `entrypoints` 非空

### 6.3 m5_injection_smoke_gate

验证在受支持 backend family 上能完成最小探测、最小注入与最小推理。

**通过条件**：

- `probe_report.status = PASS`
- `inject_report.status = PASS`
- `inference_smoke_report.status = PASS`
- 推理结果至少返回最小 `response/evidence`

### 6.4 m5_fallback_safety_gate

验证注入失败或主动卸载时，原 backend 可以恢复。

**通过条件**：

- `rollback_report.status = PASS`
- rollback 后原 backend 最小 smoke 成功
- 不出现残留 loader path 污染

### 6.5 m5_personalization_gate

验证用户可保留个性化模型与硬件 profile，而不是被强制绑定到单一 demo 环境。

**通过条件**：

- 用户自定义模型路径可被消费
- 用户自定义量化 / format family 可被识别
- 用户硬件 profile 被写入 contract / evidence
- 若不支持，必须给出结构化原因，而不是 silent fail

### 6.6 m5_dflash_sglang_route_gate

验证 `UPKG 2.1` 是否已正式消费 `M7.4`，并对 `DFlash / SGLang / DeepEP` 路径做出明确选择。

**通过条件**：

- `m74_dflash_trueorthokda_gate = PASS`
- 已给出结构化 `selected_sglang_runtime`
- 已明确记录 `DFlash` 当前采用的 `SGLang` 执行模式
- 已明确记录 `requested_dispatch_backend = deepep`
- 已明确记录 `deepep_parallel_profile = ep16_tp1`
- 已明确声明 `Spec V2` 在当前 `DFlash` 路径中的关系是“比较基线/能力参考”，而不是“必须开启的执行模式”

---

## 七、UPKG 2.1 正式要求

`UPKG 2.1` 作为 `model product gate`，要求的不只是 `M5 PASS`，而是：

- `M5` 已输出优化包
- `M6` 已证明最终本地模型运行可落地
- `M8.4` 已证明 build / release artifact 可发行
- `UPKG 1.x` 的 pipeline contract artifacts 可为该包提供上游真源
- `UPKG 3.x` 的 summary / trace / failure attribution 结构可复用到模型产品链

### 7.1 最小 accepted artifacts

`UPKG 2.1` 必须接受以下产物：

- `package_manifest.json`
- `abi_manifest.json`
- `compatibility_matrix.json`
- `install_recipe.json`
- `rollback_recipe.json`
- `probe_report.json`
- `inject_report.json`
- `inference_smoke_report.json`
- `rollback_report.json`
- `summary.json`
- `artifact_index.json`
- `failure_attribution.json`

### 7.2 最小通过条件

`UPKG 2.1 = PASS` 必须同时满足：

- `m5_artifact_package_gate = PASS`
- `m5_abi_compatibility_gate = PASS`
- `m5_injection_smoke_gate = PASS`
- `m5_fallback_safety_gate = PASS`
- `m5_personalization_gate = PASS`
- `m5_dflash_sglang_route_gate = PASS`

当前代码验证快照已经满足上述组合要求，因此这里的 PASS 条件既是规范条文，也是当前 repo 已跑通的收口条件。

### 7.3 新增聚合要求

`UPKG 2.1` 从 `v0.2` 起应以独立 gate 形式聚合以下对象：

- `M5`
- `M7.4`
- `selected_sglang_runtime`
- `SGLang + DFlash + DeepEP ep16/tp1 route contract`
- `DFlash / Spec V2 comparison statement`

推荐对外入口：

- `cgc gate upkg21`

最小聚合产物新增：

- `upkg21_report.json`
- `upkg21_sglang_selection.json`

---

## 八、首批兼容矩阵建议

第一批不要追求“全版本通吃”，而应只声明受控版本族。

### 8.1 mlx

建议首批仅支持：

- `backend_family = mlx`
- `backend_version_range = 0.20.x ~ 0.21.x`
- `platform = macOS`
- `arch = arm64`

### 8.2 llama.cpp

建议首批仅支持：

- `backend_family = llama.cpp`
- 指定 release family 或 commit family
- `platform = macOS`
- `arch = arm64`

### 8.3 blocked family

首批直接声明 `BLOCKED`：

- 未测试 `mlx` 大版本漂移
- 未测试 Linux / Windows 交叉注入
- 未测试 CUDA / ROCm / Ascend 的统一注入 ABI

---

## 九、失败归因

`M5 / UPKG 2.1` 的失败必须结构化归因，禁止只返回 `build failed`。

### 9.1 建议 failure_code

- `package_manifest_missing`
- `abi_manifest_incomplete`
- `unsupported_backend_family`
- `backend_version_out_of_range`
- `required_symbol_missing`
- `inject_recipe_invalid`
- `inject_smoke_failed`
- `rollback_failed`
- `model_format_unsupported`
- `hardware_profile_mismatch`

### 9.2 failure attribution 最小字段

- `failure_code`
- `failure_stage`
- `backend_family`
- `backend_version`
- `platform`
- `arch`
- `device_family`
- `model_format_family`
- `reason`
- `suggested_action`

---

## 十、与现有 Gate 的关系

### 10.1 对 M5 的升级

旧 `M5` 主要证明：

- fullgraph AOT compile
- 编译产物存在

新 `M5` 进一步证明：

- 编译产物已被产品化为优化包
- 优化包可插入 backend
- 优化包可回退
- 优化包可表达版本族兼容性

### 10.2 对 M6 的关系

`M6` 继续负责最终本地运行落地；
`M5` 则负责“可插入优化包”的生成与最小注入验证。

### 10.3 对 M8.4 的关系

`M8.4` 负责 release build / dist manifest / installable artifact；
`M5 / UPKG 2.1` 负责优化包本身的兼容性与注入式交付。

### 10.4 对 M7.4 的关系

`M7.4` 继续保持独立 verification gate 身份；
`UPKG 2.1` 则负责正式消费 `M7.4` 的结论，并将其纳入模型产品化验收。

这意味着：

- `M7.4` 回答 “`DFlash + TrueOrthoKDA` 是否存在且可被验证”
- `UPKG 2.1` 回答 “该路由是否已进入产品级交付与验收口径”

### 10.5 对 SGLang Spec V2 的关系

`Spec V2` 在当前文档中的角色是：

- 用来比较 `SGLang` 当前 speculative stack 的能力边界
- 用来说明 `DFlash` 不应被误写成“必须开启 overlap scheduler”

当前 `UPKG 2.1` 的正式要求是：

- 必须选择一个明确的 `SGLang` 路径
- 必须说明 `DFlash` 当前运行模式
- 不强制要求 `DFlash == Spec V2`

---

## 十一、第一批实施建议

### 11.1 第一阶段

- 先为 `mlx` 定义 `stable plugin ABI`
- 生成 `package_manifest.json / abi_manifest.json`
- 先接 `bundle_patch + loader_relink`

### 11.2 第二阶段

- 为 `llama.cpp` 增加 plugin / hook 模式
- 引入 `required_symbols` 自动探测
- 增加 version family smoke matrix

### 11.3 第三阶段

- 把 `compatibility_matrix.json` 接进正式 `UPKG 2.1`
- 把 `failure_attribution` 接进 `summary.json / artifact_index.json`
- 把 personalization profile 接进 contract / evidence

---

## 十二、正式结论

`M5 / UPKG 2.1` 的升级方向不是“继续做源码级插入”，而是：

- 把优化能力正式收敛为**版本化、可探测、可回退的 backend optimization package**

其正式产品语义应为：

- 用户不需要改源代码

补充到当前代码口径：

- `UPKG 2.1` 已不再停留在“仅有规范草案”
- `DFlash + TrueOrthoKDA + vendored SGLang + DeepEP` 路由已由当前 `upkg21` 实跑结果支撑
- 因此本文应被理解为“仍保留草案结构、但核心要求已被当前工程验证补强”的规范文档
- 用户可以保留自己的模型与硬件路径
- 系统可以自动判断兼容性与注入方式
- 注入失败时可以回退

这才是 `M5` 从“编译 gate”升级为“可交付 optimization package gate”的正式边界。
