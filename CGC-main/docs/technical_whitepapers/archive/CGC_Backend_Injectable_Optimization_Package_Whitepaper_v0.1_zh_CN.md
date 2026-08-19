# CGC Backend Injectable Optimization Package 技术白皮书 v0.1

适用场景：`M5 / UPKG 2.1`、本地模型产品化交付、`mlx` / `llama.cpp` 版本族兼容、非源码级优化包注入

核心目标：把 “编译产物” 升级为 “可交付、可插入、可回退、可审计” 的优化包

判定原则：不追求任意版本通吃，只追求**版本族受控兼容**

## 1. 概述

当前 CGC 在 `M5` 已经能完成 `fullgraph AOT` 编译并生成优化产物，但这仍然属于工程内部能力。对最终用户而言，真正有价值的不是“生成了某个 `.so/.dylib`”，而是：

- 能否不改用户源码直接使用
- 能否挂到用户已有 `mlx` / `llama.cpp` 安装上
- 能否继续使用用户自选模型与量化版本
- 能否在失败时安全回退

因此，本白皮书定义一种新的交付对象：

- `Backend Injectable Optimization Package`

它把单一编译结果升级成一套正式产品包，使 CGC 的优化能力可以以“插件式交付”的方式进入用户已有 runtime。

## 2. 为什么不能继续停留在源码级插入

源码级 patch 的问题不在于技术上做不到，而在于它无法成为正式产品交付方式。

源码级方式会带来：

- 与用户私有 fork 冲突
- 与用户现有 backend 安装割裂
- 难以表达版本兼容范围
- 无法建立统一 rollback 机制
- 失败后难以归因

从产品交付角度，源码 patch 只能是研发手段，不应是正式 UX。

## 3. 新交付对象：Optimization Package

新的最小交付单元不是 “单个优化库”，而是一套包：

```text
optimization_package/
  package_manifest.json
  abi_manifest.json
  compatibility_matrix.json
  install_recipe.json
  rollback_recipe.json
  binaries/
  smoke/
```

这意味着：

- 二进制只是包的一部分
- 兼容性声明是正式产物的一部分
- 注入方法是正式产物的一部分
- 回退路径也是正式产物的一部分

## 4. 设计原则

### 4.1 版本族相容，不做全版本通吃

白皮书明确拒绝下面这种目标：

- 任意 `mlx`
- 任意 `llama.cpp`
- 任意 Python ABI
- 任意平台一次性统一

系统应只声明：

- 哪些版本族被支持
- 哪些版本族被测试
- 哪些版本族被明确阻止

### 4.2 不篡改用户模型语义

优化包可以改变执行方式，但不能隐式改变：

- 模型格式语义
- 推理契约
- 输入输出 contract

如果优化路径需要额外假设，必须进入 manifest。

### 4.3 先探测，再注入

正式注入前必须先做 preflight probe，至少验证：

- backend family
- backend version
- platform / arch
- required symbols
- model format family
- loader / codesign 条件

### 4.4 注入失败必须可回退

注入不是一次性覆盖原 backend，而是：

- 先保存原始状态
- 再切入优化路径
- 失败后恢复原始状态

回退不是附加功能，而是正式 contract。

## 5. Stable Plugin ABI

为了让优化包不再依赖源码级 patch，系统必须对外定义稳定插件契约。

最小 `Stable Plugin ABI` 字段如下：

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

### 5.1 为什么需要 ABI Manifest

如果没有 `abi_manifest.json`，系统只能做下面两件事：

- 猜当前环境是否可插
- 出错后让用户手工排查

而 `ABI manifest` 的作用是把这些隐性假设转成显式契约。

## 6. 面向不同 backend 的注入策略

### 6.1 mlx

`mlx` 更适合采用：

- `bundle_patch`
- `loader_relink`
- 受控 wheel / app bundle family

原因是其 Python extension、动态库路径、codesign 与 loader 关系更紧。

### 6.2 llama.cpp

`llama.cpp` 更适合采用：

- backend plugin
- shared library hook
- sidecar runtime loader

原因是它的共享库与推理 backend 结构更适合做插件式适配。

### 6.3 统一原则

尽管具体注入方式不同，但对外 contract 应一致：

- 先 probe
- 再 inject
- 然后 smoke inference
- 最后可 rollback

## 7. 与用户个性化模型/硬件的关系

本方案的核心价值之一，是不把用户锁死在单一 demo 环境中。

系统应允许用户继续使用：

- 自定义模型路径
- 自定义量化形式
- 自定义 `GGUF` / `MLX` 模型版本
- 自定义硬件 profile

前提是：

- 这些选择落在 optimization package 已声明的兼容边界内

换句话说，系统不是无条件兼容，而是**在声明边界内给用户自由**。

## 8. 正式 Gate 化

### 8.1 M5 新定位

`M5` 不再只是：

- `fullgraph compile gate`

而应升级为：

- `backend-injectable optimization package generation gate`

### 8.2 UPKG 2.1 新定位

`UPKG 2.1` 不再只是继承 `UPKG 2.0` 的模型运行语义，而应新增：

- optimization package 的生成
- optimization package 的兼容性声明
- optimization package 的注入 smoke
- optimization package 的安全回退

### 8.3 与现有 M6 / M8.4 的关系

- `M5`
  - 负责产出可插入优化包
- `M6`
  - 负责证明最终本地推理仍可落地
- `M8.4`
  - 负责证明最终 release artifact 可发行

因此三者关系不是替代，而是链式互补。

## 9. 最小运行流程

正式执行顺序应为：

1. 读取 `package_manifest.json`
2. 执行 preflight probe
3. 根据 `install_recipe.json` 决定注入方式
4. 完成二进制接线 / loader patch / plugin drop-in
5. 执行最小 inference smoke
6. 写出 `inject_report.json` 与 `inference_smoke_report.json`
7. 若失败则执行 `rollback_recipe.json`
8. 写出 `rollback_report.json`

## 10. 推荐的首批落地边界

为了降低实现风险，首批建议只覆盖：

- `macOS arm64`
- `Apple Silicon`
- `mlx 0.20.x ~ 0.21.x`
- 指定 `llama.cpp` commit family
- 受控的 `MLX` 与 `GGUF` 模型格式族

首批明确不承诺：

- Linux / Windows 通用注入
- CUDA / ROCm / Ascend 统一插件 ABI
- 任意版本跨代兼容

## 11. 风险与控制

### 11.1 ABI 漂移

风险：

- backend 升级后 required symbols 改变

控制：

- 明确 `backend_version_range`
- 在 preflight 阶段执行符号探测

### 11.2 loader / codesign 风险

风险：

- macOS loader path 与 codesign 导致注入失败

控制：

- 将 `bundle_patch` 与 `codesign` 步骤正式写入 recipe
- 把失败归因写成结构化 evidence

### 11.3 模型格式漂移

风险：

- 用户模型 layout 不满足优化假设

控制：

- 在 manifest 中声明 `model_format_family`
- 不满足条件时 fail-close，而不是 silent fallback

### 11.4 回退不完整

风险：

- 注入失败后原 backend 受污染

控制：

- rollback 成为正式 gate
- rollback 后必须重新跑最小 smoke

## 12. 结论

`M5 / UPKG 2.1` 的升级方向，本质上是在回答一个产品问题：

> 如何让 CGC 的优化能力不再依赖源码 patch，而是成为用户可直接消费的正式交付物。

这件事有机会做到，但前提不是“全版本通吃”，而是：

- 明确版本族
- 明确 ABI
- 明确 probe
- 明确 inject
- 明确 rollback

一旦这条链成立，CGC 输出的就不再只是“某次编译结果”，而是一套真正可分发、可验证、可回退、可审计的优化包产品能力。
