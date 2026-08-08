# MPP Tensor 路径启用验证 Checklist

> 目标：解锁 `MPPPrefillInt4QMM`（`mpp_prefill_affine_threadgroup_f16`，Metal 4 张量核 int4 prefill QMM），
> 验证它在目标机器上是否真正可用、性能是否真的有收益。

---

## 0. 背景：这条路径为什么现在没启用

`tensorops.metal` 的整个 MPP 实现被 `#if defined(__HAVE_TENSOR__)` 包住：

```metal
#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// ... matmul2d<descriptor, execution_simdgroups<4>> operation ...
#endif
```

宏未定义时：`mpp_prefill_affine_threadgroup_f16` 函数**不存在** →
`MPPPrefillInt4QMM.init` 里 `library.makeFunction` 返回 nil → `pipeline = nil` →
`isAvailable == false` → prefill 回落到普通 kernel（`PrefillProjectionDispatchPolicy`）。

### 2026-08-07 实测（本机）
| 检查项 | 结果 |
|---|---|
| macOS | 15.2（Build 24C2101） |
| 工具链 | CommandLineTools only（**无完整 Xcode**） |
| SDK | MacOSX15.2.sdk |
| `MTLGPUFamily` 枚举 | apple1 ~ apple9 支持；**apple10 无** |
| `__HAVE_TENSOR__`（clang -dM） | **未定义** |
| MPP framework 头 | 系统内**不存在**（`MetalPerformancePrimitives.framework` 缺失） |
| 结论 | MPP 路径**当前不可用**（编译期即被排除） |

---

## 1. 启用条件（全部满足才会亮）

| # | 条件 | 说明 |
|---|---|---|
| 1 | **macOS ≥ 26.2** | 张量语言特性按部署目标 ≥ 26.2 门控（Rigel 论文实测：26.5/26.6 仍发 Metal 4.0；Metal 4.1 需 Xcode 27 + macOS 27） |
| 2 | **完整 Xcode**（非 CommandLineTools） | `MetalPerformancePrimitives/MetalPerformancePrimitives.h` 只随 Xcode SDK 分发；CommandLineTools SDK 里没有 |
| 3 | **MSL 4.0+ 编译器** | `__HAVE_TENSOR__` 由新版 metal 编译器按语言版本 + 部署目标定义 |
| 4 | **GPU 支持 tensor 指令** | apple10 家族（M4 不是！实测只到 apple9）。M4 即使前三条满足，也只能"功能可用、不加速"——**真正硬件加速需 M5/A19 的 GPU Neural Accelerator** |
| 5 | `Package.swift` 部署目标 ≥ `.macOS(.v26)` | 当前是 `.v15`，需同步提升（顺带解锁 apple10 枚举，修复之前 `MTLGPUFamily.apple10` 编译错误） |

> ⚠️ **最重要的预期管理**：Rigel（2026，M4 Max 实测）明确——"M4 generation predates the GPU
> Neural Accelerators that Apple introduced with the A19 and M5 parts… supports the low-precision
> formats **functionally but does not accelerate them**"。
> 即：**在 M4 上即使路径编译通过，MPP 大概率不比现有 kernel 快**（张量指令软模拟）。真正的
> 加速红利在 **M5+ / A19+** 机器。升级 macOS 的收益在 M4 上主要是编译器/驱动红利（0~5%）。

---

## 2. 升级后验证 Checklist（按顺序执行）

### 2.1 环境前置确认
- [ ] `sw_vers` → ProductVersion ≥ 26.2
- [ ] `xcode-select -p` → 指向完整 Xcode（`/Applications/Xcode.app`），不是 `/Library/Developer/CommandLineTools`
- [ ] `xcrun --show-sdk-path` → 出现 `MacOSX26.x.sdk`
- [ ] `xcrun -sdk macosx metal --version` → 能调用 metal 编译器（≥ 4.0 语言）

### 2.2 宏与编译验证
- [ ] **宏检查**：
  ```sh
  printf '#if defined(__HAVE_TENSOR__)\nYES\n#else\nNO\n#endif\n' > /tmp/t.metal
  xcrun -sdk macosx metal -E /tmp/t.metal | grep -E 'YES|NO'
  ```
  期望 `YES`。仍为 `NO` → 部署目标或编译器版本不够，直接跳到 2.5 回退评估。
- [ ] **编译 tensorops.metal**：确认 `mpp_prefill_affine_threadgroup_f16` 出现在编译产物
  （`strings *.metallib | grep mpp_prefill` 应有输出）
- [ ] **Swift 层**：`MPPPrefillInt4QMM(context:).isAvailable == true`

### 2.3 功能正确性验证
- [ ] 跑 prefill 相关单测（TensorCore / PrefillProjection 套件），MPP 路径必须有逐位等价或容差内的结果：
  ```sh
  swift test --filter 'TurboFieldfareTestsCore' 2>&1 | tail -3
  ```
- [ ] 特别关注 `PrefillProjectionDispatchPolicy` 是否真的切到 `.affineThreadgroupF16`
  （`RealForwardRunner` 里 `candidate.encode(...)` 返回的 `Path` 值）

### 2.4 性能验证（关键：A/B 必须交错，防页缓存漂移）
- [ ] 同一模型、同一 prompt、同一 max-new，各跑 3 次取中位数：
  ```sh
  # A = 新（MPP 启用）
  # B = 旧（临时把 tensorops.metal 的 #if 改为 0，或换回 .v15 编译产物）
  # 顺序 A,B,A,B,A,B
  TurboFieldfareCLI --model ... --prompt "..." --max-new 64 --trust-receipt
  ```
- [ ] 记录并对比：
  - `ttft=`（prefill 时间）——MPP 的目标指标
  - `decode=... tok/s`——预期不变（decode 是 GEMV/IO bound，不经过 MPP）
- [ ] **阈值判定**：TTFT 提升 < 5% 且存在功能风险 → 在 M4 上建议**保持关闭**
  （MPP 只对 M5+ 值得开）

### 2.5 回退方案（不满足预期时）
- 方法 A（零代码）：`Package.swift` 部署目标降回 `.v15`，`__HAVE_TENSOR__` 不定义 → 路径自动排除
- 方法 B（显式）：把 `tensorops.metal` 的 `#if defined(__HAVE_TENSOR__)` 改为 `#if 0`
- 方法 C（运行时）：不动源码，靠 `MPPPrefillInt4QMM` 的 `pipeline == nil` 自然回落
  （注意：这需要编译期宏未定义，运行时无法禁用已编译的 MPP 路径）

---

## 3. 升级动作清单（要做的事）

1. 备份当前项目（git 或 tar）
2. 升级 macOS 至 26.x + 安装完整 Xcode 26.x
3. `Package.swift`：`.macOS(.v15)` → `.macOS(.v26)`（同时修掉 `MTLGPUFamily.apple10` 编译错误——
   之前 `PrefillAttentionTests` 用 `#if swift(>=6.4)` 绕过的 guard 可以还原成直接 `supportsFamily(.apple10)`）
4. 按 2.1-2.4 执行验证
5. 更新本 checklist 的"实测"表 + 写入项目 memory

---

## 4. 相关代码位置

| 文件 | 作用 |
|---|---|
| `Sources/TurboFieldfare/Metal/TensorCore/tensorops.metal` | MPP kernel（`#if __HAVE_TENSOR__` 门控） |
| `Sources/TurboFieldfare/Kernels/TensorCore/MPPPrefillInt4QMM.swift` | Swift 封装 + `isAvailable` |
| `Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift` | prefill 调用点（~line 1336） |
| `Sources/TurboFieldfare/Kernels/Attention/PrefillAttention.swift` | Apple10 TensorOps prefill 路径（line ~83 fatal） |
| `Package.swift` | 部署目标 `.v15` → 需升 `.v26` |
| `Tests/.../PrefillAttentionTests.swift` | `apple10` guard（`#if swift(>=6.4)` 临时绕过处） |
