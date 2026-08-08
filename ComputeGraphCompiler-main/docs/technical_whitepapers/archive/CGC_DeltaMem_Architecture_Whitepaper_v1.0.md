# CGC DeltaMem Architecture Whitepaper v1.0（归档说明）

本文件路径沿用历史命名，但本文已不再作为当前正式技术主文档。

---

## 0. 当前状态

`DeltaMem` 现在只保留为历史术语，不再作为当前正式产品层命名。

仍然有效、并已经迁移到现行主文档的内容包括：

- `CGC KV Header`
- state externalization
- raw state bytes / state compressor
- zero-copy / `UMA` / VRAM handoff
- state resume / replay

这些内容现在统一收敛到：

- `CGC_Edge_Engine_Whitepaper_v1.0.md`
- `CGC_M75_TRUEORTHOKDA_ACTIVE_RUNTIME_WHITEPAPER_v1.0_zh_CN.md`

---

## 1. 保留的历史术语对照

为避免旧文件引用失效，本文件仅保留以下术语映射：

| 历史术语 | 当前正式口径 |
| :--- | :--- |
| `DeltaMem` | `state transport` / `zero-copy VRAM handoff` |
| `Delta KV` | 增量 state payload 或可恢复 state bytes |
| `矩阵直写` | `UMA` buffer handoff / device resume |
| `端云 KV 协议` | `CGC KV Header` + state payload contract |

---

## 2. 已删除的过时内容

以下内容已从本文件移除，不再作为当前正式路线维护：

- `Q2RL` 主线
- `AI Runtime` / `ONNX Runtime` 中枢叙事
- 旧 `M2 / M4 / M5` 交付结论混写
- 将某一种压缩算法写成唯一正式协议
- 以 `DeltaMem` 为中心的独立产品化命名

---

## 3. 现行文档边界

当前应以以下文档为准：

- `CGC_M1_8STEP_CLI_TECH_WHITEPAPER_v1.0_zh_CN.md`
  - `M1 / 8-step` 基座与 `llama.cpp / ggml-cgc` 最小产物
- `CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md`
  - `M1-M7.5` 统一 gate 契约与 artifact
- `CGC_Edge_Engine_Whitepaper_v1.0.md`
  - edge runtime、state transport、API compatibility、`M6` 产品化边界
- `CGC_M75_TRUEORTHOKDA_ACTIVE_RUNTIME_WHITEPAPER_v1.0_zh_CN.md`
  - 当前 formal runtime evidence 与 zero-copy handoff 证据

---

## 4. 结论

`DeltaMem` 对当前版本仍有价值的部分，已经全部收敛为：

- 协议头
- 状态传输
- 状态恢复
- `UMA` / zero-copy handoff

除此之外的旧叙事均已删除，不再继续维护。
