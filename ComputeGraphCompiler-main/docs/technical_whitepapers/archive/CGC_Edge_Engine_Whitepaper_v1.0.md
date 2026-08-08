# CGC Edge Engine: 端侧代理与协同推理技术白皮书 v1.1

本文件路径沿用历史命名（文件名仍含 v1.0），内容版本以本文标题为准。

---

## 0. 范围收敛说明

本文件现在作为 `CGC Edge Engine` 的正式主文档，统一承接以下仍未过时的内容：

- 端侧运行时与交付边界
- `cgc serve / cgc claude / cgc run` 等 CLI 入口
- 端云状态传输、`CGC KV Header`、state externalization 与 resume
- Apple Silicon `UMA / zero-copy / VRAM handoff`
- `M6` 产品化 bundle 与双侧 `report.json` 聚合验收

以下旧口径不再作为当前正式边界维护：

- 把 `DeltaMem` 当成当前主契约中心命名
- `Q2RL`、`AI Runtime`、`ONNX Runtime` 中枢叙事
- 旧 `M7.6 / M8` roadmap、`Ascend` 扩展蓝图与宣传型章节
- 把某一种压缩算法写成唯一正式协议

当前正式口径以：

- `CGC_UPKG_1_1_UNIFIED_PIPELINE_KERNEL_GATE_WHITEPAPER_v1.0_zh_CN.md`
- `CGC_M75_TRUEORTHOKDA_ACTIVE_RUNTIME_WHITEPAPER_v1.0_zh_CN.md`
- 本文件

三者共同收口。

---

## 1. Edge Engine 的定位

`CGC Edge Engine` 是 `UPKG_1.1` 之下的端侧运行时与交付层，负责把统一 kernel、端云协议、状态恢复与 API 兼容能力落实为可运行、可验收、可交付的产品边界。

当前版本只维护四类正式职责：

1. **端侧入口**
   - 提供 `cgc serve`、`cgc claude`、`cgc run`、`cgc build`、`cgc product`、`cgc verify`
2. **协议适配**
   - 提供本地 API surface 与 Ollama / OpenAI / Anthropic 相容入口
3. **状态传输**
   - 负责云侧状态封装、端侧接收、恢复与 runtime evidence
4. **产品化验收**
   - 负责 bundle、真实运行、cache hit、双侧 evidence 聚合

`M1` 保留的 `8-step` 与 `report.json` 最小语义仍是底座，但 `M1` 自身的 graph dump、`partitions.json`、`taps/` 与 workspace 统计，继续留在 `CGC_M1_8STEP_CLI_TECH_WHITEPAPER_v1.0_zh_CN.md`，不在本文件重复展开。

---

## 2. 运行时边界与 CLI

### 2.1 端侧服务边界

`Edge Engine` 当前的最小正式运行路径包括：

- `cgc serve`
  - 启动本地 API server 与协议代理，承接 edge runtime 请求
- `cgc claude`
  - 为 Claude Code 类客户端注入本地代理入口与必要环境变量
- `cgc run`
  - 直接启动本地模型或端云协同运行路径
- `cgc build`
  - 将模板固化为 bundle
- `cgc product`
  - 执行 `build + run` 的正式产品化烟雾路径
- `cgc verify`
  - 聚合 edge / cloud 两侧 `report.json`

### 2.2 本地模型与端云协同

当前仍有效的本地能力包括：

- 通过 `cgc run <model>` 加载 `.gguf`、`.safetensors`、`.mlx` 等本地权重
- 通过 `--use-omlx` 启用 macOS / MLX 路径
- 通过 `--use-flashmoe` 启用本地 MoE 的显存分页优化
- 在本地无法完整容纳时，回退到端云分离状态传输路径

这里的正式边界是“运行时能决定本地执行还是端云协同”，而不是承诺某一条历史实现分支永久不变。

---

## 3. 端云协议主契约

### 3.1 当前仍有效的协议语义

从旧 `DeltaMem` 白皮书中保留、并继续有效的协议语义只有以下几项：

- `CGC KV Header`
  - 负责描述长度、模式、shape、dtype、payload size 等元信息
- state payload
  - 以原始 tensor/state bytes 或压缩后的 bytes 作为传输主体
- state externalization
  - 将可恢复状态从 prompt 文本中剥离，作为显式 runtime object 传输
- resume / replay
  - 允许端侧在接收状态后继续 decode 或重放

### 3.2 当前不再保留的旧口径

以下内容不再是正式主契约：

- 指定 `Bit-Packing + RLE` 为唯一协议
- 把 `Delta KV` 作为唯一命名
- 把 `DeltaMem` 视为单独产品层

当前正式规则是：

- 压缩属于 implementation optimization，不是主契约中心
- 传输层只要求 state bytes 可被稳定封装、传输、恢复，并留下 runtime evidence
- 具体 codec 以实际 evidence 为准，而不是以旧白皮书的设想为准

### 3.3 当前 formal runtime 字段

`M7.5 TrueOrthoKDA active runtime` 已把当前主契约落到可验证字段，至少包括：

- `state_kind = kda_state_v1`
- `state_codec = zlib_torch_save_bytes`
- `compression_ratio`
- `cpu_copy_count`
- `uma_buffer_used`
- `device_resume_consumed`

这组字段说明当前正式口径已经从“历史上的 DeltaMem 架构想象”收敛为“可验收的 state transport + zero-copy runtime evidence”。

---

## 4. Zero-Copy 与 VRAM Handoff

### 4.1 当前保留的核心能力

旧 `DeltaMem` 文档里真正仍可用的能力，是以下几项底层语义：

- 云侧完成状态导出与压缩
- 端侧只做最小必要的状态恢复
- 利用 `UMA` / shared buffer 降低 CPU 复制次数
- 将恢复后的状态直接交给设备侧继续 decode

### 4.2 当前正式命名

本项目当前不再以 `DeltaMem` 作为主命名，而使用以下正式口径：

- `TrueOrthoKDA active runtime`
- `zero-copy VRAM handoff`
- `state transport`
- `device resume`

### 4.3 当前正式 evidence

当前可引用的正式 runtime evidence 包括：

- `ComputeGraphCompiler-main/Output/cli_gate_m75_trueorthokda_active/m75_trueorthokda_active/m75_trueorthokda_active_report.json`
- `ComputeGraphCompiler-main/Output/cli_gate_m75_trueorthokda_active/runtime_evidence/m75_trueorthokda_active_runtime.json`

这组 evidence 说明当前保留的是：

- `cpu_copy_count = 0`
- `uma_buffer_used = true`
- `device_resume_consumed = true`

而不是继续维护“旧 DeltaMem 叙事中的所有旁支概念”。

---

## 5. API 兼容层与客户端接入

`Edge Engine` 的产品层价值，不只是状态传输，还包括让外部客户端能通过统一入口接上正式 runtime。

当前仍有效的 API / client boundary 包括：

- Ollama 相容路由
- OpenAI 相容工具调用格式
- Anthropic / Claude 类客户端的本地代理接入
- release-facing 入口说明与 client entrypoints

当前可引用的正式 evidence 包括：

- `ComputeGraphCompiler-main/Output/cli_gate_m75/m75_api_compat/m75_report.json`
- `CGC_Release/README.md`

其中 `M7.5 API compatibility` 说明本地 loopback、client entrypoints、tool-call hotfix 与 edge router 证据已经进入正式验收边界。

---

## 6. M6 产品化边界

### 6.1 最小可交付闭环

`M6` 仍是 `Edge Engine` 的正式产品化里程碑，当前最小闭环包括：

- 模板化 bundle 描述
- `cgc build`
- `cgc run`
- `cgc product`
- `cgc verify`

### 6.2 PASS 条件

以 `report.json` 为单一真源，`M6` 当前最小 PASS 条件为：

- `gate_result.m6.build_bundle_gate = PASS`
- `gate_result.m6.run_bundle_gate = PASS`
- 状态层真实落地
- `second.cache_hit = true`
- 端侧与云侧都满足上述条件

### 6.3 当前 evidence

当前仍有效的 `M6` evidence 为：

- 端侧：`/tmp/cgc_m6_local_product_run1/report.json`
- 云侧：`/tmp/cgc_m6_cloud_product_run1/report.json`

这表示 `Edge Engine` 当前不仅是“能跑的边缘代理”，也是已有 bundle、cache、双侧 verify 证据的正式产品化运行层。

---

## 7. 与 M1 / DeltaMem 的关系

### 7.1 与 M1 的关系

`M1` 仍负责：

- `8-step` 的最小工程语义
- `partitions.json`
- `stats.json`
- `taps/`
- pipeline 汇总出的 `report.json`

`Edge Engine` 则负责把这些底层能力接到：

- 可运行的端侧入口
- 可恢复的状态传输
- 可交付的 bundle 与 verify

### 7.2 与 DeltaMem 的关系

`DeltaMem` 现在只保留为历史术语，不再作为当前正式产品命名。

可继承的有效概念为：

- `CGC KV Header`
- raw state bytes / compressor
- state externalization
- zero-copy / `UMA` / VRAM handoff

已经删去的过时概念为：

- `Q2RL` 主线
- `AI Runtime` 中枢化叙事
- `ONNX Runtime` 统一主平台叙事
- 旧 `M7.6 / M8 / Ascend` 蓝图

---

## 8. 当前收口结论

截至当前版本，`CGC Edge Engine` 的正式边界已经收敛为：

- `UPKG_1.1` 负责统一 gate 与跨里程碑契约
- `M1` 白皮书负责最小 `8-step` 基座与 `llama.cpp / ggml-cgc` 产物
- 本文件负责 edge runtime、state transport、zero-copy handoff、API compatibility 与 `M6` 产品化边界
- `DeltaMem` 只保留历史名词与术语迁移说明，不再维护为独立现行架构主文档
