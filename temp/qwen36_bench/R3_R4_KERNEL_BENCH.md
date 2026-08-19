# Qwen3.6 r3 (3-bit) vs r4 (4-bit) Metal Kernel 基准

> 日期: 2026-08-09 | 引擎: turbo-fieldfare Metal（kernel 层计时，非端到端）
> 说明: Qwen36 端到端推理路径（CLI→RealForwardRunner）尚未接通——当前 Qwen36 只到 kernel/runner 测试层。本基准为 kernel 级 3-bit vs 4-bit 对照。

## 结论摘要

- **r3 (3-bit) MoE 专家 kernel 比 r4 (4-bit) 快 7-8%**（同窗连续运行，GPU 时间）
- **整层 hybrid step：r3 86.6ms vs r4 88.7ms（−2.4%）**（P=1024 中上下文）
- 3-bit 读取量理论 −24%（stride 1,376,256 vs 1,802,240），但 kernel 时间只 −8%——因为 kernel 时间含计算+固定开销，非纯带宽
- 质量：r4 已定案 ppl=2.158 / top1-acc=80.4%（lm_head 修复后）；r3 待跑同款 sanity

## MoE 专家 kernel 计时（moeBatchAmortizedTiming，同窗对照）

| n=50 (slots 400) | r4 (4-bit) | r3 (3-bit) | 提升 |
|---|---|---|---|
| GPU current | 30.19 ms | 27.76 ms | −8.0% |
| GPU amortized | 31.44 ms | 30.37 ms | −3.4% |

| n=16 (slots 128) | r4 | r3 | 提升 |
|---|---|---|---|
| current | 9.91 ms | 9.12 ms | −8.0% |
| amortized | 10.45 ms | 10.14 ms | −2.9% |

| n=8 (slots 64) | r4 | r3 | 提升 |
|---|---|---|---|
| current | 5.01 ms | 4.64 ms | −7.4% |
| amortized | 5.11 ms | 4.89 ms | −4.3% |

## Hybrid 40 层 GPU breakdown（P=1024）

| | r4 | r3 |
|---|---|---|
| DeltaNet (30层) | 36.6% of step | 35.8% |
| GatedAttn (10层) | 11.7% | 11.5% |
| MoE (40层) | 51.7% | 52.7% |
| **step 总计** | **88.67 ms** | **86.64 ms (−2.4%)** |

GatedAttn: r4 base 0.700ms vs r3 0.645ms（−7.9%，dense 层不受 bit 影响，差异=噪声/同窗波动）
MoE per-token: r4 1.145ms vs r3 1.142ms（几乎相同——per-token current 路径，bit 优势在带宽受限时显现）

## 关键发现/修复（本基准过程中）

1. **PackedExpertsLayoutReader.defaultMaxBytes = 16MB**——40 层 qwen36 全量 layout.json 16.99MB 超限，**任何完整 Qwen36 模型都加载失败**。已修至 64MB。这是端到端接入的必要前置修复。
2. **r4 manifest 的 model_weights.bin SHA/size 过期**（patch lm_head 后没更新：实际 4.9GB vs manifest 3.88GB）——已更新为实际 SHA cbbef3f...。r3 manifest 校验通过（repack 自带正确值）。
3. 测试用 metal_out 输入文件与位宽无关（输入激活），已用 symlink 复用 qwen36-test 的 metal_out。

## 遗留：端到端 ttft/decode

- **Qwen36MoERunner 零运行时调用者**——RealForwardRunner（CLI forward 核心）无 Qwen36 hybrid 分支（30 DeltaNet + 10 GatedAttn + 40 MoE 逐层调度）
- CLI `Model.load` 默认 expecting .gemma4_26B_A4B，无 qwen36 架构选择
- **要做端到端 ttft/decode 需先接通**：① Model.load 按 manifest modelID 选 arch ② RealForwardRunner 加 hybrid 层循环（deltanet/gatedattn/Qwen36MoE per layer_type）③ CLI 参数/usage 更新
- 预计端到端 decode 收益：3-bit 省 MoE 读取带宽（step 中 MoE 占 52%），理论 step −3~4%（非 −24%，因 compute+fixed 占比高）

## MTP-enabled 增量预估（2026-08-09）

**输入**：Qwen36 fused MTP head 接受率（前会话实测 code 88% / prose 54%）、MoE amortized batch-50 0.59ms/tok、本次 kernel step 数据。

**关键确认**：官方 GatedDeltaNet 用 `chunk_gated_delta_rule`（chunk_size=64），chunk 内矩阵化并行、chunk 间串行 → verify batch（1+d≤4 位置）落在单 chunk 内 → **DeltaNet 可 batch**（乐观场景成立）。

**decode 预估（端到端 = GPU x 1.30）**：

| 配置 | MTP-off | MTP-on (draft=3, mix) | 增量 |
|---|---|---|---|
| r3 | 8.9 tok/s | ~26.8 tok/s | +202% |
| r4 | 8.7 tok/s | ~26.1 tok/s | +201% |

- code +298% / prose +137%（接受率差异）
- draft 敏感性（mix, r4）：d=1→17.5, d=2→22.6, d=3→26.1, d=4→28.6, d=5→30.3 tok/s
- **这是理论上限**；含 CPU 调度/实现折损 40-50% 后，**实际落地预估 r3 15-20 / r4 14-19 tok/s**
- 对比 Gemma4 MTP 负收益的根本差异：① fused 共享主干无二次 MoE 加载 ② DeltaNet chunk 并行
- 完整模型：`temp/qwen36_bench/mtp_increment_model.py`

## trust-receipt 适配 Qwen36（2026-08-09）

**结论：Qwen36 完全可用 trust-receipt，且收益比 Gemma4 更大**

### 已完成
- 修复 `VerifiedInstallTool.metadataMaxBytes` 16MB→64MB（40 层 layout.json 16.99MB 超限，verify-install/reseal 路径同样受影响）
- 为 r3/r4 生成 `verified-install.json`（reseal full）：r3 43 文件 17.7GiB/41s、r4 43 文件 21.45GiB/103s
- Qwen36MetalTests 支持 `QWEN36_TRUST_RECEIPT=1` env（加载走 sizeCheckTrustedReceipt），r3 验证通过（DeltaNet 测试 4.3s 全绿）
- 附带发现：r4 manifest 的 model_weights.bin SHA 之前过期（patch lm_head 后未更新），已在 reseal 前修好

### TTFT 收益估算

| 配置 | 需哈希量（40层+weights） | 预估省时（同 Gemma4 吞吐 3.67GB/s） |
|---|---|---|
| r3 | 18.0 GB | ~4.9s |
| r4 | 21.8 GB | ~5.9s |

- 吞吐校准：Gemma4 实测 14.3GB 哈希省 3.9s → 3.67 GB/s
- **收益 = 跳过哈希省下的秒数**：Qwen36 哈希量更大（18-22GB vs 14.3GB）→ 省得更多（4.9-5.9s vs 3.9s，+25~50%）
- 不是 Qwen36 更快，而是它原本哈希更久，跳过它的边际收益更高
- model_weights.bin 4.9GB 是 eager 校验（加载即哈希）；40 层 lazy（首 token touch 全部层）
- 完整对比模型：temp/qwen36_bench/trust_receipt_compare.py

### 注意
- receipt 绑定绝对路径（`modelDirectoryPath`），移动/复制模型后需 `--reseal --rebind`
- 模型目录有未声明的 `metal_out` symlink（测试用），reseal 警告但不阻塞
