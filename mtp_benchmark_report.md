# TurboFieldfare GPU MTP Benchmark — Baseline vs MTP (2026-08-07)

## 环境
- 设备: Mac M4 16GB
- 模型: Gemma4 26B-A4B (4bit, repacked gturbo), 位于 M4 内部 SSD
  `/Users/alexchuang/Documents/flashkv0516/models/gemma4.gturbo`
- MTP drafter: 官方 `gemma-4-mtp-head` (Gemma4AssistantForCausalLM, 4 层, hidden=1024)
- CLI: `turbo-fieldfare-github-official/.build/release/TurboFieldfareCLI`
- 注意: 先删了 Trae 的 57G 缓存腾出 SSD 空间 (gturbo 原在 USB 盘, 导致 baseline TTFT 高达 97s)

## Benchmark 结果 (max-new=64, temperature=0)

| 场景 | TTFT | decode tok/s | 草稿接受率 | 输出 |
|------|------|------|------|------|
| **Baseline** | **6.3–6.6s** | **9.7–10.5** | — | 正常 (但部分 prompt 退化, 见下) |
| **MTP-GPU** (draft=4) | 7.25s | 7.83 | **0/252 (0%)** | 退化循环 |

## 关键结论
1. **MTP 这条路目前不可用** — 0% 草稿接受率意味着 drafter 的预测从不等于 base 的 greedy token, 每步都被拒, 退化为普通逐 token 解码 (且因额外 draft 前向反而更慢: 7.83 vs 9.7 tok/s)。
2. **确认性测试**: 在 base 能答对的 "1 + 1 =" 上, MTP 仍是 0/44 (0%) accept → 0% 与 prompt 退化无关, 是 **drafter/target 集成 bug**。
3. 内存/磁盘层面无误: MTP 现在能完整跑完 (之前 `Metal function missing` 已修)。

## 已修复
- **Metal 模块未注册**: `MetalContext.swift` 的模块清单漏了 `assistant`, 导致 `assistant_bf16_gemv` 缺失 → MTP 启动即崩。已补 `"assistant": "Metal/Assistant"` 并把文件改名为 `assistant.metal`。重编后 MTP 能运行。

## 0% Accept 根因 (已排查, 仍未修)
约定/权重/维度全部核对无误:
- ✅ 权重 key 名与 drafter 期望完全吻合 (embed_tokens / layers.N.* / norm / pre+post_projection; 无 k/v_proj=复用 target KV; 无 lm_head=tied embed)
- ✅ 架构匹配 config (hidden=1024, 4 层, backbone_hidden=2816, sliding×3+full×1)
- ✅ README 证实约定 `(backbone_hidden_i, embed(token_i)) → token_{i+1}` 与代码吻合
- ✅ `copyLastHiddenState()` 返回 pre-final-norm residual, 符合标准 MTP 约定

**未定位** (需 HF 参考 forward 逐算子对比): forward 计算层面系统性偏差, 最可能:
1. MTP head 的 RoPE/position 对齐 (drafter: full 层用 `encodeProportionalNeox`, sliding 层用 `encodeDefaultNeox`)
2. `layer_scalar` 残差缩放的应用方式
3. backbone hidden 归一化细节与 HF `Gemma4AssistantForCausalLM` 参考实现不符

## Base Model 4-bit 质量警告
- "1 + 1 = 2" 正确 → 权重加载基本正确
- 但 "The capital of France is" → 乱码, Fibonacci prompt → "abilitying" 循环 → 4-bit 量化 + 可能缺 chat template 导致部分 prompt 质量下降, benchmark 数字可测但输出质量不可尽信

## 下一步 (修 MTP 0% accept)
1. 获取/复现 HF `Gemma4AssistantForCausalLM.forward` 参考实现, 逐算子对比 drafter 的 Metal forward
2. 重点核对: backbone hidden 归一化层级、RoPE 实现、layer_scalar 用法
3. 或写 Python 参考 forward (safetensors + 合成 hidden) 做 token 级对齐调试
