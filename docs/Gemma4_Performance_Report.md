# Gemma 4 26B-A4B 性能测试报告 (更新版)

> 日期: 2026-07-26 | 状态: Active

## 1. 测试环境

| 组件 | 配置 |
|---|---|
| **Mac (端侧)** | M4 16GB, 10核, 120GB/s 内存带宽 |
| **Host2 (云端)** | 8×RTX PRO 5000 72GB Blackwell, 1600GB/s |
| **网络** | mTLS 直连 (RTT ~15ms) |
| **sglang** | 0.5.16 (patched activation kernel) |
| **模型** | Gemma 4 26B-A4B (BF16, MoE 128专家, 3.8B 激活) |
| **MTP head** | google/gemma-4-26B-a4b-it-assistant (76M, 4层) |

## 2. 性能对比 (完整)

### 同场景对比 (Mac→Host2 mTLS 流式)

| 模型 | 框架 | cuda-graph | MTP 投机 | tok/s | 说明 |
|---|---|:---:|:---:|:---:|---|
| V4-Flash 37B | sglang 0.5.13 | ✅ | ❌ | **17.4** | 含网络 RTT |
| **Gemma 4 26B** | **sglang 0.5.16** | **❌** | **✅** | **52.5** | **含网络 RTT** |

### Host2 本地 (无网络)

| 模型 | cuda-graph | MTP | tok/s | 说明 |
|---|:---:|:---:|:---:|---|
| V4-Flash 37B | ✅ | ❌ | **38** | 已知最高 |
| Gemma 4 26B (transformers) | ❌ | ❌ | 24.7 | 无优化基线 |
| Gemma 4 26B (sglang, 无 cuda-graph) | ❌ | ✅ | **待测** | 预期 ~70 |
| Gemma 4 26B (sglang + cuda-graph) | ✅ | ✅ | **待测** | 预期 ~100-115 |

### Mac 本地 (全量加载)

| 模型 | 内存需求 | Mac 8GB 可用 | 可行? |
|---|:---:|:---:|:---:|
| V4-Flash 37B (4bit) | ~18.5GB | ❌ | ❌ 跑不动 |
| Gemma 4 26B (4bit) | ~13GB | ❌ | ❌ 跑不动 |
| Gemma 4 26B (端侧主干 4bit) | ~0.8GB | ✅ | ✅ 需端云混合 |
| MTP head (端侧 draft) | ~0.3GB | ✅ | ✅ 极轻量 |

## 3. Gemma 4 优势分析

### vs V4-Flash

| 维度 | V4-Flash 37B | Gemma 4 26B | 优势 |
|---|---|---|---|
| 总参数 | 37B | 25.2B | Gemma 更小 (0.68x) |
| 激活参数 | ~4B | 3.8B | Gemma 更少 (0.95x) |
| 专家数 | ? | 128 (8+1 active) | Gemma 更多专家 |
| 官方 MTP | ✅ (NEXTN, accept 28%) | ✅ (Assistant, accept ~60%+) | **Gemma accept 更高** |
| 许可证 | 自有 | Apache 2.0 | **Gemma 商用无限制** |
| 端侧适配 | ❌ (37B 太大) | ✅ (3.8B 激活, PLE) | **Gemma 端侧友好** |
| Mac 本地全量 | ❌ | ❌ | 都跑不动 |
| 端云混合 | PD 分离 | MTP + Pipeline | **Gemma 更适合** |

### 52.5 tok/s 加速来源

```
V4-Flash 17.4 tok/s (Mac→Host2, cuda-graph, 无投机)
Gemma 4 52.5 tok/s (Mac→Host2, 无 cuda-graph, 有 MTP)

加速 3.0x 来源:
  ① 模型更小 (26B vs 37B): ~1.4x
  ② MTP 投机 (accept 60%+, batch verify): ~1.8x
  ③ mTLS 直连 (共用): 1x
  总计: 1.4 × 1.8 ≈ 2.5x (实际 3x, MoE batch 效率好)
```

## 4. cuda-graph 修复方案

### 问题
```
Gemma 4 26B BF16: 52GB
TP=8: 6.5GB/卡 权重
mem-fraction 0.75: 47GB KV cache
cuda-graph-max-bs 8: ~16GB
MTP head: ~1.6GB
总计: 6.5 + 47 + 16 + 1.6 = 71.1GB ≈ 72GB (临界)
```

### 修复
```
方案1: mem-fraction 0.65 + cuda-graph-max-bs 4
  6.5 + 39 + 8 + 1.6 = 55.1GB < 72GB ✅
  预期: cuda-graph 加速 1.5-2x → 79-105 tok/s

方案2: mem-fraction 0.70 + cuda-graph-max-bs 2
  6.5 + 42 + 4 + 1.6 = 54.1GB < 72GB ✅
  预期: cuda-graph 加速 1.3-1.5x → 68-79 tok/s
```

## 5. Pipeline 预期

```
当前 (sglang + MTP, 无 cuda-graph, Mac→Host2):
  52.5 tok/s

加 cuda-graph (修复后):
  ~79-105 tok/s (1.5-2x)

加 Pipeline (端侧 MTP head draft + 云端 verify):
  端侧 draft (0.74ms/token, 不占云端)
  云端 batch verify (更快)
  RTT 隐藏 (draft/verify 重叠)
  → ~100-136 tok/s (理论)

最终预期:
  cuda-graph + Pipeline: 100-136 tok/s
```

## 6. 完整性能矩阵

| 场景 | V4-Flash 37B | Gemma 4 26B | 说明 |
|---|:---:|:---:|---|
| Mac 本地全量 | ❌ | ❌ | 都跑不动 (16GB) |
| Host2 本地 (cuda-graph) | 38 tok/s | ~100 (预期) | 待测 |
| Mac→Host2 mTLS (流式) | 17.4 tok/s | 52.5 (已测) | ✅ |
| Mac→Host2 + cuda-graph | ~38 (预期) | ~100 (预期) | 待测 |
| Mac→Host2 + cuda-graph + Pipeline | N/A | ~136 (理论) | 待实现 |
| Mac 端侧 MTP + 云端 Pipeline | N/A | ~85-136 (预期) | 待实现 |

## 7. 结论

```
Gemma 4 26B-A4B vs V4-Flash 37B:

Mac 本地: 都跑不动 (16GB)
Mac→Host2 mTLS: Gemma 52.5 tok/s vs V4-Flash 17.4 tok/s (3x)
Host2 本地: Gemma 预期 ~100 vs V4-Flash 38 (2.6x)

Gemma 4 优势:
  ① 更小 (26B vs 37B)
  ② 官方 MTP head (accept 60%+ vs 28%)
  ③ Apache 2.0 (商用无限制)
  ④ 端侧友好 (PLE + 3.8B 激活)

待修复:
  ① cuda-graph 内存 (mem-fraction 0.65 + bs 4)
  ② Pipeline 实现 (端侧 draft + 云端 verify)
```
