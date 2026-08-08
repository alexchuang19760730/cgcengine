# MTP Head 训练规划

## 目标

为 Qwen3-VL-2B 训练一个轻量 MTP (Multi-Token Prediction) head，用于：
1. **首 token 预测**（端侧首包，降 TTFT 66-87ms）
2. **投机 decode draft**（Mac 本地 decode 加速到 40+ tok/s）

## 背景实测数据

| Draft | forward 估算 | accept rate | decode tok/s | TTFT |
|-------|-------------|-------------|-------------|------|
| 无投机 (baseline) | — | — | 26.1 | 592ms |
| Qwen2.5-0.5B 4bit | ~15ms | 40-43% | 14-18 | 267-671ms |
| Qwen2.5-0.5B BF16 | ~10ms | 57-60% | 14-19 | 347-528ms |
| **MTP head (~50M)** | **~1ms** | **目标 60%+** | **40+ tok/s** | **~350ms** |

**核心问题**：0.5B draft 在 Metal GPU 上 forward 太慢（~10-15ms），超过投机收益。需要 ~50M 参数的 MTP head（forward ~1ms）。

## MTP Head 设计（参考 DeepSeek-V3 论文）

### 架构

```
输入:
  hidden_state (from base model last layer) [batch, seq, hidden_size=2048]
  token_embedding (current token) [batch, seq, hidden_size=2048]

MTP Head 结构 (1 层 transformer):
  ┌─ Concat(hidden_state, token_embedding) → [batch, seq, 2*hidden_size]
  ├─ Linear(2*hidden_size → hidden_size)  # projection
  ├─ RMSNorm
  ├─ Self-Attention (hidden_size, num_heads=16, head_dim=128)
  ├─ RMSNorm
  ├─ MLP (hidden_size → intermediate_size=5632 → hidden_size)
  ├─ RMSNorm
  └─ shared lm_head (hidden_size → vocab_size=151936)  # 与 base model 共享

输出: next_token_logits [batch, seq, vocab_size]
```

### 参数量估算

| 组件 | 参数量 |
|------|--------|
| Projection (2*2048 → 2048) | 8.4M |
| Attention (Q/K/V/O, 2048×2048×4) | 16.8M |
| MLP (2048×5632×2 + 2048×5632) | 23M |
| RMSNorm ×3 | 6K |
| lm_head (shared, 不训练) | 0 |
| **总计** | **~48M** |

### 关键设计决策

1. **共享 lm_head**：与 base model 共享 lm_head（不训练），减少参数量 + 保持词表对齐
2. **共享 embedding**：token_embedding 从 base model 的 embed_tokens 获取（不训练）
3. **1 层 transformer**：足够预测 next token，forward ~1ms
4. **BF16 精度**：不用量化（避免 4bit 的精度损失，accept rate 40%→57% 的教训）

## 训练流程

### 阶段 1: 数据收集（~2h）

用 base model (Qwen3-VL-2B) 在语料上 forward，收集 (hidden_state, next_token) 对：

```python
# 数据收集脚本: collect_mtp_training_data.py
for text in corpus:
    input_ids = tokenizer.encode(text)
    with torch.no_grad():
        outputs = base_model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # 最后一层 [seq, hidden]
        # 训练样本: (hidden_states[i], input_ids[i+1]) for i in range(seq-1)
    save_samples(hidden_states, input_ids)
```

**数据源**：
- 开源对话数据（ShareGPT, OpenOrca, Alpaca）
- 代码数据（The Stack, CodeAlpaca）
- 通用文本（WikiText, C4 子集）
- 目标：~500K 样本（~2GB hidden states）

### 阶段 2: 训练（~3-5 天，单 GPU）

```python
# 训练脚本: train_mtp_head.py
mtp_head = MTPHead(hidden_size=2048, vocab_size=151936, num_heads=16)
optimizer = AdamW(mtp_head.parameters(), lr=1e-4)

for epoch in range(3):
    for hidden_states, next_tokens in dataloader:
        logits = mtp_head(hidden_states, token_embeddings)
        loss = cross_entropy(logits, next_tokens)
        loss.backward()
        optimizer.step()
```

**训练配置**：
- 硬件：Host2 单卡 RTX PRO 5000（24GB）
- batch_size: 32（序列长度 512）
- 学习率: 1e-4 (cosine decay)
- epochs: 3
- 预计时间: ~3-5 天

### 阶段 3: 评估 + 转换

```python
# 评估脚本: eval_mtp_head.py
# 1. 单 token 预测准确率（首包用）
accuracy = eval_single_token_prediction(mtp_head, test_set)
# 2. 投机 accept rate（decode 用）
accept_rate = eval_spec_decode_accept(mtp_head, base_model, test_prompts)
# 3. 转换为 MLX 格式
convert_to_mlx(mtp_head, "qwen3vl_2b_mtp_head_mlx")
```

## 文件结构

```
CGC_Phase2/mtp_head/
├── model.py                 # MTPHead 模型定义
├── collect_data.py          # 数据收集脚本
├── train.py                 # 训练脚本
├── eval.py                  # 评估脚本
├── convert_mlx.py           # PyTorch → MLX 转换
└── README.md                # 本文件
```

## 预期成果

| 指标 | 目标 | 说明 |
|------|------|------|
| 参数量 | ~48M | 轻量 |
| forward 时间 | ~1ms | Metal GPU |
| 首 token 预测准确率 | 80%+ | 用 hidden_L（完整 prefill） |
| 投机 accept rate | 60%+ | 优于 0.5B BF16 的 57-60% |
| TTFT (首包预测) | 66-87ms | cloud prefill + MTP head |
| decode (投机 N=10) | 40+ tok/s | 超过 baseline 26 tok/s |

## 风险 + 备选方案

| 风险 | 备选方案 |
|------|---------|
| 训练准确率不够 (<60%) | 用 EAGLE-2/3 替代（从 hidden states 学习，非 token 级） |
| 训练时间太长 (>7天) | 用更小数据集（100K 样本）+ 更少 epochs（1-2） |
| Metal forward 仍慢 | 用 INT8 量化 MTP head（参数量减半，forward 更快） |
| accept rate 不稳定 | 加 N-gram fallback（零计算成本补充） |

## 与现有 CGC 架构集成

```
Cloud (Host2):
  prefill → emit hidden_L + KV
      ↓ (TCP, 55ms RTT)
Mac:
  recv hidden_L + KV
  ├─ MTP head(hidden_L) → 预测首 token (~1ms) → 返回用户 [TTFT 66-87ms]
  └─ 注入 KV → target decode + MTP 投机 [decode 40+ tok/s]
```

**复用现有组件**：
- `cgc_pd_patch.py` — cloud emit hidden_L + KV
- `cgc_handoff_transport.py` — TCP/NIXL 传输
- `cgc_control_protocol.py` — TrueOrthoKDA 压缩 hidden（256x）
- `mac_mlx_decode_v2.py` — Mac MLX decode + hidden 捕获
- `edge_first_proxy.py` — 路由 + warm cache
