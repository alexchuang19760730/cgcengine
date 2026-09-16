# Whittle-MoE Complete Technical Whitepaper

> Dense Qwen3.8-27B to MoE conversion pipeline: architecture, training recipe, pitfalls, M4 deployment
>
> Based on logic65/Whittle-MoE-27B-A18B-v2.1 public info + DenseMixer/smolMoELM open source

---

## 1. Architecture Overview

### 1.1 Dense Parent: Qwen3.8-27B

| Parameter | Value |
|---|---|
| Architecture | Qwen3.5 hybrid (3:1 gated deltanet + full attention) |
| Layers | 64 (16 repeated blocks x 4 layers/block) |
| hidden_size | 5120 |
| FFN intermediate_size | 17408 |
| Attention | 24 heads, 4 KV heads, head_dim=256 |
| Vocab | 248,320 |
| Context | 262,144 tokens |
| Total params | 27B |

### 1.2 MoE Variant: Whittle-MoE-27B-A18B

| Parameter | Value | Notes |
|---|---|---|
| Model type | qwen3_5_moe | Qwen3.5 MoE architecture |
| Experts | 64 | Each layer FFN split into 64 experts |
| Active experts/token | 16 (top-k) | Dynamic routing |
| moe_intermediate_size | 192 | Per-expert FFN width |
| shared_expert_intermediate_size | 5120 | Always-on shared expert |
| **Key arithmetic** | 64 x 192 + 5120 = **17408** | Exactly equals dense FFN width |
| Active params | 17.8B / 27B total | 47.1% FFN activation rate |
| Attention side | **Unchanged** | Keeps original deltanet + full attention |

### 1.3 Why 64 x 192 + 5120 = 17408?

This is the core design: **zero new weights**.

```
Original dense FFN: [gate_proj] [up_proj] [down_proj]  <-- 17408 wide
                         | split
Shared expert:    [gate_proj] [up_proj] [down_proj]  <-- 5120 wide (always active)
64 routed experts:[gate_proj] [up_proj] [down_proj]  <-- each 192 wide (top-16 selected)
                         | verify
64 x 192 + 5120 = 12288 + 5120 = 17408
```

Each token FFN = shared(5120) + 16 routed(16 x 192 = 3072) = **8192 wide** (47.1% of 17408).

### 1.4 Parameter Breakdown

```
Total 27.09B:
  Attention + GDN:  7.42B (unchanged from dense)
  Shared expert:    5.03B
  Routed experts:   12.06B (64 x 192 x 3 proj x 2 scale factors)
  Embeddings:       2.54B
  Router gates:     0.04B (64 layers x 64 experts)

Active 18.03B:
  Attention + GDN:  7.42B
  Shared expert:    5.03B (always active)
  Routed experts:   3.04B (16 x 192 x 3 x 2)
  Embeddings:       2.54B
```

**Floor: 14.96B** (attention + shared expert + embeddings, no routing can go below this).

---

## 2. Stage 1: Dense to MoE Upcycling

### 2.1 Core Principle

**Partition** dense FFN weights into N slices, each becoming an expert. Not copying -- partitioning.

### 2.2 Splitting (MoEfication-style)

```python
import torch

def upcycle_ffn(dense_weights, num_experts=64, expert_dim=192, shared_dim=5120):
    """
    Split dense FFN weights into MoE experts.
    dense_weights keys: gate_proj [5120, 17408], up_proj [5120, 17408], down_proj [17408, 5120]
    """
    shared = {
        'gate_proj': dense_weights['gate_proj'][:, :shared_dim],
        'up_proj':   dense_weights['up_proj'][:, :shared_dim],
        'down_proj': dense_weights['down_proj'][:shared_dim, :],
    }
    experts = {}
    for i in range(num_experts):
        start = shared_dim + i * expert_dim
        end = start + expert_dim
        experts[i] = {
            'gate_proj': dense_weights['gate_proj'][:, start:end],
            'up_proj':   dense_weights['up_proj'][:, start:end],
            'down_proj': dense_weights['down_proj'][start:end, :],
        }
    return shared, experts
```

### 2.3 Two Scale Folds in down_proj

```python
# 1. Shared gate scaling: sigmoid(0) = 0.5
shared_down_proj = dense_down_proj[:shared_dim, :] * 0.5

# 2. Top-k normalization scaling for routed experts
routed_down_proj = dense_down_proj[shared_dim:, :] * (1.0 / top_k)
```

**Why scale?** Shared expert is always active, routed experts only 16/64 = 25%. Without scaling, shared contribution drowns out routed (or vice versa).

### 2.4 Router Initialization

```python
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, hidden_size=5120, num_experts=64):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, hidden_state):
        logits = self.gate(hidden_state)
        top_k_logits, top_k_indices = torch.topk(logits, k=16, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)
        return top_k_weights, top_k_indices
```

### 2.5 Post-upcycle Verification

```python
def verify_upcycle(dense_weights, shared, experts, num_experts=64):
    reconstructed = torch.cat(
        [shared['down_proj']] + [experts[i]['down_proj'] for i in range(num_experts)], dim=0
    )
    assert torch.allclose(reconstructed, dense_weights['down_proj'], atol=1e-6)
    print("Split verified: reconstructed == original weights")
```

---

## 3. Stage 2: Router Training (Router-Healing)

### 3.1 Goal

Train 64 router gates (one per layer), **freeze all expert weights**.

### 3.2 Why Only Router?

- Expert weights from dense copy already contain knowledge
- Router just needs to learn "which token goes to which expert"
- Tiny training: 64 layers x 64 experts x 5120 = ~21M params (0.08% of total)

### 3.3 Training Recipe

```python
# CRITICAL: Training router alone is HARMFUL!
# logic65 experiment: router-only training doubled Unknown answers
# Router and experts are co-adapted, cannot move independently

# Correct: DenseMixer STE (Straight-Through Estimator)
# export DENSEMIXER_ENABLED=1
# Forward: computes ALL expert outputs (dense mode)
# Backward: STE for more precise router gradients
# Cost: 1.46x FLOPs vs conventional training
```

### 3.4 Training Data

```python
# Router distillation: dense model logits as teacher
# Router needs to learn "correct routing decisions"
# Loss: 1) KL divergence  2) load-balancing  3) router z-loss
```

### 3.5 Load Balancing

```python
def load_balancing_loss(router_logits, num_experts=64):
    probs = torch.softmax(router_logits, dim=-1)
    freq = probs.mean(dim=[0, 1])
    target = torch.ones(num_experts) / num_experts
    return F.kl_div(freq.log(), target, reduction='sum')
```

### 3.6 Training Hyperparameters

```yaml
optimizer: paged_adamw_8bit  # NOT adamw_torch (fp32 state = 8.2GB)
learning_rate: 1e-4
lr_scheduler: cosine
warmup_steps: 5%
batch_size: 4-8
max_seq_length: 4096
gradient_checkpointing: true
save_steps: 5           # save_steps <= steps/3
save_total_limit: 3
```

---

## 4. Stage 3: Anti-Loop Distillation (v2 Core)

### 4.1 The Problem: 69% Loop Rate

Root causes:
1. 64 experts copied from same dense weights -> identical
2. Router falls into local optima, repeatedly selecting same expert
3. FFN capacity inflates but activation shrinks -> poor long-sequence tracking
4. **Most critical: model doesn't know "when to stop"**

### 4.2 Core Insight: Looping = Failure to Stop

logic65's key finding (5 experiments):

```
Repetition and truncation are TWO ENDS OF ONE AXIS, not two independent bugs.

Long document data    -> teaches continuation (never stopping)
On-policy data        -> teaches stopping (but at wrong time)
Complete answer data  -> teaches "stop because finished" (currently missing)
```

### 4.3 Dataset Design (v2)

```python
# 1. Long-context teacher-forced KD
#    FineWeb-Edu complete docs, 3000-8000 tokens, 874 docs, 4.00M tokens
#    Every row must end on real EOS

# 2. On-policy KD (GKD paper, arXiv 2306.13649)
#    Release Q4 generates degenerate outputs, dense teacher scores them
#    142 degenerate generations

# 3. Complete correct answers (245 entries)
#    Dense model generates with top-32 logprobs
#    Coverage: short factual -> long enumeration -> code -> multi-turn
```

### 4.4 KD Loss Function (Three Parts)

```python
def anti_loop_kd_loss(student_logits, teacher_logits, true_tokens, on_policy_mask):
    """Three-part KD loss. V = 248320"""
    # 1. Binary KL: probability mass inside vs outside top-k set
    top_k_mask = get_top_k_mask(teacher_logits, k=64)
    student_in = student_logits[top_k_mask].sum(dim=-1)
    teacher_in = teacher_logits[top_k_mask].sum(dim=-1)
    binary_kl = F.k
    binary_kl = F.kl_div(student_in.log(), teacher_in)

    # 2. Conditional KL: shape within top-k set
    student_cond = F.log_softmax(student_logits[top_k_mask], dim=-1)
    teacher_cond = F.softmax(teacher_logits[top_k_mask], dim=-1)
    conditional_kl = F.kl_div(student_cond, teacher_cond)

    # 3. CE anchor: true next token
    ce = F.cross_entropy(student_logits.view(-1, V), true_tokens.view(-1))

    # 4. On-policy rows: mask EOS (CRITICAL!)
    eos_mask = (true_tokens == IM_END) & on_policy_mask
    teacher_logits[eos_mask] = -1e30

    return binary_kl + conditional_kl + ce
```

### 4.5 Critical Pitfall: EOS Masking Must Be Per-Row

```python
# WRONG: Global EOS mask -> model cannot stop -> 90% loop rate
# CORRECT: Only mask EOS on on-policy rows
#    Clean docs keep EOS (teaches "correct stopping")
#    On-policy rows mask EOS (teach "what to say" not "to stop")
```

### 4.6 Training Hyperparameters

```yaml
optimizer: paged_adamw_8bit
learning_rate: 1e-5
lr_scheduler: cosine
warmup_steps: 5%
batch_size: 2-4
max_seq_length: 8192
gradient_checkpointing: true
```

### 4.7 Windowed Teacher Pass

```python
def windowed_teacher_forward(teacher, input_ids, window_size=2048):
    all_top64 = []
    past_kv = None
    for start in range(0, len(input_ids), window_size):
        end = min(start + window_size, len(input_ids))
        chunk = input_ids[start:end]
        outputs = teacher(chunk, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits
        top64_logits, top64_indices = torch.topk(logits, k=64, dim=-1)
        all_top64.append((top64_logits.bfloat16(), top64_indices))
    return all_top64
```

---

## 5. Stage 4: Answer Distillation (v2.1 Balance)

v2 fixed looping (69%->11%) but introduced truncation. v2.1 balances by expanding answer diversity.

### v2.1 Measured Results

| Metric | Baseline | v2 | v2.1 |
|---|---|---|---|
| Single-turn loop rate | 69% | 11% | **8%** |
| Multi-turn loop rate | 64% | 7% | **7%** |
| Structured output loop rate | ~75% | 39% | **22%** |
| Median answer length | 268 words | 386 words | **388 words** |
| Knowledge battery | 28/39 | 27/39 | **28/39** |

---

## 6. M4 Training Practical Configuration

### 6.1 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Memory | 32GB unified | 48GB+ (M4 Pro/Max) |
| Storage | 100GB+ SSD | 200GB+ |
| Training time | ~2-4 hours (router) | ~6-12 hours (anti-loop KD) |

### 6.2 Quick Start

```bash
pip install mlx mlx-lm densemixer transformers
densemixer setup
huggingface-cli download Qwen/Qwen3.8-27B --local-dir ./models/dense
huggingface-cli download logic65/whittle-teacher32-complete-answers --local-dir ./data/teacher
# Stage 1: Upcycling (~10 min)
# Stage 2: Router training (~1-2 hours)
# Stage 3: Anti-loop KD (~4-6 hours)
# Stage 4: v2.1 balance (~1-2 hours)
```

---

## 7. Pitfalls (WHITTLE_FINDINGS Highlights)

| Pitfall | Consequence | Fix |
|---|---|---|
| Router-only training | Unknown answers doubled | Co-adapted, cannot move independently |
| adamw_torch default | fp32 state = 8.2GB | Use paged_adamw_8bit |
| EOS masking global | Model cannot stop -> 90% loop | Per-row masking only |
| Repetition = failure to stop | Not two bugs, one axis | Complete answer distillation |

---

## 8. References

| Project | Link |
|---|---|
| DenseMixer | github.com/yaof20/DenseMixer |
| smolMoELM-custom | github.com/pranavktrpl/smolMoELM-custom |
| MLX | github.com/ml-explore/mlx |
| mlx-lm | github.com/ml-explore/mlx-lm |
| whittle-teacher32 | logic65/whittle-teacher32-complete-answers |

---

*Based on logic65/Whittle-MoE-27B-A18B-v2.1 model card, WHITTLE_FINDINGS.md, DenseMixer, smolMoELM-custom. Last updated: 2026-08-26*
