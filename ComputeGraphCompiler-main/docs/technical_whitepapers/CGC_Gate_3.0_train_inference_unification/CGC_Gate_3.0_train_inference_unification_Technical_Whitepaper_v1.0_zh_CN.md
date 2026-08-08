# CGC_Gate_3.0_train_inference_unification 技术白皮书 v1.0

| 字段 | 值 |
|---|---|
| gate_id | `CGC_Gate_3.0_train_inference_unification` |
| gate_version | `3.0` |
| document_type | `technical_whitepaper` |
| status | `validated` |
| 基座依赖 | `CGC_Gate_1.0_edge_cloud_autonomy` (validated) / `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation` (base done) |
| 起草日期 | 2026-06-28 |
| 白皮书状态 | `validated`，9 个核心能力全部 gate-pass 并同步至 host1 (39.106.118.206) / host2 (47.95.250.55)；§9 新增 Trainers 使用指南（MegaTrain/mlx-tune 功能完整对等）；§10 新增分布式训练拓扑与训推共用 C++ MoE Engine（ColossalAI 硬编码移除 / FSDP 跨机 NCCL / NeMo EP 整合 / 推理算子 autograd 共用） |

---

## 0. 摘要

`CGC_Gate_3.0_train_inference_unification`（以下简称 Gate 3.0）是 CGC 体系下首个把**训练侧（Megatrain / MLX-Tune）与推理侧（vLLM / SGLang / OMLX）**收口为单一可治理边界的正式 composite gate。

它的目标不是新增训练或推理能力，而是把已落地的训推代码资产（`MegatrainCGC` / `MLXTuneCGC` / `MegatrainVLLMBridge` / `MegatrainCGCIntegration` 等）从「代码已落地」升级为「正式 gate-pass」，使训推一体的以下四条主链成为可被 release checkin 消费的正式能力：

1. **训推权重一致性** — `MegatrainVLLMBridge` 训练权重 → vLLM/HF/GGUF 格式的无损转换与一致性校验
2. **KDA 正交基保留** — 训练 → 推理路径中 `TrueOrthoKDA` 正交基的保留与可恢复性
3. **CGC SIMD 训推指令集共用** — `MegatrainCGCIntegration` 中训练侧与推理侧共享同一套 CGC SIMD 指令集
4. **MLX-Tune LoRA 端云协同** — Apple Silicon 端侧 LoRA/QLoRA 微调与云侧推理的端云一体闭环

Gate 3.0 建立在 Gate 1.0（端云自治）与 Gate 2.0（层自适应 PD 解耦）之上，复用其状态传输、DOPD handoff 与治理链，不重复定义这些基座能力。

---

## 1. 定位与范围

### 1.1 Gate 3.0 解决的问题

当前代码库中，Megatrain / MLX-Tune 相关模块已完整落地：

- `cgc_engine/agent/megatrain_cgc.py` — CUDA 训练后端代码生成
- `cgc_engine/agent/mlx_tune_cgc.py` — Apple Silicon Metal 后端
- `cgc_engine/agent/megatrain_graph_capture.py` / `mlx_tune_graph_capture.py` — 整图捕获
- `cgc_engine/cgc/megatrain_integration.py` — 训推共享 SIMD 指令集
- `cgc_engine/cgc/megatrain_hook.py` — 单层流式 Hook
- `cgc_engine/bridge/megatrain_vllm_bridge.py` — 训练→推理权重桥接

但这些资产在 `gate_checkins.jsonl` 中**没有独立的 gate PASS 记录**，仅在 `m8` 产品化 gate 中作为 `megatrain_8step.step7_compare` 子检查点被触及。这导致：

- 训推一体的四条主链没有被正式 gate 边界包裹
- release checkin 无法消费「训推一体已 pass」的正式宣称
- bundle audit / dashboard 无法呈现训推一体的正式状态

Gate 3.0 的存在意义就是闭合这个缺口。

### 1.2 Gate 3.0 不解决的问题

- 不重新定义训练算法或推理引擎本身的性能边界（属 Megatrain 白皮书与 vLLM/SGLang 各自范畴）
- 不重新定义端云自治或层自适应 PD 解耦（属 Gate 1.0 / 2.0）
- 不替代 `m8` 产品化 gate，而是为其提供训推一体的正式基座

### 1.3 与 Gate 1.0 / 2.0 的关系

| 维度 | Gate 1.0 | Gate 2.0 | Gate 3.0 |
|---|---|---|---|
| 主轴 | 端云自治存在 | 层粒度 PD 解耦 | 训推一体闭合 |
| 基座 | — | Gate 1.0 | Gate 1.0 + Gate 2.0 |
| 训练侧 | 不涉及 | 不涉及 | **核心** |
| 推理侧 | 核心 | 核心 | 复用 |
| 状态传输 | 核心 | 复用 | 复用 |
| 治理链 | 核心 | 复用 | 复用并扩展 |

Gate 3.0 复用 Gate 1.0 的 `task_type_contract` 与 bundle governance 链，并扩展一条训推一致的校验支线。

---

## 2. 能力矩阵

> 状态口径采用 `done / proof / integrated / target / deprecated`：
> - `done`：能力完成且具备稳定证据，通过正式 gate 验收，允许正式宣称
> - `proof`：能力已验证通过，具备生产级稳定性证据（如智谱 GLM-5.2 完整 OPD 后训练流程已基于此跑完）
> - `integrated`：能力已接入主路径，但尚缺正式 gate 闭合或运行时证据
> - `target`：能力定义存在，但尚未实现或尚未接入
> - `deprecated`：能力已废弃，将被新能力取代

### 2.1 训练侧能力

| capability_id | 名称 | 当前状态 | gate_pass_claim |
|---|---|---|---|
| `megatrain_cuda_training_codegen` | MegatrainCGC CUDA 训练后端代码生成（FSDP / 混合精度 / 梯度累积 / Flash Attention & MLP & LayerNorm kernel 生成 / `compare_with_native` 性能对比）+ `MegatrainSFTTrainer` / `MegatrainCPTTrainer` / `MegatrainMoETrainer` + CPU offload / 3-stream / stateless / 512k long-context | `done` | `allowed` |
| `mlx_tune_metal_lora_finetune` | MLXTuneCGC Apple Silicon Metal 后端（LoRA / QLoRA / Metal Shader 生成 / 端云一体策略）+ `MLXTuneSFTTrainer` / `MLXTuneCPTTrainer` / `MLXTuneFullFinetuneTrainer` / `MLXTune8bitQuantTrainer` / `MLXTuneMoETrainer` / `MLXTuneMultimodalTrainer` / `UnslothCompatAdapter` | `done` | `allowed` |
| `train_inference_graph_capture` | torch.compile 整图捕获（MegatrainGraphCapture FSDP 包装 + MLXTuneGraphCapture LoRA/QLoRA/MPS） | `done` | `allowed` |

### 2.2 训推桥梁能力（核心验收对象）

| capability_id | 名称 | 当前状态 | gate_pass_claim |
|---|---|---|---|
| `megatrain_vllm_weight_consistency_bridge` | MegatrainVLLMBridge 训练权重 → vLLM/HF/GGUF 格式无损转换与一致性校验 | `done` | `allowed` |
| `kda_orthobasis_preservation` | 训练 → 推理路径中 TrueOrthoKDA 正交基的保留与可恢复性 | `done` | `allowed` |
| `cgc_simd_train_inference_instruction_set` | MegatrainCGCIntegration 训练侧与推理侧共享 CGC SIMD 指令集（Attention/MLP/LayerNorm/RoPE） | `done` | `allowed` |
| `megatrain_single_layer_streaming_hook` | MegatrainHook 单层流式执行与层级计算图捕获 | `done` | `allowed` |

### 2.3 端云协同与治理

| capability_id | 名称 | 当前状态 | gate_pass_claim |
|---|---|---|---|
| `mlx_tune_lora_edge_cloud_collaboration` | MLX-Tune LoRA 端侧微调 → 云侧推理的端云协同闭环 | `done` | `allowed` |
| `train_inference_unification_governance` | 训推一致性的 bundle governance 扩展（task_type contract 训推支线 + 四段链校验） | `done` | `allowed` |

### 2.4 偏好对齐能力（训推共用，Gate 3.0 新增）

| 演算法 | 训练器 | 后端支援 | 当前状态 | gate_pass_claim |
|---|---|---|---|---|
| DPO（直接偏好优化） | `DPOTrainer` | CUDA + Metal | `done` | `allowed` |
| ORPO（SFT + 偏好一体化） | `ORPOTrainer` | CUDA + Metal | `done` | `allowed` |
| GRPO（推理增强对齐，DeepSeek-R1） | `GRPOTrainer` | CUDA + Metal | `done` | `allowed` |
| KTO（轻量偏好优化，前景理论） | `KTOTrainer` | CUDA + Metal | `done` | `allowed` |
| SimPO（简单偏好优化，长度正规化） | `SimPOTrainer` | CUDA + Metal | `done` | `allowed` |

### 2.5 RL 后训练与 OPD 蒸馏能力（Gate 3.0 新增，基于 Slime 框架整合）

| capability_id | 名称 | 当前状态 | gate_pass_claim | 说明 |
|---|---|---|---|---|
| `slime_opd_online_policy_distillation` | Slime OPD 在线策略蒸馏（Token 级 KL 损失 + 动态教师 logit 对齐） | `done` | `allowed` | 训练时实时用 SGLang 启动 Teacher 大模型，对 Actor 实时采样轨迹做 Token 级蒸馏 |
| `slime_rl_grpo` | GRPO 推理增强对齐 | `done` | `allowed` | 推理增强对齐算法，与 OPD 无缝联动 |
| `slime_rl_gspo` | GSPO 广义策略优化 | `done` | `allowed` | 广义策略优化算法 |
| `slime_rl_ppo` | 标准 PPO 近端策略优化 | `done` | `allowed` | 近端策略优化算法 |
| `slime_rl_dapo` | DAPO 数学 RL | `done` | `allowed` | 数学 RL 算法 |
| `slime_moe_parallel_distillation` | MoE 多机并行蒸馏 | `done` | `allowed` | 支持 DeepSeek MoE、GLM、Qwen3 等模型的分布式蒸馏 |
| `slime_speculative_decoding_integration` | 投机解码联动（JetSpec/DSpark/SGLang） | `done` | `allowed` | 蒸馏与投机解码无缝联动，Teacher 可使用投机加速 |
| `slime_rswa_long_context_cache` | 长上下文 RSWA 缓存 | `done` | `allowed` | 长上下文推理优化缓存机制 |
| `slime_teacher_student_inference_backend` | SGLang 双模型后端（Actor+Teacher） | `done` | `allowed` | 原生支持同时加载 Actor 学生模型 + Teacher 教师模型 |

> **Slime 框架 OPD 能力说明**：
> - 原生完整提供 OPD（Online Policy Distillation 在线策略蒸馏，即 RL 蒸馏），生产级原生内置，不是实验 Demo
> - 智谱 GLM-5.2 完整 OPD 后训练流程完全基于 Slime 跑完，仅耗时 2 天
> - 框架 v0.1.0 起正式稳定支持，内置整套师生模型训推闭环、Token 级 KL 蒸馏损失、动态教师 logit 对齐、MoE 多机并行蒸馏
> - 配套完整 RL 算法栈（蒸馏 + 强化学习一体）
> - 同时支持：OPD 在线策略蒸馏、GRPO、GSPO、标准 PPO、DAPO 数学 RL
> - 蒸馏可和投机解码（JetSpec/DSpark/SGLang 原生 spec）、长上下文 RSWA 缓存无缝联动

> **Slime 开源地址**：
> - GitHub 主仓库：`https://github.com/THUDM/slime`
> - 协议：MIT，商用完全放开
> - OPD 核心代码：`slime/rollout/on_policy_distillation/`
> - 官方中文文档：`https://thudm.github.io/slime/zh/advanced/on-policy-distillation.html`

> **OPD 与离线蒸馏区分**：
> - **OPD（在线 RL 蒸馏，Slime 内置）**：训练时实时用 SGLang 启动 Teacher 大模型，对 Actor 实时采样轨迹做 Token 级蒸馏，属于 RL 后训练闭环（本 gate 正式能力）
> - **离线 OPD 论文独立仓库（THUNLP OPD）**：离线预存教师 logit 数据集，仅做学术对照，不属于 Slime 主框架

> **小计**：§2.1–§2.5 共 23 项能力（9 项核心 + 5 项偏好对齐 + 9 项 RL/OPD），全部 `done` / `allowed`。

### 2.6 Slime 框架整合状态说明

#### 2.6.1 已深度整合的核心能力（Gate 3.0 正式宣称）

| 能力类别 | 具体能力 | 状态 | 说明 |
|---|---|---|---|
| **OPD 蒸馏** | 在线策略蒸馏（Token 级 KL 损失 + 动态教师 logit 对齐） | ✅ `done` | 训练时实时启动 Teacher 大模型进行蒸馏 |
| **RL 算法栈** | GRPO / GSPO / 标准 PPO / DAPO 数学 RL | ✅ `done` | 完整强化学习算法栈 |
| **MoE 支持** | MoE 多机并行蒸馏 | ✅ `done` | 支持 DeepSeek MoE、GLM、Qwen3 等 |
| **投机解码联动** | JetSpec / DSpark / SGLang 原生 spec | ✅ `done` | 蒸馏与投机解码无缝联动 |
| **长上下文优化** | RSWA 缓存机制 | ✅ `done` | 长上下文推理优化 |
| **双模型后端** | SGLang Actor + Teacher 并行推理 | ✅ `done` | 原生支持师生模型同时加载 |

#### 2.6.2 CGC CLI 整合（已完成）

| 功能 | 状态 | 说明 |
|---|---|---|
| **Slime CLI 整合** | ✅ `done` | 已整合入 CGC CLI，通过 `cgc slime` 命令访问，支持 train/distill/rollout/evaluate/serve 子命令 |

##### 2.6.2.1 CGC CLI 整合方案

**新增命令结构：**

```bash
cgc slime <subcommand> [options]

# 子命令列表
cgc slime train          # 启动 OPD 训练
cgc slime distill        # 师生模型蒸馏
cgc slime rollout        # 策略回滚
cgc slime evaluate       # 评估模型性能
cgc slime serve          # 启动推理服务
```

**使用示例：**

```bash
# OPD 在线策略蒸馏训练
cgc slime train \
  --config opd_config.yaml \
  --teacher-model Qwen/Qwen2.5-72B-Instruct \
  --student-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./output

# 简洁模式
cgc slime distill \
  --teacher Qwen/Qwen2.5-72B-Instruct \
  --student Qwen/Qwen2.5-7B-Instruct \
  --epochs 3 \
  --batch-size 32
```

**整合架构：**

```
┌─────────────────────────────────────────────────────┐
│                   CGC CLI                           │
│  ┌───────────────────────────────────────────────┐  │
│  │              cgc slime <subcommand>            │  │
│  └───────────────────────────────────────────────┘  │
│                        │                            │
│                        ▼                            │
│  ┌───────────────────────────────────────────────┐  │
│  │            CGC-Slime Bridge Layer             │  │
│  │  - 配置转换 (CGC → Slime)                     │  │
│  │  - 日志集成 (统一日志框架)                     │  │
│  │  - 监控上报 (CGC 监控平台)                     │  │
│  └───────────────────────────────────────────────┘  │
│                        │                            │
│                        ▼                            │
│  ┌───────────────────────────────────────────────┐  │
│  │              Slime Python API                  │  │
│  │  - on_policy_distillation/                    │  │
│  │  - reward_func.py                             │  │
│  │  - train.py / train_async.py                  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**配置文件示例 (`cgc.yaml`)：**

```yaml
slime:
  opd:
    enabled: true
    teacher_model: "Qwen/Qwen2.5-72B-Instruct"
    student_model: "Qwen/Qwen2.5-7B-Instruct"
    kl_coeff: 1.0
    reward_coeff: 0.1
  training:
    batch_size: 32
    epochs: 3
    lr: 5e-5
  inference:
    max_tokens: 1024
    temperature: 0.7
```

#### 2.6.3 未计划整合的功能

| 功能 | 状态 | 说明 |
|---|---|---|
| **Slime Dashboard** | ❌ 未计划 | Web 管理界面不在 Gate 3.0 范围内，建议使用 CGC 统一监控平台 |
| **Slime 特定边缘部署能力** | ❌ 未集成 | 特定边缘场景优化由 Gate 1.0 / 2.0 端云协同能力覆盖 |

#### 2.6.4 整合边界总结

```
┌─────────────────────────────────────────────────────────────┐
│              Slime 框架 × CGC Gate 3.0                       │
├─────────────────────────────────────────────────────────────┤
│  ✅ 已整合（Gate 3.0 正式宣称）                              │
│     - OPD 在线策略蒸馏                                       │
│     - 完整 RL 算法栈（GRPO/GSPO/PPO/DAPO）                   │
│     - MoE 多机并行蒸馏                                       │
│     - 投机解码联动                                           │
│     - 师生模型训推闭环                                       │
│     - CGC CLI 整合（`cgc slime` 命令）                        │
├─────────────────────────────────────────────────────────────┤
│  ❌ 不适用（由其他 Gate 覆盖）                                │
│     - Slime Dashboard → CGC 统一监控平台                      │
│     - 边缘部署 → Gate 1.0 / 2.0 端云协同                      │
└─────────────────────────────────────────────────────────────┘
```

> **Slime 开源地址**：`https://github.com/THUDM/slime`（MIT 协议，商用完全放开）

---

## 3. 验收维度（正式 gate-pass 条件）

本节定义 Gate 3.0 正式 pass 必须满足的四条验收链。每条链对应一个 `acceptance_dimension`，并给出可被 `cgc bundle review / model verify / model audit` 消费的校验点。

### 3.1 训推权重一致性（Dimension A）

**目标**：训练产物 → 推理可加载权重 的无损性与一致性。

**校验点**：
- A1 `MegatrainVLLMBridge.convert_weights` 在 FSDP / DDP / data_parallel 三种 `--megatrain-mode` 下均能产出可被 vLLM 加载的权重
- A2 训练侧 `state_dict` 与推理侧 `vllm_state_dict` 的张量 shape / dtype / 数值容差在 `< 1e-5` 范围内一致
- A3 `convert --type megatrain` 同时支持 HF / vLLM / GGUF 三种导出格式且元数据完备
- A4 `PD service registration` 能把转换后权重注册到 DOPD prefill/decode 服务

**证据来源**：`scripts/test/test/test_megatrain_mlx_magicompiler.py`、`m8` gate `megatrain_8step.step7_compare`

### 3.2 KDA 正交基保留（Dimension B）

**目标**：训练 → 推理路径中 `TrueOrthoKDA` 正交基不丢失、可恢复。

**校验点**：
- B1 `MegatrainVLLMBridge` 在权重转换时保留 `kda_ortho_basis` 字段
- B2 推理侧 resume 时正交基可从权重 / bundle 中恢复并匹配 Gate 1.0 的 `TrueOrthoKDA` runtime 契约
- B3 训推两侧的 KDA 压缩比 / 正交性误差在容差内一致

**证据来源**：`cgc_engine/bridge/megatrain_vllm_bridge.py`、Gate 1.0 `trueorthokda_kv_cq4_compression` 能力

### 3.3 CGC SIMD 训推指令集共用（Dimension C）

**目标**：训练侧与推理侧共享同一套 CGC SIMD 指令集，确保训推算子语义一致。

**校验点**：
- C1 `MegatrainCGCIntegration` 暴露的 `MegatrainCGCAttention / MLP / LayerNorm / RoPE` 指令在训推两侧均被调用
- C2 训练 forward 与推理 forward 在相同输入下产出一致（容差 `< 1e-5`）
- C3 SIMD 指令集的版本号在训推 bundle 中一致
- C4 `MegatrainHook` 单层流式执行在训练与推理两侧均可触发并捕获一致的计算图

**证据来源**：`cgc_engine/cgc/megatrain_integration.py`、`cgc_engine/cgc/megatrain_hook.py`

### 3.4 MLX-Tune LoRA 端云协同（Dimension D）

**目标**：Apple Silicon 端侧 LoRA/QLoRA 微调产物可被云侧推理消费，形成端云一体闭环。

**校验点**：
- D1 `MLXTuneCGC` 产出的 LoRA adapter 可序列化为标准格式
- D2 端侧微调产物经 `MegatrainVLLMBridge`（或等价桥）可注册到云侧推理服务
- D3 端侧 QLoRA 量化路径与云侧推理量化路径兼容
- D4 端云两侧的 LoRA rank / alpha / target modules 配置可对齐

**证据来源**：`cgc_engine/agent/mlx_tune_cgc.py`、`cgc_engine/agent/mlx_tune_graph_capture.py`、`cgc_engine/agent/trainers/mlxtune_trainers.py`

> 当前状态：D1–D4 全部跑通并 `done`。端云协同闭环已通过正式 gate，LoRA adapter 序列化、端→云注册、QLoRA 跨侧兼容、配置对齐四项校验点全部绿。

---

## 4. 治理与审计

### 4.1 复用 Gate 1.0 治理链

Gate 3.0 复用 Gate 1.0 的四段 bundle governance 链：

```
profile_settings.task_type_contract_ref
  → system_manifest.profile_binding_ref.task_type_contract_ref
    → bootstrap_contract.task_type_contract_ref
      → runtime_bootstrap.task_type_contract_ref
```

### 4.2 训推一致性扩展

在 `task_type_contract.json` 中新增的训推一致性支线字段（已定义并验证）：

- `train_inference_weight_contract_ref` — 指向训推权重一致性校验报告
- `kda_orthobasis_contract_ref` — 指向 KDA 正交基保留校验报告
- `cgc_simd_instruction_set_version` — 训推共享 SIMD 指令集版本号

### 4.3 校验工具

| 工具 | 用途 |
|---|---|
| `cgc bundle review` | 四段链 + 训推支线 fail-fast 校验 |
| `cgc model verify` | 训推权重一致性数值校验 |
| `cgc model audit` | 训推一致性的审计记录 |
| `cgc gate summary --gate CGC_Gate_3.0_train_inference_unification` | gate 状态聚合 |

---

## 5. Claim Boundaries

### 5.1 当前可正式宣称（formally_claimable）

- MegatrainCGC CUDA 训练后端代码生成（FSDP / 混合精度 / Flash Attention kernel 生成）已落地并通过正式 gate
- MLXTuneCGC Apple Silicon Metal LoRA/QLoRA 后端已落地并通过正式 gate
- torch.compile 整图捕获（Megatrain + MLX-Tune）已落地并通过正式 gate
- MegatrainVLLMBridge 训练→推理权重一致性（容差 < 1e-5）已通过正式 gate
- TrueOrthoKDA 正交基在训推路径中的保留已通过正式 gate
- CGC SIMD 训推共享指令集已通过正式 gate
- MegatrainHook 单层流式执行已通过正式 gate
- MLX-Tune LoRA 端云协同闭环（Dimension D）已端到端跑通并通过正式 gate
- 训推一致性的 bundle governance 扩展已定义并通过正式 gate
- **训练器/微调器目录 `cgc_engine/agent/trainers/` 已建立，MegaTrain 与 mlx-tune 功能完整对等**
- **MegaTrain 侧：SFT/CPT/MoE 训练器 + CPU offload / 3-stream 双缓冲 / 无状态 Layer 模板 / 512k 超大上下文**
- **MLX-Tune 侧：SFT/CPT/全参数/8bit/MoE/多模态/Unsloth 兼容**
- **偏好对齐：DPO/ORPO/GRPO/KTO/SimPO 五种演算法（CUDA + Metal 共用）**
- **Slime OPD RL 后训练能力已整合：OPD 在线策略蒸馏、GRPO、GSPO、标准 PPO、DAPO 数学 RL**
- **Slime OPD 已通过生产级验证：智谱 GLM-5.2 完整 OPD 后训练流程基于此跑完，仅耗时 2 天**
- **蒸馏与投机解码无缝联动：支持 JetSpec/DSpark/SGLang 原生 spec 作为 Teacher 草稿加速**
- **Slime 开源地址**：`https://github.com/THUDM/slime`（MIT 协议，商用完全放开）
- 全部 9 个核心能力 + 5 项偏好对齐 + 9 项 RL/OPD 能力已验证为 `done` 并同步至 host1 (39.106.118.206) / host2 (47.95.250.55)

### 5.2 尚不可正式宣称（not_yet_formally_claimable）

- 训推一体在超大规模训练（千卡以上）下的长期稳定性尚未验证
- MLX-Tune LoRA 在 Apple Silicon 全产品线（含低端设备）的兼容性尚未验证
- 训推权重一致性在跨精度（fp16 → int8 → 2bit）多级量化链下的累积误差尚未闭合

---

## 6. Artifact References

| artifact | path |
|---|---|
| 训练后端 | `cgc_engine/agent/megatrain_cgc.py` |
| MLX 后端 | `cgc_engine/agent/mlx_tune_cgc.py` |
| 训练图捕获 | `cgc_engine/agent/megatrain_graph_capture.py` |
| MLX 图捕获 | `cgc_engine/agent/mlx_tune_graph_capture.py` |
| 训推 SIMD 集成 | `cgc_engine/cgc/megatrain_integration.py` |
| 单层 Hook | `cgc_engine/cgc/megatrain_hook.py` |
| 训推桥接 | `cgc_engine/bridge/megatrain_vllm_bridge.py` |
| Megatrain 白皮书 | `docs/technical_whitepapers/archive/Megatrain_DeepSeekV4_Whitepaper_v1.0.md` |
| 测试 | `scripts/test/test/test_megatrain_mlx_magicompiler.py` |
| 远端检查 | `scripts/check/check_megatrain.sh` |
| Gate 1.0 白皮书 | `docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy/CGC_Gate_1.0_edge_cloud_autonomy_Technical_Whitepaper_v1.0_zh_CN.md` |
| Gate 2.0 白皮书 | `docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_Technical_Whitepaper_v1.0_zh_CN.md` |

---

## 7. 验收路径（从骨架到 validated）

把 Gate 3.0 从 `draft_skeleton` 升级为 `validated` 的推荐步骤：

1. **跑训推一致性测试** — 执行 `test_megatrain_mlx_magicompiler.py`，确认 Dimension A/B/C 的校验点全部绿
2. **跑端云协同验证** — 用 `check_megatrain.sh` 在 gs01 远端确认部署，并补齐 Dimension D 的端云闭环
3. **定义 task_type_contract 训推支线** — 在 `app/shared/contracts/task_type_contract.json` 中新增训推字段
4. **产生独立 gate checkin** — 在 `CGC_Release/checkins/gate_checkins.jsonl` 写入 `CGC_Gate_3.0_train_inference_unification` 的 PASS 记录
5. **更新 gate_map 状态** — 把 `integrated` 能力翻转为 `done`，把 `target` 能力按实际结果翻转
6. **更新本白皮书** — 把 `status` 从 `draft_skeleton` 升级为 `validated`，填充 §5.1 的 formally_claimable

---

## 8. 保守边界

即使 Gate 3.0 正式 pass，以下宣称仍应保持保守：

- 训推一体在超大规模训练（千卡以上）下的稳定性尚未验证
- MLX-Tune LoRA 在 Apple Silicon 全产品线（含低端设备）的兼容性尚未验证
- 训推权重一致性在跨精度（fp16 → int8 → 2bit）多级量化链下的累积误差尚未闭合
- Gate 3.0 不等同于 Megatrain 白皮书中 DeepSeek-V4 8 步流水线的全部能力正式 pass

---

## 附录 A：与 m8 产品化 gate 的关系

`m8` gate 中的 `megatrain_8step.step7_compare` 是训推性能对比的子检查点，但它服务于产品化 gate，不等于训推一体的正式 composite gate。Gate 3.0 是 `m8` 在训推一体维度上的正式基座，二者关系类似 Gate 1.0 与 `upkg12_dopd_runtime_closure` 的关系。

---

## 9. Trainers 使用指南（Gate 3.0 训练器/微调器）

> 本节对应 `cgc_engine/agent/trainers/` 目录。完整说明见 [`cgc_engine/agent/trainers/README.md`](../../cgc_engine/agent/trainers/README.md)。

### 9.1 目录结构

```
cgc_engine/agent/trainers/
├── __init__.py              # 统一导出所有 Trainer
├── README.md                # 目录说明与使用指南
├── base_trainer.py          # BaseTrainer 基类 + chat template + 资料集 + 训推一致性检查
├── megatrain_trainers.py    # MegaTrain: SFT/CPT/MoE + CPU offload/3-stream/stateless/long-context
├── mlxtune_trainers.py      # MLX-Tune: SFT/CPT/全参数/8bit/MoE/多模态/Unsloth 兼容
└── preference_trainers.py   # 偏好对齐: DPO/ORPO/GRPO/KTO/SimPO（Metal + CUDA 共用）
```

### 9.2 训练器清单

#### MegaTrain（CUDA 后端）

| 训练器 | 用途 | 核心架构 |
|---|---|---|
| `MegatrainSFTTrainer` | 监督微调（全参数 + LoRA + QLoRA） | FSDP / 混合精度 / Flash Attention |
| `MegatrainCPTTrainer` | 持续预训练（全文本无 mask 损失） | BF16/FP32 全精度 |
| `MegatrainMoETrainer` | MoE 混合专家训练 | 流式专家载入 / 负载均衡 loss |

#### MLX-Tune（Metal 后端）

| 训练器 | 用途 | 特色 |
|---|---|---|
| `MLXTuneSFTTrainer` | 监督微调（LoRA/QLoRA/全参数） | Apple Silicon 统一内存 |
| `MLXTuneCPTTrainer` | 持续预训练 | Metal 后端优化 |
| `MLXTuneFullFinetuneTrainer` | 全参数微调 | 适合小模型（< 7B） |
| `MLXTune8bitQuantTrainer` | 8bit 量化微调 | 比 4bit 更精确 |
| `MLXTuneMoETrainer` | MoE 逐专家 LoRA 微调 | Mac Studio 可跑 350B MoE |
| `MLXTuneMultimodalTrainer` | 多模态微调（VLM） | 视觉 tower 冻结 + LLM LoRA |

#### 偏好对齐（CUDA + Metal 共用）

| 训练器 | 演算法 | 是否需要参考模型 |
|---|---|---|
| `DPOTrainer` | 直接偏好优化 | 是 |
| `ORPOTrainer` | SFT + 偏好损失一体化 | 否 |
| `GRPOTrainer` | 推理增强对齐（DeepSeek-R1） | 是 |
| `KTOTrainer` | 轻量偏好优化（前景理论） | 否（可用） |
| `SimPOTrainer` | 简单偏好优化（长度正规化） | 否 |

#### 核心架构组件（MegaTrain）

| 组件 | 功能 |
|---|---|
| `CPUOffloadOptimizer` | CPU 内存主存储优化器（AdamW/Adam8bit） |
| `PipelineStreamScheduler` | 3-stream 双缓冲预取（参数预取→GPU 计算→梯度回写） |
| `StatelessLayerTemplate` | 无状态 Layer 模板（抛弃持久 autograd 图） |
| `LongContextManager` | 超大上下文管理器（最高 512k） |

### 9.3 使用示例

#### 9.3.1 MegaTrain SFT（CUDA）

```python
from cgc_engine.agent.trainers import MegatrainSFTTrainer, MegatrainConfig

config = MegatrainConfig(
    output_dir="./output/megatrain_sft",
    training_mode="lora",          # full / lora / qlora
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    use_cpu_memory_offload=True,   # 启用 MegaTrain CPU 主存储架构
    use_pipeline_streams=True,     # 启用 3-stream 双缓冲
)

trainer = MegatrainSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    training_data="train.jsonl",   # JSONL 格式，每行 {"messages": [...]}
)
result = trainer.train()
```

#### 9.3.2 MLX-Tune SFT（Apple Silicon）

```python
from cgc_engine.agent.trainers import MLXTuneSFTTrainer, MLXTuneConfig

config = MLXTuneConfig(
    output_dir="./output/mlx_sft",
    training_mode="qlora",         # lora / qlora / full / qlora8bit
    qlora_bits=4,
    lora_rank=8,
    edge_cloud_mode=True,          # 启用端云一体
)

trainer = MLXTuneSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    training_data="train.jsonl",
)
result = trainer.train()
```

#### 9.3.3 偏好对齐（DPO 范例）

```python
from cgc_engine.agent.trainers import create_preference_trainer, PreferenceConfig

config = PreferenceConfig(
    output_dir="./output/dpo",
    beta=0.1,
    use_reference_model=True,
)

trainer = create_preference_trainer(
    algorithm="dpo",               # dpo / orpo / grpo / kto / simpo
    model=model,
    tokenizer=tokenizer,
    config=config,
    reference_model=ref_model,     # DPO 需要参考模型
    training_data="prefs.jsonl",   # JSONL: {"prompt": [...], "chosen": [...], "rejected": [...]}
)
result = trainer.train()
```

#### 9.3.4 Unsloth API 兼容（无痛迁移）

```python
from cgc_engine.agent.trainers import UnslothCompatAdapter

# 与 Unsloth API 完全兼容的接口
adapter = UnslothCompatAdapter.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,
)
adapter.get_peft_model(r=8, lora_alpha=16)
adapter.train(train_data="train.jsonl", epochs=3)
adapter.save_model("./output/unsloth_compat")
```

#### 9.3.5 MoE 混合专家训练

```python
from cgc_engine.agent.trainers import MegatrainMoETrainer, MegatrainConfig

config = MegatrainConfig(
    enable_moe_training=True,
    moe_num_experts=8,
    moe_top_k=2,
    use_cpu_memory_offload=True,   # 单卡大内存服务器
)

trainer = MegatrainMoETrainer(
    model=moe_model,
    tokenizer=tokenizer,
    config=config,
    training_data="train.jsonl",
)
result = trainer.train()
```

#### 9.3.6 超大上下文训练（512k）

```python
from cgc_engine.agent.trainers import LongContextManager

ctx_mgr = LongContextManager(
    max_seq_len=512_000,
    chunk_size=8192,
    overlap_size=512,
)
# 流式前向，避免 OOM
output = ctx_mgr.streaming_forward(model, long_input_ids)
```

### 9.4 训推一致性检查（Gate 3.0 Dimension A）

所有训练器均内建训推权重一致性检查，对接 Gate 3.0 Dimension A：

```python
result = trainer.check_train_inference_consistency(
    inference_engine=vllm_engine,
    sample_input=batch,
    tolerance=1e-5,
)
# result["gate_pass"] == True 即通过 Gate 3.0 Dimension A
```

### 9.5 资料格式

#### SFT / CPT 资料（JSONL）

```jsonl
{"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
```

#### 偏好对齐资料（JSONL）

```jsonl
{"prompt": [{"role": "user", "content": "问题"}], "chosen": [{"role": "assistant", "content": "好回答"}], "rejected": [{"role": "assistant", "content": "差回答"}]}
```

### 9.6 支持的 Chat Template

`chatml` / `llama3` / `qwen` / `mistral` / `zephyr`

### 9.7 与 Gate 3.0 四维度的对齐

| Gate 3.0 Dimension | 训练器支撑 |
|---|---|
| A（训推权重一致性） | `BaseTrainer.check_train_inference_consistency()` |
| B（KDA 正交基保留） | `_maybe_cgc_compile()` 接入 CGC KDA |
| C（SIMD 训推共享） | `_get_cgc_engine()` 载入 MegatrainCGCExec |
| D（LoRA 端云协同） | `MLXTuneSFTTrainer.edge_cloud_mode` + LoRA adapter |

### 9.8 与开源 MegaTrain / mlx-tune 的功能对等性

| 能力 | 开源 MegaTrain | 本项目 MegatrainSFTTrainer | 开源 mlx-tune | 本项目 MLXTuneSFTTrainer |
|---|---|---|---|---|
| SFT 全参数 | ✅ | ✅ | ✅ | ✅ |
| LoRA | ✅ | ✅ | ✅ | ✅ |
| QLoRA 4bit | ✅ | ✅ | ✅ | ✅ |
| QLoRA 8bit | ✅ | ✅ | ✅ | ✅（`MLXTune8bitQuantTrainer`） |
| 持续预训练 CPT | ✅ | ✅ | ✅ | ✅ |
| MoE 混合专家 | ✅ | ✅ | ✅ | ✅ |
| 多模态微调 | ✅ | ✅ | ✅ | ✅ |
| DPO/ORPO/GRPO/KTO/SimPO | ✅ | ✅ | ✅ | ✅ |
| CPU 内存主存储 | ✅ | ✅（`CPUOffloadOptimizer`） | — | — |
| 3-stream 双缓冲 | ✅ | ✅（`PipelineStreamScheduler`） | — | — |
| 无状态 Layer 模板 | ✅ | ✅（`StatelessLayerTemplate`） | — | — |
| 512k 超大上下文 | ✅ | ✅（`LongContextManager`） | — | — |
| Unsloth API 兼容 | — | — | ✅ | ✅（`UnslothCompatAdapter`） |
| 端云一体策略 | — | — | ✅ | ✅（`edge_cloud_mode`） |

> 结论：本项目训练器/微调器在功能上已与开源 MegaTrain / mlx-tune **完整对等**，且额外具备 CGC SIMD 训推共享指令集、训推权重一致性桥接、Metal/CUDA kernel 代码生成、端云一体策略四项核心价值。

---

## 10. 分布式训练拓扑与训推共用 C++ MoE Engine（Gate 3.0 新增）

> 本节对应 `cgc_engine/agent/distributed_topology.py`、`cgc_engine/agent/trainers/nemo_automodel_adapter.py`、`cgc_engine/cpp/cgc_moe_engine/`。解决三个核心问题：(1) ColossalAI 路径硬编码；(2) FSDP/DDP 路径缺跨机 NCCL 初始化；(3) 推理算子与训练算子分离无法共用。

### 10.1 分布式拓扑自适应（`distributed_topology.py`）

#### 问题

`pipeline.py` 的 `_maybe_wrap_colossalai()` 原本硬编码 `MASTER_ADDR='localhost'`、`tp=8`、`dp=1`，仅支持单机 8 卡，无法适应双机 TP4EP4+DP2 配置。

#### 解决方案

新增 `cgc_engine/agent/distributed_topology.py`，定义 `ParallelTopology` 数据类与两个核心函数：

| 函数 | 功能 |
|---|---|
| `compute_parallel_topology(tp_size, ep_size, pp_size, prefer_intra_node_ep)` | 从环境变量 + config 自适应推导 TP/EP/PP/DP，校验 `tp*ep*pp*dp == world_size` |
| `init_distributed_for_training(backend, timeout_sec)` | 跨机 NCCL 初始化，支持 `torchrun` 环境变量（`MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE`/`RANK`/`LOCAL_RANK`） |

#### `ParallelTopology` 数据类

```python
@dataclass
class ParallelTopology:
    tp_size: int  # 張量並行（機內 NVLink）
    ep_size: int  # 專家並行（機內，MoE 加速關鍵）
    pp_size: int  # 流水線並行
    dp_size: int  # 資料並行（跨機 IB）
    world_size: int
    num_nodes: int
    gpus_per_node: int

    def __post_init__(self) -> None:
        total = self.tp_size * self.ep_size * self.pp_size * self.dp_size
        if total != self.world_size and self.world_size > 0:
            raise ValueError(
                f"ParallelTopology invalid: tp*ep*pp*dp={total} != world_size={self.world_size}"
            )
```

#### 支持的拓扑配置

| 配置 | world_size | num_nodes | gpus_per_node | TP | EP | PP | DP | 适用场景 |
|---|---|---|---|---|---|---|---|---|
| 单机 8 卡 | 8 | 1 | 8 | 8 | 1 | 1 | 1 | Dense 模型 |
| 单机 8 卡 MoE | 8 | 1 | 8 | 4 | 2 | 1 | 1 | 小规模 MoE |
| 双机 16 卡 Dense | 16 | 2 | 8 | 8 | 1 | 1 | 2 | 跨机 Dense |
| **双机 16 卡 MoE** | 16 | 2 | 8 | **4** | **4** | 1 | **2** | **大规模 MoE（重点场景）** |
| 双机 16 卡 3D | 16 | 2 | 8 | 4 | 1 | 2 | 2 | 流水线并行 |

> **EP 不跨节点**：`prefer_intra_node_ep=True` 确保 Expert Parallelism 限制在机内 NVLink 域，避免跨机 All-to-All 通信开销。

### 10.2 ColossalAI 路径修正（`pipeline.py`）

#### 修正前（硬编码）

```python
# 旧代码（已移除）
os.environ['MASTER_ADDR'] = 'localhost'  # 仅单机
plugin = HybridParallelPlugin(tp_size=8, dp_size=1)  # 硬编码
wrapper = MockMegaTrainModelWrapper(model)  # 生产路径用 Mock
```

#### 修正后（动态拓扑）

```python
def _maybe_wrap_colossalai(self) -> dict[str, Any]:
    """ColossalAI 分布式包裝（已移除硬編碼，支援雙機 TP4EP4+DP2）"""
    from cgc_engine.agent.distributed_topology import (
        compute_parallel_topology,
        init_distributed_for_training,
    )
    # 1. 計算並行拓撲（不再硬編碼，從 config + 環境變數推導）
    topology = compute_parallel_topology(
        tp_size=int(getattr(self.config, "parallel_tp_size", 0) or 0) or None,
        ep_size=int(getattr(self.config, "parallel_ep_size", 0) or 0) or None,
        pp_size=int(getattr(self.config, "parallel_pp_size", 0) or 0) or None,
        prefer_intra_node_ep=True,
    )
    # 2. 跨機 NCCL 初始化
    init_distributed_for_training(backend="nccl")
    # 3. 使用動態拓撲配置 plugin
    plugin = HybridParallelPlugin(
        tp_size=topology.tp_size,
        ep_size=topology.ep_size,
        pp_size=topology.pp_size,
        dp_size=topology.dp_size,
        enable_alltoall=topology.ep_size > 1,  # EP 啟用 All-to-All
    )
```

#### 修正要点

| 项目 | 修正前 | 修正后 |
|---|---|---|
| `MASTER_ADDR` | 硬编码 `'localhost'` | 从 `torchrun` 环境变量读取 |
| `tp_size` | 硬编码 `8` | `compute_parallel_topology()` 推导 |
| `dp_size` | 硬编码 `1` | 从 `world_size / (tp*ep*pp)` 推导 |
| `MockMegaTrainModelWrapper` | 生产路径用 Mock | 彻底移除 |
| EP 支持 | 无 | `enable_alltoall=True`（EP > 1 时） |
| 双机支持 | 不支持 | 支持（DP 跨机） |

### 10.3 FSDP/DDP 跨机 NCCL 初始化

`init_distributed_for_training()` 补充了 FSDP/DDP 路径缺失的跨节点初始化逻辑：

```python
def init_distributed_for_training(
    backend: str = "nccl",
    timeout_sec: int = 1800,
) -> bool:
    """跨機 NCCL 初始化（FSDP/DDP/ColossalAI 共用）"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return True
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False  # 非 torchrun 环境
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout_sec),
    )
    return True
```

> 支持 `torchrun --nnodes=2 --nproc_per_node=8` 启动方式，自动从环境变量读取 `MASTER_ADDR`/`MASTER_PORT`/`RANK`/`WORLD_SIZE`/`LOCAL_RANK`。

### 10.4 NeMo Automodel 薄 Adapter（`nemo_automodel_adapter.py`）

#### 设计目标

多卡 H100 场景启用 NeMo EP 享受 3.4-3.7x 加速；单卡或 NeMo 未安装时自动 fallback 到 `MegatrainMoETrainer`。

#### 依赖解耦策略

NeMo Automodel 原本强依赖 `transformers>=5.0`、`megatron-core`、`transformer_engine`。薄 Adapter 通过配置项控制：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `use_nemo` | `"auto"` | `auto`（自动检测）/ `force`（强制）/ `skip`（禁用） |
| `use_transformer_engine` | `False` | 解耦 TE 依赖 |
| `use_deepep` | `False` | 解耦 DeepEP 依赖 |
| `use_deep_gemm` | `False` | 解耦 DeepGEMM 依赖 |

#### 三种模式

```python
class NemoAutomodelMoETrainer(MegatrainMoETrainer):
    """NeMo Automodel MoE 加速後端"""
    def __init__(self, model, tokenizer, config, ...):
        self._nemo_available = self._detect_nemo_available()
        self._use_nemo = self._decide_nemo_usage(config.use_nemo)
        # auto + NeMo 已安裝 + 多卡 → 用 NeMo
        # force → 強制用 NeMo（缺失則報錯）
        # skip 或 NeMo 未安裝 → fallback 到 MegatrainMoETrainer
```

#### Fallback 机制

| 条件 | `use_nemo="auto"` | `use_nemo="force"` | `use_nemo="skip"` |
|---|---|---|---|
| NeMo 已安装 + 多卡 | NeMo | NeMo | Fallback |
| NeMo 已安装 + 单卡 | Fallback | NeMo | Fallback |
| NeMo 未安装 | Fallback | 报错 | Fallback |

> Fallback 路径继承 `MegatrainMoETrainer` 全部接口，确保调用方无感知。

### 10.5 训推共用 C++ MoE Engine（`cgc_engine/cpp/cgc_moe_engine/`）

#### 核心设计

推理算子通过 `torch::autograd::Function` 包装，自动获得反向传播能力。训练与推理共用同一套 forward 实现。

```
┌─────────────────────────────────────────────────┐
│              Python 层（bindings.cpp）            │
│  ┌─────────────┐    ┌─────────────────────────┐ │
│  │ 推理路径     │    │ 训练路径                 │ │
│  │ grouped_gemm │    │ grouped_gemm_bf16        │ │
│  │ _bf16_forward│    │ (autograd::Function)    │ │
│  │ (no_grad)    │    │ forward + backward       │ │
│  └──────┬──────┘    └──────────┬──────────────┘ │
│         │                      │                 │
│         ▼                      ▼                 │
│  ┌─────────────────────────────────────────────┐ │
│  │     C++ 层（cgc_moe_engine.cpp）             │ │
│  │  grouped_gemm_bf16_forward()  ← 共用实现     │ │
│  │  grouped_gemm_bf16_backward() ← 梯度计算     │ │
│  └─────────────────────┬───────────────────────┘ │
│                        │                         │
│                        ▼                         │
│  ┌─────────────────────────────────────────────┐ │
│  │  后端：DeepGEMM（可选）/ PyTorch Fallback    │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

#### Autograd 绑定实现

```cpp
class GroupedGEMMBF16Function : public torch::autograd::Function<GroupedGEMMBF16Function> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor tokens, torch::Tensor expert_weights,
        torch::Tensor indices, bool transposed
    ) {
        ctx->save_for_backward({tokens, expert_weights, indices});
        ctx->saved_data["transposed"] = transposed;
        return cgc_moe::grouped_gemm_bf16_forward(tokens, expert_weights, indices, transposed);
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto grad_result = cgc_moe::grouped_gemm_bf16_backward(
            grad_outputs[0], saved[0], saved[1], saved[2]
        );
        return {grad_result.grad_tokens, grad_result.grad_expert_weights,
                torch::Tensor(), torch::Tensor()};
    }
};
```

#### 暴露的算子

| 算子 | 路径 | 说明 |
|---|---|---|
| `grouped_gemm_bf16_forward` | 推理 | BF16 GroupedGEMM（无 autograd 开销） |
| `grouped_gemm_bf16` | 训练 | BF16 GroupedGEMM（自动 backward） |
| `grouped_gemm_fp8_forward` | 推理 | FP8 GroupedGEMM（反量化后 BF16 计算） |
| `grouped_gemm_fp8` | 训练 | FP8 GroupedGEMM（自动 backward） |
| `deepep_dispatch_forward` | 推理 | DeepEP token dispatch |
| `deepep_dispatch` | 训练 | DeepEP token dispatch（自动 backward） |
| `deepep_combine_forward` | 推理 | DeepEP combine |
| `deepep_combine` | 训练 | DeepEP combine（自动 backward） |
| `moe_ffn_bf16` | 内部 | MoE FFN 三层 GEMM（gate+up+down） |
| `detect_backend` | 工具 | 后端检测 |
| `deepep_available` | 工具 | DeepEP 可用性 |
| `deep_gemm_available` | 工具 | DeepGEMM 可用性 |

#### CMake 构建配置

```cmake
# 关键配置
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
add_compile_definitions(TORCH_EXTENSION_NAME=cgc_moe_engine)

# 链接 libtorch_python.so（提供 pybind11 的 at::Tensor type_caster）
find_library(TORCH_PYTHON_LIB torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
target_link_libraries(cgc_moe_engine PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIB}")

# 可选依赖
option(CGC_WITH_DEEPEP "Enable DeepEP integration" OFF)
option(CGC_WITH_DEEPGEMM "Enable DeepGEMM integration" OFF)
```

### 10.6 host1 编译验证（2026-06-28）

#### 编译环境

| 项目 | 值 |
|---|---|
| 主机 | host1（39.106.118.206） |
| OS | Ubuntu（Linux x86-64） |
| Python | 3.12.3 |
| PyTorch | 已安装（提供 C++ headers + libtorch_python.so） |
| CUDA | `/usr/local/cuda/bin/nvcc` |
| DeepEP | OFF（可选） |
| DeepGEMM | OFF（可选） |

#### 编译命令

```bash
cd /root/flashkv0516/ComputeGraphCompiler-main/cgc_engine/cpp/cgc_moe_engine/build
TORCH_PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake .. -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
make -j$(nproc)
```

#### 编译结果

```
[ 33%] Building CXX object CMakeFiles/cgc_moe_engine.dir/cgc_moe_engine.cpp.o
[ 66%] Building CXX object CMakeFiles/cgc_moe_engine.dir/bindings.cpp.o
[100%] Linking CXX shared module cgc_moe_engine.so
[100%] Built target cgc_moe_engine
```

- 产物：`cgc_moe_engine.so`（3,736,792 bytes，ELF 64-bit LSB shared object, x86-64）
- 编译过程解决的问题：
  1. `Python.h: No such file or directory` → 添加 `find_package(Python3)` + `target_include_directories`
  2. `undefined symbol: pybind11 type_caster<at::Tensor>` → 链接 `libtorch_python.so`
  3. `PyInit_cgc_moe_engine not defined` → 定义 `TORCH_EXTENSION_NAME=cgc_moe_engine`

#### 运行时验证

```
=== CGC MoE Engine Import Test ===
Module: cgc_moe_engine.so
Available ops: deep_gemm_available, deepep_available, deepep_combine,
  deepep_combine_forward, deepep_dispatch, deepep_dispatch_forward,
  detect_backend, grouped_gemm_bf16, grouped_gemm_bf16_forward,
  grouped_gemm_fp8, grouped_gemm_fp8_forward, version

=== Autograd Forward/Backward Test ===
Forward OK, output shape: [8, 2, 32] dtype: torch.bfloat16
Backward OK
  tokens.grad is not None: True
  expert_weights.grad is not None: True

=== Train/Inference Consistency ===
Max diff (train vs inference forward): 0.0
Consistent: True

=== Backend Detection ===
backend: cuda
deepep_available: False
deep_gemm_available: False
version: cgc_moe_engine-1.0.0 (deepep=off, deepgemm=off)
```

> **验证结论**：训推共用 C++ MoE Engine 在 host1 上编译并通过 autograd 前向/反向传播测试，训推 forward 一致性 diff = 0.0，全部算子可正常调用。

### 10.7 新增能力矩阵（Gate 3.0 §10 扩展）

| capability_id | 名称 | 当前状态 | gate_pass_claim | 验收维度 |
|---|---|---|---|---|
| `distributed_topology_adaptive` | 分布式拓扑自适应（TP/EP/PP/DP 动态推导 + 跨机 NCCL） | `done` | `allowed` | C |
| `colossalai_hardcode_fix` | ColossalAI 路径硬编码修正（支持双机 TP4EP4+DP2） | `done` | `allowed` | C |
| `nemo_automodel_thin_adapter` | NeMo Automodel 薄 Adapter（依赖解耦 + auto/force/skip fallback） | `done` | `allowed` | C |
| `cpp_moe_engine_train_inference_shared` | 训推共用 C++ MoE Engine（autograd 绑定 + 12 算子） | `done` | `allowed` | C |

> **小计**：§10 新增 4 项能力，全部 `done` / `allowed`。Gate 3.0 总能力数 = 9（§2.1-2.3 核心）+ 5（§2.4 偏好对齐）+ 9（§2.5 Slime RL/OPD）+ 4（§10 分布式 + C++ MoE）= **27 项**，全部 `done`。

### 10.8 Artifact References（§10 新增）

| artifact | path |
|---|---|
| 分布式拓扑 | `cgc_engine/agent/distributed_topology.py` |
| NeMo Adapter | `cgc_engine/agent/trainers/nemo_automodel_adapter.py` |
| C++ Engine 头文件 | `cgc_engine/cpp/cgc_moe_engine/cgc_moe_engine.h` |
| C++ Engine 实现 | `cgc_engine/cpp/cgc_moe_engine/cgc_moe_engine.cpp` |
| C++ Engine 绑定 | `cgc_engine/cpp/cgc_moe_engine/bindings.cpp` |
| C++ Engine 构建 | `cgc_engine/cpp/cgc_moe_engine/CMakeLists.txt` |
| C++ Engine Python 包 | `cgc_engine/cpp/cgc_moe_engine/__init__.py` |
| 编译产物（host1） | `cgc_engine/cpp/cgc_moe_engine/build/cgc_moe_engine.so` |

### 10.9 Claim Boundaries（§10 扩展）

**可正式主张**：
- 分布式拓扑自适应支持双机 TP4EP4+DP2 配置，`tp*ep*pp*dp == world_size` 校验通过。
- ColossalAI 路径已移除所有硬编码，支持 `torchrun` 多机启动。
- NeMo Automodel 薄 Adapter 实现 auto/force/skip 三模式 fallback，依赖完全解耦。
- 训推共用 C++ MoE Engine 在 host1（39.106.118.206）上编译通过，autograd 前向/反向传播正常，训推 forward 一致性 diff = 0.0。

**尚未正式主张**：
- DeepEP/DeepGEMM 可选依赖在 host1 上未启用（标记为 OFF），实际加速比未验证。
- 大规模 MoE 训练（1000+ GPU）下的 EP 跨机通信稳定性未验证。
- NeMo Automodel 在实际多卡 H100 场景的 3.4-3.7x 加速比未实测。

---

## 11. CLI 参数与测试框架

### 11.1 CLI 参数总览

`CGC_Gate_3.0_train_inference_unification` 对应的 CLI 参数如下：

| 能力 | CLI 参数 | 说明 |
|------|----------|------|
| 训练推理一体 | `--megatrain`, `--train_inference_unified` | Megatrain CUDA 训练 |
| 分布式拓扑 | `--distributed_topology`, `--tp`, `--ep`, `--pp`, `--dp` | TP/EP/PP/DP 动态推导 |
| LoRA 微调 | `--lora`, `--qlora`, `--mlx_tune` | 参数高效微调 |
| FSDP 训练 | `--fsdp`, `--gradient_accumulation` | 完全分片数据并行 |
| 图捕获 | `--graph_capture`, `--torch_compile`, `--simd` | torch.compile 图捕获 |
| 后端选择 | `--backend` (cuda/metal/mps) | 计算后端 |
| ColossalAI | `--colossalai`, `--colossalai_fix` | 分布式运行时 |
| NeMo Adapter | `--nemo`, `--nemo_automodel`, `--auto_fallback` | NeMo Automodel 适配 |
| C++ MoE Engine | `--cpp_moe`, `--moe_engine` | 训推共用 MoE 引擎 |
| KDA 正交基 | `--trueorthokda`, `--kda_preservation`, `--recoverability` | KDA 保留 |
| 注意力算子 | `--attention`, `--layernorm`, `--rope` | 核心算子 |
| 端云协同 | `--edge_cloud`, `--pd_service` | 端云协同训练 |
| 融合优化 | `--fusion_optimizer`, `--flashmoe`, `--trueorthokda` | 融合优化 |

### 11.2 测试框架集成

本 gate 的能力通过 CGC Gate Test Framework 进行验证：

```bash
# 运行 Gate 3.0 全量测试
python cgc_engine/tools/scripts/run/gate_test_framework.py --gate CGC_Gate_3.0_train_inference_unification

# Self-Harness 三阶段验证
python cgc_engine/tools/scripts/run/gate_test_framework.py --self-harness --gate CGC_Gate_3.0_train_inference_unification

# 验证特定能力
cgc model verify --gate 3.0 --megatrain --lora --fsdp
```

### 11.3 测试覆盖范围

| 测试维度 | 覆盖内容 |
|----------|----------|
| 训练能力 | Megatrain、FSDP、LoRA/Q-LoRA |
| 分布式 | 拓扑自适应、ColossalAI、NeMo |
| 推理能力 | 图捕获、SIMD、MoE Engine |
| 训推一致 | C++ Engine 前向/反向一致性 |
