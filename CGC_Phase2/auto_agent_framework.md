# CGC 自动化 Agent 框架设计

> 目标：基于云端 `cgc_engine` 已落地的核心概念（八步流水线 / 4D 感知矩阵 / state ABI / Unified Pipeline Kernel / bootstrap / system profile / profile binding），设计一套能**自动适配不同模型**（如 `qwen3_vl_moe.py`）的 agent 框架，把目前**手动**编写 resume patch 的流程自动化。

---

## 0. 概念溯源（云端文件引用）

以下定义全部来自云端 `47.95.250.55:/root/flashkv0516/cgc_engine/`，只读提取。

| 概念 | 主源文件 | 关键行 |
|------|----------|--------|
| 八步流水线（agent 面） | `agent/llm_auto_pipeline.py` | 2980–3313 |
| 八步流水线（compile 2.0 面） | `pipeline.py` | 2153–2163 |
| 八步合约校验 | `gate_verifiers/g22_deepep_l20n_verifier.py` | 148–160 |
| 4D 感知矩阵 | `pipeline.py` / `agent/llm_auto_pipeline.py` | 2155 / 3338 |
| state ABI | `pipeline_contract_common.py` / `pd/dopd_schema.py` | 9,19 / 47 |
| state ABI 校验 | `gate_verifiers/g21_fusion_governance_verifier.py` | 198–273 |
| Unified Pipeline Kernel (UPK) | `pipeline_contract_common.py` | 7–21 |
| UPK 校验 | `gate_verifiers/g21_...` / `g22_...` | 113 / 298 |
| bootstrap | `integrated_gate_pipeline.py` | 76 |
| system profile | `agent/llm_auto_pipeline.py` | 2345–2410 |
| profile binding | `gate_verifiers/g22_deepep_l20n_verifier.py` | 93,330 |
| 手动 resume patch（参照） | 本地 `CGC_Phase2/qwen3_resume_patch.py` | 全文 |

---

## 1. 核心概念定义

### 1.1 八步流水线（Eight-Step Pipeline）

CGC 有两条八步流水线，agent 框架以**第一条（agent/inference 面）**为驱动主轴，**第二条（compile 2.0 面）**为内核生成辅轴。

**(A) Agent 面 — 统一八步（`agent/llm_auto_pipeline.py`，CLI 标注 "unified 8-step pipeline"，见 `agent/cli.py:6,62`）**

```
step0_scenario   场景识别（edge-cloud / dev / user）
step1_hardware   硬件探测（device / VRAM / topology）
step2_capture    模型捕获（HF config / GGUF header / 图捕获）
step3_analyze    分析（等价门 equivalence_gate / skvm / graph）
step4_identify   识别优化点（含 4D 感知矩阵 → step4_hardware_perception）
step5_generate   生成最优代码（编译 .so / kernel codegen）
step6_dispatch   派发到后端（vllm / sglang / mlx / llama.cpp / megatrain）
step7_compare    原生 vs 优化对比（fullgraph bench）
step8_combine    合并/部署（deploy_unit）
```

合约校验（`g22_deepep_l20n_verifier.py:148`）要求 m76 暴露 `_validate_eight_step_pipeline`，并打印 `[1/8]..[8/8]` 全部 8 个 marker，且包含 `step7_kernel_codegen`、`step8_runtime`。

**(B) Compile 2.0 面 — 内核生成八步（`pipeline.py:2153`）**

```
step1_staticize → step2_graph_capture → step3_partition → step4_skvm_verify
→ step5_passes → step6_memory_planning → step7_kernel_codegen → step8_runtime
```

> 训练/harness 任务还会再走 `step2_capture..step8_combine` 的 7 步闭环。两条线在 step7/step8 汇合于「内核代码生成 + 运行时」。

### 1.2 4D 感知矩阵（4D Perception Matrix）

**四维定义**（`pipeline.py:2155`）：

> **4D 矩阵：环境 × 任务 × 硬件 × 模型**
> （Environment × Task × Hardware × Model）

注：同文件 `pipeline.py:2153` 另有 **5 轴矩阵：任务 × 模式 × 环境 × 模型侧 × 硬件**，5 轴是 4D 的扩展（多了"模式/模型侧"）。`matrix_axes`（`pipeline.py:2123`）字段：`task_entity / task_domain / runtime_mode / environment / hardware_scope / hardware_platform / hardware_topology / model_scope / model_assembly / model_name`。

**在 step4 的落地**（`agent/llm_auto_pipeline.py:3338` `[M7.4] 4D Perception Matrix Logic`，键名 `step4_hardware_perception`）：

- 云端 = RTX 5090 (SGLang)，端侧 = RTX Spark/Mac (Llama.cpp)
- `hardware_maximized_partitioning`：按端侧 VRAM 上限**动态计算层切分 N**
- 动作：分配 UMA 0-copy 内存池 + 注入 KV Bridge 算子
- 输出字段：`total_layers` / `estimated_vram_gb` / `allocated_edge_layers`
- 驱动 `step6_dispatch` 的动态 token 路由：短上下文(<1000)→`local_only`，长上下文→`cloud_edge_split`

> 作用：用四维状态空间描述"在什么环境/做什么任务/跑在什么硬件/用什么模型"，并据此算出层切分点与路由策略。

### 1.3 state ABI（状态应用二进制接口）

**两层含义：**

1. **合约工件路径**（`pipeline_contract_common.py:9,19`）：`state_abi_path` 是 UPK 的 7 个 artifact key 之一，且属于 4 个 REQUIRED key（`execution_context_path / state_abi_path / contract_manifest_path / system_execution_manifest_path`）。缺一即 `ready=False`。

2. **跨模型 resume 载荷规范**（`pd/dopd_schema.py:47` `DOPDResumePayloadV2`）——这是 agent 框架最关键的接口。字段：

```python
@dataclass
class DOPDResumePayloadV2:
    session_id, handoff_id, phase_role, cache_schema, kv_variant   # 必填
    model_name = ""
    abi_descriptor: Dict        # 模型特定 ABI 元信息（核心扩展点）
    layout_meta: Dict           # 布局元信息
    prefix_state_ref, kv_state_ref, kda_state_ref = ""             # 状态引用
    resume_position, token_position = 0
    prefill_done = True; decode_resume = True
    transport_codec = "cq4"; compression_codec = "trueorthokda"
    zero_copy_vram = True
    state_bytes_b64 = ""; metadata: Dict
    version = 2; payload_kind = "dopd_resume"; integrity_checksum
    # encode_dopd_resume_payload_v2 / decode_dopd_resume_payload_v2（带 magic + canonical json + sha256）
```

**模型桥接策略**（`g21_fusion_governance_verifier.py:198` `G21StateABIExtensionHookVerifier`）：pipeline 暴露 `state_abi_policy`（如 `"deepseek_v2_to_v4_min_state_abi_v1_2"`），带 `deepseek_abi_bridge` 插件，含 `qk_nope_head_dim=128`、`legacy_o_proj_in_dim=16384` 等层级 ABI 字段。`g22:347 G22StateABIL20NVerifier` 进一步要求 `DOPDResumePayloadV2` 暴露层级 ABI 字段并与 L20N profile 绑定同一 runtime shape。

> 作用：state ABI 是**跨模型/跨后端 resume 的统一契约**——只要某模型能产出/消费 `DOPDResumePayloadV2`（尤其 `abi_descriptor` + `finished_layer` 语义），端云就能在任意层切分点交接。

### 1.4 Unified Pipeline Kernel（UPK，统一流水线内核）

`pipeline_contract_common.py:7` 定义 `PIPELINE_KERNEL_ARTIFACT_KEYS`（7 件套）：

```
execution_context_path         执行上下文
state_abi_path                 状态 ABI（见 1.3）
strategy_decision_path         策略决策
compatibility_report_path      兼容性报告
distributed_runtime_bootstrap_path   分布式运行时 bootstrap
contract_manifest_path         契约清单
system_execution_manifest_path 系统执行清单
```

`pipeline_kernel_contract_artifacts_from_report()` 从 pipeline report 解析；`pipeline_contract_descriptor_from_artifacts()` 计算 `ready`（required key 齐全且路径存在）。UPK bundle 由 `_write_profile_settings_bundle()` 写出，`validate_profile_bundle()` 校验，`upkg_target="2.2"`（`g22:298 G22UPKL20NOptimizationVerifier`）。

> 作用：UPK 是把整条流水线的「执行上下文 + 状态接口 + 策略 + 兼容性 + bootstrap + 契约 + 系统清单」打包成一体的**统一内核合约**，是 agent 框架的"输出物壳"。

### 1.5 bootstrap（启动引导）

- `integrated_gate_pipeline.py:76`（Gate 1.0）定义 `bootstrap_contract` = **"引导契约 — 端云启动阶段的契约协商"**。
- UPK 中 `distributed_runtime_bootstrap_path` 是 7 件套之一。
- `g22:243` 校验 `bootstrap_contract_binding_key` / `bootstrap_contract_path` / `bootstrap_contract_id`，并要求 `bootstrap.requested_dispatch_backend`（如 `"deepep"`）。

> 作用：bootstrap 是运行时启动前**协商 dispatch backend / 并行度 / 拓扑**的契约阶段，agent 在 step6 dispatch 前必须先通过 bootstrap 协商。

### 1.6 system profile（系统画像）

`agent/llm_auto_pipeline.py:2345` `_derive_agent_system_profile()`，`schema_version="cgc.system_profile.v0.1"`，含：

- `mode_mapping`：development_cli→cgc / user_cli→cgc_edge / m76→cgc m76-dev
- `context_profile`：`execution_context{runtime_mode, environment, backend, model_name, task_type, exec_mode}` + `strategy_plan{contexts, runs, warmup_runs}`
- `routing_topology_profile`：`routing_mode(fusionroute/local_native), router_model, cloud_instance_count, fusion_group_size, cloud_instance_role, cloud_model, edge_model, gateway_ports, service_topology_backend(ray_cluster_dual_host/...), distributed_runtime_backend(colossalai/nccl/single_process), edge_decode_enabled, cloud_prefill_enabled`

另一面（`g22:283 G22SystemProfileL20NVerifier`）：`system_manifest.system_profile` 含 `deployment_mode(cloud_cluster)`、`hardware_profile.hardware_topology(2x8_blackwell_sm120)`、`component_families.route_policy_family(deepep_ep16_tp1_dualnode)`、`environment_bootstrap_ref.requested_dispatch_backend`。

> 作用：系统画像是「这台机器/这个集群在当前任务下的完整运行画像」，驱动路由、并行、dispatch 决策。

### 1.7 profile binding（画像绑定机制）

绑定发生在 `profile_settings` ↔ `system_manifest` ↔ `bootstrap_contract` ↔ `runtime_shape` 四者之间，靠**共享 binding key** 串联：

- `execution_profile_binding_key`（如 `dualnode_dsv4_qwen_dflash_exec_v1`）
- `bootstrap_contract_binding_key`（如 `dualnode_blackwell_deepep_ep16_tp1_runtime_v1`）
- `flow_parameter_contract_binding_key`（如 `dualnode_dsv4_qwen_dflash_flow_v1`）
- `distributed_binding.parallel_profile`（如 `ep16_tp1`）
- `system_profile_ref.source_path` / `profile_binding_ref`（manifest 侧反向引用）

校验链（`g21:128`、`g22:93,330`）：profile 的 binding key 必须等于 manifest `system_profile.profile_binding_ref` 里的对应 key；runtime_shape 的 `ep_size/nnodes` 必须与 `distributed_binding.parallel_profile` 一致；`validate_profile_bundle()` 读取同一三元组（profile→bootstrap→system_manifest）。

> 作用：profile binding 把"一个执行画像"钉死到"一个系统清单 + 一个 bootstrap 契约 + 一个 runtime 形状"上，保证端云两侧用的是同一套兼容配置。agent 生成新模型适配时，本质就是**为该模型生成一套自洽的 binding key 三元组**。

---

## 2. 自动化 Agent 框架设计

### 2.1 设计原则

1. **不手写 patch**：把 `qwen3_resume_patch.py` 里"人读模型源码 → 抄 deepseek_v4 → 改层签名"的过程，拆成可机读的「模型结构指纹 → ABI 模板渲染」。
2. **复用云端契约**：所有产物落成 UPK 7 件套，用 `validate_profile_bundle` + g21/g22 verifier 校验，而不是自造校验。
3. **八步驱动**：agent 主循环就是八步流水线，每步产出契约工件，缺件则 `ready=False` 自动回退。

### 2.2 架构总览

```
                    ┌─────────────────────────────────────────────┐
                    │           Auto-Adapt Agent (主循环)          │
                    │   = agent 面 八步流水线 (1.1-A)              │
                    └─────────────────────────────────────────────┘
          step0..step3                step4                  step5..step8
  ┌────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
  │ 4D 感知矩阵输入侧   │   │  Model ABI Extractor │   │  Patch Generator   │
  │ env×task×hw×model   │→  │  (AST 分析模型 .py)  │→  │  (state ABI 模板)  │
  │ → system_profile    │   │  → abi_descriptor    │   │  → resume_patch.py │
  └────────────────────┘   └──────────────────────┘   └─────────┬──────────┘
                                                               │
                ┌──────────────────────────────────────────────┘
                ▼
   ┌────────────────────────┐    ┌──────────────────────────┐
   │ UPK Bundle Writer      │    │ g21/g22 Verifier (校验)   │
   │ profile_settings +     │←── │ state_abi / upk /         │
   │ binding key 三元组      │    │ profile_binding / system  │
   └───────────┬────────────┘    └──────────────────────────┘
               ▼
   bootstrap_contract → dispatch → step7_compare(测试) → step8_combine(部署)
```

### 2.3 八步流水线如何驱动自动适配

| 步骤 | 自动化内容 | 产物（UPK 工件） |
|------|-----------|------------------|
| step0_scenario | 读 `--model qwen3_vl_moe --backend sglang --mode edge-cloud` | execution_context_path |
| step1_hardware | 探测云端 RTX 5090 / 端侧 Mac，写 system_profile | system_execution_manifest_path |
| step2_capture | **AST 解析** `qwen3_vl_moe.py`（类层级/forward 签名/层循环/residual 处理） | compatibility_report_path |
| step3_analyze | 等价门：原 forward vs resume forward 数值一致性 | strategy_decision_path |
| step4_identify | **4D 感知矩阵**算 `allocated_edge_layers`(P)、路由策略 | (并入 strategy_decision) |
| step5_generate | **渲染 state ABI 模板** → `resume_patch.py` | state_abi_path |
| step6_dispatch | bootstrap 协商 dispatch_backend，apply patch | contract_manifest_path |
| step7_compare | 短上下文/长上下文双场景 bench | compatibility_report |
| step8_combine | deploy_unit + 写 UPK bundle + binding key | pipeline_kernel_contract_artifacts |

`ready`（`pipeline_contract_common.py`）= 4 个 required key 全齐 → 框架判定"该模型已自动适配完成"。

### 2.4 4D 感知矩阵如何描述模型/系统状态

框架把 4D 作为**结构化输入**喂给 agent：

```
environment : cloud_cluster / edge_mac / edge_rtspark
task        : prefill / decode / pd_split / train
hardware    : {gpu, vram_gb, topology, uma_zero_copy}
model       : {name, num_layers, hidden_size, moe, layer_signature, residual_mode}
```

`step4` 复用 `[M7.4]` 逻辑：按端侧 VRAM 算 `allocated_edge_layers = P`（Mac 跑 0..P-1，cloud resume P..end），并决定 `step6` 路由 `local_only` vs `cloud_edge_split`。**4D 的 model 维直接来自 step2 的 AST 指纹**，使矩阵对每个模型自动定制。

### 2.5 state ABI 如何定义跨模型 resume 接口

核心：**resume patch 的产出/消费必须等于 `DOPDResumePayloadV2`**。框架把模型差异收敛进 `abi_descriptor`：

```jsonc
{
  "model_name": "qwen3_vl_moe",
  "finished_layer": 12,                 // = step4 算出的 P
  "max_local_layer": 12,
  "hidden_states_ref": "mac_emit://rank0/step0",
  "residual_ref": "mac_emit://rank0/step0#res",
  "transport_codec": "cq4",
  "abi_descriptor": {                   // ← 模型特定，AST 提取
    "layer_signature": "two_arg",       // (hidden, residual) vs single
    "return_arity": 2,                  // 2 元组 vs deepseek 4 元组
    "has_hc_mult": false,               // qwen3 无 hc_mult 维
    "residual_mode": "separate",        // separate vs fused
    "norm_call": "norm(hs, residual)",  // RMSNorm 两参
    "resume_direction": "mac_to_cloud"  // role=receiver
  }
}
```

只要 `abi_descriptor` 字段对，同一份 `DOPDResumePayloadV2` encode/decode 即可跨 deepseek_v4 / qwen3_moe / qwen3_vl_moe 复用——这正是 `state_abi_policy` + `deepseek_abi_bridge` 设计的初衷。

### 2.6 Model ABI Extractor（AST 分析器，step2 核心）

对目标模型 `.py` 做静态分析，提取以下指纹（对照本地 `qwen3_vl_moe.py`）：

| 指纹项 | 提取方式 | qwen3_vl_moe 实测值 |
|--------|---------|---------------------|
| ForCausalLM 类 | `class X(...ForCausalLM)` | `Qwen3VLMoeForConditionalGeneration` |
| Model 类 | `class Y(Qwen3MoeModel)` | `Qwen3MoeLLMModel` |
| 层循环 | `for ... in self.layers[start:end]` | `self.layers[start_layer:end_layer]` |
| 层 forward 签名 | `layer(...)` 实参 | `(positions, hidden_states, forward_batch, residual)` |
| 层返回 | 赋值左侧 | `hidden_states, residual`（2 元组） |
| residual 处理 | `residual is None` 分支 | 首层 `residual=None` 回退 |
| norm 调用 | `self.norm(...)` | `norm(hs, residual)` 两参 |
| MoE | `FusedMoE` / `num_experts` | 是（`config.num_experts`） |
| hc_mult 维 | hidden shape 探测 | 无 |

这些指纹直接映射到 2.5 的 `abi_descriptor`，是模板渲染的全部参数来源——**无需人读源码**。

### 2.7 Patch Generator（resume patch 自动生成，step5 核心）

模板 = 现有 `qwen3_resume_patch.py` 的 `PATCH_BLOCK` 抽象化，参数化以下变量：

```
${MODEL_FORCAUSAL_CLASS}   Qwen3VLMoeForConditionalGeneration
${MODEL_MODEL_CLASS}       Qwen3MoeLLMModel
${LAYER_FORWARD_ARGS}      positions, hidden_states, forward_batch, residual
${LAYER_RETURN}            hidden_states, residual
${HAS_HC_MULT}             false
${RESIDUAL_INJECT}         true
${NORM_CALL}               self.norm(hidden_states, residual)
${TRANSPORT_ROLE}          receiver
${PATCH_MARKER}            cgc_qwen3vl_resume_v1
```

生成器产出 `<model>_resume_patch.py`，含 `apply/revert/dry_run`（与手动版接口一致），并自带 `_CGC_PATCH_MARKER` 幂等标记。非 resume 路径零侵入委托原 forward。

---

## 3. 自动适配 qwen3_vl_moe 的端到端流程

```
$ python -m cgc.auto_agent adapt \
    --model qwen3_vl_moe \
    --model-py Backend/CGC/cloud_sglang/python/sglang/srt/models/qwen3_vl_moe.py \
    --backend sglang --mode edge-cloud --edge-vram-gb 48

step0  scenario        → edge-cloud PD split
step1  hardware        → cloud RTX5090 / edge Mac, system_profile 写入
step2  capture         → AST: Qwen3MoeLLMModel, 层签名 two_arg, residual separate, 无 hc_mult
step3  analyze         → 等价门：resume@P 与 full-forward 数值差 < 1e-4
step4  identify(4D)    → environment=edge_mac, task=pd_split, hardware=mac48G,
                         model=qwen3_vl_moe(28L) → allocated_edge_layers=12 (P=12)
step5  generate        → 渲染 qwen3_vl_moe_resume_patch.py
                         abi_descriptor={layer_signature:two_arg, return_arity:2,
                                         has_hc_mult:false, residual_mode:separate,
                                         norm_call:norm(hs,res), direction:mac_to_cloud}
                         state_abi_path 落盘
step6  dispatch        → bootstrap 协商 dispatch_backend=sglang
                         apply patch to qwen3_vl_moe.py (backup + append PATCH_BLOCK)
                         binding key: qwen3vl_edge_cloud_exec_v1
step7  compare         → 短 ctx local_only / 长 ctx cloud_edge_split 双 bench PASS
step8  combine         → UPK bundle ready=True (4 required keys 齐)
                         validate_profile_bundle PASS → 部署
```

**与手动 `qwen3_resume_patch.py` 的差异**（自动化的价值）：

| 维度 | 手动版 | 自动版 |
|------|--------|--------|
| 适配新模型 | 人读源码、抄 deepseek_v4、手改层签名 | AST 提取指纹 → 模板渲染 |
| 残差/ hc_mult 判断 | 注释里手写差异（易错） | `abi_descriptor` 字段化 |
| 校验 | 无 | g21/g22 state_abi + upk + profile_binding verifier |
| 层切分 P | 硬编码 | step4 4D 矩阵按 VRAM 动态算 |
| 产物一致性 | 单文件 | UPK 7 件套 + binding 三元组 |
| 跨模型复用 | 每模型重抄 | 同一 `DOPDResumePayloadV2` + 不同 abi_descriptor |

> 关键洞察：`qwen3_vl_moe.py` 的 `Qwen3MoeLLMModel` 直接继承 `Qwen3MoeModel`，层签名与 `qwen3_moe` **完全一致**（`(positions, hidden_states, forward_batch, residual)` → `(hidden_states, residual)`，无 hc_mult）。因此自动版只需把模板的 `${MODEL_*}` 换成 VL 类名，resume 逻辑零改动——这正是 state ABI + abi_descriptor 抽象的回报：**模型结构同族时 patch 可自动迁移**。

---

## 4. 与云端 Gate 体系的对接

框架产出的 UPK bundle 直接被现有 verifier 消费：

- `G21StateABIExtensionHookVerifier`（`g21:198`）：校验 `DOPDResumePayloadV2` + `state_abi_policy` + abi bridge → 自动版用 `qwen3vl_min_state_abi_v1` policy。
- `G21UPKFusionBindingVerifier`（`g21:113`）：校验 binding key 三元组 ↔ manifest `profile_binding_ref`。
- `G22StateABIL20NVerifier`（`g22:347`）：校验层级 ABI 字段 ↔ runtime shape。
- `G22UPKL20NOptimizationVerifier`（`g22:298`）：校验 `_write_profile_settings_bundle` + `execution_profile_binding_key` + `upkg_target`。

框架不需新造校验，只需让产物满足这些合约。

---

## 5. 实现路线图

1. **P0 — Model ABI Extractor**：用 `ast` 模块解析 sglang/vllm/mlx 的模型 `.py`，输出 `abi_descriptor` JSON。先覆盖 `qwen3_moe / qwen3_vl_moe / deepseek_v4`。
2. **P1 — Patch Generator**：把 `qwen3_resume_patch.py` 的 `PATCH_BLOCK` 模板化（Jinja2），按 `abi_descriptor` 渲染。
3. **P2 — 八步驱动器**：实现 `AutoAdaptAgent.run()`，串起 step0–step8，每步写 UPK 工件。
4. **P3 — 4D 矩阵 step4**：移植 `[M7.4]` 动态层切分逻辑，按 `--edge-vram-gb` 算 P。
5. **P4 — verifier 对接**：跑 g21/g22 全套校验，`ready=True` 才放行 step8 部署。
6. **P5 — 扩模型**：验证同一框架适配 `deepseek_v4`（single hidden + hc_mult）、`qwen3_vl_moe`（two_arg residual）两个不同 ABI 族。

---

## 附录 A：关键源码定位速查

```
云端 47.95.250.55:/root/flashkv0516/cgc_engine/
├── agent/llm_auto_pipeline.py:2345   _derive_agent_system_profile (system profile)
├── agent/llm_auto_pipeline.py:2980   step0_scenario..step8_combine (agent 八步)
├── agent/llm_auto_pipeline.py:3338   [M7.4] 4D Perception Matrix (step4)
├── agent/cli.py:6,62,1370            "unified 8-step pipeline" CLI
├── pipeline.py:2123                  matrix_axes (4D/5轴字段)
├── pipeline.py:2153                  "CGC 2.0 八步流水線" + "4D 矩陣：環境×任務×硬體×模型"
├── pipeline_contract_common.py:7     PIPELINE_KERNEL_ARTIFACT_KEYS (UPK 7件套)
├── pipeline_contract_common.py:9,19  state_abi_path (REQUIRED)
├── pd/dopd_schema.py:47              DOPDResumePayloadV2 (state ABI 载荷)
├── integrated_gate_pipeline.py:76    bootstrap_contract (引导契约)
├── gate_verifiers/g21_fusion_governance_verifier.py:113   G21UPKFusionBindingVerifier
├── gate_verifiers/g21_fusion_governance_verifier.py:198   G21StateABIExtensionHookVerifier
├── gate_verifiers/g22_deepep_l20n_verifier.py:93          has_profile_binding
├── gate_verifiers/g22_deepep_l20n_verifier.py:243         bootstrap binding
├── gate_verifiers/g22_deepep_l20n_verifier.py:283         G22SystemProfileL20NVerifier
├── gate_verifiers/g22_deepep_l20n_verifier.py:298         G22UPKL20NOptimizationVerifier
└── gate_verifiers/g22_deepep_l20n_verifier.py:347         G22StateABIL20NVerifier

本地 /Users/alexchuang/Documents/flashkv0516/
├── CGC_Phase2/qwen3_resume_patch.py          手动 resume patch（自动化参照基线）
├── CGC_Phase2/deepseek_v4.py                  deepseek_v4 resume（patch 移植源）
├── CGC_Phase2/cgc_handoff_transport.py        MacEmitHandoff transport
└── ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python/sglang/srt/models/qwen3_vl_moe.py
                                               Qwen3MoeLLMModel / Qwen3VLMoeForConditionalGeneration
```
