# 4DSP 统一架构 — 两层六抽象十步流水线

> 4DSP = 4D perception + State ABI + Pipeline kernel
> 目标: 跨后端 (llama.cpp/MLX/SGLang) x 跨模型 (Gemma4/Qwen/DSV4) x 跨平台 (Mac/Linux) x 跨硬件 (Metal/CUDA/CPU)

## 一、两层硬核结构

### 下层: MTP Decode 引擎 (ALU / Tensor Core)

| 属性 | 值 |
|------|-----|
| 定位 | 端侧 AI 的解码执行单元 |
| 功能 | 多候选 token 并行预测, 跨模型兼容 draft 生成, 跨后端统一执行, 固定算力开销 |
| 代码 | `app/shared/spec_decode_ir.py` (SpecDecodeBackend), `app/shared/unified_mtp_ir.py` (MTPHeadIR), `CGC_Phase2/mtp_head/` (训练) |
| 流水线步骤 | [3/10] 格式解析 ~ [7.7/10] MTP同步 + [11.5] AutoTunner + [11.6] Magicompiler |

### 上层: edge_first_proxy 调度引擎 (Cache + MMU)

| 属性 | 值 |
|------|-----|
| 定位 | 端侧 AI 的体验控制单元 + 云边网关 |
| 功能 | 语义级缓存匹配 (L1-L5), 模式识别 & 快速回复, 认知路由决策, 体验兜底与流式输出 |
| 代码 | `app/servers/edge_first_proxy.py`, `app/shared/seamless_switcher.py` |
| 流水线步骤 | [1/10] 系统侦测 ~ [2/10] CPU侦测 + [8/10] 上下文 + [9/10] 4D矩阵 + [10/10] 磁碟 + [11] 切换器 |

## 二、七个统一抽象

已有代码中的 7 个抽象, 各自角色:

| # | 抽象 | 角色 | 代码位置 | schema |
|---|------|------|---------|--------|
| 1 | 4D 感知矩陣 | INPUT — 感知输入 | `app/shared/route_decision.py` `build_4d_matrix()` | D1网络/D2硬件/D3模型/D4路由 |
| 2 | State ABI | CONTRACT — 层间合约 | `app/cli/cgc.py` `_materialize_pipeline_contract_seed_compat()` | `generic_state_abi_v1` |
| 3 | Pipeline Kernel | EXECUTION — 共享执行产物 | `ComputeGraphCompiler-main/cgc_engine/pipeline_contract_common.py` | 7 artifact paths |
| 4 | Bootstrap | ACTIVATION — 激活序列 | `app/cli/embodied_contract_profiles.py` `CANONICAL_BOOTSTRAP_TEMPLATES` | 4 bootstrap families |
| 5 | System Profile | CONFIG — 配置选择器 | `app/cli/embodied_contract_profiles.py` `CANONICAL_EXECUTION_PROFILES` | 4 profiles |
| 6 | Profile Binding | BINDING — 配置绑定运行时 | `app/cli/cgc.py` `_profile_binding_fields()` + `_infer_canonical_profile_binding()` | binding keys |
| 7 | Unified Runtime IR | LOWERING — 统一运行时意图与 adapter lowering 边界 | `docs/UNIFIED_RUNTIME_IR_V0.md`, `app/shared/colibri_backend.py`, `app/shared/spec_decode_ir.py` | `unified_runtime_ir_v0` |

### 抽象之间的数据流

```
4D 感知矩陣 (感知)
    |
    v
System Profile (选配置: local_infer / edge_cloud_infer / ...)
    |
    v
Profile Binding (绑定: 把 profile 绑到运行时)
    |
    v
Bootstrap (激活: 按 bootstrap_steps 顺序启动两层)
    |
    v
Unified Runtime IR (lowering: 统一运行时意图 -> backend request)
    |
    v
State ABI (合约: 定义上层 produces/consumes + 下层 produces/consumes)
    |
    v
Pipeline Kernel (执行: 7 个 artifact path 共享)
    |
    v
上层 edge_first_proxy <-> 下层 MTP decode (双向执行)
```

### 新增抽象 7: Unified Runtime IR

`Unified Runtime IR` 不是 kernel IR, 而是控制面 IR。

它负责统一描述:

- model identity
- backend family / execution intent
- decode strategy intent
- residency / placement intent
- adapter capability contract

它不负责统一:

- `gemma4.c` 的 `slot1` / sticky / bootstrap 内部参数
- TurboFieldfare 的 shared-core packing / SSD paging 内部算法
- MLX / llama.cpp / SGLang 的 backend-specific launch knobs

它的角色是:

```text
Hermes 4D Matrix / TenStepPipeline / ProfileBinding
    -> Unified Runtime IR
    -> Adapter.lower(ir)
    -> BackendRequest
    -> begin_request / prefill / generate / snapshot
```

正式草案见: `docs/UNIFIED_RUNTIME_IR_V0.md`

## 三、十步流水线 — 四阶段两层映射

### Phase 1: Bootstrap 引导 (上层, Steps 1-2)

| 步骤 | 名称 | 层级 | 抽象 | 代码 |
|------|------|------|------|------|
| [1/10] | 系统侦测 | 上层 | 4D-D2 | `hardware_sensing.detect_os()` |
| [2/10] | CPU 侦测 | 上层 | 4D-D2 | `hardware_sensing.detect_cpu()` |

**产出**: `HardwareInfo` (os, cpu_brand, cpu_cores) → 喂给 4D 矩阵 D2

**Dev Phase 2 映射**: edge_first_proxy 启动时执行, 检测 OS/CPU 决定本地是否能跑 MLX

### Phase 2: Perception 感知 (共享, Steps 3-5.5)

| 步骤 | 名称 | 层级 | 抽象 | 代码 |
|------|------|------|------|------|
| [3/10] | 格式解析 | 共享 | State ABI: model_format | 内联 (GGUF/MLX/SafeTensors) |
| [4/10] | 云路架构 | 共享 | 4D-D3 | `route_decision.get_model_info()` |
| [5/10] | 记忆体水位 | 共享 | 4D-D2 | `hardware_sensing.detect_memory()` |
| [5.5/10] | 算力等级 | 共享 | 4D-D2 | `hardware_sensing.classify_compute_tier()` |

**产出**: 模型格式 + MoE/Dense 拓扑 + 内存水位 + compute tier → 统一 4D 矩阵

**Dev Phase 1 映射**: sglang 重启时执行, 解析 Gemma4 格式, 检测 GPU VRAM, 决定 TP 数

### Phase 3: Configuration 配置 (下层, Steps 6-7.7)

| 步骤 | 名称 | 层级 | 抽象 | 代码 |
|------|------|------|------|------|
| [6/10] | 引擎路由 | 下层 | Bootstrap: engine_binding | `hardware_sensing.recommend_engine()` |
| [7/10] | 记忆体策略 | 下层 | State ABI: memory_plan | 内联 (FlashMoE) |
| [7.5/10] | 路由决策 | 共享 | Profile Binding: route_mode | `route_decision.compute_route()` |
| [7.6/10] | 模型分发 | 下层 | Bootstrap: dispatch_action | `model_dispatcher.ModelDispatcher.decide()` |
| [7.7/10] | MTP 同步 | 下层 | State ABI: draft_model_binding | `model_dispatcher.MTPDraftSyncer.check_and_sync()` |
| [7.8/10]* | Runtime IR Lowering | 共享 | Unified Runtime IR: adapter lowering | `docs/UNIFIED_RUNTIME_IR_V0.md`, `app/shared/colibri_backend.py` |

**产出**: 引擎绑定 (OMLX/CUDA/ROCm) + 路由模式 (PD/Cloud/Local) + MTP draft model 状态 + backend request lowering

\* `7.8/10` 是内部架构挂载点, 不要求对外 CLI 显示新增编号; 它表示 `Step 7.6 -> Step 8` 之间需要显式的统一 IR -> adapter lowering 边界。

**Dev Phase 3 映射**: MTP head 训练时执行, Step 7.7 触发 `mtp_trainer.integrate_with_step_77()` 自动训练流水线

### Phase 4: Execution 执行 (共享, Steps 8-11.6)

| 步骤 | 名称 | 层级 | 抽象 | 代码 |
|------|------|------|------|------|
| [8/10] | 上下文构建 | 上层 | Pipeline Kernel: execution_context | 内联 (参数注入) |
| [9/10] | 4D 矩阵上报 | 共享 | Pipeline Kernel: state_abi | `route_decision.build_4d_matrix()` |
| [10/10] | 磁碟检查 | 上层 | Bootstrap: resource_check | `hardware_sensing.detect_disk()` |
| [11] | 无缝切换器 | 上层 | Profile Binding: runtime_switch | `seamless_switcher.SeamlessSwitcher` |
| [11.5] | AutoTunner | 下层 | State ABI: spec_config | `spec_decode_ir.AutoTunner.get_optimal_config()` |
| [11.6] | Magicompiler IR | 下层 | Pipeline Kernel: ir_pass | `rswa_magicompiler_ir.AutoTunnerMagicompiler` |

**产出**: execution_context + state_abi + SeamlessSwitcher (运行时监控) + AutoTunner (投机配置) + Magicompiler (IR pass)

其中 `execution_context` 的上游新增为:

```text
Profile Binding
    -> Unified Runtime IR
    -> backend adapter lowering
    -> execution_context
```

**Dev 全部 3 phase 汇聚**: Phase 4 是运行时, 三个 dev phase 的产出都在这里集成

## 四、State ABI — 层间合约

`state_abi.json` (`schema_version: generic_state_abi_v1`) 定义两层之间的 produces/consumes:

```json
{
  "schema_version": "generic_state_abi_v1",
  "layers": {
    "decode_engine": {
      "produces": [
        "draft_tokens",
        "hidden_states",
        "accept_rate",
        "decode_tps"
      ],
      "consumes": [
        "model_artifact",
        "spec_config",
        "memory_budget"
      ]
    },
    "proxy_engine": {
      "produces": [
        "cache_hit",
        "prediction",
        "route_decision",
        "ttft_ms"
      ],
      "consumes": [
        "draft_tokens",
        "4d_matrix",
        "execution_context"
      ]
    }
  },
  "binding": {
    "profile": "edge_cloud_infer",
    "pipeline_kernel_artifacts": [
      "execution_context_path",
      "state_abi_path",
      "strategy_decision_path",
      "compatibility_report_path",
      "distributed_runtime_bootstrap_path",
      "contract_manifest_path",
      "system_execution_manifest_path"
    ]
  }
}
```

**关键**: 上层 consumes `draft_tokens` (下层 produces), 下层 consumes `spec_config` (上层通过 AutoTunner 产出). 双向依赖, State ABI 是合约.

## 五、Profile Binding → Bootstrap 映射

4 个 canonical execution profiles, 每个绑定不同的 bootstrap 模板:

| Profile | 下层行为 | 上层行为 | Bootstrap Steps | 适用场景 |
|---------|---------|---------|----------------|---------|
| `local_infer` | MLX 本地 decode, 无 cloud | cache only, 无 cloud route | resolve_model_runtime, bind_local_runtime_host, load_model_artifact | Mac 离线, 隐私模式 |
| `local_train` | MTP head 训练 (PyTorch) | N/A (训练无 proxy) | resolve_training_workspace, bind_local_training_runtime, materialize_state_abi | Host1 训练 MTP head |
| `edge_cloud_infer` | MTP draft → cloud verify | cache + prediction + route to cloud | load_publish_manifest, bind_bridge_contract, prepare_edge_delivery_channel, activate_edge_consume_runtime | **CGC 生产模式** |
| `edge_cloud_train` | MTP head 跨设备训练 | N/A | 同 edge_cloud_infer + 训练步骤 | Host1 训练 + Host2 验证 |

**Profile Binding 流程**:
1. 4D 矩阵产出 → `_infer_canonical_profile_binding()` 推断 profile
2. profile → `CANONICAL_BOOTSTRAP_TEMPLATES[profile]` 取 bootstrap 模板
3. bootstrap 模板 → 按 `bootstrap_steps` 顺序激活两层
4. 激活后 → State ABI 合约生效 → Pipeline Kernel 共享 artifacts

## 六、跨后端统一执行

### SpecDecodeConfig — 一份配置跨三后端

`app/shared/spec_decode_ir.py` 的 `SpecDecodeConfig` 是跨后端统一配置:

```python
SpecDecodeConfig(
    backend="mlx",        # mlx | pytorch | sglang
    mode="chain",         # chain | eagle | pipeline
    num_draft_tokens=16,  # chain: N
    top_k=4,              # eagle: top-k
    pipeline_cloud_url="http://47.95.250.55:30001",  # pipeline: cloud verify URL
    pipeline_overlap=True, # draft/verify 重叠
)
```

**一份 JSON 配置 → 三个后端共享**:
- `MLXBackend` (Mac Metal): `mlx_target_path`, `mlx_draft_model`
- `PyTorchBackend` (GPU): `pytorch_target_path`, `pytorch_device`
- `SGLangBackend` (Cloud): `sglang_target_url`, `sglang_speculative_algorithm`

### MTPHeadIR — 一份算法描述跨三后端

`app/shared/unified_mtp_ir.py` 的 `MTPHeadIR` 定义 MTP head 计算图一次:

```python
MTPHeadIR(
    hidden_size=2816,     # Gemma4
    vocab_size=262144,    # Gemma4
    num_heads=16,
    head_dim=256,
    layers=[...],         # 自动构建计算图
)
```

**改 IR 一处 → 三后端自动同步**:
- PyTorch model (训练) — `to_pytorch()`
- MLX model (Mac 推理) — `to_mlx()`
- SGLang EAGLE config (cloud 投机) — `to_sglang_config()`

## 七、三个开发 Phase 在统一架构中的位置

| Dev Phase | 内容 | 流水线阶段 | 层级 | 抽象角色 |
|-----------|------|-----------|------|---------|
| **Dev Phase 1** | sglang 重启 (MTP5+topk2, cuda-graph) | Phase 2 Perception + Phase 3 Configuration | 下层 | 4D-D3 模型感知 → Bootstrap: engine_binding → State ABI: draft_model_binding |
| **Dev Phase 2** | calibrate_first_token + edge_first_proxy 启动 | Phase 1 Bootstrap + Phase 4 Execution | 上层 | 4D-D2 硬件感知 → Profile Binding: edge_cloud_infer → Pipeline Kernel: execution_context |
| **Dev Phase 3** | MTP head 训练 (decode data 收集 → 训练 → 验证) | Phase 3 Configuration + State ABI | 下层 | Bootstrap: local_train → State ABI: draft_model_binding → Pipeline Kernel: ir_pass |

**三个 dev phase 汇聚点**: Phase 4 Execution — sglang (下层) + edge_first_proxy (上层) + SeamlessSwitcher (共享) 同时运行, 通过 State ABI 合约双向通信.

## 八、跨平台/跨模型适配清单

### Gemma4-26B-A4B 适配 (当前目标)

| 参数 | DSV4/Qwen (现有) | Gemma4 (目标) | 代码修改点 |
|------|------------------|--------------|-----------|
| hidden_size | 2048 | 2816 | `unified_mtp_ir.py` MTPHeadIR |
| vocab_size | 151936 | 262144 | `unified_mtp_ir.py` MTPHeadIR |
| head_dim | 128 | 256 | `unified_mtp_ir.py` MTPHeadIR |
| model_type | qwen3 | gemma4 | `model_loader.py` detect_model_type() |
| EOS tokens | 151644/151645 | [1, 106] | `train_chained_decode_multi.py` |
| MoE experts | 无 | 128 | `route_decision.py` MODEL_PRESETS |

### 跨后端部署路径

```
训练 (Host1, PyTorch)
  → MTPHeadIR.to_pytorch() → 训练 → checkpoint
  → MTPHeadIR.to_mlx() → Mac MLX 推理 (edge draft)
  → MTPHeadIR.to_sglang_config() → Cloud SGLang 投机 (cloud verify)
```

## 九、实现优先级

1. **Dev Phase 1** (Host1 sglang 重启) — 验证下层 MTP decode 引擎基线
2. **Dev Phase 2** (edge_first_proxy for Gemma4) — 验证上层调度引擎
3. **Dev Phase 3** (MTP head 训练) — 打通 State ABI 合约 (accept 0% → 30-50%)

Phase 1 + 2 完成后, 两层各自能跑但**没有合约** (State ABI 未激活). Phase 3 完成后, State ABI 合约生效, 两层通过 draft_tokens / spec_config 双向通信, 形成完整的 4DSP 统一架构.
