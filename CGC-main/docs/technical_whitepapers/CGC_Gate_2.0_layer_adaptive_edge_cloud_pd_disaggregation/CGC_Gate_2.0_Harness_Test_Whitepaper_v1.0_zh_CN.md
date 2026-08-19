# CGC Gate 2.0 Harness Test 技术白皮书 v1.0

**版本**: v1.0（复合 gate 版）
**归属**: CGC Self-Harness 测试框架 — Gate 2.0 测试套件
**对应 Gate**: `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation`
**测试入口**: `SelfHarnessValidator.run_gate_2_0_harness()`（`cgc_engine/tools/scripts/run/self_harness_validation_framework.py`）

> 本白皮书由原 `CGC_SelfHarness_Test_Framework_Technical_Whitepaper_v1.0_zh_CN.md` 拆分而来，专门覆盖 Gate 2.0 复合 gate（吸收原 Gate 2.1 / 2.2 / 2.3）的测试范围。Gate 1.0 测试内容见 `CGC_Gate_1.0_edge_cloud_autonomy/CGC_Gate_1.0_Harness_Test_Whitepaper_v1.0_zh_CN.md`。

---

## 0. 复合 gate 合并声明

本白皮书 v1.0 已将以下子 gate 的测试范围正式收口为 Gate 2.0 测试套件：

| 原 subgate | 测试能力数 | 合併後能力 ID 前綴 |
|---|---|---|
| `CGC_Gate_2.1_speculative_decode_fusion_optimization` | 11 | `g21_` |
| `CGC_Gate_2.2_deepep_moe_load_balancing` | 7 | `g22_deepep_` |
| `CGC_Gate_2.2_kv_cache_optimization` | 4 | `g22_kv_` |
| `CGC_Gate_2.3_unlimited_rswa_prefill_pool` | 7 | `g23_` |

**Gate 2.0 测试套件总能力数**：
- Self-Harness 真实硬件验证代表性能力：10 项（dflash / jetspec / dspk / flashmoe / omlx / rswa_double_layer_kv / prefill_pool / gds_direct_io / nfsordma / trueorthokda_kv_management）
- 完整能力矩阵（gate_test_framework.py）：51 项（22 本体 + 11 Gate 2.1 + 7 Gate 2.2 DeepEP + 4 Gate 2.2 KV + 7 Gate 2.3）

原 Gate 2.1 / 2.2 / 2.3 的独立测试入口已废弃，调用旧 gate id 会被重定向到 Gate 2.0 复合 gate（见 `_LEGACY_GATE_REDIRECT`）。

---

## 1. 文档目标

本文档定义 `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation` 的 Self-Harness 测试套件边界，回答以下问题：

1. Gate 2.0 测试套件覆盖哪些能力（含原 2.1/2.2/2.3 复合能力）
2. 三阶段闭环在 Gate 2.0 复合 gate 上如何执行
3. 各子能力组（2.1/2.2/2.3）的 CLI 验证参数
4. UnifiedIRInjector 整图注入如何被测试验证
5. 端云 MoE 一层一层张量传输主路径如何被测试覆盖
6. DSpark / JetSpec 上游引用能力的测试边界

---

## 2. Gate 2.0 测试套件范围

### 2.1 Gate 2.0 本体能力（22 项）

| 能力 ID | 能力名称 | 验证参数 | 代码位置 |
|---------|----------|----------|----------|
| `edge_omlx_flashmoe_autonomous_entry` | OMLX + FlashMoE 端侧自治入口 | `--omlx --flashmoe --edge-autonomy --memory-budget` | `app/edge_engine/local_infer.py` |
| `cq4_transport_plane` | CQ4 端云协议承载层 | `--cq4 --state-transport --protocol-contract` | `cgc_engine/pd/pd_client.py` |
| `trueorthokda_zero_copy_state_runtime` | TrueOrthoKDA + Zero-Copy 状态运行时 | `--trueorthokda --zero-copy --device-resume` | `cgc_engine/cgc/true_ortho_kda.py` |
| `dopd_handoff_control_plane` | PD → DOPD handoff 控制面 | `--dopd --prepare-commit-resume --session-handoff` | `cgc_engine/pd/dopd_schema.py` |
| `real_prefill_producer_and_auto_publish` | 云侧真实 prefill producer + auto-publish | `--prefill-producer --auto-publish --streaming-handoff` | `app/servers/cgc_api_server.py` |
| `task_type_contract_and_bundle_governance` | task_type contract + 四段 bundle governance | `--task-type-contract --profile-bundle-validator --bundle-review` | `app/shared/profile_bundle_validator.py` |
| `sglang_deepep_tp4ep4_prefill_foundation` | SGLang TP4EP4 云侧 prefill 主干 | `--sglang --tp4 --ep4 --cloud-prefill` | `cloud_sglang/python/sglang/srt/models/deepseek_v4.py` |
| `deepep_route_contract_dispatch_profile` | DeepEP route contract + dispatch profile | `--deepep --route-contract --dispatch-profile` | `Backend/CGC/deepep_sglang_patch.py` |
| `ray_engine_dual_host_service_topology` | Ray engine 双主机 service topology | `--ray --ray-serve --sglang-gateway --dual-host` | `app/servers/cgc_api_server.py` |
| `colossalai_distributed_runtime_candidate` | ColossalAI distributed runtime 候选 | `--colossalai --distributed-runtime-candidate --hybrid-parallel-plugin` | M7.6 候选 backend contract |
| `deepseek_v4_flash_resume_decode_path` | DeepSeek-V4-Flash 云侧 resume/decode 路径 | `--deepseek-v4-flash --resume-decode` | `cloud_sglang/python/sglang/srt/models/deepseek_v4.py` |
| `m77_cloud_edge_q2rl_consumption_anchor` | m77 cloud-edge Q2RL 消费锚点 | `--m77 --cloud-edge-q2rl --validated-run` | `cgc_engine/product/m77_gate.py` |
| `m78_teaching_pure_llm_consumption_anchor` | m78 GUI teaching / pure LLM 消费锚点 | `--m78 --gui-teaching --pure-llm` | `cgc_engine/product/m78_gate.py` |
| `upkg20_model_product_binding` | UPKG 2.0 模型产品 gate 承接 | `--upkg --version=2.0 --model-product-gate` | `app/cli/cgc.py` |
| `upkg3x_agent_product_binding` | UPKG 3.x agent product chain 承接 | `--upkg3x --agent-product-chain --validated-run` | `cgc_engine/product/m77_gate.py` + `m78_gate.py` |
| `m76_dev_gate_proof_anchor` | m7.6 dev gate 异构集成 bring-up 锚点 | `--m76 --heterogeneous-integration --dflash-contract` | `app/shared/contracts/task_type_contract.json` |
| `max_local_layer_dynamic_partition` | 端侧 max_local_layer 层粒度动态切分 | `--max-local-layer=16 --layer-partition --adaptive-split --vram-watermark` | `app/edge_engine/local_infer.py` |
| `finished_layer_prefill_continuation` | finished_layer 驱动云侧按层接续 Prefill | `--finished-layer --prefill-continuation --layer-resume` | `cloud_sglang/python/sglang/srt/models/deepseek_v4.py` |
| `hidden_states_partial_kv_abi` | hidden_states + partial_kv 正式中间态 ABI | `--hidden-states --partial-kv --abi-version=v2 --dopd-resume-payload-v2` | `cgc_engine/pd/dopd_schema.py` |
| `layer_wise_kv_streaming_to_decode` | 层流式 KV 同步至 Decode 集群 | `--layer-wise --streaming-kv --decode-cluster --mooncake` | `deepseek_v4.py` + `mooncake_transfer_engine.py` |
| `udiq2_kda_joint_transport_profile` | UD-IQ2 2bit + KDA 联合传输档位 | `--udiq2 --kda --transport-profile --low-bit` | `cgc_engine/cgc/true_ortho_kda.py` |
| `moe_route_consistency_across_edge_cloud` | MoE 专家路由跨端云一致性 | `--moe --route-consistency --unified-ir-inject --topk-inject` | `Backend/CGC/compiler/unified_compiler.py` |

### 2.2 原 Gate 2.1 — 投机解码融合优化（11 项）

| 能力 ID | 能力名称 | 验证参数 |
|---------|----------|----------|
| `g21_dflash_control_baseline` | DFlash 控制基线 | `--dflash --control-baseline` |
| `g21_trace_replay_governance_chain` | host1-host2 trace + replay 治理链 | `--trace-replay --governance` |
| `g21_machine_consumable_fusion_artifacts` | 机读 fusion artifacts | `--fusion-artifacts --machine-consumable` |
| `g21_bootstrap_contract_binding_surface` | Bootstrap contract 绑定面 | `--bootstrap-contract --binding-surface` |
| `g21_system_profile_and_profile_settings_binding_surface` | System profile + profile settings 绑定面 | `--system-profile --profile-settings` |
| `g21_eight_step_pipeline_governance_integration` | 8-step pipeline 治理整合 | `--eight-step --pipeline-governance` |
| `g21_upk_binding_for_fusion_variants` | fusion variants UPK 绑定 | `--upk --variant-binding --fusion-variants` |
| `g21_state_abi_extension_hook` | State ABI 扩展钩子 | `--state-abi --extension-hook --tree-verify` |
| `g21_dspark_scheduler_runtime_adapter` | DSpark scheduler runtime adapter | `--dspark --scheduler --upstream-open-source`（上游: DeepSpec） |
| `g21_jetspec_draft_runtime_adapter` | JetSpec draft runtime adapter | `--jetspec --draft-runtime --upstream-open-source`（上游: hao-ai-lab/JetSpec） |
| `g21_verified500_speedup_closure` | Verified 500 加速闭环 | `--verified-500 --speedup-closure` |

### 2.3 原 Gate 2.2 — DeepEP MoE 负载均衡（7 项）

| 能力 ID | 能力名称 | 验证参数 |
|---------|----------|----------|
| `g22_deepep_l20n_dualnode_16gpus` | L20N 双节点 16-GPU 优化 | `--deepep --l20n --dual-node --gpu-count=16` |
| `g22_deepep_l20n_megatrain_8step` | L20N 训练 8-step pipeline | `--deepep --l20n --megatrain-8step` |
| `g22_deepep_l20n_inference_8step` | L20N 推理 8-step pipeline | `--deepep --l20n --inference-8step` |
| `g22_deepep_bootstrap_deepep_compat` | Bootstrap DeepEP 兼容性 | `--bootstrap --deepep-compat` |
| `g22_deepep_system_profile_l20n` | System Profile L20N 支持 | `--system-profile --l20n` |
| `g22_deepep_upk_l20n_optimization` | UPK L20N 优化 | `--upk --l20n --optimization` |
| `g22_deepep_state_abi_l20n` | State ABI L20N 支持 | `--state-abi --l20n` |

### 2.4 原 Gate 2.2 — KV Cache 优化（4 项）

| 能力 ID | 能力名称 | 验证参数 |
|---------|----------|----------|
| `g22_kv_kv_cache_management` | KV 缓存管理 | `--kv-cache-management --kv-cache-builder` |
| `g22_kv_cache_reuse` | 缓存复用优化 | `--kv-cache-reuse --radix-cache` |
| `g22_kv_dynamic_cache_sizing` | 动态缓存大小 | `--dynamic-cache-sizing` |
| `g22_kv_cache_prefetching` | 缓存预取优化 | `--kv-cache-prefetching --mooncake-prefetch` |

### 2.5 原 Gate 2.3 — 无限 R-SWA + Prefill Pool（7 项）

| 能力 ID | 能力名称 | 验证参数 |
|---------|----------|----------|
| `g23_rswa_double_layer_kv` | R-SWA 双层 KV 结构 | `--rswa --double-layer-kv --reference-kv --output-kv --window-size=128` |
| `g23_prefill_pool_dynamic_management` | Prefill Pool 动态块管理 | `--prefill-pool --dynamic-block --hot-chunk-load --cold-chunk-unload` |
| `g23_gds_nfsordma_direct_io` | GDS + NFSoRDMA 直写显存 | `--gds --nfsordma --direct-io --zero-cpu-copy` |
| `g23_trueorthokda_adapter` | TrueOrthoKDA 适配 | `--trueorthokda --rswa-adapter` |
| `g23_cloud_l20n_tp4_adaptation` | 云端 L20N 双 TP4 适配 | `--l20n --tp4 --no-pcie-storm` |
| `g23_unified_ir_inject_sglang_compute_graph` | UnifiedIRInjector 整图注入 | `--unified-ir --inject-sglang --attention-inject --topk-inject --fusedmoe-inject` |
| `g23_endtoend_moe_tensor_transport` | 端云 MoE 一层一层张量传输 | `--endtoend --moe-tensor-transport --deep-ep --nfsordma --cq4` |

### 2.6 移除的 target 能力（不进入测试套件）

- `multimodal_input_support`（原 Gate 2.3）— 移至 future scope
- `edge_npu_adaptation`（原 Gate 2.3）— 移至 future scope

---

## 3. 三阶段闭环流程

### 3.1 阶段定义

| 阶段 | 名称 | CLI 命令 | Gate 2.0 职责 |
|------|------|----------|--------------|
| Stage 1 | 策略决策 | `cgc model verify` | 验证层粒度切分 / 接续 / KV 流式 / MoE 路由一致性 |
| Stage 2 | 图捕获 | `cgc model audit` | 审计 DOPDResumePayloadV2 / UnifiedIRInjector 注入配置 / RSWA KV 结构 |
| Stage 3 | 执行校验 | `cgc model deploy` | 端云 MoE 一层一层张量传输端到端执行 |

### 3.2 执行模式

- **direct_cli**：直接调用 CGC CLI 命令
- **cgc_agent**：通过 CGC Agent 代理执行（`cgc agent model verify ...`）
- **self_harness_three_stage**：三阶段闭环模式（默认）

### 3.3 复合 gate 重定向

调用旧 gate id 会被自动重定向到 Gate 2.0 复合 gate：

```python
_LEGACY_GATE_REDIRECT = {
    'CGC_Gate_2.1_speculative_decode_fusion_optimization': 'CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation',
    'CGC_Gate_2.2_deepep_moe_load_balancing': 'CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation',
    'CGC_Gate_2.2_kv_cache_optimization': 'CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation',
    'CGC_Gate_2.3_unlimited_rswa_prefill_pool': 'CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation',
}
```

---

## 4. CLI 调用示例

### 4.1 运行 Gate 2.0 harness test

```bash
# 通过 self_harness_validation_framework.py
python3 cgc_engine/tools/scripts/run/self_harness_validation_framework.py --gate 2.0

# 通过 gate_test_framework.py
python3 cgc_engine/tools/scripts/run/gate_test_framework.py \
    --self-harness \
    --gate CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation

# 调用旧 gate id（自动重定向到 2.0）
python3 cgc_engine/tools/scripts/run/gate_test_framework.py \
    --gate CGC_Gate_2.1_speculative_decode_fusion_optimization
# → redirected to CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation
```

### 4.2 验证核心 2.0 能力

```bash
# 层粒度切分 + 接续
cgc model verify --gate 2.0 --max-local-layer --finished-layer --partial-kv

# UnifiedIRInjector 整图注入
cgc model verify --gate 2.0 --unified-ir --inject-sglang --attention-inject --topk-inject --fusedmoe-inject

# 端云 MoE 一层一层张量传输
cgc model verify --gate 2.0 --endtoend --moe-tensor-transport --deep-ep --nfsordma --cq4

# 投机解码融合
cgc model verify --gate 2.0 --enable-speculative --speculative-mode=fusion --dspark-budget=64 --jetspec-branches=8

# RSWA + Prefill Pool
cgc model verify --gate 2.0 --rswa --prefill-pool --gds --nfsordma
```

### 4.3 SWE Verified 500 加速闭环

```bash
cgc validate --capability swe_verified_500
cgc model verify --task-type swe --enable-swe-optimization
cgc model verify --fusion-config swe --gate 6.0
```

---

## 5. DSpark / JetSpec 上游引用测试边界

`g21_dspark_scheduler_runtime_adapter` 与 `g21_jetspec_draft_runtime_adapter` 引用上游开源实现：

- DSpark: `https://github.com/deepseek-ai/DeepSpec`
- JetSpec: `https://github.com/hao-ai-lab/JetSpec`

测试套件验证：
- ✅ 上游引用 URL 可访问
- ✅ `evidence_tags` 包含 `upstream_open_source`
- ✅ vendored SGLang runtime 整合为后续工程项目（不在当前测试范围）

---

## 6. 性能指标

| 指标 | 预期值 |
|------|--------|
| 单能力测试 | < 30 秒 |
| Gate 2.0 self-harness 真实硬件验证（10 项） | < 10 分钟 |
| Gate 2.0 完整能力矩阵（51 项） | < 25 分钟 |
| 端云 MoE 张量传输端到端 | < 2 分钟/请求 |
| Verified 500 加速闭环 | < 45 分钟 |

---

## 7. 成功率标准

| 状态 | 定义 |
|------|------|
| PASS | CLI 命令返回码为 0 或 -2 |
| FAIL | CLI 命令返回码非零且非 -2 |
| ERROR | 测试执行异常 |
| SKIP | 能力被跳过（如缺少硬件：GDS / NFSoRDMA / L20N GPU） |

---

## 8. 主机同步

Gate 2.0 测试相关文件需同步至 host1 (39.106.118.206) 与 host2 (47.95.250.55)：

- `cgc_engine/tools/scripts/run/self_harness_validation_framework.py`
- `cgc_engine/tools/scripts/run/gate_test_framework.py`
- `docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/CGC_Gate_2.0_Harness_Test_Whitepaper_v1.0_zh_CN.md`
- `docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation_gate_map.json`

```bash
rsync -av cgc_engine/tools/scripts/run/self_harness_validation_framework.py root@39.106.118.206:/path/to/dest/
rsync -av docs/technical_whitepapers/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/ root@39.106.118.206:/path/to/docs/CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation/
```

---

## 9. 报告格式

```json
{
  "gate_id": "CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation",
  "gate_version": "2.0",
  "execution_mode": "self_harness_three_stage",
  "self_harness_real_hardware_caps": 10,
  "full_capability_matrix": 51,
  "passed": 51,
  "failed": 0,
  "skipped": 0,
  "errors": 0,
  "composite_structure": {
    "core_2_0": 22,
    "gate_2_1": 11,
    "gate_2_2_deepep": 7,
    "gate_2_2_kv": 4,
    "gate_2_3": 7
  },
  "capabilities": [
    {
      "capability_id": "dflash",
      "name": "DFlash 投机解码",
      "status": "PASS",
      "evidence": ["..."]
    }
  ]
}
```

---

**文档版本**: v1.0（复合 gate 版）
**创建日期**: 2026-07-05
**状态**: 正式发布
**拆分自**: `CGC_SelfHarness_Test_Framework_Technical_Whitepaper_v1.0_zh_CN.md`
**适用版本**: CGC Engine v2.1+
