# CGC Gate 1.0 Harness Test 技术白皮书 v1.0

**版本**: v1.0
**归属**: CGC Self-Harness 测试框架 — Gate 1.0 测试套件
**对应 Gate**: `CGC_Gate_1.0_edge_cloud_autonomy`
**测试入口**: `SelfHarnessValidator.run_gate_1_0_harness()`（`cgc_engine/tools/scripts/run/self_harness_validation_framework.py`）

> 本白皮书由原 `CGC_SelfHarness_Test_Framework_Technical_Whitepaper_v1.0_zh_CN.md` 拆分而来，专门覆盖 Gate 1.0 端云自治 (edge-cloud autonomy) 的测试范围。Gate 2.0 及以上测试内容见各自 gate 文件夹下的 Harness Test 白皮书。

---

## 1. 文档目标

本文档定义 `CGC_Gate_1.0_edge_cloud_autonomy` 的 Self-Harness 测试套件边界，回答以下问题：

1. Gate 1.0 测试套件覆盖哪些能力
2. 三阶段闭环（策略决策 → 图捕获 → 执行校验）在 Gate 1.0 上如何执行
3. Gate 1.0 测试使用的 CLI 命令与验证参数
4. 测试通过状态语义（PASS / FAIL / SKIP / ERROR）
5. Gate 1.0 测试报告字段与主机同步要求

---

## 2. Gate 1.0 测试套件范围

### 2.1 能力清单

Gate 1.0 测试套件覆盖 `gate_version="1.0"` 的代表性能力（真实硬件验证套件），对应 `CGC_Gate_1.0_edge_cloud_autonomy_gate_map.json` 的完整能力矩阵：

**Self-Harness 真实硬件验证能力（3 项）**：

| 能力 ID | 能力名称 | 验证方法 | 代码位置 |
|---------|----------|----------|----------|
| `edge_cloud_autonomy` | 端云自治入口 | 真实 sglang + llama.cpp 端云协同推理 | `cgc_engine/tools/scripts/run/self_harness_validation_framework.py` → `EdgeCloudInferenceValidator` |
| `cq4_protocol` | CQ4 协议承载 | 真实 CQ4 协议端云传输验证 | `cgc_engine/pd/pd_client.py` |
| `real_edge_cloud_inference` | 真实端云推理 | sglang + llama.cpp 端云推理闭环 | `cgc_engine/tools/scripts/run/self_harness_validation_framework.py` |

**完整能力矩阵（gate_test_framework.py，8 项）**：

| 能力 ID | 能力名称 | 验证参数 | 代码位置 |
|---------|----------|----------|----------|
| `dopd_handoff` | DOPD Handoff 控制面 | `--dopd --handoff-prepare --handoff-commit --handoff-resume` | `cgc_engine/pd/dopd_schema.py` + `cgc_engine/pd/pd_client.py` |
| `cq4_transport` | CQ4 传输协议承载层 | `--cq4 --state-transport --protocol-contract` | `cgc_engine/pd/pd_client.py` |
| `trueorthokda` | TrueOrthoKDA KV + CQ4 压缩 | `--trueorthokda --kv-compression --compression-ratio=high --portable-state` | `cgc_engine/cgc/true_ortho_kda.py` |
| `zero_copy` | Zero-Copy VRAM 零拷贝显存 | `--zero-copy --uma-buffer --device-resume --cpu-copy-count=0` | `cgc_engine/cgc/true_ortho_kda.py` |
| `prefill_producer` | Prefill Producer + Auto-Publish | `--prefill-producer --auto-publish --streaming-path --non-streaming-path` | `app/servers/cgc_api_server.py` |
| `task_type_contract` | Task Type Contract | `--task-type-contract --profile-bundle-validator --bundle-review --fail-fast-governance` | `app/shared/profile_bundle_validator.py` |
| `ray_dual_host` | Ray 双主机拓扑 | `--ray --dual-host --distributed-runtime` | `app/servers/cgc_api_server.py` |
| `moe_route_consistency` | MoE 路由一致性 | `--moe --route-consistency --expert-assignment` | `Backend/CGC/compiler/unified_compiler.py` |

### 2.2 与 Gate 2.0 的边界

Gate 1.0 测试套件**不覆盖**以下内容（属于 Gate 2.0 测试套件）：

- `max_local_layer` 层粒度动态切分
- `finished_layer + 1` 云侧按层接续 Prefill
- `hidden_states + partial_kv` 中间态 ABI（DOPDResumePayloadV2 扩展字段）
- 层流式 KV 同步至 Decode 集群
- UnifiedIRInjector 整图注入 SGLang compute 计算图（Attention + TopK + FusedMoE 三注入点）
- 投机解码 / DeepEP MoE / KV cache 优化 / RSWA Prefill Pool

---

## 3. 三阶段闭环流程

### 3.1 阶段定义

| 阶段 | 名称 | CLI 命令 | Gate 1.0 职责 |
|------|------|----------|--------------|
| Stage 1 | 策略决策 | `cgc model verify` | 验证端云自治能力是否符合 Gate 1.0 要求 |
| Stage 2 | 图捕获 | `cgc model audit` | 审计 DOPD handoff / CQ4 / TrueOrthoKDA 配置正确性 |
| Stage 3 | 执行校验 | `cgc model deploy` | 端云协同真实推理执行（sglang + llama.cpp） |

### 3.2 执行模式

- **direct_cli**：直接调用 CGC CLI 命令
- **cgc_agent**：通过 CGC Agent 代理执行（`cgc agent model verify ...`）
- **self_harness_three_stage**：三阶段闭环模式（默认）

---

## 4. CLI 调用示例

### 4.1 运行 Gate 1.0 harness test

```bash
# 通过 self_harness_validation_framework.py
python3 cgc_engine/tools/scripts/run/self_harness_validation_framework.py --gate 1.0

# 通过 gate_test_framework.py
python3 cgc_engine/tools/scripts/run/gate_test_framework.py \
    --self-harness \
    --gate CGC_Gate_1.0_edge_cloud_autonomy

# 验证特定能力
cgc model verify --gate 1.0 --dopd --cq4 --trueorthokda
```

### 4.2 验证 UPK / UPKG 基础能力

```bash
# 验证 UPK 基础能力
cgc model verify --upk

# 验证 UPKG 1.0 里程碑
cgc model verify --upkg --upkg_version 1.0

# 验证核心能力
cgc model verify --bootstrap --state_abi --system_profile
```

---

## 5. 性能指标

| 指标 | 预期值 |
|------|--------|
| 单能力测试 | < 30 秒 |
| Gate 1.0 self-harness 真实硬件验证（3 项） | < 3 分钟 |
| Gate 1.0 完整能力矩阵（8 项） | < 5 分钟 |
| 真实端云推理（sglang + llama.cpp） | < 60 秒/请求 |

---

## 6. 成功率标准

| 状态 | 定义 |
|------|------|
| PASS | CLI 命令返回码为 0 或 -2 |
| FAIL | CLI 命令返回码非零且非 -2 |
| ERROR | 测试执行异常 |
| SKIP | 能力被跳过（如缺少硬件） |

---

## 7. 主机同步

Gate 1.0 测试相关文件需同步至 host1 (39.106.118.206) 与 host2 (47.95.250.55)：

- `cgc_engine/tools/scripts/run/self_harness_validation_framework.py`
- `cgc_engine/tools/scripts/run/gate_test_framework.py`
- `docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy/CGC_Gate_1.0_Harness_Test_Whitepaper_v1.0_zh_CN.md`

```bash
rsync -av cgc_engine/tools/scripts/run/self_harness_validation_framework.py root@39.106.118.206:/path/to/dest/
rsync -av docs/technical_whitepapers/CGC_Gate_1.0_edge_cloud_autonomy/CGC_Gate_1.0_Harness_Test_Whitepaper_v1.0_zh_CN.md root@39.106.118.206:/path/to/docs/
```

---

## 8. 报告格式

```json
{
  "gate_id": "CGC_Gate_1.0_edge_cloud_autonomy",
  "gate_version": "1.0",
  "execution_mode": "self_harness_three_stage",
  "self_harness_real_hardware_caps": 3,
  "full_capability_matrix": 8,
  "passed": 8,
  "failed": 0,
  "skipped": 0,
  "errors": 0,
  "capabilities": [
    {
      "capability_id": "edge_cloud_autonomy",
      "name": "端云自治入口",
      "status": "PASS",
      "evidence": ["..."]
    }
  ]
}
```

---

**文档版本**: v1.0
**创建日期**: 2026-07-05
**状态**: 正式发布
**拆分自**: `CGC_SelfHarness_Test_Framework_Technical_Whitepaper_v1.0_zh_CN.md`
**适用版本**: CGC Engine v2.1+
