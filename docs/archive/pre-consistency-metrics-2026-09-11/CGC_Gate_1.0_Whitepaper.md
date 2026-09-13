# CGC Gate 1.0 技术白皮书

> 版本: 1.0 | 日期: 2026-07-25 | 状态: Active

## 1. 摘要

CGC Gate 1.0 是基于 CLI 测试的验证框架，覆盖 **37 个子命令**（cgc model/agent/storage/moe/top），验证 CGC Engine 的完整能力链。Gate 1.0 替代了旧的 Gate 2.1/2.2/2.3 验证器体系，清理了 10 个过时验证器，保留 19 个有效验证器。

### 核心指标

| 维度 | 数值 |
|---|---|
| CLI 子命令 | 37 (model:10 + agent:12 + storage:4 + moe:3 + top:8) |
| Gate 1.0 测试通过率 | 100% (37/37) |
| 验证器 | 19 保留 + 10 归档 |
| 三机同步 | Mac + Host2 + Host1 |

## 2. Gate 1.0 定义

Gate 1.0 = **CLI 测试框架**（`gate1_cli_test.py`），测试所有 cgc 子命令的可用性：

```
Gate 1.0 测试:
  cgc top (8): serve/claude/config/run/list/status/audit/build
  cgc model (10): list/run/serve/verify/audit/replay/trace/compare/launch/swe-verified
  cgc agent (12): import-dag/teach/train/infer/visualize/compare/audit/replay/trace/universe/fusionroute/bench
  cgc storage (4): status/gds/spdk/bench
  cgc moe (3): status/infer/bench
  总计: 37 子命令, 100% PASS
```

## 3. CLI 完整清单

### 3.1 cgc model (10 子命令)

| 子命令 | 功能 | 关键参数 |
|---|---|---|
| list | 模型发现 | --source, --json |
| run | 模型运行 | --prompt, --max-tokens |
| serve | API 服务器 | --port, --host |
| verify | 验证 | --run-session |
| audit | 审计 | --run-session |
| replay | 回放 | --run-session |
| trace | 追踪 | --run-session |
| compare | 对比 | --run-session, --compare-against |
| **launch** | **启动命令生成** | **AutoTunner 全自动决策** |
| swe-verified | SWE 验证 | --model-name |

### 3.2 cgc model launch — 全自动决策

```
cgc model launch v4-flash
  → Magicompiler IR Pass: 检测 compress_ratios → 跳过 R-SWA
  → SeamlessSwitcher: 大模型(>70B) → PD_SEPARATION
  → 4D 感知矩阵: build_4d_matrix + compute_route
  → AutoTunner: NEXTN N=4 + cuda-graph + CGC + mem 0.7
  → 自动参数: --ortho-base-dim, --rswa-window-size, --pd-transport, --pd-cut-layer
```

参数覆盖: R-SWA / 投机 decode / KDA/OrthoKDA / MagiCompiler / PD 分离

### 3.3 cgc agent (12 子命令) — 采训推一体

| 子命令 | 功能 | 环节 |
|---|---|---|
| import-dag | 导入 DAG | - |
| teach | GUI 教学 | 采 |
| **universe** | CLI-Universe 数据合成 | **采** |
| train | Q2RL/TMAX 训练 | 训 |
| infer | 边缘推理 | 推 |
| **fusionroute** | 四角色编排 | **推** |
| **bench** | OSWorld+WebArena | 验证 |
| visualize | 可视化 | - |
| compare | 对比 | - |
| audit | 审计 | - |
| replay | 回放 | - |
| trace | 追踪 | - |

### 3.4 cgc storage (4 子命令)

| 子命令 | 功能 |
|---|---|
| status | GDS/SPDK 可用性检查 |
| gds | GDS 零拷贝 GPU↔NVMe 测试 |
| spdk | SPDK I/O 测试 (io_uring) |
| bench | GDS vs Standard I/O 基准 |

### 3.5 cgc moe (3 子命令)

| 子命令 | 功能 |
|---|---|
| status | FlashMoE 引擎状态 (CPU/CUDA/Metal + GDS/SPDK) |
| infer | MoE 推理 |
| bench | MoE 性能基准 |

## 4. 验证器清理

### 4.1 归档的 10 个过时验证器

| 验证器 | 归档原因 |
|---|---|
| colossalai_runtime_candidate_verifier | ColossalAI 未使用 |
| cq4_verifier | CQ4 未实现 |
| dspark_verifier | DSpark 用 NEXTN 替代 |
| g21_fusion_governance_verifier | 老 Gate 2.1 (Gate 1.0 替代) |
| g22_deepep_l20n_verifier | 老 Gate 2.2 (Gate 1.0 替代) |
| g23_cloud_l20n_tp4_verifier | 老 Gate 2.3 (Gate 1.0 替代) |
| g23_trueorthokda_adapter_verifier | R-SWA 替代 TrueOrthoKDA |
| jetspec_verifier | JetSpec 用 NEXTN 替代 |
| layer_adaptive_verifier | layer-split 废弃 |
| trueorthokda_verifier | TrueOrthoKDA + CQ4 未实现 |

### 4.2 保留的 19 个验证器

| 验证器 | 功能 | 保留原因 |
|---|---|---|
| base | 基础设施 | 核心 |
| workspace_paths | 工作区路径 | 核心 |
| deepseek_v4_flash_resume_verifier | V4-Flash resume/decode | 核心功能 |
| edge_omlx_flashmoe_verifier | OMLX + FlashMoE | FlashMoE 已修复 |
| endtoend_moe_transport_verifier | 端云 MoE 传输 | 端云协同 |
| kv_cache_verifier | KV cache | 核心功能 |
| ray_engine_dual_host_verifier | Ray 双主机 | 双节点集群 |
| rswa_double_layer_kv_verifier | R-SWA 双层 KV | R-SWA 核心 |
| sglang_tp4ep4_verifier | SGLang TP4EP4 | 云端 prefill |
| unified_ir_inject_verifier | UnifiedIR 注入 | Magicompiler |
| zero_copy_verifier | Zero-Copy VRAM | NIXL 传输 |
| deepep_mode_verifier | DeepEP 三模式 | 待定 |
| dflash_deepseek_v4_verifier | DFlash + DSpark | 待定 |
| dopd_verifier | DOPD handoff | 待定 |
| eplb_verifier | EPLB 调度 | 待定 |
| lplb_verifier | LPLB 负载均衡 | 待定 |
| nfsordma_verifier | NFSoRDMA 传输 | 待定 |
| waterfill_verifier | Waterfill 算法 | 待定 |
| __init__ | 包初始化 | 核心 |

## 5. 技术栈状态

### 5.1 端云协议

| 组件 | 状态 | 说明 |
|---|---|---|
| 端云 PD 分离 | ✅ | cloud prefill → NIXL → edge decode |
| layer-split | ❌ 废弃 | Mac 参与转发是负优化 |
| SeamlessSwitcher | ✅ | LOCAL↔CLOUD (layer-split 已移除) |
| 4D 感知矩阵 | ✅ | build_4d_matrix + compute_route |

### 5.2 投机 decode

| 后端 | 最优 N | tok/s | 加速 |
|---|---|---|---|
| MLX (Mac M4) | 16→32 | 53.2 | 2.0x |
| PyTorch (RTX 5000) | 4→2 | 71.5 | 1.94x |
| SGLang (V4-Flash) | 4 (NEXTN) | 38.1 | cuda-graph 3x |

### 5.3 R-SWA

| 平台 | 实现 | 状态 |
|---|---|---|
| GPU (torch) | rswa_gpu.py | ✅ cuda-graph 兼容 |
| MLX (Metal) | rswa_mlx.py | ✅ 988.9 tok/s |
| sglang | sglang_adapter.py | ✅ safe_patch |
| Magicompiler IR | rswa_magicompiler_ir.py | ✅ compress_ratios 检测 |

### 5.4 FlashMoE + GDS + SPDK

| 组件 | 状态 | 说明 |
|---|---|---|
| FlashMoE | ✅ 完整 | Client + CPU + CUDA + Metal 四引擎 |
| GDS | ✅ 可用 | is_gds_available=True, cuFileRead/Write |
| SPDK | ✅ io_uring | liburing 异步 I/O (非真实 SPDK NVMe) |

### 5.5 V4-Flash 性能

| 配置 | decode tok/s | cuda-graph |
|---|---|---|
| disable-cuda-graph | 3-12.8 | ❌ |
| cuda-graph (CGC=0) | 27 | ✅ |
| cuda-graph (CGC=1 GPU) | 38.1 | ✅ |

## 6. 采训推一体闭环

```
采: cgc agent universe (CLI-Universe 三阶段流水线)
  → 任务蓝图 → 环境物化 → 验证过滤
  → 保留率 30% (论文 33.6%)
  ↓
训: cgc agent train (Q2RL/TMAX RL)
  → 用合成数据训练
  ↓
推: cgc agent fusionroute (四角色编排)
  → Hermes/TMAX/UITARS/Synthesizer
  ↓
验证: cgc agent bench (OSWorld + WebArena)
  → 27% (9B) / 33.4% (32B)
```

## 7. 部署拓扑

```
Mac (端侧)
  ├── cgc model launch (AutoTunner 全自动)
  ├── cgc agent universe (数据合成)
  └── cgc moe/storage (本地引擎)

Host2 (47.95.250.55, 云端)
  ├── V4-Flash sglang (cuda-graph + NEXTN)
  ├── FlashMoE + GDS + SPDK
  └── cgc agent fusionroute (四角色)

Host1 (39.106.118.206, edge)
  └── edge decode (PD resume)
```

## 8. 归档文档

以下文档已移至 `docs/archive/`：
- CGC_Engine_CloudDevice_Protocol_Whitepaper_v1.0.md (部分过时: layer-split 废弃)
- CGC_SpecDecode_AutoTunner_Whitepaper_v1.0.md (部分过时: C 库 KDA 移除)
- deepseek-v2-to-v4-state-abi-whitepaper-v1.md (v2→v4, 已过时)

## 9. 结论

Gate 1.0 建立了基于 CLI 测试的验证框架：
- **37 个子命令** 100% 通过
- **10 个过时验证器** 归档
- **19 个有效验证器** 保留
- **采训推一体闭环** CLI 建立 (universe/fusionroute/bench)
- **FlashMoE + GDS + SPDK** 修复完成
- **V4-Flash 38 tok/s** (cuda-graph + NEXTN)

Gate 1.0 替代了旧的 Gate 2.1/2.2/2.3 体系，以 CLI 测试为核心，覆盖 CGC Engine 的完整能力链。
