# CGC Engine Cloud-Device Protocol 技术白皮书

> **版本**: v1.0
> **日期**: 2026-07-25
> **作者**: CGC Team
> **适用模型**: >70B (V4-Flash 等), <70B (Qwen3-VL-2B 等), <13B (Mac 本地)
> **适用拓扑**: Mac (端) → Host2 (云 prefill) → Host1 (云 decode/gateway)

---

## 1. 摘要

CGC (Compute Graph Compiler) Engine Cloud-Device Protocol 是端云协同推理的核心协议，定义了云端 prefill → hidden+KV 传输 → 端侧 decode resume 的完整链路。协议根据模型大小自动路由：**>70B 纯云 PD 分离**、**<70B layer-split 或端侧**、**<13B Mac 完整本地**。通过 TrueOrthoKDA 256x 压缩、NIXL VRAM→VRAM 零拷贝传输、SeamlessSwitcher 运行时切换、AutoTunner 自适应调优，实现低延迟、低成本的端云协同推理。

### 核心指标

| 模型大小 | 路由策略 | TTFT | decode | 协议 |
|---|---|:---:|:---:|---|
| >70B (V4-Flash) | 全云 PD 分离 | 10-31ms (cloud prefill) | 27-37 tok/s | emit+resume+NIXL |
| 13-70B | layer-split / 全云 | 可变 | 可变 | emit+resume (cut=P) |
| <13B (Qwen3-VL-2B) | Mac MLX 本地 | 592ms | 26 tok/s | 无 (全本地) |
| 2B (投机 decode) | chain/eagle | 54ms (连接池) | 53-71 tok/s | spec_decode_ir |

---

## 2. 端云协议架构

```
┌─────────────────────────────────────────────────────────┐
│                    CGC Engine Protocol                    │
├──────────────┬──────────────┬───────────────────────────┤
│  控制协议     │  传输协议     │  路由协议                  │
│ (control)    │ (transport)  │ (route)                   │
├──────────────┼──────────────┼───────────────────────────┤
│ cgc_control  │ cgc_handoff  │ route_decision            │
│ _protocol.py │ _transport.py│ model_dispatcher          │
│              │              │ seamless_switcher         │
│ · TrueOrtho  │ · file       │ · 4D 感知矩阵              │
│   KDA 256x   │ · tcp        │ · >70B → 全云 PD           │
│ · RSWA       │ · nixl       │ · <70B → layer-split       │
│ · PrefillPool│              │ · <13B → Mac 本地           │
│ · GDS/CQ4    │              │ · AutoTunner 自适应         │
├──────────────┴──────────────┴───────────────────────────┤
│                    十步流水线 (cgc.py)                     │
│  Step 1-10: 硬件检测+路由+分发                             │
│  Step 11:   SeamlessSwitcher (运行时切换)                  │
│  Step 11.5: AutoTunner (切换时自动调优)                    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 模型路由策略

### 3.1 路由决策矩阵 (4D 感知)

| 维度 | 参数 | 来源 |
|---|---|---|
| D1 网络 | RTT, 带宽 | hardware_sensing |
| D2 硬件 | 算力, 内存 | hardware_sensing |
| D3 模型 | 大小, MoE/Dense | model_loader |
| D4 路由 | PD/layer-split/cloud/local | route_decision |

### 3.2 按模型大小路由

| 模型大小 | 路由模式 | 端侧角色 | 云端角色 | 协议 |
|---|---|---|---|---|
| **>70B** (V4-Flash) | 全云 PD 分离 | 无 (只路由) | prefill + emit | hidden+KV 传输 |
| **13-70B** | layer-split | prefill P 层 | decode 剩余 | cut=P emit+resume |
| **<13B** | Mac MLX 本地 | 完整推理 | 无 (降级 cover) | 无 (全本地) |

### 3.3 负载分层决策 (2026-07-25 最终确认)

```
<13B: Mac 完整本地 (MLX, 隐私/离线/低成本)
  ↓ 爆显存 → 云 cover 降级
13-70B: Mac 本地 (MLX, 算力够) or 全云 (时延高)
  ↓ Mac 算力不够 → 全云
>70B (V4-Flash): 纯云 PD (cloud prefill → NIXL → edge decode)
  ↓ 消费者设备装不下 → 只能服务器端
```

**layer-split 废弃决策 (2026-07-25)**: Mac MLX prefill 比 cloud prefill 慢 10-100x (100-2000ms vs 10-31ms)。layer-split decode (P<L) 受 55ms RTT 限制 (~7-18 tok/s)。Mac 要么完全不参与 (全云), 要么完全本地 (MLX)。

---

## 4. PD 分离协议 (>70B 全云)

### 4.1 协议流程

```
Cloud (Host2):                    Edge (Host1):
  ┌─────────────┐                   ┌──────────────┐
  │ 1. Prefill  │                   │              │
  │    (V4-Flash)│                   │  4. Receive  │
  │    input_ids │                   │    hidden+KV │
  │       ↓     │    3. Emit         │    (NIXL/TCP)│
  │ 2. hidden   │ ─────────────────→│       ↓      │
  │    +KV cache│   TrueOrthoKDA    │  5. Resume   │
  │    (43层)   │   256x 压缩        │    Decode    │
  │             │                   │    (~37 tok/s)│
  └─────────────┘                   └──────────────┘
```

### 4.2 Emit 机制 (cloud → edge)

**文件**: `CGC_Phase2/cgc_pd_patch.py`, `CGC_Phase2/deepseek_v4.py`

```python
# V4-Flash emit 机制: layer_kv_callback 旁路 capture
# 每次 forward (EXTEND+DECODE) 都 emit, edge 对应 recv+resume
# emit 是旁路, 不改变 forward 路径

# M1v1 退化基线 (cut=42):
#   Host2(prefill, cut=42) emit hidden_states → Host1(decode, 0层) → byte-match ✓

# M1v2 zero-copy TCP:
#   cgc_handoff_transport.py (file/tcp/nixl 三后端)
#   TCP 经 SSH 隧道跨机验证 18/18 byte-identical ✓

# NIXL 真传输:
#   cloud emit cut=21 hidden → NIXL VRAM→VRAM → edge resume → byte-match ALL_MATCH ✓
```

### 4.3 关键修复

| 问题 | 修复 |
|---|---|
| emit 用 module 级 `_CGC_EMIT_DONE` 仅捕 prefill | ForwardMode: EXTEND=1/DECODE=2 区分 |
| resume 输出按 input token 数切片 | 兼容 SGLang EXTEND+DECODE 双 forward |
| TCP `conn.shutdown(SHUT_WR)` 截包 | 去掉, 改 send 后 `recv(1)` 等客户端关 |
| NIXL `NIXL_ERR_NOT_FOUND` | metadata-refresh: recv 重取 `fetch_remote_metadata` |
| cuda-graph 崩溃 | CGC_ENABLE_ORTHO_KDA=0 关闭 instrumentation |

---

## 5. 传输协议 (cgc_handoff_transport.py)

### 5.1 三后端传输

| 后端 | 场景 | 延迟 | 带宽 |
|---|---|:---:|:---:|
| **file** | 本地调试 / 跨机 scp | 高 (磁盘 I/O) | 低 |
| **tcp** | SSH 隧道跨机 | 中 (RTT ~55ms) | 中 |
| **nixl** | VRAM→VRAM 零拷贝 | 低 (~1ms) | 高 (IB/RDMA) |

### 5.2 数据格式

```python
# 传输帧格式 (TCP):
# [4B magic] [4B version] [8B payload_len] [4B seq_len] [8B expected_seq]
# [payload: torch.save(hidden_states) + torch.save(kv_cache)]

# NIXL:
# metadata (shape/dtype/device) → transfer(handle) → recv → resume
```

### 5.3 KV cache 序列化

```python
# kv_layers: list[dict[str, Tensor]] (每层 {key: Tensor, value: Tensor})
# torch.save/load 无损序列化
# seq-aware store: _TcpStore PUT frame + seq_len + expected_seq (防竞争)
```

---

## 6. 控制协议 (rswaengine GPU 版本)

### 6.1 实际状态 (2026-07-25 修正)

> ⚠️ **重要修正**: 之前白皮书描述的 "TrueOrthoKDA 256x 压缩 + C 库 KDA replace mode + CGC env vars" 是设计目标, 非实际状态。经审计, C 库注入从未真正生效。已移除 C 库依赖, 改用 **R-SWA GPU 版本 (纯 torch, cuda-graph 兼容)**。

### 6.2 R-SWA GPU 版本 (rswaengine/python/rswa_gpu.py)

| 组件 | 实现方式 | cuda-graph | 生效 |
|---|---|:---:|:---:|
| **RSWAOrthoAttentionGPU** | 纯 torch GPU (F.scaled_dot_product_attention + torch.matmul) | ✅ | ✅ |
| **Reference attention** | 标准 attention (全维度, 永久区) | ✅ | ✅ |
| **窗口 OrthoKDA** | 正交基投影降维 K, V 保持原维度 | ✅ | ✅ |
| ~~C 库 KDA replace mode~~ | ~~已移除~~ (只是标志位, sglang 不读取) | — | ❌ |
| ~~CGC env vars~~ | ~~已移除~~ (无代码读取) | — | ❌ |

### 6.3 三个宣称验证 (GPU, 2026-07-25)

| 宣称 | 结果 | 数据 |
|---|:---:|---|
| **无限上下文** | ✅ | visible_count 恒定 = 132 (feed 2000 tokens) |
| **O(n) 计算** | ✅ | RSWA 恒定 ~75μs, 标准 O(n²) 增长 9→25μs (n=64→512) |
| **显存不长大** | ✅ | GPU mem=14.3MB 恒定 (feed 2000 tokens) |

### 6.4 cuda-graph 兼容性

```
cuda-graph 捕获: ✅ 成功
cuda-graph 重放: ✅ 成功
feed + 重放: ✅ 窗口更新生效
cuda-graph 加速: 2.8x (26.6μs vs 73.3μs)
```

### 6.5 sglang_adapter

| 函数 | 功能 | cuda-graph |
|---|---|:---:|
| `patch_sglang_gpu(model, config)` | 替换 attention 为 RSWAOrthoAttentionGPU | ✅ |
| `unpatch_sglang(model)` | 恢复原始 attention | ✅ |
| `safe_patch_sglang(model, config, cuda_graph_active)` | cuda-graph 安全 patch | ✅ |

### 6.6 prefill/decode 性能 (Qwen3-VL-2B + cuda-graph)

| Prompt 长度 | TTFT | prefill tok/s | decode tok/s |
|---|:---:|:---:|:---:|
| 短 (10 tok) | 197ms | 56 | 123.7 |
| 中 (50 tok) | 236ms | 221 | 269.4 |
| 长 (200 tok) | 219ms | 238 | 281.9 |

---

## 7. 投机 decode 协议 (spec_decode_ir.py)

### 7.1 三后端投机

| Backend | 模式 | draft model | N | tok/s | speedup |
|---|---|---|:---:|:---:|:---:|
| MLX (Mac) | chain | 0.5B 4bit | 16 | 53.2 | 2.0x |
| PyTorch (GPU) | chain | 0.5B BF16 | 4 | 71.5 | 1.94x |
| SGLang (Qwen3-VL) | plain | 无 | - | 155 | 1.0x |
| SGLang (V4-Flash) | NEXTN | 内置 MTP | 4 | 29 | 1.07x |

### 7.2 AutoTunner 自适应

```
--auto → AutoTunner.detect(backend) → HardwareProfile
  → apply_model_params(model_path) → 自动设置:
    V4-Flash: CGC_ENABLE_ORTHO_KDA=0 + cuda-graph + NEXTN + mem 0.7
    Qwen3-VL: 默认参数
    Mac MLX: N=16, chain, int4
    GPU PyTorch: N=4, chain, bfloat16
  → auto_bench: chain → EAGLE → 选最优
  → runtime_tune: accept 驱动 N 动态调整
```

### 7.3 V4-Flash 投机路线

| 方案 | 结果 | 根因 |
|---|---|---|
| 外接 EAGLE (0.5B) | ❌ | hidden 896≠4096 |
| sglang n-gram | ❌ | eagle_topk>1 需 flashinfer |
| **原生 NEXTN (内置 MTP)** | **✅** | num_nextn_predict_layers=1, hidden 匹配 |

**战略决策**: 放弃外接 EAGLE, 用原生 MTP/DSpark。公开现成、能搭配 V4-Flash (hidden=4096) 的外部 EAGLE draft 不存在。

---

## 8. SeamlessSwitcher + AutoTunner (十步流水线 Step 11.5)

### 8.1 十步流水线

| Step | 内容 | 模块 |
|:---:|---|---|
| 1-5.5 | 硬件检测 (OS/CPU/模型/内存/算力) | hardware_sensing |
| 6-7 | 引擎路由 (OMLX/CUDA/FlashMoE) | hardware_sensing |
| 7.5 | PD/Layer-split 路由决策 | route_decision |
| 7.6 | 模型分发 | model_dispatcher |
| 7.7 | MTP draft 同步 | model_dispatcher |
| 8-10 | 上下文/4D 矩阵/磁盘 | route_decision |
| 11 | SeamlessSwitcher 初始化 | seamless_switcher |
| **11.5** | **AutoTunner 集成** | **spec_decode_ir** |

### 8.2 切换流程

```
SeamlessSwitcher 后台监控 (内存/网络/decode)
  ↓ 触发切换 (内存<1GB → 本地切云)
  ↓
_on_switch_with_autotune(event)
  ↓ AutoTunner.get_optimal_config(new_backend, model_path)
  ↓ 自动设置目标后端参数 (N, cuda-graph, NEXTN, mem-fraction)
```

### 8.3 切换触发

| 触发 | 动作 | 阈值 |
|---|---|---|
| 内存不足 | 本地→云 | <1GB |
| 内存恢复 | 云→本地 | >3GB |
| 网络超时 | 云→本地 | RTT>500ms |
| 网络恢复 | 本地→云 | RTT<200ms |
| decode 太慢 | 本地→云 | <5 tok/s |

---

## 9. 统一 model_loader

```python
from app.shared.model_loader import load_base_model

# target + draft 都用统一加载, 零硬编码
target, tokenizer = load_base_model(target_path, device, dtype)
draft, _ = load_base_model(draft_path, device, dtype)

# 自动检测:
#   VL 模型 → AutoModelForImageTextToText
#   纯文本 → AutoModelForCausalLM
#   兜底   → AutoModel
```

### 统一接口

| 函数 | 功能 | 支持路径 |
|---|---|---|
| `load_base_model()` | 统一加载 | VL/纯文本/MoE |
| `get_embed_weight()` | embed_tokens | language_model.model.embed_tokens |
| `get_lm_head_weight()` | lm_head | language_model.lm_head |
| `get_text_model()` | text model | language_model |
| `get_layers()` | transformer layers | layers |
| `get_model_info()` | 模型信息 | hidden/layers/vocab/is_vl/is_moe |

---

## 10. 部署拓扑

### 10.1 三机拓扑

```
Mac (端侧, M4 16GB)
  ├── edge_first_proxy.py (首包预测, TTFT < 30ms)
  ├── spec_decode_ir.py (AutoTunner, MLX backend)
  └── cgc.py CLI (十步流水线)
        ↓ SSH 隧道 (port 22)
Host2 (47.95.250.55, 云 prefill)
  ├── sglang server (V4-Flash TP=8, cuda-graph)
  ├── cgc_launch_dual_node.py (CGC wrapper)
  ├── cgc_pd_patch.py (emit hidden+KV)
  └── cgc_handoff_transport.py (NIXL/TCP 传输)
        ↓ NIXL VRAM→VRAM / SSH 隧道
Host1 (39.106.118.206, 云 decode/gateway)
  ├── sglang server (V4-Flash decode)
  ├── cloud_resume_endpoint.py (resume decode)
  ├── cgc_api_server.py (OpenAI 兼容 API)
  └── fusionroute_cloud_orchestrator.py (融合路由)
```

### 10.2 启动命令 (AutoTunner 自动生成)

```bash
# V4-Flash (cuda-graph + NEXTN):
CGC_ENABLE_ORTHO_KDA=0 python3 cgc_launch_dual_node.py \
  --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2 \
  --tp-size 8 --context-length 16384 \
  --mem-fraction-static 0.7 --cuda-graph-max-bs 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 4

# Qwen3-VL (默认):
python3 -m sglang.launch_server \
  --model-path /data2/models/Qwen3-VL-2B-Instruct \
  --tp 1 --mem-fraction-static 0.88
```

---

## 11. 文件清单

### 11.1 核心协议文件

| 文件 | 功能 | 大小 |
|---|---|---|
| `CGC_Phase2/cgc_control_protocol.py` | CGC 控制协议核心 | - |
| `CGC_Phase2/cgc_handoff_transport.py` | 移交传输 (file/tcp/nixl) | - |
| `CGC_Phase2/cgc_pd_patch.py` | PD 分离补丁 (emit) | - |
| `CGC_Phase2/pd_separation_decode.py` | PD 分离解码 | - |
| `CGC_Phase2/layer_split_decode.py` | 层分割解码 | - |
| `CGC_Phase2/cloud_resume_endpoint.py` | 云端恢复端点 (resume) | - |
| `CGC_Phase2/deepseek_v4.py` | DeepSeek V4 支持 | - |
| `CGC_Phase2/qwen3_vl_resume_patch.py` | Qwen3-VL 恢复补丁 | - |
| `app/servers/fusionroute_cloud_orchestrator.py` | 融合路由编排 | - |
| `app/servers/edge_first_proxy.py` | 边缘首包预测 | - |
| `app/servers/cgc_api_server.py` | OpenAI 兼容 API | - |

### 11.2 路由+调优文件

| 文件 | 功能 |
|---|---|
| `app/shared/spec_decode_ir.py` | 统一 IR + AutoTunner (35KB) |
| `app/shared/model_loader.py` | 统一加载器 (8KB) |
| `app/shared/seamless_switcher.py` | 无缝切换器 (25KB) |
| `app/shared/hardware_sensing.py` | 硬件感知 (14KB) |
| `app/shared/route_decision.py` | 路由决策 (12KB) |
| `app/shared/model_dispatcher.py` | 模型分发 (20KB) |
| `app/shared/mtp_trainer.py` | MTP 训练 (18KB) |
| `app/shared/model_registry.py` | 模型注册表 |
| `app/shared/unified_mtp_ir.py` | 统一 MTP IR |
| `app/cli/cgc.py` | 十步流水线 CLI (460KB) |

### 11.3 启动脚本

| 文件 | 功能 |
|---|---|
| `cgc_launch_dual_node.py` | CGC wrapper (sglang 启动) |
| `launch_cgc_pd_emit.sh` | PD emit 启动 |
| `launch_cgc_pd_resume.sh` | PD resume 启动 |
| `launch_edge_resume_native_nixl.sh` | NIXL 恢复启动 |
| `launch_fusion_stack.py` | 融合栈启动 |
| `start_h2.sh` | Host2 sglang 启动 (Host2 专用) |

---

## 12. 验证结果汇总

### 12.1 PD 分离验证

| 阶段 | 结果 |
|---|---|
| M1v1 退化基线 (cut=42) | byte-match ✓ (France→Paris, Germany→in) |
| M1v2 TCP 传输 | 18/18 byte-identical ✓ |
| NIXL VRAM→VRAM | byte-match ALL_MATCH ✓ |
| V4-Flash PD (cloud prefill → edge decode) | ~37 tok/s ✓ |

### 12.2 投机 decode 验证

| Backend | Model | tok/s | speedup |
|---|---|:---:|:---:|
| MLX (Mac) | Qwen3-VL-2B | 53.2 | 2.0x |
| PyTorch (GPU) | Qwen3-VL-2B | 71.5 | 1.94x |
| SGLang (Cloud) | Qwen3-VL-2B | 155 | plain |
| SGLang (Cloud) | V4-Flash + NEXTN | 29 | +7% |

### 12.3 cuda-graph 修复

| 配置 | tok/s |
|---|:---:|
| disable-cuda-graph | 3-12.8 |
| **cuda-graph (CGC_ENABLE_ORTHO_KDA=0)** | **27** |
| cuda-graph + NEXTN | 29 |

---

## 13. 结论

1. **端云协议**: cloud prefill → emit hidden+KV (TrueOrthoKDA 256x) → edge resume decode, byte-match 验证通过
2. **模型路由**: >70B 全云 PD, <70B layer-split/全云, <13B Mac 本地 (4D 感知自动决策)
3. **传输层**: file/tcp/nixl 三后端, NIXL VRAM→VRAM 零拷贝最快
4. **投机 decode**: AutoTunner 自动检测硬件+模型 → 最优参数 (MLX 2.0x, GPU 1.94x, V4-Flash NEXTN)
5. **运行时切换**: SeamlessSwitcher 监控内存/网络/decode → AutoTunner 自动调优目标后端
6. **十步流水线**: Step 1-11.5 完整链路 (硬件检测→路由→分发→切换→调优)
7. **统一加载**: model_loader 零硬编码 (VL/纯文本/MoE 自动检测)

---

*End of Document*
