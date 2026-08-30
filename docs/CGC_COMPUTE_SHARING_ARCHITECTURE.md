# CGC 算力共享架构方案

> 基于已有 PD 分离基础设施（coordinator + discovery + protocol + DOPD + MoT-h）构建
>
> 版本: v1.1 | 日期: 2026-08-30
> 更新: 新增鸿蒙手机节点 (Mate 70 Pro, 12GB RAM)

---

## 1. 现有基础

### 1.1 已实现的 PD 分离能力

| 组件 | 文件 | 功能 |
|---|---|---|
| PD Coordinator | `pd/coordinator.py` | FastAPI 编排 emit→MoT-h→resume |
| PD Protocol | `pd/protocol.py` | Wire protocol + PDMode enum |
| PD Discovery | `pd/discovery.py` | etcd 节点注册/发现/心跳 + DeviceProfile |
| ComputeRouter | `pd/router.py` | 4D 矩阵路由 (network×hardware×model×runtime) |
| DOPD Runtime | `pd/dopd_runtime.py` | Session 追踪 + handoff |
| Channel Mapping | `pd/channel_mapping.py` | 异构模型层映射 (Gemma4→Qwen3.6) |
| MoT-h | `CGC_Phase2/mot_h/` | 跨模型 hidden state 翻译 |
| Expert Streaming | `cpp/pd_expert_streamer.py` | MoE expert 权重传输 |
| Edge Server | `pd/edge_server.py` | 端点 HTTP server (Mac/Win/Phone) |

### 1.2 已支持的 4 种 PD 模式

| 模式 | Prefill 节点 | Decode 节点 | 适用场景 |
|---|---|---|---|
| **端云** | 云侧 (Mac/Server) | 端侧 (Windows/手机) | 长 prompt 加速 |
| **端端** | Mac A | Mac B | 双机协作 |
| **纯云** | Cloud GPU | Cloud GPU | 大模型全量推理 |
| **纯端** | 本地 GPU | 本地 CPU | 离线/隐私场景 |

---

## 2. 四平台算力池

### 2.1 设备矩阵

```
┌─────────────────────────────────────────────────────────┐
│                    四平台算力池                           │
├─────────────┬───────────────┬───────────────┬───────────┤
│  Mac M4     │  Windows 8GB  │  鸿蒙PC 24GB  │ 鸿蒙手机  │
│  32-64GB    │  MX250 2GB    │  麒麟9030     │ 12GB      │
│  Metal GPU  │  CPU+GPU      │  CPU only     │ CPU only  │
├─────────────┼───────────────┼───────────────┼───────────┤
│  主 Prefill │  Decode       │  Decode       │ Decode    │
│  + 主 Decode│  + 小任务      │  + Expert     │ + 轻量    │
│  25-29 t/s  │  1.4-3.7 t/s  │  3-5 t/s      │ 2-4 t/s   │
│  Qwen3.6-35B│  Qwen3.6-35B  │  Qwen3.6-35B  │ Qwen3-14B │
└─────────────┴───────────────┴───────────────┴───────────┘
```

### 2.2 DeviceProfile 扩展

```python
@dataclass
class DeviceProfile:
    """设备能力画像（扩展 PDNode）"""
    # 硬件
    total_ram_gb: float          # 总内存
    available_ram_gb: float      # 可用内存
    gpu_type: str                # "MX250" / "M4-Max" / "none"
    gpu_vram_gb: float           # GPU 显存
    cpu_cores: int               # CPU 核数

    # 推理能力
    compute_score: float         # 综合算力分 (0-100)
    prefill_tok_per_sec: {       # 各模型 prefill 速度
        "qwen36_35b": 120.0,    # Mac M4
        "qwen3_14b": 1.5,       # Mate 70 Pro
    }
    decode_tok_per_sec: {        # 各模型 decode 速度
        "qwen36_35b": 22.0,     # Mac M4
        "qwen3_14b": 3.0,       # Mate 70 Pro
    }

    # 网络
    network_latency_ms: float    # 到其他节点的延迟
    bandwidth_mbps: float        # 带宽

    # 角色
    role: str                    # "prefill" / "decode" / "both"
    max_concurrent: int          # 最大并发数
```

### 2.3 四平台 DeviceProfile 示例

| 平台 | RAM | GPU | compute_score | prefill (qwen36) | decode (qwen36) |
|---|---|---|---|---|---|
| Mac M4 Max | 64GB | Metal | 95 | 120 t/s | 22 t/s |
| Windows 8GB | 8GB | MX250 | 32 | 2 t/s | 1.4 t/s |
| 鸿蒙 PC | 24GB | CPU | 45 | 3 t/s | 4 t/s |
| 鸿蒙手机 | 12GB | CPU | 28 | 1.5 t/s | 3 t/s |

---

## 3. Dynamic Router (4D 矩阵)

### 3.1 路由维度

| 维度 | 内容 | 权重 |
|---|---|---|
| **D1 Network** | RTT, bandwidth | 20% |
| **D2 Hardware** | RAM, GPU VRAM, CPU, compute_score | 40% |
| **D3 Model** | params, layers, MoE, per_layer_gb, MTP | 40% |
| **D4 Runtime** | memory pressure, load | 15% |

### 3.2 路由逻辑

```
请求到达 → ComputeRouter
  ├─ 查询 Resource Registry (所有在线节点)
  ├─ 计算每个节点对的 cost = f(latency, memory, compute, load)
  ├─ 选择 cost 最低的 PD 模式 + 节点对
  └─ 转发到 PD Coordinator 执行
```

### 3.3 调度策略矩阵

| 请求特征 | 优先模式 | 节点选择逻辑 |
|---|---|---|
| 短 prompt (<512 tok) | 纯端 | 本地能跑就本地 |
| 长 prompt (>2048 tok) | 端云 | 云侧 prefill，端侧 decode |
| 跨模型 (Gemma→Qwen) | 端端 | MoT-h 翻译 + resume |
| 批量推理 | 纯云 | 云侧 GPU 全量处理 |
| 隐私敏感 | 纯端 | 不出本地网络 |
| 高吞吐 | 端云 + expert streaming | 云侧 prefill + expert 分发 |
| **手机请求** | **端云** | **Mac prefill → 手机 decode** |

### 3.4 手机端路由示例

```
用户在 Mate 70 Pro 输入 4096 token prompt
  → Router 检测: 手机 12GB < 14B 模型 → 不适合本地 prefill
  → 选择: 端云模式，Mac M4 做 prefill
  → Mac M4: emit(prompt) → hidden_state [seq, 2816]
  → MoT-h: 翻译 (2816 → 2048)
  → Mate 70 Pro: resume(hidden_state) → SSE decode stream
  → 总延迟: ~3s prefill + ~3s/100tok decode
```

---

## 4. 三平台部署拓扑

### 4.1 端云联调流程

```
1. Mac M4 启动 edge_server (prefill 节点)
   python3 edge_server.py --binary llama-simple --model qwen36.gguf --port 1234

2. Windows/MateBook 启动 edge_server (decode 节点)
   python3 edge_server.py --binary llama-server --model qwen36.gguf --port 1234

3. Mate 70 Pro 启动 edge_server (decode 节点)
   python3 edge_server.py --binary llama-server --model qwen3-14b.gguf --port 8080

4. 客户端发起请求
   py -3 pd_e2e_test.py --model qwen36_35b \
       --local http://127.0.0.1:1234 \
       --remote http://192.168.101.87:1234
```

### 4.2 实测结果 (Mac + Windows)

| 指标 | 第1次 | 第2次 | 第3次 | 第4次 |
|---|---|---|---|---|
| decode_tps | 1.71 | 16.64 | 15.99 | **27.02** |
| wall | 29.3s | 7.2s | 31.8s | **36.2s** |
| 说明 | 冷启动 | 热缓存 | 稳定 | **完全预热** |

---

## 5. 实现路径

### Phase 1: Resource Registry (已完成 ✅)

```python
# 扩展 PDNode (discovery.py)
class PDNode:
    ...
    profile: Optional[DeviceProfile] = None

# 节点启动时自动上报
POST /v1/cgc/register
{
    "node_id": "mate-70-pro",
    "profile": {
        "total_ram_gb": 12,
        "gpu_type": "none",
        "compute_score": 28,
        "prefill_tok_per_sec": {"qwen3_14b": 1.5},
        "decode_tok_per_sec": {"qwen3_14b": 3.0}
    }
}
```

### Phase 2: Dynamic Router (已完成 ✅)

```python
# 新增 router.py (4D 矩阵路由)
class ComputeRouter:
    def select_mode(self, request: GenerateRequest) -> PDRoute:
        """根据请求特征 + 设备状态选择最优 PD 模式"""
        candidates = self.registry.get_available_nodes()
        scored = [(node, self.score(node, request)) for node in candidates]
        best = max(scored, key=lambda x: x[1])
        return PDRoute(mode=best[0].preferred_mode, nodes=[best[0]])
```

### Phase 3: Adaptive Scheduler (3-5 天)

- 实时监控各节点 load_factor
- 动态切换模式（端云↔端端↔纯端）
- 故障转移（节点掉线自动切换）

### Phase 4: 鸿蒙手机集成 (进行中)

- [ ] DevEco Studio 5.0+ 安装
- [ ] HDC 连接 Mate 70 Pro
- [ ] 交叉编译 llama.cpp for HarmonyOS NEXT
- [ ] 部署 Qwen3-14B IQ4_XS 到手机
- [ ] 手机端 edge_server 启动
- [ ] 端云联调 (Mac prefill → 手机 decode)

---

## 6. 与现有代码的整合点

| 新增功能 | 整合位置 | 改动量 |
|---|---|---|
| DeviceProfile | `pd/discovery.py` PDNode 扩展 | 小 |
| ComputeRouter | 新增 `pd/router.py` | 中 |
| 资源上报 API | `pd/coordinator.py` 新增 endpoint | 小 |
| 自适应调度 | `pd/coordinator.py` generate 逻辑 | 中 |
| 节点监控 | `pd/discovery.py` heartbeat 扩展 | 小 |
| 手机端 edge_server | `pd/edge_server.py` 轻量化 | 小 |
| 手机端 build | `deploy-harmonyos/phone/build_phone.sh` | 新增 |

---

## 7. 性能预估

| 场景 | 单机 | 四机共享 | 加速比 |
|---|---|---|---|
| 4096 tok prefill | 30s (Windows) | 2-3s (Mac M4) | **10-15x** |
| 100 tok decode | 70s (Windows) | 4-5s (Mac M4) | **14-17x** |
| 批量 10 请求 | 串行 300s | 并行 30s | **10x** |
| 跨模型推理 | 不可能 | Gemma4→Qwen3.6 | **∞** (新增能力) |
| 手机端云 | 2-4 t/s (本地) | 10x prefill 加速 | **∞** (新增能力) |

---

## 8. 附录: 部署检查清单

### Mac M4
- [x] llama-server 编译 (Metal + CGC)
- [x] edge_server.py 启动
- [x] Expert Cache 8GB 配置
- [x] 端云联调通过 (27 t/s)

### Windows
- [x] llama-server.exe 编译 (MinGW/Clang)
- [x] edge_server.py 启动
- [x] 端云联调通过

### 鸿蒙 PC
- [x] llama-simple 编译 (麒麟9030)
- [x] build.sh / run.sh 脚本
- [ ] 端云联调

### 鸿蒙手机
- [ ] DevEco Studio 5.0+ 安装
- [ ] HDC 连接 Mate 70 Pro
- [ ] llama-server 交叉编译
- [ ] Qwen3-14B IQ4_XS 下载
- [ ] 部署到手机
- [ ] 本地推理测试
- [ ] 端云联调

---

> 基于 CGC-main/cgc_engine/pd/ 已有基础设施扩展，不破坏现有 4 种 PD 模式。
> 手机端作为 decode 节点，Mac M4 作为 prefill 节点，实现端云协同。
