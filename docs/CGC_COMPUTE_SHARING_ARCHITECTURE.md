# CGC 算力共享架构方案

> 基于已有 PD 分离基础设施（coordinator + discovery + protocol + DOPD + MoT-h）构建
>
> 版本: v1.0 | 日期: 2026-08-29

---

## 1. 现有基础

### 1.1 已实现的 PD 分离能力

| 组件 | 文件 | 功能 |
|---|---|---|
| PD Coordinator | `pd/coordinator.py` | FastAPI 编排 emit→MoT-h→resume |
| PD Protocol | `pd/protocol.py` | Wire protocol + PDMode enum |
| PD Discovery | `pd/discovery.py` | etcd 节点注册/发现/心跳 |
| DOPD Runtime | `pd/dopd_runtime.py` | Session 追踪 + handoff |
| Channel Mapping | `pd/channel_mapping.py` | 异构模型层映射 (Gemma4→Qwen3.6) |
| MoT-h | `CGC_Phase2/mot_h/` | 跨模型 hidden state 翻译 |
| Expert Streaming | `cpp/pd_expert_streamer.py` | MoE expert 权重传输 |

### 1.2 已支持的 4 种 PD 模式

| 模式 | Prefill 节点 | Decode 节点 | 适用场景 |
|---|---|---|---|
| **端云** | 云侧 (Mac/Server) | 端侧 (Windows/手机) | 长 prompt 加速 |
| **端端** | Mac A | Mac B | 双机协作 |
| **纯云** | Cloud GPU | Cloud GPU | 大模型全量推理 |
| **纯端** | 本地 GPU | 本地 CPU | 离线/隐私场景 |

---

## 2. 算力共享新增架构

### 2.1 核心思路

在已有 PD 分离基础上，新增三层：

```
┌──────────────────────────────────────────────────┐
│  Layer 3: Adaptive Scheduler (自适应调度)         │
│    实时感知设备状态 → 动态选择 PD 模式 + 节点对    │
├──────────────────────────────────────────────────┤
│  Layer 2: Resource Registry (资源注册)            │
│    扩展 PDNode → 内存/GPU/算力/网络延迟画像       │
├──────────────────────────────────────────────────┤
│  Layer 1: PD Infrastructure (已有)                │
│    coordinator + discovery + protocol + DOPD      │
└──────────────────────────────────────────────────┘
```

### 2.2 Resource Registry

扩展现有 `PDNode`（discovery.py），增加设备画像：

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
    tok_per_sec: {               # 各模型实测速度
        "qwen36_35b": 1.4,
        "gemma4_26b": 25.0,
    }

    # 网络
    network_latency_ms: float    # 到其他节点的延迟
    bandwidth_mbps: float        # 带宽

    # 角色
    role: str                    # "prefill" / "decode" / "both"
    max_concurrent: int          # 最大并发数
```

### 2.3 Dynamic Router

根据设备画像 + 实时负载，选择最优调度：

```
请求到达 → Router
  ├─ 查询 Resource Registry (所有在线节点)
  ├─ 计算每个节点对的 cost = f(latency, memory, compute, load)
  ├─ 选择 cost 最低的 PD 模式 + 节点对
  └─ 转发到 PD Coordinator 执行
```

**调度策略矩阵：**

| 请求特征 | 优先模式 | 节点选择逻辑 |
|---|---|---|
| 短 prompt (<512 tok) | 纯端 | 本地能跑就本地 |
| 长 prompt (>2048 tok) | 端云 | 云侧 prefill，端侧 decode |
| 跨模型 (Gemma→Qwen) | 端端 | MoT-h 翻译 + resume |
| 批量推理 | 纯云 | 云侧 GPU 全量处理 |
| 隐私敏感 | 纯端 | 不出本地网络 |
| 高吞吐 | 端云 + expert streaming | 云侧 prefill + expert 分发 |

---

## 3. 三平台部署拓扑

### 3.1 设备角色分配

```
┌─────────────────────────────────────────────────┐
│               三平台算力池                       │
├─────────────────┬───────────────┬───────────────┤
│  Mac M4 Max     │  Windows 8GB  │  鸿蒙 PC 16GB │
│  24/32GB RAM    │  MX250 2GB    │  麒麟9030     │
│  Metal GPU      │  CPU+GPU      │  CPU only     │
├─────────────────┼───────────────┼───────────────┤
│  Role: 主 Prefill│  Role: Decode │  Role: Decode │
│  + 主 Decode    │  + 小任务      │  + expert      │
│  Gemma4 26B     │  Qwen3.6 35B  │  Qwen3.6 35B  │
│  25-29 t/s      │  1.4-3.7 t/s  │  3-5 t/s      │
└─────────────────┴───────────────┴───────────────┘
```

### 3.2 调度流程示例

**示例 1：长 prompt → 端云模式**
```
1. 用户在 Windows 输入 4096 token prompt
2. Router 检测: Windows prefill 需 30s+, Mac M4 只需 2-3s
3. 选择: 端云模式，Mac M4 做 prefill
4. Mac M4: emit(prompt) → hidden_state [seq, 2816]
5. MoT-h: 翻译 (2816 → 2048)
6. Windows: resume(hidden_state) → SSE decode stream
7. 总延迟: ~3s prefill + ~2s/100tok decode
```

**示例 2：跨模型推理 → 端端模式**
```
1. 请求需要 Gemma4 的多模态理解 + Qwen3.6 的文本生成
2. Mac M4 (Gemma4): emit → hidden_state
3. MoT-h: 翻译 Gemma4→Qwen3.6
4. Windows/鸿蒙 (Qwen3.6): resume → decode
```

**示例 3：高吞吐批量 → 纯云 + expert streaming**
``1. 批量 100 个请求
2. Mac M4: 全部 prefill (GPU 并行)
3. Expert streaming: 分发 MoE expert 权重到 Windows/鸿蒙
4. 各节点并行 decode
```

---

## 4. 实现路径

### Phase 1: Resource Registry (1-2 天)

```python
# 扩展 PDNode (discovery.py)
class PDNode:
    ...
    profile: Optional[DeviceProfile] = None

# 节点启动时自动上报
POST /v1/cgc/register
{
    "node_id": "mac-m4-max",
    "profile": {
        "total_ram_gb": 32,
        "gpu_type": "M4-Max",
        "compute_score": 85,
        "tok_per_sec": {"qwen36_35b": 22.0}
    }
}
```

### Phase 2: Dynamic Router (2-3 天)

```python
# 新增 router.py
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

---

## 5. 与现有代码的整合点

| 新增功能 | 整合位置 | 改动量 |
|---|---|---|
| DeviceProfile | `pd/discovery.py` PDNode 扩展 | 小 |
| ComputeRouter | 新增 `pd/router.py` | 中 |
| 资源上报 API | `pd/coordinator.py` 新增 endpoint | 小 |
| 自适应调度 | `pd/coordinator.py` generate 逻辑 | 中 |
| 节点监控 | `pd/discovery.py` heartbeat 扩展 | 小 |

---

## 6. 性能预估

| 场景 | 单机 | 三机共享 | 加速比 |
|---|---|---|---|
| 4096 tok prefill | 30s (Windows) | 2-3s (Mac M4) | **10-15x** |
| 100 tok decode | 70s (Windows) | 4-5s (Mac M4) | **14-17x** |
| 批量 10 请求 | 串行 300s | 并行 30s | **10x** |
| 跨模型推理 | 不可能 | Gemma4→Qwen3.6 | **∞** (新增能力) |

---

> 基于 CGC-main/cgc_engine/pd/ 已有基础设施扩展，不破坏现有 4 种 PD 模式。
