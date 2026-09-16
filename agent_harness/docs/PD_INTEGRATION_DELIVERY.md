# 端云 PD + Master/Slave 同步 + 算力共享 整合交付总结

> **日期**: 2026-09-16
> **项目**: Embodibrain — Agent Harness
> **版本**: v1.0
> **状态**: 已完成，56/56 测试通过

---

## 一、整合概述

将端云 PD（Prefill-Decode 分离）代码、Master/Slave 代码同步机制、端端/端云算力共享框架三大组件整合进 `agent_harness/` 目录，实现：

1. **代码共享** — PD 代码、Loop MoE、Qwen36 推理层、Agent 适配器统一在 `agent_harness/` 下
2. **Master/Slave 定时同步** — slave 节点定时从 master 拉取代码，保证多设备代码一致性
3. **端端/端云算力共享** — 分布式算力调度，支持 Mac↔Mac↔Windows↔云服务器的算力借用

---

## 二、整合后目录架构

```
agent_harness/
├── pd/                               ← 端云 PD 分离（35 文件）
│   ├── 核心 PD 模块（从 CGC-main 复制，31 文件）
│   │   ├── coordinator.py            FastAPI coordinator (:9000)
│   │   ├── edge_server.py            Mac 端 prefill 服务器
│   │   ├── pd_server.py              PD 服务端
│   │   ├── pd_client.py              PD 客户端
│   │   ├── router.py                 计算路由器
│   │   ├── discovery.py              设备发现
│   │   ├── dopd_runtime.py           DOPD 运行时
│   │   ├── dopd_schema.py            DOPD 数据结构
│   │   ├── protocol.py               PD 协议定义
│   │   ├── mot_h.py                  MoT-h 隐状态翻译模型
│   │   ├── train_mot_h.py            MoT-h 训练
│   │   ├── channel_mapping.py        通道映射（Gemma4↔Qwen3.6）
│   │   ├── context_replay.py         上下文回放
│   │   ├── kv_async_prefetch.py      KV 异步预取
│   │   ├── kv_quantizer.py           KV 量化
│   │   ├── spdk_kv_cache.py          SPDK KV 缓存
│   │   ├── nccl_sync.py              NCCL 多卡同步
│   │   ├── collect_batch.py          批量数据采集
│   │   ├── collect_parallel_data.py  并行数据采集
│   │   ├── gen_sample_corpus.py      样本语料生成
│   │   ├── turbofieldfare_adapter.py TurboFieldfare 客户端适配器
│   │   ├── verify_pd_emit.py         PD emit 验证
│   │   ├── pd_e2e_test.py            PD 端到端测试
│   │   ├── router_selftest.py        路由器自测
│   │   ├── mock_demo.py              Mock 演示
│   │   ├── pd_service.proto          gRPC 协议定义
│   │   ├── pd_service_pb2.py         gRPC 生成代码
│   │   └── pd_service_pb2_grpc.py    gRPC 生成代码
│   │
│   ├── master_slave_sync.py          ← 【新增】Master/Slave 代码同步（21KB）
│   ├── compute_sharing.py            ← 【新增】端端/端云算力共享（24KB）
│   ├── pd_config.py                  ← 【新增】PD 统一配置（6KB）
│   ├── CGC_PD_Whitepaper.md          PD 白皮书（39.9 KB）
│   ├── DOPD_UPGRADE_PLAN.md          DOPD 升级计划
│   └── tests/
│       └── test_pd_integration.py    ← 【新增】PD 整合测试（7 tests）
│
├── loopmoe/                          ← 【已有】Loop MoE 模型（22 文件）
├── qwen36/                           ← 【已有】Qwen36 推理接入层
├── agents/                           ← 【已有】Agent 适配器
│   └── loopmoe_agent_adapter.py      ← 【已有】Loop MoE Agent 适配器
├── docs/                             ← 设计文档
│   ├── CGC_PD_Whitepaper.md          ← 【新增】PD 白皮书
│   ├── DOPD_UPGRADE_PLAN.md          ← 【新增】DOPD 升级计划
│   └── whittle-moe-whitepaper.md     （已有）
├── finetune/
│   └── finetune_loopmoe.sh           ← 【已有】Loop MoE 微调脚本
└── config.env                        ← 【更新】新增 PD 配置段（40 行）
```

---

## 三、三大核心功能

### 3.1 代码共享（Code Sharing）

PD 代码、Loop MoE、Qwen36 推理层、Agent 适配器全部统一在 `agent_harness/` 目录下，通过 Python 包导入共享：

```python
# 任意位置导入
from agent_harness.pd import ComputeScheduler, CodeSyncer
from agent_harness.loopmoe import LoopMoEModel
from agent_harness.qwen36 import Qwen36Inference
```

### 3.2 Master/Slave 代码同步（slave 定时对齐 master）

**文件**: `pd/master_slave_sync.py`

支持四种同步方式：

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| `git` | slave 执行 `git pull` 从 master 远程拉取 | 有 git 仓库的环境 |
| `incremental` | 通过 HTTP API 对比文件 MD5，只同步变更文件 | 端端/端云跨设备同步 |
| `file` | 直接从 master 路径复制文件 | 共享存储/SMB 环境 |
| `http` | 通过 HTTP API 全量同步 | 简单部署 |

**使用方式**:

```bash
# Master 节点（提供同步服务）
python pd/master_slave_sync.py --role master --port 9000

# Slave 节点（定时同步，每 300 秒一次）
python pd/master_slave_sync.py --role slave --master-url http://192.168.1.101:9000 --interval 300

# 单次同步
python pd/master_slave_sync.py --role slave --master-url http://192.168.1.101:9000 --once
```

**Python API**:

```python
from pd.master_slave_sync import CodeSyncer, SyncConfig, SyncRole, SyncMethod

syncer = CodeSyncer(SyncConfig(
    role=SyncRole.SLAVE,
    method=SyncMethod.INCREMENTAL,
    master_url="http://192.168.1.101:9000",
    sync_interval=300,
    sync_directories=["pd", "loopmoe", "qwen36", "agents"],
))

# 单次同步
result = syncer.sync_once()
print(result.summary())  # [SUCCESS] updated=5 added=2 deleted=0 ...

# 后台定时同步
syncer.start_background_sync()
```

**核心类**:
- `SyncConfig` — 同步配置（角色、方式、间隔、目录、冲突策略）
- `CodeSyncer` — 同步器（扫描、hash 对比、执行同步、后台线程）
- `SyncResult` — 同步结果（更新/新增/删除/跳过/冲突/错误）
- `FileState` — 文件状态（路径、MD5、大小、修改时间）

### 3.3 端端/端云算力共享（Compute Sharing）

**文件**: `pd/compute_sharing.py`

分布式算力调度框架：

| 概念 | 说明 |
|------|------|
| `ComputeNode` | 算力节点（Mac A、Mac B、Windows、云服务器） |
| `ComputeTask` | 算力任务（推理、prefill、decode、训练、数据处理） |
| `ComputeScheduler` | 调度器，根据模型匹配度、GPU 显存、负载自动选最优节点 |
| `SharingPolicy` | 共享策略：exclusive（独占）/ shared（共享）/ priority（优先级） |

**支持的共享模式**:
- **端端共享（Edge-Edge）**: Mac A ↔ Mac B ↔ Windows 互相借用算力
- **端云共享（Edge-Cloud）**: 端侧将训练/大模型推理卸载到云侧
- **混合共享**: 根据任务类型自动选择端侧或云侧

**使用方式**:

```bash
# 启动算力节点（Mac A）
python pd/compute_sharing.py --role node --name mac-a-gemma4 --node-role edge --gpu-count 1 --port 9100

# 启动算力节点（云服务器）
python pd/compute_sharing.py --role node --name cloud-a100 --node-role cloud --gpu-count 4 --port 9100
```

**Python API**:

```python
from pd.compute_sharing import ComputeScheduler, ComputeTask, TaskType, NodeRole

scheduler = ComputeScheduler()

# 注册算力节点
scheduler.register_node("mac-a", "Mac A Gemma4", "http://192.168.1.101:9100",
                        role=NodeRole.EDGE, gpu_count=1)
scheduler.register_node("cloud", "Cloud A100", "http://10.0.0.1:9100",
                        role=NodeRole.CLOUD, gpu_count=4)

# 调度推理任务（自动选最优节点）
task = ComputeTask(task_type=TaskType.INFERENCE, model_id="gemma-4-26b", prompt="hello")
result = scheduler.schedule(task)
print(result.summary())  # Task xxx → Mac A Gemma4 (score=232.2)

# 端端算力共享：Windows 借用 Mac A 的 Gemma4 推理
share = scheduler.share_compute(
    from_node="mac-a", to_node="windows",
    task_type=TaskType.INFERENCE, model_id="gemma-4-26b",
)
# → {"success": True, "share_id": "abc123", "endpoint": "http://.../v1/compute/execute"}

# 查看算力拓扑
topology = scheduler.get_sharing_topology()
# → {"total_nodes": 2, "edge_nodes": [...], "cloud_nodes": [...], ...}
```

**调度评分机制**:
- 模型匹配度（权重最高，+100）
- GPU 显存充裕度（+2/GB）
- GPU 利用率（越低越好）
- 内存充裕度（+0.5/GB）
- CPU 利用率（越低越好）
- 当前负载（-10/任务）
- 角色偏好（云端 +50，端侧 +20）

---

## 四、统一配置（pd_config.py）

**文件**: `pd/pd_config.py`

四个配置类：

| 配置类 | 说明 |
|--------|------|
| `PDServerConfig` | PD 服务端（coordinator 端口、MoT-h 训练、数据目录） |
| `PDEdgeConfig` | PD 边缘节点（prefill 服务器、模型、上报） |
| `MasterSlaveConfig` | Master/Slave 同步（角色、方式、间隔、目录） |
| `ComputeSharingConfig` | 算力共享（节点角色、端口、资源、策略） |

支持从环境变量加载：`PDConfig.from_env()`

---

## 五、config.env 新增配置段

```bash
# PD 服务端
PD_SERVER_PORT=9000
PD_COORDINATOR_URL="http://127.0.0.1:9000"

# Master/Slave 同步
PD_SYNC_ENABLED=0
PD_SYNC_ROLE="slave"
PD_SYNC_METHOD="incremental"
PD_SYNC_MASTER_URL=""
PD_SYNC_INTERVAL=300

# 端端/端云算力共享
PD_COMPUTE_ENABLED=0
PD_COMPUTE_NODE_ROLE="edge"
PD_COMPUTE_PORT=9100
PD_COMPUTE_SCHEDULER_URL=""
PD_COMPUTE_GPU_COUNT=0
PD_COMPUTE_POLICY="shared"
```

---

## 六、测试验证结果

| 测试套件 | 结果 |
|----------|------|
| Loop MoE 原始单元测试 | **42/42 通过** |
| Loop MoE 整合测试 | **7/7 通过** |
| PD 整合测试（核心+同步+算力共享+配置） | **7/7 通过** |
| **合计** | **56/56 通过** |

**PD 整合测试覆盖**:
1. PD 核心模块导入（protocol、HiddenStatePacket 等）
2. PD 统一配置创建和环境变量加载
3. Master/Slave 同步配置创建
4. Master/Slave 文件扫描和 hash（35 个文件）
5. 算力节点创建和任务评分（score=232.2）
6. 算力调度器（2 节点、任务分配、算力共享、拓扑）
7. PD 整合包完整导入（所有模块可用）

---

## 七、典型部署场景

```
场景：Mac A (Gemma4) + Mac B (Qwen3.6) + Windows (decode) 三设备协同

Mac A (192.168.1.101)          Mac B (192.168.1.102)          Windows (192.168.1.8)
┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
│ PD edge_server :8080│        │ PD edge_server :8080│        │ coordinator :9000   │
│ Gemma4 prefill      │        │ Qwen3.6 prefill     │        │ MoT-h 翻译          │
│                     │        │                     │        │ decode              │
│ Master sync :9000   │◄───────│ Slave sync (300s)  │        │ Slave sync (300s)  │
│ Compute node :9100  │◄───────│ Compute node :9100  │◄───────│ Compute node :9100  │
│ 共享: Gemma4 推理   │        │ 共享: Qwen3.6 推理  │        │ 共享: decode 算力   │
└─────────────────────┘        └─────────────────────┘        └─────────────────────┘
         │                              │                              │
         └──────────────┬───────────────┴──────────────┬───────────────┘
                        ▼                               ▼
              ComputeScheduler (任一节点运行)     Master/Slave 代码同步
              - 自动选最优节点执行推理             - Mac A 为 master
              - 端端/端云算力共享                  - Mac B / Windows 为 slave
              - 模型匹配度 + GPU 显存评分           - 每 300s 自动对齐代码
```

---

## 八、文件清单

### 新增文件（本次整合）

| 文件 | 大小 | 说明 |
|------|------|------|
| `pd/master_slave_sync.py` | 21 KB | Master/Slave 代码同步 |
| `pd/compute_sharing.py` | 24 KB | 端端/端云算力共享 |
| `pd/pd_config.py` | 6 KB | PD 统一配置 |
| `pd/tests/test_pd_integration.py` | 7 KB | PD 整合测试 |
| `docs/CGC_PD_Whitepaper.md` | 39.9 KB | PD 白皮书（副本） |
| `docs/DOPD_UPGRADE_PLAN.md` | 2.9 KB | DOPD 升级计划（副本） |
| `docs/PD_INTEGRATION_DELIVERY.md` | 本文档 | 交付总结 |

### 复制文件（从 CGC-main/cgc_engine/pd/）

31 个核心 PD 文件 + `mot_h.py`（从 CGC_Phase2/mot_h/），详见目录架构。

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `pd/__init__.py` | 新增整合模块导出（pd_config、master_slave_sync、compute_sharing） |
| `pd/coordinator.py` | `from mot_h import` → `from .mot_h import` |
| `pd/router.py` | `from discovery import` → `from .discovery import` |
| `pd/collect_batch.py` 等 6 个脚本 | 新增 sys.path shim（支持直接运行） |
| `config.env` | 新增 PD 配置段（40 行） |

---

## 九、与远端 cgcengine0907 的差异

- 远端 `agent_harness/`: 630 个文件
- 本地 `agent_harness/`: 758 个文件（排除缓存）
- **远端独有文件: 0**（本地是远端完全超集）
- **本地独有文件: 128**（含本次整合的 loopmoe/pd/qwen36 等）
- `agents/` 和 `harness/` 核心代码**完全一致**

---

## 十、后续工作

1. **提交到 git** — 本地新增的 128 个文件建议整理后提交
2. **推送到远端** — 将整合内容推送到 cgcengine0907 仓库
3. **部署 Master/Slave 同步** — 在 Mac A/B、Windows 上配置同步角色
4. **部署算力共享节点** — 在各设备上启动 `compute_sharing.py --role node`
5. **端到端 PD 验证** — 运行 `pd_e2e_test.py` 验证 Mac prefill → Windows decode 全链路
