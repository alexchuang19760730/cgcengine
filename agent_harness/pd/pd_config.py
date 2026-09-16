"""
PD (Prefill-Decode) 统一配置
==============================
端云 PD 分离、master/slave 代码同步、端端/端云算力共享的统一配置。

从环境变量或 config.env 加载配置，也支持 Python API 直接设置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PDServerConfig:
    """PD 服务端配置（Windows 端侧 coordinator）"""
    host: str = "0.0.0.0"
    port: int = 9000
    coordinator_url: str = "http://127.0.0.1:9000"
    ingest_endpoint: str = "/v1/cgc/ingest"
    emit_endpoint: str = "/v1/cgc/emit"

    # MoT-h 训练
    mot_h_model_path: str = ""
    mot_h_batch_size: int = 32
    mot_h_learning_rate: float = 1e-4
    mot_h_epochs: int = 10

    # 数据存储
    data_dir: str = "./pd_data"
    hidden_state_dir: str = "./pd_data/hidden_states"
    training_data_dir: str = "./pd_data/training"


@dataclass
class PDEdgeConfig:
    """PD 边缘节点配置（Mac 端 prefill 服务器）"""
    node_id: str = ""
    node_name: str = ""
    host: str = "0.0.0.0"
    port: int = 8080
    model_id: str = ""
    model_path: str = ""
    backend: str = "llama.cpp"  # llama.cpp / sglang / vllm

    # 上报到 coordinator
    coordinator_url: str = ""
    auto_register: bool = True
    heartbeat_interval: int = 30

    # Prefill 配置
    max_prefill_tokens: int = 8192
    emit_hidden_state: bool = True
    hidden_state_layers: str = "last"  # last / all / specific


@dataclass
class MasterSlaveConfig:
    """Master/Slave 代码同步配置"""
    enabled: bool = False
    role: str = "slave"  # master / slave
    method: str = "incremental"  # git / file / http / incremental

    # Master 信息
    master_url: str = ""
    master_path: str = ""
    master_git_remote: str = "origin"
    master_git_branch: str = "main"

    # 同步策略
    sync_interval: int = 300  # 秒
    sync_directories: List[str] = field(default_factory=lambda: [
        "pd", "loopmoe", "qwen36", "agents", "finetune", "scripts"
    ])
    auto_restart_after_sync: bool = False
    verify_after_sync: bool = True

    # Master 服务
    master_port: int = 9000


@dataclass
class ComputeSharingConfig:
    """端端/端云算力共享配置"""
    enabled: bool = False
    node_id: str = ""
    node_name: str = ""
    node_role: str = "edge"  # edge / cloud / hybrid
    host: str = "0.0.0.0"
    port: int = 9100

    # 调度器
    scheduler_url: str = ""
    auto_register: bool = True

    # 资源声明
    gpu_count: int = 0
    gpu_memory_total_gb: float = 0.0
    cpu_cores: int = 0
    memory_total_gb: float = 0.0

    # 共享策略
    sharing_policy: str = "shared"  # exclusive / shared / priority
    max_concurrent_tasks: int = 4

    # 可用模型
    available_models: List[str] = field(default_factory=list)


@dataclass
class PDConfig:
    """PD 统一配置"""
    server: PDServerConfig = field(default_factory=PDServerConfig)
    edge: PDEdgeConfig = field(default_factory=PDEdgeConfig)
    master_slave: MasterSlaveConfig = field(default_factory=MasterSlaveConfig)
    compute_sharing: ComputeSharingConfig = field(default_factory=ComputeSharingConfig)

    # 全局
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "PDConfig":
        """从环境变量加载配置"""
        config = cls()

        # Server
        config.server.host = os.environ.get("PD_SERVER_HOST", config.server.host)
        config.server.port = int(os.environ.get("PD_SERVER_PORT", config.server.port))
        config.server.coordinator_url = os.environ.get("PD_COORDINATOR_URL", config.server.coordinator_url)
        config.server.data_dir = os.environ.get("PD_DATA_DIR", config.server.data_dir)

        # Edge
        config.edge.node_id = os.environ.get("PD_NODE_ID", config.edge.node_id)
        config.edge.node_name = os.environ.get("PD_NODE_NAME", config.edge.node_name)
        config.edge.port = int(os.environ.get("PD_EDGE_PORT", config.edge.port))
        config.edge.model_id = os.environ.get("PD_MODEL_ID", config.edge.model_id)
        config.edge.coordinator_url = os.environ.get("PD_COORDINATOR_URL", config.edge.coordinator_url)

        # Master/Slave
        config.master_slave.enabled = os.environ.get("PD_SYNC_ENABLED", "0") == "1"
        config.master_slave.role = os.environ.get("PD_SYNC_ROLE", config.master_slave.role)
        config.master_slave.method = os.environ.get("PD_SYNC_METHOD", config.master_slave.method)
        config.master_slave.master_url = os.environ.get("PD_SYNC_MASTER_URL", config.master_slave.master_url)
        config.master_slave.sync_interval = int(os.environ.get("PD_SYNC_INTERVAL", config.master_slave.sync_interval))

        # Compute Sharing
        config.compute_sharing.enabled = os.environ.get("PD_COMPUTE_ENABLED", "0") == "1"
        config.compute_sharing.node_id = os.environ.get("PD_COMPUTE_NODE_ID", config.compute_sharing.node_id)
        config.compute_sharing.node_name = os.environ.get("PD_COMPUTE_NODE_NAME", config.compute_sharing.node_name)
        config.compute_sharing.node_role = os.environ.get("PD_COMPUTE_NODE_ROLE", config.compute_sharing.node_role)
        config.compute_sharing.port = int(os.environ.get("PD_COMPUTE_PORT", config.compute_sharing.port))
        config.compute_sharing.scheduler_url = os.environ.get("PD_COMPUTE_SCHEDULER_URL", config.compute_sharing.scheduler_url)
        config.compute_sharing.gpu_count = int(os.environ.get("PD_COMPUTE_GPU_COUNT", config.compute_sharing.gpu_count))
        config.compute_sharing.sharing_policy = os.environ.get("PD_COMPUTE_POLICY", config.compute_sharing.sharing_policy)

        # Global
        config.debug = os.environ.get("PD_DEBUG", "0") == "1"
        config.log_level = os.environ.get("PD_LOG_LEVEL", config.log_level)

        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server": self.server.__dict__,
            "edge": self.edge.__dict__,
            "master_slave": self.master_slave.__dict__,
            "compute_sharing": self.compute_sharing.__dict__,
            "debug": self.debug,
            "log_level": self.log_level,
        }
