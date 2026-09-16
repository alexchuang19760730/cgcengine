"""
端端 / 端云算力共享框架
========================
在多设备（Mac A、Mac B、Windows、Linux 服务器）之间共享算力资源，
支持模型推理、训练、数据处理等任务的分布式调度。

核心概念：
- ComputeNode：算力节点（一台设备）
- ComputeResource：节点上的可用资源（GPU、CPU、内存、模型实例）
- ComputeTask：需要算力的任务
- ComputeScheduler：任务调度器，将任务分配到最优节点
- SharingPolicy：共享策略（独占/共享/优先级）

支持的共享模式：
1. 端端共享（Edge-Edge）：Mac A ↔ Mac B ↔ Windows 之间互相借用算力
2. 端云共享（Edge-Cloud）：端侧设备将任务卸载到云侧服务器
3. 混合共享：根据任务类型自动选择端侧或云侧

使用方式：
    # 启动算力节点
    python compute_sharing.py --role node --name mac-a-gemma4 --gpu 1 --port 9100

    # 启动调度器
    python compute_sharing.py --role scheduler --nodes mac-a:9100,mac-b:9100

    # Python API
    from agent_harness.pd.compute_sharing import ComputeScheduler, ComputeTask
    scheduler = ComputeScheduler()
    scheduler.register_node("mac-a", "http://192.168.1.101:9100", gpu_count=1)
    task = ComputeTask(task_type="inference", model="gemma4-26b", prompt="hello")
    result = scheduler.schedule(task)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 枚举与数据类
# ----------------------------------------------------------------------

class NodeRole(str, Enum):
    EDGE = "edge"            # 端侧设备（Mac、Windows 桌面）
    CLOUD = "cloud"          # 云侧服务器
    HYBRID = "hybrid"        # 混合角色


class TaskType(str, Enum):
    INFERENCE = "inference"          # 模型推理
    PREFILL = "prefill"              # Prefill 阶段
    DECODE = "decode"                # Decode 阶段
    TRAINING = "training"            # 模型训练
    FINE_TUNING = "fine_tuning"      # 微调
    DATA_PROCESSING = "data_processing"  # 数据处理
    EMBEDDING = "embedding"          # 向量计算


class SharingPolicy(str, Enum):
    EXCLUSIVE = "exclusive"      # 独占（任务运行时不接受新任务）
    SHARED = "shared"            # 共享（并发处理多个任务）
    PRIORITY = "priority"        # 优先级（高优先级任务可抢占）


class TaskStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GPUInfo:
    """GPU 信息"""
    index: int
    name: str = ""
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    utilization_percent: float = 0.0
    temperature_c: float = 0.0

    @property
    def memory_free_gb(self) -> float:
        return max(0.0, self.memory_total_gb - self.memory_used_gb)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
            "memory_free_gb": self.memory_free_gb,
            "utilization_percent": self.utilization_percent,
            "temperature_c": self.temperature_c,
        }


@dataclass
class ModelInstance:
    """运行中的模型实例"""
    model_id: str
    backend: str = ""            # llama.cpp / vllm / sglang / pytorch
    quantization: str = ""       # q4_k_m / fp16 / etc.
    context_size: int = 0
    running: bool = False
    gpu_index: int = -1
    port: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "quantization": self.quantization,
            "context_size": self.context_size,
            "running": self.running,
            "gpu_index": self.gpu_index,
            "port": self.port,
        }


@dataclass
class ComputeNode:
    """算力节点"""
    node_id: str
    name: str
    role: NodeRole
    address: str = ""                 # HTTP API 地址
    hostname: str = ""
    os: str = ""

    # 硬件资源
    cpu_cores: int = 0
    cpu_usage_percent: float = 0.0
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    gpus: List[GPUInfo] = field(default_factory=list)

    # 模型实例
    models: List[ModelInstance] = field(default_factory=list)

    # 共享策略
    policy: SharingPolicy = SharingPolicy.SHARED
    max_concurrent_tasks: int = 4
    enabled: bool = True

    # 状态
    last_heartbeat: float = 0.0
    current_tasks: int = 0
    total_tasks_completed: int = 0

    @property
    def memory_free_gb(self) -> float:
        return max(0.0, self.memory_total_gb - self.memory_used_gb)

    @property
    def gpu_memory_free_total(self) -> float:
        return sum(g.memory_free_gb for g in self.gpus)

    @property
    def is_alive(self) -> bool:
        return (time.time() - self.last_heartbeat) < 60  # 60秒内有心跳

    def can_accept_task(self, task: "ComputeTask") -> bool:
        """检查节点是否能接受任务"""
        if not self.enabled or not self.is_alive:
            return False
        if self.policy == SharingPolicy.EXCLUSIVE and self.current_tasks > 0:
            return False
        if self.current_tasks >= self.max_concurrent_tasks:
            return False
        # 检查模型是否可用
        if task.model_id:
            has_model = any(m.model_id == task.model_id and m.running for m in self.models)
            if not has_model and task.require_model:
                return False
        # 检查 GPU 显存
        if task.gpu_memory_required_gb > 0:
            if self.gpu_memory_free_total < task.gpu_memory_required_gb:
                return False
        return True

    def score_task(self, task: "ComputeTask") -> float:
        """评估节点处理任务的适合度（分数越高越好）"""
        score = 0.0

        # 模型匹配度（权重最高）
        if task.model_id:
            matching_models = [m for m in self.models if m.model_id == task.model_id and m.running]
            if matching_models:
                score += 100.0
                # 偏好上下文更大的实例
                score += max(m.context_size for m in matching_models) / 1000.0
            elif not task.require_model:
                score += 30.0  # 可以加载模型
            else:
                return -1.0  # 不匹配

        # 资源充裕度
        if self.gpus:
            avg_gpu_util = sum(g.utilization_percent for g in self.gpus) / len(self.gpus)
            score += (100.0 - avg_gpu_util) * 0.3
            score += self.gpu_memory_free_total * 2.0

        # 内存
        score += self.memory_free_gb * 0.5

        # CPU
        score += (100.0 - self.cpu_usage_percent) * 0.1

        # 负载（当前任务数越少越好）
        score -= self.current_tasks * 10.0

        # 角色偏好
        if task.prefer_cloud and self.role == NodeRole.CLOUD:
            score += 50.0
        if task.prefer_edge and self.role == NodeRole.EDGE:
            score += 20.0

        return score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "role": self.role.value,
            "address": self.address,
            "hostname": self.hostname,
            "os": self.os,
            "cpu_cores": self.cpu_cores,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
            "memory_free_gb": self.memory_free_gb,
            "gpus": [g.to_dict() for g in self.gpus],
            "models": [m.to_dict() for m in self.models],
            "policy": self.policy.value,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "enabled": self.enabled,
            "is_alive": self.is_alive,
            "current_tasks": self.current_tasks,
            "total_tasks_completed": self.total_tasks_completed,
        }


@dataclass
class ComputeTask:
    """算力任务"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: TaskType = TaskType.INFERENCE
    model_id: str = ""
    require_model: bool = True

    # 任务参数
    prompt: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # 资源需求
    gpu_memory_required_gb: float = 0.0
    cpu_cores_required: int = 0
    memory_required_gb: float = 0.0

    # 调度偏好
    prefer_cloud: bool = False
    prefer_edge: bool = False
    preferred_nodes: List[str] = field(default_factory=list)
    excluded_nodes: List[str] = field(default_factory=list)
    timeout_seconds: int = 300

    # 状态
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: str = ""
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def duration_seconds(self) -> float:
        if self.completed_at > 0:
            return self.completed_at - self.started_at
        if self.started_at > 0:
            return time.time() - self.started_at
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "model_id": self.model_id,
            "status": self.status.value,
            "assigned_node": self.assigned_node,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class ScheduleResult:
    """调度结果"""
    success: bool
    task: Optional[ComputeTask] = None
    node: Optional[ComputeNode] = None
    reason: str = ""
    candidates: List[Tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        if self.success:
            return f"Task {self.task.task_id} → {self.node.name} (score={self.candidates[0][1]:.1f})"
        return f"Schedule failed: {self.reason}"


# ----------------------------------------------------------------------
# 算力调度器
# ----------------------------------------------------------------------

class ComputeScheduler:
    """算力调度器：将任务分配到最优算力节点"""

    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.tasks: Dict[str, ComputeTask] = {}
        self._lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # 节点管理
    # ------------------------------------------------------------------

    def register_node(
        self,
        node_id: str,
        name: str,
        address: str,
        role: NodeRole = NodeRole.EDGE,
        gpu_count: int = 0,
        cpu_cores: int = 0,
        memory_total_gb: float = 0.0,
        policy: SharingPolicy = SharingPolicy.SHARED,
    ) -> ComputeNode:
        """注册算力节点"""
        node = ComputeNode(
            node_id=node_id,
            name=name,
            role=role,
            address=address,
            cpu_cores=cpu_cores,
            memory_total_gb=memory_total_gb,
            policy=policy,
            last_heartbeat=time.time(),
        )
        # 初始化 GPU 信息
        for i in range(gpu_count):
            node.gpus.append(GPUInfo(index=i, name=f"GPU-{i}", memory_total_gb=memory_total_gb / max(1, gpu_count)))

        with self._lock:
            self.nodes[node_id] = node

        logger.info(f"Registered node: {name} ({role.value}) at {address}")
        return node

    def unregister_node(self, node_id: str) -> None:
        """注销算力节点"""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered node: {node_id}")

    def update_node_heartbeat(self, node_id: str, stats: Optional[Dict[str, Any]] = None) -> None:
        """更新节点心跳和资源状态"""
        with self._lock:
            if node_id not in self.nodes:
                return
            node = self.nodes[node_id]
            node.last_heartbeat = time.time()

            if stats:
                if "cpu_usage_percent" in stats:
                    node.cpu_usage_percent = stats["cpu_usage_percent"]
                if "memory_used_gb" in stats:
                    node.memory_used_gb = stats["memory_used_gb"]
                if "current_tasks" in stats:
                    node.current_tasks = stats["current_tasks"]
                if "gpus" in stats:
                    node.gpus = [GPUInfo(**g) for g in stats["gpus"]]
                if "models" in stats:
                    node.models = [ModelInstance(**m) for m in stats["models"]]

    def get_alive_nodes(self) -> List[ComputeNode]:
        """获取所有存活节点"""
        with self._lock:
            return [n for n in self.nodes.values() if n.is_alive]

    def get_node_status(self) -> List[Dict[str, Any]]:
        """获取所有节点状态摘要"""
        with self._lock:
            return [n.to_dict() for n in self.nodes.values()]

    # ------------------------------------------------------------------
    # 任务调度
    # ------------------------------------------------------------------

    def schedule(self, task: ComputeTask) -> ScheduleResult:
        """调度任务到最优节点"""
        with self._lock:
            # 筛选候选节点
            candidates = []
            for node in self.nodes.values():
                if node.node_id in task.excluded_nodes:
                    continue
                if not node.can_accept_task(task):
                    continue
                score = node.score_task(task)
                if score >= 0:
                    candidates.append((node, score))

            # 偏好节点加权
            if task.preferred_nodes:
                for i, (node, score) in enumerate(candidates):
                    if node.node_id in task.preferred_nodes:
                        candidates[i] = (node, score + 100.0)

            # 按分数排序
            candidates.sort(key=lambda x: x[1], reverse=True)

            if not candidates:
                task.status = TaskStatus.FAILED
                task.error = "No available compute node"
                self.tasks[task.task_id] = task
                return ScheduleResult(
                    success=False,
                    task=task,
                    reason="No available compute node matching requirements",
                )

            # 分配到最优节点
            best_node, best_score = candidates[0]
            task.assigned_node = best_node.node_id
            task.status = TaskStatus.SCHEDULED
            best_node.current_tasks += 1
            self.tasks[task.task_id] = task

            result = ScheduleResult(
                success=True,
                task=task,
                node=best_node,
                candidates=[(n.node_id, s) for n, s in candidates[:5]],
            )

            logger.info(f"Scheduled task {task.task_id} ({task.task_type.value}) → {best_node.name} (score={best_score:.1f})")
            return result

    def execute_task(self, task: ComputeTask) -> ComputeTask:
        """执行任务（通过 HTTP API 调用节点）"""
        if task.status != TaskStatus.SCHEDULED:
            schedule_result = self.schedule(task)
            if not schedule_result.success:
                return task

        node = self.nodes.get(task.assigned_node)
        if not node:
            task.status = TaskStatus.FAILED
            task.error = "Assigned node not found"
            return task

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        try:
            # 通过 HTTP API 调用节点执行任务
            import urllib.request

            payload = {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "model_id": task.model_id,
                "prompt": task.prompt,
                "max_tokens": task.max_tokens,
                "temperature": task.temperature,
                "extra_params": task.extra_params,
            }

            url = f"{node.address.rstrip('/')}/v1/compute/execute"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=task.timeout_seconds) as resp:
                result_data = json.loads(resp.read())

            task.result = result_data.get("result", result_data)
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()

            with self._lock:
                if node.node_id in self.nodes:
                    self.nodes[node.node_id].current_tasks = max(0, self.nodes[node.node_id].current_tasks - 1)
                    self.nodes[node.node_id].total_tasks_completed += 1

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            with self._lock:
                if node.node_id in self.nodes:
                    self.nodes[node.node_id].current_tasks = max(0, self.nodes[node.node_id].current_tasks - 1)
            logger.error(f"Task {task.task_id} failed on {node.name}: {e}")

        return task

    def get_task(self, task_id: str) -> Optional[ComputeTask]:
        """获取任务状态"""
        with self._lock:
            return self.tasks.get(task_id)

    def get_pending_tasks(self) -> List[ComputeTask]:
        """获取待处理任务"""
        with self._lock:
            return [t for t in self.tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.SCHEDULED)]

    # ------------------------------------------------------------------
    # 端端 / 端云共享策略
    # ------------------------------------------------------------------

    def share_compute(
        self,
        from_node: str,
        to_node: str,
        task_type: TaskType = TaskType.INFERENCE,
        model_id: str = "",
        duration_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """
        算力共享：将一个节点的算力共享给另一个节点使用。

        端端共享示例：Mac A 将 Gemma4 推理算力共享给 Windows 使用
        端云共享示例：Windows 将训练任务卸载到云侧服务器
        """
        with self._lock:
            if from_node not in self.nodes or to_node not in self.nodes:
                return {"success": False, "error": "Node not found"}

            source = self.nodes[from_node]
            target = self.nodes[to_node]

            # 检查源节点是否有可用资源
            if source.current_tasks >= source.max_concurrent_tasks:
                return {"success": False, "error": "Source node is fully loaded"}

            # 检查模型是否可用
            if model_id:
                has_model = any(m.model_id == model_id and m.running for m in source.models)
                if not has_model:
                    return {"success": False, "error": f"Model {model_id} not available on {source.name}"}

            share_id = hashlib.md5(f"{from_node}:{to_node}:{time.time()}".encode()).hexdigest()[:8]

            return {
                "success": True,
                "share_id": share_id,
                "from_node": source.name,
                "to_node": target.name,
                "task_type": task_type.value,
                "model_id": model_id,
                "duration_seconds": duration_seconds,
                "endpoint": f"{source.address}/v1/compute/execute",
                "created_at": time.time(),
            }

    def get_sharing_topology(self) -> Dict[str, Any]:
        """获取算力共享拓扑图"""
        alive_nodes = self.get_alive_nodes()
        edge_nodes = [n for n in alive_nodes if n.role == NodeRole.EDGE]
        cloud_nodes = [n for n in alive_nodes if n.role == NodeRole.CLOUD]

        return {
            "total_nodes": len(alive_nodes),
            "edge_nodes": [n.name for n in edge_nodes],
            "cloud_nodes": [n.name for n in cloud_nodes],
            "total_gpu_memory_gb": sum(n.gpu_memory_free_total for n in alive_nodes),
            "total_memory_gb": sum(n.memory_free_gb for n in alive_nodes),
            "running_models": [
                f"{n.name}:{m.model_id}"
                for n in alive_nodes
                for m in n.models
                if m.running
            ],
            "nodes": [n.to_dict() for n in alive_nodes],
        }


# ----------------------------------------------------------------------
# 算力节点服务（HTTP API）
# ----------------------------------------------------------------------

def start_compute_node(
    node_id: str,
    name: str,
    address: str = "0.0.0.0",
    port: int = 9100,
    role: NodeRole = NodeRole.EDGE,
    scheduler_url: str = "",
) -> None:
    """启动算力节点服务（FastAPI）"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        logger.error("fastapi and uvicorn are required for compute node")
        sys.exit(1)

    app = FastAPI(title=f"Compute Node: {name}", version="1.0")
    node_info = {
        "node_id": node_id,
        "name": name,
        "role": role.value,
        "port": port,
        "started_at": time.time(),
    }

    @app.get("/v1/compute/status")
    def get_status():
        return {**node_info, "uptime_seconds": time.time() - node_info["started_at"]}

    @app.post("/v1/compute/execute")
    def execute_task(payload: Dict[str, Any]):
        task_type = payload.get("task_type", "inference")
        model_id = payload.get("model_id", "")
        prompt = payload.get("prompt", "")
        max_tokens = payload.get("max_tokens", 256)
        temperature = payload.get("temperature", 0.7)

        # 这里调用本地模型推理
        # 实际实现中会调用 llama.cpp / vllm / sglang 等后端
        result = {
            "task_id": payload.get("task_id", ""),
            "model_id": model_id,
            "task_type": task_type,
            "output": f"[Simulated] Response from {name} for prompt: {prompt[:50]}...",
            "node": name,
        }
        return result

    @app.get("/v1/compute/models")
    def list_models():
        return {"models": [], "node": name}

    logger.info(f"Starting compute node '{name}' on {address}:{port}")
    uvicorn.run(app, host=address, port=port)


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PD Compute Sharing Framework")
    parser.add_argument("--role", choices=["node", "scheduler", "client"], default="node")
    parser.add_argument("--node-id", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--address", default="0.0.0.0")
    parser.add_argument("--node-role", choices=["edge", "cloud", "hybrid"], default="edge")
    parser.add_argument("--gpu-count", type=int, default=0)
    parser.add_argument("--scheduler-url", default="")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.role == "node":
        node_id = args.node_id or f"node-{uuid.uuid4().hex[:6]}"
        name = args.name or node_id
        start_compute_node(
            node_id=node_id,
            name=name,
            address=args.address,
            port=args.port,
            role=NodeRole(args.node_role),
            scheduler_url=args.scheduler_url,
        )
    elif args.role == "scheduler":
        logger.info("Scheduler mode: use Python API to manage nodes and tasks")
        logger.info("Example:")
        logger.info("  from agent_harness.pd.compute_sharing import ComputeScheduler, ComputeTask")
        logger.info("  scheduler = ComputeScheduler()")
        logger.info("  scheduler.register_node('mac-a', 'Mac A Gemma4', 'http://192.168.1.101:9100')")
        logger.info("  task = ComputeTask(task_type='inference', model_id='gemma4-26b', prompt='hello')")
        logger.info("  result = scheduler.schedule(task)")
    elif args.role == "client":
        logger.info("Client mode: use Python API to submit tasks")


if __name__ == "__main__":
    main()
