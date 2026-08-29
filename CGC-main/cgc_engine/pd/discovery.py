# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PD Service Discovery - etcd Integration

功能:
- PD 节点自动注册/注销
- 节点健康检查
- 负载均衡路由
- 集群拓扑发现
"""

import asyncio
import json
import time
import hashlib
import platform
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading

try:
    import etcd3
    ETCD_AVAILABLE = True
except ImportError:
    ETCD_AVAILABLE = False
    etcd3 = None



@dataclass
class DeviceProfile:
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    gpu_type: str = "none"
    gpu_vram_gb: float = 0.0
    cpu_cores: int = 0
    cpu_arch: str = ""
    compute_score: float = 0.0
    prefill_tok_per_sec: Dict = field(default_factory=dict)
    decode_tok_per_sec: Dict = field(default_factory=dict)
    network_latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    role: str = "both"
    max_concurrent: int = 1
    pd_modes: List = field(default_factory=lambda: ["端云","端端","纯端"])

    @classmethod
    def detect_local(cls):
        import os
        p = cls()
        try:
            if platform.system() == "Windows":
                import ctypes
                k32 = ctypes.windll.kernel32
                cu = ctypes.c_ulonglong
                class MS(ctypes.Structure):
                    _fields_ = [("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",cu),("ullAvailPhys",cu),("ullTotalPageFile",cu),("ullAvailPageFile",cu),("ullTotalVirtual",cu),("ullAvailVirtual",cu),("ullAvailExtendedVirtual",cu)]
                m = MS(); m.dwLength = ctypes.sizeof(MS)
                k32.GlobalMemoryStatusEx(ctypes.byref(m))
                p.total_ram_gb = round(m.ullTotalPhys/(1024**3),1)
                p.available_ram_gb = round(m.ullAvailPhys/(1024**3),1)
            else:
                pages = os.sysconf("SC_PHYS_PAGES")
                avail = os.sysconf("SC_AVPHYS_PAGES")
                ps = os.sysconf("SC_PAGE_SIZE")
                p.total_ram_gb = round(pages*ps/(1024**3),1)
                p.available_ram_gb = round(avail*ps/(1024**3),1)
        except: pass
        p.cpu_cores = os.cpu_count() or 0
        p.cpu_arch = platform.machine()
        try:
            if platform.system() == "Windows":
                r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True,timeout=5)
                if r.returncode == 0 and r.stdout.strip():
                    parts = r.stdout.strip().split(",")
                    p.gpu_type = parts[0].strip()
                    if len(parts)>1: p.gpu_vram_gb = float(parts[1].strip().replace("MiB","").replace("GiB",""))/1024
            elif platform.machine() == "arm64":
                p.gpu_type = "Apple Silicon"
                p.gpu_vram_gb = p.total_ram_gb * 0.7
        except: pass
        p.compute_score = round(min(p.total_ram_gb/32,1)*40 + min(p.gpu_vram_gb/24,1)*30 + min(p.cpu_cores/12,1)*30,1)
        return p

    def to_dict(self): return {k:v for k,v in self.__dict__.items()}
    @classmethod
    def from_dict(cls, d): return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__})

class NodeStatus(Enum):
    """节点状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class PDNode:
    """PD 节点"""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.STARTING
    weight: int = 100
    max_blocks: int = 10000
    current_load: int = 0
    region: str = "default"
    capabilities: List[str] = field(default_factory=lambda: ["kv_cache", "cgc", "flashkda"])
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    version: str = "1.0.0"
    profile: Optional[DeviceProfile] = None

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def load_factor(self) -> float:
        if self.max_blocks == 0:
            return 1.0
        return self.current_load / self.max_blocks


class EtcdServiceDiscovery:
    """
    PD 服务发现 - etcd 实现

    功能:
    - 节点注册/注销
    - 健康检查/心跳
    - 节点发现/路由
    - 负载均衡
    """

    def __init__(
        self,
        etcd_host: str = "localhost",
        etcd_port: int = 2379,
        service_name: str = "pd-service",
        ttl_seconds: int = 30,
    ):
        if not ETCD_AVAILABLE:
            raise RuntimeError("etcd3 not installed. Run: pip install etcd3")

        self.etcd_host = etcd_host
        self.etcd_port = etcd_port
        self.service_name = service_name
        self.ttl_seconds = ttl_seconds

        self.client = etcd3.client(host=etcd_host, port=etcd_port)

        self.prefix = f"/services/{service_name}"
        self.nodes: Dict[str, PDNode] = {}
        self._lock = threading.RLock()
        self._watcher = None

        print(f"[Etcd Discovery] Connected to etcd: {etcd_host}:{etcd_port}")

    def register_node(self, node: PDNode) -> bool:
        """
        注册 PD 节点

        Args:
            node: PD 节点信息

        Returns:
            success
        """
        key = f"{self.prefix}/{node.node_id}"

        value = json.dumps({
            "node_id": node.node_id,
            "host": node.host,
            "port": node.port,
            "status": node.status.value,
            "weight": node.weight,
            "max_blocks": node.max_blocks,
            "region": node.region,
            "capabilities": node.capabilities,
            "version": node.version,
            "registered_at": node.registered_at,
        })

        try:
            self.client.put(key, value)

            with self._lock:
                self.nodes[node.node_id] = node

            print(f"[Etcd Discovery] Node registered: {node.node_id} @ {node.address}")
            return True

        except Exception as e:
            print(f"[Etcd Discovery] Failed to register node: {e}")
            return False

    def register_node_with_lease(self, node: PDNode, lease_id: int) -> bool:
        """
        使用 lease 注册节点 (推荐)

        Args:
            node: PD 节点信息
            lease_id: etcd lease ID

        Returns:
            success
        """
        key = f"{self.prefix}/{node.node_id}"

        value = json.dumps({
            "node_id": node.node_id,
            "host": node.host,
            "port": node.port,
            "status": node.status.value,
            "weight": node.weight,
            "max_blocks": node.max_blocks,
            "region": node.region,
            "capabilities": node.capabilities,
            "version": node.version,
        })

        try:
            self.client.put(key, value, lease=lease_id)

            with self._lock:
                self.nodes[node.node_id] = node

            print(f"[Etcd Discovery] Node registered with lease: {node.node_id}")
            return True

        except Exception as e:
            print(f"[Etcd Discovery] Failed to register node: {e}")
            return False

    def deregister_node(self, node_id: str) -> bool:
        """
        注销节点

        Args:
            node_id: 节点 ID

        Returns:
            success
        """
        key = f"{self.prefix}/{node_id}"

        try:
            self.client.delete(key)

            with self._lock:
                if node_id in self.nodes:
                    del self.nodes[node_id]

            print(f"[Etcd Discovery] Node deregistered: {node_id}")
            return True

        except Exception as e:
            print(f"[Etcd Discovery] Failed to deregister node: {e}")
            return False

    def discover_nodes(
        self,
        status: Optional[NodeStatus] = None,
        region: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> List[PDNode]:
        """
        发现节点

        Args:
            status: 节点状态过滤
            region: 区域过滤
            capability: 能力过滤

        Returns:
            符合条件的节点列表
        """
        try:
            values, _ = self.client.get_prefix(self.prefix)

            nodes = []
            for value, metadata in values:
                if value is None:
                    continue

                try:
                    data = json.loads(value.decode() if isinstance(value, bytes) else value)
                    node = PDNode(
                        node_id=data["node_id"],
                        host=data["host"],
                        port=data["port"],
                        status=NodeStatus(data.get("status", "healthy")),
                        weight=data.get("weight", 100),
                        max_blocks=data.get("max_blocks", 10000),
                        region=data.get("region", "default"),
                        capabilities=data.get("capabilities", []),
                        version=data.get("version", "1.0.0"),
                    )

                    if status and node.status != status:
                        continue
                    if region and node.region != region:
                        continue
                    if capability and capability not in node.capabilities:
                        continue

                    nodes.append(node)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[Etcd Discovery] Failed to parse node data: {e}")
                    continue

            with self._lock:
                self.nodes = {n.node_id: n for n in nodes}

            return nodes

        except Exception as e:
            print(f"[Etcd Discovery] Failed to discover nodes: {e}")
            return []

    def get_least_loaded_node(self, region: Optional[str] = None) -> Optional[PDNode]:
        """
        获取负载最低的节点 (负载均衡)

        Args:
            region: 区域过滤

        Returns:
            负载最低的节点
        """
        nodes = self.discover_nodes(status=NodeStatus.HEALTHY, region=region)

        if not nodes:
            return None

        return min(nodes, key=lambda n: n.load_factor)

    def get_node_by_id(self, node_id: str) -> Optional[PDNode]:
        """根据 ID 获取节点"""
        key = f"{self.prefix}/{node_id}"

        try:
            value, _ = self.client.get(key)

            if value is None:
                return None

            data = json.loads(value.decode() if isinstance(value, bytes) else value)
            return PDNode(
                node_id=data["node_id"],
                host=data["host"],
                port=data["port"],
                status=NodeStatus(data.get("status", "healthy")),
                weight=data.get("weight", 100),
                max_blocks=data.get("max_blocks", 10000),
                region=data.get("region", "default"),
                capabilities=data.get("capabilities", []),
            )

        except Exception as e:
            print(f"[Etcd Discovery] Failed to get node: {e}")
            return None

    def watch_nodes(self, callback):
        """
        监听节点变化

        Args:
            callback: 节点变化回调 (added, modified, deleted)
        """
        def watch_callback(event):
            if event.type == "PUT":
                try:
                    data = json.loads(event.value.decode())
                    node = PDNode(
                        node_id=data["node_id"],
                        host=data["host"],
                        port=data["port"],
                    )
                    callback("added", node)
                except:
                    pass
            elif event.type == "DELETE":
                node_id = event.key.decode().split("/")[-1]
                callback("deleted", node_id)

        try:
            self._watcher = self.client.add_watch_callback(self.prefix, watch_callback)
            print(f"[Etcd Discovery] Watching {self.prefix}")
        except Exception as e:
            print(f"[Etcd Discovery] Failed to watch: {e}")

    def stop_watch(self):
        """停止监听"""
        if self._watcher:
            self.client.cancel_watch(self._watcher)
            self._watcher = None

    def create_lease(self, ttl_seconds: int = 30) -> int:
        """
        创建 lease

        Args:
            ttl_seconds: TTL 秒数

        Returns:
            lease_id
        """
        try:
            lease_id = self.client.lease(ttl_seconds)
            return lease_id.id
        except Exception as e:
            print(f"[Etcd Discovery] Failed to create lease: {e}")
            return 0

    def close(self):
        """关闭连接"""
        self.stop_watch()
        if self.client:
            self.client.close()


class PDNodeRegistrar:
    """
    PD 节点注册器

    用于 PD Server 启动时自动注册到 etcd
    """

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        etcd_host: str = "localhost",
        etcd_port: int = 2379,
        max_blocks: int = 10000,
        region: str = "default",
    ):
        self.node = PDNode(
            node_id=node_id,
            host=host,
            port=port,
            status=NodeStatus.HEALTHY,
            max_blocks=max_blocks,
            region=region,
        )

        self.discovery = EtcdServiceDiscovery(
            etcd_host=etcd_host,
            etcd_port=etcd_port,
        )

        self.lease_id = 0
        self._heartbeat_task = None
        self._running = False

    def start(self):
        """启动注册"""
        self.lease_id = self.discovery.create_lease(ttl_seconds=self.discovery.ttl_seconds)

        if self.lease_id:
            self.discovery.register_node_with_lease(self.node, self.lease_id)
        else:
            self.discovery.register_node(self.node)

        self._running = True
        self._start_heartbeat()

        print(f"[Registrar] Node {self.node.node_id} started")

    def _start_heartbeat(self):
        """启动心跳"""
        def heartbeat():
            while self._running:
                time.sleep(self.discovery.ttl_seconds // 3)

                try:
                    self.node.last_heartbeat = time.time()
                    self.node.status = NodeStatus.HEALTHY

                    if self.lease_id:
                        self.discovery.client.refresh_lease(self.lease_id)
                    else:
                        self.discovery.register_node(self.node)

                except Exception as e:
                    print(f"[Registrar] Heartbeat failed: {e}")

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()

    def stop(self):
        """停止注册"""
        self._running = False
        self.node.status = NodeStatus.STOPPING
        self.discovery.deregister_node(self.node.node_id)
        self.discovery.close()

        print(f"[Registrar] Node {self.node.node_id} stopped")


class PDNodeSelector:
    """
    PD 节点选择器 (客户端用)

    支持:
    - 随机选择
    - 轮询选择
    - 负载最低
    - 一致性哈希
    """

    def __init__(self, etcd_host: str = "localhost", etcd_port: int = 2379):
        self.discovery = EtcdServiceDiscovery(
            etcd_host=etcd_host,
            etcd_port=etcd_port,
        )

        self._round_robin_index = 0
        self._lock = threading.Lock()

    def select_random(self) -> Optional[PDNode]:
        """随机选择"""
        nodes = self.discovery.discover_nodes(status=NodeStatus.HEALTHY)

        if not nodes:
            return None

        import random
        return random.choice(nodes)

    def select_round_robin(self) -> Optional[PDNode]:
        """轮询选择"""
        nodes = self.discovery.discover_nodes(status=NodeStatus.HEALTHY)

        if not nodes:
            return None

        with self._lock:
            node = nodes[self._round_robin_index % len(nodes)]
            self._round_robin_index += 1
            return node

    def select_least_loaded(self, region: Optional[str] = None) -> Optional[PDNode]:
        """负载最低选择"""
        return self.discovery.get_least_loaded_node(region=region)

    def select_by_hash(self, key: str) -> Optional[PDNode]:
        """一致性哈希选择"""
        nodes = self.discovery.discover_nodes(status=NodeStatus.HEALTHY)

        if not nodes:
            return None

        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        index = hash_value % len(nodes)

        return nodes[index]

    def select(self, strategy: str = "round_robin", **kwargs) -> Optional[PDNode]:
        """
        根据策略选择节点

        Args:
            strategy: 策略 (random, round_robin, least_loaded, hash)
            **kwargs: 策略参数

        Returns:
            选择的节点
        """
        if strategy == "random":
            return self.select_random()
        elif strategy == "round_robin":
            return self.select_round_robin()
        elif strategy == "least_loaded":
            return self.select_least_loaded(**kwargs)
        elif strategy == "hash":
            return self.select_by_hash(**kwargs)
        else:
            return self.select_round_robin()
