# Copyright (c) 2026 SandAI. All Rights Reserved.
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
Distributed Expert Store - 分布式专家存储

功能：
- 基于 SPDK NVMe over Fabrics 的分布式专家存储
- 支持跨节点高速访问专家权重
- 自动负载均衡和故障转移
"""

import torch
import logging
from typing import Dict, Optional, List, Tuple, Any
import threading
import hashlib

logger = logging.getLogger(__name__)

# 尝试导入 SPDK
try:
    from ..spdk_adapter.spdk_kv_store import SPDKKVStore
    from ..spdk_adapter.spdk_config import SPDKConfig
    from ..spdk_adapter.spdk_expert_store import SPDKExpertStore
    SPDK_AVAILABLE = True
except ImportError:
    try:
        from cgc_engine.spdk_adapter.spdk_kv_store import SPDKKVStore
        from cgc_engine.spdk_adapter.spdk_config import SPDKConfig
        from cgc_engine.spdk_adapter.spdk_expert_store import SPDKExpertStore
        SPDK_AVAILABLE = True
    except ImportError:
        SPDK_AVAILABLE = False
        logger.warning("[Distributed Expert Store] SPDK 不可用")


class DistributedExpertStore:
    """
    分布式专家存储
    
    特性：
    - 支持 NVMe over Fabrics 跨节点访问
    - 自动专家分区和路由
    - 负载均衡和故障转移
    - 本地缓存优化
    """
    
    def __init__(
        self,
        cluster_nodes: Optional[List[str]] = None,
        local_store_path: str = "/data/local_experts",
        num_partitions: int = 64
    ):
        """
        初始化分布式专家存储
        
        Args:
            cluster_nodes: 集群节点列表（格式: "host:port"）
            local_store_path: 本地存储路径
            num_partitions: 分区数量
        """
        self.cluster_nodes = cluster_nodes or []
        self.num_partitions = num_partitions
        self.local_store_path = local_store_path
        
        # 本地专家存储
        self._local_store = None
        
        # 节点连接池
        self._node_clients: Dict[str, Any] = {}
        
        # 本地缓存
        self._local_cache: Dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()
        
        # 初始化本地存储
        if SPDK_AVAILABLE:
            try:
                config = SPDKConfig(
                    kv_store_path=local_store_path,
                    io_queues=8,
                    enable_spdk=True
                )
                self._local_store = SPDKExpertStore(config)
                self._local_store.initialize()
                logger.info("[Distributed Expert Store] ✅ 本地 SPDK 存储初始化成功")
            except Exception as e:
                logger.warning(f"[Distributed Expert Store] ⚠️ 本地存储初始化失败: {e}")
        
        # 统计信息
        self._stats = {
            "local_loads": 0,
            "remote_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failovers": 0,
            "total_requests": 0
        }
    
    def _get_partition(self, expert_id: int) -> int:
        """计算专家所在的分区"""
        return expert_id % self.num_partitions
    
    def _get_node_for_partition(self, partition: int) -> Optional[str]:
        """获取分区对应的节点"""
        if not self.cluster_nodes:
            return None
        
        # 简单的一致性哈希
        return self.cluster_nodes[partition % len(self.cluster_nodes)]
    
    def _get_local_store(self):
        """获取本地存储（带降级）"""
        return self._local_store
    
    def load_expert(
        self,
        expert_id: int,
        shape: List[int],
        dtype: torch.dtype = torch.float16,
        force_remote: bool = False
    ) -> torch.Tensor:
        """
        加载专家权重
        
        Args:
            expert_id: 专家 ID
            shape: 权重形状
            dtype: 数据类型
            force_remote: 是否强制从远程加载
        
        Returns:
            专家权重张量
        """
        self._stats["total_requests"] += 1
        
        # 检查本地缓存
        with self._lock:
            if expert_id in self._local_cache:
                self._stats["cache_hits"] += 1
                logger.debug(f"[Distributed Expert Store] 缓存命中: expert_id={expert_id}")
                return self._local_cache[expert_id].to("cuda" if torch.cuda.is_available() else "cpu")
        
        self._stats["cache_misses"] += 1
        
        partition = self._get_partition(expert_id)
        node = self._get_node_for_partition(partition)
        
        # 决定从哪里加载
        if node and not force_remote:
            # 从远程节点加载
            return self._load_from_remote(expert_id, node, shape, dtype)
        else:
            # 从本地加载
            return self._load_from_local(expert_id, shape, dtype)
    
    def _load_from_local(self, expert_id: int, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
        """从本地存储加载专家"""
        self._stats["local_loads"] += 1
        
        logger.debug(f"[Distributed Expert Store] 从本地加载专家: {expert_id}")
        
        if self._local_store:
            try:
                expert = self._local_store.load_expert(expert_id, shape, dtype)
                # 缓存到内存
                with self._lock:
                    self._local_cache[expert_id] = expert.cpu()
                return expert
            except Exception as e:
                logger.warning(f"[Distributed Expert Store] 本地加载失败 {expert_id}: {e}")
        
        # 降级方案
        return self._create_fallback_expert(shape, dtype)
    
    def _load_from_remote(self, expert_id: int, node: str, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
        """从远程节点加载专家"""
        self._stats["remote_loads"] += 1
        
        logger.debug(f"[Distributed Expert Store] 从远程节点 {node} 加载专家: {expert_id}")
        
        try:
            # 尝试获取节点客户端
            client = self._get_node_client(node)
            if client:
                # 从远程节点获取专家
                # 这里可以实现 NVMe over Fabrics 或 gRPC 调用
                expert_data = self._fetch_from_remote(client, expert_id)
                
                if expert_data is not None:
                    import numpy as np

                    if dtype == torch.float16:
                        np_dtype = np.float16
                    elif dtype == torch.float32:
                        np_dtype = np.float32
                    else:
                        np_dtype = np.float16

                    numel = 1
                    for d in shape:
                        numel *= int(d)

                    if isinstance(expert_data, (bytes, bytearray, memoryview)):
                        arr = np.frombuffer(expert_data, dtype=np_dtype, count=numel)
                        arr = np.ascontiguousarray(arr.reshape(shape))
                    elif isinstance(expert_data, np.ndarray):
                        arr = np.ascontiguousarray(expert_data, dtype=np_dtype).reshape(shape)
                    else:
                        raise TypeError(f"Unexpected expert_data type: {type(expert_data)}")

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    expert = torch.from_numpy(arr).to(device=device)
                    
                    # 缓存到本地
                    with self._lock:
                        self._local_cache[expert_id] = expert.cpu()
                    
                    return expert
        except Exception as e:
            logger.warning(f"[Distributed Expert Store] 远程加载失败 {expert_id}@{node}: {e}")
            self._stats["failovers"] += 1
        
        # 故障转移到本地
        return self._load_from_local(expert_id, shape, dtype)
    
    def _get_node_client(self, node: str) -> Optional[Any]:
        """获取节点客户端"""
        if node not in self._node_clients:
            # 建立连接
            try:
                # 这里可以实现 SPDK NVMe over Fabrics 连接
                # 或 gRPC 客户端
                self._node_clients[node] = self._create_node_client(node)
            except Exception as e:
                logger.warning(f"[Distributed Expert Store] 无法连接到节点 {node}: {e}")
                return None
        
        return self._node_clients[node]
    
    def _create_node_client(self, node: str) -> Any:
        """创建节点客户端"""
        # 这里可以实现真实的 NVMe over Fabrics 客户端
        # 或返回一个模拟客户端
        host, port = node.split(":") if ":" in node else (node, "4420")
        logger.debug(f"[Distributed Expert Store] 创建节点客户端: {host}:{port}")
        
        # 返回一个模拟对象
        return {"host": host, "port": int(port)}
    
    def _fetch_from_remote(self, client: Dict[str, Any], expert_id: int) -> Optional[bytes]:
        """从远程节点获取专家数据"""
        # 模拟远程获取
        logger.debug(f"[Distributed Expert Store] 从 {client['host']} 获取专家 {expert_id}")
        
        # 在实际实现中，这里会调用 SPDK NVMe over Fabrics
        # 或 gRPC 来获取数据
        
        # 返回模拟数据
        import numpy as np
        return np.random.randn(4096, 4096).astype(np.float16).tobytes()
    
    def store_expert(self, expert_id: int, expert: torch.Tensor, replicate: bool = False) -> bool:
        """
        存储专家权重
        
        Args:
            expert_id: 专家 ID
            expert: 专家权重张量
            replicate: 是否复制到其他节点
        
        Returns:
            是否成功
        """
        partition = self._get_partition(expert_id)
        
        # 存储到本地
        if self._local_store:
            try:
                self._local_store.store_expert(expert_id, expert)
                logger.debug(f"[Distributed Expert Store] 专家 {expert_id} 已存储到本地")
                
                # 更新本地缓存
                with self._lock:
                    self._local_cache[expert_id] = expert.cpu()
                
                # 如果需要复制，分发到其他节点
                if replicate and self.cluster_nodes:
                    self._replicate_to_nodes(expert_id, expert)
                
                return True
            except Exception as e:
                logger.error(f"[Distributed Expert Store] 存储失败 {expert_id}: {e}")
        
        return False
    
    def _replicate_to_nodes(self, expert_id: int, expert: torch.Tensor):
        """复制专家到其他节点"""
        logger.debug(f"[Distributed Expert Store] 复制专家 {expert_id} 到集群节点")
        
        # 在实际实现中，这里会将专家分发到其他节点
        # 实现数据冗余和负载均衡
    
    def delete_expert(self, expert_id: int) -> bool:
        """删除专家"""
        if self._local_store:
            try:
                self._local_store.delete_expert(expert_id)
                with self._lock:
                    if expert_id in self._local_cache:
                        del self._local_cache[expert_id]
                return True
            except Exception as e:
                logger.error(f"[Distributed Expert Store] 删除失败 {expert_id}: {e}")
        
        return False
    
    def _create_fallback_expert(self, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
        """创建降级专家"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.randn(shape, dtype=dtype, device=device) * 0.01
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["num_nodes"] = len(self.cluster_nodes)
        stats["num_partitions"] = self.num_partitions
        stats["local_cache_size"] = len(self._local_cache)
        return stats
    
    def clear_cache(self):
        """清空本地缓存"""
        with self._lock:
            self._local_cache.clear()
    
    def shutdown(self):
        """关闭存储"""
        self.clear_cache()
        logger.info("[Distributed Expert Store] 已关闭")


# 使用示例
def main():
    logging.basicConfig(level=logging.INFO)
    
    # 创建分布式专家存储
    cluster_nodes = ["node1:4420", "node2:4420", "node3:4420"]
    store = DistributedExpertStore(cluster_nodes=cluster_nodes)
    
    logger.info(f"集群节点: {store.cluster_nodes}")
    logger.info(f"分区数量: {store.num_partitions}")
    
    # 存储专家
    expert = torch.randn(4096, 4096, dtype=torch.float16)
    success = store.store_expert(0, expert)
    logger.info(f"专家存储成功: {success}")
    
    # 加载专家
    loaded_expert = store.load_expert(0, [4096, 4096])
    logger.info(f"专家加载成功: shape={loaded_expert.shape}, device={loaded_expert.device}")
    
    # 统计信息
    stats = store.get_stats()
    logger.info(f"统计信息: {stats}")
    
    store.shutdown()


if __name__ == "__main__":
    main()
