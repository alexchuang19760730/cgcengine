
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

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class SPDKConfig:
    """SPDK 适配器配置"""
    
    # 是否启用 SPDK
    enable_spdk: bool = False
    
    # SPDK 环境配置
    spdk_env_path: Optional[Path] = None
    
    # NVMe 设备列表
    nvme_devices: List[str] = field(default_factory=lambda: ["nvme0n1"])
    
    # 块大小 (字节)
    block_size: int = 4096
    
    # 队列深度
    queue_depth: int = 128
    
    # I/O 队列数量
    io_queues: int = 4
    
    # 是否启用零拷贝
    enable_zero_copy: bool = True
    
    # 是否启用批量 I/O
    enable_batch_io: bool = True
    
    # 批量 I/O 大小
    batch_io_size: int = 32
    
    # 缓存配置
    cache_size_mb: int = 1024
    
    # KV 存储配置
    kv_store_path: Optional[Path] = None
    
    # 专家权重存储路径
    expert_store_path: Optional[Path] = None
    
    # Fallback 到文件系统（SPDK 不可用时）
    fallback_to_fs: bool = True
    
    # 日志级别
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.kv_store_path is None:
            self.kv_store_path = Path("/tmp/spdk_kv")
        if self.expert_store_path is None:
            self.expert_store_path = Path("/tmp/spdk_experts")

