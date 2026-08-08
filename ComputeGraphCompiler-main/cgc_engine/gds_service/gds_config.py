
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
class GDSConfig:
    """GDS (GPUDirect Storage) 配置"""
    
    # 是否启用 GDS
    enable_gds: bool = False
    
    # GDS 库路径
    gds_library_path: Optional[Path] = None
    
    # 设备 ID
    device_id: int = 0
    
    # 是否使用 GDS 注册的内存
    use_registered_memory: bool = True
    
    # 注册的内存池大小 (MB)
    registered_memory_size_mb: int = 4096
    
    # 传输对齐要求
    alignment: int = 4096
    
    # 队列深度
    queue_depth: int = 128
    
    # I/O 大小 (KB)
    io_size_kb: int = 128
    
    # 是否启用异步 I/O
    enable_async: bool = True
    
    # Fallback 到普通 POSIX I/O（GDS 不可用时）
    fallback_to_posix: bool = True
    
    # 支持的文件系统
    supported_filesystems: List[str] = field(
        default_factory=lambda: ["xfs", "ext4", "nfs", "nfs4"]
    )

    # 当文件系统是 NFS 时，是否要求底层传输必须是 RDMA
    require_rdma_for_nfs: bool = True
    
    # 日志级别
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.gds_library_path is None:
            # 默认搜索路径
            self.gds_library_path = Path("/usr/local/cuda/lib64/libcufile.so")
