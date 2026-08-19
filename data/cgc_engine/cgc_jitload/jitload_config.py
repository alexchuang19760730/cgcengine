
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
class JITLoadConfig:
    """JITLoad 配置"""
    
    # 是否启用 JITLoad
    enable_jitload: bool = True
    
    # 缓存目录
    cache_dir: Path = field(default_factory=lambda: Path("/tmp/magi_cache"))
    
    # CGC 指令缓存大小
    cgc_cache_size: int = 10000
    
    # 是否启用 AOT 预编译
    enable_aot: bool = True
    
    # AOT 预编译模型标签列表
    aot_model_tags: List[str] = field(default_factory=lambda: [])
    
    # 是否使用 SPDK 存储缓存
    use_spdk_cache: bool = False
    
    # 是否使用 GDS 加载缓存
    use_gds_load: bool = False
    
    # 缓存版本（用于失效）
    cache_version: int = 1
    
    # 缓存策略
    cache_policy: str = "lru"  # "lru", "fifo", "lfu"
    
    # 最大缓存磁盘大小 (MB)
    max_cache_size_mb: int = 10240
    
    # 日志级别
    log_level: str = "INFO"
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

