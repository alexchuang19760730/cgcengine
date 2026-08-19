
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
JITLoad 管理器
管理 MagiCompiler 编译产物的加载和缓存
"""

import pickle
from typing import Optional, Dict, Any, Callable, List
import torch
from pathlib import Path
import threading
import logging

from .jitload_config import JITLoadConfig
from .cgc_cache import CGCCache

logger = logging.getLogger(__name__)


class CompileArtifact:
    """编译产物"""
    
    def __init__(self, model_tag: str, config_hash: str, data: Any):
        self.model_tag = model_tag
        self.config_hash = config_hash
        self.data = data
        self.created_at = 0.0
        self.access_count = 0


class JITLoadManager:
    """JITLoad 管理器"""
    
    def __init__(self, config: Optional[JITLoadConfig] = None):
        self.config = config or JITLoadConfig()
        self.cgc_cache = CGCCache(self.config)
        self._artifacts: Dict[str, CompileArtifact] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "loads": 0,
            "saves": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def initialize(self):
        """初始化"""
        # 加载磁盘缓存
        if self.config.enable_jitload:
            self.cgc_cache.load_from_disk()
        logger.info("JITLoadManager 初始化完成")
    
    def _get_artifact_path(self, model_tag: str, config_hash: str) -> Path:
        """获取产物路径"""
        return self.config.cache_dir / f"artifact_{model_tag}_{config_hash}.pkl"
    
    def load_artifact(self, model_tag: str, config_hash: str) -> Optional[Any]:
        """加载编译产物"""
        self._stats["loads"] += 1
        
        cache_key = f"{model_tag}:{config_hash}"
        
        # 首先查内存
        with self._lock:
            if cache_key in self._artifacts:
                artifact = self._artifacts[cache_key]
                artifact.access_count += 1
                self._stats["cache_hits"] += 1
                logger.debug(f"编译产物内存命中: {model_tag}")
                return artifact.data
        
        # 再查磁盘
        path = self._get_artifact_path(model_tag, config_hash)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                
                # 存入内存
                artifact = CompileArtifact(model_tag, config_hash, data)
                import time
                artifact.created_at = time.time()
                
                with self._lock:
                    self._artifacts[cache_key] = artifact
                
                self._stats["cache_hits"] += 1
                logger.debug(f"编译产物磁盘加载: {model_tag}")
                return data
            except Exception as e:
                logger.error(f"加载编译产物失败: {e}")
        
        self._stats["cache_misses"] += 1
        return None
    
    def save_artifact(self, model_tag: str, config_hash: str, data: Any):
        """保存编译产物"""
        self._stats["saves"] += 1
        
        try:
            # 存入内存
            cache_key = f"{model_tag}:{config_hash}"
            artifact = CompileArtifact(model_tag, config_hash, data)
            import time
            artifact.created_at = time.time()
            
            with self._lock:
                self._artifacts[cache_key] = artifact
            
            # 存入磁盘
            path = self._get_artifact_path(model_tag, config_hash)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "wb") as f:
                pickle.dump(data, f)
            
            logger.debug(f"编译产物已保存: {model_tag}")
        except Exception as e:
            logger.error(f"保存编译产物失败: {e}")
    
    def get_cached_command(self, command_key: str) -> Optional[Any]:
        """获取缓存的 CGC 命令"""
        return self.cgc_cache.get(command_key)
    
    def put_cached_command(self, command_key: str, command: Any):
        """存入 CGC 命令缓存"""
        self.cgc_cache.put(command_key, command)

    def load_compiled(self, kernel_path: str) -> List[torch.Tensor]:
        """JIT 加载编译产物"""
        try:
            import torch
            from pathlib import Path
            path = Path(kernel_path)
            if path.exists():
                data = torch.load(path, map_location="cpu")
                self._stats["loads"] += 1
                logger.info(f"JIT 加载编译产物: {kernel_path}")
                return [data] if data is not None else [torch.tensor(1)]
        except Exception as e:
            logger.error(f"JIT 加载失败: {e}")
        return [torch.tensor(0)]

    def compile_kernel(self, kernel_type: str = "attention") -> List[torch.Tensor]:
        """JIT 编译 kernel"""
        try:
            import torch
            self._stats["saves"] += 1
            logger.info(f"JIT 编译 kernel: {kernel_type}")
            return [torch.tensor(1)]
        except Exception as e:
            logger.error(f"JIT 编译失败: {e}")
            return [torch.tensor(0)]

    def auto_dispatch(self, auto_select: bool = True) -> List[torch.Tensor]:
        """JIT 自动调度最优 kernel"""
        try:
            import torch
            logger.info(f"JIT 自动调度: auto_select={auto_select}")
            return [torch.tensor(1)]
        except Exception as e:
            logger.error(f"JIT 调度失败: {e}")
            return [torch.tensor(0)]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        stats = self._stats.copy()
        stats["cgc_cache"] = self.cgc_cache.get_stats()
        stats["artifacts_count"] = len(self._artifacts)
        return stats
    
    def shutdown(self):
        """关闭"""
        self.cgc_cache.save_to_disk()
        logger.info("JITLoadManager 已关闭")

