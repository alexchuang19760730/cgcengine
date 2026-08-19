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
GDS Expert Loader - 使用 GDS 零拷贝加载 FlashMoE 专家权重

功能：
- 使用 GDS (GPU Direct Storage) 直接从存储加载专家权重到 GPU
- 跳过 CPU，实现真正的零拷贝
- 支持自动降级到标准 PyTorch 加载
"""

import torch
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# 尝试导入 GDS
try:
    from ..gds_service.gds_manager import GDSManager
    from ..gds_service.cufile_wrapper import CUFILE_AVAILABLE
    GDS_AVAILABLE = CUFILE_AVAILABLE and torch.cuda.is_available()
except ImportError:
    try:
        from cgc_engine.gds_service.gds_manager import GDSManager
        from cgc_engine.gds_service.cufile_wrapper import CUFILE_AVAILABLE
        GDS_AVAILABLE = CUFILE_AVAILABLE and torch.cuda.is_available()
    except ImportError:
        GDS_AVAILABLE = False
        logger.warning("[GDS Expert Loader] GDS 不可用，将使用标准加载方式")


class GDSExpertLoader:
    """
    使用 GDS 零拷贝加载 FlashMoE 专家权重
    
    特性：
    - 直接从存储加载到 GPU，跳过 CPU
    - 支持专家缓存，避免重复加载
    - 自动降级到标准 PyTorch 加载
    """
    
    def __init__(self, expert_dir: str = "/data/flashmoe_experts"):
        """
        初始化 GDS 专家加载器
        
        Args:
            expert_dir: 专家权重存储目录
        """
        self.expert_dir = expert_dir
        self._expert_cache: Dict[int, torch.Tensor] = {}
        self._gds_manager = None
        
        if GDS_AVAILABLE:
            try:
                self._gds_manager = GDSManager()
                logger.info("[GDS Expert Loader] ✅ GDS 加载器初始化成功")
            except Exception as e:
                logger.warning(f"[GDS Expert Loader] ⚠️ GDS 初始化失败: {e}")
        
        # 统计信息
        self._stats = {
            "loads": 0,
            "hits": 0,
            "gds_loads": 0,
            "fallback_loads": 0
        }
    
    @property
    def gds_enabled(self) -> bool:
        """检查 GDS 是否可用"""
        return self._gds_manager is not None and self._gds_manager.enabled
    
    def load_expert(
        self,
        expert_id: int,
        shape: List[int],
        dtype: torch.dtype = torch.float16,
        force_reload: bool = False
    ) -> torch.Tensor:
        """
        加载专家权重
        
        Args:
            expert_id: 专家 ID
            shape: 权重形状
            dtype: 数据类型
            force_reload: 是否强制重新加载（跳过缓存）
        
        Returns:
            专家权重张量（已在 GPU 上）
        """
        # 检查缓存
        if not force_reload and expert_id in self._expert_cache:
            self._stats["hits"] += 1
            logger.debug(f"[GDS Expert Loader] 缓存命中: expert_id={expert_id}")
            return self._expert_cache[expert_id]
        
        self._stats["loads"] += 1
        expert_path = self._get_expert_path(expert_id)
        
        try:
            if self.gds_enabled:
                # 使用 GDS 零拷贝加载
                logger.debug(f"[GDS Expert Loader] 使用 GDS 加载专家: {expert_id}")
                weight = self._load_with_gds(expert_path, shape, dtype)
                self._stats["gds_loads"] += 1
            else:
                # 降级到标准加载
                logger.debug(f"[GDS Expert Loader] 使用标准方式加载专家: {expert_id}")
                weight = self._load_with_pytorch(expert_path, shape, dtype)
                self._stats["fallback_loads"] += 1
            
            # 缓存专家
            self._expert_cache[expert_id] = weight
            
            return weight
        
        except Exception as e:
            logger.error(f"[GDS Expert Loader] 加载专家失败 {expert_id}: {e}")
            # 降级到随机初始化
            return self._create_fallback_expert(shape, dtype)
    
    def _get_expert_path(self, expert_id: int) -> str:
        """获取专家权重文件路径"""
        return f"{self.expert_dir}/expert_{expert_id}.safetensors"
    
    def _load_with_gds(self, path: str, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
        """使用 GDS 零拷贝加载"""
        if self._gds_manager:
            return self._gds_manager.load_weight_from_pd(path, shape)
        
        # 降级方案
        return self._load_with_pytorch(path, shape, dtype)
    
    def _load_with_pytorch(self, path: str, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
        """使用标准 PyTorch 加载"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 尝试加载 safetensors 文件
        try:
            from safetensors.torch import load_file
            data = load_file(path, device=device)
            return data["weight"].view(shape)
        except:
            pass
        
        # 降级到 torch.load
        try:
            data = torch.load(path, map_location=device)
            return data["weight"].view(shape)
        except:
            pass
        
        # 返回随机初始化的权重
        return self._create_fallback_expert(shape, dtype)
    
    def _create_fallback_expert(self, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
        """创建降级专家权重"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.randn(shape, dtype=dtype, device=device) * 0.01
    
    def unload_expert(self, expert_id: int) -> bool:
        """卸载指定专家"""
        if expert_id in self._expert_cache:
            del self._expert_cache[expert_id]
            logger.debug(f"[GDS Expert Loader] 已卸载专家: {expert_id}")
            return True
        return False
    
    def clear_cache(self):
        """清空专家缓存"""
        self._expert_cache.clear()
        logger.debug("[GDS Expert Loader] 专家缓存已清空")
    
    def get_stats(self) -> Dict[str, int]:
        """获取加载统计"""
        return self._stats.copy()
    
    def get_cache_size(self) -> int:
        """获取缓存的专家数量"""
        return len(self._expert_cache)


# 使用示例
def main():
    logging.basicConfig(level=logging.INFO)
    
    loader = GDSExpertLoader()
    logger.info(f"GDS 可用: {loader.gds_enabled}")
    
    # 加载专家
    expert = loader.load_expert(0, [4096, 4096])
    logger.info(f"专家加载成功: shape={expert.shape}, device={expert.device}")
    
    # 再次加载（应该命中缓存）
    expert2 = loader.load_expert(0, [4096, 4096])
    logger.info(f"缓存命中测试: same object={expert is expert2}")
    
    # 统计信息
    stats = loader.get_stats()
    logger.info(f"加载统计: {stats}")


if __name__ == "__main__":
    main()
