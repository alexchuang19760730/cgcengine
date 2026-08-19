#!/usr/bin/env python3
"""
CGC Engine Storage Layer - Complete Integration
Integrates: KDA, SPDK, GDS, TurboQuant
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from collections import OrderedDict
import threading

# 核心 I/O 控制器
from cgc_engine.io_unified.unified_io_controller import UnifiedIOController, UnifiedIOConfig

# KDA 集成
try:
    from cgc_engine.cgc.kda_pass import KDAPass
    from cgc_engine.cgc.flashkda_integration import FlashKDAIntegration
    HAS_KDA = True
except Exception:
    HAS_KDA = False

# TurboQuant 量化支持
try:
    from cgc_engine.quantization.turboquant import TurboQuant
    HAS_TURBOQUANT = True
except ImportError:
    HAS_TURBOQUANT = False

class ExpertCacheManager:
    """专家缓存管理器 - 完整集成版"""
    
    def __init__(self, max_size: int = 8, enable_kda: bool = True):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.access_order = []
        
        # 集成 UnifiedIOController
        self.io_controller = UnifiedIOController.get_instance()
        
        # KDA 集成
        self.enable_kda = bool(enable_kda and HAS_KDA)
        if self.enable_kda:
            self.kda_integration = FlashKDAIntegration()
            self.kda_pass = KDAPass()
        
        # TurboQuant 集成
        if HAS_TURBOQUANT:
            self.turboquant = TurboQuant()
        
        self._lock = threading.Lock()
    
    def get(self, expert_id: int) -> Optional[Any]:
        if expert_id in self.cache:
            self._update_access(expert_id)
            return self.cache[expert_id]
        return None
    
    def set(self, expert_id: int, weight: Any):
        if expert_id in self.cache:
            self.cache[expert_id] = weight
            self._update_access(expert_id)
        else:
            if len(self.cache) >= self.max_size:
                self.evict_oldest()
            self.cache[expert_id] = weight
            self.access_order.append(expert_id)
    
    def _update_access(self, expert_id: int):
        if expert_id in self.access_order:
            self.access_order.remove(expert_id)
        self.access_order.append(expert_id)
    
    def evict_oldest(self) -> Optional[int]:
        if not self.access_order:
            return None
        oldest = self.access_order.pop(0)
        if oldest in self.cache:
            del self.cache[oldest]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return oldest
        return None
    
    def evict_specific(self, expert_id: int) -> bool:
        if expert_id in self.cache:
            del self.cache[expert_id]
            if expert_id in self.access_order:
                self.access_order.remove(expert_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False
    
    def contains(self, expert_id: int) -> bool:
        return expert_id in self.cache
    
    def keys(self) -> List[int]:
        return list(self.cache.keys())
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========== KDA 集成方法 ==========
    
    def apply_kda_optimization(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """
        对专家输出应用 KDA 优化
        """
        if not self.enable_kda or self.kda_integration is None:
            return x
        
        return self.kda_integration.apply_kda(x, expert_ids)
    
    def optimize_with_kda_pass(self, model):
        """
        对模型应用 KDA Pass
        """
        if not self.enable_kda or self.kda_pass is None:
            return model
        
        return self.kda_pass.apply(model)
    
    # ========== TurboQuant 集成方法 ==========
    
    def quantize_expert(self, expert_id: int, bits: int = 4) -> bool:
        """
        使用 TurboQuant 量化专家权重
        """
        if not HAS_TURBOQUANT or expert_id not in self.cache:
            return False
        
        weight = self.cache[expert_id]
        if isinstance(weight, dict):
            quantized_weight = {k: self.turboquant.quantize(v, bits=bits) for k, v in weight.items()}
        else:
            quantized_weight = self.turboquant.quantize(weight, bits=bits)
        self.cache[expert_id] = quantized_weight
        return True
    
    def dequantize_expert(self, expert_id: int) -> bool:
        """
        反量化专家权重
        """
        if not HAS_TURBOQUANT or expert_id not in self.cache:
            return False
        
        weight = self.cache[expert_id]
        if isinstance(weight, dict):
            dequantized_weight = {k: self.turboquant.dequantize(v) for k, v in weight.items()}
        else:
            dequantized_weight = self.turboquant.dequantize(weight)
        self.cache[expert_id] = dequantized_weight
        return True
    
    # ========== UnifiedIOController 集成方法 ==========
    
    def load_expert_from_storage(self, expert_id: int, path: str) -> torch.Tensor:
        expert = self.io_controller.load_expert(expert_id, path)
        self.set(expert_id, expert)
        return expert
    
    def save_expert_to_storage(self, expert_id: int, tensor: torch.Tensor) -> bool:
        success = self.io_controller.save_expert(expert_id, tensor)
        if success:
            self.set(expert_id, tensor)
        return success

class ExpertLoader:
    """专家加载器 - 完整功能版"""
    
    def __init__(self, expert_dir: str = "/home/gs01/models", expert_dim: int = 4096, intermediate_dim: int = 14336):
        self.expert_dir = expert_dir
        self.io_controller = UnifiedIOController.get_instance()
        self.expert_dim = expert_dim
        self.intermediate_dim = intermediate_dim
    
    def load_expert(self, expert_id: int) -> Dict[str, torch.Tensor]:
        expert_dim = int(self.expert_dim)
        intermediate_dim = int(self.intermediate_dim)
        base = self.get_expert_path(expert_id)
        return self.io_controller.load_expert_mlp(
            expert_id=expert_id,
            base_path=base,
            expert_dim=expert_dim,
            intermediate_dim=intermediate_dim,
            dtype=torch.float16,
        )
    
    def load_multiple(self, expert_ids: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
        weights = {}
        for eid in expert_ids:
            weights[eid] = self.load_expert(eid)
        return weights
    
    def get_expert_path(self, expert_id: int) -> str:
        return f"{self.expert_dir}/expert_{expert_id}.pt"
    
    def load_weight_with_gds(self, path: str, shape: List[int], dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """使用 GDS 零拷贝加载"""
        return self.io_controller.load_weight(path, shape, dtype)

class KVCacheManager:
    """KV Cache 管理器 - 完整功能版"""
    
    def __init__(self):
        self.io_controller = UnifiedIOController.get_instance()
    
    def load_kv(self, key: str, seq_len: int, head_dim: int, num_heads: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.io_controller.load_kv(key, seq_len, head_dim, num_heads)
    
    def save_kv(self, key: str, k: torch.Tensor, v: torch.Tensor) -> bool:
        return self.io_controller.save_kv(key, k, v)
    
    def prefetch_kv(self, keys: List[str]) -> None:
        self.io_controller.prefetch(keys)
    
    def evict_kv(self, keys: List[str]) -> None:
        self.io_controller.evict(keys)
    
    def get_io_stats(self):
        return self.io_controller.get_stats()

class StorageLayer:
    """统一存储层 - 整合所有功能"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        self.expert_cache = ExpertCacheManager(
            max_size=self.config.get('max_cached_experts', 8),
            enable_kda=self.config.get('enable_kda', True)
        )
        
        self.expert_loader = ExpertLoader(
            expert_dir=self.config.get('expert_dir', '/home/gs01/models')
        )
        
        self.kv_cache = KVCacheManager()
    
    @property
    def backend_name(self):
        return self.expert_cache.io_controller.name
    
    @property
    def platform(self):
        return self.expert_cache.io_controller.platform_name
    
    def get_all_stats(self):
        """获取所有统计信息"""
        io_stats = self.kv_cache.get_io_stats()
        return {
            'backend': self.backend_name,
            'platform': self.platform,
            'cache_size': len(self.expert_cache),
            'io_stats': io_stats.__dict__ if io_stats else None,
            'kda_enabled': self.expert_cache.enable_kda,
            'turboquant_enabled': HAS_TURBOQUANT
        }
