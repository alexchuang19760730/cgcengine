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
InferenceService - 推理服务，集成 GDS/SPDK 加速

这个服务展示了如何在实际推理场景中使用 GDS/SPDK：
1. 使用 GDS 零拷贝加载模型权重
2. 使用 SPDK 存储/读取 KV Cache
3. 自动降级机制
"""

import torch
import logging
from typing import Optional, Tuple, Dict, Any

from io_unified.unified_io_controller import UnifiedIOController, UnifiedIOConfig
from spdk_adapter.spdk_kv_store import SPDKKVStore
from spdk_adapter.spdk_config import SPDKConfig

logger = logging.getLogger(__name__)


class InferenceService:
    """
    推理服务类，集成 GDS/SPDK 加速
    
    功能：
    - 模型权重加载（GDS 零拷贝）
    - KV Cache 管理（SPDK 异步存储）
    - 会话管理
    """
    
    def __init__(self, config: Optional[UnifiedIOConfig] = None):
        """
        初始化推理服务
        
        Args:
            config: UnifiedIO 配置
        """
        # 初始化统一 IO 控制器
        self.io_controller = UnifiedIOController(config or UnifiedIOConfig())
        
        # 初始化 SPDK KV 存储（用于缓存历史对话）
        spdk_config = SPDKConfig(kv_store_path="/data/spdk_kv_cache", io_queues=8)
        self.kv_store = SPDKKVStore(spdk_config)
        self.kv_store.initialize()
        
        # 模型权重缓存
        self._model_weights: Dict[str, torch.Tensor] = {}
        
        logger.info(f"✅ 推理服务初始化完成")
        logger.info(f"   - 平台: {self.io_controller.platform_name}")
        logger.info(f"   - 活跃后端: {self.io_controller.name}")
        logger.info(f"   - GDS 可用: {'gds' in self.io_controller.backends}")
        logger.info(f"   - SPDK 可用: {'spdk' in self.io_controller.backends}")
    
    def load_model_weight(self, weight_path: str, shape: list, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        加载模型权重（使用 GDS 零拷贝）
        
        Args:
            weight_path: 权重文件路径
            shape: 权重张量形状
            dtype: 数据类型
            
        Returns:
            加载的权重张量（已在 GPU 上）
        """
        logger.debug(f"加载模型权重: {weight_path}, shape={shape}")
        
        # 使用统一 IO 控制器加载（自动选择 GDS 或降级方案）
        weight = self.io_controller.load_weight(weight_path, shape, dtype)
        
        # 缓存权重
        self._model_weights[weight_path] = weight
        
        return weight
    
    def load_history_kv(self, session_id: str, seq_len: int, head_dim: int, layer_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 SPDK 加载历史 KV Cache
        
        Args:
            session_id: 会话 ID
            seq_len: 序列长度
            head_dim: 头维度
            layer_id: 层 ID
            
        Returns:
            (k, v) 张量对
        """
        logger.debug(f"加载历史 KV Cache: session={session_id}, layer={layer_id}")
        
        # 使用统一 IO 控制器加载
        k, v = self.io_controller.load_kv(session_id, seq_len, head_dim)
        
        return k, v
    
    def save_history_kv(self, session_id: str, k: torch.Tensor, v: torch.Tensor, layer_id: int = 0) -> bool:
        """
        保存 KV Cache 到 SPDK
        
        Args:
            session_id: 会话 ID
            k: Key 张量
            v: Value 张量
            layer_id: 层 ID
            
        Returns:
            是否保存成功
        """
        logger.debug(f"保存 KV Cache: session={session_id}, layer={layer_id}")
        
        # 使用统一 IO 控制器保存
        success = self.io_controller.save_kv(session_id, k, v)
        
        return success
    
    def generate(self, session_id: str, prompt: str, max_tokens: int = 100) -> str:
        """
        生成响应（模拟推理流程）
        
        Args:
            session_id: 会话 ID
            prompt: 输入提示
            max_tokens: 最大生成长度
            
        Returns:
            生成的文本
        """
        logger.info(f"开始推理: session={session_id}, prompt={prompt[:30]}...")
        
        # 1. 加载历史 KV Cache（如果存在）
        try:
            k, v = self.load_history_kv(session_id, seq_len=128, head_dim=64)
            logger.debug(f"历史 KV 加载成功: k.shape={k.shape}, v.shape={v.shape}")
        except Exception as e:
            logger.debug(f"无历史 KV 或加载失败: {e}")
            k, v = None, None
        
        # 2. 模拟推理计算
        device = "cuda" if torch.cuda.is_available() else "cpu"
        new_k = torch.randn(1, 32, max_tokens, 64, device=device)
        new_v = torch.randn(1, 32, max_tokens, 64, device=device)
        
        # 3. 保存新的 KV Cache
        try:
            self.save_history_kv(session_id, new_k, new_v)
            logger.debug("新 KV Cache 保存成功")
        except Exception as e:
            logger.warning(f"KV Cache 保存失败: {e}")
        
        # 4. 模拟生成结果
        result = f"Response to: {prompt} [generated {max_tokens} tokens]"
        
        logger.info(f"推理完成: session={session_id}, length={len(result)}")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取服务统计信息
        
        Returns:
            统计字典
        """
        return {
            "io_stats": self.io_controller.get_stats().__dict__,
            "kv_stats": self.kv_store.get_stats(),
            "active_backend": self.io_controller.name,
            "platform": self.io_controller.platform_name,
            "num_weights_loaded": len(self._model_weights),
        }
    
    def shutdown(self):
        """
        关闭服务
        """
        logger.info("关闭推理服务...")
        self.io_controller.shutdown()
        logger.info("✅ 推理服务已关闭")


# 使用示例
def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建推理服务
    service = InferenceService()
    
    # 加载模型权重（模拟）
    logger.info("=== 测试权重加载 ===")
    try:
        weight = service.load_model_weight("/models/llama-7b/layer_0.weight", [4096, 4096])
        logger.info(f"权重加载成功: shape={weight.shape}, device={weight.device}")
    except Exception as e:
        logger.warning(f"权重加载失败（可能是测试环境）: {e}")
    
    # 测试推理
    logger.info("\n=== 测试推理 ===")
    result = service.generate("user_123", "Hello, how are you?")
    logger.info(f"生成结果: {result}")
    
    # 获取统计
    logger.info("\n=== 统计信息 ===")
    stats = service.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # 关闭服务
    service.shutdown()


if __name__ == "__main__":
    main()
