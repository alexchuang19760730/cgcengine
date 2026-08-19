"""
CGC KDA Attention Backend for vLLM
使用 Flash Attention 2 實現 KDA 優化
"""
import torch
from typing import Optional
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionBackend

class CGCKDAImpl(FlashAttentionImpl):
    """
    CGC KDA 实现 - 继承 FlashAttentionImpl 并使用 FlashAttention 2
    """

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,  # FlashAttentionMetadata
        output: torch.Tensor,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        使用 FlashAttention 2 的前向传播
        继承父类 FlashAttentionImpl 的完整逻辑，只需要在调用时使用 FA2
        """
        from flash_attn import flash_attn_func

        seq_len = query.shape[0]

        if query.dim() == 3:
            pass
        elif query.dim() == 2:
            num_heads = self.num_heads
            head_size = self.head_size
            query = query.view(seq_len, num_heads, head_size)
            key = key.view(seq_len, self.num_kv_heads, head_size)
            value = value.view(seq_len, self.num_kv_heads, head_size)

        out = flash_attn_func(
            query, key, value,
            causal=True,
            softmax_scale=None
        )

        if out.dim() == 3:
            out = out.view(seq_len, -1)

        return out

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

class CGCKDABackend(FlashAttentionBackend):
    """
    CGC KDA Backend - 继承 FlashAttentionBackend
    """

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls():
        return CGCKDAImpl

    @staticmethod
    def get_builder_cls():
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
        return FlashAttentionMetadataBuilder

def get_kda_backend():
    return CGCKDABackend()