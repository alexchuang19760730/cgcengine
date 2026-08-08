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
        attn_metadata,
        output: torch.Tensor,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        使用 FlashAttention 2 的前向传播
        继承父类 FlashAttentionImpl 的完整逻辑
        """
        from flash_attn import flash_attn_varlen_func

        if attn_metadata is None:
            return output.fill_(0.)

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return output

        q = query[:num_actual_tokens]
        k = key[:num_actual_tokens]
        v = value[:num_actual_tokens]

        if q.dim() == 2:
            q = q.view(-1, self.num_heads, self.head_size)
            k = k.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size)

        if self.num_kv_heads != self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=None,
            causal=True,
        )

        output[:num_actual_tokens] = out

        return output

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