# Copyright (c) 2026 SandAI. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from cgc_engine.prefill_pool.prefill_pool import PrefillPool
from cgc_engine.gds_service.cufile_wrapper import is_gds_available


class CGCUnlimitedRSWAAttention(nn.Module):
    """
    R-SWA (Reference Sliding Window Attention) - 双层注意力机制
    
    核心设计：
    - Reference KV: 全局常驻，永不淘汰（来自 Prefill Pool）
    - Output KV: 滑动窗口，只保留最近 W 个 token
    
    显存复杂度: O(L + W)
    - L: 参考长度（可无限扩展，由 Prefill Pool 管理）
    - W: 滑动窗口大小（固定）
    """
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 128, init_projs: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        # 投影层
        self.init_projs = init_projs
        if init_projs:
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
        else:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None

        # Prefill Pool 实例
        self.prefill_pool = PrefillPool(
            max_hot_chunks=4,
            chunk_size=8192,
            storage_path="/tmp/cgc_prefill_pool",
            enable_gds=False, # CPU test environment disables GDS
        )

        # OrthoKDA 實例（內部 O(1) attention 計算引擎）
        self._ortho_kda = None
        self._ortho_kda_initialized = False
        
        # 缓存的 Output KV
        self.register_buffer('_past_k', None, persistent=False)
        self.register_buffer('_past_v', None, persistent=False)
        
        print(f"[R-SWA] 初始化完成")
        print(f"[R-SWA] 维度: {dim}, 头数: {num_heads}, 窗口: {window_size}")
        print(f"[R-SWA] GDS 直写: {'✅ 启用' if is_gds_available() else '❌ 禁用'}")
    
    def _reset_output_kv(self):
        """重置 Output KV 缓存"""
        self._past_k = None
        self._past_v = None

    def _init_ortho_kda(self, device):
        """延遲初始化 OrthoKDA（需要實際 head_dim 才能初始化）"""
        if self._ortho_kda_initialized:
            return
        try:
            import os
            from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4
            ortho_base_dim = int(os.environ.get("CGC_ORTHO_BASE_DIM", "128"))
            self._ortho_kda = OrthoKDAV4(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                ortho_base_dim=ortho_base_dim,
                use_cuda=(device.type == "cuda"),
            )
            self._ortho_kda_initialized = True
            print(f"[R-SWA] OrthoKDA 內部計算引擎已啟用 (heads={self.num_heads}, head_dim={self.head_dim}, base_dim={ortho_base_dim})")
        except Exception as e:
            print(f"[R-SWA] OrthoKDA 初始化失敗，回退到標準 attention: {e}")
            self._ortho_kda = None
            self._ortho_kda_initialized = True
    
    def add_reference_chunk(self, token_ids: torch.Tensor, ref_k: torch.Tensor, ref_v: torch.Tensor) -> str:
        """
        向 Prefill Pool 添加参考块
        
        Args:
            token_ids: (seq_len,) 或 (B, seq_len)
            ref_k: (B, num_heads, seq_len, head_dim)
            ref_v: (B, num_heads, seq_len, head_dim)
        
        Returns:
            chunk_id: 块 ID
        """
        if token_ids.dim() == 2:
            token_ids = token_ids.squeeze(0)
        
        chunk_id = self.prefill_pool.add_hot_chunk(
            token_ids=token_ids,
            ref_k=ref_k,
            ref_v=ref_v,
        )
        return chunk_id
    
    def get_all_reference_kv(self, device="cpu"):
        ref_k_list, ref_v_list = [], []
        # Get from hot chunks in prefill pool
        for chunk_id, chunk in self.prefill_pool.hot_chunks.items():
            ref_k_list.append(chunk.k.to(device))
            ref_v_list.append(chunk.v.to(device))
            
        if not ref_k_list:
            return None, None
            
        ref_k = torch.cat(ref_k_list, dim=2)
        ref_v = torch.cat(ref_v_list, dim=2)
        return ref_k, ref_v
    
    def forward(
        self,
        x: torch.Tensor,
        use_reference: bool = True,
        update_output_kv: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        R-SWA 前向传播
        
        Args:
            x: (B, T, C) 输入张量
            use_reference: 是否使用参考 KV
            update_output_kv: 是否更新 Output KV 缓存
        
        Returns:
            out: (B, T, C) 输出
            new_k: 新的 Output K
            new_v: 新的 Output V
        """
        B, T, C = x.shape

        # 延遲初始化 OrthoKDA
        self._init_ortho_kda(x.device)

        # 投影
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # 獲取參考 KV
        ref_k, ref_v = None, None
        if use_reference:
            ref_k, ref_v = self.get_all_reference_kv(device=x.device)

        # 拼接 Reference KV
        full_k = k
        full_v = v
        if ref_k is not None and ref_v is not None:
            full_k = torch.cat([ref_k, k], dim=2)  # (B, H, L + T, D)
            full_v = torch.cat([ref_v, v], dim=2)  # (B, H, L + T, D)

        # 拼接歷史 Output KV
        if self._past_k is not None and self._past_v is not None:
            full_k = torch.cat([self._past_k, full_k], dim=2)
            full_v = torch.cat([self._past_v, full_v], dim=2)

        # 更新 Output KV（只保留窗口大小）
        if update_output_kv:
            new_k = full_k[:, :, -self.window_size:]
            new_v = full_v[:, :, -self.window_size:]
            self._past_k = new_k.detach()
            self._past_v = new_v.detach()
        else:
            new_k = full_k[:, :, -self.window_size:]
            new_v = full_v[:, :, -self.window_size:]

        # === Attention 計算：優先用 OrthoKDA（O(1) 內存），回退到 SDPA ===
        if self._ortho_kda is not None:
            # OrthoKDA 路徑：逐 token update KV，然後 forward(Q)
            # 更新 KV 到 OrthoKDA（full_k/full_v shape: [B, H, L+T, D]）
            seq_len = full_k.size(2)
            for t in range(seq_len):
                # 取每個位置的 KV: [H, D]
                k_t = full_k[0, :, t, :]  # [H, D]
                v_t = full_v[0, :, t, :]  # [H, D]
                self._ortho_kda.update(k_t, v_t)

            # Q shape: [B, H, T, D] -> [B, H, T, D] for OrthoKDA
            # OrthoKDA.forward 期望 [batch, num_heads, head_dim] 或 [num_heads, head_dim]
            # 逐 token 計算
            out_list = []
            for t in range(T):
                q_t = q[:, :, t, :]  # [B, H, D]
                out_t = self._ortho_kda.forward(q_t)  # [B, H, D] 或 [H, D]
                if out_t.dim() == 2:
                    out_t = out_t.unsqueeze(0)  # [1, H, D]
                out_list.append(out_t)
            attn = torch.stack(out_list, dim=2)  # [B, H, T, D]
            attn = attn.transpose(1, 2).reshape(B, T, C)
        else:
            # 回退路徑：標準 scaled_dot_product_attention
            full_len = full_k.size(2)
            output_start = full_len - T
            ref_len = ref_k.size(2) if ref_k is not None else 0

            attn_mask = torch.ones((T, full_len), device=x.device, dtype=torch.bool)
            for i in range(T):
                pos = output_start + i
                l_bound = max(ref_len, pos - self.window_size)
                attn_mask[i, :l_bound] = False

            attn = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=attn_mask)
            attn = attn.transpose(1, 2).reshape(B, T, C)

        out = self.out_proj(attn)

        return out, new_k, new_v
    
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_len: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        生成模式
        
        Args:
            start_tokens: (B, seq_len) 起始 token
            max_len: 最大生成长度
            temperature: 温度系数
        
        Returns:
            generated: (B, seq_len + max_len) 完整序列
        """
        self._reset_output_kv()
        
        device = start_tokens.device
        generated = [start_tokens]
        current = start_tokens
        
        for _ in range(max_len):
            # 前向传播
            x = self._get_embedding(current)  # 假设存在 embedding 层
            out, _, _ = self.forward(x, use_reference=True, update_output_kv=True)
            
            # 取最后一个 token 的 logits
            logits = self._get_logits(out[:, -1:])  # 假设存在 LM head
            
            # 采样
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            generated.append(next_token)
            current = next_token
        
        return torch.cat(generated, dim=1)
    
    def _get_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """占位：实际实现应从模型获取 embedding"""
        return torch.randn(tokens.shape[0], tokens.shape[1], self.dim, device=tokens.device)
    
    def _get_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """占位：实际实现应从模型获取 logits"""
        return torch.randn(hidden.shape[0], hidden.shape[1], 32000, device=hidden.device)
    
    def get_pool_info(self) -> Dict:
        """获取 Prefill Pool 状态"""
        pool_status = self.prefill_pool.get_pool_status()
        return {
            "window_size": self.window_size,
            "pool_status": pool_status
        }
    
    def clear_pool(self):
        """清空 Prefill Pool"""
        self.prefill_pool.clear()


class RSWAPrefillPoolEngine:
    """
    R-SWA + Prefill Pool 推理引擎
    
    端到端的无限上下文推理解决方案
    """
    
    def __init__(
        self,
        dim: int = 4096,
        num_heads: int = 32,
        window_size: int = 128,
        max_hot_chunks: int = 4,
        chunk_size: int = 8192,
    ):
        self.attention = CGCUnlimitedRSWAAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.chunk_size = chunk_size
        
        # 移动到 GPU（如果可用）
        if torch.cuda.is_available():
            self.attention = self.attention.cuda()
    
    def prefill_reference(self, reference_text: List[str]) -> List[str]:
        """
        Prefill 参考文本
        
        Args:
            reference_text: 参考文本列表
        
        Returns:
            chunk_ids: 块 ID 列表
        """
        chunk_ids = []
        
        for text in reference_text:
            # 模拟 tokenize
            token_ids = self._tokenize(text)
            
            # 分块
            for i in range(0, len(token_ids), self.chunk_size):
                chunk_tokens = token_ids[i:i + self.chunk_size]
                chunk_tokens = torch.tensor(chunk_tokens, device=self._get_device())
                
                # 模拟 Prefill 生成 KV
                batch_size = 1
                seq_len = len(chunk_tokens)
                ref_k = torch.randn(batch_size, self.attention.num_heads, seq_len, self.attention.head_dim,
                                   device=self._get_device(), dtype=torch.bfloat16)
                ref_v = torch.randn(batch_size, self.attention.num_heads, seq_len, self.attention.head_dim,
                                   device=self._get_device(), dtype=torch.bfloat16)
                
                # 添加到 Prefill Pool
                chunk_id = self.attention.add_reference_chunk(chunk_tokens, ref_k, ref_v)
                chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    def infer(self, query: str, max_tokens: int = 100) -> str:
        """
        执行推理
        
        Args:
            query: 查询文本
            max_tokens: 最大生成 token 数
        
        Returns:
            response: 响应文本
        """
        # 模拟 tokenize
        query_tokens = self._tokenize(query)
        query_tensor = torch.tensor([query_tokens], device=self._get_device())
        
        # 生成响应
        result = self.attention.generate(query_tensor, max_len=max_tokens)
        
        # 模拟 detokenize
        response = self._detokenize(result[0].cpu().tolist())
        
        return response
    
    def _tokenize(self, text: str) -> List[int]:
        """占位：实际实现应使用真实 tokenizer"""
        return [1] * min(len(text) * 2, 8192)
    
    def _detokenize(self, tokens: List[int]) -> str:
        """占位：实际实现应使用真实 tokenizer"""
        return " ".join(str(t) for t in tokens)
    
    def _get_device(self) -> torch.device:
        """获取当前设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def info(self) -> Dict:
        """获取引擎状态"""
        return {
            "dim": self.attention.dim,
            "num_heads": self.attention.num_heads,
            "window_size": self.attention.window_size,
            "chunk_size": self.chunk_size,
            "gds_enabled": is_gds_available(),
            "pool_info": self.attention.get_pool_info(),
        }
