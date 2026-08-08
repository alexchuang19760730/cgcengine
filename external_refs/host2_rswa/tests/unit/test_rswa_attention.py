# Copyright (c) 2026 SandAI. All Rights Reserved.

"""
R-SWA Attention 单元测试
"""

import torch
import pytest
import sys
sys.path.insert(0, sys.path[0] + '/../..')

from cgc_engine.rswa_integration import CGCUnlimitedRSWAAttention


class TestRSWAAttention:
    """R-SWA 注意力机制单元测试"""
    
    @pytest.fixture
    def attention(self):
        """创建注意力模块实例"""
        return CGCUnlimitedRSWAAttention(
            dim=512,
            num_heads=8,
            window_size=32
        )
    
    @pytest.fixture
    def device(self):
        """获取测试设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_init(self, attention):
        """测试初始化"""
        assert attention.dim == 512
        assert attention.num_heads == 8
        assert attention.head_dim == 64
        assert attention.window_size == 32
        assert attention.q_proj is not None
        assert attention.k_proj is not None
        assert attention.v_proj is not None
        assert attention.out_proj is not None
    
    def test_forward_shape(self, attention, device):
        """测试前向传播形状"""
        attention = attention.to(device)
        batch_size, seq_len = 2, 128
        
        x = torch.randn(batch_size, seq_len, 512, device=device)
        out, new_k, new_v = attention(x, use_reference=False)
        
        assert out.shape == (batch_size, seq_len, 512)
        assert new_k.shape == (batch_size, 8, 32, 64)
        assert new_v.shape == (batch_size, 8, 32, 64)
    
    def test_forward_with_reference(self, attention, device):
        """测试带参考 KV 的前向传播"""
        attention = attention.to(device)
        batch_size, seq_len = 2, 64
        ref_len = 256
        
        x = torch.randn(batch_size, seq_len, 512, device=device)
        ref_k = torch.randn(batch_size, 8, ref_len, 64, device=device)
        ref_v = torch.randn(batch_size, 8, ref_len, 64, device=device)
        ref_token_ids = torch.arange(ref_len, device=device)
        
        # 添加参考块
        chunk_id = attention.add_reference_chunk(ref_token_ids, ref_k, ref_v)
        assert chunk_id is not None
        
        # 前向传播
        out, new_k, new_v = attention(x, use_reference=True)
        
        assert out.shape == (batch_size, seq_len, 512)
        assert new_k.shape == (batch_size, 8, 32, 64)
    
    def test_sliding_window_mask(self, attention, device):
        """测试滑动窗口掩码"""
        attention = attention.to(device)
        attention._reset_output_kv()
        
        batch_size, seq_len = 1, 100
        
        for i in range(5):
            x = torch.randn(batch_size, seq_len, 512, device=device)
            out, new_k, new_v = attention(x, use_reference=False, update_output_kv=True)
        
        # 验证历史 KV 大小
        assert attention._past_k is not None
        assert attention._past_k.shape[2] == attention.window_size
    
    def test_reference_kv_preservation(self, attention, device):
        """测试参考 KV 保留"""
        attention = attention.to(device)
        
        # 添加参考块
        ref_len = 1024
        ref_k = torch.randn(1, 8, ref_len, 64, device=device)
        ref_v = torch.randn(1, 8, ref_len, 64, device=device)
        ref_token_ids = torch.arange(ref_len, device=device)
        chunk_id = attention.add_reference_chunk(ref_token_ids, ref_k, ref_v)
        
        # 多次推理
        for _ in range(10):
            x = torch.randn(1, 64, 512, device=device)
            out, _, _ = attention(x, use_reference=True)
        
        # 验证参考块仍然存在
        pool_info = attention.get_pool_info()
        assert pool_info["hot_chunks"] >= 1
    
    def test_device_migration(self, attention):
        """测试设备迁移"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        attention = attention.cuda()
        assert next(attention.parameters()).device.type == "cuda"
        
        attention = attention.cpu()
        assert next(attention.parameters()).device.type == "cpu"
    
    def test_reset_output_kv(self, attention, device):
        """测试重置输出 KV"""
        attention = attention.to(device)
        
        # 执行几次推理
        for _ in range(3):
            x = torch.randn(1, 64, 512, device=device)
            attention(x, use_reference=False, update_output_kv=True)
        
        assert attention._past_k is not None
        
        # 重置
        attention._reset_output_kv()
        assert attention._past_k is None
        assert attention._past_v is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])