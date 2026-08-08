# Copyright (c) 2026 SandAI. All Rights Reserved.

"""
R-SWA + Prefill Pool 集成测试
"""

import torch
import pytest
import sys
sys.path.insert(0, sys.path[0] + '/../..')

from cgc_engine.rswa_integration import CGCUnlimitedRSWAAttention, RSWAPrefillPoolEngine
from cgc_engine.prefill_pool import PrefillPool


class TestRSWAPrefillPoolIntegration:
    """R-SWA 与 Prefill Pool 集成测试"""
    
    @pytest.fixture
    def device(self):
        """获取测试设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_rswa_with_prefill_pool(self, device):
        """测试 R-SWA 与 Prefill Pool 协同工作"""
        # 初始化注意力模块
        attention = CGCUnlimitedRSWAAttention(
            dim=512,
            num_heads=8,
            window_size=32
        ).to(device)
        
        # 添加参考块到 Prefill Pool
        ref_len = 1024
        ref_k = torch.randn(1, 8, ref_len, 64, device=device)
        ref_v = torch.randn(1, 8, ref_len, 64, device=device)
        ref_token_ids = torch.arange(ref_len, device=device)
        
        chunk_id = attention.add_reference_chunk(ref_token_ids, ref_k, ref_v)
        assert chunk_id is not None
        
        # 验证 Prefill Pool 状态
        pool_info = attention.get_pool_info()
        assert pool_info["hot_chunks"] == 1
        assert pool_info["hot_tokens"] == ref_len
        
        # 执行推理
        batch_size, seq_len = 1, 64
        x = torch.randn(batch_size, seq_len, 512, device=device)
        out, new_k, new_v = attention(x, use_reference=True)
        
        # 验证输出
        assert out.shape == (batch_size, seq_len, 512)
        assert new_k.shape == (batch_size, 8, 32, 64)
        
        # 验证参考块仍然可用
        pool_info = attention.get_pool_info()
        assert pool_info["hot_chunks"] == 1
    
    def test_engine_initialization(self, device):
        """测试推理引擎初始化"""
        engine = RSWAPrefillPoolEngine(
            dim=512,
            num_heads=8,
            window_size=32,
            max_hot_chunks=2,
            chunk_size=1024,
        )
        
        # 验证引擎配置
        info = engine.info()
        assert info["dim"] == 512
        assert info["num_heads"] == 8
        assert info["window_size"] == 32
        assert info["chunk_size"] == 1024
        assert "pool_info" in info
    
    def test_prefill_reference(self, device):
        """测试预填充参考文本"""
        engine = RSWAPrefillPoolEngine(
            dim=512,
            num_heads=8,
            window_size=32,
            max_hot_chunks=4,
            chunk_size=1024,
        )
        
        # 添加参考文本
        reference_texts = [
            "法律知识: 中华人民共和国刑法第三百零二条...",
            "技术知识: GPU Direct Storage 是 NVIDIA 的技术...",
            "历史知识: 唐朝是中国历史上强盛的朝代...",
        ]
        
        chunk_ids = engine.prefill_reference(reference_texts)
        
        # 验证块被添加
        assert len(chunk_ids) > 0
        pool_info = engine.attention.get_pool_info()
        assert pool_info["hot_chunks"] == len(chunk_ids)
    
    def test_inference_with_reference(self, device):
        """测试带参考上下文的推理"""
        engine = RSWAPrefillPoolEngine(
            dim=512,
            num_heads=8,
            window_size=32,
            max_hot_chunks=4,
            chunk_size=1024,
        )
        
        # 添加参考知识
        reference_texts = [
            "GDS 是 GPU Direct Storage 的缩写，是 NVIDIA 公司开发的技术...",
        ]
        engine.prefill_reference(reference_texts)
        
        # 执行推理
        response = engine.infer("什么是 GDS 技术？", max_tokens=50)
        
        # 验证响应
        assert response is not None
        assert len(response) > 0
    
    def test_chunk_management_during_inference(self, device):
        """测试推理过程中的块管理"""
        attention = CGCUnlimitedRSWAAttention(
            dim=512,
            num_heads=8,
            window_size=32
        ).to(device)
        
        # 添加超过最大热块数的参考块
        max_hot = 2
        chunk_ids = []
        
        for i in range(max_hot + 1):
            ref_len = 512
            ref_k = torch.randn(1, 8, ref_len, 64, device=device)
            ref_v = torch.randn(1, 8, ref_len, 64, device=device)
            ref_token_ids = torch.arange(ref_len, device=device)
            chunk_ids.append(attention.add_reference_chunk(ref_token_ids, ref_k, ref_v))
        
        # 验证块管理
        pool_info = attention.get_pool_info()
        assert pool_info["hot_chunks"] == max_hot
        assert pool_info["cold_chunks"] == 1
        
        # 执行推理，验证系统稳定性
        x = torch.randn(1, 64, 512, device=device)
        out, _, _ = attention(x, use_reference=True)
        assert out is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])