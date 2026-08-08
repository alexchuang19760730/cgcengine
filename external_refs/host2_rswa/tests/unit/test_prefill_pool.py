# Copyright (c) 2026 SandAI. All Rights Reserved.

"""
Prefill Pool 单元测试
"""

import torch
import pytest
import os
import sys
sys.path.insert(0, sys.path[0] + '/../..')

from cgc_engine.prefill_pool import PrefillPool, Chunk, ChunkMetadata


class TestPrefillPool:
    """Prefill Pool 单元测试"""
    
    @pytest.fixture
    def pool(self, tmp_path):
        """创建 Prefill Pool 实例"""
        return PrefillPool(
            max_hot_chunks=2,
            chunk_size=1024,
            storage_path=str(tmp_path / "prefill_pool"),
            enable_gds=False  # 单元测试禁用 GDS
        )
    
    @pytest.fixture
    def device(self):
        """获取测试设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_init(self, pool):
        """测试初始化"""
        assert pool.max_hot_chunks == 2
        assert pool.chunk_size == 1024
        assert pool.hot_chunks == {}
        assert pool.cold_metadata == {}
        assert pool.step == 0
    
    def test_add_hot_chunk(self, pool, device):
        """测试添加热块"""
        chunk_len = 512
        token_ids = torch.arange(chunk_len, device=device)
        ref_k = torch.randn(1, 8, chunk_len, 64, device=device)
        ref_v = torch.randn(1, 8, chunk_len, 64, device=device)
        
        chunk_id = pool.add_hot_chunk(token_ids, ref_k, ref_v)
        
        assert chunk_id is not None
        assert len(chunk_id) == 16  # MD5 前16位
        assert chunk_id in pool.hot_chunks
        
        chunk = pool.hot_chunks[chunk_id]
        assert torch.equal(chunk.token_ids, token_ids)
        assert torch.equal(chunk.ref_k, ref_k)
        assert torch.equal(chunk.ref_v, ref_v)
        assert chunk.metadata.is_hot is True
    
    def test_chunk_eviction(self, pool, device):
        """测试块驱逐策略"""
        # 添加超过最大热块数的块
        for i in range(3):
            token_ids = torch.arange(512, device=device)
            ref_k = torch.randn(1, 8, 512, 64, device=device)
            ref_v = torch.randn(1, 8, 512, 64, device=device)
            pool.add_hot_chunk(token_ids, ref_k, ref_v)
        
        # 验证热块数不超过限制
        assert len(pool.hot_chunks) == 2
        
        # 验证冷块数
        assert len(pool.cold_metadata) == 1
    
    def test_load_chunk(self, pool, device):
        """测试加载块"""
        # 添加初始块
        token_ids = torch.arange(512, device=device)
        ref_k = torch.randn(1, 8, 512, 64, device=device)
        ref_v = torch.randn(1, 8, 512, 64, device=device)
        chunk_id = pool.add_hot_chunk(token_ids, ref_k, ref_v)
        
        # 加载已存在的块
        loaded = pool.load_chunk(chunk_id, device)
        assert loaded is not None
        assert loaded.chunk_id == chunk_id
        assert torch.equal(loaded.token_ids, token_ids)
    
    def test_load_cold_chunk(self, pool, device):
        """测试加载冷块"""
        # 添加足够多的块以触发驱逐
        chunk_ids = []
        for i in range(3):
            token_ids = torch.arange(512, device=device)
            ref_k = torch.randn(1, 8, 512, 64, device=device)
            ref_v = torch.randn(1, 8, 512, 64, device=device)
            chunk_ids.append(pool.add_hot_chunk(token_ids, ref_k, ref_v))
        
        # 第一个块应该被驱逐到冷存储
        first_chunk_id = chunk_ids[0]
        assert first_chunk_id not in pool.hot_chunks
        assert first_chunk_id in pool.cold_metadata
        
        # 加载冷块
        loaded = pool.load_chunk(first_chunk_id, device)
        assert loaded is not None
        assert loaded.chunk_id == first_chunk_id
        assert loaded in pool.hot_chunks.values()
    
    def test_get_all_ref_kv(self, pool, device):
        """测试获取所有参考 KV"""
        # 添加多个块
        for i in range(2):
            token_ids = torch.arange(512, device=device)
            ref_k = torch.randn(1, 8, 512, 64, device=device)
            ref_v = torch.randn(1, 8, 512, 64, device=device)
            pool.add_hot_chunk(token_ids, ref_k, ref_v)
        
        # 获取所有参考 KV
        all_k, all_v = pool.get_all_ref_kv(device)
        
        assert all_k is not None
        assert all_v is not None
        assert all_k.shape == (1, 8, 1024, 64)  # 2 * 512
        assert all_v.shape == (1, 8, 1024, 64)
    
    def test_info(self, pool, device):
        """测试获取 Pool 信息"""
        # 添加块
        token_ids = torch.arange(512, device=device)
        ref_k = torch.randn(1, 8, 512, 64, device=device)
        ref_v = torch.randn(1, 8, 512, 64, device=device)
        pool.add_hot_chunk(token_ids, ref_k, ref_v)
        
        info = pool.info()
        
        assert info["hot_chunks"] == 1
        assert info["cold_chunks"] == 0
        assert info["hot_tokens"] == 512
        assert info["max_hot_chunks"] == 2
        assert info["chunk_size"] == 1024
    
    def test_clear(self, pool, device):
        """测试清空 Pool"""
        # 添加块
        token_ids = torch.arange(512, device=device)
        ref_k = torch.randn(1, 8, 512, 64, device=device)
        ref_v = torch.randn(1, 8, 512, 64, device=device)
        pool.add_hot_chunk(token_ids, ref_k, ref_v)
        
        assert len(pool.hot_chunks) == 1
        
        # 清空
        pool.clear()
        
        assert len(pool.hot_chunks) == 0
        assert len(pool.cold_metadata) == 0
    
    def test_remove_chunk(self, pool, device):
        """测试移除块"""
        # 添加块
        token_ids = torch.arange(512, device=device)
        ref_k = torch.randn(1, 8, 512, 64, device=device)
        ref_v = torch.randn(1, 8, 512, 64, device=device)
        chunk_id = pool.add_hot_chunk(token_ids, ref_k, ref_v)
        
        assert chunk_id in pool.hot_chunks
        
        # 移除块
        result = pool.remove_chunk(chunk_id)
        
        assert result is True
        assert chunk_id not in pool.hot_chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])