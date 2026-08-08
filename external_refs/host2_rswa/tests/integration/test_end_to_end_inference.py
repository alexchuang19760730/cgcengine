# Copyright (c) 2026 SandAI. All Rights Reserved.

"""
端到端推理集成测试
"""

import torch
import pytest
import sys
sys.path.insert(0, sys.path[0] + '/../..')

from cgc_engine.rswa_integration import RSWAPrefillPoolEngine


class TestEndToEndInference:
    """端到端推理测试"""
    
    @pytest.fixture
    def engine(self):
        """创建推理引擎"""
        return RSWAPrefillPoolEngine(
            dim=512,
            num_heads=8,
            window_size=32,
            max_hot_chunks=4,
            chunk_size=1024,
        )
    
    def test_full_inference_pipeline(self, engine):
        """测试完整推理流程"""
        # Step 1: 添加参考知识
        reference_texts = [
            "法律知识: 《中华人民共和国刑法》第三百零二条规定：盗窃、侮辱、故意毁坏尸体、尸骨、骨灰的，处三年以下有期徒刑、拘役或者管制。",
            "技术知识: GPU Direct Storage (GDS) 是 NVIDIA 公司开发的一项技术，允许 GPU 直接从存储设备读取数据，无需经过 CPU 内存中转。",
            "历史知识: 唐朝（618年—907年）是中国历史上继隋朝之后的大一统中原王朝，共历二十一帝，享国二百八十九年。",
        ]
        
        chunk_ids = engine.prefill_reference(reference_texts)
        assert len(chunk_ids) == len(reference_texts)
        
        # Step 2: 验证参考块加载
        pool_info = engine.attention.get_pool_info()
        assert pool_info["hot_chunks"] == len(reference_texts)
        
        # Step 3: 执行多个推理查询
        queries = [
            "什么是 GDS 技术？",
            "盗窃尸体的法律后果是什么？",
            "唐朝存在了多少年？",
        ]
        
        for query in queries:
            response = engine.infer(query, max_tokens=50)
            assert response is not None
            assert len(response) > 0
        
        # Step 4: 验证系统稳定性
        assert engine.attention.get_pool_info()["hot_chunks"] == len(reference_texts)
    
    def test_inference_without_reference(self, engine):
        """测试无参考上下文的推理"""
        response = engine.infer("你好，世界！", max_tokens=30)
        assert response is not None
        assert len(response) > 0
    
    def test_multiple_rounds(self, engine):
        """测试多轮对话"""
        # 添加参考知识
        reference_texts = [
            "用户偏好: 用户喜欢科技类新闻，对人工智能特别感兴趣。",
        ]
        engine.prefill_reference(reference_texts)
        
        # 多轮对话
        queries = [
            "最近有什么科技新闻？",
            "人工智能有什么新进展？",
            "能详细说说吗？",
        ]
        
        for i, query in enumerate(queries):
            response = engine.infer(query, max_tokens=40)
            assert response is not None
            print(f"Round {i+1}: {query} -> {response[:50]}...")
    
    def test_engine_reset(self, engine):
        """测试引擎重置"""
        # 添加参考知识
        reference_texts = ["测试知识: 这是一个测试。"]
        engine.prefill_reference(reference_texts)
        
        assert engine.attention.get_pool_info()["hot_chunks"] == 1
        
        # 重置
        engine.attention.clear_pool()
        
        assert engine.attention.get_pool_info()["hot_chunks"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])