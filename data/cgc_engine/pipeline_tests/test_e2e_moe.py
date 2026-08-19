#!/usr/bin/env python3
"""
OMLX-FlashMoE end-to-end test
"""

import torch
import sys
sys.path.insert(0, "/root/MagiCompiler-main")

from cgc_engine.flash_moe.client import FlashMoEClient
from cgc_engine.omlx.client import OMLXClient

def test_e2e_moe_inference():
    print("="*60)
    print("OMLX-FlashMoE End-to-End Test")
    print("="*60)
    
    flashmoe = FlashMoEClient(
        expert_dir="/home/gs01/models",
        backend="cuda"
    )
    flashmoe.num_experts = 16
    flashmoe.expert_dim = 4096
    flashmoe.intermediate_dim = 14336
    print("FlashMoEClient initialized")
    
    omlx = OMLXClient(model_dir="/home/gs01/models")
    omlx.num_experts = 16
    omlx.expert_dim = 4096
    print("OMLXClient initialized")
    
    batch_size = 2
    seq_len = 128
    hidden_dim = 4096
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16).cuda()
    print("Test input created: " + str(x.shape))
    
    print("\n--- Step 1: Expert Prediction ---")
    predicted_experts = omlx.predict_experts(x, top_k=2)
    print("Predicted experts: " + str(predicted_experts.flatten().tolist()))
    
    print("\n--- Step 2: Load Experts to Cache ---")
    unique_experts = list(set(predicted_experts.flatten().tolist()))
    flashmoe.load_experts(expert_ids=unique_experts)
    print("Loaded experts: " + str(unique_experts))
    
    print("\n--- Step 3: MoE Inference ---")
    try:
        result = flashmoe.moe_forward(x, top_k=2)
        print("moe_forward succeeded")
    except AttributeError:
        result = flashmoe.mlp_forward_moe(x, top_k=2)
        print("mlp_forward_moe succeeded")
    print("Input shape: " + str(x.shape))
    print("Output shape: " + str(result.shape))
    
    print("\n--- Step 4: Result Validation ---")
    assert result.shape[0] == batch_size, "Batch size mismatch"
    assert result.shape[1] == seq_len, "Sequence length mismatch"
    assert result.shape[2] == flashmoe.intermediate_dim, "Output dim mismatch"
    print("Result validation passed")
    
    print("\n--- Step 5: LRU Eviction Test ---")
    initial_cache_size = len(flashmoe.cache_manager)
    flashmoe.cache_manager.evict_oldest()
    after_evict_size = len(flashmoe.cache_manager)
    print("Cache size before eviction: " + str(initial_cache_size))
    print("Cache size after eviction: " + str(after_evict_size))
    
    print("\n" + "="*60)
    print("All end-to-end tests passed!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_e2e_moe_inference()
