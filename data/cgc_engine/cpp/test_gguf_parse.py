#!/usr/bin/env python3
"""Quick test: verify GGUF parsing and ExpertStreamerLite with an actual model."""

import sys
sys.path.insert(0, "D:/alex/flashkv0516/app/edge_engine")
from llama_monkey_patch import parse_gguf_header, ExpertStreamerLite

MODELS = [
    (r"D:\alex\flashkv0516\models\qwen2.5-1.5b-instruct-q4_k_m.gguf", "qwen2.5 (dense)"),
    (r"D:\alex\flashkv0516\models\gemma4_gguf\gemma-4-26B-A4B-it-UD-IQ3_S.gguf", "gemma4 (check)"),
]

for model_path, label in MODELS:
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Path: {model_path}")
    
    try:
        header = parse_gguf_header(model_path)
        print(f"  GGUF version: {header['version']}")
        print(f"  Tensor count: {header['n_tensors']}")
        print(f"  KV count: {header['n_kv']}")
        
        kv = header['kv']
        arch = kv.get("general.architecture", "unknown")
        print(f"  Architecture: {arch}")
        
        has_experts = any("expert" in t["name"] for t in header["tensors"])
        print(f"  Has MoE experts: {has_experts}")
        
        # Try ExpertStreamerLite
        streamer = ExpertStreamerLite(model_path)
        stats = streamer.cache_stats()
        print(f"  ExpertStreamerLite:")
        print(f"    Architecture: {stats['architecture']}")
        print(f"    Hidden: {stats['hidden']}")
        print(f"    Inter: {stats['inter']}")
        print(f"    Num experts: {stats['num_experts']}")
        print(f"    Has experts: {stats['has_experts']}")
        print(f"    Num layers: {stats['num_layers']}")
        
        if stats["has_experts"]:
            eids = streamer.list_experts()
            print(f"    Expert IDs: {eids[:8]}{'...' if len(eids) > 8 else ''}")
        else:
            print(f"    (dense model, no MoE experts)")
        
        print(f"  [OK] PASSED")
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("GGUF parsing tests complete")