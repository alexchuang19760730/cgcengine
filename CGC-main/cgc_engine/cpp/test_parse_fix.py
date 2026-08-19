#!/usr/bin/env python3
"""Test the updated llama_monkey_patch parse_gguf_header with existing GGUF files."""
import sys
sys.path.insert(0, r"D:\alex\flashkv0516")

from app.edge_engine.llama_monkey_patch import parse_gguf_header

models = [
    r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
    r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf",
]

for path in models:
    print(f"\n{'='*60}")
    print(f"Testing: {path}")
    print(f"{'='*60}")
    try:
        result = parse_gguf_header(path)
        print(f"  Version: {result['version']}")
        print(f"  Tensors parsed: {len(result['tensors'])}/{result['n_tensors']}")
        print(f"  KV entries parsed: {len(result['kv'])}/{result['n_kv']}")
        print(f"  Data start: 0x{result['data_start']:X}")
        
        if result['tensors']:
            print(f"  First tensor: '{result['tensors'][0]['name']}'")
            print(f"  Last tensor: '{result['tensors'][-1]['name']}'")
        
        arch = result['kv'].get('general.architecture', 'N/A')
        print(f"  Architecture: {arch}")
        
        if result['tensors']:
            expert_count = sum(1 for t in result['tensors'] if 'expert' in t['name'] or 'exps' in t['name'])
            print(f"  Expert tensors: {expert_count}")
        
        print(f"  ✅ SUCCESS")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
