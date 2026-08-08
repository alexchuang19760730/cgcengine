#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/gs01/MagiCompiler-main')

print('='*60)
print('Test: vLLM + OrthoKDA')
print('='*60)

# Check CUDA
print('\n[1] CUDA Check')
try:
    import torch
    print('CUDA available:', torch.cuda.is_available())
    print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    print('CUDA error:', e)

# Check vLLM
print('\n[2] vLLM Check')
try:
    import vllm
    print('vLLM version:', vllm.__version__)
except Exception as e:
    print('vLLM error:', e)

# Check OrthoKDA
print('\n[3] OrthoKDA Check')
try:
    from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4
    kda = OrthoKDAV4(num_heads=32, head_dim=128, ortho_base_dim=128, use_cuda=True)
    print('OrthoKDA initialized: OK')
    print('Config:', kda.num_heads, 'heads x', kda.head_dim, 'dim')
except Exception as e:
    print('OrthoKDA error:', e)

print('\n' + '='*60)
print('Test completed')
print('='*60)

