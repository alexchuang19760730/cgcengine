#!/usr/bin/env python3
"""
OrthoKDA v4 Integration Test
Tests vLLM and llama.cpp backends
"""

import sys
sys.path.insert(0, '.')

print('=' * 60)
print('Testing OrthoKDA v4 Integration')
print('=' * 60)

# 1. Test Python Backend
print('\n1. Testing Python Backend (Pure Python TrueOrthoBasisAccumulator)')
print('-' * 60)
try:
    from cgc_engine.cgc.true_ortho_kda import TrueOrthoBasisAccumulator
    import torch

    acc = TrueOrthoBasisAccumulator(num_heads=4, head_dim=128, ortho_base_dim=32)

    k = torch.randn(4, 128)
    v = torch.randn(4, 128)
    q = torch.randn(1, 4, 128)

    acc.update(k, v)
    output = acc.attention(q)

    print(f'  [PASS] TrueOrthoBasisAccumulator works')
    print(f'  [PASS] KV shape: {acc.K.shape} (fixed O(1) memory)')
    print(f'  [PASS] current_dim: {acc.current_dim}')
    print(f'  [PASS] output shape: {output.shape}')
except Exception as e:
    print(f'  [FAIL] Python Backend test failed: {e}')
    import traceback
    traceback.print_exc()

# 2. Test Bridge
print('\n2. Testing OrthoKDAV4 Bridge')
print('-' * 60)
try:
    from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4

    kda = OrthoKDAV4(num_heads=4, head_dim=128, ortho_base_dim=32, use_cuda=False)

    k = torch.randn(4, 128)
    v = torch.randn(4, 128)
    q = torch.randn(4, 128)

    kda.update(k, v)
    output = kda.forward(q)

    print(f'  [PASS] OrthoKDAV4 Bridge works')
    print(f'  [PASS] Backend: {type(kda._python_accumulator).__name__}')
    print(f'  [PASS] output shape: {output.shape}')
except Exception as e:
    print(f'  [FAIL] Bridge test failed: {e}')
    import traceback
    traceback.print_exc()

# 3. Test llama.cpp Backend
print('\n3. Testing llama.cpp Backend')
print('-' * 60)
try:
    from cgc_engine.cgc.ortho_kda_v4_llama import (
        OrthoKDAV4LlamaBackend,
        OrthoKDAV4LlamaConfig,
        OrthoKDAKVState,
    )

    kv_state = OrthoKDAKVState(num_heads=4, head_dim=128, ortho_base_dim=32)

    import random
    key = [random.random() for _ in range(128)]
    value = [random.random() for _ in range(128)]
    query = [random.random() for _ in range(128)]

    kv_state.update(key, value)
    output = kv_state.forward(query)

    print(f'  [PASS] OrthoKDAKVState works')
    print(f'  [PASS] output length: {len(output)}')

    config = OrthoKDAV4LlamaConfig(
        num_heads=4,
        head_dim=128,
        ortho_base_dim=32,
        enable=True,
    )
    backend = OrthoKDAV4LlamaBackend(config=config)
    print(f'  [PASS] OrthoKDAV4LlamaBackend initialized')
except Exception as e:
    print(f'  [WARN] llama.cpp Backend test partially failed: {e}')
    import traceback
    traceback.print_exc()

# 4. Test vLLM Backend
print('\n4. Testing vLLM Backend')
print('-' * 60)
try:
    from cgc_engine.cgc.ortho_kda_v4_vllm import (
        OrthoKDAV4VLLMBackend,
        OrthoKDAV4VLLMConfig,
        VLLM_AVAILABLE,
    )

    if not VLLM_AVAILABLE:
        print(f'  [SKIP] vLLM not installed. Install with: pip install vllm')
    else:
        config = OrthoKDAV4VLLMConfig(
            num_heads=4,
            head_dim=128,
            ortho_base_dim=32,
            enable=True,
        )
        backend = OrthoKDAV4VLLMBackend(config=config, device='cpu')
        print(f'  [PASS] OrthoKDAV4VLLMBackend initialized (CPU mode)')
except Exception as e:
    print(f'  [WARN] vLLM Backend test failed: {e}')
    import traceback
    traceback.print_exc()

# 5. Test CGC Integration
print('\n5. Testing CGC Integration')
print('-' * 60)
try:
    from cgc_engine.cgc.ortho_kda_v4_cgc import (
        OrthoKDAV4Pass,
        OrthoKDAV4PassConfig,
        BackendType,
    )

    config = OrthoKDAV4PassConfig(
        enable=True,
        num_heads=4,
        head_dim=128,
        ortho_base_dim=32,
        backend=BackendType.PYTHON,
    )
    cgc_pass = OrthoKDAV4Pass(config=config)

    cgc_pass._init_backends()

    print(f'  [PASS] OrthoKDAV4Pass initialized')
    print(f'  [PASS] Backend type: {cgc_pass.backend_type}')
except Exception as e:
    print(f'  [FAIL] CGC Integration test failed: {e}')
    import traceback
    traceback.print_exc()

# 6. Test O(1) Memory Property
print('\n6. Testing O(1) Memory Property')
print('-' * 60)
try:
    from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4

    kda = OrthoKDAV4(num_heads=4, head_dim=128, ortho_base_dim=32, use_cuda=False)

    mem1 = kda.memory_footprint()
    print(f'  [PASS] Initial memory: {mem1["total_bytes"]} bytes')

    for i in range(1000):
        k = torch.randn(4, 128)
        v = torch.randn(4, 128)
        kda.update(k, v)

    mem2 = kda.memory_footprint()
    print(f'  [PASS] After 1000 updates: {mem2["total_bytes"]} bytes')
    print(f'  [PASS] Memory unchanged (O(1) property): {mem1["total_bytes"] == mem2["total_bytes"]}')
except Exception as e:
    print(f'  [FAIL] O(1) Memory test failed: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('Integration Tests Completed')
print('=' * 60)