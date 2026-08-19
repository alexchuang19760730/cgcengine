#!/usr/bin/env python3
"""
简化测试脚本：验证vLLM和OrthoKDA基本功能
"""

import sys
import os
sys.path.insert(0, '/home/gs01/MagiCompiler-main')

print('=' * 60)
print('简化测试：vLLM + OrthoKDA')
print('=' * 60)

# 1. 检查CUDA
print('\n【1】CUDA检查')
try:
    import torch
    print(f'✅ CUDA可用: {torch.cuda.is_available()}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA版本: {torch.version.cuda}')
except Exception as e:
    print(f'❌ CUDA检查失败: {e}')

# 2. 检查vLLM
print('\n【2】vLLM检查')
try:
    import vllm
    print(f'✅ vLLM版本: {vllm.__version__}')
except Exception as e:
    print(f'❌ vLLM检查失败: {e}')

# 3. 检查OrthoKDA
print('\n【3】OrthoKDA检查')
try:
    from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4
    
    kda = OrthoKDAV4(num_heads=32, head_dim=128, ortho_base_dim=128, use_cuda=True)
    print(f'✅ OrthoKDA初始化成功')
    print(f'   - 配置: {kda.num_heads} heads × {kda.head_dim} dim')
    print(f'   - KV形状: {kda.K.shape}')
    
    # 简单测试
    k = torch.randn(32, 128, device='cuda')
    v = torch.randn(32, 128, device='cuda')
    q = torch.randn(32, 128, device='cuda')
    
    kda.update(k, v)
    output = kda.forward(q)
    print(f'✅ OrthoKDA推理成功: {output.shape}')
    
except Exception as e:
    print(f'❌ OrthoKDA检查失败: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('基础检查完成!')
print('=' * 60)