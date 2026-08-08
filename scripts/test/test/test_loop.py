import torch
import flash_kda
import subprocess
import os

def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return float(result.stdout.strip().split('\n')[0]) / 1024.0

BATCH_SIZE = 4
H, K, V = 28, 128, 128
scale = 1.0 / K**0.5

for ctx_len in [256, 512]:
    print(f'\nContext Length = {ctx_len}')
    T = ctx_len

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    q = torch.randn(BATCH_SIZE, T, H, K, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(BATCH_SIZE, T, H, K, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(BATCH_SIZE, T, H, V, device='cuda', dtype=torch.bfloat16)
    g = torch.ones(BATCH_SIZE, T, H, K, device='cuda', dtype=torch.bfloat16)
    beta = torch.zeros(BATCH_SIZE, T, H, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(BATCH_SIZE, T, H, V, device='cuda', dtype=torch.bfloat16)

    A_log = torch.zeros(H, device='cuda', dtype=torch.float32)
    dt_bias = torch.zeros(H, K, device='cuda', dtype=torch.float32)
    lower_bound = -5.0

    torch.cuda.synchronize()

    try:
        flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, None, None, None)
        torch.cuda.synchronize()
        print(f'  fwd SUCCESS! GPU Mem: {get_gpu_memory():.2f} GB')
    except Exception as e:
        print(f'  ERROR: {e}')

    del q, k, v, g, beta, out, A_log, dt_bias
    torch.cuda.empty_cache()

print('\nDone!')