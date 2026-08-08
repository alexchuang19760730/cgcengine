import torch
import flash_kda
import subprocess
import sys

ctx_len = int(sys.argv[1]) if len(sys.argv) > 1 else 256

BATCH_SIZE = 4
H, K, V = 28, 128, 128
scale = 1.0 / K**0.5
T = ctx_len

print(f'Context Length = {ctx_len}')

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

print('Call 1...')
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, None, None, None)
torch.cuda.synchronize()
print('Call 1 SUCCESS!')

print('Call 2...')
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, None, None, None)
torch.cuda.synchronize()
print('Call 2 SUCCESS!')