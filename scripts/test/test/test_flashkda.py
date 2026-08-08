import torch
import flash_kda

B, T, H, K, V = 4, 256, 28, 128, 128
scale = 1.0 / K**0.5

q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
g = torch.ones(B, T, H, K, device='cuda', dtype=torch.bfloat16)
beta = torch.zeros(B, T, H, device='cuda', dtype=torch.bfloat16)
out = torch.empty(B, T, H, V, device='cuda', dtype=torch.bfloat16)

A_log = torch.zeros(H, device='cuda', dtype=torch.float32)
dt_bias = torch.zeros(H, K, device='cuda', dtype=torch.float32)
lower_bound = -5.0

T_total = B * T
workspace_size = flash_kda.get_workspace_size(T_total, H, B)
workspace = torch.empty(workspace_size, device='cuda', dtype=torch.uint8)

print(f'Input shapes: q={q.shape}, k={k.shape}, v={v.shape}')
print(f'Output shape: out={out.shape}')
print(f'Workspace size: {workspace_size}')

torch.cuda.synchronize()
print('Calling flash_kda.fwd...')
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, None, None, None)
torch.cuda.synchronize()
print('FlashKDA fwd completed and synchronized!')
print(f'Output before access: device={out.device}, dtype={out.dtype}')
result = out.cpu()
print(f'Output copied to CPU successfully!')
print(f'Output stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}')