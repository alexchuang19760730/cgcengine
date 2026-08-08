import os

path = '/root/flashkv0516/ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python/sglang/srt/ray/engine.py'
with open(path, 'r') as f:
    content = f.read()

func_def = '''
def _get_effective_model_parallel_size(server_args: ServerArgs) -> int:
    return max(
        int(server_args.tp_size),
        int(server_args.ep_size) * max(int(server_args.moe_dp_size), 1),
    )

def _compute_world_size'''

if 'def _get_effective_model_parallel_size' not in content:
    content = content.replace('def _compute_world_size', func_def)
    with open(path, 'w') as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Already patched")
