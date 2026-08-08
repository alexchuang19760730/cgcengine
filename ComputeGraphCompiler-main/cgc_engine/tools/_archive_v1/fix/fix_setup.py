import os

path = os.path.expanduser('~/FlashKDA/setup.py')
with open(path, 'r') as f:
    content = f.read()

old = '''def get_arch_flags():
    # TODO: add compile flags here to support more architectures
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    DISABLE_SM90 = is_flag_set("FLASH_KDA_DISABLE_SM90")
    arch_flags = []
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    return arch_flags'''

new = '''def get_arch_flags():
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    arch_flags = []
    arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    arch_flags.extend(["-gencode", "arch=compute_120a,code=sm_120a"])
    return arch_flags'''

content = content.replace(old, new)

with open(path, 'w') as f:
    f.write(content)

print('Done')