#!/usr/bin/env python3
content = open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py').read()

# 找到我们添加的 CGC KDA 代码并替换
lines = content.split('\n')
new_lines = []
in_kda_block = False

for line in lines:
    if line.strip() == '# ==============================================' and 'CGC KDA Auto-Registration' in line:
        in_kda_block = True
        # 添加新的 KDA 块
        new_lines.append('')
        new_lines.append('# ==============================================')
        new_lines.append('# CGC KDA Auto-Registration')
        new_lines.append('# ==============================================')
        new_lines.append('import os')
        new_lines.append('')
        new_lines.append('def _try_register_cgc_kda():')
        new_lines.append('    """Auto-register CGC KDA backend if environment variable is set."""')
        new_lines.append('    if os.environ.get("VLLM_USE_CGC_KDA", "0") == "1":')
        new_lines.append('        try:')
        new_lines.append('            import sys')
        new_lines.append('            sys.path.insert(0, "/home/gs01")')
        new_lines.append('            # 使用独立的 stub 文件来避免循环导入')
        new_lines.append('            register_backend(AttentionBackendEnum.FLASH_ATTN,')
        new_lines.append('                           "vllm_attention_backend.kda_backend_stub.CGCKDAAttentionBackend")')
        new_lines.append('            print("[CGC-KDA] Auto-registered KDA backend via stub (VLLM_USE_CGC_KDA=1)")')
        new_lines.append('        except Exception as e:')
        new_lines.append('            print(f"[CGC-KDA] Failed to auto-register: {e}")')
        new_lines.append('')
        new_lines.append('_try_register_cgc_kda()')
        continue
    
    if in_kda_block:
        # 跳过旧的 KDA 块内容
        if line.strip() == '':
            in_kda_block = False
        continue
    
    new_lines.append(line)

open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py', 'w').write('\n'.join(new_lines))
print('Done')
