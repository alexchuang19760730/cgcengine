#!/usr/bin/env python3
content = open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py').read()

# 找到我们添加的 CGC KDA 代码并修复
lines = content.split('\n')
new_lines = []
in_kda_block = False

for line in lines:
    if line.strip() == '# ==============================================' and 'CGC KDA Auto-Registration' in line:
        in_kda_block = True
        # 开始新的 KDA 块
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
        new_lines.append('            # 直接使用字符串路径注册，不导入类')
        new_lines.append('            register_backend(AttentionBackendEnum.FLASH_ATTN,')
        new_lines.append('                           "vllm_attention_backend.cgc_kda_backend.CGCKDAAttentionBackend")')
        new_lines.append('            print("[CGC-KDA] Auto-registered KDA backend via string path (VLLM_USE_CGC_KDA=1)")')
        new_lines.append('        except Exception as e:')
        new_lines.append('            print(f"[CGC-KDA] Failed to auto-register: {e}")')
        new_lines.append('')
        new_lines.append('# 延迟调用，确保所有模块都已加载')
        new_lines.append('import threading')
        new_lines.append('threading.Timer(0.01, _try_register_cgc_kda).start()')
        continue
    
    if in_kda_block:
        # 跳过旧的 KDA 块内容
        if line.strip() == '' and len(new_lines) > 0 and new_lines[-1].strip() == 'threading.Timer(0.01, _try_register_cgc_kda).start()':
            in_kda_block = False
        continue
    
    new_lines.append(line)

open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py', 'w').write('\n'.join(new_lines))
print('Done')
