#!/usr/bin/env python3
content = open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py').read()

# 找到我们之前添加的代码并修复
lines = content.split('\n')
new_lines = []
i = 0
while i < len(lines):
    if i < len(lines) - 1 and lines[i].strip() == '# ==============================================' and 'CGC KDA Auto-Registration' in lines[i+1]:
        # 找到我们添加的代码块，替换它
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
        new_lines.append('            # 直接使用字符串路径注册，避免循环导入')
        new_lines.append('            register_backend(AttentionBackendEnum.FLASH_ATTN, ')
        new_lines.append('                           "vllm_attention_backend.cgc_kda_backend.CGCKDAAttentionBackend")')
        new_lines.append('            print("[CGC-KDA] Auto-registered KDA backend (VLLM_USE_CGC_KDA=1)")')
        new_lines.append('        except Exception as e:')
        new_lines.append('            print(f"[CGC-KDA] Failed to auto-register: {e}")')
        new_lines.append('')
        new_lines.append('_try_register_cgc_kda()')
        # 跳过旧代码块
        while i < len(lines) and (i == 0 or lines[i] != '# ==============================================' or 'CGC KDA' not in lines[i] or i == len(lines)-1):
            i += 1
            if i < len(lines) and lines[i].strip() == '# ==============================================':
                i += 1
                break
    else:
        new_lines.append(lines[i])
        i += 1

open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py', 'w').write('\n'.join(new_lines))
print('Done')
