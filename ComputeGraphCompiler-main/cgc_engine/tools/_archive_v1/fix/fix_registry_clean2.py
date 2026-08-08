#!/usr/bin/env python3
# 完全清理并重新写入 registry.py
content = open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py').read()

# 移除所有 CGC KDA 相关代码块
lines = content.split('\n')
new_lines = []
in_kda_block = False

for line in lines:
    # 检测 KDA 块的开始
    if '# ==============================================' in line:
        # 检查下几行是否包含 CGC KDA
        context = ''.join(lines[lines.index(line):min(lines.index(line)+5, len(lines))])
        if 'CGC KDA' in line or 'CGC KDA' in context:
            in_kda_block = True
            continue
    
    if in_kda_block:
        if line.strip() == '' and len(new_lines) > 0 and new_lines[-1].strip() != '':
            in_kda_block = False
            new_lines.append(line)
        continue
    
    new_lines.append(line)

# 在文件末尾添加干净的 KDA 注册代码 - os 导入放在函数外面
new_lines.append('')
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

open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py', 'w').write('\n'.join(new_lines))
print('Done')
