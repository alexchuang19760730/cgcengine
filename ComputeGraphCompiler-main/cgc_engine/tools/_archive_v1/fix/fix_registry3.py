#!/usr/bin/env python3
content = open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py').read()

# 找到所有重复的 CGC KDA 代码块并删除，只保留一个正确的版本
lines = content.split('\n')
new_lines = []
in_kda_block = False
kda_block_count = 0

for line in lines:
    if line.strip() == '# ==============================================' and 'CGC KDA Auto-Registration' in ''.join(lines[lines.index(line):lines.index(line)+3] if lines.index(line)+3 < len(lines) else lines[lines.index(line):]):
        in_kda_block = True
        kda_block_count += 1
        if kda_block_count == 1:
            # 只保留第一个块，但修复它
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
            new_lines.append('            # 直接使用字符串路径注册，不需要导入类')
            new_lines.append('            register_backend(AttentionBackendEnum.FLASH_ATTN,')
            new_lines.append('                           "vllm_attention_backend.cgc_kda_backend.CGCKDAAttentionBackend")')
            new_lines.append('            print("[CGC-KDA] Auto-registered KDA backend via string path (VLLM_USE_CGC_KDA=1)")')
            new_lines.append('        except Exception as e:')
            new_lines.append('            print(f"[CGC-KDA] Failed to auto-register: {e}")')
            new_lines.append('')
            new_lines.append('_try_register_cgc_kda()')
        # 跳过其他重复块
        continue
    
    if in_kda_block:
        # 检查是否到达块的末尾（下一个空行或特定模式）
        if line.strip() == '' and kda_block_count > 1:
            in_kda_block = False
        elif kda_block_count > 1:
            continue  # 跳过重复块的内容
    
    if not (in_kda_block and kda_block_count > 1):
        new_lines.append(line)

open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py', 'w').write('\n'.join(new_lines))
print('Done')
