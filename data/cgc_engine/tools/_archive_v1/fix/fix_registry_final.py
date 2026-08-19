#!/usr/bin/env python3

# 读取原始文件
with open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py', 'r') as f:
    content = f.read()

# 找到最后一个有效的函数定义位置（register_backend函数）
lines = content.split('\n')
clean_lines = []
in_bad_section = False

for i, line in enumerate(lines):
    # 检测损坏的部分
    if line.strip().startswith('import os') and i > len(lines) - 20:
        in_bad_section = True
    if in_bad_section and '==============================================' in line:
        break
    if not in_bad_section:
        clean_lines.append(line)

# 添加干净的注册代码
clean_lines.append('')
clean_lines.append('')
clean_lines.append('# ==============================================')
clean_lines.append('# CGC KDA Auto-Registration')
clean_lines.append('# ==============================================')
clean_lines.append('import os')
clean_lines.append('')
clean_lines.append('def _try_register_cgc_kda():')
clean_lines.append('    """Auto-register CGC KDA backend if environment variable is set."""')
clean_lines.append('    if os.environ.get("VLLM_USE_CGC_KDA", "0") == "1":')
clean_lines.append('        try:')
clean_lines.append('            import sys')
clean_lines.append('            sys.path.insert(0, "/home/gs01")')
clean_lines.append('            # 使用独立的 stub 文件来避免循环导入')
clean_lines.append('            register_backend(AttentionBackendEnum.FLASH_ATTN,')
clean_lines.append('                           "vllm_attention_backend.kda_backend_stub.CGCKDAAttentionBackend")')
clean_lines.append('            print("[CGC-KDA] Auto-registered KDA backend via stub (VLLM_USE_CGC_KDA=1)")')
clean_lines.append('        except Exception as e:')
clean_lines.append('            print(f"[CGC-KDA] Failed to auto-register: {e}")')
clean_lines.append('')
clean_lines.append('_try_register_cgc_kda()')

# 写入修复后的文件
with open('/home/gs01/.local/lib/python3.10/site-packages/vllm/v1/attention/backends/registry.py', 'w') as f:
    f.write('\n'.join(clean_lines))

print('Done')
