#!/usr/bin/env python3

# 读取原始文件
with open('/home/gs01/vllm_attention_backend/cgc_kda_backend.py', 'r') as f:
    content = f.read()

lines = content.split('\n')
new_lines = []
in_vllm_ok_block = False
vllm_ok_block_content = []
cgckda_impl_content = []
cgckda_backend_content = []
monkey_patch_content = []

for i, line in enumerate(lines):
    # 检测 if VLLM_OK: 块的开始
    if line.strip() == 'if VLLM_OK:' and i < len(lines) - 1 and lines[i+1].strip() == '':
        in_vllm_ok_block = True
        vllm_ok_block_content.append(line)
        continue
    
    if in_vllm_ok_block:
        vllm_ok_block_content.append(line)
        
        # 检测 CGCKDAImpl 类的开始
        if 'class CGCKDAImpl' in line:
            cgckda_impl_content = []
            j = i
            while j < len(lines):
                cgckda_impl_content.append(lines[j])
                # 检测类的结束（下一个 class 或 def 或文件结束）
                j += 1
                if j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith('class ') or next_line.startswith('def ') or next_line.startswith('if '):
                        break
        
        # 检测 CGCKDAAttentionBackend 类
        if 'class CGCKDAAttentionBackend' in line:
            cgckda_backend_content = []
            j = i
            while j < len(lines):
                cgckda_backend_content.append(lines[j])
                j += 1
                if j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith('class ') or next_line.startswith('def ') or next_line.startswith('if '):
                        break
        
        # 检测 monkey_patch_vllm 函数
        if 'def monkey_patch_vllm' in line:
            monkey_patch_content = []
            j = i
            while j < len(lines):
                monkey_patch_content.append(lines[j])
                j += 1
                if j < len(lines):
                    next_line = lines[j].strip()
                    if (next_line.startswith('class ') or next_line.startswith('def ') or 
                        next_line.startswith('if ') and not next_line.startswith('if os.environ')):
                        break
        
        # 检测块的结束
        if line.strip() == '' and i > 0 and lines[i-1].strip() == '':
            in_vllm_ok_block = False
    else:
        new_lines.append(line)

# 创建新文件内容
new_content = '\n'.join(new_lines)

# 在文件末尾添加修改后的内容
new_content += '\n\n'
new_content += '# ==============================================\n'
new_content += '# CGCKDAAttentionBackend - 定义在顶层以避免循环导入问题\n'
new_content += '# ==============================================\n\n'

# 添加 CGCKDAAttentionBackend 类，使用延迟导入
new_content += '''
class CGCKDAAttentionBackend:
    @staticmethod
    def get_impl_cls():
        # 延迟导入以避免循环导入
        from vllm_attention_backend.cgc_kda_backend import CGCKDAImpl
        return CGCKDAImpl

    @staticmethod
    def get_name():
        return "FLASH_ATTN"
'''

# 添加 monkey_patch_vllm 函数
new_content += '\n' + '\n'.join(monkey_patch_content)

# 添加最后的自动注册逻辑
new_content += '''

if os.environ.get("VLLM_USE_CGC_KDA", "0") == "1":
    print(f"[CGC-KDA] Auto-registering (VLLM_USE_CGC_KDA=1)...")
    monkey_patch_vllm()


def register_cgc_kda_now():
    """立即注册 CGC KDA Backend"""
    if os.environ.get("VLLM_USE_CGC_KDA", "0") != "1":
        os.environ["VLLM_USE_CGC_KDA"] = "1"
    monkey_patch_vllm()
'''

# 写入新文件
with open('/home/gs01/vllm_attention_backend/cgc_kda_backend.py', 'w') as f:
    f.write(new_content)

print('Done')
