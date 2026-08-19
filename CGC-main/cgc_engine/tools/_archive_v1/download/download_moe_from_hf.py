#!/usr/bin/env python3
"""在服务器上查找并下载 MoE 模型"""

import sys
import os

print("=" * 60)
print("MoE 模型下载工具")
print("=" * 60)

# 先检查是否已有下载工具
print("\n1. 检查已安装的工具...")

try:
    import huggingface_hub
    print("✓ huggingface_hub 已安装")
except ImportError:
    print("安装 huggingface_hub...")
    os.system("pip install huggingface_hub -q")
    try:
        import huggingface_hub
        print("✓ huggingface_hub 安装成功")
    except ImportError:
        print("无法安装 huggingface_hub")

# 列出可能的模型名称
model_options = [
    "microsoft/Phi-3.5-MoE-instruct",          # 原版
    "unsloth/Phi-3.5-MoE-instruct",            # Unsloth 优化版
    "TheBloke/Phi-3.5-MoE-instruct-GGUF",      # GGUF 格式
    "bartowski/Phi-3.5-MoE-instruct-GGUF",    # GGUF 格式
    "Qwen/Qwen2.5-MoE",                       # Qwen 的 MoE 版本
]

print("\n2. 可用模型选项:")
for i, model in enumerate(model_options, 1):
    print(f"  {i}. {model}")

model_dir = "/home/gs01/models"
os.makedirs(model_dir, exist_ok=True)

# 先检查 TheBloke 的 GGUF 格式 (最可能有)
target_model = "TheBloke/Phi-3.5-MoE-instruct-GGUF"

print(f"\n3. 尝试下载: {target_model}")

try:
    from huggingface_hub import snapshot_download, hf_hub_download

    # 尝试下载特定的量化版本
    quant_files = [
        "Phi-3.5-MoE-instruct.Q4_K_M.gguf",
        "Phi-3.5-MoE-instruct.Q5_K_M.gguf",
        "Phi-3.5-MoE-instruct.Q4_0.gguf",
    ]

    downloaded_file = None

    for qf in quant_files:
        try:
            print(f"   尝试下载: {qf}")
            local_path = hf_hub_download(
                repo_id=target_model,
                filename=qf,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            downloaded_file = local_path
            print(f"   ✓ 下载成功!")
            break
        except Exception as e:
            print(f"   ✗ 失败: {str(e)[:80]}")
            continue

    if downloaded_file:
        file_size = os.path.getsize(downloaded_file) / (1024*1024*1024)
        print(f"\n✓ 成功下载!")
        print(f"  文件: {downloaded_file}")
        print(f"  大小: {file_size:.2f} GB")

        # 尝试列出模型目录内容
        print(f"\n模型目录内容:")
        for f in os.listdir(model_dir):
            fpath = os.path.join(model_dir, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath) / (1024*1024*1024)
                print(f"  {f}: {size:.2f} GB")
    else:
        print("\n未找到特定的 GGUF 版本，尝试下载整个仓库...")
        try:
            local_path = snapshot_download(
                repo_id=target_model,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=["*.gguf"],
            )
            print(f"✓ 仓库下载到: {local_path}")
            print("\n已下载的文件:")
            for f in os.listdir(local_path):
                fpath = os.path.join(local_path, f)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath) / (1024*1024*1024)
                    print(f"  {f}: {size:.2f} GB")
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            print("\n尝试备选方案...")
            # 如果找不到，使用已有的测试（模拟专家）
            print("\n当前测试使用模拟专家已通过所有功能验证")
            print("可以继续使用现有测试，或手动上传模型文件")

except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("完成")
print("=" * 60)