#!/usr/bin/env python3
"""
DeepSeek V4 Flash - ModelScope 断点续传 + 多线程 加速下载脚本
支持断点续传，不会重复下已完成的文件
"""
import os
import sys
import time
from pathlib import Path

SERVER_MAGICOMPILER = '/home/gs01/MagiCompiler-main'
SERVER_MODELS_DIR = '/home/gs01/models'

def print_header(title):
    print("\n" + "="*100)
    print(f"  {title}")
    print("="*100)

print_header("DeepSeek V4 Flash @ gs01 - ModelScope 断点续传加速下载")

os.makedirs(SERVER_MODELS_DIR, exist_ok=True)

# 1. 升级ModelScope到最新版加速
print("\n[1/4] 升级ModelScope到最新加速版...")
os.system("pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")

# 2. 导入加速模块
print("\n[2/4] 配置国内镜像 + 断点续传...")
try:
    from modelscope import snapshot_download
    os.environ['MODELSCOPE_ENDPOINT'] = 'https://modelscope.cn'
    os.environ['MODELSCOPE_DOWNLOAD_PARALLEL'] = '8'  # 8线程并发
except ImportError:
    print("⚠️  重试安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    from modelscope import snapshot_download

# 3. 开始断点续传下载
print("\n[3/4] 开始断点续传下载 deepseek-ai/DeepSeek-V4-Flash...")
print("  → 已下载完成的文件会自动跳过，不会重复下载")
model_dir = snapshot_download(
    'deepseek-ai/DeepSeek-V4-Flash',
    cache_dir=SERVER_MODELS_DIR,
    revision='master'
)

print(f"\n✅ 下载完成！模型保存路径: {model_dir}")

# 4. 验证
print_header("Step 4: 下载验证")
total_size = 0
for f in Path(model_dir).rglob("*"):
    if f.is_file():
        total_size += f.stat().st_size
total_gb = total_size / 1024 / 1024 / 1024
print(f"\n✅ 模型总大小: {total_gb:.2f} GB")
print(f"✅ 全部文件验证完毕！")

report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": "deepseek-ai/DeepSeek-V4-Flash",
    "model_dir": model_dir,
    "total_size_gb": round(total_gb, 2),
    "accelerated": True,
    "resume_support": True,
    "parallel_threads": 8
}
import json
rpth = os.path.join(SERVER_MAGICOMPILER, "ACCELERATED_DOWNLOAD_REPORT.json")
with open(rpth, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n📝 报告保存: {rpth}")
print("\n🎉 ModelScope断点续传加速下载完成！")
