#!/usr/bin/env python3
"""
修复 MinGW 编译的 llama-bench.exe。
通过正确设置 DLL 搜索路径和环境变量来解决依赖问题。
"""

import os
import sys
import subprocess

# 路径配置
LLAMA_BENCH_DIR = r"D:\alex\toolchains\llama-build\bin"
MINGW_BIN = r"D:\alex\toolchains\winlibs-gcc162\mingw64\bin"
LLAMA_BENCH = os.path.join(LLAMA_BENCH_DIR, "llama-bench.exe")

print("=" * 70)
print("FIXING MINGW LLAMA-BECH DEPENDENCIES")
print("=" * 70)

# 1. 检查所有 DLL 是否就绪
print("\n1. 检查 DLL 文件...")
required_dlls = [
    "libgcc_s_seh-1.dll",
    "libstdc++-6.dll",
    "libwinpthread-1.dll",
    "libgomp-1.dll",
    "vulkan-1.dll",
]

for dll in required_dlls:
    # 检查 llama-bench 目录
    our_path = os.path.join(LLAMA_BENCH_DIR, dll)
    # 检查 MinGW bin
    mingw_path = os.path.join(MINGW_BIN, dll)
    # 检查 System32
    sys32_path = os.path.join(r"C:\Windows\System32", dll)
    
    locations = []
    for path, name in [(our_path, "llama-bench/bin"), (mingw_path, "mingw64/bin"), (sys32_path, "System32")]:
        if os.path.exists(path):
            locations.append((path, name))
    
    if locations:
        path, name = locations[0]
        print(f"  ✅ {dll}: found in {name}")
    else:
        print(f"  ❌ {dll}: NOT FOUND anywhere!")

# 2. 创建完整的环境变量
print("\n2. 构建环境变量...")
env = os.environ.copy()

# 将所有必要的目录加入 PATH
path_dirs = [
    LLAMA_BENCH_DIR,
    MINGW_BIN,
    r"C:\Windows\System32",
]

# 获取当前 PATH 并扩展
current_path = env.get("PATH", "")
new_path = os.pathsep.join(path_dirs) + os.pathsep + current_path
env["PATH"] = new_path

# 添加其他可能需要的环境变量
env["LLAMA_BENCH_PATH"] = LLAMA_BENCH_DIR

print(f"  PATH includes:")
for d in path_dirs:
    print(f"    {d}")

# 3. 测试运行
print("\n3. 测试 llama-bench...")
try:
    result = subprocess.run(
        [LLAMA_BENCH, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=LLAMA_BENCH_DIR,
        env=env
    )
    
    if result.returncode == 0:
        print("  ✅ SUCCESS! llama-bench runs!")
        print(f"  Output:\n{result.stdout[:1000]}")
    else:
        print(f"  ❌ Exit code: {result.returncode}")
        print(f"  Stdout: {result.stdout[:500]}")
        print(f"  Stderr: {result.stderr[:500]}")
        
except subprocess.TimeoutExpired:
    print("  ⚠️  Timeout after 10s (might be loading model)")
except Exception as e:
    print(f"  ❌ Error: {type(e).__name__}: {e}")

# 4. 编写启动脚本
print("\n4. 创建启动脚本...")

# 创建一个 PowerShell 启动脚本
ps_script = f'''# llama-bench 启动脚本
# 自动设置正确的 DLL 搜索路径

$env:PATH = "{LLAMA_BENCH_DIR};{MINGW_BIN};$env:PATH"
& "{LLAMA_BENCH}" $args
'''

script_path = os.path.join(LLAMA_BENCH_DIR, "run-llama-bench.ps1")
with open(script_path, 'w') as f:
    f.write(ps_script)
print(f"  ✅ Created: {script_path}")

# 创建一个 CMD 启动脚本
cmd_script = f'''@echo off
REM llama-bench 启动脚本 (CMD 版本)
set PATH={LLAMA_BENCH_DIR};{MINGW_BIN};%PATH%
"{LLAMA_BENCH}" %*
'''

cmd_path = os.path.join(LLAMA_BENCH_DIR, "run-llama-bench.cmd")
with open(cmd_path, 'w') as f:
    f.write(cmd_script)
print(f"  ✅ Created: {cmd_path}")

# 5. 最终测试
print("\n5. 使用启动脚本测试...")
try:
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=LLAMA_BENCH_DIR
    )
    
    if result.returncode == 0:
        print("  ✅ PowerShell script works!")
        print(f"  Output:\n{result.stdout[:500]}")
    else:
        print(f"  Exit code: {result.returncode}")
        print(f"  Output: {result.stdout[:500]}")
        
except Exception as e:
    print(f"  Error: {e}")

# 6. 运行带模型的测试
print("\n6. 测试实际模型加载...")
MODEL = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
if os.path.exists(MODEL):
    print(f"  Model: {os.path.basename(MODEL)} ({os.path.getsize(MODEL)/1024**3:.2f}GB)")
    
    # 小测试: 只加载 50 tokens 的 prompt
    try:
        cmd = [
            LLAMA_BENCH,
            "-m", MODEL,
            "-c", "512",
            "-n", "10",
            "-p", "50",
            "-t", "4",
            "--no-mmap"
        ]
        
        print(f"  Running: {' '.join(cmd[:8])}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=LLAMA_BENCH_DIR,
            env=env
        )
        
        print(f"  Exit code: {result.returncode}")
        if result.returncode == 0:
            print("  ✅ Model loads and runs!")
            print(f"  Output:\n{result.stdout[:1500]}")
        else:
            print(f"  Output: {result.stdout[:500]}{result.stderr[:500]}")
            
    except subprocess.TimeoutExpired:
        print("  ⚠️  Timeout (model is large, might take time to load)")
    except Exception as e:
        print(f"  ❌ Error: {e}")
else:
    print(f"  ⚠️  Model not found: {MODEL}")

print("\n" + "=" * 70)
print("FIX COMPLETE")
print("=" * 70)
print(f"""
使用方法:
  1. PowerShell: .\\run-llama-bench.ps1 -m model.gguf -ngl 20
  2. CMD:       run-llama-bench.cmd -m model.gguf -ngl 20
  3. Python:    import os; os.environ["PATH"] = "{LLAMA_BENCH_DIR};{MINGW_BIN};" + os.environ["PATH"]
                 然后 subprocess.run(["{LLAMA_BENCH}", ...], env=os.environ)
""")
