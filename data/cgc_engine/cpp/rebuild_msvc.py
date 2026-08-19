#!/usr/bin/env python3
"""
使用 MSVC 重新编译 llama.cpp with Vulkan 支持。

步骤:
1. 设置 MSVC 环境
2. 创建新的构建目录 (build-msvc)
3. 使用 CMake 配置 (MSVC + Vulkan)
4. 编译
5. 验证
"""

import os
import sys
import subprocess
import shutil

# 路径配置
LLAMA_SOURCE = r"D:\alex\flashkv0516\CGC-main\Backend\Llama.cpp\llama.cpp"
BUILD_DIR = r"D:\alex\toolchains\llama-build-msvc"
VULKAN_SDK = r"D:\alex\toolchains\VulkanSDK"

# MSVC 路径
MSVC_ROOT = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
MSVC_VERSION = "14.44.35207"  # 可能需要调整
VC_TOOLS = os.path.join(MSVC_ROOT, "VC", "Tools", "MSVC")

def find_msvc_version():
    """查找最新的 MSVC 版本."""
    if os.path.exists(VC_TOOLS):
        versions = sorted(os.listdir(VC_TOOLS), reverse=True)
        if versions:
            return versions[0]
    return None

def find_vctools_version():
    """查找最新的 VCTools 版本."""
    sdk_path = r"C:\Program Files (x86)\Windows Kits\10\Include"
    if os.path.exists(sdk_path):
        versions = sorted(os.listdir(sdk_path), reverse=True)
        if versions:
            return versions[0]
    return None

def run_cmd(cmd, description="", timeout=600):
    """运行命令并捕获输出."""
    print(f"\n{'='*60}")
    print(f"CMD: {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd[:5])}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=LLAMA_SOURCE
        )
        
        if result.returncode == 0:
            print(f"  ✅ Success!")
            # 显示最后几行输出
            output_lines = (result.stdout + result.stderr).split('\n')
            for line in output_lines[-10:]:
                if line.strip():
                    print(f"     {line}")
            return True
        else:
            print(f"  ❌ Failed with code {result.returncode}")
            # 显示错误
            error_lines = (result.stdout + result.stderr).split('\n')
            for line in error_lines[-15:]:
                if line.strip() and ('error' in line.lower() or 'failed' in line.lower()):
                    print(f"     ❌ {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("=" * 70)
    print("MSVC LLAMA.CPP REBUILD WITH VULKAN")
    print("=" * 70)
    
    # 检查源目录
    if not os.path.exists(LLAMA_SOURCE):
        print(f"❌ Source directory not found: {LLAMA_SOURCE}")
        return 1
    
    print(f"✅ Source: {LLAMA_SOURCE}")
    
    # 检查 MSVC
    msvc_version = find_msvc_version()
    if not msvc_version:
        print(f"❌ MSVC not found in {VC_TOOLS}")
        return 1
    print(f"✅ MSVC version: {msvc_version}")
    
    cl_path = os.path.join(VC_TOOLS, msvc_version, "bin", "Hostx64", "x64", "cl.exe")
    if not os.path.exists(cl_path):
        print(f"❌ cl.exe not found: {cl_path}")
        return 1
    print(f"✅ cl.exe: {cl_path}")
    
    # 检查 Vulkan SDK
    if not os.path.exists(VULKAN_SDK):
        print(f"❌ Vulkan SDK not found: {VULKAN_SDK}")
        return 1
    print(f"✅ Vulkan SDK: {VULKAN_SDK}")
    
    vulkan_include = os.path.join(VULKAN_SDK, "Include")
    vulkan_lib = os.path.join(VULKAN_SDK, "Lib")
    
    if not os.path.exists(vulkan_include):
        print(f"❌ Vulkan include not found: {vulkan_include}")
        return 1
    if not os.path.exists(vulkan_lib):
        print(f"❌ Vulkan lib not found: {vulkan_lib}")
        return 1
    
    # 查找 Vulkan 库文件
    vulkan_lib_files = {
        'x64': os.path.join(vulkan_lib, "x64", "vulkan-1.lib"),
    }
    
    vulkan_lib_path = None
    for key, path in vulkan_lib_files.items():
        if os.path.exists(path):
            vulkan_lib_path = path
            print(f"✅ Vulkan lib: {path}")
            break
    
    if not vulkan_lib_path:
        # 尝试在 Lib 根目录查找
        for f in os.listdir(vulkan_lib):
            if 'vulkan' in f.lower() and f.endswith('.lib'):
                vulkan_lib_path = os.path.join(vulkan_lib, f)
                print(f"✅ Found vulkan lib: {vulkan_lib_path}")
                break
    
    if not vulkan_lib_path:
        print(f"❌ Vulkan .lib file not found")
        print(f"   Contents of {vulkan_lib}:")
        for f in os.listdir(vulkan_lib):
            print(f"     {f}")
        return 1
    
    # 步骤 1: 创建干净的构建目录
    print("\n" + "=" * 70)
    print("STEP 1: SETUP BUILD DIRECTORY")
    print("=" * 70)
    
    if os.path.exists(BUILD_DIR):
        print(f"  Removing old build directory...")
        try:
            shutil.rmtree(BUILD_DIR)
            print(f"  ✅ Removed")
        except PermissionError as e:
            print(f"  ⚠️  Cannot remove: {e}")
            print(f"  Trying to use existing...")
    else:
        os.makedirs(BUILD_DIR, exist_ok=True)
        print(f"  ✅ Created {BUILD_DIR}")
    
    # 步骤 2: CMake 配置
    print("\n" + "=" * 70)
    print("STEP 2: CMake CONFIGURATION (MSVC + VULKAN)")
    print("=" * 70)
    
    cmake_config_cmd = [
        "cmake",
        "-B", BUILD_DIR,
        "-S", LLAMA_SOURCE,
        "-G", "Visual Studio 17 2022",
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_VULKAN=ON",
        f"-DVULKAN_INCLUDE_DIR={vulkan_include}",
        f"-DVULKAN_LIBRARY={vulkan_lib_path}",
        "-DGGML_VULKAN_CHECK_RESULTS=OFF",
        "-DGGML_VULKAN_DEBUG=OFF",
        "-DGGML_VULKAN_MEMORY_DEBUG=OFF",
        "-DGGML_OPENMP=OFF",  # 避免 OpenMP 问题
    ]
    
    if not run_cmd(cmake_config_cmd, "CMake Configuration", timeout=120):
        print("\n❌ CMake configuration failed!")
        print("  Checking CMakeCache.txt for errors...")
        cmake_cache = os.path.join(BUILD_DIR, "CMakeCache.txt")
        if os.path.exists(cmake_cache):
            with open(cmake_cache, 'r') as f:
                for line in f:
                    if 'error' in line.lower() or 'vulkan' in line.lower():
                        print(f"    {line.strip()}")
        return 1
    
    # 步骤 3: 编译
    print("\n" + "=" * 70)
    print("STEP 3: BUILD LLAMA.CPP")
    print("=" * 70)
    
    cmake_build_cmd = [
        "cmake",
        "--build", BUILD_DIR,
        "--config", "Release",
        "--parallel", "8",
        "--target", "llama-bench",  # 只编译 llama-bench
    ]
    
    if not run_cmd(cmake_build_cmd, "CMake Build", timeout=600):
        print("\n❌ Build failed!")
        return 1
    
    # 步骤 4: 验证
    print("\n" + "=" * 70)
    print("STEP 4: VERIFY BUILD")
    print("=" * 70)
    
    # 查找编译产物
    bin_dirs = [
        os.path.join(BUILD_DIR, "bin", "Release"),
        os.path.join(BUILD_DIR, "Release"),
        os.path.join(BUILD_DIR, "bin"),
    ]
    
    llama_bench = None
    for bin_dir in bin_dirs:
        candidate = os.path.join(bin_dir, "llama-bench.exe")
        if os.path.exists(candidate):
            llama_bench = candidate
            print(f"  ✅ Found: {candidate}")
            break
    
    if not llama_bench:
        # 全局搜索
        print("  Searching for llama-bench.exe...")
        for root, dirs, files in os.walk(BUILD_DIR):
            for f in files:
                if f == "llama-bench.exe":
                    llama_bench = os.path.join(root, f)
                    print(f"  ✅ Found: {llama_bench}")
                    break
            if llama_bench:
                break
    
    if not llama_bench:
        print("  ❌ llama-bench.exe not found after build")
        return 1
    
    # 运行测试
    print("\n  Testing llama-bench...")
    try:
        result = subprocess.run(
            [llama_bench, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.dirname(llama_bench)
        )
        if result.returncode == 0:
            print("  ✅ llama-bench runs successfully!")
            print(f"  Output: {result.stdout[:500]}")
        else:
            print(f"  ⚠️  Exit code: {result.returncode}")
            print(f"  Output: {(result.stdout or '')[:300]}{(result.stderr or '')[:300]}")
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
    
    # 检查 Vulkan 设备
    print("\n  Checking Vulkan devices...")
    try:
        result = subprocess.run(
            [llama_bench, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=os.path.dirname(llama_bench)
        )
        print(f"  Devices output: {result.stdout[:1000]}")
    except Exception as e:
        print(f"  Device check failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ BUILD COMPLETE!")
    print("=" * 70)
    print(f"\n  Binary location: {llama_bench}")
    print(f"  To run with Vulkan:")
    print(f'    "{llama_bench}" -m model.gguf -dev Vulkan0 -ngl 20')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
