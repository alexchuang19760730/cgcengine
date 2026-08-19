#!/usr/bin/env python3
"""
Check CMake to understand how llama.cpp was built and what it needs.
"""

import os
import sys
import subprocess

CMAKE_CACHE = r"D:\alex\toolchains\llama-build\CMakeCache.txt"

print("=" * 60)
print("CMAKE BUILD ANALYSIS")
print("=" * 60)

# Read CMakeCache.txt
if os.path.exists(CMAKE_CACHE):
    with open(CMAKE_CACHE, 'r') as f:
        content = f.read()
    
    # Find compiler info
    for line in content.split('\n'):
        if 'CMAKE_CXX_COMPILER' in line and not line.startswith('//'):
            print(f"  Compiler: {line}")
        if 'CMAKE_C_COMPILER' in line and not line.startswith('//'):
            print(f"  C Compiler: {line}")
        if 'CMAKE_BUILD_TYPE' in line and not line.startswith('//'):
            print(f"  Build Type: {line}")
    
    # Check for MinGW vs MSVC
    if 'mingw' in content.lower():
        print("\n  ⚠️  Built with MinGW/GCC")
        print("  MinGW binaries need libxxx.dll naming convention")
        print("  Required DLLs typically: libstdc++-6.dll, libgcc_s_seh-1.dll")
    elif 'MSVC' in content or 'cl.exe' in content.lower():
        print("\n  Built with MSVC/Visual Studio")
        print("  MSVC binaries need vcruntime140.dll, msvcp140.dll")

# Check MinGW runtime DLLs
print("\nCHECKING MinGW runtime DLLs...")
mingw_bin = r"D:\alex\toolchains\winlibs-gcc162\mingw64\bin"
required_mingw_dlls = [
    "libstdc++-6.dll",
    "libgcc_s_seh-1.dll",
    "libwinpthread-1.dll",
]
for dll in required_mingw_dlls:
    path = os.path.join(mingw_bin, dll)
    if os.path.exists(path):
        print(f"  ✅ {dll}: found at {path}")
    else:
        print(f"  ❌ {dll}: NOT FOUND")

# Check if these DLLs are needed by llama-bench
print("\nANALYZING llama-bench.exe imports...")
llama_bench = r"D:\alex\toolchains\llama-build\bin\llama-bench.exe"
with open(llama_bench, 'rb') as f:
    data = f.read()

# Search for DLL name patterns
import re
dll_pattern = rb'(?:libstdc\+\+|libgcc|libwinpthread|vcruntime|msvcp|kernel32|advapi32|ws2_32|vulkan|ggml|llama)[\w.]*\.dll'
matches = set()
for match in re.finditer(dll_pattern, data, re.IGNORECASE):
    dll = match.group().decode('ascii', errors='ignore')
    matches.add(dll)

print(f"  Found {len(matches)} DLL references in binary:")
for dll in sorted(matches):
    print(f"    {dll}")

# Check for .dll.a import libraries
print("\nSEARCHING for import libraries...")
for root, dirs, files in os.walk(r"D:\alex\toolchains\llama-build"):
    for f in files:
        if f.endswith('.dll.a') or f.endswith('.lib'):
            print(f"  {os.path.join(root, f)}")

# Try to copy MinGW DLLs to llama-bench directory
print("\nCOPYING required MinGW DLLs to llama-bench directory...")
dest_dir = r"D:\alex\toolchains\llama-build\bin"
for dll in required_mingw_dlls:
    src = os.path.join(mingw_bin, dll)
    dst = os.path.join(dest_dir, dll)
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil
        shutil.copy2(src, dst)
        print(f"  ✅ Copied {dll}")
    elif os.path.exists(dst):
        print(f"  ℹ️  {dll} already exists in dest")
    else:
        print(f"  ❌ {dll} not found in MinGW bin")

# Check for ggml DLL
print("\nSEARCHING for ggml DLL...")
for root, dirs, files in os.walk(r"D:\alex\toolchains\llama-build"):
    for f in files:
        if 'ggml' in f.lower() and (f.endswith('.dll') or f.endswith('.a')):
            print(f"  Found: {os.path.join(root, f)}")
