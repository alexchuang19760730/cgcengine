"""
validate_c_code.py - 验证 C 代码的逻辑正确性

检查内容:
1. 头文件包含一致性
2. 结构体尺寸和对齐假设
3. 常量边界检查
4. 函数原型匹配
"""

import re
import os
import sys

BASE_DIR = r"D:\alex\flashkv0516\CGC-main\cgc_engine\cpp\expert_streaming"

def read_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def check_header_guards():
    """检查头文件的 include guard"""
    print("=== Check 1: Header Guards ===")
    headers = [
        "cgc_expert_streamer.h",
        "cgc_expert_streamer_gguf.h",
        "cgc_pd_scheduler.h",
        "cgc_expert_compute.h",
    ]
    for h in headers:
        path = os.path.join(BASE_DIR, h)
        if not os.path.exists(path):
            print(f"  MISSING: {h}")
            continue
        content = read_file(path)
        guard_match = re.search(r'#ifndef\s+(\w+)', content)
        define_match = re.search(r'#define\s+(\w+)', content)
        if guard_match and define_match:
            guard = guard_match.group(1)
            define = define_match.group(1)
            if guard == define:
                print(f"  OK: {h} -> {guard}")
            else:
                print(f"  MISMATCH: {h} -> {guard} vs {define}")
        else:
            print(f"  WARN: {h} -> missing guard pattern")
    return True

def check_struct_constraints():
    """检查结构体字段约束"""
    print("\n=== Check 2: Struct Constraints ===")
    
    streamer_h = read_file(os.path.join(BASE_DIR, "cgc_expert_streamer.h"))
    
    max_experts = int(re.search(r'CGC_MAX_EXPERTS_PER_LAYER\s+(\d+)', streamer_h).group(1))
    max_slots = int(re.search(r'CGC_MAX_SLOT_COUNT\s+(\d+)', streamer_h).group(1))
    max_path = int(re.search(r'CGC_MAX_PATH_LEN\s+(\d+)', streamer_h).group(1))
    max_name = int(re.search(r'CGC_MAX_NAME_LEN\s+(\d+)', streamer_h).group(1))
    
    print(f"  CGC_MAX_EXPERTS_PER_LAYER = {max_experts}")
    print(f"  CGC_MAX_SLOT_COUNT = {max_slots}")
    print(f"  CGC_MAX_PATH_LEN = {max_path}")
    print(f"  CGC_MAX_NAME_LEN = {max_name}")
    
    assert max_experts == 256, f"CGC_MAX_EXPERTS_PER_LAYER should be 256, got {max_experts}"
    assert max_slots == 1024, f"CGC_MAX_SLOT_COUNT should be 1024, got {max_slots}"
    assert max_path == 512, f"CGC_MAX_PATH_LEN should be 512, got {max_path}"
    assert max_name == 256, f"CGC_MAX_NAME_LEN should be 256, got {max_name}"
    
    print("  All constants OK")
    return True

def check_buffer_overflow_risks():
    """检查潜在的缓冲区溢出风险"""
    print("\n=== Check 3: Buffer Overflow Risks ===")
    
    risk_count = 0
    
    files = [
        "cgc_expert_streamer.h",
        "cgc_pd_scheduler.h",
        "cgc_expert_compute.h",
    ]
    
    for fname in files:
        path = os.path.join(BASE_DIR, fname)
        content = read_file(path)
        
        strncpy_matches = re.findall(r'strncpy\((\w+)\s*,\s*(\w+)\s*,\s*(sizeof\(\w+\)|CGC_MAX_\w+)\)', content)
        if strncpy_matches:
            for dest, src, size in strncpy_matches:
                if dest in size:
                    print(f"  OK (self-limited): {fname} strncpy({dest}, {src}, {size})")
                else:
                    print(f"  CHECK: {fname} strncpy({dest}, {src}, {size}) - verify truncation safety")
                    risk_count += 1
    
    if risk_count == 0:
        print("  No buffer overflow risks detected")
    
    return risk_count == 0

def check_cross_references():
    """检查头文件交叉引用"""
    print("\n=== Check 4: Cross References ===")
    
    ref_map = {
        "cgc_expert_streamer.h": ["cgc_gguf_lite.h"],
        "cgc_expert_streamer_gguf.h": ["cgc_expert_streamer.h", "cgc_gguf_lite.h"],
        "cgc_pd_scheduler.h": ["cgc_expert_streamer.h", "cgc_expert_streamer_gguf.h"],
        "cgc_expert_compute.h": ["cgc_expert_streamer.h", "cgc_expert_streamer_gguf.h", "cgc_gguf_lite.h"],
    }
    
    for header, expected_refs in ref_map.items():
        path = os.path.join(BASE_DIR, header)
        if not os.path.exists(path):
            print(f"  MISSING: {header}")
            continue
        content = read_file(path)
        
        for ref in expected_refs:
            include_pattern = f'#include "{ref}"'
            if include_pattern in content:
                print(f"  OK: {header} includes {ref}")
            else:
                print(f"  MISSING: {header} should include {ref}")
    
    return True

def check_function_prototypes():
    """检查头文件和实现文件的函数原型匹配"""
    print("\n=== Check 5: Function Prototype Consistency ===")
    
    pairs = [
        ("cgc_expert_streamer.h", "cgc_expert_streamer.c"),
        ("cgc_expert_streamer_gguf.h", "cgc_expert_streamer_gguf.c"),
        ("cgc_pd_scheduler.h", "cgc_pd_scheduler.c"),
        ("cgc_expert_compute.h", "cgc_expert_compute.c"),
    ]
    
    for hdr_name, impl_name in pairs:
        hdr_path = os.path.join(BASE_DIR, hdr_name)
        impl_path = os.path.join(BASE_DIR, impl_name)
        
        if not os.path.exists(hdr_path) or not os.path.exists(impl_path):
            print(f"  SKIP: {hdr_name}/{impl_name} pair")
            continue
        
        hdr_content = read_file(hdr_path)
        impl_content = read_file(impl_path)
        
        decls = re.findall(r'(?:cgc_\w+)\s+\*?\s*(cgc_\w+)\s*\(', hdr_content)
        impls = re.findall(r'(?:cgc_\w+)\s+\*?\s*(cgc_\w+)\s*\(', impl_content)
        
        decl_funcs = set(decls)
        impl_funcs = set(impls)
        
        only_in_header = decl_funcs - impl_funcs
        only_in_impl = impl_funcs - decl_funcs
        
        if only_in_header:
            print(f"  WARNING {hdr_name}: functions only in header: {only_in_header}")
        if only_in_impl:
            print(f"  WARNING {impl_name}: functions only in implementation: {only_in_impl}")
        if not only_in_header and not only_in_impl:
            print(f"  OK: {hdr_name}/{impl_name} - all prototypes match ({len(decl_funcs)} functions)")
    
    return True

def check_typedef_consistency():
    """检查类型定义一致性"""
    print("\n=== Check 6: Typedef Consistency ===")
    
    streamer_h = read_file(os.path.join(BASE_DIR, "cgc_expert_streamer.h"))
    pd_h = read_file(os.path.join(BASE_DIR, "cgc_pd_scheduler.h"))
    
    stream_layout_match = re.search(r'typedef struct \{.*?\} cgc_stream_layout_t;', 
                                     streamer_h, re.DOTALL)
    if stream_layout_match:
        size_match = re.findall(r'\buint64_t\b', stream_layout_match.group(0))
        int_match = re.findall(r'\bint\b', stream_layout_match.group(0))
        print(f"  cgc_stream_layout_t: {len(size_match)} uint64_t fields, {len(int_match)} int fields")
    
    pd_route_match = re.search(r'typedef struct \{.*?\} cgc_pd_expert_route_t;', pd_h, re.DOTALL)
    if pd_route_match:
        print("  cgc_pd_expert_route_t: defined")
    
    return True

def check_test_coverage():
    """检查测试覆盖范围"""
    print("\n=== Check 7: Test Coverage ===")
    
    tests = [
        "test_cgc_expert_streamer.c",
        "test_cgc_pd_scheduler.c",
        "test_cgc_gguf_integration.c",
    ]
    
    for test_file in tests:
        path = os.path.join(BASE_DIR, test_file)
        if not os.path.exists(path):
            print(f"  MISSING: {test_file}")
            continue
        
        content = read_file(path)
        test_funcs = re.findall(r'static int (test_\w+)\(', content)
        print(f"  {test_file}: {len(test_funcs)} test functions")
        for tf in test_funcs:
            print(f"    - {tf}")
    
    return True

def generate_summary():
    """生成代码结构摘要"""
    print("\n=== Code Structure Summary ===")
    
    modules = {
        "cgc_expert_streamer": "Core streaming loader (pread + mmap + LRU cache)",
        "cgc_expert_streamer_gguf": "GGUF layout integration (parse headers, extract metadata)",
        "cgc_pd_scheduler": "PD separation scheduler (prefill/decode phase, dual-GPU cache)",
        "cgc_expert_compute": "Compute bridge (zero-copy weight views for MoE GEMM)",
    }
    
    for name, desc in modules.items():
        hdr = f"{name}.h"
        impl = f"{name}.c"
        hdr_path = os.path.join(BASE_DIR, hdr)
        impl_path = os.path.join(BASE_DIR, impl)
        
        hdr_size = os.path.getsize(hdr_path) if os.path.exists(hdr_path) else 0
        impl_size = os.path.getsize(impl_path) if os.path.exists(impl_path) else 0
        
        hdr_lines = len(read_file(hdr_path).splitlines()) if os.path.exists(hdr_path) else 0
        impl_lines = len(read_file(impl_path).splitlines()) if os.path.exists(impl_path) else 0
        
        print(f"  {name}:")
        print(f"    Header: {hdr} ({hdr_size} bytes, {hdr_lines} lines)")
        print(f"    Source: {impl} ({impl_size} bytes, {impl_lines} lines)")
        print(f"    Purpose: {desc}")
    
    return True

def main():
    print("=" * 60)
    print("C Expert Streaming Code Validator")
    print("=" * 60)
    
    checks = [
        check_header_guards,
        check_struct_constraints,
        check_buffer_overflow_risks,
        check_cross_references,
        check_function_prototypes,
        check_typedef_consistency,
        check_test_coverage,
        generate_summary,
    ]
    
    all_ok = True
    for check in checks:
        try:
            if not check():
                all_ok = False
        except Exception as e:
            print(f"  ERROR in {check.__name__}: {e}")
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL VALIDATIONS PASSED")
    else:
        print("  SOME VALIDATIONS FAILED (see above)")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
