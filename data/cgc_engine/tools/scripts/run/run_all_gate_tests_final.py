#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import datetime
import re

# 配置
HOST1_IP = "39.106.118.206"
HOST2_IP = "47.95.250.55"
USER = "root"
HOST1_PASS = "Gen@song@2026622"
HOST2_PASS = "Gen@song123"
LOCAL_DIR = "/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main"
REMOTE_DIR = "/root/flashkv0516/ComputeGraphCompiler-main"

# Gate 列表及其预期版本
GATE_VERSIONS = {
    "CGC_Gate_1.0_edge_cloud_autonomy": "1.0",
    "CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation": "2.0",
    "CGC_Gate_2.1_speculative_decode_fusion_optimization": "2.1",
    "CGC_Gate_2.2_deepep_moe_load_balancing": "2.2",
    "CGC_Gate_2.3_unlimited_rswa_prefill_pool": "2.3",
    "CGC_Gate_3.0_train_inference_unification": "3.0",
    "CGC_Gate_3.1_self_harness": "3.1",
    "CGC_Gate_5.0_audit_trace_replay_visualization": "5.0"
}

# 颜色定义
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

class TestResult:
    def __init__(self):
        self.pass_count = 0
        self.fail_count = 0
        self.total_tests = 0
        self.gate_pass = {}
        self.gate_fail = {}
        
    def record_pass(self, gate_name):
        self.pass_count += 1
        self.total_tests += 1
        self.gate_pass[gate_name] = self.gate_pass.get(gate_name, 0) + 1
        
    def record_fail(self, gate_name):
        self.fail_count += 1
        self.total_tests += 1
        self.gate_fail[gate_name] = self.gate_fail.get(gate_name, 0) + 1

def log_info(msg):
    print(f"{BLUE}[INFO]{NC} {msg}")
    with open(REPORT_FILE, "a") as f:
        f.write(f"[INFO] {msg}\n")

def log_pass(msg):
    print(f"{GREEN}[PASS]{NC} {msg}")
    with open(REPORT_FILE, "a") as f:
        f.write(f"[PASS] {msg}\n")

def log_fail(msg):
    print(f"{RED}[FAIL]{NC} {msg}")
    with open(REPORT_FILE, "a") as f:
        f.write(f"[FAIL] {msg}\n")

def log_warn(msg):
    print(f"{YELLOW}[WARN]{NC} {msg}")
    with open(REPORT_FILE, "a") as f:
        f.write(f"[WARN] {msg}\n")

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip()
    except:
        return False, ""

def run_gate10_legacy_mapping_validation(result, gate):
    script_path = os.path.join(
        LOCAL_DIR,
        "cgc_engine",
        "tools",
        "scripts",
        "run",
        "validate_gate10_legacy_mapping.py",
    )
    ok, output = run_command(f"python3 '{script_path}'")
    if ok:
        log_pass(f"{gate} - legacy capability mapping 一致性验证通过")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - legacy capability mapping 一致性验证通过")
        if output:
            log_info(output)
        result.record_fail(gate)

def run_gate20_legacy_mapping_validation(result, gate):
    script_path = os.path.join(
        LOCAL_DIR,
        "cgc_engine",
        "tools",
        "scripts",
        "run",
        "validate_gate20_legacy_mapping.py",
    )
    ok, output = run_command(f"python3 '{script_path}'")
    if ok:
        log_pass(f"{gate} - legacy capability mapping 一致性验证通过")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - legacy capability mapping 一致性验证通过")
        if output:
            log_info(output)
        result.record_fail(gate)

def validate_gate_local(gate, result):
    log_info(f"===================== {gate} (本地) =====================")
    with open(REPORT_FILE, "a") as f:
        f.write(f"===================== {gate} (本地) =====================\n")
    
    gate_path = os.path.join(LOCAL_DIR, "docs", "technical_whitepapers", gate)
    expected_version = GATE_VERSIONS[gate]
    
    log_info("【初始化测试】")
    
    # 测试 1: 目录存在
    if os.path.isdir(gate_path):
        log_pass(f"{gate} - 目录存在")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - 目录存在")
        result.record_fail(gate)
        return
    
    # 测试 2: gate_id 验证
    if re.match(r'^CGC_Gate_[0-9]+\.[0-9]+_[a-z0-9_]+$', gate):
        log_pass(f"{gate} - gate_id 符合命名规范")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - gate_id 符合命名规范")
        result.record_fail(gate)
    
    # 测试 3: gate_version 提取验证
    if re.search(r'_[0-9]+\.[0-9]+_', gate):
        log_pass(f"{gate} - gate_version 格式正确")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - gate_version 格式正确")
        result.record_fail(gate)
    
    log_info("【文档测试】")
    
    # 测试 4: 技术白皮书存在
    whitepaper_exists = False
    for f in os.listdir(gate_path):
        if "Technical_Whitepaper" in f and f.endswith(".md"):
            whitepaper_exists = True
            break
    if whitepaper_exists:
        log_pass(f"{gate} - 技术白皮书存在")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - 技术白皮书存在")
        result.record_fail(gate)
    
    # 测试 5: gate_map.json 存在
    gate_map_file = None
    for f in os.listdir(gate_path):
        if "gate_map" in f and f.endswith(".json"):
            gate_map_file = os.path.join(gate_path, f)
            break
    if gate_map_file and os.path.isfile(gate_map_file):
        log_pass(f"{gate} - gate_map.json 存在")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - gate_map.json 存在")
        result.record_fail(gate)
    
    # 测试 6: checkin.json 存在
    checkin_exists = False
    for f in os.listdir(gate_path):
        if "checkin" in f and f.endswith(".json"):
            checkin_exists = True
            break
    if checkin_exists:
        log_pass(f"{gate} - checkin.json 存在")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - checkin.json 存在")
        result.record_fail(gate)
    
    # 测试 7: summary.json 存在
    summary_exists = False
    for f in os.listdir(gate_path):
        if "summary" in f and f.endswith(".json"):
            summary_exists = True
            break
    if summary_exists:
        log_pass(f"{gate} - summary.json 存在")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - summary.json 存在")
        result.record_fail(gate)
    
    # 测试 8: README.md 存在
    readme_path = os.path.join(gate_path, "README.md")
    if os.path.isfile(readme_path):
        log_pass(f"{gate} - README.md 存在")
        result.record_pass(gate)
    else:
        log_fail(f"{gate} - README.md 存在")
        result.record_fail(gate)
    
    log_info("【gate_map.json 内容验证】")
    
    if gate_map_file and os.path.isfile(gate_map_file):
        try:
            with open(gate_map_file, "r") as f:
                gate_map = json.load(f)
            
            # 测试 9: JSON 格式正确
            log_pass(f"{gate} - gate_map.json JSON 格式正确")
            result.record_pass(gate)
            
            # 测试 10: gate_id 字段存在
            if gate_map.get("gate_id") == gate:
                log_pass(f"{gate} - gate_map.json 包含 gate_id")
                result.record_pass(gate)
            else:
                log_fail(f"{gate} - gate_map.json 包含 gate_id")
                result.record_fail(gate)
            
            # 测试 11: gate_version 字段存在且正确
            if gate_map.get("gate_version") == expected_version:
                log_pass(f"{gate} - gate_map.json gate_version 正确")
                result.record_pass(gate)
            else:
                log_fail(f"{gate} - gate_map.json gate_version 正确")
                result.record_fail(gate)
            
            # 测试 12: capabilities 字段存在
            if "capabilities" in gate_map:
                log_pass(f"{gate} - gate_map.json 包含 capabilities")
                result.record_pass(gate)
                
                # 测试 13: capabilities 非空
                caps = gate_map.get("capabilities", [])
                if len(caps) > 0:
                    log_pass(f"{gate} - gate_map.json capabilities 非空 ({len(caps)})")
                    result.record_pass(gate)
                    
                    # 测试 14: 所有 capabilities 有 id
                    all_have_id = all("capability_id" in cap for cap in caps)
                    if all_have_id:
                        log_pass(f"{gate} - 所有 capabilities 包含 capability_id")
                        result.record_pass(gate)
                    else:
                        log_fail(f"{gate} - 所有 capabilities 包含 capability_id")
                        result.record_fail(gate)
                    
                    # 测试 15: 所有 capabilities 状态合法
                    valid_statuses = ["done", "allowed", "proof", "integrated", "target", "stub"]
                    all_status_valid = all(cap.get("status", "") in valid_statuses for cap in caps)
                    if all_status_valid:
                        log_pass(f"{gate} - 所有 capabilities 状态合法")
                        result.record_pass(gate)
                    else:
                        log_fail(f"{gate} - 所有 capabilities 状态合法")
                        result.record_fail(gate)
                else:
                    log_fail(f"{gate} - gate_map.json capabilities 非空")
                    result.record_fail(gate)
            else:
                log_fail(f"{gate} - gate_map.json 包含 capabilities")
                result.record_fail(gate)
        except json.JSONDecodeError:
            log_fail(f"{gate} - gate_map.json JSON 格式正确")
            result.record_fail(gate)
    else:
        log_warn("gate_map.json 不存在，跳过内容验证")
    
    log_info("【能力测试】")
    
    # 测试 16: gate_coverage_matrix.md 引用验证
    matrix_path = os.path.join(LOCAL_DIR, "docs", "technical_whitepapers", "gate_coverage_matrix.md")
    if os.path.isfile(matrix_path):
        with open(matrix_path, "r") as f:
            content = f.read()
            if gate in content:
                log_pass(f"{gate} - 在 gate_coverage_matrix.md 中被引用")
                result.record_pass(gate)
            else:
                log_warn(f"{gate} - 未在 gate_coverage_matrix.md 中被引用")

    if gate == "CGC_Gate_1.0_edge_cloud_autonomy":
        log_info("【Gate 1.0 legacy mapping 测试】")
        run_gate10_legacy_mapping_validation(result, gate)

    if gate == "CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation":
        log_info("【Gate 2.0 legacy mapping 测试】")
        run_gate20_legacy_mapping_validation(result, gate)
    
    with open(REPORT_FILE, "a") as f:
        f.write("\n")
    print()

def show_summary(result):
    print()
    print("============================================================")
    print("                    📊 测试结果汇总")
    print("============================================================")
    with open(REPORT_FILE, "a") as f:
        f.write("\n")
        f.write("============================================================")
        f.write("\n                    测试结果汇总\n")
        f.write("============================================================")
        f.write("\n")
    
    print("\n  📈 总体统计:")
    print(f"    测试总数: {result.total_tests}")
    print(f"    ✅ 通过: {result.pass_count}")
    print(f"    ❌ 失败: {result.fail_count}")
    with open(REPORT_FILE, "a") as f:
        f.write("\n  总体统计:\n")
        f.write(f"    测试总数: {result.total_tests}\n")
        f.write(f"    通过: {result.pass_count}\n")
        f.write(f"    失败: {result.fail_count}\n")
    
    percentage = (result.pass_count * 100 // result.total_tests) if result.total_tests > 0 else 0
    print(f"    📊 通过率: {percentage}%")
    with open(REPORT_FILE, "a") as f:
        f.write(f"    通过率: {percentage}%\n")
    
    print("\n  🏷️  各 Gate 测试统计:")
    with open(REPORT_FILE, "a") as f:
        f.write("\n  各 Gate 测试统计:\n")
    
    all_passed = True
    
    for gate in GATE_VERSIONS:
        pass_count = result.gate_pass.get(gate, 0)
        fail_count = result.gate_fail.get(gate, 0)
        total = pass_count + fail_count
        gate_percentage = (pass_count * 100 // total) if total > 0 else 0
        
        print(f"    {gate}:")
        print(f"      通过: {pass_count} / {total} ({gate_percentage}%)")
        with open(REPORT_FILE, "a") as f:
            f.write(f"    {gate}:\n")
            f.write(f"      通过: {pass_count} / {total} ({gate_percentage}%)\n")
        
        if fail_count > 0:
            all_passed = False
    
    print(f"\n  📝 测试报告已保存到: {REPORT_FILE}")
    with open(REPORT_FILE, "a") as f:
        f.write(f"\n  测试报告已保存到: {REPORT_FILE}\n")
    
    print()
    if all_passed and result.fail_count == 0:
        print("  🎉 所有测试通过！")
        with open(REPORT_FILE, "a") as f:
            f.write("  所有测试通过！\n")
    else:
        print("  ⚠️  存在失败测试，请检查上述输出和测试报告")
        with open(REPORT_FILE, "a") as f:
            f.write("  存在失败测试，请检查上述输出和测试报告\n")
    
    print("============================================================")
    with open(REPORT_FILE, "a") as f:
        f.write("============================================================\n")

def main():
    global REPORT_FILE
    REPORT_FILE = f"/tmp/cgc_gate_test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(REPORT_FILE, "w") as f:
        pass
    
    print("============================================================")
    print("  🔥 CGC Gate 统一测试执行脚本 v2.0")
    print("  测试 Gate 版本: 1.0 / 2.0 / 2.1 / 2.2 / 2.3 / 3.0 / 3.1 / 5.0")
    print("  测试类型: 初始化测试 + 能力测试 + 功能测试")
    print(f"  测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("============================================================")
    print()
    
    with open(REPORT_FILE, "a") as f:
        f.write("============================================================\n")
        f.write("  CGC Gate 统一测试执行脚本 v2.0\n")
        f.write("  测试 Gate 版本: 1.0 / 2.0 / 2.1 / 2.2 / 2.3 / 3.0 / 3.1 / 5.0\n")
        f.write(f"  测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("============================================================\n")
        f.write("\n")
    
    result = TestResult()
    
    log_info("【阶段 1/1】本地测试")
    log_info("----------------------------------------")
    
    for gate in GATE_VERSIONS:
        validate_gate_local(gate, result)
    
    show_summary(result)

if __name__ == "__main__":
    main()
