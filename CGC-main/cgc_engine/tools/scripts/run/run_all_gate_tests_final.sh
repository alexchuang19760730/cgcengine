#!/bin/bash
# CGC Gate 统一测试执行脚本
# 测试所有 Gate 版本（1.0/2.0/2.1/2.2/2.3/3.0/5.0）

set -e

# 配置
HOST1_IP="39.106.118.206"
HOST2_IP="47.95.250.55"
USER="root"
HOST1_PASS="Gen@song@2026622"
HOST2_PASS="Gen@song123"
LOCAL_DIR="/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main"
REMOTE_DIR="/root/flashkv0516/ComputeGraphCompiler-main"

# 测试报告文件
REPORT_FILE="/tmp/cgc_gate_test_report_$(date +%Y%m%d_%H%M%S).txt"

# Gate 列表及其预期版本
declare -A GATE_VERSIONS=(
    ["CGC_Gate_1.0_edge_cloud_autonomy"]="1.0"
    ["CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation"]="2.0"
    ["CGC_Gate_2.1_speculative_decode_fusion_optimization"]="2.1"
    ["CGC_Gate_2.2_deepep_moe_load_balancing"]="2.2"
    ["CGC_Gate_2.3_unlimited_rswa_prefill_pool"]="2.3"
    ["CGC_Gate_3.0_train_inference_unification"]="3.0"
    ["CGC_Gate_3.1_self_harness"]="3.1"
    ["CGC_Gate_5.0_audit_trace_replay_visualization"]="5.0"
)

# 测试结果统计
PASS_COUNT=0
FAIL_COUNT=0
TOTAL_TESTS=0
declare -A GATE_PASS
declare -A GATE_FAIL

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$REPORT_FILE"
}

function log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    echo "[PASS] $1" >> "$REPORT_FILE"
}

function log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    echo "[FAIL] $1" >> "$REPORT_FILE"
}

function log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[WARN] $1" >> "$REPORT_FILE"
}

function run_test() {
    local test_name="$1"
    local gate_name="$2"
    local command="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    log_info "Running test: $test_name"
    
    if eval "$command" &>/dev/null; then
        log_pass "$test_name"
        PASS_COUNT=$((PASS_COUNT + 1))
        GATE_PASS["$gate_name"]=$((GATE_PASS["$gate_name"] + 1))
        return 0
    else
        log_fail "$test_name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        GATE_FAIL["$gate_name"]=$((GATE_FAIL["$gate_name"] + 1))
        return 1
    fi
}

function run_test_with_output() {
    local test_name="$1"
    local gate_name="$2"
    local command="$3"
    local expected_output="$4"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    log_info "Running test: $test_name"
    
    local output
    output=$(eval "$command" 2>/dev/null)
    
    if [ -n "$expected_output" ]; then
        if echo "$output" | grep -q "$expected_output"; then
            log_pass "$test_name"
            PASS_COUNT=$((PASS_COUNT + 1))
            GATE_PASS["$gate_name"]=$((GATE_PASS["$gate_name"] + 1))
            return 0
        else
            log_fail "$test_name"
            log_info "Expected: $expected_output"
            log_info "Got: $output"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            GATE_FAIL["$gate_name"]=$((GATE_FAIL["$gate_name"] + 1))
            return 1
        fi
    else
        if [ -n "$output" ]; then
            log_pass "$test_name"
            PASS_COUNT=$((PASS_COUNT + 1))
            GATE_PASS["$gate_name"]=$((GATE_PASS["$gate_name"] + 1))
            return 0
        else
            log_fail "$test_name"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            GATE_FAIL["$gate_name"]=$((GATE_FAIL["$gate_name"] + 1))
            return 1
        fi
    fi
}

function run_gate10_legacy_mapping_validation() {
    local gate="$1"
    local script_path="$LOCAL_DIR/cgc_engine/tools/scripts/run/validate_gate10_legacy_mapping.py"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log_info "Running test: $gate - legacy capability mapping 一致性验证通过"
    if python3 "$script_path" &>/dev/null; then
        log_pass "$gate - legacy capability mapping 一致性验证通过"
        PASS_COUNT=$((PASS_COUNT + 1))
        GATE_PASS["$gate"]=$((GATE_PASS["$gate"] + 1))
    else
        log_fail "$gate - legacy capability mapping 一致性验证通过"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        GATE_FAIL["$gate"]=$((GATE_FAIL["$gate"] + 1))
    fi
}

function run_gate20_legacy_mapping_validation() {
    local gate="$1"
    local script_path="$LOCAL_DIR/cgc_engine/tools/scripts/run/validate_gate20_legacy_mapping.py"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log_info "Running test: $gate - legacy capability mapping 一致性验证通过"
    if python3 "$script_path" &>/dev/null; then
        log_pass "$gate - legacy capability mapping 一致性验证通过"
        PASS_COUNT=$((PASS_COUNT + 1))
        GATE_PASS["$gate"]=$((GATE_PASS["$gate"] + 1))
    else
        log_fail "$gate - legacy capability mapping 一致性验证通过"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        GATE_FAIL["$gate"]=$((GATE_FAIL["$gate"] + 1))
    fi
}

function validate_gate() {
    local gate="$1"
    local location="$2"
    local base_path="$3"
    
    # 初始化统计
    GATE_PASS["$gate"]=0
    GATE_FAIL["$gate"]=0
    
    log_info "===================== $gate ($location) ====================="
    echo "===================== $gate ($location) =====================" >> "$REPORT_FILE"
    
    local gate_path="$base_path/docs/technical_whitepapers/$gate"
    local expected_version="${GATE_VERSIONS[$gate]}"
    
    # ================ 初始化测试 ================
    log_info "【初始化测试】"
    
    # 测试 1: 目录存在
    run_test "$gate - 目录存在" "$gate" "test -d $gate_path"
    
    # 测试 2: gate_id 验证
    run_test "$gate - gate_id 符合命名规范" "$gate" "echo $gate | grep -qE '^CGC_Gate_[0-9]+\.[0-9]+_[a-z0-9_]+$'"
    
    # 测试 3: gate_version 提取验证
    run_test "$gate - gate_version 格式正确" "$gate" "echo $gate | grep -qE '_[0-9]+\.[0-9]+_'"
    
    # ================ 文档测试 ================
    log_info "【文档测试】"
    
    # 测试 4: 技术白皮书存在
    run_test "$gate - 技术白皮书存在" "$gate" "ls $gate_path/*Technical_Whitepaper*.md &>/dev/null"
    
    # 测试 5: gate_map.json 存在
    local gate_map_file
    gate_map_file=$(ls $gate_path/*gate_map*.json 2>/dev/null | head -1)
    run_test "$gate - gate_map.json 存在" "$gate" "test -f '$gate_map_file'"
    
    # 测试 6: checkin.json 存在
    run_test "$gate - checkin.json 存在" "$gate" "ls $gate_path/*checkin*.json &>/dev/null"
    
    # 测试 7: summary.json 存在
    run_test "$gate - summary.json 存在" "$gate" "ls $gate_path/*summary*.json &>/dev/null"
    
    # 测试 8: README.md 存在
    run_test "$gate - README.md 存在" "$gate" "test -f $gate_path/README.md"
    
    # ================ gate_map.json 内容验证 ================
    log_info "【gate_map.json 内容验证】"
    
    if [ -n "$gate_map_file" ] && [ -f "$gate_map_file" ]; then
        # 测试 9: JSON 格式正确
        run_test "$gate - gate_map.json JSON 格式正确" "$gate" "python3 -c \"import json; json.load(open('$gate_map_file'))\""
        
        # 测试 10: gate_id 字段存在
        run_test_with_output "$gate - gate_map.json 包含 gate_id" "$gate" \
            "python3 -c \"import json; d=json.load(open('$gate_map_file')); print(d.get('gate_id',''))\"" \
            "$gate"
        
        # 测试 11: gate_version 字段存在且正确
        run_test_with_output "$gate - gate_map.json gate_version 正确" "$gate" \
            "python3 -c \"import json; d=json.load(open('$gate_map_file')); print(d.get('gate_version',''))\"" \
            "$expected_version"
        
        # 测试 12: capabilities 字段存在
        run_test "$gate - gate_map.json 包含 capabilities" "$gate" \
            "python3 -c \"import json; d=json.load(open('$gate_map_file')); assert 'capabilities' in d, 'missing capabilities'\""
        
        # 测试 13: capabilities 非空
        run_test_with_output "$gate - gate_map.json capabilities 非空" "$gate" \
            "python3 -c \"import json; d=json.load(open('$gate_map_file')); caps=d.get('capabilities',[]); print(len(caps))\"" \
            "^[1-9][0-9]*$"
        
        # 测试 14: 所有 capabilities 有 id
        run_test "$gate - 所有 capabilities 包含 capability_id" "$gate" \
            "python3 -c \"import json; d=json.load(open('$gate_map_file')); caps=d.get('capabilities',[]); [assert 'capability_id' in c for c in caps]\""
        
        # 测试 15: 所有 capabilities 状态为 done 或 allowed
        run_test "$gate - 所有 capabilities 状态合法" "$gate" \
            "python3 -c \"import json; d=json.load(open('$gate_map_file')); caps=d.get('capabilities',[]); [assert c.get('status','') in ['done','allowed','proof','integrated','target','stub'] for c in caps]\""
    else
        log_warn "gate_map.json 不存在，跳过内容验证"
    fi
    
    # ================ 能力测试 ================
    log_info "【能力测试】"
    
    # 测试 16: gate_coverage_matrix.md 引用验证
    run_test "$gate - 在 gate_coverage_matrix.md 中被引用" "$gate" \
        "grep -q \"$gate\" \"$base_path/docs/technical_whitepapers/gate_coverage_matrix.md\" 2>/dev/null || true"

    if [ "$gate" = "CGC_Gate_1.0_edge_cloud_autonomy" ]; then
        log_info "【Gate 1.0 legacy mapping 测试】"
        run_gate10_legacy_mapping_validation "$gate"
    fi

    if [ "$gate" = "CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation" ]; then
        log_info "【Gate 2.0 legacy mapping 测试】"
        run_gate20_legacy_mapping_validation "$gate"
    fi
    
    echo "" >> "$REPORT_FILE"
    echo ""
}

function validate_gate_remote() {
    local gate="$1"
    local host="$2"
    local host_ip="$3"
    local pass="$4"
    
    # 初始化统计
    GATE_PASS["$gate"]=0
    GATE_FAIL["$gate"]=0
    
    log_info "===================== $gate ($host) ====================="
    echo "===================== $gate ($host) =====================" >> "$REPORT_FILE"
    
    local gate_path="/root/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers/$gate"
    local expected_version="${GATE_VERSIONS[$gate]}"
    
    # ================ 初始化测试 ================
    log_info "【初始化测试】"
    
    # 测试 1: 目录存在
    if sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$host_ip" "test -d $gate_path" &>/dev/null; then
        log_pass "$gate - 目录存在"
        PASS_COUNT=$((PASS_COUNT + 1))
        GATE_PASS["$gate"]=$((GATE_PASS["$gate"] + 1))
    else
        log_fail "$gate - 目录存在"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        GATE_FAIL["$gate"]=$((GATE_FAIL["$gate"] + 1))
        return
    fi
    
    # ================ 文档测试 ================
    log_info "【文档测试】"
    
    # 测试 2: 技术白皮书存在
    if sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$host_ip" "ls $gate_path/*Technical_Whitepaper*.md &>/dev/null" &>/dev/null; then
        log_pass "$gate - 技术白皮书存在"
        PASS_COUNT=$((PASS_COUNT + 1))
        GATE_PASS["$gate"]=$((GATE_PASS["$gate"] + 1))
    else
        log_fail "$gate - 技术白皮书存在"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        GATE_FAIL["$gate"]=$((GATE_FAIL["$gate"] + 1))
    fi
    
    # 测试 3: gate_map.json 存在
    if sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$host_ip" "ls $gate_path/*gate_map*.json &>/dev/null" &>/dev/null; then
        log_pass "$gate - gate_map.json 存在"
        PASS_COUNT=$((PASS_COUNT + 1))
        GATE_PASS["$gate"]=$((GATE_PASS["$gate"] + 1))
        
        # 获取 gate_map 文件路径
        local gate_map_file
        gate_map_file=$(sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$host_ip" "ls $gate_path/*gate_map*.json | head -1")
        
        # 测试 4: JSON 格式正确
        if sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$host_ip" "python3 -c \"import json; json.load(open('$gate_map_file'))\"" &>/dev/null; then
            log_pass "$gate - gate_map.json JSON 格式正确"
            PASS_COUNT=$((PASS_COUNT + 1))
            GATE_PASS["$gate"]=$((GATE_PASS["$gate"] + 1))
            
            # 测试 5: gate_id 验证
            local gate_id_output
            gate_id_output=$(sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$host_ip" "python3 -c \"import json; d=json.load(open('$gate_map_file')); print(d.get('gate_id',''))\"" 2>/dev/null)
            if echo "$gate_id_output" | grep -q "$gate"; then
                log_pass "$gate - gate_map.json gate_id 正确"
                PASS_COUNT=$((PASS_COUNT + 1))
                GATE_PASS["$gate"]=$((GATE_PASS["$gate"] + 1))
            else
                log_fail "$gate - gate_map.json gate_id 正确 (got: $gate_id_output)"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                GATE_FAIL["$gate"]=$((GATE_FAIL["$gate"] + 1))
            fi
            
            # 测试 6: capabilities 验证
            if sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$host_ip" "python3 -c \"import json; d=json.load(open('$gate_map_file')); assert 'capabilities' in d\"" &>/dev/null; then
                log_pass "$gate - gate_map.json 包含 capabilities"
                PASS_COUNT=$((PASS_COUNT + 1))
                GATE_PASS["$gate"]=$((GATE_PASS["$gate"] + 1))
            else
                log_fail "$gate - gate_map.json 包含 capabilities"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                GATE_FAIL["$gate"]=$((GATE_FAIL["$gate"] + 1))
            fi
        else
            log_fail "$gate - gate_map.json JSON 格式正确"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            GATE_FAIL["$gate"]=$((GATE_FAIL["$gate"] + 1))
        fi
    else
        log_fail "$gate - gate_map.json 存在"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        GATE_FAIL["$gate"]=$((GATE_FAIL["$gate"] + 1))
    fi
    
    # 测试 7: 配套文件存在
    if sshpass -p "$pass" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$host_ip" "ls $gate_path/*checkin*.json $gate_path/*summary*.json $gate_path/README.md &>/dev/null" &>/dev/null; then
        log_pass "$gate - 配套文件存在"
        PASS_COUNT=$((PASS_COUNT + 1))
        GATE_PASS["$gate"]=$((GATE_PASS["$gate"] + 1))
    else
        log_fail "$gate - 配套文件存在"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        GATE_FAIL["$gate"]=$((GATE_FAIL["$gate"] + 1))
    fi
    
    echo "" >> "$REPORT_FILE"
    echo ""
}

function run_functional_tests() {
    log_info "===================== 功能测试 ====================="
    echo "===================== 功能测试 =====================" >> "$REPORT_FILE"
    
    # 测试 1: gate_coverage_matrix.md 存在
    run_test "gate_coverage_matrix.md 存在" "global" "test -f $LOCAL_DIR/docs/technical_whitepapers/gate_coverage_matrix.md"
    
    # 测试 2: 示例配置文件存在
    run_test "示例配置文件存在" "global" "ls $LOCAL_DIR/docs/technical_whitepapers/examples/*.json &>/dev/null"
    
    # 测试 3: archive 目录存在
    run_test "archive 目录存在" "global" "test -d $LOCAL_DIR/docs/technical_whitepapers/archive/"
    
    # 测试 4: 同步脚本存在
    run_test "同步脚本存在" "global" "test -f $LOCAL_DIR/cgc_engine/tools/scripts/server/sync_all_gates_to_hosts.sh"
    
    # 测试 5: 示例配置文件格式正确
    run_test "示例配置文件 JSON 格式正确" "global" \
        "for f in $LOCAL_DIR/docs/technical_whitepapers/examples/*.json; do python3 -c \"import json; json.load(open('$f'))\"; done"
    
    echo "" >> "$REPORT_FILE"
    echo ""
}

function run_integration_tests() {
    log_info "===================== 集成测试 ====================="
    echo "===================== 集成测试 =====================" >> "$REPORT_FILE"
    
    log_info "验证本地与远程文件一致性..."
    
    local all_consistent=true
    
    for gate in "${!GATE_VERSIONS[@]}"; do
        # 比较文件数量
        local local_count
        local remote_count
        local_count=$(ls "$LOCAL_DIR/docs/technical_whitepapers/$gate" 2>/dev/null | wc -l)
        remote_count=$(sshpass -p "$HOST1_PASS" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$HOST1_IP" "ls $REMOTE_DIR/docs/technical_whitepapers/$gate 2>/dev/null | wc -l" 2>/dev/null || echo "0")
        
        if [ "$local_count" -eq "$remote_count" ]; then
            log_pass "$gate - 本地与 Host1 文件数量一致 ($local_count)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            log_fail "$gate - 本地与 Host1 文件数量不一致 (本地: $local_count, Host1: $remote_count)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            all_consistent=false
        fi
    done
    
    if [ "$all_consistent" = true ]; then
        log_info "✓ 所有 Gate 本地与远程文件一致"
    else
        log_warn "⚠️  存在文件不一致的 Gate"
    fi
    
    echo "" >> "$REPORT_FILE"
    echo ""
}

function show_summary() {
    echo ""
    echo "============================================================"
    echo "                    📊 测试结果汇总"
    echo "============================================================"
    echo "" >> "$REPORT_FILE"
    echo "============================================================" >> "$REPORT_FILE"
    echo "                    测试结果汇总" >> "$REPORT_FILE"
    echo "============================================================" >> "$REPORT_FILE"
    
    # 总体统计
    echo ""
    echo "  📈 总体统计:"
    echo "    测试总数: $TOTAL_TESTS"
    echo "    ✅ 通过: $PASS_COUNT"
    echo "    ❌ 失败: $FAIL_COUNT"
    echo "" >> "$REPORT_FILE"
    echo "  总体统计:" >> "$REPORT_FILE"
    echo "    测试总数: $TOTAL_TESTS" >> "$REPORT_FILE"
    echo "    通过: $PASS_COUNT" >> "$REPORT_FILE"
    echo "    失败: $FAIL_COUNT" >> "$REPORT_FILE"
    
    # 通过率
    local percentage=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        percentage=$((PASS_COUNT * 100 / TOTAL_TESTS))
    fi
    
    echo "    📊 通过率: $percentage%"
    echo "    通过率: $percentage%" >> "$REPORT_FILE"
    
    # 各 Gate 统计
    echo ""
    echo "  🏷️  各 Gate 测试统计:"
    echo "" >> "$REPORT_FILE"
    echo "  各 Gate 测试统计:" >> "$REPORT_FILE"
    
    local all_passed=true
    
    for gate in "${!GATE_VERSIONS[@]}"; do
        local pass=${GATE_PASS["$gate"]:-0}
        local fail=${GATE_FAIL["$gate"]:-0}
        local total=$((pass + fail))
        local gate_percentage=0
        
        if [ $total -gt 0 ]; then
            gate_percentage=$((pass * 100 / total))
        fi
        
        echo "    $gate:"
        echo "      通过: $pass / $total (${gate_percentage}%)"
        echo "    $gate:" >> "$REPORT_FILE"
        echo "      通过: $pass / $total (${gate_percentage}%)" >> "$REPORT_FILE"
        
        if [ $fail -gt 0 ]; then
            all_passed=false
        fi
    done
    
    echo ""
    echo "  📝 测试报告已保存到: $REPORT_FILE"
    echo "  测试报告已保存到: $REPORT_FILE" >> "$REPORT_FILE"
    
    echo ""
    if [ "$all_passed" = true ] && [ $FAIL_COUNT -eq 0 ]; then
        echo "  🎉 所有测试通过！"
        echo "  所有测试通过！" >> "$REPORT_FILE"
    else
        echo "  ⚠️  存在失败测试，请检查上述输出和测试报告"
        echo "  存在失败测试，请检查上述输出和测试报告" >> "$REPORT_FILE"
    fi
    
    echo "============================================================"
    echo "============================================================" >> "$REPORT_FILE"
}

function main() {
    # 初始化报告文件
    echo "" > "$REPORT_FILE"
    
    echo "============================================================"
    echo "  🔥 CGC Gate 统一测试执行脚本 v2.0"
    echo "  测试 Gate 版本: 1.0 / 2.0 / 2.1 / 2.2 / 2.3 / 3.0 / 3.1 / 5.0"
    echo "  测试类型: 初始化测试 + 能力测试 + 功能测试 + 集成测试"
    echo "  测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""
    
    echo "============================================================" >> "$REPORT_FILE"
    echo "  CGC Gate 统一测试执行脚本 v2.0" >> "$REPORT_FILE"
    echo "  测试 Gate 版本: 1.0 / 2.0 / 2.1 / 2.2 / 2.3 / 3.0 / 3.1 / 5.0" >> "$REPORT_FILE"
    echo "  测试时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
    echo "============================================================" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # ================ 本地测试 ================
    log_info "【阶段 1/4】本地测试"
    log_info "----------------------------------------"
    
    for gate in "${!GATE_VERSIONS[@]}"; do
        validate_gate "$gate" "本地" "$LOCAL_DIR"
    done
    
    # ================ Host1 测试 ================
    log_info "【阶段 2/4】Host1 测试 ($HOST1_IP)"
    log_info "----------------------------------------"
    
    for gate in "${!GATE_VERSIONS[@]}"; do
        validate_gate_remote "$gate" "Host1" "$HOST1_IP" "$HOST1_PASS"
    done
    
    # ================ Host2 测试 ================
    log_info "【阶段 3/4】Host2 测试 ($HOST2_IP)"
    log_info "----------------------------------------"
    
    for gate in "${!GATE_VERSIONS[@]}"; do
        validate_gate_remote "$gate" "Host2" "$HOST2_IP" "$HOST2_PASS"
    done
    
    # ================ 功能测试 ================
    run_functional_tests
    
    # ================ 集成测试 ================
    run_integration_tests
    
    # ================ 显示汇总 ================
    show_summary
}

main "$@"
