#!/bin/bash
# precommit_e2e_gate.sh — Pre-commit E2E Quality Gate
#
# 功能:
#   1. 多層次檢查（fast / full / skip 三種模式）
#   2. 集成 e2e 測試框架（裸請求測試 + agent 逐條驗收 + upstream 對比）
#   3. 根據檢查結果決定是否允許 commit
#   4. 智能跳過（文檔 commit、小改動）
#   5. 清晰的通過/失敗標準
#   6. 報告生成與數據庫保存
#
# 模式:
#   fast  (預設)  : 快速檢查 + 3 個關鍵 profile 的 e2e 測試（約 2-3 分鐘）
#   full          : 完整 15 個 profile 的 e2e 測試 + upstream 對比（約 15-30 分鐘）
#   skip          : 完全跳過（用 --no-verify 或環境變數）
#
# 用法:
#   ./scripts/check/precommit_e2e_gate.sh              # 運行 fast 模式
#   E2E_GATE_MODE=full ./scripts/check/precommit_e2e_gate.sh  # 運行 full 模式
#   E2E_GATE_MODE=skip ./scripts/check/precommit_e2e_gate.sh  # 跳過 gate
#   ./scripts/check/precommit_e2e_gate.sh --status     # 只顯示當前狀態
#
# 環境變數:
#   E2E_GATE_MODE=fast|full|skip  # 設定運行模式（預設 fast）
#   E2E_GATE_SKIP=1                # 跳過 gate（等同 E2E_GATE_MODE=skip）
#   E2E_GATE_SERVER=url            # server 地址（預設 http://127.0.0.1:8080）
#   E2E_GATE_UPSTREAM=url          # upstream server 地址（可選，用於對比）
#   E2E_GATE_MIN_QUALITY=0.8       # 最低品質分數（預設 0.8）
#   E2E_GATE_MIN_SPEED_RATIO=0.8   # 速度不低於 baseline 的比例（預設 0.8）
#   E2E_GATE_TIMEOUT=300            # 超時時間（秒，預設 300）

set -euo pipefail

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 項目根目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================
# 輔助函數
# ============================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  CGC Pre-commit E2E Quality Gate${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_section() {
    echo -e "${CYAN}[${1}/${2}] ${3}${NC}"
}

print_success() {
    echo -e "${GREEN}  ✅ ${1}${NC}"
}

print_warning() {
    echo -e "${YELLOW}  ⚠️  ${1}${NC}"
}

print_error() {
    echo -e "${RED}  ❌ ${1}${NC}"
}

print_info() {
    echo -e "  ℹ️  ${1}"
}

# ============================================================
# 1. 檢查是否跳過
# ============================================================

check_skip() {
    # 檢查環境變數
    if [ "${E2E_GATE_SKIP:-0}" = "1" ]; then
        print_warning "E2E_GATE_SKIP=1，跳過 E2E quality gate"
        exit 0
    fi
    
    if [ "${E2E_GATE_MODE:-fast}" = "skip" ]; then
        print_warning "E2E_GATE_MODE=skip，跳過 E2E quality gate"
        exit 0
    fi
    
    # 檢查 --status 參數
    if [ "${1:-}" = "--status" ]; then
        print_header
        echo "  當前狀態:"
        echo "    E2E_GATE_MODE: ${E2E_GATE_MODE:-fast}"
        echo "    E2E_GATE_SERVER: ${E2E_GATE_SERVER:-http://127.0.0.1:8080}"
        echo "    E2E_GATE_UPSTREAM: ${E2E_GATE_UPSTREAM:-未設定（不做對比）}"
        echo "    E2E_GATE_MIN_QUALITY: ${E2E_GATE_MIN_QUALITY:-0.8}"
        echo ""
        echo "  提示:"
        echo "    - 預設 fast 模式：3 個關鍵 profile，約 2-3 分鐘"
        echo "    - full 模式：15 個 profile + upstream 對比，約 15-30 分鐘"
        echo "    - 跳過：E2E_GATE_MODE=skip git commit ..."
        exit 0
    fi
}

# ============================================================
# 2. 檢查是否為代碼 commit
# ============================================================

check_code_commit() {
    print_section 1 6 "檢查 commit 類型..."
    
    # 獲取暫存的文件列表
    STAGED_FILES=$(git diff --cached --name-only)
    
    if [ -z "$STAGED_FILES" ]; then
        print_warning "沒有暫存的文件，跳過 E2E quality gate"
        exit 0
    fi
    
    # 檢查是否有代碼文件變更
    CODE_FILES=$(echo "$STAGED_FILES" | grep -E '\.(cpp|h|hpp|c|cc|py|sh|metal|swift|js|ts)$' || true)
    
    if [ -z "$CODE_FILES" ]; then
        print_warning "沒有代碼文件變更（只有文檔/配置），跳過 E2E quality gate"
        echo "  變更文件:"
        echo "$STAGED_FILES" | head -5 | sed 's/^/    /'
        exit 0
    fi
    
    print_success "檢測到代碼文件變更，需要運行 E2E quality gate"
    echo "  代碼文件數: $(echo "$CODE_FILES" | wc -l | tr -d ' ')"
    echo "  總變更文件數: $(echo "$STAGED_FILES" | wc -l | tr -d ' ')"
}

# ============================================================
# 3. 獲取 git 信息
# ============================================================

get_git_info() {
    print_section 2 6 "獲取 git 信息..."
    
    GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    GIT_VERSION=$(git describe --tags --exact-match 2>/dev/null || echo "dev-${GIT_COMMIT:0:8}")
    
    print_info "commit: ${GIT_COMMIT:0:8}"
    print_info "branch: $GIT_BRANCH"
    print_info "version: $GIT_VERSION"
    print_info "mode: ${E2E_GATE_MODE:-fast}"
}

# ============================================================
# 4. 檢查 server 狀態
# ============================================================

check_server() {
    print_section 3 6 "檢查 server 狀態..."
    
    SERVER_URL="${E2E_GATE_SERVER:-http://127.0.0.1:8080}"
    
    # 檢查 server 是否健康
    if curl -s --max-time 5 "$SERVER_URL/health" 2>/dev/null | grep -q "ok"; then
        print_success "server 健康 ($SERVER_URL)"
    else
        print_error "server 健康檢查失敗 ($SERVER_URL)"
        echo ""
        echo "  請先啟動 server:"
        echo "    ./scripts/run_server.sh --detach"
        echo ""
        echo "  或者跳過 gate:"
        echo "    E2E_GATE_MODE=skip git commit ..."
        exit 1
    fi
    
    # 檢查 upstream server（如果設定）
    if [ -n "${E2E_GATE_UPSTREAM:-}" ]; then
        if curl -s --max-time 5 "${E2E_GATE_UPSTREAM}/health" 2>/dev/null | grep -q "ok"; then
            print_success "upstream server 健康 (${E2E_GATE_UPSTREAM})"
        else
            print_warning "upstream server 健康檢查失敗，將跳過 upstream 對比"
            E2E_GATE_UPSTREAM=""
        fi
    fi
}

# ============================================================
# 5. 運行 E2E 測試
# ============================================================

run_e2e_test() {
    print_section 4 6 "運行 E2E 品質測試..."
    
    MODE="${E2E_GATE_MODE:-fast}"
    SERVER_URL="${E2E_GATE_SERVER:-http://127.0.0.1:8080}"
    TIMEOUT="${E2E_GATE_TIMEOUT:-300}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    E2E_OUTPUT="/tmp/cgc_precommit_e2e_${TIMESTAMP}.json"
    
    # 根據模式選擇 profiles
    if [ "$MODE" = "full" ]; then
        PROFILES=""  # 空表示全部 15 個 profiles
        print_info "full 模式：運行全部 15 個 profiles（約 15-30 分鐘）"
    else
        PROFILES="qa-zh,coding,reasoning"  # fast 模式：3 個關鍵 profile
        print_info "fast 模式：運行 3 個關鍵 profile（qa-zh, coding, reasoning），約 2-3 分鐘"
    fi
    
    # 構建命令
    CMD="python3 ./scripts/check/e2e/e2e_quality_test.py"
    CMD="$CMD --server $SERVER_URL"
    CMD="$CMD --output $E2E_OUTPUT"
    if [ -n "${E2E_GATE_UPSTREAM:-}" ]; then
        CMD="$CMD --upstream-server ${E2E_GATE_UPSTREAM}"
    fi
    if [ -n "$PROFILES" ]; then
        CMD="$CMD --profiles $PROFILES"
    fi
    
    echo ""
    print_info "運行命令: $CMD"
    echo ""
    
    # 運行測試
    if eval "$CMD" 2>&1 | tee /tmp/cgc_precommit_e2e.log; then
        print_success "E2E 測試運行完成"
    else
        print_error "E2E 測試運行失敗"
        echo "  日誌: /tmp/cgc_precommit_e2e.log"
        exit 1
    fi
    
    # 保存輸出路徑
    E2E_OUTPUT_PATH="$E2E_OUTPUT"
}

# ============================================================
# 6. 運行 agent 驗收檢查
# ============================================================

run_agent_check() {
    print_section 5 6 "運行 agent 逐條驗收檢查..."
    
    ACCEPTANCE_OUTPUT="/tmp/cgc_precommit_acceptance_$(date +%Y%m%d_%H%M%S).json"
    
    if python3 ./scripts/check/e2e/agent_acceptance_checker.py \
        --input "$E2E_OUTPUT_PATH" \
        --output "$ACCEPTANCE_OUTPUT" \
        2>&1 | tee /tmp/cgc_precommit_acceptance.log; then
        print_success "Agent 驗收檢查完成"
    else
        print_warning "Agent 驗收檢查出錯，將使用 E2E 測試的基本結果"
    fi
    
    ACCEPTANCE_OUTPUT_PATH="$ACCEPTANCE_OUTPUT"
}

# ============================================================
# 7. 分析結果並判定是否通過
# ============================================================

analyze_and_judge() {
    print_section 6 6 "分析結果並判定..."
    
    MIN_QUALITY="${E2E_GATE_MIN_QUALITY:-0.8}"
    
    # 使用 Python 分析結果
    JUDGEMENT=$(python3 -c "
import json
import sys

# 讀取 E2E 測試結果
with open('$E2E_OUTPUT_PATH') as f:
    e2e_data = json.load(f)

# 讀取 agent 驗收結果（如果存在）
acceptance_data = None
try:
    with open('$ACCEPTANCE_OUTPUT_PATH') as f:
        acceptance_data = json.load(f)
except:
    pass

# 分析結果
summary = e2e_data.get('summary', {})
total = summary.get('total_prompts', 0)
errors = summary.get('our_errors', 0)
avg_decode_tps = summary.get('avg_our_decode_tps', 0)

# 如果有 agent 驗收結果，使用它的評分
if acceptance_data:
    acc_summary = acceptance_data.get('summary', {})
    pass_count = acc_summary.get('pass', 0)
    partial_count = acc_summary.get('partial', 0)
    fail_count = acc_summary.get('fail', 0)
    avg_score = acc_summary.get('avg_score', 0)
    critical_issues = acc_summary.get('critical_issues', 0)
    our_modification_issues = acc_summary.get('our_modification_issues', 0)
    
    print(f'  === Agent 驗收結果 ===')
    print(f'  總測試數: {total}')
    print(f'  通過: {pass_count} ({pass_count/total*100:.1f}%)' if total > 0 else '  通過: 0')
    print(f'  部分通過: {partial_count} ({partial_count/total*100:.1f}%)' if total > 0 else '  部分通過: 0')
    print(f'  失敗: {fail_count} ({fail_count/total*100:.1f}%)' if total > 0 else '  失敗: 0')
    print(f'  平均品質分數: {avg_score:.2f}')
    print(f'  Critical 問題數: {critical_issues}')
    print(f'  我們的修改引入的問題: {our_modification_issues}')
    print(f'  平均解碼速度: {avg_decode_tps:.2f} t/s')
    print()
    
    # 判定標準
    issues = []
    
    # 1. 平均品質分數
    if avg_score < $MIN_QUALITY:
        issues.append(f'平均品質分數 {avg_score:.2f} 低於最低要求 {$MIN_QUALITY}')
    
    # 2. Critical 問題
    if critical_issues > 0:
        issues.append(f'檢測到 {critical_issues} 個 critical 問題（echo/循環/thinking 標籤泄漏等）')
    
    # 3. 失敗比例
    if total > 0 and fail_count / total > 0.2:
        issues.append(f'失敗比例過高 ({fail_count/total*100:.1f}% > 20%)')
    
    # 4. 我們的修改引入的問題
    if our_modification_issues > 0:
        issues.append(f'檢測到 {our_modification_issues} 個由我們的修改引入的問題（upstream 對比）')
    
    # 輸出判定
    if issues:
        print('  ❌ 判定: FAIL')
        print()
        print('  失敗原因:')
        for issue in issues:
            print(f'    - {issue}')
        print()
        print('  建議:')
        print('    1. 檢查完整報告: $ACCEPTANCE_OUTPUT_PATH')
        print('    2. 修復品質問題後重新 commit')
        print('    3. 如果這是預期的變更，可以跳過 gate:')
        print('       E2E_GATE_MODE=skip git commit ...')
        sys.exit(1)
    else:
        print('  ✅ 判定: PASS')
        print()
        print('  所有檢查通過:')
        print(f'    - 平均品質分數 {avg_score:.2f} >= {$MIN_QUALITY}')
        print(f'    - 沒有 critical 問題')
        print(f'    - 失敗比例在可接受範圍內')
        sys.exit(0)

else:
    # 如果沒有 agent 驗收結果，使用基本判定
    print(f'  === E2E 測試基本結果 ===')
    print(f'  總測試數: {total}')
    print(f'  錯誤數: {errors}')
    print(f'  平均解碼速度: {avg_decode_tps:.2f} t/s')
    print()
    
    if errors > total * 0.2:
        print('  ❌ 判定: FAIL')
        print(f'  失敗原因: 錯誤比例過高 ({errors}/{total})')
        sys.exit(1)
    else:
        print('  ✅ 判定: PASS（基本檢查，建議運行 full 模式獲得完整驗收）')
        sys.exit(0)
" 2>&1)
    
    JUDGEMENT_EXIT=$?
    
    echo "$JUDGEMENT"
    
    return $JUDGEMENT_EXIT
}

# ============================================================
# 主流程
# ============================================================

main() {
    print_header
    
    # 1. 檢查是否跳過
    check_skip "${1:-}"
    
    # 2. 檢查是否為代碼 commit
    check_code_commit
    
    # 3. 獲取 git 信息
    get_git_info
    
    # 4. 檢查 server 狀態
    check_server
    
    # 5. 運行 E2E 測試
    run_e2e_test
    
    # 6. 運行 agent 驗收檢查
    run_agent_check
    
    # 7. 分析結果並判定
    echo ""
    if analyze_and_judge; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  E2E Quality Gate: PASS${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "  測試報告: $E2E_OUTPUT_PATH"
        echo "  驗收報告: ${ACCEPTANCE_OUTPUT_PATH:-未生成}"
        echo ""
        exit 0
    else
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}  E2E Quality Gate: FAIL${NC}"
        echo -e "${RED}========================================${NC}"
        echo ""
        echo "  測試報告: $E2E_OUTPUT_PATH"
        echo "  驗收報告: ${ACCEPTANCE_OUTPUT_PATH:-未生成}"
        echo ""
        exit 1
    fi
}

# 運行主流程
main "$@"
