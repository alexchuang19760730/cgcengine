#!/usr/bin/env python3
"""
CGC E2E Agent Acceptance Checker (agent_acceptance_checker.py)

由 AI agent 逐條檢查 e2e 測試結果，評估每個 prompt 的輸出品質。
檢查維度：
1. 正確性（答案是否正確）
2. 完整性（輸出是否完整，沒有被截斷）
3. 邏輯性（推理是否合理）
4. 循環檢測（是否有重複循環）
5. echo 檢測（是否回顯 prompt）
6. thinking 標籤檢測（是否有 <think> 標籤泄漏）
7. 代碼質量（代碼是否正確、可運行）
8. 語言正確性（是否用正確的語言回答）
9. upstream 對比（與 upstream 原版 server 的輸出對比）
10. 歸因分析（問題是我們的修改導致的，還是模型/量化本身的問題）

用法:
  python3 agent_acceptance_checker.py --input /tmp/cgc_e2e_quality_results.json --output /tmp/e2e_acceptance_report.json
  python3 agent_acceptance_checker.py --input /tmp/cgc_e2e_quality_results.json --output /tmp/e2e_acceptance_report.json --verbose
"""

import argparse
import json
import re
import sys
from datetime import datetime


# ============================================================
# 規則檢查函數（機器可自動檢測的問題）
# ============================================================

def check_empty_output(content):
    """檢查輸出是否為空或過短"""
    if not content or len(content.strip()) < 5:
        return {
            "issue": "empty_or_too_short",
            "severity": "critical",
            "description": f"輸出為空或過短（長度: {len(content)}）",
            "confidence": 1.0
        }
    return None


def check_echo_prompt(prompt, content):
    """檢查是否回顯 prompt（echo）"""
    if not content:
        return None
    
    prompt_lower = prompt.lower().strip()
    content_lower = content.lower().strip()
    
    # 檢查 content 的前 50% 是否包含 prompt 的大部分內容
    prompt_words = set(prompt_lower.split())
    content_first_half = content_lower[:len(content_lower)//2]
    content_words = set(content_first_half.split())
    
    if len(prompt_words) > 5:
        overlap = len(prompt_words & content_words) / len(prompt_words)
        if overlap > 0.7:
            return {
                "issue": "echo_prompt",
                "severity": "critical",
                "description": f"輸出回顯了 prompt（詞重疊率: {overlap:.0%}）",
                "confidence": 0.9
            }
    return None


def check_loop(content):
    """檢查是否有循環（重複的短語或句子）"""
    if not content or len(content) < 50:
        return None
    
    # 方法 1：檢查最後 100 個字符是否有重複模式
    last_100 = content[-100:] if len(content) > 100 else content
    words = last_100.split()
    
    if len(words) >= 20:
        # 檢查最後 10 個詞是否與前 10 個詞完全相同
        if words[-10:] == words[-20:-10]:
            return {
                "issue": "word_loop",
                "severity": "critical",
                "description": "檢測到詞級別循環（最後 10 個詞與前 10 個詞完全相同）",
                "confidence": 0.95
            }
    
    # 方法 2：檢查是否有連續重複的短語（≥6 字符，重複 ≥3 次）
    phrases = re.findall(r'(.{6,}?)\1{2,}', content)
    if phrases:
        return {
            "issue": "phrase_loop",
            "severity": "high",
            "description": f"檢測到短語循環: '{phrases[0][:30]}...'",
            "confidence": 0.9
        }
    
    # 方法 3：檢查代碼 fence 循環（``` 出現次數異常）
    code_fence_count = content.count("```")
    if code_fence_count > 6:
        return {
            "issue": "code_fence_loop",
            "severity": "high",
            "description": f"代碼 fence 出現次數異常（{code_fence_count} 次）",
            "confidence": 0.85
        }
    
    return None


def check_thinking_tags(content):
    """檢查是否有 <think> 標籤泄漏"""
    if not content:
        return None
    
    if "<think>" in content or "</think>" in content:
        return {
            "issue": "thinking_tag_leak",
            "severity": "high",
            "description": "檢測到 <think> 標籤泄漏到輸出中",
            "confidence": 1.0
        }
    
    # 檢查開頭是否有 "thinking" 或 "reasoning" 相關的 scaffold
    first_50 = content[:50].lower()
    if first_50.startswith("thinking") or first_50.startswith("reasoning"):
        if "\n\n" not in content[:100]:  # 沒有正常的分隔
            return {
                "issue": "thinking_scaffold",
                "severity": "medium",
                "description": "檢測到 thinking scaffold 開頭",
                "confidence": 0.7
            }
    
    return None


def check_truncation(content, max_tokens=512):
    """檢查輸出是否被截斷（達到 max_tokens 限制）"""
    if not content:
        return None
    
    # 粗略估計 token 數（1 token ≈ 4 字符英文，≈ 2 字符中文）
    estimated_tokens = len(content) / 3  # 平均估算
    
    if estimated_tokens > max_tokens * 0.95:
        # 檢查結尾是否不完整（沒有句號、沒有正常結束）
        ending = content[-20:].strip()
        if not ending.endswith(('.', '。', '!', '?', '！', '？', '```', '}')):
            return {
                "issue": "possible_truncation",
                "severity": "medium",
                "description": f"輸出可能被截斷（估計 {estimated_tokens:.0f} tokens，接近 max_tokens={max_tokens}）",
                "confidence": 0.6
            }
    
    return None


def check_code_quality(prompt, content):
    """檢查代碼質量（僅對 coding 相關 profile）"""
    if not content:
        return None
    
    # 判斷是否是代碼生成請求
    code_keywords = ["write", "function", "code", "python", "javascript", "java", "c++", "class", "algorithm", "sort", "fibonacci", "palindrome", "linked list", "binary tree"]
    prompt_lower = prompt.lower()
    is_code_request = any(kw in prompt_lower for kw in code_keywords)
    
    if not is_code_request:
        return None
    
    issues = []
    
    # 檢查是否有代碼塊
    if "```" not in content and "def " not in content and "function " not in content:
        issues.append("no_code_block")
    
    # 檢查是否有語法錯誤（簡單檢查）
    if "def " in content:
        # 檢查函數定義後是否有冒號
        func_defs = re.findall(r'def \w+\([^)]*\)(?!:)', content)
        if func_defs:
            issues.append("python_syntax_error_missing_colon")
    
    if issues:
        return {
            "issue": "code_quality_issues",
            "severity": "medium",
            "description": f"代碼質量問題: {', '.join(issues)}",
            "confidence": 0.7
        }
    
    return None


def check_language_correctness(prompt, content):
    """檢查是否用正確的語言回答"""
    if not content:
        return None
    
    # 簡單檢查：如果 prompt 是中文，content 應該主要是中文
    chinese_chars_prompt = len(re.findall(r'[\u4e00-\u9fff]', prompt))
    chinese_chars_content = len(re.findall(r'[\u4e00-\u9fff]', content))
    
    if chinese_chars_prompt > 10 and len(content) > 20:
        chinese_ratio = chinese_chars_content / len(content)
        if chinese_ratio < 0.1:
            return {
                "issue": "wrong_language",
                "severity": "medium",
                "description": f"prompt 是中文但輸出主要不是中文（中文比例: {chinese_ratio:.0%}）",
                "confidence": 0.7
            }
    
    return None


# ============================================================
# 綜合檢查函數
# ============================================================

def check_single_prompt(prompt, our_result, upstream_result=None, profile_name=""):
    """
    檢查單個 prompt 的輸出品質
    
    Args:
        prompt: 用戶輸入
        our_result: 我們的 server 的結果
        upstream_result: upstream server 的結果（可選）
        profile_name: profile 名稱
    
    Returns:
        dict: 檢查結果
    """
    our_content = our_result.get("content", "") if our_result else ""
    upstream_content = upstream_result.get("content", "") if upstream_result else None
    
    check_result = {
        "prompt": prompt,
        "profile": profile_name,
        "our_content": our_content,
        "upstream_content": upstream_content,
        "our_decode_tps": our_result.get("decode_tps", 0) if our_result else 0,
        "upstream_decode_tps": upstream_result.get("decode_tps", 0) if upstream_result else 0,
        "our_error": our_result.get("error") if our_result else None,
        "upstream_error": upstream_result.get("error") if upstream_result else None,
        "issues": [],
        "score": 1.0,
        "verdict": "pass",
        "attribution": "unknown",
        "agent_notes": "",
    }
    
    # 執行所有規則檢查
    checks = [
        check_empty_output(our_content),
        check_echo_prompt(prompt, our_content),
        check_loop(our_content),
        check_thinking_tags(our_content),
        check_truncation(our_content),
        check_code_quality(prompt, our_content),
        check_language_correctness(prompt, our_content),
    ]
    
    for check in checks:
        if check:
            check_result["issues"].append(check)
    
    # upstream 對比
    if upstream_content is not None:
        if our_error and not upstream_error:
            check_result["attribution"] = "our_modification"
            check_result["issues"].append({
                "issue": "our_server_error",
                "severity": "critical",
                "description": f"我們的 server 出錯但 upstream 正常: {our_error}",
                "confidence": 0.95
            })
        elif upstream_error and not our_error:
            check_result["attribution"] = "model_quantization"
            check_result["issues"].append({
                "issue": "upstream_error",
                "severity": "high",
                "description": f"upstream 出錯但我們的 server 正常: {upstream_error}",
                "confidence": 0.9
            })
        elif our_content and upstream_content:
            # 比較內容相似度
            our_set = set(our_content.lower().split())
            upstream_set = set(upstream_content.lower().split())
            if our_set or upstream_set:
                similarity = len(our_set & upstream_set) / len(our_set | upstream_set)
                check_result["content_similarity"] = similarity
                
                # 如果我們的輸出有問題但 upstream 沒有，歸因為我們的修改
                our_has_critical = any(i["severity"] == "critical" for i in check_result["issues"])
                upstream_has_critical = any([
                    check_empty_output(upstream_content),
                    check_echo_prompt(prompt, upstream_content),
                    check_loop(upstream_content),
                    check_thinking_tags(upstream_content),
                ])
                
                if our_has_critical and not upstream_has_critical:
                    check_result["attribution"] = "our_modification"
                elif our_has_critical and upstream_has_critical:
                    check_result["attribution"] = "both"
                elif not our_has_critical:
                    check_result["attribution"] = "none"
    
    # 計算分數
    critical_count = sum(1 for i in check_result["issues"] if i["severity"] == "critical")
    high_count = sum(1 for i in check_result["issues"] if i["severity"] == "high")
    medium_count = sum(1 for i in check_result["issues"] if i["severity"] == "medium")
    
    score = 1.0
    score -= critical_count * 0.5
    score -= high_count * 0.2
    score -= medium_count * 0.1
    score = max(0.0, min(1.0, score))
    
    check_result["score"] = score
    
    # 判定結果
    if score >= 0.8:
        check_result["verdict"] = "pass"
    elif score >= 0.5:
        check_result["verdict"] = "partial"
    else:
        check_result["verdict"] = "fail"
    
    return check_result


def run_acceptance_check(input_path, output_path, verbose=False):
    """
    運行 agent 驗收檢查
    
    Args:
        input_path: e2e 測試結果 JSON 文件路徑
        output_path: 輸出報告 JSON 文件路徑
        verbose: 是否打印詳細信息
    """
    with open(input_path, "r", encoding="utf-8") as f:
        e2e_results = json.load(f)
    
    report = {
        "report_info": {
            "timestamp": datetime.now().isoformat(),
            "input_file": input_path,
            "test_info": e2e_results.get("test_info", {}),
            "checker_version": "1.0",
        },
        "profiles": {},
        "summary": {
            "total_prompts": 0,
            "pass": 0,
            "partial": 0,
            "fail": 0,
            "avg_score": 0,
            "our_modification_issues": 0,
            "model_quantization_issues": 0,
            "both_issues": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "issue_breakdown": {},
        },
        "failed_cases": [],
        "critical_cases": [],
    }
    
    all_scores = []
    all_checks = []
    
    for profile_name, profile_data in e2e_results.get("profiles", {}).items():
        profile_report = {
            "description": profile_data.get("description", ""),
            "prompts": [],
            "summary": {
                "total": 0,
                "pass": 0,
                "partial": 0,
                "fail": 0,
                "avg_score": 0,
            }
        }
        
        profile_scores = []
        
        for prompt_data in profile_data.get("prompts", []):
            prompt = prompt_data.get("prompt", "")
            our_result = prompt_data.get("our_result", {})
            upstream_result = prompt_data.get("upstream_result")
            
            check = check_single_prompt(prompt, our_result, upstream_result, profile_name)
            profile_report["prompts"].append(check)
            all_checks.append(check)
            profile_scores.append(check["score"])
            all_scores.append(check["score"])
            
            # 統計
            profile_report["summary"]["total"] += 1
            if check["verdict"] == "pass":
                profile_report["summary"]["pass"] += 1
            elif check["verdict"] == "partial":
                profile_report["summary"]["partial"] += 1
            else:
                profile_report["summary"]["fail"] += 1
            
            # 收集失敗和關鍵案例
            if check["verdict"] == "fail":
                report["failed_cases"].append({
                    "profile": profile_name,
                    "prompt": prompt[:100],
                    "score": check["score"],
                    "issues": check["issues"],
                    "our_content": check["our_content"][:200],
                    "attribution": check["attribution"],
                })
            
            if any(i["severity"] == "critical" for i in check["issues"]):
                report["critical_cases"].append({
                    "profile": profile_name,
                    "prompt": prompt[:100],
                    "score": check["score"],
                    "critical_issues": [i for i in check["issues"] if i["severity"] == "critical"],
                    "our_content": check["our_content"][:200],
                    "attribution": check["attribution"],
                })
            
            # 統計問題類型
            for issue in check["issues"]:
                issue_type = issue["issue"]
                report["summary"]["issue_breakdown"][issue_type] = report["summary"]["issue_breakdown"].get(issue_type, 0) + 1
                
                if issue["severity"] == "critical":
                    report["summary"]["critical_issues"] += 1
                elif issue["severity"] == "high":
                    report["summary"]["high_issues"] += 1
                elif issue["severity"] == "medium":
                    report["summary"]["medium_issues"] += 1
            
            # 統計歸因
            if check["attribution"] == "our_modification":
                report["summary"]["our_modification_issues"] += 1
            elif check["attribution"] == "model_quantization":
                report["summary"]["model_quantization_issues"] += 1
            elif check["attribution"] == "both":
                report["summary"]["both_issues"] += 1
            
            if verbose:
                print(f"  [{check['verdict'].upper()}] {profile_name}: {prompt[:50]}... (score: {check['score']:.2f})")
                for issue in check["issues"]:
                    print(f"    - [{issue['severity']}] {issue['issue']}: {issue['description']}")
        
        if profile_scores:
            profile_report["summary"]["avg_score"] = sum(profile_scores) / len(profile_scores)
        
        report["profiles"][profile_name] = profile_report
    
    # 計算整體統計
    report["summary"]["total_prompts"] = len(all_checks)
    report["summary"]["pass"] = sum(1 for c in all_checks if c["verdict"] == "pass")
    report["summary"]["partial"] = sum(1 for c in all_checks if c["verdict"] == "partial")
    report["summary"]["fail"] = sum(1 for c in all_checks if c["verdict"] == "fail")
    if all_scores:
        report["summary"]["avg_score"] = sum(all_scores) / len(all_scores)
    
    # 保存報告
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print(f"\n{'=' * 80}")
    print(f"CGC E2E Agent Acceptance Check Report")
    print(f"{'=' * 80}")
    print(f"Total prompts: {report['summary']['total_prompts']}")
    print(f"Pass: {report['summary']['pass']} ({report['summary']['pass']/report['summary']['total_prompts']*100:.1f}%)")
    print(f"Partial: {report['summary']['partial']} ({report['summary']['partial']/report['summary']['total_prompts']*100:.1f}%)")
    print(f"Fail: {report['summary']['fail']} ({report['summary']['fail']/report['summary']['total_prompts']*100:.1f}%)")
    print(f"Avg score: {report['summary']['avg_score']:.2f}")
    print(f"{'=' * 80}")
    print(f"\nIssue breakdown:")
    for issue_type, count in sorted(report["summary"]["issue_breakdown"].items(), key=lambda x: -x[1]):
        print(f"  {issue_type}: {count}")
    print(f"\nAttribution:")
    print(f"  Our modification issues: {report['summary']['our_modification_issues']}")
    print(f"  Model/quantization issues: {report['summary']['model_quantization_issues']}")
    print(f"  Both: {report['summary']['both_issues']}")
    print(f"\nReport saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="CGC E2E Agent Acceptance Checker")
    parser.add_argument("--input", required=True, help="Input e2e test results JSON file")
    parser.add_argument("--output", default="/tmp/cgc_e2e_acceptance_report.json", help="Output acceptance report JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    run_acceptance_check(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()
