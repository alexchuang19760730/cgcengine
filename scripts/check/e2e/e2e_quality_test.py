#!/usr/bin/env python3
"""
CGC End-to-End Quality Test with Upstream Comparison (e2e_quality_test.py)

端到端品質測試 + upstream 對比：
- 模擬真實客戶端（Windows/Claude Code CLI）的裸請求
- 不帶 assistant_prefill、不帶 stop tokens、用預設參數
- 覆蓋 15+ profiles，每個 profile 多個 prompts
- 同時測試我們的 server 和 upstream 原版 server
- 對比輸出品質，標記哪些是我們的修改導致的問題，哪些是模型本身的問題
- 記錄完整輸出，供後續 agent 逐條驗收檢查

用法:
  # 只測我們的 server
  python3 e2e_quality_test.py --server http://127.0.0.1:8080 --output /tmp/e2e_results.json

  # 同時測試我們的 server 和 upstream server（推薦）
  python3 e2e_quality_test.py \
    --server http://127.0.0.1:8080 \
    --upstream-server http://127.0.0.1:8081 \
    --output /tmp/e2e_results.json

  # 只測特定 profiles
  python3 e2e_quality_test.py --server http://127.0.0.1:8080 --profiles coding,math --output /tmp/e2e_results.json

  # 列出所有可用 profiles
  python3 e2e_quality_test.py --list-profiles
"""

import argparse
import json
import time
import requests
import sys
from datetime import datetime


# ============================================================
# Profiles 定義（15+ profiles，每個 profile 多個 prompts）
# ============================================================

PROFILES = {
    # === 原有 7 profiles ===
    "qa-zh": {
        "description": "中文短問答",
        "prompts": [
            "2+2等於多少？只回答數字",
            "15+27等於多少？只回答數字",
            "法國的首都是哪裡？只回答城市名",
            "水的沸點是多少度？只回答數字",
            "一年有多少個月？只回答數字",
        ]
    },
    "longform-zh": {
        "description": "中文長文本生成",
        "prompts": [
            "請用一段中文說明為什麼巴黎會成為法國的政治與文化中心，包括歷史、地理和文化三個方面。",
            "請寫一篇關於人工智慧發展歷史的短文，從1950年代到現在，包括重要的里程碑。",
            "請解釋什麼是量子計算，以及它與傳統計算的區別，用淺顯易懂的語言。",
        ]
    },
    "coding": {
        "description": "代碼生成",
        "prompts": [
            "Write a Python function to calculate fibonacci.",
            "Write a Python function to check if a string is a palindrome.",
            "Write a Python function to sort a list using bubble sort.",
            "Write a Python class for a binary search tree with insert and search methods.",
            "Write a Python function to reverse a linked list.",
        ]
    },
    "math": {
        "description": "數學計算",
        "prompts": [
            "What is 123 * 456? Show your work.",
            "What is the square root of 144?",
            "Solve for x: 2x + 5 = 15",
            "What is 15% of 200?",
            "What is the derivative of x^2 + 3x + 1?",
        ]
    },
    "reasoning": {
        "description": "邏輯推理",
        "prompts": [
            "If all cats are animals and all animals are living things, are all cats living things? Answer yes or no, then explain.",
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
        ]
    },
    "writing": {
        "description": "寫作",
        "prompts": [
            "Write a short poem about spring.",
            "Write a haiku about the ocean.",
            "Write a short story about a robot learning to paint.",
        ]
    },
    "translation": {
        "description": "翻譯",
        "prompts": [
            "Translate 'Hello, how are you?' to French.",
            "Translate '人工智慧正在改變世界' to English.",
            "Translate 'The quick brown fox jumps over the lazy dog' to Chinese.",
        ]
    },

    # === 新增 8+ profiles ===
    "qa-en": {
        "description": "英文短問答",
        "prompts": [
            "2+2 is? Answer with one word.",
            "What is the capital of Japan? Answer with one word.",
            "How many continents are there? Answer with one number.",
            "What is the chemical symbol for gold? Answer with one word.",
        ]
    },
    "coding-debug": {
        "description": "代碼調試",
        "prompts": [
            "Find and fix the bug in this Python code:\n\ndef calculate_average(numbers):\n    total = 0\n    for num in numbers:\n        total = total + num\n    return total / len(numbers)\n\nprint(calculate_average([]))",
            "Why does this code throw an error? Fix it:\n\nmy_list = [1, 2, 3]\nprint(my_list[3])",
        ]
    },
    "coding-explain": {
        "description": "代碼解釋",
        "prompts": [
            "Explain what this Python code does in simple terms:\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
            "Explain the difference between a list and a tuple in Python.",
        ]
    },
    "math-word-problem": {
        "description": "數學應用題",
        "prompts": [
            "A train travels 60 miles per hour. How far will it travel in 2.5 hours?",
            "If a shirt costs $25 and is on sale for 20% off, what is the sale price?",
            "A recipe calls for 2 cups of flour for 12 cookies. How much flour is needed for 18 cookies?",
        ]
    },
    "creative-writing": {
        "description": "創意寫作",
        "prompts": [
            "Write the opening paragraph of a science fiction novel set on Mars.",
            "Write a dialogue between a teacher and a student about the importance of curiosity.",
            "Write a product description for a smartwatch that can read minds.",
        ]
    },
    "summarization": {
        "description": "摘要總結",
        "prompts": [
            "Summarize the following text in 3 sentences:\n\nArtificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term 'artificial intelligence' had previously been used to describe machines that mimic and display human cognitive skills that are associated with the human mind, such as learning and problem-solving. This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.",
            "Summarize the main points of this article:\n\nClimate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, such as through variations in the solar cycle, but since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures. Examples of greenhouse gas emissions that are causing climate change include carbon dioxide and methane. These come from using gasoline for driving a car or coal for heating a building, for example. Clearing land and forests can also release carbon dioxide. Agriculture, oil and gas operations are major sources of methane emissions. Energy, industry, transport, buildings, agriculture and land use are among the main emitters.",
        ]
    },
    "extraction": {
        "description": "信息提取",
        "prompts": [
            "Extract all the names and dates from this text:\n\nAlbert Einstein was born on March 14, 1879, in Ulm, Germany. He published his theory of special relativity in 1905 and general relativity in 1915. He received the Nobel Prize in Physics in 1921. He died on April 18, 1955, in Princeton, New Jersey.",
            "List all the programming languages mentioned in this text:\n\nPython is a versatile language used for web development, data science, and automation. JavaScript is essential for front-end web development. Java remains popular for enterprise applications. C++ is used for system programming and game development. Rust is gaining popularity for systems programming due to its memory safety guarantees.",
        ]
    },
    "role-play": {
        "description": "角色扮演",
        "prompts": [
            "You are a customer service representative. A customer says: 'I received my order but it's damaged. What can I do?' Respond professionally.",
            "You are a tour guide. Welcome a group of tourists to Paris and give them a brief overview of the city.",
        ]
    },
}


# ============================================================
# 測試函數
# ============================================================

def send_prompt(server_url, prompt, max_tokens=512, temperature=0.7, timeout=120):
    """
    發送裸請求（模擬真實客戶端，不帶 assistant_prefill、不帶 stop tokens）
    
    Args:
        server_url: server 地址（如 http://127.0.0.1:8080）
        prompt: 用戶輸入
        max_tokens: 最大生成 token 數
        temperature: 溫度
        timeout: 超時時間（秒）
    
    Returns:
        dict: 包含 content, decode_tps, prompt_tps, elapsed, error 等字段
    """
    start_time = time.time()
    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            return {
                "content": "",
                "decode_tps": 0,
                "prompt_tps": 0,
                "elapsed": elapsed,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "raw_response": response.text[:500]
            }
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        timings = result.get("timings", {})
        decode_tps = timings.get("predicted_per_second", 0)
        prompt_tps = timings.get("prompt_per_second", 0)
        
        return {
            "content": content,
            "decode_tps": decode_tps,
            "prompt_tps": prompt_tps,
            "elapsed": elapsed,
            "error": None,
            "raw_response": None
        }
        
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return {
            "content": "",
            "decode_tps": 0,
            "prompt_tps": 0,
            "elapsed": elapsed,
            "error": "Timeout",
            "raw_response": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "content": "",
            "decode_tps": 0,
            "prompt_tps": 0,
            "elapsed": elapsed,
            "error": str(e),
            "raw_response": None
        }


def compare_outputs(our_result, upstream_result):
    """
    對比我們的 server 和 upstream server 的輸出
    
    Args:
        our_result: 我們的 server 的結果
        upstream_result: upstream server 的結果
    
    Returns:
        dict: 對比結果，包含 difference_type, our_issues, upstream_issues 等
    """
    comparison = {
        "both_error": False,
        "our_error": False,
        "upstream_error": False,
        "both_empty": False,
        "our_empty": False,
        "upstream_empty": False,
        "content_match": False,
        "content_similarity": 0,
        "difference_type": "unknown",
        "our_issues": [],
        "upstream_issues": [],
        "attribution": "unknown",  # our_modification, model_quantization, both, unknown
    }
    
    # 檢查錯誤
    if our_result.get("error") and upstream_result.get("error"):
        comparison["both_error"] = True
        comparison["attribution"] = "both"
        comparison["difference_type"] = "both_error"
    elif our_result.get("error"):
        comparison["our_error"] = True
        comparison["attribution"] = "our_modification"
        comparison["difference_type"] = "our_error_only"
    elif upstream_result.get("error"):
        comparison["upstream_error"] = True
        comparison["attribution"] = "model_quantization"
        comparison["difference_type"] = "upstream_error_only"
    
    # 檢查空輸出
    our_content = our_result.get("content", "")
    upstream_content = upstream_result.get("content", "")
    
    if not our_content and not upstream_content:
        comparison["both_empty"] = True
        comparison["attribution"] = "both"
        comparison["difference_type"] = "both_empty"
    elif not our_content:
        comparison["our_empty"] = True
        comparison["attribution"] = "our_modification"
        comparison["difference_type"] = "our_empty_only"
    elif not upstream_content:
        comparison["upstream_empty"] = True
        comparison["attribution"] = "model_quantization"
        comparison["difference_type"] = "upstream_empty_only"
    
    # 檢查明顯問題（echo、循環、thinking 標籤）
    def check_issues(content):
        issues = []
        if not content:
            return issues
        # echo prompt（輸出包含 prompt 的大部分）
        # 循環檢測（連續重複的短語）
        words = content.split()
        if len(words) > 20:
            # 檢查最後 20 個詞是否有重複模式
            last_20 = words[-20:]
            # 簡單循環檢測：如果最後 10 個詞和前 10 個詞完全相同
            if len(last_20) >= 20 and last_20[:10] == last_20[10:]:
                issues.append("loop_detected")
        # thinking 標籤泄漏
        if "<think>" in content or "</think>" in content or "thinking" in content.lower()[:50]:
            issues.append("thinking_tag_leak")
        # 代碼圍欄循環
        if content.count("```") > 4:
            issues.append("code_fence_loop")
        return issues
    
    comparison["our_issues"] = check_issues(our_content)
    comparison["upstream_issues"] = check_issues(upstream_content)
    
    # 歸因分析
    if comparison["our_issues"] and not comparison["upstream_issues"]:
        comparison["attribution"] = "our_modification"
        if not comparison["difference_type"] or comparison["difference_type"] == "unknown":
            comparison["difference_type"] = "our_issues_only"
    elif comparison["upstream_issues"] and not comparison["our_issues"]:
        comparison["attribution"] = "model_quantization"
        if not comparison["difference_type"] or comparison["difference_type"] == "unknown":
            comparison["difference_type"] = "upstream_issues_only"
    elif comparison["our_issues"] and comparison["upstream_issues"]:
        comparison["attribution"] = "both"
        if not comparison["difference_type"] or comparison["difference_type"] == "unknown":
            comparison["difference_type"] = "both_have_issues"
    
    # 簡單內容相似度（基於共同字符比例）
    if our_content and upstream_content:
        our_set = set(our_content.lower())
        upstream_set = set(upstream_content.lower())
        if our_set or upstream_set:
            intersection = our_set & upstream_set
            union = our_set | upstream_set
            comparison["content_similarity"] = len(intersection) / len(union) if union else 0
        if our_content.strip() == upstream_content.strip():
            comparison["content_match"] = True
            if not comparison["difference_type"] or comparison["difference_type"] == "unknown":
                comparison["difference_type"] = "identical"
    
    return comparison


def run_e2e_test(server_url, upstream_server_url=None, profiles_to_run=None, output_path=None):
    """
    運行端到端品質測試（可選 upstream 對比）
    
    Args:
        server_url: 我們的 server 地址
        upstream_server_url: upstream 原版 server 地址（可選）
        profiles_to_run: 要運行的 profiles 列表（None 表示全部）
        output_path: 輸出 JSON 文件路徑
    
    Returns:
        dict: 完整的測試結果
    """
    has_upstream = upstream_server_url is not None
    
    results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "server_url": server_url,
            "upstream_server_url": upstream_server_url,
            "has_upstream_comparison": has_upstream,
            "test_type": "e2e_quality_test_with_upstream_comparison",
            "description": "End-to-end quality test with bare requests + upstream comparison (no assistant_prefill, no stop tokens, default parameters)",
            "total_profiles": 0,
            "total_prompts": 0,
        },
        "profiles": {},
        "summary": {
            "total_prompts": 0,
            "our_errors": 0,
            "upstream_errors": 0,
            "our_modification_issues": 0,
            "model_quantization_issues": 0,
            "both_issues": 0,
            "identical_outputs": 0,
            "avg_our_decode_tps": 0,
            "avg_upstream_decode_tps": 0,
        }
    }
    
    # 確定要運行的 profiles
    if profiles_to_run is None:
        profiles_to_run = list(PROFILES.keys())
    else:
        profiles_to_run = [p.strip() for p in profiles_to_run.split(",") if p.strip() in PROFILES]
    
    results["test_info"]["total_profiles"] = len(profiles_to_run)
    total_prompts = sum(len(PROFILES[p]["prompts"]) for p in profiles_to_run)
    results["test_info"]["total_prompts"] = total_prompts
    
    print(f"=" * 80)
    print(f"CGC End-to-End Quality Test with Upstream Comparison")
    print(f"=" * 80)
    print(f"Our server: {server_url}")
    if has_upstream:
        print(f"Upstream server: {upstream_server_url}")
    else:
        print(f"Upstream server: NOT PROVIDED (no comparison)")
    print(f"Profiles: {len(profiles_to_run)}")
    print(f"Total prompts: {total_prompts}")
    print(f"=" * 80)
    print()
    
    all_our_decode_tps = []
    all_upstream_decode_tps = []
    
    for profile_name in profiles_to_run:
        profile = PROFILES[profile_name]
        print(f"\n{'=' * 80}")
        print(f"Profile: {profile_name} ({profile['description']})")
        print(f"Prompts: {len(profile['prompts'])}")
        print(f"{'=' * 80}")
        
        profile_results = {
            "description": profile["description"],
            "prompts": [],
            "summary": {
                "total": len(profile["prompts"]),
                "our_errors": 0,
                "upstream_errors": 0,
                "our_modification_issues": 0,
                "model_quantization_issues": 0,
                "both_issues": 0,
                "identical_outputs": 0,
                "avg_our_decode_tps": 0,
                "avg_upstream_decode_tps": 0,
            }
        }
        
        profile_our_decode_tps = []
        profile_upstream_decode_tps = []
        
        for i, prompt in enumerate(profile["prompts"], 1):
            print(f"\n  [{i}/{len(profile['prompts'])}] Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            
            # 測試我們的 server
            our_result = send_prompt(server_url, prompt)
            
            # 測試 upstream server（如果提供）
            upstream_result = None
            if has_upstream:
                upstream_result = send_prompt(upstream_server_url, prompt)
            
            # 對比結果
            comparison = None
            if has_upstream and upstream_result is not None:
                comparison = compare_outputs(our_result, upstream_result)
            
            prompt_result = {
                "prompt": prompt,
                "our_result": our_result,
                "upstream_result": upstream_result,
                "comparison": comparison,
            }
            
            profile_results["prompts"].append(prompt_result)
            
            # 更新統計
            if our_result.get("error"):
                profile_results["summary"]["our_errors"] += 1
                results["summary"]["our_errors"] += 1
            elif our_result.get("decode_tps", 0) > 0:
                profile_our_decode_tps.append(our_result["decode_tps"])
                all_our_decode_tps.append(our_result["decode_tps"])
            
            if has_upstream and upstream_result is not None:
                if upstream_result.get("error"):
                    profile_results["summary"]["upstream_errors"] += 1
                    results["summary"]["upstream_errors"] += 1
                elif upstream_result.get("decode_tps", 0) > 0:
                    profile_upstream_decode_tps.append(upstream_result["decode_tps"])
                    all_upstream_decode_tps.append(upstream_result["decode_tps"])
                
                if comparison is not None:
                    attribution = comparison.get("attribution", "unknown")
                    if attribution == "our_modification":
                        profile_results["summary"]["our_modification_issues"] += 1
                        results["summary"]["our_modification_issues"] += 1
                    elif attribution == "model_quantization":
                        profile_results["summary"]["model_quantization_issues"] += 1
                        results["summary"]["model_quantization_issues"] += 1
                    elif attribution == "both":
                        profile_results["summary"]["both_issues"] += 1
                        results["summary"]["both_issues"] += 1
                    
                    if comparison.get("content_match", False):
                        profile_results["summary"]["identical_outputs"] += 1
                        results["summary"]["identical_outputs"] += 1
            
            # 打印結果
            if our_result.get("error"):
                print(f"    ❌ Our server error: {our_result['error']}")
            else:
                print(f"    ✅ Our server: {our_result['decode_tps']:.2f} t/s | Length: {len(our_result['content'])} chars")
                print(f"    Content preview: {our_result['content'][:100]}{'...' if len(our_result['content']) > 100 else ''}")
            
            if has_upstream and upstream_result is not None:
                if upstream_result.get("error"):
                    print(f"    ❌ Upstream error: {upstream_result['error']}")
                else:
                    print(f"    ✅ Upstream: {upstream_result['decode_tps']:.2f} t/s | Length: {len(upstream_result['content'])} chars")
                    print(f"    Content preview: {upstream_result['content'][:100]}{'...' if len(upstream_result['content']) > 100 else ''}")
                
                if comparison is not None:
                    print(f"    📊 Comparison: {comparison['difference_type']} | Attribution: {comparison['attribution']}")
                    if comparison.get("our_issues"):
                        print(f"    ⚠️  Our issues: {comparison['our_issues']}")
                    if comparison.get("upstream_issues"):
                        print(f"    ⚠️  Upstream issues: {comparison['upstream_issues']}")
        
        # 計算 profile 平均值
        if profile_our_decode_tps:
            profile_results["summary"]["avg_our_decode_tps"] = sum(profile_our_decode_tps) / len(profile_our_decode_tps)
        if profile_upstream_decode_tps:
            profile_results["summary"]["avg_upstream_decode_tps"] = sum(profile_upstream_decode_tps) / len(profile_upstream_decode_tps)
        
        results["profiles"][profile_name] = profile_results
        
        print(f"\n  Profile summary:")
        print(f"    Total: {profile_results['summary']['total']}")
        print(f"    Our errors: {profile_results['summary']['our_errors']}")
        if has_upstream:
            print(f"    Upstream errors: {profile_results['summary']['upstream_errors']}")
            print(f"    Our modification issues: {profile_results['summary']['our_modification_issues']}")
            print(f"    Model quantization issues: {profile_results['summary']['model_quantization_issues']}")
            print(f"    Both issues: {profile_results['summary']['both_issues']}")
            print(f"    Identical outputs: {profile_results['summary']['identical_outputs']}")
        print(f"    Avg our decode: {profile_results['summary']['avg_our_decode_tps']:.2f} t/s")
        if has_upstream:
            print(f"    Avg upstream decode: {profile_results['summary']['avg_upstream_decode_tps']:.2f} t/s")
    
    # 計算整體平均值
    results["summary"]["total_prompts"] = total_prompts
    if all_our_decode_tps:
        results["summary"]["avg_our_decode_tps"] = sum(all_our_decode_tps) / len(all_our_decode_tps)
    if all_upstream_decode_tps:
        results["summary"]["avg_upstream_decode_tps"] = sum(all_upstream_decode_tps) / len(all_upstream_decode_tps)
    
    print(f"\n\n{'=' * 80}")
    print(f"Overall Summary")
    print(f"{'=' * 80}")
    print(f"Total prompts: {results['summary']['total_prompts']}")
    print(f"Our errors: {results['summary']['our_errors']}")
    if has_upstream:
        print(f"Upstream errors: {results['summary']['upstream_errors']}")
        print(f"Our modification issues: {results['summary']['our_modification_issues']}")
        print(f"Model quantization issues: {results['summary']['model_quantization_issues']}")
        print(f"Both issues: {results['summary']['both_issues']}")
        print(f"Identical outputs: {results['summary']['identical_outputs']}")
    print(f"Avg our decode: {results['summary']['avg_our_decode_tps']:.2f} t/s")
    if has_upstream:
        print(f"Avg upstream decode: {results['summary']['avg_upstream_decode_tps']:.2f} t/s")
    print(f"{'=' * 80}")
    
    # 保存結果
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CGC End-to-End Quality Test with Upstream Comparison")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="Our server URL")
    parser.add_argument("--upstream-server", default=None, help="Upstream server URL (optional, for comparison)")
    parser.add_argument("--profiles", default=None, help="Comma-separated list of profiles to run (default: all)")
    parser.add_argument("--output", default="/tmp/cgc_e2e_quality_results.json", help="Output JSON file path")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    
    args = parser.parse_args()
    
    if args.list_profiles:
        print("Available profiles:")
        for name, profile in PROFILES.items():
            print(f"  - {name}: {profile['description']} ({len(profile['prompts'])} prompts)")
        print(f"\nTotal: {len(PROFILES)} profiles, {sum(len(p['prompts']) for p in PROFILES.values())} prompts")
        return
    
    run_e2e_test(args.server, args.upstream_server, args.profiles, args.output)


if __name__ == "__main__":
    main()
