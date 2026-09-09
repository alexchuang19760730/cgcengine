#!/usr/bin/env python3
"""
CGC ngl=99 vs ngl=30 對比測試腳本
在同一個 Server 配置下，對比不同 ngl 值的結果，確定是 ngl 配置的問題還是其他問題
"""
import subprocess
import json
import time
import sys
import os
import urllib.request

# 配置
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MAX_TOKENS = 80
TEMPERATURE = 0.3

# 測試用例
TEST_CASES = [
    {
        "name": "T1: Short EN Q&A",
        "prompt": "2+2 is? Answer with one word.",
        "check": lambda c: "4" in c and c.count("2+2 is") <= 1,
        "expected": "包含 4，無 echo 循環"
    },
    {
        "name": "T2: Short CN Q&A",
        "prompt": "15+27等於多少？只回答數字",
        "check": lambda c: "42" in c and len(c.strip()) > 0,
        "expected": "包含 42，Content 不為空"
    },
    {
        "name": "T3: Code Gen",
        "prompt": "Write a Python function to calculate fibonacci.",
        "check": lambda c: "def " in c and c.count("```") <= 4,
        "expected": "包含函數定義，無代碼 fence 循環"
    },
    {
        "name": "T4: Logic",
        "prompt": "If all cats are animals and all animals are living things, are all cats living things? Answer yes or no.",
        "check": lambda c: ("yes" in c.lower() or "no" in c.lower()) and c.strip() not in ["1.", "1"],
        "expected": "包含 yes/no，不只輸出 1."
    },
    {
        "name": "T5: Long Context",
        "prompt": "請用一段中文說明為什麼巴黎會成為法國的政治與文化中心，包括歷史、地理和文化三個方面。",
        "check": lambda c: len(c) >= 50 and "<think>" not in c,
        "expected": "輸出長度 >= 50，無 <think> 標籤"
    },
]

def test_server(tc):
    """Server 模式測試"""
    try:
        data = {
            "model": "default",
            "messages": [{"role": "user", "content": tc["prompt"]}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }
        req = urllib.request.Request(
            SERVER_URL,
            data=json.dumps(data).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        start = time.time()
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
        elapsed = time.time() - start
        
        content = result["choices"][0]["message"]["content"]
        timings = result.get("timings", {})
        decode_tps = timings.get("predicted_per_second", 0)
        prompt_tps = timings.get("prompt_per_second", 0)
        draft_accept = timings.get("draft_acceptance", 0)
        
        passed = tc["check"](content)
        return {
            "content": content,
            "decode_tps": decode_tps,
            "prompt_tps": prompt_tps,
            "draft_accept": draft_accept,
            "elapsed": elapsed,
            "passed": passed,
            "error": None
        }
    except Exception as e:
        return {"content": "", "decode_tps": 0, "prompt_tps": 0, "draft_accept": 0, "elapsed": 0, "passed": False, "error": str(e)}

def run_all_tests(label):
    """運行所有測試用例"""
    print(f"\n{'='*80}")
    print(f"運行測試: {label}")
    print(f"{'='*80}")
    
    results = []
    for tc in TEST_CASES:
        print(f"\n  [{tc['name']}] 測試中...")
        result = test_server(tc)
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"  {status} | Decode: {result['decode_tps']:.2f} t/s | Prefill: {result['prompt_tps']:.2f} t/s | Draft Accept: {result['draft_accept']:.1%}")
        print(f"  Content: {repr(result['content'][:150])}")
        if result["error"]:
            print(f"  Error: {result['error']}")
        results.append({"name": tc["name"], "result": result})
    
    return results

def main():
    print("=" * 80)
    print("CGC ngl=99 vs ngl=30 對比測試")
    print("=" * 80)
    print(f"Server: {SERVER_URL}")
    print(f"Temp: {TEMPERATURE}, Max Tokens: {MAX_TOKENS}")
    print()
    print("注意：需要手動切換 ngl 配置並重啟 server")
    print("  1. 先在常前 ngl 配置下跑所有測試")
    print("  2. 然後手動重啟 server 用另一個 ngl 配置")
    print("  3. 再跑所有測試")
    print("  4. 最後對比結果")
    print()
    
    # 檢查當前 server 的 ngl 配置
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "llama-server" in line and "grep" not in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "-ngl" and i+1 < len(parts):
                        current_ngl = parts[i+1]
                        print(f"當前 Server ngl 配置: {current_ngl}")
                        break
                break
    except:
        pass
    
    print()
    print("請問要在哪個 ngl 配置下跑測試？")
    print("  1. ngl=99 (全部 GPU)")
    print("  2. ngl=30 (部分 GPU)")
    print("  3. 兩個都跑（需要手動重啟 server）")
    print()
    
    choice = input("請輸入選擇 (1/2/3): ").strip()
    
    if choice == "1":
        results_99 = run_all_tests("ngl=99")
        results_30 = None
    elif choice == "2":
        results_30 = run_all_tests("ngl=30")
        results_99 = None
    elif choice == "3":
        print("\n>>> 第一步：在當前 ngl 配置下跑所有測試")
        results_first = run_all_tests("當前配置")
        
        print("\n>>> 請手動重啟 server 用另一個 ngl 配置，然後按 Enter 繼續...")
        input()
        
        print("\n>>> 第二步：在新 ngl 配置下跑所有測試")
        results_second = run_all_tests("新配置")
        
        # 對比
        print(f"\n{'='*80}")
        print("對比結果")
        print(f"{'='*80}")
        print(f"{'測試用例':<25} {'配置1':<15} {'配置2':<15} {'對比'}")
        print("-" * 80)
        
        for i, tc in enumerate(TEST_CASES):
            r1 = results_first[i]["result"]
            r2 = results_second[i]["result"]
            s1 = "✅" if r1["passed"] else "❌"
            s2 = "✅" if r2["passed"] else "❌"
            d1 = f"{r1['decode_tps']:.1f}"
            d2 = f"{r2['decode_tps']:.1f}"
            
            if r1["passed"] and r2["passed"]:
                comparison = "兩者都正常"
            elif not r1["passed"] and not r2["passed"]:
                comparison = "兩者都失敗"
            elif not r1["passed"] and r2["passed"]:
                comparison = "⚠️ 配置1失敗，配置2正常"
            else:
                comparison = "配置1正常，配置2失敗"
            
            print(f"{tc['name']:<25} {s1} {d1}t/s{'':<5} {s2} {d2}t/s{'':<5} {comparison}")
        
        # 保存結果
        with open("/tmp/cgc_ngl_comparison.json", "w") as f:
            json.dump({
                "config1": results_first,
                "config2": results_second
            }, f, indent=2, ensure_ascii=False)
        print(f"\n結果已保存到: /tmp/cgc_ngl_comparison.json")
        return
    
    # 單一配置的結果
    results = results_99 if results_99 else results_30
    label = "ngl=99" if results_99 else "ngl=30"
    
    print(f"\n{'='*80}")
    print(f"{label} 測試總結")
    print(f"{'='*80}")
    passed = sum(1 for r in results if r["result"]["passed"])
    avg_decode = sum(r["result"]["decode_tps"] for r in results) / len(results)
    print(f"通過: {passed}/{len(results)}")
    print(f"平均 Decode: {avg_decode:.2f} t/s")
    
    with open(f"/tmp/cgc_ngl_{label.replace('=', '')}_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
