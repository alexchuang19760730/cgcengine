#!/usr/bin/env python3
"""
CGC Server vs CLI 對比測試腳本
同時跑 Server 模式和 CLI 模式，比較結果，確定問題是出在 server 配置還是模型/量化
"""
import subprocess
import json
import time
import sys
import os

# 配置
MODEL_PATH = "/Users/alexchuang/Documents/flashkv-devserver/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"
CLI_PATH = "/Users/alexchuang/Documents/flashkv-devserver/src/llama.cpp/build/bin/llama-cli"
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
NGL = 30  # CLI 用 ngl=30 避免 OOM
CTX = 4096
MAX_TOKENS = 50  # 縮短以加快 CLI 測試
TEMPERATURE = 0.3

# 測試用例
TEST_CASES = [
    {
        "name": "T1: Short EN Q&A",
        "prompt": "2+2 is? Answer with one word.",
        "check": lambda c: "4" in c and "2+2 is" not in c,
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
        "check": lambda c: "def " in c and c.count("```") <= 2,
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
        "prompt": "請用一段中文說明為什麼巴黎會成為法國的政治與文化中心。",
        "check": lambda c: len(c) >= 30 and "<think>" not in c,
        "expected": "輸出長度 >= 30，無 <think> 標籤"
    },
]

def test_server(tc):
    """Server 模式測試"""
    try:
        import urllib.request
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
        
        passed = tc["check"](content)
        return {
            "content": content,
            "decode_tps": decode_tps,
            "prompt_tps": prompt_tps,
            "elapsed": elapsed,
            "passed": passed,
            "error": None
        }
    except Exception as e:
        return {"content": "", "decode_tps": 0, "prompt_tps": 0, "elapsed": 0, "passed": False, "error": str(e)}

def test_cli(tc):
    """CLI 模式測試（upstream 原始配置）"""
    try:
        cmd = [
            CLI_PATH,
            "-m", MODEL_PATH,
            "-ngl", str(NGL),
            "-c", str(CTX),
            "--temp", str(TEMPERATURE),
            "-n", str(MAX_TOKENS),
            "-p", tc["prompt"],
            "--no-mmap",
            "--no-display-prompt"
        ]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start
        
        # CLI 輸出包含 prompt 和生成的文本，需要提取生成的部分
        output = result.stdout
        # 去掉 prompt 部分（CLI 會重複 prompt）
        if tc["prompt"] in output:
            content = output.split(tc["prompt"])[-1].strip()
        else:
            content = output.strip()
        
        # 計算速度（粗略估計）
        tokens_generated = len(content.split()) if content else 0
        decode_tps = tokens_generated / elapsed if elapsed > 0 else 0
        
        passed = tc["check"](content)
        return {
            "content": content,
            "decode_tps": decode_tps,
            "prompt_tps": 0,  # CLI 不提供 prefill 速度
            "elapsed": elapsed,
            "passed": passed,
            "error": None if result.returncode == 0 else result.stderr[:200]
        }
    except subprocess.TimeoutExpired:
        return {"content": "", "decode_tps": 0, "prompt_tps": 0, "elapsed": 300, "passed": False, "error": "Timeout"}
    except Exception as e:
        return {"content": "", "decode_tps": 0, "prompt_tps": 0, "elapsed": 0, "passed": False, "error": str(e)}

def main():
    print("=" * 80)
    print("CGC Server vs CLI 對比測試")
    print("=" * 80)
    print(f"模型: {MODEL_PATH}")
    print(f"CLI: {CLI_PATH}")
    print(f"Server: {SERVER_URL}")
    print(f"NGL: {NGL}, CTX: {CTX}, Temp: {TEMPERATURE}, Max Tokens: {MAX_TOKENS}")
    print()
    
    results = []
    
    for tc in TEST_CASES:
        print("-" * 80)
        print(f"測試: {tc['name']}")
        print(f"Prompt: {tc['prompt'][:80]}...")
        print()
        
        # Server 模式
        print("  [Server 模式] 測試中...")
        server_result = test_server(tc)
        server_status = "✅ PASS" if server_result["passed"] else "❌ FAIL"
        print(f"  {server_status} | Decode: {server_result['decode_tps']:.2f} t/s | 耗時: {server_result['elapsed']:.1f}s")
        print(f"  Content: {repr(server_result['content'][:150])}")
        if server_result["error"]:
            print(f"  Error: {server_result['error']}")
        print()
        
        # CLI 模式
        print("  [CLI 模式] 測試中...（可能需要 1-5 分鐘）")
        cli_result = test_cli(tc)
        cli_status = "✅ PASS" if cli_result["passed"] else "❌ FAIL"
        print(f"  {cli_status} | Decode: {cli_result['decode_tps']:.2f} t/s | 耗時: {cli_result['elapsed']:.1f}s")
        print(f"  Content: {repr(cli_result['content'][:150])}")
        if cli_result["error"]:
            print(f"  Error: {cli_result['error']}")
        print()
        
        # 對比
        if server_result["passed"] and cli_result["passed"]:
            comparison = "兩者都正常"
        elif not server_result["passed"] and not cli_result["passed"]:
            comparison = "兩者都失敗 → 模型/量化問題"
        elif not server_result["passed"] and cli_result["passed"]:
            comparison = "⚠️ 只有 Server 失敗 → 我們的 server 配置問題！"
        else:
            comparison = "只有 CLI 失敗 → CLI 配置/chat template 問題"
        
        print(f"  對比: {comparison}")
        print()
        
        results.append({
            "name": tc["name"],
            "server": server_result,
            "cli": cli_result,
            "comparison": comparison
        })
    
    # 總結
    print("=" * 80)
    print("測試總結")
    print("=" * 80)
    print(f"{'測試用例':<25} {'Server':<10} {'CLI':<10} {'對比結果'}")
    print("-" * 80)
    
    server_pass = 0
    cli_pass = 0
    server_issues = 0
    
    for r in results:
        s = "✅" if r["server"]["passed"] else "❌"
        c = "✅" if r["cli"]["passed"] else "❌"
        print(f"{r['name']:<25} {s:<10} {c:<10} {r['comparison']}")
        if r["server"]["passed"]:
            server_pass += 1
        if r["cli"]["passed"]:
            cli_pass += 1
        if "我們的 server 配置問題" in r["comparison"]:
            server_issues += 1
    
    print("-" * 80)
    print(f"Server 通過: {server_pass}/{len(results)}")
    print(f"CLI 通過: {cli_pass}/{len(results)}")
    print(f"Server 配置問題: {server_issues}/{len(results)}")
    print()
    
    if server_issues > 0:
        print("⚠️  結論: 有測試用例在 CLI 正常但 Server 失敗，說明是我們的 server 配置問題！")
    elif server_pass == cli_pass:
        print("✅ 結論: Server 和 CLI 表現一致，問題可能是模型/量化本身的限制")
    else:
        print("📊 結論: Server 和 CLI 表現不同，需要進一步分析")
    
    # 保存結果
    with open("/tmp/cgc_server_vs_cli_comparison.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果已保存到: /tmp/cgc_server_vs_cli_comparison.json")

if __name__ == "__main__":
    main()
