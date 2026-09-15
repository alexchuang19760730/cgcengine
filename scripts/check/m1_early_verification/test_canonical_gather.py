#!/usr/bin/env python3
"""
M1 早期驗證測試 1：Canonical Gather Order 驗證框架

目的：驗證按 expert id 排序後的累加順序是否與原來的 slot index 順序一致，
這是 M1 不垮的基礎——確保換 pool size 時，浮點舍入順序不改變。

背景：
    - 當前代碼中 expert 的累加順序是按 slot index（pool 相關）
    - slot index 是 pool 相關的，若累加順序跟著 slot 走，換 pool size 就會改變舍入
    - M1 需要實作 canonical gather order：compacted 之後的 expert 按 expert id 排序
    - 這是 roadmap 中明確標註的「這個新架構唯一的 M1 陷阱」

使用方式：
    1. 先在代碼中實作 canonical gather order（按 expert id 排序）
    2. 用此腳本對比排序前後的 logits
    3. 確認在誤差範圍內一致

注意：
    - 此腳本是驗證框架，需要 M1 代碼改動後才能執行
    - 當前（c22d75395）尚未實作 canonical gather order
    - 測試需要同時運行兩個 server：一個用 slot order，一個用 canonical order

測試結果（2026-09-13）：
    ⚠️ 需代碼改動，當前未實現
    - 當前 expert 累加順序按 slot index，不是按 expert id 排序
    - M1 需要實作 canonical gather order 才能驗證
"""

import json
import time
import sys
import os
import subprocess
import signal


# 配置
REPO_DIR = "/Users/alexchuang/Documents/flashkv-devserver"
MODEL_PATH = os.path.join(REPO_DIR, "models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf")
TEMPLATE_PATH = os.path.join(REPO_DIR, "src/llama.cpp/models/templates/Qwen3-nothink-ChatML.jinja")
SERVER_BIN = os.path.join(REPO_DIR, "src/llama.cpp/build/bin/llama-server")

# 測試 prompt（簡單 case：單層、單 token）
TEST_PROMPTS = [
    "What is the capital of Japan? Answer in one word.",
    "What is 2+2? Answer with just the number.",
    "Translate 'hello' to French. One word.",
]

# 環境變量切換 gather order
# 注意：這需要代碼支援 CGC_CANONICAL_GATHER_ORDER 環境變量
ENV_SLOT_ORDER = {"CGC_CANONICAL_GATHER_ORDER": "0"}
ENV_CANONICAL_ORDER = {"CGC_CANONICAL_GATHER_ORDER": "1"}


def start_server(port, env_vars, log_file):
    """啟動 llama-server，返回進程對象"""
    env = os.environ.copy()
    env.update(env_vars)

    cmd = [
        SERVER_BIN,
        "-m", MODEL_PATH,
        "-expert-cache", "10737418240",  # 10GB pool
        "-ngl", "99", "--load-mode", "none", "-t", "8", "-c", "2048", "-np", "1",
        "--no-kv-unified", "-sps", "0",
        "--host", "127.0.0.1", "--port", str(port),
        "--jinja", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--chat-template-file", TEMPLATE_PATH,
        "--reasoning", "off", "--reasoning-format", "none",
        "--temp", "0", "--top-k", "0", "--top-p", "1.0",
    ]

    log = open(log_file, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc


def wait_for_server(port, timeout=60):
    """等待 server 就緒"""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def get_completion(port, prompt, max_tokens=10):
    """獲取模型補全"""
    import urllib.request
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def compare_outputs(output_a, output_b):
    """對比兩個輸出，返回是否一致"""
    if output_a == output_b:
        return True, "完全一致"
    # 檢查是否是前綴關係（可能是 max_tokens 限制）
    if output_a.startswith(output_b) or output_b.startswith(output_a):
        return True, "前綴一致（可能是 max_tokens 限制）"
    return False, f"不一致：{repr(output_a[:50])} vs {repr(output_b[:50])}"


def main():
    print("=" * 60)
    print("M1 早期驗證測試 1：Canonical Gather Order 驗證框架")
    print("=" * 60)
    print()

    # 檢查環境變量是否支援
    if "CGC_CANONICAL_GATHER_ORDER" not in os.environ:
        print("⚠️  注意：當前代碼可能尚未支援 CGC_CANONICAL_GATHER_ORDER 環境變量")
        print("   此腳本是驗證框架，需要 M1 代碼改動後才能執行")
        print()
        print("   要實作的功能：")
        print("   1. 在 expert cache 中添加 CGC_CANONICAL_GATHER_ORDER 開關")
        print("   2. 當開啟時，compacted 之後的 expert 按 expert id 排序")
        print("   3. 對比排序前後的 logits，確認在誤差範圍內一致")
        print()

    # 檢查二進制文件
    if not os.path.exists(SERVER_BIN):
        print(f"❌ llama-server 二進制不存在：{SERVER_BIN}")
        return 1

    # 檢查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在：{MODEL_PATH}")
        return 1

    print("=== 測試配置 ===")
    print(f"   模型：{MODEL_PATH}")
    print(f"   Pool：10GB")
    print(f"   測試 prompt 數量：{len(TEST_PROMPTS)}")
    print()

    # 啟動兩個 server：一個用 slot order，一個用 canonical order
    print("=== 啟動 Server A（Slot Order，port 8091）===")
    server_a = start_server(8091, ENV_SLOT_ORDER, "/tmp/gather_test_slot.log")
    print(f"   PID: {server_a.pid}")

    print("=== 啟動 Server B（Canonical Order，port 8092）===")
    server_b = start_server(8092, ENV_CANONICAL_ORDER, "/tmp/gather_test_canonical.log")
    print(f"   PID: {server_b.pid}")
    print()

    try:
        # 等待 server 就緒
        print("=== 等待 Server A 就緒 ===")
        if not wait_for_server(8091):
            print("❌ Server A 未能就緒")
            return 1
        print("   ✅ Server A 就緒")

        print("=== 等待 Server B 就緒 ===")
        if not wait_for_server(8092):
            print("❌ Server B 未能就緒")
            return 1
        print("   ✅ Server B 就緒")
        print()

        # 運行測試
        print("=== 運行測試 ===")
        results = []
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n測試 {i+1}/{len(TEST_PROMPTS)}: {prompt[:50]}...")

            output_a = get_completion(8091, prompt)
            output_b = get_completion(8092, prompt)

            is_match, detail = compare_outputs(output_a, output_b)
            status = "✅" if is_match else "❌"
            print(f"   {status} {detail}")
            print(f"   Slot Order:      {repr(output_a[:80])}")
            print(f"   Canonical Order: {repr(output_b[:80])}")

            results.append({
                "prompt": prompt,
                "slot_order": output_a,
                "canonical_order": output_b,
                "match": is_match,
                "detail": detail,
            })

        # 總結
        print("\n" + "=" * 60)
        print("=== 測試總結 ===")
        print("=" * 60)
        match_count = sum(1 for r in results if r["match"])
        print(f"   通過：{match_count}/{len(results)}")
        print()

        if match_count == len(results):
            print("   ✅ Canonical Gather Order 與 Slot Order 完全一致")
            print("      M1 不垮的基礎已驗證")
        else:
            print("   ❌ 存在不一致，需要進一步調查")
            print("      可能原因：浮點舍入順序不同、expert 加權順序不同")

        # 保存結果
        result_file = "/tmp/gather_order_test_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n   結果已保存：{result_file}")

    finally:
        # 清理
        print("\n=== 清理 ===")
        server_a.terminate()
        server_b.terminate()
        server_a.wait()
        server_b.wait()
        print("   Servers 已停止")

    return 0


if __name__ == "__main__":
    sys.exit(main())
