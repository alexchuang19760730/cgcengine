#!/usr/bin/env python3
"""校准 Gemma4 26B 首 token 预测规则。

直接发请求到 sglang (绕过 proxy), 记录实际首 token,
然后更新 edge_first_proxy 的模式规则。
"""
import asyncio
import aiohttp
import json
import time

SGLANG_URL = "http://localhost:30001/v1/chat/completions"

# 测试各种 prompt 类型, 记录 Gemma4 实际首 token
CALIBRATION_PROMPTS = [
    # 代码补全
    ("py_def", [{"role": "user", "content": "def calculate_sum(a, b):"}]),
    ("py_class", [{"role": "user", "content": "class UserProfile:"}]),
    ("py_import", [{"role": "user", "content": "import json\nimport os\n"}]),
    ("py_self", [{"role": "user", "content": "self.get_user_data()"}]),
    ("py_return", [{"role": "user", "content": "return self.result"}]),
    ("js_const", [{"role": "user", "content": "const result = await fetch"}]),
    ("js_func", [{"role": "user", "content": "function handleSubmit(event)"}]),
    ("js_export", [{"role": "user", "content": "export default function"}]),
    # 聊天 - 代码相关
    ("write_code", [{"role": "user", "content": "Write a Python function to reverse a string"}]),
    ("write_code2", [{"role": "user", "content": "Write a Python function to check if a number is prime"}]),
    ("fix_bug", [{"role": "user", "content": "Fix this bug: IndexError: list index out of range"}]),
    ("explain", [{"role": "user", "content": "Explain how this code works: def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"}]),
    ("debug", [{"role": "user", "content": "Debug this error: TypeError: 'NoneType' object is not subscriptable"}]),
    ("list", [{"role": "user", "content": "List all Python design patterns"}]),
    ("algo", [{"role": "user", "content": "What is the time complexity of binary search?"}]),
    # 聊天 - 通用
    ("generic1", [{"role": "user", "content": "Hello, how are you?"}]),
    ("generic2", [{"role": "user", "content": "What is machine learning?"}]),
    ("generic3", [{"role": "user", "content": "Tell me about Python decorators"}]),
]


async def get_first_token(session, messages, max_tokens=5):
    """发送请求, 获取前几个 token。"""
    payload = {
        "model": "gemma-4-26b-a4b-it",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.monotonic()
    first_token = None
    first_5_tokens = []
    
    try:
        async with session.post(SGLANG_URL, json=payload) as resp:
            async for line in resp.content:
                line = line.decode().strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    choices = obj.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            if first_token is None:
                                first_token = content
                            if len(first_5_tokens) < 5:
                                first_5_tokens.append(content)
                except:
                    continue
    except Exception as e:
        return None, [], str(e)
    
    elapsed = round((time.monotonic() - t0) * 1000, 1)
    return first_token, first_5_tokens, f"{elapsed}ms"


async def main():
    print("=" * 70)
    print("Gemma4 26B 首 Token 校准")
    print("=" * 70)
    
    results = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        for name, messages in CALIBRATION_PROMPTS:
            first_token, first_5, info = await get_first_token(session, messages)
            results[name] = {
                "first_token": first_token,
                "first_5": first_5,
                "info": info,
            }
            # 显示首 token (repr 以看到空白字符)
            ft_repr = repr(first_token) if first_token else "None"
            print(f"  {name:15s}  first={ft_repr:20s}  tokens={''.join(first_5[:3])!r}  ({info})")
    
    # 汇总首 token 频率
    print("\n" + "=" * 70)
    print("首 Token 频率分析")
    print("=" * 70)
    
    token_freq = {}
    for name, data in results.items():
        ft = data["first_token"]
        if ft:
            # 取第一个非空白字符组
            stripped = ft.strip()
            if stripped:
                token_freq[stripped] = token_freq.get(stripped, 0) + 1
    
    for token, count in sorted(token_freq.items(), key=lambda x: -x[1]):
        print(f"  '{token}': {count}x")
    
    # 生成建议的 pattern rules
    print("\n" + "=" * 70)
    print("建议的 _PROMPT_FAMILY_RULES 更新")
    print("=" * 70)
    
    # 按类别分组
    categories = {
        "code_completion": ["py_def", "py_class", "py_import", "py_self", "py_return",
                           "js_const", "js_func", "js_export"],
        "chat_code": ["write_code", "write_code2", "fix_bug", "explain", "debug", "list", "algo"],
        "chat_generic": ["generic1", "generic2", "generic3"],
    }
    
    for cat, names in categories.items():
        tokens = [results[n]["first_token"] for n in names if results[n]["first_token"]]
        print(f"\n  {cat}:")
        for n in names:
            ft = results[n].get("first_token", "None")
            print(f"    {n:15s} → {ft!r}")
    
    print("\n" + "=" * 70)
    print("完整结果 (JSON)")
    print("=" * 70)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
