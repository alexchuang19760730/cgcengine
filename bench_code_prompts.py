#!/usr/bin/env python3
"""Test edge_first_proxy with code-containing prompts."""
import asyncio, aiohttp, time, json, sys

PROXY_URL = "http://127.0.0.1:30001/v1/chat/completions"

# Code-containing prompts — _has_code() should return True
CODE_PROMPTS = [
    {
        "label": "fix_python_code",
        "content": "fix this code:\n```python\ndef reverse_string(s):\n    return s[::-1]\n```",
        "expected_family": "fix",
        "expected_has_code": True,
    },
    {
        "label": "debug_traceback",
        "content": 'debug this error:\nTraceback (most recent call last):\n  File "test.py", line 42, in <module>\n    result = func(args)\nTypeError: cannot concatenate str and int',
        "expected_family": "debug",
        "expected_has_code": True,
    },
    {
        "label": "review_js_code",
        "content": "review this code:\n```javascript\nconst fetchData = async (url) => {\n  const res = await fetch(url);\n  return res.json();\n}\n```",
        "expected_family": "review",
        "expected_has_code": True,
    },
    {
        "label": "refactor_java_code",
        "content": "refactor this code:\n```java\npublic class Foo {\n  private int bar;\n  public Foo(int bar) {\n    this.bar = bar;\n  }\n}\n```",
        "expected_family": "refactor",
        "expected_has_code": True,
    },
    {
        "label": "fix_python_def",
        "content": "fix this:\ndef calculate_sum(a, b):\n    return a + b\n\nresult = calculate_sum(5, '10')",
        "expected_family": "fix",
        "expected_has_code": True,
    },
    {
        "label": "debug_with_stack",
        "content": "debug this:\nError: Cannot read property 'map' of undefined\n  at UserList.render (UserList.js:23)\n  at ReactRenderer.render (react.js:145)",
        "expected_family": "debug",
        "expected_has_code": True,
    },
]

async def send_request(session, prompt_content, label):
    """Send a single request and return metrics."""
    data = {
        "model": "gemma-4-26b-a4b-it",
        "messages": [{"role": "user", "content": prompt_content}],
        "max_tokens": 50,
        "stream": True,
    }
    
    start = time.time()
    first_token_time = None
    first_token = None
    spec_status = None
    spec_predicted = None
    cloud_first_token = None
    full_content = ""
    
    try:
        async with session.post(PROXY_URL, json=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and first_token_time is None:
                        first_token_time = (time.time() - start) * 1000
                        first_token = content
                    if content:
                        full_content += content
                    # Check for spec markers in headers/chunk
                    if "x-cgc-speculation" in str(chunk):
                        spec_status = chunk.get("x-cgc-speculation", spec_status)
                        spec_predicted = chunk.get("x-cgc-predicted", spec_predicted)
                    if "x-cgc-cloud-first-token" in str(chunk):
                        cloud_first_token = chunk.get("x-cgc-cloud-first-token", cloud_first_token)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"label": label, "error": str(e), "ttft_ms": -1}
    
    total_time = (time.time() - start) * 1000
    return {
        "label": label,
        "ttft_ms": first_token_time or total_time,
        "total_ms": total_time,
        "first_token": first_token,
        "spec": spec_status,
        "predicted": spec_predicted,
        "cloud_token": cloud_first_token,
        "content_preview": full_content[:100],
    }

async def main():
    print("=== Code Prompt Test (edge_first_proxy + sglang MTP) ===\n")
    
    async with aiohttp.ClientSession() as session:
        # Reset stats first
        try:
            async with session.post("http://127.0.0.1:30001/stats/reset", timeout=aiohttp.ClientTimeout(total=5)) as r:
                print("[INFO] Stats reset\n")
        except:
            pass
        
        results = []
        for prompt in CODE_PROMPTS:
            print(f"--- {prompt['label']} ---")
            r = await send_request(session, prompt["content"], prompt["label"])
            results.append(r)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
            else:
                hit_miss = "HIT" if r["spec"] != "miss" else "MISS"
                print(f"  TTFT: {r['ttft_ms']:.1f}ms, spec: {r['spec'] or 'none'}, {hit_miss}")
                print(f"  first_token: {r['first_token']!r}, predicted: {r['predicted']!r}")
                print(f"  content: {r['content_preview'][:80]}...")
            print()
        
        # Check tracker
        try:
            async with session.get("http://127.0.0.1:30001/acceptance", timeout=aiohttp.ClientTimeout(total=5)) as r:
                tracker = await r.json()
                print(f"=== Tracker ===")
                print(f"  State: {tracker['state']}")
                print(f"  Accept rate: {tracker['global_accept_rate']:.3f}")
                print(f"  Samples: {tracker['global_samples']}")
                print(f"  Hits: {tracker['total_hits']}, Misses: {tracker['total_misses']}")
                print(f"  Family stats: {tracker.get('family_rates', {})}")
        except Exception as e:
            print(f"Tracker check failed: {e}")
    
    # Summary
    print(f"\n=== Summary ===")
    hits = sum(1 for r in results if r.get("spec") and r["spec"] != "miss")
    misses = sum(1 for r in results if r.get("spec") == "miss")
    errors = sum(1 for r in results if "error" in r)
    ttfts = [r["ttft_ms"] for r in results if "ttft_ms" in r and r["ttft_ms"] > 0]
    print(f"  Total: {len(results)}, Hits: {hits}, Misses: {misses}, Errors: {errors}")
    if ttfts:
        print(f"  TTFT: min={min(ttfts):.1f}ms, avg={sum(ttfts)/len(ttfts):.1f}ms, max={max(ttfts):.1f}ms")

if __name__ == "__main__":
    asyncio.run(main())
