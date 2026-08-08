#!/usr/bin/env python3
"""Test parallel preflight + wrong first token correction.

Starts a mock cloud server (simulating sglang streaming) and the real edge_first_proxy.
Tests both HIT and MISS scenarios, verifying:
  - HIT: client gets [prediction][cloud tokens 2..N] (no duplicate first token)
  - MISS: client gets [wrong_prediction][correction marker][correct_first_token][rest]
"""
import asyncio
import json
import os
import sys
import time
import subprocess
import signal
import urllib.request
import http.server
import threading

REPO = "/Users/alexchuang/Documents/flashkv0516"
PROXY_SCRIPT = os.path.join(REPO, "app", "servers", "edge_first_proxy.py")

# --- Mock Cloud Server ---
# Simulates sglang's /v1/chat/completions streaming endpoint.
# Configurable to return predetermined first token to trigger HIT or MISS.

MOCK_CLOUD_RESPONSES = {
    # prompt substring -> list of tokens to stream back
    "hello": ["Hello", " there", "!", " How", " can", " I", " help?"],
    "write": ["I", "'ll", " write", " a", " function", " for", " you."],
    "debug": ["Let", "'s", " debug", " this", " step", " by", " step."],
    "fix": ["The", " issue", " is", " in", " the", " loop", " logic."],
}

class MockCloudHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len))
        messages = body.get("messages", [])
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "").lower()
                break

        # Find matching response
        tokens = ["Default", " response", " here", "."]
        for key, toks in MOCK_CLOUD_RESPONSES.items():
            if key in user_msg:
                tokens = toks
                break

        stream = body.get("stream", False)
        if not stream:
            # Non-streaming response
            content = "".join(tokens)
            resp = {
                "id": "mock-cloud-1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mock-cloud",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(tokens), "total_tokens": 10 + len(tokens)},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        else:
            # Streaming response (SSE)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # Simulate cloud TTFT (small delay before first token)
            time.sleep(0.030)  # 30ms cloud TTFT

            # First chunk: role + first token
            first_chunk = {
                "id": "mock-cloud-1",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "mock-cloud",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": tokens[0]}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(first_chunk)}\n\n".encode())
            self.wfile.flush()

            # Subsequent tokens
            for tok in tokens[1:]:
                time.sleep(0.005)  # 5ms per token
                chunk = {
                    "id": "mock-cloud-1",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "mock-cloud",
                    "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            # Done
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {"data": [{"id": "mock-cloud"}]}
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress logs


def start_mock_cloud(port):
    server = http.server.HTTPServer(("127.0.0.1", port), MockCloudHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def start_proxy(port, cloud_url):
    """Start edge_first_proxy via runpy wrapper (fixes PYTHONPATH issue)."""
    wrapper_code = f"""
import sys, os
REPO = {REPO!r}
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "app"))
os.chdir(REPO)
import runpy
runpy.run_path({PROXY_SCRIPT!r}, run_name="__main__")
"""
    cmd = [sys.executable, "-c", wrapper_code,
           "--port", str(port),
           "--cloud-url", cloud_url]
    env = os.environ.copy()
    env["EDGE_FIRST_ENABLED"] = "1"
    env["CLOUD_URL"] = cloud_url
    # Don't set PYTHONPATH - it breaks site-packages
    env.pop("PYTHONPATH", None)
    # Redirect stderr to file to avoid PIPE deadlock
    stderr_file = open("/tmp/proxy_stderr.log", "w")
    proc = subprocess.Popen(cmd, env=env, stderr=stderr_file, stdout=subprocess.DEVNULL)
    return proc


def send_streaming_request(proxy_url, user_msg, max_tokens=10):
    """Send a streaming chat completion request and return all chunks + timing."""
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        proxy_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    chunks = []
    ttft = None
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            buffer = ""
            for byte_chunk in iter(lambda: resp.read(1024), b""):
                buffer += byte_chunk.decode()
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.split("\n"):
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                continue
                            try:
                                obj = json.loads(data_str)
                                tt = time.monotonic() - t0
                                if ttft is None:
                                    ttft = tt
                                chunks.append((tt, obj))
                            except json.JSONDecodeError:
                                pass
    except Exception as e:
        print(f"  Request error: {e}")
    return chunks, ttft


def extract_text(chunks):
    """Extract accumulated text from SSE chunks."""
    text = ""
    for _, obj in chunks:
        choices = obj.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                text += content
    return text


def extract_metadata(chunks):
    """Extract CGC-specific metadata from chunks."""
    metadata = []
    for _, obj in chunks:
        if "x-cgc-speculation" in obj:
            metadata.append({
                "type": obj["x-cgc-speculation"],
                "predicted": obj.get("x-cgc-predicted", ""),
                "t": _,
            })
    return metadata


def main():
    cloud_port = 19200
    proxy_port = 19201

    print("=" * 70)
    print("Parallel Preflight + Wrong First Token Correction Test")
    print("=" * 70)

    # 1. Start mock cloud
    print("\n[1] Starting mock cloud server on port", cloud_port)
    cloud = start_mock_cloud(cloud_port)
    cloud_url = f"http://127.0.0.1:{cloud_port}"
    print("    Mock cloud ready (TTFT=30ms, 5ms/token)")

    # 2. Start proxy
    print(f"\n[2] Starting edge_first_proxy on port {proxy_port}")
    proc = start_proxy(proxy_port, cloud_url)
    time.sleep(3)  # wait for proxy to start

    if proc.poll() is not None:
        print("    ERROR: proxy died immediately!")
        stderr = proc.stderr.read().decode()
        print(f"    stderr: {stderr[:500]}")
        return

    # Check if proxy is responding
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/health", timeout=5)
        print("    Proxy is healthy")
    except Exception as e:
        print(f"    Proxy health check failed: {e}")
        stderr = proc.stderr.read().decode()
        print(f"    stderr: {stderr[:500]}")
        proc.terminate()
        return

    proxy_url = f"http://127.0.0.1:{proxy_port}"

    # 3. Test scenarios
    results = []

    # Scenario A: "hello" — cloud returns "Hello" (capital H)
    #   Proxy might predict "The" (fallback) or something else → likely MISS
    print("\n[3] Testing scenarios...")
    print("-" * 50)

    test_cases = [
        ("hello world", "Hello there! How can I help?", "MISS expected (fallback 'The' vs cloud 'Hello')"),
        ("write a function", "I'll write a function for you.", "MISS expected (fallback vs cloud 'I')"),
        ("debug this code", "Let's debug this step by step.", "MISS expected (fallback vs cloud 'Let')"),
        ("fix the bug", "The issue is in the loop logic.", "Possible HIT (cloud starts with 'The')"),
    ]

    for i, (prompt, expected, description) in enumerate(test_cases):
        print(f"\n  Test {i+1}: '{prompt}'")
        print(f"  Expected cloud response: '{expected}'")
        print(f"  {description}")

        chunks, ttft = send_streaming_request(proxy_url, prompt)
        text = extract_text(chunks)
        meta = extract_metadata(chunks)

        print(f"  TTFT: {ttft*1000:.0f}ms")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Accumulated text: '{text}'")
        print(f"  Expected text:    '{expected}'")

        if meta:
            for m in meta:
                print(f"  CGC metadata: type={m['type']} predicted='{m['predicted']}'")
        else:
            print(f"  CGC metadata: none (HIT or no speculation)")

        # Verify correctness
        if text == expected:
            print(f"  ✅ CORRECT: text matches expected")
            results.append(("PASS", prompt, text, expected))
        else:
            # Check if expected is a substring (miss with correction)
            if expected in text:
                print(f"  ⚠️  PARTIAL: expected text is substring of actual (miss + correction, extra prefix)")
                results.append(("PARTIAL", prompt, text, expected))
            else:
                print(f"  ❌ WRONG: text does not match")
                results.append(("FAIL", prompt, text, expected))

    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for status, prompt, actual, expected in results:
        emoji = "✅" if status == "PASS" else ("⚠️" if status == "PARTIAL" else "❌")
        print(f"  {emoji} {status}: '{prompt}' → '{actual[:50]}...'")

    # 5. Check proxy logs for correction behavior
    print("\n[4] Proxy stderr (correction logs):")
    print("-" * 50)
    try:
        with open("/tmp/proxy_stderr.log", "r") as f:
            for line in f:
                if "CORRECTION" in line or "spec HIT" in line or "spec MISS" in line or "TTFT" in line:
                    print(f"  {line.rstrip()}")
    except Exception as e:
        print(f"  (error reading log: {e})")

    # Cleanup
    proc.terminate()
    proc.wait()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
