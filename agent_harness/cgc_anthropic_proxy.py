#!/usr/bin/env python3
"""
Anthropic → CGC Protocol Proxy
让 Claude Code CLI 通过 Mac M4 的 CGC edge_server (Qwen3.6-35B) 运行

用法:
  python cgc_anthropic_proxy.py --port 8082 --cgc-url http://192.168.101.87:1237

  然后:
  ANTHROPIC_BASE_URL=http://127.0.0.1:8082 ANTHROPIC_API_KEY=dummy claude

协议转换:
  Claude Code CLI → Anthropic /v1/messages → proxy → CGC /v1/cgc/resume → Mac M4
"""

import argparse
import json
import os
import sys
import time
import uuid
import re

# ---------------------------------------------------------------------------
# Lightweight HTTP server (stdlib only, no deps)
# ---------------------------------------------------------------------------
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError


def messages_to_prompt(messages: list[dict]) -> str:
    """把 Anthropic messages 数组转成单个 prompt 字符串"""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # multimodal: 提取 text 部分
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            content = "\n".join(texts)
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def sse_format(event: str, data: dict) -> str:
    """格式化为 Anthropic SSE 格式"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class ProxyHandler(BaseHTTPRequestHandler):
    cgc_url: str = "http://192.168.101.87:1237"
    model_name: str = "qwen3.6-35b-mtp"

    def log_message(self, format, *args):
        # 静默日志，避免刷屏
        pass

    def do_GET(self):
        """处理 /v1/models 等 GET 请求"""
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "cgc-mac-m4"
                }]
            }).encode())
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "proxy": "cgc-anthropic"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """处理 /v1/messages (Anthropic API)"""
        if self.path != "/v1/messages":
            self.send_response(404)
            self.end_headers()
            return

        # 读取请求体
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "invalid json"}).encode())
            return

        messages = req.get("messages", [])
        model = req.get("model", self.model_name)
        max_tokens = req.get("max_tokens", 4096)
        stream = req.get("stream", False)
        system = req.get("system", "")

        # 构建 prompt
        prompt_parts = []
        if system:
            if isinstance(system, list):
                system_text = "\n".join(
                    b.get("text", "") for b in system if isinstance(b, dict)
                )
            else:
                system_text = str(system)
            prompt_parts.append(f"[system]\n{system_text}")

        prompt_parts.append(messages_to_prompt(messages))
        prompt = "\n\n".join(prompt_parts)

        print(f"[proxy] request: model={model}, max_tokens={max_tokens}, "
              f"stream={stream}, prompt_len={len(prompt)}", flush=True)

        if stream:
            self._handle_streaming(prompt, max_tokens, model)
        else:
            self._handle_non_streaming(prompt, max_tokens, model)

    def _handle_streaming(self, prompt: str, max_tokens: int, model: str):
        """流式响应: CGC SSE → Anthropic SSE"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # 发送 message_start
        self.wfile.write(sse_format("message_start", {
            "type": "message",
            "id": msg_id,
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }).encode())
        self.wfile.flush()

        # 发送 content_block_start
        self.wfile.write(sse_format("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }).encode())
        self.wfile.flush()

        # 调用 CGC resume
        cgc_payload = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "seed": 42
        }).encode()

        req = Request(
            f"{self.cgc_url}/v1/cgc/resume",
            data=cgc_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            resp = urlopen(req, timeout=600)
            full_text = ""

            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event.get("event") == "token":
                    token = event.get("t", "")
                    full_text += token
                    # 发送 token delta
                    self.wfile.write(sse_format("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": token}
                    }).encode())
                    self.wfile.flush()

                elif event.get("event") == "summary":
                    n_decoded = event.get("n_decoded", 0)
                    decode_tps = event.get("decode_tps", 0)
                    print(f"[proxy] CGC done: {n_decoded} tokens, {decode_tps:.1f} t/s",
                          flush=True)

        except URLError as e:
            error_text = f"\n[CGC error: {e}]"
            full_text += error_text
            self.wfile.write(sse_format("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": error_text}
            }).encode())
            self.wfile.flush()

        # 发送 content_block_stop
        self.wfile.write(sse_format("content_block_stop", {
            "type": "content_block_stop",
            "index": 0
        }).encode())
        self.wfile.flush()

        # 发送 message_delta (stop_reason)
        self.wfile.write(sse_format("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": len(full_text.split())}
        }).encode())
        self.wfile.flush()

        # 发送 message_stop
        self.wfile.write(sse_format("message_stop", {
            "type": "message_stop"
        }).encode())
        self.wfile.flush()

    def _handle_non_streaming(self, prompt: str, max_tokens: int, model: str):
        """非流式响应"""
        cgc_payload = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "seed": 42
        }).encode()

        req = Request(
            f"{self.cgc_url}/v1/cgc/resume",
            data=cgc_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            resp = urlopen(req, timeout=600)
            full_text = ""

            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "token":
                    full_text += event.get("t", "")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": full_text}],
                "model": model,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": len(full_text.split())}
            }).encode())

        except URLError as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


def main():
    parser = argparse.ArgumentParser(description="Anthropic → CGC Protocol Proxy")
    parser.add_argument("--port", type=int, default=8082, help="本地监听端口 (default: 8082)")
    parser.add_argument("--cgc-url", default="http://192.168.101.87:1237",
                        help="Mac CGC edge_server URL")
    parser.add_argument("--model", default="qwen3.6-35b-mtp",
                        help="模型名称 (default: qwen3.6-35b-mtp)")
    args = parser.parse_args()

    ProxyHandler.cgc_url = args.cgc_url
    ProxyHandler.model_name = args.model

    server = HTTPServer(("127.0.0.1", args.port), ProxyHandler)
    print(f"[proxy] Anthropic → CGC proxy listening on http://127.0.0.1:{args.port}")
    print(f"[proxy] CGC backend: {args.cgc_url}")
    print(f"[proxy] Model: {args.model}")
    print(f"[proxy] Usage:")
    print(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:{args.port} ANTHROPIC_API_KEY=dummy claude")
    print(flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[proxy] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
