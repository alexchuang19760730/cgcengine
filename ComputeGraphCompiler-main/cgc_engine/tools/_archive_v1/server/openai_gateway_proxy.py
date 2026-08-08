#!/usr/bin/env python3
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable

import requests


LOG = logging.getLogger("openai_gateway_proxy")


def _filtered_response_headers(headers: requests.structures.CaseInsensitiveDict) -> Iterable[tuple[str, str]]:
    for key, value in headers.items():
        lower = key.lower()
        if lower in {
            "connection",
            "content-encoding",
            "transfer-encoding",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "upgrade",
        }:
            continue
        yield key, value


class OpenAIProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    backend_base_url = ""
    gateway_name = ""
    protocol_family = ""

    def log_message(self, fmt: str, *args) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _json_response(self, status_code: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_proxy(self, method: str) -> None:
        upstream_url = f"{self.backend_base_url}{self.path}"
        body = None
        if method in {"POST", "PUT", "PATCH"}:
            body_len = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(body_len) if body_len > 0 else b""
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        try:
            upstream = requests.request(
                method,
                upstream_url,
                data=body,
                headers=headers,
                stream=True,
                timeout=(10, 1800),
            )
        except requests.RequestException as exc:
            self._json_response(
                503,
                {
                    "error": {
                        "message": f"Upstream unavailable: {exc}",
                        "type": "backend_unavailable",
                    }
                },
            )
            return

        try:
            self.send_response(upstream.status_code)
            for key, value in _filtered_response_headers(upstream.headers):
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in upstream.iter_content(chunk_size=64 * 1024):
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            self.close_connection = True
        finally:
            upstream.close()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response(
                200,
                {
                    "status": "ok",
                    "gateway": self.gateway_name,
                    "backend_url": self.backend_base_url,
                    "protocol_family": self.protocol_family,
                },
            )
            return
        self._handle_proxy("GET")

    def do_POST(self) -> None:
        self._handle_proxy("POST")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible reverse proxy.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, required=True)
    parser.add_argument("--gateway-name", default="cgc-openai-proxy")
    parser.add_argument("--protocol-family", default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    handler = type(
        "ConfiguredOpenAIProxyHandler",
        (OpenAIProxyHandler,),
        {
            "backend_base_url": f"http://{args.backend_host}:{args.backend_port}",
            "gateway_name": args.gateway_name,
            "protocol_family": args.protocol_family,
        },
    )
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), handler)
    LOG.info(
        "Proxy listening on http://%s:%s -> http://%s:%s protocol_family=%s",
        args.listen_host,
        args.listen_port,
        args.backend_host,
        args.backend_port,
        args.protocol_family,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
