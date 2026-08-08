"""
LlamaCpp Draft Backend - 使用 llama-server 做真实 draft token 生成。

架构:
  llama-server (子进程, port 8082) ← HTTP /v1/completions
       ↑
  LlamaCppDraftBackend (async, aiohttp)
       ↑
  DraftSequenceEngine (Mode B)

优势:
  - llama.cpp Metal kernel 优化成熟, decode 149.8 tok/s (1B Q4)
  - 模型常驻内存, 无进程启动开销
  - HTTP API 与现有 aiohttp 基础设施集成
  - 5 tokens @ 149.8 t/s ≈ 33ms (vs MLX 192ms, 5.8x faster)
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class DraftResult:
    """Draft generation result."""
    tokens: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    backend: str = "llamacpp"
    error: Optional[str] = None


class LlamaCppDraftBackend:
    """
    llama.cpp draft backend via llama-server HTTP API.

    Manages a persistent llama-server subprocess that keeps the model
    loaded in GPU memory. Provides async draft token generation.
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8082
    DEFAULT_MODEL = "/Users/alexchuang/Documents/embodied/.cgc_local/models/minicpm5-1b/MiniCPM5-1B-Q4_K_M.gguf"

    def __init__(
        self,
        model_path: Optional[str] = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        n_gpu_layers: int = 99,
        n_threads: int = 4,
        ctx_size: int = 4096,
        auto_start: bool = True,
    ):
        self.model_path = model_path or self.DEFAULT_MODEL
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.ctx_size = ctx_size
        self._process: Optional[subprocess.Popen] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = f"http://{host}:{port}"
        self._started = False
        self._model_loaded = False

        if auto_start:
            self._start_sync()

    def _start_sync(self):
        """Start llama-server subprocess (sync, for __init__)."""
        # Check if server is already running on this port
        import urllib.request
        try:
            resp = urllib.request.urlopen(f"http://{self.host}:{self.port}/health", timeout=2)
            if resp.status == 200:
                logger.info(f"llama-server already running on :{self.port}")
                self._started = True
                self._model_loaded = True
                return
        except Exception:
            pass

        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found: {self.model_path}, skip start")
            return

        cmd = [
            "llama-server",
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "-t", str(self.n_threads),
            "-c", str(self.ctx_size),
            "--temp", "0",
            "--no-webui",
            "-np", "4",
        ]

        logger.info(f"Starting llama-server: {self.model_path} on :{self.port}")
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            self._started = True
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")

    async def ensure_ready(self, timeout: float = 30.0) -> bool:
        """Wait for llama-server to be ready."""
        if self._model_loaded:
            return True
        if not self._started:
            logger.warning("llama-server not started, cannot ensure_ready")
            return False

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(
                        f"{self._base_url}/health",
                        timeout=aiohttp.ClientTimeout(total=2.0),
                    ) as resp:
                        if resp.status == 200:
                            self._model_loaded = True
                            logger.info("llama-server ready!")
                            return True
            except Exception:
                pass
            await asyncio.sleep(0.5)

        logger.error("llama-server failed to become ready within timeout")
        return False

    async def generate_draft(
        self,
        prompt: str,
        n_tokens: int = 5,
        temperature: float = 0.0,
        timeout: float = 10.0,
    ) -> DraftResult:
        """
        Generate draft tokens using llama-server.

        Args:
            prompt: Input prompt (partial code/text to continue)
            n_tokens: Number of draft tokens to generate
            temperature: 0.0 for deterministic (matches CGC确定性解码策略)
            timeout: HTTP timeout in seconds

        Returns:
            DraftResult with tokens, latency, throughput
        """
        if not await self.ensure_ready():
            return DraftResult(error="llama-server not ready")

        # Create fresh session per call (safe across asyncio.run() boundaries)
        payload = {
            "prompt": prompt,
            "n_predict": n_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "stream": False,
            "cache_prompt": True,
        }

        t0 = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/completion",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return DraftResult(error=f"HTTP {resp.status}: {text[:200]}")

                    data = await resp.json()
                    t1 = time.time()

                    latency_ms = (t1 - t0) * 1000
                    text = data.get("content", "")
                    tokens_generated = data.get("tokens_predicted", n_tokens)
                    tokens_evaluated = data.get("tokens_evaluated", 0)

                    if latency_ms > 0 and tokens_generated > 0:
                        tps = tokens_generated / (latency_ms / 1000.0)
                    else:
                        tps = 0.0

                    token_ids = []
                    tokens = text.split() if text else []

                    return DraftResult(
                        tokens=tokens,
                        token_ids=token_ids,
                        text=text,
                        latency_ms=latency_ms,
                        tokens_per_sec=tps,
                        backend="llamacpp",
                    )

        except asyncio.TimeoutError:
            return DraftResult(error="Request timed out")
        except Exception as e:
            return DraftResult(error=str(e))

    async def generate_draft_stream(
        self,
        prompt: str,
        n_tokens: int = 5,
        temperature: float = 0.0,
    ):
        """
        Stream draft tokens one by one (for real-time TTFT measurement).

        Yields:
            (token_text, elapsed_ms) tuples
        """
        if not await self.ensure_ready():
            yield ("", 0.0)
            return

        payload = {
            "prompt": prompt,
            "n_predict": n_tokens,
            "temperature": temperature,
            "stream": True,
            "cache_prompt": True,
        }

        t0 = time.time()
        first_token_time = None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/completion",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    async for line in resp.content:
                        line = line.decode("utf-8", errors="ignore").strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            token_text = chunk.get("content", "")
                            elapsed = (time.time() - t0) * 1000
                            if first_token_time is None:
                                first_token_time = elapsed
                            if token_text:
                                yield (token_text, elapsed)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield ("", 0.0)

    def is_available(self) -> bool:
        """Check if llama-server is running and model is loaded."""
        return self._model_loaded

    def get_stats(self) -> dict:
        """Get backend statistics."""
        return {
            "backend": "llamacpp",
            "model_path": self.model_path,
            "port": self.port,
            "started": self._started,
            "model_loaded": self._model_loaded,
            "process_alive": self._process is not None and self._process.poll() is None,
        }

    async def close(self):
        """Clean shutdown."""
        if self._session:
            await self._session.close()
            self._session = None
        if self._process:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._process = None
        self._started = False
        self._model_loaded = False


# Singleton
_global_backend: Optional[LlamaCppDraftBackend] = None


def get_llamacpp_backend(
    model_path: Optional[str] = None,
    port: int = LlamaCppDraftBackend.DEFAULT_PORT,
) -> LlamaCppDraftBackend:
    """Get or create global llama.cpp draft backend."""
    global _global_backend
    if _global_backend is None or not _global_backend.get_stats()["process_alive"]:
        _global_backend = LlamaCppDraftBackend(
            model_path=model_path,
            port=port,
            auto_start=True,
        )
    return _global_backend


async def reset_llamacpp_backend():
    """Reset global backend (for testing)."""
    global _global_backend
    if _global_backend:
        await _global_backend.close()
    _global_backend = None
