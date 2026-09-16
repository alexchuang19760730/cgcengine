"""
Qwen3.6-35B-A3B inference integration layer.

Provides unified interface to Qwen3.6 inference via:
1. CGC edge_server (OpenAI-compatible API, with expert-cache + MTP + DOPD)
2. llama.cpp llama-server (local GGUF inference)
3. Direct PyTorch (for Loop MoE model after grafting)

Used by:
- loopmoe_agent_adapter.py (Terminal-Bench agent)
- finetune/finetune_loopmoe.sh (training data generation)
- SFT data generation pipeline
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None  # Lazy import; only needed for API backend


@dataclass
class Qwen36Config:
    """Configuration for Qwen3.6-35B-A3B inference."""

    # Inference backend
    backend: str = "cgc"  # "cgc" | "llama_cpp" | "pytorch" | "mlx"

    # API endpoint (OpenAI-compatible)
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "sk-local"
    model_name: str = "qwen3.6-35b-a3b"

    # Generation params
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.05

    # CGC-specific
    cgc_expert_cache: bool = True
    cgc_mtp: bool = False  # MTP speculative decoding
    cgc_l4_skip: bool = True

    # Timeout
    request_timeout: int = 120

    # GGUF path (for llama.cpp / MLX)
    gguf_path: str = ""

    # Loop MoE specific (when backend=pytorch)
    loopmoe_config_path: str = ""
    loopmoe_checkpoint_path: str = ""
    loopmoe_max_recurrent_steps: int = 8


class Qwen36Inference:
    """
    Unified Qwen3.6 inference client.

    Usage:
        config = Qwen36Config(backend="cgc", base_url="http://127.0.0.1:1234/v1")
        infer = Qwen36Inference(config)
        response = infer.chat("Write a hello world program")
    """

    def __init__(self, config: Qwen36Config):
        self.config = config
        self._pytorch_model = None
        self._pytorch_tokenizer = None

        # Session is only needed for API backends; create lazily
        self._session = None

    @property
    def session(self):
        """Lazy-create requests session (only when API backend is used)."""
        if self._session is None:
            if requests is None:
                raise ImportError(
                    "requests package is required for API backends. "
                    "Install with: pip install requests"
                )
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            })
        return self._session

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Chat completion.

        Args:
            messages: list of {"role": "system|user|assistant", "content": "..."}
            **kwargs: override generation params

        Returns:
            assistant response text
        """
        cfg = self.config

        if cfg.backend in ("cgc", "llama_cpp", "mlx"):
            return self._chat_api(messages, **kwargs)
        elif cfg.backend == "pytorch":
            return self._chat_pytorch(messages, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {cfg.backend}")

    def _chat_api(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat via OpenAI-compatible API (CGC edge_server / llama.cpp)."""
        cfg = self.config
        payload = {
            "model": cfg.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", cfg.max_tokens),
            "temperature": kwargs.get("temperature", cfg.temperature),
            "top_p": kwargs.get("top_p", cfg.top_p),
            "stream": False,
        }

        # CGC-specific extras
        if cfg.backend == "cgc":
            payload["expert_cache"] = cfg.cgc_expert_cache
            payload["l4_skip"] = cfg.cgc_l4_skip
            if cfg.cgc_mtp:
                payload["mtp"] = True

        try:
            resp = self.session.post(
                f"{cfg.base_url}/chat/completions",
                json=payload,
                timeout=cfg.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"[Qwen36] API error: {e}")
            return f"[ERROR] Inference request failed: {e}"

    def _chat_pytorch(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat via direct PyTorch Loop MoE model."""
        if self._pytorch_model is None:
            self._load_pytorch_model()

        import torch
        from ..config import LoopMoEConfig

        cfg = self.config
        model = self._pytorch_model
        tokenizer = self._pytorch_tokenizer

        # Format messages
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(text, return_tensors="pt").to(next(model.parameters()).device)
        else:
            # Fallback: simple concatenation
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            text += "\nassistant: "
            input_ids = torch.tensor([[ord(c) for c in text[-2048:]]]).to(next(model.parameters()).device)

        # Generate
        model.eval()
        max_new_tokens = kwargs.get("max_tokens", cfg.max_tokens)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids=input_ids)
                logits = outputs["logits"][:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if tokenizer is not None and next_token.item() == tokenizer.eos_token_id:
                    break

        if tokenizer is not None:
            return tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return "".join(chr(t) for t in input_ids[0].tolist())

    def _load_pytorch_model(self):
        """Load Loop MoE model in PyTorch (for backend=pytorch)."""
        import torch
        from ..config import LoopMoEConfig
        from ..models.loop_moe_model import LoopMoEModel

        cfg = self.config
        print("[Qwen36] Loading Loop MoE PyTorch model...")

        # Load config
        if cfg.loopmoe_config_path and os.path.exists(cfg.loopmoe_config_path):
            model_config = LoopMoEConfig.from_json(cfg.loopmoe_config_path)
        else:
            model_config = LoopMoEConfig()

        model = LoopMoEModel(model_config)

        # Load checkpoint
        if cfg.loopmoe_checkpoint_path and os.path.exists(cfg.loopmoe_checkpoint_path):
            ckpt = torch.load(cfg.loopmoe_checkpoint_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            print(f"[Qwen36] Loaded checkpoint: {cfg.loopmoe_checkpoint_path}")

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device).eval()
        self._pytorch_model = model

        # Try load tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer_path = os.path.dirname(cfg.loopmoe_config_path) if cfg.loopmoe_config_path else "."
            self._pytorch_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception:
            print("[Qwen36] Warning: could not load tokenizer, using char-level fallback")
            self._pytorch_tokenizer = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Convenience: single-turn generation."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def health_check(self) -> Dict[str, Any]:
        """Check if inference backend is reachable."""
        cfg = self.config
        result = {"backend": cfg.backend, "reachable": False, "latency_ms": 0}

        if cfg.backend == "pytorch":
            result["reachable"] = self._pytorch_model is not None
            return result

        try:
            start = time.time()
            resp = self.session.get(
                f"{cfg.base_url}/models",
                timeout=5,
            )
            result["latency_ms"] = int((time.time() - start) * 1000)
            result["reachable"] = resp.status_code == 200
            if resp.status_code == 200:
                result["models"] = [m.get("id") for m in resp.json().get("data", [])]
        except Exception as e:
            result["error"] = str(e)

        return result

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """Batch generation (sequential, for API backends)."""
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, system_prompt, **kwargs))
        return results


def create_inference_from_env() -> Qwen36Inference:
    """Create inference client from environment variables (agent_harness config.env)."""
    config = Qwen36Config(
        backend=os.environ.get("LOOPMOE_BACKEND", "cgc"),
        base_url=os.environ.get("QWEN36_BASE_URL", "http://127.0.0.1:1234/v1"),
        api_key=os.environ.get("QWEN36_API_KEY", "sk-local"),
        model_name=os.environ.get("QWEN36_MODEL", "qwen3.6-35b-a3b"),
        gguf_path=os.environ.get("QWEN36_GGUF_PATH", ""),
        loopmoe_config_path=os.environ.get("LOOPMOE_CONFIG_PATH", ""),
        loopmoe_checkpoint_path=os.environ.get("LOOPMOE_CHECKPOINT_PATH", ""),
    )
    return Qwen36Inference(config)
