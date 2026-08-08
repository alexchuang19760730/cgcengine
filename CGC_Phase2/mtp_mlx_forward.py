"""MLX Metal MTP Head — 高速 draft 生成 (替代 PyTorch CPU forward).

将训练好的 PyTorch MTP head checkpoint 转换为 MLX 权重, 在 Metal GPU 上运行.
性能: ~8ms/draft (vs PyTorch CPU ~25-45ms), 3-5x 加速.

用法:
    from mtp_mlx_forward import MTPMLXForward

    mlp = MTPMLXForward(
        checkpoint="mtp_head.pt",
        embed_head_path="embed_head.pt",
        hidden_size=896, vocab_size=151936,
        num_heads=14, head_dim=64, intermediate_size=4864,
    )
    draft_tokens = mlp.draft_chain(hidden_numpy, token_id, num_draft=4)
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as mx_nn
import numpy as np
import torch


class MTPMLXForward:
    """MLX Metal MTP head forward pass.

    Loads PyTorch checkpoint, converts to MLX, runs on Metal GPU.
    Drop-in replacement for PyTorch MTPHead in verify loop.
    """

    def __init__(
        self,
        checkpoint: str,
        embed_head_path: str,
        hidden_size: int = 896,
        vocab_size: int = 151936,
        num_heads: int = 14,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        rms_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_eps = rms_eps
        self.scale = head_dim ** -0.5
        self.rope_theta = rope_theta

        print(f"[MLX-MTP] Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)

        # Handle nested checkpoint format: {model_state_dict: {...}, config: {...}, ...}
        if "model_state_dict" in ckpt:
            raw_weights = ckpt["model_state_dict"]
        else:
            raw_weights = ckpt

        # Filter out lm_head/embed (shared, loaded separately)
        mtp_weights = {}
        for k, v in raw_weights.items():
            if not isinstance(v, torch.Tensor):
                continue
            if "lm_head" in k or "embed" in k:
                continue
            mtp_weights[k] = v
        print(f"  MTP weights: {len(mtp_weights)} tensors")

        # Convert to MLX arrays
        self.weights = {}
        for k, v in mtp_weights.items():
            self.weights[k] = mx.array(v.float().numpy())

        # Load embed + lm_head
        print(f"  Loading embed+head: {embed_head_path}")
        eh = torch.load(embed_head_path, map_location="cpu", weights_only=True)
        lm_head_w = eh.get("lm_head_weight")
        if lm_head_w is None:
            lm_head_w = eh.get("lm_head")
        embed_w = eh.get("embed_weight")
        if embed_w is None:
            embed_w = eh.get("embed")
        if lm_head_w is None and bool(eh.get("lm_head_tied_to_embed")):
            lm_head_w = embed_w

        if lm_head_w is None or embed_w is None:
            raise ValueError("embed_head.pt missing lm_head_weight or embed_weight")

        self.lm_head = mx.array(lm_head_w.float().numpy())  # [vocab, hidden]
        self.embed = mx.array(embed_w.float().numpy())       # [vocab, hidden]
        print(f"  lm_head: {self.lm_head.shape}, embed: {self.embed.shape}")

        # Precompute RoPE
        self._rope_cos = None
        self._rope_sin = None

        # Warmup Metal kernels
        self._warmup()
        print(f"  MLX MTP ready (Metal GPU)")

    def _get_rope(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        """Compute RoPE cos/sin."""
        if self._rope_cos is not None and self._rope_cos.shape[0] >= seq_len:
            return self._rope_cos[:seq_len], self._rope_sin[:seq_len]

        inv_freq = 1.0 / (self.rope_theta ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim))
        positions = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        self._rope_cos = cos
        self._rope_sin = sin
        return cos[:seq_len], sin[:seq_len]

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Apply RoPE to x. x: [heads, seq, head_dim], cos/sin: [seq, head_dim]"""
        cos = mx.broadcast_to(cos, (x.shape[0], cos.shape[0], cos.shape[1]))
        sin = mx.broadcast_to(sin, (x.shape[0], sin.shape[0], sin.shape[1]))
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated * sin

    def _rmsnorm(self, x: mx.array, weight_key: str) -> mx.array:
        """RMSNorm."""
        w = self.weights[weight_key]
        variance = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.rms_eps)
        return (w * x).astype(x.dtype)

    def _attention(self, x: mx.array) -> mx.array:
        """Self-attention with RoPE. x: [1, seq, hidden]"""
        seq_len = x.shape[1]
        hidden = x.shape[2]

        q = x @ self.weights["attn.q_proj.weight"].T  # [1, seq, num_heads*head_dim]
        k = x @ self.weights["attn.k_proj.weight"].T
        v = x @ self.weights["attn.v_proj.weight"].T

        # Reshape: [1, seq, num_heads*head_dim] -> [num_heads, seq, head_dim]
        q = q.reshape(1, seq_len, self.num_heads, self.head_dim).squeeze(0).transpose(0, 1, 2).squeeze(0)  # Actually need [heads, seq, head_dim]
        # Simpler: [1, seq, heads, dim] -> transpose to [heads, seq, dim]
        q = q.reshape(1, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3).reshape(self.num_heads, seq_len, self.head_dim)
        k = k.reshape(1, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3).reshape(self.num_heads, seq_len, self.head_dim)
        v = v.reshape(1, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3).reshape(self.num_heads, seq_len, self.head_dim)

        cos, sin = self._get_rope(seq_len)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Scaled dot-product attention (causal)
        # scores: [heads, seq, seq]
        scores = (q @ k.transpose(0, 2, 1)) * self.scale

        # Causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
        scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v  # [heads, seq, head_dim]

        # Reshape back: [heads, seq, dim] -> [1, seq, hidden]
        out = out.transpose(1, 0, 2).reshape(1, seq_len, self.num_heads * self.head_dim)
        return out @ self.weights["attn.o_proj.weight"].T

    def _mlp(self, x: mx.array) -> mx.array:
        """SwiGLU MLP. x: [1, seq, hidden]"""
        gate = x @ self.weights["mlp.gate_proj.weight"].T
        up = x @ self.weights["mlp.up_proj.weight"].T
        return (mx_nn.silu(gate) * up) @ self.weights["mlp.down_proj.weight"].T

    def forward_hidden(self, hidden: mx.array, token_embed: mx.array) -> mx.array:
        """Forward pass, returns hidden state (before lm_head).

        Args:
            hidden: [1, 1, hidden_size] base model hidden state
            token_embed: [1, 1, hidden_size] current token embedding

        Returns:
            mtp_hidden: [1, 1, hidden_size] MTP hidden state for chaining
        """
        # Concat + projection
        x = mx.concatenate([hidden, token_embed], axis=-1)  # [1, 1, 2*hidden]
        x = x @ self.weights["proj.weight"].T  # [1, 1, hidden]

        # Transformer layer (pre-norm)
        h = x + self._attention(self._rmsnorm(x, "norm1.weight"))
        h = h + self._mlp(self._rmsnorm(h, "norm2.weight"))
        mtp_hidden = self._rmsnorm(h, "norm_out.weight")

        return mtp_hidden

    def draft_chain(
        self,
        hidden_np: np.ndarray,
        token_id: int,
        num_draft: int,
    ) -> Tuple[List[int], float]:
        """Chain draft generation.

        Args:
            hidden_np: base model hidden state [hidden_size] (numpy float32)
            token_id: last token ID
            num_draft: number of draft tokens to generate

        Returns:
            draft_tokens: list of draft token IDs
            elapsed_ms: total draft time in ms
        """
        t0 = time.time()

        current_hidden = mx.array(hidden_np.astype(np.float32)).reshape(1, 1, self.hidden_size)
        current_token = token_id

        draft_tokens = []
        for i in range(num_draft):
            # Token embedding lookup
            token_embed = self.embed[current_token].reshape(1, 1, self.hidden_size)

            # Forward → hidden
            mtp_hidden = self.forward_hidden(current_hidden, token_embed)

            # lm_head → logits → argmax
            logits = mtp_hidden.reshape(1, -1) @ self.lm_head.T  # [1, vocab]
            draft_token = int(mx.argmax(logits, axis=-1).item())

            draft_tokens.append(draft_token)
            current_hidden = mtp_hidden
            current_token = draft_token

        mx.eval(draft_tokens)
        elapsed_ms = (time.time() - t0) * 1000
        return draft_tokens, elapsed_ms

    def _warmup(self):
        """Warmup Metal kernels with a dummy forward pass."""
        dummy_hidden = mx.zeros((1, 1, self.hidden_size))
        dummy_embed = mx.zeros((1, 1, self.hidden_size))
        h = self.forward_hidden(dummy_hidden, dummy_embed)
        logits = h.reshape(1, -1) @ self.lm_head.T
        mx.eval(logits)
