from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx


@dataclass
class MLXOrthoKDAConfig:
    num_heads: int
    head_dim: int
    ortho_base_dim: int = 32
    decay_rate: float = 0.01


class OrthoKDACache:
    def __init__(self, config: MLXOrthoKDAConfig):
        self.config = config
        self.offset = 0
        self.K = mx.zeros((config.num_heads, config.ortho_base_dim, config.head_dim), dtype=mx.float32)
        self.V = mx.zeros((config.num_heads, config.ortho_base_dim, config.head_dim), dtype=mx.float32)
        self.decay = mx.zeros((config.num_heads, config.ortho_base_dim), dtype=mx.float32)
        self.current_dim = 0
        self._update_decay()

    def _update_decay(self):
        i = mx.arange(self.config.ortho_base_dim, dtype=mx.float32)
        decay_1d = mx.exp(-i * mx.array(self.config.decay_rate, dtype=mx.float32))
        self.decay = mx.broadcast_to(decay_1d[None, :], (self.config.num_heads, self.config.ortho_base_dim))

    def empty(self):
        return self.offset == 0

    def size(self):
        return self.offset

    @property
    def nbytes(self):
        return self.K.nbytes + self.V.nbytes + self.decay.nbytes

    def trim(self, n: int):
        n = min(self.offset, int(n))
        self.offset -= n
        return n

    def is_trimmable(self):
        return True

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        L = int(keys.shape[2])
        self.offset += L

        if self.current_dim == 0 and L > 1:
            take = min(self.config.ortho_base_dim, L)
            k0 = keys[0, :, :take, :].astype(mx.float32)
            v0 = values[0, :, :take, :].astype(mx.float32)
            norms = mx.sqrt(mx.sum(k0 * k0, axis=-1, keepdims=True) + 1e-6)
            k0 = k0 / norms
            self.K[:, :take, :] = k0
            self.V[:, :take, :] = v0
            self.current_dim = take
            return self.K[None, :, : self.current_dim, :], self.V[None, :, : self.current_dim, :]

        for t in range(L):
            k_new = keys[0, :, t, :].astype(mx.float32)
            v_new = values[0, :, t, :].astype(mx.float32)
            self._update_one(k_new, v_new)

        return self.K[None, :, : self.current_dim, :], self.V[None, :, : self.current_dim, :]

    def _update_one(self, k_new: mx.array, v_new: mx.array) -> None:
        if self.current_dim < self.config.ortho_base_dim:
            i = self.current_dim
            k_ortho = self._gram_schmidt(k_new, i)
            self.K[:, i, :] = self.K[:, i, :] + k_ortho
            self.V[:, i, :] = self.V[:, i, :] + v_new
            self.current_dim += 1
            return

        self.K[:, 1:, :] = self.K[:, :-1, :]
        self.V[:, 1:, :] = self.V[:, :-1, :]
        k_ortho = self._gram_schmidt(k_new, 0)
        self.K[:, 0, :] = self.K[:, 0, :] + k_ortho
        self.V[:, 0, :] = self.V[:, 0, :] + v_new

    def _gram_schmidt(self, v: mx.array, idx: int) -> mx.array:
        v_out = v
        for i in range(idx):
            basis_i = self.K[:, i, :]
            dot = mx.sum(v_out * basis_i, axis=-1, keepdims=True)
            v_out = v_out - dot * basis_i
        norm = mx.sqrt(mx.sum(v_out * v_out, axis=-1, keepdims=True) + 1e-6)
        return v_out / norm
