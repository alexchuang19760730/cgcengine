import platform
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TorchLoraRopeReference(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 8, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.w = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, device=device, dtype=dtype))
        self.a = nn.Parameter(torch.randn(self.hidden_dim, self.rank, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.randn(self.rank, self.hidden_dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        base_out = x2 @ self.w
        lora_out = (x2 @ self.a) @ self.b
        y = base_out + lora_out
        y = y.reshape(*x.shape)
        cos = torch.cos(y)
        sin = torch.sin(y)
        z = y * cos + _rotate_half(y) * sin
        return z


class MlxLoraRopeRuntime(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 8, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        if platform.system() != "Darwin":
            raise RuntimeError("MLX runtime is macOS-only")
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.w = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, device=device, dtype=dtype))
        self.a = nn.Parameter(torch.randn(self.hidden_dim, self.rank, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.randn(self.rank, self.hidden_dim, device=device, dtype=dtype))

        from cgc_engine.cgc.mlx_tune_integration import mlx_lora_rope_fwd, get_mlx_tune_info

        info = get_mlx_tune_info()
        if not bool(info.get("mlx_available", False)):
            raise RuntimeError("MLX is not available on this environment")

        self._mlx_lora_rope_fwd = mlx_lora_rope_fwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        y2 = self._mlx_lora_rope_fwd(x2, self.w, self.a, self.b, scale=1.0)
        return y2.reshape(*x.shape)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def build_mlx_step67_pair(
    *,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    hidden_dim = int(input_shape[-1])
    device = torch.device("cpu")
    dtype = torch.float32
    ref = TorchLoraRopeReference(hidden_dim=hidden_dim, rank=8, device=device, dtype=dtype)
    mlx = MlxLoraRopeRuntime(hidden_dim=hidden_dim, rank=8, device=device, dtype=dtype)
    with torch.no_grad():
        mlx.w.copy_(ref.w)
        mlx.a.copy_(ref.a)
        mlx.b.copy_(ref.b)
    return ref, mlx


class MlxLoraRopeNative:
    is_mlx_model = True

    def __init__(self, hidden_dim: int, rank: int = 8, dtype: str = "float16"):
        if platform.system() != "Darwin":
            raise RuntimeError("MLX runtime is macOS-only")
        import mlx.core as mx

        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.mx = mx
        self.mlx_dtype = getattr(mx, dtype)

        self.w = mx.random.normal((self.hidden_dim, self.hidden_dim), dtype=self.mlx_dtype)
        self.a = mx.random.normal((self.hidden_dim, self.rank), dtype=self.mlx_dtype)
        self.b = mx.random.normal((self.rank, self.hidden_dim), dtype=self.mlx_dtype)

    def __call__(self, x):
        mx = self.mx
        x2 = x.reshape((-1, x.shape[-1]))
        y2 = x2 @ self.w + (x2 @ self.a) @ self.b
        y = y2.reshape(x.shape)
        cos = mx.cos(y)
        sin = mx.sin(y)
        z = y * cos + _mx_rotate_half(y, mx) * sin
        return z


class MlxLoraRopeOptimized:
    is_mlx_model = True

    def __init__(self, hidden_dim: int, rank: int = 8, dtype: str = "float16"):
        if platform.system() != "Darwin":
            raise RuntimeError("MLX runtime is macOS-only")
        import mlx.core as mx

        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.mx = mx
        self.mlx_dtype = getattr(mx, dtype)

        self.w = mx.random.normal((self.hidden_dim, self.hidden_dim), dtype=self.mlx_dtype)
        self.a = mx.random.normal((self.hidden_dim, self.rank), dtype=self.mlx_dtype)
        self.b = mx.random.normal((self.rank, self.hidden_dim), dtype=self.mlx_dtype)

        def _f(x, w, a, b):
            x2 = x.reshape((-1, x.shape[-1]))
            y2 = x2 @ w + (x2 @ a) @ b
            y = y2.reshape(x.shape)
            cos = mx.cos(y)
            sin = mx.sin(y)
            return y * cos + _mx_rotate_half(y, mx) * sin

        self._compiled = mx.compile(_f)

    def __call__(self, x):
        return self._compiled(x, self.w, self.a, self.b)


def _mx_rotate_half(x, mx):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def build_mlx_step67_pair_metal(
    *,
    input_shape: Tuple[int, ...],
) -> Tuple[object, object]:
    hidden_dim = int(input_shape[-1])
    native = MlxLoraRopeNative(hidden_dim=hidden_dim, rank=8, dtype="float16")
    opt = MlxLoraRopeOptimized(hidden_dim=hidden_dim, rank=8, dtype="float16")
    opt.w = native.w
    opt.a = native.a
    opt.b = native.b
    return native, opt
