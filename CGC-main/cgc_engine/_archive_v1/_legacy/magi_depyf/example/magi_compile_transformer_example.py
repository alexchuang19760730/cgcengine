# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Magi compile demo: two-layer Transformer with wrapped middle attention.

Run:
    PYTHONPATH=. python pkgs/MagiCompiler/cgc_engine/magi_depyf/example/magi_compile_transformer_example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgc_engine import magi_compile, magi_register_custom_op

DEBUG_CACHE_ROOT_DIR = "./magi_depyf_torch_compile_debug"


def _patch_debug_cache_root(conf):
    conf.cache_root_dir = DEBUG_CACHE_ROOT_DIR
    return conf


@magi_register_custom_op("magi_depyf::wrapped_attention_boundary", is_subgraph_boundary=True)
def wrapped_attention_boundary(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WrappedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, f"hidden dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        y = wrapped_attention_boundary(q, k, v)
        y = y.transpose(1, 2).reshape(bsz, seqlen, self.dim)
        return self.out_proj(y)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WrappedAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@magi_compile(dynamic_arg_dims={"x": 0}, config_patch=_patch_debug_cache_root)
class TwoLayerTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.block1 = TransformerLayer(dim=dim, num_heads=num_heads)
        self.block2 = TransformerLayer(dim=dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.mha.set_fastpath_enabled(False)

    model = TwoLayerTransformer(dim=256, num_heads=8).to(device).eval()
    x = torch.randn(2, 128, 256, device=device)

    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    print(f"cache/debug output root: {DEBUG_CACHE_ROOT_DIR}")


if __name__ == "__main__":
    main()
