# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import torch

try:
    from ...magi_depyf.timeline import observe_lifecycle
except ImportError:
    try:
        from ..._legacy.magi_depyf.timeline import observe_lifecycle
    except ImportError:
        def observe_lifecycle(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator if not callable(args[0]) else args[0]
from .remove_item import RemoveItemPass
from .replace_sage_atten import ReplaceSageAttentionPass
from .insert_kda import InsertKDAPass


class CGCFullGraphPassManager:
    """
    CGC-aware Full Graph Pass Manager

    Extends the base FullGraphPassManager to include CGC-specific passes
    for Kimi KDA instruction insertion based on the CGC SIMD command set.

    Pass Order:
    1. RemoveItemPass - Remove unnecessary item operations
    2. ReplaceSageAttentionPass - Replace flash attention with sage attention (optional)
    3. InsertKDAPass - Insert Kimi KDA instructions (CGC-specific)
    """

    def __init__(
        self,
        pass_config,
        enable_kda_insertion: bool = True,
        kda_config: dict | None = None,
    ):
        self.pass_config = pass_config
        self.passes = []

        if self.pass_config.enable_sage_attn:
            self.passes.append(ReplaceSageAttentionPass())

        self.passes.append(RemoveItemPass())

        if enable_kda_insertion:
            kda_kwargs = kda_config or {}
            self.passes.append(InsertKDAPass(**kda_kwargs))

    @observe_lifecycle("cgc_full_graph_manager")
    def __call__(self, gm: torch.fx.GraphModule):
        for pass_ in self.passes:
            if pass_.is_applicable(gm.graph):
                pass_(gm.graph)


class CGCKDAConfig:
    """Configuration for CGC KDA instruction insertion"""

    def __init__(
        self,
        enable_ortho_basis_update: bool = True,
        enable_flashkda_fusion: bool = True,
        kda_scale: float = 1.0,
        use_gate: bool = True,
        use_qk_l2norm: bool = True,
        use_beta_sigmoid: bool = True,
        ortho_kda_base_dim: int = 128,
        chunk_size: int = 64,
        k_dim: int = 128,
        v_dim: int = 128,
        fixed_basis_size: int = 1024,
        ortho_decay: float = 0.99,
    ):
        self.enable_ortho_basis_update = enable_ortho_basis_update
        self.enable_flashkda_fusion = enable_flashkda_fusion
        self.kda_scale = kda_scale
        self.use_gate = use_gate
        self.use_qk_l2norm = use_qk_l2norm
        self.use_beta_sigmoid = use_beta_sigmoid
        self.ortho_kda_base_dim = int(ortho_kda_base_dim)
        self.chunk_size = chunk_size
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.fixed_basis_size = fixed_basis_size
        self.ortho_decay = ortho_decay

    def to_dict(self) -> dict:
        return {
            "enable_ortho_basis_update": self.enable_ortho_basis_update,
            "enable_flashkda_fusion": self.enable_flashkda_fusion,
            "kda_scale": self.kda_scale,
            "use_gate": self.use_gate,
            "use_qk_l2norm": self.use_qk_l2norm,
            "use_beta_sigmoid": self.use_beta_sigmoid,
            "ortho_kda_base_dim": self.ortho_kda_base_dim,
            "chunk_size": self.chunk_size,
            "k_dim": self.k_dim,
            "v_dim": self.v_dim,
            "fixed_basis_size": self.fixed_basis_size,
            "ortho_decay": self.ortho_decay,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CGCKDAConfig":
        return cls(**config_dict)

    def get_kda_pass_kwargs(self) -> dict:
        return {
            "enable_ortho_basis_update": self.enable_ortho_basis_update,
            "enable_flashkda_fusion": self.enable_flashkda_fusion,
            "kda_scale": self.kda_scale,
            "use_gate": self.use_gate,
            "use_qk_l2norm": self.use_qk_l2norm,
            "use_beta_sigmoid": self.use_beta_sigmoid,
            "ortho_kda_base_dim": self.ortho_kda_base_dim,
        }


DEFAULT_KDA_CONFIG = CGCKDAConfig()
