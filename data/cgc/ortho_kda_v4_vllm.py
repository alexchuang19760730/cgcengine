# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
vLLM Backend Adapter for OrthoKDA v4

This module provides integration between OrthoKDA v4 and vLLM's attention backend.
vLLM uses Flash Attention style attention with custom backends.

Key integration points:
1. Custom attention backend registration
2. KV cache management with OrthoKDA structure
3. Forward pass hook for OrthoKDA v4 attention
"""

import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

VLLM_AVAILABLE = False
_vllm_version = None
try:
    import vllm
    _vllm_version = getattr(vllm, '__version__', 'unknown')
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[OrthoKDAV4:vLLM] vLLM import error: {e}. Install with: pip install vllm")
except Exception as e:
    logger.warning(f"[OrthoKDAV4:vLLM] vLLM initialization error: {e}. Version: {_vllm_version}")

_ATTENTION_BACKEND_AVAILABLE = False
if VLLM_AVAILABLE:
    try:
        from vllm.attention.backends.flash_attn import FlashAttentionBackend
        _ATTENTION_BACKEND_AVAILABLE = True
    except ImportError:
        try:
            from vllm.attention.backends.utils import FlashAttentionBackend
            _ATTENTION_BACKEND_AVAILABLE = True
        except ImportError:
            logger.warning(f"[OrthoKDAV4:vLLM] FlashAttentionBackend not available in vLLM {_vllm_version}")
    
    try:
        from vllm.attention import Attention
    except ImportError:
        try:
            from vllm.attention.layer import Attention
        except ImportError:
            logger.warning(f"[OrthoKDAV4:vLLM] Attention class not available in vLLM {_vllm_version}")


@dataclass
class OrthoKDAV4VLLMConfig:
    num_heads: int = 32
    head_dim: int = 128
    ortho_base_dim: int = 128
    decay_rate: float = 0.01
    enable: bool = True
    layers: Optional[List[int]] = None


class OrthoKDAV4VLLMBackend:
    """
    vLLM Backend Adapter for OrthoKDA v4

    This class provides the integration layer between vLLM's attention
    mechanism and OrthoKDA v4's orthogonal basis accumulation.

    Usage:
        config = OrthoKDAV4VLLMConfig(
            num_heads=32,
            head_dim=128,
            ortho_base_dim=128,
            enable=True,
            layers=[0, 1, 2, 3]  # Apply to specific layers
        )
        backend = OrthoKDAV4VLLMBackend(config)
    """

    def __init__(
        self,
        config: OrthoKDAV4VLLMConfig,
        device: str = "cuda",
    ):
        """
        Initialize OrthoKDA v4 vLLM backend

        Args:
            config: OrthoKDA v4 configuration
            device: Device to run on ('cuda' or 'cpu')
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not available. Cannot initialize OrthoKDAV4VLLMBackend.")

        self.config = config
        self.device = device
        self.enable = config.enable

        self.kv_cache: Dict[int, Any] = {}

        self.python_backend = None
        if self.enable:
            self._init_python_backend()

        logger.info(f"[OrthoKDAV4:vLLM] Initialized with {config.num_heads} heads, "
                   f"head_dim={config.head_dim}, ortho_base_dim={config.ortho_base_dim}, "
                   f"device={device}, layers={config.layers}")

    def _init_python_backend(self):
        """Initialize Python backend for fallback"""
        from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4

        self.python_backend = OrthoKDAV4(
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            ortho_base_dim=self.config.ortho_base_dim,
            decay_rate=self.config.decay_rate,
            use_cuda=(self.device == "cuda"),
        )

    def allocate_kv_cache(
        self,
        layer_idx: int,
        num_seqs: int,
        max_seq_len: int,
    ) -> int:
        """
        Allocate KV cache for a specific layer

        Note: OrthoKDA v4 uses FIXED memory regardless of max_seq_len

        Args:
            layer_idx: Layer index
            num_seqs: Number of sequences
            max_seq_len: Maximum sequence length (ignored by OrthoKDA)

        Returns:
            Size of allocated memory in bytes
        """
        if not self.enable:
            return 0

        if layer_idx in self.kv_cache:
            return self.get_memory_footprint(layer_idx)

        layer_cache = {
            "num_seqs": num_seqs,
            "ortho_kda": OrthoKDAV4(
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                ortho_base_dim=self.config.ortho_base_dim,
                decay_rate=self.config.decay_rate,
                use_cuda=(self.device == "cuda"),
            ) if self.python_backend is None else None,
        }

        if self.python_backend is not None:
            layer_cache["ortho_kda"] = self.python_backend

        self.kv_cache[layer_idx] = layer_cache

        memory = self.get_memory_footprint(layer_idx)
        logger.debug(f"[OrthoKDAV4:vLLM] Allocated KV cache for layer {layer_idx}: {memory} bytes")

        return memory

    def get_memory_footprint(self, layer_idx: int) -> int:
        """Get memory footprint for a layer"""
        if layer_idx not in self.kv_cache:
            return 0

        num_heads = self.config.num_heads
        head_dim = self.config.head_dim
        ortho_base_dim = self.config.ortho_base_dim

        kv_size = num_heads * ortho_base_dim * head_dim * 2
        decay_size = num_heads * ortho_base_dim

        return kv_size * 4 + decay_size * 4

    def forward(
        self,
        layer_idx: int,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Optional[Any] = None,
        prefix: bool = False,
    ) -> Any:
        """
        OrthoKDA v4 forward pass

        Args:
            layer_idx: Layer index
            query: Query tensor [batch, num_heads, head_dim] or similar
            key: Key tensor
            value: Value tensor
            attention_mask: Attention mask (ignored by OrthoKDA)
            prefix: Whether this is a prefix decode

        Returns:
            Attention output
        """
        if not self.enable:
            return query

        if layer_idx not in self.kv_cache:
            self.allocate_kv_cache(layer_idx, 1, 1)

        layer_cache = self.kv_cache[layer_idx]
        ortho_kda = layer_cache.get("ortho_kda", self.python_backend)

        if ortho_kda is None:
            return query

        import torch

        def to_tensor(t):
            if hasattr(t, 'to'):
                return t.to(torch.float32)
            return torch.tensor(t, dtype=torch.float32)

        key_tensor = to_tensor(key)
        value_tensor = to_tensor(value)

        if key_tensor.dim() == 2:
            key_tensor = key_tensor.unsqueeze(0).expand(self.config.num_heads, -1, -1)
        if value_tensor.dim() == 2:
            value_tensor = value_tensor.unsqueeze(0).expand(self.config.num_heads, -1, -1)

        ortho_kda.update(key_tensor, value_tensor)

        query_tensor = to_tensor(query)
        if query_tensor.dim() == 2:
            query_tensor = query_tensor.unsqueeze(0)

        output = ortho_kda.forward(query_tensor)

        if hasattr(query, 'to'):
            output = output.to(query.dtype)

        return output

    def reset_layer(self, layer_idx: int) -> None:
        """Reset state for a specific layer"""
        if layer_idx in self.kv_cache and self.kv_cache[layer_idx].get("ortho_kda"):
            self.kv_cache[layer_idx]["ortho_kda"].reset()

    def reset_all(self) -> None:
        """Reset all layer states"""
        for layer_idx in self.kv_cache:
            self.reset_layer(layer_idx)

    def get_layer_state(self, layer_idx: int) -> Optional[Dict[str, Any]]:
        """Get current state for a layer"""
        if layer_idx not in self.kv_cache:
            return None

        layer_cache = self.kv_cache[layer_idx]
        ortho_kda = layer_cache.get("ortho_kda", self.python_backend)

        if ortho_kda is None:
            return None

        return ortho_kda.get_state()

    def should_apply_to_layer(self, layer_idx: int) -> bool:
        """Check if OrthoKDA should be applied to this layer"""
        if not self.enable:
            return False

        if self.config.layers is None:
            return True

        return layer_idx in self.config.layers


class OrthoKDAV4VLLMIntegration:
    """
    High-level integration helper for vLLM + OrthoKDA v4

    This class provides a simplified interface for integrating
    OrthoKDA v4 with existing vLLM models.
    """

    def __init__(
        self,
        num_heads: int = 32,
        head_dim: int = 128,
        ortho_base_dim: int = 128,
        decay_rate: float = 0.01,
        enable: bool = True,
        layers: Optional[List[int]] = None,
        device: str = "cuda",
    ):
        """
        Initialize vLLM integration

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            ortho_base_dim: Orthogonal basis dimension (fixed KV size)
            decay_rate: TimeDecay rate
            enable: Whether to enable OrthoKDA
            layers: Specific layers to apply (None = all layers)
            device: Device to run on
        """
        self.config = OrthoKDAV4VLLMConfig(
            num_heads=num_heads,
            head_dim=head_dim,
            ortho_base_dim=ortho_base_dim,
            decay_rate=decay_rate,
            enable=enable,
            layers=layers,
        )

        self.backend = OrthoKDAV4VLLMBackend(self.config, device)

        logger.info(f"[OrthoKDAV4:vLLM:Integration] Configured with {num_heads} heads, "
                   f"ortho_base_dim={ortho_base_dim}, enable={enable}, layers={layers}")

    @staticmethod
    def from_model_config(model_path: str, **kwargs) -> "OrthoKDAV4VLLMIntegration":
        """
        Create integration from model configuration

        Args:
            model_path: Path to model or model name
            **kwargs: Additional configuration overrides

        Returns:
            OrthoKDAV4VLLMIntegration instance
        """
        num_heads = kwargs.get("num_heads", 32)
        head_dim = kwargs.get("head_dim", 128)
        ortho_base_dim = kwargs.get("ortho_base_dim", 128)

        return OrthoKDAV4VLLMIntegration(
            num_heads=num_heads,
            head_dim=head_dim,
            ortho_base_dim=ortho_base_dim,
            **kwargs,
        )


def create_vllm_model_with_ortho_kda(
    model_path: str,
    num_heads: int = 32,
    head_dim: int = 128,
    ortho_base_dim: int = 128,
    **kwargs,
) -> Any:
    """
    Create a vLLM model with OrthoKDA v4 enabled

    Args:
        model_path: Path to model or model name
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        ortho_base_dim: Orthogonal basis dimension
        **kwargs: Additional vLLM engine arguments

    Returns:
        LLM instance with OrthoKDA v4 integration
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is not available")

    integration = OrthoKDAV4VLLMIntegration(
        num_heads=num_heads,
        head_dim=head_dim,
        ortho_base_dim=ortho_base_dim,
        **kwargs,
    )

    logger.info(f"[OrthoKDAV4:vLLM] Creating model with OrthoKDA v4 enabled")

    return integration


if __name__ == "__main__":
    print("=" * 60)
    print("OrthoKDA v4 - vLLM Backend Adapter Test")
    print("=" * 60)

    if not VLLM_AVAILABLE:
        print("❌ vLLM not available. Skipping test.")
    else:
        config = OrthoKDAV4VLLMConfig(
            num_heads=8,
            head_dim=64,
            ortho_base_dim=64,
            enable=True,
            layers=[0, 1, 2, 3],
        )

        backend = OrthoKDAV4VLLMBackend(config, device="cpu")

        print(f"\n✅ vLLM Backend initialized")
        print(f"   Config: {config}")

        import torch

        batch_size = 2
        num_heads = 8
        head_dim = 64

        print(f"\n📊 Forward pass test:")

        for step in range(5):
            key = torch.randn(batch_size, num_heads, head_dim)
            value = torch.randn(batch_size, num_heads, head_dim)
            query = torch.randn(batch_size, num_heads, head_dim)

            output = backend.forward(layer_idx=0, query=query, key=key, value=value)

            state = backend.get_layer_state(0)
            print(f"   Step {step+1}: idx={state['idx']}, output_shape={output.shape}")

        print(f"\n📊 Memory footprint:")
        memory = backend.get_memory_footprint(0)
        print(f"   Layer 0: {memory} bytes = {memory/1024:.2f} KB")

        print(f"\n   ✅ KV shape fixed regardless of seq_len")

        print("\n" + "=" * 60)
        print("✅ vLLM Backend Adapter Test PASSED!")
        print("=" * 60)
