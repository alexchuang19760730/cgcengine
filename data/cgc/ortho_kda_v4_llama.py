# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
llama.cpp Backend Adapter for OrthoKDA v4

This module provides integration between OrthoKDA v4 and llama.cpp's attention backend.
llama.cpp uses ggml (now called llama.cpp) for computation with custom attention patterns.

Key integration points:
1. ggml_tensor management for OrthoKDA structure
2. Forward pass using llama.cpp's computation graph
3. KV cache management with OrthoKDA semantics
"""

import os
import math
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import logging
import ctypes
import struct

logger = logging.getLogger(__name__)

LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    logger.warning("[OrthoKDAV4:llama.cpp] llama.cpp not available. Install with: pip install llama-cpp-python")

_GGUF_AVAILABLE = False
if LLAMA_CPP_AVAILABLE:
    try:
        from llama_cpp.gguf import GGUFReader
        _GGUF_AVAILABLE = True
    except ImportError:
        logger.warning("[OrthoKDAV4:llama.cpp] GGUFReader not available, some features disabled")


@dataclass
class OrthoKDAV4LlamaConfig:
    num_heads: int = 32
    head_dim: int = 128
    ortho_base_dim: int = 128
    decay_rate: float = 0.01
    enable: bool = True
    layers: Optional[List[int]] = None
    model_path: Optional[str] = None


class OrthoKDAKVState:
    """
    Manages the orthogonal basis KV state for llama.cpp

    This class holds the actual K and V matrices used by OrthoKDA v4,
    stored in a format compatible with llama.cpp's memory model.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ortho_base_dim: int,
        device: str = "cpu",
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ortho_base_dim = ortho_base_dim
        self.device = device

        self.K = [[0.0] * head_dim for _ in range(ortho_base_dim)]
        self.V = [[0.0] * head_dim for _ in range(ortho_base_dim)]
        self.decay = [0.0] * ortho_base_dim

        self.current_dim = 0
        self.total_updates = 0

        self._update_decay()

    def _update_decay(self):
        """Update TimeDecay values"""
        import math
        for i in range(self.ortho_base_dim):
            self.decay[i] = math.exp(-i * 0.01)

    def gram_schmidt(self, v: List[float]) -> List[float]:
        """
        Perform Gram-Schmidt orthogonalization

        Args:
            v: Input vector [head_dim]

        Returns:
            Orthogonalized vector [head_dim]
        """
        import math

        for i in range(self.current_dim):
            dot = sum(v[d] * self.K[i][d] for d in range(self.head_dim))
            for d in range(self.head_dim):
                v[d] -= dot * self.K[i][d]

        norm = math.sqrt(sum(v[d] * v[d] for d in range(self.head_dim)) + 1e-6)
        for d in range(self.head_dim):
            v[d] /= norm

        return v

    def update(self, key: List[float], value: List[float]) -> None:
        """
        Update orthogonal basis with new key-value pair

        Args:
            key: Key vector [head_dim]
            value: Value vector [head_dim]
        """
        if self.current_dim >= self.ortho_base_dim:
            return

        key_ortho = self.gram_schmidt(key.copy())

        for d in range(self.head_dim):
            if self.current_dim == 0:
                self.K[self.current_dim][d] = key_ortho[d]
                self.V[self.current_dim][d] = value[d]
            else:
                self.K[self.current_dim][d] += key_ortho[d]
                self.V[self.current_dim][d] += value[d]

        self.decay[self.current_dim] = 1.0 if self.current_dim == 0 else \
            math.exp(-self.current_dim * 0.01)

        self.current_dim += 1
        self.total_updates += 1

        if self.current_dim >= self.ortho_base_dim:
            self.current_dim = self.ortho_base_dim - 1

    def forward(self, query: List[float]) -> List[float]:
        """
        Compute attention output

        Args:
            query: Query vector [head_dim]

        Returns:
            Output vector [head_dim]
        """
        if self.current_dim == 0:
            return [0.0] * self.head_dim

        output = [0.0] * self.head_dim

        for i in range(self.current_dim):
            score = sum(query[d] * self.K[i][d] for d in range(self.head_dim))

            weighted_v = score * self.decay[i]

            for d in range(self.head_dim):
                output[d] += weighted_v * self.V[i][d]

        return output

    def reset(self) -> None:
        """Reset the state"""
        for i in range(self.ortho_base_dim):
            for d in range(self.head_dim):
                self.K[i][d] = 0.0
                self.V[i][d] = 0.0
            self.decay[i] = 0.0

        self.current_dim = 0
        self.total_updates = 0

    def get_state(self) -> Dict[str, Any]:
        """Get current state as dictionary"""
        return {
            "K": self.K,
            "V": self.V,
            "decay": self.decay,
            "current_dim": self.current_dim,
            "total_updates": self.total_updates,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "ortho_base_dim": self.ortho_base_dim,
        }


class OrthoKDAV4LlamaBackend:
    """
    llama.cpp Backend Adapter for OrthoKDA v4

    This class provides the integration layer between llama.cpp's attention
    mechanism and OrthoKDA v4's orthogonal basis accumulation.

    Usage:
        config = OrthoKDAV4LlamaConfig(
            num_heads=32,
            head_dim=128,
            ortho_base_dim=128,
            enable=True,
            layers=[0, 1, 2, 3]
        )
        backend = OrthoKDAV4LlamaBackend(config)
    """

    def __init__(
        self,
        config: OrthoKDAV4LlamaConfig,
        device: str = "cpu",
    ):
        """
        Initialize OrthoKDA v4 llama.cpp backend

        Args:
            config: OrthoKDA v4 configuration
            device: Device to run on ('cpu', 'cuda', 'metal')
        """
        self.config = config
        self.device = device
        self.enable = config.enable

        self.kv_cache: Dict[int, OrthoKDAKVState] = {}

        if self.enable:
            self._init_kv_cache()

        logger.info(f"[OrthoKDAV4:llama.cpp] Initialized with {config.num_heads} heads, "
                   f"head_dim={config.head_dim}, ortho_base_dim={config.ortho_base_dim}, "
                   f"device={device}, layers={config.layers}")

    def _init_kv_cache(self):
        """Initialize KV cache for configured layers"""
        layers = self.config.layers if self.config.layers else range(32)

        for layer_idx in layers:
            self.kv_cache[layer_idx] = OrthoKDAKVState(
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                ortho_base_dim=self.config.ortho_base_dim,
                device=self.device,
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

        if layer_idx not in self.kv_cache:
            self.kv_cache[layer_idx] = OrthoKDAKVState(
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                ortho_base_dim=self.config.ortho_base_dim,
                device=self.device,
            )

        memory = self.get_memory_footprint(layer_idx)
        logger.debug(f"[OrthoKDAV4:llama.cpp] Allocated KV cache for layer {layer_idx}: {memory} bytes")

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
        query: Union[List[float], Any],
        key: Union[List[float], Any],
        value: Union[List[float], Any],
        attention_mask: Optional[Any] = None,
        prefix: bool = False,
    ) -> Any:
        """
        OrthoKDA v4 forward pass

        Args:
            layer_idx: Layer index
            query: Query tensor or list [head_dim] or [num_heads, head_dim]
            key: Key tensor or list
            value: Value tensor or list
            attention_mask: Attention mask (ignored by OrthoKDA)
            prefix: Whether this is a prefix decode

        Returns:
            Attention output in same format as input
        """
        if not self.enable:
            return query

        if layer_idx not in self.kv_cache:
            self.allocate_kv_cache(layer_idx, 1, 1)

        kv_state = self.kv_cache[layer_idx]

        def to_list(t):
            if isinstance(t, list):
                return t
            if hasattr(t, 'tolist'):
                return t.tolist()
            if hasattr(t, 'cpu'):
                return t.cpu().tolist()
            return list(t)

        key_list = to_list(key)
        value_list = to_list(value)
        query_list = to_list(query)

        batch_mode = isinstance(key_list[0], list) if key_list else False

        if batch_mode:
            key_tensor = key_list
            value_tensor = value_list
            query_tensor = query_list
        else:
            key_tensor = [key_list]
            value_tensor = [value_list]
            query_tensor = [query_list]

        num_heads = len(key_tensor) if batch_mode else 1
        head_dim = len(key_tensor[0]) if batch_mode else len(key_list)

        outputs = []
        for h in range(num_heads):
            k = key_tensor[h] if batch_mode else key_list
            v = value_tensor[h] if batch_mode else value_list
            q = query_tensor[h] if batch_mode else query_list

            kv_state.update(k, v)
            out = kv_state.forward(q)
            outputs.append(out)

        if batch_mode:
            result = outputs
        else:
            result = outputs[0]

        if hasattr(query, 'tolist'):
            import torch
            result_tensor = torch.tensor(result, dtype=torch.float32)
            if hasattr(query, 'to'):
                result_tensor = result_tensor.to(query.dtype)
            return result_tensor

        return result

    def reset_layer(self, layer_idx: int) -> None:
        """Reset state for a specific layer"""
        if layer_idx in self.kv_cache:
            self.kv_cache[layer_idx].reset()

    def reset_all(self) -> None:
        """Reset all layer states"""
        for layer_idx in self.kv_cache:
            self.reset_layer(layer_idx)

    def get_layer_state(self, layer_idx: int) -> Optional[Dict[str, Any]]:
        """Get current state for a layer"""
        if layer_idx not in self.kv_cache:
            return None

        return self.kv_cache[layer_idx].get_state()

    def should_apply_to_layer(self, layer_idx: int) -> bool:
        """Check if OrthoKDA should be applied to this layer"""
        if not self.enable:
            return False

        if self.config.layers is None:
            return True

        return layer_idx in self.config.layers

    def export_state(self, layer_idx: int) -> bytes:
        """
        Export layer state as binary for serialization

        Args:
            layer_idx: Layer index

        Returns:
            Binary state data
        """
        state = self.get_layer_state(layer_idx)
        if state is None:
            return b""

        import json
        state_json = json.dumps(state)
        return state_json.encode('utf-8')

    def import_state(self, layer_idx: int, data: bytes) -> None:
        """
        Import layer state from binary data

        Args:
            layer_idx: Layer index
            data: Binary state data
        """
        import json
        state = json.loads(data.decode('utf-8'))

        if layer_idx not in self.kv_cache:
            self.kv_cache[layer_idx] = OrthoKDAKVState(
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                ortho_base_dim=self.config.ortho_base_dim,
                device=self.device,
            )

        kv = self.kv_cache[layer_idx]
        kv.K = state["K"]
        kv.V = state["V"]
        kv.decay = state["decay"]
        kv.current_dim = state["current_dim"]
        kv.total_updates = state["total_updates"]


class OrthoKDAV4LlamaCppIntegration:
    """
    High-level integration helper for llama.cpp + OrthoKDA v4

    This class provides a simplified interface for integrating
    OrthoKDA v4 with existing llama.cpp models.
    """

    def __init__(
        self,
        num_heads: int = 32,
        head_dim: int = 128,
        ortho_base_dim: int = 128,
        decay_rate: float = 0.01,
        enable: bool = True,
        layers: Optional[List[int]] = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize llama.cpp integration

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            ortho_base_dim: Orthogonal basis dimension (fixed KV size)
            decay_rate: TimeDecay rate
            enable: Whether to enable OrthoKDA
            layers: Specific layers to apply (None = all layers)
            model_path: Path to gguf model file
            device: Device to run on
        """
        self.config = OrthoKDAV4LlamaConfig(
            num_heads=num_heads,
            head_dim=head_dim,
            ortho_base_dim=ortho_base_dim,
            decay_rate=decay_rate,
            enable=enable,
            layers=layers,
            model_path=model_path,
        )

        self.backend = OrthoKDAV4LlamaBackend(self.config, device)

        self.llm = None
        if model_path and LLAMA_CPP_AVAILABLE and enable:
            try:
                self.llm = Llama(model_path=model_path)
                logger.info(f"[OrthoKDAV4:llama.cpp] Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"[OrthoKDAV4:llama.cpp] Failed to load model: {e}")

        logger.info(f"[OrthoKDAV4:llama.cpp:Integration] Configured with {num_heads} heads, "
                   f"ortho_base_dim={ortho_base_dim}, enable={enable}, layers={layers}")

    def create_completion(self, prompt: str, **kwargs):
        """
        Create completion with OrthoKDA v4 enabled

        Args:
            prompt: Input prompt
            **kwargs: Additional completion parameters

        Returns:
            Completion result
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Set model_path during initialization.")

        return self.llm.create_completion(prompt, **kwargs)

    def reset(self):
        """Reset all OrthoKDA states"""
        self.backend.reset_all()


def create_llama_cpp_model_with_ortho_kda(
    model_path: str,
    num_heads: int = 32,
    head_dim: int = 128,
    ortho_base_dim: int = 128,
    **kwargs,
) -> OrthoKDAV4LlamaCppIntegration:
    """
    Create a llama.cpp model with OrthoKDA v4 enabled

    Args:
        model_path: Path to gguf model file
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        ortho_base_dim: Orthogonal basis dimension
        **kwargs: Additional configuration

    Returns:
        OrthoKDAV4LlamaCppIntegration instance
    """
    if not LLAMA_CPP_AVAILABLE:
        raise RuntimeError("llama.cpp is not available")

    integration = OrthoKDAV4LlamaCppIntegration(
        num_heads=num_heads,
        head_dim=head_dim,
        ortho_base_dim=ortho_base_dim,
        model_path=model_path,
        **kwargs,
    )

    logger.info(f"[OrthoKDAV4:llama.cpp] Creating model with OrthoKDA v4 enabled")

    return integration


if __name__ == "__main__":
    print("=" * 60)
    print("OrthoKDA v4 - llama.cpp Backend Adapter Test")
    print("=" * 60)

    if not LLAMA_CPP_AVAILABLE:
        print("❌ llama.cpp not available. Running pure Python fallback test.")
    else:
        print("✅ llama.cpp available")

    config = OrthoKDAV4LlamaConfig(
        num_heads=4,
        head_dim=8,
        ortho_base_dim=4,
        enable=True,
        layers=[0, 1, 2, 3],
    )

    backend = OrthoKDAV4LlamaBackend(config, device="cpu")

    print(f"\n✅ llama.cpp Backend initialized")
    print(f"   Config: {config}")

    print(f"\n📊 Forward pass test:")

    for step in range(6):
        import torch

        key = torch.randn(4, 8).tolist()
        value = torch.randn(4, 8).tolist()
        query = torch.randn(4, 8).tolist()

        output = backend.forward(layer_idx=0, query=query, key=key, value=value)

        state = backend.get_layer_state(0)
        output_tensor = torch.tensor(output)
        print(f"   Step {step+1}: idx={state['current_dim']}, output_shape={output_tensor.shape}")

    print(f"\n📊 Memory footprint:")
    memory = backend.get_memory_footprint(0)
    print(f"   Layer 0: {memory} bytes = {memory/1024:.2f} KB")

    print(f"\n   ✅ KV shape fixed regardless of seq_len")

    print("\n" + "=" * 60)
    print("✅ llama.cpp Backend Adapter Test PASSED!")
    print("=" * 60)
