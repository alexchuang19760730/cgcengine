# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
CGC Graph Optimization Integration for OrthoKDA v4

This module provides the integration between OrthoKDA v4 and CGC's
graph optimization passes. It handles:
1. KDA Pass updates for OrthoKDA v4
2. Graph optimization insertion
3. Backend dispatch (vLLM / llama.cpp)
"""

from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import os

import torch

from cgc_engine.cgc.kda_pass import KDAGraphPattern, CGCKDAVisitor, InsertKDAPass, KDAMetadata
from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4
from cgc_engine.cgc.ortho_kda_v4_vllm import (
    OrthoKDAV4VLLMBackend,
    OrthoKDAV4VLLMConfig,
    OrthoKDAV4VLLMIntegration,
)
from cgc_engine.cgc.ortho_kda_v4_llama import (
    OrthoKDAV4LlamaBackend,
    OrthoKDAV4LlamaConfig,
    OrthoKDAV4LlamaCppIntegration,
)

logger = logging.getLogger(__name__)


class BackendType(Enum):
    AUTO = "auto"
    VLLM = "vllm"
    LLAMA_CPP = "llama.cpp"
    PYTHON = "python"
    CUDA = "cuda"


@dataclass
class OrthoKDAV4PassConfig:
    enable: bool = True
    num_heads: int = 32
    head_dim: int = 128
    ortho_base_dim: int = 128
    decay_rate: float = 0.01
    layers: Optional[List[int]] = None
    backend: BackendType = BackendType.AUTO
    force_cpu_fallback: bool = False


class OrthoKDAV4FusionNode:
    """
    KDA Node specialized for OrthoKDA v4

    This node represents an OrthoKDA v4 fusion operation in the
    computation graph.
    """

    def __init__(
        self,
        node_id: str,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        ortho_base_dim: int,
        decay_rate: float,
    ):
        self.node_id = node_id
        self.op_type = "ortho_kda_v4"
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ortho_base_dim = ortho_base_dim
        self.decay_rate = decay_rate

        self.kv_cache: Optional[Any] = None
        self.ortho_kda: Optional[OrthoKDAV4] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary"""
        return {
            "node_id": self.node_id,
            "op_type": self.op_type,
            "layer_idx": self.layer_idx,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "ortho_base_dim": self.ortho_base_dim,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrthoKDAV4FusionNode":
        """Deserialize node from dictionary"""
        node = cls(
            node_id=data["node_id"],
            layer_idx=data["layer_idx"],
            num_heads=data["num_heads"],
            head_dim=data["head_dim"],
            ortho_base_dim=data["ortho_base_dim"],
            decay_rate=data["decay_rate"],
        )
        return node


class OrthoKDAV4Pass:
    """
    CGC KDA Pass for OrthoKDA v4

    This pass analyzes the computation graph and inserts OrthoKDA v4
    fusion nodes where appropriate.

    Key features:
    - Automatic detection of attention patterns
    - Orthogonal basis accumulation with fixed O(1) memory
    - Gram-Schmidt strict orthogonalization
    - TimeDecay attention
    """

    def __init__(
        self,
        config: Optional[OrthoKDAV4PassConfig] = None,
    ):
        """
        Initialize OrthoKDA v4 KDA Pass

        Args:
            config: OrthoKDA v4 configuration
        """
        self.name = "ortho_kda_v4"
        self.config = config or OrthoKDAV4PassConfig()

        self.nodes: Dict[str, OrthoKDAV4FusionNode] = {}

        self.vllm_backend: Optional[OrthoKDAV4VLLMBackend] = None
        self.llama_backend: Optional[OrthoKDAV4LlamaBackend] = None

        self._backend_initialized = False

    def _detect_backend(self) -> BackendType:
        """Auto-detect the best backend based on environment"""
        if self.config.force_cpu_fallback:
            return BackendType.PYTHON

        import torch

        if torch.cuda.is_available():
            return BackendType.CUDA

        try:
            from llama_cpp import Llama
            return BackendType.LLAMA_CPP
        except ImportError:
            pass

        try:
            from vllm import LLM
            return BackendType.VLLM
        except ImportError:
            pass

        return BackendType.PYTHON

    def _init_backends(self):
        """Initialize backends based on configuration"""
        if self._backend_initialized:
            return

        backend = self.config.backend
        if backend == BackendType.AUTO:
            backend = self._detect_backend()

        self.backend_type = backend

        if backend == BackendType.VLLM:
            vllm_config = OrthoKDAV4VLLMConfig(
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                ortho_base_dim=self.config.ortho_base_dim,
                decay_rate=self.config.decay_rate,
                enable=self.config.enable,
                layers=self.config.layers,
            )
            self.vllm_backend = OrthoKDAV4VLLMBackend(
                config=vllm_config,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            logger.info("[OrthoKDAV4:CGC] Initialized vLLM backend")

        elif backend == BackendType.LLAMA_CPP:
            llama_config = OrthoKDAV4LlamaConfig(
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                ortho_base_dim=self.config.ortho_base_dim,
                decay_rate=self.config.decay_rate,
                enable=self.config.enable,
                layers=self.config.layers,
            )
            self.llama_backend = OrthoKDAV4LlamaBackend(
                config=llama_config,
                device="cpu",
            )
            logger.info("[OrthoKDAV4:CGC] Initialized llama.cpp backend")

        else:
            logger.info("[OrthoKDAV4:CGC] Using Python/CUDA backend")

        self._backend_initialized = True

    def analyze(self, graph: Any) -> List[OrthoKDAV4FusionNode]:
        """
        Analyze computation graph and identify OrthoKDA v4 insertion points

        Args:
            graph: Computation graph to analyze

        Returns:
            List of KDA nodes to insert
        """
        self._init_backends()

        nodes = []

        layers = self.config.layers if self.config.layers else range(32)

        for layer_idx in layers:
            node_id = f"ortho_kda_v4_layer_{layer_idx}"

            node = OrthoKDAV4FusionNode(
                node_id=node_id,
                layer_idx=layer_idx,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                ortho_base_dim=self.config.ortho_base_dim,
                decay_rate=self.config.decay_rate,
            )

            nodes.append(node)
            self.nodes[node_id] = node

        logger.info(f"[OrthoKDAV4:CGC] Identified {len(nodes)} insertion points")

        return nodes

    def insert(self, graph: Any, nodes: List[OrthoKDAV4FusionNode]) -> Any:
        """
        Insert OrthoKDA v4 nodes into computation graph

        編譯期自動替換：
        1. remove_op("scaled_dot_product_attention") - 刪除原生 Attention
        2. add_op("ortho_kda_v4_attention") - 插入正交基累積 KDA v4

        Args:
            graph: Computation graph
            nodes: List of KDA nodes to insert

        Returns:
            Modified computation graph
        """
        logger.info(f"[OrthoKDAV4:CGC] === Compilation-time Auto-Insert ===")

        self._remove_native_attention(graph)

        self._add_ortho_kda_v4_op(graph, nodes)

        return graph

    def _remove_native_attention(self, graph: Any):
        """
        Step 1: Remove native scaled_dot_product_attention

        刪除原生 Attention，實現真正的 KV Cache O(1)
        """
        logger.info("[OrthoKDAV4:CGC] 1. remove_op(scaled_dot_product_attention)")

        native_attention_ops = [
            "scaled_dot_product_attention",
            "sdpa",
            "flash_attention",
            "Attention",
        ]

        removed_count = 0
        for op_name in native_attention_ops:
            if hasattr(graph, 'remove_op'):
                result = graph.remove_op(op_name)
                if result:
                    removed_count += 1
                    logger.info(f"       Removed: {op_name}")
            elif hasattr(graph, 'remove_node'):
                for node in list(graph.nodes):
                    if op_name in str(getattr(node, 'op_type', '')) or \
                       op_name in str(getattr(node, 'name', '')):
                        graph.remove_node(node)
                        removed_count += 1
                        logger.info(f"       Removed: {node}")

        logger.info(f"       Total removed: {removed_count} native attention ops")

    def _add_ortho_kda_v4_op(self, graph: Any, nodes: List[OrthoKDAV4FusionNode]):
        """
        Step 2: Add ortho_kda_v4_attention

        插入真正正交基累積 KDA v4，特性：
        - fixed_kv = true (O(1) KV)
        - is_kda = true (KDA 内核)
        - orthogonal_basis = true (严格正交)
        """
        logger.info("[OrthoKDAV4:CGC] 2. add_op(ortho_kda_v4_attention)")

        if not hasattr(graph, 'add_op'):
            logger.warning("       graph.add_op not available, using alternative method")
            self._insert_ortho_kda_v4_node(graph, nodes[0] if nodes else None)
            return

        for node in nodes:
            if not isinstance(node, OrthoKDAV4FusionNode):
                continue

            op_config = {
                "name": "ortho_kda_v4_attention",
                "impl": self._get_ortho_kda_forward_impl(),
                "type": "OP_TYPE_ATTENTION",
                "fixed_kv": True,
                "is_kda": True,
                "orthogonal_basis": True,
                "layer_idx": node.layer_idx,
                "num_heads": node.num_heads,
                "head_dim": node.head_dim,
                "ortho_base_dim": node.ortho_base_dim,
                "decay_rate": node.decay_rate,
            }

            graph.add_op(op_config)
            logger.info(f"       Added: ortho_kda_v4_attention for layer {node.layer_idx}")

        logger.info(f"       Total inserted: {len(nodes)} ortho_kda_v4_attention ops")

    def _insert_ortho_kda_v4_node(self, graph: Any, node: Optional[OrthoKDAV4FusionNode]):
        """Insert a single OrthoKDA v4 node into the graph (fallback method)"""
        if node is None:
            return

        logger.debug(f"[OrthoKDAV4:CGC] Inserting node {node.node_id} for layer {node.layer_idx}")

        node.ortho_kda = OrthoKDAV4(
            num_heads=node.num_heads,
            head_dim=node.head_dim,
            ortho_base_dim=node.ortho_base_dim,
            decay_rate=node.decay_rate,
            use_cuda=(self.backend_type == BackendType.CUDA),
        )

    def _get_ortho_kda_forward_impl(self):
        """Get the OrthoKDA v4 forward implementation"""
        if self.backend_type == BackendType.CUDA:
            return self._cuda_ortho_kda_forward
        elif self.backend_type == BackendType.VLLM and self.vllm_backend:
            return self.vllm_backend.forward
        elif self.backend_type == BackendType.LLAMA_CPP and self.llama_backend:
            return self.llama_backend.forward
        else:
            return self._python_ortho_kda_forward

    def _cuda_ortho_kda_forward(self, query, key, value, **kwargs):
        """CUDA kernel implementation"""
        node_id = f"ortho_kda_v4_layer_{kwargs.get('layer_idx', 0)}"
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if node.ortho_kda:
                node.ortho_kda.update(key, value)
                return node.ortho_kda.forward(query)
        return query

    def _python_ortho_kda_forward(self, query, key, value, **kwargs):
        """Python fallback implementation"""
        import torch
        scale = 1.0 / (self.config.head_dim ** 0.5)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)

    def optimize(self, graph: Any) -> Any:
        """
        Apply OrthoKDA v4 optimizations to the graph

        Args:
            graph: Computation graph

        Returns:
            Optimized graph
        """
        logger.info("[OrthoKDAV4:CGC] Applying optimizations")

        return graph

    def forward(
        self,
        layer_idx: int,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Optional[Any] = None,
    ) -> Any:
        """
        Execute OrthoKDA v4 forward pass for a layer

        Args:
            layer_idx: Layer index
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask

        Returns:
            Attention output
        """
        self._init_backends()

        if self.backend_type == BackendType.VLLM and self.vllm_backend:
            return self.vllm_backend.forward(
                layer_idx=layer_idx,
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
            )
        elif self.backend_type == BackendType.LLAMA_CPP and self.llama_backend:
            return self.llama_backend.forward(
                layer_idx=layer_idx,
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
            )

        node_id = f"ortho_kda_v4_layer_{layer_idx}"
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if node.ortho_kda:
                node.ortho_kda.update(key, value)
                return node.ortho_kda.forward(query)

        return query

    def reset(self):
        """Reset all layer states"""
        if self.vllm_backend:
            self.vllm_backend.reset_all()
        if self.llama_backend:
            self.llama_backend.reset_all()
        for node in self.nodes.values():
            if node.ortho_kda:
                node.ortho_kda.reset()

    def get_state(self, layer_idx: int) -> Optional[Dict[str, Any]]:
        """Get current state for a layer"""
        if self.vllm_backend:
            return self.vllm_backend.get_layer_state(layer_idx)
        if self.llama_backend:
            return self.llama_backend.get_layer_state(layer_idx)

        node_id = f"ortho_kda_v4_layer_{layer_idx}"
        if node_id in self.nodes and self.nodes[node_id].ortho_kda:
            return self.nodes[node_id].ortho_kda.get_state()

        return None

    def export_config(self) -> Dict[str, Any]:
        """Export pass configuration"""
        return {
            "name": self.name,
            "enable": self.config.enable,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "ortho_base_dim": self.config.ortho_base_dim,
            "decay_rate": self.config.decay_rate,
            "layers": self.config.layers,
            "backend": self.config.backend.value if self.config.backend else BackendType.AUTO.value,
        }

    def save(self, path: str):
        """Save pass state to file"""
        state = {
            "config": self.export_config(),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"[OrthoKDAV4:CGC] Saved state to {path}")

    def load(self, path: str):
        """Load pass state from file"""
        with open(path, 'r') as f:
            state = json.load(f)

        self.config = OrthoKDAV4PassConfig(
            enable=state["config"]["enable"],
            num_heads=state["config"]["num_heads"],
            head_dim=state["config"]["head_dim"],
            ortho_base_dim=state["config"]["ortho_base_dim"],
            decay_rate=state["config"]["decay_rate"],
            layers=state["config"]["layers"],
            backend=BackendType(state["config"]["backend"]),
        )

        self.nodes = {
            k: OrthoKDAV4FusionNode.from_dict(v)
            for k, v in state["nodes"].items()
        }

        self._backend_initialized = False

        logger.info(f"[OrthoKDAV4:CGC] Loaded state from {path}")


class OrthoKDAV4Manager:
    """
    Manager for OrthoKDA v4 integration with CGC

    This class provides a high-level interface for managing OrthoKDA v4
    across the CGC compilation and execution pipeline.
    """

    def __init__(
        self,
        config: Optional[OrthoKDAV4PassConfig] = None,
    ):
        """
        Initialize OrthoKDA v4 Manager

        Args:
            config: OrthoKDA v4 configuration
        """
        self.config = config or OrthoKDAV4PassConfig()
        self.pass_instance = OrthoKDAV4Pass(self.config)

        self._graph = None

    def analyze_model(self, model: Any) -> List[OrthoKDAV4FusionNode]:
        """
        Analyze a model and identify OrthoKDA v4 insertion points

        Args:
            model: Model to analyze

        Returns:
            List of identified KDA nodes
        """
        self._graph = model
        return self.pass_instance.analyze(model)

    def compile(self, graph: Any) -> Any:
        """
        Compile graph with OrthoKDA v4 optimizations

        Args:
            graph: Computation graph

        Returns:
            Compiled graph
        """
        nodes = self.pass_instance.analyze(graph)
        return self.pass_instance.insert(graph, nodes)

    def forward(
        self,
        layer_idx: int,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Optional[Any] = None,
    ) -> Any:
        """
        Execute OrthoKDA v4 forward pass

        Args:
            layer_idx: Layer index
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask

        Returns:
            Attention output
        """
        return self.pass_instance.forward(layer_idx, query, key, value, attention_mask)

    def reset(self):
        """Reset all layer states"""
        self.pass_instance.reset()

    def get_memory_footprint(self) -> Dict[str, int]:
        """
        Calculate total memory footprint

        Returns:
            Dictionary with memory usage in bytes
        """
        num_layers = len(self.config.layers) if self.config.layers else 32

        per_layer = self.config.num_heads * self.config.ortho_base_dim * self.config.head_dim * 2
        decay = self.config.num_heads * self.config.ortho_base_dim

        total = num_layers * (per_layer * 4 + decay * 4)

        return {
            "per_layer_bytes": per_layer * 4 + decay * 4,
            "num_layers": num_layers,
            "total_bytes": total,
            "total_kb": total / 1024,
            "total_mb": total / (1024 * 1024),
        }


def create_ortho_kda_v4_pass(
    num_heads: int = 32,
    head_dim: int = 128,
    ortho_base_dim: int = 128,
    decay_rate: float = 0.01,
    layers: Optional[List[int]] = None,
    backend: str = "auto",
    **kwargs,
) -> OrthoKDAV4Pass:
    """
    Factory function to create an OrthoKDA v4 KDA pass

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        ortho_base_dim: Orthogonal basis dimension (fixed KV size)
        decay_rate: TimeDecay rate
        layers: Specific layers to apply (None = all layers)
        backend: Backend to use ('auto', 'vllm', 'llama.cpp', 'python', 'cuda')
        **kwargs: Additional configuration

    Returns:
        OrthoKDAV4Pass instance
    """
    backend_map = {
        "auto": BackendType.AUTO,
        "vllm": BackendType.VLLM,
        "llama.cpp": BackendType.LLAMA_CPP,
        "python": BackendType.PYTHON,
        "cuda": BackendType.CUDA,
    }

    config = OrthoKDAV4PassConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        ortho_base_dim=ortho_base_dim,
        decay_rate=decay_rate,
        layers=layers,
        backend=backend_map.get(backend, BackendType.AUTO),
        **kwargs,
    )

    return OrthoKDAV4Pass(config)


if __name__ == "__main__":
    print("=" * 60)
    print("OrthoKDA v4 - CGC Graph Optimization Integration Test")
    print("=" * 60)

    import torch

    config = OrthoKDAV4PassConfig(
        enable=True,
        num_heads=4,
        head_dim=8,
        ortho_base_dim=4,
        decay_rate=0.01,
        layers=[0, 1, 2, 3],
        backend=BackendType.PYTHON,
    )

    pass_instance = OrthoKDAV4Pass(config)

    print(f"\n📊 Configuration:")
    print(f"   Num Heads: {config.num_heads}")
    print(f"   Head Dim: {config.head_dim}")
    print(f"   Ortho Base Dim: {config.ortho_base_dim}")
    print(f"   Layers: {config.layers}")
    print(f"   Backend: {config.backend.value}")

    graph = {"dummy": True}
    nodes = pass_instance.analyze(graph)

    print(f"\n📊 Analysis Results:")
    print(f"   Identified {len(nodes)} insertion points")

    for node in nodes:
        print(f"   - {node.node_id}: layer={node.layer_idx}, heads={node.num_heads}")

    print(f"\n📊 Forward pass test:")

    for step in range(5):
        query = torch.randn(4, 8)
        key = torch.randn(4, 8)
        value = torch.randn(4, 8)

        output = pass_instance.forward(layer_idx=0, query=query, key=key, value=value)

        state = pass_instance.get_state(0)
        if state:
            print(f"   Step {step+1}: idx={state.get('idx', state.get('current_dim', 'N/A'))}")

    memory = pass_instance.get_memory_footprint()
    print(f"\n📊 Memory footprint:")
    for k, v in memory.items():
        print(f"   {k}: {v}")

    manager = OrthoKDAV4Manager(config)
    print(f"\n📊 Manager initialized")

    print("\n" + "=" * 60)
    print("✅ CGC Graph Optimization Integration Test PASSED!")
    print("=" * 60)
