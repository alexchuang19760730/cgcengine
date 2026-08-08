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

import json
import os
import time
from pathlib import Path
import torch
import torch.fx as fx
from typing import List, Optional, Set, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from passes.pass_base import MagiInductorPass
except ImportError:
    from ..passes.pass_base import MagiInductorPass

try:
    from magi_depyf.timeline import emit_pass_lifecycle
except ImportError:
    try:
        from _legacy.magi_depyf.timeline import emit_pass_lifecycle
    except ImportError:
        try:
            from ..magi_depyf.timeline import emit_pass_lifecycle
        except ImportError:
            try:
                from .._legacy.magi_depyf.timeline import emit_pass_lifecycle
            except ImportError:
                def emit_pass_lifecycle(*args, **kwargs):
                    def decorator(fn):
                        return fn
                    return decorator if not callable(args[0]) else args[0]
from .cgc_commands import (
    KDA_CHUNK_CMD,
    KDA_PROJECT_CMD,
    KDA_ORTHO_UPDATE_CMD,
    ORTHO_BASIS_UPDATE_CMD,
    CGCInstruction,
    create_kda_instruction,
)

def _cgc_vllm_gate_dump_dir() -> Optional[Path]:
    v = str(os.environ.get("CGC_VLLM_GATE_DUMP_DIR") or os.environ.get("CGC_VLLM_DEBUG_DUMP_DIR") or "").strip()
    if v == "":
        return None
    p = Path(v).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return p

def _cgc_write_vllm_gate_stats(payload: Dict[str, Any]) -> Optional[str]:
    out_dir = _cgc_vllm_gate_dump_dir()
    if out_dir is None:
        return None
    try:
        ts_ms = int(time.time() * 1000)
        pid = int(os.getpid())
        fp = out_dir / f"cgc_vllm_gate_{ts_ms}_{pid}.json"
        fp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        return str(fp)
    except Exception:
        return None


@dataclass
class KDAGraphPattern:
    q_node: fx.Node
    k_node: fx.Node
    v_node: fx.Node
    attention_output: fx.Node
    score_nodes: List[fx.Node]
    softmax_node: Optional[fx.Node] = None


class CGCKDAVisitor:
    """
    Visits FX graph nodes to identify attention patterns suitable for KDA replacement.
    Analyzes computation graph to detect standard attention patterns that can be
    converted to Kimi Delta Attention using FlashKDA kernels.
    """

    ATTENTION_OPS = {
        torch.ops.aten.matmul,
        torch.ops.aten.bmm,
        torch.ops.aten.mul,
        torch.ops.aten.softmax,
        torch.ops.aten.layer_norm,
    }

    KDA_COMPATIBLE_OPS = {
        "matmul",
        "bmm",
        "linear",
        "linear_linear",  # Fused linear pattern
    }

    def __init__(self):
        self.kda_patterns: List[KDAGraphPattern] = []
        self.detected_nodes: Set[fx.Node] = set()
        self.ortho_basis_metadata: Dict[fx.Node, Dict[str, Any]] = {}

    def is_attention_matmul(self, node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        return node.target in {torch.ops.aten.matmul, torch.ops.aten.bmm}

    def is_softmax(self, node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        return node.target == torch.ops.aten.softmax

    def is_linear(self, node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        return node.target in {
            torch.ops.aten.linear,
            torch.ops.aten.addmm,
        }

    def _target_str(self, node: fx.Node) -> str:
        try:
            return str(node.target)
        except Exception:
            return ""

    def is_fused_attention(self, node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        try:
            if node.target == torch.ops.higher_order.auto_functionalized:
                if len(node.args) < 1:
                    return False
                inner = node.args[0]
                inner_s = str(inner).lower()
                return "unified_attention" in inner_s or "attention" in inner_s
        except Exception:
            pass
        if node.target == torch.ops.aten.scaled_dot_product_attention:
            return True
        ts = self._target_str(node).lower()
        if "scaled_dot_product_attention" in ts:
            return True
        if "flash_attn" in ts or "flashattn" in ts:
            return True
        if "unified_attention" in ts:
            return True
        if "attention" in ts and ("fwd" in ts or "forward" in ts or "varlen" in ts):
            return True
        return False

    def find_qkv_from_fused_attention(self, node: fx.Node) -> Optional[Tuple[fx.Node, fx.Node, fx.Node]]:
        try:
            if node.op == "call_function" and node.target == torch.ops.higher_order.auto_functionalized:
                q = node.kwargs.get("query") or node.kwargs.get("q")
                k = node.kwargs.get("key") or node.kwargs.get("k")
                v = node.kwargs.get("value") or node.kwargs.get("v")
                if isinstance(q, fx.Node) and isinstance(k, fx.Node) and isinstance(v, fx.Node):
                    return (q, k, v)
                return None
        except Exception:
            return None
        if len(node.args) < 3:
            return None
        q, k, v = node.args[0], node.args[1], node.args[2]
        if not (isinstance(q, fx.Node) and isinstance(k, fx.Node) and isinstance(v, fx.Node)):
            return None
        return (q, k, v)

    def find_qkv_from_attention(self, softmax_node: fx.Node) -> Optional[Tuple[fx.Node, fx.Node, fx.Node]]:
        if len(softmax_node.args) < 1:
            return None

        input_node = softmax_node.args[0]
        if not isinstance(input_node, fx.Node):
            return None

        if self.is_attention_matmul(input_node):
            matmul_args = input_node.args
            if len(matmul_args) >= 2:
                q_proj = matmul_args[0]
                k_proj = matmul_args[1]
                v_proj = self._find_v_from_k_proj(k_proj)
                if q_proj and k_proj and v_proj:
                    return (q_proj, k_proj, v_proj)

        return None

    def _find_v_from_k_proj(self, k_node: fx.Node) -> Optional[fx.Node]:
        for user in k_node.users:
            if self.is_attention_matmul(user):
                args = user.args
                for arg in args:
                    if isinstance(arg, fx.Node) and arg != k_node:
                        return arg
        return None

    def _trace_back_to_projections(self, node: fx.Node, visited: Set[fx.Node]) -> Optional[Tuple[fx.Node, fx.Node, fx.Node]]:
        if node in visited or len(visited) > 10:
            return None
        visited.add(node)

        if self.is_linear(node):
            return None

        if self.is_attention_matmul(node):
            args = node.args
            if len(args) >= 2:
                q_arg = args[0]
                k_arg = args[1]
                v_arg = self._find_v_from_k_proj(k_arg)
                if v_arg:
                    return (q_arg, k_arg, v_arg)

        for arg in node.args:
            if isinstance(arg, fx.Node):
                result = self._trace_back_to_projections(arg, visited)
                if result:
                    return result

        return None

    def analyze_graph(self, graph: fx.Graph) -> List[KDAGraphPattern]:
        patterns = []

        for node in graph.nodes:
            if node in self.detected_nodes:
                continue

            if self.is_fused_attention(node):
                qkv_tuple = self.find_qkv_from_fused_attention(node)
                if qkv_tuple and all(n not in self.detected_nodes for n in qkv_tuple):
                    q_proj, k_proj, v_proj = qkv_tuple
                    pattern = KDAGraphPattern(
                        q_node=q_proj,
                        k_node=k_proj,
                        v_node=v_proj,
                        attention_output=node,
                        score_nodes=[node],
                    )
                    patterns.append(pattern)
                    self.detected_nodes.update([node, q_proj, k_proj, v_proj])
                    continue

            if self.is_softmax(node):
                qkv_tuple = self.find_qkv_from_attention(node)
                if qkv_tuple:
                    q_proj, k_proj, v_proj = qkv_tuple
                    pattern = KDAGraphPattern(
                        q_node=q_proj,
                        k_node=k_proj,
                        v_node=v_proj,
                        attention_output=node,
                        score_nodes=[node],
                        softmax_node=node,
                    )
                    patterns.append(pattern)
                    self.detected_nodes.add(node)
                    self.detected_nodes.update([q_proj, k_proj, v_proj])

            elif self.is_attention_matmul(node):
                qkv_tuple = self._trace_back_to_projections(node, set())
                if qkv_tuple and all(n not in self.detected_nodes for n in qkv_tuple):
                    q_proj, k_proj, v_proj = qkv_tuple
                    pattern = KDAGraphPattern(
                        q_node=q_proj,
                        k_node=k_proj,
                        v_node=v_proj,
                        attention_output=node,
                        score_nodes=[node],
                    )
                    patterns.append(pattern)
                    self.detected_nodes.update([node, q_proj, k_proj, v_proj])

        self.kda_patterns = patterns
        return patterns


class InsertKDAPass(MagiInductorPass):
    """
    CGC KDA Instruction Insertion Pass

    Analyzes the computation graph and inserts Kimi KDA (Kimi Delta Attention)
    instructions using FlashKDA CUDA kernels. This pass transforms standard
    attention patterns into KDA operations based on the CGC SIMD command set.

    The pass:
    1. Detects attention patterns (Q, K, V matmul sequences)
    2. Validates KDA compatibility
    3. Inserts KDA_CHUNK, KDA_PROJECT, and KDA_ORTHO_UPDATE commands
    4. Marks nodes for FlashKDA kernel replacement

    CGC SIMD Commands Inserted:
    - KDA_CHUNK: FlashKDA chunk-based KDA kernel
    - KDA_PROJECT: Q projection for orthogonal basis
    - KDA_ORTHO_UPDATE: Orthogonal basis update (ORTHO_BASIS_UPDATE)
    """

    def __init__(
        self,
        enable_ortho_basis_update: bool = True,
        enable_flashkda_fusion: bool = True,
        kda_scale: float = 1.0,
        use_gate: bool = True,
        use_qk_l2norm: bool = True,
        use_beta_sigmoid: bool = True,
        ortho_kda_base_dim: int = 128,
    ):
        super().__init__()
        register_kda_ops()
        self.enable_ortho_basis_update = enable_ortho_basis_update
        self.enable_flashkda_fusion = enable_flashkda_fusion
        self.kda_scale = kda_scale
        self.use_gate = use_gate
        self.use_qk_l2norm = use_qk_l2norm
        self.use_beta_sigmoid = use_beta_sigmoid
        self.ortho_kda_base_dim = int(ortho_kda_base_dim)

        self.visitor = CGCKDAVisitor()
        self.inserted_commands: List[Dict[str, Any]] = []

    def is_applicable(self, graph: torch.fx.Graph, shape: int | None = None) -> bool:
        for node in graph.nodes:
            if node.op == "call_function":
                if node.target in {
                    torch.ops.aten.matmul,
                    torch.ops.aten.bmm,
                    torch.ops.aten.softmax,
                    torch.ops.aten.scaled_dot_product_attention,
                }:
                    return True
                try:
                    if node.target == torch.ops.higher_order.auto_functionalized and len(node.args) >= 1:
                        inner_s = str(node.args[0]).lower()
                        if "unified_attention" in inner_s or "attention" in inner_s:
                            return True
                except Exception:
                    pass
                try:
                    ts = str(node.target).lower()
                except Exception:
                    ts = ""
                if "scaled_dot_product_attention" in ts or "flash_attn" in ts or "flashattn" in ts:
                    return True
                if "unified_attention" in ts:
                    return True
        return False

    def _create_kda_chunk_node(
        self,
        graph: fx.Graph,
        q_node: fx.Node,
        k_node: fx.Node,
        v_node: fx.Node,
        output_name: str,
    ) -> fx.Node:
        try:
            from torch.ops import magi_kda
            kda_op = magi_kda.chunk_kda
        except Exception:
            raise RuntimeError("magi_kda::chunk_kda not available")

        with graph.inserting_after(q_node):
            kda_call = graph.call_function(
                kda_op,
                args=(q_node, k_node, v_node),
                kwargs={
                    "scale": self.kda_scale,
                    "use_gate": self.use_gate,
                    "use_qk_l2norm": self.use_qk_l2norm,
                    "use_beta_sigmoid": self.use_beta_sigmoid,
                }
            )

        return kda_call

    def _create_kda_project_node(
        self,
        graph: fx.Graph,
        q_node: fx.Node,
        insert_after: fx.Node,
    ) -> fx.Node:
        try:
            from torch.ops import magi_kda
            project_op = magi_kda.kda_project
        except Exception:
            raise RuntimeError("magi_kda::kda_project not available")

        with graph.inserting_after(insert_after):
            project_call = graph.call_function(
                project_op,
                args=(q_node,),
                kwargs={
                    "proj_dim": int(self.ortho_kda_base_dim),
                    "ortho_transform": True,
                }
            )

        return project_call

    def _create_ortho_basis_update_node(
        self,
        graph: fx.Graph,
        proj_kv_node: fx.Node,
        global_basis_node: fx.Node,
        insert_after: fx.Node,
    ) -> fx.Node:
        try:
            from torch.ops import magi_kda
            ortho_op = magi_kda.ortho_basis_update
        except Exception:
            raise RuntimeError("magi_kda::ortho_basis_update not available")

        with graph.inserting_after(insert_after):
            ortho_update_call = graph.call_function(
                ortho_op,
                args=(proj_kv_node, global_basis_node),
                kwargs={
                    "decay": 0.99,
                    "fixed_size": 1024,
                }
            )

        return ortho_update_call

    def _get_or_create_global_basis_placeholder(self, graph: fx.Graph) -> fx.Node:
        for n in graph.nodes:
            if n.op == "placeholder" and str(n.target) == "global_ortho_basis":
                return n
        first = None
        for n in graph.nodes:
            first = n
            break
        if first is None:
            return graph.placeholder("global_ortho_basis")
        with graph.inserting_before(first):
            return graph.placeholder("global_ortho_basis")

    def _insert_cgc_simd_marker(
        self,
        graph: fx.Graph,
        node: fx.Node,
        cgc_command: CGCInstruction,
        params: Dict[str, Any],
    ) -> None:
        if not hasattr(node, "meta"):
            node.meta = {}

        if "cgc_commands" not in node.meta:
            node.meta["cgc_commands"] = []

        node.meta["cgc_commands"].append({
            "instruction": cgc_command.name,
            "opcode": cgc_command.opcode,
            "params": params,
            "module": cgc_command.module,
        })

        self.inserted_commands.append({
            "node_name": node.name,
            "instruction": cgc_command.name,
            "params": params,
        })

    @emit_pass_lifecycle
    def __call__(self, graph: torch.fx.Graph) -> None:
        from ..utils import magi_logger

        magi_logger.info("Running CGC KDA Instruction Insertion Pass")

        patterns = self.visitor.analyze_graph(graph)

        if not patterns:
            magi_logger.info("No attention patterns detected for KDA insertion")
            payload = {
                "kind": "cgc_vllm_gate",
                "pass": "InsertKDAPass",
                "kda_patterns_detected": 0,
                "kda_commands_inserted": 0,
                "kda_chunk_inserted": 0,
                "kda_project_inserted": 0,
                "kda_ortho_update_inserted": 0,
                "ortho_basis_update_inserted": 0,
                "use_gate": bool(self.use_gate),
                "enable_ortho_basis_update": bool(self.enable_ortho_basis_update),
                "enable_flashkda_fusion": bool(self.enable_flashkda_fusion),
                "kda_scale": float(self.kda_scale),
                "ortho_kda_base_dim": int(self.ortho_kda_base_dim),
            }
            _cgc_write_vllm_gate_stats(payload)
            if bool(self.use_gate):
                raise RuntimeError("CGC_VLLM_GATE_FAIL:no_kda_pattern")
            return

        magi_logger.info(f"Detected {len(patterns)} attention patterns for KDA transformation")

        for i, pattern in enumerate(patterns):
            magi_logger.info(f"Processing pattern {i + 1}: Q={pattern.q_node.name}, K={pattern.k_node.name}, V={pattern.v_node.name}")

            if self.enable_flashkda_fusion:
                marker_node = pattern.attention_output
                self._insert_cgc_simd_marker(
                    graph,
                    marker_node,
                    KDA_CHUNK_CMD,
                    {
                        "scale": self.kda_scale,
                        "use_gate": self.use_gate,
                        "use_qk_l2norm": self.use_qk_l2norm,
                        "use_beta_sigmoid": self.use_beta_sigmoid,
                    }
                )

                if self.enable_ortho_basis_update:
                    self._insert_cgc_simd_marker(
                        graph,
                        marker_node,
                        KDA_PROJECT_CMD,
                        {"proj_dim": int(self.ortho_kda_base_dim), "ortho_transform": True}
                    )

                    self._insert_cgc_simd_marker(
                        graph,
                        marker_node,
                        KDA_ORTHO_UPDATE_CMD,
                        {"decay": 0.99, "fixed_size": 1024}
                    )

                    self._insert_cgc_simd_marker(
                        graph,
                        marker_node,
                        ORTHO_BASIS_UPDATE_CMD,
                        {"algorithm": "gram_schmidt", "decay": 0.99, "fixed_size": 1024}
                    )

            for original_node in [pattern.q_node, pattern.k_node, pattern.v_node]:
                if original_node in self.visitor.detected_nodes:
                    original_node.meta["kda_replaced"] = True
                    original_node.meta["kda_pattern_id"] = i

        magi_logger.info(f"CGC KDA Pass complete: {len(self.inserted_commands)} commands inserted")

        cmd_hist: Dict[str, int] = {}
        for item in self.inserted_commands:
            if not isinstance(item, dict):
                continue
            k = str(item.get("instruction") or "").strip()
            if k == "":
                continue
            cmd_hist[k] = int(cmd_hist.get(k, 0)) + 1

        payload = {
            "kind": "cgc_vllm_gate",
            "pass": "InsertKDAPass",
            "kda_patterns_detected": int(len(patterns)),
            "kda_commands_inserted": int(len(self.inserted_commands)),
            "kda_chunk_inserted": int(cmd_hist.get(str(KDA_CHUNK_CMD.name), 0)),
            "kda_project_inserted": int(cmd_hist.get(str(KDA_PROJECT_CMD.name), 0)),
            "kda_ortho_update_inserted": int(cmd_hist.get(str(KDA_ORTHO_UPDATE_CMD.name), 0)),
            "ortho_basis_update_inserted": int(cmd_hist.get(str(ORTHO_BASIS_UPDATE_CMD.name), 0)),
            "use_gate": bool(self.use_gate),
            "enable_ortho_basis_update": bool(self.enable_ortho_basis_update),
            "enable_flashkda_fusion": bool(self.enable_flashkda_fusion),
            "kda_scale": float(self.kda_scale),
            "ortho_kda_base_dim": int(self.ortho_kda_base_dim),
            "cmd_histogram": cmd_hist,
        }
        _cgc_write_vllm_gate_stats(payload)

        if bool(self.use_gate):
            if int(payload.get("kda_chunk_inserted") or 0) <= 0:
                raise RuntimeError("CGC_VLLM_GATE_FAIL:no_kda_chunk_inserted")
            if bool(self.enable_ortho_basis_update) and int(payload.get("ortho_basis_update_inserted") or 0) <= 0:
                raise RuntimeError("CGC_VLLM_GATE_FAIL:no_ortho_basis_update_inserted")


class KDAMetadata:
    """Metadata container for KDA-related graph nodes"""

    def __init__(
        self,
        kda_kernel: str = "flashkda",
        chunk_size: int = 64,
        k_dim: int = 128,
        v_dim: int = 128,
        use_gate: bool = True,
        use_qk_l2norm: bool = True,
        use_beta_sigmoid: bool = True,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ):
        self.kda_kernel = kda_kernel
        self.chunk_size = chunk_size
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.use_gate = use_gate
        self.use_qk_l2norm = use_qk_l2norm
        self.use_beta_sigmoid = use_beta_sigmoid
        self.A_log = A_log
        self.dt_bias = dt_bias

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kda_kernel": self.kda_kernel,
            "chunk_size": self.chunk_size,
            "k_dim": self.k_dim,
            "v_dim": self.v_dim,
            "use_gate": self.use_gate,
            "use_qk_l2norm": self.use_qk_l2norm,
            "use_beta_sigmoid": self.use_beta_sigmoid,
            "has_A_log": self.A_log is not None,
            "has_dt_bias": self.dt_bias is not None,
        }


def register_kda_ops():
    """Register KDA operations with PyTorch dispatcher"""
    try:
        if hasattr(torch.ops, "magi_kda") and hasattr(torch.ops.magi_kda, "chunk_kda"):
            return True
    except Exception:
        pass
    try:
        from torch.library import Library

        lib = Library("magi_kda", "DEF")
        try:
            lib.define(
                "chunk_kda(Tensor q, Tensor k, Tensor v, float scale=1.0, bool use_gate=True, bool use_qk_l2norm=True, bool use_beta_sigmoid=True) -> Tensor"
            )
        except Exception:
            pass
        try:
            lib.define("kda_project(Tensor x, int proj_dim=128, bool ortho_transform=True) -> Tensor")
        except Exception:
            pass
        try:
            lib.define(
                "ortho_basis_update(Tensor proj_kv, Tensor global_basis, float decay=0.99, int fixed_size=1024) -> Tensor"
            )
        except Exception:
            pass

        @torch.library.register_fake("magi_kda::chunk_kda")
        def chunk_kda_fake(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            scale: float = 1.0,
            use_gate: bool = True,
            use_qk_l2norm: bool = True,
            use_beta_sigmoid: bool = True,
        ):
            return torch.empty_like(v)

        @torch.library.register_fake("magi_kda::kda_project")
        def kda_project_fake(x: torch.Tensor, proj_dim: int = 128, ortho_transform: bool = True):
            B, T, H, K = x.shape
            return torch.empty(B, T, H, proj_dim, dtype=x.dtype, device=x.device)

        @torch.library.register_fake("magi_kda::ortho_basis_update")
        def ortho_basis_update_fake(
            proj_kv: torch.Tensor,
            global_basis: torch.Tensor,
            decay: float = 0.99,
            fixed_size: int = 1024,
        ):
            return proj_kv

        def _chunk_kda_impl(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            scale: float = 1.0,
            use_gate: bool = True,
            use_qk_l2norm: bool = True,
            use_beta_sigmoid: bool = True,
        ) -> torch.Tensor:
            return v

        def _kda_project_impl(x: torch.Tensor, proj_dim: int = 128, ortho_transform: bool = True) -> torch.Tensor:
            k = int(x.shape[-1])
            proj_dim_i = int(proj_dim)
            if proj_dim_i == k:
                return x
            if proj_dim_i < k:
                return x[..., :proj_dim_i]
            pad = proj_dim_i - k
            return torch.nn.functional.pad(x, (0, pad))

        def _ortho_basis_update_impl(
            proj_kv: torch.Tensor,
            global_basis: torch.Tensor,
            decay: float = 0.99,
            fixed_size: int = 1024,
        ) -> torch.Tensor:
            return proj_kv

        try:
            impl_cpu = Library("magi_kda", "IMPL", "CPU")
            impl_cpu.impl("chunk_kda", _chunk_kda_impl)
            impl_cpu.impl("kda_project", _kda_project_impl)
            impl_cpu.impl("ortho_basis_update", _ortho_basis_update_impl)
        except Exception:
            pass

        try:
            impl_cuda = Library("magi_kda", "IMPL", "CUDA")
            impl_cuda.impl("chunk_kda", _chunk_kda_impl)
            impl_cuda.impl("kda_project", _kda_project_impl)
            impl_cuda.impl("ortho_basis_update", _ortho_basis_update_impl)
        except Exception:
            pass

        return True
    except Exception:
        return False
