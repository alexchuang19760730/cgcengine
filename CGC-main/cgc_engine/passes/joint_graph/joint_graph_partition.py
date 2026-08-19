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

import operator
import os
from typing import Any, Optional, Sequence
from unittest.mock import patch

import torch
import torch.fx as fx
from torch._functorch.compile_utils import get_aten_target
from torch._functorch.partitioners import NodeInfo, OpTypes, get_default_op_list, min_cut_rematerialization_partition
from torch.utils._ordered_set import OrderedSet

# 临时处理 CustomPartitionerFn 不存在的问题！
try:
    from torch._inductor.custom_graph_pass import CustomPartitionerFn
    HAS_CUSTOM_PARTITIONER = True
except ImportError:
    HAS_CUSTOM_PARTITIONER = False
    # 定义一个基类作为 fallback！
    class CustomPartitionerFn:
        def uuid(self) -> Optional[Any]:
            return None

from cgc_engine.config import RecomputePolicy, get_compile_config
from cgc_engine.magi_backend.partition_rules import resolve_defined_ops
from cgc_engine.utils import compute_code_hash, magi_logger
from cgc_engine.utils.visualize import joint_graph_vis


def is_memory_increase_by_node(node: fx.Node) -> bool:
    """Check if the operation increases memory size (e.g., casting fp16 to fp32)."""
    assert get_aten_target(node) == torch.ops.prims.convert_element_type, "Only aten.to is supported"
    input_dtype = node.args[0].meta["tensor_meta"].dtype
    output_dtype = node.args[1]
    assert output_dtype is not None, "Output dtype must be specified"
    return output_dtype.itemsize > input_dtype.itemsize


def is_compute_sensitive_op(
    node: fx.Node, op_types: OpTypes, custom_compute_sensitive_ops: list[torch._ops.OpOverload]
) -> bool:
    """Check if the node is a compute-intensive operation."""
    if op_types.is_compute_intensive(node):
        return True
    if node.op != "call_function":
        return False
    if node.target in custom_compute_sensitive_ops:
        return True
    if hasattr(node.target, "default") and node.target.default in custom_compute_sensitive_ops:
        return True
    return False


def _primal_contributes_to_bwd_directly(
    primal_node: fx.Node, node_info: NodeInfo, op_types: OpTypes, custom_compute_sensitive_ops: list[torch._ops.OpOverload]
) -> bool:
    """
    FSDP ensures that weights already reside in memory.
    If there is a path from the primal (weight) to the backward pass that does not contain
    any compute-intensive operations (like matmul), it contributes to the backward pass directly,
    and we should save this primal node.
    """
    if node_info.is_required_bw(primal_node):
        return True

    worklist = {primal_node}
    visited = {primal_node}

    while worklist:
        cur_node = worklist.pop()
        for user in cur_node.users:
            if node_info.is_required_bw(user):
                return True
            if is_compute_sensitive_op(user, op_types, custom_compute_sensitive_ops):
                continue
            if user not in visited:
                visited.add(user)
                worklist.add(user)

    return False


def _push_down_save_node(node: fx.Node, node_info: NodeInfo, op_types: OpTypes) -> Optional[fx.Node]:
    """
    Starting from a compute-intensive node, walk forward through memory-efficient ops
    (views, type-narrowing casts) to find the optimal save point.

    For example, for `matmul -> view -> to(fp16)`, we save the fp16 tensor rather than
    the matmul output, since they carry the same information at a lower memory cost.

    Returns None if the node is a direct output of the forward graph (no explicit save needed).
    """
    cur_node = node
    save_node = node

    while True:
        fwd_user_nodes = [u for u in cur_node.users if node_info.is_required_fw(u)]

        if len(fwd_user_nodes) > 1:  # branch point: multiple users, save here
            return save_node
        if len(fwd_user_nodes) == 0:  # fwd graph output: autograd handles it
            return None

        next_node = fwd_user_nodes[0]

        if next_node.op == "output":
            return None

        # Try to push save_node down through memory-efficient ops
        is_view = op_types.is_view(next_node)
        is_type_convert = get_aten_target(next_node) == torch.ops.prims.convert_element_type

        if is_view:
            if save_node == cur_node:
                save_node = next_node
            cur_node = next_node
        elif is_type_convert:
            if not is_memory_increase_by_node(next_node):
                save_node = next_node
            cur_node = next_node
        else:
            return save_node


def _decide_save_node(
    node: fx.Node, node_info: NodeInfo, primal_set: frozenset, op_types: OpTypes, custom_ops: list[torch._ops.OpOverload]
) -> Optional[fx.Node]:
    """
    Unified decision function: given any node in the joint graph, return the optimal
    node to save, or None if no save is needed.

    Two cases trigger saving:
      1. Primal node: backward needs it via a path with no compute-intensive barrier.
         Save the primal as-is (it is already the smallest representation of itself).
      2. Compute-intensive forward node: push the save point down through memory-efficient
         ops (views, type-narrowing casts) to minimize memory footprint.
    """
    if node in primal_set:
        if _primal_contributes_to_bwd_directly(node, node_info, op_types, custom_ops):
            return node
        return None

    if node_info.is_required_fw(node) and is_compute_sensitive_op(node, op_types, custom_ops):
        return _push_down_save_node(node, node_info, op_types)

    return None


def _collect_save_node(save_node: fx.Node, output: OrderedSet) -> None:
    """
    Add save_node to the output set.
    If the node's output is a tuple (e.g., from ops like `torch.var_mean`),
    collect all getitem users instead of the node itself.
    """
    out_val = save_node.meta.get("val")
    assert out_val is not None, f"save_node {save_node} must have output, otherwise it's no need to save"

    if isinstance(out_val, tuple):
        for user in save_node.users:
            assert (
                user.op == "call_function" and user.target == operator.getitem
            ), f"save_node {save_node} must have a getitem user"
            output.add(user)
    else:
        output.add(save_node)


def heuristic_choose_saved_values_set(joint_graph: fx.Graph, node_info: NodeInfo, memory_budget=1) -> list[fx.Node]:
    """
    Heuristic to select which forward nodes to save for the backward pass.

    Rather than reasoning about primals and intermediates separately, we make a single
    pass over all joint-graph nodes and apply a unified decision (_decide_save_node):
      - Primal nodes that backward directly needs (no compute-intensive barrier) are saved as-is.
      - Compute-intensive forward nodes whose outputs are consumed by later forward ops
        are saved at their memory-optimal downstream position.
    """
    op_types = get_default_op_list()
    custom_ops: list[torch._ops.OpOverload] = resolve_defined_ops(
        get_compile_config().recompute_config.custom_compute_sensitive_ops
    )
    primal_set = frozenset(node_info.inputs)
    output: OrderedSet[fx.Node] = OrderedSet()

    for node in joint_graph.nodes:
        save_node = _decide_save_node(node, node_info, primal_set, op_types, custom_ops)
        if save_node is not None:
            _collect_save_node(save_node, output)

    magi_logger.info("MagiCompiler: saved_output = %s", output)
    return list(output)


def custom_joint_graph_partition_fn(
    joint_module: fx.GraphModule,
    _joint_inputs,
    compiler="inductor",
    *,
    num_fwd_outputs: int,
    static_lifetime_input_indices: Optional[list[int]] = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    recompute_config = get_compile_config().recompute_config
    partition_kwargs = dict(num_fwd_outputs=num_fwd_outputs, static_lifetime_input_indices=static_lifetime_input_indices)

    save_tensor_nodes: list[fx.Node] = []
    policy = recompute_config.recompute_policy

    if policy == RecomputePolicy.HANDCRAFT:
        magi_logger.info("MagiCompiler using handcraft recompute policy")
        # TODO: different memory budget definition from torch
        ctx = patch("torch._functorch.config.activation_memory_budget", recompute_config.memory_budget)
    elif policy == RecomputePolicy.HEURISTIC:
        magi_logger.info("MagiCompiler using heuristic recompute policy")

        def _tracked_choose(joint_graph, node_info, memory_budget=1):
            result = heuristic_choose_saved_values_set(joint_graph, node_info, memory_budget)
            save_tensor_nodes.extend(result)
            return result

        ctx = patch("torch._functorch.partitioners.choose_saved_values_set", _tracked_choose)
    elif policy == RecomputePolicy.AUTOSEARCH:
        raise ValueError("AutoSearch recompute policy is not supported yet")
    else:
        raise ValueError(f"Invalid recompute policy: {policy}")

    with ctx:
        fwd_module, bwd_module = min_cut_rematerialization_partition(joint_module, _joint_inputs, compiler, **partition_kwargs)

    joint_graph_vis(joint_module, fwd_module, bwd_module, save_tensor_nodes=save_tensor_nodes or None)

    return fwd_module, bwd_module


class CustomJointGraphPartitionFn(CustomPartitionerFn):
    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        """
        Implementation of the custom partitioner.
        """
        return custom_joint_graph_partition_fn(gm, joint_inputs, **kwargs)

    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom partitioner implementation.
        Return None to skip inductor code caching entirely.
        """
        return compute_code_hash({os.path.abspath(__file__)})
