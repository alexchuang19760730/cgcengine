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

import os
import textwrap
from typing import Dict, List, Optional, Set, Tuple

import graphviz
import torch
import torch.fx as fx

from cgc_engine.config import get_compile_config
from cgc_engine.utils import envs, magi_logger


class NodeCategory:
    FWD = "fwd"
    BWD = "bwd"
    SAVE_TENSOR = "save_tensor"
    INPUT = "input"
    FWD_OUTPUT = "fwd_output"
    BWD_OUTPUT = "bwd_output"
    TANGENT = "tangent"


NODE_CATEGORY_STYLES = {
    NodeCategory.INPUT: {"shape": "box", "style": "filled,bold", "fillcolor": "#E8F4FD", "color": "#2196F3"},
    NodeCategory.FWD: {"shape": "ellipse", "style": "filled", "fillcolor": "#E8F5E9", "color": "#4CAF50"},
    NodeCategory.BWD: {"shape": "ellipse", "style": "filled", "fillcolor": "#FFF3E0", "color": "#FF9800"},
    NodeCategory.SAVE_TENSOR: {"shape": "hexagon", "style": "filled,bold", "fillcolor": "#FFEB3B", "color": "#D32F2F"},
    NodeCategory.TANGENT: {"shape": "box", "style": "filled,bold", "fillcolor": "#B3E5FC", "color": "#0288D1"},
    NodeCategory.FWD_OUTPUT: {"shape": "box", "style": "filled,bold", "fillcolor": "#C8E6C9", "color": "#388E3C"},
    NodeCategory.BWD_OUTPUT: {"shape": "box", "style": "filled,bold", "fillcolor": "#FFCCBC", "color": "#E64A19"},
}


def get_graph_node_names(graph: fx.Graph) -> Set[str]:
    """Extract all node names from a graph."""
    return {node.name for node in graph.nodes}


def is_tangent_node(node: fx.Node) -> bool:
    """Check if a node is a tangent node (gradient input for backward)."""
    return node.name.startswith("tangent") or "tangent" in node.name.lower()


def categorize_joint_nodes(
    joint_graph: fx.Graph, fwd_graph: fx.Graph, bwd_graph: fx.Graph, save_tensor_nodes: Optional[List[fx.Node]] = None
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Categorize nodes in joint graph with priority: save_tensor > fwd > bwd.

    Returns:
        - node_categories: dict mapping node name to category
        - input_save_tensors: set of input node names that are also save tensors
    """
    fwd_names = get_graph_node_names(fwd_graph)
    bwd_names = get_graph_node_names(bwd_graph)
    save_tensor_names = {node.name for node in save_tensor_nodes} if save_tensor_nodes else set()

    node_categories = {}
    input_save_tensors = set()

    for node in joint_graph.nodes:
        if node.op == "placeholder":
            if is_tangent_node(node):
                node_categories[node.name] = NodeCategory.TANGENT
            else:
                node_categories[node.name] = NodeCategory.INPUT
                # Track if this input is also a save tensor
                if node.name in save_tensor_names:
                    input_save_tensors.add(node.name)
        elif node.op == "output":
            node_categories[node.name] = NodeCategory.BWD_OUTPUT
        elif node.name in save_tensor_names:
            node_categories[node.name] = NodeCategory.SAVE_TENSOR
        elif node.name in fwd_names:
            node_categories[node.name] = NodeCategory.FWD
        elif node.name in bwd_names:
            node_categories[node.name] = NodeCategory.BWD
        else:
            node_categories[node.name] = NodeCategory.FWD

    return node_categories, input_save_tensors


def extract_joint_graph_structure(
    graph: fx.Graph, node_categories: Dict[str, str], input_save_tensors: Optional[Set[str]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Extract nodes and edges from joint graph with category-based styling."""

    def wrap_str(text, width=40):
        return textwrap.fill(text, width=width, break_long_words=True, replace_whitespace=False)

    nodes, edges = [], []
    input_save_tensors = input_save_tensors or set()

    for node in graph.nodes:
        name_str = wrap_str(str(node.name))

        if callable(node.target):
            target_str = getattr(node.target, "__name__", str(node.target))
        elif hasattr(node.target, "_op"):
            target_str = str(node.target._op)
        else:
            target_str = str(node.target)
        target_str = wrap_str(target_str)

        category = node_categories.get(node.name, NodeCategory.FWD)
        style_info = NODE_CATEGORY_STYLES.get(category, NODE_CATEGORY_STYLES[NodeCategory.FWD])

        # Add annotation if input node is also a save tensor
        if node.name in input_save_tensors:
            node_label = f"{name_str}\n[{target_str}]\n(SaveTensor)"
        else:
            node_label = f"{name_str}\n[{target_str}]"

        nodes.append({"id": node.name, "style": style_info, "node_label": node_label, "category": category})

        def traverse_args(args_kwargs):
            if isinstance(args_kwargs, (tuple, list)):
                for arg in args_kwargs:
                    traverse_args(arg)
            elif isinstance(args_kwargs, dict):
                for val in args_kwargs.values():
                    traverse_args(val)
            elif isinstance(args_kwargs, fx.Node):
                d = {"source": args_kwargs.name, "target": node.name}
                if d not in edges:
                    edges.append(d)

        traverse_args(node.args)
        traverse_args(node.kwargs)

    return nodes, edges


def create_joint_graph_dot(nodes: List[Dict], edges: List[Dict]) -> graphviz.Digraph:
    """
    Create a graphviz Digraph for joint graph visualization.

    Layout (using rankdir=BT, bottom-to-top):
    - Top: BWD output (gradients)
    - Middle-Left: FWD cluster (inputs, fwd ops, save_tensors)
    - Middle-Right: BWD cluster (bwd ops)
    - Bottom: Tangent inputs + FWD output
    """
    dot = graphviz.Digraph(
        name="joint_graph",
        format="pdf",
        graph_attr={
            "rankdir": "BT",
            "nodesep": "0.4",
            "ranksep": "0.6",
            "overlap": "false",
            "splines": "true",
            "newrank": "true",
            "label": "Joint Graph Visualization\\n"
            "Blue: Input | Cyan: Tangent | Green: FWD | Yellow: SaveTensor | Orange: BWD | Top: BWD Output",
            "labelloc": "t",
            "fontsize": "12",
        },
        node_attr={"fontname": "Helvetica", "fontsize": "9", "fixedsize": "false", "margin": "0.12"},
    )

    input_nodes = []
    fwd_nodes = []
    save_tensor_nodes = []
    tangent_nodes = []
    bwd_nodes = []
    bwd_output_nodes = []

    for node in nodes:
        cat = node["category"]
        if cat == NodeCategory.INPUT:
            input_nodes.append(node)
        elif cat == NodeCategory.FWD:
            fwd_nodes.append(node)
        elif cat == NodeCategory.SAVE_TENSOR:
            save_tensor_nodes.append(node)
        elif cat == NodeCategory.TANGENT:
            tangent_nodes.append(node)
        elif cat == NodeCategory.BWD:
            bwd_nodes.append(node)
        elif cat == NodeCategory.BWD_OUTPUT:
            bwd_output_nodes.append(node)
        else:
            fwd_nodes.append(node)

    with dot.subgraph(name="cluster_fwd") as fwd_cluster:
        fwd_cluster.attr(label="Forward Pass", style="rounded,dashed", color="#4CAF50", bgcolor="#F1F8E9", penwidth="2")
        for node in input_nodes:
            fwd_cluster.node(node["id"], node["node_label"], **node["style"])
        for node in fwd_nodes:
            fwd_cluster.node(node["id"], node["node_label"], **node["style"])
        for node in save_tensor_nodes:
            fwd_cluster.node(node["id"], node["node_label"], **node["style"])

    with dot.subgraph(name="cluster_bwd") as bwd_cluster:
        bwd_cluster.attr(label="Backward Pass", style="rounded,dashed", color="#FF9800", bgcolor="#FFF8E1", penwidth="2")
        for node in bwd_nodes:
            bwd_cluster.node(node["id"], node["node_label"], **node["style"])

    for node in tangent_nodes:
        dot.node(node["id"], node["node_label"], **node["style"])

    for node in bwd_output_nodes:
        dot.node(node["id"], node["node_label"], **node["style"])

    with dot.subgraph() as s:
        s.attr(rank="min")
        for node in tangent_nodes:
            s.node(node["id"])
        for node in save_tensor_nodes:
            s.node(node["id"])

    with dot.subgraph() as s:
        s.attr(rank="max")
        for node in bwd_output_nodes:
            s.node(node["id"])

    if input_nodes and fwd_nodes:
        input_nodes[0]["id"]
        first_fwd = fwd_nodes[0]["id"] if fwd_nodes else None
        if first_fwd:
            pass

    if bwd_nodes and tangent_nodes:
        first_bwd = bwd_nodes[0]["id"]
        first_tangent = tangent_nodes[0]["id"]
        dot.edge(first_tangent, first_bwd, style="invis", constraint="true")

    for edge in edges:
        dot.edge(edge["source"], edge["target"])

    return dot


def get_joint_graph_path(sub_dir: str = "", filename: str = "") -> str:
    """Get the path for saving joint graph visualization."""
    cache_root_dir = get_compile_config().cache_root_dir
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    joint_graph_dir = os.path.join(cache_root_dir, "joint_graph_views", f"rank_{rank}")
    os.makedirs(joint_graph_dir, exist_ok=True)
    if sub_dir:
        joint_graph_sub_dir = os.path.join(joint_graph_dir, sub_dir)
        os.makedirs(joint_graph_sub_dir, exist_ok=True)
        if filename:
            return os.path.join(joint_graph_sub_dir, filename)
        return joint_graph_sub_dir
    if filename:
        return os.path.join(joint_graph_dir, filename)
    return joint_graph_dir


def joint_graph_vis(
    joint_module: fx.GraphModule,
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    save_tensor_nodes: Optional[List[fx.Node]] = None,
    file_path: str = None,
):
    """
    Visualize joint graph with coloring priority: save_tensor > fwd > bwd.

    Layout (bottom-to-top flow):
    - Top: BWD output (gradient outputs)
    - Middle-Left: FWD cluster (inputs + fwd ops + save_tensors)
    - Middle-Right: BWD cluster (bwd ops)
    - Bottom: Tangent inputs + FWD outputs (save_tensors)

    Node colors:
    - Blue box: Input nodes (fwd placeholders)
    - Cyan box: Tangent nodes (bwd gradient inputs)
    - Green ellipse: FWD nodes
    - Yellow hexagon: SaveTensor nodes (in FWD cluster)
    - Orange ellipse: BWD nodes
    - Orange box: BWD output (gradient outputs, at top)

    Args:
        joint_module: The joint graph module containing both fwd and bwd
        fwd_module: The forward graph module
        bwd_module: The backward graph module
        save_tensor_nodes: List of nodes that are saved tensors for backward
        file_path: Optional path to save the visualization. If None, uses default path.
    """
    if not envs.MAGI_ENABLE_FX_GRAPH_VIZ:
        magi_logger.info("Joint graph visualization is disabled. Set MAGI_ENABLE_FX_GRAPH_VIZ=true to enable it.")
        return

    joint_graph = joint_module.graph if isinstance(joint_module, fx.GraphModule) else joint_module
    fwd_graph = fwd_module.graph if isinstance(fwd_module, fx.GraphModule) else fwd_module
    bwd_graph = bwd_module.graph if isinstance(bwd_module, fx.GraphModule) else bwd_module

    node_categories, input_save_tensors = categorize_joint_nodes(joint_graph, fwd_graph, bwd_graph, save_tensor_nodes)

    nodes, edges = extract_joint_graph_structure(joint_graph, node_categories, input_save_tensors)

    dot = create_joint_graph_dot(nodes, edges)

    if file_path is None:
        file_path = get_joint_graph_path(filename="joint_graph")

    dot.render(filename=file_path, view=False, cleanup=True)
    magi_logger.info("Joint graph visualization saved to: %s.pdf", file_path)
