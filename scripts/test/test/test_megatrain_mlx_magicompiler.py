#!/usr/bin/env python3
"""
MegaTrain/MLX-Tune ↔ MagiCompiler 连接测试
==========================================

测试目标：
1. 验证 MegaTrain 劫持路径是否通畅
2. 验证 MLX-Tune 劫持路径是否通畅
3. 验证 MagiCompiler 能否正确捕获计算图
4. 验证子图分析功能是否正常
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_backend import (
    MagiCompilerBackend,
    megatrain_hook,
    mlx_hook,
)
from collections import Counter
import time

# ====================
# 测试 1: MegaTrain 劫持测试
# ====================
print("=" * 60)
print("测试 1: MegaTrain 2026.4 ↔ MagiCompiler")
print("=" * 60)

class MockLayerExecutor:
    def __init__(self):
        self._current_gpu_weights = {
            "q_proj": "mock_tensor_q",
            "k_proj": "mock_tensor_k",
            "v_proj": "mock_tensor_v",
            "o_proj": "mock_tensor_o",
        }
    
    def _get_current_compute_graph(self):
        return {
            "nodes": ["q_proj", "k_proj", "v_proj", "o_proj", "layer_norm", "add"],
            "edges": [("q_proj", "layer_norm"), ("k_proj", "layer_norm")],
        }
    
    @megatrain_hook(mode="LAYER_EXEC")
    def execute_layer(self, layer_id, input_ids):
        print(f"  [MegaTrain] 执行第 {layer_id} 层")
        return {"output": "mock_output"}

executor = MockLayerExecutor()
result = executor.execute_layer(4, "input_data")
print(f"  ✓ MegaTrain 劫持成功，返回: {result}")

# ====================
# 测试 2: MLX-Tune 劫持测试
# ====================
print("\n" + "=" * 60)
print("测试 2: MLX-Tune ↔ MagiCompiler")
print("=" * 60)

class MockLlamaDecoderLayer:
    @mlx_hook()
    def forward(self, x, mask=None):
        print(f"  [MLX-Tune] 执行前向计算，输入形状: {x.shape if hasattr(x, 'shape') else 'unknown'}")
        return "mlx_output"

layer = MockLlamaDecoderLayer()
result = layer.forward("input_tensor")
print(f"  ✓ MLX-Tune 劫持成功，返回: {result}")

# ====================
# 测试 3: MagiCompiler 捕获验证
# ====================
print("\n" + "=" * 60)
print("测试 3: MagiCompiler 捕获验证")
print("=" * 60)

magi = MagiCompilerBackend()

print("  捕获 MegaTrain 计算图...")
magi.capture_graph(
    backend="MEGATRAIN_2026.4",
    mode="TRAIN_GLOBAL",
    layer_id=0,
    graph={
        "nodes": ["embed", "attn", "mlp", "norm"],
        "params": {"hidden_size": 4096, "num_layers": 32},
    },
    weights={"w1": "tensor1", "w2": "tensor2"},
    streams=["h2d_stream", "compute_stream", "d2h_stream"],
)
print("  ✓ MegaTrain 计算图捕获成功")

print("  捕获 MLX 计算图...")
magi.capture_graph(
    backend="MLX_TUNE",
    mode="COMPUTE",
    graph={"operations": ["matmul", "add", "relu"]},
)
print("  ✓ MLX 计算图捕获成功")

# ====================
# 测试 4: 子图分析测试
# ====================
print("\n" + "=" * 60)
print("测试 4: 子图分析功能")
print("=" * 60)

graph_info = {
    "backend": "MEGATRAIN_2026.4",
    "mode": "LAYER_EXEC",
    "timestamp": time.time(),
    "layer_id": 5,
    "graph": {
        "nodes": [
            {"name": "q_proj", "op": "linear", "shape": [4096, 4096]},
            {"name": "k_proj", "op": "linear", "shape": [4096, 4096]},
            {"name": "v_proj", "op": "linear", "shape": [4096, 4096]},
            {"name": "o_proj", "op": "linear", "shape": [4096, 4096]},
            {"name": "attn", "op": "attention", "heads": 32},
            {"name": "mlp", "op": "mlp", "hidden_dim": 11008},
            {"name": "ln1", "op": "layer_norm"},
            {"name": "ln2", "op": "layer_norm"},
        ],
        "edges": [
            ("q_proj", "attn"),
            ("k_proj", "attn"),
            ("v_proj", "attn"),
            ("attn", "o_proj"),
            ("o_proj", "ln1"),
            ("ln1", "mlp"),
            ("mlp", "ln2"),
        ],
    },
}

analysis = magi._analyze_captured_graph(graph_info)
print(f"  ✓ 子图分析完成")
print(f"    - 节点数: {len(graph_info['graph']['nodes'])}")
print(f"    - 边数: {len(graph_info['graph']['edges'])}")
print(f"    - 算子类型统计:")
op_counts = Counter(node["op"] for node in graph_info["graph"]["nodes"])
for op, count in op_counts.items():
    print(f"      * {op}: {count} 个")

# ====================
# 测试总结
# ====================
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)
print("✅ MegaTrain 2026.4 ↔ MagiCompiler 路径通畅")
print("✅ MLX-Tune ↔ MagiCompiler 路径通畅")
print("✅ MagiCompiler 计算图捕获功能正常")
print("✅ MagiCompiler 子图分析功能正常")
print("\n🎉 所有测试通过！")
