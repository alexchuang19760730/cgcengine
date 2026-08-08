#!/usr/bin/env python3
"""
自动生成测试演示
验证：后端感知 + 硬件感知 + 图结构感知 + 模式感知
"""

import sys

from cgc_engine.utils.knowledge_storage import auto_generate


def create_test_graph():
    """创建一个模拟的Transformer推理计算图"""
    return {
        'nodes': [
            # Embedding层
            {'id': 0, 'op_type': 'embedding', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            
            # Transformer层 0
            {'id': 1, 'op_type': 'linear', 'output_shape': [32, 512, 12288], 'dtype': 'float16'},
            {'id': 2, 'op_type': 'gelu', 'output_shape': [32, 512, 12288], 'dtype': 'float16'},
            {'id': 3, 'op_type': 'linear', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            {'id': 4, 'op_type': 'scaled_dot_product_attention', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            {'id': 5, 'op_type': 'layer_norm', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            
            # Transformer层 1
            {'id': 6, 'op_type': 'linear', 'output_shape': [32, 512, 12288], 'dtype': 'float16'},
            {'id': 7, 'op_type': 'gelu', 'output_shape': [32, 512, 12288], 'dtype': 'float16'},
            {'id': 8, 'op_type': 'linear', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            {'id': 9, 'op_type': 'scaled_dot_product_attention', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            {'id': 10, 'op_type': 'layer_norm', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            
            # Transformer层 2
            {'id': 11, 'op_type': 'linear', 'output_shape': [32, 512, 12288], 'dtype': 'float16'},
            {'id': 12, 'op_type': 'gelu', 'output_shape': [32, 512, 12288], 'dtype': 'float16'},
            {'id': 13, 'op_type': 'linear', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            {'id': 14, 'op_type': 'scaled_dot_product_attention', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            {'id': 15, 'op_type': 'layer_norm', 'output_shape': [32, 512, 4096], 'dtype': 'float16'},
            
            # LM Head
            {'id': 16, 'op_type': 'linear', 'output_shape': [32, 512, 32000], 'dtype': 'float16'},
            {'id': 17, 'op_type': 'softmax', 'output_shape': [32, 512, 32000], 'dtype': 'float16'}
        ],
        'edges': [
            # Embedding -> Transformer 0
            {'from': 0, 'to': 1},
            {'from': 0, 'to': 4},
            
            # Transformer 0内部
            {'from': 1, 'to': 2},
            {'from': 2, 'to': 3},
            {'from': 3, 'to': 5},
            {'from': 4, 'to': 5},
            
            # Transformer 0 -> Transformer 1
            {'from': 5, 'to': 6},
            {'from': 5, 'to': 9},
            
            # Transformer 1内部
            {'from': 6, 'to': 7},
            {'from': 7, 'to': 8},
            {'from': 8, 'to': 10},
            {'from': 9, 'to': 10},
            
            # Transformer 1 -> Transformer 2
            {'from': 10, 'to': 11},
            {'from': 10, 'to': 14},
            
            # Transformer 2内部
            {'from': 11, 'to': 12},
            {'from': 12, 'to': 13},
            {'from': 13, 'to': 15},
            {'from': 14, 'to': 15},
            
            # Transformer 2 -> LM Head
            {'from': 15, 'to': 16},
            {'from': 16, 'to': 17}
        ]
    }


def print_results(result):
    """打印自动生成结果"""
    print("=" * 100)
    print("🎯 自动生成结果")
    print("=" * 100)
    
    # 平台信息
    print("\n📱 后端感知 + 硬件感知")
    print("-" * 50)
    platform = result['platform']
    print(f"  操作系统: {platform['os']}")
    print(f"  后端: {platform['backend']}")
    print(f"  硬件: {platform['hardware']}")
    print(f"  设备数量: {platform['num_devices']}")
    print(f"  内存: {platform['memory_gb']:.1f} GB")
    
    # 图分析
    print("\n🔍 图结构感知")
    print("-" * 50)
    graph = result['graph_analysis']
    print(f"  节点数量: {graph['nodes_count']}")
    print(f"  并行分组: {graph['parallel_groups']}")
    print(f"  关键路径长度: {graph['critical_path']}")
    print(f"  内存估算: {graph['memory_estimate_gb']:.2f} GB")
    
    # 模式匹配
    print("\n✨ 模式感知")
    print("-" * 50)
    optimizations = result['optimizations']
    print(f"  识别到 {len(optimizations)} 个优化模式")
    
    for i, opt in enumerate(optimizations, 1):
        print(f"\n  [{i}] {opt['pattern_id']}")
        print(f"     类型: {opt['pattern_type']}")
        print(f"     置信度: {opt['confidence']:.2f}")
        print(f"     预估收益: {opt['estimated_gain']:.2f}x")
        print(f"     优化代码:")
        code_lines = opt['code'].strip().split('\n')
        for line in code_lines[:5]:  # 只显示前5行
            print(f"       {line}")
        if len(code_lines) > 5:
            print("       ...")
    
    print(f"\n📊 总预估加速比: {result['total_estimated_gain']:.2f}x")
    print(f"✅ 最优后端: {result['optimal_backend']}")
    print("\n" + "=" * 100)


def main():
    """主函数"""
    print("🚀 启动自动生成演示...")
    print("自动生成 = 后端感知 + 硬件感知 + 图结构感知 + 模式感知\n")
    
    # 创建测试图
    graph = create_test_graph()
    print(f"📈 测试图: {len(graph['nodes'])} 个节点, {len(graph['edges'])} 条边")
    
    # 执行自动生成
    result = auto_generate(graph)
    
    # 打印结果
    print_results(result)
    
    print("\n💾 优化知识已自动保存到知识库")


if __name__ == "__main__":
    main()
