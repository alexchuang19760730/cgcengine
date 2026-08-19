#!/usr/bin/env python3
"""
诊断 GGUF MoE Tensor 结构
"""

import sys
import os

sys.path.insert(0, "/home/gs01/MagiCompiler-main")

import gguf
from pathlib import Path


def main():
    gguf_path = "/home/gs01/models/Phi-3.5-MoE-instruct-Q4_K_M.gguf"

    print(f"📂 读取: {gguf_path}")
    print(f"📊 文件大小: {os.path.getsize(gguf_path) / (1024**3):.2f} GB")

    reader = gguf.GGUFReader(gguf_path)

    print(f"\n📋 元数据:")
    for key in ['general.architecture', 'general.name', 'expert_count']:
        if key in reader.fields:
            value = reader.fields[key].parts[0]
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='ignore')
            print(f"   {key}: {value}")

    print(f"\n📦 MoE 相关 Tensors:")

    for tensor in reader.tensors:
        name = tensor.name.lower()

        if 'exps' in name and any(x in name for x in ['ffn_up', 'ffn_down', 'ffn_gate']):
            # 获取数据
            tensor_data = tensor.data

            # 检查数据类型
            data_type = type(tensor_data).__name__

            # 获取 shape
            shape = tensor.shape
            n_params = len(tensor_data)

            print(f"\n   {tensor.name}:")
            print(f"      shape (from tensor): {shape}")
            print(f"      data length: {n_params}")
            print(f"      data type: {data_type}")

            # 尝试推断原始维度
            if len(shape) == 1:
                # 一维: 可能是 gate_up, gate_down 或 experts
                print(f"      可能已展平")

            # 计算合理的大小
            print(f"      shape product: {shape[0] if len(shape) == 1 else shape[0]*shape[1]*shape[2]}")

    # 特别检查第一个 MoE tensor
    print(f"\n🔍 详细分析第一个 MoE tensor:")
    first_moe_tensor = None
    for tensor in reader.tensors:
        name = tensor.name.lower()
        if 'exps' in name:
            first_moe_tensor = tensor
            break

    if first_moe_tensor:
        print(f"   名称: {first_moe_tensor.name}")
        print(f"   n_elements: {first_moe_tensor.n_elements}")
        print(f"   tensor_type: {first_moe_tensor.tensor_type}")
        print(f"   shape: {first_moe_tensor.shape}")

        # 尝试直接读取数据
        data = first_moe_tensor.data
        print(f"   数据类型: {type(data)}")
        print(f"   数据长度: {len(data)}")


if __name__ == "__main__":
    main()