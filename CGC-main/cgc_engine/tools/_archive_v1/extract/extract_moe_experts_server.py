#!/usr/bin/env python3
"""
GGUF MoE 专家提取器 - 修正版
从 GGUF 中提取真正的专家权重
"""

import sys
import os
import torch

sys.path.insert(0, "/home/gs01/MagiCompiler-main")

import gguf
from pathlib import Path


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def extract_moe_experts_from_gguf(gguf_path: str, output_dir: str = "/tmp/flash_moe_experts"):
    """
    从 GGUF 文件中提取 MoE 专家权重
    """
    print_header(f"从 GGUF 提取 MoE 专家权重")

    print(f"\n📂 输入: {gguf_path}")
    print(f"📂 输出: {output_dir}")

    if not os.path.exists(gguf_path):
        print(f"❌ 文件不存在: {gguf_path}")
        return False

    file_size_gb = os.path.getsize(gguf_path) / (1024 ** 3)
    print(f"📊 文件大小: {file_size_gb:.2f} GB")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ 输出目录创建: {output_path}")

    print(f"\n🔍 读取 GGUF 文件...")
    try:
        reader = gguf.GGUFReader(gguf_path)
        print(f"✅ GGUF 读取成功")
    except Exception as e:
        print(f"❌ GGUF 读取失败: {e}")
        return False

    print(f"\n📋 模型元数据:")
    for key in ['general.architecture', 'general.name']:
        if key in reader.fields:
            value = reader.fields[key].parts[0]
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='ignore')
            print(f"   {key}: {value}")

    print(f"\n🔍 搜索 MoE 专家权重 tensor...")
    expert_tensors = {}

    for tensor in reader.tensors:
        name = tensor.name.lower()
        if 'exps' in name and any(x in name for x in ['ffn_up', 'ffn_down', 'ffn_gate']):
            expert_tensors[tensor.name] = tensor

    print(f"📊 找到 {len(expert_tensors)} 个 MoE tensor")

    # 分析第一个 tensor 来确定结构
    if len(expert_tensors) > 0:
        first_tensor_name = list(expert_tensors.keys())[0]
        first_tensor = expert_tensors[first_tensor_name]
        first_data = first_tensor.data

        if isinstance(first_data, (list, tuple)):
            first_tensor_py = torch.tensor(first_data, dtype=torch.float32)
        else:
            import numpy as np
            first_tensor_py = torch.from_numpy(np.array(first_data)).float()

        print(f"\n📐 第一个 tensor: {first_tensor_name}")
        print(f"   原始 shape: {first_tensor.shape}")
        print(f"   数据长度: {len(first_data)}")
        print(f"   转换后 shape: {first_tensor_py.shape}")

        # 推断专家数量和形状
        if first_tensor_py.dim() == 1:
            # 是一维的，需要从 tensor 名称推断
            print(f"   ⚠️ 数据是一维的，需要分析")

    # 收集每个专家的权重
    num_experts = 16
    expert_gate_weights = [None] * num_experts
    expert_up_weights = [None] * num_experts
    expert_down_weights = [None] * num_experts

    for tensor_name, tensor_info in expert_tensors.items():
        name_lower = tensor_name.lower()
        tensor_data = tensor_info.data

        # 转换为 torch tensor
        if isinstance(tensor_data, (list, tuple)):
            tensor = torch.tensor(tensor_data, dtype=torch.float32)
        else:
            import numpy as np
            tensor = torch.from_numpy(np.array(tensor_data)).float()

        print(f"\n   📦 {tensor_name}")
        print(f"      原始 shape: {tensor_info.shape}")
        print(f"      数据长度: {len(tensor_data)}")
        print(f"      转换后 shape: {tensor.shape}")

        # 根据数据类型和名称分配权重
        # MoE 专家权重在 GGUF 中通常是量化的，形状信息在 tensor_info.shape 中
        tensor_shape = list(tensor_info.shape)

        # 如果是 3D shape: [experts, dim1, dim2]
        if len(tensor_shape) == 3:
            # 假设第一个维度是专家数
            n_experts_tensor = tensor_shape[0]
            if n_experts_tensor <= 16:  # 合理范围
                num_experts = max(num_experts, n_experts_tensor)
                for exp_id in range(n_experts_tensor):
                    if 'gate' in name_lower:
                        if expert_gate_weights[exp_id] is None:
                            expert_gate_weights[exp_id] = []
                        expert_gate_weights[exp_id].append(tensor[exp_id].clone())
                    elif 'up' in name_lower:
                        if expert_up_weights[exp_id] is None:
                            expert_up_weights[exp_id] = []
                        expert_up_weights[exp_id].append(tensor[exp_id].clone())
                    elif 'down' in name_lower:
                        if expert_down_weights[exp_id] is None:
                            expert_down_weights[exp_id] = []
                        expert_down_weights[exp_id].append(tensor[exp_id].clone())
        # 如果是 2D shape: [total_params] (已经展平)
        elif len(tensor_shape) == 2:
            # 尝试展平并分割
            total_params = tensor_shape[0] * tensor_shape[1]
            if total_params == len(tensor_data):
                # 数据已经展平，需要根据专家数量分割
                params_per_expert = total_params // num_experts
                for exp_id in range(num_experts):
                    start = exp_id * params_per_expert
                    end = start + params_per_expert
                    exp_params = tensor.flatten()[start:end]

                    if 'gate' in name_lower:
                        if expert_gate_weights[exp_id] is None:
                            expert_gate_weights[exp_id] = []
                        expert_gate_weights[exp_id].append(exp_params)
                    elif 'up' in name_lower:
                        if expert_up_weights[exp_id] is None:
                            expert_up_weights[exp_id] = []
                        expert_up_weights[exp_id].append(exp_params)
                    elif 'down' in name_lower:
                        if expert_down_weights[exp_id] is None:
                            expert_down_weights[exp_id] = []
                        expert_down_weights[exp_id].append(exp_params)
        # 如果是 1D shape: [total_params] (展平)
        elif len(tensor_shape) == 1:
            params_per_expert = len(tensor_data) // num_experts
            for exp_id in range(num_experts):
                start = exp_id * params_per_expert
                end = start + params_per_expert
                exp_params = tensor.flatten()[start:end]

                if 'gate' in name_lower:
                    if expert_gate_weights[exp_id] is None:
                        expert_gate_weights[exp_id] = []
                    expert_gate_weights[exp_id].append(exp_params)
                elif 'up' in name_lower:
                    if expert_up_weights[exp_id] is None:
                        expert_up_weights[exp_id] = []
                    expert_up_weights[exp_id].append(exp_params)
                elif 'down' in name_lower:
                    if expert_down_weights[exp_id] is None:
                        expert_down_weights[exp_id] = []
                    expert_down_weights[exp_id].append(exp_params)

    print(f"\n💾 保存 {num_experts} 个专家的权重...")

    saved_count = 0
    for exp_id in range(num_experts):
        print(f"\n   处理专家 {exp_id}...")

        if expert_gate_weights[exp_id]:
            gate_tensor = torch.cat([w.flatten() for w in expert_gate_weights[exp_id]])
            gate_file = output_path / f"expert_{exp_id}_gate.bin"
            gate_tensor.numpy().tofile(str(gate_file))
            print(f"      ✅ gate: {len(gate_tensor)} 参数 -> {gate_file.name}")
            saved_count += 1

        if expert_up_weights[exp_id]:
            up_tensor = torch.cat([w.flatten() for w in expert_up_weights[exp_id]])
            up_file = output_path / f"expert_{exp_id}_up.bin"
            up_tensor.numpy().tofile(str(up_file))
            print(f"      ✅ up: {len(up_tensor)} 参数 -> {up_file.name}")
            saved_count += 1

        if expert_down_weights[exp_id]:
            down_tensor = torch.cat([w.flatten() for w in expert_down_weights[exp_id]])
            down_file = output_path / f"expert_{exp_id}_down.bin"
            down_tensor.numpy().tofile(str(down_file))
            print(f"      ✅ down: {len(down_tensor)} 参数 -> {down_file.name}")
            saved_count += 1

    print(f"\n✅ 专家权重提取完成!")
    print(f"   输出目录: {output_path}")
    print(f"   专家数量: {num_experts}")
    print(f"   保存文件: {saved_count} 个")

    return True


def test_omlx_flashmoe():
    """测试 OMLX + FlashMoE 使用提取的专家权重"""
    print_header("测试 OMLX + FlashMoE")

    try:
        from cgc_engine.omlx.client import OMLXClient
        from cgc_engine.flash_moe.client import FlashMoEClient

        expert_dir = "/tmp/flash_moe_experts"

        expert_files = list(Path(expert_dir).glob("expert_*_gate.bin"))
        if len(expert_files) == 0:
            print(f"⚠️ 未找到提取的专家权重: {expert_dir}")
            return False

        print(f"✅ 找到 {len(expert_files)} 个专家权重文件")

        omlx = OMLXClient(model_dir=expert_dir)
        omlx.num_experts = 16
        omlx.expert_dim = 4096
        print(f"✅ OMLXClient 初始化成功")

        flashmoe = FlashMoEClient(
            expert_dir=expert_dir,
            backend="cuda"
        )
        flashmoe.num_experts = 16
        flashmoe.expert_dim = 4096
        flashmoe.intermediate_dim = 6400
        print(f"✅ FlashMoEClient 初始化成功")

        print(f"\n🔮 测试专家预测...")
        x = torch.randn(1, 4096, dtype=torch.float32).cuda()
        predicted = omlx.predict_experts(x, top_k=2)
        print(f"   预测激活的专家: {predicted.flatten().tolist()}")

        expert_ids = predicted.flatten().tolist()
        unique_experts = list(set(expert_ids))
        print(f"\n📥 按需加载专家: {unique_experts}")

        flashmoe.load_experts(expert_ids=unique_experts)
        print(f"   缓存中的专家: {len(flashmoe.cache_manager)}")

        print(f"\n🚀 测试 MoE 推理...")
        result = flashmoe.mlp_forward_moe(x, top_k=2)
        print(f"   ✅ 推理成功, 输出 shape: {result.shape}")

        print(f"\n🗑️ 测试 LRU 淘汰...")
        flashmoe.cache_manager.evict_oldest()
        print(f"   淘汰后缓存专家数: {len(flashmoe.cache_manager)}")

        return True

    except Exception as e:
        print(f"❌ OMLX + FlashMoE 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print_header("🚀 GGUF MoE 专家提取器 (服务器版 v2)")

    gguf_path = "/home/gs01/models/Phi-3.5-MoE-instruct-Q4_K_M.gguf"

    success = extract_moe_experts_from_gguf(gguf_path)

    if success:
        test_omlx_flashmoe()
    else:
        print(f"\n❌ 专家提取失败，跳过 OMLX + FlashMoE 测试")

    print_header("完成")


if __name__ == "__main__":
    main()