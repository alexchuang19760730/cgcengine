"""转换 MTP Head 为 MLX 格式 (Mac 本地推理用).

用法:
  python convert_mlx.py \
    --mtp-checkpoint /data/mtp_head_output/mtp_head_final.pt \
    --base-model /Users/alexchuang/models/Qwen3-VL-2B-bf16 \
    --output /Users/alexchuang/models/Qwen3VL-2B-MTP-Head

输出 MLX 格式:
  - model.safetensors (MTP head 权重)
  - config.json (MTP head 配置)
  - 可被 mlx_lm.load() 加载 (需要自定义模型类)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MTPHead, MTPHeadConfig, create_mtp_head_for_qwen3vl_2b


def convert_to_mlx(
    checkpoint_path: str,
    base_model_path: str,
    output_dir: str,
):
    """转换 PyTorch MTP head → MLX 格式.

    1. 加载 PyTorch checkpoint
    2. 提取权重 (重命名对齐 MLX 命名)
    3. 保存为 safetensors
    4. 生成 config.json
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 checkpoint
    print(f"[convert] Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    # 2. 提取 + 重命名权重
    state_dict = ckpt["model_state_dict"]
    mlx_weights = {}

    for name, param in state_dict.items():
        if "lm_head" in name:
            # shared lm_head 不保存 (运行时从 base model 加载)
            continue
        # PyTorch → MLX 命名转换
        mlx_name = name
        # MLX 用 .weight 后缀, PyTorch 也是
        mlx_weights[mlx_name] = param.to(torch.float16).contiguous()  # MLX 常用 float16

    print(f"[convert] Extracted {len(mlx_weights)} weight tensors")

    # 3. 保存 safetensors
    weights_path = os.path.join(output_dir, "mtp_head.safetensors")
    save_file(mlx_weights, weights_path)
    print(f"[convert] Saved weights to {weights_path}")

    # 4. 保存 config.json
    config = {
        "architectures": ["MTPHead"],
        "model_type": "mtp_head",
        "hidden_size": 2048,
        "vocab_size": 151936,
        "num_heads": 16,
        "head_dim": 128,
        "intermediate_size": 5632,
        "num_hidden_layers": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "torch_dtype": "float16",
        "transformers_version": "4.45.0",
        "base_model_path": base_model_path,  # 关联的 base model
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[convert] Saved config to {config_path}")

    # 5. 保存元信息
    meta = {
        "checkpoint_path": checkpoint_path,
        "base_model_path": base_model_path,
        "step": ckpt.get("step"),
        "epochs": ckpt.get("epochs"),
        "loss": ckpt.get("loss"),
        "num_parameters": sum(p.numel() for p in state_dict.values()),
        "weight_files": ["mtp_head.safetensors"],
    }
    meta_path = os.path.join(output_dir, "mtp_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[convert] Saved meta to {meta_path}")

    print(f"\n[convert] Done. MLX MTP head saved to {output_dir}")
    print(f"[convert] To use in mlx_lm:")
    print(f"  from mlx_lm import load")
    print(f"  mtp_model, mtp_tok = load('{output_dir}')")
    print(f"  # Then use as draft_model in stream_generate")


def main():
    parser = argparse.ArgumentParser(description="Convert MTP Head to MLX format")
    parser.add_argument("--mtp-checkpoint", required=True, help="PyTorch checkpoint path")
    parser.add_argument("--base-model", required=True, help="Base model path (for reference)")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    convert_to_mlx(
        checkpoint_path=args.mtp_checkpoint,
        base_model_path=args.base_model,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
