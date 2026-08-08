#!/usr/bin/env python3
"""创建测试用 Qwen3MoE safetensors 模型.

用于验证 oMLX+FlashMoE expert streaming 路径.
随机权重, 计算开销与真实模型一致.
"""
import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file as np_save_file


def create_test_moe_model(
    output_dir: str,
    num_layers: int = 8,
    hidden_size: int = 512,
    intermediate_size: int = 1408,
    moe_intermediate_size: int = 1408,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 4,
    head_dim: int = 64,
    vocab_size: int = 32000,
    max_position_embeddings: int = 4096,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 1000000.0,
    tie_word_embeddings: bool = True,
):
    """创建 Qwen3MoE safetensors 模型.

    Args:
        output_dir: 输出目录
        num_layers: 层数 (小模型用 8 层, 减少测试时间)
        hidden_size: hidden 维度
        intermediate_size: dense MLP intermediate
        moe_intermediate_size: MoE expert intermediate
        num_experts: expert 数量
        num_experts_per_tok: top-k
        其他参数同 Qwen3MoE config
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 写 config.json
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_position_embeddings,
        "max_window_layers": num_layers,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": moe_intermediate_size,
        "norm_topk_prob": True,
        "num_attention_heads": num_attention_heads,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_key_value_heads,
        "output_router_logits": False,
        "rms_norm_eps": rms_norm_eps,
        "rope_scaling": None,
        "rope_theta": rope_theta,
        "router_aux_loss_coef": 0.001,
        "sliding_window": None,
        "tie_word_embeddings": tie_word_embeddings,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.52.4",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": vocab_size,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 2. 生成权重 (per-expert 格式, mlx_lm sanitize 会 stack)
    weights = {}
    scale = 0.02

    # embed_tokens
    weights["model.embed_tokens.weight"] = (
        np.random.randn(vocab_size, hidden_size).astype(np.float32) * scale
    )

    # layers
    for l in range(num_layers):
        prefix = f"model.layers.{l}"

        # attention
        n_heads = num_attention_heads
        n_kv_heads = num_key_value_heads
        weights[f"{prefix}.self_attn.q_proj.weight"] = (
            np.random.randn(n_heads * head_dim, hidden_size).astype(np.float32) * scale
        )
        weights[f"{prefix}.self_attn.k_proj.weight"] = (
            np.random.randn(n_kv_heads * head_dim, hidden_size).astype(np.float32) * scale
        )
        weights[f"{prefix}.self_attn.v_proj.weight"] = (
            np.random.randn(n_kv_heads * head_dim, hidden_size).astype(np.float32) * scale
        )
        weights[f"{prefix}.self_attn.o_proj.weight"] = (
            np.random.randn(hidden_size, n_heads * head_dim).astype(np.float32) * scale
        )
        weights[f"{prefix}.self_attn.q_norm.weight"] = np.ones((head_dim,), dtype=np.float32)
        weights[f"{prefix}.self_attn.k_norm.weight"] = np.ones((head_dim,), dtype=np.float32)

        # layernorm
        weights[f"{prefix}.input_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)
        weights[f"{prefix}.post_attention_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)

        # MoE: gate (router) + per-expert weights
        weights[f"{prefix}.mlp.gate.weight"] = (
            np.random.randn(num_experts, hidden_size).astype(np.float32) * scale
        )
        for e in range(num_experts):
            weights[f"{prefix}.mlp.experts.{e}.gate_proj.weight"] = (
                np.random.randn(moe_intermediate_size, hidden_size).astype(np.float32) * scale
            )
            weights[f"{prefix}.mlp.experts.{e}.up_proj.weight"] = (
                np.random.randn(moe_intermediate_size, hidden_size).astype(np.float32) * scale
            )
            weights[f"{prefix}.mlp.experts.{e}.down_proj.weight"] = (
                np.random.randn(hidden_size, moe_intermediate_size).astype(np.float32) * scale
            )

    # final norm
    weights["model.norm.weight"] = np.ones((hidden_size,), dtype=np.float32)

    # lm_head (if not tied)
    if not tie_word_embeddings:
        weights["lm_head.weight"] = (
            np.random.randn(vocab_size, hidden_size).astype(np.float32) * scale
        )

    # 3. 保存 safetensors (numpy 格式, mlx_lm 可加载)
    np_save_file(weights, str(output_path / "model.safetensors"))

    # 4. 写 generation_config.json
    gen_config = {
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "max_length": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "transformers_version": "4.52.4",
    }
    with open(output_path / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    # 5. 写一个简单的 tokenizer_config (复用 Qwen 格式)
    tok_config = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|im_end|>",
        "model_max_length": 4096,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}",
    }
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(tok_config, f, indent=2)

    # 6. 写 tokenizer.json (从 Qwen2.5 复制)
    import shutil
    qwen_tok = "/Users/alexchuang/Documents/flashkv0516/models/qwen25_0.5b_hf/tokenizer.json"
    if os.path.exists(qwen_tok):
        shutil.copy(qwen_tok, output_path / "tokenizer.json")

    total_params = sum(v.size for v in weights.values())
    total_mb = total_params * 2 / 1e6  # bf16

    print(f"Model created at {output_path}")
    print(f"  layers: {num_layers}, hidden: {hidden_size}")
    print(f"  experts: {num_experts}, top-k: {num_experts_per_tok}")
    print(f"  total params: {total_params:,} ({total_mb:.1f} MB bf16)")
    print(f"  files: {list(output_path.iterdir())}")


if __name__ == "__main__":
    # 小模型: 8层, 8 expert, top-2, hidden=512
    # 参数量约 50M, bf16 约 100MB
    # 计算开销与真实模型一致, 但加载快
    create_test_moe_model(
        output_dir="/Users/alexchuang/Documents/flashkv0516/models/test_moe_small",
        num_layers=8,
        hidden_size=512,
        moe_intermediate_size=1408,
        num_experts=8,
        num_experts_per_tok=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        vocab_size=32000,
    )
