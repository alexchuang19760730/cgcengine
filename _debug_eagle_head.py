#!/usr/bin/env python3
"""Debug script: diagnose why EAGLE accept rate is only 7%.

Key tests:
1. Compare safetensors weights vs original .pt checkpoint (structural comparison)
2. Test manual sglang-style forward using individual QKV vs stacked QKV
3. Verify that the stacked QKV approach gives same result
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

CKPT_PATH = "/data/mtp_output/qwen3vl/mtp_head_qwen3vl_decode.pt"
SAFETENSORS_PATH = "/data/eagle_drafts/qwen3vl/model.safetensors"
DRAFT_CONFIG_PATH = "/data/eagle_drafts/qwen3vl/config.json"


def test_1_weight_comparison():
    """Compare individual weights between .pt and safetensors."""
    print("\n" + "=" * 60)
    print("TEST 1: Weight comparison")
    print("=" * 60)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    pt_sd = ckpt.get("model_state_dict", ckpt)

    st_sd = load_file(SAFETENSORS_PATH)

    # Map .pt keys to safetensors keys
    pt_to_st = {
        "proj.weight": "model.fc.weight",
        "norm1.weight": "model.layers.0.input_layernorm.weight",
        "norm2.weight": "model.layers.0.post_attention_layernorm.weight",
        "norm_out.weight": "model.norm_out.weight",
        "attn.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
        "attn.k_proj.weight": "model.layers.0.self_attn.k_proj.weight",
        "attn.v_proj.weight": "model.layers.0.self_attn.v_proj.weight",
        "attn.o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
        "mlp.gate_proj.weight": "model.layers.0.mlp.gate_proj.weight",
        "mlp.up_proj.weight": "model.layers.0.mlp.up_proj.weight",
        "mlp.down_proj.weight": "model.layers.0.mlp.down_proj.weight",
    }

    all_ok = True
    for pt_key, st_key in pt_to_st.items():
        if pt_key not in pt_sd:
            print(f"  MISSING (.pt): {pt_key}")
            all_ok = False
            continue
        if st_key not in st_sd:
            print(f"  MISSING (safetensors): {st_key}")
            all_ok = False
            continue

        pt_w = pt_sd[pt_key].to(torch.bfloat16).to(torch.float32)
        st_w = st_sd[st_key].to(torch.bfloat16).to(torch.float32)

        if pt_w.shape != st_w.shape:
            print(f"  SHAPE MISMATCH: {pt_key} pt={tuple(pt_w.shape)} st={tuple(st_w.shape)}")
            all_ok = False
            continue

        max_diff = (pt_w - st_w).abs().max().item()
        mean_diff = (pt_w - st_w).abs().mean().item()
        status = "OK" if max_diff < 1e-5 else "MISMATCH!"
        if status != "OK":
            all_ok = False
        print(f"  {pt_key:35s} max_diff={max_diff:.8f} [{status}]")

    print(f"\n  Result: {'ALL OK' if all_ok else 'SOME MISMATCHES'}")
    return pt_sd


def test_2_stacked_vs_individual_qkv(pt_sd):
    """Test that QKV stacked forward gives same result as individual QKV forward."""
    print("\n" + "=" * 60)
    print("TEST 2: Stacked QKV vs Individual QKV forward")
    print("=" * 60)

    with open(DRAFT_CONFIG_PATH) as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]

    # Build both models
    # Model A: Individual QKV (like training)
    fc_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    q_a = nn.Linear(hidden_size, hidden_size, bias=False)
    k_a = nn.Linear(hidden_size, hidden_size, bias=False)
    v_a = nn.Linear(hidden_size, hidden_size, bias=False)
    o_a = nn.Linear(hidden_size, hidden_size, bias=False)
    gate_a = nn.Linear(hidden_size, intermediate_size, bias=False)
    up_a = nn.Linear(hidden_size, intermediate_size, bias=False)
    down_a = nn.Linear(intermediate_size, hidden_size, bias=False)

    fc_a.weight.data = pt_sd["proj.weight"].to(torch.bfloat16)
    q_a.weight.data = pt_sd["attn.q_proj.weight"].to(torch.bfloat16)
    k_a.weight.data = pt_sd["attn.k_proj.weight"].to(torch.bfloat16)
    v_a.weight.data = pt_sd["attn.v_proj.weight"].to(torch.bfloat16)
    o_a.weight.data = pt_sd["attn.o_proj.weight"].to(torch.bfloat16)
    gate_a.weight.data = pt_sd["mlp.gate_proj.weight"].to(torch.bfloat16)
    up_a.weight.data = pt_sd["mlp.up_proj.weight"].to(torch.bfloat16)
    down_a.weight.data = pt_sd["mlp.down_proj.weight"].to(torch.bfloat16)

    # Model B: Stacked QKV (like sglang's QKVParallelLinear)
    fc_b = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    qkv_b = nn.Linear(hidden_size, hidden_size * 3, bias=False)
    o_b = nn.Linear(hidden_size, hidden_size, bias=False)
    gate_b = nn.Linear(hidden_size, intermediate_size, bias=False)
    up_b = nn.Linear(hidden_size, intermediate_size, bias=False)
    down_b = nn.Linear(intermediate_size, hidden_size, bias=False)

    fc_b.weight.data = pt_sd["proj.weight"].to(torch.bfloat16)
    # Stack Q, K, V along dim 0
    qkv_stacked = torch.cat([
        pt_sd["attn.q_proj.weight"].to(torch.bfloat16),
        pt_sd["attn.k_proj.weight"].to(torch.bfloat16),
        pt_sd["attn.v_proj.weight"].to(torch.bfloat16),
    ], dim=0)
    qkv_b.weight.data = qkv_stacked
    o_b.weight.data = pt_sd["attn.o_proj.weight"].to(torch.bfloat16)
    gate_b.weight.data = pt_sd["mlp.gate_proj.weight"].to(torch.bfloat16)
    up_b.weight.data = pt_sd["mlp.up_proj.weight"].to(torch.bfloat16)
    down_b.weight.data = pt_sd["mlp.down_proj.weight"].to(torch.bfloat16)

    # Create random inputs with similar statistics to real hidden states
    torch.manual_seed(42)
    target_hidden = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16) * 2.0
    token_embed = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16) * 0.5

    print(f"  target_hidden: mean={target_hidden.mean().item():.4f}, std={target_hidden.std().item():.4f}")
    print(f"  token_embed: mean={token_embed.mean().item():.4f}, std={token_embed.std().item():.4f}")

    # === Forward with Model A (individual QKV) ===
    concat_a = torch.cat([target_hidden, token_embed], dim=-1)
    h_a = fc_a(concat_a)

    n1_a_weight = pt_sd["norm1.weight"].to(torch.bfloat16)
    n2_a_weight = pt_sd["norm2.weight"].to(torch.bfloat16)
    no_a_weight = pt_sd["norm_out.weight"].to(torch.bfloat16)

    # RMSNorm
    def rms_norm(x, w, eps=1e-6):
        rrms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        return x * rrms * w

    normed_a = rms_norm(h_a, n1_a_weight)
    q_a_out = q_a(normed_a)
    k_a_out = k_a(normed_a)
    v_a_out = v_a(normed_a)

    # With seq_len=1, attention = V
    attn_a = v_a_out
    attn_out_a = o_a(attn_a)
    h_a = h_a + attn_out_a

    normed2_a = rms_norm(h_a, n2_a_weight)
    gate_a_out = gate_a(normed2_a)
    up_a_out = up_a(normed2_a)
    mlp_out_a = down_a(F.silu(gate_a_out) * up_a_out)
    h_a = h_a + mlp_out_a

    hidden_a = rms_norm(h_a, no_a_weight)

    # === Forward with Model B (stacked QKV) ===
    concat_b = torch.cat([target_hidden, token_embed], dim=-1)
    h_b = fc_b(concat_b)

    normed_b = rms_norm(h_b, n1_a_weight)  # same norm weights
    qkv_out = qkv_b(normed_b)  # [1, 1, hidden*3]

    # Split: chunk(3, dim=-1) — sglang EAGLE approach
    chunks = qkv_out.chunk(3, dim=-1)
    q_b = chunks[0]; k_b = chunks[1]; v_b = chunks[2]

    # Verify individual chunks match individual projections
    q_match = (q_b - q_a_out).abs().max().item()
    k_match = (k_b - k_a_out).abs().max().item()
    v_match = (v_b - v_a_out).abs().max().item()
    print(f"  Q chunk vs Q individual: max_diff={q_match:.8f} ({'OK' if q_match < 1e-4 else 'MISMATCH!'})")
    print(f"  K chunk vs K individual: max_diff={k_match:.8f} ({'OK' if k_match < 1e-4 else 'MISMATCH!'})")
    print(f"  V chunk vs V individual: max_diff={v_match:.8f} ({'OK' if v_match < 1e-4 else 'MISMATCH!'})")

    attn_out_b = o_b(v_b)  # Using V (seq_len=1)
    h_b = h_b + attn_out_b

    normed2_b = rms_norm(h_b, n2_a_weight)
    gate_b_out = gate_b(normed2_b)
    up_b_out = up_b(normed2_b)
    mlp_out_b = down_b(F.silu(gate_b_out) * up_b_out)
    h_b = h_b + mlp_out_b

    hidden_b = rms_norm(h_b, no_a_weight)

    # === Compare ===
    diff = (hidden_a - hidden_b).abs().max().item()
    print(f"\n  Final output diff (individual vs stacked): {diff:.8f}")
    print(f"  Result: {'IDENTICAL' if diff < 1e-4 else 'DIFFERENT!'}")

    # Also test: what if we use split instead of chunk?
    qkv_split = qkv_out.split(hidden_size, dim=-1)
    v_split = qkv_split[2]
    split_match = (v_split - v_a_out).abs().max().item()
    print(f"  V split[2] vs V individual: max_diff={split_match:.8f}")

    # Test chunk ordering: chunk(3, dim=-1) gives [0:2048, 2048:4096, 4096:6144]
    # This should be [Q, K, V]
    # But what if sglang stacks as [K, Q, V] or [Q, V, K]?
    # Let's check: if we take chunk(3)[0] as V
    v_as_chunk0 = chunks[0]
    v_match0 = (v_as_chunk0 - v_a_out).abs().max().item()
    print(f"\n  Order check:")
    print(f"    chunks[0] vs V: diff={v_match0:.8f} -> {'V is chunk[0]!' if v_match0 < 1e-4 else 'V is NOT chunk[0]'}")
    print(f"    chunks[2] vs V: diff={v_match:.8f} -> {'V is chunk[2]!' if v_match < 1e-4 else 'V is NOT chunk[2]'}")

    return hidden_a, hidden_b


def test_3_qkv_ordering_sglang_style(pt_sd):
    """Check how sglang's QKVParallelLinear actually stacks weights.

    In sglang, QKVParallelLinear has a 'num_heads' and 'num_kv_heads' parameter.
    The stacking order depends on the implementation.
    """
    print("\n" + "=" * 60)
    print("TEST 3: SGLang QKV stacking order analysis")
    print("=" * 60)

    q_w = pt_sd["attn.q_proj.weight"]
    k_w = pt_sd["attn.k_proj.weight"]
    v_w = pt_sd["attn.v_proj.weight"]

    print(f"  Q weight shape: {tuple(q_w.shape)}")
    print(f"  K weight shape: {tuple(k_w.shape)}")
    print(f"  V weight shape: {tuple(v_w.shape)}")

    # In sglang, the QKVParallelLinear for Qwen2 stacks [Q, K, V] along the
    # output dimension (dim 0 of weight). The order is:
    # total_out_size = num_heads * head_dim + num_kv_heads * head_dim * 2
    # portion_sizes = [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim]
    # portion = qkv.transpose(0, 1).chunk(3, dim=-1)
    # qkv_split = [portion[0].transpose(0, 1), portion[1].transpose(0, 1), portion[2].transpose(0, 1)]
    # So Q=portion[0], K=portion[1], V=portion[2]

    # With MHA (num_kv_heads == num_heads), all portions are equal size.

    # The key question: when sglang loads individual q_proj/k_proj/v_proj weights
    # and stacks them into qkv_proj, does it stack as [Q, K, V] or [V, K, Q] or what?

    # Let's check by looking at the sglang source. The standard order should be [Q, K, V].
    # But let's also test [K, Q, V] and [V, K, Q] to be safe.

    # Our cgc_mtp_eagle.py does: v = qkv.chunk(3, dim=-1)[2]
    # This assumes [Q, K, V] order in the concatenated output.

    # But wait — the QKVParallelLinear output is along the last dimension.
    # If stacking is [Q, K, V]:
    #   output[:, :hidden] = Q
    #   output[:, hidden:2*hidden] = K
    #   output[:, 2*hidden:3*hidden] = V
    #   So chunk(3)[2] = V ✓

    # Let's verify by loading the actual sglang QKVParallelLinear and checking.
    try:
        import sglang.srt.layers.linear as sgl_linear
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        hidden_size = q_w.shape[1]
        num_heads = 16
        head_dim = 128
        num_kv_heads = 16

        # Try to create a QKVParallelLinear instance
        qkv_layer = sgl_linear.QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            quant_config=None,
            prefix="test",
        )

        # Load weights
        st_sd = load_file(SAFETENSORS_PATH)
        q_key = "model.layers.0.self_attn.q_proj.weight"
        k_key = "model.layers.0.self_attn.k_proj.weight"
        v_key = "model.layers.0.self_attn.v_proj.weight"

        for name, param in qkv_layer.named_parameters():
            if "qkv_proj" in name:
                print(f"\n  QKV param name: {name}, shape: {tuple(param.shape)}")

        # Now simulate sglang's weight loading by stacking manually
        # and check which portion maps to Q, K, V
        stacked = load_file(SAFETENSORS_PATH)

        # Actually, let me just look at how QKVParallelLinear.load_weights works.
        # In the code: weights loader stacks q/k/v along the column dimension.
        # For ColumnParallelLinear, the weight is split into portions.
        # With TP=1, no splitting occurs.

        # Let me directly check: load the safetensors qkv weights into the QKVParallelLinear
        weight_loader = qkv_layer.weight_loader_v2
        if weight_loader is None:
            weight_loader = qkv_layer.weight_loader

        print(f"  weight_loader type: {type(weight_loader).__name__}")

        # Create dummy input
        x = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)

        # Test loading individual weights
        qkv_layer.load_weights([
            (q_key, st_sd[q_key]),
            (k_key, st_sd[k_key]),
            (v_key, st_sd[v_key]),
        ])

        # Forward
        with torch.no_grad():
            output, bias = qkv_layer(x)

        print(f"  QKV output shape: {output.shape}")

        # Now compare: output chunk(3)[0] should match q_proj(x)
        q_indiv = torch.mm(x.squeeze(0).squeeze(0).unsqueeze(0), q_w.T).squeeze(0).unsqueeze(0)
        k_indiv = torch.mm(x.squeeze(0).squeeze(0).unsqueeze(0), k_w.T).squeeze(0).unsqueeze(0)
        v_indiv = torch.mm(x.squeeze(0).squeeze(0).unsqueeze(0), v_w.T).squeeze(0).unsqueeze(0)

        chunks = output.chunk(3, dim=-1)
        for i, label in enumerate(["Q", "K", "V"]):
            for j, indiv, ilabel in [(0, q_indiv, "Q"), (1, k_indiv, "K"), (2, v_indiv, "V")]:
                diff = (chunks[i] - indiv).abs().max().item()
                marker = "✓" if diff < 1e-4 else ""
                if diff < 1e-4:
                    print(f"  chunk[{i}] == {ilabel} {marker}")
                    break

    except ImportError as e:
        print(f"  Cannot import sglang: {e}")
        print(f"  Falling back to manual analysis...")

        # Without sglang, we can still verify:
        # Our conversion stacks q/k/v as individual files
        # sglang's load_weights maps them to the stacked qkv_proj.
        # The standard order is [Q, K, V].
        print(f"  Assumed stacking order: [Q, K, V]")
        print(f"  cgc_mtp_eagle.py uses chunk(3)[2] = V")


def main():
    print("=" * 60)
    print("EAGLE Draft Model Debug Suite")
    print("=" * 60)

    pt_sd = test_1_weight_comparison()

    hidden_a, hidden_b = test_2_stacked_vs_individual_qkv(pt_sd)

    try:
        test_3_qkv_ordering_sglang_style(pt_sd)
    except Exception as e:
        print(f"\n  TEST 3 failed: {e}")

    print("\n" + "=" * 60)
    print("Debug complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
