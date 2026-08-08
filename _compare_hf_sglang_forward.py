#!/usr/bin/env python3
"""Comprehensive comparison: HF MTP Head vs SGLang-style MTP Head on Host2.

Loads Qwen3-VL target model via HF on GPU, gets hidden states, runs both
HF-style forward and sglang-style forward, and compares predictions.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Must be set before importing flashinfer
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

CKPT_PATH = "/data/mtp_output/qwen3vl/mtp_head_qwen3vl_decode.pt"
TARGET_MODEL_PATH = "/data2/models/Qwen3-VL-2B-Instruct"


def load_pt_weights():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    return ckpt.get("model_state_dict", ckpt)


def load_target_model():
    """Load target model on GPU."""
    print("Loading target model...")
    from transformers import AutoModel, AutoTokenizer, AutoProcessor

    # Qwen3-VL uses Qwen3VLForConditionalGeneration, not AutoModelForCausalLM
    # Load via AutoModel for the transformer backbone
    model = AutoModel.from_pretrained(
        TARGET_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH, trust_remote_code=True)

    print(f"  Model: {type(model).__name__}")
    if hasattr(model.config, 'hidden_size'):
        print(f"  Hidden size: {model.config.hidden_size}")
        vocab_size = model.config.vocab_size
    else:
        text_config = getattr(model.config, 'text_config', None)
        if text_config:
            print(f"  Hidden size (text_config): {text_config.hidden_size}")
            vocab_size = text_config.vocab_size
        else:
            print(f"  Config keys: {list(model.config.__dict__.keys())[:10]}")
            vocab_size = None
    print(f"  Vocab size: {vocab_size}")
    return model, tokenizer


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm matching training (non-sglang version)."""
    rrms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return x * rrms * weight


def hf_style_forward(pt_sd, target_hidden, input_ids, embed, lm_head):
    """Forward using HF-style individual QKV layers (like training)."""
    hidden_size = target_hidden.shape[-1]

    # Individual QKV (like training)
    q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    q_proj.weight = nn.Parameter(pt_sd["attn.q_proj.weight"].to(torch.bfloat16).cuda())
    k_proj.weight = nn.Parameter(pt_sd["attn.k_proj.weight"].to(torch.bfloat16).cuda())
    v_proj.weight = nn.Parameter(pt_sd["attn.v_proj.weight"].to(torch.bfloat16).cuda())
    o_proj.weight = nn.Parameter(pt_sd["attn.o_proj.weight"].to(torch.bfloat16).cuda())

    # FC (concat -> hidden)
    fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    fc.weight = nn.Parameter(pt_sd["proj.weight"].to(torch.bfloat16).cuda())

    # RMSNorm params
    n1_w = pt_sd["norm1.weight"].to(torch.bfloat16).cuda()
    n2_w = pt_sd["norm2.weight"].to(torch.bfloat16).cuda()
    no_w = pt_sd["norm_out.weight"].to(torch.bfloat16).cuda()

    # MLP
    intermediate = pt_sd["mlp.gate_proj.weight"].shape[0]
    gate = nn.Linear(hidden_size, intermediate, bias=False)
    up = nn.Linear(hidden_size, intermediate, bias=False)
    down = nn.Linear(intermediate, hidden_size, bias=False)
    gate.weight = nn.Parameter(pt_sd["mlp.gate_proj.weight"].to(torch.bfloat16).cuda())
    up.weight = nn.Parameter(pt_sd["mlp.up_proj.weight"].to(torch.bfloat16).cuda())
    down.weight = nn.Parameter(pt_sd["mlp.down_proj.weight"].to(torch.bfloat16).cuda())

    with torch.no_grad():
        # Build token embeddings
        token_embed = embed(input_ids)

        # Concat
        h = fc(torch.cat([target_hidden, token_embed], dim=-1))

        # norm1 -> qkv -> attention (seq_len=1 → V only)
        normed = rms_norm(h, n1_w)
        q = q_proj(normed)
        k = k_proj(normed)
        v = v_proj(normed)

        # Manual attention: seq_len=1, output = V
        attn_out = o_proj(v)
        h = h + attn_out

        # norm2 -> mlp
        normed2 = rms_norm(h, n2_w)
        mlp_out = down(F.silu(gate(normed2)) * up(normed2))
        h = h + mlp_out

        # norm_out
        h = rms_norm(h, no_w)

        # lm_head
        logits = lm_head(h)

    return logits


def sglang_style_forward(pt_sd, target_hidden, input_ids, embed, lm_head):
    """Forward using sglang-style stacked QKV (like cgc_mtp_eagle.py)."""
    hidden_size = target_hidden.shape[-1]

    # Stacked QKV
    qkv_weight = torch.cat([
        pt_sd["attn.q_proj.weight"].to(torch.bfloat16),
        pt_sd["attn.k_proj.weight"].to(torch.bfloat16),
        pt_sd["attn.v_proj.weight"].to(torch.bfloat16),
    ], dim=0).cuda()

    qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
    qkv_proj.weight = nn.Parameter(qkv_weight)

    o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    o_proj.weight = nn.Parameter(pt_sd["attn.o_proj.weight"].to(torch.bfloat16).cuda())

    fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    fc.weight = nn.Parameter(pt_sd["proj.weight"].to(torch.bfloat16).cuda())

    n1_w = pt_sd["norm1.weight"].to(torch.bfloat16).cuda()
    n2_w = pt_sd["norm2.weight"].to(torch.bfloat16).cuda()
    no_w = pt_sd["norm_out.weight"].to(torch.bfloat16).cuda()

    intermediate = pt_sd["mlp.gate_proj.weight"].shape[0]
    gate = nn.Linear(hidden_size, intermediate, bias=False)
    up = nn.Linear(hidden_size, intermediate, bias=False)
    down = nn.Linear(intermediate, hidden_size, bias=False)
    gate.weight = nn.Parameter(pt_sd["mlp.gate_proj.weight"].to(torch.bfloat16).cuda())
    up.weight = nn.Parameter(pt_sd["mlp.up_proj.weight"].to(torch.bfloat16).cuda())
    down.weight = nn.Parameter(pt_sd["mlp.down_proj.weight"].to(torch.bfloat16).cuda())

    with torch.no_grad():
        token_embed = embed(input_ids)

        # Concat
        h = fc(torch.cat([target_hidden, token_embed], dim=-1))

        # norm1 -> qkv_proj (stacked)
        normed = rms_norm(h, n1_w)
        qkv = qkv_proj(normed)  # [1, 1, hidden*3]

        # chunk(3, dim=-1)[2] = V (sglang style)
        v = qkv.chunk(3, dim=-1)[2]

        # Verify chunk order
        q_chunk = qkv.chunk(3, dim=-1)[0]
        k_chunk = qkv.chunk(3, dim=-1)[1]

        # Manual attention (seq_len=1)
        attn_out = o_proj(v)
        h = h + attn_out

        # norm2 -> mlp
        normed2 = rms_norm(h, n2_w)
        mlp_out = down(F.silu(gate(normed2)) * up(normed2))
        h = h + mlp_out

        # norm_out
        h = rms_norm(h, no_w)

        logits = lm_head(h)

    return logits


def main():
    print("=" * 60)
    print("HF vs SGLang Draft Forward Comparison")
    print("=" * 60)

    # Load weights
    pt_sd = load_pt_weights()
    print(f"\nLoaded {len(pt_sd)} weight tensors from checkpoint")

    # Load target model
    target, tokenizer = load_target_model()

    # Get embed from target
    embed = target.get_input_embeddings()

    # Load full model for lm_head and target predictions
    from transformers import Qwen3VLForConditionalGeneration
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        TARGET_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    lm_head = full_model.lm_head
    print(f"  lm_head: {type(lm_head).__name__}, weight shape: {lm_head.weight.shape}")

    # Test inputs
    test_texts = [
        "Write a Python function to compute Fibonacci numbers",
        "def hello_world():",
        "The capital of France is",
        "import numpy as",
    ]

    total_matches = 0
    total_tests = 0

    for text in test_texts:
        print(f"\n--- Text: '{text[:60]}' ---")
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            # Target forward (use base model for hidden states)
            target_out = target(
                input_ids=inputs["input_ids"],
                output_hidden_states=True,
            )
            last_hidden = target_out.last_hidden_state[:, -1:, :]  # [1, 1, hidden]
            last_input_id = inputs["input_ids"][:, -1:]  # [1, 1]

        print(f"  Target hidden: mean={last_hidden.float().mean().item():.4f}, "
              f"std={last_hidden.float().std().item():.4f}")
        print(f"  Last input token: {last_input_id.item()} '{tokenizer.decode([last_input_id.item()])}'")

        # HF-style forward
        hf_logits = hf_style_forward(pt_sd, last_hidden, last_input_id, embed, lm_head)
        hf_pred = hf_logits.argmax(dim=-1).item()
        hf_text = tokenizer.decode([hf_pred])
        hf_top5 = hf_logits[0, 0].topk(5)
        print(f"  HF-style pred: {hf_pred} '{hf_text}'")
        hf_t5 = [(hf_top5.indices[0, j].item(), tokenizer.decode([hf_top5.indices[0, j].item()])[:20]) for j in range(5)]
        print(f"  HF top-5: {hf_t5}")

        # SGLang-style forward
        sgl_logits = sglang_style_forward(pt_sd, last_hidden, last_input_id, embed, lm_head)
        sgl_pred = sgl_logits.argmax(dim=-1).item()
        sgl_text = tokenizer.decode([sgl_pred])
        sgl_top5 = sgl_logits[0, 0].topk(5)
        print(f"  SGL-style pred: {sgl_pred} '{sgl_text}'")
        sgl_t5 = [(sgl_top5.indices[0, j].item(), tokenizer.decode([sgl_top5.indices[0, j].item()])[:20]) for j in range(5)]
        print(f"  SGL top-5: {sgl_t5}")

        # Compare logits
        logit_diff = (hf_logits - sgl_logits).abs().max().item()
        logit_mean_diff = (hf_logits - sgl_logits).abs().mean().item()
        print(f"  Logit max_diff: {logit_diff:.8f}")
        print(f"  Logit mean_diff: {logit_mean_diff:.8f}")

        # Target model prediction
        with torch.no_grad():
            target_full_out = full_model(input_ids=inputs["input_ids"])
            target_pred = target_full_out.logits[:, -1, :].argmax(dim=-1).item()
            target_text = tokenizer.decode([target_pred])
            target_top5 = target_full_out.logits[:, -1, :].topk(5)
            print(f"  Target pred: {target_pred} '{target_text}'")
            if target_top5.indices.dim() == 2:
                t5 = [(target_top5.indices[0, j].item(), tokenizer.decode([target_top5.indices[0, j].item()])[:20]) for j in range(min(5, target_top5.indices.shape[1]))]
            else:
                t5 = [(target_top5.indices[j].item(), tokenizer.decode([target_top5.indices[j].item()])[:20]) for j in range(min(5, len(target_top5.indices)))]
            print(f"  Target top-5: {t5}")

        # Check match
        hf_match_target = (hf_pred == target_pred)
        sgl_match_target = (sgl_pred == target_pred)
        hf_match_sgl = (hf_logits - sgl_logits).abs().max().item() < 1e-3

        print(f"  HF == Target: {hf_match_target}")
        print(f"  SGL == Target: {sgl_match_target}")
        print(f"  HF == SGL: {hf_match_sgl}")

        total_tests += 1
        if hf_match_target:
            total_matches += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {total_matches}/{total_tests} HF-to-target matches")

    if total_matches == total_tests:
        print("✓ All HF predictions match target — architecture is correct")
    else:
        print("✗ Some predictions don't match — architecture or weights have issues")

    # Additional test: sglang hidden states format
    print(f"\n{'=' * 60}")
    print("Additional: Hidden state format check")
    print("=" * 60)

    # Get FULL hidden states (all tokens, like sglang FULL capture)
    text = "Write a Python function"
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        target_out = target(
            input_ids=inputs["input_ids"],
            output_hidden_states=True,
        )
        all_hidden = target_out.last_hidden_state  # [1, L, hidden]
        print(f"  FULL hidden states shape: {all_hidden.shape}")

        # Simulate sglang's hidden state layout
        # sglang returns [total_tokens, hidden] for FULL mode
        flat_hidden = all_hidden.view(-1, all_hidden.shape[-1])
        print(f"  Flattened shape (sglang FULL mode): {flat_hidden.shape}")

        # In draft-extend, input_ids are shifted by 1
        shifted_ids = inputs["input_ids"][:, 1:]  # drop first token
        tail_token = torch.tensor([[target_pred]], device="cuda:0")
        shifted_ids = torch.cat([shifted_ids, tail_token], dim=1)
        print(f"  Original ids: {inputs['input_ids'].shape}")
        print(f"  Shifted ids: {shifted_ids.shape}")

        # For each position i in shifted_ids:
        # hidden_states[i] = target hidden at position i
        # token_embed[i] = embedding of shifted_ids[i] (= original_ids[i+1])
        # This is the shift-by-1 EAGLE pattern.
        print(f"  Position 0: hidden[0] for '{tokenizer.decode([inputs['input_ids'][0, 0].item()])}' → token '{tokenizer.decode([inputs['input_ids'][0, 0].item()])}'")
        print(f"              embed[0] for '{tokenizer.decode([shifted_ids[0, 0].item()])}' → token '{tokenizer.decode([shifted_ids[0, 0].item()])}' (shifted)")
        print(f"              This predicts token after '{tokenizer.decode([shifted_ids[0, 0].item()])}'")

        # Test: for each position, can the MTP head predict the correct next token?
        # For position i: hidden[i] + embed(shifted_ids[i]) → predict actual token at position i+2
        print(f"\n  === Per-position prediction test ===")
        L = all_hidden.shape[1]
        for i in range(min(L, 5)):
            pos_hidden = flat_hidden[i:i+1].unsqueeze(0)  # [1, 1, hidden]
            pos_input_id = shifted_ids[:, i:i+1]  # [1, 1]

            # HF-style forward on this position
            pos_logits = hf_style_forward(pt_sd, pos_hidden, pos_input_id, embed, lm_head)
            pos_pred = pos_logits.argmax(dim=-1).item()
            pos_text = tokenizer.decode([pos_pred])

            # What token actually comes after shifted_ids[i]?
            if i + 1 < shifted_ids.shape[1]:
                actual_next = shifted_ids[0, i + 1].item()
                actual_text = tokenizer.decode([actual_next])
                match = "✓" if pos_pred == actual_next else "✗"
            else:
                actual_next = target_pred
                actual_text = tokenizer.decode([actual_next])
                match = "✓" if pos_pred == actual_next else "✗"

            print(f"  pos[{i}]: hidden for '{tokenizer.decode([inputs['input_ids'][0, i].item()])[:15]}' + "
                  f"embed('{tokenizer.decode([shifted_ids[0, i].item()])[:15]}') → "
                  f"pred='{pos_text[:15]}', actual='{actual_text[:15]}' {match}")


if __name__ == "__main__":
    main()
