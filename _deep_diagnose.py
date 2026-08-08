#!/usr/bin/env python3
"""Deep diagnose: check MTP head per-token accuracy on a real prompt.
Simplified version - directly load weights, no sglang class dependency."""

import torch
import torch.nn.functional as F
import sys

TARGET_PATH = "/data2/models/Qwen3-VL-2B-Instruct"
CKPT_PATH = "/data/mtp_output/qwen3vl/mtp_head_qwen3vl_decode.pt"
PROMPT = "Write a Python function to compute Fibonacci numbers recursively"


def manual_mtp_forward(weights, hidden, token_embed):
    """Manual MTP head forward matching training code distill_train.py.
    
    x = cat(hidden, token_embed)
    x = proj(x)
    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))
    x = norm_out(x)
    """
    # Concatenate [B, hidden] + [B, hidden] -> [B, 2*hidden]
    x = torch.cat([hidden, token_embed], dim=-1)  # [B, 2*hidden]
    x = F.linear(x, weights["proj.weight"], weights.get("proj.bias"))  # [B, hidden]
    
    # Attention block (simplified: seq_len=1 just use linear)
    # For single token, attention degrades to: Q,K,V projections then V output
    # But we need to match the training code which does full self-attention
    # Training code: x = x + attn(norm1(x))
    # Here we need the full attention because weights matter
    
    h = x.unsqueeze(1)  # [B, 1, hidden]
    
    # Layer norm 1
    norm1 = weights["norm1.weight"]
    eps = 1e-6
    h_norm = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm1
    
    # Q, K, V projections (separate)
    q = F.linear(h_norm, weights["attn.q_proj.weight"], weights.get("attn.q_proj.bias"))
    k = F.linear(h_norm, weights["attn.k_proj.weight"], weights.get("attn.k_proj.bias"))
    v = F.linear(h_norm, weights["attn.v_proj.weight"], weights.get("attn.v_proj.bias"))
    
    # Simple attention: V (since seq_len=1, attention is identity after softmax)
    attn_out = F.linear(v, weights["attn.o_proj.weight"], weights.get("attn.o_proj.bias"))
    
    # Residual
    x = x.unsqueeze(1) + attn_out  # [B, 1, hidden]
    
    # MLP block
    h2 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weights["norm2.weight"]
    # MLP: gate projection + up projection -> silu(gate) * up -> down
    gate = F.linear(h2, weights["mlp.gate_proj.weight"])
    up = F.linear(h2, weights["mlp.up_proj.weight"])
    mlp_hidden = F.silu(gate) * up
    mlp_out = F.linear(mlp_hidden, weights["mlp.down_proj.weight"])
    
    x = x + mlp_out  # [B, 1, hidden]
    
    # Final norm
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weights["norm_out.weight"]
    
    return x.squeeze(1)  # [B, hidden]


def main():
    print("=" * 60)
    print("Deep Diagnose: MTP Head Per-Token Accuracy")
    print("=" * 60)

    # Load target model
    print("\n1. Loading target model...")
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        TARGET_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True)
    
    config = model.config
    text_config = config.text_config
    hidden_size = text_config.hidden_size
    vocab_size = text_config.vocab_size
    print(f"  Hidden: {hidden_size}, Vocab: {vocab_size}")

    # Extract embed and lm_head
    lang_model = model.model.language_model
    embed = lang_model.embed_tokens
    lm_head = model.lm_head
    
    # Load checkpoint
    print("\n2. Loading MTP checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    print(f"  Checkpoint keys ({len(state_dict)}):")
    for k, v in state_dict.items():
        print(f"    {k}: {tuple(v.shape)} {v.dtype}")
    
    # Check proj.weight dimensions
    proj_w = state_dict["proj.weight"]
    print(f"\n  proj.weight: {tuple(proj_w.shape)}, expects 2*hidden_size={2*hidden_size}")
    if proj_w.shape[1] != 2 * hidden_size:
        print(f"  WARNING: proj input dim {proj_w.shape[1]} != 2*{hidden_size}={2*hidden_size}!")
    
    # Move to GPU
    device = torch.device("cuda:0")
    weights_gpu = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in state_dict.items()}

    # Tokenize
    print(f"\n3. Testing prompt: '{PROMPT}'")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # [1, L]
    L = input_ids.shape[1]
    print(f"  Tokens: {L}")
    
    # Get hidden states from target model
    print("\n4. Getting target hidden states...")
    with torch.no_grad():
        lang_output = lang_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        all_hidden = lang_output.last_hidden_state  # [1, L, hidden]
        target_logits = lm_head(all_hidden)  # [1, L, vocab]
    
    print(f"  Hidden shape: {all_hidden.shape}")
    print(f"  Logits shape: {target_logits.shape}")
    
    # Hidden stats
    print(f"  Hidden norm (avg): {all_hidden.norm(dim=-1).mean().item():.4f}")
    print(f"  Hidden mean: {all_hidden.mean().item():.6f}")
    print(f"  Hidden std: {all_hidden.std().item():.6f}")
    
    # Test per-position prediction
    print("\n5. Per-position prediction accuracy:")
    print(f"{'Pos':>4} {'Input Token':>25} {'Target(→next)':>25} {'MTP Pred':>25} {'Match':>6} {'Hidden norm':>12}")
    print("-" * 105)
    
    match_count = 0
    top5_count = 0
    total = L - 1
    
    for i in range(total):
        hidden_i = all_hidden[0, i, :]  # [hidden]
        token_i = input_ids[0, i:i+1]   # [1]
        
        token_embed = embed(token_i.unsqueeze(0)).squeeze(0).squeeze(0)  # [hidden]
        
        # Target prediction
        target_pred_id = target_logits[0, i].argmax(dim=-1).item()
        target_pred_text = tokenizer.decode([target_pred_id])
        
        # MTP prediction
        mtp_hidden = manual_mtp_forward(weights_gpu, hidden_i.unsqueeze(0), token_embed.unsqueeze(0))
        mtp_logits = lm_head(mtp_hidden)
        mtp_pred_id = mtp_logits[0].argmax(dim=-1).item()
        mtp_pred_text = tokenizer.decode([mtp_pred_id])
        
        match = mtp_pred_id == target_pred_id
        top5_ids = mtp_logits[0].topk(5).indices.tolist()
        in_top5 = target_pred_id in top5_ids
        
        if match:
            match_count += 1
        if in_top5:
            top5_count += 1
        
        input_text = tokenizer.decode([input_ids[0, i].item()])
        hidden_norm = hidden_i.norm().item()
        marker = "✓" if match else ("▴" if in_top5 else "✗")
        
        # Only print if not match (to see failures)
        if not match:
            print(f"{i:>4} {input_text[:25]:>25} {target_pred_text[:25]:>25} "
                  f"{mtp_pred_text[:25]:>25} {marker:>6} {hidden_norm:>12.2f}")
    
    print("-" * 105)
    print(f"\n  Exact match: {match_count}/{total} = {match_count/max(total,1)*100:.1f}%")
    print(f"  Top-5 match: {top5_count}/{total} = {top5_count/max(total,1)*100:.1f}%")
    
    # Last token detail (EAGLE decode scenario)
    print("\n6. Last token prediction detail:")
    last_hidden = all_hidden[0, -1, :].unsqueeze(0)
    last_token = input_ids[0, -1:]
    last_token_embed = embed(last_token.unsqueeze(0)).squeeze(0).squeeze(0).unsqueeze(0)
    
    mtp_h = manual_mtp_forward(weights_gpu, last_hidden, last_token_embed)
    mtp_logits_last = lm_head(mtp_h)
    mtp_top5 = mtp_logits_last[0].topk(5)
    
    target_logits_last = target_logits[0, -1]
    target_top5 = target_logits_last.topk(5)
    
    print(f"  Input token: '{tokenizer.decode([last_token.item()])}' (id={last_token.item()})")
    print(f"  Hidden norm: {last_hidden.norm().item():.2f}")
    print(f"  MTP top-5:    {[(t.item(), tokenizer.decode([t.item()])[:20]) for t in mtp_top5.indices]}")
    print(f"  Target top-5: {[(t.item(), tokenizer.decode([t.item()])[:20]) for t in target_top5.indices]}")
    
    # Check: does MTP predict the same as embedding-only?
    print("\n7. Sanity check: what does embed+lm_head predict?")
    # If we just project the embed + hidden through lm_head, what do we get?
    x = torch.cat([last_hidden, last_token_embed], dim=-1)
    proj_out = F.linear(x, weights_gpu["proj.weight"], weights_gpu.get("proj.bias"))
    lm_out = lm_head(proj_out)
    print(f"  Direct (hidden+embed) → proj → lm_head → top-5:")
    for t in lm_out[0].topk(5).indices:
        print(f"    {t.item()}: '{tokenizer.decode([t.item()])[:20]}'")
    
    # Compare: target logits vs MTP logits distribution similarity
    print("\n8. Distribution comparison (last token):")
    # Compute KL divergence between MTP and target distributions
    mtp_probs = F.softmax(mtp_logits_last.float(), dim=-1)
    target_probs = F.softmax(target_logits_last.float(), dim=-1)
    kl = (target_probs * (target_probs.log() - mtp_probs.log())).sum()
    print(f"  KL(target || MTP): {kl.item():.4f}")
    
    # Correlation of logits
    from scipy.stats import pearsonr
    corr, _ = pearsonr(mtp_logits_last[0].float().cpu().numpy(), target_logits_last.float().cpu().numpy())
    print(f"  Pearson correlation: {corr:.4f}")
    
    # Check weight statistics
    print("\n9. Weight statistics:")
    for k in ["proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
        if k in weights_gpu:
            w = weights_gpu[k]
            print(f"  {k}: shape={tuple(w.shape)}, mean={w.float().mean().item():.6f}, "
                  f"std={w.float().std().item():.6f}")
    
    print("\n" + "=" * 60)
    acc = match_count / max(total, 1)
    if acc > 0.7:
        print("CONCLUSION: MTP head works well on standalone! (>70% exact)")
        print("           → Low accept rate in sglang EAGLE = hidden state passing issue")
    elif acc > 0.3:
        print("CONCLUSION: MTP head moderate accuracy")
    else:
        print("CONCLUSION: MTP head accuracy is critically low!")
        print("           → Training failed or checkpoint corrupted")
        print("           → Need to re-train or investigate training data")

if __name__ == "__main__":
    main()
