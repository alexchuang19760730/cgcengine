"""Local MTP overhead benchmark on Mac M4.

Measures:
1. MTP head forward pass time (Qwen3-VL config, 48M params, on MPS)
2. MTP head forward pass time (DSV4 config, larger, with real checkpoint)
3. Base model decode speed (via mlx_lm, if model available)
4. Full local MTP verify loop (if base model available)

Key question: Is MTP head forward pass < 3ms on M4?
If yes, local MTP is viable (crossover ~333-1000 tok/s, well above base model speed).
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project paths
REPO = "/Users/alexchuang/Documents/flashkv0516"
sys.path.insert(0, os.path.join(REPO, "CGC_Phase2", "mtp_head"))

from model import (
    MTPHead, MTPHeadConfig,
    create_mtp_head_for_qwen3vl_2b,
    create_mtp_head_for_dsv4_flash,
)

device_mps = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device_cpu = torch.device("cpu")

print(f"Device: {device_mps}")
print(f"PyTorch: {torch.__version__}")
print()


# ============================================================================
# Part 1: MTP Head Forward Pass Timing
# ============================================================================

def bench_mtp_head(name, mtp_head, hidden_size, vocab_size, device, n_warmup=10, n_iter=100):
    """Benchmark MTP head forward pass.
    
    Simulates decode-mode: seq_len=1, batch=1.
    """
    mtp_head = mtp_head.to(device)
    mtp_head.eval()
    
    # Set fake shared lm_head (must match model dtype)
    model_dtype = next(mtp_head.parameters()).dtype
    lm_head_w = torch.randn(vocab_size, hidden_size, dtype=model_dtype)
    mtp_head.set_shared_lm_head(lm_head_w)
    mtp_head = mtp_head.to(device)  # move everything to device
    mtp_head.eval()
    
    # Create synthetic inputs (decode mode: seq=1, batch=1)
    hidden = torch.randn(1, 1, hidden_size, device=device, dtype=model_dtype)
    embed = torch.randn(1, 1, hidden_size, device=device, dtype=model_dtype)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = mtp_head(hidden, embed)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            t0 = time.perf_counter()
            logits = mtp_head(hidden, embed)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
    
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    p10 = times_sorted[len(times_sorted) // 10]
    p90 = times_sorted[len(times_sorted) * 9 // 10]
    mean = sum(times) / len(times)
    
    params = mtp_head.num_parameters() / 1e6
    
    print(f"=== {name} MTP Head ({params:.1f}M params) on {device} ===")
    print(f"  hidden_size={hidden_size}, vocab={vocab_size}")
    print(f"  Forward pass (seq=1, batch=1):")
    print(f"    median: {median:.3f}ms")
    print(f"    mean:   {mean:.3f}ms")
    print(f"    p10:    {p10:.3f}ms")
    print(f"    p90:    {p90:.3f}ms")
    print(f"    logits shape: {tuple(logits.shape)}")
    print()
    
    return median


def bench_mtp_head_multi_token(name, mtp_head, hidden_size, vocab_size, device, n_draft=4, n_warmup=10, n_iter=100):
    """Benchmark MTP head for multi-token draft (chained, seq=1 each step).
    
    In EAGLE/chain mode, each draft token requires one MTP head forward pass.
    So n_draft tokens = n_draft forward passes.
    """
    mtp_head = mtp_head.to(device)
    mtp_head.eval()
    
    model_dtype = next(mtp_head.parameters()).dtype
    hidden = torch.randn(1, 1, hidden_size, device=device, dtype=model_dtype)
    embed = torch.randn(1, 1, hidden_size, device=device, dtype=model_dtype)
    lm_head_w = torch.randn(vocab_size, hidden_size, dtype=model_dtype)
    mtp_head.set_shared_lm_head(lm_head_w)
    mtp_head = mtp_head.to(device)
    mtp_head.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            for _ in range(n_draft):
                _ = mtp_head(hidden, embed)
    if device.type == "mps":
        torch.mps.synchronize()
    
    # Benchmark: time n_draft chained forward passes
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            t0 = time.perf_counter()
            for _ in range(n_draft):
                logits = mtp_head(hidden, embed)
                # In real chain: would update hidden/embed with new token
                # For timing, just measure raw forward passes
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    per_token = median / n_draft
    
    print(f"=== {name} MTP Head — {n_draft} draft tokens (chained) on {device} ===")
    print(f"  Total {n_draft} forward passes: median {median:.3f}ms")
    print(f"  Per draft token: {per_token:.3f}ms")
    print()
    
    return per_token


# ============================================================================
# Part 2: Base Model Decode Speed
# ============================================================================

def bench_base_model_mlx(model_name="Qwen/Qwen3-1.7B", n_tokens=50):
    """Benchmark base model decode speed via mlx_lm.
    
    Returns: (tok/s, model_info) or (None, None) if failed.
    """
    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        
        print(f"=== Base Model: {model_name} (mlx_lm) ===")
        print(f"  Loading...")
        t0 = time.perf_counter()
        model, tokenizer = load(model_name)
        t_load = time.perf_counter() - t0
        print(f"  Loaded in {t_load:.1f}s")
        
        # Get model info
        config = model.config if hasattr(model, 'config') else {}
        print(f"  Config: {config}")
        
        # Generate tokens and measure speed
        prompt = "def fibonacci(n):"
        
        # Warmup
        response = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
        
        # Benchmark
        t0 = time.perf_counter()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=n_tokens, verbose=False)
        t1 = time.perf_counter()
        
        elapsed = t1 - t0
        tps = n_tokens / elapsed
        
        print(f"  Generated {n_tokens} tokens in {elapsed:.2f}s")
        print(f"  Decode speed: {tps:.1f} tok/s")
        print(f"  Per-token: {1000/tps:.2f}ms")
        print(f"  Output: {response[:100]}...")
        print()
        
        return tps, model
        
    except Exception as e:
        print(f"  Failed: {e}")
        print()
        return None, None


def bench_base_model_transformers(model_name="Qwen/Qwen3-1.7B", n_tokens=50):
    """Benchmark base model decode speed via transformers with MPS.
    
    Also tests hidden state extraction.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"=== Base Model: {model_name} (transformers + MPS) ===")
        print(f"  Loading...")
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="mps",
            trust_remote_code=True,
        )
        t_load = time.perf_counter() - t0
        print(f"  Loaded in {t_load:.1f}s")
        
        # Get model config
        config = model.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        print(f"  hidden_size={hidden_size}, vocab_size={vocab_size}")
        
        # Test hidden state extraction
        input_ids = tokenizer("def fibonacci(n):", return_tensors="pt").to("mps")
        
        with torch.no_grad():
            outputs = model(**input_ids, output_hidden_states=True)
        
        last_hidden = outputs.hidden_states[-1]  # [1, seq, hidden]
        print(f"  Hidden states: {tuple(last_hidden.shape)}, dtype={last_hidden.dtype}")
        print(f"  Hidden state extraction: OK")
        
        # Benchmark decode speed
        # Generate one token at a time
        generated = input_ids["input_ids"]
        torch.mps.synchronize()
        
        times = []
        with torch.no_grad():
            for i in range(n_tokens):
                t0 = time.perf_counter()
                outputs = model(generated, output_hidden_states=True)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.mps.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
                generated = torch.cat([generated, next_token], dim=-1)
        
        times_sorted = sorted(times)
        median = times_sorted[len(times_sorted) // 2]
        mean = sum(times) / len(times)
        tps = 1000 / mean
        
        print(f"  Decode speed: {tps:.1f} tok/s (mean {mean:.2f}ms/tok, median {median:.2f}ms/tok)")
        print()
        
        return tps, model, hidden_size, vocab_size, last_hidden
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None, None, None, None, None


# ============================================================================
# Part 3: Full Local MTP Verify Loop
# ============================================================================

def bench_local_mtp_loop(model, tokenizer, mtp_head, hidden_size, vocab_size, n_steps=30):
    """Full local MTP verify loop:
    
    1. Base model forward → hidden states + logits → argmax → token
    2. MTP head forward (hidden, embed) → draft logits → draft tokens (n_draft)
    3. Base model forward (draft tokens) → verify logits
    4. Accept/reject by comparing argmax
    
    Measures: effective tok/s, accept rate, overhead breakdown.
    """
    print(f"=== Local MTP Verify Loop ===")
    print(f"  Base model hidden={hidden_size}, vocab={vocab_size}")
    print(f"  MTP head params={mtp_head.num_parameters()/1e6:.1f}M")
    
    # Check compatibility
    mtp_hidden = mtp_head.config.hidden_size
    mtp_vocab = mtp_head.config.vocab_size
    if mtp_hidden != hidden_size:
        print(f"  WARNING: hidden mismatch! base={hidden_size}, mtp={mtp_hidden}")
        print(f"  Cannot run full loop — need matching models.")
        return None
    if mtp_vocab != vocab_size:
        print(f"  WARNING: vocab mismatch! base={vocab_size}, mtp={mtp_vocab}")
        return None
    
    # Set shared lm_head from base model
    try:
        base_lm_head = model.get_output_embeddings()
        if base_lm_head is not None:
            mtp_head.set_shared_lm_head(base_lm_head.weight.detach().cpu())
            mtp_head = mtp_head.to("mps")
            mtp_head.eval()
            print(f"  Shared lm_head: loaded from base model")
        else:
            print(f"  WARNING: Cannot get base model lm_head, using random")
            lm_head_w = torch.randn(vocab_size, hidden_size)
            mtp_head.set_shared_lm_head(lm_head_w)
            mtp_head = mtp_head.to("mps")
            mtp_head.eval()
    except Exception as e:
        print(f"  WARNING: lm_head load failed: {e}")
        return None
    
    # Get embedding from base model
    try:
        embed_layer = model.get_input_embeddings()
        print(f"  Embedding layer: {type(embed_layer).__name__}")
    except:
        print(f"  WARNING: Cannot get embedding layer")
        return None
    
    input_ids = tokenizer("def fibonacci(n):", return_tensors="pt")["input_ids"].to("mps")
    
    n_draft = 4  # draft 4 tokens per step
    total_accepted = 0
    total_draft = 0
    total_steps = 0
    
    base_times = []
    mtp_times = []
    verify_times = []
    total_times = []
    
    generated = input_ids
    
    with torch.no_grad():
        for step in range(n_steps):
            step_t0 = time.perf_counter()
            
            # 1. Base model forward (get hidden states + logits for last token)
            base_t0 = time.perf_counter()
            outputs = model(generated, output_hidden_states=True)
            torch.mps.synchronize()
            base_t1 = time.perf_counter()
            base_times.append((base_t1 - base_t0) * 1000)
            
            last_hidden = outputs.hidden_states[-1][:, -1:, :]  # [1, 1, hidden]
            base_logits = outputs.logits[:, -1, :]  # [1, vocab]
            base_token = base_logits.argmax(dim=-1)  # [1]
            
            # 2. MTP head: generate n_draft draft tokens (chained)
            draft_tokens = []
            cur_hidden = last_hidden.clone()
            cur_token = base_token.clone()
            
            mtp_t0 = time.perf_counter()
            for d in range(n_draft):
                # Get embedding for current token
                token_embed = embed_layer(cur_token).unsqueeze(0)  # [1, 1, hidden]
                
                # MTP head forward
                draft_logits = mtp_head(cur_hidden.to("mps"), token_embed.to("mps"))
                draft_token = draft_logits[:, -1, :].argmax(dim=-1)  # [1]
                draft_tokens.append(draft_token)
                
                # Update for next chain step (in real impl, would use new hidden)
                # For timing: just measure forward passes
                cur_token = draft_token
            
            torch.mps.synchronize()
            mtp_t1 = time.perf_counter()
            mtp_times.append((mtp_t1 - mtp_t0) * 1000)
            
            # 3. Verify: run base model with draft tokens
            # Append all draft tokens to sequence and verify
            draft_seq = torch.cat([generated, base_token.unsqueeze(0), 
                                   torch.stack(draft_tokens, dim=1).squeeze(0).unsqueeze(0)], dim=1)
            
            verify_t0 = time.perf_counter()
            verify_outputs = model(draft_seq, output_hidden_states=True)
            torch.mps.synchronize()
            verify_t1 = time.perf_counter()
            verify_times.append((verify_t1 - verify_t0) * 1000)
            
            # 4. Accept/reject
            # Compare draft tokens with base model's actual predictions
            verify_logits = verify_outputs.logits  # [1, seq_len, vocab]
            
            # The base model's prediction at position of base_token should match draft_tokens[0]
            # Position of base_token is len(generated), prediction is at that position
            base_pred_pos = generated.shape[1]  # position of the token after base_token
            accepted = 0
            for d in range(n_draft):
                verify_pred = verify_logits[:, base_pred_pos + d - 1, :].argmax(dim=-1)
                if d < len(draft_tokens) and verify_pred.item() == draft_tokens[d].item():
                    accepted += 1
                else:
                    break  # reject rest
            
            total_accepted += accepted
            total_draft += n_draft
            total_steps += 1
            
            step_t1 = time.perf_counter()
            total_times.append((step_t1 - step_t0) * 1000)
            
            # Append accepted tokens + base token to generated
            new_tokens = [base_token] + draft_tokens[:accepted]
            generated = torch.cat([generated, torch.stack(new_tokens, dim=1).squeeze(0).unsqueeze(0)], dim=1)
    
    # Calculate results
    accept_rate = total_accepted / total_draft if total_draft > 0 else 0
    mean_base = sum(base_times) / len(base_times)
    mean_mtp = sum(mtp_times) / len(mtp_times)
    mean_verify = sum(verify_times) / len(verify_times)
    mean_total = sum(total_times) / len(total_times)
    
    # Without MTP: 1 token per base forward = 1000/mean_base tok/s
    # With MTP: (1 + accepted) tokens per (base + mtp + verify) step
    tokens_per_step = 1 + (total_accepted / total_steps)
    tps_with_mtp = tokens_per_step / (mean_total / 1000)
    tps_without_mtp = 1000 / mean_base
    
    print(f"\n  Results ({total_steps} steps, {n_draft} draft/step):")
    print(f"  Accept rate: {accept_rate:.1%} ({total_accepted}/{total_draft})")
    print(f"  Tokens per step: {tokens_per_step:.2f}")
    print(f"  Timing breakdown (mean):")
    print(f"    Base forward:  {mean_base:.2f}ms")
    print(f"    MTP head ({n_draft} drafts): {mean_mtp:.2f}ms ({mean_mtp/n_draft:.2f}ms/draft)")
    print(f"    Verify forward: {mean_verify:.2f}ms")
    print(f"    Total/step:    {mean_total:.2f}ms")
    print(f"  Speed:")
    print(f"    Without MTP: {tps_without_mtp:.1f} tok/s ({mean_base:.2f}ms/tok)")
    print(f"    With MTP:    {tps_with_mtp:.1f} tok/s ({mean_total/tokens_per_step:.2f}ms/tok)")
    print(f"    Speedup:     {tps_with_mtp/tps_without_mtp:.2f}x")
    
    crossover = 1000 / (mean_mtp / n_draft) if mean_mtp > 0 else 9999
    print(f"  MTP crossover: ~{crossover:.0f} tok/s (base below this → MTP beneficial)")
    print()
    
    return {
        "accept_rate": accept_rate,
        "tokens_per_step": tokens_per_step,
        "tps_with_mtp": tps_with_mtp,
        "tps_without_mtp": tps_without_mtp,
        "speedup": tps_with_mtp / tps_without_mtp,
        "mean_base_ms": mean_base,
        "mean_mtp_ms": mean_mtp,
        "mean_verify_ms": mean_verify,
        "mean_total_ms": mean_total,
        "crossover_tps": crossover,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = {}
    
    # --- Part 1a: Qwen3-VL MTP Head (48M params, small) ---
    print("=" * 70)
    print("PART 1: MTP Head Forward Pass Timing")
    print("=" * 70)
    print()
    
    mtp_qwen = create_mtp_head_for_qwen3vl_2b()
    t_qwen_mps = bench_mtp_head("Qwen3-VL-2B", mtp_qwen, 2048, 151936, device_mps)
    t_qwen_draft = bench_mtp_head_multi_token("Qwen3-VL-2B", mtp_qwen, 2048, 151936, device_mps, n_draft=4)
    results["qwen3vl_mtp_mps"] = t_qwen_mps
    results["qwen3vl_mtp_per_draft"] = t_qwen_draft
    
    # --- Part 1b: DSV4 MTP Head (larger) ---
    mtp_dsv4 = create_mtp_head_for_dsv4_flash()
    t_dsv4_mps = bench_mtp_head("DSV4-Flash", mtp_dsv4, 4096, 129280, device_mps)
    t_dsv4_draft = bench_mtp_head_multi_token("DSV4-Flash", mtp_dsv4, 4096, 129280, device_mps, n_draft=4)
    results["dsv4_mtp_mps"] = t_dsv4_mps
    results["dsv4_mtp_per_draft"] = t_dsv4_draft
    
    # --- Part 1c: DSV4 with real checkpoint ---
    ckpt_path = os.path.join(REPO, "CGC_Phase2/mtp_head/checkpoints/mtp_head_dsv4_decode_v1.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading real DSV4 checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        mtp_dsv4_real = create_mtp_head_for_dsv4_flash()
        # Load only non-lm_head weights
        filtered = {k: v for k, v in state_dict.items() if "lm_head" not in k}
        try:
            mtp_dsv4_real.load_state_dict(filtered, strict=False)
            print(f"  Loaded {len(filtered)} keys from checkpoint")
            t_dsv4_real = bench_mtp_head("DSV4-Flash (real ckpt)", mtp_dsv4_real, 4096, 129280, device_mps)
            results["dsv4_mtp_real_mps"] = t_dsv4_real
        except Exception as e:
            print(f"  Checkpoint load failed: {e}")
        
        # Check if there's chain accuracy info
        if isinstance(checkpoint, dict):
            for key in ["chain_acc", "train_acc", "loss", "epoch"]:
                if key in checkpoint:
                    print(f"  checkpoint[{key}] = {checkpoint[key]}")
    print()
    
    # --- Part 1d: BF16 precision (production realistic) ---
    print("--- BF16 precision (production) ---")
    mtp_qwen_bf16 = create_mtp_head_for_qwen3vl_2b()
    mtp_qwen_bf16 = mtp_qwen_bf16.to(torch.bfloat16)
    t_qwen_bf16 = bench_mtp_head("Qwen3-VL-2B (BF16)", mtp_qwen_bf16, 2048, 151936, device_mps)
    results["qwen3vl_mtp_bf16_mps"] = t_qwen_bf16
    print()
    
    # --- Part 2: Base Model ---
    print("=" * 70)
    print("PART 2: Base Model Decode Speed")
    print("=" * 70)
    print()
    
    # Try transformers first (needed for hidden states)
    tps_base, model, hidden_size, vocab_size, last_hidden = bench_base_model_transformers(
        "Qwen/Qwen3-1.7B", n_tokens=30
    )
    
    if tps_base is None:
        # Fallback to mlx_lm
        tps_base, _ = bench_base_model_mlx("Qwen/Qwen3-1.7B", n_tokens=50)
    
    if tps_base:
        results["base_tps"] = tps_base
        results["base_ms_per_tok"] = 1000 / tps_base
    
    # --- Part 3: Full Local MTP Loop ---
    print("=" * 70)
    print("PART 3: Full Local MTP Verify Loop")
    print("=" * 70)
    print()
    
    if model is not None and hidden_size is not None:
        # Check if model is compatible with our MTP head
        # Qwen3-1.7B: hidden_size might differ from Qwen3-VL-2B (2048)
        print(f"Base model hidden_size={hidden_size}, MTP head expects=2048")
        
        if hidden_size == 2048:
            loop_result = bench_local_mtp_loop(model, None, mtp_qwen, hidden_size, vocab_size, n_steps=20)
            if loop_result:
                results["mtp_loop"] = loop_result
        else:
            print(f"  Hidden size mismatch ({hidden_size} vs 2048). Creating custom MTP head...")
            # Create MTP head matching base model
            custom_config = MTPHeadConfig(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_heads=16,
                head_dim=hidden_size // 16 if hidden_size >= 1024 else 64,
                intermediate_size=int(hidden_size * 2.75),
            )
            custom_mtp = MTPHead(custom_config)
            print(f"  Custom MTP head: {custom_mtp.num_parameters()/1e6:.1f}M params")
            t_custom = bench_mtp_head("Custom (base-matched)", custom_mtp, hidden_size, vocab_size, device_mps)
            results["custom_mtp_mps"] = t_custom
            
            # Run full loop with untrained MTP head (accept rate will be low, but timing is valid)
            print("\n  NOTE: Using untrained MTP head — accept rate will be near-random.")
            print("  Timing is still valid for overhead measurement.\n")
            loop_result = bench_local_mtp_loop(model, None, custom_mtp, hidden_size, vocab_size, n_steps=15)
            if loop_result:
                results["mtp_loop_untrained"] = loop_result
    else:
        print("  Skipped: base model not available")
    
    # --- Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"MTP Head Forward Pass (Qwen3-VL config, 48M params):")
    print(f"  MPS FP32:  {results.get('qwen3vl_mtp_mps', 'N/A'):.3f}ms")
    print(f"  MPS BF16:  {results.get('qwen3vl_mtp_bf16_mps', 'N/A'):.3f}ms")
    print(f"  Per draft token (4 chained): {results.get('qwen3vl_mtp_per_draft', 'N/A'):.3f}ms")
    print()
    print(f"MTP Head Forward Pass (DSV4 config, larger):")
    print(f"  MPS FP32:  {results.get('dsv4_mtp_mps', 'N/A'):.3f}ms")
    if "dsv4_mtp_real_mps" in results:
        print(f"  MPS FP32 (real ckpt): {results['dsv4_mtp_real_mps']:.3f}ms")
    print()
    if "base_tps" in results:
        print(f"Base Model (Qwen3-1.7B via transformers+MPS):")
        print(f"  Speed: {results['base_tps']:.1f} tok/s ({results['base_ms_per_tok']:.2f}ms/tok)")
    print()
    if "mtp_loop" in results:
        r = results["mtp_loop"]
        print(f"Full MTP Loop (trained head):")
        print(f"  Accept rate: {r['accept_rate']:.1%}")
        print(f"  Speedup: {r['speedup']:.2f}x ({r['tps_without_mtp']:.1f} → {r['tps_with_mtp']:.1f} tok/s)")
        print(f"  Crossover: ~{r['crossover_tps']:.0f} tok/s")
    if "mtp_loop_untrained" in results:
        r = results["mtp_loop_untrained"]
        print(f"Full MTP Loop (untrained, timing only):")
        print(f"  Accept rate: {r['accept_rate']:.1%} (expected low)")
        print(f"  Overhead: base={r['mean_base_ms']:.1f}ms + mtp={r['mean_mtp_ms']:.1f}ms + verify={r['mean_verify_ms']:.1f}ms = {r['mean_total_ms']:.1f}ms")
        print(f"  Crossover: ~{r['crossover_tps']:.0f} tok/s (base below this → MTP beneficial)")
        print(f"  Verdict: {'VIABLE' if r['crossover_tps'] > r['tps_without_mtp'] else 'NOT VIABLE'}")
    
    print()
    print("--- Key verdict ---")
    qwen_mtp_ms = results.get('qwen3vl_mtp_mps', 999)
    if qwen_mtp_ms < 3.0:
        print(f"  MTP head forward = {qwen_mtp_ms:.2f}ms < 3ms target → LOCAL MTP VIABLE")
        print(f"  Crossover ~{1000/qwen_mtp_ms:.0f} tok/s, base model well below this")
    else:
        print(f"  MTP head forward = {qwen_mtp_ms:.2f}ms >= 3ms target → NEED OPTIMIZATION")
    
    # Save results
    out_path = os.path.join(REPO, "CGC_Phase2/bench_local_mtp_results.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = v
        elif isinstance(v, (int, float)):
            serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")
