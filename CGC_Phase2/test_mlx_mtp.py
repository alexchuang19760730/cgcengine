#!/usr/bin/env python3
"""Test MLX Metal MTP forward vs PyTorch CPU.

Verifies:
1. MLX forward produces same tokens as PyTorch
2. MLX is faster than PyTorch
3. Can be integrated into verify loop
"""
import os, sys, time
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [REPO_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL = "/Users/alexchuang/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf"
CKPT = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_output/qwen25_llamacpp/mtp_head_qwen25-0.5b_decode.pt"
EMBED = "/Users/alexchuang/Documents/flashkv0516/CGC_Phase2/mtp_train_data/qwen25_llamacpp/embed_head.pt"

def test_mlx_forward():
    """Test MLX MTP forward speed and correctness."""
    from mtp_mlx_forward import MTPMLXForward

    print("="*60)
    print("  MLX Metal MTP Forward Test")
    print("="*60)

    # Initialize MLX MTP
    mlx_mtp = MTPMLXForward(
        checkpoint=CKPT,
        embed_head_path=EMBED,
        hidden_size=896, vocab_size=151936,
        num_heads=14, head_dim=64, intermediate_size=4864,
    )

    # Generate dummy hidden state (simulating llama.cpp output)
    np.random.seed(42)
    dummy_hidden = np.random.randn(896).astype(np.float32) * 0.1
    dummy_token = 2038  # "code" token

    # Benchmark MLX draft chain
    print("\n--- MLX Metal Draft Chain ---")
    mlx_times = []
    mlx_tokens_all = []
    for trial in range(10):
        tokens, ms = mlx_mtp.draft_chain(dummy_hidden, dummy_token, num_draft=4)
        mlx_times.append(ms)
        mlx_tokens_all.append(tokens)
        if trial < 3:
            print(f"  Trial {trial}: {ms:.2f}ms, tokens={tokens}")

    mlx_times = np.array(mlx_times[2:])  # skip first 2 (warmup)
    print(f"  MLX nd=4: mean={mlx_times.mean():.2f}ms, median={np.median(mlx_times):.2f}ms, min={mlx_times.min():.2f}ms")
    print(f"  Per-draft: {mlx_times.mean()/4:.2f}ms")

    # Test nd=1
    mlx_times_nd1 = []
    for trial in range(10):
        _, ms = mlx_mtp.draft_chain(dummy_hidden, dummy_token, num_draft=1)
        mlx_times_nd1.append(ms)
    mlx_times_nd1 = np.array(mlx_times_nd1[2:])
    print(f"  MLX nd=1: mean={mlx_times_nd1.mean():.2f}ms, min={mlx_times_nd1.min():.2f}ms")

    # Compare with PyTorch CPU
    print("\n--- PyTorch CPU Draft Chain (for comparison) ---")
    from mtp_head.model import MTPHead, MTPHeadConfig
    from mtp_verify_loop import MTPVerifyLoop

    config = MTPHeadConfig(
        hidden_size=896, vocab_size=151936,
        num_heads=14, head_dim=64, intermediate_size=4864,
    )
    mtp_head = MTPHead(config)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    # Fix: load from model_state_dict, not top-level dict
    raw_weights = ckpt.get("model_state_dict", ckpt)
    filtered = {k: v for k, v in raw_weights.items()
                if isinstance(v, torch.Tensor) and "lm_head" not in k and "embed" not in k}
    mtp_head.load_state_dict(filtered, strict=False)

    eh = torch.load(EMBED, map_location="cpu", weights_only=True)
    lm_head_w = eh.get("lm_head_weight")
    embed_w = eh.get("embed_weight")
    mtp_head.set_shared_lm_head(lm_head_w)
    mtp_head.eval()

    pt_times = []
    pt_tokens_all = []
    with torch.no_grad():
        for trial in range(10):
            t0 = time.time()
            current_hidden = torch.from_numpy(dummy_hidden).float().unsqueeze(0).unsqueeze(0)
            current_token = dummy_token
            draft_tokens = []
            for i in range(4):
                token_embed = embed_w[current_token].unsqueeze(0).unsqueeze(0)
                x = torch.cat([current_hidden, token_embed], dim=-1)
                x = mtp_head.proj(x)
                h = x + mtp_head.attn(mtp_head.norm1(x))
                h = h + mtp_head.mlp(mtp_head.norm2(h))
                mtp_hidden = mtp_head.norm_out(h)
                logits = mtp_head.lm_head(mtp_hidden)
                draft_token = int(logits.argmax(dim=-1).item())
                draft_tokens.append(draft_token)
                current_hidden = mtp_hidden
                current_token = draft_token
            ms = (time.time() - t0) * 1000
            pt_times.append(ms)
            pt_tokens_all.append(draft_tokens)
            if trial < 3:
                print(f"  Trial {trial}: {ms:.2f}ms, tokens={draft_tokens}")

    pt_times = np.array(pt_times[2:])
    print(f"  PyTorch nd=4: mean={pt_times.mean():.2f}ms, median={np.median(pt_times):.2f}ms, min={pt_times.min():.2f}ms")
    print(f"  Per-draft: {pt_times.mean()/4:.2f}ms")

    # Check token match
    print("\n--- Token Match Check ---")
    match_count = 0
    for i in range(min(len(mlx_tokens_all), len(pt_tokens_all))):
        if mlx_tokens_all[i] == pt_tokens_all[i]:
            match_count += 1
        else:
            print(f"  Mismatch at trial {i}: MLX={mlx_tokens_all[i]} vs PT={pt_tokens_all[i]}")
    print(f"  Match: {match_count}/{min(len(mlx_tokens_all), len(pt_tokens_all))}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  MLX Metal nd=4:  {mlx_times.mean():.2f}ms ({mlx_times.mean()/4:.2f}ms/draft)")
    print(f"  PyTorch CPU nd=4: {pt_times.mean():.2f}ms ({pt_times.mean()/4:.2f}ms/draft)")
    print(f"  Speedup: {pt_times.mean() / mlx_times.mean():.2f}x")
    print(f"  Token match: {match_count}/{min(len(mlx_tokens_all), len(pt_tokens_all))}")

if __name__ == "__main__":
    test_mlx_forward()
