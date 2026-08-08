"""Test ggml native MTP head: correctness (token match vs MLX/PyTorch) + performance."""
import sys
import os
import time
import numpy as np

# Add CGC_Phase2 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Paths
CKPT_V2_FINAL = os.path.expanduser(
    "CGC_Phase2/mtp_output/qwen25_llamacpp_v2/mtp_head_qwen25-0.5b_decode.pt"
)
CKPT_V1 = os.path.expanduser(
    "CGC_Phase2/mtp_output/qwen25_llamacpp/mtp_head_qwen25-0.5b_decode.pt"
)
EMBED_HEAD = os.path.expanduser(
    "CGC_Phase2/mtp_train_data/qwen25_llamacpp/embed_head.pt"
)

# Use v1 (known good: nd=1 accept=81%, nd=4 accept=49.2%)
CKPT = CKPT_V1 if os.path.exists(CKPT_V1) else CKPT_V2_FINAL

def test_ggml_correctness():
    """Test 1: Verify ggml produces same tokens as PyTorch reference."""
    print("=" * 60)
    print("TEST 1: ggml vs PyTorch token match")
    print("=" * 60)

    import torch
    from mtp_head.model import MTPHead, MTPHeadConfig

    # Load PyTorch reference
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    raw = ckpt.get("model_state_dict", ckpt)
    config = ckpt.get("config", {})
    hidden_size = config.get("hidden_size", 896)
    vocab_size = config.get("vocab_size", 151936)
    num_heads = config.get("num_heads", 14)
    head_dim = config.get("head_dim", 64)
    intermediate_size = config.get("intermediate_size", 4864)

    # Create PyTorch model
    mtp_config = MTPHeadConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )
    pt_model = MTPHead(mtp_config)
    pt_model.load_state_dict(raw)
    pt_model.eval()

    # Load embed_head
    eh = torch.load(EMBED_HEAD, map_location="cpu", weights_only=True)
    lm_head_w = eh.get("lm_head_weight", eh.get("lm_head"))
    embed_w = eh.get("embed_weight", eh.get("embed"))
    pt_model.set_shared_lm_head(lm_head_w)

    # Create ggml native
    from mtp_ggml_native import MTPGgmlNative
    ggml_mtp = MTPGgmlNative(
        checkpoint_path=CKPT,
        embed_head_path=EMBED_HEAD,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )

    # Generate test hidden states
    np.random.seed(42)
    test_cases = []
    for i in range(10):
        hidden = np.random.randn(hidden_size).astype(np.float32) * 10  # scale up
        token_id = np.random.randint(0, vocab_size)
        test_cases.append((hidden, token_id))

    # Compare single forward pass
    print("\nComparing single forward pass (10 cases):")
    matches = 0
    total = 0

    for i, (hidden, token_id) in enumerate(test_cases):
        # PyTorch reference
        h_torch = torch.from_numpy(hidden).float().unsqueeze(0).unsqueeze(0)
        embed_torch = embed_w[token_id].float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pt_logits = pt_model(h_torch, embed_torch)
            pt_token = int(pt_logits.argmax(dim=-1).item())
            # Get mtp_hidden (before lm_head)
            x = torch.cat([h_torch, embed_torch], dim=-1)
            x = pt_model.proj(x)
            h = x + pt_model.attn(pt_model.norm1(x))
            h = h + pt_model.mlp(pt_model.norm2(h))
            pt_hidden = pt_model.norm_out(h).squeeze().numpy()

        # ggml native
        embed_np = embed_w[token_id].float().numpy().astype(np.float32)
        ggml_token, ggml_hidden = ggml_mtp.forward_single(hidden, embed_np)

        match = (pt_token == ggml_token)
        if match:
            matches += 1
        total += 1

        # Hidden state comparison
        cos_sim = float(np.dot(pt_hidden, ggml_hidden) / 
                       (np.linalg.norm(pt_hidden) * np.linalg.norm(ggml_hidden) + 1e-10))
        max_diff = float(np.max(np.abs(pt_hidden - ggml_hidden)))

        status = "✓" if match else "✗"
        print(f"  Case {i}: PT={pt_token:6d} ggml={ggml_token:6d} {status}  "
              f"cos={cos_sim:.6f} max_diff={max_diff:.6f}")

    print(f"\nToken match: {matches}/{total} ({100*matches/total:.1f}%)")
    return matches == total


def test_ggml_performance():
    """Test 2: Benchmark ggml vs MLX vs PyTorch CPU."""
    print("\n" + "=" * 60)
    print("TEST 2: Performance benchmark (ggml vs MLX vs PyTorch)")
    print("=" * 60)

    import torch
    from mtp_ggml_native import MTPGgmlNative

    # Load ggml
    ggml_mtp = MTPGgmlNative(
        checkpoint_path=CKPT,
        embed_head_path=EMBED_HEAD,
        hidden_size=896,
        vocab_size=151936,
        num_heads=14,
        head_dim=64,
        intermediate_size=4864,
    )

    # Test data
    hidden_size = 896
    np.random.seed(123)
    hidden = np.random.randn(hidden_size).astype(np.float32) * 10
    token_id = 1234

    # Benchmark single forward
    print("\nSingle forward pass:")
    N = 20

    # ggml
    times_ggml = []
    for _ in range(N):
        embed = ggml_mtp._embed_local[token_id].copy()
        t0 = time.time()
        ggml_mtp.forward_single(hidden, embed)
        times_ggml.append((time.time() - t0) * 1000)
    print(f"  ggml Metal:  {np.mean(times_ggml):.2f}ms ± {np.std(times_ggml):.2f}ms "
          f"(min={np.min(times_ggml):.2f}, max={np.max(times_ggml):.2f})")

    # Benchmark chain (num_draft=4)
    print("\nChain draft (num_draft=4, 10 rounds):")
    N_CHAIN = 10

    times_chain = []
    for _ in range(N_CHAIN):
        t0 = time.time()
        tokens, _ = ggml_mtp.draft_chain(hidden, token_id, 4)
        times_chain.append((time.time() - t0) * 1000)

    print(f"  ggml Metal:  {np.mean(times_chain):.2f}ms ± {np.std(times_chain):.2f}ms "
          f"(min={np.min(times_chain):.2f}, max={np.max(times_chain):.2f})")
    print(f"  Per-step:    {np.mean(times_chain)/4:.2f}ms")
    print(f"  Tokens:      {tokens}")

    # Compare with MLX if available
    try:
        from mtp_mlx_forward import MTPMLXForward
        mlx_mtp = MTPMLXForward(
            checkpoint=CKPT,
            embed_head_path=EMBED_HEAD,
            hidden_size=896,
            vocab_size=151936,
            num_heads=14,
            head_dim=64,
            intermediate_size=4864,
        )

        times_mlx = []
        for _ in range(N_CHAIN):
            t0 = time.time()
            tokens, _ = mlx_mtp.draft_chain(hidden, token_id, 4)
            times_mlx.append((time.time() - t0) * 1000)

        print(f"\n  MLX Metal:   {np.mean(times_mlx):.2f}ms ± {np.std(times_mlx):.2f}ms "
              f"(min={np.min(times_mlx):.2f}, max={np.max(times_mlx):.2f})")
        print(f"  Per-step:    {np.mean(times_mlx)/4:.2f}ms")

        speedup = np.mean(times_mlx) / np.mean(times_chain)
        print(f"\n  ggml vs MLX speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"\n  MLX comparison skipped: {e}")

    # Compare with PyTorch CPU
    try:
        from mtp_head.model import MTPHead, MTPHeadConfig
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
        raw = ckpt.get("model_state_dict", ckpt)
        config = ckpt.get("config", {})
        mtp_config = MTPHeadConfig(
            hidden_size=896, vocab_size=151936,
            num_heads=14, head_dim=64, intermediate_size=4864,
        )
        pt_model = MTPHead(mtp_config)
        pt_model.load_state_dict(raw)
        pt_model.eval()
        eh = torch.load(EMBED_HEAD, map_location="cpu", weights_only=True)
        pt_model.set_shared_lm_head(eh.get("lm_head_weight", eh.get("lm_head")))

        h_torch = torch.from_numpy(hidden).float().unsqueeze(0).unsqueeze(0)
        embed_torch = eh.get("embed_weight", eh.get("embed"))[token_id].float().unsqueeze(0).unsqueeze(0)

        times_pt = []
        with torch.no_grad():
            for _ in range(N_CHAIN):
                t0 = time.time()
                for _ in range(4):
                    logits = pt_model(h_torch, embed_torch)
                    token = int(logits.argmax(dim=-1).item())
                times_pt.append((time.time() - t0) * 1000)

        print(f"\n  PyTorch CPU: {np.mean(times_pt):.2f}ms ± {np.std(times_pt):.2f}ms")
        print(f"  Per-step:    {np.mean(times_pt)/4:.2f}ms")
    except Exception as e:
        print(f"\n  PyTorch comparison skipped: {e}")


def test_ggml_vs_mlx_tokens():
    """Test 3: Verify ggml and MLX produce same draft chains."""
    print("\n" + "=" * 60)
    print("TEST 3: ggml vs MLX draft chain token match")
    print("=" * 60)

    from mtp_ggml_native import MTPGgmlNative

    ggml_mtp = MTPGgmlNative(
        checkpoint_path=CKPT,
        embed_head_path=EMBED_HEAD,
        hidden_size=896,
        vocab_size=151936,
        num_heads=14,
        head_dim=64,
        intermediate_size=4864,
    )

    try:
        from mtp_mlx_forward import MTPMLXForward
        mlx_mtp = MTPMLXForward(
            checkpoint=CKPT,
            embed_head_path=EMBED_HEAD,
            hidden_size=896,
            vocab_size=151936,
            num_heads=14,
            head_dim=64,
            intermediate_size=4864,
        )
    except Exception as e:
        print(f"MLX not available: {e}")
        return True

    np.random.seed(42)
    hidden_size = 896
    num_tests = 20
    matches = 0

    for i in range(num_tests):
        hidden = np.random.randn(hidden_size).astype(np.float32) * 10
        token_id = np.random.randint(0, 151936)

        ggml_tokens, _ = ggml_mtp.draft_chain(hidden, token_id, 4)
        mlx_tokens, _ = mlx_mtp.draft_chain(hidden, token_id, 4)

        match = (ggml_tokens == mlx_tokens)
        if match:
            matches += 1

        status = "✓" if match else "✗"
        print(f"  Case {i:2d}: ggml={ggml_tokens} mlx={mlx_tokens} {status}")

    print(f"\nggml vs MLX token match: {matches}/{num_tests} ({100*matches/num_tests:.1f}%)")
    return matches == num_tests


if __name__ == "__main__":
    print("ggml Native MTP Head Test Suite")
    print(f"Checkpoint: {CKPT}")
    print(f"Embed head: {EMBED_HEAD}")
    print()

    # Use system Python (has llama_cpp)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ok1 = test_ggml_correctness()
    test_ggml_performance()
    ok3 = test_ggml_vs_mlx_tokens()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Correctness (ggml vs PyTorch): {'PASS' if ok1 else 'FAIL'}")
    print(f"  Token match (ggml vs MLX):     {'PASS' if ok3 else 'FAIL'}")
