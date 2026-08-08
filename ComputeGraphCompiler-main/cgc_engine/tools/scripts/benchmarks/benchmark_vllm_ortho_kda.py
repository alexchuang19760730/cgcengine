#!/usr/bin/env python3
"""vLLM + OrthoKDA v4 Benchmark on NVIDIA GPU"""

import time
import gc
import torch

def main():
    print("="*60)
    print("vLLM + OrthoKDA v4 Benchmark")
    print("="*60)

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    MODEL_PATH = "/home/gs01/models/Qwen/Qwen2___5-7B-Instruct"

    print("\n" + "="*60)
    print("TEST 1: Native vLLM (Baseline)")
    print("="*60)

    from vllm import LLM, SamplingParams

    print(f"Loading native vLLM model: {MODEL_PATH}...")
    t0 = time.time()
    native_llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

    print("\nRunning prefill + decode test...")
    t0 = time.time()
    outputs = native_llm.generate(["Hello, how are you?"], sampling_params)
    native_time = time.time() - t0
    print(f"Total time: {native_time:.3f}s")
    print(f"Generated: {len(outputs[0].outputs[0].token_ids)} tokens")

    native_tps = len(outputs[0].outputs[0].token_ids) / native_time
    print(f"Throughput: {native_tps:.1f} tokens/s")

    del native_llm
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("TEST 2: OrthoKDA v4 Kernel Test")
    print("="*60)

    try:
        from cgc_engine.cgc.ortho_kda_v4_vllm import (
            OrthoKDAV4VLLMIntegration,
            OrthoKDAV4VLLMConfig,
        )

        print("Initializing OrthoKDA v4 integration...")
        kda_integration = OrthoKDAV4VLLMIntegration(
            num_heads=32,
            head_dim=128,
            ortho_base_dim=128,
            decay_rate=0.01,
            enable=True,
        )

        print(f"\n  OrthoKDA v4 Configuration:")
        print(f"    num_heads: {kda_integration.config.num_heads}")
        print(f"    head_dim: {kda_integration.config.head_dim}")
        print(f"    ortho_base_dim: {kda_integration.config.ortho_base_dim}")
        print(f"    decay_rate: {kda_integration.config.decay_rate}")
        print(f"    enable: {kda_integration.config.enable}")

        print(f"\n  Testing OrthoKDA v4 forward pass on CUDA...")

        backend = kda_integration.backend
        if hasattr(backend, 'forward'):
            k_test = torch.randn(32, 128, device="cuda")
            v_test = torch.randn(32, 128, device="cuda")

            t0 = time.time()
            for _ in range(100):
                k_out, v_out = backend.forward(k_test, v_test)
            torch.cuda.synchronize()
            forward_time = (time.time() - t0) / 100

            print(f"    Forward pass time: {forward_time*1000:.3f}ms")
            print(f"    Output shapes: K={k_out.shape}, V={v_out.shape}")
        else:
            print("    Backend forward method not available (C++ backend not compiled)")

        native_kv = 32 * 2048 * 128 * 2 * 2 / 1024 / 1024
        ortho_kv = 32 * 128 * 128 * 2 * 2 / 1024 / 1024

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        print(f"\n{'Metric':<30} {'Native vLLM':<20}")
        print(f"{'-'*30} {'-'*20}")
        print(f"{'Throughput':<30} {native_tps:.1f} tok/s")

        print(f"\nKV Cache Memory:")
        print(f"  Native vLLM: {native_kv:.1f} MB (grows with context)")
        print(f"  OrthoKDA v4: {ortho_kv:.1f} MB (fixed O(1))")
        print(f"  Memory Savings: {(1 - ortho_kv/native_kv)*100:.1f}%")

        print("\nNote: OrthoKDA v4 uses fixed-size orthogonal basis accumulation")
        print("      for O(1) KV cache memory instead of growing with context.")

    except Exception as e:
        print(f"\n❌ OrthoKDA v4 test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60)


if __name__ == '__main__':
    main()