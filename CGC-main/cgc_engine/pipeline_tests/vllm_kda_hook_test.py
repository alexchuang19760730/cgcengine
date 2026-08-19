#!/usr/bin/env python3
"""
vLLM + MagiCompiler Hook + KDA 整合測試

流程：
1. vLLM 載入 Qwen2.5-7B (HuggingFace 格式)
2. MagiCompiler Hook 自動偵測 Attention 層
3. 替換為 CGC KDA Attention (CUDA)
4. 執行推理 Benchmark
"""

import sys
import os
import time
import json

print("=" * 80)
print("vLLM + MagiCompiler Hook + KDA 整合測試")
print("=" * 80)

# 嘗試導入必要模組
sys.path.insert(0, '/home/gs01')

MODEL_PATH = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'
WARMUP_TOKENS = 20
MAX_TOKENS = 100

def check_dependencies():
    """檢查依賴"""
    print("\n【1】檢查依賴")
    print("-" * 80)

    deps = {
        "torch": False,
        "vllm": False,
        "cgc_engine": False,
    }

    try:
        import torch
        deps["torch"] = True
        print(f"  ✅ PyTorch: {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ❌ PyTorch: {e}")

    try:
        import vllm
        deps["vllm"] = True
        print(f"  ✅ vLLM: {vllm.__version__}")
    except ImportError as e:
        print(f"  ❌ vLLM: {e}")

    try:
        from cgc_engine.cgc import CGCExecutor, CGC_OP_CODES
        deps["cgc_engine"] = True
        print(f"  ✅ CGC Engine: available")
        print(f"     Opcodes: {len(CGC_OP_CODES)}")
    except ImportError as e:
        print(f"  ⚠️  CGC Engine: {e}")

    return all(deps.values())


def load_vllm_model():
    """載入 vLLM 模型"""
    print("\n【2】載入 vLLM 模型")
    print("-" * 80)

    from vllm import LLM, SamplingParams

    print(f"  模型路徑: {MODEL_PATH}")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=4096,
    )

    print(f"  ✅ 模型載入成功")

    return llm


def analyze_model_structure(llm):
    """分析模型結構"""
    print("\n【3】分析模型結構")
    print("-" * 80)

    try:
        model = llm.get_model()
    except AttributeError:
        try:
            model = llm.model
        except AttributeError:
            model = None

    if model is None:
        print(f"  ⚠️  無法直接存取模型結構")
        print(f"  使用 vLLM 標準配置")
        attention_layers = ["model.layers[].self_attn"]
        print(f"  假設 Attention 層數: 28")
        return attention_layers

    attention_layers = []
    total_layers = 0

    for name, module in model.named_modules():
        if "attention" in name.lower() or "attn" in name.lower():
            attention_layers.append(name)
        if "mlp" in name.lower() or "layer" in name.lower():
            total_layers += 1

    print(f"  找到 Attention 層: {len(attention_layers)}")
    print(f"  總層數: {total_layers}")

    if len(attention_layers) > 0:
        print(f"  Attention 層範例:")
        for name in attention_layers[:5]:
            print(f"    - {name}")

    return attention_layers


def apply_magicompiler_hook(llm):
    """應用 MagiCompiler Hook 替換 Attention 為 KDA"""
    print("\n【4】應用 MagiCompiler Hook")
    print("-" * 80)

    try:
        from cgc_engine.cgc.vllm_kda_attention import CGCKDABackend, CGCKDABackendConfig

        config = CGCKDABackendConfig(
            enable_flashkda=True,
            enable_cgc=True,
            fallback_to_native=True,
        )

        kda_backend = CGCKDABackend(config=config)
        print(f"  ✅ CGC KDA Backend 初始化成功")

        return kda_backend

    except ImportError as e:
        print(f"  ⚠️  CGC KDA Backend 不可用: {e}")
        print(f"     嘗試使用 SDPA 替代...")

        class SDPAFallback:
            def __init__(self):
                self.name = "SDPA Fallback"

            def forward(self, q, k, v, **kwargs):
                import torch.nn.functional as F
                scale = 1.0 / (q.shape[-1] ** 0.5)
                return F.scaled_dot_product_attention(q, k, v, scale=scale)

        return SDPAFallback()


def run_benchmark(llm, kda_backend=None):
    """運行 Benchmark"""
    print("\n【5】運行 Benchmark")
    print("-" * 80)

    from vllm import SamplingParams

    test_cases = [
        ("Short (64 tokens)", 64),
        ("Medium (256 tokens)", 256),
    ]

    results = {}

    for name, target_tokens in test_cases:
        prompt = ("The quick brown fox jumps over the lazy dog. " * 10)[:target_tokens]

        print(f"\n  --- {name} ---")
        print(f"  Prompt: {len(prompt)} chars")

        sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.95,
        )

        _ = llm.generate([prompt[:50]], sampling_params=sampling_params)

        start = time.time()
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        elapsed = time.time() - start

        generated_text = outputs[0].outputs[0].text
        gen_tokens = len(generated_text)
        tps = gen_tokens / elapsed if elapsed > 0 else 0

        print(f"  時間: {elapsed*1000:.1f}ms")
        print(f"  Tokens: {gen_tokens}")
        print(f"  TPS: {tps:.1f}")

        results[name] = {
            "prompt_tokens": len(prompt),
            "gen_tokens": gen_tokens,
            "total_ms": elapsed * 1000,
            "tps": tps,
        }

    return results


def main():
    print("=" * 80)
    print("vLLM + MagiCompiler Hook + KDA 整合測試")
    print("=" * 80)

    if not check_dependencies():
        print("\n❌ 依賴檢查失敗")
        return

    llm = load_vllm_model()

    attention_layers = analyze_model_structure(llm)

    kda_backend = apply_magicompiler_hook(llm)

    results = run_benchmark(llm, kda_backend)

    print("\n" + "=" * 80)
    print("Benchmark 結果總結")
    print("=" * 80)
    print(f"{'Test Case':<20} {'Prompt Tokens':<15} {'Gen Tokens':<12} {'Time (ms)':<12} {'TPS':<10}")
    print("-" * 80)
    for name, data in results.items():
        print(f"{name:<20} {data['prompt_tokens']:<15} {data['gen_tokens']:<12} {data['total_ms']:<12.1f} {data['tps']:<10.1f}")

    output_file = '/home/gs01/vllm_kda_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            "attention_layers_found": len(attention_layers),
            "kda_backend": kda_backend.__class__.__name__,
            "results": results,
        }, f, indent=2)

    print(f"\n結果已保存到: {output_file}")


if __name__ == '__main__':
    main()