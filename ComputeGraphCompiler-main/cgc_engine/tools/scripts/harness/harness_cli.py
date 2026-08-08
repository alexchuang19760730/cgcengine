#!/usr/bin/env python3
"""
Harness Agent CLI - 策略生成接口

功能：
- 從基準測試結果分析生成優化策略
- 支持 llama.cpp 和 vLLM 參考
- 自動注入策略到 SIMD Engine

使用方法：
    python -m cgc_engine.agent.harness_cli \
        --benchmark-results results.json \
        --model-type llama \
        --device cuda \
        --output strategy.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch

# 添加項目路徑
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from cgc_engine.agent import (
    HarnessAgent,
    HarnessCompileStrategy,
    GraphAnalyzer,
    GraphFeatures,
    OptimizationSpaceBuilder,
    OptimizationSpace,
)
from cgc_engine.model_parsers import ParsedModel


def load_benchmark_results(path: str) -> Dict[str, Any]:
    """加載基準測試結果"""
    with open(path, 'r') as f:
        return json.load(f)


def analyze_benchmark_for_features(benchmark: Dict[str, Any]) -> GraphFeatures:
    """
    從基準測試結果推斷 GraphFeatures

    這是一個 heuristic 分析，基於 opcode 使用情況推斷模型特徵
    """
    features = GraphFeatures()
    opcode_counts = benchmark.get("opcode_counts", {})

    # 檢測 Attention
    attention_opcodes = ["0x01", "0x02", "0x03", "0x04"]
    if any(op in opcode_counts for op in attention_opcodes):
        features.has_attention = True
        features.attention_patterns.append("inference_attention")

    # 檢測 Flash Attention
    if "0x04" in opcode_counts:
        features.has_flash_attention = True

    # 檢測 MoE (如果有 MoE 相關的 opcode)
    moe_opcodes = ["0xE0", "0xE1", "0xE2", "0xCA", "0xCB"]
    if any(op in opcode_counts for op in moe_opcodes):
        features.has_moe = True
        features.moe_patterns.append("inference_moe")

    # 檢測 Tensor Parallel
    tp_opcodes = ["0x60", "0x61", "0x62"]  # all_reduce, all_gather, etc.
    if any(op in opcode_counts for op in tp_opcodes):
        features.has_tensor_parallel = True

    # 檢測 Vision/VLM
    vision_opcodes = ["0x70", "0x71", "0x72"]
    if any(op in opcode_counts for op in vision_opcodes):
        features.has_vlm = True

    # 推斷 hidden_dim 和 num_layers
    features.hidden_dim = benchmark.get("hidden_dim", 4096)
    features.num_layers = benchmark.get("num_layers", 32)
    features.num_heads = benchmark.get("num_heads", 32)

    # 檢測 GEMM 規模
    gemm_opcodes = ["0x10", "0x11", "0x12"]
    total_gemm = sum(opcode_counts.get(op, 0) for op in gemm_opcodes)
    if total_gemm > 1000:
        features.large_gemm = True
    else:
        features.small_gemm = True

    return features


def generate_strategy_from_benchmark(
    benchmark: Dict[str, Any],
    model_type: str,
    device: str,
    enable_llama_cpp_ref: bool = True,
    enable_vllm_ref: bool = True,
) -> HarnessCompileStrategy:
    """
    從基準測試結果生成優化策略

    Args:
        benchmark: 基準測試結果
        model_type: 模型類型 (llama, mistral, qwen, etc.)
        device: 設備 (cuda, cpu, metal)
        enable_llama_cpp_ref: 是否參考 llama.cpp
        enable_vllm_ref: 是否參考 vLLM

    Returns:
        HarnessCompileStrategy
    """
    # 創建 Harness Agent
    agent = HarnessAgent(
        device=device,
        enable_llama_cpp_reference=enable_llama_cpp_ref,
        enable_vllm_reference=enable_vllm_ref,
    )

    # 從基準測試結果推斷特徵
    features = analyze_benchmark_for_features(benchmark)

    # 從基準測試結果推斷輸入形狀
    num_tokens = benchmark.get("num_tokens", 100)
    hidden_dim = benchmark.get("hidden_dim", 4096)
    input_shape = (1, num_tokens, hidden_dim)

    # 構建優化空間
    space = OptimizationSpaceBuilder.build(
        model=None,  # 我們從 benchmark 推斷
        input_shape=input_shape,
        device=device,
    )
    space.model_type = model_type

    # Agent 決策
    strategy = agent.decide(
        model=None,
        input_shape=input_shape,
        graph_features=features,
        optimization_space=space,
        user_hints=None,
    )

    # 從 benchmark 添加額外信息
    if benchmark.get("fusion_opportunities"):
        strategy.fusion_regions.extend(benchmark["fusion_opportunities"])

    return strategy


def inject_strategy_to_simd_engine(strategy: HarnessCompileStrategy) -> bool:
    """
    將策略注入到 SIMD Engine

    Returns:
        是否成功
    """
    try:
        from cgc_engine.cgc.cgc_strategy_injection import inject_strategy

        success = inject_strategy(strategy)
        if success:
            print("[HarnessCLI] Strategy injected successfully")
        else:
            print("[HarnessCLI] Failed to inject strategy")
        return success
    except ImportError:
        print("[HarnessCLI] Strategy injection module not available")
        return False
    except Exception as e:
        print(f"[HarnessCLI] Strategy injection error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Harness Agent CLI - Strategy Generation"
    )
    parser.add_argument(
        "--benchmark-results",
        type=str,
        required=True,
        help="Path to benchmark JSON results"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="llama",
        choices=["llama", "mistral", "qwen", "phi", "moe", "vlm"],
        help="Model type"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "metal", "auto"],
        help="Target device"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="strategy.json",
        help="Output strategy JSON file"
    )
    parser.add_argument(
        "--inject",
        action="store_true",
        help="Inject strategy to SIMD Engine after generation"
    )
    parser.add_argument(
        "--no-llama-cpp-ref",
        action="store_true",
        help="Disable llama.cpp reference"
    )
    parser.add_argument(
        "--no-vllm-ref",
        action="store_true",
        help="Disable vLLM reference"
    )

    args = parser.parse_args()

    # 檢查文件
    if not Path(args.benchmark_results).exists():
        print(f"[HarnessCLI] Error: Benchmark results not found: {args.benchmark_results}")
        sys.exit(1)

    # 加載基準測試結果
    print(f"[HarnessCLI] Loading benchmark results from {args.benchmark_results}")
    benchmark = load_benchmark_results(args.benchmark_results)

    # 生成策略
    print(f"[HarnessCLI] Generating strategy for {args.model_type} on {args.device}")
    strategy = generate_strategy_from_benchmark(
        benchmark=benchmark,
        model_type=args.model_type,
        device=args.device,
        enable_llama_cpp_ref=not args.no_llama_cpp_ref,
        enable_vllm_ref=not args.no_vllm_ref,
    )

    # 保存策略
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(strategy.to_dict(), f, indent=2)
    print(f"[HarnessCLI] Strategy saved to {args.output}")

    # 打印策略摘要
    print("\n" + "=" * 60)
    print("📋 Strategy Summary")
    print("=" * 60)
    print(f"Backend: {strategy.backend}")
    print(f"Op Fusion: {strategy.enable_op_fusion}")
    print(f"Tile sizes: {strategy.tile_sizes}")
    print(f"TP degree: {strategy.tp_degree}")
    print(f"PP degree: {strategy.pp_degree}")
    print(f"Quantization: {strategy.quantization_mode}")
    print(f"Op hints: {[h.value for h in strategy.op_hints]}")
    print(f"Fusion regions: {len(strategy.fusion_regions)}")
    for i, region in enumerate(strategy.fusion_regions):
        print(f"  {i+1}. {region}")
    print("=" * 60 + "\n")

    # 可選：注入策略
    if args.inject:
        print("[HarnessCLI] Injecting strategy to SIMD Engine...")
        if inject_strategy_to_simd_engine(strategy):
            print("[HarnessCLI] Done!")
        else:
            print("[HarnessCLI] Injection failed, but strategy is saved.")


if __name__ == "__main__":
    main()
