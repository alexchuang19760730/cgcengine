#!/usr/bin/env python3
"""
CGC SIMD Engine 基準測試 - llama.cpp 驅動

功能：
1. 追蹤 llama.cpp 推理時使用的所有 CGC Commands
2. 測量各 command 的執行時間和內存使用
3. 識別優化機會 (高頻模式、Gap)

使用方法：
    python benchmark_llama_cpp_driven.py \
        --gguf-path /path/to/model.gguf \
        --num-tokens 100 \
        --output results.json
"""

import argparse
import json
import time
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

import torch

# 添加項目路徑
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OpcodeStats:
    """單個 Opcode 的統計"""
    count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0

    def update(self, elapsed: float):
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.avg_time = self.total_time / self.count if self.count > 0 else 0


@dataclass
class BenchmarkResult:
    """基準測試結果"""
    model_name: str
    model_path: str
    total_time: float
    prefill_time: float = 0.0
    decode_time: float = 0.0
    num_tokens: int = 0
    num_layers: int = 0
    hidden_dim: int = 0
    num_heads: int = 0

    opcode_stats: Dict[str, OpcodeStats] = field(default_factory=dict)
    opcode_counts: Dict[str, int] = field(default_factory=dict)
    opcode_times: Dict[str, float] = field(default_factory=dict)

    peak_memory_mb: float = 0.0
    fusion_opportunities: List[List[str]] = field(default_factory=list)
    missing_opcodes: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "total_time": self.total_time,
            "prefill_time": self.prefill_time,
            "decode_time": self.decode_time,
            "num_tokens": self.num_tokens,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "opcode_counts": self.opcode_counts,
            "opcode_times": self.opcode_times,
            "peak_memory_mb": self.peak_memory_mb,
            "fusion_opportunities": self.fusion_opportunities,
            "missing_opcodes": self.missing_opcodes,
            "metadata": self.metadata,
        }


class ProfilingCGCExecutor:
    """帶 Profiling 功能的 CGC Executor"""

    def __init__(self):
        self.opcode_stats: Dict[str, OpcodeStats] = {}
        self.enabled = False
        self._original_execute = None

    def enable(self):
        """啟用 Profiling"""
        self.enabled = True
        logger.info("[ProfilingCGCExecutor] Enabled")

    def disable(self):
        """停用 Profiling"""
        self.enabled = False
        logger.info("[ProfilingCGCExecutor] Disabled")

    def record_opcode(self, opcode: int, elapsed: float):
        """記錄 Opcode 執行"""
        if not self.enabled:
            return

        opcode_hex = f"0x{opcode:02X}"
        if opcode_hex not in self.opcode_stats:
            self.opcode_stats[opcode_hex] = OpcodeStats()
        self.opcode_stats[opcode_hex].update(elapsed)

    def get_stats(self) -> Dict[str, Any]:
        """獲取統計"""
        return {
            "opcode_counts": {
                k: v.count for k, v in self.opcode_stats.items()
            },
            "opcode_times": {
                k: v.total_time for k, v in self.opcode_stats.items()
            },
            "opcode_avg_times": {
                k: v.avg_time for k, v in self.opcode_stats.items()
            },
        }

    def reset(self):
        """重置統計"""
        self.opcode_stats.clear()
        logger.info("[ProfilingCGCExecutor] Reset")


def detect_fusion_opportunities(opcode_stats: Dict[str, OpcodeStats]) -> List[List[str]]:
    """
    識別可融合的算子模式

    基於 llama.cpp 的常見模式：
    1. QKV Projection + RoPE: [q_proj, k_proj, v_proj, rope]
    2. MLP: [linear, silu, linear]
    3. Attention: [rms_norm, attention, rms_norm]
    """
    opportunities = []

    # 檢測 QKV 模式
    qkv_opcodes = ["0x10", "0x11", "0x12"]  # 假設的線性層 opcode
    rope_opcode = "0x24"  # 假設的 RoPE opcode

    has_qkv = any(op in opcode_stats for op in qkv_opcodes)
    has_rope = rope_opcode in opcode_stats

    if has_qkv and has_rope:
        opportunities.append(["q_proj", "k_proj", "v_proj", "rope"])

    # 檢測 MLP 模式
    silu_opcode = "0x31"
    linear_opcodes = ["0x10", "0x11", "0x12"]

    has_silu = silu_opcode in opcode_stats
    if has_silu and any(op in opcode_stats for op in linear_opcodes):
        opportunities.append(["linear", "silu", "linear"])

    return opportunities


def identify_missing_opcodes(
    used_opcodes: set,
    implemented_opcodes: set
) -> List[str]:
    """識別未實現的 Opcode"""
    missing = used_opcodes - implemented_opcodes
    return sorted(list(missing))


def run_llama_cpp_benchmark(
    gguf_path: str,
    num_tokens: int = 100,
    temperature: float = 0.7,
    profile: bool = True,
) -> BenchmarkResult:
    """
    運行 llama.cpp 基準測試

    Args:
        gguf_path: GGUF 模型路徑
        num_tokens: 生成 token 數量
        temperature: 采樣溫度
        profile: 是否啟用 Profiling

    Returns:
        BenchmarkResult
    """
    from cgc_engine.model_parsers import GGUFParser
    from cgc_engine.cgc import CGCExecutor

    logger.info(f"[Benchmark] Starting benchmark for {gguf_path}")

    # 初始化 Profiling Executor
    profiler = ProfilingCGCExecutor()
    if profile:
        profiler.enable()

    # 解析 GGUF 模型
    parser = GGUFParser(gguf_path)
    parsed_model = parser.parse()

    # 獲取模型信息
    result = BenchmarkResult(
        model_name=parsed_model.model_type or Path(gguf_path).stem,
        model_path=gguf_path,
        num_tokens=num_tokens,
        num_layers=parsed_model.num_layers,
        hidden_dim=parsed_model.hidden_dim,
        num_heads=parsed_model.num_heads,
    )

    # 記錄初始內存
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        initial_memory = 0

    # 創建 CGC Executor
    executor = CGCExecutor(mode="profiling" if profile else "normal")

    # 嘗試加載 llama.cpp 模型
    llama_model = None
    try:
        from llama_cpp import Llama
        llama_model = Llama(
            model_path=gguf_path,
            n_ctx=2048,
            n_threads=8,
            use_mmap=True,
            use_mlock=False,
        )
        logger.info("[Benchmark] llama.cpp model loaded")
    except ImportError:
        logger.warning("[Benchmark] llama.cpp not available, using PyTorch fallback")
    except Exception as e:
        logger.warning(f"[Benchmark] Failed to load llama.cpp: {e}")

    # 推理
    prompt = "Hello, I am a"
    start_time = time.time()

    if llama_model:
        # 使用 llama.cpp 推理
        try:
            prefill_start = time.time()
            output = llama_model(
                prompt,
                max_tokens=num_tokens,
                temperature=temperature,
                echo=False,
            )
            prefill_time = time.time() - prefill_start

            result.prefill_time = prefill_time
            result.decode_time = result.total_time - prefill_time
            result.total_time = time.time() - start_time

            logger.info(f"[Benchmark] llama.cpp inference completed in {result.total_time:.2f}s")
        except Exception as e:
            logger.error(f"[Benchmark] llama.cpp inference failed: {e}")
            result.metadata["error"] = str(e)
    else:
        # 使用 PyTorch fallback
        result.metadata["backend"] = "pytorch_fallback"
        result.total_time = time.time() - start_time

    # 記錄內存
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        result.peak_memory_mb = peak_memory - initial_memory

    # 獲取 opcode 統計
    if profile:
        try:
            exec_stats = executor.get_stats()
            result.opcode_counts = exec_stats.get("opcode_counts", {})
            result.opcode_times = exec_stats.get("opcode_times", {})

            # 重建 OpcodeStats
            profiler.opcode_stats = {
                k: OpcodeStats(
                    count=v if isinstance(v, int) else 1,
                    total_time=t if isinstance(t, (int, float)) else 0.0
                )
                for k, v, t in zip(
                    result.opcode_counts.keys(),
                    result.opcode_counts.values(),
                    result.opcode_times.values()
                )
            }
        except Exception as e:
            logger.warning(f"[Benchmark] Failed to get executor stats: {e}")

    # 識別融合機會
    result.fusion_opportunities = detect_fusion_opportunities(profiler.opcode_stats)

    # 識別缺失的 Opcode
    implemented_opcodes = {
        "0x01", "0x02", "0x03", "0x04",  # Attention
        "0x10", "0x11", "0x12",  # Linear
        "0x20", "0x21",  # Norm
        "0x30", "0x31",  # Activation
    }
    result.missing_opcodes = identify_missing_opcodes(
        set(result.opcode_counts.keys()),
        implemented_opcodes
    )

    logger.info(f"[Benchmark] Total time: {result.total_time:.2f}s")
    logger.info(f"[Benchmark] Peak memory: {result.peak_memory_mb:.2f} MB")
    logger.info(f"[Benchmark] Fusion opportunities: {len(result.fusion_opportunities)}")
    logger.info(f"[Benchmark] Missing opcodes: {result.missing_opcodes}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CGC SIMD Engine Benchmark - llama.cpp Driven"
    )
    parser.add_argument(
        "--gguf-path",
        type=str,
        required=True,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_result.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable profiling"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 檢查文件是否存在
    if not os.path.exists(args.gguf_path):
        logger.error(f"Model file not found: {args.gguf_path}")
        sys.exit(1)

    # 運行基準測試
    result = run_llama_cpp_benchmark(
        gguf_path=args.gguf_path,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
        profile=not args.no_profile,
    )

    # 保存結果
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"[Benchmark] Results saved to {args.output}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("📊 Benchmark Summary")
    print("=" * 60)
    print(f"Model: {result.model_name}")
    print(f"Total time: {result.total_time:.2f}s")
    print(f"Peak memory: {result.peak_memory_mb:.2f} MB")
    print(f"\nTop 5 Opcode usage:")
    sorted_opcodes = sorted(
        result.opcode_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for opcode, count in sorted_opcodes:
        print(f"  {opcode}: {count} times")
    print(f"\nFusion opportunities: {len(result.fusion_opportunities)}")
    print(f"Missing opcodes: {len(result.missing_opcodes)}")
    if result.missing_opcodes:
        print(f"  {result.missing_opcodes}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
