#!/usr/bin/env python3
"""
Comprehensive Benchmark: llama.cpp vs MLX vs PyTorch MPS vs CGC Engine
測試不同後端的效能

後端:
- llama.cpp: CPU 推理 (GGUF 格式)
- MLX: Apple Silicon GPU 加速
- PyTorch MPS: Apple Silicon Metal 備用
- CGC Engine: PyTorch fallback 模式
"""

import sys
import os
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main')

MODEL_PATH = "/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf"

CONTEXT_LENGTHS = [512, 2048, 4096]


class LlamaCppBenchmark:
    """llama.cpp Benchmark (CPU + Metal)"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        self.backend = "cpu"
        self._load_model()

    def _load_model(self):
        try:
            from llama_cpp import Llama
            import torch

            mps_available = torch.backends.mps.is_available()
            n_gpu_layers = 32 if mps_available else 0
            self.backend = "metal" if mps_available else "cpu"

            print(f"   Loading llama.cpp model... (backend: {self.backend.upper()})")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=8192,
                n_gpu_layers=n_gpu_layers,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
            print(f"   llama.cpp loaded ({self.backend.upper()})")
        except Exception as e:
            print(f"   llama.cpp load failed: {e}")
            self.llm = None

    def get_memory_usage(self) -> Dict:
        if not self.llm:
            return {}
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                mem = torch.mps.current_allocated_memory() / 1024 / 1024
                if mem == 0:
                    return {"allocated_mb": 0, "note": "llama.cpp uses ggml memory, not MPS API"}
            else:
                mem = 0
            return {"allocated_mb": mem}
        except:
            return {}

    def benchmark_prefill(self, prompt: str, n_tokens: int = 64) -> Dict:
        if not self.llm:
            return {"error": "Model not loaded"}

        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        start_time = time.time()
        try:
            result = self.llm(prompt, max_tokens=n_tokens, stop=["</s>"], echo=False)
            elapsed = time.time() - start_time

            mem = self.get_memory_usage()

            return {
                "success": True,
                "elapsed_ms": elapsed * 1000,
                "tokens_per_sec": n_tokens / elapsed,
                "memory_mb": mem.get("allocated_mb", 0),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def benchmark_decode(self, prompt: str, n_tokens: int = 64) -> Dict:
        if not self.llm:
            return {"error": "Model not loaded"}

        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self.llm(prompt, max_tokens=10, stop=["</s>"])
        start_time = time.time()
        try:
            result = self.llm(prompt, max_tokens=n_tokens, stop=["</s>"], echo=False)
            elapsed = time.time() - start_time

            mem = self.get_memory_usage()

            return {
                "success": True,
                "elapsed_ms": elapsed * 1000,
                "tokens_per_sec": n_tokens / elapsed,
                "memory_mb": mem.get("allocated_mb", 0),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class MLXBenchmark:
    """MLX Benchmark (Apple Silicon GPU) - Simplified"""

    def __init__(self):
        self.available = False
        self.mx = None
        self._init_mlx()

    def _init_mlx(self):
        try:
            import mlx.core as mx
            import mlx.nn as mx_nn
            self.mx = mx
            self.mx_nn = mx_nn
            self.available = True
            print(f"   MLX device: {mx.default_device()}")
        except Exception as e:
            print(f"   MLX init failed: {e}")
            self.available = False

    def _matmul_attention(self, q, k, v, head_dim):
        """MLX attention with proper dimension handling"""
        B, T, H, D = q.shape
        scale = 1.0 / self.mx.sqrt(self.mx.array(D))

        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = self.mx.softmax(scores, axis=-1)
        out = attn @ v
        return out.transpose(0, 2, 1, 3).reshape(B, T, H, D)

    def _get_memory_mb(self) -> float:
        try:
            import mlx.core as mx
            mem_info = mx.get_peak_memory()
            return mem_info / 1024 / 1024
        except:
            return 0

    def benchmark_prefill(
        self,
        batch_size: int = 1,
        seq_len: int = 512,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> Dict:
        if not self.available:
            return {"error": "MLX not available"}

        try:
            self.mx.reset_peak_memory()

            start_time = time.time()

            q = self.mx.random.normal(shape=(batch_size, seq_len, num_heads, head_dim))
            k = self.mx.random.normal(shape=(batch_size, seq_len, num_heads, head_dim))
            v = self.mx.random.normal(shape=(batch_size, seq_len, num_heads, head_dim))

            for _ in range(3):
                _ = self._matmul_attention(q, k, v, head_dim)
                self.mx.eval(_)

            output = self._matmul_attention(q, k, v, head_dim)
            self.mx.eval(output)

            elapsed = time.time() - start_time
            mem_mb = self._get_memory_mb()

            return {
                "success": True,
                "elapsed_ms": elapsed * 1000,
                "seq_len": seq_len,
                "memory_mb": mem_mb,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def benchmark_decode(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> Dict:
        if not self.available:
            return {"error": "MLX not available"}

        try:
            start_time = time.time()

            q = self.mx.random.normal(shape=(batch_size, 1, num_heads, head_dim))
            k = self.mx.random.normal(shape=(batch_size, seq_len, num_heads, head_dim))
            v = self.mx.random.normal(shape=(batch_size, seq_len, num_heads, head_dim))

            output = self._matmul_attention(q, k, v, head_dim)
            self.mx.eval(output)

            elapsed = time.time() - start_time

            return {
                "success": True,
                "elapsed_ms": elapsed * 1000,
                "tokens_per_sec": 1.0 / elapsed,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class MPSBenchmark:
    """PyTorch MPS Benchmark (Apple Silicon Metal)"""

    def __init__(self):
        self.available = torch.backends.mps.is_available()
        print(f"   MPS available: {self.available}")

    def _get_memory_mb(self) -> float:
        try:
            torch.mps.empty_cache()
            return torch.mps.current_allocated_memory() / 1024 / 1024
        except:
            return 0

    def benchmark_prefill(
        self,
        batch_size: int = 1,
        seq_len: int = 512,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> Dict:
        if not self.available:
            return {"error": "MPS not available"}

        device = torch.device("mps")
        torch.mps.empty_cache()

        start_time = time.time()

        try:
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            _ = output.cpu()

            elapsed = time.time() - start_time
            mem_mb = self._get_memory_mb()

            return {
                "success": True,
                "elapsed_ms": elapsed * 1000,
                "seq_len": seq_len,
                "memory_mb": mem_mb,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def benchmark_decode(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> Dict:
        if not self.available:
            return {"error": "MPS not available"}

        device = torch.device("mps")
        torch.mps.empty_cache()

        start_time = time.time()

        try:
            q = torch.randn(batch_size, 1, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            _ = output.cpu()

            elapsed = time.time() - start_time

            return {
                "success": True,
                "elapsed_ms": elapsed * 1000,
                "tokens_per_sec": 1.0 / elapsed,
            }
        except RuntimeError as e:
            if "number of heads" in str(e):
                return {
                    "success": False,
                    "error": "MPS SDPA limitation: seq_len mismatch not supported",
                    "elapsed_ms": 0,
                    "tokens_per_sec": 0,
                }
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}


class CGCCEngineBenchmark:
    """CGC Engine Benchmark (Auto-detect: Metal/CUDA/CPU)"""

    def __init__(self):
        self.executor = None
        self.backend = "unknown"
        self.kda_available = False
        self._detect_and_init()

    def _detect_and_init(self):
        import torch

        if torch.cuda.is_available():
            self.backend = "cuda"
        elif torch.backends.mps.is_available():
            self.backend = "mps"
        else:
            self.backend = "cpu"

        print(f"   Detected backend: {self.backend.upper()}")

        try:
            from cgc_engine.cgc.flashkda_integration import _check_flashkda_available
            self.kda_available = _check_flashkda_available()
            print(f"   KDA available: {'✓' if self.kda_available else '✗ (using PyTorch SDPA)'}")
        except Exception as e:
            print(f"   KDA check failed: {e}")
            self.kda_available = False

        try:
            from cgc_engine.cgc.cgc_simd_executor import CGCExecutor
            print(f"   Initializing CGC Executor...")
            self.executor = CGCExecutor(enable_profiling=False)
            print(f"   CGC Executor initialized")
        except Exception as e:
            print(f"   CGC Executor init failed: {e}")

    def _get_memory_mb(self) -> float:
        try:
            import torch
            if self.backend == "mps":
                torch.mps.empty_cache()
                return torch.mps.current_allocated_memory() / 1024 / 1024
            elif self.backend == "cuda":
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return 0

    def benchmark_prefill(
        self,
        seq_len: int = 512,
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> Dict:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.mps.empty_cache() if device == "mps" else None

        start_time = time.time()

        try:
            q = torch.randn(1, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(1, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(1, seq_len, num_heads, head_dim, device=device)

            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

            if device == "mps":
                _ = output.cpu()

            elapsed = time.time() - start_time
            mem_mb = self._get_memory_mb()

            return {
                "success": True,
                "elapsed_ms": elapsed * 1000,
                "seq_len": seq_len,
                "memory_mb": mem_mb,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def benchmark_decode(
        self,
        seq_len: int = 128,
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> Dict:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.mps.empty_cache() if device == "mps" else None

        start_time = time.time()

        try:
            q = torch.randn(1, 1, num_heads, head_dim, device=device)
            k = torch.randn(1, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(1, seq_len, num_heads, head_dim, device=device)

            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

            if device == "mps":
                _ = output.cpu()

            elapsed = time.time() - start_time

            return {
                "success": True,
                "elapsed_ms": elapsed * 1000,
                "tokens_per_sec": 1.0 / elapsed,
            }
        except RuntimeError as e:
            if "number of heads" in str(e):
                return {
                    "success": False,
                    "error": "MPS SDPA limitation",
                    "elapsed_ms": 0,
                    "tokens_per_sec": 0,
                }
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}


def run_benchmarks():
    print("=" * 80)
    print("Comprehensive Benchmark: llama.cpp vs MLX vs MPS vs CGC Engine")
    print("=" * 80)
    print(f"\n模型: {MODEL_PATH}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"MPS: {torch.backends.mps.is_available()}")
    print()

    results = {}

    # 1. Llama.cpp Benchmark
    print("-" * 80)
    print("【1】Llama.cpp Benchmark (CPU)")
    print("-" * 80)
    llama = LlamaCppBenchmark(MODEL_PATH)
    if llama.llm:
        print("   Model loaded successfully")
        results["llama_cpp"] = {}
        for ctx_len in CONTEXT_LENGTHS:
            print(f"\n   [Context: {ctx_len}]")
            prefill = llama.benchmark_prefill("Hello world. " * (ctx_len // 5), n_tokens=32)
            decode = llama.benchmark_decode("Hello world. " * (ctx_len // 5), n_tokens=32)
            if prefill.get("success"):
                print(f"      Prefill: {prefill['elapsed_ms']:.2f}ms, {prefill['tokens_per_sec']:.2f} tokens/sec")
            if decode.get("success"):
                print(f"      Decode:  {decode['elapsed_ms']:.2f}ms, {decode['tokens_per_sec']:.2f} tokens/sec")
            results["llama_cpp"][ctx_len] = {"prefill": prefill, "decode": decode}

    # 2. MLX Benchmark
    print("\n" + "-" * 80)
    print("【2】MLX Benchmark (Apple Silicon GPU)")
    print("-" * 80)
    mlx_bench = MLXBenchmark()
    if mlx_bench.available:
        results["mlx"] = {}
        for ctx_len in CONTEXT_LENGTHS:
            print(f"\n   [Context: {ctx_len}]")
            try:
                prefill = mlx_bench.benchmark_prefill(seq_len=ctx_len)
                decode = mlx_bench.benchmark_decode(seq_len=ctx_len)
                if prefill.get("success"):
                    print(f"      Prefill: {prefill['elapsed_ms']:.2f}ms")
                if decode.get("success"):
                    print(f"      Decode:  {decode['elapsed_ms']:.2f}ms, {decode['tokens_per_sec']:.2f} tokens/sec")
                results["mlx"][ctx_len] = {"prefill": prefill, "decode": decode}
            except Exception as e:
                print(f"      MLX Error: {e}")
                results["mlx"][ctx_len] = {"prefill": {"error": str(e)}, "decode": {"error": str(e)}}
    else:
        print("   MLX not available")

    # 3. MPS Benchmark
    print("\n" + "-" * 80)
    print("【3】PyTorch MPS Benchmark (Metal)")
    print("-" * 80)
    mps_bench = MPSBenchmark()
    if mps_bench.available:
        results["mps"] = {}
        for ctx_len in CONTEXT_LENGTHS:
            print(f"\n   [Context: {ctx_len}]")
            prefill = mps_bench.benchmark_prefill(seq_len=ctx_len)
            decode = mps_bench.benchmark_decode(seq_len=ctx_len)
            if prefill.get("success"):
                print(f"      Prefill: {prefill['elapsed_ms']:.2f}ms")
            if decode.get("success"):
                print(f"      Decode:  {decode['elapsed_ms']:.2f}ms, {decode['tokens_per_sec']:.2f} tokens/sec")
            results["mps"][ctx_len] = {"prefill": prefill, "decode": decode}

    # 4. CGC Engine Benchmark
    print("\n" + "-" * 80)
    print("【4】CGC Engine Benchmark (Auto-detect)")
    print("-" * 80)
    cgc_bench = CGCCEngineBenchmark()
    if cgc_bench.executor:
        results["cgc"] = {}
        for ctx_len in CONTEXT_LENGTHS:
            print(f"\n   [Context: {ctx_len}]")
            prefill = cgc_bench.benchmark_prefill(seq_len=ctx_len)
            decode = cgc_bench.benchmark_decode(seq_len=ctx_len)
            if prefill.get("success"):
                print(f"      Prefill: {prefill['elapsed_ms']:.2f}ms")
            if decode.get("success"):
                print(f"      Decode:  {decode['elapsed_ms']:.2f}ms, {decode['tokens_per_sec']:.2f} tokens/sec")
            results["cgc"][ctx_len] = {"prefill": prefill, "decode": decode}

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Context':<10} {'llama.cpp':<20} {'MLX':<20} {'MPS':<20} {'CGC':<20}")
    print(f"{'':10} {'Prefill/Decode/Mem':<20} {'Prefill/Decode/Mem':<20} {'Prefill/Decode/Mem':<20} {'Prefill/Decode/Mem':<20}")
    print("-" * 100)

    for ctx_len in CONTEXT_LENGTHS:
        row = f"[{ctx_len:<8}]"
        for backend in ["llama_cpp", "mlx", "mps", "cgc"]:
            if backend in results and ctx_len in results[backend]:
                prefill = results[backend][ctx_len]["prefill"]
                decode = results[backend][ctx_len]["decode"]
                pf_ms = prefill.get("elapsed_ms", 0) if prefill.get("success") else 0
                dc_ms = decode.get("elapsed_ms", 0) if decode.get("success") else 0
                mem = prefill.get("memory_mb", 0) if prefill.get("success") else 0
                row += f" {pf_ms:.1f}/{dc_ms:.1f}/{mem:.0f}MB{'':5}"
            else:
                row += f" {'N/A':<20}"
        print(row)

    # KDA Status
    print("\n" + "-" * 80)
    print("KDA STATUS")
    print("-" * 80)
    if "cgc" in results:
        kda_status = cgc_bench.kda_available
        print(f"   KDA Enabled: {'✓ YES' if kda_status else '✗ NO (PyTorch SDPA fallback)'}")
        if not kda_status:
            print("   To enable KDA, compile C++ extension:")
            print("   cd cgc_engine/cgc/cgc_cpp && mkdir build && cd build && cmake .. && make")

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_benchmarks()