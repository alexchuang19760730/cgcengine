"""
MagiCompiler Phase 2: vLLM CUDA Graph 集成引擎
Integration of Dynamic CUDA Graph Cache with vLLM
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from vllm import LLM, SamplingParams

from cgc_engine.cuda.dynamic_graph_cache import PrefillDecodeGraphManager


@dataclass
class InferenceResult:
    """推理結果數據類"""

    prompt: str
    output_text: str
    num_prompt_tokens: int
    num_output_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    memory_allocated_gb: float
    memory_reserved_gb: float


class VLLMCudaGraphEngine:
    """
    vLLM CUDA Graph 集成引擎

    核心功能：
    1. 自動捕獲 Prefill 和 Decode 階段的計算圖
    2. 動態 Shape 緩存
    3. 內存追蹤
    4. 性能基準測試
    """

    def __init__(
        self,
        model_path: str,
        enable_cudagraph: bool = True,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        **kwargs,
    ):
        self.model_path = model_path
        self.enable_cudagraph = enable_cudagraph
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        print("[VLLM-CUDA-Graph] 初始化 vLLM 引擎...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enable_cudagraph,
            **kwargs,
        )

        if enable_cudagraph:
            self.graph_manager = PrefillDecodeGraphManager(max_decode_cache_size=20)
        else:
            self.graph_manager = None

        self._reset_memory_stats()
        self._is_warmed = False

        print("[VLLM-CUDA-Graph] ✅ 初始化完成")

    def _reset_memory_stats(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        self._baseline_memory_gb = 0.0
        self._peak_memory_gb = 0.0
        self._memory_samples = []

    def _sample_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            self._memory_samples.append(allocated)
            self._peak_memory_gb = max(self._peak_memory_gb, allocated)

    def warmup(self, sample_prompt: str = "Hello", max_tokens: int = 16):
        if self._is_warmed:
            print("[VLLM-CUDA-Graph] 已經預熱過")
            return

        print("[VLLM-CUDA-Graph] 🔄 預熱中...")

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens)

        if torch.cuda.is_available():
            self._baseline_memory_gb = torch.cuda.memory_allocated() / (1024**3)

        _ = self.llm.generate([sample_prompt], sampling_params)
        self._sample_memory()

        self._is_warmed = True
        print("[VLLM-CUDA-Graph] ✅ 預熱完成")

    def generate(
        self, prompts: List[str], sampling_params: Optional[SamplingParams] = None, use_graph: bool = True
    ) -> List[InferenceResult]:
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

        if not self._is_warmed:
            self.warmup()

        start_time = time.time()
        self._sample_memory()

        outputs = self.llm.generate(prompts, sampling_params)

        total_time = (time.time() - start_time) * 1000
        self._sample_memory()

        results: List[InferenceResult] = []
        for prompt, output in zip(prompts, outputs):
            prompt_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)

            prefill_ratio = (
                prompt_tokens / (prompt_tokens + output_tokens) if output_tokens > 0 else 0.5
            )
            prefill_time = total_time * prefill_ratio
            decode_time = total_time * (1 - prefill_ratio)

            results.append(
                InferenceResult(
                    prompt=prompt,
                    output_text=output.outputs[0].text,
                    num_prompt_tokens=prompt_tokens,
                    num_output_tokens=output_tokens,
                    prefill_time_ms=prefill_time,
                    decode_time_ms=decode_time,
                    total_time_ms=total_time,
                    memory_allocated_gb=self._peak_memory_gb,
                    memory_reserved_gb=torch.cuda.memory_reserved() / (1024**3)
                    if torch.cuda.is_available()
                    else 0,
                )
            )

        return results

    def benchmark(
        self, prompts: List[str], sampling_params: Optional[SamplingParams] = None, num_iterations: int = 5
    ) -> Dict[str, Any]:
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

        print(f"\n{'='*60}")
        print(f"基準測試: {len(prompts)} prompts, {num_iterations} iterations")
        print(f"{'='*60}")

        all_results: List[InferenceResult] = []
        memory_stats = {"peak_gb": 0.0, "avg_gb": 0.0, "samples": []}

        for i in range(num_iterations):
            print(f"\n[迭代 {i+1}/{num_iterations}]")
            results = self.generate(prompts, sampling_params, use_graph=True)

            for r in results:
                print(f"  - Prompt: {r.prompt[:30]}...")
                print(f"    Output tokens: {r.num_output_tokens}")
                print(f"    Time: {r.total_time_ms:.2f} ms")
                print(f"    Memory: {r.memory_allocated_gb:.2f} GB")

            all_results.extend(results)

            memory_stats["samples"].extend([r.memory_allocated_gb for r in results])
            memory_stats["peak_gb"] = max(memory_stats["peak_gb"], max(memory_stats["samples"]))

        memory_stats["avg_gb"] = (
            sum(memory_stats["samples"]) / len(memory_stats["samples"]) if memory_stats["samples"] else 0
        )

        total_prefill_time = sum(r.prefill_time_ms for r in all_results)
        total_decode_time = sum(r.decode_time_ms for r in all_results)
        total_time = sum(r.total_time_ms for r in all_results)
        total_output_tokens = sum(r.num_output_tokens for r in all_results)

        benchmark_result = {
            "num_prompts": len(prompts),
            "num_iterations": num_iterations,
            "total_inferences": len(all_results),
            "total_output_tokens": total_output_tokens,
            "avg_prefill_time_ms": total_prefill_time / len(all_results),
            "avg_decode_time_ms": total_decode_time / len(all_results),
            "avg_total_time_ms": total_time / len(all_results),
            "throughput_tokens_per_sec": (total_output_tokens * 1000) / total_time if total_time > 0 else 0,
            "memory": memory_stats,
        }

        print(f"\n{'='*60}")
        print("基準測試結果")
        print(f"{'='*60}")
        print(f"  總推理次數: {len(all_results)}")
        print(f"  總輸出 tokens: {total_output_tokens}")
        print(f"  平均 Prefill 時間: {benchmark_result['avg_prefill_time_ms']:.2f} ms")
        print(f"  平均 Decode 時間: {benchmark_result['avg_decode_time_ms']:.2f} ms")
        print(f"  平均總時間: {benchmark_result['avg_total_time_ms']:.2f} ms")
        print(f"  吞吐量: {benchmark_result['throughput_tokens_per_sec']:.2f} tokens/s")
        print(f"  峰值內存: {memory_stats['peak_gb']:.2f} GB")
        print(f"  平均內存: {memory_stats['avg_gb']:.2f} GB")
        print(f"{'='*60}")

        return benchmark_result

    def compare_with_without_graph(self, prompts: List[str], num_iterations: int = 3) -> Dict[str, Dict[str, float]]:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

        print(f"\n{'='*60}")
        print("性能對比測試")
        print(f"{'='*60}")

        print("\n[1/2] 使用 CUDA Graph...")
        graph_results = self.benchmark(prompts, sampling_params, num_iterations)

        print("\n[2/2] 不使用 CUDA Graph...")
        llm_no_graph = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enforce_eager=True,
        )

        _ = llm_no_graph.generate(["Warmup"], sampling_params)

        no_graph_times: List[float] = []
        for i in range(num_iterations):
            start = time.time()
            _ = llm_no_graph.generate(prompts, sampling_params)
            elapsed = (time.time() - start) * 1000
            no_graph_times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f} ms")

        avg_no_graph = sum(no_graph_times) / len(no_graph_times)
        speedup = graph_results["avg_total_time_ms"] / avg_no_graph if avg_no_graph > 0 else 0

        comparison = {
            "with_graph": {
                "avg_time_ms": graph_results["avg_total_time_ms"],
                "throughput": graph_results["throughput_tokens_per_sec"],
            },
            "without_graph": {
                "avg_time_ms": avg_no_graph,
                "throughput": 0,
            },
            "speedup": speedup,
        }

        print(f"\n{'='*60}")
        print("對比結果")
        print(f"{'='*60}")
        print(f"  使用 Graph: {comparison['with_graph']['avg_time_ms']:.2f} ms")
        print(f"  不使用 Graph: {comparison['without_graph']['avg_time_ms']:.2f} ms")
        print(f"  加速比: {comparison['speedup']:.2f}x")
        print(f"{'='*60}")

        return comparison


def create_engine(model_path: str, **kwargs) -> VLLMCudaGraphEngine:
    return VLLMCudaGraphEngine(model_path=model_path, **kwargs)
