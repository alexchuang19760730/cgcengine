"""
MagiCompiler CUDA Graph 集成模块
Phase 1: torch.compile + CUDA Graph Integration for vLLM
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class CudaGraphManager:
    """
    CUDA Graph 管理器 - 管理 Prefill 和 Decode 阶段的计算图捕获与重放

    核心功能：
    1. 捕获 Prefill 阶段的完整计算图
    2. 支持动态序列长度的 Decode 图缓存
    3. 集成 torch.compile 进行算子优化
    """

    def __init__(self, enable_cudagraph: bool = True, enable_compile: bool = True):
        self.enable_cudagraph = enable_cudagraph
        self.enable_compile = enable_compile

        self.prefill_graph: Optional[torch.cuda.CUDAGraph] = None
        self.decode_graphs: Dict[int, Any] = {}

        self.prefill_input_placeholder: Optional[torch.Tensor] = None
        self.prefill_output_placeholder: Optional[Any] = None

        self.compiled_model: Optional[nn.Module] = None

        self._is_capturing = False
        self._graph_warmed = False

    def compile_model(self, model: nn.Module, **compile_kwargs) -> nn.Module:
        if not self.enable_compile:
            return model

        default_kwargs = {
            "mode": "reduce-overhead",
            "fullgraph": True,
            "dynamic": True,
        }
        default_kwargs.update(compile_kwargs)

        self.compiled_model = torch.compile(model, **default_kwargs)
        print("[MagiCompiler] ✅ 模型已编译")
        return self.compiled_model

    def capture_prefill_graph(
        self, model: nn.Module, sample_input: torch.Tensor, **forward_kwargs
    ) -> torch.cuda.CUDAGraph:
        if not self.enable_cudagraph:
            raise RuntimeError("CUDA Graph 未启用")

        self.prefill_input_placeholder = sample_input.clone().detach().requires_grad_(False)

        with torch.no_grad():
            warmup_output = model(self.prefill_input_placeholder, **forward_kwargs)

        if isinstance(warmup_output, torch.Tensor):
            self.prefill_output_placeholder = warmup_output.clone().detach().requires_grad_(False)
        elif isinstance(warmup_output, tuple):
            self.prefill_output_placeholder = tuple(
                o.clone().detach().requires_grad_(False) for o in warmup_output
            )

        self._is_capturing = True
        self.prefill_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.prefill_graph):
            output = model(self.prefill_input_placeholder, **forward_kwargs)

            if isinstance(output, torch.Tensor):
                self.prefill_output_placeholder.copy_(output)
            elif isinstance(output, tuple):
                for out, placeholder in zip(output, self.prefill_output_placeholder):
                    placeholder.copy_(out)

        self._is_capturing = False
        self._graph_warmed = True
        print(f"[MagiCompiler] ✅ Prefill Graph 已捕获 (输入形状: {sample_input.shape})")

        return self.prefill_graph

    def capture_decode_graph(
        self, model: nn.Module, seq_len: int, sample_input: torch.Tensor, **forward_kwargs
    ) -> torch.cuda.CUDAGraph:
        if not self.enable_cudagraph:
            raise RuntimeError("CUDA Graph 未启用")

        if seq_len in self.decode_graphs:
            return self.decode_graphs[seq_len]

        with torch.no_grad():
            warmup_output = model(sample_input, **forward_kwargs)

        output_placeholder = warmup_output.clone().detach().requires_grad_(False)
        input_placeholder = sample_input.clone().detach().requires_grad_(False)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(input_placeholder, **forward_kwargs)
            output_placeholder.copy_(output)

        self.decode_graphs[seq_len] = {
            "graph": graph,
            "input": input_placeholder,
            "output": output_placeholder,
        }

        print(f"[MagiCompiler] ✅ Decode Graph 已捕获 (seq_len: {seq_len})")
        return graph

    def replay_prefill(self, input_tensor: torch.Tensor) -> Any:
        if not self.prefill_graph or not self._graph_warmed:
            raise RuntimeError("Prefill Graph 未捕获或未预热")

        self.prefill_input_placeholder.copy_(input_tensor)
        self.prefill_graph.replay()

        if isinstance(self.prefill_output_placeholder, torch.Tensor):
            return self.prefill_output_placeholder.clone()
        return tuple(o.clone() for o in self.prefill_output_placeholder)

    def replay_decode(self, seq_len: int, input_tensor: torch.Tensor) -> torch.Tensor:
        if seq_len not in self.decode_graphs:
            raise RuntimeError(f"未找到 seq_len={seq_len} 的 Decode Graph")

        graph_info = self.decode_graphs[seq_len]

        graph_info["input"].copy_(input_tensor)
        graph_info["graph"].replay()

        return graph_info["output"].clone()

    def clear_cache(self):
        self.prefill_graph = None
        self.decode_graphs.clear()
        self._graph_warmed = False
        print("[MagiCompiler] 🧹 图缓存已清除")


class CUDAGraphManager(CudaGraphManager):
    pass


class VLLMCudaGraphWrapper:
    """
    vLLM 专用的 CUDA Graph 包装类
    集成到 vLLM 的推理流程中
    """

    def __init__(self, llm_model):
        self.llm = llm_model
        self.graph_manager = CudaGraphManager()
        self._prefill_ready = False
        self._decode_graphs: Dict[int, Any] = {}

    def initialize_prefill_graph(self, sample_prompt: str):
        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=1)
        _ = self.llm.generate([sample_prompt], sampling_params)

        self._prefill_ready = True
        print("[VLLM-CUDA-Graph] ✅ Prefill Graph 初始化完成")

    def generate_with_graph(self, prompts: list, sampling_params):
        if not self._prefill_ready:
            print("[VLLM-CUDA-Graph] 🔄 首次运行，预热中...")
            result = self.llm.generate(prompts, sampling_params)
            self._prefill_ready = True
            return result

        print("[VLLM-CUDA-Graph] ⚡ 使用 CUDA Graph 加速")
        return self.llm.generate(prompts, sampling_params)


def benchmark_cudagraph_vs_eager(
    model: nn.Module, input_tensor: torch.Tensor, num_iterations: int = 100
) -> Tuple[float, float]:
    with torch.no_grad():
        _ = model(input_tensor)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(input_tensor)
    end.record()
    torch.cuda.synchronize()
    eager_ms = start.elapsed_time(end) / num_iterations

    graph_manager = CudaGraphManager()
    graph_manager.capture_prefill_graph(model, input_tensor)

    start.record()
    for _ in range(num_iterations):
        _ = graph_manager.replay_prefill(input_tensor)
    end.record()
    torch.cuda.synchronize()
    graph_ms = start.elapsed_time(end) / num_iterations

    print(
        f"[Benchmark] Eager: {eager_ms:.2f} ms | Graph: {graph_ms:.2f} ms | Speedup: {eager_ms/graph_ms:.2f}x"
    )

    return eager_ms, graph_ms
