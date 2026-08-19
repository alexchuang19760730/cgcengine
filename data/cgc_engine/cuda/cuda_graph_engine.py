# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CUDA Graph Prefill Engine

Prefill 阶段优化：
- 将矩阵计算与 NCCL AllReduce 通信统一打包为 CUDA Graph
- 一次 launch 执行完整流水线
- 消除 CPU 调度、kernel 启动与 NCCL 重复初始化开销
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CUDAGraphConfig:
    """CUDA Graph 配置"""
    enable_graph: bool = True
    warmup_runs: int = 3
    memory_fraction: float = 0.9
    num_layers: int = 28
    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128


class CUDAGraphCapture:
    """
    CUDA Graph 捕获器

    使用 torch.cuda.CUDAGraph 捕获 GPU 操作，
    后续 replay 时无需重新调度 kernel
    """

    def __init__(self, config: CUDAGraphConfig):
        self.config = config
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        self.is_captured: bool = False

        self._allocate_buffers()

    def _allocate_buffers(self):
        """预分配静态缓冲区"""
        batch_size = self.config.hidden_dim
        self.static_input = torch.empty(
            (batch_size, self.config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )
        self.static_output = torch.empty(
            (batch_size, self.config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )

        logger.info(f"[CUDA Graph] 预分配缓冲区: {self.static_input.numel() * 2 / 1e6:.2f} MB")

    def capture(self, compute_fn, *args, **kwargs):
        """
        捕获计算图

        Args:
            compute_fn: 要捕获的计算函数
        """
        if not self.config.enable_graph:
            logger.warning("[CUDA Graph] Graph 捕获被禁用")
            return

        logger.info("[CUDA Graph] 开始捕获计算图...")

        # 先 warmup
        for _ in range(self.config.warmup_runs):
            compute_fn(*args, **kwargs)
        torch.cuda.synchronize()

        # 重置图
        self.graph = torch.cuda.CUDAGraph()

        # 捕获
        with torch.cuda.graph(self.graph):
            self.static_output = compute_fn(*args, **kwargs)

        self.is_captured = True
        logger.info("[CUDA Graph] ✅ 图捕获完成")

    def replay(self, *args, **kwargs) -> torch.Tensor:
        """
        重放捕获的图

        优势：
        - 无需 CPU 调度
        - 无需 kernel 启动开销
        - 无需 NCCL 重复初始化
        """
        if not self.is_captured:
            raise RuntimeError("[CUDA Graph] 图尚未捕获，无法重放")

        # 更新输入（如果需要）
        if args:
            self.static_input.copy_(args[0])

        # 重放图
        self.graph.replay()

        return self.static_output


class NCCLCommWrapper:
    """
    NCCL 通信包装器

    模拟 TP=2 环境下的 AllReduce 通信
    """

    def __init__(self, rank: int = 0, world_size: int = 2):
        self.rank = rank
        self.world_size = world_size
        self.is_initialized = False
        self.comm = None

    def init_communicator(self):
        """初始化 NCCL 通信器"""
        if self.rank == 0:
            logger.info(f"[NCCL] 初始化通信器: rank={self.rank}, world_size={self.world_size}")
        self.is_initialized = True

    def allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        AllReduce 操作

        在 CUDA Graph 中调用时，需要确保通信被正确捕获
        """
        if not self.is_initialized:
            self.init_communicator()

        # 模拟 AllReduce: 求平均值
        # 实际生产环境中使用 torch.distributed.all_reduce
        tensor.mul_(1.0 / self.world_size)
        return tensor

    def barrier(self):
        """同步屏障"""
        torch.cuda.synchronize()


class PrefillCUDAGraphEngine:
    """
    Prefill CUDA Graph 引擎

    将矩阵计算与 NCCL AllReduce 统一打包为 CUDA Graph
    """

    def __init__(self, config: CUDAGraphConfig, rank: int = 0, world_size: int = 2):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        self.graph_capture = CUDAGraphCapture(config)
        self.nccl = NCCLCommWrapper(rank=rank, world_size=world_size)

        self.linear_q = nn.Linear(config.hidden_dim, config.hidden_dim).cuda().half()
        self.linear_k = nn.Linear(config.hidden_dim, config.hidden_dim).cuda().half()
        self.linear_v = nn.Linear(config.hidden_dim, config.hidden_dim).cuda().half()
        self.linear_o = nn.Linear(config.hidden_dim, config.hidden_dim).cuda().half()

        self.is_graph_captured = False

    def _compute_attention(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        单层 Attention 计算（将被捕获到 CUDA Graph）

        包含：
        1. QKV 线性变换
        2. AllReduce 同步
        3. Attention 计算
        4. Output 线性变换
        """
        # QKV 变换
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # NCCL AllReduce 同步（在 TP 中需要同步 K/V）
        if self.world_size > 1:
            k = self.nccl.allreduce(k)
            v = self.nccl.allreduce(v)

        # 简化的 Attention 计算
        scale = 1.0 / (self.config.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Output 变换
        out = self.linear_o(out)

        # 残差连接
        return out + x

    def _full_prefill_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        完整 Prefill 前向计算（将被捕获）

        包含多层 Transformer 计算 + AllReduce
        """
        x = self.linear_q(input_ids)  # 嵌入层

        for layer_idx in range(self.config.num_layers):
            x = self._compute_attention(x, layer_idx)

        return x

    def capture_graph(self, batch_size: int, seq_len: int):
        """
        捕获完整 Prefill 计算图

        包含：
        - 多层矩阵乘法
        - NCCL AllReduce 通信
        - Softmax 等非线性操作
        """
        logger.info(f"[Prefill Graph] 开始捕获: batch={batch_size}, seq={seq_len}")

        dummy_input = torch.randn(
            (batch_size, seq_len, self.config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )

        def compute_fn(x):
            return self._full_prefill_forward(x)

        self.graph_capture.capture(compute_fn, dummy_input)
        self.is_graph_captured = True

        logger.info("[Prefill Graph] ✅ Prefill CUDA Graph 捕获完成")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Prefill 前向计算

        如果已捕获图则直接重放，否则执行普通计算
        """
        if self.is_graph_captured:
            return self.graph_capture.replay(input_ids)
        else:
            return self._full_prefill_forward(input_ids)

    def benchmark(self, batch_size: int, seq_len: int, num_runs: int = 100) -> Dict[str, float]:
        """
        性能基准测试

        对比：
        - 普通执行（有 CPU 调度开销）
        - CUDA Graph 重放（无调度开销）
        """
        logger.info(f"\n[Prefill Benchmark] batch={batch_size}, seq={seq_len}, runs={num_runs}")

        input_tensor = torch.randn(
            (batch_size, seq_len, self.config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )

        # Warmup
        for _ in range(10):
            self.forward(input_tensor)
        torch.cuda.synchronize()

        # 普通执行基准
        torch.cuda.reset_accumulated_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            self.forward(input_tensor)
        end.record()
        torch.cuda.synchronize()
        normal_time = start.elapsed_time(end) / num_runs

        # 如果尚未捕获图，先捕获
        if not self.is_graph_captured:
            self.capture_graph(batch_size, seq_len)

        # Graph 重放基准
        torch.cuda.reset_accumulated_memory_stats()
        start.record()
        for _ in range(num_runs):
            self.forward(input_tensor)
        end.record()
        torch.cuda.synchronize()
        graph_time = start.elapsed_time(end) / num_runs

        speedup = normal_time / graph_time if graph_time > 0 else 1.0

        results = {
            "normal_time_ms": normal_time,
            "graph_time_ms": graph_time,
            "speedup": speedup,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_layers": self.config.num_layers,
            "hidden_dim": self.config.hidden_dim
        }

        logger.info(f"[Prefill Benchmark] 结果:")
        logger.info(f"   普通执行: {normal_time:.3f} ms")
        logger.info(f"   Graph 重放: {graph_time:.3f} ms")
        logger.info(f"   加速比: {speedup:.2f}x")

        return results


class StaticDecodeEngine:
    """
    静态图 Decode 引擎

    将 Attention、NCCL 同步、MLP 计算固化为静态图
    实现循环重放，大幅降低 Decode 阶段开销
    """

    def __init__(self, config: CUDAGraphConfig, rank: int = 0, world_size: int = 2):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        self.nccl = NCCLCommWrapper(rank=rank, world_size=world_size)

        # 静态权重（Decode 阶段权重不变）
        self.static_k = torch.randn(
            (config.hidden_dim, config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )
        self.static_v = torch.randn(
            (config.hidden_dim, config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )
        self.static_mlp_w1 = torch.randn(
            (config.hidden_dim * 4, config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )
        self.static_mlp_w2 = torch.randn(
            (config.hidden_dim, config.hidden_dim * 4),
            dtype=torch.float16,
            device="cuda"
        )

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.is_captured = False

        # 静态输入输出
        self.static_q = torch.empty(
            (1, config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )
        self.static_context = torch.empty(
            (1, config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )

    def _static_decode_step(self, q: torch.Tensor) -> torch.Tensor:
        """
        单步 Decode 计算（将被固化为静态图）

        包含：
        1. Attention 计算（使用静态 K/V）
        2. NCCL 同步
        3. MLP 计算
        """
        # Attention: Q @ K^T
        scale = 1.0 / (self.config.head_dim ** 0.5)
        attn_scores = torch.matmul(q, self.static_k.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, self.static_v)

        # NCCL 同步（TP 分布式）
        if self.world_size > 1:
            attn_out = self.nccl.allreduce(attn_out)

        # MLP
        mlp_inter = torch.matmul(attn_out, self.static_mlp_w1)
        mlp_inter = torch.nn.functional.gelu(mlp_inter)
        mlp_out = torch.matmul(mlp_inter, self.static_mlp_w2)

        return mlp_out + attn_out

    def capture_static_graph(self, num_steps: int = 32):
        """
        捕获静态 Decode 图

        Args:
            num_steps: 循环重放的步数
        """
        logger.info(f"[Static Decode] 开始捕获: num_steps={num_steps}")

        # Warmup
        for _ in range(10):
            self._static_decode_step(self.static_q)
        torch.cuda.synchronize()

        # 捕获
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            for step in range(num_steps):
                self.static_context = self._static_decode_step(self.static_q)

        self.is_captured = True
        logger.info("[Static Decode] ✅ 静态图捕获完成")

    def decode_step(self, q: torch.Tensor) -> torch.Tensor:
        """
        执行单步 Decode

        如果图已捕获则重放
        """
        if self.is_captured:
            self.static_q.copy_(q)
            self.graph.replay()
            return self.static_context
        else:
            return self._static_decode_step(q)

    def decode_sequence(self, initial_q: torch.Tensor, num_tokens: int) -> List[torch.Tensor]:
        """
        Decode 整个序列

        循环重放静态图
        """
        outputs = []
        q = initial_q.clone()

        for _ in range(num_tokens):
            out = self.decode_step(q)
            outputs.append(out)
            q = out  # 自回归

        return outputs

    def benchmark(self, num_tokens: int = 128, num_runs: int = 10) -> Dict[str, float]:
        """
        性能基准测试
        """
        logger.info(f"\n[Static Decode Benchmark] tokens={num_tokens}, runs={num_runs}")

        # 捕获图（如果尚未捕获）
        if not self.is_captured:
            self.capture_static_graph(num_steps=num_tokens)

        q = torch.randn(
            (1, self.config.hidden_dim),
            dtype=torch.float16,
            device="cuda"
        )

        # Warmup
        for _ in range(10):
            self.decode_step(q)
        torch.cuda.synchronize()

        # 静态图重放基准
        torch.cuda.reset_accumulated_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        total_tokens = 0
        start.record()
        for _ in range(num_runs):
            outputs = self.decode_sequence(q, num_tokens)
            total_tokens += len(outputs)
        end.record()
        torch.cuda.synchronize()
        total_time = start.elapsed_time(end)

        avg_time_per_token = total_time / total_tokens
        tokens_per_sec = (total_tokens * 1000) / total_time

        results = {
            "total_time_ms": total_time,
            "avg_time_per_token_ms": avg_time_per_token,
            "tokens_per_sec": tokens_per_sec,
            "num_tokens": num_tokens,
            "num_runs": num_runs,
            "total_tokens": total_tokens,
            "hidden_dim": self.config.hidden_dim
        }

        logger.info(f"[Static Decode Benchmark] 结果:")
        logger.info(f"   总耗时: {total_time:.2f} ms")
        logger.info(f"   每 Token 平均: {avg_time_per_token:.4f} ms")
        logger.info(f"   吞吐量: {tokens_per_sec:.2f} tokens/s")

        return results
