# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
MagiCompiler 整合层 (v1.1.0 API)

统一编排层，负责：
1. 全计算图分析  —— 基于 GraphAnalyzer 捕获整图，识别 MoE / Attention / 量化等特征；
2. 统一 IR       —— 把图分析结果降级为统一的编译 IR（splitting_ops、compute_sensitive_ops、
                     memory_hierarchy、scheduling_plan），同时可下发给 C++ expert-streamer；
3. 编译执行      —— 通过新的 `magi_compile` / `magi_register_custom_op` API 驱动
                     torch.compile（piecewise 后端），MoE 专家流式算子注册为 subgraph
                     boundary + compute_sensitive，实现全后端（cpu/cuda/metal/inductor）统一接入；
4. 兼容层        —— 保留旧版 `MagiCompiler` / `MagiCompiledModel` 接口，供
                     cgc_engine_test.py 等既有调用方使用。

无 torch 环境（仅做静态分析 / py_compile）下自动回退为 eager 包装，不报错。
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 可选依赖：torch 与 MagiCompiler v1.1.0 API
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

try:
    from cgc_engine.api import magi_compile, magi_register_custom_op
    from cgc_engine.config import CompileConfig, CompileMode, get_compile_config
    _MAGI_API_AVAILABLE = True
except Exception:  # pragma: no cover
    magi_compile = None
    magi_register_custom_op = None
    get_compile_config = None
    _MAGI_API_AVAILABLE = False

try:
    from .agent.graph_analyzer import GraphAnalyzer
    _GRAPH_ANALYZER_AVAILABLE = True
except Exception:  # pragma: no cover - torch 缺失等场景
    GraphAnalyzer = None
    _GRAPH_ANALYZER_AVAILABLE = False

# ---------------------------------------------------------------------------
# 统一 IR 常量：C++ expert-streamer 与 torch 编译共用
# ---------------------------------------------------------------------------
# 统一的 MoE 专家流式算子名（torch 侧 custom op / C++ 侧 build_moe_ffn hook 同名）。
MOE_EXPERT_STREAM_OP = "cgc::moe_expert_stream"

# C++ expert-streamer 环境变量（cgc_engine/cpp/expert_streaming 与 llama.cpp 端共用）
EX_STREAM_ENV = {
    "enable": "CGC_EXPERT_STREAM",
    "slots": "CGC_EX_STREAM_SLOTS",
    "hot_pool": "CGC_EX_HOT_POOL",
    "mtp": "CGC_EX_MTP_TOKENS",
    "pd": "CGC_EX_PD_SPLIT",
    "prefill_gpu": "CGC_EX_PREFILL_GPU",
    "decode_gpu": "CGC_EX_DECODE_GPU",
    "prefetch_threads": "CGC_EX_PREFETCH_THREADS",
}

_UNIFIED_IR_DEFAULT = {
    "splitting_ops": [MOE_EXPERT_STREAM_OP],
    "compute_sensitive_ops": [MOE_EXPERT_STREAM_OP],
    "memory_hierarchy": {
        "weight": "mmap",        # 专家权重零拷贝驻留在 mmap 页
        "activation": "device",  # 激活驻留计算设备
    },
    "scheduling": {
        "pd_split": False,
        "prefill_gpu": 0,
        "decode_gpu": 1,
        "prefetch_threads": 4,
    },
}


def _expert_stream_env() -> Dict[str, str]:
    """把统一 IR 的调度配置落成 C++ expert-streamer 读到的环境变量。"""
    return {v: os.environ.get(k, "") for k, v in EX_STREAM_ENV.items()}


def register_moe_expert_op():
    """注册统一的 MoE 专家流式 custom op（subgraph boundary + compute sensitive）。

    仅当 torch 可用时注册；重复注册由 torch.library 幂等处理。
    该算子与 C++ 侧 `build_moe_ffn` 中注入的 expert-streamer hook 一一对应：
    torch 图在 `cgc::moe_expert_stream` 处切分（piecewise），
    C++ 静态图在 topk 处触发专家预取——二者共享同一份"统一 IR"语义。
    """
    if not (_TORCH_AVAILABLE and _MAGI_API_AVAILABLE and magi_register_custom_op is not None):
        logger.debug("[MagiCompiler] skip moe op registration: torch/magi API unavailable")
        return None

    @magi_register_custom_op(
        name=MOE_EXPERT_STREAM_OP,
        is_compute_sensitive=True,
        is_subgraph_boundary=True,
    )
    def moe_expert_stream(
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """统一 IR：MoE 专家流式前向。

        - hidden_states: [n_embd, n_tokens]
        - expert_weights: 完整 3D 专家张量 [n_ff, n_embd, n_expert]（静态图只访问 topk 选中专家）
        - expert_ids:     [n_expert_used, n_tokens] topk 路由结果（来自 build_moe_ffn 的 selected_experts）
        - scale:          专家权重缩放
        返回路由后输出 [n_embd, n_tokens]。
        """
        sel = expert_ids.long()
        # 仅 gather topk 选中的专家列，实现"静态图只访问 topk 专家"的等价语义
        w = expert_weights.index_select(-1, sel.reshape(-1).unique())
        out = torch.matmul(w.transpose(0, 1), hidden_states.unsqueeze(-1)).squeeze(-1)
        return out * scale

    logger.info("[MagiCompiler] registered unified MoE expert-stream op: %s", MOE_EXPERT_STREAM_OP)
    return moe_expert_stream


class MagiCompiler:
    """
    MagiCompiler 整合层 (v1.1.0)

    接收 Agent 决策的编译策略，通过 `magi_compile` / `magi_register_custom_op`
    执行编译。对外保留旧版方法接口。
    """

    def __init__(self, model: Optional[Any] = None):
        self.model = model
        self.graph = None
        self.fusion_boundary: List[List[str]] = []
        self.tiling_config: Dict[str, int] = {}
        self.memory_hierarchy: Dict[str, str] = dict(_UNIFIED_IR_DEFAULT["memory_hierarchy"])
        self.scheduling_plan: Dict[str, Any] = dict(_UNIFIED_IR_DEFAULT["scheduling"])
        self.backend = "auto"
        self.op_hints: List[Any] = []
        self.model_tag: Optional[str] = None
        self.dynamic_arg_dims: Optional[Dict[str, int]] = None
        self._compiled_model = None
        self.moe_ffn_patch = None

    # ------------------------------------------------------------------
    # 全计算图分析
    # ------------------------------------------------------------------
    def capture_full_graph(self, inputs: Optional[tuple] = None) -> Any:
        """
        捕获完整计算图。

        通过 GraphAnalyzer 分析模块结构（并可选地对输入做 trace），
        识别 MoE / Attention / VLM 等特征。若为 MoE 模型，自动把统一 IR
        的 splitting/compute-sensitive op 写入全局 CompileConfig。
        """
        if self.model is None or not _GRAPH_ANALYZER_AVAILABLE:
            logger.warning("[MagiCompiler] no model set / GraphAnalyzer unavailable; graph analysis skipped")
            return None
        logger.info("[MagiCompiler] capturing full graph...")
        self.graph = GraphAnalyzer.analyze(self.model, inputs)
        n_ops = getattr(self.graph, "ops", [])
        logger.info("[MagiCompiler] graph captured: %s ops", len(n_ops) if n_ops else "n/a")

        if getattr(self.graph, "has_moe", False) and get_compile_config is not None:
            cfg = get_compile_config()
            if MOE_EXPERT_STREAM_OP not in cfg.splitting_ops:
                cfg.splitting_ops.append(MOE_EXPERT_STREAM_OP)
            if MOE_EXPERT_STREAM_OP not in cfg.recompute_config.custom_compute_sensitive_ops:
                cfg.recompute_config.custom_compute_sensitive_ops.append(MOE_EXPERT_STREAM_OP)
            logger.info("[MagiCompiler] MoE detected -> unified IR ops injected into CompileConfig")
        return self.graph

    # ------------------------------------------------------------------
    # 策略注入（Agent 驱动，兼容旧接口）
    # ------------------------------------------------------------------
    def set_fusion_boundary(self, boundary: List[List[str]]):
        logger.info("[MagiCompiler] setting fusion boundary: %s", boundary)
        self.fusion_boundary = boundary

    def set_tiling_config(self, cfg: Dict[str, int]):
        logger.info("[MagiCompiler] setting tiling config: %s", cfg)
        self.tiling_config = cfg

    def set_memory_hierarchy(self, hierarchy: Dict[str, str]):
        logger.info("[MagiCompiler] setting memory hierarchy: %s", hierarchy)
        self.memory_hierarchy.update(hierarchy)

    def set_scheduling_plan(self, plan: Dict[str, Any]):
        logger.info("[MagiCompiler] setting scheduling plan: %s", plan)
        self.scheduling_plan.update(plan)

    def set_backend(self, backend: str):
        logger.info("[MagiCompiler] setting backend: %s", backend)
        self.backend = backend

    def apply_op_hint(self, hint: Any):
        logger.info("[MagiCompiler] applying op hint: %s", hint)
        self.op_hints.append(hint)

    # ------------------------------------------------------------------
    # MoE 专家流式配置（统一 IR 下发到 C++ expert-streamer）
    # ------------------------------------------------------------------
    def enable_moe_expert_streaming(
        self,
        enabled: bool = True,
        slots: int = 16,
        hot_pool: int = 8,
        mtp_tokens: int = 4,
        pd_split: bool = False,
        prefill_gpu: int = 0,
        decode_gpu: int = 1,
        prefetch_threads: int = 4,
    ):
        """把统一 IR 的专家流式配置写入环境变量，供 C++ expert-streamer 读取。"""
        os.environ[EX_STREAM_ENV["enable"]] = "1" if enabled else "0"
        os.environ[EX_STREAM_ENV["slots"]] = str(slots)
        os.environ[EX_STREAM_ENV["hot_pool"]] = str(hot_pool)
        os.environ[EX_STREAM_ENV["mtp"]] = str(mtp_tokens)
        os.environ[EX_STREAM_ENV["pd"]] = "1" if pd_split else "0"
        os.environ[EX_STREAM_ENV["prefill_gpu"]] = str(prefill_gpu)
        os.environ[EX_STREAM_ENV["decode_gpu"]] = str(decode_gpu)
        os.environ[EX_STREAM_ENV["prefetch_threads"]] = str(prefetch_threads)
        self.scheduling_plan.update(
            {"pd_split": pd_split, "prefill_gpu": prefill_gpu, "decode_gpu": decode_gpu,
             "prefetch_threads": prefetch_threads, "slots": slots, "hot_pool": hot_pool}
        )
        logger.info(
            "[MagiCompiler] expert streaming IR -> env: enabled=%s slots=%d hot_pool=%d "
            "mtp=%d pd_split=%s p_gpu=%d d_gpu=%d",
            enabled, slots, hot_pool, mtp_tokens, pd_split, prefill_gpu, decode_gpu,
        )

    # ------------------------------------------------------------------
    # 方案 2：统一 IR monkey patch 接入 build_moe_ffn 接入点
    # ------------------------------------------------------------------
    def enable_moe_ffn_patch(
        self,
        gguf_path: str,
        config: Optional[Any] = None,
        backend: Optional[Any] = None,
        backend_kind: str = "auto",
    ) -> Tuple[Optional[Any], Optional[Any], bool]:
        """
        方案 2：全计算图分析 -> 统一 IR -> monkey patch 接入所有后端。

        接入点（gemma4.cpp build_moe_ffn）：
        - llama.cpp 静态图把"完整 3D 专家张量"（ffn_*_exps）传入 build_moe_ffn，
          但 mmid 只访问 topk 选中的专家列 —— 这正是 expert streaming 的切入口；
        - 这里用 MagiCompiler 做全计算图分析：把 GGUF 每个 MoE 层还原为
          MoeFfnGraph 中的一个 build_moe_ffn 接入点（统一 IR）；
        - 统一 IR 下发到 C++ expert-streamer（cgc_stream.dll），并通过
          monkey patch 包装 llama-cpp-python / vllm / mlx 后端推理入口，
          在 prefill/decode 阶段并行预取/流式加载 topk 专家。

        Args:
            gguf_path:    MoE GGUF 模型路径
            config:       MoeFfnStreamConfig（统一 IR 调度配置；默认读环境变量）
            backend:      要接入的后端对象（None 时仅做图分析）
            backend_kind: "auto" | "llama_cpp" | "vllm" | "mlx"

        Returns:
            (graph, patch_driver, applied)
        """
        from .moe_ffn_unified_patch import enable_moe_ffn_unified_patch

        graph, patch, applied = enable_moe_ffn_unified_patch(
            gguf_path, config=config, backend=backend, backend_kind=backend_kind,
        )
        if graph is not None:
            self.graph = graph
        self.moe_ffn_patch = patch
        logger.info(
            "[MagiCompiler] unified IR monkey patch (build_moe_ffn): "
            "graph=%s applied=%s",
            graph is not None, applied,
        )
        return graph, patch, applied

    # ------------------------------------------------------------------
    # 编译执行（新 API）
    # ------------------------------------------------------------------
    def compile(self) -> "MagiCompiledModel":
        """
        编译模型。

        torch 可用：注册统一 MoE op，并用 `magi_compile`（imperative 风格）
        编译模型，采用 MAGI_COMPILE / piecewise 后端。
        无 torch：回退为 eager 包装（保留 MagiCompiledModel 契约）。
        """
        logger.info("\n" + "=" * 60)
        logger.info("MagiCompiler compile (v1.1.0 API)")
        logger.info("=" * 60)
        logger.info("backend=%s fusion_boundary=%s tiling=%s", self.backend, self.fusion_boundary, self.tiling_config)
        logger.info("memory_hierarchy=%s scheduling=%s", self.memory_hierarchy, self.scheduling_plan)

        if not (_TORCH_AVAILABLE and _MAGI_API_AVAILABLE and magi_compile is not None):
            logger.warning("[MagiCompiler] torch/magi API unavailable -> eager fallback wrapper")
            self._compiled_model = MagiCompiledModel(self.model, self)
            return self._compiled_model

        register_moe_expert_op()

        compiled_model = self.model
        if self.model is not None:
            model_tag = self.model_tag or getattr(self.model, "__class__", type(None)).__name__
            try:
                compiled_model = magi_compile(
                    self.model,
                    model_tag=model_tag,
                    dynamic_arg_dims=self.dynamic_arg_dims,
                    config_patch=self._config_patch,
                )
            except Exception as e:  # 编译失败不阻断推理，回退 eager
                logger.warning("[MagiCompiler] magi_compile failed (%s) -> eager fallback", e)
                compiled_model = self.model

        self._compiled_model = MagiCompiledModel(compiled_model, self)
        logger.info("[MagiCompiler] compile done")
        logger.info("=" * 60)
        return self._compiled_model

    def _config_patch(self, cfg) -> Any:
        """把 Agent 策略 / 统一 IR 映射到 CompileConfig 字段。"""
        if cfg is None:
            return cfg
        # splitting: 统一 IR 的 MoE 专家流式算子作为 subgraph boundary
        for op in _UNIFIED_IR_DEFAULT["splitting_ops"]:
            if op not in cfg.splitting_ops:
                cfg.splitting_ops.append(op)
        for op in _UNIFIED_IR_DEFAULT["compute_sensitive_ops"]:
            if op not in cfg.recompute_config.custom_compute_sensitive_ops:
                cfg.recompute_config.custom_compute_sensitive_ops.append(op)
        # backend 映射
        backend = self.backend
        if backend and backend != "auto":
            if backend in ("cuda", "gpu", "metal", "cpu"):
                pass  # torch.compile 后端由设备自动选择，这里保持 inductor
        return cfg


class MagiCompiledModel:
    """
    编译后的模型。

    包装原始模型（或 magi_compile 编译产物），执行统一 IR 优化后的计算。
    """

    def __init__(self, raw_model, mgc: MagiCompiler):
        self.raw_model = raw_model
        self.mgc = mgc
        self.expert_stream_env: Dict[str, str] = _expert_stream_env()

    def __call__(self, *args, **kwargs):
        """
        执行编译后模型。

        Returns:
            模型输出
        """
        if not _TORCH_AVAILABLE or torch is None:
            raise RuntimeError("torch is required to run a compiled model")

        with torch.no_grad():
            return self.raw_model(*args, **kwargs)
