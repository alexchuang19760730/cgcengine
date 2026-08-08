# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
知识存储系统 - 增强版
包含：后端感知 + 硬件感知 + 图结构感知 + 模式感知
支持自动检测、智能匹配和端云一体切换
"""

import sqlite3
import json
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from cgc_engine.utils.envs import cgc_output_dir

@dataclass
class BackendKnowledge:
    """后端知识"""
    backend_id: str
    name: str
    type: str  # inference/training
    supported_ops: List[str]
    optimization_capabilities: List[str]
    hardware_requirements: Dict[str, Any]
    performance_profiles: Dict[str, float]
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareKnowledge:
    """硬件知识"""
    hardware_id: str
    device_type: str  # cpu/gpu/metal
    vendor: str
    model: str
    compute_capability: Optional[str]
    memory_gb: float
    unified_memory: bool
    supported_backends: List[str]
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPattern:
    """图模式知识"""
    pattern_id: str
    pattern_type: str  # attention/mlp/fusion/prefill/decode
    description: str
    node_patterns: List[Dict[str, Any]]
    optimizations: List[str]
    applicable_backends: List[str]
    performance_impact: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationCode:
    """优化代码存储"""
    code_id: str
    pattern_id: str
    backend_id: str
    hardware_id: str
    code_type: str  # kernel/strategy/pass/adapter
    code: str
    code_path: str  # 代码所在文件路径
    dependencies: List[str]
    performance_gains: Dict[str, float]
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeCloudStrategy:
    """端云一体策略"""
    strategy_id: str
    name: str
    description: str
    conditions: Dict[str, Any]  # 触发条件
    actions: List[Dict[str, Any]]  # 执行动作
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationStrategy:
    """优化策略"""
    strategy_id: str
    name: str
    strategy_type: str  # heuristic/pattern/edge_cloud/reference
    conditions: Dict[str, Any]  # 触发条件
    actions: List[Dict[str, Any]]  # 执行动作
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternMatchResult:
    """模式匹配结果"""
    pattern_id: str
    pattern_type: str
    confidence: float
    matched_nodes: List[int]
    suggested_optimizations: List[str]
    estimated_gain: float


@dataclass
class GraphAnalysisResult:
    """图结构分析结果"""
    nodes_count: int
    edges_count: int
    parallel_groups: List[List[int]]
    critical_path_length: int
    memory_footprint_estimate: float
    suggested_optimizations: List[str]


class KnowledgeStorage:
    """增强版知识存储系统 - 具备后端感知 + 硬件感知 + 图结构感知 + 模式感知"""
    
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.path.join(cgc_output_dir(), "knowledge.db")
        self._conn = sqlite3.connect(self.db_path)
        self._init_database(self._conn)
        self._load_enhanced_knowledge(self._conn)
        # 初始化分析引擎
        self.graph_analyzer = GraphStructureAnalyzer(self)
        self.pattern_detector = PatternRecognitionEngine(self)
    
    def close(self):
        if hasattr(self, '_conn'):
            self._conn.close()
    
    def _init_database(self, conn):
        """初始化数据库表"""
        cursor = conn.cursor()
        
        # 后端知识表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backend_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backend_id TEXT UNIQUE,
                name TEXT,
                type TEXT,
                supported_ops TEXT,
                optimization_capabilities TEXT,
                hardware_requirements TEXT,
                performance_profiles TEXT,
                version TEXT,
                metadata TEXT,
                created_at TEXT
            )
        ''')
        
        # 硬件知识表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hardware_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hardware_id TEXT UNIQUE,
                device_type TEXT,
                vendor TEXT,
                model TEXT,
                compute_capability TEXT,
                memory_gb REAL,
                unified_memory BOOLEAN,
                supported_backends TEXT,
                performance_metrics TEXT,
                metadata TEXT,
                created_at TEXT
            )
        ''')
        
        # 图模式知识表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT UNIQUE,
                pattern_type TEXT,
                description TEXT,
                node_patterns TEXT,
                optimizations TEXT,
                applicable_backends TEXT,
                performance_impact REAL,
                metadata TEXT,
                created_at TEXT
            )
        ''')
        
        # 优化代码表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_id TEXT UNIQUE,
                pattern_id TEXT,
                backend_id TEXT,
                hardware_id TEXT,
                code_type TEXT,
                code TEXT,
                code_path TEXT,  -- 代码所在文件路径
                dependencies TEXT,
                performance_gains TEXT,
                version TEXT,
                metadata TEXT,
                created_at TEXT
            )
        ''')
        
        # 端云策略表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edge_cloud_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT UNIQUE,
                name TEXT,
                description TEXT,
                conditions TEXT,
                actions TEXT,
                priority INTEGER,
                metadata TEXT,
                created_at TEXT
            )
        ''')
        
        # 优化策略表（包含六大策略）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT UNIQUE,
                name TEXT,
                strategy_type TEXT,
                conditions TEXT,
                actions TEXT,
                priority INTEGER,
                metadata TEXT,
                created_at TEXT
            )
        ''')
        
        # 索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_backend_id ON backend_knowledge(backend_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hardware_id ON hardware_knowledge(hardware_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_id ON graph_patterns(pattern_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_code_backend ON optimization_codes(backend_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_priority ON edge_cloud_strategies(priority)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opt_strategy_type ON optimization_strategies(strategy_type)')
        
        conn.commit()
    
    def _load_enhanced_knowledge(self, conn):
        """加载增强版默认知识"""
        # 检查是否已有知识
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM backend_knowledge')
        count = cursor.fetchone()[0]
        
        if count > 0:
            return  # 已有知识，跳过加载
        
        # ========== 后端知识 ==========
        # vLLM
        self.save_backend_knowledge(BackendKnowledge(
            backend_id="vllm",
            name="vLLM",
            type="inference",
            supported_ops=["scaled_dot_product_attention", "linear", "layer_norm", "mlp", "rope"],
            optimization_capabilities=["flash_attention", "paged_attention", "continuous_batching", 
                                      "cuda_graph", "tensor_parallel", "pipeline_parallel"],
            hardware_requirements={"min_gpu_memory_gb": 16, "cuda_version": ">=11.8"},
            performance_profiles={"throughput": 2000, "latency": 15, "memory_efficiency": 0.95},
            version="0.5.0",
            metadata={"author": "UC Berkeley", "license": "Apache 2.0", "scaling": "horizontal"}
        ), conn)
        
        # llama.cpp
        self.save_backend_knowledge(BackendKnowledge(
            backend_id="llama.cpp",
            name="llama.cpp",
            type="inference",
            supported_ops=["matmul", "attention", "layer_norm", "mlp"],
            optimization_capabilities=["ggml", "quantization", "cpu_offloading", "metal"],
            hardware_requirements={"min_memory_gb": 8},
            performance_profiles={"throughput": 100, "latency": 50, "memory_efficiency": 0.85},
            version="0.2.0",
            metadata={"author": "Georgi Gerganov", "license": "MIT", "scaling": "edge"}
        ), conn)
        
        # MLX
        self.save_backend_knowledge(BackendKnowledge(
            backend_id="mlx",
            name="MLX",
            type="inference",
            supported_ops=["matmul", "attention", "layer_norm", "mlp", "rope"],
            optimization_capabilities=["metal", "unified_memory", "flash_attention"],
            hardware_requirements={"device_type": "apple_metal"},
            performance_profiles={"throughput": 200, "latency": 30, "memory_efficiency": 0.90},
            version="0.13.0",
            metadata={"author": "Apple", "license": "MIT", "scaling": "edge"}
        ), conn)
        
        # mlx-tune（MLX调优版本 - 第四后端）
        self.save_backend_knowledge(BackendKnowledge(
            backend_id="mlx-tune",
            name="mlx-tune",
            type="inference",
            supported_ops=["matmul", "attention", "layer_norm", "mlp", "rope", "lora"],
            optimization_capabilities=["metal", "unified_memory", "flash_attention", "lora", "mpsgraph"],
            hardware_requirements={"device_type": "apple_metal"},
            performance_profiles={"throughput": 300, "latency": 25, "memory_efficiency": 0.93},
            version="0.13.0",
            metadata={"author": "Apple", "license": "MIT", "scaling": "edge", "custom_backend": True}
        ), conn)
        
        # MegaTrain（云端训练 - 第五后端）
        self.save_backend_knowledge(BackendKnowledge(
            backend_id="megatrain",
            name="MegaTrain",
            type="training",
            supported_ops=["matmul", "attention", "layer_norm", "mlp", "rope", "loss"],
            optimization_capabilities=["tensor_parallel", "pipeline_parallel", "zero", "flash_attention"],
            hardware_requirements={"min_gpu_memory_gb": 80, "cuda_version": ">=12.0", "num_gpus": ">=8"},
            performance_profiles={"throughput": 5000, "latency": 100, "memory_efficiency": 0.96},
            version="2026.4",
            metadata={"author": "SandAI", "license": "Proprietary", "scaling": "vertical"}
        ), conn)
        
        # ========== 硬件知识 ==========
        # NVIDIA RTX 5090（支持五大后端中的 CUDA、vLLM、Native）
        self.save_hardware_knowledge(HardwareKnowledge(
            hardware_id="nvidia-rtx-5090",
            device_type="gpu",
            vendor="NVIDIA",
            model="RTX 5090",
            compute_capability="8.9",
            memory_gb=32.0,
            unified_memory=False,
            supported_backends=["vllm", "cuda", "native", "tensorrt-llm"],
            performance_metrics={"tflops_fp16": 100, "memory_bandwidth_gbs": 800, 
                                "ncc_latency_us": 10, "max_batch_size": 128},
            metadata={"release_year": 2025, "power_watts": 575}
        ), conn)
        
        # Apple M4 Pro（支持五大后端中的 MLX、llama.cpp、Native）
        self.save_hardware_knowledge(HardwareKnowledge(
            hardware_id="apple-m4-pro",
            device_type="metal",
            vendor="Apple",
            model="M4 Pro",
            compute_capability=None,
            memory_gb=36.0,
            unified_memory=True,
            supported_backends=["mlx", "llama.cpp", "native", "metal"],
            performance_metrics={"tflops_fp16": 40, "memory_bandwidth_gbs": 400,
                                "unified_memory_gb": 36, "max_batch_size": 32},
            metadata={"release_year": 2024, "cpu_cores": 12}
        ), conn)
        
        # Apple M4 Ultra（支持五大后端中的 MLX、llama.cpp、Native、Multi-GPU）
        self.save_hardware_knowledge(HardwareKnowledge(
            hardware_id="apple-m4-ultra",
            device_type="metal",
            vendor="Apple",
            model="M4 Ultra",
            compute_capability=None,
            memory_gb=192.0,
            unified_memory=True,
            supported_backends=["mlx", "llama.cpp", "native", "metal"],
            performance_metrics={"tflops_fp16": 80, "memory_bandwidth_gbs": 800,
                                "unified_memory_gb": 192, "max_batch_size": 64, "gpu_count": 2},
            metadata={"release_year": 2024, "cpu_cores": 24}
        ), conn)
        
        # Intel Xeon Platinum（支持五大后端中的 llama.cpp、Native）
        self.save_hardware_knowledge(HardwareKnowledge(
            hardware_id="intel-xeon-platinum",
            device_type="cpu",
            vendor="Intel",
            model="Xeon Platinum 8592+",
            compute_capability=None,
            memory_gb=512.0,
            unified_memory=False,
            supported_backends=["llama.cpp", "native"],
            performance_metrics={"tflops_fp32": 5, "memory_bandwidth_gbs": 80,
                                "cpu_cores": 64, "max_batch_size": 16},
            metadata={"release_year": 2024, "socket": "LGA4677"}
        ), conn)
        
        # ========== 图模式知识 ==========
        # Flash Attention（支持五大后端）
        self.save_graph_pattern(GraphPattern(
            pattern_id="attention-flash",
            pattern_type="attention",
            description="Flash Attention 模式 - 高效内存优化注意力计算",
            node_patterns=[
                {"op_type": "scaled_dot_product_attention", "input_count": 3},
                {"op_type": "dropout", "input_from": "attention"},
                {"op_type": "linear", "input_from": "dropout"}
            ],
            optimizations=["flash_attention_v2", "flash_attention_v3", "paged_attention"],
            applicable_backends=["vllm", "mlx", "cuda", "native"],
            performance_impact=0.75,
            metadata={"category": "attention", "complexity": "O(n^2)"}
        ), conn)
        
        # MLP 融合（支持五大后端）
        self.save_graph_pattern(GraphPattern(
            pattern_id="mlp-fusion",
            pattern_type="mlp",
            description="MLP 融合模式 - 合并多层感知机计算",
            node_patterns=[
                {"op_type": "linear", "output_shape": "hidden"},
                {"op_type": "gelu", "input_from": "linear"},
                {"op_type": "linear", "input_from": "gelu"}
            ],
            optimizations=["mlp_fusion", "activation_fusion", "swiglu"],
            applicable_backends=["vllm", "mlx", "cuda", "native"],
            performance_impact=0.30,
            metadata={"category": "mlp", "complexity": "O(n)"}
        ), conn)
        
        # Prefill 阶段（支持五大后端中的 vLLM、CUDA、Native）
        self.save_graph_pattern(GraphPattern(
            pattern_id="prefill-cudagraph",
            pattern_type="prefill",
            description="Prefill 阶段 CUDA Graph 优化 - 消除调度开销",
            node_patterns=[
                {"op_type": "embedding", "input_type": "tokens"},
                {"op_type": "transformer_layers", "layer_count": "N"},
                {"op_type": "lm_head", "output_type": "logits"}
            ],
            optimizations=["cuda_graph", "tensor_parallel", "nccl_allreduce", "torch_compile"],
            applicable_backends=["vllm", "cuda", "native"],
            performance_impact=0.50,
            metadata={"category": "stage", "phase": "prefill"}
        ), conn)
        
        # Decode 阶段（支持五大后端）
        self.save_graph_pattern(GraphPattern(
            pattern_id="decode-static",
            pattern_type="decode",
            description="Decode 阶段静态图优化 - 循环重放",
            node_patterns=[
                {"op_type": "embedding", "input_type": "token"},
                {"op_type": "transformer_layer", "layer_count": "1"},
                {"op_type": "lm_head", "output_type": "logit"}
            ],
            optimizations=["static_graph", "paged_attention", "speculative_decoding"],
            applicable_backends=["vllm", "mlx", "llama.cpp", "native"],
            performance_impact=0.40,
            metadata={"category": "stage", "phase": "decode"}
        ), conn)
        
        # PD 分离（支持五大后端中的 vLLM、CUDA、Native）
        self.save_graph_pattern(GraphPattern(
            pattern_id="pd-separation",
            pattern_type="pd",
            description="Prefill/Decode 分离调度 - 专用卡干专用事",
            node_patterns=[
                {"stage": "prefill", "device": "gpu0", "compute_intensive": True},
                {"stage": "decode", "device": "gpu1", "memory_intensive": True}
            ],
            optimizations=["pd_scheduler", "kv_cache_sharing", "device_isolation"],
            applicable_backends=["vllm", "cuda", "native"],
            performance_impact=0.25,
            metadata={"category": "scheduling", "requires_multigpu": True}
        ), conn)
        
        # 通用 Attention 模式（简化版，支持五大后端）
        self.save_graph_pattern(GraphPattern(
            pattern_id="attention-generic",
            pattern_type="attention",
            description="通用注意力模式 - 匹配任何注意力计算",
            node_patterns=[
                {"op_type": "scaled_dot_product_attention"},
                {"op_type": "attention"}
            ],
            optimizations=["flash_attention", "paged_attention", "flash_inference"],
            applicable_backends=["vllm", "mlx", "cuda", "llama.cpp", "native"],
            performance_impact=0.60,
            metadata={"category": "attention", "generic": True}
        ), conn)
        
        # 通用 MLP 模式（简化版，支持五大后端）
        self.save_graph_pattern(GraphPattern(
            pattern_id="mlp-generic",
            pattern_type="mlp",
            description="通用 MLP 模式 - 匹配线性层+激活函数组合",
            node_patterns=[
                {"op_type": "linear"},
                {"op_type": "gelu"},
                {"op_type": "relu"}
            ],
            optimizations=["mlp_fusion", "activation_fusion", "swiglu"],
            applicable_backends=["vllm", "mlx", "cuda", "llama.cpp", "native"],
            performance_impact=0.25,
            metadata={"category": "mlp", "generic": True}
        ), conn)
        
        # Transformer 层模式（支持五大后端）
        self.save_graph_pattern(GraphPattern(
            pattern_id="transformer-layer",
            pattern_type="transformer",
            description="Transformer 层模式 - 匹配 transformer 基本结构",
            node_patterns=[
                {"op_type": "layer_norm"},
                {"op_type": "scaled_dot_product_attention"},
                {"op_type": "linear"},
                {"op_type": "gelu"}
            ],
            optimizations=["layer_fusion", "flash_attention", "tensor_parallel"],
            applicable_backends=["vllm", "mlx", "cuda", "llama.cpp", "native"],
            performance_impact=0.45,
            metadata={"category": "transformer", "generic": True}
        ), conn)
        
        # ========== 优化代码 ==========
        # Flash Attention CUDA
        self.save_optimization_code(OptimizationCode(
            code_id="flash-attn-v2-cuda",
            pattern_id="attention-flash",
            backend_id="vllm",
            hardware_id="nvidia-rtx-5090",
            code_type="kernel",
            code="""
# Flash Attention V2 CUDA Implementation
def flash_attention(q, k, v, causal=True):
    from flash_attn import flash_attn_qkvpacked_func
    return flash_attn_qkvpacked_func(
        torch.cat([q, k, v], dim=-1), 
        causal=causal,
        return_attn_probs=False
    )
""",
            code_path="cgc_engine/cuda/flash_attention.py",
            dependencies=["flash_attn>=2.0"],
            performance_gains={"speedup": 3.5, "memory_savings": 0.5, "latency_reduction": 0.6},
            version="2.0",
            metadata={"source": "flash-attn", "precision": "fp16"}
        ), conn)
        
        # MLX Attention
        self.save_optimization_code(OptimizationCode(
            code_id="mlx-attention-metal",
            pattern_id="attention-flash",
            backend_id="mlx",
            hardware_id="apple-m4-pro",
            code_type="kernel",
            code="""
# MLX Attention Metal Implementation
import mlx.core as mx

def mlx_attention(q, k, v, causal=True):
    # MLX 原生支持 Flash Attention
    scores = (q @ k.transpose(1, 2)) / mx.sqrt(q.shape[-1])
    if causal:
        mask = mx.tril(mx.ones((scores.shape[1], scores.shape[2])))
        scores = scores * mask + (1 - mask) * float('-inf')
    attn = mx.softmax(scores, axis=-1)
    return attn @ v
""",
            code_path="cgc_engine/mlx/mlx_attention.py",
            dependencies=["mlx>=0.13"],
            performance_gains={"speedup": 2.0, "memory_savings": 0.3, "unified_memory": True},
            version="0.13",
            metadata={"source": "mlx", "precision": "bf16"}
        ), conn)
        
        # CUDA Graph Prefill
        self.save_optimization_code(OptimizationCode(
            code_id="cuda-graph-prefill",
            pattern_id="prefill-cudagraph",
            backend_id="cuda",
            hardware_id="nvidia-rtx-5090",
            code_type="strategy",
            code="""
# CUDA Graph Prefill Strategy
def build_prefill_graph(model, batch_size, seq_len):
    import torch
    from torch.cuda import graph
    
    # 创建 CUDA Graph
    g = graph.CUDAGraph()
    
    # 预热
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device='cuda')
    with torch.cuda.graph(g):
        # 捕获完整 Prefill 计算
        outputs = model(input_ids)
    
    def run_prefill(input_ids):
        # 一次 launch 执行完整流水线
        return g.replay()
    
    return run_prefill
""",
            code_path="cgc_engine/cuda/cuda_graph_engine.py",
            dependencies=["torch>=2.0"],
            performance_gains={"speedup": 1.5, "overhead_reduction": 0.8, "latency_ms": 50},
            version="1.0",
            metadata={"source": "pytorch", "feature": "cuda_graph"}
        ), conn)
        
        # PD 分离调度器
        self.save_optimization_code(OptimizationCode(
            code_id="pd-scheduler",
            pattern_id="pd-separation",
            backend_id="vllm",
            hardware_id="nvidia-rtx-5090",
            code_type="strategy",
            code="""
# PD 分离调度策略
class PDScheduler:
    def __init__(self, prefill_devices=[0], decode_devices=[1]):
        self.prefill_devices = prefill_devices
        self.decode_devices = decode_devices
        
    def schedule_prefill(self, requests):
        # Prefill 在专用设备上执行
        with torch.cuda.device(self.prefill_devices[0]):
            results = self._run_prefill(requests)
        return results
    
    def schedule_decode(self, requests):
        # Decode 在专用设备上执行
        with torch.cuda.device(self.decode_devices[0]):
            results = self._run_decode(requests)
        return results
    
    def _run_prefill(self, requests):
        # TP=2 分布式计算
        pass
    
    def _run_decode(self, requests):
        # 专用解码
        pass
""",
            code_path="cgc_engine/cgc/pd_scheduler.py",
            dependencies=["vllm>=0.5"],
            performance_gains={"speedup": 1.3, "utilization_improvement": 0.2, "throughput_gain": 0.5},
            version="1.0",
            metadata={"source": "vllm", "feature": "pd_separation"}
        ), conn)
        
        # ========== 端云策略 ==========
        self._save_edge_cloud_strategy(EdgeCloudStrategy(
            strategy_id="edge-only-low-latency",
            name="端侧优先（低延迟）",
            description="当请求量小、延迟敏感时，使用端侧推理",
            conditions={
                "requests_per_second": {"$lt": 30},  # 低QPS触发
                "model_size_gb": {"$lt": 10},
                "edge_hardware_available": True
            },
            actions=[
                {"action": "select_backend", "backend": "mlx"},
                {"action": "set_role", "role": "full_inference"},
                {"action": "optimization", "enable": ["metal", "unified_memory"]}
            ],
            priority=1,
            metadata={"scenario": "edge_only", "latency_target": "<100ms"}
        ), conn)
        
        # DFlash-DFlash 端云一体策略（最高优先级）
        self._save_edge_cloud_strategy(EdgeCloudStrategy(
            strategy_id="dflash-dflash-hybrid",
            name="DFlash-DFlash端云一体",
            description="双端DFlash配置：云端DFlash Prefill + 端侧DFlash Decode，最优性能",
            conditions={
                "min_requests_per_second": 30,
                "prefill_seq_len": {"$gt": 256},
                "decode_tokens": {"$gt": 32},
                "cloud_hardware_available": True,
                "edge_hardware_available": True,
                "cloud_has_dflash": True,
                "edge_has_dflash": True
            },
            actions=[
                {"action": "select_backend", "cloud_backend": "vllm", "edge_backend": "mlx"},
                {"action": "set_role", "cloud_role": "prefill", "edge_role": "decode"},
                {"action": "optimization", "cloud": ["dflash", "tp=2", "cuda_graph", "spdk"], "edge": ["dflash", "mtp=2", "mps_graph", "unified_memory"]},
                {"action": "kv_transfer", "protocol": "grpc", "compression": "lz4"}
            ],
            priority=1.5,
            metadata={"scenario": "dflash_hybrid", "latency_target": "<150ms", "throughput_target": ">100 tok/s"}
        ), conn)
        
        self._save_edge_cloud_strategy(EdgeCloudStrategy(
            strategy_id="cloud-prefill-edge-decode",
            name="云端Prefill+端侧Decode",
            description="大模型场景：云端做Prefill，端侧做Decode（非DFlash配置）",
            conditions={
                "min_requests_per_second": 50,
                "prefill_seq_len": {"$gt": 512},
                "decode_tokens": {"$gt": 64},
                "cloud_hardware_available": True,
                "edge_hardware_available": True,
                "cloud_has_dflash": {"$ne": True},
                "edge_has_dflash": {"$ne": True}
            },
            actions=[
                {"action": "select_backend", "cloud_backend": "vllm", "edge_backend": "mlx"},
                {"action": "set_role", "cloud_role": "prefill", "edge_role": "decode"},
                {"action": "optimization", "cloud": ["tp=2", "cuda_graph"], "edge": ["metal"]},
                {"action": "kv_transfer", "protocol": "grpc", "compression": "gzip"}
            ],
            priority=2,
            metadata={"scenario": "hybrid", "latency_target": "<200ms"}
        ), conn)
        
        self._save_edge_cloud_strategy(EdgeCloudStrategy(
            strategy_id="cloud-only-high-throughput",
            name="云端优先（高吞吐量）",
            description="高并发场景：全部在云端执行",
            conditions={
                "min_requests_per_second": 100,
                "model_size_gb": {"$gt": 20},
                "cloud_hardware_available": True
            },
            actions=[
                {"action": "select_backend", "backend": "vllm"},
                {"action": "set_role", "role": "full_inference"},
                {"action": "optimization", "enable": ["tp=2", "cuda_graph", "spdk"]},
                {"action": "scaling", "strategy": "horizontal", "max_workers": 8}
            ],
            priority=3,
            metadata={"scenario": "cloud_only", "throughput_target": ">1000 tok/s"}
        ), conn)
        
        self._save_edge_cloud_strategy(EdgeCloudStrategy(
            strategy_id="fallback-edge",
            name="端侧降级",
            description="云端不可用时，降级到端侧（非DFlash）",
            conditions={
                "cloud_hardware_available": False,
                "edge_hardware_available": True,
                "model_size_gb": {"$lt": 15},
                "edge_has_dflash": {"$ne": True}
            },
            actions=[
                {"action": "select_backend", "backend": "llama.cpp"},
                {"action": "optimization", "enable": ["quantization", "cpu_offloading"]},
                {"action": "notify", "message": "Cloud unavailable, falling back to edge"}
            ],
            priority=4,
            metadata={"scenario": "fallback", "reliability": "high"}
        ), conn)
        
        # 端侧DFlash策略（只有端侧设备时启用DFlash和MPSGraph）
        self._save_edge_cloud_strategy(EdgeCloudStrategy(
            strategy_id="edge-only-dflash",
            name="端侧DFlash（Metal）",
            description="只有端侧设备时，启用DFlash和MPSGraph优化",
            conditions={
                "cloud_hardware_available": False,
                "edge_hardware_available": True,
                "edge_has_dflash": True,
                "model_size_gb": {"$lt": 15}
            },
            actions=[
                {"action": "select_backend", "backend": "mlx"},
                {"action": "optimization",
                 "edge": ["dflash", "mtp=2", "mps_graph", "unified_memory"]},
                {"action": "notify", "message": "Edge-only mode with DFlash and MPSGraph"}
            ],
            priority=1,
            metadata={"scenario": "edge_dflash", "latency_target": "<100ms"}
        ), conn)
        
        # 端侧CUDA策略（Windows/Linux通用：本地PD + CUDA Graph + 计算并行，与SPDK解耦）
        self._save_edge_cloud_strategy(EdgeCloudStrategy(
            strategy_id="edge-cuda-local-pd",
            name="端侧CUDA（本地PD）",
            description="端侧NVIDIA GPU场景：本地PD + CUDA Graph + 计算并行，与SPDK解耦，适用于Windows/Linux",
            conditions={
                "cloud_hardware_available": False,
                "edge_hardware_available": True,
                "edge_has_cuda": True,
                "model_size_gb": {"$lt": 20},
                "enable_local_pd": True
            },
            actions=[
                {"action": "select_backend", "backend": "cgc"},
                {"action": "set_role", "role": "full_inference"},
                {"action": "optimization", "enable": ["cuda_graph", "tensor_parallel", "local_pd", "kda"]},
                {"action": "set_pd_config", "mode": "local", "kv_cache": "enabled"},
                {"action": "notify", "message": "Edge-only CUDA mode: Local PD + CUDA Graph + Tensor Parallel"}
            ],
            priority=1,
            metadata={"scenario": "edge_cuda_local_pd", "latency_target": "<120ms", "platform": ["windows", "linux"]}
        ), conn)
        
        # ========== 优化策略（六大策略） ==========
        self._save_optimization_strategy(OptimizationStrategy(
            strategy_id="heuristic-default",
            name="启发式默认策略",
            strategy_type="heuristic",
            conditions={
                "device_type": {"$in": ["cuda", "metal", "cpu"]},
                "optimization_space_available": True
            },
            actions=[
                {"action": "set_backend", "backend": "auto"},
                {"action": "set_tile_sizes", "cuda": {"M": 128, "N": 128, "K": 128}, "default": {"M": 64, "N": 64, "K": 64}},
                {"action": "set_schedules", "prefetch": 2, "unroll": 2, "pipeline": 2}
            ],
            priority=1,
            metadata={"source": "heuristic", "category": "default"}
        ), conn)
        
        self._save_optimization_strategy(OptimizationStrategy(
            strategy_id="heuristic-flash-attention",
            name="Flash Attention 启发式",
            strategy_type="heuristic",
            conditions={
                "has_flash_attention": True,
                "device_type": {"$in": ["cuda", "metal"]}
            },
            actions=[
                {"action": "enable_op_fusion", "value": True},
                {"action": "set_attention_config", "flash_attention": True, "causal": True},
                {"action": "add_fusion_region", "ops": ["q_proj", "k_proj", "v_proj", "rope", "sdpa"]},
                {"action": "add_op_hint", "hint": "FLASH_ATTENTION"}
            ],
            priority=2,
            metadata={"source": "heuristic", "category": "attention"}
        ), conn)
        
        self._save_optimization_strategy(OptimizationStrategy(
            strategy_id="heuristic-moe",
            name="MoE 启发式策略",
            strategy_type="heuristic",
            conditions={
                "has_moe": True
            },
            actions=[
                {"action": "set_moe_config", "num_experts": 8, "top_k": 2, "routing_impl": "cgc_routing"},
                {"action": "add_fusion_region", "ops": ["moe_gate", "expert_forward", "moe_aggregate"]},
                {"action": "add_op_hint", "hint": "MOE_ROUTING"}
            ],
            priority=3,
            metadata={"source": "heuristic", "category": "moe"}
        ), conn)
        
        self._save_optimization_strategy(OptimizationStrategy(
            strategy_id="heuristic-tensor-parallel",
            name="张量并行启发式",
            strategy_type="heuristic",
            conditions={
                "has_tensor_parallel": True,
                "num_devices": {"$gte": 2}
            },
            actions=[
                {"action": "set_tp_degree", "degree": 2},
                {"action": "add_op_hint", "hint": "TENSOR_PARALLEL"}
            ],
            priority=4,
            metadata={"source": "heuristic", "category": "parallel"}
        ), conn)
        
        self._save_optimization_strategy(OptimizationStrategy(
            strategy_id="heuristic-vlm",
            name="VLM 启发式策略",
            strategy_type="heuristic",
            conditions={
                "has_vlm": True
            },
            actions=[
                {"action": "set_vlm_config", "vision_encoder": True, "cross_attention": True},
                {"action": "add_op_hint", "hint": "VLM_CROSS_ATTENTION"}
            ],
            priority=5,
            metadata={"source": "heuristic", "category": "vlm"}
        ), conn)
        
        self._save_optimization_strategy(OptimizationStrategy(
            strategy_id="reference-llama-cpp",
            name="llama.cpp 参考策略",
            strategy_type="reference",
            conditions={
                "enable_llama_cpp_reference": True,
                "backend": {"$in": ["metal", "cpu"]}
            },
            actions=[
                {"action": "apply_llama_cpp_optimizations"},
                {"action": "enable_quantization", "modes": ["q4_0", "q4_k", "q5_k"]},
                {"action": "enable_cpu_offloading", "value": True}
            ],
            priority=10,
            metadata={"source": "reference", "backend": "llama.cpp"}
        ), conn)
        
        self._save_optimization_strategy(OptimizationStrategy(
            strategy_id="reference-vllm",
            name="vLLM 参考策略",
            strategy_type="reference",
            conditions={
                "enable_vllm_reference": True,
                "backend": "cuda",
                "has_moe": True
            },
            actions=[
                {"action": "set_attention_config", "paged_attention": True},
                {"action": "enable_continuous_batching", "value": True}
            ],
            priority=11,
            metadata={"source": "reference", "backend": "vllm"}
        ), conn)
        
        print("✅ 增强版知识库已加载完成")
    
    # ... 原有方法保持不变 ...
    
    def save_backend_knowledge(self, knowledge: BackendKnowledge, conn=None):
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            close_conn = True
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO backend_knowledge (
                backend_id, name, type, supported_ops, optimization_capabilities,
                hardware_requirements, performance_profiles, version, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            knowledge.backend_id, knowledge.name, knowledge.type,
            json.dumps(knowledge.supported_ops),
            json.dumps(knowledge.optimization_capabilities),
            json.dumps(knowledge.hardware_requirements),
            json.dumps(knowledge.performance_profiles),
            knowledge.version,
            json.dumps(knowledge.metadata),
            datetime.now().isoformat()
        ))
        conn.commit()
        if close_conn:
            conn.close()
    
    def save_hardware_knowledge(self, knowledge: HardwareKnowledge, conn=None):
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            close_conn = True
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO hardware_knowledge (
                hardware_id, device_type, vendor, model, compute_capability,
                memory_gb, unified_memory, supported_backends, performance_metrics, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            knowledge.hardware_id, knowledge.device_type, knowledge.vendor,
            knowledge.model, knowledge.compute_capability, knowledge.memory_gb,
            knowledge.unified_memory, json.dumps(knowledge.supported_backends),
            json.dumps(knowledge.performance_metrics),
            json.dumps(knowledge.metadata),
            datetime.now().isoformat()
        ))
        conn.commit()
        if close_conn:
            conn.close()
    
    def save_graph_pattern(self, pattern: GraphPattern, conn=None):
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            close_conn = True
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO graph_patterns (
                pattern_id, pattern_type, description, node_patterns, optimizations,
                applicable_backends, performance_impact, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id, pattern.pattern_type, pattern.description,
            json.dumps(pattern.node_patterns),
            json.dumps(pattern.optimizations),
            json.dumps(pattern.applicable_backends),
            pattern.performance_impact,
            json.dumps(pattern.metadata),
            datetime.now().isoformat()
        ))
        conn.commit()
        if close_conn:
            conn.close()
    
    def save_optimization_code(self, code: OptimizationCode, conn=None):
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            close_conn = True
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO optimization_codes (
                code_id, pattern_id, backend_id, hardware_id, code_type, code,
                code_path, dependencies, performance_gains, version, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            code.code_id, code.pattern_id, code.backend_id, code.hardware_id,
            code.code_type, code.code, code.code_path,
            json.dumps(code.dependencies),
            json.dumps(code.performance_gains),
            code.version,
            json.dumps(code.metadata),
            datetime.now().isoformat()
        ))
        conn.commit()
        if close_conn:
            conn.close()
    
    def _save_edge_cloud_strategy(self, strategy: EdgeCloudStrategy, conn=None):
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            close_conn = True
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO edge_cloud_strategies (
                strategy_id, name, description, conditions, actions, priority, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy.strategy_id, strategy.name, strategy.description,
            json.dumps(strategy.conditions),
            json.dumps(strategy.actions),
            strategy.priority,
            json.dumps(strategy.metadata),
            datetime.now().isoformat()
        ))
        conn.commit()
        if close_conn:
            conn.close()
    
    def _save_optimization_strategy(self, strategy: OptimizationStrategy, conn=None):
        """保存优化策略"""
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            close_conn = True
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO optimization_strategies (
                strategy_id, name, strategy_type, conditions, actions, priority, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy.strategy_id, strategy.name, strategy.strategy_type,
            json.dumps(strategy.conditions),
            json.dumps(strategy.actions),
            strategy.priority,
            json.dumps(strategy.metadata),
            datetime.now().isoformat()
        ))
        conn.commit()
        if close_conn:
            conn.close()
    
    def get_optimization_strategies(self, strategy_type: Optional[str] = None) -> List[OptimizationStrategy]:
        """获取优化策略列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if strategy_type:
            cursor.execute('SELECT * FROM optimization_strategies WHERE strategy_type = ? ORDER BY priority ASC', (strategy_type,))
        else:
            cursor.execute('SELECT * FROM optimization_strategies ORDER BY priority ASC')
        
        rows = cursor.fetchall()
        strategies = []
        
        for row in rows:
            strategies.append(OptimizationStrategy(
                strategy_id=row[1],
                name=row[2],
                strategy_type=row[3],
                conditions=json.loads(row[4]),
                actions=json.loads(row[5]),
                priority=row[6],
                metadata=json.loads(row[7]) if row[7] else {}
            ))
        
        conn.close()
        return strategies
    
    def match_optimization_strategies(self, context: Dict[str, Any]) -> List[OptimizationStrategy]:
        """根据上下文匹配优化策略"""
        all_strategies = self.get_optimization_strategies()
        matched = []
        
        for strategy in all_strategies:
            if self._check_conditions(strategy.conditions, context):
                matched.append(strategy)
        
        return sorted(matched, key=lambda x: x.priority)
    
    # ========== 智能感知与匹配 ==========
    
    def detect_current_platform(self) -> Dict[str, Any]:
        """检测当前平台（后端感知 + 硬件感知）"""
        platform_info = {
            "os": sys.platform,
            "backend": None,
            "hardware": None,
            "num_devices": 0,
            "memory_gb": 0
        }
        
        # 检测 CUDA
        try:
            import torch
            if torch.cuda.is_available():
                platform_info["backend"] = "cuda"
                platform_info["num_devices"] = torch.cuda.device_count()
                platform_info["memory_gb"] = sum([
                    torch.cuda.get_device_properties(i).total_memory / 1e9 
                    for i in range(torch.cuda.device_count())
                ])
                platform_info["hardware"] = f"nvidia-{torch.cuda.get_device_name(0).lower().replace(' ', '-')}"
        except:
            pass
        
        # 检测 Metal（如果没有 CUDA）
        if platform_info["backend"] is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    platform_info["backend"] = "metal"
                    platform_info["num_devices"] = 1
                    # 获取 Metal 内存信息
                    platform_info["memory_gb"] = 32  # 默认值
                    platform_info["hardware"] = "apple-m4-pro"
            except:
                pass
        
        # 检测 CPU
        if platform_info["backend"] is None:
            platform_info["backend"] = "cpu"
            platform_info["num_devices"] = 1
            platform_info["memory_gb"] = 8
        
        return platform_info
    
    def find_matching_strategy(self, request_info: Dict[str, Any]) -> Optional[EdgeCloudStrategy]:
        """根据请求信息找到最佳端云策略"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM edge_cloud_strategies ORDER BY priority ASC')
        rows = cursor.fetchall()
        
        for row in rows:
            conditions = json.loads(row[4])
            if self._check_conditions(conditions, request_info):
                return EdgeCloudStrategy(
                    strategy_id=row[1],
                    name=row[2],
                    description=row[3],
                    conditions=conditions,
                    actions=json.loads(row[5]),
                    priority=row[6],
                    metadata=json.loads(row[7]) if row[7] else {}
                )
        
        conn.close()
        return None
    
    def _check_conditions(self, conditions: Dict[str, Any], request_info: Dict[str, Any]) -> bool:
        """检查条件是否满足"""
        for key, value in conditions.items():
            if key not in request_info:
                continue

            req_value = request_info[key]

            if isinstance(value, dict):
                # 处理比较操作
                for op, threshold in value.items():
                    if op == "$lt" and req_value >= threshold:
                        return False
                    elif op == "$gt" and req_value <= threshold:
                        return False
                    elif op == "$eq" and req_value != threshold:
                        return False
                    elif op == "$ne" and req_value == threshold:
                        return False
            else:
                if req_value != value:
                    return False

        return True
    
    def find_optimal_backend(self, hardware_id: str, task_type: str = "inference") -> Optional[BackendKnowledge]:
        """为硬件找到最优后端"""
        hardware = self.get_hardware_knowledge(hardware_id)
        if not hardware:
            return None
        
        best_backend = None
        best_score = 0
        
        for backend_id in hardware.supported_backends:
            backend = self.get_backend_knowledge(backend_id)
            if backend and backend.type == task_type:
                # 简单评分：性能 * 内存效率
                score = backend.performance_profiles.get("throughput", 0) * \
                        backend.performance_profiles.get("memory_efficiency", 0)
                if score > best_score:
                    best_score = score
                    best_backend = backend
        
        return best_backend
    
    def generate_optimization_code(self, pattern_id: str, backend_id: str, hardware_id: str) -> Optional[str]:
        """生成最优优化代码"""
        code = self.find_optimal_code(pattern_id, backend_id, hardware_id)
        if code:
            return code.code
        return None
    
    def get_all_patterns(self) -> List[GraphPattern]:
        """获取所有图模式"""
        cursor = self._conn.cursor()
        cursor.execute('SELECT * FROM graph_patterns')
        rows = cursor.fetchall()
        
        patterns = []
        for row in rows:
            patterns.append(GraphPattern(
                pattern_id=row[1],
                pattern_type=row[2],
                description=row[3],
                node_patterns=json.loads(row[4]),
                optimizations=json.loads(row[5]),
                applicable_backends=json.loads(row[6]),
                performance_impact=row[7],
                metadata=json.loads(row[8]) if row[8] else {}
            ))
        
        return patterns
    
    def get_backend_knowledge(self, backend_id: str) -> Optional[BackendKnowledge]:
        cursor = self._conn.cursor()
        cursor.execute('SELECT * FROM backend_knowledge WHERE backend_id = ?', (backend_id,))
        row = cursor.fetchone()
        
        if row:
            return BackendKnowledge(
                backend_id=row[1],
                name=row[2],
                type=row[3],
                supported_ops=json.loads(row[4]),
                optimization_capabilities=json.loads(row[5]),
                hardware_requirements=json.loads(row[6]),
                performance_profiles=json.loads(row[7]),
                version=row[8],
                metadata=json.loads(row[9]) if row[9] else {}
            )
        
        return None
    
    def get_hardware_knowledge(self, hardware_id: str) -> Optional[HardwareKnowledge]:
        cursor = self._conn.cursor()
        cursor.execute('SELECT * FROM hardware_knowledge WHERE hardware_id = ?', (hardware_id,))
        row = cursor.fetchone()
        
        if row:
            return HardwareKnowledge(
                hardware_id=row[1],
                device_type=row[2],
                vendor=row[3],
                model=row[4],
                compute_capability=row[5],
                memory_gb=row[6],
                unified_memory=bool(row[7]),
                supported_backends=json.loads(row[8]),
                performance_metrics=json.loads(row[9]),
                metadata=json.loads(row[10]) if row[10] else {}
            )
        
        conn.close()
        return None
    
    def find_optimal_code(self, pattern_id: str, backend_id: str, hardware_id: str) -> Optional[OptimizationCode]:
        cursor = self._conn.cursor()
        cursor.execute('''
            SELECT * FROM optimization_codes 
            WHERE pattern_id = ? AND backend_id = ? AND hardware_id = ?
            ORDER BY json_extract(performance_gains, '$.speedup') DESC
            LIMIT 1
        ''', (pattern_id, backend_id, hardware_id))
        
        row = cursor.fetchone()
        
        if row:
            return OptimizationCode(
                code_id=row[1],
                pattern_id=row[2],
                backend_id=row[3],
                hardware_id=row[4],
                code_type=row[5],
                code=row[6],
                code_path=row[7],
                dependencies=json.loads(row[8]),
                performance_gains=json.loads(row[9]),
                version=row[10],
                metadata=json.loads(row[11]) if row[11] else {}
            )
        
        return None
    
    def export_knowledge(self, output_file: str):
        """导出所有知识到 JSON"""
        cursor = self._conn.cursor()
        
        cursor.execute('SELECT * FROM backend_knowledge')
        backends = []
        for row in cursor.fetchall():
            backends.append({
                'backend_id': row[1], 'name': row[2], 'type': row[3],
                'supported_ops': json.loads(row[4]),
                'optimization_capabilities': json.loads(row[5]),
                'hardware_requirements': json.loads(row[6]),
                'performance_profiles': json.loads(row[7]),
                'version': row[8], 'metadata': json.loads(row[9]) if row[9] else {}
            })
        
        cursor.execute('SELECT * FROM hardware_knowledge')
        hardwares = []
        for row in cursor.fetchall():
            hardwares.append({
                'hardware_id': row[1], 'device_type': row[2], 'vendor': row[3],
                'model': row[4], 'compute_capability': row[5],
                'memory_gb': row[6], 'unified_memory': bool(row[7]),
                'supported_backends': json.loads(row[8]),
                'performance_metrics': json.loads(row[9]),
                'metadata': json.loads(row[10]) if row[10] else {}
            })
        
        cursor.execute('SELECT * FROM graph_patterns')
        patterns = []
        for row in cursor.fetchall():
            patterns.append({
                'pattern_id': row[1], 'pattern_type': row[2], 'description': row[3],
                'node_patterns': json.loads(row[4]),
                'optimizations': json.loads(row[5]),
                'applicable_backends': json.loads(row[6]),
                'performance_impact': row[7],
                'metadata': json.loads(row[8]) if row[8] else {}
            })
        
        cursor.execute('SELECT * FROM optimization_codes')
        codes = []
        for row in cursor.fetchall():
            codes.append({
                'code_id': row[1], 'pattern_id': row[2], 'backend_id': row[3],
                'hardware_id': row[4], 'code_type': row[5], 'code': row[6],
                'code_path': row[7],
                'dependencies': json.loads(row[8]),
                'performance_gains': json.loads(row[9]),
                'version': row[10],
                'metadata': json.loads(row[11]) if row[11] else {}
            })
        
        cursor.execute('SELECT * FROM edge_cloud_strategies')
        strategies = []
        for row in cursor.fetchall():
            strategies.append({
                'strategy_id': row[1], 'name': row[2], 'description': row[3],
                'conditions': json.loads(row[4]),
                'actions': json.loads(row[5]),
                'priority': row[6],
                'metadata': json.loads(row[7]) if row[7] else {}
            })
        
        data = {
            'export_time': datetime.now().isoformat(),
            'backends': backends,
            'hardwares': hardwares,
            'patterns': patterns,
            'optimization_codes': codes,
            'edge_cloud_strategies': strategies
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ 知识已导出到: {output_file}")


# ========== 图结构感知引擎 ==========

class GraphStructureAnalyzer:
    """图结构分析器 - 分析计算图的节点、依赖、并行性"""
    
    def __init__(self, knowledge_storage: 'KnowledgeStorage'):
        self.knowledge = knowledge_storage
    
    def analyze_graph(self, graph_data: Dict[str, Any]) -> GraphAnalysisResult:
        """分析计算图结构"""
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # 基础统计
        nodes_count = len(nodes)
        edges_count = len(edges)
        
        # 构建依赖图
        dependency_graph = self._build_dependency_graph(nodes, edges)
        
        # 识别并行节点组
        parallel_groups = self._find_parallel_groups(dependency_graph)
        
        # 计算关键路径
        critical_path_length = self._compute_critical_path(dependency_graph)
        
        # 估算内存占用
        memory_footprint = self._estimate_memory_footprint(nodes)
        
        # 生成优化建议
        suggestions = self._generate_optimization_suggestions(nodes, parallel_groups)
        
        return GraphAnalysisResult(
            nodes_count=nodes_count,
            edges_count=edges_count,
            parallel_groups=parallel_groups,
            critical_path_length=critical_path_length,
            memory_footprint_estimate=memory_footprint,
            suggested_optimizations=suggestions
        )
    
    def _build_dependency_graph(self, nodes, edges):
        """构建依赖图"""
        graph = {node['id']: {'deps': [], 'users': [], 'op_type': node.get('op_type')} 
                 for node in nodes}
        
        for edge in edges:
            src = edge['from']
            dst = edge['to']
            if src in graph and dst in graph:
                graph[src]['users'].append(dst)
                graph[dst]['deps'].append(src)
        
        return graph
    
    def _find_parallel_groups(self, dependency_graph):
        """识别可并行执行的节点组"""
        groups = []
        visited = set()
        
        # 按拓扑顺序处理节点
        in_degree = {node_id: len(info['deps']) for node_id, info in dependency_graph.items()}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        
        while queue:
            # 当前层级的所有节点可以并行执行
            current_level = queue.copy()
            queue.clear()
            
            if current_level:
                groups.append(current_level)
            
            for node_id in current_level:
                visited.add(node_id)
                for user in dependency_graph[node_id]['users']:
                    in_degree[user] -= 1
                    if in_degree[user] == 0:
                        queue.append(user)
        
        return groups
    
    def _compute_critical_path(self, dependency_graph):
        """计算关键路径长度"""
        # 动态规划计算最长路径
        longest_path = {}
        
        def dfs(node_id):
            if node_id in longest_path:
                return longest_path[node_id]
            
            max_length = 1
            for user in dependency_graph[node_id]['users']:
                length = 1 + dfs(user)
                if length > max_length:
                    max_length = length
            
            longest_path[node_id] = max_length
            return max_length
        
        # 从所有源节点开始计算
        max_path = 0
        for node_id, info in dependency_graph.items():
            if not info['deps']:  # 源节点
                path_len = dfs(node_id)
                if path_len > max_path:
                    max_path = path_len
        
        return max_path
    
    def _estimate_memory_footprint(self, nodes):
        """估算内存占用（GB）"""
        total_bytes = 0
        
        for node in nodes:
            shape = node.get('output_shape', [])
            dtype = node.get('dtype', 'float32')
            
            if shape:
                size = 1
                for dim in shape:
                    size *= dim
                
                dtype_sizes = {
                    'float32': 4,
                    'float16': 2,
                    'bfloat16': 2,
                    'int64': 8,
                    'int32': 4
                }
                total_bytes += size * dtype_sizes.get(dtype, 4)
        
        return total_bytes / (1024 ** 3)  # 转换为 GB
    
    def _generate_optimization_suggestions(self, nodes, parallel_groups):
        """生成优化建议"""
        suggestions = []
        
        # 检查是否有大规模矩阵运算
        matmul_count = sum(1 for node in nodes if node.get('op_type') in ['matmul', 'linear'])
        if matmul_count > 5:
            suggestions.append('启用 TensorRT 或 cuBLAS 优化')
        
        # 检查是否有可融合的模式
        fusion_opp = sum(1 for node in nodes if node.get('op_type') in ['add', 'mul', 'relu', 'gelu'])
        if fusion_opp > 10:
            suggestions.append('算子融合优化')
        
        # 检查并行机会
        if len(parallel_groups) > 3 and max(len(g) for g in parallel_groups) > 4:
            suggestions.append('分布式并行（TP/PP）')
        
        # 检查内存压力
        mem_estimate = self._estimate_memory_footprint(nodes)
        if mem_estimate > 10:
            suggestions.append('启用 FP8/INT8 量化')
            suggestions.append('KV Cache 优化')
        
        return suggestions


# ========== 模式感知引擎 ==========

class PatternRecognitionEngine:
    """模式识别引擎 - 识别计算图中的优化模式"""
    
    def __init__(self, knowledge_storage: 'KnowledgeStorage'):
        self.knowledge = knowledge_storage
        self.patterns = knowledge_storage.get_all_patterns()
    
    def recognize_patterns(self, graph_data: Dict[str, Any]) -> List[PatternMatchResult]:
        """识别计算图中的模式"""
        results = []
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        for pattern in self.patterns:
            match = self._match_pattern(pattern, nodes, edges)
            if match and match.confidence > 0.3:  # 置信度阈值
                results.append(match)
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    def _match_pattern(self, pattern: GraphPattern, nodes: List[Dict], edges: List[Dict]) -> Optional[PatternMatchResult]:
        """匹配单个模式 - 支持多种匹配策略"""
        matched_nodes = []
        node_patterns = pattern.node_patterns
        
        # 统计匹配的模式数量
        matched_pattern_count = 0
        
        for node_pattern in node_patterns:
            matched = self._match_node_pattern(node_pattern, nodes, edges)
            if matched:
                matched_nodes.extend(matched)
                matched_pattern_count += 1
        
        if matched_pattern_count > 0:
            # 计算置信度：匹配的模式数 / 总模式数 * 匹配节点数的归一化
            base_confidence = matched_pattern_count / len(node_patterns)
            node_factor = min(len(set(matched_nodes)) / len(nodes), 0.5) + 0.5
            confidence = base_confidence * node_factor
            
            return PatternMatchResult(
                pattern_id=pattern.pattern_id,
                pattern_type=pattern.pattern_type,
                confidence=confidence,
                matched_nodes=list(set(matched_nodes)),
                suggested_optimizations=pattern.optimizations,
                estimated_gain=pattern.performance_impact * confidence
            )
        
        return None
    
    def _match_node_pattern(self, node_pattern: Dict, nodes: List[Dict], edges: List[Dict]) -> List[int]:
        """匹配节点模式 - 支持模糊匹配"""
        matched = []
        
        for node in nodes:
            score = 0
            total_checks = 0
            
            for key, value in node_pattern.items():
                total_checks += 1
                
                if key == 'input_from':
                    # 检查输入来源关系
                    if self._has_input_from(node['id'], value, nodes, edges):
                        score += 1
                elif key == 'output_shape':
                    # 模糊匹配输出形状
                    if self._match_shape(node.get('output_shape'), value):
                        score += 1
                elif key == 'op_type':
                    # 操作类型匹配（支持通配符）
                    node_op = node.get('op_type', '')
                    if isinstance(value, list):
                        if node_op in value:
                            score += 1
                    elif value in node_op or node_op in value:
                        score += 1
                else:
                    # 精确匹配其他属性
                    if node.get(key) == value:
                        score += 1
            
            # 如果匹配度超过50%则认为匹配
            if total_checks > 0 and score / total_checks >= 0.5:
                matched.append(node['id'])
        
        return matched
    
    def _has_input_from(self, node_id: int, source_op_type: str, nodes: List[Dict], edges: List[Dict]) -> bool:
        """检查节点是否有来自特定操作类型的输入"""
        for edge in edges:
            if edge['to'] == node_id:
                src_node = next((n for n in nodes if n['id'] == edge['from']), None)
                if src_node and src_node.get('op_type') == source_op_type:
                    return True
        return False
    
    def _match_shape(self, actual_shape, expected_shape) -> bool:
        """模糊匹配形状"""
        if not actual_shape or not expected_shape:
            return True
        
        if isinstance(expected_shape, dict):
            # 处理条件形状匹配
            for key, value in expected_shape.items():
                if key == '$gt' and actual_shape[0] <= value:
                    return False
                elif key == '$lt' and actual_shape[0] >= value:
                    return False
            return True
        
        # 简单形状匹配
        if isinstance(expected_shape, str):
            return True  # 描述性字符串
        
        return len(actual_shape) == len(expected_shape)


# ========== 自动生成接口 ==========

class AutoGenerationEngine:
    """自动生成引擎 - 根据感知结果生成优化代码"""
    
    def __init__(self, knowledge_storage: 'KnowledgeStorage'):
        self.knowledge = knowledge_storage
    
    def generate_optimized_code(self, 
                               platform_info: Dict[str, Any],
                               graph_analysis: GraphAnalysisResult,
                               pattern_matches: List[PatternMatchResult]) -> Dict[str, Any]:
        """生成优化代码"""
        backend_id = platform_info.get('backend', 'cpu')
        hardware_id = platform_info.get('hardware', 'cpu-generic')
        
        # 获取最优后端
        optimal_backend = self.knowledge.find_optimal_backend(hardware_id)
        if not optimal_backend:
            optimal_backend = self.knowledge.get_backend_knowledge(backend_id)
        
        # 收集需要优化的模式
        optimizations = []
        for match in pattern_matches:
            code = self.knowledge.generate_optimization_code(
                match.pattern_id, 
                optimal_backend.backend_id if optimal_backend else backend_id,
                hardware_id
            )
            if code:
                optimizations.append({
                    'pattern_id': match.pattern_id,
                    'pattern_type': match.pattern_type,
                    'confidence': match.confidence,
                    'estimated_gain': match.estimated_gain,
                    'code': code
                })
        
        # 添加分布式策略建议
        if len(graph_analysis.parallel_groups) > 2:
            if platform_info.get('num_devices', 1) > 1:
                optimizations.append({
                    'pattern_id': 'distributed-parallel',
                    'pattern_type': 'strategy',
                    'confidence': 1.0,
                    'estimated_gain': 0.5 * (platform_info['num_devices'] - 1),
                    'code': self._generate_distributed_code(platform_info)
                })
        
        return {
            'platform': platform_info,
            'optimal_backend': optimal_backend.name if optimal_backend else 'unknown',
            'graph_analysis': {
                'nodes_count': graph_analysis.nodes_count,
                'parallel_groups': len(graph_analysis.parallel_groups),
                'critical_path': graph_analysis.critical_path_length,
                'memory_estimate_gb': graph_analysis.memory_footprint_estimate
            },
            'optimizations': optimizations,
            'total_estimated_gain': sum(o['estimated_gain'] for o in optimizations)
        }
    
    def _generate_distributed_code(self, platform_info: Dict[str, Any]) -> str:
        """生成分布式并行代码"""
        num_devices = platform_info.get('num_devices', 1)
        backend = platform_info.get('backend', 'cuda')
        
        if backend == 'cuda' and num_devices > 1:
            return f"""
# 分布式并行配置（TP={num_devices}）
def setup_tensor_parallel(model, world_size={num_devices}):
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # 初始化分布式环境
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    
    # 应用张量并行
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    return model, local_rank
"""
        return ""


# ========== 完整的自动生成流程 ==========

def auto_generate(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    自动生成 = 后端感知 + 硬件感知 + 图结构感知 + 模式感知
    
    Args:
        graph_data: 计算图数据，包含 nodes 和 edges
    
    Returns:
        优化建议和代码
    """
    # 1. 初始化知识库
    knowledge = KnowledgeStorage()
    
    # 2. 后端感知 + 硬件感知
    platform_info = knowledge.detect_current_platform()
    
    # 3. 图结构感知
    graph_analysis = knowledge.graph_analyzer.analyze_graph(graph_data)
    
    # 4. 模式感知
    pattern_matches = knowledge.pattern_detector.recognize_patterns(graph_data)
    
    # 5. 生成优化代码
    generator = AutoGenerationEngine(knowledge)
    result = generator.generate_optimized_code(platform_info, graph_analysis, pattern_matches)
    
    # 6. 保存优化知识到知识库
    for opt in result['optimizations']:
        knowledge.save_optimization_code(OptimizationCode(
            code_id=f"auto-gen-{opt['pattern_id']}-{platform_info['hardware']}",
            pattern_id=opt['pattern_id'],
            backend_id=platform_info.get('backend', 'cpu'),
            hardware_id=platform_info.get('hardware', 'cpu-generic'),
            code_type='strategy',
            code=opt['code'],
            dependencies=[],
            performance_gains={'speedup': opt['estimated_gain']},
            version='auto-gen',
            metadata={'source': 'auto_generation', 'confidence': opt['confidence']}
        ))
    
    return result
