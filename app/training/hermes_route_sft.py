#!/usr/bin/env python3
"""Hermes Router SFT 数据合成 v4 — 用 4D矩阵 + 十步流水线数据微调 Hermes.

用 Hermes Router v4 的 PlatformBenchmark + FourDMatrix + TenStepPipeline
生成 SFT 训练对, 用于微调 Hermes 认知路由模型.

三阶段训练:
  阶段 1: SFT Bootstrap (本脚本) — 5K-10K 配对, CE loss, LoRA r=16
  阶段 2: DPO 自适应 (在线学习) — 滚动 1K-5K 配对/天
  阶段 3: 蒸馏小模型 (可选)

数据格式: JSONL, 每行一个对话:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

v4 变更 (vs v2):
  - 输入: FourDMatrix (D1网络+D2硬件+D3模型+D4决策) + PlatformBenchmark + 十步流水线摘要
  - 输出: 5 种新模式 (cache_hit, edge_draft, cloud_mtp, cloud_only, local_only)
  - 新增: edge_backend (mlx/llamacpp), cloud_compute_savings_pct
  - 新增: StateABI 状态转换作为特征
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import asdict
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from app.shared.hermes_router import (
    SystemProfile,
    PlatformBenchmark,
    PlatformBenchmarkResult,
    ProfileBinding,
    FourDMatrix,
    TenStepPipeline,
    HermesRouter,
    Bootstrap,
    StateABI,
    D1Network,
    D2Hardware,
    D3Model,
    D4Decision,
)


# === 硬件参数随机空间 (覆盖主流端侧平台) ===
HARDWARE_SPACE = [
    # (chip, total_mem_gb, avail_mem_gb, gpu_type, tflops, engine, rtt_ms, disk_gb)
    ("Apple M4", 16, 8, "apple_metal", 38, "mlx", 5, 100),
    ("Apple M4", 16, 10, "apple_metal", 38, "mlx", 15, 200),
    ("Apple M4", 16, 12, "apple_metal", 38, "mlx", 30, 300),
    ("Apple M4 Pro", 24, 16, "apple_metal", 52, "mlx", 5, 200),
    ("Apple M4 Pro", 24, 18, "apple_metal", 52, "mlx", 20, 400),
    ("Apple M4 Pro", 24, 20, "apple_metal", 52, "omlx", 10, 500),
    ("Apple M4 Max", 64, 48, "apple_metal", 75, "omlx", 5, 1000),
    ("Apple M4 Max", 64, 50, "apple_metal", 75, "omlx", 25, 2000),
    ("Apple M3 Pro", 18, 10, "apple_metal", 42, "mlx", 15, 150),
    ("Apple M2", 16, 8, "apple_metal", 30, "mlx", 40, 80),
    ("Intel i9-13900K", 64, 40, "nvidia", 80, "cuda", 10, 500),
    ("Intel i7-12700K", 32, 16, "nvidia", 60, "cuda", 20, 300),
    ("AMD Ryzen 9 7950X", 64, 48, "amd", 70, "rocm", 15, 600),
    ("Qualcomm Snapdragon X Elite", 32, 20, "qualcomm", 45, "cpu", 50, 200),
]

# === 平台后端性能基线 (用于 SFT 数据中的 PlatformBenchmark 特征) ===
PLATFORM_BENCHMARK_SPACE = [
    # (mlx_available, llamacpp_available, mlx_tps, llamacpp_tps, preferred)
    (True, True, 26.0, 149.8, "llamacpp"),    # M4 typical
    (True, True, 35.0, 180.0, "llamacpp"),    # M4 Pro
    (True, True, 60.0, 250.0, "llamacpp"),    # M4 Max
    (True, False, 26.0, 0.0, "mlx"),           # M4 no llama.cpp
    (True, True, 22.0, 130.0, "llamacpp"),    # M3
    (True, True, 18.0, 110.0, "llamacpp"),    # M2
    (False, True, 0.0, 200.0, "llamacpp"),     # Linux + CUDA no MLX
    (False, False, 0.0, 0.0, "none"),           # No local backend
]

# === Draft 模型空间 (不同架构, 决定能否在端侧跑) ===
DRAFT_MODEL_SPACE = [
    {
        "name": "gemma4",
        "display_name": "Gemma4-26B-A4B",
        "draft_architecture": "Gemma4AssistantForCausalLM",
        "draft_model_type": "gemma4_assistant",
        "draft_model_size_gb": 0.84,
        "draft_num_layers": 4,
        "draft_hidden_size": 1024,
        "draft_vocab_size": 262144,
        "draft_params_m": 183.1,
        "cloud_accept_rate": 0.95,
    },
    {
        "name": "dsv4",
        "display_name": "DeepSeek V4 Flash",
        "draft_architecture": "DeepseekV4ForCausalLM",
        "draft_model_type": "deepseek_v4",
        "draft_model_size_gb": 0.90,
        "draft_num_layers": 4,
        "draft_hidden_size": 4096,
        "draft_vocab_size": 129280,
        "draft_params_m": 188.8,
        "cloud_accept_rate": 0.85,
    },
    {
        "name": "qwen3vl",
        "display_name": "Qwen3-VL-2B",
        "draft_architecture": "MTPHead",
        "draft_model_type": "mtphead",
        "draft_model_size_gb": 0.30,
        "draft_num_layers": 4,
        "draft_hidden_size": 2048,
        "draft_vocab_size": 151936,
        "draft_params_m": 59.8,
        "cloud_accept_rate": 0.80,
    },
    {
        "name": "llama3",
        "display_name": "Llama3-8B",
        "draft_architecture": "LlamaForCausalLM",
        "draft_model_type": "llama",
        "draft_model_size_gb": 0.40,
        "draft_num_layers": 4,
        "draft_hidden_size": 4096,
        "draft_vocab_size": 128256,
        "draft_params_m": 80.0,
        "cloud_accept_rate": 0.82,
    },
    {
        "name": "qwen25",
        "display_name": "Qwen2.5-7B",
        "draft_architecture": "Qwen2ForCausalLM",
        "draft_model_type": "qwen2",
        "draft_model_size_gb": 0.35,
        "draft_num_layers": 4,
        "draft_hidden_size": 3584,
        "draft_vocab_size": 152064,
        "draft_params_m": 65.0,
        "cloud_accept_rate": 0.78,
    },
]

# === Prompt 模板 ===
PROMPT_TEMPLATES = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return ",
    "class User:\n    def __init__(self, name):\n        self.name = name\n\n    def ",
    "import asyncio\n\nasync def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        ",
    "const express = require('express');\nconst app = express();\n\napp.get('/', (req, res) => {\n    ",
    "function calculateTotal(items) {\n    let total = 0;\n    for (const item of items) {\n        ",
    "What is the capital of France?",
    "Explain how neural networks work in simple terms.",
    "What are the benefits of edge computing?",
    "How does speculative decoding improve LLM inference?",
    "What is the difference between TCP and UDP?",
    "Write a poem about autumn leaves.",
    "Generate a SQL query to find the top 10 customers by revenue.",
    "Create a REST API endpoint for user registration.",
    "Write a unit test for a binary search function.",
    "Design a database schema for an e-commerce platform.",
]

CODE_KEYWORDS = ["def ", "class ", "import ", "function ", "const ", "var ",
                 "```", "    ", "\t", "return ", "if ", "for ", "while ",
                 "async ", "await ", "SELECT ", "CREATE ", "INSERT "]


def has_code(prompt: str) -> bool:
    return any(kw in prompt for kw in CODE_KEYWORDS)


class MockSystemProfile(SystemProfile):
    """模拟 SystemProfile (用于 SFT 数据生成, 绕过真实硬件检测)."""

    @classmethod
    def from_hardware_spec(
        cls,
        chip: str,
        total_mem: float,
        avail_mem: float,
        gpu_type: str,
        tflops: float,
        engine: str,
        rtt: float,
        disk: float,
        bench_spec: tuple,
    ) -> "MockSystemProfile":
        """从硬件规格创建模拟 SystemProfile."""
        profile = cls()
        profile.os_name = "Darwin" if "Apple" in chip or "Qualcomm" in chip else "Linux"
        profile.arch = "arm64" if "Apple" in chip or "Qualcomm" in chip else "x86_64"
        profile.cpu_brand = chip
        profile.cpu_cores = 8 if "Pro" in chip else 12 if "Max" in chip else 6
        profile.total_mem_gb = total_mem
        profile.avail_mem_gb = avail_mem
        profile.disk_total_gb = disk * 2
        profile.disk_available_gb = disk
        profile.gpu_type = gpu_type
        profile.gpu_name = chip if "Apple" in chip else "NVIDIA RTX 4090" if "nvidia" in gpu_type else "AMD RX 7900"
        profile.gpu_vram_gb = 0 if "apple" in gpu_type else 24
        profile.compute_tier = "ultra" if tflops > 60 else "medium" if tflops > 35 else "weak"
        profile.tflops = tflops
        profile.recommended_engine = engine
        profile.rtt_ms = rtt
        profile.bandwidth_mbps = 1000 if rtt < 20 else 100 if rtt < 100 else 10
        profile.online = rtt < 9999

        # PlatformBenchmark 结果
        mlx_avail, llamacpp_avail, mlx_tps, llamacpp_tps, preferred = bench_spec
        profile.mlx_available = mlx_avail
        profile.llamacpp_available = llamacpp_avail
        profile.mlx_benchmark_tps = mlx_tps
        profile.llamacpp_benchmark_tps = llamacpp_tps
        profile.preferred_edge_backend = preferred

        return profile


def generate_sft_pair(
    prompt: str,
    hw_spec: tuple,
    bench_spec: tuple,
    draft_model: dict,
    add_noise: bool = True,
) -> dict:
    """用 Hermes Router v4 生成一条 SFT 配对.

    Args:
        prompt: 用户 prompt
        hw_spec: 硬件规格 tuple
        bench_spec: PlatformBenchmark 规格 tuple
        draft_model: draft 模型信息 dict
        add_noise: 是否对决策加 noise (让 Hermes 学会边界情况)

    Returns:
        {"messages": [system, user, assistant]} 格式的对话
    """
    # 1. 创建模拟 SystemProfile (含 PlatformBenchmark 结果)
    profile = MockSystemProfile.from_hardware_spec(*hw_spec, bench_spec)

    # 2. 评估 ProfileBinding
    binding = ProfileBinding.evaluate(
        model_name=draft_model["name"],
        model_display_name=draft_model["display_name"],
        draft_model_path="",
        draft_model_size_gb=draft_model["draft_model_size_gb"],
        draft_architecture=draft_model["draft_architecture"],
        draft_model_type=draft_model["draft_model_type"],
        draft_num_layers=draft_model["draft_num_layers"],
        draft_hidden_size=draft_model["draft_hidden_size"],
        draft_vocab_size=draft_model["draft_vocab_size"],
        draft_params_m=draft_model["draft_params_m"],
        system_profile=profile,
        cloud_accept_rate=draft_model["cloud_accept_rate"],
    )

    # 3. 构建 4D 矩阵
    matrix = FourDMatrix.from_system_profile(
        profile=profile,
        model_name=draft_model["name"],
        draft_model_path="",
        draft_architecture=draft_model["draft_architecture"],
        draft_model_size_gb=draft_model["draft_model_size_gb"],
        draft_params_m=draft_model["draft_params_m"],
        prompt=prompt,
    )
    matrix.history_accept_rate = random.uniform(0.6, 0.9) if add_noise else 0.75
    matrix.cache_hit_rate = random.uniform(0.3, 0.8) if add_noise else 0.5

    # 随机模拟离线场景 (10% 概率)
    cache_hit = False
    if add_noise and random.random() < 0.15:
        cache_hit = True
    if add_noise and random.random() < 0.10:
        matrix.D1.stability = "offline"
        matrix.D1.rtt_ms = 9999
        profile.online = False

    # 4. 执行十步流水线
    pipeline = TenStepPipeline()
    pipeline_steps = pipeline.execute(
        profile=profile,
        binding=binding,
        prompt=prompt,
        verbose=False,
    )

    # 5. Hermes 路由决策 (ground truth)
    # 模拟 MTP 可用性和 accept rate
    mtp_available = random.random() > 0.05 if add_noise else True
    mtp_accept_rate = random.uniform(0.3, 1.0) if add_noise else binding.estimated_cloud_mtp_accept_rate

    # 用 Bootstrap 创建一个 mini router
    bootstrap = Bootstrap.__new__(Bootstrap)
    bootstrap.system_profile = profile
    bootstrap.bindings = {draft_model["name"]: binding}
    bootstrap.state_abi = StateABI()
    bootstrap.state_abi.transition("READY", "SFT 生成")
    bootstrap.cloud_mtp_url = "http://cloud:30001"
    bootstrap.cloud_plain_url = "http://cloud:30000"
    bootstrap.result = type("R", (), {"cloud_reachable": mtp_available and profile.online})()

    router = HermesRouter(
        bootstrap=bootstrap,
        cloud_mtp_url="http://cloud:30001",
        cloud_plain_url="http://cloud:30000",
    )

    decision = router.decide(
        model_name=draft_model["name"],
        prompt=prompt,
        cache_hit=cache_hit,
        online=profile.online if not profile.online else None,
        mtp_available=mtp_available if profile.online else None,
        mtp_accept_rate=mtp_accept_rate if profile.online else None,
    )

    # 加 noise: 5% 概率切换到相邻模式 (让 Hermes 学边界)
    if add_noise and random.random() < 0.05:
        modes = ["cache_hit", "local_only", "edge_draft", "cloud_mtp", "cloud_only"]
        current_idx = modes.index(decision.mode) if decision.mode in modes else 3
        new_idx = max(0, min(len(modes) - 1, current_idx + random.choice([-1, 1])))
        decision.mode = modes[new_idx]
        decision.confidence = random.uniform(0.6, 0.8)
        decision.reason = f"boundary_case: {decision.reason}"

    # 6. 构造 SFT 对话
    system_msg = {
        "role": "system",
        "content": (
            "You are Hermes, a cognitive routing agent for edge-cloud LLM inference. "
            "Given a 4D perception matrix (D1 network, D2 hardware, D3 model, D4 context), "
            "platform benchmark results (MLX vs llama.cpp), and ten-step pipeline summary, "
            "output a JSON routing decision.\n"
            "\nRouting modes:\n"
            "- cache_hit: L1-L5 cache hit, return immediately\n"
            "- local_only: offline/privacy, local inference\n"
            "- edge_draft: Hermes orchestrates local draft (MLX/llama.cpp) + cloud verify (saves cloud compute)\n"
            "- cloud_mtp: Hermes orchestrates cloud NEXTN MTP (draft+verify both on cloud)\n"
            "- cloud_only: direct cloud (MTP degraded/unavailable)\n"
            "\nOutput ONLY valid JSON with these fields:\n"
            "- mode: one of cache_hit, local_only, edge_draft, cloud_mtp, cloud_only\n"
            "- edge_backend: mlx / llamacpp / none (which local backend for edge_draft)\n"
            "- confidence: 0.0-1.0\n"
            "- reason: brief explanation\n"
            "- expected_ttft_ms: estimated time to first token\n"
            "- expected_decode_tps: estimated decode throughput\n"
            "- cloud_compute_savings_pct: 0.0-1.0 (cloud compute saved by edge_draft)\n"
            "- use_mtp: boolean"
        ),
    }

    # User message: 4D matrix + benchmark + pipeline summary
    user_content = {
        "four_d_matrix": matrix.to_dict(),
        "platform_benchmark": {
            "mlx_available": profile.mlx_available,
            "llamacpp_available": profile.llamacpp_available,
            "mlx_tps": profile.mlx_benchmark_tps,
            "llamacpp_tps": profile.llamacpp_benchmark_tps,
            "preferred_backend": profile.preferred_edge_backend,
        },
        "ten_step_pipeline_summary": {
            "step_1_os": pipeline_steps.get("1_os", {}),
            "step_2_cpu": pipeline_steps.get("2_cpu", {}),
            "step_4_arch": pipeline_steps.get("4_arch", {}),
            "step_5_mem": pipeline_steps.get("5_mem", {}),
            "step_5.5_compute": pipeline_steps.get("5.5_compute", {}),
            "step_6_engine": pipeline_steps.get("6_engine", {}),
            "step_7.5_route": pipeline_steps.get("7.5_route", {}),
            "step_7.7_mtp": pipeline_steps.get("7.7_mtp", {}),
            "step_9_4d": pipeline_steps.get("9_4d", {}),
        },
        "profile_binding": {
            "can_run_on_edge": binding.can_run_on_edge,
            "edge_backend": binding.edge_backend,
            "draft_architecture": binding.draft_architecture,
            "draft_model_size_gb": binding.draft_model_size_gb,
            "estimated_edge_tps": binding.estimated_edge_tps,
            "cloud_compute_savings_pct": binding.cloud_compute_savings_pct,
        },
        "request_context": {
            "prompt_preview": prompt[:200],
            "prompt_has_code": has_code(prompt),
            "cache_hit": cache_hit,
            "online": profile.online,
            "rtt_ms": profile.rtt_ms,
        },
    }

    user_msg = {
        "role": "user",
        "content": json.dumps(user_content, ensure_ascii=False),
    }

    assistant_msg = {
        "role": "assistant",
        "content": json.dumps({
            "mode": decision.mode,
            "edge_backend": binding.edge_backend if decision.mode == "edge_draft" else "none",
            "confidence": round(decision.confidence, 2),
            "reason": decision.reason,
            "expected_ttft_ms": round(decision.expected_ttft_ms, 1),
            "expected_decode_tps": round(decision.expected_decode_tps, 1),
            "cloud_compute_savings_pct": round(decision.cloud_compute_savings_pct, 3),
            "use_mtp": decision.use_mtp,
        }, ensure_ascii=False),
    }

    return {"messages": [system_msg, user_msg, assistant_msg]}


def generate_sft_dataset(
    num_samples: int = 5000,
    output_path: str = "",
) -> str:
    """生成完整 SFT 数据集.

    Args:
        num_samples: 配对数量
        output_path: 输出 JSONL 路径

    Returns:
        输出文件路径
    """
    if not output_path:
        output_path = os.path.join(PROJECT_ROOT, "data", "hermes_sft_train_v4.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mode_counts = {}
    backend_counts = {}
    total = 0

    with open(output_path, "w") as f:
        for i in range(num_samples):
            hw_spec = random.choice(HARDWARE_SPACE)
            bench_spec = random.choice(PLATFORM_BENCHMARK_SPACE)
            draft_model = random.choice(DRAFT_MODEL_SPACE)
            prompt = random.choice(PROMPT_TEMPLATES)

            pair = generate_sft_pair(prompt, hw_spec, bench_spec, draft_model, add_noise=True)

            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            assistant_content = json.loads(pair["messages"][2]["content"])
            mode = assistant_content["mode"]
            backend = assistant_content.get("edge_backend", "none")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            total += 1

            if (i + 1) % 1000 == 0:
                logger.info(f"[sft] Generated {i+1}/{num_samples} pairs")

    logger.info(f"\n[sft] Dataset generated: {output_path}")
    logger.info(f"[sft] Total pairs: {total}")
    logger.info("[sft] Mode distribution:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        logger.info(f"  {mode}: {count} ({pct:.1f}%)")
    logger.info("[sft] Backend distribution:")
    for backend, count in sorted(backend_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        logger.info(f"  {backend}: {count} ({pct:.1f}%)")

    return output_path


def generate_eval_dataset(
    num_samples: int = 500,
    output_path: str = "",
) -> str:
    """生成评估数据集 (不带 noise, 用于准确率测量)."""
    if not output_path:
        output_path = os.path.join(PROJECT_ROOT, "data", "hermes_sft_eval_v4.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for i in range(num_samples):
            hw_spec = random.choice(HARDWARE_SPACE)
            bench_spec = random.choice(PLATFORM_BENCHMARK_SPACE)
            draft_model = random.choice(DRAFT_MODEL_SPACE)
            prompt = random.choice(PROMPT_TEMPLATES)

            pair = generate_sft_pair(prompt, hw_spec, bench_spec, draft_model, add_noise=False)
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"[sft] Eval dataset: {output_path} ({num_samples} pairs)")
    return output_path


def validate_dataset(filepath: str) -> dict:
    """验证数据集格式 + 统计."""
    total = 0
    valid = 0
    mode_counts = {}
    backend_counts = {}
    errors = []

    with open(filepath) as f:
        for i, line in enumerate(f):
            total += 1
            try:
                data = json.loads(line)
                assert "messages" in data
                assert len(data["messages"]) == 3
                assert data["messages"][0]["role"] == "system"
                assert data["messages"][1]["role"] == "user"
                assert data["messages"][2]["role"] == "assistant"

                decision = json.loads(data["messages"][2]["content"])
                assert "mode" in decision
                assert "confidence" in decision
                assert "edge_backend" in decision

                mode = decision["mode"]
                backend = decision.get("edge_backend", "none")
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
                valid += 1
            except Exception as e:
                errors.append(f"Line {i+1}: {e}")

    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "mode_distribution": mode_counts,
        "backend_distribution": backend_counts,
        "errors": errors[:5],
    }


def show_sample(filepath: str, n: int = 2) -> None:
    """显示前 n 条 SFT 样例."""
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            print(f"\n{'='*70}")
            print(f"Sample {i+1}:")
            print(f"{'='*70}")

            # System
            print(f"\n[System]:")
            print(f"  {data['messages'][0]['content'][:200]}...")

            # User (4D matrix + benchmark + pipeline)
            user_data = json.loads(data["messages"][1]["content"])
            print(f"\n[User]:")
            print(f"  4D Matrix:")
            print(f"    D1 Network: rtt={user_data['four_d_matrix']['D1_network']['rtt_ms']}ms, "
                  f"stability={user_data['four_d_matrix']['D1_network']['stability']}")
            print(f"    D2 Hardware: {user_data['four_d_matrix']['D2_hardware']['chip']}, "
                  f"tier={user_data['four_d_matrix']['D2_hardware']['compute_tier']}")
            print(f"    D3 Model: {user_data['four_d_matrix']['D3_model']['name']}")
            print(f"  Platform Benchmark:")
            print(f"    MLX={user_data['platform_benchmark']['mlx_tps']} tok/s, "
                  f"llama.cpp={user_data['platform_benchmark']['llamacpp_tps']} tok/s, "
                  f"preferred={user_data['platform_benchmark']['preferred_backend']}")
            print(f"  Pipeline Step 7.5 (Route):")
            print(f"    {json.dumps(user_data['ten_step_pipeline_summary']['step_7.5_route'], ensure_ascii=False)}")
            print(f"  Profile Binding:")
            print(f"    can_run_on_edge={user_data['profile_binding']['can_run_on_edge']}, "
                  f"backend={user_data['profile_binding']['edge_backend']}")

            # Assistant (decision)
            decision = json.loads(data["messages"][2]["content"])
            print(f"\n[Assistant]:")
            print(f"  mode: {decision['mode']}")
            print(f"  edge_backend: {decision['edge_backend']}")
            print(f"  confidence: {decision['confidence']}")
            print(f"  reason: {decision['reason']}")
            print(f"  TTFT: {decision['expected_ttft_ms']}ms")
            print(f"  decode: {decision['expected_decode_tps']} tok/s")
            print(f"  savings: {decision['cloud_compute_savings_pct']:.0%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hermes Router SFT Data Synthesis v4")
    parser.add_argument("--num", type=int, default=5000, help="Number of SFT pairs")
    parser.add_argument("--eval-num", type=int, default=500, help="Number of eval pairs")
    parser.add_argument("--output", type=str, default="", help="Output path")
    parser.add_argument("--validate", type=str, default="", help="Validate dataset")
    parser.add_argument("--show", type=str, default="", help="Show samples from dataset")
    parser.add_argument("--show-n", type=int, default=2, help="Number of samples to show")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.validate:
        result = validate_dataset(args.validate)
        print(json.dumps(result, indent=2))
    elif args.show:
        show_sample(args.show, args.show_n)
    else:
        train_path = generate_sft_dataset(args.num, args.output)
        eval_path = generate_eval_dataset(args.eval_num)

        print(f"\nTrain: {train_path}")
        print(f"Eval:  {eval_path}")

        print("\n=== Validation ===")
        train_stats = validate_dataset(train_path)
        print(f"Train: {train_stats['valid']}/{train_stats['total']} valid")
        print(f"  Modes: {train_stats['mode_distribution']}")
        print(f"  Backends: {train_stats['backend_distribution']}")

        eval_stats = validate_dataset(eval_path)
        print(f"Eval: {eval_stats['valid']}/{eval_stats['total']} valid")
        print(f"  Modes: {eval_stats['mode_distribution']}")

        print("\n=== Sample ===")
        show_sample(train_path, 2)
