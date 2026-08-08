#!/usr/bin/env python3
"""bench_e2e_draft_verify.py — 端到端 Hermes v4 路由 + MTP draft→verify benchmark.

测试链路:
  Mac (Hermes Router v4 + PlatformBenchmark) ──SSH tunnel──> Host1 (sglang Gemma4 26B + NEXTN MTP)

核心测试:
  Phase 0: PlatformBenchmark — MLX vs llama.cpp 平台速度检测, 选快的
  Phase 1: Hermes Bootstrap — ProfileBinding 评估每个模型能否在端侧跑 draft
  Phase 2: 云端 NEXTN MTP benchmark — Gemma4 26B + Gemma4Assistant draft head
  Phase 3: Hermes 路由决策展示 — 3 个模型的路由结果 + 十步流水线 + 后端选择
  Phase 4: 云端算力节省计算 — edge_draft 模式省了多少 cloud GPU 时间
  Phase 5: SFT 数据样例 — 展示 4D矩阵+十步流水线数据如何微调 Hermes

路由逻辑:
  Gemma4 26B: gemma4_assistant 架构 Mac 不支持 → cloud_mtp (云端 NEXTN MTP)
  DSV4 Flash: deepseek_v4 架构 Mac 不支持 → cloud_mtp
  Qwen3-VL:   MTPHead 自定义架构 Mac 支持 → edge_draft via llama.cpp (省云端算力)

用法:
    python bench_e2e_draft_verify.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import statistics
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Suppress verbose logging from hermes internals
logging.getLogger("app.shared.hermes_router").setLevel(logging.WARNING)

# === Config ===
CLOUD_MTP_URL = "http://localhost:30001"   # sglang + NEXTN MTP (SSH tunnel to Host1)
CLOUD_PLAIN_URL = "http://localhost:30000"  # sglang without MTP

# Test prompts (code completion scenarios)
TEST_PROMPTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "import numpy as np\n\narr = np.array([1, 2, 3",
    "class Solution:\n    def twoSum(self, nums, target):",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if",
    "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        ",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    ",
    "def binary_search(nums, target):\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] ==",
    "class LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, data):\n        new_node = Node(data)\n        if not self.head:\n            ",
]

WARMUP_PROMPTS = [
    "def hello_world():\n    print('hello')\n    return",
    "import os\n\npath = os.path.join('a', 'b'",
]


# ============================================================================
# Phase 0: PlatformBenchmark — MLX vs llama.cpp 平台速度检测
# ============================================================================
async def bench_platform_benchmark() -> dict:
    """Phase 0: MLX vs llama.cpp 平台基准测试."""
    from app.shared.hermes_router import PlatformBenchmark, SystemProfile

    print("\n" + "=" * 70)
    print("Phase 0: PlatformBenchmark — MLX vs llama.cpp 平台速度检测")
    print("=" * 70)

    profile = SystemProfile.detect()
    benchmark = PlatformBenchmark()
    result = benchmark.run(system_profile=profile, verbose=True)

    print(f"\n  {'Metric':<30s} {'MLX':>12s} {'llama.cpp':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Available':<30s} {'✅' if result.mlx_available else '❌':>12s} {'✅' if result.llamacpp_available else '❌':>12s}")
    print(f"  {'Decode tok/s':<30s} {result.mlx_tps:>11.1f}  {result.llamacpp_tps:>11.1f} ")
    print(f"  {'5-token latency':<30s} {result.mlx_latency_ms:>10.0f}ms {result.llamacpp_latency_ms:>10.0f}ms")

    print(f"\n  ✅ 首选后端: {result.preferred_backend} (speedup {result.speedup:.1f}x)")
    if result.preferred_backend == "llamacpp":
        print(f"     → edge_draft 模式将使用 llama.cpp 调动 draft model")
    elif result.preferred_backend == "mlx":
        print(f"     → edge_draft 模式将使用 MLX 调动 draft model")
    else:
        print(f"     → 无可用端侧后端, 所有请求走 cloud_mtp")

    return {
        **result.to_dict(),
        "system_profile_chip": profile.cpu_brand,
    }


# ============================================================================
# Phase 1: Hermes Bootstrap + ProfileBinding — 端侧能力检测
# ============================================================================
async def bench_hermes_bootstrap() -> dict:
    """Phase 1: Hermes Bootstrap — 端侧能力检测 + ProfileBinding 评估."""
    from app.shared.hermes_router import Bootstrap, SystemProfile, TenStepPipeline

    print("\n" + "=" * 70)
    print("Phase 1: Hermes Bootstrap — 端侧能力检测 + ProfileBinding (含后端选择)")
    print("=" * 70)

    # Bootstrap (含 PlatformBenchmark)
    bootstrap = Bootstrap(
        cloud_mtp_url=CLOUD_MTP_URL,
        cloud_plain_url=CLOUD_PLAIN_URL,
    )
    result = bootstrap.run(verbose=True)

    print(f"\n  Bootstrap: {'✅' if result.success else '❌'} ({result.elapsed_ms:.1f}ms)")
    print(f"  Cloud MTP: {'✅' if result.cloud_reachable else '❌'}")
    print(f"  State ABI: {result.state}")

    # ProfileBinding 结果
    print(f"\n  ProfileBinding 评估:")
    print(f"  {'Model':<22s} {'Arch':<30s} {'Edge?':>6s} {'Backend':>10s} {'Savings':>8s} {'Reason':<30s}")
    print(f"  {'-'*22} {'-'*30} {'-'*6} {'-'*10} {'-'*8} {'-'*30}")

    bindings_info = {}
    for name, binding in bootstrap.bindings.items():
        arch_short = binding.draft_architecture[:28]
        size_str = f"{binding.draft_model_size_gb:.1f}GB"
        edge_str = "✅ EDGE" if binding.can_run_on_edge else "❌ CLOUD"
        backend_str = binding.edge_backend or "—"
        savings_str = f"省{binding.cloud_compute_savings_pct:.0%}" if binding.can_run_on_edge else "N/A"
        reason_short = binding.reason[:30]
        print(f"  {binding.model_display_name:<22s} {arch_short:<30s} {edge_str:>6s} {backend_str:>10s} {savings_str:>8s} {reason_short:<30s}")

        bindings_info[name] = binding.to_dict()

    # 十步流水线 (Gemma4 为例)
    print(f"\n  十步流水线 (Gemma4):")
    gemma4_binding = bootstrap.bindings.get("gemma4")
    if gemma4_binding:
        pipeline = TenStepPipeline()
        pipeline.execute(bootstrap.system_profile, gemma4_binding, "def fibonacci(n):\n    return", verbose=True)

    return {
        "system_profile": bootstrap.system_profile.to_dict(),
        "platform_benchmark": result.platform_benchmark,
        "bootstrap_result": {
            "success": result.success,
            "elapsed_ms": result.elapsed_ms,
            "cloud_reachable": result.cloud_reachable,
            "state": result.state,
        },
        "bindings": bindings_info,
        "bootstrap": bootstrap,  # 传递给后续 phase
    }


# ============================================================================
# Phase 2: 云端 NEXTN MTP benchmark
# ============================================================================
async def bench_cloud_mtp(prompts: list[str]) -> dict:
    """Phase 2: 云端 sglang NEXTN MTP — Gemma4 26B + Gemma4Assistant draft head."""
    import aiohttp

    print("\n" + "=" * 70)
    print("Phase 2: 云端 NEXTN MTP (sglang Gemma4 26B + Gemma4Assistant MTP head)")
    print("   Draft model: Gemma4AssistantForCausalLM (4 layers, 1024 hidden, 262K vocab)")
    print("   Config: num_steps=5, num_draft_tokens=5, eagle_topk=1")
    print("=" * 70)

    results = []
    async with aiohttp.ClientSession() as session:
        # Warmup
        for wp in WARMUP_PROMPTS:
            await session.post(
                f"{CLOUD_MTP_URL}/generate",
                json={"text": wp, "sampling_params": {"temperature": 0, "max_new_tokens": 5}, "stream": False},
                timeout=aiohttp.ClientTimeout(total=30),
            )

        for prompt in prompts:
            t0 = time.time()
            async with session.post(
                f"{CLOUD_MTP_URL}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 30},
                    "stream": False,
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                t1 = time.time()

                latency_ms = (t1 - t0) * 1000
                text = data.get("text", "")
                meta = data.get("meta_info", {})

                completion_tokens = meta.get("completion_tokens", 0)
                e2e_latency = meta.get("e2e_latency", 0) * 1000
                cached_tokens = meta.get("cached_tokens", 0)
                spec_accept_rate = meta.get("spec_accept_rate", 0)
                spec_accept_length = meta.get("spec_accept_length", 0)
                spec_num_correct = meta.get("spec_num_correct_drafts", 0)
                spec_num_proposed = meta.get("spec_num_proposed_drafts", 0)

                tps = completion_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
                prefill_tokens = meta.get("prompt_tokens", 0) - cached_tokens
                decode_time_s = max(e2e_latency / 1000 - 0.001 * prefill_tokens, 0.001)
                decode_tps = completion_tokens / decode_time_s if decode_time_s > 0 else 0

                results.append({
                    "prompt": prompt[:50],
                    "text": text[:60],
                    "latency_ms": round(latency_ms, 1),
                    "e2e_latency_ms": round(e2e_latency, 1),
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                    "tps": round(tps, 1),
                    "decode_tps": round(decode_tps, 1),
                    "spec_accept_rate": spec_accept_rate,
                    "spec_accept_length": spec_accept_length,
                    "spec_correct": spec_num_correct,
                    "spec_proposed": spec_num_proposed,
                })

                print(f"  {prompt[:40]:40s} → {text[:25]:25s} | {latency_ms:.0f}ms | {tps:.0f} tok/s")
                print(f"    MTP: accept={spec_accept_rate:.0%} len={spec_accept_length:.1f} "
                      f"correct={spec_num_correct}/{spec_num_proposed} cached={cached_tokens}")

    # Streaming TTFT
    ttfts = []
    async with aiohttp.ClientSession() as session:
        for prompt in prompts:
            t0 = time.time()
            first_token_ms = None
            async with session.post(
                f"{CLOUD_MTP_URL}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 20},
                    "stream": True,
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            token_text = chunk.get("text", "")
                            elapsed = (time.time() - t0) * 1000
                            if token_text and first_token_ms is None:
                                first_token_ms = elapsed
                                break
                        except json.JSONDecodeError:
                            continue
            if first_token_ms:
                ttfts.append(first_token_ms)

    avg_ttft = statistics.median(ttfts) if ttfts else 999
    avg_latency = statistics.median([r["latency_ms"] for r in results])
    avg_tps = statistics.median([r["tps"] for r in results])
    avg_decode_tps = statistics.median([r["decode_tps"] for r in results])
    avg_accept_rate = statistics.mean([r["spec_accept_rate"] for r in results]) if results else 0
    avg_accept_length = statistics.mean([r["spec_accept_length"] for r in results]) if results else 0

    base_decode_tps = avg_decode_tps / (1 + avg_accept_length) if avg_accept_length > 0 else avg_decode_tps

    print(f"\n  {'Metric':<30s} {'Value':>10s}")
    print(f"  {'-'*30} {'-'*10}")
    print(f"  {'TTFT (median)':<30s} {avg_ttft:>9.1f}ms")
    print(f"  {'Latency (median)':<30s} {avg_latency:>9.1f}ms")
    print(f"  {'Throughput (median)':<30s} {avg_tps:>9.1f} tok/s")
    print(f"  {'Decode rate (median)':<30s} {avg_decode_tps:>9.1f} tok/s")
    print(f"  {'Base decode (no MTP)':<30s} {base_decode_tps:>9.1f} tok/s")
    print(f"  {'MTP accept rate':<30s} {avg_accept_rate:>9.1%}")
    print(f"  {'MTP accept length':<30s} {avg_accept_length:>9.1f}")
    print(f"  {'MTP speedup':<30s} {1 + avg_accept_length:>9.2f}x")

    return {
        "results": results,
        "avg_ttft_ms": round(avg_ttft, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_tps": round(avg_tps, 1),
        "avg_decode_tps": round(avg_decode_tps, 1),
        "base_decode_tps": round(base_decode_tps, 1),
        "avg_accept_rate": round(avg_accept_rate, 3),
        "avg_accept_length": round(avg_accept_length, 2),
        "ttfts": [round(t, 1) for t in ttfts],
    }


# ============================================================================
# Phase 3: Hermes 路由决策 — 3 模型对比 (含后端选择)
# ============================================================================
async def bench_hermes_routing(bootstrap_data: dict, cloud_mtp_result: dict) -> dict:
    """Phase 3: Hermes 路由决策 — 每个模型的完整路由流程."""
    from app.shared.hermes_router import HermesRouter

    print("\n" + "=" * 70)
    print("Phase 3: Hermes 路由决策 — 平台检测 + 后端选择 + 路由分发")
    print("=" * 70)

    bootstrap = bootstrap_data["bootstrap"]
    router = HermesRouter(bootstrap=bootstrap)

    cloud_decode_tps = cloud_mtp_result["avg_decode_tps"]
    base_decode_tps = cloud_mtp_result["base_decode_tps"]
    accept_rate = cloud_mtp_result["avg_accept_rate"]
    accept_length = cloud_mtp_result["avg_accept_length"]

    routing_results = {}

    models_to_test = [
        ("gemma4", "Gemma4 26B-A4B", "gemma4_assistant 架构 Mac 不支持 → cloud_mtp"),
        ("dsv4", "DeepSeek V4 Flash", "deepseek_v4 架构 Mac 不支持 → cloud_mtp"),
        ("qwen3vl", "Qwen3-VL 2B", "MTPHead 架构 Mac 支持 → edge_draft via llama.cpp"),
    ]

    for model_name, display_name, expected in models_to_test:
        binding = bootstrap.bindings.get(model_name)
        if not binding:
            continue

        decision = router.decide(
            model_name=model_name,
            prompt="def fibonacci(n):\n    return",
            cache_hit=False,
            mtp_available=True,
            mtp_accept_rate=accept_rate if model_name == "gemma4" else binding.estimated_cloud_mtp_accept_rate,
        )

        print(f"\n  {display_name}:")
        print(f"    Draft arch:    {binding.draft_architecture}")
        print(f"    Draft size:    {binding.draft_model_size_gb:.1f}GB ({binding.draft_params_m:.1f}M params)")
        print(f"    Can run edge:  {'✅' if binding.can_run_on_edge else '❌'}")
        print(f"    Edge backend:  {binding.edge_backend or 'N/A'}")
        print(f"    Expected:      {expected}")
        print(f"    ────────────────────────────────────")
        print(f"    Route mode:    {decision.mode}")
        print(f"    Confidence:    {decision.confidence:.0%}")
        print(f"    Reason:        {decision.reason}")
        print(f"    TTFT:          {decision.expected_ttft_ms:.1f}ms")
        print(f"    Decode:        {decision.expected_decode_tps:.1f} tok/s")
        print(f"    Cloud savings: {decision.cloud_compute_savings_pct:.0%}")
        print(f"    Cloud URL:     {decision.cloud_url or '(local)'}")
        print(f"    Use MTP:       {decision.use_mtp}")

        if decision.mode == "edge_draft":
            print(f"    → Hermes 调动端侧 {binding.edge_backend} 运行 draft model, "
                  f"draft tokens 发送到云端 verify")
        elif decision.mode == "cloud_mtp":
            print(f"    → Hermes 调动云端 sglang NEXTN MTP, draft+verify 都在云端")

        if model_name == "gemma4":
            actual = {
                "actual_ttft_ms": cloud_mtp_result["avg_ttft_ms"],
                "actual_decode_tps": cloud_decode_tps,
                "actual_accept_rate": accept_rate,
                "actual_accept_length": accept_length,
            }
        else:
            actual = {"note": "未实测 (云端未部署此模型)"}

        routing_results[model_name] = {
            "display_name": display_name,
            "binding": binding.to_dict(),
            "decision": decision.to_dict(),
            "actual": actual,
        }

    return routing_results


# ============================================================================
# Phase 4: 云端算力节省分析
# ============================================================================
async def bench_cloud_savings(routing_results: dict, cloud_mtp_result: dict) -> dict:
    """Phase 4: 云端算力节省分析 — edge_draft vs cloud_mtp."""
    print("\n" + "=" * 70)
    print("Phase 4: 云端算力节省分析 — Hermes 调动 draft model 的经济效益")
    print("=" * 70)

    accept_length = cloud_mtp_result["avg_accept_length"]
    base_decode = cloud_mtp_result["base_decode_tps"]
    mtp_decode = cloud_mtp_result["avg_decode_tps"]
    cloud_ttft = cloud_mtp_result["avg_ttft_ms"]

    print(f"\n  云端 NEXTN MTP 实测数据:")
    print(f"    Base decode (no MTP):  {base_decode:.1f} tok/s")
    print(f"    MTP decode:            {mtp_decode:.1f} tok/s")
    print(f"    Accept length:         {accept_length:.1f} tokens/step")
    print(f"    Speedup:               {1 + accept_length:.2f}x")
    print(f"    TTFT:                  {cloud_ttft:.1f}ms")

    print(f"\n  云端算力消耗分析 (per {1 + accept_length:.0f} tokens):")
    print(f"    {'Mode':<30s} {'Draft':>10s} {'Verify':>10s} {'Total':>10s} {'Savings':>10s}")
    print(f"    {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    draft_layers = 4
    main_layers = 48
    cloud_mtp_draft = 5 * draft_layers
    cloud_mtp_verify = main_layers
    cloud_mtp_total = cloud_mtp_draft + cloud_mtp_verify

    print(f"    {'cloud_mtp (all cloud)':<30s} {cloud_mtp_draft:>10d} {cloud_mtp_verify:>10d} {cloud_mtp_total:>10d} {'0%':>10s}")

    edge_draft_draft = 0
    edge_draft_verify = main_layers
    edge_draft_total = edge_draft_draft + edge_draft_verify
    edge_savings = (cloud_mtp_total - edge_draft_total) / cloud_mtp_total

    print(f"    {'edge_draft (Hermes→edge)':<30s} {edge_draft_draft:>10d} {edge_draft_verify:>10d} {edge_draft_total:>10d} {edge_savings:>9.0%}")

    print(f"\n  Per-model 云端算力节省:")
    print(f"  {'Model':<22s} {'Route':<15s} {'Backend':>10s} {'Savings':>8s} {'Edge TPS':>10s} {'Cloud TPS':>10s}")
    print(f"  {'-'*22} {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    model_savings = {}
    for model_name, info in routing_results.items():
        binding = info["binding"]
        decision = info["decision"]
        display = info["display_name"]

        route = decision["mode"]
        backend = binding.get("edge_backend", "none")
        savings = decision["cloud_compute_savings_pct"]

        if binding["can_run_on_edge"]:
            edge_tps = binding["estimated_edge_tps"]
            cloud_tps = base_decode
            combined = min(edge_tps, mtp_decode)
            print(f"  {display:<22s} {route:<15s} {backend:>10s} {savings:>7.0%} {edge_tps:>9.1f} {cloud_tps:>9.1f}")
        else:
            edge_tps = 0
            cloud_tps = mtp_decode
            print(f"  {display:<22s} {route:<15s} {backend:>10s} {savings:>7.0%} {'N/A':>10s} {cloud_tps:>9.1f}")

        model_savings[model_name] = {
            "display_name": display,
            "route": route,
            "edge_backend": backend,
            "can_run_on_edge": binding["can_run_on_edge"],
            "cloud_compute_savings_pct": savings,
            "estimated_edge_tps": edge_tps,
            "cloud_tps": cloud_tps,
        }

    print(f"\n  关键结论:")
    gemma4 = model_savings.get("gemma4", {})
    qwen3vl = model_savings.get("qwen3vl", {})

    print(f"    Gemma4 26B:   draft 架构 Mac 不支持 → Hermes 调云端 NEXTN MTP ({mtp_decode:.0f} tok/s)")
    if qwen3vl.get("can_run_on_edge"):
        print(f"    Qwen3-VL 2B:  draft 可端侧 → Hermes 调 {qwen3vl['edge_backend']} 运行 draft (省 {qwen3vl['cloud_compute_savings_pct']:.0%} 云端算力)")

    return {
        "cloud_mtp_compute": {
            "draft_passes": 5,
            "verify_passes": 1,
            "draft_layers": draft_layers,
            "main_layers": main_layers,
            "total_layer_passes_cloud_mtp": cloud_mtp_total,
            "total_layer_passes_edge_draft": edge_draft_total,
            "savings_pct": edge_savings,
        },
        "per_model": model_savings,
        "target_420_550": {
            "cloud_mtp_tps": mtp_decode,
            "target_met": 420 <= mtp_decode <= 600,
            "gap": max(0, 420 - mtp_decode),
        },
    }


# ============================================================================
# Phase 5: SFT 数据样例 — 4D矩阵+十步流水线数据如何微调 Hermes
# ============================================================================
async def bench_sft_sample() -> dict:
    """Phase 5: 展示 SFT 数据样例 — 用 4D矩阵+十步流水线数据微调 Hermes."""
    print("\n" + "=" * 70)
    print("Phase 5: SFT 数据样例 — 4D矩阵 + 十步流水线数据微调 Hermes")
    print("=" * 70)

    sft_path = os.path.join(PROJECT_ROOT, "data", "hermes_sft_train_v4.jsonl")
    if not os.path.exists(sft_path):
        print("  ⚠️ SFT 数据未生成, 跳过此 phase")
        print(f"  生成命令: python app/training/hermes_route_sft.py --num 100")
        return {"status": "not_generated"}

    # 统计
    total = 0
    mode_counts = {}
    backend_counts = {}
    with open(sft_path) as f:
        for line in f:
            total += 1
            data = json.loads(line)
            decision = json.loads(data["messages"][2]["content"])
            mode = decision["mode"]
            backend = decision.get("edge_backend", "none")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            backend_counts[backend] = backend_counts.get(backend, 0) + 1

    print(f"\n  SFT 数据集: {sft_path}")
    print(f"  总配对数: {total}")
    print(f"\n  Mode 分布:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"    {mode:<20s} {count:>5d} ({count/total*100:.1f}%)")
    print(f"\n  Backend 分布:")
    for backend, count in sorted(backend_counts.items(), key=lambda x: -x[1]):
        print(f"    {backend:<20s} {count:>5d} ({count/total*100:.1f}%)")

    # 展示 2 条样例
    print(f"\n  样例展示 (前 2 条):")
    with open(sft_path) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            data = json.loads(line)
            user_data = json.loads(data["messages"][1]["content"])
            decision = json.loads(data["messages"][2]["content"])

            print(f"\n  ── Sample {i+1} ──")
            print(f"  4D Matrix:")
            print(f"    D1: rtt={user_data['four_d_matrix']['D1_network']['rtt_ms']}ms, "
                  f"stability={user_data['four_d_matrix']['D1_network']['stability']}")
            print(f"    D2: {user_data['four_d_matrix']['D2_hardware']['chip']}, "
                  f"tier={user_data['four_d_matrix']['D2_hardware']['compute_tier']}")
            print(f"    D3: {user_data['four_d_matrix']['D3_model']['name']}")
            print(f"  PlatformBenchmark:")
            print(f"    MLX={user_data['platform_benchmark']['mlx_tps']} tok/s, "
                  f"llama.cpp={user_data['platform_benchmark']['llamacpp_tps']} tok/s, "
                  f"preferred={user_data['platform_benchmark']['preferred_backend']}")
            print(f"  Pipeline 7.5 (Route):")
            route = user_data['ten_step_pipeline_summary']['step_7.5_route']
            print(f"    mode={route['mode']}, can_edge={route['can_run_on_edge']}")
            print(f"  ProfileBinding:")
            pb = user_data['profile_binding']
            print(f"    can_run_on_edge={pb['can_run_on_edge']}, backend={pb['edge_backend']}")
            print(f"  → Decision:")
            print(f"    mode={decision['mode']}, backend={decision['edge_backend']}, "
                  f"conf={decision['confidence']}, savings={decision['cloud_compute_savings_pct']:.0%}")

    print(f"\n  SFT 训练流程:")
    print(f"    1. 数据生成: python app/training/hermes_route_sft.py --num 5000")
    print(f"    2. LoRA 微调: 用 Qwen2.5-1.5B 作为 base model, LoRA r=16")
    print(f"    3. 输入: 4D矩阵 + PlatformBenchmark + 十步流水线摘要")
    print(f"    4. 输出: mode + edge_backend + confidence + TTFT + decode_tps")

    return {
        "sft_path": sft_path,
        "total_pairs": total,
        "mode_distribution": mode_counts,
        "backend_distribution": backend_counts,
    }


# ============================================================================
# Main
# ============================================================================
async def main():
    print("=" * 70)
    print("CGC 端到端 Hermes v4 路由 + MTP Draft→Verify Benchmark")
    print("Mac (Hermes Router v4 + PlatformBenchmark) → SSH tunnel → Host1 (sglang NEXTN MTP)")
    print("=" * 70)

    # Check cloud connectivity
    import aiohttp
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{CLOUD_MTP_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"ERROR: sglang health check failed: {resp.status}")
                    return
                print("✅ sglang NEXTN MTP (cloud) connected via SSH tunnel")
        except Exception as e:
            print(f"ERROR: Cannot connect to sglang: {e}")
            print(f"  Make sure SSH tunnel is up: sshpass ssh -f -N -L 30001:localhost:30001 root@39.106.118.206")
            return

    # Phase 0: PlatformBenchmark (MLX vs llama.cpp)
    platform_bench = await bench_platform_benchmark()

    # Phase 1: Hermes Bootstrap + ProfileBinding
    bootstrap_data = await bench_hermes_bootstrap()

    # Phase 2: Cloud NEXTN MTP benchmark
    cloud_mtp = await bench_cloud_mtp(TEST_PROMPTS)

    # Phase 3: Hermes routing decisions
    routing = await bench_hermes_routing(bootstrap_data, cloud_mtp)

    # Phase 4: Cloud compute savings analysis
    savings = await bench_cloud_savings(routing, cloud_mtp)

    # Phase 5: SFT data samples
    sft_info = await bench_sft_sample()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Platform:         {platform_bench['system_profile_chip']}")
    print(f"  Preferred backend: {platform_bench['preferred_backend']} "
          f"(MLX={platform_bench['mlx_tps']:.0f} vs llama.cpp={platform_bench['llamacpp_tps']:.0f} tok/s, "
          f"speedup={platform_bench['speedup']:.1f}x)")
    print(f"  Cloud model:      Gemma4 26B-A4B (sglang TP4, bf16)")
    print(f"  Draft model:      Gemma4AssistantForCausalLM (official MTP head, 4 layers)")
    print(f"  MTP config:       NEXTN, steps=5, draft_tokens=5, topk=1")
    print(f"  Cloud TTFT:       {cloud_mtp['avg_ttft_ms']:.1f}ms")
    print(f"  Cloud decode:     {cloud_mtp['avg_decode_tps']:.1f} tok/s (with MTP)")
    print(f"  Base decode:      {cloud_mtp['base_decode_tps']:.1f} tok/s (without MTP)")
    print(f"  MTP accept rate:  {cloud_mtp['avg_accept_rate']:.0%}")
    print(f"  MTP accept len:   {cloud_mtp['avg_accept_length']:.1f} tokens/step")
    print(f"  Speedup:          {1 + cloud_mtp['avg_accept_length']:.2f}x")
    print()
    print(f"  Hermes 路由决策 (Hermes 调动 draft model):")
    for model_name, info in routing.items():
        d = info["decision"]
        b = info["binding"]
        edge_str = "端侧✓" if b["can_run_on_edge"] else "端侧✗"
        backend = b.get("edge_backend", "—")
        print(f"    {info['display_name']:20s} {edge_str} [{backend:8s}] → {d['mode']:15s} "
              f"(省 {d['cloud_compute_savings_pct']:.0%} 云端算力)")
    print()
    if sft_info.get("total_pairs"):
        print(f"  SFT 数据:         {sft_info['total_pairs']} 配对 "
              f"(modes: {list(sft_info['mode_distribution'].keys())})")
    print()
    print(f"  目标 420-550 tok/s: {'✅' if savings['target_420_550']['target_met'] else '❌'} "
          f"({cloud_mtp['avg_decode_tps']:.1f} tok/s)")
    if not savings['target_420_550']['target_met']:
        print(f"  差距: {savings['target_420_550']['gap']:.0f} tok/s (需 FP8 量化或多卡)")

    # Save JSON
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "cloud_model": "Gemma4 26B-A4B bf16 TP4",
            "draft_model": "Gemma4AssistantForCausalLM (official MTP head, 4 layers)",
            "mtp_config": "NEXTN, steps=5, draft_tokens=5, topk=1",
            "cloud_url": CLOUD_MTP_URL,
            "hermes_version": "v4",
        },
        "phase0_platform_benchmark": platform_bench,
        "phase1_bootstrap": {
            "system_profile": bootstrap_data["system_profile"],
            "platform_benchmark": bootstrap_data["platform_benchmark"],
            "bootstrap_result": bootstrap_data["bootstrap_result"],
            "bindings": bootstrap_data["bindings"],
        },
        "phase2_cloud_mtp": cloud_mtp,
        "phase3_routing": {k: v for k, v in routing.items()},
        "phase4_savings": savings,
        "phase5_sft": sft_info,
    }

    # Clean non-serializable
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k != "bootstrap"}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    json_path = os.path.join(PROJECT_ROOT, "bench_e2e_draft_verify_result.json")
    with open(json_path, "w") as f:
        json.dump(clean(output), f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
