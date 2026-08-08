"""路由决策模块: 基于 4D 感知矩阵选择 PD分离/Layer-split/全云.

4D 感知矩阵:
  D1: 网络 (RTT, 带宽)
  D2: 硬件 (内存, 算力, GPU, 磁盘)
  D3: 模型 (参数量, 层数, MoE, 量化)
  D4: 路由决策 (mode, P, 预期性能)

路由模式:
  pd_separation: cloud prefill → Mac 全部层 decode (无 RTT)
  layer_split:   cloud prefill → Mac P 层 → cloud L-P 层 (每 token RTT)
  cloud_only:    全云 prefill + decode
  local_only:    Mac 本地 (无 cloud, 离线/隐私)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class ModelInfo:
    """模型信息 (D3)."""
    name: str = ""
    params_b: float = 0.0          # 参数量 (B)
    num_layers: int = 0            # 层数
    is_moe: bool = False           # MoE?
    num_experts: int = 0           # expert 总数
    experts_per_tok: int = 0       # 激活 expert
    hidden_size: int = 0
    vocab_size: int = 0
    quantization: str = "bf16"     # bf16 / 4bit / 8bit
    model_size_gb: float = 0.0     # 量化后大小
    per_layer_gb: float = 0.0      # 每层权重大小


@dataclass
class RouteDecision:
    """路由决策 (D4)."""
    mode: str = "cloud_only"       # pd_separation / layer_split / cloud_only / local_only
    P: int = 0                     # Mac 做的层数
    expected_ttft_ms: float = 0.0
    expected_decode_tps: float = 0.0
    cloud_save_pct: int = 0        # 省 cloud 成本 %
    reason: str = ""

    # Layer-split 详情
    per_layer_forward_ms: float = 0.0  # 每层 forward 时间
    mac_forward_ms: float = 0.0        # Mac 总 forward
    rtt_ms: float = 0.0                # RTT
    spec_n: int = 21                   # 投机批量
    spec_accept: float = 0.28          # accept rate

    def to_dict(self) -> dict:
        return asdict(self)


# === 模型预设 ===

MODEL_PRESETS = {
    "qwen3-vl-2b": ModelInfo(
        name="Qwen3-VL-2B", params_b=2.0, num_layers=28, is_moe=False,
        hidden_size=2048, vocab_size=151936,
        quantization="bf16", model_size_gb=4.26, per_layer_gb=0.15,
    ),
    "qwen3-vl-2b-4bit": ModelInfo(
        name="Qwen3-VL-2B-4bit", params_b=2.0, num_layers=28, is_moe=False,
        hidden_size=2048, vocab_size=151936,
        quantization="4bit", model_size_gb=1.5, per_layer_gb=0.054,
    ),
    "qwen3-vl-30b": ModelInfo(
        name="Qwen3-VL-30B-A3B", params_b=30.0, num_layers=48, is_moe=True,
        num_experts=128, experts_per_tok=8,
        hidden_size=2048, vocab_size=151936,
        quantization="bf16", model_size_gb=60.0, per_layer_gb=1.25,
    ),
    "qwen3-vl-30b-4bit": ModelInfo(
        name="Qwen3-VL-30B-A3B-4bit", params_b=30.0, num_layers=48, is_moe=True,
        num_experts=128, experts_per_tok=8,
        hidden_size=2048, vocab_size=151936,
        quantization="4bit", model_size_gb=15.0, per_layer_gb=0.31,
    ),
    "deepseek-v4-flash": ModelInfo(
        name="DeepSeek-V4-Flash", params_b=671.0, num_layers=61, is_moe=True,
        num_experts=256, experts_per_tok=8,
        hidden_size=7168, vocab_size=129280,
        quantization="fp8", model_size_gb=300.0, per_layer_gb=4.9,
    ),
}


def get_model_info(model_name: str, quantization: str = "auto") -> ModelInfo:
    """获取模型信息 (从预设或 config.json)."""
    key = model_name.lower().replace(" ", "-")
    if quantization != "auto":
        key = f"{key}-{quantization}"

    if key in MODEL_PRESETS:
        return MODEL_PRESETS[key]

    # 尝试模糊匹配
    for k, v in MODEL_PRESETS.items():
        if k in key or key in k:
            return v

    # 默认: 未知模型
    return ModelInfo(name=model_name)


def estimate_per_layer_ms(hardware_info, model_info: ModelInfo) -> float:
    """估算每层 forward 时间 (ms, 1 token decode).

    基于:
      - 算力 (TFLOPS)
      - 模型类型 (dense/MoE)
      - 量化 (bf16/4bit)
    """
    tflops = hardware_info.tflops
    is_moe = model_info.is_moe
    quant = model_info.quantization

    # 每层激活参数量
    if is_moe:
        # MoE: 只激活 experts_per_tok 个 expert
        expert_params = model_info.params_b * 1e9 / model_info.num_layers * \
                        model_info.experts_per_tok / model_info.num_experts
        # attention 参数 (约 hidden_size^2 × 4)
        attn_params = model_info.hidden_size ** 2 * 4 * 2 / model_info.num_layers / 1e9
        per_layer_params_b = (expert_params + attn_params) / 1e9
    else:
        # Dense: 全部参数 / 层数
        per_layer_params_b = model_info.params_b / model_info.num_layers

    # FLOPs per layer (forward = 2 × params × seq_len, decode seq=1)
    flops = 2 * per_layer_params_b * 1e9

    # 量化加速因子
    quant_factor = {"bf16": 1.0, "4bit": 0.5, "8bit": 0.75, "fp8": 0.6}.get(quant, 1.0)

    # 内存带宽限制 (更现实)
    # MoE 需要读取激活的 expert 权重
    mem_bandwidth_gbs = {
        "apple_metal": hardware_info.total_mem_gb * 8,  # ~8GB/s per GB (unified)
        "nvidia": 900,  # RTX 4090 ~900GB/s
        "amd": 800,
        "cpu": 50,
    }.get(hardware_info.gpu_type, 50)

    # 权重读取时间 (受内存带宽限制)
    weight_bytes = per_layer_params_b * 1e9 * (2 if quant == "bf16" else 0.5 if quant == "4bit" else 1)
    mem_time_ms = weight_bytes / (mem_bandwidth_gbs * 1e9) * 1000

    # 计算时间 (受 TFLOPS 限制)
    compute_time_ms = flops / (tflops * 1e12) * 1000 * quant_factor

    # MoE routing 开销
    moe_overhead = 1.3 if is_moe else 1.0

    return max(mem_time_ms, compute_time_ms) * moe_overhead


def compute_route(hardware_info, model_info: ModelInfo) -> RouteDecision:
    """基于 4D 感知矩阵计算路由决策.

    D1: network (RTT)
    D2: hardware (mem, tflops, gpu)
    D3: model (size, layers, moe)
    D4: route decision
    """
    avail_mem = hardware_info.available_mem_gb
    model_size = model_info.model_size_gb
    per_layer = model_info.per_layer_gb
    num_layers = model_info.num_layers
    rtt = hardware_info.rtt_ms
    tflops = hardware_info.tflops
    engine = hardware_info.recommended_engine

    per_layer_ms = estimate_per_layer_ms(hardware_info, model_info)

    # KV cache 估算 (decode, 假设 512 tokens context)
    kv_cache_gb = 0.5  # 保守估算
    embed_gb = 0.2     # embed_tokens
    overhead_gb = 1.0  # 系统/其他

    # === 决策 1: Mac 本地 (离线/隐私, 无 cloud) ===
    if avail_mem >= model_size + kv_cache_gb + embed_gb + overhead_gb:
        # Mac 能放完整模型
        decode_tps = 1000 / (per_layer_ms * num_layers) if per_layer_ms > 0 else 0

        # 比较: 本地 vs 全云
        cloud_ttft = rtt + 10  # cloud prefill
        local_ttft = per_layer_ms * num_layers * 5  # Mac prefill (估算, 5x decode)

        if local_ttft < cloud_ttft * 2 and decode_tps >= 10:
            # 本地够快 → 本地
            return RouteDecision(
                mode="local_only",
                P=num_layers,
                expected_ttft_ms=local_ttft,
                expected_decode_tps=decode_tps,
                cloud_save_pct=100,
                reason=f"本地完整: mem={avail_mem}GB >= {model_size}+{kv_cache_gb}+{overhead_gb}",
                per_layer_forward_ms=per_layer_ms,
                mac_forward_ms=per_layer_ms * num_layers,
                rtt_ms=0,
            )

        # 本地慢 → PD 分离 (cloud prefill + Mac decode)
        return RouteDecision(
            mode="pd_separation",
            P=num_layers,
            expected_ttft_ms=cloud_ttft,
            expected_decode_tps=decode_tps,
            cloud_save_pct=100,
            reason=f"PD分离: cloud prefill ({cloud_ttft:.0f}ms) + Mac decode ({decode_tps:.0f} tok/s)",
            per_layer_forward_ms=per_layer_ms,
            mac_forward_ms=per_layer_ms * num_layers,
            rtt_ms=rtt,
        )

    # === 决策 2: Layer-split (Mac 部分层) ===
    elif avail_mem >= per_layer * 6 + overhead_gb:
        # 计算最大 P
        P_mem = int((avail_mem - overhead_gb - kv_cache_gb) / per_layer)
        P_mem = min(P_mem, num_layers - 1)  # 至少 cloud 做 1 层

        # 算力约束: Mac forward 不超过总延迟的 50%
        target_total_ms = 250  # 目标 250ms/batch
        P_compute = int((target_total_ms - rtt * 2 - 10) / per_layer_ms)
        P_compute = max(P_compute, 6)  # 至少 6 层

        P = min(P_mem, P_compute, num_layers - 1)
        P = max(P, 6)  # 最少 6 层

        mac_forward = P * per_layer_ms
        cloud_forward = (num_layers - P) * 0.1  # cloud 每层 0.1ms (TP=8)
        verify = 3

        # 投机编码估算
        spec_n = 21
        spec_accept = 0.28
        accept_tokens = spec_n * spec_accept

        total_batch = mac_forward + rtt + cloud_forward + verify + rtt
        decode_tps = accept_tokens / total_batch * 1000 if total_batch > 0 else 0

        save_pct = int(P / num_layers * 100)

        return RouteDecision(
            mode="layer_split",
            P=P,
            expected_ttft_ms=rtt + 10,
            expected_decode_tps=decode_tps,
            cloud_save_pct=save_pct,
            reason=f"Layer-split: P={P} (mem={avail_mem}GB, per_layer={per_layer}GB, {per_layer_ms:.1f}ms/层)",
            per_layer_forward_ms=per_layer_ms,
            mac_forward_ms=mac_forward,
            rtt_ms=rtt,
            spec_n=spec_n,
            spec_accept=spec_accept,
        )

    # === 决策 3: 全云 ===
    else:
        return RouteDecision(
            mode="cloud_only",
            P=0,
            expected_ttft_ms=rtt + 10,
            expected_decode_tps=112,  # cloud 30B 估算
            cloud_save_pct=0,
            reason=f"全云: mem={avail_mem}GB < per_layer×6={per_layer*6:.1f}GB",
            rtt_ms=rtt,
        )


def build_4d_matrix(hardware_info, model_info: ModelInfo) -> dict:
    """构建完整 4D 感知矩阵."""
    route = compute_route(hardware_info, model_info)

    return {
        "D1_network": {
            "rtt_ms": hardware_info.rtt_ms,
            "bandwidth_mbps": 0,  # TODO: 实测
        },
        "D2_hardware": {
            "os": hardware_info.os_name,
            "arch": hardware_info.arch,
            "chip": hardware_info.cpu_brand,
            "cores": hardware_info.cpu_cores,
            "total_mem_gb": hardware_info.total_mem_gb,
            "available_mem_gb": hardware_info.available_mem_gb,
            "disk_available_gb": hardware_info.disk_available_gb,
            "gpu_type": hardware_info.gpu_type,
            "gpu_name": hardware_info.gpu_name,
            "gpu_vram_gb": hardware_info.gpu_vram_gb,
            "compute_tier": hardware_info.compute_tier,
            "tflops": hardware_info.tflops,
            "engine": hardware_info.recommended_engine,
        },
        "D3_model": {
            "name": model_info.name,
            "params_b": model_info.params_b,
            "num_layers": model_info.num_layers,
            "is_moe": model_info.is_moe,
            "num_experts": model_info.num_experts,
            "experts_per_tok": model_info.experts_per_tok,
            "hidden_size": model_info.hidden_size,
            "quantization": model_info.quantization,
            "model_size_gb": model_info.model_size_gb,
            "per_layer_gb": model_info.per_layer_gb,
        },
        "D4_route": route.to_dict(),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516")
    from app.shared.hardware_sensing import detect_all

    print("=" * 60)
    print("4D 感知矩阵 + 路由决策")
    print("=" * 60)

    hw = detect_all()
    print(f"\n硬件: {hw.cpu_brand}, {hw.available_mem_gb}GB, {hw.compute_tier}, RTT={hw.rtt_ms}ms")

    for model_key in ["qwen3-vl-2b", "qwen3-vl-2b-4bit", "qwen3-vl-30b-4bit", "deepseek-v4-flash"]:
        model = MODEL_PRESETS[model_key]
        route = compute_route(hw, model)

        print(f"\n  {model.name} ({model.quantization}, {model.model_size_gb}GB):")
        print(f"    → {route.mode} P={route.P}")
        print(f"    TTFT={route.expected_ttft_ms:.0f}ms, decode={route.expected_decode_tps:.1f} tok/s, 省{route.cloud_save_pct}%")
        print(f"    {route.reason}")
