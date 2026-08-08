#!/usr/bin/env python3
"""4D 感知矩阵 v2 Schema -- Hermes 认知路由版.

扩展 v1 (route_decision.py):
  D1: + bandwidth + jitter + stability
  D2: + unified_memory + tflops_int8 + engine 5 options
  D3: + per_layer_gb (by-layer 关键)
  D4: + draft_n + pivot_layer + use_flashmoe + confidence + reason

Hermes 路由模型输入 = FourDMatrixV2.to_dict()
Hermes 路由模型输出 = RouteDecisionV2 (JSON schema 约束)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Literal, Optional


# === D1 Network ===
@dataclass
class D1Network:
    """D1: 网络感知 (实测)."""
    rtt_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    jitter_ms: float = 0.0
    stability: Literal["stable", "unstable", "offline"] = "stable"

    def to_dict(self) -> dict:
        return asdict(self)


# === D2 Hardware ===
@dataclass
class D2Hardware:
    """D2: 硬件感知 (来自 hardware_sensing)."""
    chip: str = ""
    avail_mem_gb: float = 0.0
    total_mem_gb: float = 0.0
    disk_free_gb: float = 0.0
    tflops_fp16: float = 0.0
    tflops_int8: float = 0.0
    engine: Literal["mlx", "cuda", "cpu", "rocm", "omlx"] = "mlx"
    unified_memory: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    compute_tier: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# === D3 Model ===
@dataclass
class D3Model:
    """D3: 模型感知 (来自 model_registry)."""
    name: str = ""
    params_b: float = 0.0
    num_layers: int = 0
    is_moe: bool = False
    num_experts: int = 0
    experts_per_tok: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    quantization: Literal["bf16", "fp8", "int4", "int8"] = "bf16"
    model_size_gb: float = 0.0
    per_layer_gb: float = 0.0              # 关键: 决定 by-layer 切粒度
    has_native_mtp: bool = False           # Tier 0: 原生 MTP
    draft_model_path: str = ""             # Tier 1: 训练的 draft head

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DynamicHeatFeatures:
    """Colibri 风格动态热度特征."""
    expert_hit_rate_ema: float = 0.0
    hot_expert_ratio: float = 0.0
    recent_expert_heat_entropy: float = 0.0
    layer_hotness_topk: list[float] = field(default_factory=list)
    warm_pin_gb: float = 0.0
    repin_recent_count: int = 0
    prefetch_hit_rate_ema: float = 0.0
    predicted_cold_bytes_mb: float = 0.0
    predicted_bytes_to_read_mb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StorageRuntimeFeatures:
    """存储、驻留与 I/O 特征."""
    dense_resident_tier: Literal["memory", "nvme", "mixed", "unknown"] = "unknown"
    expert_resident_tier: Literal["memory", "nvme", "mixed", "unknown"] = "unknown"
    predicted_residency_mode: Literal["full_resident", "warm_resident", "streamed", "hybrid_tier"] = "full_resident"
    can_partial_resident: bool = False
    nvme_bw_gbps: float = 0.0
    io_queue_depth: int = 0
    secondary_nvme_available: bool = False
    disk_mirror_mode: Literal["none", "mirror", "split"] = "none"
    multi_store_read_gain_estimate: float = 0.0
    io_compute_overlap_gain: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpeculationROIFeatures:
    """投机与 grammar/json 路径收益特征."""
    accept_rate_ema: float = 0.0
    verify_cost_ms: float = 0.0
    draft_cost_ms: float = 0.0
    recent_speculation_roi: float = 0.0
    recent_json_success_rate: float = 0.0
    grammar_accept_rate_ema: float = 0.0
    grammar_mode_roi: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RequestContractFeatures:
    """请求合约与结构化输出提示."""
    prompt_has_code: bool = False
    prompt_is_json_task: bool = False
    prompt_is_tool_task: bool = False
    response_contract_hint: Literal["plain", "json", "tool"] = "plain"
    cache_hit_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TransportRouteFeatures:
    """edge_first 静态 transport 估算特征."""
    mode: Literal["local_full", "layer_split_pd", "cloud_pd", "cloud_fallback", "unknown"] = "unknown"
    mode_hint: Literal["local_full", "layer_split_pd", "cloud_pd", "cloud_fallback", "unknown"] = "unknown"
    reason: str = ""
    arch_supported: bool = True
    local_model_configured: bool = False
    prompt_len_est: int = 0
    max_tokens: int = 0
    mac_available_bytes: int = 0
    mac_available_safe_bytes: int = 0
    model_weight_bytes: int = 0
    kv_bytes_est: int = 0
    needed_full_bytes: int = 0
    local_num_layers: int = 0
    per_layer_bytes: float = 0.0
    activation_bytes_per_layer: int = 0
    partial_layer_capacity: int = 0
    latency_split_sec_est: float = 0.0
    latency_cloud_sec_est: float = 0.0
    rtt_sec: float = 0.0
    mac_prefill_sec_est: float = 0.0
    desired_mode: Literal["local_full", "layer_split_pd", "cloud_pd", "cloud_fallback", "unknown"] = "unknown"
    memory_pressure: Literal["normal", "elevated", "critical", "unknown"] = "unknown"
    mode_switch_reason: str = ""
    sticky_active: bool = False
    sticky_until_epoch_ms: int = 0
    sticky_window_sec: float = 0.0
    degrade_suggested: bool = False
    degrade_target_mode: Literal["local_full", "layer_split_pd", "cloud_pd", "cloud_fallback", "unknown"] = "unknown"
    downgrade_chain: list[str] = field(default_factory=list)
    moe_candidate: bool = False
    moe_streaming_admissible: bool = False
    moe_streaming_required_bytes: int = 0
    moe_streaming_headroom_bytes: int = 0
    external_low_memory_detected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryPressureFeatures:
    """运行时内存压力与 MoE streaming 准入特征."""
    memory_pressure: Literal["normal", "elevated", "critical", "unknown"] = "unknown"
    moe_streaming_admissible: bool = False
    moe_streaming_required_bytes: int = 0
    moe_streaming_headroom_bytes: int = 0
    external_low_memory_detected: bool = False
    degrade_suggested: bool = False
    degrade_target_mode: Literal["local_full", "layer_split_pd", "cloud_pd", "cloud_fallback", "unknown"] = "unknown"
    mode_switch_reason: str = ""
    sticky_active: bool = False
    sticky_until_epoch_ms: int = 0
    sticky_window_sec: float = 0.0
    downgrade_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# === D4 Route Decision ===
@dataclass
class RouteDecisionV2:
    """D4: 路由决策 (Hermes 模型输出).

    9 种模式:
      cache_hit              → L1-L5 缓存命中, 直接返回
      local_only             → Mac 本地完整推理 (离线/隐私)
      local_full             → edge_first 本地完整推理
      layer_split_pd         → edge_first 前 P 层本地 + 云端续跑
      cloud_pd               → edge_first 全云预填充/解码
      cloud_fallback         → edge_first 降级全云
      edge_pivot_draft       → 端侧 MTP draft 分层前向抢首包 + 上云 verify
      edge_draft_cloud_verify → 端侧 MTP draft 完整生成 + 上云 verify (无 pivot)
      cloud_only             → 直连云端
    """
    mode: Literal[
        "cache_hit",
        "local_only",
        "local_full",
        "layer_split_pd",
        "cloud_pd",
        "cloud_fallback",
        "edge_pivot_draft",
        "edge_draft_cloud_verify",
        "cloud_only",
    ] = "cloud_only"

    draft_n_tokens: int = 0          # 1-16, MTP chain length
    pivot_layer: int = 0             # 分层前向在哪层抢首包 (0 = 不抢)
    use_flashmoe: bool = False       # Draft 为 MoE 时启用
    draft_model_path: str = ""       # DraftRegistry.get(model_name)
    confidence: float = 0.0          # 0-1, Hermes 自评置信度
    reason: str = ""                 # Hermes 自然语言解释

    # 性能预估
    expected_ttft_ms: float = 0.0
    expected_decode_tps: float = 0.0
    expected_accept_rate: float = 0.0
    speculation_expected_roi: float = 0.0

    # Colibri/Hermes v2 policy fields
    residency_policy: Literal["full_resident", "warm_resident", "streamed", "hybrid_tier"] = "full_resident"
    prefetch_policy: Literal["off", "conservative", "aggressive"] = "off"
    streaming_policy: Literal["buffered", "direct_io", "overlap_io_compute"] = "buffered"
    fallback_policy: Literal["cloud_only", "plain_mtp", "disable_speculation"] = "cloud_only"
    response_contract: Literal["plain", "json", "tool"] = "plain"
    grammar_mode: Literal["off", "json", "schema", "tool_call"] = "off"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "RouteDecisionV2":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> "RouteDecisionV2":
        return cls.from_dict(json.loads(s))

    @classmethod
    def schema_json(cls) -> dict:
        """JSON Schema for constrained generation."""
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": [
                        "cache_hit",
                        "local_only",
                        "local_full",
                        "layer_split_pd",
                        "cloud_pd",
                        "cloud_fallback",
                        "edge_pivot_draft",
                        "edge_draft_cloud_verify",
                        "cloud_only",
                    ],
                },
                "draft_n_tokens": {"type": "integer", "minimum": 0, "maximum": 16},
                "pivot_layer": {"type": "integer", "minimum": 0, "maximum": 64},
                "use_flashmoe": {"type": "boolean"},
                "draft_model_path": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"},
                "expected_ttft_ms": {"type": "number"},
                "expected_decode_tps": {"type": "number"},
                "expected_accept_rate": {"type": "number"},
                "speculation_expected_roi": {"type": "number"},
                "residency_policy": {
                    "type": "string",
                    "enum": ["full_resident", "warm_resident", "streamed", "hybrid_tier"],
                },
                "prefetch_policy": {
                    "type": "string",
                    "enum": ["off", "conservative", "aggressive"],
                },
                "streaming_policy": {
                    "type": "string",
                    "enum": ["buffered", "direct_io", "overlap_io_compute"],
                },
                "fallback_policy": {
                    "type": "string",
                    "enum": ["cloud_only", "plain_mtp", "disable_speculation"],
                },
                "response_contract": {
                    "type": "string",
                    "enum": ["plain", "json", "tool"],
                },
                "grammar_mode": {
                    "type": "string",
                    "enum": ["off", "json", "schema", "tool_call"],
                },
            },
            "required": ["mode", "draft_n_tokens", "pivot_layer", "use_flashmoe",
                         "confidence", "reason"],
            "additionalProperties": False,
        }


# === 完整 4D 矩阵 ===
@dataclass
class FourDMatrixV2:
    """4D 感知矩阵 v2 -- Hermes 路由输入."""
    D1: D1Network = field(default_factory=D1Network)
    D2: D2Hardware = field(default_factory=D2Hardware)
    D3: D3Model = field(default_factory=D3Model)
    D4: RouteDecisionV2 = field(default_factory=RouteDecisionV2)
    heat: DynamicHeatFeatures = field(default_factory=DynamicHeatFeatures)
    storage_runtime: StorageRuntimeFeatures = field(default_factory=StorageRuntimeFeatures)
    speculation_roi: SpeculationROIFeatures = field(default_factory=SpeculationROIFeatures)
    request_contract: RequestContractFeatures = field(default_factory=RequestContractFeatures)
    transport_route: TransportRouteFeatures = field(default_factory=TransportRouteFeatures)
    memory_pressure: MemoryPressureFeatures = field(default_factory=MemoryPressureFeatures)

    # 上下文信息 (非 4D, 但影响决策)
    prompt_preview: str = ""          # prompt 前 200 字符
    prompt_has_code: bool = False     # 是否包含代码
    history_accept_rate: float = 0.0  # 最近 accept rate (AcceptanceTracker)
    cache_hit_rate: float = 0.0       # 最近缓存命中率

    def to_dict(self) -> dict:
        return {
            "D1_network": self.D1.to_dict(),
            "D2_hardware": self.D2.to_dict(),
            "D3_model": self.D3.to_dict(),
            "D4_route": self.D4.to_dict(),
            "heat": self.heat.to_dict(),
            "storage_runtime": self.storage_runtime.to_dict(),
            "speculation_roi": self.speculation_roi.to_dict(),
            "request_contract": self.request_contract.to_dict(),
            "transport_route": self.transport_route.to_dict(),
            "memory_pressure": self.memory_pressure.to_dict(),
            "context": {
                "prompt_preview": self.prompt_preview[:200],
                "prompt_has_code": self.prompt_has_code,
                "history_accept_rate": self.history_accept_rate,
                "cache_hit_rate": self.cache_hit_rate,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_hardware_model(
        cls,
        hardware_info,
        model_info,
        prompt: str = "",
        history_accept_rate: float = 0.0,
        cache_hit_rate: float = 0.0,
        dynamic_heat: Optional[dict[str, Any]] = None,
        storage_runtime: Optional[dict[str, Any]] = None,
        speculation_roi: Optional[dict[str, Any]] = None,
        request_contract: Optional[dict[str, Any]] = None,
        transport_route: Optional[dict[str, Any]] = None,
        memory_pressure: Optional[dict[str, Any]] = None,
    ) -> "FourDMatrixV2":
        """从现有 hardware_sensing + model_info 构建 v2 矩阵.

        兼容 v1 的 HardwareInfo / ModelInfo 接口.
        """
        d1 = D1Network(
            rtt_ms=getattr(hardware_info, "rtt_ms", 0.0),
            bandwidth_mbps=getattr(hardware_info, "bandwidth_mbps", 0.0),
            jitter_ms=getattr(hardware_info, "jitter_ms", 0.0),
            stability="stable" if getattr(hardware_info, "rtt_ms", 0) < 200 else "unstable",
        )

        d2 = D2Hardware(
            chip=getattr(hardware_info, "cpu_brand", ""),
            avail_mem_gb=getattr(hardware_info, "available_mem_gb", 0.0),
            total_mem_gb=getattr(hardware_info, "total_mem_gb", 0.0),
            disk_free_gb=getattr(hardware_info, "disk_available_gb", 0.0),
            tflops_fp16=getattr(hardware_info, "tflops", 0.0),
            tflops_int8=getattr(hardware_info, "tflops", 0.0) * 2,  # int8 ≈ 2x fp16
            engine=getattr(hardware_info, "recommended_engine", "mlx"),
            unified_memory="apple" in getattr(hardware_info, "gpu_type", "").lower(),
            gpu_name=getattr(hardware_info, "gpu_name", ""),
            gpu_vram_gb=getattr(hardware_info, "gpu_vram_gb", 0.0),
            compute_tier=getattr(hardware_info, "compute_tier", ""),
        )

        d3 = D3Model(
            name=getattr(model_info, "name", ""),
            params_b=getattr(model_info, "params_b", 0.0),
            num_layers=getattr(model_info, "num_layers", 0),
            is_moe=getattr(model_info, "is_moe", False),
            num_experts=getattr(model_info, "num_experts", 0),
            experts_per_tok=getattr(model_info, "experts_per_tok", 0),
            hidden_size=getattr(model_info, "hidden_size", 0),
            vocab_size=getattr(model_info, "vocab_size", 0),
            quantization=getattr(model_info, "quantization", "bf16"),
            model_size_gb=getattr(model_info, "model_size_gb", 0.0),
            per_layer_gb=getattr(model_info, "per_layer_gb", 0.0),
            has_native_mtp=getattr(model_info, "has_native_mtp", False),
            draft_model_path=getattr(model_info, "draft_model_path", ""),
        )

        # 简单代码检测
        has_code = any(kw in prompt.lower() for kw in [
            "def ", "class ", "import ", "function ", "const ", "var ",
            "```", "    ", "\t", "return ", "if ", "for ", "while ",
        ])
        lowered_prompt = prompt.lower()
        prompt_is_json_task = any(kw in lowered_prompt for kw in [
            "json", "json only", "strict json", "json_schema",
        ])
        prompt_is_tool_task = any(kw in lowered_prompt for kw in [
            "tool", "function call", "tool call",
        ])

        request_contract_features = RequestContractFeatures(
            prompt_has_code=has_code,
            prompt_is_json_task=prompt_is_json_task,
            prompt_is_tool_task=prompt_is_tool_task,
            response_contract_hint="tool" if prompt_is_tool_task else ("json" if prompt_is_json_task else "plain"),
            cache_hit_rate=cache_hit_rate,
        )
        if request_contract:
            for key, value in request_contract.items():
                if hasattr(request_contract_features, key):
                    setattr(request_contract_features, key, value)

        return cls(
            D1=d1,
            D2=d2,
            D3=d3,
            heat=DynamicHeatFeatures(**{
                k: v for k, v in (dynamic_heat or {}).items()
                if k in DynamicHeatFeatures.__dataclass_fields__
            }),
            storage_runtime=StorageRuntimeFeatures(**{
                k: v for k, v in (storage_runtime or {}).items()
                if k in StorageRuntimeFeatures.__dataclass_fields__
            }),
            speculation_roi=SpeculationROIFeatures(**{
                k: v for k, v in (speculation_roi or {}).items()
                if k in SpeculationROIFeatures.__dataclass_fields__
            }),
            request_contract=request_contract_features,
            transport_route=TransportRouteFeatures(**{
                k: v for k, v in (transport_route or {}).items()
                if k in TransportRouteFeatures.__dataclass_fields__
            }),
            memory_pressure=MemoryPressureFeatures(**{
                k: v for k, v in (memory_pressure or {}).items()
                if k in MemoryPressureFeatures.__dataclass_fields__
            }),
            prompt_preview=prompt[:200],
            prompt_has_code=has_code,
            history_accept_rate=history_accept_rate,
            cache_hit_rate=cache_hit_rate,
        )


def transport_route_report_contract() -> dict[str, Any]:
    return {
        "mode": "unknown",
        "mode_hint": "unknown",
        "desired_mode": "unknown",
        "reason": "",
        "mode_switch_reason": "",
        "memory_pressure": "unknown",
        "degrade_suggested": False,
        "degrade_target_mode": "unknown",
        "downgrade_chain": [],
        "sticky_active": False,
        "sticky_until_epoch_ms": 0,
        "sticky_window_sec": 0.0,
        "moe_candidate": False,
        "moe_streaming_admissible": False,
        "moe_streaming_required_bytes": 0,
        "moe_streaming_headroom_bytes": 0,
        "external_low_memory_detected": False,
    }


def route_policy_v2_report_contract() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "feature_schema_version": "v2",
        "policy_schema_version": "v2",
        "feature_schema": {},
        "transport_runtime": {},
        "transport_route": transport_route_report_contract(),
        "hermes_policy": {},
        "final_policy": {},
        "guard_overrides": [],
    }


def route_heat_snapshot_report_contract() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "expert_hit_rate_ema": 0.0,
        "hot_expert_ratio": 0.0,
        "warm_pin_gb": 0.0,
        "predicted_bytes_to_read_mb": 0.0,
        "predicted_cold_bytes_mb": 0.0,
        "prefetch_hit_rate_ema": 0.0,
        "transport_runtime": {},
        "transport_route": transport_route_report_contract(),
        "frontier_thread": {
            "execution_success": False,
            "content_success": False,
        },
        "expert_data_plane": {},
        "storage_topology": {},
    }


def draft_mode_acceptance_report_contract() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "draft_mode": "off",
        "response_contract": "plain",
        "accept_rate_ema": 0.0,
        "grammar_accept_rate_ema": 0.0,
        "json_success_rate": 0.0,
        "recent_speculation_roi": 0.0,
        "auto_disabled": False,
        "disable_reason": "",
    }


def single_node_candidate_matrix_report_contract() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "production_target": {},
        "hardware_profile": {},
        "transport_runtime": {},
        "transport_route": transport_route_report_contract(),
        "routing_summary": {
            "routing_status": "UNKNOWN",
            "admissible_degrade": False,
            "transport_route_mode": "unknown",
            "degrade_target_mode": "",
            "mode_switch_reason": "",
            "memory_pressure": "unknown",
        },
        "frontier_thread": {
            "execution_success": False,
            "content_success": False,
        },
        "expert_data_plane": {},
        "acceptance_summary": {
            "cold_path": {},
            "warm_hot_path": {},
            "path_bottlenecks": {
                "cold_path": {},
                "warm_hot_path": {},
            },
        },
        "production_gates": {
            "cold_path": {},
            "warm_hot_path": {},
            "shared": {},
        },
        "production_readiness": {
            "routing_status": "UNKNOWN",
            "admissible_degrade": False,
            "cold_path_failed_gates": [],
            "cold_path_failed_gate_summaries": [],
            "failed_gates": [],
            "failed_gate_summaries": [],
        },
        "candidates": [
            {
                "model": "",
                "priority": "P0",
                "status": "PARTIAL",
                "ttft_ms": 0.0,
                "decode_tps": 0.0,
                "stable_rounds_passed": 0,
                "notes": "",
            }
        ],
    }


def report_contracts() -> dict[str, dict[str, Any]]:
    return {
        "route_policy_v2": route_policy_v2_report_contract(),
        "route_heat_snapshot": route_heat_snapshot_report_contract(),
        "draft_mode_acceptance": draft_mode_acceptance_report_contract(),
        "single_node_candidate_matrix": single_node_candidate_matrix_report_contract(),
    }


# === 规则引擎兼容层 (Hermes 不可用时的 fallback) ===
def rule_based_decision_v2(matrix: FourDMatrixV2) -> RouteDecisionV2:
    """规则引擎 fallback: 当 Hermes 不可用时用规则生成决策.

    逻辑继承 route_decision.compute_route() 但输出 v2 格式.
    """
    d1 = matrix.D1
    d2 = matrix.D2
    d3 = matrix.D3
    heat = matrix.heat
    roi = matrix.speculation_roi
    request_contract = matrix.request_contract
    memory_pressure = matrix.memory_pressure

    if memory_pressure.degrade_suggested and memory_pressure.degrade_target_mode in {
        "local_full",
        "layer_split_pd",
        "cloud_pd",
        "cloud_fallback",
    }:
        downgraded_mode = str(memory_pressure.degrade_target_mode)
        downgraded_reason = str(memory_pressure.mode_switch_reason or "memory_pressure_degrade")
        return RouteDecisionV2(
            mode=downgraded_mode,  # type: ignore[arg-type]
            draft_n_tokens=0 if downgraded_mode != "layer_split_pd" else 4,
            pivot_layer=0,
            confidence=0.9,
            reason=f"内存动态模式机: {downgraded_reason}",
            expected_ttft_ms=0.0,
            expected_decode_tps=0.0,
            residency_policy="streamed" if matrix.D3.is_moe else "warm_resident",
            prefetch_policy="conservative",
            streaming_policy="overlap_io_compute" if matrix.D3.is_moe else "buffered",
            fallback_policy="cloud_only" if downgraded_mode in {"cloud_pd", "cloud_fallback"} else "plain_mtp",
            response_contract=request_contract.response_contract_hint,
            grammar_mode="json" if request_contract.response_contract_hint == "json" else ("tool_call" if request_contract.response_contract_hint == "tool" else "off"),
        )

    # 离线
    if d1.stability == "offline":
        return RouteDecisionV2(
            mode="local_only",
            draft_n_tokens=0,
            pivot_layer=0,
            confidence=0.95,
            reason="离线模式: 本地完整推理",
            expected_ttft_ms=500,
            expected_decode_tps=26,
            residency_policy="warm_resident" if d3.is_moe else "full_resident",
            prefetch_policy="aggressive" if heat.predicted_bytes_to_read_mb > 0 else "off",
            streaming_policy="overlap_io_compute" if heat.predicted_bytes_to_read_mb > 0 else "buffered",
            fallback_policy="plain_mtp",
            response_contract=request_contract.response_contract_hint,
            grammar_mode="json" if request_contract.response_contract_hint == "json" else ("tool_call" if request_contract.response_contract_hint == "tool" else "off"),
        )

    # 缓存命中 (由 proxy 层判断, 这里只是 placeholder)
    # 实际 cache_hit 由 edge_first_proxy 直接返回, 不走 Hermes

    # 弱网: 有 draft 但网络差 → edge_pivot_draft (本地抢首包)
    if d1.rtt_ms > 200 and d3.draft_model_path:
        return RouteDecisionV2(
            mode="edge_pivot_draft",
            draft_n_tokens=8,
            pivot_layer=6,
            use_flashmoe=d3.is_moe,
            draft_model_path=d3.draft_model_path,
            confidence=0.85,
            reason=f"弱网 (RTT={d1.rtt_ms:.0f}ms): 端侧抢首包 + draft",
            expected_ttft_ms=40 + d1.rtt_ms * 0.1,
            expected_decode_tps=420,
            expected_accept_rate=0.75,
            speculation_expected_roi=max(roi.recent_speculation_roi, 0.1),
            residency_policy="streamed" if d3.is_moe else "warm_resident",
            prefetch_policy="aggressive",
            streaming_policy="overlap_io_compute",
            fallback_policy="disable_speculation" if roi.recent_speculation_roi < 0 else "plain_mtp",
            response_contract=request_contract.response_contract_hint,
            grammar_mode="json" if request_contract.response_contract_hint == "json" else ("tool_call" if request_contract.response_contract_hint == "tool" else "off"),
        )

    # 正常网络 + 有 draft → edge_draft_cloud_verify
    if d3.draft_model_path and d1.rtt_ms < 100:
        return RouteDecisionV2(
            mode="edge_draft_cloud_verify",
            draft_n_tokens=8,
            pivot_layer=0,
            use_flashmoe=d3.is_moe,
            draft_model_path=d3.draft_model_path,
            confidence=0.90,
            reason=f"正常网络 (RTT={d1.rtt_ms:.0f}ms): draft + 云端 verify",
            expected_ttft_ms=d1.rtt_ms + 15,
            expected_decode_tps=500,
            expected_accept_rate=0.80,
            speculation_expected_roi=max(roi.recent_speculation_roi, 0.15),
            residency_policy="streamed" if d3.is_moe else "warm_resident",
            prefetch_policy="aggressive" if heat.prefetch_hit_rate_ema >= 0.3 else "conservative",
            streaming_policy="overlap_io_compute",
            fallback_policy="disable_speculation" if roi.recent_speculation_roi < 0 else "plain_mtp",
            response_contract=request_contract.response_contract_hint,
            grammar_mode="json" if request_contract.response_contract_hint == "json" else ("tool_call" if request_contract.response_contract_hint == "tool" else "off"),
        )

    # 有 draft + 中等网络 → edge_pivot_draft
    if d3.draft_model_path:
        return RouteDecisionV2(
            mode="edge_pivot_draft",
            draft_n_tokens=8,
            pivot_layer=6,
            use_flashmoe=d3.is_moe,
            draft_model_path=d3.draft_model_path,
            confidence=0.82,
            reason=f"中等网络 (RTT={d1.rtt_ms:.0f}ms): 抢首包 + draft verify",
            expected_ttft_ms=50,
            expected_decode_tps=450,
            expected_accept_rate=0.78,
            speculation_expected_roi=max(roi.recent_speculation_roi, 0.1),
            residency_policy="streamed" if d3.is_moe else "warm_resident",
            prefetch_policy="conservative",
            streaming_policy="overlap_io_compute" if heat.predicted_bytes_to_read_mb > 0 else "buffered",
            fallback_policy="disable_speculation" if roi.recent_speculation_roi < 0 else "plain_mtp",
            response_contract=request_contract.response_contract_hint,
            grammar_mode="json" if request_contract.response_contract_hint == "json" else ("tool_call" if request_contract.response_contract_hint == "tool" else "off"),
        )

    # 无 draft → cloud_only
    return RouteDecisionV2(
        mode="cloud_only",
        draft_n_tokens=0,
        pivot_layer=0,
        confidence=0.95,
        reason="无 draft model: 直连云端",
        expected_ttft_ms=d1.rtt_ms + 10,
        expected_decode_tps=273,
        expected_accept_rate=0.0,
        speculation_expected_roi=0.0,
        residency_policy="streamed" if d3.is_moe else "full_resident",
        prefetch_policy="off",
        streaming_policy="buffered",
        fallback_policy="cloud_only",
        response_contract=request_contract.response_contract_hint,
        grammar_mode="json" if request_contract.response_contract_hint == "json" else ("tool_call" if request_contract.response_contract_hint == "tool" else "off"),
    )


if __name__ == "__main__":
    # 自测
    from app.shared.route_decision import MODEL_PRESETS
    from app.shared.hardware_sensing import detect_all

    hw = detect_all()
    model = MODEL_PRESETS["qwen3-vl-2b-4bit"]

    matrix = FourDMatrixV2.from_hardware_model(hw, model, prompt="def hello():\n    print('hi')")
    print("4D Matrix V2:")
    print(matrix.to_json())

    decision = rule_based_decision_v2(matrix)
    print("\nRule-based decision:")
    print(decision.to_json())

    # JSON Schema
    print("\nRouteDecisionV2 JSON Schema:")
    print(json.dumps(RouteDecisionV2.schema_json(), indent=2))
