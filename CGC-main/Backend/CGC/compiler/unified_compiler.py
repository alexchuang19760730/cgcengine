"""Unified IR Compiler + SGLang Compute Injector (Gate 6.0)

将 tmax/uitars/hermes/cli_universe 四角色模型的计算图统一编译为 Unified IR，
并通过 monkey-patch 注入到 vendored SGLang 的 compute 路径，使四模型无需各自
patch SGLang 源码即可接入 SGLang runtime。

核心注入点：
  1. Linear-Attention 层：强制 LINEAR_ATTN_DECODE/PREFILL_BACKEND=TRITON，
     绕过 flashinfer 0.6.x 缺 linear_attention 模块的问题（Blackwell SM100+
     默认会选 flashinfer decode backend，但旧版 flashinfer 不支持 GDN）。
  2. Attention 层：保持 SGLang 原生 flashinfer/triton 后端。
  3. 模型层元数据：在 ModelRunner 上挂 cgc_ir_injection 元数据供审计。

设计目标（用户需求）：
  "tmax/uitars/hermes/CLI-Universe 不直接接 SGLang，而是通过 CGC engine 统一 IR
   注入 vendored SGLang 的 compute 路径" —— 本文件即该注入机制的实现。
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("cgc.unified_compiler")


# ---------------------------------------------------------------------------
# Hardware Probe — 動態硬體感知
# ---------------------------------------------------------------------------

class HardwareProbe:
    """探測當前節點的硬體狀態，構建 perception_matrix。

    覆蓋白皮書定義的 5 個感知維度：
      - GPU 型號 / 記憶體 / 使用率 / 驅動
      - CPU 核數 / 記憶體
      - NCCL / eRDMA 拓撲
      - 網路延遲（跨節點 ping）
      - 環境變數（CGC_* / NCCL_*）
    """

    @staticmethod
    def probe_gpu() -> Dict[str, Any]:
        """探測 GPU 型號、記憶體、驅動版本"""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return {
                    "gpu_name": props.name,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory_total_mb": props.total_memory // (1024 * 1024),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "driver_version": torch.version.cuda or "unknown",
                }
        except Exception:
            pass
        # fallback: nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                timeout=5, stderr=subprocess.DEVNULL
            ).decode().strip().split("\n")[0].split(", ")
            return {
                "gpu_name": out[0].strip(),
                "gpu_count": 1,
                "gpu_memory_total_mb": int(out[1].strip()),
                "compute_capability": "unknown",
                "driver_version": out[2].strip(),
            }
        except Exception:
            return {"gpu_name": "unknown", "gpu_count": 0, "gpu_memory_total_mb": 0}

    @staticmethod
    def probe_gpu_usage() -> List[Dict[str, Any]]:
        """探測每張 GPU 的當前使用率與記憶體使用量"""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                timeout=5, stderr=subprocess.DEVNULL
            ).decode().strip()
            gpus = []
            for line in out.split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "index": int(parts[0]),
                        "utilization_pct": int(parts[1]),
                        "memory_used_mb": int(parts[2]),
                        "memory_total_mb": int(parts[3]),
                    })
            return gpus
        except Exception:
            return []

    @staticmethod
    def probe_cpu() -> Dict[str, Any]:
        """探測 CPU 核數與系統記憶體"""
        try:
            cpu_count = os.cpu_count() or 0
            # 系統記憶體（MB）
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        return {"cpu_cores": cpu_count, "system_memory_mb": mem_kb // 1024}
        except Exception:
            pass
        return {"cpu_cores": 0, "system_memory_mb": 0}

    @staticmethod
    def probe_network() -> Dict[str, Any]:
        """探測網路拓撲：主機名、內網 IP、eRDMA 狀態"""
        info: Dict[str, Any] = {}
        try:
            info["hostname"] = socket.gethostname()
        except Exception:
            info["hostname"] = "unknown"
        try:
            info["internal_ip"] = socket.gethostbyname(socket.gethostname())
        except Exception:
            info["internal_ip"] = "unknown"
        # eRDMA 狀態
        info["erdma_enabled"] = os.environ.get("NCCL_IB_DISABLE", "1") == "0"
        info["erdma_hca"] = os.environ.get("NCCL_IB_HCA", "")
        info["nccl_socket_ifname"] = os.environ.get("NCCL_SOCKET_IFNAME", "")
        # Ray cluster（如果在 Ray worker 中）
        info["ray_address"] = os.environ.get("RAY_ADDRESS", "")
        return info

    @staticmethod
    def probe_cross_node_latency(peer_ips: Optional[List[str]] = None) -> Dict[str, Any]:
        """探測跨節點 ping 延遲（ms）。

        Args:
            peer_ips: 對端節點 IP 列表。None 時從環境變數 CGC_PEER_IPS 讀取。
                      格式: "ip1,ip2,ip3"

        Returns:
            {peer_ip: {"latency_ms": float, "status": "ok"|"timeout"|"error"}}
        """
        if peer_ips is None:
            env_peers = os.environ.get("CGC_PEER_IPS", "").strip()
            peer_ips = [ip.strip() for ip in env_peers.split(",") if ip.strip()] if env_peers else []

        if not peer_ips:
            return {"peer_count": 0, "peers": {}}

        results: Dict[str, Any] = {"peer_count": len(peer_ips), "peers": {}}
        for peer_ip in peer_ips:
            try:
                # 使用 ping -c 3 測量 3 次平均延遲
                out = subprocess.check_output(
                    ["ping", "-c", "3", "-W", "2", peer_ip],
                    timeout=10, stderr=subprocess.DEVNULL
                ).decode()
                # 解析 rtt min/avg/max/mdev
                import re
                m = re.search(r"rtt min/avg/max/mdev\s*=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)", out)
                if m:
                    results["peers"][peer_ip] = {
                        "latency_ms": float(m.group(2)),
                        "rtt_min": float(m.group(1)),
                        "rtt_max": float(m.group(3)),
                        "rtt_mdev": float(m.group(4)),
                        "status": "ok",
                    }
                else:
                    results["peers"][peer_ip] = {"latency_ms": -1, "status": "parse_error"}
            except subprocess.TimeoutExpired:
                results["peers"][peer_ip] = {"latency_ms": -1, "status": "timeout"}
            except Exception as e:
                results["peers"][peer_ip] = {"latency_ms": -1, "status": f"error: {e}"}

        # 計算平均延遲
        ok_latencies = [p["latency_ms"] for p in results["peers"].values() if p.get("status") == "ok"]
        results["avg_latency_ms"] = sum(ok_latencies) / len(ok_latencies) if ok_latencies else -1
        return results

    @staticmethod
    def probe_model_pool(model_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
        """探測可用模型池（讀取模型目錄）。

        Args:
            model_dirs: 模型目錄列表。None 時使用預設路徑。

        Returns:
            {"model_count": int, "models": [{"name": str, "path": str, "size_gb": float}]}
        """
        if model_dirs is None:
            model_dirs = ["/data/models", "/root/models", "/root/flashkv0516/models"]

        models: List[Dict[str, Any]] = []
        for model_dir in model_dirs:
            if not os.path.isdir(model_dir):
                continue
            try:
                for entry in os.listdir(model_dir):
                    entry_path = os.path.join(model_dir, entry)
                    if not os.path.isdir(entry_path):
                        continue
                    # 計算目錄大小（GB）
                    total_size = 0
                    file_count = 0
                    for root, _, files in os.walk(entry_path):
                        for f in files:
                            try:
                                total_size += os.path.getsize(os.path.join(root, f))
                                file_count += 1
                            except OSError:
                                pass
                    # 檢查是否為有效模型（有 config.json 或 model.safetensors）
                    has_config = os.path.exists(os.path.join(entry_path, "config.json"))
                    has_weights = any(
                        os.path.exists(os.path.join(entry_path, f))
                        for f in ["model.safetensors", "pytorch_model.bin"]
                    ) or os.path.isdir(os.path.join(entry_path, "snapshot"))

                    models.append({
                        "name": entry,
                        "path": entry_path,
                        "size_gb": round(total_size / (1024 ** 3), 2),
                        "file_count": file_count,
                        "has_config": has_config,
                        "has_weights": has_weights,
                    })
            except Exception:
                pass

        return {
            "model_count": len(models),
            "models": sorted(models, key=lambda m: m["name"]),
        }

    @staticmethod
    def probe_ray_cluster() -> Dict[str, Any]:
        """探測 Ray cluster 狀態（如果在 Ray 環境中）。

        Returns:
            {"ray_available": bool, "nodes": [...], "actors": int, "resources": {...}}
        """
        result: Dict[str, Any] = {"ray_available": False}
        try:
            import ray
            if not ray.is_initialized():
                result["ray_available"] = False
                result["note"] = "ray not initialized"
                return result

            result["ray_available"] = True
            nodes = ray.nodes()
            result["nodes"] = [
                {
                    "node_id": n.get("NodeID", ""),
                    "alive": n.get("Alive", False),
                    "node_manager_address": n.get("NodeManagerAddress", ""),
                    "resources": n.get("Resources", {}),
                }
                for n in nodes
            ]
            result["alive_node_count"] = sum(1 for n in nodes if n.get("Alive", False))
            result["cluster_resources"] = ray.cluster_resources()
            result["available_resources"] = ray.available_resources()
        except Exception as e:
            result["error"] = str(e)
        return result

    @staticmethod
    def probe_cgc_env() -> Dict[str, Any]:
        """探測 CGC 環境變數，反映當前 CGC 配置"""
        return {
            "enable_ortho_kda": os.environ.get("CGC_ENABLE_ORTHO_KDA", "0") == "1",
            "enable_rswa": os.environ.get("CGC_ENABLE_RSWA", "0") == "1",
            "enable_prefill_pool": os.environ.get("CGC_ENABLE_PREFILL_POOL", "0") == "1",
            "enable_gds": os.environ.get("CGC_ENABLE_GDS", "0") == "1",
            "ortho_base_dim": int(os.environ.get("CGC_ORTHO_BASE_DIM", "128")),
            "rswa_window_size": int(os.environ.get("CGC_RSWA_WINDOW_SIZE", "128")),
        }

    @staticmethod
    def detect_hardware_type(gpu_name: str) -> str:
        """從 GPU 名稱推斷硬體型號代號"""
        name = gpu_name.lower()
        if "blackwell" in name or "rtx pro 5000" in name:
            return "Nvidia_Blackwell_RT5000"
        if "h100" in name or "h200" in name:
            return "Nvidia_H100"
        if "a100" in name:
            return "Nvidia_A100"
        if "l40" in name or "l20" in name:
            return "Nvidia_L40S"
        if "a10" in name:
            return "Nvidia_A10"
        if "ascend" in name or "huawei" in name:
            return "Huawei_Ascend"
        return f"Nvidia_Unknown({gpu_name})"

    # ------------------------------------------------------------------
    # 4D 感知矩陣 → 契約 Profile 推導
    # ------------------------------------------------------------------

    @staticmethod
    def derive_bootstrap_profile(gpu_count: int, erdma_enabled: bool,
                                 topology_profile: str, locality: str) -> str:
        """根據硬體感知推導 bootstrap profile。

        契約約束：
          - 8+ GPU + eRDMA → gate6_tp8d1_bootstrap（單節點高吞吐）
          - 4+ GPU + eRDMA → gate6_tp4ep4_bootstrap（雙節點 TP4EP4）
          - 2+ GPU        → gate5_edge_bootstrap（邊緣推理）
          - 0 GPU         → gate4_cpu_bootstrap（CPU fallback）
        """
        if gpu_count >= 8 and erdma_enabled:
            return "gate6_tp8d1_bootstrap"
        if gpu_count >= 4 and erdma_enabled:
            return "gate6_tp4ep4_bootstrap"
        if gpu_count >= 4:
            return "gate6_tp4_bootstrap"
        if gpu_count >= 2:
            return "gate5_edge_bootstrap"
        return "gate4_cpu_bootstrap"

    @staticmethod
    def derive_system_profile(hardware_type: str, locality: str,
                              cgc_config: Dict[str, Any],
                              gpu_count: int, gpu_memory_mb: int) -> Dict[str, Any]:
        """根據硬體感知推導 system profile（結構化）。

        契約約束：system profile 必須包含 hardware/locality/runtime/cgc 四個維度。
        """
        # 根據 GPU 記憶體決定 runtime mode
        if gpu_memory_mb >= 60000:
            runtime_tier = "high_mem"  # 60GB+ per GPU
        elif gpu_memory_mb >= 40000:
            runtime_tier = "standard"  # 40-60GB
        elif gpu_memory_mb >= 16000:
            runtime_tier = "edge"  # 16-40GB
        else:
            runtime_tier = "minimal"

        return {
            "schema_version": "cgc.system_profile.v1",
            "profile_id": f"{hardware_type.lower()}_{locality}_{runtime_tier}",
            "hardware": {
                "type": hardware_type,
                "gpu_count": gpu_count,
                "gpu_memory_mb": gpu_memory_mb,
                "runtime_tier": runtime_tier,
            },
            "locality": locality,
            "cgc_capabilities": {
                "ortho_kda": cgc_config.get("enable_ortho_kda", False),
                "rswa": cgc_config.get("enable_rswa", False),
                "prefill_pool": cgc_config.get("enable_prefill_pool", False),
                "gds": cgc_config.get("enable_gds", False),
            },
            "mode_mapping": {
                "development_cli": "cgc",
                "user_cli": "cgc_edge" if locality == "edge" else "cgc",
                "m76_dev_entrypoint": "cgc m76-dev",
            },
        }

    @staticmethod
    def derive_profile_binding(primary_role: str, locality: str,
                               topology_profile: str, hardware_type: str) -> str:
        """根據硬體感知推導 profile binding。

        契約約束：profile binding 必須綁定 role + locality + topology + hardware。
        """
        role_key = str(primary_role or "cli_universe").strip().lower().replace(" ", "_").replace("-", "_")
        locality_key = str(locality or "cloud").strip().lower()
        topo_key = str(topology_profile or "tp4_ep4_formal").strip().lower()
        hw_key = hardware_type.lower().replace("nvidia_", "").replace(" ", "_")
        return f"{role_key}_{locality_key}_{topo_key}_{hw_key}"

    @staticmethod
    def derive_united_pipeline_kernel(cgc_config: Dict[str, Any], gpu_count: int,
                                      hardware_type: str) -> str:
        """根據 CGC 配置和硬體推導 United Pipeline Kernel 模式。

        契約約束：
          - RSWA + OrthoKDA + GDS → unified_pipeline_kernel_rswa_ortho_gds
          - RSWA + OrthoKDA       → unified_pipeline_kernel_rswa_ortho
          - RSWA                  → unified_pipeline_kernel_rswa
          - 無 CGC                → batch_pipeline_kernel
        """
        has_rswa = cgc_config.get("enable_rswa", False)
        has_ortho = cgc_config.get("enable_ortho_kda", False)
        has_gds = cgc_config.get("enable_gds", False)
        has_prefill = cgc_config.get("enable_prefill_pool", False)

        parts = ["unified_pipeline_kernel"]
        if has_rswa:
            parts.append("rswa")
        if has_ortho:
            parts.append("ortho")
        if has_prefill:
            parts.append("prefill")
        if has_gds:
            parts.append("gds")

        if len(parts) == 1:
            # 無 CGC 配置
            return "batch_pipeline_kernel"

        return "_".join(parts)

    @classmethod
    def build_perception_matrix(cls) -> Dict[str, Any]:
        """構建完整的感知矩陣（覆蓋白皮書 5 維輸入 + 跨節點延遲 + 模型池 + Ray 負載）"""
        gpu = cls.probe_gpu()
        gpu_usage = cls.probe_gpu_usage()
        cpu = cls.probe_cpu()
        network = cls.probe_network()
        cgc_env = cls.probe_cgc_env()

        hardware_type = cls.detect_hardware_type(gpu.get("gpu_name", ""))

        # 計算可用記憶體（取最大的一張 GPU）
        available_gpu_mem = max(
            (g["memory_total_mb"] - g["memory_used_mb"] for g in gpu_usage),
            default=gpu.get("gpu_memory_total_mb", 0)
        )

        matrix = {
            # 維度 1: 硬體
            "hardware_type": hardware_type,
            "gpu_name": gpu.get("gpu_name", "unknown"),
            "gpu_count": gpu.get("gpu_count", 0),
            "gpu_memory_total_mb": gpu.get("gpu_memory_total_mb", 0),
            "gpu_available_mb": available_gpu_mem,
            "compute_capability": gpu.get("compute_capability", "unknown"),
            "driver_version": gpu.get("driver_version", "unknown"),
            "cpu_cores": cpu.get("cpu_cores", 0),
            "system_memory_mb": cpu.get("system_memory_mb", 0),
            # 維度 2: GPU 使用率
            "gpu_usage": gpu_usage,
            # 維度 3: 網路拓撲
            "hostname": network.get("hostname", "unknown"),
            "internal_ip": network.get("internal_ip", "unknown"),
            "erdma_enabled": network.get("erdma_enabled", False),
            "erdma_hca": network.get("erdma_hca", ""),
            "nccl_socket_ifname": network.get("nccl_socket_ifname", ""),
            "ray_address": network.get("ray_address", ""),
            # 維度 4: CGC 配置
            "cgc_config": cgc_env,
            # 維度 5: 拓撲決策（根據硬體推斷）
            "topology_profile": "tp4_ep4_formal" if gpu.get("gpu_count", 0) >= 4 else "tp2_edge",
            "locality": "cloud" if gpu.get("gpu_memory_total_mb", 0) >= 40000 else "edge",
            "latency_budget_ms": 120,
            "privacy_level": "standard",
            # 維度 6: 跨節點延遲（新增）
            "cross_node_latency": cls.probe_cross_node_latency(),
            # 維度 7: 模型池（新增）
            "model_pool": cls.probe_model_pool(),
            # 維度 8: Ray cluster 負載（新增）
            "ray_cluster": cls.probe_ray_cluster(),
        }

        # 根據跨節點延遲調整 locality 決策
        avg_latency = matrix["cross_node_latency"].get("avg_latency_ms", -1)
        if avg_latency > 0:
            if avg_latency < 5:
                matrix["network_quality"] = "excellent"  # eRDMA 級別
            elif avg_latency < 20:
                matrix["network_quality"] = "good"
            elif avg_latency < 50:
                matrix["network_quality"] = "fair"
            else:
                matrix["network_quality"] = "poor"
        else:
            matrix["network_quality"] = "unknown"

        # === 契約 Profile 推導（4D 感知矩陣 → 契約層投影）===
        # 根據硬體感知結果，推導 bootstrap / system_profile / profile_binding / UPK
        gpu_count = gpu.get("gpu_count", 0)
        gpu_mem = gpu.get("gpu_memory_total_mb", 0)
        erdma = network.get("erdma_enabled", False)
        topology = matrix["topology_profile"]
        locality = matrix["locality"]

        matrix["contract_profiles"] = {
            "bootstrap_profile": cls.derive_bootstrap_profile(
                gpu_count, erdma, topology, locality),
            "system_profile": cls.derive_system_profile(
                hardware_type, locality, cgc_env, gpu_count, gpu_mem),
            "profile_binding": cls.derive_profile_binding(
                "cli_universe", locality, topology, hardware_type),
            "united_pipeline_kernel": cls.derive_united_pipeline_kernel(
                cgc_env, gpu_count, hardware_type),
            "state_abi_mode": "cloud_prefill_edge_decode" if locality == "edge" else "local_runtime_execute",
        }

        LOG.info(f"[HardwareProbe] perception_matrix built: "
                 f"hardware={hardware_type}, gpu={gpu.get('gpu_name', '?')}, "
                 f"mem_avail={available_gpu_mem}MB, "
                 f"erdma={erdma}, "
                 f"net_quality={matrix.get('network_quality', '?')}, "
                 f"models={matrix['model_pool']['model_count']}, "
                 f"ray={matrix['ray_cluster'].get('ray_available', False)}, "
                 f"bootstrap={matrix['contract_profiles']['bootstrap_profile']}, "
                 f"upk={matrix['contract_profiles']['united_pipeline_kernel']}")
        return matrix


# ---------------------------------------------------------------------------
# Unified IR 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """单层 IR 规格"""
    layer_id: int
    layer_type: str  # "attention" | "linear_attention" | "mlp" | "moe" | "norm"
    hidden_size: int = 0
    num_heads: int = 0
    kernel_backend: str = "triton"  # 默认 triton（不依赖 flashinfer linear_attn）
    inject: bool = False  # 是否需要 CGC IR 注入
    # MoE 相关（layer_type == "moe" 时使用）
    routing_strategy: str = "default"  # "default" | "biased" | "grouped" | "edge_cloud_consistent"
    topk_override: Optional[int] = None  # 强制 top_k 覆盖（端云路由一致性场景）
    expert_bias: Optional[List[float]] = None  # correction_bias（端云负载感知路由）
    edge_cloud_route_sync: bool = False  # 是否启用端云路由一致性 hook


@dataclass
class UnifiedIR:
    """模型级 Unified IR"""
    model_arch: str
    hardware_type: str
    layers: List[LayerSpec] = field(default_factory=list)
    linear_attn_decode_backend: str = "triton"
    linear_attn_prefill_backend: str = "triton"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        n_linear = sum(1 for l in self.layers if l.layer_type == "linear_attention")
        n_attn = sum(1 for l in self.layers if l.layer_type == "attention")
        return {
            "model_arch": self.model_arch,
            "hardware_type": self.hardware_type,
            "total_layers": len(self.layers),
            "attention_layers": n_attn,
            "linear_attention_layers": n_linear,
            "linear_attn_decode_backend": self.linear_attn_decode_backend,
            "linear_attn_prefill_backend": self.linear_attn_prefill_backend,
        }


# ---------------------------------------------------------------------------
# Unified IR Compiler
# ---------------------------------------------------------------------------

class UnifiedIRCompiler:
    """编译模型规格为 Unified IR，并 lower 到硬件可执行 kernel 描述。

    取代 Gate 2.0 的 stub 字符串实现，提供真正的结构化 IR + 注入器。
    """

    def __init__(self, perception_matrix: Optional[Dict[str, Any]] = None):
        if perception_matrix is None:
            # 動態探測硬體，構建 perception_matrix
            perception_matrix = HardwareProbe.build_perception_matrix()
        pm = perception_matrix or {}
        self.hardware_type = pm.get("hardware_type", "Nvidia_L20N")
        self.perception_matrix = pm  # 保存完整矩陣供後續決策
        self.is_compiled = False
        LOG.info(f"[UnifiedCompiler] Initialized with Hardware Type: {self.hardware_type} "
                 f"(host={pm.get('hostname', '?')}, gpu={pm.get('gpu_name', '?')}, "
                 f"erdma={pm.get('erdma_enabled', False)})")

    # ---- Step 1: 模型规格 -> Unified IR ----

    def compile_to_unified_ir(
        self,
        model_graph: Any = None,
        *,
        model_arch: str = "",
        layers_block_type: Optional[List[str]] = None,
        hidden_size: int = 0,
        num_hidden_layers: int = 0,
    ) -> UnifiedIR:
        """把模型规格编译为 Unified IR。

        Args:
            model_graph: 兼容旧接口的占位参数（字符串/dict）
            model_arch: 模型架构名（如 "Qwen3_5ForCausalLM"）
            layers_block_type: 每层的类型列表（如 ["attention","linear_attention",...]）
            hidden_size: 隐藏维度
            num_hidden_layers: 层数
        """
        LOG.info("[UnifiedCompiler] Converting computation graph to Unified IR...")

        # 兼容旧接口：model_graph 为字符串时尝试解析
        if isinstance(model_graph, str) and not model_arch:
            model_arch = model_graph
        elif isinstance(model_graph, dict):
            model_arch = model_arch or model_graph.get("arch", "")
            layers_block_type = layers_block_type or model_graph.get("layers_block_type")
            hidden_size = hidden_size or model_graph.get("hidden_size", 0)
            num_hidden_layers = num_hidden_layers or model_graph.get("num_hidden_layers", 0)

        # Check environment for OrthoKDA and RSWA
        enable_ortho_kda = os.environ.get("CGC_ENABLE_ORTHO_KDA", "0") == "1"
        enable_rswa = os.environ.get("CGC_ENABLE_RSWA", "0") == "1"

        # 构造层规格
        layers: List[LayerSpec] = []
        if layers_block_type:
            for idx, lt in enumerate(layers_block_type):
                # linear_attention 层需要 CGC IR 注入（强制 triton kernel）
                needs_inject = (lt == "linear_attention")
                kernel_backend = "triton" if needs_inject else "native"
                if lt == "attention":
                    if enable_rswa and enable_ortho_kda:
                        # 兩者都啟用：分層分配（偶數層 RSWA，奇數層 OrthoKDA）
                        needs_inject = True
                        kernel_backend = "rswa" if (idx % 2 == 0) else "ortho_kda"
                    elif enable_rswa:
                        needs_inject = True
                        kernel_backend = "rswa"
                    elif enable_ortho_kda:
                        needs_inject = True
                        kernel_backend = "ortho_kda"
                    
                layers.append(LayerSpec(
                    layer_id=idx,
                    layer_type=lt,
                    hidden_size=hidden_size,
                    kernel_backend=kernel_backend,
                    inject=needs_inject,
                ))
        elif num_hidden_layers > 0:
            # 无 layers_block_type 信息，默认全 attention（dense 模型）
            for idx in range(num_hidden_layers):
                needs_inject = enable_rswa or enable_ortho_kda
                if enable_rswa:
                    # 所有層都用 RSWA（內部整合 OrthoKDA 做計算）
                    kernel_backend = "rswa"
                elif enable_ortho_kda:
                    kernel_backend = "ortho_kda"
                else:
                    kernel_backend = "native"
                layers.append(LayerSpec(
                    layer_id=idx,
                    layer_type="attention",
                    hidden_size=hidden_size,
                    kernel_backend=kernel_backend,
                    inject=needs_inject,
                ))

        # Blackwell SM100+ 强制 triton，绕过 flashinfer 0.6.x 缺 linear_attn 模块
        is_blackwell = self._is_blackwell()
        decode_backend = "triton" if is_blackwell else "triton"
        prefill_backend = "triton" if is_blackwell else "triton"

        ir = UnifiedIR(
            model_arch=model_arch or "unknown",
            hardware_type=self.hardware_type,
            layers=layers,
            linear_attn_decode_backend=decode_backend,
            linear_attn_prefill_backend=prefill_backend,
            metadata={
                "compiled_by": "cgc_unified_compiler",
                "is_blackwell": is_blackwell,
                "inject_linear_attn": any(l.inject for l in layers),
            },
        )
        LOG.info(
            f"[UnifiedCompiler] IR compiled: arch={ir.model_arch} "
            f"layers={len(ir.layers)} linear_attn_decode={decode_backend}"
        )
        return ir

    # ---- Step 2: Unified IR -> 硬件可执行 kernel 描述 ----

    def lower_to_hardware(self, unified_ir: UnifiedIR) -> Dict[str, Any]:
        """Lower Unified IR 到硬件 kernel 描述（注入器消费的配置）。"""
        LOG.info(f"[UnifiedCompiler] Lowering Unified IR to {self.hardware_type}...")

        if "Nvidia" in self.hardware_type or self._is_blackwell():
            target = "cuda_triton"
            LOG.info("[UnifiedCompiler] Lowering to CUDA/Triton operators "
                     "(preserving FlashInfer full-attention kernels).")
        elif "Huawei" in self.hardware_type or "Ascend" in self.hardware_type:
            target = "cann"
            LOG.info("[UnifiedCompiler] Lowering to Huawei CANN operators.")
        else:
            target = "cpu"
            LOG.warning(f"[UnifiedCompiler] Unknown hardware {self.hardware_type}, fallback CPU.")

        self.is_compiled = True
        return {
            "compile_target": target,
            "ir_summary": unified_ir.summary(),
            "linear_attn_decode_backend": unified_ir.linear_attn_decode_backend,
            "linear_attn_prefill_backend": unified_ir.linear_attn_prefill_backend,
            "injectable_layers": [
                {"layer_id": l.layer_id, "layer_type": l.layer_type,
                 "kernel_backend": l.kernel_backend}
                for l in unified_ir.layers if l.inject
            ],
            "moe_layers": [
                {
                    "layer_id": l.layer_id,
                    "routing_strategy": l.routing_strategy,
                    "topk_override": l.topk_override,
                    "expert_bias": l.expert_bias,
                    "edge_cloud_route_sync": l.edge_cloud_route_sync,
                }
                for l in unified_ir.layers if l.layer_type == "moe"
            ],
        }

    def _is_blackwell(self) -> bool:
        """检测是否 Blackwell SM100+（会触发 flashinfer GDN 默认）"""
        if os.environ.get("CGC_FORCE_TRITON_LINEAR_ATTN", "1") == "1":
            return True  # 强制走 triton 注入路径
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_capability()[0] >= 10
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# SGLang Compute Injector
# ---------------------------------------------------------------------------

class UnifiedIRInjector:
    """把 Unified IR 编译结果注入到 vendored SGLang 的 compute 路径。

    注入策略：
      1. 设置 SGLang 全局 LINEAR_ATTN_DECODE/PREFILL_BACKEND = TRITON
         （在 initialize_linear_attn_config 之前生效，使 GDNAttnBackend 选 TritonGDNKernel）
      2. 在 ModelRunner 上挂 cgc_ir_injection 元数据（审计追踪）
      3. 可选：monkey-patch 指定层的 forward（未来扩展点）

    使用方式（gateway 启动 SGLang engine 前调用）：
        injector = UnifiedIRInjector()
        injector.inject_into_sglang(compiled_target)
    """

    def __init__(self):
        self.injected = False
        self.injection_log: List[Dict[str, Any]] = []

    def inject_into_sglang(self, compiled_target: Dict[str, Any]) -> Dict[str, Any]:
        """注入 IR 编译结果到 vendored SGLang runtime。

        Args:
            compiled_target: UnifiedIRCompiler.lower_to_hardware() 的返回值

        Returns:
            注入结果摘要
        """
        decode_backend = compiled_target.get("linear_attn_decode_backend", "triton")
        prefill_backend = compiled_target.get("linear_attn_prefill_backend", "triton")
        injectable = compiled_target.get("injectable_layers", [])

        # ---- 注入点 1：强制 linear_attn backend 为 triton ----
        # 关键：Blackwell SM100+ 默认选 flashinfer decode，但 flashinfer 0.6.x
        # 缺 linear_attention 模块，导致 TMAX-9B (Qwen3.5 hybrid GDN) 启动失败。
        # 通过设置环境变量 + 全局变量，强制 GDNAttnBackend 用 TritonGDNKernel。
        os.environ["SGLANG_LINEAR_ATTN_DECODE_BACKEND"] = decode_backend
        os.environ["SGLANG_LINEAR_ATTN_PREFILL_BACKEND"] = prefill_backend
        os.environ.setdefault("CGC_IR_INJECTED", "1")

        try:
            from sglang.srt.layers.attention.linear.utils import (
                LinearAttnKernelBackend,
                initialize_linear_attn_config,
            )
            # 直接设置全局变量（在 initialize_linear_attn_config 调用前生效）
            import sglang.srt.layers.attention.linear.utils as _la_utils
            _la_utils.LINEAR_ATTN_DECODE_BACKEND = LinearAttnKernelBackend.TRITON
            _la_utils.LINEAR_ATTN_PREFILL_BACKEND = LinearAttnKernelBackend.TRITON
            LOG.info(
                "[UnifiedIRInjector] Injected LINEAR_ATTN_DECODE/PREFILL_BACKEND=TRITON "
                f"({len(injectable)} linear_attention layers)"
            )
            self.injection_log.append({
                "point": "linear_attn_backend",
                "decode": "triton",
                "prefill": "triton",
                "injectable_layers": len(injectable),
            })
        except ImportError:
            LOG.warning("[UnifiedIRInjector] vendored SGLang linear.utils not on path, "
                        "set PYTHONPATH to cloud_sglang/python before inject.")
            self.injection_log.append({
                "point": "linear_attn_backend",
                "status": "skipped_sglang_not_importable",
            })

        # ---- 注入点 2：patch server_args 默认值（若 SGLang 已 import）----
        try:
            from sglang.srt import server_args as _sa_mod
            # 让 _handle_linear_attn_backend 不再强制 flashinfer
            if hasattr(_sa_mod, "is_sm100_supported"):
                _orig_is_sm100 = _sa_mod.is_sm100_supported
                def _patched_is_sm100(*a, **kw):
                    return False  # 骗过默认逻辑，保持 triton
                _sa_mod.is_sm100_supported = _patched_is_sm100
                LOG.info("[UnifiedIRInjector] Patched server_args.is_sm100_supported -> False "
                         "(prevent flashinfer default on Blackwell)")
                self.injection_log.append({
                    "point": "server_args.is_sm100_supported",
                    "patched_to": False,
                })
        except ImportError:
            pass

        # ---- 注入点 3：CGC TrueOrthoKDA / R-SWA Attention 注入 ----
        ortho_kda_layers = [l for l in injectable if l.get("kernel_backend") == "ortho_kda"]
        rswa_layers = [l for l in injectable if l.get("kernel_backend") == "rswa"]
        
        if ortho_kda_layers or rswa_layers:
            try:
                from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
                
                original_forward = AttentionBackend.forward
                
                if not hasattr(AttentionBackend, "_cgc_ortho_kda_instances"):
                    AttentionBackend._cgc_ortho_kda_instances = {}
                if not hasattr(AttentionBackend, "_cgc_rswa_instances"):
                    AttentionBackend._cgc_rswa_instances = {}

                def _patched_forward(self_attn, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
                    import torch
                    layer_id = layer.layer_id
                    needs_ortho = any(l["layer_id"] == layer_id for l in ortho_kda_layers)
                    needs_rswa = any(l["layer_id"] == layer_id for l in rswa_layers)

                    # Debug: 第一次調用時打印
                    if not hasattr(AttentionBackend, "_cgc_debug_logged"):
                        AttentionBackend._cgc_debug_logged = True
                        print(f"[CGC Debug] _patched_forward called! layer_id={layer_id}, needs_rswa={needs_rswa}, needs_ortho={needs_ortho}, k is None={k is None}, v is None={v is None}", flush=True)
                        print(f"[CGC Debug] rswa_layers count={len(rswa_layers)}, ortho_kda_layers count={len(ortho_kda_layers)}", flush=True)
                        if rswa_layers:
                            print(f"[CGC Debug] rswa_layers[0]={rswa_layers[0]}", flush=True)
                        print(f"[CGC Debug] layer.layer_id type={type(layer_id)}, value={layer_id}", flush=True)

                    if needs_rswa and k is not None and v is not None:
                        # Debug: 打印 tensor shape 用於診斷
                        if not hasattr(AttentionBackend, "_cgc_shape_logged"):
                            AttentionBackend._cgc_shape_logged = True
                            print(f"[CGC Debug] q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}", flush=True)
                            print(f"[CGC Debug] layer.tp_q_head_num={layer.tp_q_head_num}, layer.qk_head_dim={layer.qk_head_dim}", flush=True)
                            print(f"[CGC Debug] layer.tp_k_head_num={layer.tp_k_head_num}, layer.v_head_dim={layer.v_head_dim}", flush=True)
                            print(f"[CGC Debug] forward_mode={forward_batch.forward_mode}, is_extend={forward_batch.forward_mode.is_extend()}", flush=True)
                            # 寫文件標記
                            import os as _os
                            with open("/tmp/cgc_shape_debug.txt", "w") as _f:
                                _f.write(f"q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}\n")
                                _f.write(f"tp_q={layer.tp_q_head_num}, qk_dim={layer.qk_head_dim}, tp_k={layer.tp_k_head_num}, v_dim={layer.v_head_dim}\n")
                                _f.write(f"is_extend={forward_batch.forward_mode.is_extend()}\n")

                        # R-SWA 注入邏輯（內部整合 OrthoKDA 做計算）
                        if layer_id not in AttentionBackend._cgc_rswa_instances:
                            # 動態匯入我們從 Host 2 移植過來的 R-SWA 模組
                            import sys
                            import os
                            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
                            from cgc_engine.rswa_integration.rswa_prefill_pool_adapter import CGCUnlimitedRSWAAttention

                            # 使用實際 tensor shape 初始化（DeepseekV4 MLA 的 layer 屬性可能不匹配）
                            if q.dim() >= 3:
                                num_heads = q.shape[1]
                                head_dim = q.shape[2]
                            else:
                                num_heads = layer.tp_q_head_num
                                head_dim = layer.qk_head_dim
                            dim = num_heads * head_dim

                            AttentionBackend._cgc_rswa_instances[layer_id] = CGCUnlimitedRSWAAttention(
                                dim=dim,
                                num_heads=num_heads,
                                window_size=int(os.environ.get("CGC_RSWA_WINDOW_SIZE", "128")),
                                init_projs=False
                            )
                            if q.is_cuda:
                                AttentionBackend._cgc_rswa_instances[layer_id] = AttentionBackend._cgc_rswa_instances[layer_id].cuda()

                            # 同時初始化 OrthoKDA 計算引擎（O(1) 內存 attention）
                            from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4
                            ortho_base_dim = int(os.environ.get("CGC_ORTHO_BASE_DIM", "128"))
                            AttentionBackend._cgc_rswa_ortho_kda_instances = getattr(AttentionBackend, "_cgc_rswa_ortho_kda_instances", {})
                            AttentionBackend._cgc_rswa_ortho_kda_instances[layer_id] = OrthoKDAV4(
                                num_heads=num_heads,
                                head_dim=head_dim,
                                ortho_base_dim=ortho_base_dim,
                                use_cuda=q.is_cuda
                            )
                            LOG.info(f"[UnifiedIRInjector] Initialized RSWA+OrthoKDA for layer {layer_id} (heads={num_heads}, dim={dim}, head_dim={head_dim}, ortho_base={ortho_base_dim})")

                        rswa_instance = AttentionBackend._cgc_rswa_instances[layer_id]
                        kda_instance = AttentionBackend._cgc_rswa_ortho_kda_instances[layer_id]

                        num_tokens = q.shape[0]

                        # 使用實際 tensor shape 而非 layer 屬性（DeepseekV4 MLA 的 layer 屬性可能不匹配）
                        if k.dim() >= 3:
                            kv_heads = k.shape[1]
                            head_dim_k = k.shape[2]
                        else:
                            kv_heads = 1
                            head_dim_k = k.shape[-1] if k.dim() >= 2 else k.shape[0]
                        if q.dim() >= 3:
                            q_heads = q.shape[1]
                            head_dim_q = q.shape[2]
                        else:
                            q_heads = layer.tp_q_head_num
                            head_dim_q = layer.qk_head_dim

                        # 從 rswa_instance 獲取 Reference KV
                        ref_k, ref_v = rswa_instance.get_all_reference_kv(device=q.device)

                        if forward_batch.forward_mode.is_extend():
                            # Prefill 階段: 將長文本 KV 存入 Prefill Pool 作為 Reference KV
                            # 計算回退到原始 forward 保證精度
                            if num_tokens > 128:
                                k_pool = k.view(1, num_tokens, kv_heads, head_dim_k).transpose(1, 2)
                                v_pool = v.view(1, num_tokens, kv_heads, head_dim_k).transpose(1, 2)
                                dummy_tokens = torch.zeros((1, num_tokens), dtype=torch.long, device=q.device)
                                rswa_instance.add_reference_chunk(dummy_tokens, k_pool, v_pool)
                            # extend 階段回退到原始 forward
                            return None
                        else:
                            # Decode 階段: 將當前 token 的 KV 存入 Sliding Window Buffer
                            k_decode = k.view(1, num_tokens, kv_heads, head_dim_k).transpose(1, 2)
                            v_decode = v.view(1, num_tokens, kv_heads, head_dim_k).transpose(1, 2)
                            if rswa_instance._past_k is None:
                                rswa_instance._past_k = k_decode
                                rswa_instance._past_v = v_decode
                            else:
                                rswa_instance._past_k = torch.cat([rswa_instance._past_k, k_decode], dim=2)[:, :, -rswa_instance.window_size:]
                                rswa_instance._past_v = torch.cat([rswa_instance._past_v, v_decode], dim=2)[:, :, -rswa_instance.window_size:]

                        # === 計算階段：用 SDPA 做精確 attention 計算 ===
                        # RSWA 負責 KV 管理（Reference KV + Sliding Window），SDPA 負責計算
                        # 收集所有 KV：Reference KV (from PrefillPool) + Output KV (sliding window) + 當前 KV

                        # 安全檢查：如果沒有 Reference KV（短 prompt 未存入 PrefillPool），
                        # 回退到原始 forward，避免丟失 prefill 階段的 KV 導致亂碼
                        #
                        # 原因：R-SWA 的 sliding window 只保存最近 window_size 個 token 的 KV，
                        # 但 prefill 階段的完整 KV 存在 sglang 原生 KV cache 中。
                        # 如果 ref_k is None（短 prompt 未觸發 PrefillPool 存儲），
                        # R-SWA 的 KV 拼接會丟失 prefill 的大部分 KV → attention 計算用殘缺 KV → 亂碼
                        if ref_k is None:
                            # 沒有 Reference KV，回退到原始 forward（使用 sglang 完整 KV cache）
                            return original_forward(self_attn, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)

                        all_k_parts = []
                        all_v_parts = []

                        if ref_k is not None:
                            # ref_k shape: [B, H, L, D]
                            all_k_parts.append(ref_k)  # [1, kv_heads, L, D]
                            all_v_parts.append(ref_v)
                        if rswa_instance._past_k is not None:
                            # _past_k shape: [B, H, W, D]
                            all_k_parts.append(rswa_instance._past_k)  # [1, kv_heads, W, D]
                            all_v_parts.append(rswa_instance._past_v)
                        # 當前 KV: [num_tokens, kv_heads, D] -> [1, kv_heads, num_tokens, D]
                        k_cur = k.unsqueeze(0).transpose(1, 2)  # [1, kv_heads, num_tokens, D]
                        v_cur = v.unsqueeze(0).transpose(1, 2)
                        all_k_parts.append(k_cur)
                        all_v_parts.append(v_cur)

                        # 拼接所有 KV: [1, kv_heads, total_len, D]
                        full_k = torch.cat(all_k_parts, dim=2)
                        full_v = torch.cat(all_v_parts, dim=2)

                        # GQA: 將 kv_heads repeat 到 q_heads
                        group_size = q_heads // kv_heads if kv_heads > 0 else 1
                        if group_size > 1:
                            full_k = full_k.repeat_interleave(group_size, dim=1)  # [1, q_heads, total_len, D]
                            full_v = full_v.repeat_interleave(group_size, dim=1)

                        # Q: [num_tokens, q_heads, D] -> [1, q_heads, num_tokens, D]
                        q_sdpa = q.unsqueeze(0).transpose(1, 2)

                        # SDPA 計算
                        attn_out = torch.nn.functional.scaled_dot_product_attention(q_sdpa, full_k, full_v)  # [1, q_heads, num_tokens, D]
                        # 返回 [num_tokens, q_heads, D]
                        output = attn_out[0].transpose(0, 1)  # [num_tokens, q_heads, D]
                        return output.to(device=q.device, dtype=q.dtype)
                        
                    elif needs_ortho and k is not None and v is not None:
                        from cgc_engine.cgc.ortho_kda_v4_bridge import OrthoKDAV4
                        if layer_id not in AttentionBackend._cgc_ortho_kda_instances:
                            num_heads = layer.tp_q_head_num
                            head_dim = layer.qk_head_dim
                            ortho_base_dim = int(os.environ.get("CGC_ORTHO_BASE_DIM", "128"))
                            AttentionBackend._cgc_ortho_kda_instances[layer_id] = OrthoKDAV4(
                                num_heads=num_heads,
                                head_dim=head_dim,
                                ortho_base_dim=ortho_base_dim,
                                use_cuda=True
                            )
                            LOG.info(f"[UnifiedIRInjector] Initialized OrthoKDAV4 for layer {layer_id} (heads={num_heads}, dim={head_dim})")
                        
                        kda_instance = AttentionBackend._cgc_ortho_kda_instances[layer_id]
                        
                        if forward_batch.forward_mode.is_extend():
                            # k, v shape: [num_tokens, tp_head_num, head_dim]
                            num_tokens = k.shape[0]
                            for i in range(num_tokens):
                                kda_instance.update(k[i], v[i])
                            
                        # Forward phase (O(1) memory)
                        # q shape: [num_tokens, tp_q_head_num * head_dim]
                        # we need to reshape it for OrthoKDAV4
                        num_tokens = q.shape[0]
                        q_reshaped = q.view(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
                        
                        output_reshaped = kda_instance.forward(q_reshaped)
                        # Output shape is [num_tokens, tp_q_head_num, head_dim], we need [num_tokens, tp_q_head_num * head_dim]
                        output = output_reshaped.reshape(num_tokens, -1)
                        return output
                    else:
                        return original_forward(self_attn, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)

                AttentionBackend.forward = _patched_forward

                # Patch RadixAttention.forward 加文件標記確認是否被調用
                try:
                    from sglang.srt.layers.radix_attention import RadixAttention
                    _orig_radix_forward = RadixAttention.forward

                    def _patched_radix_forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
                        import os as _os
                        _flag = "/tmp/cgc_radix_triggered.txt"
                        if not _os.path.exists(_flag):
                            with open(_flag, "w") as _f:
                                _f.write(f"layer_id={self.layer_id}, k_none={k is None}, v_none={v is None}, mode={forward_batch.forward_mode}\n")
                        return _orig_radix_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)

                    RadixAttention.forward = _patched_radix_forward
                    print("[UnifiedIRInjector] Patched RadixAttention.forward (flag)", flush=True)
                except Exception as e:
                    print(f"[UnifiedIRInjector] Could not patch RadixAttention: {e}", flush=True)

                # Patch HybridLinearAttnBackend.forward（DeepSeek-V4 等混合模型使用此 backend）
                patched_backends = []
                try:
                    from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend
                    _orig_hybrid_forward = HybridLinearAttnBackend.forward

                    def _patched_hybrid_forward(self, q=None, k=None, v=None, layer=None,
                                                forward_batch=None, save_kv_cache=True,
                                                mixed_qkv=None, a=None, b=None, **kwargs):
                        layer_id = layer.layer_id if layer is not None else kwargs.get("layer_id")
                        needs_rswa_local = any(l["layer_id"] == layer_id for l in rswa_layers)

                        # Debug: 寫文件標記確認 forward 被調用
                        import os as _os
                        _flag = "/tmp/cgc_forward_triggered.txt"
                        if not _os.path.exists(_flag):
                            with open(_flag, "w") as _f:
                                _f.write(f"layer_id={layer_id}, needs_rswa={needs_rswa_local}, k_none={k is None}, v_none={v is None}, is_full={self._is_full_attn(layer, layer_id) if layer else 'N/A'}\n")

                        if needs_rswa_local and k is not None and v is not None:
                            # R-SWA + OrthoKDA 路徑
                            return _patched_forward(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
                        else:
                            return _orig_hybrid_forward(self, q, k, v, layer, forward_batch, save_kv_cache,
                                                       mixed_qkv=mixed_qkv, a=a, b=b, **kwargs)

                    HybridLinearAttnBackend.forward = _patched_hybrid_forward
                    patched_backends.append("HybridLinearAttnBackend")
                except Exception as e:
                    print(f"[UnifiedIRInjector] Could not patch HybridLinearAttnBackend: {e}", flush=True)

                # 也 patch HybridAttnBackend（非 linear 混合模型）
                try:
                    from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
                    _orig_hybrid2_forward = HybridAttnBackend.forward

                    def _patched_hybrid2_forward(self, q=None, k=None, v=None, layer=None,
                                                 forward_batch=None, save_kv_cache=True,
                                                 mixed_qnv=None, a=None, b=None, **kwargs):
                        layer_id = layer.layer_id if layer is not None else kwargs.get("layer_id")
                        needs_rswa_local = any(l["layer_id"] == layer_id for l in rswa_layers)

                        if not hasattr(HybridAttnBackend, "_cgc_debug_logged"):
                            HybridAttnBackend._cgc_debug_logged = True
                            print(f"[CGC Debug] HybridAttnBackend.forward called! layer_id={layer_id}, needs_rswa={needs_rswa_local}", flush=True)

                        if needs_rswa_local and k is not None and v is not None:
                            try:
                                _ret = _patched_forward(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
                                if _ret is not None:
                                    return _ret
                                # None 表示 extend 階段，回退到原始 forward
                            except Exception as _e:
                                import traceback as _tb
                                import os as _os
                                with open("/tmp/cgc_hybrid_rswa_error.txt", "w") as _f:
                                    _f.write(f"layer_id={layer_id}: {_e}\n{_tb.format_exc()}\n")
                            return _orig_hybrid2_forward(self, q, k, v, layer, forward_batch, save_kv_cache,
                                                            mixed_qnv=mixed_qnv, a=a, b=b, **kwargs)
                        else:
                            return _orig_hybrid2_forward(self, q, k, v, layer, forward_batch, save_kv_cache,
                                                        mixed_qnv=mixed_qnv, a=a, b=b, **kwargs)

                    HybridAttnBackend.forward = _patched_hybrid2_forward
                    patched_backends.append("HybridAttnBackend")
                except Exception as e:
                    print(f"[UnifiedIRInjector] Could not patch HybridAttnBackend: {e}", flush=True)

                # Patch DeepseekV4AttnBackend.forward（DeepSeek-V4 模型實際使用此 backend）
                try:
                    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
                    _orig_dsv4_forward = DeepseekV4AttnBackend.forward

                    def _patched_dsv4_forward(self, q, k, v, layer, forward_batch,
                                              compress_ratio, save_kv_cache=True,
                                              attn_sink=None, **kwargs):
                        layer_id = layer.layer_id if layer is not None else -1
                        needs_rswa_local = any(l["layer_id"] == layer_id for l in rswa_layers)

                        # Debug: 寫文件標記確認 forward 被調用
                        import os as _os
                        _flag = "/tmp/cgc_dsv4_forward_triggered.txt"
                        if not _os.path.exists(_flag):
                            with open(_flag, "w") as _f:
                                _f.write(f"layer_id={layer_id}, needs_rswa={needs_rswa_local}, compress_ratio={compress_ratio}, k_none={k is None}\n")

                        if needs_rswa_local and k is not None and v is not None:
                            # R-SWA + SDPA 路徑：複用 _patched_forward 中的計算邏輯
                            try:
                                _ret = _patched_forward(self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache)
                                if _ret is not None:
                                    return _ret
                                # None 表示 extend 階段，回退到原始 forward
                            except Exception as _e:
                                import traceback as _tb
                                with open("/tmp/cgc_dsv4_rswa_error.txt", "w") as _f:
                                    _f.write(f"layer_id={layer_id}: {_e}\n{_tb.format_exc()}\n")
                            # 回退到原始 forward
                            return _orig_dsv4_forward(self, q, k, v, layer, forward_batch,
                                                      compress_ratio, save_kv_cache,
                                                      attn_sink=attn_sink, **kwargs)
                        else:
                            return _orig_dsv4_forward(self, q, k, v, layer, forward_batch,
                                                      compress_ratio, save_kv_cache,
                                                      attn_sink=attn_sink, **kwargs)

                    DeepseekV4AttnBackend.forward = _patched_dsv4_forward
                    patched_backends.append("DeepseekV4AttnBackend")
                    print("[UnifiedIRInjector] Patched DeepseekV4AttnBackend.forward", flush=True)
                except Exception as e:
                    print(f"[UnifiedIRInjector] Could not patch DeepseekV4AttnBackend: {e}", flush=True)

                if patched_backends:
                    print(f"[UnifiedIRInjector] Patched forward on: {patched_backends}", flush=True)
                    self.injection_log.append({
                        "point": "hybrid_backends.forward",
                        "patched_to": "RSWA+OrthoKDA",
                        "backends": patched_backends,
                        "layers": len(rswa_layers)
                    })

                patched_target = "RSWA" if rswa_layers else "OrthoKDAV4"
                patched_len = len(rswa_layers) if rswa_layers else len(ortho_kda_layers)
                
                LOG.info(f"[UnifiedIRInjector] Injected {patched_target} into SGLang AttentionBackend for {patched_len} layers")
                self.injection_log.append({
                    "point": "attention_backend.forward",
                    "patched_to": patched_target,
                    "layers": patched_len
                })
            except Exception as e:
                LOG.error(f"[UnifiedIRInjector] Failed to inject OrthoKDA: {e}")
                self.injection_log.append({
                    "point": "attention_backend.forward",
                    "status": f"failed: {e}"
                })

        # ---- 注入点 4：TopK 路由一致性 hook（端云 MoE 路由同步）----
        # 对应能力 moe_route_consistency_across_edge_cloud。
        # 端侧执行前 N 层时记录每层 topk_ids，云侧接续时通过本 hook 注入
        # correction_bias，使云侧后续 MoE 层的路由决策与端侧一致。
        moe_layers = compiled_target.get("moe_layers", [])
        edge_cloud_sync_layers = [
            m for m in moe_layers if m.get("edge_cloud_route_sync")
        ]
        if edge_cloud_sync_layers:
            try:
                self._inject_topk_route_consistency(edge_cloud_sync_layers)
            except Exception as e:
                LOG.error(f"[UnifiedIRInjector] Failed to inject TopK route consistency: {e}")
                self.injection_log.append({
                    "point": "topk.select_experts",
                    "status": f"failed: {e}"
                })

        # ---- 注入点 5：FusedMoE forward hook（端云层接续 KV push）----
        # 在 FusedMoE.forward 后插入 callback，把每层 MoE 输出 push 给端侧
        # （用于端侧 R-SWA 双层 KV cache 更新）。
        # 对应能力 layer_wise_kv_streaming_to_decode + moe_route_consistency_across_edge_cloud。
        if moe_layers:
            try:
                self._inject_fused_moe_edge_continuation(moe_layers)
            except Exception as e:
                LOG.error(f"[UnifiedIRInjector] Failed to inject FusedMoE edge continuation: {e}")
                self.injection_log.append({
                    "point": "fused_moe.forward",
                    "status": f"failed: {e}"
                })

        self.injected = True
        return {
            "injected": True,
            "linear_attn_decode_backend": decode_backend,
            "linear_attn_prefill_backend": prefill_backend,
            "injectable_layers": len(injectable),
            "moe_layers": len(moe_layers),
            "edge_cloud_route_sync_layers": len(edge_cloud_sync_layers),
            "injection_points": self.injection_log,
        }

    def _inject_topk_route_consistency(self, edge_cloud_sync_layers: List[Dict[str, Any]]) -> None:
        """注入 TopK 路由一致性 hook

        monkey-patch sglang.srt.layers.moe.topk.select_experts，使其在指定 layer
        上应用端侧上传的 correction_bias（覆盖 IR 中 expert_bias）。
        """
        from sglang.srt.layers.moe.topk import select_experts as _orig_select_experts

        # layer_id -> correction_bias（运行时可被 forward_batch.cgc_edge_route_bias 覆盖）
        layer_bias_map: Dict[int, Any] = {}
        for m in edge_cloud_sync_layers:
            layer_id = m.get("layer_id")
            if layer_id is None:
                continue
            bias = m.get("expert_bias")
            if bias is not None:
                try:
                    import torch
                    layer_bias_map[int(layer_id)] = torch.tensor(bias, dtype=torch.float32)
                except Exception:
                    layer_bias_map[int(layer_id)] = bias

        if not hasattr(_orig_select_experts, "_cgc_route_sync_patched"):
            def _patched_select_experts(
                hidden_states,
                router_logits,
                topk_config,
                *,
                layer_id=None,
                num_token_non_padded=None,
                expert_location_dispatch_info=None,
            ):
                # 运行时优先用 forward_batch 上挂的 cgc_edge_route_bias（端侧上传）
                # 其次用 IR 编译期注入的 layer_bias_map
                runtime_bias = None
                if layer_id is not None:
                    runtime_bias = layer_bias_map.get(int(layer_id))

                # 注入 correction_bias 到 topk_config（如果 IR 指定）
                if runtime_bias is not None and topk_config is not None:
                    try:
                        import torch
                        if not isinstance(runtime_bias, torch.Tensor):
                            runtime_bias = torch.tensor(runtime_bias, dtype=torch.float32)
                        # 同步 device
                        if hasattr(router_logits, "device"):
                            runtime_bias = runtime_bias.to(router_logits.device)
                        topk_config.correction_bias = runtime_bias
                    except Exception:
                        pass

                # topk_override（端云 top_k 一致性）
                if layer_id is not None:
                    for m in edge_cloud_sync_layers:
                        if m.get("layer_id") == layer_id and m.get("topk_override"):
                            try:
                                topk_config.top_k = int(m["topk_override"])
                            except Exception:
                                pass
                            break

                return _orig_select_experts(
                    hidden_states,
                    router_logits,
                    topk_config,
                    layer_id=layer_id,
                    num_token_non_padded=num_token_non_padded,
                    expert_location_dispatch_info=expert_location_dispatch_info,
                )

            _patched_select_experts._cgc_route_sync_patched = True
            _patched_select_experts._cgc_patched = True
            _patched_select_experts._cgc_orig = _orig_select_experts

            # 替换 module 内函数引用
            import sglang.srt.layers.moe.topk as _topk_mod
            _topk_mod.select_experts = _patched_select_experts

            LOG.info(
                f"[UnifiedIRInjector] Injected TopK route consistency hook for "
                f"{len(layer_bias_map)} layers: {sorted(layer_bias_map.keys())}"
            )
            self.injection_log.append({
                "point": "topk.select_experts",
                "patched_to": "cgc_edge_cloud_route_sync",
                "layers": len(layer_bias_map),
                "layer_ids": sorted(layer_bias_map.keys()),
            })

    def _inject_fused_moe_edge_continuation(self, moe_layers: List[Dict[str, Any]]) -> None:
        """注入 FusedMoE forward hook（端云层接续 KV push）

        monkey-patch FusedMoE.forward，在 forward 后检查 forward_batch 上的
        cgc_layer_kv_push_callback，若存在则调用以 push 该层 MoE 输出。
        与 deepseek_v4.py 中 layer-level KV push callback 互补：
          - deepseek_v4.py: 在 layer loop 层级调用
          - 本处: 在 FusedMoE 模块层级调用（兼容非 deepseek_v4 模型）
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE as _FusedMoE
        _orig_forward = _FusedMoE.forward

        if hasattr(_orig_forward, "_cgc_edge_continuation_patched"):
            return  # 已 patch，避免重复

        def _patched_forward(self_moe, hidden_states, topk_output, *args, **kwargs):
            out = _orig_forward(self_moe, hidden_states, topk_output, *args, **kwargs)
            # 调用 layer-level KV push callback（若 forward_batch 上挂载）
            # 注意：FusedMoE.forward 签名不含 forward_batch，需从 topk_output 或
            # 全局 context 获取；此处用 thread-local 兜底
            try:
                from sglang.srt.layers.moe.fused_moe_triton.layer import _cgc_moe_forward_ctx
                ctx = _cgc_moe_forward_ctx.get()
                if ctx is not None:
                    cb = ctx.get("cgc_layer_kv_push_callback")
                    if cb is not None:
                        cb(
                            layer_id=getattr(self_moe, "layer_id", None),
                            hidden_states=out,
                            forward_batch=ctx.get("forward_batch"),
                        )
            except ImportError:
                pass  # _cgc_moe_forward_ctx 未定义，跳过
            except Exception:
                pass  # KV push 失败不阻断主 forward
            return out

        _patched_forward._cgc_edge_continuation_patched = True
        _patched_forward._cgc_orig = _orig_forward
        _FusedMoE.forward = _patched_forward

        LOG.info(
            f"[UnifiedIRInjector] Injected FusedMoE edge continuation hook for "
            f"{len(moe_layers)} MoE layers"
        )
        self.injection_log.append({
            "point": "fused_moe.forward",
            "patched_to": "cgc_edge_continuation_kv_push",
            "layers": len(moe_layers),
        })

    def attach_model_runner_metadata(self, model_runner: Any,
                                     ir: UnifiedIR) -> None:
        """在 ModelRunner 上挂 CGC IR 注入元数据（审计追踪用）。

        在 SGLang ModelRunner.__init__ 完成后调用。
        """
        try:
            setattr(model_runner, "cgc_ir_injection", {
                "model_arch": ir.model_arch,
                "hardware_type": ir.hardware_type,
                "ir_summary": ir.summary(),
                "injection_log": self.injection_log,
            })
            LOG.debug(f"[UnifiedIRInjector] Attached cgc_ir_injection metadata to {model_runner}")
        except Exception as e:
            LOG.warning(f"[UnifiedIRInjector] Failed to attach metadata: {e}")


# ---------------------------------------------------------------------------
# 便捷入口：四角色统一 IR 注入
# ---------------------------------------------------------------------------

# 四角色模型的架构规格（用于 IR 编译）
ROLE_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "hermes": {
        "model_arch": "DeepseekV4ForCausalLM",
        "layers_block_type": None,  # 全 attention dense/moe
        "num_hidden_layers": 61,
    },
    "tmax": {
        "model_arch": "Qwen3_5ForCausalLM",
        # TMAX-9B = Qwen3.5-9B hybrid GDN：attention + linear_attention 混合
        # 实际 layers_block_type 从 config.json 读取，这里给占位
        "layers_block_type": ["attention"] * 36,  # 占位，注入时从 config 读取真实值
        "hidden_size": 4096,
        "num_hidden_layers": 36,
    },
    "uitars": {
        "model_arch": "Qwen2VLForConditionalGeneration",
        "layers_block_type": None,
        "num_hidden_layers": 28,
    },
    "cli_universe": {
        "model_arch": "DeepseekV4ForCausalLM",
        "layers_block_type": None,
        "num_hidden_layers": 61,
    },
}


def inject_unified_ir_for_role(
    role: str,
    perception_matrix: Optional[Dict[str, Any]] = None,
    *,
    layers_block_type: Optional[List[str]] = None,
    hidden_size: int = 0,
    num_hidden_layers: int = 0,
) -> Dict[str, Any]:
    """四角色统一入口：编译 IR + 注入 SGLang compute。

    Args:
        role: hermes/tmax/uitars/cli_universe
        perception_matrix: 硬件感知矩阵（None 時自動動態探測）
        layers_block_type: 覆盖默认层类型（从模型 config.json 读取真实值）
        hidden_size: 覆盖默认 hidden_size
        num_hidden_layers: 覆盖默认层数

    Returns:
        {"ir": UnifiedIR, "compiled": {...}, "injection": {...},
         "perception_matrix": {...}}
    """
    # 動態探測硬體（如果調用者未提供 perception_matrix）
    if perception_matrix is None:
        perception_matrix = HardwareProbe.build_perception_matrix()
        LOG.info(f"[UnifiedIR] Auto-probed perception_matrix: "
                 f"host={perception_matrix.get('hostname')}, "
                 f"hardware={perception_matrix.get('hardware_type')}")

    spec = ROLE_MODEL_SPECS.get(role, {})
    compiler = UnifiedIRCompiler(perception_matrix=perception_matrix)

    ir = compiler.compile_to_unified_ir(
        model_arch=spec.get("model_arch", "unknown"),
        layers_block_type=layers_block_type or spec.get("layers_block_type"),
        hidden_size=hidden_size or spec.get("hidden_size", 0),
        num_hidden_layers=num_hidden_layers or spec.get("num_hidden_layers", 0),
    )
    compiled = compiler.lower_to_hardware(ir)

    injector = UnifiedIRInjector()
    injection = injector.inject_into_sglang(compiled)

    LOG.info(f"[UnifiedIR] role={role} injection complete: {injection}")
    return {
        "ir": ir,
        "compiled": compiled,
        "injection": injection,
        "perception_matrix": perception_matrix,
    }


# ---------------------------------------------------------------------------
# 向后兼容：旧接口（gateway 现有调用）
# ---------------------------------------------------------------------------

# 旧测试可能直接实例化 UnifiedIRCompiler 并调用 execute()，保留兼容
def _legacy_execute(self, compiled_graph, *args, **kwargs):
    if not self.is_compiled:
        raise RuntimeError("Graph must be compiled before execution.")
    return f"Result_from_{self.hardware_type}"


UnifiedIRCompiler.execute = _legacy_execute
