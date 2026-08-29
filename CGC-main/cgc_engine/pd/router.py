"""ComputeRouter - 4D matrix routing for PD mode selection.

Routes based on:
  D1: Network quality (RTT, bandwidth, jitter)
  D2: Hardware capability (RAM, GPU VRAM, tflops, compute_tier)
  D3: Model parameters (params, layers, MoE, quantization, per_layer_gb)
  D4: Runtime state (memory pressure, expert cache hit rate, speculation ROI)

Integrates with Hermes route_decision_v2 FourDMatrixV2 schema.
"""
from __future__ import annotations
import time
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

try:
    from .discovery import DeviceProfile, PDNode, NodeStatus
except ImportError:
    from discovery import DeviceProfile, PDNode, NodeStatus

# ──────────────────────────────────────────────────────────────────────
# Route Decision (4D-aware)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    """4D-aware route decision with prefill/decode latency estimates."""
    mode: Literal[
        "local_full", "edge_cloud", "edge_edge", "pure_cloud",
        "edge_pivot_draft", "edge_draft_cloud_verify", "cloud_only",
    ] = "local_full"
    prefill_node: Optional[PDNode] = None
    decode_node: Optional[PDNode] = None
    reason: str = ""
    confidence: float = 0.0
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    network_ms: float = 0.0
    total_latency_ms: float = 0.0
    # MTP / draft fields (Hermes D4)
    draft_n_tokens: int = 0
    pivot_layer: int = 0
    use_flashmoe: bool = False

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "prefill_node": self.prefill_node.node_id if self.prefill_node else None,
            "decode_node": self.decode_node.node_id if self.decode_node else None,
            "reason": self.reason,
            "confidence": self.confidence,
            "prefill_ms": round(self.prefill_latency_ms, 1),
            "decode_ms": round(self.decode_latency_ms, 1),
            "network_ms": round(self.network_ms, 1),
            "total_ms": round(self.total_latency_ms, 1),
            "draft_n": self.draft_n_tokens,
            "pivot_layer": self.pivot_layer,
        }


# ──────────────────────────────────────────────────────────────────────
# Model presets (from Hermes MODEL_PRESETS)
# ──────────────────────────────────────────────────────────────────────

MODEL_PRESETS: Dict[str, dict] = {
    "qwen36_35b": {
        "params_b": 35.0, "is_moe": True, "num_experts": 64,
        "experts_per_tok": 8, "num_layers": 64, "hidden_size": 2048,
        "model_size_gb": 13.0, "per_layer_gb": 0.203,
        "has_native_mtp": True,
    },
    "gemma4_26b": {
        "params_b": 26.0, "is_moe": True, "num_experts": 64,
        "experts_per_tok": 8, "num_layers": 36, "hidden_size": 3072,
        "model_size_gb": 12.0, "per_layer_gb": 0.333,
        "has_native_mtp": False,
    },
    "qwen25_7b": {
        "params_b": 7.5, "is_moe": False, "num_experts": 0,
        "experts_per_tok": 0, "num_layers": 28, "hidden_size": 3584,
        "model_size_gb": 4.0, "per_layer_gb": 0.143,
        "has_native_mtp": False,
    },
    "qwen25_15b": {
        "params_b": 1.5, "is_moe": False, "num_experts": 0,
        "experts_per_tok": 0, "num_layers": 28, "hidden_size": 1536,
        "model_size_gb": 1.0, "per_layer_gb": 0.036,
        "has_native_mtp": False,
    },
}


# ──────────────────────────────────────────────────────────────────────
# ComputeRouter: 4D matrix based
# ──────────────────────────────────────────────────────────────────────

class ComputeRouter:
    """Route requests across edge/cloud nodes using 4D capability matrix.

    Key insight: routing is NOT based on token length alone.
    It's based on (device_compute, model_params, network_quality, runtime_state).
    """
    def __init__(self, local_node: Optional[PDNode] = None):
        self._nodes: Dict[str, PDNode] = {}
        self._local_node = local_node
        if local_node:
            self._nodes[local_node.node_id] = local_node

    # ── node registry ──

    def register(self, node: PDNode):
        self._nodes[node.node_id] = node

    def deregister(self, node_id: str):
        self._nodes.pop(node_id, None)

    def heartbeat(self, node_id: str, load: float = 0):
        node = self._nodes.get(node_id)
        if node:
            node.last_heartbeat = time.time()
            node.current_load = load

    def get_online_nodes(self) -> List[PDNode]:
        now = time.time()
        return [
            n for n in self._nodes.values()
            if n.status == NodeStatus.HEALTHY and now - n.last_heartbeat < 30
        ]

    # ── 4D scoring ──

    def _device_compute_score(self, node: PDNode) -> float:
        """D2: hardware compute score (0-100)."""
        if not node.profile:
            return 0.0
        p = node.profile
        # Weighted: RAM 25%, GPU VRAM 25%, CPU 20%, network 15%, load 15%
        ram_score = min(p.total_ram_gb / 64.0, 1.0) * 25
        gpu_score = min(p.gpu_vram_gb / 24.0, 1.0) * 25
        cpu_score = min(p.cpu_cores / 16.0, 1.0) * 20
        net_score = max(0, (200 - p.network_latency_ms)) / 200 * 15
        load_score = (1 - node.current_load / 100.0) * 15
        return round(ram_score + gpu_score + cpu_score + net_score + load_score, 1)

    def _can_run_model(self, node: PDNode, model_name: str) -> bool:
        """D2+D3: can this node run this model locally?"""
        if not node.profile or model_name not in MODEL_PRESETS:
            return True  # conservative: assume yes
        preset = MODEL_PRESETS[model_name]
        model_gb = preset["model_size_gb"]
        available = node.profile.total_ram_gb * 0.7  # 70% usable
        if node.profile.gpu_vram_gb > 0:
            available += node.profile.gpu_vram_gb
        return available >= model_gb

    def _prefill_speed(self, node: PDNode, model_name: str) -> float:
        """Estimated prefill tok/s for this node + model."""
        if not node.profile:
            return 10.0
        base = node.profile.prefill_tok_per_sec.get(model_name, 0)
        if base > 0:
            return base
        # Estimate from compute score
        score = self._device_compute_score(node)
        preset = MODEL_PRESETS.get(model_name, {})
        params = preset.get("params_b", 7.0)
        # rough: higher score / larger model = slower
        return max(1.0, score * 3.0 / max(params / 7.0, 0.5))

    def _decode_speed(self, node: PDNode, model_name: str) -> float:
        """Estimated decode tok/s for this node + model."""
        if not node.profile:
            return 2.0
        base = node.profile.decode_tok_per_sec.get(model_name, 0)
        if base > 0:
            return base
        score = self._device_compute_score(node)
        preset = MODEL_PRESETS.get(model_name, {})
        params = preset.get("params_b", 7.0)
        return max(0.5, score * 1.5 / max(params / 7.0, 0.5))

    def _estimate_network_ms(self, a: PDNode, b: PDNode) -> float:
        """D1: estimated network RTT between two nodes."""
        if a == b:
            return 0.0
        # Use measured latency if available
        la = a.profile.network_latency_ms if a.profile else 50
        lb = b.profile.network_latency_ms if b.profile else 50
        # rough: average + overhead
        return (la + lb) / 2 + 5  # 5ms protocol overhead

    def _score_pair(self, prefill: PDNode, decode: PDNode, model: str) -> float:
        """Score a (prefill, decode) pair. Higher = better."""
        pf_score = self._device_compute_score(prefill)
        dc_score = self._device_compute_score(decode)
        net_ms = self._estimate_network_ms(prefill, decode)
        # Same node: no network penalty
        net_penalty = 0 if prefill == decode else min(net_ms / 200, 1.0) * 20
        return round(pf_score * 0.4 + dc_score * 0.4 + (100 - net_penalty) * 0.2, 1)

    # ── routing decisions ──

    def select(
        self,
        prompt_tokens: int = 128,
        output_tokens: int = 100,
        model: str = "qwen36_35b",
        prefer_mode: Optional[str] = None,
    ) -> RouteDecision:
        """Select optimal PD mode + node pair based on 4D matrix.

        Routing logic:
          1. If local can run model + prompt small → local_full
          2. If remote has much higher compute → edge_cloud (prefill on remote)
          3. If 2+ remotes available → edge_edge (distribute)
          4. If no local → pure_cloud / cloud_only
          5. If MTP draft available → edge_pivot_draft / edge_draft_cloud_verify
        """
        online = self.get_online_nodes()
        if not online:
            return RouteDecision(mode="cloud_only", reason="no online nodes")

        local = self._local_node
        preset = MODEL_PRESETS.get(model, {})

        # ── Check memory pressure: if critical, degrade to cloud ──
        if local and local.profile:
            local_ram = local.profile.total_ram_gb
            model_gb = preset.get("model_size_gb", 13.0)
            if local_ram < model_gb * 1.2:
                # Can't fit model locally → must use cloud for at least prefill
                remote = [n for n in online if n != local and n.profile]
                if remote:
                    best_remote = max(remote, key=lambda n: self._device_compute_score(n))
                    pf = self._prefill_speed(best_remote, model)
                    dc = self._decode_speed(local, model) if self._can_run_model(local, model) else pf
                    net_ms = self._estimate_network_ms(best_remote, local)
                    pf_ms = (prompt_tokens / pf) * 1000
                    dc_ms = (output_tokens / dc) * 1000
                    return RouteDecision(
                        mode="edge_cloud",
                        prefill_node=best_remote,
                        decode_node=local if self._can_run_model(local, model) else best_remote,
                        reason=f"local RAM {local_ram:.1f}GB < model {model_gb:.1f}GB → cloud prefill",
                        confidence=0.95,
                        prefill_latency_ms=pf_ms,
                        decode_latency_ms=dc_ms,
                        network_ms=net_ms,
                        total_latency_ms=pf_ms + net_ms + dc_ms,
                    )
                else:
                    return RouteDecision(mode="cloud_only", reason="local OOM, no remote nodes")

        # ── Compute scores for all nodes ──
        local_score = self._device_compute_score(local) if local else 0
        remote_scores = []
        for n in online:
            if n != local:
                remote_scores.append((n, self._device_compute_score(n)))

        # ── Rule 1: local can handle it well → local_full ──
        if local and self._can_run_model(local, model):
            if not remote_scores or local_score >= max(s for _, s in remote_scores) * 0.7:
                pf = self._prefill_speed(local, model)
                dc = self._decode_speed(local, model)
                pf_ms = (prompt_tokens / pf) * 1000
                dc_ms = (output_tokens / dc) * 1000
                return RouteDecision(
                    mode="local_full",
                    prefill_node=local, decode_node=local,
                    reason=f"local score {local_score} sufficient for {model}",
                    confidence=0.9,
                    prefill_latency_ms=pf_ms, decode_latency_ms=dc_ms,
                    total_latency_ms=pf_ms + dc_ms,
                )

        # ── Rule 2: remote much stronger → edge_cloud ──
        if remote_scores:
            best_remote, best_score = max(remote_scores, key=lambda x: x[1])
            if best_score > local_score * 1.3 or not self._can_run_model(local, model):
                pf = self._prefill_speed(best_remote, model)
                # Decode: use local if it can run the model, else remote
                if local and self._can_run_model(local, model):
                    dc_node = local
                    dc = self._decode_speed(local, model)
                else:
                    dc_node = best_remote
                    dc = self._decode_speed(best_remote, model)
                net_ms = self._estimate_network_ms(best_remote, dc_node)
                pf_ms = (prompt_tokens / pf) * 1000
                dc_ms = (output_tokens / dc) * 1000

                # MTP draft bonus
                draft_n = 0
                pivot = 0
                if preset.get("has_native_mtp") and local and local.profile:
                    if local.profile.gpu_vram_gb >= 2:
                        draft_n = 2
                        pivot = preset.get("num_layers", 64) // 4
                        pf_ms *= 0.6  # MTP speeds up prefill

                return RouteDecision(
                    mode="edge_cloud",
                    prefill_node=best_remote,
                    decode_node=dc_node,
                    reason=f"remote {best_remote.node_id} score {best_score} >> local {local_score}",
                    confidence=0.92,
                    prefill_latency_ms=pf_ms,
                    decode_latency_ms=dc_ms,
                    network_ms=net_ms,
                    total_latency_ms=pf_ms + net_ms + dc_ms,
                    draft_n_tokens=draft_n,
                    pivot_layer=pivot,
                )

        # ── Rule 3: multiple remotes → edge_edge ──
        if len(remote_scores) >= 2:
            ranked = sorted(remote_scores, key=lambda x: x[1], reverse=True)
            pf_node, pf_score = ranked[0]
            dc_node, dc_score = ranked[1]
            pf = self._prefill_speed(pf_node, model)
            dc = self._decode_speed(dc_node, model)
            net_ms = self._estimate_network_ms(pf_node, dc_node)
            pf_ms = (prompt_tokens / pf) * 1000
            dc_ms = (output_tokens / dc) * 1000
            return RouteDecision(
                mode="edge_edge",
                prefill_node=pf_node, decode_node=dc_node,
                reason=f"2 remotes: prefill={pf_node.node_id} decode={dc_node.node_id}",
                confidence=0.85,
                prefill_latency_ms=pf_ms, decode_latency_ms=dc_ms,
                network_ms=net_ms, total_latency_ms=pf_ms + net_ms + dc_ms,
            )

        # ── Rule 4: only one remote, local can't run → cloud_only ──
        if remote_scores:
            best_remote, best_score = remote_scores[0]
            pf = self._prefill_speed(best_remote, model)
            dc = pf
            pf_ms = (prompt_tokens / pf) * 1000
            dc_ms = (output_tokens / dc) * 1000
            return RouteDecision(
                mode="cloud_only",
                prefill_node=best_remote, decode_node=best_remote,
                reason=f"only remote {best_remote.node_id}, local can't run {model}",
                confidence=0.88,
                prefill_latency_ms=pf_ms, decode_latency_ms=dc_ms,
                total_latency_ms=pf_ms + dc_ms,
            )

        # ── Fallback: local or nothing ──
        if local:
            pf = self._prefill_speed(local, model)
            dc = self._decode_speed(local, model)
            pf_ms = (prompt_tokens / pf) * 1000
            dc_ms = (output_tokens / dc) * 1000
            return RouteDecision(
                mode="local_full",
                prefill_node=local, decode_node=local,
                reason=f"fallback: only local node",
                confidence=0.7,
                prefill_latency_ms=pf_ms, decode_latency_ms=dc_ms,
                total_latency_ms=pf_ms + dc_ms,
            )

        return RouteDecision(mode="cloud_only", reason="no nodes available")


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate Mac M4 + Windows 8GB + 鸿蒙 PC
    mac = PDNode(
        node_id="mac-m4", host="192.168.1.10", port=8000, status=NodeStatus.HEALTHY,
        profile=DeviceProfile(
            total_ram_gb=64, gpu_vram_gb=0,  # unified memory
            gpu_type="apple_m4_max", cpu_cores=14,
            network_latency_ms=5,
            prefill_tok_per_sec={"qwen36_35b": 120.0, "qwen25_7b": 400.0},
            decode_tok_per_sec={"qwen36_35b": 22.0, "qwen25_7b": 85.0},
        ),
    )
    win = PDNode(
        node_id="win-8gb", host="192.168.1.20", port=8000, status=NodeStatus.HEALTHY,
        profile=DeviceProfile(
            total_ram_gb=8, gpu_vram_gb=2, gpu_type="nvidia_mx250",
            cpu_cores=4, network_latency_ms=5,
            prefill_tok_per_sec={"qwen36_35b": 1.5},
            decode_tok_per_sec={"qwen36_35b": 1.4},
        ),
    )
    router = ComputeRouter(local_node=win)
    router.register(mac)
    router.register(win)

    for model in ["qwen36_35b", "qwen25_7b"]:
        for tokens in [128, 1024, 4096]:
            d = router.select(prompt_tokens=tokens, output_tokens=100, model=model)
            print(f"[{model:12s} {tokens:4d}tok] mode={d.mode:20s} "
                  f"prefill={d.prefill_latency_ms:7.1f}ms "
                  f"decode={d.decode_latency_ms:7.1f}ms "
                  f"total={d.total_latency_ms:7.1f}ms "
                  f"reason={d.reason}")
