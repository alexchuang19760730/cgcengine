"""ComputeRouter - select optimal PD mode + node pair based on DeviceProfile."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from .discovery import DeviceProfile, PDNode, NodeStatus
except ImportError:
    from discovery import DeviceProfile, PDNode, NodeStatus

@dataclass
class RouteDecision:
    mode: str
    prefill_node: Optional[PDNode] = None
    decode_node: Optional[PDNode] = None
    reason: str = ""
    score: float = 0.0
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

SHORT_PROMPT = 512
LONG_PROMPT = 2048

class ComputeRouter:
    def __init__(self, local_node=None):
        self._nodes = {}
        self._local_node = local_node
        if local_node:
            self._nodes[local_node.node_id] = local_node

    def register(self, node):
        self._nodes[node.node_id] = node

    def deregister(self, node_id):
        self._nodes.pop(node_id, None)

    def heartbeat(self, node_id, load=0):
        node = self._nodes.get(node_id)
        if node:
            node.last_heartbeat = time.time()
            node.current_load = load

    def get_online_nodes(self):
        now = time.time()
        return [n for n in self._nodes.values()
                if n.status == NodeStatus.HEALTHY and now - n.last_heartbeat < 30]

    def select(self, prompt_tokens=128, output_tokens=100, model="qwen36_35b", prefer_mode=None):
        online = self.get_online_nodes()
        if not online:
            return RouteDecision(mode="pure_edge", reason="no online nodes")
        mode = prefer_mode or self._select_mode(prompt_tokens, online)
        if mode == "edge_cloud":
            return self._route_edge_cloud(online, model, prompt_tokens, output_tokens)
        elif mode == "edge_edge":
            return self._route_edge_edge(online, model, prompt_tokens, output_tokens)
        elif mode == "pure_cloud":
            return self._route_pure_cloud(online, model, prompt_tokens, output_tokens)
        return self._route_pure_edge(online, model, prompt_tokens, output_tokens)

    def _select_mode(self, prompt_tokens, nodes):
        if prompt_tokens < SHORT_PROMPT:
            return "pure_edge"
        local = self._local_node
        if prompt_tokens > LONG_PROMPT:
            cloud = [n for n in nodes if n != local and n.profile and n.profile.compute_score > 50]
            if cloud:
                return "edge_cloud"
        if local and local.profile:
            best_remote = max((n.profile.compute_score for n in nodes if n != local and n.profile), default=0)
            if best_remote > local.profile.compute_score * 1.5:
                return "edge_cloud"
        return "pure_edge"

    def _route_edge_cloud(self, nodes, model, prompt_tokens, output_tokens):
        local = self._local_node
        remote = [n for n in nodes if n != local and n.profile]
        if not remote:
            return RouteDecision(mode="edge_cloud", reason="no remote nodes")
        prefill = max(remote, key=lambda n: n.profile.compute_score)
        decode = local if local and local in nodes else min(remote, key=lambda n: n.profile.compute_score)
        pf = prefill.profile
        dp = decode.profile
        pf_rate = pf.prefill_tok_per_sec.get(model, 50.0)
        dc_rate = dp.decode_tok_per_sec.get(model, 5.0)
        network_ms = pf.network_latency_ms * 2
        pf_ms = (prompt_tokens / pf_rate) * 1000
        dc_ms = (output_tokens / dc_rate) * 1000
        total = pf_ms + network_ms + dc_ms
        return RouteDecision(mode="edge_cloud", prefill_node=prefill, decode_node=decode,
                           reason=f"prefill={prefill.node_id} decode={decode.node_id}",
                           score=self._score_pair(prefill, decode),
                           prefill_latency_ms=pf_ms, decode_latency_ms=dc_ms, total_latency_ms=total)

    def _route_edge_edge(self, nodes, model, prompt_tokens, output_tokens):
        remote = [n for n in nodes if n.profile]
        if len(remote) < 2:
            return RouteDecision(mode="edge_edge", reason="need 2+ nodes")
        s = sorted(remote, key=lambda n: n.profile.compute_score, reverse=True)
        pf, dc = s[0], s[1]
        pf_rate = pf.profile.prefill_tok_per_sec.get(model, 50.0)
        dc_rate = dc.profile.decode_tok_per_sec.get(model, 5.0)
        pf_ms = (prompt_tokens / pf_rate) * 1000
        dc_ms = (output_tokens / dc_rate) * 1000
        total = pf_ms + pf.profile.network_latency_ms * 2 + dc_ms
        return RouteDecision(mode="edge_edge", prefill_node=pf, decode_node=dc,
                           score=self._score_pair(pf, dc),
                           prefill_latency_ms=pf_ms, decode_latency_ms=dc_ms, total_latency_ms=total)

    def _route_pure_cloud(self, nodes, model, prompt_tokens, output_tokens):
        remote = [n for n in nodes if n.profile]
        if not remote:
            return RouteDecision(mode="pure_cloud", reason="no nodes")
        best = max(remote, key=lambda n: n.profile.compute_score)
        pf_rate = best.profile.prefill_tok_per_sec.get(model, 50.0)
        dc_rate = best.profile.decode_tok_per_sec.get(model, 5.0)
        pf_ms = (prompt_tokens / pf_rate) * 1000
        dc_ms = (output_tokens / dc_rate) * 1000
        return RouteDecision(mode="pure_cloud", prefill_node=best, decode_node=best,
                           score=best.profile.compute_score,
                           prefill_latency_ms=pf_ms, decode_latency_ms=dc_ms, total_latency_ms=pf_ms+dc_ms)

    def _route_pure_edge(self, nodes, model, prompt_tokens, output_tokens):
        local = self._local_node
        if local and local.profile:
            pf_rate = local.profile.prefill_tok_per_sec.get(model, 5.0)
            dc_rate = local.profile.decode_tok_per_sec.get(model, 2.0)
            pf_ms = (prompt_tokens / pf_rate) * 1000
            dc_ms = (output_tokens / dc_rate) * 1000
            return RouteDecision(mode="pure_edge", prefill_node=local, decode_node=local,
                               score=local.profile.compute_score,
                               prefill_latency_ms=pf_ms, decode_latency_ms=dc_ms, total_latency_ms=pf_ms+dc_ms)
        if nodes:
            c = min(nodes, key=lambda n: n.profile.network_latency_ms if n.profile else 9999)
            return RouteDecision(mode="pure_edge", prefill_node=c, decode_node=c)
        return RouteDecision(mode="pure_edge", reason="no nodes")

    def _score_pair(self, prefill, decode):
        pp, dp = prefill.profile, decode.profile
        if not pp or not dp:
            return 0.0
        return round(
            min(pp.total_ram_gb/32,1)*30 + min(pp.gpu_vram_gb/24,1)*25 +
            max(0,100-pp.network_latency_ms)/100*20 +
            (1-prefill.load_factor)*15 + (1-decode.load_factor)*10, 1)
