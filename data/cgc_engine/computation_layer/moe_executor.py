#!/usr/bin/env python3
"""
CGC Engine Computation Layer
Responsible for: expert prediction, MoE inference
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any

class ExpertPredictor:
    def __init__(self, num_experts: int = 16, expert_dim: int = 4096, device: Optional[torch.device] = None):
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        self.router_device = torch.device("cpu") if device.type in ("mps", "cuda") else device
        self.router = torch.nn.Linear(expert_dim, num_experts).to(self.router_device).float()
    
    def predict(self, x: torch.Tensor, top_k: int = 2) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        routing_feature = x.to(self.router_device, dtype=torch.float32).mean(dim=1)
        logits = self.router(routing_feature)
        tie_break = torch.arange(self.num_experts, device=logits.device, dtype=logits.dtype) * 1e-7
        logits = logits + tie_break
        _, expert_ids = torch.topk(logits, k=top_k, dim=-1)
        expert_ids = expert_ids.unsqueeze(1).expand(-1, seq_len, -1)
        return expert_ids

class MoEExecutor:
    def __init__(self, num_experts: int = 16, expert_dim: int = 4096, intermediate_dim: int = 14336):
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.intermediate_dim = intermediate_dim
        self.expert_weights = {}
    
    def load_expert(self, expert_id: int, weight: Any):
        self.expert_weights[expert_id] = weight
    
    def unload_expert(self, expert_id: int):
        if expert_id in self.expert_weights:
            del self.expert_weights[expert_id]
    
    def moe_forward(self, x: torch.Tensor, expert_ids: torch.Tensor, top_k: int = 2) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        output = torch.zeros(batch_size, seq_len, self.expert_dim, dtype=x.dtype, device=x.device)
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                experts = expert_ids[batch_idx, seq_idx]
                expert_outputs = []
                for eid in experts:
                    eid = eid.item()
                    if eid in self.expert_weights:
                        w1 = self.expert_weights[eid]["w1"]
                        w3 = self.expert_weights[eid]["w3"]
                        w2 = self.expert_weights[eid]["w2"]
                        gate = F.silu(F.linear(x[batch_idx, seq_idx], w1))
                        up = F.linear(x[batch_idx, seq_idx], w3)
                        out = F.linear(gate * up, w2)
                        expert_outputs.append(out)
                if expert_outputs:
                    output[batch_idx, seq_idx] = torch.mean(torch.stack(expert_outputs), dim=0)
        return output

    mlp_forward_moe = moe_forward
