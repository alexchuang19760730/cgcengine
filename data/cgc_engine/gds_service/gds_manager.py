# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
gds_manager.py - PD 统一入口，架构中枢
"""

import torch
import os
from .gds_ops import GDSKVStore, GDSWeightLoader
from .cufile_wrapper import CUFILE_AVAILABLE, get_gds_capabilities


class GDSManager:
    def __init__(self, pd_endpoint: str = "localhost:50051"):
        self.pd_endpoint = pd_endpoint
        self.kv_store = GDSKVStore()
        self.weight_loader = GDSWeightLoader()
        self.enabled = CUFILE_AVAILABLE and torch.cuda.is_available()

    def load_kv_from_pd(self, key: str, seq_len: int, head_dim: int):
        device = "cuda" if self.enabled else "cpu"
        k = torch.empty((1, 32, seq_len, head_dim), device=device, dtype=torch.bfloat16)
        v = torch.empty_like(k)
        if self.enabled:
            k, v = self.kv_store.load_kv(key, device="cuda")
        return k, v

    def save_kv_to_pd(self, key: str, k: torch.Tensor, v: torch.Tensor):
        if self.enabled:
            self.kv_store.save_kv(key, k, v)

    def load_weight_from_pd(self, path: str, shape: list):
        return self.weight_loader.load_weight(path, shape)

    def info(self):
        return {
            "gds_enabled": self.enabled,
            "cufile_available": CUFILE_AVAILABLE,
            "cuda_available": torch.cuda.is_available(),
            "capabilities": get_gds_capabilities(),
        }
