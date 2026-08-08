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
gds_ops.py - KV Cache + 权重加载操作
"""

import torch
import os
from .cufile_wrapper import cuFileRead, cuFileWrite


class GDSKVStore:
    def __init__(self, root: str = "/tmp/gds_kv"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def save_kv(self, key: str, k: torch.Tensor, v: torch.Tensor):
        path = os.path.join(self.root, f"{key}.kv")
        data = torch.cat([k.contiguous(), v.contiguous()])
        cuFileWrite(path, data)

    def load_kv(self, key: str, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        path = os.path.join(self.root, f"{key}.kv")
        info = os.stat(path)
        buf = torch.empty(info.st_size, dtype=torch.uint8, device=device)
        cuFileRead(path, buf)
        half = buf.numel() // 2
        k = buf[:half].view(torch.bfloat16)
        v = buf[half:].view(torch.bfloat16)
        return k, v


class GDSWeightLoader:
    def __init__(self):
        self.cache = {}

    def load_weight(self, path: str, shape: list, dtype: torch.dtype = torch.float16):
        dev = torch.device("cuda")
        numel = torch.prod(torch.tensor(shape))
        tensor = torch.empty(numel, dtype=dtype, device=dev)
        cuFileRead(path, tensor)
        return tensor.view(shape)
