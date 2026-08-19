# Copyright (c) 2025 SandAI. All Rights Reserved.
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

from typing import Any

import torch
from torch import fx
from torch.fx.experimental.symbolic_shapes import is_symbolic

from cgc_engine.utils.logger import magi_logger


def is_func(node: fx.Node, target) -> bool:
    return node.op == "call_function" and node.target == target


def detect_symbolic_tensor_indices(fake_args: list[Any]) -> list[int]:
    """Detect indices of input tensors that have symbolic shapes."""
    sym_tensor_indices = [
        i
        for i, x in enumerate(fake_args)
        if isinstance(x, torch._subclasses.fake_tensor.FakeTensor) and any(is_symbolic(d) for d in x.size())
    ]
    if sym_tensor_indices:
        magi_logger.info(f"Detected {len(sym_tensor_indices)} symbolic input tensors (dynamic seqlen) for CUDA Graph.")
    return sym_tensor_indices
