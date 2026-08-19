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


import torch
from torch._library.utils import lookup_op

from cgc_engine.utils import magi_logger


def resolve_defined_ops(op_names: list[str]) -> list["torch._ops.OpOverload"]:
    """Resolve operator names to OpOverload objects.

    Skips operators that fail to resolve (e.g., operators not registered or
    model-specific operators not present in the current model).

    Note: Users should inspect the operator graph before lowering and ensure
    the specified operators are present in the final graph. Built-in PyTorch
    operators (aten::*, torch::*) may be decomposed, fused, or transformed
    during Inductor's compilation passes, so use them with caution.

    Args:
        op_names: List of operator names in PyTorch format
            (e.g., "athena::flash_attn_func")

    Returns:
        List of successfully resolved operator overloads
    """
    resolved = []
    for op_name in op_names:
        try:
            resolved.append(lookup_op(op_name))
        except Exception:
            magi_logger.info(f"Failed to resolve operator for graph partition: {op_name}")
            continue

    return resolved
