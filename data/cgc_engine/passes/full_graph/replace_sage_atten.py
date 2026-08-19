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

import torch

try:
    from ...magi_depyf.timeline import emit_pass_lifecycle
except ImportError:
    try:
        from ..._legacy.magi_depyf.timeline import emit_pass_lifecycle
    except ImportError:
        def emit_pass_lifecycle(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator if not callable(args[0]) else args[0]

from ..pass_base import MagiInductorPass


class ReplaceSageAttentionPass(MagiInductorPass):
    """
    A pass to replace flash attention with sage attention.
    """

    def is_applicable(self, graph: torch.fx.Graph, shape: int | None = None) -> bool:
        pass

    @emit_pass_lifecycle
    def __call__(self, graph: torch.fx.Graph):
        pass
