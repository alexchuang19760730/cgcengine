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

from torch import fx
from torch._inductor.pattern_matcher import stable_topological_sort

from ...magi_depyf.timeline import emit_pass_lifecycle
from ..pass_base import MagiInductorPass


class PostCleanupPass(MagiInductorPass):
    """
    This pass performs cleanup after custom passes.
    It topologically sorts the graph and removes unused nodes.
    This is needed because the pattern matcher does not guarantee producing
    a topologically sorted graph, and there may be unused nodes left around.
    """

    @emit_pass_lifecycle
    def __call__(self, graph: fx.Graph) -> None:
        stable_topological_sort(graph)
        graph.eliminate_dead_code()
