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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import dataclasses
from contextlib import contextmanager


@dataclasses.dataclass
class CompilationCounter:
    # How many models are decorated by @magi_compile decorator
    num_models_seen: int = 0
    # The number of graphs seen by Dynamo, usually the same as num_models_seen
    # NOTE: __init__ updates num_models_seen but __call__ updates num_graphs_seen,
    # we also use num_models_seen for cache key
    num_graphs_seen: int = 0
    # Total number of subgraphs, which includes the splitting ops
    num_piecewise_graphs_seen: int = 0
    # Total number of subgraphs that are captured, which does not include the splitting ops
    num_piecewise_capturable_graphs_seen: int = 0
    # Total number of subgraphs that are compiled by the backend, which does not include the splitting ops
    num_backend_compilations: int = 0
    # The number of cached graphs
    num_cache_entries: int = 0
    # The number of InductorStandaloneAdaptor.compile calls
    num_inductor_compiles: int = 0
    # The number of standalone_compile compiled artifacts saved, should be 0 if MAGI_DISABLE_COMPILE_CACHE is true
    num_compiled_artifacts_saved: int = 0
    # The number of EagerAdaptor.compile calls
    num_eager_compiles: int = 0
    # # Number of gpu_model_runner attempts to trigger CUDAGraphs capture
    # num_gpu_runner_capture_triggers: int = 0
    # # Number of CUDAGraphs captured
    # num_cudagraph_captured: int = 0

    def accuracy_check(self):
        # check the consistency of the counters
        assert self.num_models_seen >= self.num_graphs_seen
        assert self.num_piecewise_graphs_seen >= self.num_piecewise_capturable_graphs_seen
        assert self.num_piecewise_capturable_graphs_seen == self.num_backend_compilations
        assert self.num_cache_entries == (self.num_inductor_compiles + self.num_eager_compiles)
        assert self.num_inductor_compiles == 0 or self.num_eager_compiles == 0
        assert self.num_compiled_artifacts_saved == 0 or self.num_compiled_artifacts_saved == self.num_inductor_compiles

    def clone(self) -> "CompilationCounter":
        return copy.deepcopy(self)

    @contextmanager
    def expect(self, **kwargs):
        old = self.clone()
        yield
        for k, v in kwargs.items():
            assert getattr(self, k) - getattr(old, k) == v, (
                f"{k} not as expected, before it is {getattr(old, k)}"
                f", after it is {getattr(self, k)}, "
                f"expected diff is {v}"
            )


compilation_counter = CompilationCounter()
