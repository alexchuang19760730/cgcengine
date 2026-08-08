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

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch.fx as fx

from cgc_engine.config import CompileConfig
from cgc_engine.utils import magi_logger

if TYPE_CHECKING:
    from .magi_backend import CompilerManager


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    compiled: bool = False
    runnable: Callable = None  # type: ignore


class PiecewiseBackend:
    def __init__(
        self,
        graph: fx.GraphModule,
        compiled_graph_for_general_shape: Callable,
        compile_config: CompileConfig,
        inductor_compile_config: dict[str, Any],
        piecewise_compile_index: int,
        piecewise_submodule_number: int,
        sym_shape_indices: list[int],
        compiler_manager: "CompilerManager",
    ):
        """
        The backend for piecewise compilation. It mainly handles the compilation of static shapes and dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape, and then compile for different shapes specified in `compile_config.compile_sizes`.
        """
        self.graph = graph
        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape
        self.compile_config = compile_config
        self.inductor_compile_config = inductor_compile_config
        self.piecewise_compile_index = piecewise_compile_index
        self.piecewise_submodule_number = piecewise_submodule_number
        self.compiler_manager = compiler_manager
        self.sym_shape_indices = sym_shape_indices

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == piecewise_submodule_number - 1
        self.is_first_run = True

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = set(self.compile_config.compile_sizes)

        # the entries for different shapes that we need to compile
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}
        for shape in self.to_be_compiled_sizes:
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape, runnable=self.compiled_graph_for_general_shape
            )

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            self.compiler_manager.save_to_file()

    def __call__(self, *args) -> Any:
        if self.is_first_run:
            self.is_first_run = False
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        if len(self.sym_shape_indices) == 0:
            magi_logger.info("No symbolic shape indices found, falling back to general shape compiled graph")
            return self.compiled_graph_for_general_shape(*args)
        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.compiler_manager.compile(
                self.graph,
                args,
                self.inductor_compile_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.piecewise_submodule_number,
                runtime_shape=runtime_shape,
            )

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        return entry.runnable(*args)
