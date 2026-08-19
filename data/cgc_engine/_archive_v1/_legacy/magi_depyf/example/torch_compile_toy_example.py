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

"""Plain torch.compile + explain_compilation demo.

Run:
    PYTHONPATH=. python pkgs/MagiCompiler/cgc_engine/magi_depyf/example/torch_compile_toy_example.py
"""

import os
import shutil

import torch

from cgc_engine.magi_depyf.inspect import explain_compilation

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


@torch.compile
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = -b
    return x * b


def main() -> None:
    out = "./magi_depyf_torch_compile_debug"
    if os.path.exists(out):
        shutil.rmtree(out)

    with explain_compilation(out):
        for _ in range(10):
            toy_example(torch.randn(10), torch.randn(10))

    print(f"debug output: {out}")


if __name__ == "__main__":
    main()
