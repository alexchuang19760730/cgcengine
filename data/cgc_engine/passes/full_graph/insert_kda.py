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

"""
Insert KDA Pass - Insert Kimi KDA instructions into the computation graph.
"""

from typing import Any, Optional, Dict


class InsertKDAPass:
    def __init__(self, **kwargs):
        from cgc_engine.cgc.kda_pass import InsertKDAPass as _RealInsertKDAPass

        self._kwargs = dict(kwargs)
        self._impl = _RealInsertKDAPass(**kwargs)

    def is_applicable(self, graph, shape=None):
        ok = bool(self._impl.is_applicable(graph, shape))
        op_hist: Dict[str, int] = {}
        try:
            for n in getattr(graph, "nodes", []):
                try:
                    op = str(getattr(n, "op", ""))
                    tgt = str(getattr(n, "target", ""))
                    k = f"{op}:{tgt}" if tgt else op
                except Exception:
                    k = "unknown"
                op_hist[k] = int(op_hist.get(k, 0)) + 1
        except Exception:
            op_hist = {}
        if not ok:
            try:
                from cgc_engine.cgc.kda_pass import _cgc_write_vllm_gate_stats  # type: ignore

                payload = {
                    "kind": "cgc_vllm_gate",
                    "pass": "InsertKDAPass",
                    "kda_patterns_detected": 0,
                    "kda_commands_inserted": 0,
                    "kda_chunk_inserted": 0,
                    "kda_project_inserted": 0,
                    "kda_ortho_update_inserted": 0,
                    "ortho_basis_update_inserted": 0,
                    "use_gate": bool(self._kwargs.get("use_gate", False)),
                    "enable_ortho_basis_update": bool(self._kwargs.get("enable_ortho_basis_update", False)),
                    "enable_flashkda_fusion": bool(self._kwargs.get("enable_flashkda_fusion", False)),
                    "kda_scale": float(self._kwargs.get("kda_scale", 1.0)),
                    "ortho_kda_base_dim": int(self._kwargs.get("ortho_kda_base_dim", 0)),
                    "applicable": False,
                    "op_histogram": dict(sorted(op_hist.items(), key=lambda kv: kv[1], reverse=True)[:40]),
                }
                _cgc_write_vllm_gate_stats(payload)
            except Exception:
                pass
            if bool(self._kwargs.get("use_gate", False)):
                raise RuntimeError("CGC_VLLM_GATE_FAIL:kda_not_applicable")
        return ok

    def __call__(self, graph):
        return self._impl(graph)
