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
GDS (GPUDirect Storage) 服务 - 行业首创完整 GDS + PD + CGC 模块
"""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__all__ = [
    "GDSManager",
    "GDSKVStore",
    "GDSWeightLoader",
    "cuFileRead",
    "cuFileWrite",
    "is_gds_available",
    "gds_or_fallback",
]

_EXPORT_MAP = {
    "GDSManager": (".gds_manager", "GDSManager"),
    "GDSKVStore": (".gds_ops", "GDSKVStore"),
    "GDSWeightLoader": (".gds_ops", "GDSWeightLoader"),
    "cuFileRead": (".cufile_wrapper", "cuFileRead"),
    "cuFileWrite": (".cufile_wrapper", "cuFileWrite"),
    "is_gds_available": (".gds_fallback", "is_gds_available"),
    "gds_or_fallback": (".gds_fallback", "gds_or_fallback"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
