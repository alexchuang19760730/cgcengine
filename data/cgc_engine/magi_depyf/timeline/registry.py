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

from collections.abc import Callable
from typing import Any

AttrsResolver = Callable[[str, tuple[Any, ...], dict[str, Any], Any | None, Exception | None], dict[str, Any] | None]

_attrs_resolvers: dict[str, AttrsResolver] = {}


def register_attrs_resolver(
    lifecycle_name: str, resolver: AttrsResolver | None = None
) -> AttrsResolver | Callable[[AttrsResolver], AttrsResolver]:
    if resolver is not None:
        _attrs_resolvers[lifecycle_name] = resolver
        return resolver

    def decorator(fn: AttrsResolver) -> AttrsResolver:
        _attrs_resolvers[lifecycle_name] = fn
        return fn

    return decorator


def get_attrs_resolver(lifecycle_name: str) -> AttrsResolver | None:
    return _attrs_resolvers.get(lifecycle_name)


def clear_attrs_resolvers() -> None:
    _attrs_resolvers.clear()
