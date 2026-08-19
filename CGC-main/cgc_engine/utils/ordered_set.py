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

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, MutableSet
from typing import Generic, TypeVar

T = TypeVar("T")


class OrderedSet(MutableSet[T], Generic[T]):
    __slots__ = ("_map",)

    def __init__(self, iterable: Iterable[T] | None = None):
        self._map: OrderedDict[T, None] = OrderedDict()
        if iterable:
            self.update(iterable)

    def __contains__(self, x: T) -> bool:
        return x in self._map

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self) -> Iterator[T]:
        return iter(self._map.keys())

    def add(self, value: T) -> None:
        self._map[value] = None

    def discard(self, value: T) -> None:
        self._map.pop(value, None)

    def pop(self, last: bool = True) -> T:
        if not self._map:
            raise KeyError("pop from an empty OrderedSet")
        key = next(reversed(self._map)) if last else next(iter(self._map))
        self._map.pop(key, None)
        return key

    def update(self, iterable: Iterable[T]) -> None:
        for item in iterable:
            self._map[item] = None

    def clear(self) -> None:
        self._map.clear()

    def to_list(self) -> list[T]:
        return list(self._map.keys())

    def copy(self) -> "OrderedSet[T]":
        return OrderedSet(self)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        if not self:
            return f"{cls}()"
        return f"{cls}([{', '.join(repr(x) for x in self)}])"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return list(self) == list(other)
        if isinstance(other, set):
            return set(self) == other
        return NotImplemented

    def union(self, *others: Iterable[T]) -> "OrderedSet[T]":
        result = OrderedSet(self)
        for other in others:
            for x in other:
                if x not in result:
                    result.add(x)
        return result

    __or__ = union

    def intersection(self, *others: Iterable[T]) -> "OrderedSet[T]":
        if not others:
            return self.copy()
        common = set(self)
        for other in others:
            common &= set(other)
        return OrderedSet(x for x in self if x in common)

    __and__ = intersection

    def difference(self, *others: Iterable[T]) -> "OrderedSet[T]":
        remove = set().union(*(set(o) for o in others))
        return OrderedSet(x for x in self if x not in remove)

    __sub__ = difference

    def symmetric_difference(self, other: Iterable[T]) -> "OrderedSet[T]":
        other_set = set(other)
        left = [x for x in self if x not in other_set]
        right = [x for x in other_set if x not in self]
        return OrderedSet([*left, *right])

    __xor__ = symmetric_difference

    # Pydantic v2 compatible: parse/serialize as list
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema as cs

        def validate_from_any(v):
            if isinstance(v, OrderedSet):
                return v
            if v is None:
                return cls()
            try:
                return cls(v)
            except TypeError:
                raise TypeError("OrderedSet must be built from an iterable")

        return cs.no_info_after_validator_function(
            validate_from_any,
            cs.list_schema(cs.any_schema()),
            serialization=cs.plain_serializer_function_ser_schema(lambda v: list(v)),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        json_schema = handler(core_schema)
        json_schema.update({"type": "array"})
        return json_schema
