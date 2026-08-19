from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PrefixCacheFlow:
    snapshot: Any = None
    snapshot_service: Any = None
    stable_prefix_len: int = 0
    cache_active: bool = False

    @staticmethod
    def for_request(
        *,
        model_provider: Any,
        draft_model: Any,
        tokenizer: Any,
        prompt: list[int],
        runtime_context: Any,
    ) -> "PrefixCacheFlow":
        return PrefixCacheFlow()

    def publish_generation_snapshot(self, *args: Any, **kwargs: Any) -> None:
        return None
