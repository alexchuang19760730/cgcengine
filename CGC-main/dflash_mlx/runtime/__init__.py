from __future__ import annotations

import importlib
import time
from typing import Any, Iterable

from dflash_mlx.engine.events import SummaryEvent, TokenEvent


def get_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids:
        return [int(x) for x in eos_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        return []
    return [int(eos_id)]


def _normalize_prompt(prompt: str, prompt_tokens_override: Any) -> str | list[int]:
    if prompt_tokens_override is not None:
        if isinstance(prompt_tokens_override, list):
            return prompt_tokens_override
        try:
            return list(prompt_tokens_override)
        except TypeError:
            return prompt
    return prompt


def stream_dflash_generate(
    *,
    target_model: Any,
    target_ops: Any,
    tokenizer: Any,
    draft_model: Any,
    draft_backend: Any,
    prompt: str,
    max_new_tokens: int,
    stop_token_ids: list[int] | None = None,
    prompt_tokens_override: Any = None,
    prefix_snapshot: Any = None,
    snapshot_service: Any = None,
    stable_prefix_len: int = 0,
    prefix_cache_active: bool = False,
    publish_generation_snapshot: Any = None,
    runtime_context: Any = None,
) -> Iterable[object]:
    del target_ops, draft_model, draft_backend
    del prefix_snapshot, snapshot_service, stable_prefix_len
    del prefix_cache_active, publish_generation_snapshot, runtime_context

    generate_mod = importlib.import_module("mlx_lm.generate")
    stream_generate = getattr(generate_mod, "stream_generate")

    normalized_prompt = _normalize_prompt(prompt, prompt_tokens_override)
    prompt_token_count = (
        len(normalized_prompt)
        if isinstance(normalized_prompt, list)
        else len(tokenizer.encode(normalized_prompt))
    )
    started = time.perf_counter()
    emitted_generation_tokens = 0
    final_response = None

    for response in stream_generate(
        model=target_model,
        tokenizer=tokenizer,
        prompt=normalized_prompt,
        max_tokens=max_new_tokens,
    ):
        final_response = response
        generation_tokens = int(getattr(response, "generation_tokens", 0) or 0)
        token = int(getattr(response, "token", -1))
        should_emit_token = generation_tokens > emitted_generation_tokens
        if should_emit_token and token >= 0:
            emitted_generation_tokens = generation_tokens
            yield TokenEvent(token_id=token)

    elapsed_us = int((time.perf_counter() - started) * 1_000_000)
    final_generation_tokens = emitted_generation_tokens
    if final_response is not None:
        final_generation_tokens = int(
            getattr(final_response, "generation_tokens", emitted_generation_tokens) or 0
        )
        prompt_token_count = int(
            getattr(final_response, "prompt_tokens", prompt_token_count) or prompt_token_count
        )

    yield SummaryEvent(
        prompt_token_count=prompt_token_count,
        generation_tokens=final_generation_tokens,
        acceptance_ratio=1.0 if final_generation_tokens > 0 else 0.0,
        cycles_completed=max(final_generation_tokens, 1) if max_new_tokens > 0 else 0,
        elapsed_us=elapsed_us,
        fallback_ar=False,
    )
