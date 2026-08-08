from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenEvent:
    token_id: int


@dataclass
class SummaryEvent:
    prompt_token_count: int
    generation_tokens: int
    acceptance_ratio: float
    cycles_completed: int
    elapsed_us: int
    fallback_ar: bool = False
