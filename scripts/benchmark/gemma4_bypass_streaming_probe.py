#!/usr/bin/env python3
import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "app" / "edge_engine"))

import mlx.core as mx  # type: ignore
from omlx_mlx_engine import OMLXMLXEngine  # type: ignore


def _configure_metal() -> None:
    try:
        mx.set_memory_limit(14 * 1024 * 1024 * 1024)
    except Exception:
        try:
            mx.metal.set_memory_limit(14 * 1024 * 1024 * 1024)
        except Exception:
            pass
    try:
        mx.set_cache_limit(2 * 1024 * 1024 * 1024)
    except Exception:
        try:
            mx.metal.set_cache_limit(2 * 1024 * 1024 * 1024)
        except Exception:
            pass


def _tokenize_prompt(engine: OMLXMLXEngine, prompt: str):
    if hasattr(engine.tokenizer, "chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        formatted = engine.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        tokens = engine.tokenizer.encode(formatted)
    elif hasattr(engine.tokenizer, "encode"):
        tokens = engine.tokenizer.encode(prompt)
    else:
        tokens = engine.tokenizer(prompt)
    return mx.array(tokens)


def _run_generate_step(engine: OMLXMLXEngine, prompt: str, max_tokens: int) -> dict:
    from mlx_lm.generate import generate_step

    prompt_arr = _tokenize_prompt(engine, prompt)
    t0 = time.perf_counter()
    first_token_s = None
    pieces: List[str] = []
    completion_tokens = 0
    for token, _logprobs in generate_step(prompt_arr, engine.model, max_tokens=max_tokens):
        if first_token_s is None:
            first_token_s = time.perf_counter() - t0
        token_id = int(token) if isinstance(token, mx.array) else int(token)
        pieces.append(engine.tokenizer.decode([token_id]))
        completion_tokens += 1
        if engine.manager and engine.manager.lazy_stats:
            engine.manager.maybe_flush()
    if engine.manager and engine.manager.lazy_stats and engine.manager.stats_mode != "off":
        engine.manager.flush_pending()
    elapsed_s = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed_s,
        "ttft_ms": (first_token_s or elapsed_s) * 1000.0,
        "completion_tokens": completion_tokens,
        "tokens_per_second": (completion_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
        "output_preview": "".join(pieces)[:120],
    }


def _benchmark_mode(model_path: str, prompt: str, max_tokens: int, warmup: int, *, enable_streaming: bool, cache_size: int) -> dict:
    engine = OMLXMLXEngine(
        model_path=model_path,
        enable_streaming=enable_streaming,
        streaming_config={
            "max_experts_in_memory": cache_size,
            "stats_mode": "off",
            "lazy_stats": True,
            "enable_io_simulation": False,
        },
    )
    engine.load()
    for _ in range(warmup):
        _run_generate_step(engine, prompt, min(8, max_tokens))

    result = _run_generate_step(engine, prompt, max_tokens)
    stats = engine.get_stats()
    return {
        "mode": "streaming" if enable_streaming else "bypass",
        "elapsed_s": result["elapsed_s"],
        "ttft_ms": result["ttft_ms"],
        "completion_tokens": result["completion_tokens"],
        "tokens_per_second": result["tokens_per_second"],
        "output_preview": result["output_preview"],
        "streaming_stats": stats.summary() if stats else "N/A",
        "hit_rate": stats.hit_rate if stats else None,
        "swaps": stats.total_swaps if stats else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Gemma4 bypass vs expert streaming with exact first-yield TTFT.")
    ap.add_argument("--model", default=str(REPO_ROOT / "models" / "gemma-4-26B-A4B-it-qat-4bit"))
    ap.add_argument("--prompt", default="Explain in one short sentence why expert streaming can reduce memory but hurt latency.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--streaming-cache-size", type=int, default=2)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    _configure_metal()
    run_id = f"gemma4_bypass_streaming_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    results = [
        _benchmark_mode(args.model, args.prompt, args.max_tokens, args.warmup, enable_streaming=False, cache_size=args.streaming_cache_size),
        _benchmark_mode(args.model, args.prompt, args.max_tokens, args.warmup, enable_streaming=True, cache_size=args.streaming_cache_size),
    ]
    bypass = next((r for r in results if r["mode"] == "bypass"), {})
    streaming = next((r for r in results if r["mode"] == "streaming"), {})
    summary = {
        "run_id": run_id,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "results": results,
        "ratios": {
            "streaming_vs_bypass_tps": (
                float(streaming.get("tokens_per_second") or 0.0) / float(bypass.get("tokens_per_second") or 1.0)
                if bypass.get("tokens_per_second") else None
            ),
            "streaming_vs_bypass_ttft": (
                float(streaming.get("ttft_ms") or 0.0) / float(bypass.get("ttft_ms") or 1.0)
                if bypass.get("ttft_ms") else None
            ),
        },
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
