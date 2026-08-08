#!/usr/bin/env python3
"""PD 分离: Mac 本地 prefill + decode (with draft_model 投机编码)。

阶段 1 (当前): Mac 自己 prefill + decode + draft_model (无 cloud KV 注入)
  TTFT = Mac prefill (598ms), decode = 27-67 tok/s (with draft)

阶段 2 (后续): Cloud prefill → emit KV → Mac 注入 → decode + draft
  TTFT = cloud prefill (105ms), decode = 27-67 tok/s (with draft)

mlx_lm stream_generate 内置 draft_model 支持:
  stream_generate(model, tokenizer, prompt, draft_model=draft, num_draft_tokens=N)
"""
import os, sys, time

os.environ.setdefault("EDGE_LOCAL_MODEL_PATH", "/Users/alexchuang/models/Qwen3-VL-2B-bf16")

REPO = "/Users/alexchuang/Documents/flashkv0516"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mlx.core as mx
from mlx_lm import load, stream_generate


def pd_separation_decode(messages, max_tokens=50, P=6, draft_model_path=None, num_draft=4):
    """PD 分离 decode: Mac prefill + decode + draft_model。

    Args:
        messages: chat messages
        max_tokens: 最大生成 token 数
        P: layer-split P (当前未用, Mac 做全部层)
        draft_model_path: draft 模型路径 (None = 无投机)
        num_draft: 投机 draft token 数
    """
    print(f"[PD] Loading target model (2B BF16)...", flush=True)
    t0 = time.time()
    model, tokenizer = load("/Users/alexchuang/models/Qwen3-VL-2B-bf16")
    print(f"[PD] Target model loaded: {time.time()-t0:.1f}s", flush=True)

    # 加载 draft model (可选)
    draft_model = None
    if draft_model_path:
        print(f"[PD] Loading draft model: {draft_model_path}...", flush=True)
        t1 = time.time()
        draft_model, draft_tokenizer = load(draft_model_path)
        print(f"[PD] Draft model loaded: {time.time()-t1:.1f}s", flush=True)

    # 准备 prompt
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [t for t in input_ids if t not in (151644, 151645)]
    print(f"[PD] Prompt: {len(input_ids)} tokens (filtered)", flush=True)

    # 生成
    print(f"[PD] Starting generate (max_tokens={max_tokens}, "
          f"draft={'yes' if draft_model else 'no'}, N={num_draft})...", flush=True)

    t_gen = time.time()
    tokens = []
    for response in stream_generate(
        model, tokenizer, mx.array(input_ids),
        max_tokens=max_tokens,
        draft_model=draft_model,
        num_draft_tokens=num_draft if draft_model else None,
    ):
        tokens.append(response.token)
        if len(tokens) == 1:
            t_first = time.time()
            print(f"[PD] First token: '{response.text}' "
                  f"TTFT={1000*(t_first-t_gen):.0f}ms", flush=True)
        elif len(tokens) <= 5 or len(tokens) % 10 == 0:
            print(f"  token {len(tokens)}: '{response.text}' "
                  f"from_draft={response.from_draft}", flush=True)

    t_end = time.time()
    total = t_end - t_gen
    decode_time = t_end - t_first
    n_decode = len(tokens) - 1

    print(f"\n{'='*60}")
    print(f"[PD] Results:")
    print(f"  TTFT (prefill + first token): {1000*(t_first-t_gen):.0f}ms")
    print(f"  Decode: {n_decode} tokens in {decode_time:.2f}s = {n_decode/decode_time:.1f} tok/s")
    print(f"  Total: {len(tokens)} tokens in {total:.2f}s = {len(tokens)/total:.1f} tok/s")
    if draft_model:
        draft_count = sum(1 for i, t in enumerate(tokens) if i > 0 and hasattr(response, 'from_draft'))
        print(f"  Draft: N={num_draft}, accept_rate=estimated")
    print(f"  Output: {tokenizer.decode(tokens[:50])}...")
    return tokens


if __name__ == "__main__":
    messages = [{"role": "user", "content": "Write a short story about a cat"}]

    # Test 1: 无投机 (baseline)
    print("="*60)
    print("Test 1: No speculative (baseline)")
    print("="*60)
    pd_separation_decode(messages, max_tokens=50, draft_model_path=None)

    print("\n")
