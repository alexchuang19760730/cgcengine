#!/usr/bin/env python3
"""Test sglang return_hidden_states API.

1. Send a simple request with return_hidden_states=True
2. Check if hidden states are returned
3. Inspect their shape and dtype
4. Test with both Qwen3-VL (port 30003) and Gemma4 (port 30000)
"""
import json
import urllib.request
import sys
import numpy as np

def test_hidden_states(sglang_url: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"Testing return_hidden_states on {model_name}")
    print(f"URL: {sglang_url}")
    print(f"{'='*60}")

    # Simple test prompt
    payload = {
        "text": "Hello, how are you?",
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
        },
        "return_hidden_states": True,
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 1,
    }

    req = urllib.request.Request(
        f"{sglang_url}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    meta = result.get("meta_info", {})

    # Check for hidden states in various possible locations
    hidden_states = meta.get("hidden_states", None)
    output_ids = result.get("output_ids", [])

    print(f"  output_ids: {output_ids}")
    print(f"  meta keys: {list(meta.keys())}")

    if hidden_states is not None:
        print(f"  hidden_states type: {type(hidden_states)}")
        if isinstance(hidden_states, list):
            print(f"  hidden_states length: {len(hidden_states)}")
            if len(hidden_states) > 0:
                first = hidden_states[0]
                print(f"  first element type: {type(first)}")
                if isinstance(first, list):
                    print(f"  first element length: {len(first)}")
                    if len(first) > 0:
                        print(f"  first[0] type: {type(first[0])}")
                        if isinstance(first[0], (int, float)):
                            print(f"  first[0:5]: {first[:5]}")
                        elif isinstance(first[0], list):
                            print(f"  first[0] length: {len(first[0])}")
                            print(f"  first[0][0:5]: {first[0][:5]}")
                elif isinstance(first, str):
                    # Might be base64 encoded
                    print(f"  first element (str, first 100 chars): {first[:100]}")
        else:
            print(f"  hidden_states (raw, first 200): {str(hidden_states)[:200]}")
    else:
        print(f"  hidden_states: NOT FOUND in meta_info")
        print(f"  Full meta_info keys: {list(meta.keys())}")
        # Check top-level
        if "hidden_states" in result:
            print(f"  hidden_states found at top level!")
            hs = result["hidden_states"]
            print(f"  type: {type(hs)}, len: {len(hs) if hasattr(hs, '__len__') else 'N/A'}")

    # Also check input_token_logprobs for prompt token IDs
    input_lp = meta.get("input_token_logprobs", [])
    print(f"\n  input_token_logprobs: {len(input_lp)} entries")
    if input_lp:
        print(f"  first entry: {input_lp[0]}")

    input_top_lp = meta.get("input_top_logprobs", [])
    print(f"  input_top_logprobs: {len(input_top_lp)} entries")

    return hidden_states is not None


def test_hidden_states_with_ids(sglang_url: str, model_name: str):
    """Test with input_ids instead of text, and with max_new_tokens=0."""
    print(f"\n{'='*60}")
    print(f"Testing return_hidden_states with input_ids on {model_name}")
    print(f"{'='*60}")

    # First tokenize
    tok_payload = {
        "text": "def hello():",
        "sampling_params": {"max_new_tokens": 0},
        "return_hidden_states": True,
    }

    req = urllib.request.Request(
        f"{sglang_url}/generate",
        data=json.dumps(tok_payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    meta = result.get("meta_info", {})
    print(f"  meta keys: {list(meta.keys())}")
    print(f"  output_ids: {result.get('output_ids', [])}")

    hidden_states = meta.get("hidden_states")
    if hidden_states is not None:
        print(f"  hidden_states FOUND! type={type(hidden_states)}")
        if isinstance(hidden_states, list):
            print(f"  length: {len(hidden_states)}")
            if len(hidden_states) > 0:
                elem = hidden_states[0]
                if isinstance(elem, list):
                    print(f"  elem[0] type: {type(elem[0]) if elem else 'empty'}, len={len(elem)}")
                    if elem and isinstance(elem[0], (int, float)):
                        arr = np.array(hidden_states)
                        print(f"  numpy shape: {arr.shape}")
                        print(f"  dtype: {arr.dtype}")
                        print(f"  sample values: {arr.flatten()[:10]}")
                elif isinstance(elem, str):
                    import base64
                    print(f"  base64 string, first 80 chars: {elem[:80]}")
                    try:
                        decoded = base64.b64decode(elem)
                        arr = np.frombuffer(decoded, dtype=np.float32)
                        print(f"  decoded length: {len(arr)} floats")
                        print(f"  first 10: {arr[:10]}")
                    except Exception as e:
                        print(f"  base64 decode failed: {e}")
        return True
    else:
        print(f"  hidden_states NOT FOUND")
        # Try alternative param names
        for alt_name in ["hidden_state", "hidden_states_all", "hidden_layers", "layers"]:
            val = meta.get(alt_name) or result.get(alt_name)
            if val is not None:
                print(f"  Found {alt_name}! type={type(val)}")

    return False


if __name__ == "__main__":
    # Test Qwen3-VL on port 30003
    qwen_ok = test_hidden_states("http://127.0.0.1:30003", "Qwen3-VL 2B")

    # Test Gemma4 on port 30000
    gemma_ok = test_hidden_states("http://127.0.0.1:30000", "Gemma4 26B")

    # Test with input_ids
    if not qwen_ok:
        test_hidden_states_with_ids("http://127.0.0.1:30003", "Qwen3-VL 2B")
    if not gemma_ok:
        test_hidden_states_with_ids("http://127.0.0.1:30000", "Gemma4 26B")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Qwen3-VL hidden_states: {'YES' if qwen_ok else 'NO'}")
    print(f"  Gemma4 hidden_states: {'YES' if gemma_ok else 'NO'}")
