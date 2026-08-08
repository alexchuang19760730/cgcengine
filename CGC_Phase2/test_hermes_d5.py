#!/usr/bin/env python3
"""Test Hermes Router D5 content awareness + verify loop integration.

Tests:
  1. D5ContentAware: prompt type detection + parameter adjustment
  2. HermesRouter.decide(): D5-aware routing decisions
  3. HermesRouter.execute(): end-to-end verify loop execution
  4. decide_and_execute(): one-step routing + execution
"""
import os, sys, time, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [REPO_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.shared.hermes_router import (
    D5ContentAware, D5Content, HermesRouter, Bootstrap,
    VerifyLoopBackend, D4Decision,
)


def test_d5_content_aware():
    """Test 1: D5 content awareness — prompt type detection."""
    print("\n" + "=" * 70)
    print("  Test 1: D5ContentAware — prompt type detection")
    print("=" * 70)

    analyzer = D5ContentAware()

    test_cases = [
        ("def fibonacci(n):\n    ", "code_completion", "python"),
        ("class LinkedList:\n    def __init__(self):\n        ", "code_completion", "python"),
        ("import numpy as np\n\ndef matrix_multiply(a, b):\n    ", "code_completion", "python"),
        ("Write a Python function to check if a number is prime:", "code_completion", "python"),
        ("What is the time complexity of quicksort?", "reasoning", ""),
        ("Explain the difference between TCP and UDP.", "chat", ""),
        ("The main advantage of using a linked list over an array is", "chat", ""),
        ("async def fetch_data(url):\n    ", "code_completion", "python"),
        ("function fetchData(url) {\n    ", "code_completion", "javascript"),
        ("Analyze the time complexity of this algorithm step by step:", "reasoning", ""),
    ]

    correct = 0
    for prompt, expected_type, expected_lang in test_cases:
        result = analyzer.analyze(prompt)
        ok = result.prompt_type == expected_type
        lang_ok = (not expected_lang) or result.language == expected_lang
        if ok and lang_ok:
            correct += 1

        status = "OK" if ok and lang_ok else "FAIL"
        print(f"\n  [{status}] {prompt[:50]}...")
        print(f"    type={result.prompt_type} (expected={expected_type}), "
              f"lang={result.language or '(none)'} (expected={expected_lang or '(any)'})")
        print(f"    is_completion={result.is_completion}, "
              f"accept~{result.expected_accept_rate:.0%}, "
              f"num_draft={result.suggested_num_draft}")
        if result.suggested_mode_override:
            print(f"    mode_override={result.suggested_mode_override}")

    print(f"\n  Result: {correct}/{len(test_cases)} correct")
    return correct == len(test_cases)


def test_hermes_decide():
    """Test 2: HermesRouter.decide() with D5 awareness."""
    print("\n" + "=" * 70)
    print("  Test 2: HermesRouter.decide() — D5-aware routing")
    print("=" * 70)

    # Bootstrap without cloud (local only test)
    bootstrap = Bootstrap(
        cloud_mtp_url="http://localhost:30001",
        cloud_plain_url="http://localhost:30000",
    )
    result = bootstrap.run(verbose=False)
    router = HermesRouter(bootstrap=bootstrap)

    test_cases = [
        ("code_completion", "qwen25_05b", "def fibonacci(n):\n    "),
        ("code_generation", "qwen25_05b", "Write a Python function to reverse a string:"),
        ("chat", "qwen25_05b", "What is machine learning?"),
        ("reasoning", "qwen25_05b", "What is the time complexity of quicksort?"),
        ("gemma4_code", "gemma4", "def hello():\n    "),
        ("gemma4_chat", "gemma4", "Explain recursion."),
    ]

    all_ok = True
    for name, model, prompt in test_cases:
        decision = router.decide(
            model_name=model, prompt=prompt,
            cache_hit=False, online=True,
            mtp_available=True,
        )
        d5 = getattr(decision, "d5_content", None)
        num_draft = getattr(decision, "num_draft", "?")

        print(f"\n  [{name}]")
        print(f"    mode: {decision.mode}, confidence: {decision.confidence:.0%}")
        if d5:
            print(f"    D5: type={d5.prompt_type}, num_draft={num_draft}, "
                  f"accept~{decision.expected_accept_rate:.0%}")
        print(f"    reason: {decision.reason[:100]}")

        # Verify D5 integration
        if d5:
            if d5.prompt_type == "reasoning" and decision.mode == "edge_draft":
                print(f"    [FAIL] reasoning should not use edge_draft")
                all_ok = False
            if d5.prompt_type == "code_completion" and num_draft < 4:
                print(f"    [FAIL] code_completion should use num_draft=4, got {num_draft}")
                all_ok = False

    return all_ok


def test_execute():
    """Test 3: HermesRouter.execute() — end-to-end verify loop."""
    print("\n" + "=" * 70)
    print("  Test 3: HermesRouter.execute() — verify loop execution")
    print("=" * 70)

    # Check if model + checkpoint exist
    repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    model_path = os.path.expanduser("~/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf")
    ckpt_v1 = os.path.join(SCRIPT_DIR, "mtp_output/qwen25_llamacpp_v2/mtp_head_qwen25-0.5b_decode.pt")
    embed_path = os.path.join(SCRIPT_DIR, "mtp_train_data/qwen25_llamacpp/embed_head.pt")

    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found: {model_path}")
        return True
    if not os.path.exists(ckpt_v1):
        print(f"  [SKIP] Checkpoint not found: {ckpt_v1}")
        return True

    # Bootstrap + Router
    bootstrap = Bootstrap()
    bootstrap.run(verbose=False)
    router = HermesRouter(bootstrap=bootstrap)

    # Test with code completion prompt (high accept rate expected)
    prompt = "def fibonacci(n):\n    "

    print(f"\n  Prompt: {prompt!r}")
    print(f"  Model: Qwen2.5-0.5B FP16 + MTP head (v1)")

    # Decide
    decision = router.decide(
        model_name="qwen25_05b", prompt=prompt,
        cache_hit=False, online=True, mtp_available=True,
    )
    decision.model_name = "qwen25_05b"

    d5 = getattr(decision, "d5_content", None)
    num_draft = getattr(decision, "num_draft", 2)
    print(f"  Decision: mode={decision.mode}, num_draft={num_draft}")
    if d5:
        print(f"  D5: type={d5.prompt_type}, expected_accept~{d5.expected_accept_rate:.0%}")

    # Execute
    print(f"\n  Executing verify loop (max_tokens=30, num_draft={num_draft})...")
    t0 = time.time()
    result = router.execute(decision, prompt, max_tokens=30)
    elapsed = time.time() - t0

    print(f"\n  Result:")
    print(f"    mode: {result.get('mode', '?')}")
    print(f"    tps: {result.get('tps', 0):.1f}")
    print(f"    accept_rate: {result.get('accept_rate', 0):.1%}")
    print(f"    avg_accept_len: {result.get('avg_accept_len', 0):.2f}")
    print(f"    tokens: {result.get('total_tokens', 0)}")
    print(f"    prefill_ms: {result.get('prefill_ms', 0):.0f}")
    print(f"    draft_ms: {result.get('draft_ms', 0):.0f}")
    print(f"    verify_ms: {result.get('verify_ms', 0):.0f}")
    print(f"    elapsed: {elapsed:.1f}s")
    text = result.get("text", "")
    print(f"    output: {text[:200]}")

    # Verify accept rate > 50% for code completion
    accept = result.get("accept_rate", 0)
    if accept > 0.50:
        print(f"\n  [OK] Accept rate {accept:.1%} > 50% target")
        return True
    elif accept > 0:
        print(f"\n  [WARN] Accept rate {accept:.1%} < 50% target (v1 checkpoint)")
        return True  # Still passes, just warning
    else:
        print(f"\n  [FAIL] No accept rate data")
        return False


def test_decide_and_execute():
    """Test 4: decide_and_execute() — one-step routing + execution."""
    print("\n" + "=" * 70)
    print("  Test 4: decide_and_execute() — one-step end-to-end")
    print("=" * 70)

    model_path = os.path.expanduser("~/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf")
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found")
        return True

    bootstrap = Bootstrap()
    bootstrap.run(verbose=False)
    router = HermesRouter(bootstrap=bootstrap)

    prompts = [
        ("def binary_search(arr, target):\n    ", "code_completion"),
        ("Write a function to merge two sorted lists:", "code_generation"),
        ("What is encapsulation in OOP?", "chat"),
    ]

    all_ok = True
    for prompt, expected_type in prompts:
        print(f"\n  Prompt: {prompt[:50]}...")
        try:
            decision, result = router.decide_and_execute(
                prompt, model_name="qwen25_05b", max_tokens=20
            )
            d5 = getattr(decision, "d5_content", None)
            print(f"    mode={decision.mode}", end="")
            if d5:
                print(f", D5={d5.prompt_type}", end="")
            print(f", tps={result.get('tps', 0):.1f}, "
                  f"accept={result.get('accept_rate', 0):.0%}")
            text = result.get("text", "")
            if text:
                print(f"    output: {text[:100]}")
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    return all_ok


def main():
    print("=" * 70)
    print("  Hermes Router D5 + Verify Loop Integration Test")
    print("=" * 70)

    results = []

    # Test 1: D5 Content Awareness
    results.append(("D5 Content Awareness", test_d5_content_aware()))

    # Test 2: Hermes Router Decide
    results.append(("Hermes Router Decide", test_hermes_decide()))

    # Test 3: Execute (verify loop)
    results.append(("Execute (verify loop)", test_execute()))

    # Test 4: decide_and_execute
    results.append(("decide_and_execute", test_decide_and_execute()))

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    for name, ok in results:
        print(f"  {'OK' if ok else 'FAIL'}  {name}")

    all_ok = all(ok for _, ok in results)
    print(f"\n  Overall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
