"""用真实 30B 4bit MLX 模型测 forward 速度.

下载完成后运行:
  python bench_30b_real.py
"""
import time
import mlx.core as mx
from mlx_lm import load, stream_generate


def benchmark_30b():
    print("Loading Qwen3-VL-30B-A3B-Instruct-4bit...")
    t0 = time.time()
    model, tokenizer = load("mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit")
    print(f"Loaded in {time.time()-t0:.1f}s")

    # 获取层数
    layers = getattr(getattr(model, "model", model), "layers", None)
    num_layers = len(layers) if layers else "?"
    print(f"Model: {type(model).__name__}, layers: {num_layers}")

    # 测试 prompt
    messages = [{"role": "user", "content": "Write a short story about a cat"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Prompt: {len(input_ids)} tokens")

    # === 1. Prefill 时间 (全部层) ===
    print("\n=== Prefill (全部层) ===")
    # warmup
    list(stream_generate(model, tokenizer, prompt, max_tokens=1))

    t0 = time.time()
    list(stream_generate(model, tokenizer, prompt, max_tokens=1))
    prefill_ms = (time.time() - t0) * 1000
    print(f"Prefill ({len(input_ids)} tokens): {prefill_ms:.0f}ms")

    # === 2. Decode 速度 (全部 48 层, 无投机) ===
    print("\n=== Decode (全部层, 无投机) ===")
    t0 = time.time()
    tokens = []
    first_token_time = None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=30):
        tokens.append(resp.token)
        if len(tokens) == 1:
            first_token_time = time.time()
    total_time = time.time() - t0
    decode_time = time.time() - first_token_time
    n_decode = len(tokens) - 1

    ttft = (first_token_time - t0) * 1000
    decode_tps = n_decode / decode_time if decode_time > 0 else 0
    print(f"TTFT: {ttft:.0f}ms")
    print(f"Decode: {n_decode} tokens in {decode_time:.2f}s = {decode_tps:.1f} tok/s")
    print(f"Total: {len(tokens)} tokens in {total_time:.2f}s")

    # === 3. 每层 forward 时间 (从 decode 速度推算) ===
    print("\n=== Per-layer forward (从 decode 推算) ===")
    if isinstance(num_layers, int) and num_layers > 0:
        per_token_ms = 1000 / decode_tps if decode_tps > 0 else 0
        per_layer_ms = per_token_ms / num_layers
        print(f"Per token: {per_token_ms:.1f}ms")
        print(f"Per layer: {per_layer_ms:.2f}ms ({num_layers} layers)")

    # === 4. Layer-split 分析 ===
    print("\n=== Layer-split 分析 (N=21, accept=0.28, RTT=110ms) ===")
    if isinstance(num_layers, int) and num_layers > 0 and decode_tps > 0:
        per_layer = 1000 / decode_tps / num_layers
        rtt = 110
        cloud_per_layer = 0.1
        verify = 3
        accept = 21 * 0.28

        print(f"{'P':>4} {'Mac_fw':>8} {'Cloud':>6} {'Total':>8} {'tok/s':>6} {'省cloud':>8}")
        print("-" * 50)
        for P in [3, 6, 12, 18, 24]:
            mac_fw = P * per_layer
            cloud_fw = (num_layers - P) * cloud_per_layer
            batch = mac_fw + rtt + cloud_fw + verify + rtt
            tok_s = accept / batch * 1000
            save = P / num_layers * 100
            print(f"{P:4d} {mac_fw:7.1f}ms {cloud_fw:5.1f}ms {batch:7.1f}ms {tok_s:5.1f} {save:6.0f}%")

    # === 5. 显存使用 ===
    print("\n=== 显存 ===")
    try:
        import subprocess
        result = subprocess.run(
            ["memory_pressure"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "free" in line.lower() or "used" in line.lower():
                print(f"  {line.strip()}")
    except:
        pass

    print(f"\n{'='*50}")
    print("对比全云 30B (直连): TTFT 122ms, decode 112 tok/s")
    print(f"{'='*50}")


if __name__ == "__main__":
    benchmark_30b()
