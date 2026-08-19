#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/gs01')

from cgc_engine import CGCEngine, CGCEngineConfig
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model_path = '/home/gs01/models/Qwen/Qwen2___5-7B-Instruct'

config = CGCEngineConfig(
    model_name_or_path=model_path,
    enable_vllm=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
)

print(f"Creating CGCEngine with vLLM: {model_path}")
engine = CGCEngine(config=config)
print(f"Engine created! Mode: {engine._get_mode()}")

print("\nTesting inference...")
result = engine.generate("Hello, world!", max_tokens=50)
print(f"Result: {result}")
print("\nvLLM integration successful!")

with open('/home/gs01/test_result.log', 'w') as f:
    f.write(f"SUCCESS\nMode: {engine._get_mode()}\nResult: {result}\n")
