#!/usr/bin/env python3
import os
os.environ['VLLM_USE_CGC_KDA'] = '1'
import sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import Backend.Vllm.vllm_backend.cgc_kda_backend
from vllm import LLM, SamplingParams
print('Starting test...')
llm = LLM(
    model='/home/gs01/models/Qwen/Qwen2___5-7B-Instruct',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_model_len=4096,
    enforce_eager=True,
    disable_log_stats=True
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
outputs = llm.generate(['Hello, how are you?'], sampling_params)
print(f'Result: {outputs[0].outputs[0].text}')
del llm
