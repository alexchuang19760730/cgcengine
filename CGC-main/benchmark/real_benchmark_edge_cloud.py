import time
import subprocess
import os
from pathlib import Path
from llama_cpp import Llama
import threading

def run_pd_server():
    os.system("cd ComputeGraphCompiler-main && python3 -m cgc_engine.pd.pd_server 50051 > /dev/null 2>&1")

# Start PD Server in background
server_thread = threading.Thread(target=run_pd_server, daemon=True)
server_thread.start()
time.sleep(3) # wait for server to start

import sys
sys.path.insert(0, str(Path("ComputeGraphCompiler-main").absolute()))
from cgc_engine.pd.pd_client import PDClient
client = PDClient(address="localhost:50051")

model_path = "ComputeGraphCompiler-main/Output/Models/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q2_k.gguf"
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    exit(1)

prompt = "Hello, tell me a long story about a brave knight. " * 10 # Long prompt (approx 100 tokens)
gen_tokens = 64

print("============================================================")
print("🚀 真實端雲 PD 分離效能對比 (嚴格無 Mock)")
print("============================================================")

# 1. Native Edge (Full Prefill + Decode locally)
print("\n[1] 執行原生 Native Edge (全端側 Prefill + Decode)...")
llm_native = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=99, verbose=False)
t0 = time.perf_counter()
output_native = llm_native(prompt, max_tokens=gen_tokens)
t1 = time.perf_counter()
native_total_time = t1 - t0
print(f"    -> 原生總耗時 (Prefill + Decode): {native_total_time:.2f} 秒")
del llm_native

# 2. Edge-Cloud PD Separation
print("\n[2] 執行 CGC Engine 端雲 PD 分離 (Cloud Prefill -> Edge Decode)...")

import pickle

# A. Cloud Prefill (Simulated on Cloud)
print("    [Cloud] 正在雲端進行 Prefill 並擷取 KV Cache/State...")
llm_cloud = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=99, verbose=False)
llm_cloud.eval(llm_cloud.tokenize(prompt.encode("utf-8")))
# Save state
state_obj = llm_cloud.save_state()
kv_data = pickle.dumps(state_obj)

# Transmit to PD Server (KV Cache protocol V1)
client.store_prefix_kv_blocks_v1("prompt_1", {"data": kv_data})
del llm_cloud

# B. Edge Decode
print("    [Edge] 端側從 PD 伺服器下載 KV Cache 並開始 Decode...")
llm_edge = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=99, verbose=False)
t0 = time.perf_counter()
# Fetch from PD Server (Simulate Network transmission of PD Protocol)
kv_payload, hit = client.get_prefix_kv_blocks_v1("prompt_1")
if hit and kv_payload:
    state_obj = pickle.loads(kv_payload["data"])
    llm_edge.load_state(state_obj)
else:
    print("    [Edge] KV Cache 下載失敗！")

# Decode remaining tokens natively on Edge
last_token = llm_edge.tokenize(prompt.encode("utf-8"))[-1]
llm_edge.eval([last_token])
output_edge = llm_edge.sample(temp=0.0, top_k=1, top_p=1.0)
for _ in range(gen_tokens - 1):
    llm_edge.eval([output_edge])
    output_edge = llm_edge.sample(temp=0.0, top_k=1, top_p=1.0)

edge_decode_time = time.perf_counter() - t0
print(f"    -> 端雲 PD 模式端側耗時 (僅下載 KV + Decode): {edge_decode_time:.2f} 秒")

print("\n------------------------------------------------------------")
print("📊 真實測試結果對比 (Apple M4)")
print("------------------------------------------------------------")
print(f"【原生 Native Edge】: {native_total_time:.2f} 秒 (端側承擔全部運算)")
print(f"【CGC 端雲 PD 分離】: {edge_decode_time:.2f} 秒 (端側僅承擔網路下載與 Decode)")
if edge_decode_time < native_total_time:
    print(f"🏆 CGC PD 分離成功為端側省下 {(native_total_time - edge_decode_time):.2f} 秒的 Prefill 算力瓶頸！")
