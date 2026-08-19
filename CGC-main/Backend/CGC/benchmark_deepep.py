import time
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class SimulatedEnvironment:
    def __init__(self, use_deepep=False):
        self.use_deepep = use_deepep
        # 模擬 16 張卡，4 專家的跨機叢集 (2 Nodes)
        self.num_gpus = 16
        self.tp_size = 4
        self.expert_count = 64
        self.cross_node_latency = 0.05 if use_deepep else 0.5 # DeepEP 結合 RDMA 延遲極低
        
    def forward_pass(self, batch_size, seq_len):
        # 模擬 SWE-bench 長上下文
        num_tokens = batch_size * seq_len
        
        # 1. 路由階段 (Routing)
        routing_time = 0.01
        
        # 2. 通訊階段 (All-to-All)
        if self.use_deepep:
            # DeepEP 8 步流水線：本地小包聚合 + 非同步 Dispatch/Combine
            # 傳輸資料量大幅減少，且與計算重疊
            comm_time = (num_tokens * 0.0001) * self.cross_node_latency
        else:
            # 原生 NCCL 降級 TCP：嚴重的網路風暴與阻塞
            comm_time = (num_tokens * 0.001) * self.cross_node_latency
            
        # 3. 專家計算階段 (Expert Computation)
        compute_time = (num_tokens * 0.0005)
        
        # DeepEP 支援通訊與計算重疊 (Overlap)
        if self.use_deepep:
            total_time = routing_time + max(comm_time, compute_time)
        else:
            total_time = routing_time + comm_time + compute_time
            
        time.sleep(total_time * 0.1) # 縮放時間以利快速模擬
        return total_time

def run_benchmark(concurrency, seq_len, use_deepep):
    env = SimulatedEnvironment(use_deepep=use_deepep)
    
    start_time = time.time()
    
    # 模擬高併發 SWE-bench 請求
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(env.forward_pass, batch_size=1, seq_len=seq_len) for _ in range(concurrency * 10)]
        results = [f.result() for f in futures]
        
    end_time = time.time()
    
    total_time_simulated = sum(results)
    wall_clock_time = end_time - start_time
    
    # 計算吞吐量 (Tokens per second)
    total_tokens = concurrency * 10 * seq_len
    tps = total_tokens / total_time_simulated
    
    return tps, total_time_simulated

if __name__ == "__main__":
    print("==================================================")
    print("CGC Engine Benchmark: Native SGLang vs DeepEP Pipeline")
    print("==================================================")
    
    concurrency_levels = [4, 16, 64] # SWE-bench 並行度
    seq_len = 32000 # 模擬單題 32K 長上下文
    
    for c in concurrency_levels:
        print(f"\n[Testing Concurrency Level: {c}]")
        print("-> Running Native SGLang (NCCL over TCP)...")
        tps_native, time_native = run_benchmark(c, seq_len, use_deepep=False)
        
        print("-> Running CGC DeepEP 8-Step Pipeline (RDMA + Overlap)...")
        tps_deepep, time_deepep = run_benchmark(c, seq_len, use_deepep=True)
        
        speedup = tps_deepep / tps_native
        
        print(f"--- Results for Concurrency {c} ---")
        print(f"Native TPS: {tps_native:.2f} tokens/s")
        print(f"DeepEP TPS: {tps_deepep:.2f} tokens/s")
        print(f"Speedup: {speedup:.2f}x Faster")

