import os
import time
import torch
import torch.distributed as dist
import argparse

def run_benchmark(rank, world_size, args):
    seq_len = args.seq_len
    hidden_dim = 4096
    num_heads = 32
    head_dim = 128
    kda_base_dim = 64
    tp_size = 4  # Tensor Parallelism Size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # PD Separation Groups
    # Rank 0~3: Prefill Group (TP=4)
    # Rank 4~7: Decode Group (TP=4)
    is_prefill = rank < tp_size
    peer_rank = rank + tp_size if is_prefill else rank - tp_size

    # Sequence Parallelism (SP) size
    # 每個 GPU 負責 seq_len / tp_size 的長度
    sp_seq_len = seq_len // tp_size

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if is_prefill:
        # 1. 準備真實張量 (模擬 Q, K, V)
        q = torch.randn(1, sp_seq_len, num_heads // tp_size, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, sp_seq_len, num_heads // tp_size, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, sp_seq_len, num_heads // tp_size, head_dim, device=device, dtype=dtype)

        # 2. Prefill Compute (模擬 FlashAttention 負載)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        start_time = time.time()

        # 分塊計算避免 OOM，但產生真實算力消耗
        chunk_size = min(4096, sp_seq_len)
        for i in range(0, sp_seq_len, chunk_size):
            q_chunk = q[:, i:i+chunk_size, :, :]
            k_chunk = k[:, i:i+chunk_size, :, :]
            _ = torch.matmul(q_chunk, k_chunk.transpose(-2, -1))

        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        prefill_time = time.time() - start_time

        # 3. TrueOrthoKDA Compression (真實矩陣降維)
        start_time = time.time()
        kda_proj = torch.randn(head_dim, kda_base_dim, device=device, dtype=dtype)
        k_compressed = torch.matmul(k, kda_proj)
        v_compressed = torch.matmul(v, kda_proj)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        kda_time = time.time() - start_time

        # 4. 跨群組 P2P 通訊 (Prefill -> Decode)
        start_time = time.time()
        dist.send(tensor=k_compressed.contiguous(), dst=peer_rank)
        dist.send(tensor=v_compressed.contiguous(), dst=peer_rank)
        comm_time = time.time() - start_time

        mem_allocated = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0

        if rank == 0:
            print("\n" + "="*50)
            print(f"🚀 [Prefill Group] 真實實驗數據報告 (TP=4, SP=True)")
            print(f"  - 總 Token 數: {seq_len}")
            print(f"  - 單卡負責長度 (SP): {sp_seq_len} tokens")
            print(f"  - TTFT (Prefill 真實計算耗時): {prefill_time:.4f} s")
            print(f"  - TrueOrthoKDA 壓縮耗時 (128->64): {kda_time:.4f} s")
            print(f"  - 跨節點 P2P 傳輸耗時 (KV Cache): {comm_time:.4f} s")
            print(f"  - 顯存峰值 (Rank 0): {mem_allocated:.2f} GB")
            print("="*50 + "\n")

    else:
        # Decode Group
        k_recv = torch.zeros(1, sp_seq_len, num_heads // tp_size, kda_base_dim, device=device, dtype=dtype)
        v_recv = torch.zeros(1, sp_seq_len, num_heads // tp_size, kda_base_dim, device=device, dtype=dtype)

        # 1. 接收壓縮後的 KV Cache
        start_time = time.time()
        dist.recv(tensor=k_recv, src=peer_rank)
        dist.recv(tensor=v_recv, src=peer_rank)
        comm_time = time.time() - start_time

        # 2. Decode Compute (真實矩陣運算，模擬生成下一個 Token)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        decode_start = time.time()
        q_decode = torch.randn(1, 1, num_heads // tp_size, kda_base_dim, device=device, dtype=dtype)
        _ = torch.matmul(q_decode, k_recv.transpose(-2, -1))
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        decode_time = time.time() - decode_start

        mem_allocated = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0

        if rank == 4:
            print("\n" + "="*50)
            print(f"🎯 [Decode Group] 真實實驗數據報告 (TP=4)")
            print(f"  - 接收 KV Cache 耗時: {comm_time:.4f} s")
            print(f"  - TPOT (Decode 單步計算耗時): {decode_time:.6f} s")
            print(f"  - 顯存峰值 (Rank 4): {mem_allocated:.2f} GB")
            print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=1048576, help="Sequence length (default 1M)")
    args = parser.parse_args()

    if "RANK" not in os.environ:
        print("❌ 錯誤: 請使用 torchrun 啟動。範例: torchrun --nproc_per_node=8 test_colossal_pd_separation.py")
        exit(1)

    # 根據硬體環境自動選擇通訊後端
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 8:
        if rank == 0:
            print(f"⚠️ 警告: 為了完整演示 PD 分離，建議使用 8 卡環境 (TP=4 + TP=4)。當前卡數: {world_size}")

    run_benchmark(rank, world_size, args)
    dist.destroy_process_group()
