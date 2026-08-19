#!/usr/bin/env python3
"""
云端服务器真实性能测试脚本
"""

import subprocess
import time

def run_cloud_test():
    print("=" * 60)
    print("☁️ 云端服务器真实性能测试")
    print("=" * 60)
    
    # 检查GPU信息
    print("\n💻 GPU信息:")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    print(result.stdout.strip())
    
    # 检查系统信息
    print("\n💻 系统信息:")
    result = subprocess.run(["nproc"], capture_output=True, text=True)
    print(f"CPU核心数: {result.stdout.strip()}")
    
    result = subprocess.run(["free", "-h"], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    print(f"内存: {lines[1]}")
    
    # 使用PyTorch进行真实GPU测试
    print("\n⚡ PyTorch GPU测试:")
    try:
        import torch
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            # 矩阵乘法测试
            N = 2048
            A = torch.randn(N, N, device='cuda', dtype=torch.float16)
            B = torch.randn(N, N, device='cuda', dtype=torch.float16)
            
            # 预热
            for _ in range(3):
                C = A @ B
            torch.cuda.synchronize()
            
            # 真实测试
            start = time.time()
            for _ in range(24):  # 模拟24层Transformer
                C = A @ B
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # 计算性能
            flops = 2 * N**3 * 24 / 1e12  # TFLOPs
            tfps = flops / elapsed
            
            print(f"\n📊 Prefill性能测试:")
            print(f"  矩阵大小: {N}x{N}")
            print(f"  计算量: {flops:.1f} TFLOPs")
            print(f"  耗时: {elapsed*1000:.2f} ms")
            print(f"  有效算力: {tfps:.2f} TFLOPS")
            print(f"  Prefill吞吐量: {N/elapsed:.1f} tokens/s")
            
        else:
            print("❌ CUDA不可用")
            
    except ImportError:
        print("❌ PyTorch未安装")
        print("尝试安装PyTorch...")
        subprocess.run(["pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121"])

if __name__ == "__main__":
    run_cloud_test()