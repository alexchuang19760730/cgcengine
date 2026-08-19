#!/usr/bin/env python3
"""
云端服务器真实Prefill测试脚本
"""

import subprocess
import time

def run_cloud_prefill_test():
    print("=" * 60)
    print("☁️ 云端服务器真实Prefill测试")
    print("=" * 60)
    
    # 检查GPU信息
    print("\n💻 GPU信息:")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        print(result.stdout.strip())
    except Exception as e:
        print(f"无法获取GPU信息: {e}")
    
    # 检查系统信息
    print("\n💻 系统信息:")
    result = subprocess.run(["nproc"], capture_output=True, text=True)
    print(f"CPU核心数: {result.stdout.strip()}")
    
    # 检查内存
    result = subprocess.run(["free", "-h"], capture_output=True, text=True)
    print(f"内存:\n{result.stdout.strip()}")
    
    # 真实Prefill性能测试（使用nvcc编译的测试程序）
    print("\n⚡ Prefill性能测试:")
    
    # 编译并运行CUDA测试
    cuda_code = '''
#include <stdio.h>
#include <cuda_runtime.h>

#define N 2048
#define LAYERS 24

__global__ void prefill_kernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[i * n + k] * B[k * n + j];
        }
        C[i * n + j] = sum;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int size = N * N * sizeof(float);
    
    // 分配内存
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // 分配GPU内存
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 复制数据到GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 配置线程
    dim3 block(32, 32);
    dim3 grid(N / block.x, N / block.y);
    
    // 预热
    for (int i = 0; i < 3; i++) {
        prefill_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    
    // 真实测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < LAYERS; i++) {
        prefill_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    // 计算TFLOPS
    double flops = (double)LAYERS * 2 * N * N * N / 1e12;
    double tfps = flops / (elapsed_ms / 1000.0);
    
    printf("Prefill计算量: %.1f TFLOPs\\n", flops);
    printf("Prefill时间: %.2f ms\\n", elapsed_ms);
    printf("有效算力: %.2f TFLOPS\\n", tfps);
    printf("Prefill吞吐量: %.1f tokens/s\\n", N / (elapsed_ms / 1000.0));
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
'''
    
    # 写入CUDA代码
    with open("/tmp/prefill_test.cu", "w") as f:
        f.write(cuda_code)
    
    # 编译
    print("\n⏳ 编译CUDA测试程序...")
    result = subprocess.run(
        ["nvcc", "/tmp/prefill_test.cu", "-o", "/tmp/prefill_test"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"❌ 编译失败: {result.stderr}")
        return
    
    # 运行测试
    print("\n⏳ 运行Prefill测试...")
    result = subprocess.run(["/tmp/prefill_test"], capture_output=True, text=True)
    print(result.stdout)
    
    if result.stderr:
        print(f"stderr: {result.stderr}")

if __name__ == "__main__":
    run_cloud_prefill_test()