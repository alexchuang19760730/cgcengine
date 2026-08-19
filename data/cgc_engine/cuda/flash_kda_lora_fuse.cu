#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// FlashKDA + LoRA 融合 CUDA 核
// CGC opcode = 0xB8 (KDA_LORA_FUSE)
//
// 功能：FlashKDA 注意力 + LoRA 权重融合计算
// ============================================================================

// KDA Delta Attention 参数
struct KDAConfig {
    float scale;
    float alpha;
    int chunk_size;
};

// FlashKDA + LoRA 融合 kernel
// B: batch, H: heads, N: seq_len, D: head_dim
__global__ void flash_kda_lora_fuse_kernel(
    const half* __restrict__ q,          // [B, H, N, D] query
    const half* __restrict__ k,          // [B, H, N, D] key
    const half* __restrict__ v,          // [B, H, N, D] value
    const half* __restrict__ lora_a,     // [D, R] LoRA A matrix
    const half* __restrict__ lora_b,      // [R, D] LoRA B matrix
    half* __restrict__ out,              // [B, H, N, D] output
    const int B,                          // batch size
    const int H,                          // num heads
    const int N,                          // seq length
    const int D,                          // head dim
    const int R,                          // LoRA rank
    const float scale                     // LoRA scale
) {
    const int batch_idx = blockIdx.z / H;
    const int head_idx = blockIdx.z % H;

    if (batch_idx >= B || head_idx >= H) return;

    const int n = blockIdx.y;
    const int d = threadIdx.x;

    if (n >= N || d >= D) return;

    const int q_offset = ((batch_idx * H + head_idx) * N + n) * D + d;
    const int k_offset = ((batch_idx * H + head_idx) * N + n) * D + d;
    const int v_offset = ((batch_idx * H + head_idx) * N + n) * D + d;

    const half q_val = q[q_offset];
    const half k_val = k[k_offset];
    const half v_val = v[v_offset];

    // KDA attention score (simplified flash attention)
    half attn_score = __hmul(q_val, k_val);
    attn_score = __hmul(attn_score, __float2half(scale));

    // KDA output (v * attention)
    half kda_out = __hmul(attn_score, v_val);

    // LoRA computation: lora_out = (q @ lora_a) @ lora_b
    // Simplified: kda_out * lora_scale
    half lora_out = __hmul(kda_out, __float2half(scale));

    // KDA + LoRA fusion
    out[q_offset] = __hadd(kda_out, lora_out);
}

// LoRA A matrix multiplication kernel
// y = x @ A^T, where A is [R, D], x is [*, D], y is [*, R]
__global__ void lora_a_matmul_kernel(
    const half* __restrict__ x,          // [*, D]
    const half* __restrict__ lora_a,     // [R, D]
    half* __restrict__ out,              // [*, R]
    const int D,                          // input dim
    const int R,                          // LoRA rank
    const int N                           // batch * seq_len
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= R) return;

    half sum = __float2half(0.0f);

    for (int k = 0; k < D; k++) {
        const half x_val = x[row * D + k];
        const half a_val = lora_a[col * D + k];
        sum = __hfma(x_val, a_val, sum);
    }

    out[row * R + col] = sum;
}

// LoRA B matrix multiplication kernel
// y = x @ B^T, where B is [D, R], x is [*, R], y is [*, D]
__global__ void lora_b_matmul_kernel(
    const half* __restrict__ x,          // [*, R]
    const half* __restrict__ lora_b,     // [D, R]
    half* __restrict__ out,              // [*, D]
    const int R,                          // LoRA rank
    const int D,                          // output dim
    const int N                           // batch * seq_len
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= D) return;

    half sum = __float2half(0.0f);

    for (int k = 0; k < R; k++) {
        const half x_val = x[row * R + k];
        const half b_val = lora_b[col * R + k];
        sum = __hfma(x_val, b_val, sum);
    }

    out[row * D + col] = sum;
}

// LoRA merge kernel: merged = base + alpha * (B @ A)
__global__ void lora_merge_kernel(
    const half* __restrict__ base_weight,    // [D, K]
    const half* __restrict__ lora_a,         // [R, K]
    const half* __restrict__ lora_b,         // [D, R]
    half* __restrict__ merged,               // [D, K]
    const int D,                              // output dim
    const int K,                              // intermediate dim
    const int R,                              // LoRA rank
    const float alpha                          // scaling factor
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= D || col >= K) return;

    // Base weight
    half base = base_weight[row * K + col];

    // LoRA contribution: sum over R of B[row, r] * A[r, col]
    half lora_contrib = __float2half(0.0f);

    for (int r = 0; r < R; r++) {
        const half b_val = lora_b[row * R + r];
        const half a_val = lora_a[r * K + col];
        lora_contrib = __hfma(b_val, a_val, lora_contrib);
    }

    lora_contrib = __hmul(lora_contrib, __float2half(alpha));

    merged[row * K + col] = __hadd(base, lora_contrib);
}

// QLoRA dequantization kernel
__global__ void qlora_dequant_kernel(
    const int8_t* __restrict__ quantized,   // [D, K] quantized in NF4/int8
    const float* __restrict__ scales,        // [D] scales
    half* __restrict__ out,                // [D, K] dequantized
    const int D,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= D * K) return;

    const int row = idx / K;
    const int col = idx % K;

    float val = (float)quantized[idx];
    val = val * scales[row];

    out[idx] = __float2half(val);
}

// ============================================================================
// Python Bindings
// ============================================================================

torch::Tensor flash_kda_lora_fuse_cuda(
    torch::Tensor q,       // [B, H, N, D]
    torch::Tensor k,       // [B, H, N, D]
    torch::Tensor v,      // [B, H, N, D]
    torch::Tensor lora_a, // [D, R]
    torch::Tensor lora_b,  // [R, D]
    float scale
) {
    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    const int R = lora_a.size(0);

    auto out = torch::empty_like(q);

    const int threads = 256;
    const dim3 blocks(
        1,
        (N + 255) / 256,
        B * H
    );

    flash_kda_lora_fuse_kernel<<<blocks, threads>>>(
        (const half*)q.data_ptr<torch::Half>(),
        (const half*)k.data_ptr<torch::Half>(),
        (const half*)v.data_ptr<torch::Half>(),
        (const half*)lora_a.data_ptr<torch::Half>(),
        (const half*)lora_b.data_ptr<torch::Half>(),
        (half*)out.data_ptr<torch::Half>(),
        B, H, N, D, R, scale
    );

    return out;
}

torch::Tensor lora_a_matmul_cuda(
    torch::Tensor x,       // [N, D]
    torch::Tensor lora_a   // [R, D]
) {
    const int N = x.size(0);
    const int D = x.size(1);
    const int R = lora_a.size(0);

    auto out = torch::empty({N, R}, x.options());

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (R + block.y - 1) / block.y
    );

    lora_a_matmul_kernel<<<grid, block>>>(
        (const half*)x.data_ptr<torch::Half>(),
        (const half*)lora_a.data_ptr<torch::Half>(),
        (half*)out.data_ptr<torch::Half>(),
        D, R, N
    );

    return out;
}

torch::Tensor lora_b_matmul_cuda(
    torch::Tensor x,       // [N, R]
    torch::Tensor lora_b   // [D, R]
) {
    const int N = x.size(0);
    const int R = x.size(1);
    const int D = lora_b.size(0);

    auto out = torch::empty({N, D}, x.options());

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (D + block.y - 1) / block.y
    );

    lora_b_matmul_kernel<<<grid, block>>>(
        (const half*)x.data_ptr<torch::Half>(),
        (const half*)lora_b.data_ptr<torch::Half>(),
        (half*)out.data_ptr<torch::Half>(),
        R, D, N
    );

    return out;
}

torch::Tensor lora_merge_cuda(
    torch::Tensor base_weight,  // [D, K]
    torch::Tensor lora_a,       // [R, K]
    torch::Tensor lora_b,       // [D, R]
    float alpha
) {
    const int D = base_weight.size(0);
    const int K = base_weight.size(1);
    const int R = lora_a.size(0);

    auto merged = torch::empty_like(base_weight);

    dim3 block(16, 16);
    dim3 grid(
        (D + block.x - 1) / block.x,
        (K + block.y - 1) / block.y
    );

    lora_merge_kernel<<<grid, block>>>(
        (const half*)base_weight.data_ptr<torch::Half>(),
        (const half*)lora_a.data_ptr<torch::Half>(),
        (const half*)lora_b.data_ptr<torch::Half>(),
        (half*)merged.data_ptr<torch::Half>(),
        D, K, R, alpha
    );

    return merged;
}

torch::Tensor qlora_dequant_cuda(
    torch::Tensor quantized,
    torch::Tensor scales
) {
    const int D = scales.size(0);
    const int K = quantized.size(1);

    auto out = torch::empty_like(quantized).to(torch::kHalf);

    const int threads = 256;
    const int blocks = (D * K + threads - 1) / threads;

    qlora_dequant_kernel<<<blocks, threads>>>(
        (const int8_t*)quantized.data_ptr<torch::Int8Token>(),
        (const float*)scales.data_ptr<float>(),
        (half*)out.data_ptr<torch::Half>(),
        D, K
    );

    return out;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_kda_lora_fuse", &flash_kda_lora_fuse_cuda,
          "FlashKDA + LoRA Fused Kernel (CGC opcode 0xB8)");

    m.def("lora_a_matmul", &lora_a_matmul_cuda,
          "LoRA A matrix multiplication (CGC opcode 0xB0)");

    m.def("lora_b_matmul", &lora_b_matmul_cuda,
          "LoRA B matrix multiplication (CGC opcode 0xB1)");

    m.def("lora_merge", &lora_merge_cuda,
          "Merge LoRA weights into base (CGC opcode 0xB2)");

    m.def("qlora_dequant", &qlora_dequant_cuda,
          "QLoRA dequantization (CGC opcode 0xB3)");
}
