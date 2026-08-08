import re
import os

cpp_path = "/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/cgc/cgc_cpp/src/kernels/ortho_kda_v4.cpp"

with open(cpp_path, "r") as f:
    content = f.read()

# Replace the #else block with full implementation
old_else_idx = content.find('#else\n')
if old_else_idx == -1:
    print("Could not find #else block")
    exit(1)

new_else_block = """#else

#include <cmath>

struct OrthoKDAKV_v4;
typedef int cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaGetLastError() cudaSuccess

void ortho_kda_v4_gram_schmidt_cpu(
    float* v,
    const float (*basis)[ORTHO_KDA_V4_HEAD_DIM],
    int n,
    int head_dim
) {
    for (int i = 0; i < n; i++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += v[d] * basis[i][d];
        }
        for (int d = 0; d < head_dim; d++) {
            v[d] -= dot * basis[i][d];
        }
    }

    float norm = ORTHO_KDA_V4_EPS;
    for (int d = 0; d < head_dim; d++) {
        norm += v[d] * v[d];
    }
    norm = 1.0f / std::sqrt(norm);
    for (int d = 0; d < head_dim; d++) {
        v[d] *= norm;
    }
}

cudaError_t ortho_kda_v4_alloc_kv(
    OrthoKDAKV_v4** kv,
    int num_heads,
    int head_dim,
    int ortho_base_dim,
    cudaStream_t stream
) {
    *kv = new OrthoKDAKV_v4();
    (*kv)->num_heads = num_heads;
    (*kv)->head_dim = head_dim;
    (*kv)->ortho_base_dim = ortho_base_dim;
    (*kv)->idx = 0;

    for (int i = 0; i < ortho_base_dim; i++) {
        for (int d = 0; d < head_dim; d++) {
            (*kv)->K[i][d] = 0.0f;
            (*kv)->V[i][d] = 0.0f;
        }
        (*kv)->decay[i] = 0.0f;
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_free_kv(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
) {
    if (kv != nullptr) {
        delete kv;
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_reset(
    OrthoKDAKV_v4* kv,
    cudaStream_t stream
) {
    if (!kv) return cudaSuccess;
    kv->idx = 0;
    for (int i = 0; i < kv->ortho_base_dim; i++) {
        for (int d = 0; d < kv->head_dim; d++) {
            kv->K[i][d] = 0.0f;
            kv->V[i][d] = 0.0f;
        }
        kv->decay[i] = 0.0f;
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_update(
    OrthoKDAKV_v4* kv,
    const float* key,
    const float* value,
    cudaStream_t stream
) {
    if (!kv) return cudaSuccess;

    for (int i = 0; i < kv->ortho_base_dim; i++) {
        float k[ORTHO_KDA_V4_HEAD_DIM];
        for (int d = 0; d < kv->head_dim; d++) {
            k[d] = key[d];
        }

        ortho_kda_v4_gram_schmidt_cpu(k, kv->K, i, kv->head_dim);

        for (int d = 0; d < kv->head_dim; d++) {
            kv->K[i][d] += k[d];
            kv->V[i][d] += value[d];
        }
        kv->decay[i] = std::exp(-ORTHO_KDA_V4_DEFAULT_DECAY * static_cast<float>(i));
    }

    if (kv->idx < kv->ortho_base_dim - 1) {
        kv->idx++;
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_forward(
    OrthoKDAKV_v4* kv,
    const float* query,
    float* output,
    cudaStream_t stream
) {
    if (!kv) return cudaSuccess;

    int num_heads = kv->num_heads;
    int head_dim = kv->head_dim;

    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        float sum[ORTHO_KDA_V4_HEAD_DIM] = {0.0f};

        for (int i = 0; i < kv->idx; i++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int idx = head_idx * head_dim + d;
                score += query[idx] * kv->K[i][d];
            }

            float weighted_score = score * kv->decay[i];

            for (int d = 0; d < head_dim; d++) {
                sum[d] += weighted_score * kv->V[i][d];
            }
        }

        for (int d = 0; d < head_dim; d++) {
            int idx = head_idx * head_dim + d;
            output[idx] = sum[d];
        }
    }
    return cudaSuccess;
}

cudaError_t ortho_kda_v4_get_state(
    OrthoKDAKV_v4* kv,
    float* K_out,
    float* V_out,
    float* decay_out,
    int* idx_out,
    cudaStream_t stream
) {
    if (!kv) return cudaSuccess;

    int head_dim = kv->head_dim;
    for (int i = 0; i < kv->ortho_base_dim; i++) {
        for (int d = 0; d < head_dim; d++) {
            K_out[i * head_dim + d] = kv->K[i][d];
            V_out[i * head_dim + d] = kv->V[i][d];
        }
        decay_out[i] = kv->decay[i];
    }
    *idx_out = kv->idx;
    return cudaSuccess;
}

#endif
"""

new_content = content[:old_else_idx] + new_else_block

with open(cpp_path, "w") as f:
    f.write(new_content)

print("Updated ortho_kda_v4.cpp")

# Now update the binding cpp
binding_path = "/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/cgc/cgc_cpp/src/kernels/ortho_kda_v4_binding.cpp"
with open(binding_path, "r") as f:
    binding_content = f.read()

# Replace "#ifdef __CUDACC__\n        cudaError_t err" with just "cudaError_t err" (remove the block)
binding_content = re.sub(r'#ifdef __CUDACC__\n(.*?)#endif', r'\1', binding_content, flags=re.DOTALL)

# But wait, removing all #ifdef __CUDACC__ will also remove the one around `#include <cuda_runtime.h>` which will break CPU build!
# Let's restore the #include <cuda_runtime.h> properly
binding_content = binding_content.replace('namespace py = pybind11;\n\n#include <cuda_runtime.h>\n', 'namespace py = pybind11;\n\n#ifdef __CUDACC__\n#include <cuda_runtime.h>\n#endif\n')

# Also the benchmark cudaEvent_t needs to be protected
benchmark_cuda = """        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < 10; i++) {
            kda.update(key_arr, value_arr);
        }

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            kda.forward(Q_arr);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);"""

benchmark_cpu = """        float ms = 0.0f;
        for (int i = 0; i < 10; i++) {
            kda.update(key_arr, value_arr);
        }
        for (int i = 0; i < iterations; i++) {
            kda.forward(Q_arr);
        }"""

if benchmark_cuda in binding_content:
    binding_content = binding_content.replace(benchmark_cuda, f"#ifdef __CUDACC__\n{benchmark_cuda}\n#else\n{benchmark_cpu}\n#endif")

with open(binding_path, "w") as f:
    f.write(binding_content)

print("Updated ortho_kda_v4_binding.cpp")

# Now update CMakeLists.txt to include them
cmake_path = "/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/cgc_engine/cgc/cgc_cpp/CMakeLists.txt"
with open(cmake_path, "r") as f:
    cmake_content = f.read()

if "src/kernels/ortho_kda_v4.cpp" not in cmake_content:
    cmake_content = cmake_content.replace(
        "src/kernels/kda.cpp",
        "src/kernels/kda.cpp\n    src/kernels/ortho_kda_v4.cpp\n    src/kernels/ortho_kda_v4_binding.cpp"
    )
    with open(cmake_path, "w") as f:
        f.write(cmake_content)
    print("Updated CMakeLists.txt")
else:
    print("CMakeLists.txt already contains ortho_kda_v4.cpp")
