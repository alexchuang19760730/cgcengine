#include "cgc_backend.h"
#include <stdio.h>
#include <string.h>

#ifdef CGC_METAL_ENABLED
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Include metal_runtime library
#include "../../../../../magi_native_engine/metal_runtime.h"

static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static id<MTLLibrary> g_kda_library = nil;
static id<MTLFunction> g_kda_function = nil;
static id<MTLComputePipelineState> g_kda_pipeline = nil;

typedef struct {
    int batch;
    int n_heads;
    int seq_len;
    int head_dim;
    float beta;
} KDAParams;

static const char* kKDAShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct KDAParamsMetal {
    int batch;
    int n_heads;
    int seq_len;
    int head_dim;
    float beta;
};

kernel void kda_causal(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device const float* gamma [[buffer(4)]],
    device float* S [[buffer(5)]],
    constant KDAParamsMetal& params [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int b = gid.x;
    const int h = gid.y;
    const int t = gid.z;

    if (b >= params.batch || h >= params.n_heads || t >= params.seq_len) return;

    const int D = params.head_dim;
    const float beta = params.beta;
    const float scale = 1.0f / sqrt((float)D);

    const int s_offset = (b * params.n_heads + h) * D * D;
    const int q_offset = ((b * params.n_heads + h) * params.seq_len + t) * D;
    const int kv_base = (b * params.n_heads + h) * params.seq_len * D;

    float S_local[128][128];
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            S_local[i][j] = S[s_offset + i * D + j];
        }
    }

    for (int tt = 0; tt <= t; tt++) {
        const int kv_off = kv_base + tt * D;

        float k_vec[128];
        float v_vec[128];
        for (int d = 0; d < D; d++) {
            k_vec[d] = K[kv_off + d];
            v_vec[d] = V[kv_off + d];
        }

        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                float k_i = k_vec[i];
                float k_j = k_vec[j];
                float v_j = v_vec[j];
                float s_ij = S_local[i][j];
                S_local[i][j] = s_ij * (1.0f - beta * k_i * k_j) + beta * k_i * v_j;
            }
        }
    }

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            S[s_offset + i * D + j] = S_local[i][j];
        }
    }

    float q_vec[128];
    for (int d = 0; d < D; d++) {
        q_vec[d] = Q[q_offset + d];
    }

    for (int d = 0; d < D; d++) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++) {
            sum += q_vec[k] * S_local[k][d];
        }
        O[q_offset + d] = sum * scale;
    }
}
)METAL";

static cgc_error_t metal_init(void) {
    @autoreleasepool {
        g_metal_device = MTLCreateSystemDefaultDevice();
        if (!g_metal_device) {
            printf("[CGC Metal Backend] No Metal device found\n");
            return CGC_ERROR;
        }

        g_command_queue = [g_metal_device newCommandQueue];
        if (!g_command_queue) {
            printf("[CGC Metal Backend] Failed to create command queue\n");
            return CGC_ERROR;
        }

        NSError* error = nil;
        NSString* shaderSource = [NSString stringWithUTF8String:kKDAShaderSource];
        g_kda_library = [g_metal_device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!g_kda_library) {
            printf("[CGC Metal Backend] Failed to compile shader: %s\n",
                   [[error localizedDescription] UTF8String]);
            return CGC_ERROR;
        }

        g_kda_function = [g_kda_library newFunctionWithName:@"kda_causal"];
        if (!g_kda_function) {
            printf("[CGC Metal Backend] Failed to find kda_causal function\n");
            return CGC_ERROR;
        }

        g_kda_pipeline = [g_metal_device newComputePipelineStateWithFunction:g_kda_function error:&error];
        if (!g_kda_pipeline) {
            printf("[CGC Metal Backend] Failed to create pipeline: %s\n",
                   [[error localizedDescription] UTF8String]);
            return CGC_ERROR;
        }

        printf("[CGC Metal Backend] Initialized: %s (GPU Kernel loaded)\n",
               [[g_metal_device name] UTF8String]);
    }
    return CGC_OK;
}

static cgc_error_t metal_destroy(void) {
    @autoreleasepool {
        g_kda_pipeline = nil;
        g_kda_function = nil;
        g_kda_library = nil;
        g_command_queue = nil;
        g_metal_device = nil;
    }
    printf("[CGC Metal Backend] Destroyed\n");
    return CGC_OK;
}

static cgc_error_t metal_execute(int opcode, const float** inputs, float** outputs,
                             const int* params, int num_inputs, int num_outputs) {
    switch (opcode) {
        case 0x11: { // ATTENTION_KDA
            printf("[CGC Metal Backend] KDA (kimikda) removed\n");
            return CGC_ERROR;
        }
        case 0x20: { // LINEAR_GEMM - Metal GEMM via MPS
            int m = params[0];
            int n = params[1];
            int k = params[2];

            @autoreleasepool {
                id<MTLBuffer> bufferA = [g_metal_device newBufferWithBytes:inputs[0]
                                                                   length:m * k * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufferB = [g_metal_device newBufferWithBytes:inputs[1]
                                                                   length:k * n * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufferC = [g_metal_device newBufferWithLength:m * n * sizeof(float)
                                                                     options:MTLResourceStorageModeShared];

                // Use MPS matrix multiplication
                MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                                 columns:k
                                                                                rowBytes:k * sizeof(float)
                                                                               dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                                                 columns:n
                                                                                rowBytes:n * sizeof(float)
                                                                               dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                                 columns:n
                                                                                rowBytes:n * sizeof(float)
                                                                               dataType:MPSDataTypeFloat32];

                MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
                MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
                MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

                MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
                    initWithDevice:g_metal_device
                    transposeLeft:NO
                 transposeRight:NO
                          resultRows:m
                       resultColumns:n
                      interiorColumns:k];

                id<MTLCommandBuffer> cmdBuf = [g_command_queue commandBuffer];
                [mm encodeToCommandBuffer:cmdBuf
                       leftMatrix:matrixA
                      rightMatrix:matrixB
                      resultMatrix:matrixC];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];

                memcpy(outputs[0], bufferC.contents, m * n * sizeof(float));
            }
            break;
        }
        case 0x60: { // SOFTMAX - Simple memcpy for now
            int n = params[0];
            memcpy(outputs[0], inputs[0], n * sizeof(float));
            break;
        }
        default:
            printf("[CGC Metal Backend] Unsupported opcode: 0x%02X, falling back to CPU\n", opcode);
            return CGC_ERROR_NOT_SUPPORTED;
    }
    return CGC_OK;
}

CGCBackend cgc_metal_backend = {
    .name = "Metal",
    .platform = CGC_PLATFORM_METAL,
    .init = metal_init,
    .execute = metal_execute,
    .destroy = metal_destroy
};

#else

CGCBackend cgc_metal_backend = {
    .name = "Metal (disabled)",
    .platform = CGC_PLATFORM_METAL,
    .init = NULL,
    .execute = NULL,
    .destroy = NULL
};

#endif
