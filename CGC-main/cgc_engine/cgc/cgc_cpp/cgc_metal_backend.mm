#include "cgc_backend.h"
#include <stdio.h>
#include <string.h>

#ifdef CGC_METAL_ENABLED
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_command_queue = nil;

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

        printf("[CGC Metal Backend] Initialized: %s\n",
               [[g_metal_device name] UTF8String]);
    }
    return CGC_OK;
}

static cgc_error_t metal_destroy(void) {
    @autoreleasepool {
        g_command_queue = nil;
        g_metal_device = nil;
    }
    printf("[CGC Metal Backend] Destroyed\n");
    return CGC_OK;
}

static cgc_error_t metal_execute(int opcode, const float** inputs, float** outputs,
                             const int* params, int num_inputs, int num_outputs) {
    switch (opcode) {
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