#include "cgc_backend.h"
#include "kernels/quant_full.h"
#include "kernels/activation.h"
#include "kernels/activation_full.h"
#include "kernels/norm.h"
#include "kernels/norm_full.h"
#include "kernels/rope.h"
#include "kernels/rope_full.h"
#include "kernels/sampling.h"
#include "kernels/gemm_cpu.h"
#include <stdio.h>
#include <math.h>

static cgc_error_t cpu_init(void) {
    printf("[CGC CPU Backend] Initialized\n");
    return CGC_OK;
}

static cgc_error_t cpu_destroy(void) {
    printf("[CGC CPU Backend] Destroyed\n");
    return CGC_OK;
}

static cgc_error_t cpu_execute(int opcode, const float** inputs, float** outputs,
                           const int* params, int num_inputs, int num_outputs) {
    switch (opcode) {
        case 0x20: { // LINEAR_GEMM
            int m = params[0];
            int n = params[1];
            int k = params[2];
            gemm_cpu(inputs[0], inputs[1], outputs[0], m, n, k);
            break;
        }
        case 0x21: { // LINEAR_BIAS
            int size = params[0];
            for (int i = 0; i < size; i++) {
                outputs[0][i] = inputs[0][i] + inputs[1][i];
            }
            break;
        }
        case 0x30: { // LAYER_NORM
            int n = params[0];
            float eps = params[1] / 1000000.0f;
            float* weight = (float*)(num_inputs > 1 ? inputs[1] : nullptr);
            float* bias = (float*)(num_inputs > 2 ? inputs[2] : nullptr);
            layer_norm(inputs[0], outputs[0], 1, n, n, weight, bias, eps);
            break;
        }
        case 0x31: { // RMS_NORM
            int n = params[0];
            float eps = params[1] / 1000000.0f;
            rms_norm(inputs[0], nullptr, outputs[0], eps, 1, n, n);
            break;
        }
        case 0x32: { // GROUP_NORM
            int n = params[0];
            int group_size = params[1];
            float eps = params[2] / 1000000.0f;
            float* weight = (float*)(num_inputs > 1 ? inputs[1] : nullptr);
            float* bias = (float*)(num_inputs > 2 ? inputs[2] : nullptr);
            int num_groups = n / group_size;
            group_norm(inputs[0], outputs[0], 1, n, 1, 1, num_groups, weight, bias, eps);
            break;
        }
        case 0x40: { // ROPE
            int n = params[0];
            int head_dim = params[1];
            int offset = params[2];
            float* x = const_cast<float*>(inputs[0]);
            rope_hf(x, 1, 1, n, head_dim, offset, nullptr);
            for (int i = 0; i < n * head_dim; i++) {
                outputs[0][i] = x[i];
            }
            break;
        }
        case 0x50: { // SILU
            int n = params[0];
            activation_silu(inputs[0], outputs[0], n);
            break;
        }
        case 0x51: { // GELU
            int n = params[0];
            activation_gelu(inputs[0], outputs[0], n);
            break;
        }
        case 0x53: { // RELU
            int n = params[0];
            activation_relu(inputs[0], outputs[0], n);
            break;
        }
        case 0x54: { // SIGMOID
            int n = params[0];
            activation_sigmoid(inputs[0], outputs[0], n);
            break;
        }
        case 0x60: { // SOFTMAX
            int n = params[0];
            softmax(inputs[0], outputs[0], 1, 1, n);
            break;
        }
        case 0x61: { // LOG_SOFTMAX
            int n = params[0];
            softmax(inputs[0], outputs[0], 1, 1, n);
            for (int i = 0; i < n; i++) {
                outputs[0][i] = logf(outputs[0][i] + 1e-10f);
            }
            break;
        }
        case 0x62: { // TOP_K
            int n = params[0];
            int k = params[1];
            sample_topk(inputs[0], outputs[0], nullptr, 1, n, k);
            break;
        }
        case 0x63: { // TOP_P
            int n = params[0];
            float p = params[1] / 1000000.0f;
            sample_topp(inputs[0], outputs[0], 1, n, p);
            break;
        }
        case 0x64: { // TEMPERATURE
            int n = params[0];
            float temp = params[1] / 1000000.0f;
            sample_temperature(inputs[0], outputs[0], 1, n, temp);
            break;
        }
        case 0xA0: { // QUANTIZE_W8A16
            int size = params[0];
            int group_size = params[1];
            quantize_q8(inputs[0], (uint8_t*)outputs[0], size, outputs[1], group_size);
            break;
        }
        case 0xA1: { // QUANTIZE_W4A16
            int size = params[0];
            int group_size = params[1];
            quantize_q4(inputs[0], (uint8_t*)outputs[0], size, outputs[1], group_size);
            break;
        }
        case 0xA2: { // DEQUANTIZE
            int size = params[0];
            int group_size = params[1];
            dequantize_q4((const uint8_t*)inputs[0], outputs[0], size, inputs[1], group_size);
            break;
        }
        default:
            printf("[CGC CPU Backend] Unsupported opcode: 0x%02X\n", opcode);
            return CGC_ERROR;
    }
    return CGC_OK;
}

CGCBackend cgc_cpu_backend = {
    .name = "CPU",
    .platform = CGC_PLATFORM_CPU,
    .init = cpu_init,
    .execute = cpu_execute,
    .destroy = cpu_destroy
};