#include "cgc_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

static bool g_has_cuda = false;
static bool g_has_metal = false;
static CGCPlatform g_platform = CGC_PLATFORM_UNKNOWN;

#ifdef __APPLE__
#include <dlfcn.h>

static bool check_metal_availability(void) {
    void* metal_lib = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    if (metal_lib) {
        dlclose(metal_lib);
        return true;
    }
    return false;
}
#endif

#ifdef CGC_CUDA_ENABLED
#include <cuda_runtime.h>

static bool check_cuda_availability(void) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}
#else
static bool check_cuda_availability(void) {
    return false;
}
#endif

bool cgc_has_cuda(void) {
    return g_has_cuda;
}

bool cgc_has_metal(void) {
    return g_has_metal;
}

CGCPlatform cgc_detect_platform(void) {
    if (g_platform != CGC_PLATFORM_UNKNOWN) {
        return g_platform;
    }

    g_has_cuda = check_cuda_availability();
    g_has_metal = false;

#ifdef __APPLE__
    if (TARGET_OS_MAC && TARGET_CPU_ARM64) {
        g_has_metal = check_metal_availability();
    }
#endif

    if (g_has_cuda) {
        g_platform = CGC_PLATFORM_CUDA;
    } else if (g_has_metal) {
        g_platform = CGC_PLATFORM_METAL;
    } else {
        g_platform = CGC_PLATFORM_CPU;
    }

    return g_platform;
}

const char* cgc_platform_name(CGCPlatform platform) {
    switch (platform) {
        case CGC_PLATFORM_CPU:
            return "CPU";
        case CGC_PLATFORM_CUDA:
            return "CUDA";
        case CGC_PLATFORM_METAL:
            return "Metal";
        case CGC_PLATFORM_UNKNOWN:
            return "Unknown";
        default:
            return "Unknown";
    }
}