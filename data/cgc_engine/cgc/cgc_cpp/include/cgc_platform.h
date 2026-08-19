#ifndef CGC_PLATFORM_H
#define CGC_PLATFORM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef enum {
    CGC_PLATFORM_CPU = 0,
    CGC_PLATFORM_CUDA = 1,
    CGC_PLATFORM_METAL = 2,
    CGC_PLATFORM_UNKNOWN = 3
} CGCPlatform;

CGCPlatform cgc_detect_platform(void);

const char* cgc_platform_name(CGCPlatform platform);

bool cgc_has_cuda(void);

bool cgc_has_metal(void);

#ifdef __cplusplus
}
#endif

#endif // CGC_PLATFORM_H