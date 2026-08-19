#ifndef CGC_BACKEND_H
#define CGC_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cgc_platform.h"
#include "cgc_cpp.h"

typedef struct CGCBackend CGCBackend;

struct CGCBackend {
    const char* name;
    CGCPlatform platform;
    
    cgc_error_t (*init)(void);
    cgc_error_t (*execute)(int opcode, const float** inputs, float** outputs, 
                       const int* params, int num_inputs, int num_outputs);
    cgc_error_t (*destroy)(void);
};

cgc_error_t cgc_backend_init(CGCBackend* backend);

cgc_error_t cgc_backend_execute(CGCBackend* backend, int opcode, 
                            const float** inputs, float** outputs,
                            const int* params, int num_inputs, int num_outputs);

cgc_error_t cgc_backend_destroy(CGCBackend* backend);

extern CGCBackend cgc_cpu_backend;
#ifdef CGC_CUDA_ENABLED
extern CGCBackend cgc_cuda_backend;
#endif
#ifdef CGC_METAL_ENABLED
extern CGCBackend cgc_metal_backend;
#endif

#ifdef __cplusplus
}
#endif

#endif // CGC_BACKEND_H