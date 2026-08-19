#include "cgc_backend.h"
#include <stdio.h>

cgc_error_t cgc_backend_init(CGCBackend* backend) {
    if (!backend || !backend->init) {
        return CGC_ERROR;
    }
    printf("[CGC Backend] Initializing %s backend\n", backend->name);
    return backend->init();
}

cgc_error_t cgc_backend_execute(CGCBackend* backend, int opcode, 
                            const float** inputs, float** outputs,
                            const int* params, int num_inputs, int num_outputs) {
    if (!backend || !backend->execute) {
        return CGC_ERROR;
    }
    return backend->execute(opcode, inputs, outputs, params, num_inputs, num_outputs);
}

cgc_error_t cgc_backend_destroy(CGCBackend* backend) {
    if (!backend || !backend->destroy) {
        return CGC_ERROR;
    }
    printf("[CGC Backend] Destroying %s backend\n", backend->name);
    return backend->destroy();
}