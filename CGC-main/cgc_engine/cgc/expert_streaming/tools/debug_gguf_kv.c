#include "cgc_gguf_lite.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(argv[1]);
    if (!ctx) { printf("LOAD FAILED\n"); return 1; }
    printf("n_tensors=%llu n_kv=%llu\n",
           (unsigned long long)ctx->n_tensors, (unsigned long long)ctx->n_kv);
    for (int i = 0; i < (int)ctx->n_kv && i < 66; i++) {
        cgc_gguf_kv_t* kv = &ctx->kvs[i];
        printf("#%d: %s type=%d i64=%lld str=%.60s\n", i, kv->key, kv->value_type,
               (long long)kv->i64_val, kv->str_val);
    }
    cgc_gguf_lite_free(ctx);
    return 0;
}
