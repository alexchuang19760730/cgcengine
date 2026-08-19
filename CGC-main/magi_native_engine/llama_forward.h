#pragma once
#include <stdint.h>

void llama_forward(
    int n_layer,
    int dim,
    int n_head,
    int n_kv_head,
    int head_dim,
    int vocab_size,
    int* tokens,
    int n_tokens,
    int* out_tokens,
    int max_gen
);
