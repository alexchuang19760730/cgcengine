#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "llama_forward.h"

int main() {
    printf("\n");
    printf("🔥 MagiCompiler KDA 原生推理引擎\n");
    printf("🧠 模型：Qwen2.5-7B (Metal GPU)\n");
    printf("⚡ 加速：KDA + Metal 融合算子\n");
    printf("=====================================\n");

    const int n_layer_test = 28;
    const int dim_test = 3584;
    const int n_head_test = 28;
    const int n_kv_head_test = 2;
    const int head_dim_test = 128;
    const int vocab_size_test = 151936;

    const char* prompt = "Hello world";
    int tokens[64] = {0};
    int n_tokens = strlen(prompt);
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = (int)prompt[i] % vocab_size_test;
    }

    int output[64] = {0};

    llama_forward(
        n_layer_test,
        dim_test,
        n_head_test,
        n_kv_head_test,
        head_dim_test,
        vocab_size_test,
        tokens, n_tokens,
        output, 8
    );

    printf("\n✅ 推理完成！\n");
    printf("📝 输出 token IDs: ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");
    return 0;
}
