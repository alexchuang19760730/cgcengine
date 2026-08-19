#include "magi_compiler_integration.h"

int main() {
    // 初始化编译器
    MagiCompiler compiler;

    // 一键注入：KDA v4 + 正交固定 KV
    bool success = compiler.compile_llm_with_kda_ortho_kv(
        32,        // heads
        128,       // head_dim
        "vllm"     // backend: vllm / llama.cpp
    );

    if (success) {
        printf("\n🎉 MagiCompiler 集成成功!\n");
        printf("📦 固定 KV 配置:\n");
        printf("   - heads: 32\n");
        printf("   - head_dim: 128\n");
        printf("   - ortho_dim: 128\n");
        printf("   - KV大小: ~16 MB (固定不变)\n");
        printf("\n🚀 可以开始推理!\n");
        return 0;
    } else {
        printf("\n❌ MagiCompiler 集成失败\n");
        return 1;
    }
}