#!/usr/bin/env python3
"""
诊断 GGUF MoE Tensor 结构 - 使用 llama_cpp
"""

import sys
import os
import gc


def main():
    gguf_path = "/home/gs01/models/Phi-3.5-MoE-instruct-Q4_K_M.gguf"

    print(f"📂 读取: {gguf_path}")
    print(f"📊 文件大小: {os.path.getsize(gguf_path) / (1024**3):.2f} GB")

    print(f"\n🔍 尝试使用 llama_cpp 读取...")

    try:
        from llama_cpp import Llama

        print(f"   初始化 Llama model...")
        llm = Llama(
            model_path=gguf_path,
            n_ctx=512,
            verbose=False
        )

        print(f"✅ Llama model 加载成功")
        print(f"   n_vocab: {llm.n_vocab}")
        print(f"   n_embd: {llm.n_embd}")
        print(f"   n_layer: {llm.n_layer}")
        print(f"   n_head: {llm.n_head}")
        print(f"   n_ctx: {llm.n_ctx}")

        # 尝试获取模型参数
        print(f"\n📋 模型参数:")
        params = llm.metadata
        if params:
            for key, value in params.items():
                print(f"   {key}: {value}")

        del llm
        gc.collect()

    except Exception as e:
        print(f"❌ llama_cpp 加载失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n🔍 尝试直接读取 GGUF 二进制...")

    try:
        # 直接读取 GGUF 二进制文件来分析
        with open(gguf_path, 'rb') as f:
            # 读取 header
            magic = f.read(4)
            print(f"   Magic: {magic}")

            # 读取版本
            version = f.read(4)
            print(f"   Version: {int.from_bytes(version, 'little')}")

            # 读取 tensor 数量
            n_tensors = f.read(8)
            print(f"   n_tensors: {int.from_bytes(n_tensors, 'little')}")

    except Exception as e:
        print(f"❌ GGUF 二进制读取失败: {e}")


if __name__ == "__main__":
    main()