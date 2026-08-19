#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WSL 环境下的 vLLM Benchmark 测试"""

import os
import sys
import time
import gc

def main():
    print("=" * 80)
    print("WSL vLLM Benchmark 测试")
    print("=" * 80)

    # 路径设置
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # 检查环境
    print("\n[1] 环境检查")
    print("-" * 50)

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"PyTorch 错误: {e}")

    try:
        import vllm
        print(f"vLLM: {vllm.__version__}")
    except Exception as e:
        print(f"vLLM 错误: {e}")
        return

    try:
        from cgc_engine.agent.harness_agent import HarnessAgent
        from cgc_engine.agent.harness_strategy import StrategyDispatcher, MagiBackendType, MagiExecuteMode
        print("Harness Agent: 就绪")
    except Exception as e:
        print(f"Harness Agent 错误: {e}")
        return

    print("\n[2] 策略分发器测试")
    print("-" * 50)

    try:
        dispatcher = StrategyDispatcher()

        # 测试所有可用后端
        backends = [
            (MagiBackendType.VLLM, "vLLM"),
            (MagiBackendType.LLAMA_CPP, "llama.cpp"),
            (MagiBackendType.MEGATRAIN_2026_4, "MegaTrain"),
            (MagiBackendType.MLX_TUNE, "MLX Tune"),
        ]

        for backend, name in backends:
            try:
                strategy = dispatcher.dispatch(backend, MagiExecuteMode.INFER_DECODE)
                summary = dispatcher.get_strategy_summary()
                print(f"\n{name}:")
                print(f"  - 整图编译: {summary['compile']['full_graph']}")
                print(f"  - 分布式: {summary['distributed']['enabled']}")
                print(f"  - 内存管理: {summary['memory']['compiler_as_manager']}")
            except Exception as e:
                print(f"\n{name} 错误: {e}")

    except Exception as e:
        print(f"策略分发器错误: {e}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print("\n下一步：")
    print("1. 确保 WSL 支持 GPU（检查 CUDA 设置）")
    print("2. 下载模型权重")
    print("3. 运行完整的推理 benchmark")

if __name__ == "__main__":
    main()
