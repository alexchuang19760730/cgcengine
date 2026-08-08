# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
三層學習器優化目標文檔

本文檔說明 Harness Agent 三層（計算層、調度層、存儲層）的優化目標
"""

LAYER_OPTIMIZATION_OBJECTIVES = {
    "computation": {
        "name": "計算層",
        "description": "負責算子融合、kernel 選擇、量化等計算優化",
        "objectives": [
            {
                "metric": "latency",
                "name": "延遲",
                "target": "最小化",
                "weight": 0.4,
            },
            {
                "metric": "throughput",
                "name": "吞吐量",
                "target": "最大化",
                "weight": 0.3,
            },
            {
                "metric": "memory_bandwidth",
                "name": "內存帶寬利用率",
                "target": "最大化",
                "weight": 0.2,
            },
            {
                "metric": "kernel_selection",
                "name": "Kernel 選擇準確率",
                "target": "最大化",
                "weight": 0.1,
            },
        ],
        "key_learnings": [
            "算子融合模式",
            "Tiling 配置",
            "Quantization 策略",
            "Memory Layout 選擇",
        ],
    },
    "scheduling": {
        "name": "調度層",
        "description": "負責 Batch 調度、Prefill/Decode 分離、Prefix Caching 等調度優化",
        "objectives": [
            {
                "metric": "batch_utilization",
                "name": "Batch 利用率",
                "target": "最大化",
                "weight": 0.35,
            },
            {
                "metric": "pd_separation_efficiency",
                "name": "PD 分離效率",
                "target": "最大化",
                "weight": 0.25,
            },
            {
                "metric": "prefix_cache_hit_rate",
                "name": "Prefix Caching 命中率",
                "target": "最大化",
                "weight": 0.20,
            },
            {
                "metric": "waiting_time",
                "name": "請求等待時間",
                "target": "最小化",
                "weight": 0.15,
            },
            {
                "metric": "continuous_batching_overhead",
                "name": "Continuous Batching 開銷",
                "target": "最小化",
                "weight": 0.05,
            },
        ],
        "key_learnings": [
            "Batch Size 配置",
            "PD 分離閾值",
            "Prefix Caching 策略",
            "動態 Batch 調整",
        ],
    },
    "storage": {
        "name": "存儲層",
        "description": "負責 KV Cache 管理、KDA、Prefetch、Memory Layout 等存儲優化",
        "objectives": [
            {
                "metric": "kv_cache_hit_rate",
                "name": "KV Cache 命中率",
                "target": "最大化",
                "weight": 0.30,
            },
            {
                "metric": "memory_efficiency",
                "name": "內存效率",
                "target": "最大化",
                "weight": 0.25,
            },
            {
                "metric": "kda_bandwidth",
                "name": "KDA 帶寬",
                "target": "最大化",
                "weight": 0.20,
            },
            {
                "metric": "prefetch_accuracy",
                "name": "Prefetch 準確率",
                "target": "最大化",
                "weight": 0.15,
            },
            {
                "metric": "fragmentation",
                "name": "內存碎片",
                "target": "最小化",
                "weight": 0.10,
            },
        ],
        "key_learnings": [
            "KV Cache 策略 (LRU/LFU)",
            "PagedAttention 配置",
            "Quantization 壓縮比",
            "Prefetch 距離",
        ],
    },
}


def get_layer_objectives(layer: str) -> dict:
    """獲取指定層的優化目標"""
    return LAYER_OPTIMIZATION_OBJECTIVES.get(layer, {})


def get_all_objectives() -> dict:
    """獲取所有層的優化目標"""
    return LAYER_OPTIMIZATION_OBJECTIVES


def print_layer_objectives(layer: str) -> None:
    """打印指定層的優化目標"""
    obj = get_layer_objectives(layer)
    if not obj:
        print(f"Unknown layer: {layer}")
        return

    print(f"\n{'='*60}")
    print(f"{obj['name']} - {obj['description']}")
    print(f"{'='*60}")
    print("\n優化目標:")
    for i, metric in enumerate(obj["objectives"], 1):
        print(f"  {i}. {metric['name']} ({metric['metric']})")
        print(f"     - 目標: {metric['target']}")
        print(f"     - 權重: {metric['weight']:.0%}")

    print("\n關鍵學習內容:")
    for learning in obj["key_learnings"]:
        print(f"  - {learning}")


if __name__ == "__main__":
    for layer in ["computation", "scheduling", "storage"]:
        print_layer_objectives(layer)
