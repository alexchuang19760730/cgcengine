#!/usr/bin/env python3
"""
DFlash端云一体防造假测试
模拟端云一体推理场景，验证防造假机制能否检测出异常数据
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cgc_engine'))

def simulate_dflash_edge_cloud_inference():
    """模拟DFlash端云一体推理"""
    print("=" * 90)
    print("🌐 DFlash端云一体防造假测试")
    print("=" * 90)
    
    # 测试场景定义
    test_scenarios = [
        {
            "name": "场景1：正常端云一体推理",
            "description": "云端Prefill + 端侧Decode，所有数据正常",
            "cloud_prefill_time": 48.13,
            "kv_transfer_time": 20.0,
            "edge_decode_time": 7.66,
            "nvml_vram": 16000.0,
            "engine_vram": 16800.0,
            "gpu_utilization": 85.0,
            "tok_per_sec": 121596.0,
            "kv_read_bytes": 1024 * 1024 * 200,  # 200MB
            "kv_write_bytes": 1024 * 1024 * 50,  # 50MB
            "baseline_time": 100.0,
            "expect_valid": True
        },
        {
            "name": "场景2：加速比异常（>20x）",
            "description": "云端Prefill过快，导致加速比超过20x",
            "cloud_prefill_time": 2.0,
            "kv_transfer_time": 20.0,
            "edge_decode_time": 7.66,
            "nvml_vram": 16000.0,
            "engine_vram": 16800.0,
            "gpu_utilization": 85.0,
            "tok_per_sec": 500000.0,
            "kv_read_bytes": 1024 * 1024 * 200,
            "kv_write_bytes": 1024 * 1024 * 50,
            "baseline_time": 100.0,
            "expect_valid": False
        },
        {
            "name": "场景3：显存不一致（误差>20%）",
            "description": "引擎统计显存与NVML显存差异过大",
            "cloud_prefill_time": 48.13,
            "kv_transfer_time": 20.0,
            "edge_decode_time": 7.66,
            "nvml_vram": 16000.0,
            "engine_vram": 24000.0,  # 误差50%
            "gpu_utilization": 85.0,
            "tok_per_sec": 121596.0,
            "kv_read_bytes": 1024 * 1024 * 200,
            "kv_write_bytes": 1024 * 1024 * 50,
            "baseline_time": 100.0,
            "expect_valid": False
        },
        {
            "name": "场景4：GPU利用率低但tok/s高",
            "description": "GPU利用率只有20%，但tok/s异常高",
            "cloud_prefill_time": 48.13,
            "kv_transfer_time": 20.0,
            "edge_decode_time": 7.66,
            "nvml_vram": 16000.0,
            "engine_vram": 16800.0,
            "gpu_utilization": 20.0,  # 利用率低
            "tok_per_sec": 200000.0,  # tok/s异常高
            "kv_read_bytes": 1024 * 1024 * 200,
            "kv_write_bytes": 1024 * 1024 * 50,
            "baseline_time": 100.0,
            "expect_valid": False
        },
        {
            "name": "场景5：KV带宽异常（<100MB/s）",
            "description": "KV读写带宽过低，明显不合理",
            "cloud_prefill_time": 48.13,
            "kv_transfer_time": 20.0,
            "edge_decode_time": 7.66,
            "nvml_vram": 16000.0,
            "engine_vram": 16800.0,
            "gpu_utilization": 85.0,
            "tok_per_sec": 121596.0,
            "kv_read_bytes": 1024 * 1024 * 2,  # 只有2MB
            "kv_write_bytes": 1024 * 1024 * 1,  # 1MB
            "baseline_time": 100.0,
            "expect_valid": False
        },
        {
            "name": "场景6：KV带宽异常（>4000MB/s）",
            "description": "KV读写带宽过高，超出合理范围",
            "cloud_prefill_time": 48.13,
            "kv_transfer_time": 20.0,
            "edge_decode_time": 7.66,
            "nvml_vram": 16000.0,
            "engine_vram": 16800.0,
            "gpu_utilization": 85.0,
            "tok_per_sec": 121596.0,
            "kv_read_bytes": 1024 * 1024 * 5000,  # 5GB
            "kv_write_bytes": 1024 * 1024 * 1000,  # 1GB
            "baseline_time": 100.0,
            "expect_valid": False
        },
        {
            "name": "场景7：显存异常（>48GB）",
            "description": "显存占用超出双5090最大容量",
            "cloud_prefill_time": 48.13,
            "kv_transfer_time": 20.0,
            "edge_decode_time": 7.66,
            "nvml_vram": 50000.0,  # 50GB超出48GB
            "engine_vram": 16800.0,
            "gpu_utilization": 85.0,
            "tok_per_sec": 121596.0,
            "kv_read_bytes": 1024 * 1024 * 200,
            "kv_write_bytes": 1024 * 1024 * 50,
            "baseline_time": 100.0,
            "expect_valid": False
        },
        {
            "name": "场景8：耗时为负数",
            "description": "总耗时为负数，明显异常",
            "cloud_prefill_time": -48.13,
            "kv_transfer_time": 20.0,
            "edge_decode_time": 7.66,
            "nvml_vram": 16000.0,
            "engine_vram": 16800.0,
            "gpu_utilization": 85.0,
            "tok_per_sec": 121596.0,
            "kv_read_bytes": 1024 * 1024 * 200,
            "kv_write_bytes": 1024 * 1024 * 50,
            "baseline_time": 100.0,
            "expect_valid": False
        },
    ]
    
    print("\n📋 测试场景概览")
    print("-" * 90)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   预期结果: {'✅ 有效' if scenario['expect_valid'] else '❌ 拒绝'}")
    
    # 模拟防造假检测
    print("\n" + "=" * 90)
    print("🔍 开始防造假检测")
    print("=" * 90)
    
    results = []
    for scenario in test_scenarios:
        result = validate_dflash_scenario(scenario)
        results.append({
            "scenario": scenario["name"],
            "is_valid": result["is_valid"],
            "rejection_reason": result["rejection_reason"],
            "expect_valid": scenario["expect_valid"],
            "detection_correct": result["is_valid"] == scenario["expect_valid"]
        })
    
    # 输出检测结果
    print("\n📊 检测结果汇总")
    print("-" * 90)
    print(f"{'场景':<30} | {'检测结果':<12} | {'拒绝原因':<30} | {'检测正确':<10}")
    print("-" * 90)
    
    correct_count = 0
    for result in results:
        status = "✅ 有效" if result["is_valid"] else "❌ 拒绝"
        detection = "✅ 正确" if result["detection_correct"] else "❌ 错误"
        if result["detection_correct"]:
            correct_count += 1
        
        print(f"{result['scenario']:<30} | {status:<12} | {result['rejection_reason']:<30} | {detection:<10}")
    
    # 统计
    total = len(results)
    accuracy = (correct_count / total) * 100
    
    print("\n" + "=" * 90)
    print("📈 防造假检测统计")
    print("=" * 90)
    print(f"  总测试场景: {total}")
    print(f"  检测正确: {correct_count}")
    print(f"  检测错误: {total - correct_count}")
    print(f"  检测准确率: {accuracy:.1f}%")
    
    if accuracy == 100:
        print("\n  ✅ 所有异常场景均被正确检测！防造假机制工作正常。")
    else:
        print(f"\n  ⚠️ 检测准确率为{accuracy:.1f}%，需要进一步优化。")
    
    return results

def validate_dflash_scenario(scenario):
    """验证DFlash场景数据"""
    is_valid = True
    rejection_reason = "无"
    
    # 计算总耗时（转换为秒）
    total_latency_ms = scenario["cloud_prefill_time"] + scenario["kv_transfer_time"] + scenario["edge_decode_time"]
    total_latency_s = total_latency_ms / 1000.0
    
    # 1. 异常值校验
    # 计算加速比
    if scenario["baseline_time"] > 0 and total_latency_ms > 0:
        speedup = scenario["baseline_time"] / total_latency_ms
        if speedup > 20.0:
            is_valid = False
            rejection_reason = "加速比异常(>20x)"
    
    # 耗时异常
    if total_latency_ms <= 0:
        is_valid = False
        rejection_reason = "耗时异常(<=0ms)"
    
    # 显存异常
    if scenario["nvml_vram"] < 0 or scenario["nvml_vram"] > 48000:
        is_valid = False
        rejection_reason = "显存异常(>48GB)"
    
    # KV带宽异常（单位：MB/s）
    kv_bandwidth = (scenario["kv_read_bytes"] + scenario["kv_write_bytes"]) / total_latency_s / 1024.0 / 1024.0
    if kv_bandwidth < 100 or kv_bandwidth > 4000:
        is_valid = False
        rejection_reason = f"KV带宽异常({kv_bandwidth:.0f}MB/s)"
    
    # 2. 三端一致性校验
    # 引擎显存与NVML显存一致性
    if scenario["nvml_vram"] > 0 and scenario["engine_vram"] > 0:
        diff = abs(scenario["engine_vram"] - scenario["nvml_vram"]) / scenario["nvml_vram"]
        if diff > 0.2:
            is_valid = False
            rejection_reason = f"显存不一致(误差{diff*100:.0f}%)"
    
    # tok/s与GPU利用率一致性
    if scenario["gpu_utilization"] >= 0 and scenario["tok_per_sec"] > 0:
        if scenario["gpu_utilization"] < 30 and scenario["tok_per_sec"] > 100000:
            is_valid = False
            rejection_reason = "GPU利用率与tok/s不匹配"
    
    return {
        "is_valid": is_valid,
        "rejection_reason": rejection_reason
    }

if __name__ == "__main__":
    simulate_dflash_edge_cloud_inference()
