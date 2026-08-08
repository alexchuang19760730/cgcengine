#!/usr/bin/env python3
import os
import json
import re

LOCAL_DIR = "/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main"

GATE_VERSIONS = {
    "CGC_Gate_1.0_edge_cloud_autonomy": "1.0",
    "CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation": "2.0",
    "CGC_Gate_2.1_speculative_decode_fusion_optimization": "2.1",
    "CGC_Gate_2.2_deepep_moe_load_balancing": "2.2",
    "CGC_Gate_2.3_unlimited_rswa_prefill_pool": "2.3",
    "CGC_Gate_3.0_train_inference_unification": "3.0",
    "CGC_Gate_3.1_self_harness": "3.1",
    "CGC_Gate_5.0_audit_trace_replay_visualization": "5.0"
}

def find_gate_map(gate_path):
    for f in os.listdir(gate_path):
        if "gate_map" in f and f.endswith(".json"):
            return os.path.join(gate_path, f)
    return None

def check_non_done_capabilities():
    print("============================================================")
    print("           🔍 检测所有 Gate 中非 done 状态的能力")
    print("============================================================")
    print()
    
    all_non_done = {}
    total_capabilities = 0
    done_count = 0
    non_done_count = 0
    
    for gate, version in GATE_VERSIONS.items():
        gate_path = os.path.join(LOCAL_DIR, "docs", "technical_whitepapers", gate)
        
        if not os.path.isdir(gate_path):
            print(f"❌ {gate} - 目录不存在")
            continue
        
        gate_map_file = find_gate_map(gate_path)
        
        if not gate_map_file:
            print(f"❌ {gate} - gate_map.json 不存在")
            continue
        
        try:
            with open(gate_map_file, "r") as f:
                gate_map = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ {gate} - gate_map.json 格式错误")
            continue
        
        capabilities = gate_map.get("capabilities", [])
        gate_non_done = []
        
        for cap in capabilities:
            cap_id = cap.get("capability_id", "unknown")
            cap_name = cap.get("name", "unknown")
            status = cap.get("status", "unknown")
            
            total_capabilities += 1
            
            if status != "done":
                gate_non_done.append({
                    "capability_id": cap_id,
                    "name": cap_name,
                    "status": status
                })
                non_done_count += 1
            else:
                done_count += 1
        
        if gate_non_done:
            all_non_done[gate] = gate_non_done
    
    print("📊 总体统计:")
    print(f"   能力总数: {total_capabilities}")
    print(f"   ✅ done: {done_count}")
    print(f"   ⚠️  非 done: {non_done_count}")
    print()
    
    if not all_non_done:
        print("🎉 恭喜！所有能力均已标记为 done 状态！")
        print("============================================================")
        return
    
    print("📋 非 done 状态的能力列表:")
    print("------------------------------------------------------------")
    
    for gate, caps in all_non_done.items():
        print(f"\n【{gate}】")
        print(f"   非 done 能力数量: {len(caps)}")
        print(f"   --------------------------------------------------------")
        
        for cap in caps:
            status_color = ""
            if cap["status"] == "proof":
                status_color = "\033[1;33m"
            elif cap["status"] == "integrated":
                status_color = "\033[0;34m"
            elif cap["status"] == "target":
                status_color = "\033[0;35m"
            else:
                status_color = "\033[0;31m"
            
            print(f"   • {cap['capability_id']}")
            print(f"     名称: {cap['name']}")
            print(f"     状态: {status_color}{cap['status']}\033[0m")
            print()
    
    print("============================================================")
    print("状态说明:")
    print("   • proof: 验证中")
    print("   • integrated: 已集成")
    print("   • target: 目标状态")
    print("   • 其他: 未知状态")
    print("============================================================")

if __name__ == "__main__":
    check_non_done_capabilities()
