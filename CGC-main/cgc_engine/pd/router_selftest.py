#!/usr/bin/env python3
"""Router decode-速度感知 單元測試（2026-08-29）— 7 case 回歸保護。

跑法：python3 CGC-main/cgc_engine/pd/router_selftest.py
覆蓋：
  1. 35B + 8GB Windows（塞不下）        → pure_cloud，decode 上 Mac
  2. 7B + Windows 12 t/s vs Mac 85 t/s  → pure_cloud，decode 送最快（新行為）
  3. 7B 兩台同速（tie）                 → decode 留本地（edge-first）
  4. 無 remote                          → local_full
  5. Mac 當 local                        → local_full
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery import DeviceProfile, PDNode, NodeStatus  # noqa: E402
from router import ComputeRouter  # noqa: E402


def node(nid, ram, vram, gpu, cores, decode, prefill):
    p = DeviceProfile(total_ram_gb=ram, available_ram_gb=ram * 0.5, gpu_type=gpu,
                      gpu_vram_gb=vram, cpu_cores=cores, cpu_arch="x86_64",
                      compute_score=0.0,
                      prefill_tok_per_sec={"qwen36_35b": prefill, "qwen25_7b": prefill * 4},
                      decode_tok_per_sec={"qwen36_35b": decode, "qwen25_7b": decode * 4},
                      network_latency_ms=2.0, bandwidth_mbps=500.0)
    return PDNode(node_id=nid, host="h", port=1, status=NodeStatus.HEALTHY, profile=p)


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: got {got} want {want}")
    return not ok


def main():
    mac = node("mac-m4", 64, 24, "apple_m4_max", 16, decode=22.0, prefill=44.0)
    win8 = node("win-8gb", 8, 2, "mx250", 12, decode=1.4, prefill=6.0)
    win8.profile.decode_tok_per_sec["qwen25_7b"] = 12.0
    mac.profile.decode_tok_per_sec["qwen25_7b"] = 85.0
    win8.profile.prefill_tok_per_sec["qwen25_7b"] = 25.0
    mac.profile.prefill_tok_per_sec["qwen25_7b"] = 240.0

    fails = 0
    # Case 1: 35B + Windows 8GB（塞不下）→ pure_cloud，decode 上 Mac
    r = ComputeRouter(local_node=win8)
    r.register(mac)
    d = r.select(prompt_tokens=200, output_tokens=64, model="qwen36_35b")
    fails += check("35b/8GB mode", d.mode, "pure_cloud")
    fails += check("35b/8GB decode_node", d.decode_node.node_id, "mac-m4")

    # Case 2: 7B + Windows 能跑但 Mac decode 85 vs 12 → decode 送 Mac（新行為）
    d = r.select(prompt_tokens=200, output_tokens=64, model="qwen25_7b")
    fails += check("7b decode→fastest (mode)", d.mode, "pure_cloud")
    fails += check("7b decode→fastest (node)", d.decode_node.node_id, "mac-m4")

    # Case 3: 7B + 兩台同速（tie）→ decode 留 local（edge-first）
    mac_tie = node("mac-tie", 64, 24, "apple_m4_max", 16, decode=3.0, prefill=12.0)
    mac_tie.profile.decode_tok_per_sec["qwen25_7b"] = 12.0
    mac_tie.profile.prefill_tok_per_sec["qwen25_7b"] = 25.0
    r3 = ComputeRouter(local_node=win8)
    r3.register(mac_tie)
    d3 = r3.select(prompt_tokens=200, output_tokens=64, model="qwen25_7b")
    fails += check("7b tie decode→local", d3.decode_node.node_id, "win-8gb")

    # Case 4: 無 remote → local_full
    r4 = ComputeRouter(local_node=win8)
    d4 = r4.select(prompt_tokens=200, output_tokens=64, model="qwen25_7b")
    fails += check("solo local_full", d4.mode, "local_full")

    # Case 5: Mac 當 local → local_full
    r5 = ComputeRouter(local_node=mac)
    r5.register(win8)
    d5 = r5.select(prompt_tokens=200, output_tokens=64, model="qwen36_35b")
    fails += check("mac-local local_full", d5.mode, "local_full")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
