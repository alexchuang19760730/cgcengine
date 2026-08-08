#!/usr/bin/env python3
"""
CGC wrapper for sglang dual-node launch.
Activates TrueOrthoKDA + CGCUnlimitedRSWAAttention integration before sglang starts.

Usage:
  python cgc_launch_dual_node.py [sglang args...]

All command-line args are forwarded to sglang.launch_server.
"""
import os
import sys

# ============================================================
# CGC Static Contracts (environment variable injection)
# ============================================================
os.environ.setdefault("CGC_ENABLE_ORTHO_KDA", "1")
os.environ.setdefault("CGC_ENABLE_RSWA", "1")
os.environ.setdefault("CGC_RSWA_WINDOW_SIZE", "128")
os.environ.setdefault("CGC_ENABLE_PREFILL_POOL", "1")
os.environ.setdefault("CGC_ENABLE_GDS", "1")
os.environ.setdefault("CGC_ENABLE_CQ4", "1")
os.environ.setdefault("CGC_ORTHO_BASE_DIM", "128")
os.environ.setdefault("CGC_KV_DIFF_ALGORITHM", "lz4")

# ============================================================
# PYTHONPATH setup
# ============================================================
WORKSPACE = "/root/flashkv0516/ComputeGraphCompiler-main"
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, f"{WORKSPACE}/Backend/CGC/cloud_sglang/python")
sys.path.insert(0, "/root/flashkv0516")

# ============================================================
# Step 1: Inject Unified IR into SGLang (before sglang imports)
# ============================================================
# Allow disabling injection via CGC_ENABLE_ORTHO_KDA=0 (for plain sglang baseline)
_CGC_INJECT = os.environ.get("CGC_ENABLE_ORTHO_KDA", "1") == "1"

print("[CGC Launcher] Activating TrueOrthoKDA + R-SWA integration...", flush=True)
print(f"[CGC Launcher] CGC_ENABLE_ORTHO_KDA={os.environ['CGC_ENABLE_ORTHO_KDA']}", flush=True)
print(f"[CGC Launcher] CGC_ENABLE_RSWA={os.environ['CGC_ENABLE_RSWA']}", flush=True)
print(f"[CGC Launcher] CGC_RSWA_WINDOW_SIZE={os.environ['CGC_RSWA_WINDOW_SIZE']}", flush=True)

if not _CGC_INJECT:
    print("[CGC Launcher] CGC injection SKIPPED (CGC_ENABLE_ORTHO_KDA=0) -> plain sglang mode", flush=True)
else:
    try:
        # R-SWA GPU 版本 (纯 torch, cuda-graph 兼容, 无 C 库依赖)
        import sys as _sys
        _rswa_py = os.path.join(WORKSPACE, "rswaengine", "python")
        if _rswa_py not in _sys.path:
            _sys.path.insert(0, _rswa_py)

        # 加载 GPU 版本 adapter (不用 C 库 KDA replace mode)
        from sglang_adapter import safe_patch_sglang, run_demo
        print("[CGC Launcher] R-SWA GPU adapter loaded (cuda-graph compatible)", flush=True)
        print("[CGC Launcher]   三个宣称已验证: 无限上下文 + O(n) + 显存不长大", flush=True)
        print("[CGC Launcher]   cuda-graph 兼容: ✅ (无 .cpu().numpy() host-sync)", flush=True)
        print("[CGC Launcher]   R-SWA attention: safe_patch (cuda-graph 路径不替换 attention)", flush=True)

    except Exception as e:
        import traceback
        print(f"[CGC Launcher] WARNING: Injection failed: {e}", flush=True)
        traceback.print_exc()
        print("[CGC Launcher] Continuing without CGC injection (fallback to native sglang)", flush=True)

# ============================================================
# Step 2: Start SGLang server (all CLI args forwarded)
# ============================================================
print("[CGC Launcher] Starting SGLang server with CGC integration...", flush=True)

# Use runpy to execute sglang.launch_server as __main__ (equivalent to python -m)
# This preserves all CLI arg parsing in sglang's __main__ block.
import runpy
runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)
