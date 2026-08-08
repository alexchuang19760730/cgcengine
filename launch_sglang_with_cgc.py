import os
import sys

# Enable TrueOrthoKDA via CGC Unified IR
os.environ["CGC_ENABLE_ORTHO_KDA"] = "1"
os.environ["CGC_ORTHO_BASE_DIM"] = "128"

# Add path if necessary
workspace = "/root/flashkv0516/ComputeGraphCompiler-main"
sys.path.insert(0, workspace)
sys.path.insert(0, f"{workspace}/Backend/CGC/cloud_sglang/python")

try:
    from Backend.CGC.compiler.unified_compiler import inject_unified_ir_for_role
    print("[CGC Launcher] Injecting Unified IR for role: cli_universe")
    result = inject_unified_ir_for_role("cli_universe")
    print(f"[CGC Launcher] Injection result: {result['injection']}")
except Exception as e:
    print(f"[CGC Launcher] Failed to inject Unified IR: {e}")
    raise

# Start SGLang server natively, but with CGC injected into its runtime
print("[CGC Launcher] Starting SGLang server...")
from sglang.launch_server import main

if __name__ == "__main__":
    main()
