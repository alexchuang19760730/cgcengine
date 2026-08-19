#!/usr/bin/env python3
"""Register the qwen36 forward shader module in MetalContext so the new
q36_embed / q36_residual_add / q36_fp16_gemv kernels compile into the shared
runtime library."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift"
src = open(path).read()

# 1. shaderModules: add "qwen36" after "moe_qwen36"
old1 = """        "deltanet",
        "gatedattn",
        "moe_qwen36",
    ]"""
new1 = """        "deltanet",
        "gatedattn",
        "moe_qwen36",
        "qwen36",
    ]"""
assert old1 in src, "shaderModules anchor not found"
src = src.replace(old1, new1)

# 2. shaderSubdirectories: add entry after moe_qwen36
old2 = """        "moe_qwen36": "Metal/MoE",
    ]"""
new2 = """        "moe_qwen36": "Metal/MoE",
        "qwen36": "Metal/Qwen36",
    ]"""
assert old2 in src, "shaderSubdirectories anchor not found"
src = src.replace(old2, new2)

open(path, "w").write(src)
print("MetalContext: qwen36 module registered (shaderModules + subdir Metal/Qwen36)")
