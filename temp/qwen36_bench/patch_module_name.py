#!/usr/bin/env python3
"""The bundle file is qwen36_forward.metal but the registered module name is
"qwen36" -> Bundle.module.url(forResource: "qwen36") returns nil. The module
name must match the filename: use "qwen36_forward"."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift"
src = open(path).read()

old1 = """        "moe_qwen36",
        "qwen36",
    ]"""
new1 = """        "moe_qwen36",
        "qwen36_forward",
    ]"""
assert old1 in src, "shaderModules anchor not found"
src = src.replace(old1, new1)

old2 = """        "moe_qwen36": "Metal/MoE",
        "qwen36": "Metal/Qwen36",
    ]"""
new2 = """        "moe_qwen36": "Metal/MoE",
        "qwen36_forward": "Metal/Qwen36",
    ]"""
assert old2 in src, "shaderSubdirectories anchor not found"
src = src.replace(old2, new2)

open(path, "w").write(src)
print("MetalContext: module name qwen36 -> qwen36_forward (matches filename)")
