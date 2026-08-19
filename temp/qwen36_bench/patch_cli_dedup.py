#!/usr/bin/env python3
"""Remove the duplicate `let detectedArch = detectArch(modelURL)` at the
Model.load site (line ~117) — it is now declared at the top of run()."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfareCLI/Run.swift"
src = open(path).read()

old = """        let context = try MetalContext()
        let detectedArch = detectArch(modelURL)
        let model = try Model.load("""
new = """        let context = try MetalContext()
        let model = try Model.load("""
assert old in src, "duplicate detectedArch block not found"
src = src.replace(old, new)
open(path, "w").write(src)
print("Run.swift: duplicate detectedArch removed")
