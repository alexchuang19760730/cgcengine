#!/usr/bin/env python3
"""Fix mutable pointer for makeBuffer(bytesNoCopy:)."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Kernels/MoE/Qwen36MoERunner.swift"
src = open(path).read()

old = """        guard let base = mapping.withUnsafeBytes({ $0.baseAddress }) else {
            throw ModelError.missingFile(name: "expert blob mmap failed")
        }
        guard let blobBuf = device.makeBuffer(bytesNoCopy: base,"""

new = """        guard let rawBase = mapping.withUnsafeBytes({ $0.baseAddress }) else {
            throw ModelError.missingFile(name: "expert blob mmap failed")
        }
        // mmap'd pages are read-only; Metal reads them, so a mutating
        // view is safe here.
        let base = UnsafeMutableRawPointer(mutating: rawBase)
        guard let blobBuf = device.makeBuffer(bytesNoCopy: base,"""

assert old in src, "pointer block not found"
src = src.replace(old, new)
open(path, "w").write(src)
print("Qwen36MoERunner: mutable pointer fix applied")
