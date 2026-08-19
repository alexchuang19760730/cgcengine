#!/usr/bin/env python3
"""Fix DeltaNetRunner.reset(): MTLBuffer.contents() pointers are non-optional."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Kernels/Attention/DeltaNetRunner.swift"
src = open(path).read()

old = """    public func reset() {
        if let hp = hState.contents().assumingMemoryBound(to: Float.self) {
            memset(hp, 0, hState.length)
        }
        if let cp = convState.contents().assumingMemoryBound(to: Float16.self) {
            memset(cp, 0, convState.length)
        }
        if let tp = convStateTmp.contents().assumingMemoryBound(to: Float16.self) {
            memset(tp, 0, convStateTmp.length)
        }
    }"""
new = """    public func reset() {
        let hp = hState.contents().assumingMemoryBound(to: Float.self)
        memset(hp, 0, hState.length)
        let cp = convState.contents().assumingMemoryBound(to: Float16.self)
        memset(cp, 0, convState.length)
        let tp = convStateTmp.contents().assumingMemoryBound(to: Float16.self)
        memset(tp, 0, convStateTmp.length)
    }"""
assert old in src, "DeltaNet reset not found"
src = src.replace(old, new)
open(path, "w").write(src)
print("DeltaNetRunner.reset: fixed")
