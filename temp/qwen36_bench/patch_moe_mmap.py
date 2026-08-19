#!/usr/bin/env python3
"""Qwen36MoERunner: load the per-layer expert blob as an mmap'd zero-copy
MTLBuffer instead of Data(contentsOf:) + makeBuffer(bytes:) which fully
allocates every layer up front (40 x 432MB = 17.3GB on a 16GB Mac -> OOM).

bytesNoCopy wraps the file-backed mapping; the OS pages it in lazily as the
GPU touches experts, so RSS tracks the working set (hot experts only).
"""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Kernels/MoE/Qwen36MoERunner.swift"
src = open(path).read()

old = """        let blobData = try Data(contentsOf: layerFile)
        guard let blobBuf = device.makeBuffer(bytes: [UInt8](blobData),
                                              length: blobData.count,
                                              options: .storageModeShared) else {
            throw ModelError.missingFile(name: "expert blob alloc failed")
        }
        expertBlob = blobBuf"""
new = """        // Zero-copy mmap: OS pages the file in lazily as the GPU touches
        // experts (RSS tracks the working set, not the whole 40-layer set).
        let fileHandle = try FileHandle(forReadingFrom: layerFile)
        defer { try? fileHandle.close() }
        let fileSize = try fileHandle.seekToEnd()
        guard fileSize > 0 else {
            throw ModelError.missingFile(name: "expert blob empty")
        }
        let map = try fileHandle.map(Int(fileSize))
        guard let base = map.baseAddress else {
            throw ModelError.missingFile(name: "expert blob mmap failed")
        }
        guard let blobBuf = device.makeBuffer(bytesNoCopy: base,
                                              length: Int(fileSize),
                                              options: .storageModeShared,
                                              deallocator: { _, _ in }) else {
            throw ModelError.missingFile(name: "expert blob wrap failed")
        }
        // Keep the mapping alive for the buffer's lifetime.
        blobMappings.append(map)
        expertBlob = blobBuf"""
assert old in src, "expert blob load block not found"
src = src.replace(old, new)

# add the mappings array property
old2 = """    private let maxTokens: Int"""
new2 = """    private let maxTokens: Int
    /// File-backed mappings backing `expertBlob` (kept alive for buffer lifetime).
    private var blobMappings: [Data] = []"""
assert old2 in src, "maxTokens property not found"
src = src.replace(old2, new2)

open(path, "w").write(src)
print("Qwen36MoERunner: mmap zero-copy expert blob")
