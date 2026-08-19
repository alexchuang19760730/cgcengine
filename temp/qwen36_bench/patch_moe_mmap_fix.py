#!/usr/bin/env python3
"""Fix Qwen36MoERunner mmap: FileHandle.map is Linux-only. Use
Data(contentsOf: .mappedIfSafe) — contiguous backing that stays valid
while the Data is alive (kept in blobMappings)."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Kernels/MoE/Qwen36MoERunner.swift"
src = open(path).read()

old = """        // Zero-copy mmap: OS pages the file in lazily as the GPU touches
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

new = """        // Zero-copy mmap: OS pages the file in lazily as the GPU touches
        // experts (RSS tracks the working set, not the whole 40-layer set).
        // Data(contentsOf: .mappedIfSafe) keeps the file mapping alive for as
        // long as the Data instance lives; we hold it in blobMappings so the
        // base address stays valid for the MTLBuffer's lifetime.
        let mapping = try Data(contentsOf: layerFile, options: [.mappedIfSafe])
        guard mapping.count > 0 else {
            throw ModelError.missingFile(name: "expert blob empty")
        }
        guard let base = mapping.withUnsafeBytes({ $0.baseAddress }) else {
            throw ModelError.missingFile(name: "expert blob mmap failed")
        }
        guard let blobBuf = device.makeBuffer(bytesNoCopy: base,
                                              length: mapping.count,
                                              options: .storageModeShared,
                                              deallocator: { _, _ in }) else {
            throw ModelError.missingFile(name: "expert blob wrap failed")
        }
        // Keep the mapping alive for the buffer's lifetime.
        blobMappings.append(mapping)
        expertBlob = blobBuf"""

assert old in src, "mmap block not found"
src = src.replace(old, new)
open(path, "w").write(src)
print("Qwen36MoERunner: portable mmap fix applied")
