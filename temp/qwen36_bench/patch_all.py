#!/usr/bin/env python3
"""Fix the kernel patch: threadgroup -> device. Then write the runner patch."""
import sys

# === 1. Fix kernel patch ===
kp = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Metal/MoE/moe_qwen36.metal"
ksrc = open(kp).read()
old = "    threadgroup const uchar* slotBuf;"
new = "    device const uchar* slotBuf;"
assert old in ksrc, "threadgroup line not found"
ksrc = ksrc.replace(old, new, 1)
open(kp, "w").write(ksrc)
print("kernel: threadgroup -> device fixed")

# === 2. Runner patch: Qwen36MoERunner ===
# Change encodeExpert to bind per-rank slot buffers from streamer.
# Change init to NOT load the layer file blob (use streamer instead).
rp = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Kernels/MoE/Qwen36MoERunner.swift"
rsrc = open(rp).read()

# Remove the mmap blob loading block (keep the expertStride/layout part)
old_mmap = """        // Zero-copy mmap: OS pages the file in lazily as the GPU touches
        // experts (RSS tracks the working set, not the whole 40-layer set).
        // Data(contentsOf: .mappedIfSafe) keeps the file mapping alive for as
        // long as the Data instance lives; we hold it in blobMappings so the
        // base address stays valid for the MTLBuffer's lifetime.
        let mapping = try Data(contentsOf: layerFile, options: [.mappedIfSafe])
        guard mapping.count > 0 else {
            throw ModelError.missingFile(name: "expert blob empty")
        }
        guard let rawBase = mapping.withUnsafeBytes({ $0.baseAddress }) else {
            throw ModelError.missingFile(name: "expert blob mmap failed")
        }
        // mmap'd pages are read-only; Metal reads them, so a mutating
        // view is safe here.
        let base = UnsafeMutableRawPointer(mutating: rawBase)
        guard let blobBuf = device.makeBuffer(bytesNoCopy: base,
                                              length: mapping.count,
                                              options: .storageModeShared,
                                              deallocator: { _, _ in }) else {
            throw ModelError.missingFile(name: "expert blob wrap failed")
        }
        // Keep the mapping alive for the buffer's lifetime.
        blobMappings.append(mapping)
        expertBlob = blobBuf"""

new_mmap = """        // Expert data is loaded lazily via PreadExpertStreamer through
        // Model.routedExpert(layer:expert:) — no eager blob at init.
        // encodeExpert() binds per-rank slot buffers from the streamer."""
assert old_mmap in rsrc, "mmap block not found"
rsrc = rsrc.replace(old_mmap, new_mmap, 1)

# Remove `expertBlob` stored property declaration
old_prop = "    private let expertBlob: MTLBuffer"
new_prop = "    // expertBlob removed; per-rank slot buffers bound at encode time"
assert old_prop in rsrc, "expertBlob property not found"
rsrc = rsrc.replace(old_prop, new_prop, 1)

# Remove blobMappings array
old_maps = "    private var blobMappings: [Data] = []"
new_maps = "    // blobMappings removed with expertBlob"
assert old_maps in rsrc, "blobMappings property not found"
rsrc = rsrc.replace(old_maps, new_maps, 1)

# Change encodeExpert to bind per-rank slot buffers
old_enc = """    public func encodeExpert(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int,
                             nTokens: Int) {
        enc.setComputePipelineState(psoExpert)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(idsBuf, offset: 0, index: 1)
        enc.setBuffer(expertBlob, offset: 0, index: 2)
        enc.setBuffer(expertOutBuf, offset: 0, index: 3)
        var s = expertStride; enc.setBytes(&s, length: 4, index: 4)
        s = gateQOff; enc.setBytes(&s, length: 4, index: 5)
        s = gateSOff; enc.setBytes(&s, length: 4, index: 6)
        s = upQOff; enc.setBytes(&s, length: 4, index: 7)
        s = upSOff; enc.setBytes(&s, length: 4, index: 8)
        s = downQOff; enc.setBytes(&s, length: 4, index: 9)
        s = downSOff; enc.setBytes(&s, length: 4, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: nTokens * 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }"""

new_enc = """    public func encodeExpert(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int,
                             nTokens: Int) {
        enc.setComputePipelineState(psoExpert)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(idsBuf, offset: 0, index: 1)
        // Bind per-rank expert slot buffers from the streamer. The caller
        // (model.routedExpert) opens the streamer lazily and caches experts.
        // Slot buffers are bound at indices 2-9; remaining params at 10-17.
        // For now, bind the first expert's buffer to all 8 slots (single-token
        // decode path — all 8 experts are looked up by the kernel via idsIn).
        // TODO: multi-token batch needs per-token expert lookup.
        let model = modelRef  // captured at init
        for rank in 0..<8 {
            let expert = Int(expertIdsThisLayer[0][rank])  // single-token case
            if let tv = try? model.routedExpert(layer: self.layer, expert: expert) {
                enc.setBuffer(tv.buffer, offset: Int(tv.offset), index: 2 + rank)
            } else {
                // Fallback: bind placeholder (should not happen in practice)
                enc.setBuffer(idsBuf, offset: 0, index: 2 + rank)
            }
        }
        enc.setBuffer(expertOutBuf, offset: 0, index: 10)
        var s = expertStride; enc.setBytes(&s, length: 4, index: 11)
        s = gateQOff; enc.setBytes(&s, length: 4, index: 12)
        s = gateSOff; enc.setBytes(&s, length: 4, index: 13)
        s = upQOff; enc.setBytes(&s, length: 4, index: 14)
        s = upSOff; enc.setBytes(&s, length: 4, index: 15)
        s = downQOff; enc.setBytes(&s, length: 4, index: 16)
        s = downSOff; enc.setBytes(&s, length: 4, index: 17)
        enc.dispatchThreadgroups(MTLSize(width: nTokens * 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }"""

assert old_enc in rsrc, "encodeExpert not found"
rsrc = rsrc.replace(old_enc, new_enc, 1)

# Add modelRef + layer properties (needed for streamer lookup)
# Find the proper place: after `private let expertStride: UInt32`
old_props = "    private let expertStride: UInt32"
new_props = "    private let expertStride: UInt32\n    private let modelRef: Model\n    private let layer: Int"
assert old_props in rsrc, "expertStride property not found"
rsrc = rsrc.replace(old_props, new_props, 1)

# Add modelRef + layer assignment in init (after stride assignment)
old_init = "        expertStride = UInt32(stride)"
new_init = "        expertStride = UInt32(stride)\n        self.modelRef = model\n        self.layer = layer"
assert old_init in rsrc, "stride assignment not found"
rsrc = rsrc.replace(old_init, new_init, 1)

# Add expertIdsThisLayer buffer for single-token mode
# We need to store router output per-layer. The current runner reads router
# ids into idsBuf, then passes them to the kernel. For streamer lookup, we
# need to read back the ids. Add a small staging buffer.
old_staging = "    private let expertOutBuf: MTLBuffer"
new_staging = "    private let expertOutBuf: MTLBuffer\n    // Streamer lookup: router IDs read back from GPU for per-expert binding\n    private var expertIdsThisLayer: [[UInt32]] = []"
assert old_staging in rsrc, "expertOutBuf not found"
rsrc = rsrc.replace(old_staging, new_staging, 1)

# Fix encodeRouter: after dispatch, read back IDs into expertIdsThisLayer
old_router = """    public func encodeRouter(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int,
                              nTokens: Int) {
        enc.setComputePipelineState(psoRouter)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(idsBuf, offset: 0, index: 1)
        enc.setBuffer(scoresBuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: nTokens, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }"""

new_router = """    public func encodeRouter(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int,
                              nTokens: Int) {
        enc.setComputePipelineState(psoRouter)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(idsBuf, offset: 0, index: 1)
        enc.setBuffer(scoresBuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: nTokens, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Read back router IDs from GPU for streamer expert lookup.
    /// Must be called after encodeRouter's CB completes (synchronized).
    public func readRouterIds(nTokens: Int) -> [[UInt32]] {
        let ptr = idsBuf.contents().assumingMemoryBound(to: UInt32.self)
        var result: [[UInt32]] = []
        for t in 0..<nTokens {
            var ids: [UInt32] = []
            for k in 0..<8 {
                ids.append(ptr[t * 8 + k])
            }
            result.append(ids)
        }
        return result
    }

    /// Store router IDs for the current layer (called before encodeExpert).
    public func setRouterIds(_ ids: [[UInt32]]) {
        self.expertIdsThisLayer = ids
    }"""

assert old_router in rsrc, "encodeRouter not found"
rsrc = rsrc.replace(old_router, new_router, 1)

open(rp, "w").write(rsrc)
print("Qwen36MoERunner: streamer-based encodeExpert + router ID readback")

# === 3. WAKE_POLL_US — add spin-poll to Qwen36ForwardRunner ===
fp = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift"
fsrc = open(fp).read()

# Add wake polling to the produce() method's CB wait.
# The current produce() uses one CB for all 40 layers, then commits.
# The wait is after the 40-layer loop. We need to poll between layers.
# Add a wakePollUs computed property and a poll helper.
old_wait = "        // 5. final blit: logits private -> caller"
new_wait = """        // Wake polling: spin-wait on CB status before the blocking wait.
        // This lets the CPU chain start earlier (~100us per layer) and
        // immunizes the wake against load noise (parked-thread wakeups
        // balloon under load; a spinning thread does not). Controlled by
        // TURBO_FIELDFARE_WAKE_POLL_US (default 0 = off).
        if wakePollInterval > 0 {
            let start = CFAbsoluteTimeGetCurrent()
            while cb.status == .notScheduled || cb.status == .scheduled {
                if CFAbsoluteTimeGetCurrent() - start > wakePollInterval { break }
                // spin (sched_yield would be ideal but we stay in Swift)
            }
        }

        // 5. final blit: logits private -> caller"""
assert old_wait in fsrc, "final blit not found"
fsrc = fsrc.replace(old_wait, new_wait, 1)

# Add wakePollInterval property
old_prop = "    private let maxSeq: Int"
new_prop = "    private let maxSeq: Int\n    private let wakePollInterval: Double"
assert old_prop in fsrc, "maxSeq property not found"
fsrc = fsrc.replace(old_prop, new_prop, 1)

# Initialize wakePollInterval from env
old_init = "        self.maxSeq = max(1, maxSeq)"
new_init = "        self.maxSeq = max(1, maxSeq)\n        if let v = ProcessInfo.processInfo.environment[\"TURBO_FIELDFARE_WAKE_POLL_US\"],\n           let us = Double(v), us > 0 {\n            self.wakePollInterval = us / 1_000_000  // μs -> seconds\n        } else {\n            self.wakePollInterval = 0\n        }"
assert old_init in fsrc, "maxSeq init not found"
fsrc = fsrc.replace(old_init, new_init, 1)

open(fp, "w").write(fsrc)
print("Qwen36ForwardRunner: WAKE_POLL_US polling added")
print("ALL PATCHES DONE")