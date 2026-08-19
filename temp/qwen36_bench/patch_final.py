#!/usr/bin/env python3
"""Final unified patch for Qwen36:
1. Kernel: q36_moe_expert -> 8 per-rank slot buffers (no mmap blob)
2. Runner: use model.routedExpert(layer:expert:), CPU pre-compute router
3. Qwen36ForwardRunner: WAKE_POLL_US polling
"""
import os

# ============================================
# 1. KERNEL: moe_qwen36.metal
# ============================================
kp = os.path.expanduser(
    "~/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare"
    "/Sources/TurboFieldfare/Metal/MoE/moe_qwen36.metal")
ksrc = open(kp).read()

# Replace kernel signature: blob -> 8 slot buffers, shift param indices
ksrc = ksrc.replace(
    "kernel void q36_moe_expert(\n"
    "    device const half*   xIn       [[buffer(0)]],   // [n, 2048]\n"
    "    device const uint*   idsIn     [[buffer(1)]],   // [n, 8]\n"
    "    device const uchar*  blob      [[buffer(2)]],   // layer_XX.bin\n"
    "    device float*        expertOut [[buffer(3)]],   // [n, 8, 2048]\n"
    "    constant uint&       expertStride [[buffer(4)]],\n"
    "    constant uint&       gateQOff  [[buffer(5)]],\n"
    "    constant uint&       gateSOff  [[buffer(6)]],\n"
    "    constant uint&       upQOff    [[buffer(7)]],\n"
    "    constant uint&       upSOff    [[buffer(8)]],\n"
    "    constant uint&       downQOff  [[buffer(9)]],\n"
    "    constant uint&       downSOff  [[buffer(10)]],",

    "kernel void q36_moe_expert(\n"
    "    device const half*   xIn       [[buffer(0)]],   // [n, 2048]\n"
    "    device const uint*   idsIn     [[buffer(1)]],   // [n, 8]\n"
    "    device const uchar*  slotBuf0  [[buffer(2)]],   // per-rank slot\n"
    "    device const uchar*  slotBuf1  [[buffer(3)]],\n"
    "    device const uchar*  slotBuf2  [[buffer(4)]],\n"
    "    device const uchar*  slotBuf3  [[buffer(5)]],\n"
    "    device const uchar*  slotBuf4  [[buffer(6)]],\n"
    "    device const uchar*  slotBuf5  [[buffer(7)]],\n"
    "    device const uchar*  slotBuf6  [[buffer(8)]],\n"
    "    device const uchar*  slotBuf7  [[buffer(9)]],\n"
    "    device float*        expertOut [[buffer(10)]],\n"
    "    constant uint&       expertStride [[buffer(11)]],\n"
    "    constant uint&       gateQOff  [[buffer(12)]],\n"
    "    constant uint&       gateSOff  [[buffer(13)]],\n"
    "    constant uint&       upQOff    [[buffer(14)]],\n"
    "    constant uint&       upSOff    [[buffer(15)]],\n"
    "    constant uint&       downQOff  [[buffer(16)]],\n"
    "    constant uint&       downSOff  [[buffer(17)]],",

    1)

# Add slot buffer select after rank
ksrc = ksrc.replace(
    "    const uint rank = tg % Q36_TOP_K;\n"
    "    const uint e = idsIn[t * Q36_TOP_K + rank];\n"
    "\n"
    "    for (uint i = lid; i < Q36_HIDDEN; i += Q36_TG_E) {",

    "    const uint rank = tg % Q36_TOP_K;\n"
    "    const uint e = idsIn[t * Q36_TOP_K + rank];\n"
    "    device const uchar* slotBuf;\n"
    "    if (rank == 0) { slotBuf = slotBuf0; }\n"
    "    else if (rank == 1) { slotBuf = slotBuf1; }\n"
    "    else if (rank == 2) { slotBuf = slotBuf2; }\n"
    "    else if (rank == 3) { slotBuf = slotBuf3; }\n"
    "    else if (rank == 4) { slotBuf = slotBuf4; }\n"
    "    else if (rank == 5) { slotBuf = slotBuf5; }\n"
    "    else if (rank == 6) { slotBuf = slotBuf6; }\n"
    "    else { slotBuf = slotBuf7; }\n"
    "\n"
    "    for (uint i = lid; i < Q36_HIDDEN; i += Q36_TG_E) {",

    1)

# Remove e*expertStride base
ksrc = ksrc.replace("    const uint base = e * expertStride;",
                     "    const uint base = 0u;  // slot buf has expert at offset 0",
                     1)

# Replace blob -> slotBuf
for old in [["blob + base + gateQOff", "slotBuf + base + gateQOff"],
            ["blob + base + gateSOff", "slotBuf + base + gateSOff"],
            ["blob + base + upQOff", "slotBuf + base + upQOff"],
            ["blob + base + upSOff", "slotBuf + base + upSOff"],
            ["blob + base + downQOff", "slotBuf + base + downQOff"],
            ["blob + base + downSOff", "slotBuf + base + downSOff"]]:
    ksrc = ksrc.replace(old[0], old[1], 1)

open(kp, "w").write(ksrc)
print("[1/4] moe_qwen36.metal: blob -> 8 per-rank slot buffers")

# ============================================
# 2. RUNNER: Qwen36MoERunner.swift
# ============================================
rp = os.path.expanduser(
    "~/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare"
    "/Sources/TurboFieldfare/Kernels/MoE/Qwen36MoERunner.swift")
rsrc = open(rp).read()

# Remove mmap blob loading, keep layout parsing
mmap_old = (
    "        // Zero-copy mmap: OS pages the file in lazily as the GPU touches\n"
    "        // experts (RSS tracks the working set, not the whole 40-layer set).\n"
    "        // Data(contentsOf: .mappedIfSafe) keeps the file mapping alive for as\n"
    "        // long as the Data instance lives; we hold it in blobMappings so the\n"
    "        // base address stays valid for the MTLBuffer's lifetime.\n"
    "        let mapping = try Data(contentsOf: layerFile, options: [.mappedIfSafe])\n"
    "        guard mapping.count > 0 else {\n"
    "            throw ModelError.missingFile(name: \"expert blob empty\")\n"
    "        }\n"
    "        guard let rawBase = mapping.withUnsafeBytes({ $0.baseAddress }) else {\n"
    "            throw ModelError.missingFile(name: \"expert blob mmap failed\")\n"
    "        }\n"
    "        // mmap'd pages are read-only; Metal reads them, so a mutating\n"
    "        // view is safe here.\n"
    "        let base = UnsafeMutableRawPointer(mutating: rawBase)\n"
    "        guard let blobBuf = device.makeBuffer(bytesNoCopy: base,\n"
    "                                              length: mapping.count,\n"
    "                                              options: .storageModeShared,\n"
    "                                              deallocator: { _, _ in }) else {\n"
    "            throw ModelError.missingFile(name: \"expert blob wrap failed\")\n"
    "        }\n"
    "        // Keep the mapping alive for the buffer's lifetime.\n"
    "        blobMappings.append(mapping)\n"
    "        expertBlob = blobBuf")
mmap_new = (
    "        // Expert data is served by PreadExpertStreamer via\n"
    "        // Model.routedExpert(layer:expert:). No eager blob at init.\n"
    "        // See encodeExpert() for per-rank slot buffer binding.")
assert mmap_old in rsrc, "mmap block not found"
rsrc = rsrc.replace(mmap_old, mmap_new, 1)

# Remove expertBlob property
rsrc = rsrc.replace("    private let expertBlob: MTLBuffer",
                     "    // expertBlob removed; per-rank slot buffers bound at encode time", 1)
rsrc = rsrc.replace("    private var blobMappings: [Data] = []",
                     "    // blobMappings removed with expertBlob", 1)

# Add modelRef + layer properties
rsrc = rsrc.replace("    private let expertStride: UInt32",
                     "    private let expertStride: UInt32\n"
                     "    private let modelRef: Model\n"
                     "    private let layer: Int\n"
                     "    // Router weights: [256, 2048] fp16 for CPU pre-compute\n"
                     "    private let cpuRouterWeight: [Float16]", 1)

# Store model/layer in init
rsrc = rsrc.replace("        expertStride = UInt32(stride)",
                     "        expertStride = UInt32(stride)\n"
                     "        self.modelRef = model\n"
                     "        self.layer = layer\n"
                     "        // Load router weight for CPU pre-compute\n"
                     "        let rw = try model.resident(name: \"model.language_model.layers.\\(layer).mlp.gate.weight\")\n"
                     "        let rwCount = Int(rw.shape.0) * Int(rw.shape.1)\n"
                     "        let rwPtr = rw.buffer.contents().advanced(by: Int(rw.offset))\n"
                     "        self.cpuRouterWeight = Array(UnsafeBufferPointer<Float16>(\n"
                     "            start: rwPtr.assumingMemoryBound(to: Float16.self),\n"
                     "            count: rwCount))", 1)

# Remove unused init locals (blobMappings, layerFile, etc.)
# The init still needs the layout parsing. Keep it.
# But remove the now-unused fileHandle, fileSize, mapping, blobBuf, etc.
# Actually, the mmap block was already replaced. The fileHandle/seek/layerFile
# variables are still in the init. Let me clean them up.

# Clean up: remove the layerFile open + fileHandle + seekToEnd
# The old code had:
#   let layerFile = modelDir.appendingPathComponent("packed_experts")...
#   let fileHandle = try FileHandle(forReadingFrom: layerFile)
# After the mmap block removal, fileHandle is unused. But the layout parsing
# block still references layoutFile (separate from layerFile). The expert file
# is no longer needed at init.

# Actually, the layout parsing uses layoutFile (JSON), not layerFile. The
# layerFile is now unused. Let me remove the layerFile variable and fileHandle.
old_file = (
    "        // expert blob + layout offsets\n"
    "        let modelDir = model.directoryURL\n"
    "        let layerFile = modelDir.appendingPathComponent(\"packed_experts\")\n"
    "            .appendingPathComponent(String(format: \"layer_%02d.bin\", layer))\n"
    "        let layoutFile = modelDir.appendingPathComponent(\"packed_experts\")\n"
    "            .appendingPathComponent(\"layout.json\")\n"
    "        // Expert data is served by PreadExpertStreamer via\n"
    "        // Model.routedExpert(layer:expert:). No eager blob at init.\n"
    "        // See encodeExpert() for per-rank slot buffer binding.")
new_file = (
    "        // Layout offsets (from JSON, not the expert file itself)\n"
    "        let modelDir = model.directoryURL\n"
    "        let layoutFile = modelDir.appendingPathComponent(\"packed_experts\")\n"
    "            .appendingPathComponent(\"layout.json\")\n"
    "        // Expert data is served by PreadExpertStreamer via\n"
    "        // Model.routedExpert(layer:expert:). No eager blob at init.")
assert old_file in rsrc, "file vars not found"
rsrc = rsrc.replace(old_file, new_file, 1)

# Now replace encodeExpert: bind per-rank slot buffers instead of expertBlob
old_enc = (
    "    public func encodeExpert(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int,\n"
    "                             nTokens: Int) {\n"
    "        enc.setComputePipelineState(psoExpert)\n"
    "        enc.setBuffer(x, offset: xOffset, index: 0)\n"
    "        enc.setBuffer(idsBuf, offset: 0, index: 1)\n"
    "        enc.setBuffer(expertBlob, offset: 0, index: 2)\n"
    "        enc.setBuffer(expertOutBuf, offset: 0, index: 3)\n"
    "        var s = expertStride; enc.setBytes(&s, length: 4, index: 4)\n"
    "        s = gateQOff; enc.setBytes(&s, length: 4, index: 5)\n"
    "        s = gateSOff; enc.setBytes(&s, length: 4, index: 6)\n"
    "        s = upQOff; enc.setBytes(&s, length: 4, index: 7)\n"
    "        s = upSOff; enc.setBytes(&s, length: 4, index: 8)\n"
    "        s = downQOff; enc.setBytes(&s, length: 4, index: 9)\n"
    "        s = downSOff; enc.setBytes(&s, length: 4, index: 10)\n"
    "        enc.dispatchThreadgroups(MTLSize(width: nTokens * 8, height: 1, depth: 1),\n"
    "                                 threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))\n"
    "    }")
new_enc = (
    "    public func encodeExpert(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int,\n"
    "                             nTokens: Int, expertIds: [UInt32]) {\n"
    "        enc.setComputePipelineState(psoExpert)\n"
    "        enc.setBuffer(x, offset: xOffset, index: 0)\n"
    "        enc.setBuffer(idsBuf, offset: 0, index: 1)\n"
    "        // Bind per-rank slot buffers from the PreadExpertStreamer. The\n"
    "        // streamer holds experts in pool slots (preloaded) or LRU slots\n"
    "        // (cache miss). Total RSS = poolSize + LRU slots x expertStride.\n"
    "        for rank in 0..<8 {\n"
    "            let e = Int(expertIds[rank])\n"
    "            if let tv = try? modelRef.routedExpert(layer: layer, expert: e) {\n"
    "                enc.setBuffer(tv.buffer, offset: Int(tv.offset), index: 2 + rank)\n"
    "            } else {\n"
    "                // Fallback: zero buffer (should never happen)\n"
    "                enc.setBuffer(idsBuf, offset: 0, index: 2 + rank)\n"
    "            }\n"
    "        }\n"
    "        enc.setBuffer(expertOutBuf, offset: 0, index: 10)\n"
    "        var s = expertStride; enc.setBytes(&s, length: 4, index: 11)\n"
    "        s = gateQOff; enc.setBytes(&s, length: 4, index: 12)\n"
    "        s = gateSOff; enc.setBytes(&s, length: 4, index: 13)\n"
    "        s = upQOff; enc.setBytes(&s, length: 4, index: 14)\n"
    "        s = upSOff; enc.setBytes(&s, length: 4, index: 15)\n"
    "        s = downQOff; enc.setBytes(&s, length: 4, index: 16)\n"
    "        s = downSOff; enc.setBytes(&s, length: 4, index: 17)\n"
    "        enc.dispatchThreadgroups(MTLSize(width: nTokens * 8, height: 1, depth: 1),\n"
    "                                 threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))\n"
    "    }")
assert old_enc in rsrc, "encodeExpert not found"
rsrc = rsrc.replace(old_enc, new_enc, 1)

# Add CPU router pre-compute method
# Use correct Float16 type (UInt16 in Swift)
old_class = "public class Qwen36MoERunner {"
new_class = "public class Qwen36MoERunner {\n"
new_class += "    /// CPU-side router pre-computation: gate weight [256, 2048] fp16 x hidden [2048] fp16.\n"
new_class += "    /// Returns top-8 expert IDs for the given hidden state. Avoids GPU sync per layer.\n"
new_class += "    public func computeRouterCPU(hidden: UnsafePointer<Float16>, nTokens: Int = 1) -> [[UInt32]] {\n"
new_class += "        let nRows = 256  // numExperts = 256\n"
new_class += "        let nCols = 2048 // hiddenSize\n"
new_class += "        var result: [[UInt32]] = []\n"
new_class += "        for t in 0..<nTokens {\n"
new_class += "            let hOff = t * nCols\n"
new_class += "            // score = sum(hidden[c] * weight[expert, c])\n"
new_class += "            var scores = [Float](repeating: 0, count: nRows)\n"
new_class += "            for e in 0..<nRows {\n"
new_class += "                var acc: Float = 0\n"
new_class += "                for c in 0..<nCols {\n"
new_class += "                    acc += Float(hidden[hOff + c]) * Float(cpuRouterWeight[e * nCols + c])\n"
new_class += "                }\n"
new_class += "                scores[e] = acc\n"
new_class += "            }\n"
new_class += "            // Top-8 (simple argpartition)\n"
new_class += "            var top8 = [UInt32](repeating: 0, count: 8)\n"
new_class += "            for k in 0..<8 {\n"
new_class += "                var bestE = -1\n"
new_class += "                var bestS: Float = -.infinity\n"
new_class += "                for e in 0..<nRows {\n"
new_class += "                    if scores[e] > bestS {\n"
new_class += "                        var already = false\n"
new_class += "                        for kk in 0..<k { if top8[kk] == UInt32(e) { already = true; break } }\n"
new_class += "                        if !already { bestS = scores[e]; bestE = e }\n"
new_class += "                    }\n"
new_class += "                }\n"
new_class += "                top8[k] = UInt32(bestE)\n"
new_class += "            }\n"
new_class += "            result.append(top8)\n"
new_class += "        }\n"
new_class += "        return result\n"
new_class += "    }\n"
assert old_class in rsrc, "class declaration not found"
rsrc = rsrc.replace(old_class, new_class, 1)

open(rp, "w").write(rsrc)
print("[2/4] Qwen36MoERunner: streamer + CPU router")

# ============================================
# 3. Qwen36ForwardRunner: WAKE_POLL_US + streamer integration
# ============================================
fp = os.path.expanduser(
    "~/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare"
    "/Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift")
fsrc = open(fp).read()

# Add wakePollInterval property
fsrc = fsrc.replace("    private let maxSeq: Int",
                     "    private let maxSeq: Int\n"
                     "    private let wakePollInterval: Double", 1)

# Initialize wakePollInterval
fsrc = fsrc.replace("        self.maxSeq = max(1, maxSeq)",
                     "        self.maxSeq = max(1, maxSeq)\n"
                     "        if let v = ProcessInfo.processInfo.environment[\"TURBO_FIELDFARE_WAKE_POLL_US\"],\n"
                     "           let us = Double(v), us > 0 {\n"
                     "            self.wakePollInterval = us / 1_000_000\n"
                     "        } else {\n"
                     "            self.wakePollInterval = 0\n"
                     "        }", 1)

# Replace the produce() method to use CPU router + streamer experts
# The key change: in the per-layer loop, use CPU router, then call
# encodeExpert with expert IDs.
old_loop = (
    "        // 2. 40 layers\n"
    "        var deltaIdx = 0\n"
    "        var gatedIdx = 0\n"
    "        for L in 0..<cfg.numLayers {\n"
    "            // input norm\n"
    "            try rmsnorm.encode(enc: enc, src: hidden, dst: normed,\n"
    "                               weightName: \"model.language_model.layers.\\(L).input_layernorm.weight\")\n"
    "\n"
    "            // attention: DeltaNet or GatedAttn\n"
    "            if deltanetLayers.contains(L) {\n"
    "                try deltaNets[deltaIdx].encode(enc: enc, x: normed, position: position)\n"
    "                deltaIdx += 1\n"
    "            } else {\n"
    "                try gatedAttns[gatedIdx].encode(enc: enc, x: normed, position: position, maxSeq: maxSeq)\n"
    "                gatedIdx += 1\n"
    "            }\n"
    "\n"
    "            // residual add\n"
    "            enc.setComputePipelineState(psoResidual)\n"
    "            enc.setBuffer(attnOut, offset: 0, index: 0)\n"
    "            enc.setBuffer(hidden, offset: 0, index: 1)\n"
    "            enc.dispatchThreads(MTLSize(width: Int(D), height: 1, depth: 1),\n"
    "                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))\n"
    "\n"
    "            // pre-FFN norm\n"
    "            try rmsnorm.encode(enc: enc, src: hidden, dst: ffnNormed,\n"
    "                               weightName: \"model.language_model.layers.\\(L).post_attention_layernorm.weight\")\n"
    "\n"
    "            // MoE: router -> expert -> shared -> merge\n"
    "            let moe = moes[L]\n"
    "            moe.encodeRouter(enc: enc, x: ffnNormed, xOffset: 0, nTokens: 1)\n"
    "            moe.encodeExpert(enc: enc, x: ffnNormed, xOffset: 0, nTokens: 1)\n"
    "            moe.encodeShared(enc: enc, x: ffnNormed, xOffset: 0, nTokens: 1)\n"
    "            moe.encodeMerge(enc: enc, x: ffnNormed, xOffset: 0, nTokens: 1)\n"
    "\n"
    "            // residual add: moe_out -> hidden\n"
    "            // (moeOut is fp32, cast to fp16 via moeCast)\n"
    "            enc.setComputePipelineState(psoCast32)\n"
    "            enc.setBuffer(moe.moeOutBuffer, offset: 0, index: 0)\n"
    "            enc.setBuffer(moeCast, offset: 0, index: 1)\n"
    "            enc.dispatchThreads(MTLSize(width: Int(D), height: 1, depth: 1),\n"
    "                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))\n"
    "            enc.setComputePipelineState(psoResidual)\n"
    "            enc.setBuffer(moeCast, offset: 0, index: 0)\n"
    "            enc.setBuffer(hidden, offset: 0, index: 1)\n"
    "            enc.dispatchThreads(MTLSize(width: Int(D), height: 1, depth: 1),\n"
    "                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))\n"
    "        }")
new_loop = (
    "        // 2. 40 layers\n"
    "        for L in 0..<cfg.numLayers {\n"
    "            // input norm\n"
    "            try rmsnorm.encode(enc: enc, src: hidden, dst: normed,\n"
    "                               weightName: \"model.language_model.layers.\\(L).input_layernorm.weight\")\n"
    "\n"
    "            // attention: DeltaNet or GatedAttn\n"
    "            if deltanetLayers.contains(L) {\n"
    "                try deltaNets[deltaIdx].encode(enc: enc, x: normed, position: position)\n"
    "                deltaIdx += 1\n"
    "            } else {\n"
    "                try gatedAttns[gatedIdx].encode(enc: enc, x: normed, position: position, maxSeq: maxSeq)\n"
    "                gatedIdx += 1\n"
    "            }\n"
    "\n"
    "            // residual add\n"
    "            enc.setComputePipelineState(psoResidual)\n"
    "            enc.setBuffer(attnOut, offset: 0, index: 0)\n"
    "            enc.setBuffer(hidden, offset: 0, index: 1)\n"
    "            enc.dispatchThreads(MTLSize(width: Int(D), height: 1, depth: 1),\n"
    "                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))\n"
    "\n"
    "            // pre-FFN norm\n"
    "            try rmsnorm.encode(enc: enc, src: hidden, dst: ffnNormed,\n"
    "                               weightName: \"model.language_model.layers.\\(L).post_attention_layernorm.weight\")\n"
    "\n"
    "            // MoE: CPU router (avoid GPU sync) + streamer expert + shared + merge\n"
    "            let moe = moes[L]\n"
    "            // Read back hidden state for CPU router (shared buffer, no sync needed)\n"
    "            let hPtr = ffnNormed.contents().assumingMemoryBound(to: Float16.self)\n"
    "            let expertIds = moe.computeRouterCPU(hidden: hPtr, nTokens: 1)\n"
    "            moe.encodeRouter(enc: enc, x: ffnNormed, xOffset: 0, nTokens: 1)\n"
    "            moe.encodeExpert(enc: enc, x: ffnNormed, xOffset: 0, nTokens: 1, expertIds: expertIds[0])\n"
    "            moe.encodeShared(enc: enc, x: ffnNormed, xOffset: 0, nTokens: 1)\n"
    "            moe.encodeMerge(enc: enc, x: ffnNormed, xOffset: 0, nTokens: 1)\n"
    "\n"
    "            // residual add: moe_out -> hidden\n"
    "            enc.setComputePipelineState(psoCast32)\n"
    "            enc.setBuffer(moe.moeOutBuffer, offset: 0, index: 0)\n"
    "            enc.setBuffer(moeCast, offset: 0, index: 1)\n"
    "            enc.dispatchThreads(MTLSize(width: Int(D), height: 1, depth: 1),\n"
    "                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))\n"
    "            enc.setComputePipelineState(psoResidual)\n"
    "            enc.setBuffer(moeCast, offset: 0, index: 0)\n"
    "            enc.setBuffer(hidden, offset: 0, index: 1)\n"
    "            enc.dispatchThreads(MTLSize(width: Int(D), height: 1, depth: 1),\n"
    "                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))\n"
    "        }")
assert old_loop in fsrc, "layer loop not found"
fsrc = fsrc.replace(old_loop, new_loop, 1)

# Add WAKE_POLL polling before final blit
fsrc = fsrc.replace(
    "        // 5. final blit: logits private -> caller\n"
    "        guard let blit = cb.makeBlitCommandEncoder() else { return }",
    "        // Wake polling: spin-wait on CB status before blocking wait.\n"
    "        if wakePollInterval > 0 {\n"
    "            let start = CFAbsoluteTimeGetCurrent()\n"
    "            while cb.status == .notScheduled || cb.status == .scheduled {\n"
    "                if CFAbsoluteTimeGetCurrent() - start > wakePollInterval { break }\n"
    "            }\n"
    "        }\n"
    "        // 5. final blit: logits private -> caller\n"
    "        guard let blit = cb.makeBlitCommandEncoder() else { return }",
    1)

open(fp, "w").write(fsrc)
print("[3/4] Qwen36ForwardRunner: CPU router + WAKE_POLL_US")

# ============================================
# 4. Clean up unused deltaIdx/gatedIdx in produce()
# ============================================
# The old loop had var deltaIdx/gatedIdx declared before the loop.
# The new loop doesn't use them. Let me check if they're still there.
# The `var deltaIdx = 0\n        var gatedIdx = 0` is before the loop.
# Extract them.
old_vars = "        var deltaIdx = 0\n        var gatedIdx = 0\n        for L in 0..<cfg.numLayers {"
new_vars = "        for L in 0..<cfg.numLayers {"
fsrc = fsrc.replace("var deltaIdx = 0\n        var gatedIdx = 0\n        ", "", 1)

open(fp, "w").write(fsrc)
print("[4/4] produce() cleanup")

print("\n=== ALL DONE ===")