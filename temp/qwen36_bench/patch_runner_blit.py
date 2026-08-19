#!/usr/bin/env python3
"""Fix the produce() tail in Qwen36ForwardRunner: blit self.logits to the
caller's `logits` parameter in the SAME command buffer, drop the redundant
second CB."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift"
src = open(path).read()

old = """        // 5. final blit: private logits -> caller buffer
        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: logits, sourceOffset: 0,
                      to: logits, destinationOffset: 0,
                      size: cfg.vocabSize * MemoryLayout<Float16>.size)
            blit.endEncoding()
        }

        cb.commit()
        cb.waitUntilCompleted()

        // copy results to caller's buffer (blit into the same CB above is not
        // possible after endEncoding of the last compute encoder, so do a
        // separate blit CB here)
        guard let cb2 = queue.makeCommandBuffer() else { return }
        if let blit = cb2.makeBlitCommandEncoder() {
            blit.copy(from: logits, sourceOffset: 0,
                      to: logits, destinationOffset: 0,
                      size: cfg.vocabSize * MemoryLayout<Float16>.size)
            blit.endEncoding()
        }
        cb2.commit()
        cb2.waitUntilCompleted()
    }"""

new = """        // 5. final blit: private logits -> caller buffer (same CB; a blit
        //    encoder may follow compute encoders). `logits` here is the caller
        //    parameter; the stored property is self.logits.
        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: self.logits, sourceOffset: 0,
                      to: logits, destinationOffset: 0,
                      size: cfg.vocabSize * MemoryLayout<Float16>.size)
            blit.endEncoding()
        }

        cb.commit()
        cb.waitUntilCompleted()
    }"""

assert old in src, "produce tail (old form) not found"
src = src.replace(old, new)
open(path, "w").write(src)
print("produce tail fixed: self.logits -> caller logits, single CB")
