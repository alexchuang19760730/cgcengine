#!/usr/bin/env python3
"""Fix three review findings in Qwen36ForwardRunner:
1. embed must use the qwen36 resident name (model.language_model.embed_tokens.weight)
   — model.embedding resolves the GEMMA4 name and would crash (try!).
2. makeCommandBuffer nil should throw, not silently return.
3. (vocabSize tokenizer fix is in Tokenizer.swift separately)
"""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift"
src = open(path).read()

# 1. embed resident name
old1 = """            let emb = model.embedding  // qwen36: model.language_model.embed_tokens.weight
            enc.setBuffer(emb.buffer, offset: Int(emb.offset), index: 0)"""
new1 = """            let emb = try model.resident(name: "model.language_model.embed_tokens.weight")
            enc.setBuffer(emb.buffer, offset: Int(emb.offset), index: 0)"""
assert old1 in src, "embed block not found"
src = src.replace(old1, new1)

# 2. throw on nil CB
old2 = """        let queue = context.queue
        guard let cb = queue.makeCommandBuffer() else { return }"""
new2 = """        let queue = context.queue
        guard let cb = queue.makeCommandBuffer() else {
            throw PrefillError.chunkedUnsupported("qwen36: failed to allocate command buffer")
        }"""
assert old2 in src, "cb guard not found"
src = src.replace(old2, new2)

open(path, "w").write(src)
print("Qwen36ForwardRunner: embed resident name + throw on nil CB")
