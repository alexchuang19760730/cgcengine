#!/usr/bin/env python3
"""Restore the qwen36 tokenizer branch in Run.swift: declare detectedArch at
the top of run() and use loadQwen36 when arch is Qwen3.6."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfareCLI/Run.swift"
src = open(path).read()

old = """        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)"""
new = """        let modelURL = URL(fileURLWithPath: args.model)
        let detectedArch = detectArch(modelURL)
        let tokenizer: GFTokenizer
        if detectedArch == .qwen36_35B_A3B,
           let folder = GFTokenizer.tokenizerFolder(forModelDirectory: modelURL) {
            tokenizer = try await GFTokenizer.loadQwen36(from: folder)
        } else {
            tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        }"""
assert old in src, "tokenizer load anchor not found"
src = src.replace(old, new)
open(path, "w").write(src)
print("Run.swift: detectedArch + loadQwen36 branch restored at top of run()")
