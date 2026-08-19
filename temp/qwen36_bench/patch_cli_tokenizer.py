#!/usr/bin/env python3
"""Run.swift: load the Qwen36 tokenizer in .qwen36 compatibility mode when the
detected arch is Qwen3.6. The tokenizer load currently happens before arch
detection, so we restructure: detect arch first, then load tokenizer."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfareCLI/Run.swift"
src = open(path).read()

old = """        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)"""
new = """        let modelURL = URL(fileURLWithPath: args.model)
        let detectedArch = detectArch(modelURL)
        let tokenizer: GFTokenizer
        if detectedArch == .qwen36_35B_A3B {
            if let folder = GFTokenizer.tokenizerFolder(forModelDirectory: modelURL) {
                let underlying = try await AutoTokenizer.from(modelFolder: folder)
                tokenizer = try GFTokenizer(tokenizer: underlying, compatibility: .qwen36)
            } else {
                tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
            }
        } else {
            tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        }"""
assert old in src, "tokenizer load block not found"
src = src.replace(old, new)

# remove the later duplicate detection
old2 = """        let detectedArch = detectArch(modelURL)
        let model = try Model.load("""
new2 = """        let model = try Model.load("""
assert old2 in src, "duplicate detectArch not found"
src = src.replace(old2, new2)

open(path, "w").write(src)
print("Run.swift: qwen36 tokenizer compatibility wired")
