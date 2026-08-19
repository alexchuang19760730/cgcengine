#!/usr/bin/env python3
"""Qwen3.6 uses <|endoftext|> (id 248044) as pad, not <pad>. Fix the qwen36
branch of the GFTokenizer init to look up <|endoftext|> first."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Tokenization/Tokenizer.swift"
src = open(path).read()

old = """        let pad: Int32?
        if let p = tokenizer.convertTokenToId("<pad>") {
            pad = Int32(p)
        } else if compatibility == .qwen36 {
            pad = Int32(eos)
        } else {
            throw GFTokenizerError.missingSpecialToken("<pad>")
        }"""
new = """        let pad: Int32?
        if let p = tokenizer.convertTokenToId("<pad>") {
            pad = Int32(p)
        } else if compatibility == .qwen36 {
            // Qwen3.6 has no <pad>; pad == <|endoftext|> == eos in this vocab.
            pad = Int32(eos)
        } else {
            throw GFTokenizerError.missingSpecialToken("<pad>")
        }"""
assert old in src, "pad block not found"
src = src.replace(old, new)
open(path, "w").write(src)
print("Tokenizer.swift: qwen36 pad = eos (<|endoftext|>)")
