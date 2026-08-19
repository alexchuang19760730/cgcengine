#!/usr/bin/env python3
"""Fix the over-escaped footer string in the Qwen36 branch of Run.swift:
the Python patch writer emitted \\" (literal backslash-quote) inside the Swift
string; Swift needs plain " for String(format:) calls."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfareCLI/Run.swift"
src = open(path).read()

line_old = 'let footer = "\\n[stop=\\(String(describing: qStats.reason)) prefill=\\(qStats.prefillTokens)tok new=\\(qStats.newTokens)tok ttft=\\(String(format: \\"%.2f\\", qStats.prefillSeconds))s decode=\\(String(format: \\"%.2f\\", qStats.decodeSeconds))s tok/s=\\(String(format: \\"%.3f\\", tps))]\\n"'

line_new = 'let footer = "\\n[stop=\\(String(describing: qStats.reason)) prefill=\\(qStats.prefillTokens)tok new=\\(qStats.newTokens)tok ttft=\\(String(format: "%.2f", qStats.prefillSeconds))s decode=\\(String(format: "%.2f", qStats.decodeSeconds))s tok/s=\\(String(format: "%.3f", tps))]\\n"'

if line_old in src:
    src = src.replace(line_old, line_new)
    print("footer fixed")
else:
    # fallback: replace the escaped quote sequences that are NOT part of \\( or \\n
    import re
    # find the footer line and fix just the \\" -> "
    for m in re.finditer(r'.*let footer.*', src):
        frag = m.group(0)
        fixed = frag.replace('\\"', '"')
        src = src.replace(frag, fixed)
        print("fallback: fixed footer line")

open(path, "w").write(src)
