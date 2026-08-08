#!/usr/bin/env python3
"""Make deepseek_v4.py capture-safe: gate the per-forward [CGC_DBG] print
behind CGC_DBG env var (default off) AND shape-ify the tensor fields so even
when enabled it never triggers a device->host sync during stream capture.

Root cause of "cudaErrorStreamCaptureUnsupported" on the CGC fork of sglang
0.5.13: the [CGC_DBG] print formats forward_batch.seq_lens / extend_seq_lens
(CUDA tensors) every forward, including during cuda-graph capture warmup.
Formatting a tensor calls .item() -> host sync -> illegal inside capture.

Usage: python3 patch_dsv4_capture_safe.py <path/to/deepseek_v4.py>
"""
import shutil, sys, os

path = sys.argv[1]
shutil.copy2(path, path + ".bak_capture_safe")
s = open(path).read()
orig = s

repls = []

# 1) Gate the [CGC_DBG] print behind CGC_DBG (default off) so it does NOT run
#    during graph-capture warmup. One-line form keeps existing arg indentation.
old1 = '            print(\n                f"[CGC_DBG] sc_all={_sc_all} emit_cut={_emit_cut} "'
new1 = '            if os.environ.get("CGC_DBG"): print(\n                f"[CGC_DBG] sc_all={_sc_all} emit_cut={_emit_cut} "'
assert old1 in s, "anchor1 [CGC_DBG] print not found"
s = s.replace(old1, new1, 1)
repls.append("gated [CGC_DBG] print behind CGC_DBG")

# 2) Shape-ify extend_seq_lens (tensor -> shape tuple, no sync)
old2 = "f\"extend_seq_lens={getattr(forward_batch, 'extend_seq_lens', None)} \""
new2 = "f\"extend_seq_lens.shape={tuple(getattr(forward_batch, 'extend_seq_lens', None).shape) if getattr(forward_batch, 'extend_seq_lens', None) is not None else None} \""
if old2 in s:
    s = s.replace(old2, new2, 1)
    repls.append("shape-ified extend_seq_lens")

# 3) Shape-ify seq_lens (tensor -> shape tuple, no sync)
old3 = "f\"seq_lens={getattr(forward_batch, 'seq_lens', None)}\","
new3 = "f\"seq_lens.shape={tuple(getattr(forward_batch, 'seq_lens', None).shape) if getattr(forward_batch, 'seq_lens', None) is not None else None}\","
if old3 in s:
    s = s.replace(old3, new3, 1)
    repls.append("shape-ified seq_lens")

if s == orig:
    print("NO_CHANGE (already patched?)")
    sys.exit(0)

open(path, "w").write(s)
# syntax check
import py_compile
py_compile.compile(path, doraise=True)
print("PATCHED", path)
for r in repls:
    print("  -", r)
print("backup:", path + ".bak_capture_safe")
