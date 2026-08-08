#!/usr/bin/env python3
"""Gate the capture-unsafe debug instrumentation in deepseek_v4.py.

Root cause of cuda-graph capture crash on the CGC fork of sglang 0.5.13:
the `_log_deepseek_v4_layer_trace("input_ids_global_ready", ...)` call passes
device tensors through .item() / .cpu().tolist() as kwargs. Python evaluates
those args BEFORE the call, so each does a device->host sync -> illegal inside
stream capture -> cudaErrorStreamCaptureUnsupported.

Fix:
  1) Make _log_deepseek_v4_layer_trace a no-op unless CGC_TRACE is set
     (default off) -> avoids building 27 trace dicts per forward.
  2) Gate the one call site whose args do host syncs (input_ids_global_ready)
     behind `if os.environ.get("CGC_TRACE"):` so its .item()/.cpu() args are
     NOT evaluated during graph-capture warmup.

Other sync sites are already guarded (_FORK_DUMP_HS / _sc_all) or are CPU
tensors (seq_lens_cpu.tolist) and need no change.

Usage: python3 patch_dsv4_trace_gate.py <path/to/deepseek_v4.py>
"""
import shutil, sys, py_compile

path = sys.argv[1]
shutil.copy2(path, path + ".bak_trace_gate")
s = open(path).read()

# 1) Gate the trace function body behind CGC_TRACE (default off)
old_fn = ('def _log_deepseek_v4_layer_trace(event: str, **fields: object) -> None:\n'
          '    parts = [f"{key}={value!r}" for key, value in fields.items()]')
new_fn = ('def _log_deepseek_v4_layer_trace(event: str, **fields: object) -> None:\n'
          '    if not os.environ.get("CGC_TRACE"):\n'
          '        return\n'
          '    parts = [f"{key}={value!r}" for key, value in fields.items()]')
assert old_fn in s, "fn anchor not found"
s = s.replace(old_fn, new_fn, 1)

# 2) Gate the input_ids_global_ready call site (the one with .item()/.cpu() args)
old_call = '        _log_deepseek_v4_layer_trace(\n            "input_ids_global_ready",'
new_call = '        if os.environ.get("CGC_TRACE"): _log_deepseek_v4_layer_trace(\n            "input_ids_global_ready",'
assert old_call in s, "call anchor not found"
s = s.replace(old_call, new_call, 1)

open(path, "w").write(s)
py_compile.compile(path, doraise=True)
print("PATCHED trace gates:", path)
print("backup:", path + ".bak_trace_gate")
