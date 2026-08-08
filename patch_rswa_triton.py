#!/usr/bin/env python3
"""Patch sglang triton_backend.py to add R-SWA attention length cap.

Adds env var RSWA_MAX_ATTN_LEN:
  - 0 (default): no change, original behavior
  - 132: cap full-attention decode to reference(4) + window(128) = 132 tokens

Only affects full attention layers (SWA layers already bounded).
Only affects decode path (prefill stores all KV normally).
"""
import sys
import shutil

TARGET = sys.argv[1] if len(sys.argv) > 1 else "/data/venv_gemma4/lib/python3.12/site-packages/sglang/srt/layers/attention/triton_backend.py"

with open(TARGET, "r") as f:
    code = f.read()

# Check if already patched
if "RSWA_MAX_ATTN_LEN" in code:
    print("Already patched, skipping")
    sys.exit(0)

# Patch 1: Add import os at the top (after existing imports)
old_import = "import torch\nimport triton"
new_import = "import os as _rswa_os\nimport torch\nimport triton"
code = code.replace(old_import, new_import, 1)

# Patch 2: In _update_decode_kv_buffers, cap seq_lens
old_line = """        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]"""
new_line = """        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        # R-SWA: cap attention length for full attention layers
        _rswa_max = int(_rswa_os.environ.get("RSWA_MAX_ATTN_LEN", "0"))
        if _rswa_max > 0:
            seq_lens = torch.clamp(seq_lens, max=_rswa_max)"""
code = code.replace(old_line, new_line, 1)

# Patch 3: In _fill_kv_indptr_and_indices, also cap (covers target_verify path)
old_fill = """    def _fill_kv_indptr_and_indices(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        kv_indptr = self.kv_indptr[: bs + 1]"""
new_fill = """    def _fill_kv_indptr_and_indices(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        # R-SWA: cap attention length
        _rswa_max = int(_rswa_os.environ.get("RSWA_MAX_ATTN_LEN", "0"))
        if _rswa_max > 0:
            seq_lens = torch.clamp(seq_lens, max=_rswa_max)
        kv_indptr = self.kv_indptr[: bs + 1]"""
code = code.replace(old_fill, new_fill, 1)

# Patch 4: Add logging in forward_decode for R-SWA activation
old_decode_start = """    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
    ):"""
new_decode_start = """    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
    ):
        _rswa_max = int(_rswa_os.environ.get("RSWA_MAX_ATTN_LEN", "0"))
        if _rswa_max > 0 and not getattr(self, '_rswa_logged', False):
            print(f"[R-SWA] Active: max_attn_len={_rswa_max} (full attention layers capped)", flush=True)
            self._rswa_logged = True"""
code = code.replace(old_decode_start, new_decode_start, 1)

with open(TARGET, "w") as f:
    f.write(code)

print(f"Patched {TARGET}")
print("  - Added RSWA_MAX_ATTN_LEN env var support")
print("  - Cap applied in _update_decode_kv_buffers and _fill_kv_indptr_and_indices")
print("  - Set RSWA_MAX_ATTN_LEN=132 to enable, 0 or unset to disable")
