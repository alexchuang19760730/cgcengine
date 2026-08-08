#!/usr/bin/env python3
"""R-SWA monkey-patch for sglang triton backend.

This script patches TritonAttnBackend at import time.
No sglang source files are modified.

Usage:
  Set env RSWA_MAX_ATTN_LEN=132, then:
  PYTHONPATH=/path/to/this/script python3 -m sglang.launch_server ...

Or add to sglang launch script:
  export RSWA_MAX_ATTN_LEN=132
  export PYTHONPATH=/path/to/this/dir:$PYTHONPATH
  exec python3 -m sglang.launch_server ...
"""
import os

_RSWA_MAX = int(os.environ.get("RSWA_MAX_ATTN_LEN", "0"))

if _RSWA_MAX > 0:
    import torch
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    # Save originals
    _orig_fill_kv = TritonAttnBackend._fill_kv_indptr_and_indices
    _orig_update_decode = TritonAttnBackend._update_decode_kv_buffers
    _orig_forward_decode = TritonAttnBackend.forward_decode

    print(f"[R-SWA] Patching TritonAttnBackend: max_attn_len={_RSWA_MAX}", flush=True)

    def _rswa_fill_kv(self, bs, seq_lens, req_pool_indices, kv_indices):
        """Capped version: limit KV indices to first RSWA_MAX tokens."""
        capped = torch.clamp(seq_lens, max=_RSWA_MAX)
        return _orig_fill_kv(self, bs, capped, req_pool_indices, kv_indices)

    def _rswa_update_decode(self, bs, seq_lens, req_pool_indices):
        """Capped decode: limit full-attention layers to RSWA_MAX tokens."""
        capped = torch.clamp(seq_lens[:bs], max=_RSWA_MAX)
        return _orig_update_decode(self, bs, capped, req_pool_indices)

    def _rswa_forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True,
                             sinks=None, score_mod=None, aux_tensors=None):
        """Forward decode with R-SWA logging (first call only)."""
        if not getattr(self, '_rswa_logged', False):
            print(f"[R-SWA] forward_decode active: max_attn_len={_RSWA_MAX}", flush=True)
            self._rswa_logged = True
        return _orig_forward_decode(self, q, k, v, layer, forward_batch,
                                     save_kv_cache, sinks=sinks,
                                     score_mod=score_mod, aux_tensors=aux_tensors)

    TritonAttnBackend._fill_kv_indptr_and_indices = _rswa_fill_kv
    TritonAttnBackend._update_decode_kv_buffers = _rswa_update_decode
    TritonAttnBackend.forward_decode = _rswa_forward_decode

    print(f"[R-SWA] Patch applied: _fill_kv_indptr_and_indices + _update_decode_kv_buffers + forward_decode", flush=True)
else:
    # RSWA_MAX_ATTN_LEN=0 or unset: no patch
    pass
