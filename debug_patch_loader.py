#!/usr/bin/env python3
"""Debug patch: log tensor shapes, then pass through to original."""
import os
import torch

if os.environ.get("ORTHO_KDA_ENABLED", "0") == "1":
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    _orig = AttentionBackend.forward
    _logged = {}

    def _debug_forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        lid = layer.layer_id
        is_ext = forward_batch.forward_mode.is_extend()

        if lid not in _logged:
            _logged[lid] = 0
        _logged[lid] += 1

        # Log first 2 calls per layer
        if _logged[lid] <= 2:
            qshp = q.shape if q is not None else None
            kshp = k.shape if k is not None else None
            vshp = v.shape if v is not None else None
            qd = q.dtype if q is not None else None
            kd = k.dtype if k is not None else None
            ltpq = getattr(layer, "tp_q_head_num", "?")
            ltpk = getattr(layer, "tp_k_head_num", "?")
            lqkd = getattr(layer, "qk_head_dim", "?")
            lvd = getattr(layer, "v_head_dim", "?")
            print(f"[DBG] L{lid} call#{_logged[lid]} ext={is_ext} "
                  f"q={qshp}/{qd} k={kshp}/{kd} v={vshp} "
                  f"tpq={ltpq} tpk={ltpk} qkd={lqkd} vd={lvd} "
                  f"save_kv={save_kv_cache}", flush=True)

            # Also log the return value shape
            result = _orig(self, q, k, v, layer, forward_batch,
                           save_kv_cache=save_kv_cache, **kwargs)
            rshp = result.shape if result is not None else None
            rd = result.dtype if result is not None else None
            print(f"[DBG] L{lid} return: {rshp}/{rd}", flush=True)
            return result

        return _orig(self, q, k, v, layer, forward_batch,
                     save_kv_cache=save_kv_cache, **kwargs)

    AttentionBackend.forward = _debug_forward
    print("[DBG] Debug patch applied to AttentionBackend.forward", flush=True)
