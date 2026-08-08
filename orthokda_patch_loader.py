#!/usr/bin/env python3
"""OrthoKDA monkey-patch for sglang attention backend.

Replaces decode attention with orthogonal KV projection:
- All KV (reference + window) projected to ortho_dim for attention weights
- V kept at full dimension for weighted sum
- Softmax normalizes over ALL tokens (ref + window) together

Prefill: falls back to original forward (saves KV cache + computes attention)
Decode: uses OrthoKDA projected attention + saves KV cache for sglang consistency

Usage:
  export ORTHO_KDA_ENABLED=1
  export PYTHONPATH=/path/to/this/dir:$PYTHONPATH
  python3 -m sglang.launch_server ...
  (sitecustomize.py auto-imports this module)

Diagnostic modes:
  ORTHO_FULL_ATTN=1  Disable projection, use full-dim K for attention.
                     Useful to verify windowing logic independent of projection.
"""
import os
import sys
import torch
import torch.nn.functional as F

_ENABLED = os.environ.get("ORTHO_KDA_ENABLED", "0") == "1"

if _ENABLED:
    _REF_LEN = int(os.environ.get("ORTHO_REF_LEN", "4"))
    _WIN_SIZE = int(os.environ.get("ORTHO_WINDOW_SIZE", "128"))
    _ORTHO_DIM = int(os.environ.get("ORTHO_BASE_DIM", "64"))
    _FULL_ATTN = os.environ.get("ORTHO_FULL_ATTN", "0") == "1"

    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    _orig_forward = AttentionBackend.forward

    # Fix SEQRLEN_FIX bug: schedule_batch.py:1957 references req.fill_ids
    # which doesn't exist on Req (uses __slots__). Add property to delegate
    # to full_untruncated_fill_ids.
    try:
        from sglang.srt.managers.schedule_batch import Req
        if not isinstance(getattr(Req, 'fill_ids', None), property):
            Req.fill_ids = property(
                lambda self: self.full_untruncated_fill_ids,
                lambda self, value: setattr(self, 'full_untruncated_fill_ids', value)
            )
            print("[OrthoKDA] Added fill_ids property to Req (SEQRLEN_FIX fix)", flush=True)
    except Exception as e:
        print(f"[OrthoKDA] SEQRLEN_FIX fix skipped: {e}", flush=True)

    _layer_states = {}
    _debug_count = 0
    _error_count = 0
    _MAX_ERRORS = 5

    _mode_str = "FULL_ATTN (no projection)" if _FULL_ATTN else f"ORTHO proj dim={_ORTHO_DIM}"
    print(f"[OrthoKDA] Loader: ref={_REF_LEN} win={_WIN_SIZE} ortho_dim={_ORTHO_DIM} "
          f"mode={_mode_str}", flush=True)

    def _init_state(layer_id, kv_heads, head_dim, device, dtype):
        """Initialize OrthoKDA state for a layer."""
        basis_f32 = torch.randn(head_dim, _ORTHO_DIM, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(basis_f32)
        basis = Q.to(dtype=dtype)

        state = {
            "ref_K": torch.zeros(_REF_LEN, kv_heads, head_dim, device=device, dtype=dtype),
            "ref_V": torch.zeros(_REF_LEN, kv_heads, head_dim, device=device, dtype=dtype),
            "win_K": torch.zeros(_WIN_SIZE, kv_heads, head_dim, device=device, dtype=dtype),
            "win_V": torch.zeros(_WIN_SIZE, kv_heads, head_dim, device=device, dtype=dtype),
            "basis": basis,
            "win_count": 0,
            "ref_set": False,
        }
        print(f"[OrthoKDA] Layer {layer_id}: kv_heads={kv_heads} head_dim={head_dim} "
              f"ortho_dim={_ORTHO_DIM} dtype={dtype}", flush=True)
        return state

    def _orthokda_forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        global _debug_count, _error_count

        if _error_count >= _MAX_ERRORS:
            return _orig_forward(self, q, k, v, layer, forward_batch,
                                 save_kv_cache=save_kv_cache, **kwargs)

        try:
            layer_id = layer.layer_id
            is_extend = forward_batch.forward_mode.is_extend()
            is_idle = forward_batch.forward_mode.is_idle()

            # Idle mode: return empty like original
            if is_idle:
                return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

            if k is None or v is None:
                return _orig_forward(self, q, k, v, layer, forward_batch,
                                     save_kv_cache=save_kv_cache, **kwargs)

            num_tokens = q.shape[0]
            q_heads = q.shape[1] if q.dim() >= 3 else getattr(layer, "tp_q_head_num", 1)
            head_dim = q.shape[2] if q.dim() >= 3 else getattr(layer, "qk_head_dim", 128)
            kv_heads = k.shape[1] if k.dim() >= 3 else getattr(layer, "tp_k_head_num", 1)
            head_dim_k = k.shape[2] if k.dim() >= 3 else getattr(layer, "v_head_dim", head_dim)

            # Use layer.scaling as the attention scale (1.0 for Gemma4 with QK-norm)
            attn_scale = getattr(layer, "scaling", 1.0)

            if _debug_count < 3:
                print(f"[OrthoKDA] call#{_debug_count} L{layer_id} ext={is_extend} "
                      f"q={q.shape}/{q.dtype} k={k.shape} v={v.shape} "
                      f"qh={q_heads} kvh={kv_heads} hd={head_dim} hdk={head_dim_k} "
                      f"scale={attn_scale}", flush=True)
                _debug_count += 1

            if layer_id not in _layer_states:
                _layer_states[layer_id] = _init_state(layer_id, kv_heads, head_dim_k,
                                                       q.device, q.dtype)
            state = _layer_states[layer_id]

            if is_extend:
                # === PREFILL: capture reference + window, use original forward ===
                if num_tokens >= _REF_LEN:
                    state["ref_K"].copy_(k[:_REF_LEN])
                    state["ref_V"].copy_(v[:_REF_LEN])
                    state["ref_set"] = True

                win_start = max(_REF_LEN, num_tokens - _WIN_SIZE)
                win_len = min(num_tokens - win_start, _WIN_SIZE)
                if win_len > 0:
                    state["win_K"][:win_len].copy_(k[win_start:win_start + win_len])
                    state["win_V"][:win_len].copy_(v[win_start:win_start + win_len])
                state["win_count"] = win_len

                return _orig_forward(self, q, k, v, layer, forward_batch,
                                     save_kv_cache=save_kv_cache, **kwargs)

            else:
                # === DECODE: OrthoKDA projected attention ===

                # CRITICAL: Save KV cache to sglang's token_to_kv_pool.
                # Without this, init_forward_metadata on the next batch reads
                # uninitialized memory → cudaErrorIllegalAddress.
                # sglang 0.5.16: SWAKVPool requires KVWriteLoc (not bare tensor)
                # so swa_loc is available for sliding-window layers.
                if save_kv_cache:
                    metadata = getattr(self, 'forward_metadata', None)
                    swa_loc = getattr(metadata, 'swa_out_cache_loc', None) if metadata else None
                    full_loc = getattr(metadata, 'out_cache_loc_full_physical', None) if metadata else None
                    try:
                        from sglang.srt.mem_cache.memory_pool import KVWriteLoc
                        loc_info = KVWriteLoc(
                            forward_batch.out_cache_loc,
                            swa_loc,
                            full_loc=full_loc,
                        )
                    except ImportError:
                        loc_info = forward_batch.out_cache_loc
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        loc_info,
                        k,
                        v,
                        getattr(layer, "k_scale", None),
                        getattr(layer, "v_scale", None),
                    )

                group_size = q_heads // kv_heads if kv_heads > 0 else 1
                basis = state["basis"]

                if _debug_count < 6:
                    print(f"[OrthoKDA] DECODE L{layer_id}: q={q.shape} k={k.shape} v={v.shape} "
                          f"qh={q_heads} kvh={kv_heads} gs={group_size} "
                          f"hd={head_dim} hdk={head_dim_k} basis={basis.shape} "
                          f"scale={attn_scale} win_count={state['win_count']} "
                          f"mode={'FULL' if _FULL_ATTN else 'ORTHO'}", flush=True)
                    _debug_count += 1

                outputs = []
                for i in range(num_tokens):
                    k_i = k[i]   # [kv_heads, head_dim_k]
                    v_i = v[i]

                    # === FIX: Window append logic ===
                    # When window is not full, append at position win_count.
                    # When full, slide left by 1 and add at end.
                    # OLD BUG: always slid + placed at [-1], but read [:win_count],
                    # so new token was never in the read range.
                    wc = state["win_count"]
                    if wc < _WIN_SIZE:
                        # Append at next free slot
                        state["win_K"][wc].copy_(k_i)
                        state["win_V"][wc].copy_(v_i)
                        state["win_count"] = wc + 1
                    else:
                        # Window full: slide left, add at end
                        state["win_K"][:-1].copy_(state["win_K"][1:])
                        state["win_V"][:-1].copy_(state["win_V"][1:])
                        state["win_K"][-1].copy_(k_i)
                        state["win_V"][-1].copy_(v_i)

                    q_i = q[i]   # [q_heads, head_dim]

                    # Collect ALL KV: reference + window
                    kv_parts_k = []
                    kv_parts_v = []

                    if state["ref_set"]:
                        kv_parts_k.append(state["ref_K"])
                        kv_parts_v.append(state["ref_V"])

                    wc = state["win_count"]
                    if wc > 0:
                        kv_parts_k.append(state["win_K"][:wc])
                        kv_parts_v.append(state["win_V"][:wc])

                    # Concatenate: [total_len, kv_heads, head_dim_k]
                    all_k = torch.cat(kv_parts_k, dim=0)
                    all_v = torch.cat(kv_parts_v, dim=0)

                    # Transpose: [kv_heads, total_len, head_dim_k]
                    all_k = all_k.transpose(0, 1)
                    all_v = all_v.transpose(0, 1)

                    total_len = all_k.shape[1]

                    if _FULL_ATTN:
                        # === FULL ATTENTION (no projection) ===
                        # Use original head_dim for attention scores.
                        # This tests windowing logic without projection artifacts.
                        if group_size > 1:
                            # GQA: expand KV for each q_head in the group
                            q_grouped = q_i.view(kv_heads, group_size, head_dim)
                            # [kv_heads, group_size, 1, head_dim] @ [kv_heads, 1, total_len, head_dim]^T
                            all_k_exp = all_k.unsqueeze(1).expand(-1, group_size, -1, -1)
                            all_v_exp = all_v.unsqueeze(1).expand(-1, group_size, -1, -1)
                            all_k_exp = all_k_exp.reshape(q_heads, total_len, head_dim_k)
                            all_v_exp = all_v_exp.reshape(q_heads, total_len, head_dim_k)
                            q_3d = q_grouped.reshape(q_heads, 1, head_dim)
                        else:
                            all_k_exp = all_k
                            all_v_exp = all_v
                            q_3d = q_i.unsqueeze(1)

                        attn = torch.matmul(q_3d, all_k_exp.transpose(-1, -2)) * attn_scale
                        attn = F.softmax(attn, dim=-1)
                        out_i = torch.matmul(attn, all_v_exp).squeeze(1)  # [q_heads, head_dim_k]

                    else:
                        # === ORTHO PROJECTED ATTENTION ===
                        # Project K to ortho_dim: [kv_heads, total_len, ortho_dim]
                        all_k_proj = torch.matmul(all_k, basis)

                        # Project Q to ortho_dim and compute attention
                        if group_size > 1:
                            # GQA: project Q per kv_head group
                            q_grouped = q_i.view(kv_heads, group_size, head_dim)
                            q_proj = torch.matmul(q_grouped, basis)  # [kv_heads, group_size, ortho_dim]
                            all_k_proj_exp = all_k_proj.unsqueeze(1).expand(-1, group_size, -1, -1)
                            all_v_exp = all_v.unsqueeze(1).expand(-1, group_size, -1, -1)
                            all_k_proj_exp = all_k_proj_exp.reshape(q_heads, total_len, _ORTHO_DIM)
                            all_v_exp = all_v_exp.reshape(q_heads, total_len, head_dim_k)
                            q_3d = q_proj.reshape(q_heads, 1, _ORTHO_DIM)
                        else:
                            q_proj = torch.matmul(q_i, basis)
                            q_3d = q_proj.unsqueeze(1)
                            all_k_proj_exp = all_k_proj
                            all_v_exp = all_v

                        # Attention: Q @ K^T * scale -> softmax -> @ V
                        # For projected attention, normalize Q_proj and K_proj
                        # to unit length so dot product = cosine similarity.
                        # This makes the scale-independent of projection dim.
                        q_norm = F.normalize(q_3d, dim=-1, eps=1e-6)
                        k_norm = F.normalize(all_k_proj_exp, dim=-1, eps=1e-6)
                        attn = torch.matmul(q_norm, k_norm.transpose(-1, -2))
                        # Scale: temperature factor to sharpen/soften
                        # With cosine similarity in [-1,1], scale=1 gives uniform attention.
                        # Use sqrt(head_dim) as temperature to match original attention sharpness.
                        proj_scale = (head_dim ** 0.5) * attn_scale
                        attn = attn * proj_scale
                        attn = F.softmax(attn, dim=-1)
                        out_i = torch.matmul(attn, all_v_exp).squeeze(1)  # [q_heads, head_dim_k]

                    outputs.append(out_i)

                # Return 2D to match original forward_decode format
                output = torch.stack(outputs, dim=0)  # [num_tokens, q_heads, head_dim_k]
                output = output.reshape(num_tokens, q_heads * head_dim_k)  # [num_tokens, q_heads * head_dim_k]
                return output.to(device=q.device, dtype=q.dtype)

        except Exception as e:
            _error_count += 1
            print(f"[OrthoKDA] ERROR ({_error_count}/{_MAX_ERRORS}): {e}", flush=True)
            import traceback
            traceback.print_exc()
            return _orig_forward(self, q, k, v, layer, forward_batch,
                                 save_kv_cache=save_kv_cache, **kwargs)

    AttentionBackend.forward = _orthokda_forward

    try:
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        if "forward" in TritonAttnBackend.__dict__:
            TritonAttnBackend.forward = _orthokda_forward
            print("[OrthoKDA] Patched AttentionBackend + TritonAttnBackend.forward", flush=True)
        else:
            print("[OrthoKDA] Patched AttentionBackend.forward (Triton inherits)", flush=True)
    except ImportError:
        print("[OrthoKDA] Patched AttentionBackend.forward", flush=True)

else:
    pass
