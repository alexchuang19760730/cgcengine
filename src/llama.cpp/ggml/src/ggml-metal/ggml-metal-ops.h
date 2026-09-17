#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_op * ggml_metal_op_t;

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        struct ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion);

void ggml_metal_op_free(ggml_metal_op_t ctx);

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx);

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx);

//
// available ops:
//

// tokens per expert
size_t ggml_metal_op_mul_mat_id_extra_tpe(const struct ggml_tensor * op);

// id map [n_tokens, n_expert]
size_t ggml_metal_op_mul_mat_id_extra_ids(const struct ggml_tensor * op);

// return true if we should use the FA vector kernel for this op
bool ggml_metal_op_flash_attn_ext_use_vec(const struct ggml_tensor * op);

size_t ggml_metal_op_flash_attn_ext_extra_pad(const struct ggml_tensor * op);
size_t ggml_metal_op_flash_attn_ext_extra_blk(const struct ggml_tensor * op);
size_t ggml_metal_op_flash_attn_ext_extra_tmp(const struct ggml_tensor * op);

int ggml_metal_op_concat            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_repeat            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_acc               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_unary             (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_glu               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_sum               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_sum_rows          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_cumsum            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_get_rows          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_set_rows          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_diag              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_lightning_indexer (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_dsv4_hc           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_soft_max          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_ssm_conv          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_ssm_scan          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_rwkv              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_gated_delta_net   (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_solve_tri         (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_set               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_cpy               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pool_1d           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pool_2d           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_fwht              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_mul_mat           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_mul_mat_id        (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_mul_mat_id_down_combine (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_add_id            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_flash_attn_ext    (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_bin               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_silu_back         (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_l2_norm           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_group_norm        (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_norm              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_rope              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_im2col            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_2d           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_2d_dw        (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_3d           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_transpose_1d (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_transpose_2d (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_col2im_1d         (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_snake_fused       (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_upscale           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pad               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pad_reflect_1d    (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_roll              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_arange            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_argmax            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_argsort           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_top_k             (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_tri               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_opt_step_adamw    (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_opt_step_sgd      (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_count_equal       (ggml_metal_op_t ctx, int idx);

// [CGC 2026-09-15 S1 kernel-side ids capture; extended 2026-09-16 for the tensor-output side]
// Emit every capture slot not yet emitted. Called from ggml_metal_synchronize, i.e. only at points
// where the GPU work has provably completed. Counts as a diagnostic: it is a no-op unless
// CGC_IDS_CAPTURE and/or CGC_TENSOR_CAPTURE is set, and it changes no numerics.
//   CGC_IDS_CAPTURE=1                  -> snapshot the ids operand `mul_mat_id` consumes  (path=MV|MM)
//   CGC_TENSOR_CAPTURE=<exact node>    -> also snapshot that node's OUTPUT tensor          (path=DST)
//   CGC_TENSOR_CAPTURE_WORDS=N         -> how many int32 words of it (default 32, max 32)
// [CGC 2026-09-17 §9.18.6 r12] ...and a THIRD stream for the bytes BEHIND the ids:
//   CGC_POOL_CAPTURE=<exact node>      -> digest the rows the ids select, on the device   (path=POOL)
//   CGC_POOL_CAPTURE_ROWS=N            -> how many of the operand's ids (default 8, max 12)
//   CGC_POOL_CAPTURE_BYTES=N           -> bytes digested per row (default 4096, max 1 MiB)
// Every row carries the same `CGC-IDS-CAP` line format, so scripts/check/ids_capture_diff.py diffs the
// ids rows unchanged; the derived rows are named `<node>.dst` and `<node>.pool` (with `probe=`/`nsel=`
// instead of `ne=`/`off=`) to keep the three distinguishable. Rows are emitted in submission order
// across all three streams, which is what keeps each derived row inside its own graph.
void ggml_metal_cgc_ids_dump(void);

// [CGC 2026-09-17 §9.18.7] The POOL row's missing half: WHICH EXPERT each digested row belonged to.
//
// A POOL row says "the ids selected these slots, and those slots hold these bytes". It does NOT say
// whether two arms agreeing on a slot id means they agree on the DATA, because the pool's slot->expert
// map (`llama_expert_cache::slot_owner`, a HOST array owned by llama) is not visible from here.
// Without it the comparator cannot separate two different defects that print identically: "the same
// slot holds different experts" (a mapping/accounting difference) and "the same slot holds the same
// expert but the bytes differ" (a fill/content difference). Those call for different next steps.
//
// The reverse map has to be read where it lives, so llama registers a pull callback here through
// ggml_backend_reg_get_proc_address("ggml_metal_cgc_set_owner_fn") -- the same cross-dylib mechanism
// the other CGC readouts use. ggml-metal never dereferences a llama type.
//
//   `il`     : the MoE layer, parsed from the captured node's name (`ffn_moe_gate-17` -> 17).
//   `cap`    : how many int32 the caller can accept (the slot count is model-specific).
//   `out[s]` : expert id held by SLOT s, or -1 for a slot the pool does not own.
//   returns  : how many entries were written, or -1 if the layer is unknown to the callee.
//
// WHEN it is called, and why that matters: ONCE PER CAPTURE, AT ENCODE TIME -- i.e. immediately before
// the kernel that consumes the ids is submitted, not at dump time. The snapshot is copied into this
// side's own arena (see CGC_POOL_OWN_ARENA_MAX), so a row carries the mapping as of the encode of its
// own node. Sampling at dump time instead would read the map AFTER the whole pass, including any fill
// that completed after the consumer had already read it -- and that late-fill is one of the hypotheses
// under test, so the instrument must not use its answer to label the question.
//
// Cost: one memcpy of the slot array per matched node plus a 4 MiB arena on first use. No kernel, no
// numeric change, and a no-op unless CGC_POOL_CAPTURE is set AND a callback is registered.
//
// The read is UNLOCKED, like every other engine-side readout in this project: a background pool fill
// can be writing the same array, so a snapshot may occasionally be torn or one fill behind. That is
// why the SAME-ARM CONTROL is the premise of any verdict drawn from it (one arm, captured twice, must
// agree) -- a single run's `own=` values are never on their own evidence about anything.
typedef int32_t (*ggml_metal_cgc_owner_fn)(int32_t il, int32_t cap, int32_t * out);

void ggml_metal_cgc_set_owner_fn(ggml_metal_cgc_owner_fn fn);

// [CGC 2026-09-17 §9.18.8] The other half of the same question: the slot the HOST says the consumer
// should read, per consumer POSITION.
//
// `own=` answers "which expert does the slot the device used hold". It does NOT answer "is that the
// slot the host meant". Those are different failures and the pool row could not tell them apart:
//     the device read slot 8, which holds expert 237; the host's table says expert 237 lives in slot 8
//   -> device and host agree, and whatever went wrong was upstream of the table
//     the device read slot 0, which holds expert 163; the host's table says expert 237 lives in slot 8
//   -> the device read a DIFFERENT TABLE than the host published
// The second is a machine defect in this fork (stale table, wrong buffer, a fill that landed late);
// the first is not. Without the expectation the two print identically, because in both cases the
// device's ids "found a legal expert" -- 0 is a legal index and nothing asserts.
//
//   `il`     : the MoE layer, parsed from the captured node's name.
//   `cap`    : how many positions the caller can accept.
//   `out[j]` : the slot the host expects at consumer position j (element j of the ids operand), or
//              -1 if the host has no opinion for that position.
//   returns  : how many entries were written, or -1 if this layer has no expectation to give.
//
// Same sampling rule as the owner callback and for the same reason: ONCE PER CAPTURE, AT ENCODE TIME.
// Both annotations describe one instant, the instant the consuming kernel was submitted; sampling one
// of them at dump time would make the two disagree about a moment neither of them is looking at.
//
// Alignment is SELF-CHECKING, and that is not a nicety: on the arm whose graph actually consumes the
// host leaf (the anchor), `ids` and `exp` must come out IDENTICAL for every row. If they do not, the
// instrument is misaligned and nothing it prints is usable -- so the anchor arm doubles as its own
// alignment control, and the comparator reports the mismatch as an instrument fault rather than as an
// engine finding.
typedef int32_t (*ggml_metal_cgc_expect_fn)(int32_t il, int32_t cap, int32_t * out);

void ggml_metal_cgc_set_expect_fn(ggml_metal_cgc_expect_fn fn);

#ifdef __cplusplus
}
#endif
