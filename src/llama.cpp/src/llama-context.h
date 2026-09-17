#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <deque>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    // [CGC Phase Discrimination 2026-09-08] Set the current decode phase.
    // Must be called before llama_decode() for each batch.
    // UNKNOWN is the safe default (exact path, no ZERO-slot fast path).
    void set_cgc_phase(cgc_phase_t phase);
    cgc_phase_t get_cgc_phase() const;

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    // CGC expert-cache hook (rebuilt from build-test3 DWARF). Called from the eval callback
    // when the compute graph reaches a `ffn_moe_topk-<il>` node during decode: reads the top-k
    // expert ids, ensures the layer's union is resident in the pool, and repoints the FFN
    // mul_mat_id src0 tensors + remap leaf so the layer computes over cache-resident weights.
    void expert_cache_on_topk(ggml_tensor * t);

    // Wrapper installed as cparams.cb_eval (static so it can be passed to
    // ggml_backend_sched_set_eval_callback): dispatches expert_cache_on_topk on the top-k
    // nodes, then forwards to the user's callback. Returns the user callback's result (or
    // true when no user callback is installed).
    static bool expert_cache_eval_cb(ggml_tensor * t, bool ask, void * user_data);

    // [CGC bit-bisect v6] dump intermediate F32 graph tensors by name after each ubatch
    // compute (CGC_TENSOR_DUMP="name-substr,name-substr", CGC_TD_PMAX gates batches by
    // seq pos_max, default 7 = prefill chunk 0). Files: /tmp/cgc_td_<DEF|MTP>_p<pmax>_
    // <tensor-name>.f32 — identical names across the spec / simple runs so the files can
    // be diffed bit-by-bit. Used to bisect WHERE inside layer 3 (full-attn) rows 0-2 of
    // chunk 0 first diverge between the MTP and non-MTP paths.
    void cgc_dump_graph_tensors(ggml_cgraph * gf);

    // [CGC V2 logits oracle 2026-09-05] Per-ubatch logits summary to JSONL. Complements V1
    // (byte-identity on cache fill) by catching semantic divergence V1 cannot: V1 verifies
    // the bytes match GGUF; V2 verifies the bytes produce the same model output. Trigger:
    // CGC_LOGITS_ORACLE_DUMP=path/to/oracle.jsonl. CGC_LOGITS_ORACLE_TOPN (default 5) controls
    // top-k token records. CGC_LOGITS_ORACLE_FIRST_N (default unlimited, 0=unlimited) caps the
    // number of ubatches dumped (memory-bounded for long sequences). The function scans
    // gf for the F32 logits tensor (ne[0] == n_vocab, ne[1] == n_tokens) itself so the
    // caller only needs to pass the graph.
    //
    // Format (one JSON object per line):
    //   {"step":N,"token_idx":T,"n_tokens":NT,"n_vocab":V,"ctx_type":"DEF|MTP",
    //    "logits_fnv1a64":"<hex16>","row_fnv1a64":"<hex16>",
    //    "sum":X,"mean":Y,
    //    "argmax_token":ID,"argmax_logit":V,
    //    "top":[{"t":ID,"v":V}, ...]}
    // Compare two oracle files with scripts/check/cgc_logits_oracle_compare.py.
    void cgc_logits_oracle_dump(ggml_cgraph * gf, uint32_t n_tokens, uint32_t n_outputs);

    // [CGC bit-bisect v7] in-compute tensor dump. The CGC segmented dispatcher
    // (ggml-backend.cpp hook_seg) forwards each node to expert_cache_eval_cb with
    // ask=false right after the node's segment completed and before the next
    // segment is submitted, so the node's buffer still holds its computed value
    // (the post-compute cgc_dump_graph_tensors reads recycled work buffers).
    // CGC_TD_CB="Name,Name,FA": exact tensor-name matches (use "-<il>" suffixes,
    // exact strcmp — no substring layer-30 pollution); "FA" additionally dumps
    // every GGML_OP_FLASH_ATTN_EXT node's Q/K/V/mask inputs and its output as
    // fa<n>_q/k/v/m/o (visit order = layer order). Gated to ubatch
    // CGC_TD_CB_UB (default 1 = prefill chunk 0) via cgc_tdcb_ubatch_seq.
    // Files: /tmp/cgc_tdcb_<DEF|MTP>_<name>[_n<occ>].f32 (F16/BF16 converted to
    // F32 losslessly; other types dumped raw as .bin). ne/nb are logged to
    // stderr as "CGC-TDCB:" lines so the comparison script can decode strides.
    void cgc_tdcb_maybe_dump(ggml_tensor * t);

    // ubatch sequence counter (1 = first ubatch the context ever computed)
    int64_t cgc_tdcb_ubatch_seq = 0;

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    // CGC expert-cache hook state (rebuilt from build-test3 DWARF; L243-282)
    // Decode step counter for the eval hook (only incremented for decode n_tokens==1).
    uint64_t cache_hook_count = 0;
    // layer -> remap leaf tensor (ggml_dup_tensor of the top-k ids, marked input so the hook
    // can write the remapped ids into its buffer before the FFN mul_mat_id dispatches).
    // mutable: filled from the const graph_get_cb.
    mutable std::map<int, ggml_tensor *> cache_remap_tensors;
    // [CGC 2026-09-15 S1 slot-table] layer -> the I32 [1, n_expert] expert->slot table that the
    // graph reads through ggml_get_rows when CGC_SLOT_TABLE_GPU=1. It is the GPU-side replacement
    // for the host-written remap leaf: instead of the CPU mapping "selected expert -> slot" and
    // hand-writing the resulting id vector, the hook publishes the whole per-layer table and the
    // graph computes `slots = get_rows(table, selected_experts)`.
    //
    // INVARIANT (the whole point of stage S1): every place that writes the remap leaf must ALSO
    // publish this table under the SAME mapping, so the GPU-computed ids and the host-written ids
    // are the same sequence by construction. The four write sites are the decode fast path, the
    // exact/prefill path, the L4 layer-0 identity path and the gather (union-index) path.
    //
    // The leaf itself is still built when this is on: it is simply not consumed. That keeps every
    // other path bit-identical and makes the S1 A/B a pure scheduler/dispatch change.
    // mutable: filled from the const graph_get_cb.
    mutable std::map<int, ggml_tensor *> cache_slot_table_tensors;
    // [CGC 2026-09-17 §11.5] layer -> the `ffn_moe_ids_leaf` node: the INDEX VECTOR the S1 gather
    // reads (k*n_tokens int32, contiguous, pinned). The hook writes the step's raw expert ids into it
    // -- the same array the host leaf is written from -- so the gather never depends on the device
    // materialising a copy of `selected_experts`, which is what produced a correct token 0 and a wrong
    // token >= 1 (a strided CONT read as contiguous blocks; see the point of use in llama-graph.cpp).
    // Contract: every step that builds the S1 nodes must fill this, like the table itself.
    mutable std::map<int, ggml_tensor *> cache_ids_tensors;
    // [CGC M1 work item 4 · canonical gather order] layer -> the `ffn_moe_canon_perm` leaf: the
    // PERMUTATION the graph applies to `selected_experts` (k*n_tokens int32, 1-D, pinned) when
    // CGC_CANON_ORDER != 0. perm[p] = the ORIGINAL POSITION whose id belongs at canonical position p,
    // so the permuted id vector is ids[perm] and the permuted weight vector must be weights[perm].
    //
    // Why the permutation is written from the host: its key is the expert ID, which in the pool path
    // only exists on the host (the device holds slot indices). The hook is the one place that holds
    // the raw ids AND the pool's reverse map, and it must write this leaf in the same step it writes
    // the remap leaf -- the two are one mapping, exactly like the table/leaf pair above. A step that
    // writes one without the other mis-pairs experts with weights SILENTLY (both orders are legal
    // ids), which is why the hook permutes its host id snapshot instead of permuting each site.
    //
    // Consumed by GET_ROWS as the index vector; the graph builds the same nodes in mode 1 and mode 2
    // (identity), so mode 2 isolates "the reordering changed the numbers" from "the extra nodes did".
    // mutable: filled from the const graph_get_cb.
    mutable std::map<int, ggml_tensor *> cache_canon_tensors;
    // [CGC 2026-09-15 S1 slot-table] layer -> the `ffn_moe_slots` node, i.e. the GET_ROWS result
    // (viewed as [k, n_tokens]) that mul_mat_id consumes as its ids operand. Captured ONLY so the
    // post-synchronize readback can read back what the Metal gather actually produced, on the host,
    // after the command buffer completed.
    //
    // Why that readback is the only trustworthy one: CGC-MMID-ASSERT reads this same buffer from the
    // host at Metal ENCODE time, i.e. before the command buffer that contains the gather has run.
    // For a host-written leaf that is harmless (the CPU wrote it), but for a GPU-computed id vector
    // the host read is a race -- it sees whatever the allocator last put at that address, which is
    // exactly the shape of the reported symptoms (F32 router probabilities, +/-NaN, and a mix of
    // legal and illegal indices within one node). Reading after ggml_backend_sched_synchronize()
    // separates "the gather really produced garbage" from "the probe read too early"; the two are
    // otherwise indistinguishable and they demand opposite fixes.
    // mutable: filled from the const graph_get_cb.
    mutable std::map<int, ggml_tensor *> cache_slots_out_tensors;
    // [CGC 2026-09-15 S1 output capture] layer -> the `ffn_moe_down` node: the MoE FFN's final
    // matmul, i.e. the whole layer's expert contribution BEFORE the residual add. Filled only under
    // CGC_S1_OUT_CAP=1 so the post-synchronize readback can answer the one question §9.18 left open.
    //
    // Why this is the right next probe: the kernel-side ids capture proved the ids mul_mat_id
    // consumes are BIT-IDENTICAL between the host-leaf arm and the GPU-table arm, and layer 1's
    // router logits are identical on both sides, so the layer-1 MoE output has the same input and
    // the same index vector. If it still differs, the carrier is neither the ids nor the lookup --
    // it is the WEIGHT CONTENT those ids point at (pool/slot residency). That is a different repair
    // (a pool-fill fix) from "the table disagrees with the leaf at consume time" (a publish-order
    // fix), and only the output tells them apart.
    //
    // Unlike cache_slots_out_tensors this one must PIN the tensor with ggml_set_output: ffn_moe_down
    // is an intermediate, so ggml-alloc may recycle its buffer the moment its consumer has read it,
    // and a post-synchronize read of a dead buffer reads the next occupant (the mirror of the
    // encode-time race this file already documents). Pinning perturbs the allocator layout, so this
    // is a DIAGNOSTIC-arm-only knob: it is applied to EVERY arm of a comparison identically, and
    // never enabled on a bit-identical gate arm.
    //
    // The perturbation is bounded by the NUMBER of pinned tensors on a graph, not by their bytes:
    // pinning all 40 layers in every graph and pinning 11 layers on prefill graphs only both die at
    // the load-time warmup decode (kIOGPUCommandBufferCallbackErrorOutOfMemory -> CGC_METAL_FAIL_STOP
    // abort) even though 11 layers of ffn_moe_down at ntok=2 hold ~1.4 MB, while 4 pinned layers on
    // prefill and 40 pinned layers on decode both run clean. So a caller must budget pins per GRAPH,
    // and the safe working number on this machine is between 4 and 11. Not established: why.
    // mutable: filled from the const graph_get_cb.
    mutable std::map<int, ggml_tensor *> cache_down_out_tensors;
    // layer -> renormalized-routing mask leaf (F32 [n_expert, 1]; CGC_RN_ROUTING=1 only): the
    // hook rewrites 0.0/-inf per expert from the slot_table before the router softmax reads it
    // (Step-3: cold experts get -inf -> softmax renormalizes over resident experts).
    // mutable: filled from the const graph_get_cb.
    mutable std::map<int, ggml_tensor *> cache_rn_mask_tensors;
    // layer -> probs tensor (softmax over experts); the hook uses it to read per-token probs.
    // mutable: filled from the const graph_get_cb.
    mutable std::map<int, ggml_tensor *> cache_probs_tensors;
    // layer -> the FFN expert weight tensors (src0 of mul_mat_id) by kind: 0=gate 1=up
    // 2=down 3=gate_up. The hook repoints their data pointer at the cache pool region (pool
    // path) or a per-step gather buffer (L3-B path), restoring it on the next step.
    // mutable: filled from the const graph_get_cb.
    mutable std::map<int, std::vector<ggml_tensor *>> cache_ffn_tensors;
    // layer -> scratch buffer holding the remap ids written into cache_remap_tensors[il].
    std::map<int, std::vector<uint8_t>> cache_remap_buf;
    // [kind] per-kind gather buffers (L3-B path): the hook gathers selected expert weights
    // into these contiguous buffers and points the FFN src0 tensor at them.
    //
    // [CGC M1 Metal slab 2026-09-14] These were host std::vector<uint8_t>. That is the root cause
    // of the union>slots corruption: the FFN src0 tensor lives in a Metal buffer, and
    // ggml_metal_buffer_get_id() resolves a tensor to a buffer purely by ADDRESS RANGE --
    //
    //     ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;
    //     if (ioffs >= 0 && ioffs + ggml_nbytes(t) <= buf->buffers[i].size) -> found
    //     else -> "tensor '%s' buffer is nil"
    //
    // A host pointer lies in no Metal buffer's range, so Metal read nothing and the layer produced
    // wrong values. Measured at 2 GiB: 468 nil dispatches, M1 2/42, step-2 sum 55% off.
    // The slab is a real Metal backend buffer instead. See
    // docs/M1_POOL_GRAPH_DECOUPLE_PLAN_2026-09-14.md 3.1.
    struct cgc_gather_slab {
        int                   kind   = -1;
        size_t                stride = 0;  // bytes/expert: this kind's WHOLE-MODEL max stride
        ggml_backend_buffer_t buf    = nullptr;
        uint8_t *             base   = nullptr;
        size_t                size   = 0;  // bytes = cgc_gather_slab_cap() * stride
    };
    // At most one slab per kind in practice (2026-09-14): each is sized for its kind's whole-model
    // maximum stride on first touch, so there is nothing left to grow to -- the entries that used
    // to pile up (157.5 MiB allocated for an 89.0 MiB live boundary) are gone. Never freed early
    // and never shrunk regardless: a repointed tensor keeps pointing into its slab until the next
    // graph build restores it, so freeing on growth would leave other layers' tensors dangling.
    // All of them are released in the destructor.
    std::vector<cgc_gather_slab> cache_gather_slab;
    // Index into cache_gather_slab of the current (largest) slab per kind, or -1. An index rather
    // than a pointer, because push_back may move the vector's storage.
    int cache_gather_cur[4] = { -1, -1, -1, -1 };
    // [CGC M2 2026-09-14] CGC_PREFILL_STREAM, validated once at construction. The stream path sets
    // ne[2] = n_expert on the expert tensors and fills n_expert experts, so it is only legal when
    // the slab can hold them all: with the default cap of 64 the fill writes 256 experts into a
    // 64-expert buffer, i.e. out of bounds, and the FFN then reads past its own allocation.
    // Measured: CGC_PREFILL_STREAM=1 without CGC_GATHER_SLAB_CAP=256 allocated a 27.50 MiB slab
    // (64 x 450,560 bytes) while CGC-PREFILL-STREAM logged experts=256. Refusing here means the
    // n_batch clamp stays on and the pool path runs, which is slow but correct -- the alternative
    // is a silently wrong slab.
    bool cgc_stream_on = false;
    // [CGC M1 work item 2 · phase split] THE decode-graph width bound in tokens, computed ONCE in
    // the constructor from the pool's ROUTABLE geometry:
    //     min over pooled layers of llama_expert_cache_usable_slots(), / hparams.n_expert_used,
    //     then lowered (never raised) by cgc_pool_max_tokens().
    // It is a member, and not a call to cgc_pool_max_tokens(), because FOUR sites must agree on it:
    // the n_batch clamp, the prewarm/prefetch pool features, the hook's prefill branch, and the
    // graph's decode block (handed over through llm_graph_params). When those sites each derived the
    // phase for themselves the graph could build the decode graph for a step the hook served with the
    // prefill branch -- both variants of that disagreement are silent. See llama-cgc-phase.h.
    uint32_t cgc_decode_max_tokens = 0;
    // Return the slab to gather this kind into, or nullptr if the request cannot be served.
    // Capacity is cgc_gather_slab_cap() experts and never follows the pool or the observed union
    // -- see the comment on cgc_gather_slab_cap in llama-context.cpp. The SIZE is the kind's
    // whole-model max stride (all layers are captured before the first eval hook), so the first
    // allocation is final; `exp_bytes` only still decides whether the existing slab is big enough.
    cgc_gather_slab * cgc_gather_slab_get(int kind, size_t exp_bytes, ggml_backend_buffer_type_t buft);
    // (layer, kind) -> (original src0 data pointer); saved before the FFN tensor is repointed
    // at the cache pool, restored in process_ubatch before every build_graph.
    // mutable: written from the const graph_get_cb.
    mutable std::map<std::pair<int,int>, void *> cache_orig;
    // [CGC M1 Metal slab] (layer, kind) -> original src0 ne[2], for the tensors whose ne[2] the
    // gather path rewrites to the union size. Restored in the same pass as cache_orig, so every
    // freshly built graph starts from the full original expert geometry.
    mutable std::map<std::pair<int,int>, int64_t> cache_gather_ne2;
    // [CGC M1 Metal slab] (layer, kind) -> original src0 buffer. The gather path must move the
    // tensor's BUFFER along with its data, because Metal resolves a tensor through its own buffer
    // (see the note in llama-context.cpp); moving only data leaves the lookup computing an offset
    // against the original allocation and Metal still reports nil. Restored with the ne[2] above.
    mutable std::map<std::pair<int,int>, ggml_backend_buffer_t> cache_gather_orig_buf;

    // [CGC M2 double-buffer 2026-09-14] Two slab sets (A/B) for overlapping slab fill with
    // graph build. While layer L computes/builds against set N, a background thread fills
    // layer L+1 into set 1-N. The eval hook swaps sets at each layer boundary.
    int  cgc_db_cur       = 0;       // current slab set in use (0 or 1)
    int  cgc_db_slab[2][4] = {{-1,-1,-1,-1}, {-1,-1,-1,-1}};  // indices into cache_gather_slab
    // [CGC M2 pool reuse 2026-09-14] per (layer, kind) DESTINATION spacing for the slab fill, i.e.
    // the layer's own per-expert byte count (what mul_mat_id walks as nb[2]). It must not be the
    // slab's stride: that is the kind's whole-model maximum, so on a mixed-quant GGUF the experts
    // would be laid out on a different pitch than the reader walks (harmless only while the
    // contiguous fast path -- which ignores the parameter and writes linearly -- is the one that
    // runs). Precomputed here because the fill worker must not touch cache_ffn_tensors while the
    // main thread builds a graph.
    std::vector<std::vector<size_t>> cgc_db_stride;  // [layer][kind] bytes/expert, 0 = absent
    bool cgc_db_init       = false;   // both sets allocated
    std::thread cgc_db_thread;
    std::mutex  cgc_db_mtx;
    std::condition_variable cgc_db_cv;
    int  cgc_db_job_layer = -1;       // layer the background thread is filling (-1 = no job)
    int  cgc_db_job_set   = -1;       // which set the background thread is filling
    bool cgc_db_job_done  = false;    // background job completed
    bool cgc_db_stop      = false;    // signal background thread to exit
    void cgc_db_worker();             // background thread entry point
    void cgc_db_start_prefill(int layer, int set);  // launch background fill for layer into set
    void cgc_db_wait();               // wait for background job to finish
    // per-layer union of experts used by the current decode step (deduped, sorted).
    std::vector<std::vector<uint32_t>> cache_step_union;
    // per-layer union of experts used by the previous decode step.
    std::vector<std::vector<uint32_t>> cache_prev_union;
    // per-layer tail-union history (TAILPIN): deques of recent unions used to pre-pin.
    std::vector<std::deque<std::vector<uint32_t>>> cache_tail_union;
    bool cache_tail_unpinned = false;

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // [CGC Phase Discrimination 2026-09-08] Current decode phase.
    // Set by set_cgc_phase() before each llama_decode() call.
    // Used by expert_cache_on_topk() to distinguish prefill vs verify vs draft.
    // UNKNOWN is the safe default: exact ensure_batch path, no ZERO-slot fast path.
    cgc_phase_t cgc_current_phase = CGC_PHASE_UNKNOWN;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
