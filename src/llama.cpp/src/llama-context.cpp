#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include "llama-expert-cache.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

//
// llama_context
//

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

// [CGC M2 2026-09-14] defined next to cgc_gather_slab_cap() below; declared here because the
// constructor validates CGC_PREFILL_STREAM before the clamp decision.
static bool cgc_prefill_stream_enabled(int64_t n_expert);

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type     = params.ctx_type;
    cparams.pooling_type = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    cparams.fused_lid    = true;
    cparams.auto_flid    = true;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    // CGC expert-cache L4: the per-layer expert tensors are SHRUNK to the bounded pool capacity,
    // so a multi-token (n_tokens > cgc_pool_max_tokens()) prefill graph would read full-range
    // router ids (0..n_expert-1) against a capacity-slot tensor -> OOB -> NaN -> garbage downstream
    // (observed: layer-1 routes to experts >= capacity, layer-2+ router probs become NaN). The pool
    // path is correct for single-token AND small multi-token steps (n_tokens <= cgc_pool_max_tokens()):
    // the remap leaf is built, the hook maps expert ids to slot indices (union across all tokens),
    // and the FFN reads the pool region by slot. So when the pool is active we cap n_batch to
    // cgc_pool_max_tokens() so speculative/MTP verify batches (n_max+1 <= cap) flow through the pool
    // path, while still bounding every step to the pool (never the full model). Chunked callers
    // (mtmd, warmup) must read the capped value via llama_n_batch().
    // [CGC M1 work item 1 CGC_POOL_SPLIT, 2026-09-14] The clamp is NOT lifted for an owned pool.
    // The temptation was: with the expert tensors left full width, a wide batch's router ids are in
    // range by construction, so the clamp's stated reason disappears. Measured, it does not:
    // lifting it produced a `ntok=16 mode=wide ne2=256` build whose Metal encoder faulted, and the
    // hook at il=2 then reported router ids that are the bit pattern of a NaN. The wide path means
    // Metal reading a 256-expert tensor through the model's weight buffer, which is the exact
    // configuration this engine's loader shrink exists to avoid (docs/M1_POOL_GRAPH_DECOUPLE_PLAN
    // §1.1: `buffer is nil`, and the earlier P1 mmap-stream residency explosion). Keeping the clamp
    // means every step -- decode, MTP verify, prefill chunk -- runs the pool path, which is the only
    // expert-weight path Metal is known to read correctly here. Prefill chunk size therefore remains
    // bounded by the pool capacity, exactly as in the legacy (shrunk-tensor) mode.
    // [CGC M2 prefill streaming 2026-09-14] When CGC_PREFILL_STREAM=1 the whole-layer slab path
    // serves large prefill chunks (expert weights are read straight from the GGUF into a transient
    // Metal slab, ne[2]=n_expert, remap ids are raw expert ids), so the shrunk-tensor OOB reason
    // for the clamp no longer applies. Decode / MTP verify still run n_tokens<=pmax through the
    // pool path; only prefill chunks exceed it. Leave the clamp in place for every other config.
    cgc_stream_on = cgc_prefill_stream_enabled((int64_t) model.hparams.n_expert);
    const bool cgc_prefill_stream = cgc_stream_on;
    if (model.expert_cache_pool_capacity > 0 && cparams.n_batch > 1 && !cgc_prefill_stream) {
        const uint32_t pmax = cgc_pool_max_tokens();
        if (cparams.n_batch > pmax) {
            cparams.n_batch = pmax;
            LLAMA_LOG_INFO("%s: L4 pool capacity=%zu -> n_batch capped to %u (multi-token pool path up to %u tokens; larger batches read shrunk tensors OOB)\n",
                    __func__, model.expert_cache_pool_capacity, pmax, pmax);
        }
    }
    if (cgc_prefill_stream && model.expert_cache_pool_capacity > 0) {
        LLAMA_LOG_INFO("%s: CGC_PREFILL_STREAM=1 -> n_batch NOT capped (prefill chunks use whole-layer slab path)\n", __func__);
    }

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // CGC: multi-command-buffer encoding (CGC_N_CB). Overrides the Metal n_cb to cut command
        // buffer submission overhead (production profile: n_cb = max(1, n_parallel) unless the env
        // overrides). The proc-address form keeps this source decoupled from the Metal headers.
        {
            const char * cgc_n_cb = getenv("CGC_N_CB");
            if (cgc_n_cb && cgc_n_cb[0]) {
                typedef void (*ggml_backend_set_n_cb_t)(ggml_backend_t backend, int n_cb);
                const int n_cb = std::max(1, atoi(cgc_n_cb));
                for (auto & backend : backends) {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
                    if (!reg) {
                        continue;
                    }
                    auto set_n_cb_fn = (ggml_backend_set_n_cb_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_cb");
                    if (set_n_cb_fn) {
                        set_n_cb_fn(backend.get(), n_cb);
                    }
                }
            }
        }

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

// [CGC M1 Metal slab 2026-09-14] Capacity of the gather slab, in experts.
//
// This is deliberately a CONSTANT and not derived from the pool. The pool path's whole correctness
// argument -- and the M1 invariant gate that enforces it -- is that a given routing set is
// combined in the same order no matter how big the pool is. If the slab capacity followed the
// pool, then on any union wide enough to need more than one fill pass the pass boundaries would
// move with the pool too, and each token's sum would become an outer sum whose shape depends on
// the pool size: exactly the failure the gate exists to catch.
//
// 64 is the correct value for M1 because the union ceiling is cap * top_k = 8 * 8 = 64 -- measured,
// not estimated (docs/M1_POOL_GRAPH_DECOUPLE_PLAN_2026-09-14.md 2.1: union max=64 of usable=34 at
// 2 GiB, across layers 1-8). With C = 64 the pass loop of 3.2 is provably a single pass at cap=8.
// The env override exists for A/B only; a larger C is what M2's wider ubatches will need.
static uint32_t cgc_gather_slab_cap() {
    static const uint32_t cap = []() {
        const char * e = getenv("CGC_GATHER_SLAB_CAP");
        const int v = e != nullptr ? atoi(e) : 64;
        return (uint32_t) (v < 2 ? 2 : (v > 1024 ? 1024 : v));
    }();
    return cap;
}

// [CGC M2 2026-09-14] Is the whole-layer prefill stream path usable in this process?
//
// The stream path repoints the expert tensors with ne[2] = n_expert and fills n_expert experts
// into the slab, while the slab's capacity is cgc_gather_slab_cap() experts (default 64, which is
// the M1 wide-union ceiling). Any cap below n_expert means fill_layer_slab writes expert bytes
// past the end of the allocation and mul_mat_id walks past it, silently. Measured: with
// CGC_PREFILL_STREAM=1 and no CGC_GATHER_SLAB_CAP, the driver allocated a 27.50 MiB slab
// (64 x 450,560 B) and CGC-PREFILL-STREAM logged experts=256.
//
// Refusing (rather than aborting or silently raising the cap on the user's behalf) keeps the
// in-process n_batch clamp on, so the run falls back to the pool path: slower, but the numbers
// stay right. The caller sees a loud one-time explanation of the exact mismatch.
static bool cgc_prefill_stream_enabled(int64_t n_expert) {
    const char * e = getenv("CGC_PREFILL_STREAM");
    if (e == nullptr || e[0] == '0') {
        return false;
    }
    const uint32_t cap = cgc_gather_slab_cap();
    if (n_expert > 0 && cap < (uint32_t) n_expert) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "CGC-PREFILL-STREAM: REFUSED -- slab cap %u < n_expert %lld. The stream path "
                    "sets ne[2]=n_expert and fills that many experts, so this slab would be written out "
                    "of bounds (and read past its end). Set CGC_GATHER_SLAB_CAP=%lld (or unset "
                    "CGC_PREFILL_STREAM). Falling back to the pool path, with the n_batch clamp kept on.\n",
                    cap, (long long) n_expert, (long long) n_expert);
        }
        return false;
    }
    return true;
}

llama_context::cgc_gather_slab * llama_context::cgc_gather_slab_get(int kind, size_t exp_bytes,
                                                                 ggml_backend_buffer_type_t buft) {
    if (kind < 0 || kind >= 4 || exp_bytes == 0 || buft == nullptr) {
        return nullptr;
    }
    // ONE slab per kind, sized for the kind's **whole-model maximum stride**.
    //
    // Why the whole model and not "whatever arrived first": these GGUFs are MIXED QUANT, so one
    // kind carries several per-expert sizes across layers (Nail: kind 0/1 are iq2_s|iq3_s|q2_K at
    // 335,872 / 450,560 / 344,064 B, kind 2 is iq3_s|iq4_xs at 450,560 / 557,056 B) -- 8 distinct
    // (kind, stride) combinations inside one 41-layer model. Sizing on first touch (the earlier
    // behaviour) therefore allocated a SECOND slab the first time some layer asked for a bigger
    // stride, and pinned the first one until the context died: measured 157.5 MiB allocated to
    // serve an 89.0 MiB live boundary (docs/M1_POOL_GRAPH_DECOUPLE_PLAN_2026-09-14.md 3.1.1).
    // All layers' expert tensors are captured into cache_ffn_tensors during build_graph, i.e.
    // BEFORE the first eval hook runs, so the model's true worst case is already known here and
    // one allocation is enough.
    //
    // Keyed by kind, NOT by (kind, stride): the latter would allocate a fresh ~30 MiB buffer at
    // nearly every layer boundary and then FREE the one that other layers' live tensors still
    // point into.
    size_t want_stride = exp_bytes;
    if (cache_gather_cur[kind] < 0) {
        for (const auto & it : cache_ffn_tensors) {
            if ((size_t) kind >= it.second.size()) {
                continue;
            }
            const ggml_tensor * t = it.second[(size_t) kind];  // nullptr: kind 3 is absent in these GGUFs
            if (t == nullptr) {
                continue;
            }
            const size_t stride = ggml_row_size(t->type, t->ne[0]) * t->ne[1];
            if (stride > want_stride) {
                want_stride = stride;
            }
        }
    }
    // A larger slab always serves a smaller request: Metal's range check is
    // `ioffs + ggml_nbytes(wt) <= buffers[0].size`, and ggml_nbytes = exp_bytes * union
    // <= cap * slab_stride = the allocated size. So this test is what the per-call exp_bytes
    // still drives.
    if (cache_gather_cur[kind] >= 0) {
        llama_context::cgc_gather_slab & cur = cache_gather_slab[(size_t) cache_gather_cur[kind]];
        if (cur.size >= (size_t) cgc_gather_slab_cap() * exp_bytes) {
            return &cur;
        }
        // Unreachable now that the first allocation covers the whole model. Kept so that a model
        // whose tensor geometry grows after first touch still cannot silently overrun the slab.
        fprintf(stderr, "CGC-GATHER-SLAB: growth after first touch kind=%d from stride=%zu "
                "to %zu -- the model's geometry changed under the slab\n",
                kind, cur.stride, want_stride);
    }

    const size_t want = (size_t) cgc_gather_slab_cap() * want_stride;
    // Lazily allocated, and only for the kinds the wide-union path actually touches, so a config
    // that never takes it (the default 10 GiB pool: the pool path always wins) allocates nothing.
    //
    // The buffer type is the EXPERT TENSOR's own, deliberately not the device default. Metal only
    // hands out host-visible storage when `use_shared_buffers && shared` (ggml-metal-device.m:1666);
    // otherwise `all_data` is a virtual address and the CPU fill below would write into nothing.
    // The expert tensor's buft is the shared one by construction: the pool path CPU-memcpy's into
    // that very region (pool_region -> pool_ext), which is only possible if it is host-visible.
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, want);
    if (buf == nullptr) {
        fprintf(stderr, "CGC-GATHER-SLAB: allocation of %.2f MiB failed kind=%d stride=%zu on %s\n",
                (double) want / (1024.0 * 1024.0), kind, exp_bytes, ggml_backend_buft_name(buft));
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    cache_gather_slab.push_back(llama_context::cgc_gather_slab{});
    llama_context::cgc_gather_slab & s = cache_gather_slab.back();
    s.kind   = kind;
    s.stride = want_stride;
    s.buf    = buf;
    s.base   = (uint8_t *) ggml_backend_buffer_get_base(buf);
    s.size   = want;
    cache_gather_cur[kind] = (int) (cache_gather_slab.size() - 1);
    fprintf(stderr, "CGC-GATHER-SLAB: kind=%d cap=%u stride=%zu (requested=%zu) size=%.2f MiB "
            "base=%p buft=%s\n",
            kind, cgc_gather_slab_cap(), want_stride, exp_bytes,
            (double) want / (1024.0 * 1024.0), (void *) s.base, ggml_backend_buft_name(buft));
    return &s;
}

// [CGC M2 double-buffer 2026-09-14] Background worker: fills one layer's worth of slabs
// (all 4 kinds) into the target set. Runs while the main thread builds the graph for
// the previous layer, hiding the ~40-195ms slab-fill I/O behind graph build/compute.
void llama_context::cgc_db_worker() {
    while (true) {
        int layer = -1, set = -1;
        {
            std::unique_lock<std::mutex> lk(cgc_db_mtx);
            cgc_db_cv.wait(lk, [this]{ return cgc_db_stop || cgc_db_job_layer >= 0; });
            if (cgc_db_stop) return;
            layer = cgc_db_job_layer;
            set   = cgc_db_job_set;
            cgc_db_job_done = false;
        }
        // Fill all 4 kinds for this layer into the target set's slabs.
        // [CGC M2 pool reuse 2026-09-14] Destination pitch = this LAYER's per-expert byte count
        // (the pitch mul_mat_id walks as nb[2]), not the slab's kind-wide max stride.
        for (int kind = 0; kind < 4; ++kind) {
            const int slab_idx = cgc_db_slab[set][kind];
            if (slab_idx < 0) continue;
            cgc_gather_slab & sl = cache_gather_slab[(size_t) slab_idx];
            const size_t sp = ((size_t) layer < cgc_db_stride.size() &&
                               cgc_db_stride[(size_t) layer][(size_t) kind] > 0)
                    ? cgc_db_stride[(size_t) layer][(size_t) kind]
                    : sl.stride;
            llama_expert_cache_fill_layer_slab(
                    model.expert_cache, (uint32_t) layer, kind, sl.base, sp);
        }
        {
            std::lock_guard<std::mutex> lk(cgc_db_mtx);
            cgc_db_job_done  = true;
            cgc_db_job_layer = -1;
        }
        cgc_db_cv.notify_one();
    }
}

void llama_context::cgc_db_start_prefill(int layer, int set) {
    if (!cgc_db_thread.joinable()) {
        cgc_db_thread = std::thread(&llama_context::cgc_db_worker, this);
    }
    {
        std::lock_guard<std::mutex> lk(cgc_db_mtx);
        cgc_db_job_layer = layer;
        cgc_db_job_set   = set;
        cgc_db_job_done  = false;
    }
    cgc_db_cv.notify_one();
}

void llama_context::cgc_db_wait() {
    std::unique_lock<std::mutex> lk(cgc_db_mtx);
    cgc_db_cv.wait(lk, [this]{ return cgc_db_job_done || cgc_db_job_layer < 0; });
}

llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    // [CGC M2 double-buffer] stop background prefill thread
    {
        std::lock_guard<std::mutex> lk(cgc_db_mtx);
        cgc_db_stop = true;
    }
    cgc_db_cv.notify_one();
    if (cgc_db_thread.joinable()) {
        cgc_db_thread.join();
    }

    // [CGC M1 Metal slab 2026-09-14] release the gather slabs. Empty in any config that never took
    // the wide-union gather path, which includes the default 10 GiB pool. This is the ONLY place
    // they are freed: freeing a superseded slab earlier would leave the tensors of other layers
    // pointing into freed memory until the next graph build restores them.
    //
    // [CGC 2026-09-15] UN-REPOINT BEFORE FREEING -- this is a real bug fix, not hygiene.
    //
    // The wide-union path mutates the MODEL's FFN expert tensors in place: it saves the original
    // `data` / `buffer` / `ne[2]` into cache_orig / cache_gather_orig_buf / cache_gather_ne2 and
    // then points them at the slab (`wt->data = sl->base; wt->buffer = sl->buf;`, see the repoint
    // above). The restore is performed by the NEXT per-layer hook that visits the same layer --
    // either the same layer on the following step, or the pool branch that un-repoints whatever
    // this tensor was left in gather state. After the LAST layer of the LAST wide prefill there is
    // no further hook, so those tensors stay pointed into the slab with the originals still sitting
    // unerased in cache_orig. Freeing the slab then leaves every one of them with
    //
    //     data   -> freed slab memory
    //     buffer -> a FREED ggml_backend_buffer_t (dangling pointer)
    //     ne[2]  -> the union size (256), not the slot count
    //
    // The tensors belong to the MODEL, which outlives the context, so the damage is not confined
    // to this context: the next llama_context built from the same model adopts the L4 pool from
    // exactly these tensors and reads `wt->buffer` while doing it -- crashing before it emits a
    // single log line of its own.
    //
    // Measured (llama-bench, which creates and frees one context per (p,n,d) instance):
    //   arm prod25-stream, -p 512 -n 8 -d 0 -r 3   -> pp512 115.84 +- 7.48 t/s, then SIGSEGV
    //   arm prod25-stream-nodb (DB thread off)     -> pp512  78.33 t/s,      still SIGSEGV
    //   arm prod25 (pool path, no slab, no repoint)-> two contexts, no crash
    // The DB thread is therefore not involved; the dangling repoint is. Restoring here also makes
    // the crash impossible for the server, which today only escapes it because it is one long-lived
    // context that is never rebuilt from the same loaded model.
    {
        size_t n_unrepointed = 0;
        for (const auto & kv : cache_orig) {
            const int il = kv.first.first;
            const int k  = kv.first.second;
            auto it_ffn = cache_ffn_tensors.find(il);
            ggml_tensor * wt = (it_ffn != cache_ffn_tensors.end() &&
                                (size_t) k < it_ffn->second.size())
                    ? it_ffn->second[(size_t) k] : nullptr;
            if (wt == nullptr) {
                continue;
            }
            if (wt->data != kv.second) {
                wt->data = kv.second;
                n_unrepointed++;
            }
            if (auto itb = cache_gather_orig_buf.find(kv.first); itb != cache_gather_orig_buf.end()) {
                wt->buffer = itb->second;
            }
            if (auto itn = cache_gather_ne2.find(kv.first); itn != cache_gather_ne2.end()) {
                wt->ne[2] = itn->second;
            }
        }
        if (n_unrepointed > 0) {
            fprintf(stderr, "CGC-M2-UNREPOINT: teardown restored %zu expert tensor(s) to their "
                    "model storage before freeing %zu slab(s) (a second context built from this "
                    "model would otherwise read a freed buffer)\n",
                    n_unrepointed, cache_gather_slab.size());
        }
        cache_orig.clear();
        cache_gather_orig_buf.clear();
        cache_gather_ne2.clear();
    }

    for (llama_context::cgc_gather_slab & s : cache_gather_slab) {
        if (s.buf != nullptr) {
            ggml_backend_buffer_free(s.buf);
            s.buf  = nullptr;
            s.base = nullptr;
        }
    }
    cache_gather_slab.clear();
    for (int k = 0; k < 4; k++) {
        cache_gather_cur[k] = -1;
    }

    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    const bool cgc_gst_dbg = getenv("CGC_SAMPLER_DBG") != nullptr;
    const int64_t g0 = cgc_gst_dbg ? ggml_time_us() : 0;
    output_reorder();
    const int64_t g1 = cgc_gst_dbg ? ggml_time_us() : 0;

    if (!sampling.sampled.has_data()) {
        if (cgc_gst_dbg) fprintf(stderr, "CGC-GST: reorder=%dus no_data\n", (int)(g1 - g0));
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        const int64_t g2 = cgc_gst_dbg ? ggml_time_us() : 0;
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        const llama_token tok = sampling.sampled.data[row];
        if (cgc_gst_dbg) fprintf(stderr, "CGC-GST: reorder=%dus resolve=%dus read=%dus tok=%d\n",
                (int)(g1 - g0), (int)(g2 - g1), (int)(ggml_time_us() - g2), (int) tok);
        return tok;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

// [CGC Phase Discrimination 2026-09-08] Set/get the current decode phase.
// Must be called before llama_decode() for each batch.
// UNKNOWN is the safe default: exact ensure_batch path, no ZERO-slot fast path.
void llama_context::set_cgc_phase(cgc_phase_t phase) {
    cgc_current_phase = phase;
    // [CGC PREFILL_PROTECT A/B 2026-09-13] refresh the runtime toggle once per batch (one
    // open()/read() per decode step) so one server can run both arms of the A/B. No-op unless
    // CGC_PREFILL_PROTECT_FILE is set.
    llama_expert_cache_refresh_prefill_protect();
}

cgc_phase_t llama_context::get_cgc_phase() const {
    return cgc_current_phase;
}

// Public API wrappers (declared in llama-ext.h)
void llama_context_set_cgc_phase(llama_context * ctx, cgc_phase_t phase) {
    ctx->set_cgc_phase(phase);
}

cgc_phase_t llama_context_get_cgc_phase(const llama_context * ctx) {
    return ctx->get_cgc_phase();
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    // [CGC bit-bisect v7] ubatch sequence counter for the in-compute dump (1 = first
    // ubatch = prefill chunk 0 with the chunked-prefill driver config)
    ++cgc_tdcb_ubatch_seq;

    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    // CGC: per-phase decode timing (CGC_PHASE_TIMING=1). Accumulates and prints mean per phase
    // every 32 decode steps; batched (prefill) steps are skipped.
    const bool cgc_phase_timing = getenv("CGC_PHASE_TIMING") != nullptr;
    // [CGC decode phase decomposition 2026-09-15] `compute` is `graph_compute()`, which contains
    // BOTH the Metal encoding of all 40 layers AND the expert-cache fill wait (the hook blocks in
    // ensure_batch/fill_pool_direct inside graph_compute). A step whose whole cost lands in
    // `compute` therefore cannot be attributed to GPU vs disk from this line alone. ph_fillwait is
    // the same window's delta of the calling-thread fill-wait counter, so
    // `compute - fill_wait` = GPU/encoding, `fill_wait` = disk. See
    // llama_expert_cache_fill_wait_us() for why the delta is meaningful only for this thread.
    static int64_t ph_build=0, ph_alloc=0, ph_inputs=0, ph_compute=0, ph_fillwait=0, ph_n=0;
    const int64_t ph_t0 = cgc_phase_timing ? ggml_time_us() : 0;
    int64_t ph_t;

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // [CGC fix 2026-09-08] install the eval wrapper on EVERY ubatch, not only in the
    // fresh-graph branch. sched_reserve() recreates the sched (ggml_backend_sched_new),
    // which drops the callback; a fixed-shape context (MTP draft: constant n_tokens) then
    // reuses its graph forever and never reinstalls it -> expert_cache_on_topk never fires
    // for the draft ctx -> CGC_DRAFT_PREFETCH collect stays empty (loaded=0 across all
    // layers, prediction never populated). Idempotent and cheap (pointer store).
    ggml_backend_sched_set_eval_callback(sched.get(), expert_cache_eval_cb, this);

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        res->reset();

        // [CGC 2026-09-15 S1 slot-table] The captured table is deliberately NOT reset here.
        //
        // An earlier revision cleared it on the fresh-build branch, and that was wrong for a
        // reason worth recording: capture (the build callback) and use (the eval hook) interleave
        // PER LAYER, not per step. The hook for layer k runs before layer k's table is captured, so
        // a per-build reset emptied the map on every step and the hook could never publish
        // anything -- get_rows then gathered from uninitialised memory (measured: every id
        // out-of-bounds, first=987120956, ids ne=[8,1]).
        //
        // The leaf (cache_remap_tensors) has always worked this way and is never reset: the hook
        // publishes into the tensor captured by the PREVIOUS build, which is still correct because
        // ggml-alloc hands a same-shaped graph the same buffer, so the old pointer and the new
        // pointer are the same address. The table must follow exactly the same lifecycle -- and it
        // is built under exactly the same condition as the leaf (one or the other per layer, never
        // both), so the two can never disagree about which steps have a table.
        ggml_backend_sched_reset(sched.get());

        // CGC: restore any FFN expert weights repointed at the cache pool by the previous
        // step's graph, so every freshly built graph starts from the full original weights.
        // decode (n_tokens == 1) re-points them at the pool regions in graph_get_cb
        // (ffn_moe_topk_remap), while prefill computes over the full expert weights.
    for (auto it = cache_orig.begin(); it != cache_orig.end(); ) {
        auto it_ffn = cache_ffn_tensors.find(it->first.first);
        if (it_ffn != cache_ffn_tensors.end() && (size_t) it->first.second < it_ffn->second.size()) {
            ggml_tensor * wt = it_ffn->second[it->first.second];
            if (wt != nullptr) {
                wt->data = it->second;
                if (getenv("CGC_POOL_SPLIT_DBG") != nullptr && strstr(wt->name, "_exps") != nullptr) {
                    fprintf(stderr, "CGC-POOL-SPLIT-RESTORE-data: %s wt=%p data=%p\n",
                            wt->name, (void *) wt, wt->data);
                }
            }
        }
        it = cache_orig.erase(it);
    }

        // [CGC M1 Metal slab 2026-09-14] restore ne[2] for the FFN tensors the gather path
        // resized. Same lifecycle as cache_orig: written during the step's eval hook, put back
        // here before build_graph so the next graph sees the full original expert geometry.
        for (auto it = cache_gather_ne2.begin(); it != cache_gather_ne2.end(); ) {
            auto it_ffn = cache_ffn_tensors.find(it->first.first);
            if (it_ffn != cache_ffn_tensors.end() && (size_t) it->first.second < it_ffn->second.size()) {
                ggml_tensor * wt = it_ffn->second[it->first.second];
                if (wt != nullptr) {
                    wt->ne[2] = it->second;
                    auto it_buf = cache_gather_orig_buf.find(it->first);
                    if (it_buf != cache_gather_orig_buf.end()) {
                        wt->buffer = it_buf->second;
                        cache_gather_orig_buf.erase(it_buf);
                    }
                    if (getenv("CGC_POOL_SPLIT_DBG") != nullptr && strstr(wt->name, "_exps") != nullptr) {
                        fprintf(stderr, "CGC-POOL-SPLIT-RESTORE-geom: %s wt=%p ne2=%lld buf=%p\n",
                                wt->name, (void *) wt, (long long) wt->ne[2], (void *) wt->buffer);
                    }
                }
            }
            it = cache_gather_ne2.erase(it);
        }

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        if (cgc_phase_timing) { ph_t = ggml_time_us(); ph_build += ph_t - ph_t0; }
        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
        if (cgc_phase_timing) { int64_t t = ggml_time_us(); ph_alloc += t - ph_t; ph_t = t; }
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();
        if (cgc_phase_timing) { ph_t = ggml_time_us(); }

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
        if (cgc_phase_timing) { int64_t t = ggml_time_us(); ph_inputs += t - ph_t; ph_t = t; }
    }

    // Hot prewarm: before the first decode step, fill each layer's pool with its top-K most-
    // routed experts from prefill (recorded by the hook via record_routes). One-time sync
    // cold-start preads; a no-op on later steps (hot_prewarm_done). Together with the B
    // async prefetch this keeps the working set warm from the very first decode token.
    if (model.expert_cache_active && (uint32_t) ubatch.n_tokens <= cgc_pool_max_tokens() && getenv("CGC_NO_PREWARM") == nullptr) {
        llama_expert_cache * ec = model.expert_cache;
        if (ec != nullptr && llama_expert_cache_pool_active(ec)) {
            llama_expert_cache_prewarm_hot(ec);
        }
    }

    // [CGC decode phase decomposition 2026-09-15] sample the calling-thread fill-wait counter
    // around graph_compute so the fill wait can be split out of `compute`. The window is exactly
    // graph_compute, which is also exactly the window `compute` measures. Non-owning.
    llama_expert_cache * ec_phase = model.expert_cache;
    const uint64_t ph_fw0 = (cgc_phase_timing && ec_phase != nullptr)
                          ? llama_expert_cache_fill_wait_us(ec_phase) : 0;

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    // [CGC bit-bisect v6] post-compute intermediate tensor dump (see header comment).
    if (getenv("CGC_TENSOR_DUMP") != nullptr) {
        cgc_dump_graph_tensors(res->get_gf());
    }
    if (cgc_phase_timing) {
        int64_t t = ggml_time_us();
        if (ubatch.n_tokens == 1) {
            ph_compute += t - ph_t;
            if (ec_phase != nullptr) {
                const uint64_t fw1 = llama_expert_cache_fill_wait_us(ec_phase);
                if (fw1 >= ph_fw0) {
                    ph_fillwait += (int64_t) (fw1 - ph_fw0);
                }
            }
            ph_n++;
            if (ph_n % 32 == 0) {
                // gpu = compute - fillwait. When fillwait > compute the two windows disagree
                // (the fill counter is monotonic across all callers, and graph_compute also does
                // non-fill work outside the hook), so print it raw and let the reader see the
                // inconsistency rather than clamping it to a fake 0.
                const double cms  = ph_compute/1000.0/ph_n;
                const double fwms = ph_fillwait/1000.0/ph_n;
                fprintf(stderr, "CGC-PHASE: n=%lld build=%.3f alloc=%.3f inputs=%.3f compute=%.3f"
                                " fill_wait=%.3f gpu=%.3f ms\n",
                        (long long) ph_n, ph_build/1000.0/ph_n, ph_alloc/1000.0/ph_n,
                        ph_inputs/1000.0/ph_n, cms, fwms, cms - fwms);
            }
        }
    }

    // B: async-prefetch predicted next-step experts into the pool's bg thread so the NEXT decode
    // step's ensure_batch mostly hits (temporal locality). prefetch_slot is non-blocking
    // (free-slot-only / LRU-evict-a-non-union-slot) and overlaps the disk IO with the sampler +
    // next step's whole GPU window instead of stalling the hook on synchronous preads.
    // [CGC MTP fix] allow multi-token pool steps (speculative/MTP verify) too.
    // [CGC prefetch fix] source selection (CGC_PREFETCH_SRC):
    //   - default / "step": the OLD behaviour — this step's union, which the hook just ensured
    //     (all resident) -> prefetch_slot drops everything (prefetch=0/0), so the 2.1% cold
    //     misses still blocked the hook for ~167us/layer. Kept for A/B.
    //   - "prev": the previous step's union — partially LRU-evicted, but with a 5GiB pool most
    //     is still resident, so little to prefetch.
    //   - "hist" (recommended): a rolling window (CGC_PREFETCH_WINDOW=N, default 4) of recent
    //     step unions per layer. Miss analysis: 1062 misses = only 80 distinct experts, 72 of
    //     them repeated (e243 missed 28x) — i.e. recurring hot experts evicted by LRU between
    //     uses. Prefetching the recent window's non-resident members re-residents exactly those
    //     before the next step needs them.
    // NOTE: prefetch_slot was previously DEAD CODE — bg_loop checked slot_loading (never set to
    // 1) instead of slot_queued, so every queued prefetch was dropped as stale. Fixed in
    // llama-expert-cache.cpp bg_loop. drain_layer still drops not-yet-started (slot_queued)
    // fills for a layer whose hook fires before the bg thread reached them — inherent to the
    // single-buffer window; hist reduces the misses that remain.
    // CGC_NO_PREFETCH=1 disables the whole prefetch path (MTP verify safety: the bg thread
    // filling/evicting slots while the GPU is mid-verify overwrites slots still being read).
    static const char * pf_src_env = getenv("CGC_PREFETCH_SRC");
    static const bool pf_prev = pf_src_env != nullptr && strcmp(pf_src_env, "prev") == 0;
    static const bool pf_hist = pf_src_env != nullptr && strcmp(pf_src_env, "hist") == 0;
    // [CGC SpAc 2026-09-06] when CGC_SPAC=1 the EMA-utility source REPLACES the step/hist/prev
    // sources below (mutually exclusive — spac_prefetch re-targets each layer toward its
    // utility top-K every CGC_SPAC_REFRESH B-sections; the default sources stay dormant so the
    // A/B isolates the estimator's effect). CGC_SPAC_REFRESH=1 (default) = every B-section.
    static const bool spac_on     = cgc_spac_on();
    static const uint32_t spac_refresh = cgc_spac_refresh();
    static const size_t pf_win = []() {
        const char * w = getenv("CGC_PREFETCH_WINDOW");
        size_t v = 4;
        if (w != nullptr && w[0] != '\0') {
            v = (size_t) atoi(w);
            if (v < 1) v = 1;
            if (v > 16) v = 16;
        }
        return v;
    }();
    // [CGC SpAc EMA membership 2026-09-08] the EMA refresh is EXEMPT from CGC_NO_PREFETCH:
    // NO_PREFETCH stops the step/hist union bg prefetch (MTP-verify race mitigation), but the
    // EMA refresh is the pool's routing-driven membership driver — without it the pool stays at
    // the loader's identity-ordered prepopulate and decode's working set never becomes resident
    // (measured ~44% count-cold at 143 slots -> ZERO-slot logits collapse -> degenerate loops).
    // prefetch_slot publishes slot_table only after bytes land and its LRU victim is by
    // construction not in the current step's union (union members were just LRU-touched), so the
    // refresh cannot corrupt an in-flight remap. Runs every CGC_SPAC_REFRESH routed steps.
    if (model.expert_cache_active && (uint32_t) ubatch.n_tokens <= cgc_pool_max_tokens() &&
            (getenv("CGC_NO_PREFETCH") == nullptr || spac_on)) {
        llama_expert_cache * ec = model.expert_cache;
        if (ec != nullptr && llama_expert_cache_pool_active(ec)) {
            if (spac_on) {
                static uint64_t spac_tick = 0;
                if ((spac_tick++ % spac_refresh) == 0) {
                    const size_t spac_q = llama_expert_cache_spac_prefetch(ec);
                    if (getenv("CGC_SPAC_DBG") != nullptr) {
                        static int spac_dbg_n = 0;
                        if (spac_dbg_n++ < 40) {
                            fprintf(stderr, "CGC-SPAC: feeds=%llu queued=%zu n_prefetch=%zu dropped=%zu\n",
                                    (unsigned long long) ec->spac_feeds, spac_q,
                                    ec->n_prefetch, ec->n_prefetch_dropped);
                        }
                    }
                }
            } else if (pf_hist) {
                // roll the current step's union into the per-layer history window (do this
                // before the rotate below clears cache_step_union), then prefetch the window.
                if (cache_tail_union.size() < (size_t) model.hparams.n_layer_all) {
                    cache_tail_union.resize(model.hparams.n_layer_all);
                }
                for (size_t il = 0; il < cache_step_union.size() && il < (size_t) model.hparams.n_layer_all; ++il) {
                    auto & dq = cache_tail_union[il];
                    if (!cache_step_union[il].empty()) {
                        dq.push_back(cache_step_union[il]);
                        while (dq.size() > pf_win) {
                            dq.pop_front();
                        }
                    }
                    if (getenv("CGC_TAIL_DBG") != nullptr) {
                        static int dbg_n = 0;
                        if (il == 5 && dbg_n++ < 4) {
                            fprintf(stderr, "TAILDBG layer=%zu dq=%zu", il, dq.size());
                            for (const auto & u : dq) {
                                fprintf(stderr, " u%zu", u.size());
                            }
                            fprintf(stderr, "\n");
                        }
                    }
                    for (const auto & u : dq) {
                        for (uint32_t e : u) {
                            llama_expert_cache_prefetch_slot(ec, (uint32_t) il, e);
                        }
                    }
                    // [CGC prefetch v2] also re-prefetch the recently-evicted experts (ring
                    // recorded by pick_slot on LRU eviction): recurring hot experts that LRU
                    // dropped between uses. Re-resident before the next ensure so it HITs
                    // instead of preading. CGC_EVICTED_RING=0 disables (A/B control).
                    // NOTE: iterate a COPY — prefetch_slot can evict another slot (and thus
                    // push_back into this same ring) while holding cache->m, which would
                    // invalidate a live range-for over the vector.
                    if (il < ec->evicted_recent.size() && !ec->evicted_recent[il].empty()) {
                        const std::vector<uint32_t> ev_snap = ec->evicted_recent[il];
                        for (uint32_t e : ev_snap) {
                            llama_expert_cache_prefetch_slot(ec, (uint32_t) il, e);
                        }
                    }
                }
            } else {
                const auto & pf_src = pf_prev ? cache_prev_union : cache_step_union;
                if (!pf_src.empty()) {
                    const size_t n_l = (size_t) model.hparams.n_layer();
                    for (size_t il = 0; il < pf_src.size() && il < n_l; ++il) {
                        const auto & v = pf_src[il];
                        if (v.empty()) {
                            continue;
                        }
                        for (uint32_t e : v) {
                            llama_expert_cache_prefetch_slot(ec, (uint32_t) il, e);
                        }
                    }
                }
            }
        }
        // rotate: this step becomes the previous step; clear the buffer for the next build.
        cache_prev_union.swap(cache_step_union);
        if (cache_step_union.size() < (size_t) model.hparams.n_layer_all) {
            cache_step_union.resize(model.hparams.n_layer_all);
        }
        for (auto & v : cache_step_union) {
            v.clear();
        }
    }

    ret = GGML_STATUS_SUCCESS;

    // [CGC V2 logits oracle 2026-09-05] per-ubatch logits summary (see header comment for
    // JSONL format / env vars). Env-gated; one line of caller code, the function no-ops when
    // CGC_LOGITS_ORACLE_DUMP is unset.
    if (getenv("CGC_LOGITS_ORACLE_DUMP") != nullptr) {
        cgc_logits_oracle_dump(res->get_gf(), (uint32_t) ubatch.n_tokens, (uint32_t) ubatch.n_tokens);
    }

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_embd = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    // [CGC bit-bisect v5 TEMP] trace every decode entry: who decodes what, when.
    // Used to locate a hidden 2-token decode on ctx_tgt that runs before prefill.
    if (getenv("CGC_DECODE_TRACE")) {
        fprintf(stderr, "CGC-DEC: ctx=%p type=%s n_tokens=%d pos0=%d tok0=%d tok1=%d embd=%p\n",
                (void *) this,
                cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "MTP" : "DEF",
                batch_inp.n_tokens,
                batch_inp.pos ? (int) batch_inp.pos[0] : -1,
                batch_inp.token ? (int) batch_inp.token[0] : -1,
                (batch_inp.token && batch_inp.n_tokens > 1) ? (int) batch_inp.token[1] : -1,
                (void *) batch_inp.embd);
        fflush(stderr);
    }

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const bool    mtp_embd = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && batch_inp.embd;
    const int64_t n_embd  = mtp_embd ? hparams.n_embd_out() : hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // embedding contexts output every token even when batch.logits is not set
    if (has_samplers && (output_all || batch_inp.logits)) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (!output_all && batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
            }
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (embd_layer_inp.size() > 0) {
            for (int lid = 0; lid < (int) embd_layer_inp.size(); ++lid) {
                if (embd_layer_inp[lid].size > 0) {
                    for (uint64_t k = 0; k < n_embd; ++k) {
                        std::swap(embd_layer_inp[lid].data[i0*n_embd + k], embd_layer_inp[lid].data[i1*n_embd + k]);
                    }
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_M3) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    uint32_t n_sampling_nodes = 0;
    uint32_t n_sampling_nodes_max = 0;
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler);
        n_sampling_nodes += n_nodes;
        if (cparams.n_outputs_max_per_seq > 1) {
            n_sampling_nodes_max = std::max(n_sampling_nodes_max, n_nodes);
        }
    }

    const uint32_t n_sampling_outputs_max = std::min<uint64_t>(
            std::min(n_tokens, cparams.n_outputs_max),
            (uint64_t) cparams.n_seq_max * cparams.n_outputs_max_per_seq);

    res += n_sampling_nodes;
    if (n_sampling_outputs_max > 1) {
        res += (n_sampling_outputs_max - 1) * n_sampling_nodes_max;
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.expert_cache_active =*/ model.expert_cache_active,
        /*.n_gpu_layers =*/ model.n_gpu_layers(),
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    // [CGC Step-3 renormalized routing] rewrite every layer's rn-mask leaf (0.0 warm / -inf
    // cold per the current slot_table) before dispatch. The leaves are built only under
    // CGC_RN_ROUTING=1 (llama-graph build_moe_ffn); without them this is a no-op. Masks reflect
    // the slot state at step start (end of previous step); fills made by ensure_batch mid-step
    // only take effect from the next step, which keeps this simple and race-free (no segment
    // submitted yet at this point).
    if (!cache_rn_mask_tensors.empty() && model.expert_cache != nullptr) {
        llama_expert_cache * rnc = model.expert_cache;
        const int32_t rn_k = (int32_t) model.hparams.n_expert_used;
        for (const auto & kv : cache_rn_mask_tensors) {
            const int rn_il = kv.first;
            ggml_tensor * rn_m = kv.second;
            if (rn_m == nullptr || rn_m->data == nullptr ||
                rn_il < 0 || (uint32_t) rn_il >= rnc->slot_owner.size() ||
                (size_t) rn_il * rnc->n_expert + rnc->n_expert > rnc->slot_table.size()) {
                continue;
            }
            const uint32_t rn_ne = (uint32_t) rn_m->ne[0];
            if (rn_ne == 0 || rn_ne > rnc->n_expert) {
                continue;
            }
            const int32_t * rn_st = rnc->slot_table.data() + (size_t) rn_il * rnc->n_expert;
            float * rn_out = (float *) rn_m->data;
            int32_t rn_warm = 0;
            for (uint32_t e = 0; e < rn_ne; ++e) {
                if (rn_st[e] >= 0) {
                    rn_warm++;
                }
            }
            // safety floor: masking needs >= n_expert_used warm experts so top-k still has real
            // candidates. Fewer warm -> all-zero mask (softmax unmasked; the cold guard below
            // still protects via ensure_batch).
            // [CGC RN bisect 2026-09-06] CGC_RN_NOOP=1 writes all-zero masks (mechanism runs:
            // leaf + ADD + host write each step, but no expert excluded) — isolates a broken
            // mask/buffer path from "renorm-by-exclusion is harmful" (measured: RN quality is
            // deterministic 0.3 at 4/8/10GiB pool alike, even at ~99% residency).
            static const bool rn_dbg = getenv("CGC_RN_DBG") != nullptr;
            static const bool rn_noop = getenv("CGC_RN_NOOP") != nullptr;
            if (rn_warm < rn_k || rn_noop) {
                memset(rn_out, 0, rn_ne * sizeof(float));
                if (rn_dbg && rn_il <= 2) {
                    fprintf(stderr, "CGC-RN-FLOOR: il=%d warm=%d < k=%d -> no mask%s\n", rn_il, (int) rn_warm, (int) rn_k, rn_noop ? " (NOOP)" : "");
                }
            } else {
                for (uint32_t e = 0; e < rn_ne; ++e) {
                    rn_out[e] = (rn_st[e] >= 0) ? 0.0f : -INFINITY;
                }
                if (rn_dbg && rn_il <= 2) {
                    fprintf(stderr, "CGC-RN-MASK: il=%d warm=%d/%u masked=%u\n", rn_il, (int) rn_warm, rn_ne, rn_ne - (uint32_t) rn_warm);
                }
            }
        }
    }

    // [CGC remap write-vs-alloc bisect 2026-09-09] read back what the GPU will ACTUALLY
    // consume from the remap leaves at dispatch time (post-alloc). The hook writes the leaves
    // during build_graph, BEFORE ggml_backend_sched_alloc_graph reassigns input data
    // pointers — if alloc moved remap->data, the dispatch buffer holds garbage and the FFN
    // reads wrong experts from the first chunk (oracle divergence at [0,0,DEF]).
    static const bool cgc_rmap_dbg = getenv("CGC_REMAP_DBG") != nullptr;
    if (cgc_rmap_dbg) {
        llama_expert_cache * ec = model.expert_cache;
        for (int il = 0; il < 3 && il < (int) model.hparams.n_layer(); il++) {
            const char * rc_ctx = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "MTP" : "DEF";
            ggml_tensor * r = cache_remap_tensors[il];
            if (r == nullptr) {
                fprintf(stderr, "CGC-REMAP-READBACK: %s il=%d remap=null\n", rc_ctx, il);
                continue;
            }
            if (r->data == nullptr) {
                fprintf(stderr, "CGC-REMAP-READBACK: %s il=%d remap-data=null\n", rc_ctx, il);
                continue;
            }
            const int32_t * rd = (const int32_t *) r->data;
            const int64_t nv = r->ne[0] * r->ne[1] < 16 ? r->ne[0] * r->ne[1] : 16;
            fprintf(stderr, "CGC-REMAP-READBACK: %s il=%d data=%p ntok=%lld vals=[", rc_ctx, il,
                    (const void *) r->data, (long long) r->ne[1]);
            for (int64_t k = 0; k < nv; ++k) {
                fprintf(stderr, "%s%d", k ? " " : "", rd[k]);
            }
            fprintf(stderr, "]\n");
            // FFN weight tensor data ptr vs pool region base (L4 adoption)
            auto & ffn = cache_ffn_tensors[il];
            for (int kind = 0; kind < 4; ++kind) {
                ggml_tensor * wt = (size_t) kind < ffn.size() ? ffn[kind] : nullptr;
                if (wt == nullptr) {
                    continue;
                }
                size_t stride = 0;
                uint32_t slots = 0;
                const uint8_t * region = nullptr;
                if (ec != nullptr) {
                    region = llama_expert_cache_pool_data(ec, (uint32_t) il, kind);
                    stride = llama_expert_cache_pool_stride(ec, (uint32_t) il, kind);
                    slots = llama_expert_cache_slots_per_layer_l(ec, (uint32_t) il);
                }
                fprintf(stderr, "CGC-FFN-PTR: il=%d kind=%d wt->data=%p ne=[%lld %lld %lld] nb2=%lld region=%p stride=%zu slots=%u match=%d\n",
                        il, kind, (const void *) wt->data,
                        (long long) wt->ne[0], (long long) wt->ne[1], (long long) wt->ne[2],
                        (long long) wt->nb[2], (const void *) region, stride, slots,
                        region != nullptr && wt->data == region ? 1 : 0);
            }
        }
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // [CGC remap post-compute bisect 2026-09-09] AFTER the graph ran, read back what mul_mat_id
    // ACTUALLY consumed from the remap leaves. The pre-dispatch readback above runs before
    // ggml_backend_sched_alloc_graph reassigns input data pointers AND before the ggml_cont copy
    // node executes — if alloc moved remap->data, or the cont op overwrites the hook-written slot
    // ids with the raw expert ids (0..255 on a 143-slot tensor), the post-compute values diverge
    // from the pre-dispatch ones and the FFN reads wrong/OOB experts from the first chunk.
    static const bool cgc_rmap_post = getenv("CGC_REMAP_POST_DBG") != nullptr;
    if (cgc_rmap_post) {
        ggml_backend_sched_synchronize(sched.get());
        const int nq = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? 3 : 3;
        const int il_base = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? (int) model.hparams.n_layer() : 0;
        for (int qi = 0; qi < nq; qi++) {
            const int il = il_base + qi;
            if (il >= (int) model.hparams.n_layer_all) continue;
            const char * rc_ctx = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "MTP" : "DEF";
            ggml_tensor * r = cache_remap_tensors[il];
            if (r == nullptr || r->data == nullptr) {
                fprintf(stderr, "CGC-REMAP-POST: %s il=%d remap=null\n", rc_ctx, il);
                continue;
            }
            const int64_t ntot = r->ne[0] * r->ne[1];
            std::vector<int32_t> buf((size_t) ntot);
            ggml_backend_tensor_get(r, buf.data(), 0, (size_t) ntot * sizeof(int32_t));
            const int64_t nv = ntot < 16 ? ntot : 16;
            fprintf(stderr, "CGC-REMAP-POST: %s il=%d ntok=%lld vals=[", rc_ctx, il, (long long) r->ne[1]);
            for (int64_t k = 0; k < nv; ++k) {
                fprintf(stderr, "%s%d", k ? " " : "", buf[(size_t) k]);
            }
            fprintf(stderr, "]\n");
        }
    }

    // [CGC 2026-09-15 S1 slot-table] The decisive readback: what the Metal GET_ROWS actually
    // produced, read on the host AFTER ggml_backend_sched_synchronize(). CGC-MMID-ASSERT reads the
    // same buffer at encode time (before the command buffer runs) and therefore reports the
    // allocator's leftovers for any GPU-computed id vector -- F32 router probabilities, +NaN, or a
    // mix of legal and illegal indices inside a single node, which is exactly what it reported.
    // Comparing the gather's output against the host leaf for the SAME step splits the two
    // remaining possibilities cleanly:
    //   equal    -> the GPU path is correct and every id_oob line was a probe artifact; the logits
    //               divergence then has a different cause and the gate stays the only judge.
    //   differs  -> the gather really produced wrong ids, and the printed pair says by how much.
    static const bool cgc_s1_post = getenv("CGC_S1_DBG") != nullptr;
    if (cgc_s1_post) {
        static int cgc_s1_post_n = 0;
        if (cgc_s1_post_n < 6) {
            cgc_s1_post_n++;
            ggml_backend_sched_synchronize(sched.get());
            // [CGC 2026-09-15 S1 window fix] This loop used to be `for (int il = 0; il < 4; ++il)`,
            // i.e. it read back layers 0..3 only, while the divergence the ladder localized sits at
            // layers 10..19 -- so the probe kept confirming a mapping nobody had questioned. It now
            // walks every captured `ffn_moe_slots` node, which is what "the gather is correct" was
            // always supposed to mean. Only the readback cost changes: these are host reads after a
            // synchronize that already happened, on a diagnostic arm whose throughput nobody quotes.
            for (const auto & cgc_kv : cache_slots_out_tensors) {
                const int il = cgc_kv.first;
                ggml_tensor * s  = cgc_kv.second;
                ggml_tensor * tb = cache_slot_table_tensors.count(il) ? cache_slot_table_tensors[il] : nullptr;
                ggml_tensor * rm = cache_remap_tensors.count(il) ? cache_remap_tensors[il] : nullptr;
                if (s == nullptr || s->data == nullptr) {
                    fprintf(stderr, "CGC-S1: POST il=%d gather=null (leaf path)\n", il);
                    continue;
                }
                const int64_t ntot = s->ne[0] * s->ne[1];
                std::vector<int32_t> sbuf((size_t) ntot);
                ggml_backend_tensor_get(s, sbuf.data(), 0, (size_t) ntot * sizeof(int32_t));
                std::vector<int32_t> rbuf;
                if (rm != nullptr && rm->data != nullptr && rm->ne[0] == s->ne[0] && rm->ne[1] == s->ne[1]) {
                    rbuf.resize((size_t) ntot);
                    ggml_backend_tensor_get(rm, rbuf.data(), 0, (size_t) ntot * sizeof(int32_t));
                }
                // Then attempt the complete differential: read the index vector and the whole table
                // on the host, recompute table[idx[j]] for every j, and compare against the gather's
                // own output.
                //
                // IT ONLY MEANS SOMETHING WHEN ids_src_valid=1, and that is the whole point of this
                // comment. `ids_cont` (the CONT that feeds the gather) is NOT an output, so
                // ggml-alloc may recycle its buffer as soon as the gather has consumed it. Reading it
                // AFTER the graph completed therefore reads whatever tensor took the buffer next --
                // measured: layer 1 read small ints, layers 2 and 3 read F32 bit patterns
                // (idx=1043923934), producing 16/16 "mismatches" that are an artifact of the probe,
                // not of the gather. This is the MIRROR of the encode-time race: that one read the
                // buffer too early, this one reads a dead buffer too late. The two valid ways to
                // compare a transient against a GPU product are to PIN it (ggml_set_output, which
                // perturbs the graph and so is not acceptable on a gate arm) or to sample it at the
                // right moment -- which is what the hook-side `CGC-S1: EXPECT-*` lines already do.
                // Cross-referencing EXPECT (what the table holds for this step's ids, read at hook
                // time) with POST (the gather output, read post-sync) is what actually closes the
                // loop; see docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md §8.2.
                ggml_tensor * ids_src = (s->view_src != nullptr) ? s->view_src->src[1] : nullptr;
                const void * ids_src_data = ids_src != nullptr ? ids_src->data : nullptr;
                std::vector<int32_t> ibuf;
                std::vector<int32_t> tbuf;
                if (ids_src != nullptr && ids_src->data != nullptr &&
                        ggml_nelements(ids_src) == ntot) {
                    ibuf.resize((size_t) ntot);
                    ggml_backend_tensor_get(ids_src, ibuf.data(), 0, (size_t) ntot * sizeof(int32_t));
                }
                if (tb != nullptr && tb->data != nullptr) {
                    tbuf.resize((size_t) tb->ne[1]);
                    ggml_backend_tensor_get(tb, tbuf.data(), 0, (size_t) tb->ne[1] * sizeof(int32_t));
                }
                int ids_src_valid = 0;
                for (int32_t v : ibuf) {
                    if (v < 0 || v >= (int32_t) (tb != nullptr ? tb->ne[1] : 0)) {
                        ids_src_valid = 0;
                        break;
                    }
                    ids_src_valid = 1;
                }                int64_t n_gather_mis = -1;
                if (ids_src_valid == 1 && !tbuf.empty()) {
                    n_gather_mis = 0;
                    for (int64_t k = 0; k < ntot; ++k) {
                        if (tbuf[(size_t) ibuf[(size_t) k]] != sbuf[(size_t) k]) {
                            if (n_gather_mis < 4) {
                                fprintf(stderr, "CGC-S1: POST il=%d MISMATCH k=%lld idx=%d table=%d gather=%d\n",
                                        il, (long long) k, ibuf[(size_t) k],
                                        tbuf[(size_t) ibuf[(size_t) k]], sbuf[(size_t) k]);
                            }
                            n_gather_mis++;
                        }
                    }
                }
                const int64_t nv = ntot < 16 ? ntot : 16;
                fprintf(stderr, "CGC-S1: POST il=%d ntok=%lld n_expert=%lld gather=[",
                        il, (long long) s->ne[1], tb != nullptr ? (long long) tb->ne[1] : -1);
                for (int64_t k = 0; k < nv; ++k) {
                    fprintf(stderr, "%s%d", k ? " " : "", sbuf[(size_t) k]);
                }
                fprintf(stderr, "] leaf=[");
                if (rbuf.empty()) {
                    fprintf(stderr, "n/a");
                } else {
                    for (int64_t k = 0; k < nv; ++k) {
                        fprintf(stderr, "%s%d", k ? " " : "", rbuf[(size_t) k]);
                    }
                }
                char vtb[64];
                if (ids_src_valid == 1) {
                    snprintf(vtb, sizeof(vtb), "%lld", (long long) n_gather_mis);
                } else {
                    snprintf(vtb, sizeof(vtb), "n/a (index vector buffer recycled)");
                }
                fprintf(stderr, "] gather_data=%p ids_src_data=%p table_data=%p same=%d ids_src_valid=%d gather_vs_table=[%s]%s\n",
                        s->data, ids_src_data, tb != nullptr ? tb->data : nullptr,
                        (!rbuf.empty() && memcmp(sbuf.data(), rbuf.data(), (size_t) ntot * sizeof(int32_t)) == 0) ? 1 : 0,
                        ids_src_valid, vtb,
                        (s->data != nullptr && s->data == ids_src_data)
                                ? " *** mmid consumes the INDEX vector, not the gather output ***" : "");
            }
        }
    }

    // [CGC 2026-09-15 S1 output capture] The layer's MoE FFN output, read on the host after a
    // synchronize. This is the probe §9.18 ended on: on a shared input the ids are bit-identical AND
    // the lookup that produced them is bit-identical, so if this output still differs between the
    // host-leaf arm and the GPU-table arm, the carrier is the WEIGHT CONTENT those ids point at
    // (pool/slot residency) -- a pool-fill repair -- and not "the table disagreed with the leaf at
    // consume time", which would be a publish-order repair. Those two demand opposite fixes and
    // only the output tells them apart. The tensor is PINNED in graph_get_cb (see
    // cache_down_out_tensors) precisely so this reads a live buffer rather than whatever the
    // allocator put there next.
    //
    // Gated on CGC_S1_OUT_CAP, NOT on CGC_S1_DBG. It was written inside the CGC_S1_DBG block first,
    // and the consequence is worth keeping written down: the outcap arms captured and pinned the
    // tensor and then printed nothing, because the knob that armed the capture was not the knob that
    // gated the readback. The run looked like a probe that worked, and an empty log reads as "the
    // outputs agree" -- which is the opposite of the truth and would have been believed.
    //
    // [CGC 2026-09-15 second fix] Gated on `!batched` (decode) as well, and it prints a decode-graph
    // ordinal. Without that this loop spent its entire budget -- the counter is capped -- on the
    // first few graphs of the process, which are warmup and prefill: the arms would have been
    // compared at a graph where the expert-cache hook takes a different branch (prefill builds no
    // remap leaf), i.e. at a graph where the mechanism under test is not even running. "Graph 3" is
    // only a meaningful address if graphs are numbered inside the phase that has the question.
    // It also reports which PHASE it is reading, because the two phases answer different questions
    // and only one of them is confounded. A decode capture cannot localize anything: decode step 0's
    // layer-0 input is derived from the KV cache the PREFILL produced, and the two arms' prefills
    // already disagree (the M1 numeric gate is 5/9), so every decode layer is downstream of a
    // divergence that happened earlier. A prefill capture is the clean ladder -- at layer 0 the input
    // is the embedding, which is identical by construction, so the FIRST layer at which the arms
    // differ names the carrier directly. `CGC_S1_OUT_CAP=pre` selects it, and it pins only the layers
    // named by CGC_S1_OUT_LAYERS (default "0") because a prefill ffn_moe_down is [2048, 8, ntok] --
    // 16 MB per layer at a 250-token prompt, but 402 MB per layer at a 6144-token chunk.
    //
    // [CGC 2026-09-15 RESULT, two rungs of 0-3 and 4-7] The ladder came back layer-INDEPENDENT, which
    // is the opposite of the shape it was built to find. Across eight tested layers and six prefill
    // graphs per arm, the partition is exactly by SERVICE PATH and not by depth:
    //
    //   ntok=182 (real prompt)   served by CGC-PREFILL-STREAM, 256/256 resident -> EQ on layers 0-7
    //   ntok=22  (asst prefix)   served by CGC-PREFILL-STREAM, 256/256 resident -> EQ on layers 0-7
    //   ntok=2   (warmup x2)     served by the pool,              143/256 resident -> NE on 1-7
    //   ntok=4   (KV reuse x2)   served by the pool,              143/256 resident -> NE on 1-7
    //
    // The log's own `CGC-PREFILL-STREAM` lines are the marker, and they appear for ntok=182 and 22
    // and for nothing else -- the marker and the outcome coincide on all 96 (graph, layer, arm)
    // triples. Layer 0 never diverges because it is the UNPOOLED layer (identity map; the pool logs
    // L0=0), so "the first diverging layer" is really "the first POOLED layer". Two consequences for
    // whoever reads this next: a deeper window will not find a different answer -- the variable is
    // the path, not the index, so rungs below 8-11 are lower value than they look; and both of
    // §9.18's candidate carriers are now excluded on evidence (ids equal per the kernel capture;
    // the table equals the leaf on every consumed id, clamped_selected=0), so the next measurement
    // must be on the slot CONTENT behind the consumed ids, not on the mapping that names them.
    static const char * cgc_s1_outcap_mode = getenv("CGC_S1_OUT_CAP");
    static const bool cgc_s1_outcap  = cgc_s1_outcap_mode != nullptr;
    static const bool cgc_s1_outpre  = cgc_s1_outcap && strcmp(cgc_s1_outcap_mode, "pre") == 0;
    static long long cgc_s1_out_graph = 0;
    static long long cgc_s1_pre_graph = 0;
    if (cgc_s1_outcap && (cgc_s1_outpre ? batched : !batched)) {
        const long long cgc_out_graph = cgc_s1_outpre ? cgc_s1_pre_graph++ : cgc_s1_out_graph++;
        if (cgc_out_graph < 12) {
            ggml_backend_sched_synchronize(sched.get());
            for (const auto & cgc_kv : cache_down_out_tensors) {
                const int il = cgc_kv.first;
                ggml_tensor * d = cgc_kv.second;
                if (d == nullptr || d->data == nullptr) {
                    continue;
                }
                const int64_t ntot = ggml_nelements(d);
                std::vector<float> dbuf((size_t) ntot);
                ggml_backend_tensor_get(d, dbuf.data(), 0, (size_t) ntot * sizeof(float));
                // fnv1a64 over the raw bytes: a whole-tensor verdict that does not depend on how
                // many values we choose to print. Printed as 16 hex digits so two arms can be
                // diffed by eye as well as by script.
                uint64_t h = 1469598103934665603ULL;
                const uint8_t * raw = (const uint8_t *) dbuf.data();
                for (size_t b = 0; b < (size_t) ntot * sizeof(float); ++b) {
                    h = (h ^ raw[b]) * 1099511628211ULL;
                }
                const int64_t nv = ntot < 8 ? ntot : 8;
                fprintf(stderr, "CGC-S1: OUT phase=%s graph=%lld il=%d ne=[%lld %lld %lld] fnv1a64=%016llx vals=[",
                        cgc_s1_outpre ? "pre" : "dec", cgc_out_graph, il,
                        (long long) d->ne[0], (long long) d->ne[1], (long long) d->ne[2],
                        (unsigned long long) h);
                for (int64_t k = 0; k < nv; ++k) {
                    fprintf(stderr, "%s%.6g", k ? " " : "", (double) dbuf[(size_t) k]);
                }
                fprintf(stderr, "]\n");
                // [CGC 2026-09-15 S1 output capture, row split] Per-expert-row hashes plus a hash of
                // the SORTED row-hash multiset.
                //
                // Why the whole-tensor hash above is not enough to close this: ffn_moe_down comes out
                // of mul_mat_id as [n_embd, n_expert_used, n_tokens], i.e. n_expert_used SEPARATE
                // expert contributions that are weighted and summed only later. Two arms that agree
                // on the weighted SUM can disagree on this tensor in two completely different ways --
                //   (a) the same per-expert values arranged in a different ORDER (the top-k ids
                //       vector is permuted, or the ids map to different rows), which is invisible in
                //       the layer output except for float summation order, and
                //   (b) genuinely different values (the ids resolve to different WEIGHTS),
                // and (a) and (b) demand opposite repairs: reorder the ids vector vs fix what the ids
                // point at. `set` is the hash over the sorted per-row hashes, so set==set with
                // fnv1a64!=fnv1a64 is exactly case (a), and set!=set is case (b).
                //
                // This also explains why it is worth splitting at all: the arms' router logits were
                // measured identical while their MoE outputs differ, which is only self-consistent if
                // the difference can be order-insensitive. The row split is the instrument that
                // decides whether the order-insensitivity is where the divergence lives.
                {
                    const int64_t nrows = d->ne[1] * d->ne[2];
                    const int64_t per_row = d->ne[0];
                    std::vector<uint64_t> rowh((size_t) nrows, 0);
                    for (int64_t r = 0; r < nrows; ++r) {
                        uint64_t rh = 1469598103934665603ULL;
                        const uint8_t * rb = (const uint8_t *) (dbuf.data() + (size_t) r * (size_t) per_row);
                        for (size_t b = 0; b < (size_t) per_row * sizeof(float); ++b) {
                            rh = (rh ^ rb[b]) * 1099511628211ULL;
                        }
                        rowh[(size_t) r] = rh;
                    }
                    std::vector<uint64_t> srt = rowh;
                    std::sort(srt.begin(), srt.end());
                    uint64_t sh = 1469598103934665603ULL;
                    const uint8_t * sb = (const uint8_t *) srt.data();
                    for (size_t b = 0; b < srt.size() * sizeof(uint64_t); ++b) {
                        sh = (sh ^ sb[b]) * 1099511628211ULL;
                    }
                    fprintf(stderr, "CGC-S1: OUTSET phase=%s graph=%lld il=%d nrows=%lld set=%016llx rows=[",
                            cgc_s1_outpre ? "pre" : "dec", cgc_out_graph, il,
                            (long long) nrows, (unsigned long long) sh);
                    for (int64_t r = 0; r < nrows; ++r) {
                        fprintf(stderr, "%s%016llx", r ? " " : "", (unsigned long long) rowh[(size_t) r]);
                    }
                    fprintf(stderr, "]\n");
                }
            }
        }
    }

    static int cgc_sched_dbg = 0;
    if (cgc_sched_dbg < 12) {
        cgc_sched_dbg++;
        LLAMA_LOG_INFO("CGC-SCHED: %d splits\n", ggml_backend_sched_get_n_splits(sched.get()));
        for (int il = 0; il < 2 && il < (int) model.hparams.n_layer(); il++) {
            ggml_tensor * r = cache_remap_tensors[il];
            auto & ffn = cache_ffn_tensors[il];
            const char * rb = "-";
            const char * wb = "-";
            if (r) rb = ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched.get(), r));
            if (ffn.size() > 1 && ffn[1]) wb = ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched.get(), ffn[1]));
            LLAMA_LOG_INFO("  CGC-SCHED: il=%d remap=%s up_w=%s\n", il, rb, wb);
        }
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::expert_cache_eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    llama_context * ctx = static_cast<llama_context *>(user_data);
    const bool cgc_verify_op_timing = getenv("CGC_VERIFY_OP_TIMING") != nullptr;
    auto cgc_is_verify_timing_target = [](const ggml_tensor * node) -> bool {
        if (node == nullptr || node->name[0] == '\0') {
            return false;
        }
        if (strncmp(node->name, "ffn_moe_topk-", 13) == 0) {
            return true;
        }
        // Extra boundary markers used only to keep the down-path chunk from swallowing
        // the interleaved shared-expert FFN work that sits between routed swiglu and
        // routed down in the final execution order.
        if (strncmp(node->name, "ffn_gate-",       9) == 0 ||
            strncmp(node->name, "ffn_up-",         7) == 0 ||
            strncmp(node->name, "ffn_swiglu-",    11) == 0 ||
            strncmp(node->name, "ffn_moe_swiglu-", 15) == 0 ||
            strncmp(node->name, "shared_expert_gate_sigmoid-", sizeof("shared_expert_gate_sigmoid-") - 1) == 0 ||
            strncmp(node->name, "mtp_shared_expert_gate_sigmoid-", sizeof("mtp_shared_expert_gate_sigmoid-") - 1) == 0) {
            return true;
        }
        if (strncmp(node->name, "ffn_moe_gate-", 13) != 0 &&
            strncmp(node->name, "ffn_moe_up-",   11) != 0 &&
            strncmp(node->name, "ffn_moe_down-", 13) != 0) {
            return false;
        }
        if (node->op != GGML_OP_MUL_MAT_ID || node->src[2] == nullptr) {
            return false;
        }
        return node->src[2]->type == GGML_TYPE_I32 && node->src[2]->ne[1] > 1;
    };
    if (ask && cgc_verify_op_timing) {
        return cgc_is_verify_timing_target(t);
    }
    static int cgc_cb_mtp = 0;
    static int cgc_cb_def = 0;
    const bool cgc_cb_is_mtp = ctx->cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP;
    const int cgc_cb_cnt = cgc_cb_is_mtp ? cgc_cb_mtp : cgc_cb_def;
    if (strncmp(t->name, "ffn_moe_topk", 12) == 0 && cgc_cb_cnt < 200 && getenv("CGC_CB_DBG") != nullptr) {
        if (cgc_cb_is_mtp) { cgc_cb_mtp++; } else { cgc_cb_def++; }
        const char * dash = strrchr(t->name, '-');
        const int cb_il = dash ? atoi(dash + 1) : -1;
        fprintf(stderr, "CGC-CB: %s name=%s ask=%d active=%d map_has=%d\n",
                cgc_cb_is_mtp ? "MTP" : "DEF",
                t->name, ask ? 1 : 0, ctx->model.expert_cache_active ? 1 : 0,
                (int) ctx->cache_remap_tensors.count(cb_il));
    }
    // [CGC Step-2/3 weighted cold guard] produce moved into expert_cache_on_topk head: the
    // async segmented dispatcher (CGC_OA_ASYNC) only forwards each segment's ffn_moe_topk node
    // here, so the ffn_moe_logits* branch below could never fire. on_topk recovers the layer's
    // logits by walking the topk node's src chain (cgc_topk_find_logits) in every dispatch
    // mode, then computes the weighted cold ratio and stores it for the fast-path guard.
    // only react on the actual (non-ask) dispatch of a top-k node
    if (!ask && ctx->model.expert_cache_active &&
        strncmp(t->name, "ffn_moe_topk", 12) == 0) {
        ctx->expert_cache_on_topk(t);
    }
    // [CGC bit-bisect v7] in-compute tensor dump: the CGC segmented dispatcher
    // forwards each completed segment's nodes here (ask=false) — see header comment.
    if (!ask) {
        ctx->cgc_tdcb_maybe_dump(t);
    }
    if (ctx->cparams.cb_eval != nullptr) {
        return ctx->cparams.cb_eval(t, ask, ctx->cparams.cb_eval_user_data);
    }
    return true;
}

void llama_context::cgc_tdcb_maybe_dump(ggml_tensor * t) {
    static const std::vector<std::string> td_filters = []() {
        std::vector<std::string> v;
        const char * e = getenv("CGC_TD_CB");
        if (e != nullptr && e[0] != '\0') {
            std::string s(e);
            size_t p = 0;
            while (p <= s.size()) {
                size_t q = s.find(',', p);
                std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
                if (!tok.empty()) {
                    v.push_back(tok);
                }
                if (q == std::string::npos) {
                    break;
                }
                p = q + 1;
            }
        }
        return v;
    }();
    static const bool td_fa = []() {
        for (const auto & f : td_filters) {
            if (f == "FA") {
                return true;
            }
        }
        return false;
    }();
    static const int64_t td_ub = []() {
        const char * e = getenv("CGC_TD_CB_UB");
        return e != nullptr && e[0] != '\0' ? (int64_t) atoll(e) : 1;
    }();
    if (td_filters.empty() || t == nullptr) {
        return;
    }
    // only dump the configured ubatch (default 1 = prefill chunk 0)
    if (cgc_tdcb_ubatch_seq != td_ub) {
        return;
    }

    const char * ctype = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "MTP" : "DEF";

    // per-ubatch occurrence state: names can repeat (e.g. Kcur before/after RoPE);
    // occurrence index follows node visit order, identical across the two paths
    static int64_t last_seq = -1;
    static std::unordered_map<std::string, int> seen;
    static int fa_occ = 0;
    if (last_seq != cgc_tdcb_ubatch_seq) {
        last_seq = cgc_tdcb_ubatch_seq;
        seen.clear();
        fa_occ = 0;
    }

    auto dump_tensor = [&](const ggml_tensor * ts, const char * name) {
        if (ts == nullptr || ts->data == nullptr) {
            return;
        }
        const std::string key(name);
        const int occ = ++seen[key];
        char path[256];
        if (occ == 1) {
            snprintf(path, sizeof(path), "/tmp/cgc_tdcb_%s_%s", ctype, name);
        } else {
            snprintf(path, sizeof(path), "/tmp/cgc_tdcb_%s_%s_n%d", ctype, name, occ);
        }

        const size_t nbytes = ggml_nbytes(ts);
        // F16/BF16 -> F32 (lossless, 2x size) so the compare script can treat those
        // files as f32 too; other types are dumped raw for a byte-level comparison.
        // [CGC fix 2026-08-30] the old version memcpy'd the widened f32 data back into
        // the nbytes-sized buf -> 2x heap overflow -> SIGABRT a few allocations later.
        if (ts->type == GGML_TYPE_F16 || ts->type == GGML_TYPE_BF16) {
            strcat(path, ".f32");
            std::vector<uint8_t> buf(nbytes);
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(ts), buf.data(), 0, nbytes);
            const size_t n = nbytes / ggml_type_size(ts->type);
            std::vector<float> f(n);
            if (ts->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), f.data(), (int64_t) n);
            } else {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) buf.data(), f.data(), (int64_t) n);
            }
            FILE * fo = fopen(path, "wb");
            if (fo != nullptr) {
                fwrite(f.data(), sizeof(float), n, fo);
                fclose(fo);
            }
        } else {
            if (ts->type == GGML_TYPE_F32) {
                strcat(path, ".f32");
            } else {
                strcat(path, ".bin");
            }
            std::vector<uint8_t> buf(nbytes);
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(ts), buf.data(), 0, nbytes);
            FILE * fo = fopen(path, "wb");
            if (fo != nullptr) {
                fwrite(buf.data(), 1, buf.size(), fo);
                fclose(fo);
            }
        }
        fprintf(stderr, "CGC-TDCB: %s type=%s ne=[%lld %lld %lld %lld] nb=[%zu %zu %zu %zu] %s\n",
                path, ggml_type_name(ts->type),
                (long long) ts->ne[0], (long long) ts->ne[1], (long long) ts->ne[2], (long long) ts->ne[3],
                (size_t) ts->nb[0], (size_t) ts->nb[1], (size_t) ts->nb[2], (size_t) ts->nb[3],
                ctype);
        fflush(stderr);
    };

    // flash-attention node: dump its Q/K/V/mask inputs and its output. Visit order
    // across the whole graph = layer order, so fa1_* is the first full-attn layer.
    if (td_fa && t->op == GGML_OP_FLASH_ATTN_EXT) {
        fa_occ++;
        char nm[32];
        snprintf(nm, sizeof(nm), "fa%d_o", fa_occ);
        dump_tensor(t, nm);
        snprintf(nm, sizeof(nm), "fa%d_q", fa_occ);
        dump_tensor(t->src[0], nm);
        snprintf(nm, sizeof(nm), "fa%d_k", fa_occ);
        dump_tensor(t->src[1], nm);
        snprintf(nm, sizeof(nm), "fa%d_v", fa_occ);
        dump_tensor(t->src[2], nm);
        if (t->src[3] != nullptr) {
            snprintf(nm, sizeof(nm), "fa%d_m", fa_occ);
            dump_tensor(t->src[3], nm);
        }
        return;
    }

    // exact-name match (no substring: "Kcur-3" must not hit "Kcur-31")
    if (t->name[0] == '\0') {
        return;
    }
    for (const auto & flt : td_filters) {
        if (flt == t->name) {
            dump_tensor(t, t->name);
            return;
        }
    }
}

void llama_context::cgc_dump_graph_tensors(ggml_cgraph * gf) {
    static const std::vector<std::string> td_filters = []() {
        std::vector<std::string> v;
        const char * e = getenv("CGC_TENSOR_DUMP");
        if (e != nullptr && e[0] != '\0') {
            std::string s(e);
            size_t p = 0;
            while (p <= s.size()) {
                size_t q = s.find(',', p);
                std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
                if (!tok.empty()) {
                    v.push_back(tok);
                }
                if (q == std::string::npos) {
                    break;
                }
                p = q + 1;
            }
        }
        return v;
    }();
    static const int td_pmax_max = []() {
        const char * e = getenv("CGC_TD_PMAX");
        return e != nullptr ? atoi(e) : 7;
    }();
    if (td_filters.empty() || gf == nullptr) {
        return;
    }

    const llama_pos pmax = llama_memory_seq_pos_max(memory.get(), 0);
    if (pmax < 0 || pmax > td_pmax_max) {
        return;
    }

    const char * ctype = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "MTP" : "DEF";

    // the async pipeline may still be draining — sync before reading work buffers
    ggml_backend_sched_synchronize(sched.get());

    std::unordered_map<std::string, int> seen;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t == nullptr || t->name[0] == '\0') {
            continue;
        }
        if ((t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) || t->data == nullptr) {
            continue;
        }
        bool match = false;
        for (const auto & f : td_filters) {
            if (strstr(t->name, f.c_str()) != nullptr) {
                match = true;
                break;
            }
        }
        if (!match) {
            continue;
        }

        // same name can appear on several nodes (e.g. Kcur before/after RoPE) —
        // disambiguate by graph order, which is identical across the two paths
        const std::string key(t->name);
        const int occ = ++seen[key];
        char path[256];
        if (occ == 1) {
            snprintf(path, sizeof(path), "/tmp/cgc_td_%s_p%d_%s.f32", ctype, (int) pmax, t->name);
        } else {
            snprintf(path, sizeof(path), "/tmp/cgc_td_%s_p%d_%s_n%d.f32", ctype, (int) pmax, t->name, occ);
        }

        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> buf(nbytes);
        ggml_backend_tensor_get(t, buf.data(), 0, nbytes);

        FILE * f = fopen(path, "wb");
        if (f != nullptr) {
            fwrite(buf.data(), 1, nbytes, f);
            fclose(f);
            LLAMA_LOG_INFO("CGC-TD: %s type=%s ne=[%lld %lld %lld %lld] %zu bytes\n", path,
                    ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], nbytes);
        }
    }
}

// [CGC V2 logits oracle 2026-09-05] FNV-1a 64-bit hash. Self-contained (no external crypto).
// FNV-1a 64-bit: hash = offset_basis; for each byte: hash ^= byte; hash *= FNV_prime.
// offset_basis = 0xcbf29ce484222325, FNV_prime = 0x100000001b3.
static uint64_t cgc_logits_fnv1a64(const void * data, size_t bytes) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= (uint64_t) p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

void llama_context::cgc_logits_oracle_dump(ggml_cgraph * gf, uint32_t n_tokens, uint32_t n_outputs) {
    (void) n_outputs; // log every token row in the ubatch (n_outputs is informational)
    if (gf == nullptr) {
        return;
    }
    // env-gated; static so the lookup + file open only happen once
    static const char * env_path = getenv("CGC_LOGITS_ORACLE_DUMP");
    if (env_path == nullptr || env_path[0] == '\0') {
        return;
    }
    static FILE * f_out = fopen(env_path, "w");
    if (f_out == nullptr) {
        return;
    }
    static int topn = -1;
    if (topn < 0) {
        const char * e = getenv("CGC_LOGITS_ORACLE_TOPN");
        topn = (e != nullptr) ? atoi(e) : 5;
        if (topn < 1) topn = 1;
        if (topn > 64) topn = 64;
    }
    // first_n_max: -2=unset, -1=unlimited, >=0 = cap
    static int first_n_max = -2;
    static int dump_seq = 0;
    if (first_n_max == -2) {
        const char * e = getenv("CGC_LOGITS_ORACLE_FIRST_N");
        first_n_max = (e != nullptr) ? atoi(e) : -1;
    }
    if (first_n_max >= 0 && dump_seq >= first_n_max) {
        return;
    }
    // find logits tensor. Prefer the node named "result_output" (the LM-head output that
    // llama.cpp model files register; the only F32 [n_vocab, n_tokens] tensor in the graph).
    // Shape heuristics are unreliable here: async/segmented graphs contain thousands of F32
    // intermediates with ne[1]==n_tokens (hidden states, KV views, MTP heads), so fall back
    // to largest-ne[0] only when no result_output node exists.
    ggml_tensor * t_logits = nullptr;
    const int n_nodes = ggml_graph_n_nodes(gf);
    int n_cand = 0;
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t == nullptr || t->type != GGML_TYPE_F32) {
            continue;
        }
        // name match first, without the ne[1] constraint: prefill graphs compute logits
        // only for the final token (ne[1]==1 even when n_tokens>1), while verify batches
        // need all rows (ne[1]==n_tokens).
        if (strncmp(t->name, "result_output", 13) == 0) {
            t_logits = t;
            break;
        }
        if (t->ne[1] != (int64_t) n_tokens) {
            continue;
        }
        if (t->ne[0] < 1024) {
            continue;
        }
        n_cand++;
        if (t_logits == nullptr || t->ne[0] > t_logits->ne[0]) {
            t_logits = t;
        }
    }
    if (t_logits == nullptr) {
        return;
    }
    const int64_t n_vocab = t_logits->ne[0];
    const int64_t n_tok   = t_logits->ne[1];
    if (n_vocab <= 0 || n_tok <= 0) {
        return;
    }
    const size_t n_floats = (size_t) (n_vocab * n_tok);
    const size_t nbytes = n_floats * sizeof(float);
    std::vector<float> buf(n_floats);
    // sync the async pipeline before pulling the work buffer
    ggml_backend_sched_synchronize(sched.get());
    ggml_backend_tensor_get(t_logits, buf.data(), 0, nbytes);

    // [CGC-LOGITS-VALID 2026-09-15] P0: never emit a dump we cannot stand behind.
    //
    // Motivating incident: a Metal command buffer failed with Insufficient Memory, the
    // graph never ran, and this function read the *previous* compute's bytes out of the
    // output buffer. The dump looked structurally fine (a JSON object per row) but held
    // sum=-6.83e38 / mean=-2.75e33 and a top-8 of consecutive ids 59400..59407 all tied
    // at 0.125. A watchdog now aborts on that failure, but a dump must be defensible on
    // its own: any consumer may treat the file as an oracle reference, and a poisoned
    // reference silently destroys every comparison built on it. So validate first, and
    // on failure emit a sidecar `<dump>.invalid` that the harness refuses to read.
    //
    // Bounds are deliberately loose - they only have to separate real logits from garbage:
    // a plausible LM logit is O(10..100), so |sum| over <=150k vocab stays under ~1e10 and
    // f32 representation tops out at 3.4e38. These thresholds are ~5 and ~8 orders of
    // magnitude away from anything a real forward pass can produce.
    bool dump_invalid = false;
    char invalid_reason[256] = {0};
    {
        double vsum = 0.0;
        double vmax_abs = 0.0;
        int64_t n_nonfinite = 0;
        for (size_t i = 0; i < n_floats; ++i) {
            const float v = buf[i];
            if (!std::isfinite(v)) {
                n_nonfinite++;
                continue;
            }
            vsum += (double) v;
            const double a = std::fabs((double) v);
            if (a > vmax_abs) {
                vmax_abs = a;
            }
        }
        const double vmean = (n_floats > 0) ? vsum / (double) n_floats : 0.0;
        if (n_nonfinite > 0) {
            dump_invalid = true;
            snprintf(invalid_reason, sizeof(invalid_reason), "non-finite values: %lld of %zu", (long long) n_nonfinite, n_floats);
        } else if (vmax_abs > 1.0e30) {
            dump_invalid = true;
            snprintf(invalid_reason, sizeof(invalid_reason), "|max|=%.6e exceeds 1e30 (stale/failed compute)", vmax_abs);
        } else if (std::fabs(vsum) > 1.0e15) {
            dump_invalid = true;
            snprintf(invalid_reason, sizeof(invalid_reason), "|sum|=%.6e exceeds 1e15 (%zu floats)", std::fabs(vsum), n_floats);
        } else if (std::fabs(vmean) > 1.0e9) {
            dump_invalid = true;
            snprintf(invalid_reason, sizeof(invalid_reason), "|mean|=%.6e exceeds 1e9", std::fabs(vmean));
        }
        if (dump_invalid) {
            LLAMA_LOG_ERROR("%s: CGC-LOGITS-INVALID: refusing to emit oracle dump (%s)\n", __func__, invalid_reason);
            const std::string marker = std::string(env_path) + ".invalid";
            if (FILE * fm = fopen(marker.c_str(), "w")) {
                fprintf(fm, "{\"reason\":\"%s\",\"n_floats\":%zu,\"n_tokens\":%lld,\"n_vocab\":%lld,\"pmax\":%lld}\n",
                        invalid_reason, n_floats, (long long) n_tok, (long long) n_vocab,
                        (long long) llama_memory_seq_pos_max(memory.get(), 0));
                fclose(fm);
            }
            return;
        }
    }

    const uint64_t h_full = cgc_logits_fnv1a64(buf.data(), nbytes);
    const char * ctype = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "MTP" : "DEF";
    const char * tname = t_logits->name[0] != '\0' ? t_logits->name : "?";

    // Per-token row dump: one JSON object per (ubatch_step, token_idx) so single-token
    // divergence in a multi-token ubatch is detectable (vs. only the last-token argmax).
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * row = buf.data() + t * n_vocab;
        double sum = 0.0;
        float maxv = row[0];
        int64_t argmax = 0;
        for (int64_t v = 0; v < n_vocab; ++v) {
            sum += (double) row[v];
            if (row[v] > maxv) {
                maxv = row[v];
                argmax = v;
            }
        }
        const double mean = sum / (double) n_vocab;
        // top-N via bounded insertion: keep an array of size topn, sort descending
        // n_vocab is ~150K, topn is small (default 5) — O(n_vocab * topn) is fine.
        std::vector<std::pair<int64_t, float>> top;
        top.reserve((size_t) topn);
        for (int64_t v = 0; v < n_vocab; ++v) {
            const float val = row[v];
            if ((int) top.size() < topn) {
                top.push_back({v, val});
                // bubble the new entry into place
                for (int i = (int) top.size() - 1; i > 0; --i) {
                    if (top[i].second > top[i - 1].second) {
                        std::swap(top[i], top[i - 1]);
                    } else {
                        break;
                    }
                }
            } else if (val > top.back().second) {
                top.back() = {v, val};
                for (int i = (int) top.size() - 1; i > 0; --i) {
                    if (top[i].second > top[i - 1].second) {
                        std::swap(top[i], top[i - 1]);
                    } else {
                        break;
                    }
                }
            }
        }
        // per-row hash so single-token divergence is detectable even if total hash collides
        const uint64_t h_row = cgc_logits_fnv1a64(row, sizeof(float) * (size_t) n_vocab);

        // [CGC-LOGITS-VALID] row-level degeneracy screen. Two shapes that a real forward
        // pass cannot produce but a stale/partial buffer can:
        //   (a) the whole row is one repeated bit pattern
        //   (b) the top-N is a run of consecutive ids tied on the exact same bits - this is
        //       the 2026-09-15 fingerprint (ids 59400..59407, all 0.125). Ties in f32 are
        //       possible in principle, but a *consecutive-id* run of >=4 bit-identical
        //       values is not something a trained LM head emits.
        {
            bool row_uniform = true;
            for (int64_t v = 1; v < n_vocab; ++v) {
                if (memcmp(&row[v], &row[0], sizeof(float)) != 0) {
                    row_uniform = false;
                    break;
                }
            }
            int n_tied_consecutive = 0;
            for (size_t i = 1; i < top.size(); ++i) {
                const bool consecutive = (top[i].first == top[i - 1].first + 1);
                const bool identical = memcmp(&top[i].second, &top[i - 1].second, sizeof(float)) == 0;
                if (consecutive && identical) {
                    n_tied_consecutive++;
                }
            }
            if (row_uniform || n_tied_consecutive >= 3) {
                dump_invalid = true;
                snprintf(invalid_reason, sizeof(invalid_reason),
                        "%s at token_idx=%lld (topn_tied_consecutive=%d, %zu topn entries)",
                        row_uniform ? "row is a single repeated value" : "top-N is a consecutive tie run",
                        (long long) t, n_tied_consecutive, top.size());
                LLAMA_LOG_ERROR("%s: CGC-LOGITS-INVALID: %s - truncating dump\n", __func__, invalid_reason);
                break;
            }
        }

        // build top JSON array inline
        std::string top_json = "[";
        for (size_t i = 0; i < top.size(); ++i) {
            char tb[64];
            if (i > 0) top_json += ",";
            snprintf(tb, sizeof(tb), "{\"t\":%lld,\"v\":%.6f}",
                    (long long) top[i].first, (double) top[i].second);
            top_json += tb;
        }
        top_json += "]";

        const long long cgc_dump_pmax = llama_memory_seq_pos_max(memory.get(), 0);
        fprintf(f_out,
            "{\"step\":%d,\"token_idx\":%lld,\"n_tokens\":%lld,\"n_vocab\":%lld,\"ctx_type\":\"%s\","
            "\"node\":\"%s\",\"n_cand\":%d,\"pmax\":%lld,"
            "\"logits_fnv1a64\":\"%016llx\",\"row_fnv1a64\":\"%016llx\","
            "\"sum\":%.6f,\"mean\":%.6e,"
            "\"argmax_token\":%lld,\"argmax_logit\":%.6f,"
            "\"top\":%s}\n",
            dump_seq, (long long) t, (long long) n_tok, (long long) n_vocab, ctype,
            tname, n_cand, cgc_dump_pmax,
            (unsigned long long) h_full, (unsigned long long) h_row,
            sum, mean,
            (long long) argmax, (double) maxv,
            top_json.c_str());
    }
    if (dump_invalid) {
        // [CGC-LOGITS-VALID] the row loop broke early, so the file holds a partial dump.
        // Do NOT advance dump_seq and do NOT flush-and-forget: stamp the sidecar so the
        // harness can never mistake this file for a usable oracle reference.
        const std::string marker = std::string(env_path) + ".invalid";
        if (FILE * fm = fopen(marker.c_str(), "w")) {
            fprintf(fm, "{\"reason\":\"%s\",\"partial\":true,\"stopped_at_step\":%d,\"n_tokens\":%lld,\"n_vocab\":%lld}\n",
                    invalid_reason, dump_seq, (long long) n_tok, (long long) n_vocab);
            fclose(fm);
        }
        fflush(f_out);
        return;
    }
    dump_seq++;
    fflush(f_out);
}

// [CGC Step-2a produce 2026-09-06] find this layer's router-logits node by walking the
// dispatched top-k tensor's src chain. The async segmented dispatcher (CGC_OA_ASYNC) only
// forwards each segment's ffn_moe_topk node to the eval callback — the ffn_moe_logits* nodes
// never reach expert_cache_eval_cb in production, which is why the produce side stayed dormant
// (weighted ratio always at the 1e9 sentinel -> count fallback). At hook time the whole segment
// (including the logits matmul) has completed and no later segment has been submitted yet, so
// the logits buffer still holds its final values and a host read is safe.
static ggml_tensor * cgc_topk_find_logits(ggml_tensor * t, int il) {
    if (t == nullptr) {
        return nullptr;
    }
    char want[24];
    snprintf(want, sizeof(want), "-%d", il);
    const size_t want_len = strlen(want);
    std::vector<ggml_tensor *> stack;
    std::unordered_set<ggml_tensor *> seen;
    stack.push_back(t);
    seen.insert(t);
    while (!stack.empty()) {
        ggml_tensor * n = stack.back();
        stack.pop_back();
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = n->src[s];
            if (src == nullptr || !seen.insert(src).second) {
                continue;
            }
            if (src->name[0] != '\0' && src->type == GGML_TYPE_F32 &&
                (strncmp(src->name, "ffn_moe_logits_raw", 18) == 0 ||
                 strncmp(src->name, "ffn_moe_logits_biased", 21) == 0)) {
                const size_t nl = strlen(src->name);
                if (nl > want_len && strcmp(src->name + nl - want_len, want) == 0) {
                    return src;
                }
            }
            stack.push_back(src);
        }
    }
    return nullptr;
}

// [CGC 2026-09-15 S1 diagnostic] Print the slot vector the hook just published for THIS step's
// ids, together with the site that published it. The number to compare it against is
// CGC-MMID-ASSERT's `first=` for the same layer: the assertion reports what the GPU consumed, this
// reports what the host wrote. If they differ, the GPU is not reading the buffer the hook wrote,
// and no amount of staring at the mapping will show that. If they agree, the mapping is right and
// the divergence is downstream of the table (the gather's index or the gather output's buffer).
// `site` also answers "which of the four remap write sites ran", which the missing EXPECT in the
// first diagnostic run could not tell us.
static void cgc_s1_expect_dbg(const char * site, int il, int64_t n_tokens, int64_t n_expert_used,
                             const int32_t * ids, const ggml_tensor * tbl) {
    static const bool on = getenv("CGC_S1_DBG") != nullptr;
    // [CGC 2026-09-15 S1 window fix] This used to be `!on || il > 3` with a cap of 8 lines, i.e. the
    // hook-side probe looked at layers 0..3 only. Every probe run of the day therefore reported "the
    // table is correct" while the bit-identical gate kept failing, and the reason was not that the
    // probe was wrong -- it was that the bisect put the divergence at layers 10..19, OUTSIDE the
    // window the probe could see. An instrument whose scan range is narrower than the hypothesis
    // cannot falsify it, and its silence then reads as confirmation (the same failure as
    // CONVENTIONS.md B2, one level up: there the probe printed nothing at all, here it printed
    // something true about the wrong place). The window is now the whole stack; the cap is sized so
    // one full layer sweep over a few steps fits. Cost is stderr volume, which is opt-in and gated.
    if (!on) {
        return;
    }
    static int n = 0;
    if (n >= 512) {
        return;
    }
    n++;
    if (tbl == nullptr || tbl->data == nullptr) {
        fprintf(stderr, "CGC-S1: EXPECT-%s il=%d ntok=%lld TBL=null\n",
                site, il, (long long) n_tokens);
        return;
    }
    const int32_t * tb = (const int32_t *) tbl->data;
    fprintf(stderr, "CGC-S1: EXPECT-%s il=%d ntok=%lld data=%p host=%d buf=%s ids=[",
            site, il, (long long) n_tokens, (void *) tbl->data,
            tbl->buffer ? (int) ggml_backend_buffer_is_host(tbl->buffer) : -1,
            tbl->buffer ? ggml_backend_buffer_name(tbl->buffer) : "nil");
    for (int64_t i = 0; i < n_expert_used; ++i) {
        fprintf(stderr, "%s%d->%d", i ? " " : "", ids[i], tb[ids[i]]);
    }
    fprintf(stderr, "]\n");
}

// [CGC 2026-09-15 S1 slot-table equivalence] Differential check between the TWO implementations of
// the expert->slot mapping, for every expert this step actually selects.
//
//   llama_expert_cache_publish_slot_table()  sweeps the whole layer in one call (what the GPU table
//                                            gets), and is reached only from the S1 path.
//   llama_expert_cache_slot_table_safe()     is asked once per selected expert (what the host leaf
//                                            gets), and is the only path the bit-identical
//                                            reference was ever produced with.
//
// They are deliberately separate code paths -- a bulk export and a per-lookup -- so they can drift,
// and a drift is invisible in the logits oracle until it changes which slot a cold expert reads.
// The leaf is the authority here: the gate's reference is a leaf-path product. So this reports every
// SELECTED expert whose published table entry differs from what slot_table_safe() returns for it.
// The interesting case is the one the two functions were written to disagree on: a NON-resident
// expert on a layer with no reserved ZERO slot. publish clamps it to 0 (and counts a clamp); safe
// returns the raw -1. 0 is a legal slot index pointing at some other expert's weights; -1 is not an
// index at all. Neither is what the leaf path needs, and which one is in the table decides whether
// the failure is silent (wrong expert) or loud (id_oob), so it must be printed rather than assumed.
static void cgc_s1_equiv_dbg(const char * site, const llama_expert_cache * cache, int il,
                             int64_t n_tokens, int64_t n_expert_used, const int32_t * ids,
                             const ggml_tensor * tbl, bool identity) {
    static const bool on = getenv("CGC_S1_DBG") != nullptr;
    if (!on || ids == nullptr || tbl == nullptr || tbl->data == nullptr) {
        return;
    }
    const int32_t * tb = (const int32_t *) tbl->data;
    const int64_t n_sel = n_tokens * n_expert_used;
    int64_t n_mismatch = 0;
    int32_t first_e = -1, first_got = 0, first_want = 0;
    for (int64_t j = 0; j < n_sel; ++j) {
        const uint32_t e = (uint32_t) ids[j];
        const int32_t want = identity ? (int32_t) e
                                      : llama_expert_cache_slot_table_safe(cache, (uint32_t) il, e);
        if (tb[e] != want) {
            if (n_mismatch == 0) {
                first_e = (int32_t) e;
                first_got = tb[e];
                first_want = want;
            }
            n_mismatch++;
        }
    }
    const int32_t zs = identity ? -1 : llama_expert_cache_zero_slot(cache, (uint32_t) il);
    fprintf(stderr, "CGC-S1: EQUIV-%s il=%d ntok=%lld n_sel=%lld mismatch=%lld"
                    " first_e=%d table=%d safe=%d zero_slot=%d map=%s\n",
            site, il, (long long) n_tokens, (long long) n_sel, (long long) n_mismatch,
            first_e, first_got, first_want, zs, identity ? "identity" : "slot");
}

// [CGC 2026-09-15 §8.3 + premise B] The single publish-and-count entry point for the GPU slot
// table. Both remap write sites must go through this, and not for tidiness: the two sites
// previously disagreed in two independent ways and BOTH disagreements were invisible.
//
//   (1) The pool path called llama_expert_cache_publish_slot_table and DISCARDED its return value.
//       That is the path every MTP-off S1 arm actually takes: CGC_SERVER_MTP=0 makes verify_fast and
//       draft_fast both false, so the `(verify_fast || draft_fast) && cgc_fast_eligible` guard is
//       never entered and the fast-path site never runs. So on every S1 measurement taken to date,
//       the clamp count -- the number §8.3 identified as the silent-corruption signal -- was
//       computed and thrown away on the floor.
//   (2) The two sites write the LEAF under different mappings while publishing the TABLE through the
//       same function. Fast path writes slot_table_safe (non-resident -> ZERO slot when reserved,
//       else -1); pool path writes the raw slot table (non-resident -> -1 always). The table write
//       clamps a non-resident expert to 0 whenever no ZERO slot is reserved. Since zero_slot is
//       itself gated on the MTP fast path, that is exactly the S1 configuration: leaf = -1 (loud --
//       the consumer reports an out-of-range id), table = 0 (SILENT -- index 0 is a legal index, so
//       mul_mat_id reads a DIFFERENT expert's weights and the answer is quietly wrong).
//
// Counting here, inside the one function both sites call, is what makes them agree by construction.
// The counting itself is unconditional (an integer add each, no branch on the hot path beyond the
// publish that already happened); every print is env-gated.
static int64_t cgc_publish_slot_table_counted(llama_expert_cache * cache, int il, int64_t n_tokens,
        int64_t n_expert, ggml_tensor * table, bool is_draft, const char * site,
        const int32_t * ids, int64_t n_expert_used) {
    if (cache == nullptr || table == nullptr || table->data == nullptr) {
        return 0;
    }
    const int64_t clamped = llama_expert_cache_publish_slot_table(
            cache, (uint32_t) il, (int32_t *) table->data, (uint32_t) n_expert);
    cache->n_slot_table_publishes++;
    if (clamped > 0) {
        cache->n_slot_table_clamped += (size_t) clamped;
        // [CGC 2026-09-15 §8.3 gate quantity, take 2] The whole-table count is kept only as a
        // secondary number. On the first churn run it printed 480363, and 480363/1088256 = 44.1% --
        // which is simply the non-resident fraction of a 143-slot pool over 256 experts. A quantity
        // that is large in EVERY run carries no information (CONVENTIONS B1: a field holding one
        // value across all observations says nothing when it does not change), so the first version
        // of this gate quantity violated the very rule it was added to satisfy. The number that
        // decides whether the answer can be silently wrong is the SELECTED subset: of the ids the
        // consumer actually reads this step, how many were clamped to slot 0 instead of -1.
        // Verify-strict is supposed to make that zero; this is what says whether it did.
        if (ids != nullptr && n_expert_used > 0) {
            const int32_t * tb = (const int32_t *) table->data;
            const int32_t * st = llama_expert_cache_slot_table(cache, (uint32_t) il);
            for (int64_t j = 0; j < n_tokens * n_expert_used; ++j) {
                const int32_t e = ids[j];
                if (st != nullptr && e >= 0 && (int64_t) e < n_expert && st[e] < 0 && tb[e] == 0) {
                    cache->n_slot_table_clamped_selected++;
                }
            }
        }
        static long long cgc_clamp_lines = 0;
        if (cgc_clamp_lines++ < 16) {
            fprintf(stderr, "CGC-SLOT-TABLE-CLAMP: site=%s %s il=%d clamped=%lld/%lld"
                            "  <-- TABLE/LEAF EQUIVALENCE BROKEN: gather reads slot 0 (another"
                            " expert's weights) while the host leaf writes -1 (loud)\n",
                    site, is_draft ? "draft" : "verify", il,
                    (long long) clamped, (long long) n_expert);
        }
        if (getenv("CGC_S1_CLAMP_ABORT") != nullptr) {
            GGML_ABORT("CGC_S1_CLAMP_ABORT: publish_slot_table clamped %lld/%lld entries"
                       " (site=%s il=%d)\n",
                       (long long) clamped, (long long) n_expert, site, il);
        }
    }

    // Premise B: does the published CONTENT move between steps? Only decode steps are counted --
    // prefill rewrites the table for reasons nobody is questioning, and mixing the two would drown
    // the decode answer. Keyed by (context, layer) so verify and draft cannot contaminate each
    // other's snapshots.
    static const bool cgc_churn_on = getenv("CGC_S1_TABLE_CHURN") != nullptr;
    if (cgc_churn_on && n_tokens == 1) {
        static std::map<int, std::vector<int32_t>> cgc_churn_last;
        static long long cgc_churn_graph_pub = 0;
        static long long cgc_churn_graph_chg = 0;
        static long long cgc_churn_graph_idx = 0;
        static long long cgc_churn_lines     = 0;
        // The hook walks layers in order, so coming back round to the first served layer is a graph
        // boundary. CGC_S1_MIN_IL defaults to 1 and layer 0 keeps its host leaf, so il == 1 is the
        // first served layer on the default configuration.
        if (il <= 1 && cgc_churn_graph_pub > 0) {
            fprintf(stderr, "CGC-S1: TABLE-CHURN graph=%lld publishes=%lld changed_entries=%lld\n",
                    cgc_churn_graph_idx, cgc_churn_graph_pub, cgc_churn_graph_chg);
            cgc_churn_graph_idx++;
            cgc_churn_graph_pub = 0;
            cgc_churn_graph_chg = 0;
        }
        const int key = il + (is_draft ? 100000 : 0);
        std::vector<int32_t> & prev = cgc_churn_last[key];
        const int32_t * now = (const int32_t *) table->data;
        const bool seeded = prev.size() == (size_t) n_expert;
        long long chg = 0;
        if (seeded) {
            for (int64_t e = 0; e < n_expert; ++e) {
                if (prev[(size_t) e] != now[e]) {
                    chg++;
                }
            }
        }
        // [CGC 2026-09-15 premise B, take 2] `chg` above is the WHOLE-TABLE count and is the wrong
        // question. A 143-slot pool over 256 experts reshuffles a large number of entries every step
        // by construction, so a big number there carries no information -- the first churn run
        // printed changed_entries=1976 and that is simply what a shuffled table looks like
        // (CONVENTIONS B1). Premise B needs "does the mapping of an id the consumer actually READS
        // move", because only a consumed move can create a per-step host->GPU ordering requirement.
        // Counted per publish so the teardown can put it next to publishes as a rate; the question
        // is whether it is 0/N or N/N, and anything in between is a mixed answer worth seeing.
        if (seeded && ids != nullptr && n_expert_used > 0) {
            long long consumed_moved = 0;
            long long consumed_total = 0;
            for (int64_t j = 0; j < n_tokens * n_expert_used; ++j) {
                const int32_t e = ids[j];
                if (e < 0 || (int64_t) e >= n_expert) {
                    continue;
                }
                consumed_total++;
                if (prev[(size_t) e] != now[e]) {
                    consumed_moved++;
                }
            }
            if (consumed_total > 0) {
                if (consumed_moved > 0) {
                    cache->n_slot_table_consumed_changed++;
                } else {
                    cache->n_slot_table_consumed_same++;
                }
            }
        }
        cgc_churn_graph_pub++;
        cgc_churn_graph_chg += chg;
        prev.assign(now, now + n_expert);
        if (seeded) {
            if (chg > 0) {
                cache->n_slot_table_changed += (size_t) chg;
                if (cgc_churn_lines++ < 24) {
                    fprintf(stderr, "CGC-S1: TABLE-CHURN il=%d changed=%lld/%lld\n",
                            il, chg, (long long) n_expert);
                }
            } else {
                cache->n_slot_table_unchanged++;
            }
        }
    }
    return clamped;
}

void llama_context::expert_cache_on_topk(ggml_tensor * t) {
    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];
    const char * dash = strrchr(t->name, '-');
    if (dash == nullptr) {
        return;
    }
    const int il = atoi(dash + 1);
    // [CGC MTP fix] accept the MTP draft block layer (il == n_layer()) too: its layer-N expert
    // tensors are shrunk to the bounded pool capacity and the draft graph repoints them at the
    // pool regions, so the hook MUST ensure/fill layer-N slots and write its remap leaf. Using
    // n_layer_all (trunk + nextn) keeps the original non-MTP behaviour for the trunk layers.
    if (il < 0 || il >= (int) model.hparams.n_layer_all) {
        return;
    }

    llama_expert_cache * cache = model.expert_cache;
    if (cache == nullptr) {
        return;
    }

    // [CGC MTP Draft Prefetch DEBUG] trace every on_topk call to verify draft ctx reaches here
    static int cgc_draft_pf_dbg_count = 0;
    if (cgc_draft_prefetch_on() && cgc_draft_pf_dbg_count < 500) {  // increased from 50 to 500 to capture decode phase
        const char * ctype = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "MTP" : "DEF";
        fprintf(stderr, "CGC-DRAFT-PF-DBG: on_topk il=%d n_tokens=%lld ctx_type=%s n_expert_used=%lld\n",
                il, (long long) n_tokens, ctype, (long long) n_expert_used);
        cgc_draft_pf_dbg_count++;
    }

    // [CGC DBUF2 full double-buffer 2026-09-06] at the first MoE layer of each VERIFY decode
    // step (il==1 AND n_tokens==1), atomically swap active<->scratch slot tables. This is the
    // GPU-idle step boundary: the previous step's FFN has completed and consumed its remap leaves,
    // and the next step's FFN hasn't started. Async bg fills that landed in scratch since the last
    // swap become active; the old active retains its full mappings and becomes the new scratch
    // (incrementally updated by the next round of fills).
    // CRITICAL: only swap when n_tokens==1 (decode). During prefill (n_tokens>1) the graph
    // dispatches all layers at once and there is no per-step idle window; swapping there caused
    // severe cache->m contention with bg workers and collapsed prefill t/s (measured 5.5 vs
    // 15-26). Draft ctx (il >= n_layer) never triggers il==1. CGC_DBUF2=0 = no-op.
    if (il == 1 && n_tokens == 1 && cgc_dbuf2_on()) {
        llama_expert_cache_dbuf2_swap(cache);
    }

    // [CGC Step-2a produce 2026-09-06] weighted cold ratio measurement: walk upstream from the
    // top-k node to this layer's ffn_moe_logits* tensor (all dispatched modes converge here),
    // softmax it on the host and store sum(cold routing mass)/total for the fast-path cold
    // guard below. Runs before the guard reads cgc_weighted_cold_ratio. Unmeasured layers keep
    // the 1e9 sentinel -> count-based fallback in the guard.
    // [2026-09-06 gate] CGC_WCOLD_EN=1 required: the weighted metric is MORE permissive than
    // the count-based one (a layer with many cold experts that carry little routing mass stays
    // on the ZERO-slot fast path), and without renormalization (CGC_RN_ROUTING, Step-3) that
    // silently-dropped mass degrades quality (measured: coding 1.0 -> 0.3 deterministic loop).
    // Only enable together with CGC_RN_ROUTING once pool membership tracks real routing.
    static const bool cgc_wcold_en = getenv("CGC_WCOLD_EN") != nullptr;
    if (cgc_wcold_en) {
        ggml_tensor * lg = cgc_topk_find_logits(t, il);
        if (lg != nullptr && (size_t) il < cache->cgc_weighted_cold_ratio.size()) {
            const uint32_t lg_n_expert = (uint32_t) lg->ne[0];
            const uint32_t lg_n_tokens = (uint32_t) lg->ne[1];
            if (lg_n_expert > 0 && lg_n_tokens > 0) {
                const int32_t * st = cache->slot_table.data() + (size_t) il * cache->n_expert;
                std::vector<float> lbuf((size_t) lg_n_expert * lg_n_tokens);
                ggml_backend_tensor_get(lg, lbuf.data(), 0,
                                        (size_t) lg_n_expert * lg_n_tokens * sizeof(float));
                double total_w = 0.0, cold_w = 0.0;
                for (uint32_t j = 0; j < lg_n_tokens; ++j) {
                    // softmax over experts for token j (row-major: logit[e,j] at e + j*n_expert)
                    float max_l = -1e30f;
                    for (uint32_t e = 0; e < lg_n_expert; ++e) {
                        const float v = lbuf[e + j * lg_n_expert];
                        if (v > max_l) {
                            max_l = v;
                        }
                    }
                    float sum_exp = 0.0f;
                    for (uint32_t e = 0; e < lg_n_expert; ++e) {
                        const float v = lbuf[e + j * lg_n_expert];
                        const float ex = (float) expf(v - max_l);
                        sum_exp += ex;
                        if (e < cache->n_expert && st[e] < 0) {
                            cold_w += ex;
                        }
                    }
                    total_w += sum_exp;
                }
                cache->cgc_weighted_cold_ratio[(size_t) il] =
                    total_w > 0.0 ? (double) cold_w / total_w : 0.0;
                if (il <= 1) {
                    fprintf(stderr, "CGC-LOGITS il=%d weighted_cold=%.3f%% ntok=%u n_expert=%u\n",
                            il, cache->cgc_weighted_cold_ratio[(size_t) il] * 100.0,
                            lg_n_tokens, lg_n_expert);
                }
            }
        }
    }

    // [CGC M1 Metal slab 2026-09-14] The per-kind gather buffers used to be eagerly resized here.
    // They are now lazily allocated Metal buffers (cache_gather_slab, created on first wide union),
    // so there is nothing to pre-create: a default-config server never allocates one at all.

    const int32_t * ids = (const int32_t *) t->data;
    if (ids == nullptr) {
        return;
    }
    // CGC fix: the top-k ids live in the graph work buffer. With the pipelined async segmented
    // dispatch (CGC_OA_ASYNC) the GPU pipeline keeps advancing while this hook runs, and ggml-alloc
    // can REUSE the top-k buffer for later tensors of the same step. Re-reading t->data after the
    // blocking ensure_batch/drain_layer below then returns clobbered values (segfault in the
    // CGC-POST/st slot lookup + a corrupted remap). Snapshot the ids once, up front, so every later
    // read (remap write, debug prints) is stable.
    std::vector<int32_t> ids_snap(ids, ids + (size_t) n_tokens * n_expert_used);
    ids = ids_snap.data();
    static int cgc_hook_dbg_n = 0;
    if (cgc_hook_dbg_n < 80) {
        cgc_hook_dbg_n++;
        fprintf(stderr, "CGC-HOOK: ctx=%p il=%d ntok=%lld ids=[%d %d %d %d %d %d %d %d]\n",
                (void *) this, il, (long long) n_tokens,
                ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7]);
    }
    // [CGC MTP Draft Prefetch 2026-09-07] COLLECT phase: during MTP draft decode
    // (ctx_type == MTP, n_tokens >= 1), the draft ctx computes the top-8 expert ids for the
    // NEXT token one step ahead of the trunk verify ctx. Since draft_accept is 92-98%, these
    // draft-predicted expert ids are highly likely to be selected by the verify step. We collect
    // them per layer here, then the verify ctx's il==1 hook triggers the prefetch (see below).
    // This attacks the count-cold problem (15% of selected experts are non-resident) by prefetching
    // the EXACT experts the draft predicts will be needed next step. CGC_DRAFT_PREFETCH=1 enables.
    //
    // [FIX 2026-09-08] Changed n_tokens == 1 to n_tokens >= 1:
    //   MTP draft ctx processes multiple tokens at once (n_tokens = draft_n_max, typically 3).
    //   The ids tensor layout is [n_expert_used, n_tokens], so token j's experts are at
    //   ids[j*n_expert_used ... (j+1)*n_expert_used - 1].
    //   We take the FIRST token (j=0) because verify ctx validates from the first predicted token.
    //   Previously n_tokens == 1 caused collect to never fire under MTP (loaded=0 always).
    if (cgc_draft_prefetch_on() &&
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && n_tokens >= 1 &&
        il >= 0 && (size_t) il < cache->draft_prefetch_ids.size()) {
        // snapshot the top-8 expert ids for this layer (draft predicts next token's routing)
        // Take the FIRST token (j=0) from the ids tensor — verify validates from the first.
        std::vector<uint32_t> &dst = cache->draft_prefetch_ids[il];
        dst.resize((size_t) n_expert_used);
        for (int64_t i = 0; i < n_expert_used; ++i) {
            dst[i] = (uint32_t) ids[i];  // j=0 offset is 0
        }
        cache->draft_prefetch_valid[il] = true;
        // print for the trunk start (il<=1) AND the MTP block layer (il == n_layer): the MTP
        // draft graph only executes the single nextn block, so collect fires at il == n_layer.
        if (il <= 1 || il == (int) model.hparams.n_layer()) {
            fprintf(stderr, "CGC-DRAFT-PF: collect il=%d ntok=%lld ids=[%u %u %u %u %u %u %u %u]\n",
                    il, (long long) n_tokens,
                    dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
        }
    }
    // Note: #12 collect_skipped counter removed — with n_tokens >= 1, batched draft steps
    // are now collected (first token), not skipped. The counter remains in prefetch_drop_stats
    // for backward compatibility but will always be 0 under this fixed path.

    // [CGC prev-token prefetch 2026-09-08] For verify ctx (DEFAULT), collect the current token's
    // per-layer expert ids into curr_token_expert_ids. At il==0, swap prev<->curr so prev holds
    // the COMPLETE previous token's ids (all layers) for prefetch. This attacks count-cold by
    // predicting the current token's experts from the previous token (adjacent tokens have ~70-90%
    // routing overlap). CGC_PREV_TOKEN_PREFETCH=1 enables; default OFF = byte-identical legacy.
    static const bool cgc_prev_token_prefetch = getenv("CGC_PREV_TOKEN_PREFETCH") != nullptr &&
                                                  getenv("CGC_PREV_TOKEN_PREFETCH")[0] == '1';
    if (cgc_prev_token_prefetch &&
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && n_tokens >= 1 &&
        il >= 0 && (size_t) il < cache->curr_token_expert_ids.size()) {
        // At il==0, swap prev<->curr: prev now holds the complete previous token's ids,
        // curr starts collecting the current token's ids.
        if (il == 0) {
            cache->prev_token_expert_ids.swap(cache->curr_token_expert_ids);
            // Mark all prev layers as valid (they were collected in the previous token)
            for (size_t l = 0; l < cache->prev_token_valid.size(); ++l) {
                cache->prev_token_valid[l] = !cache->prev_token_expert_ids[l].empty();
            }
        }
        // Collect current token's first-token expert ids (j=0 offset is 0)
        std::vector<uint32_t> &dst = cache->curr_token_expert_ids[il];
        dst.resize((size_t) n_expert_used);
        for (int64_t i = 0; i < n_expert_used; ++i) {
            dst[i] = (uint32_t) ids[i];  // j=0 offset is 0
        }
        if (il <= 1 && getenv("CGC_PREV_PF_DBG") != nullptr) {
            fprintf(stderr, "CGC-PREV-PF: collect il=%d ntok=%lld ids=[%u %u %u %u %u %u %u %u]\n",
                    il, (long long) n_tokens,
                    dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
        }
    }
    // [CGC bit-bisect 2026-08-30] small batches (prefill chunks / tail / verify / draft): dump
    // the FULL ids tensor so the expert selection of the SAME token can be compared across
    // paths (simp last-chunk row2 vs spec verify row0). ids layout: [n_expert_used, n_tokens],
    // element (i, j) at i + j*n_expert_used — so token row j = ids[j*n_expert_used ... ].
    // [CGC v4] threshold raised 4 -> 8 so the M=8 prefill chunks (present in BOTH paths with
    // identical shapes) are captured too — this is where an early expert flip could hide.
    // Line budget env-tunable (default 4000: ~21 decode calls x 41 layers per path).
    // [CGC 2026-09-15] Default flipped 4000 -> 0. decode runs with n_tokens = 2 (MTP verify),
    // so `n_tokens <= 8` was true on every decode call and the whole 4000-line budget was spent
    // inside a single short request -- ~4000 unbuffered stderr writes per run, on the same hot
    // path the assertion work is being measured against. Opt back in with CGC_IDS_MAX_LINES=N.
    static int cgc_ids_max_lines = -1;
    if (cgc_ids_max_lines < 0) {
        const char * e = getenv("CGC_IDS_MAX_LINES");
        cgc_ids_max_lines = e ? atoi(e) : 0;
    }
    if (cgc_ids_max_lines > 0 && n_tokens <= 8 && cgc_hook_dbg_n < cgc_ids_max_lines) {
        cgc_hook_dbg_n++;
        // [CGC bit-bisect] pos_max identifies the batch (async dispatch scrambles the
        // stderr order, so ctx+ntok alone can't attribute a line to a decode call).
        const llama_pos cgc_pmax = llama_memory_seq_pos_max(memory.get(), 0);
        fprintf(stderr, "CGC-IDS: ctx=%p pmax=%d il=%d ntok=%lld", (void *) this, (int) cgc_pmax, il, (long long) n_tokens);
        const int64_t n_show = n_tokens * n_expert_used;
        for (int64_t k = 0; k < n_show; ++k) {
            fprintf(stderr, " %d", ids[k]);
        }
        fprintf(stderr, "\n");
    }
    if (getenv("LLAMA_EXPERT_CACHE_DISABLE_WRITE")) {
        return; // diagnostic: keep graph structure, do not write remap / repoint weights
    }

    // [CGC mass-coverage measurement 2026-09-06] CGC_MASSCOV=1 only. Read this layer's UNMASKED
    // router logits (same upstream walk as Step-2a produce), softmax them per token, and record
    // the actual softmax MASS of each selected expert (not just its count). The destructor
    // report then computes how much routing mass the CURRENT pool membership serves resident vs
    // what a top-K-by-mass prewarm would serve — the decisive metric for renorm feasibility.
    static const bool cgc_masscov = getenv("CGC_MASSCOV") != nullptr;
    if (cgc_masscov) {
        ggml_tensor * lg = cgc_topk_find_logits(t, il);
        if (lg != nullptr && lg->type == GGML_TYPE_F32) {
            const uint32_t mc_ne = (uint32_t) lg->ne[0];
            const uint32_t mc_nt = (uint32_t) lg->ne[1];
            if (mc_ne > 0 && mc_nt > 0 && mc_ne <= cache->n_expert) {
                std::vector<float> mc_buf((size_t) mc_ne * mc_nt);
                ggml_backend_tensor_get(lg, mc_buf.data(), 0,
                                        (size_t) mc_ne * mc_nt * sizeof(float));
                std::vector<float> mc_w((size_t) n_tokens * n_expert_used, 0.0f);
                for (int64_t j = 0; j < n_tokens && (uint32_t) j < mc_nt; ++j) {
                    float mx = -1e30f;
                    for (uint32_t e = 0; e < mc_ne; ++e) {
                        const float v = mc_buf[e + (size_t) j * mc_ne];
                        if (v > mx) {
                            mx = v;
                        }
                    }
                    double denom = 0.0;
                    for (uint32_t e = 0; e < mc_ne; ++e) {
                        denom += (double) expf(mc_buf[e + (size_t) j * mc_ne] - mx);
                    }
                    for (int64_t i = 0; i < n_expert_used; ++i) {
                        const uint32_t e = (uint32_t) ids[i + j * n_expert_used];
                        if (e < mc_ne && denom > 0.0) {
                            mc_w[i + j * n_expert_used] =
                                expf(mc_buf[e + (size_t) j * mc_ne] - mx) / (float) denom;
                        }
                    }
                }
                llama_expert_cache_masscov_record(cache, (uint32_t) il,
                        (const uint32_t *) ids, mc_w.data(), (size_t) n_tokens * n_expert_used);
            }
        }
    }

    // NOTE: FFN tensor restore/repoint is done in process_ubatch (restore all before every
    // build_graph) and in graph_get_cb (repoint at the L4 pool regions when the remap leaf is
    // built). The hook here only ensures the slot data and writes the remap ids, because with
    // the pipelined segmented dispatch (CGC_OA_ASYNC) the following segments are already
    // submitted — repointing the weights after submit would be too late.

    // build the per-step union (dedup + sorted) and the raw route list. The top-k ids tensor is
    // [n_expert_used, n_tokens] in ggml layout (ne[0] fastest), so element (i, j) is at i + j*n_expert_used.
    std::unordered_map<uint32_t, uint32_t> umap;
    std::vector<uint32_t> uni;
    std::vector<uint32_t> routes;
    routes.reserve(n_tokens * n_expert_used);
    uni.reserve(n_tokens * n_expert_used);
    for (int64_t j = 0; j < n_tokens; ++j) {
        for (int64_t i = 0; i < n_expert_used; ++i) {
            const uint32_t e = (uint32_t) ids[i + j * n_expert_used];
            routes.push_back(e);
            if (!umap.count(e)) {
                umap[e] = (uint32_t) uni.size();
                uni.push_back(e);
            }
        }
    }
    if (uni.empty()) {
        return;
    }
    std::sort(uni.begin(), uni.end());
    // the L3-B gather buffer is laid out in sorted-union order, so remap must use the expert's
    // position in the sorted union (insertion-order umap is NOT valid after the sort).
    std::unordered_map<uint32_t, uint32_t> uidx;
    uidx.reserve(uni.size());
    for (size_t k = 0; k < uni.size(); ++k) {
        uidx[uni[k]] = (uint32_t) k;
    }

    // [CGC SpAc 2026-09-06] EMA utility feed (CGC_SPAC=1 only; zero cost default): every routed
    // step updates this layer's utility vector (full-EMA decay + routed bump). Fires on every
    // path that reaches here — large prefill, MTP draft (ctx MTP) and trunk verify alike — so
    // the draft's next-token routing reaches the estimator one step before trunk needs it (free
    // look-ahead, no extra compute). Unpooled layers update a row spac_prefetch never reads.
    if (cgc_spac_on()) {
        llama_expert_cache_spac_update(cache, (uint32_t) il, routes.data(), routes.size());
    }

    // [CGC MTP Draft Prefetch 2026-09-07] TRIGGER phase: during trunk verify decode
    // (ctx_type == DEFAULT, n_tokens >= 1), at the first MoE layer (il==1), load ALL layers'
    // draft-predicted expert ids SYNCHRONOUSLY. The draft ctx collected these one step ahead
    // (see COLLECT phase above); now we load them directly in this thread so the verify
    // decode's later layers find them resident. This is the exact-prediction counterpart to
    // SpAc's frequency-based prefetch: draft_accept 92-98% = prediction accuracy, so most
    // loaded experts will actually be selected. CGC_DRAFT_PREFETCH=1 enables.
    // SYNC mode (CGC_DRAFT_PREFETCH_SYNC=1): load in this thread (blocks decode, but safe under MTP+OA_ASYNC)
    // ASYNC mode (default): queue to bg thread (faster, but may crash under MTP+OA_ASYNC)
    //
    // [FIX 2026-09-08] Changed n_tokens > 1 to n_tokens >= 1:
    //   MTP verify ctx validates one token at a time (n_tokens == 1), so n_tokens > 1
    //   caused trigger to never fire (loaded=0 always). The batched n_tokens > 1 case is
    //   handled by the same code path (first token's experts are at ids[0..n_expert_used-1]).
    if (cgc_draft_prefetch_on() &&
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && n_tokens >= 1 && il == 1) {
        size_t queued = 0;
        size_t sync_loaded = 0;
        uint64_t sync_usec = 0;
        const size_t n_layers = cache->draft_prefetch_ids.size();
        const bool sync_mode = getenv("CGC_DRAFT_PREFETCH_SYNC") != nullptr &&
                               getenv("CGC_DRAFT_PREFETCH_SYNC")[0] == '1';
        for (size_t layer = 0; layer < n_layers; ++layer) {
            if (!cache->draft_prefetch_valid[layer]) continue;
            const std::vector<uint32_t> &pred = cache->draft_prefetch_ids[layer];
            const int32_t *st = cache->slot_table.data() + layer * cache->n_expert;
            size_t layer_queued = 0;
            size_t layer_cold = 0;
            for (uint32_t e : pred) {
                if (e >= cache->n_expert) continue;
                if (st[e] >= 0) continue;  // already resident
                layer_cold++;
                if (sync_mode) {
                    // SYNC: load directly in this thread (blocks decode, but safe)
                    const auto t0 = std::chrono::high_resolution_clock::now();
                    int32_t slot = llama_expert_cache_ensure_slot(cache, (uint32_t) layer, e, /*count=*/false);
                    const auto t1 = std::chrono::high_resolution_clock::now();
                    sync_usec += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                    if (slot >= 0) {
                        ++queued;
                        ++sync_loaded;
                        ++layer_queued;
                    }
                } else {
                    // ASYNC: queue for bg prefetch (non-blocking; slot_table publishes only after bytes land)
                    int32_t slot = llama_expert_cache_prefetch_slot(cache, (uint32_t) layer, e);
                    if (slot >= 0) {
                        ++queued;
                        ++layer_queued;
                    }
                }
            }
            // #9: One-shot consumption — experts that failed to queue (cap/slot) are never retried;
            // prediction consumed anyway. Count the cold experts in this layer that didn't get loaded.
            if (layer_cold > layer_queued) {
                const size_t skipped = layer_cold - layer_queued;
                cache->n_prefetch_dropped += skipped;
                cache->drop_stats.one_shot_consumed += skipped;
            }
            cache->draft_prefetch_valid[layer] = false;  // consumed
        }
        cache->n_draft_prefetch_queued += queued;
        if (sync_mode) {
            fprintf(stderr, "CGC-DRAFT-PF: trigger SYNC loaded=%zu experts across %zu layers (total_queued=%zu, sync_usec=%llu)\n",
                    sync_loaded, n_layers, (unsigned long long) cache->n_draft_prefetch_queued,
                    (unsigned long long) sync_usec);
        } else {
            fprintf(stderr, "CGC-DRAFT-PF: trigger ASYNC queued=%zu experts across %zu layers (total_queued=%zu)\n",
                    queued, n_layers, (unsigned long long) cache->n_draft_prefetch_queued);
        }
    }

    // [CGC prev-token prefetch 2026-09-08] TRIGGER phase: at il==1 of verify decode, prefetch
    // the PREVIOUS token's per-layer expert ids. Adjacent tokens have strong routing correlation
    // (~70-90% overlap), so this provides high-hit-rate prediction covering ALL layers (unlike
    // MTP draft which only has the last layer). CGC_PREV_TOKEN_PREFETCH=1 enables.
    // SYNC mode (CGC_PREV_TOKEN_PREFETCH_SYNC=1): load in this thread (blocks decode, but safe)
    // ASYNC mode (default): queue for bg prefetch (faster, but may need CGC_NO_PREFETCH removed)
    if (cgc_prev_token_prefetch &&
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && n_tokens >= 1 && il == 1) {
        size_t queued = 0;
        size_t sync_loaded = 0;
        uint64_t sync_usec = 0;
        const size_t n_layers = cache->prev_token_expert_ids.size();
        const bool sync_mode = getenv("CGC_PREV_TOKEN_PREFETCH_SYNC") != nullptr &&
                               getenv("CGC_PREV_TOKEN_PREFETCH_SYNC")[0] == '1';
        size_t valid_layers = 0;
        size_t total_cold = 0;
        size_t total_resident = 0;
        for (size_t layer = 0; layer < n_layers; ++layer) {
            if (!cache->prev_token_valid[layer]) continue;
            valid_layers++;
            const std::vector<uint32_t> &pred = cache->prev_token_expert_ids[layer];
            const int32_t *st = cache->slot_table.data() + layer * cache->n_expert;
            size_t layer_queued = 0;
            size_t layer_cold = 0;
            for (uint32_t e : pred) {
                if (e >= cache->n_expert) continue;
                if (st[e] >= 0) { total_resident++; continue; }  // already resident
                layer_cold++;
                total_cold++;
                if (sync_mode) {
                    // SYNC: load directly in this thread (blocks decode, but safe)
                    const auto t0 = std::chrono::high_resolution_clock::now();
                    int32_t slot = llama_expert_cache_ensure_slot(cache, (uint32_t) layer, e, /*count=*/false);
                    const auto t1 = std::chrono::high_resolution_clock::now();
                    sync_usec += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                    if (slot >= 0) {
                        ++queued;
                        ++sync_loaded;
                        ++layer_queued;
                    }
                } else {
                    // ASYNC: queue for bg prefetch (non-blocking)
                    int32_t slot = llama_expert_cache_prefetch_slot(cache, (uint32_t) layer, e);
                    if (slot >= 0) {
                        ++queued;
                        ++layer_queued;
                    }
                }
            }
            if (layer_cold > layer_queued) {
                const size_t skipped = layer_cold - layer_queued;
                cache->n_prefetch_dropped += skipped;
                cache->drop_stats.one_shot_consumed += skipped;
            }
        }
        cache->n_prev_token_prefetch_queued += queued;
        if (sync_mode) {
            fprintf(stderr, "CGC-PREV-PF: trigger SYNC loaded=%zu/%zu cold (valid_layers=%zu/%zu, resident=%zu, total_queued=%zu, sync_usec=%llu)\n",
                    sync_loaded, total_cold, valid_layers, n_layers, total_resident,
                    (unsigned long long) cache->n_prev_token_prefetch_queued,
                    (unsigned long long) sync_usec);
        } else {
            fprintf(stderr, "CGC-PREV-PF: trigger ASYNC queued=%zu/%zu cold (valid_layers=%zu/%zu, resident=%zu, total_queued=%zu)\n",
                    queued, total_cold, valid_layers, n_layers, total_resident,
                    (unsigned long long) cache->n_prev_token_prefetch_queued);
        }
    }

    // [CGC MTP Draft Prefetch 2026-09-07] HIT-RATE measurement: during trunk verify decode,
    // compare this layer's ACTUAL selected experts (uni) against the draft-predicted experts
    // (draft_prefetch_ids[il]) to measure prediction accuracy. This tells us how much of the
    // prefetch is wasted (miss) vs useful (hit). Draft_accept ~92-98% should translate to
    // similar hit rates here.
    //
    // [FIX 2026-09-08] Changed n_tokens > 1 to n_tokens >= 1 (same reason as TRIGGER phase).
    if (cgc_draft_prefetch_on() &&
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && n_tokens >= 1 &&
        il >= 0 && (size_t) il < cache->draft_prefetch_ids.size() &&
        cache->draft_prefetch_valid[il]) {
        const std::vector<uint32_t> &pred = cache->draft_prefetch_ids[il];
        std::unordered_set<uint32_t> pred_set(pred.begin(), pred.end());
        size_t hit = 0, miss = 0;
        for (uint32_t e : uni) {
            if (pred_set.count(e)) ++hit;
            else ++miss;
        }
        cache->n_draft_prefetch_hit += hit;
        cache->n_draft_prefetch_miss += miss;
        if (il <= 1) {
            fprintf(stderr, "CGC-DRAFT-PF: hitrate il=%d hit=%zu miss=%zu rate=%.1f%% (cum_hit=%zu cum_miss=%zu)\n",
                    il, hit, miss,
                    (hit + miss) > 0 ? (double) hit / (hit + miss) * 100.0 : 0.0,
                    cache->n_draft_prefetch_hit, cache->n_draft_prefetch_miss);
        }
    }

    // large prefill (multi-token beyond the pool path): record route frequencies for the hot
    // prewarm; the FFN tensors were restored above, so it computes over the full expert weights.
    // No remap leaf is created for such batches (graph.cpp builds it only when n_tokens <=
    // cgc_pool_max_tokens()), so there is nothing to fill here. Small multi-token batches
    // (speculative/MTP verify) fall through to the pool/gather path below, where the union spans
    // ALL tokens and the remap leaf is written for every token.
    //
    // [CGC M2 prefill streaming 2026-09-14] When CGC_PREFILL_STREAM=1, instead of returning here
    // we fill a whole-layer slab (all 256 experts, read straight from the GGUF via preadv) and
    // repoint each FFN weight tensor at it. The slab is the same cgc_gather_slab infrastructure
    // used by the wide-union path, but sized to n_expert (set CGC_GATHER_SLAB_CAP=256). Because
    // no remap leaf exists for large batches, mul_mat_id reads RAW expert ids — and with ne[2] set
    // to n_expert those ids are in range by construction. This is what lifts the n_batch clamp:
    // prefill chunks of 2048 run the FFN against a freshly-streamed full expert set instead of
    // the shrunk pool tensor.
    const bool cgc_prefill_stream = cgc_stream_on;  // validated once in the constructor
    if ((uint64_t) n_tokens > cgc_pool_max_tokens()) {
        // [CGC 2026-09-15] env-gated: this fired unconditionally on every large-batch step
        // (10 lines per run, on the hot prefill path) and was committed by accident. Opt-in via
        // CGC_M2_DBG. Static-initialized once so the getenv cost is paid a single time, not per
        // layer per step.
        static const bool cgc_m2_dbg = getenv("CGC_M2_DBG") != nullptr;
        static int dbg_cnt = 0;
        if (cgc_m2_dbg && dbg_cnt < 10) {
            dbg_cnt++;
            fprintf(stderr, "CGC-M2-DBG: il=%d n_tokens=%lld pmax=%u stream=%d\n",
                    il, (long long) n_tokens, cgc_pool_max_tokens(), (int) cgc_prefill_stream);
        }
        // [M2 debug 2026-09-14] Print standard-path prefill tensor state
        static const bool m2_state_dbg = getenv("CGC_M2_STATE_DBG") != nullptr;
        if (m2_state_dbg && il == 0) {
            for (int kind = 0; kind < 1; ++kind) {
                ggml_tensor * wt = cache_ffn_tensors[il][kind];
                if (wt != nullptr) {
                    fprintf(stderr, "CGC-STD-PREFILL: il=%d kind=%d ntok=%lld "
                            "ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p buf=%p buft=%s\n",
                            il, kind, (long long) n_tokens,
                            (long long) wt->ne[0], (long long) wt->ne[1],
                            (long long) wt->ne[2], (long long) wt->ne[3],
                            wt->nb[0], wt->nb[1], wt->nb[2], wt->nb[3],
                            wt->data, (void *) wt->buffer,
                            wt->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(wt->buffer)) : "null");
                }
            }
        }
        llama_expert_cache_record_routes(cache, (uint32_t) il, routes.data(), routes.size());
        if (!cgc_prefill_stream) {
            return;
        }
        // [CGC M2 double-buffer 2026-09-14] Overlap slab fill with graph build:
        // - Layer 0: synchronously fill set 0, launch background fill of layer 1 into set 1
        // - Layer L>0: wait for background fill of layer L into current set, then launch
        //   background fill of layer L+1 into the other set
        // - The swap happens at each layer boundary, so the current set always has fresh data.
        static const bool cgc_db_enable = getenv("CGC_M2_DB_DISABLE") == nullptr;
        const int n_layers = (int) model.hparams.n_layer();
        if (cgc_db_enable) {
            // Initialize both slab sets on first layer (il == 0)
            if (il == 0) {
                // Wait for any leftover background job from a previous graph build
                if (cgc_db_init) {
                    cgc_db_wait();
                }
                cgc_db_cur = 0;
                if (!cgc_db_init) {
                    // First time: allocate both slab sets
                    for (int set = 0; set < 2; ++set) {
                        for (int kind = 0; kind < 4; ++kind) {
                            ggml_tensor * wt = cache_ffn_tensors[0][kind];
                            if (wt == nullptr) continue;
                            const size_t exp_bytes = ggml_row_size(wt->type, wt->ne[0]) * wt->ne[1];
                            ggml_backend_buffer_type_t wt_buft = wt->buffer != nullptr
                                    ? ggml_backend_buffer_get_type(wt->buffer) : nullptr;
                            if (llama_expert_cache_pool_buffer(cache, 0, kind) != nullptr) {
                                wt_buft = ggml_backend_buffer_get_type(
                                        llama_expert_cache_pool_buffer(cache, 0, kind));
                            }
                            if (wt_buft != nullptr && strcmp(ggml_backend_buft_name(wt_buft), "CPU") == 0) {
                                for (size_t bi = 0; bi < backends.size(); ++bi) {
                                    auto dev = ggml_backend_get_device(backends[bi].get());
                                    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                                        auto * dev_buft = ggml_backend_get_default_buffer_type(backends[bi].get());
                                        if (dev_buft) { wt_buft = dev_buft; break; }
                                    }
                                }
                            }
                            cgc_gather_slab * sl = cgc_gather_slab_get(kind, exp_bytes, wt_buft);
                            if (sl != nullptr) {
                                if (set == 1) {
                                    // Allocate a second buffer for set 1
                                    const size_t want = (size_t) cgc_gather_slab_cap() * sl->stride;
                                    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(wt_buft, want);
                                    if (buf != nullptr) {
                                        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                                        cache_gather_slab.push_back(cgc_gather_slab{});
                                        cgc_gather_slab & s2 = cache_gather_slab.back();
                                        s2.kind   = kind;
                                        s2.stride = sl->stride;
                                        s2.buf    = buf;
                                        s2.base   = (uint8_t *) ggml_backend_buffer_get_base(buf);
                                        s2.size   = want;
                                        cgc_db_slab[set][kind] = (int) (cache_gather_slab.size() - 1);
                                        fprintf(stderr, "CGC-M2-DB: allocated set=%d kind=%d slab=%.2f MiB\n",
                                                set, kind, (double) want / (1024.0*1024.0));
                                    }
                                } else {
                                    for (size_t i = 0; i < cache_gather_slab.size(); ++i) {
                                        if (&cache_gather_slab[i] == sl) {
                                            cgc_db_slab[set][kind] = (int) i;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    cgc_db_init = true;
                }
                // [CGC M2 pool reuse 2026-09-14] Cache every (layer, kind)'s own per-expert byte
                // count once: that is the pitch the fill must use (see cgc_db_stride in the header).
                // The slab's stride is the kind's whole-model MAX, so on this mixed-quant GGUF the
                // two differ for most layers.
                if (cgc_db_stride.empty()) {
                    const int nl = (int) model.hparams.n_layer();
                    cgc_db_stride.assign((size_t) nl, std::vector<size_t>(4, 0));
                    for (const auto & it : cache_ffn_tensors) {
                        const int l = it.first;
                        if (l < 0 || l >= nl) continue;
                        for (size_t k = 0; k < it.second.size() && k < 4; ++k) {
                            const ggml_tensor * t = it.second[k];
                            if (t == nullptr) continue;
                            cgc_db_stride[(size_t) l][k] = ggml_row_size(t->type, t->ne[0]) * t->ne[1];
                        }
                    }
                }
                // Synchronously fill layer 0 into set 0
                for (int kind = 0; kind < 4; ++kind) {
                    const int si = cgc_db_slab[0][kind];
                    if (si < 0) continue;
                    cgc_gather_slab & sl = cache_gather_slab[(size_t) si];
                    const size_t sp = (!cgc_db_stride.empty() && cgc_db_stride[0][(size_t) kind] > 0)
                            ? cgc_db_stride[0][(size_t) kind]
                            : sl.stride;
                    llama_expert_cache_fill_layer_slab(cache, 0, kind, sl.base, sp);
                }
                // Launch background fill of layer 1 into set 1
                if (n_layers > 1) {
                    cgc_db_start_prefill(1, 1);
                }
            } else if (cgc_db_init) {
                // Wait for background fill of this layer into current set
                cgc_db_wait();
                // Swap to the set that was just filled
                cgc_db_cur = 1 - cgc_db_cur;
                // Launch background fill of next layer into the other set
                if (il + 1 < n_layers) {
                    cgc_db_start_prefill(il + 1, 1 - cgc_db_cur);
                }
            }
        }
        // Whole-layer slab path: fill + repoint each kind's expert tensor.
        const uint32_t n_expert_full = model.hparams.n_expert;
        for (int kind = 0; kind < 4; ++kind) {
            ggml_tensor * wt = cache_ffn_tensors[il][kind];
            if (wt == nullptr) {
                continue;
            }
            // [M2 debug 2026-09-14] Print tensor state BEFORE repoint
            static const bool m2_state_dbg = getenv("CGC_M2_STATE_DBG") != nullptr;
            if (m2_state_dbg && il == 0 && kind == 0) {
                fprintf(stderr, "CGC-M2-STATE-BEFORE: il=%d kind=%d ntok=%lld "
                        "ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p buf=%p buft=%s\n",
                        il, kind, (long long) n_tokens,
                        (long long) wt->ne[0], (long long) wt->ne[1],
                        (long long) wt->ne[2], (long long) wt->ne[3],
                        wt->nb[0], wt->nb[1], wt->nb[2], wt->nb[3],
                        wt->data, (void *) wt->buffer,
                        wt->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(wt->buffer)) : "null");
            }
            const size_t exp_bytes = ggml_row_size(wt->type, wt->ne[0]) * wt->ne[1];
            ggml_backend_buffer_type_t wt_buft = wt->buffer != nullptr
                    ? ggml_backend_buffer_get_type(wt->buffer) : nullptr;
            // Use the pool's buffer type (host-visible Metal shared storage) so the CPU pread
            // fill below actually lands where Metal can read it (same reasoning as the wide-union
            // path in cgc_apply_expert_geometry).
            if (llama_expert_cache_pool_buffer(cache, (uint32_t) il, kind) != nullptr) {
                wt_buft = ggml_backend_buffer_get_type(
                        llama_expert_cache_pool_buffer(cache, (uint32_t) il, kind));
            }
            // [M2 fix 2026-09-14] If expert tensor was skip-loaded (--load-mode none), its buffer
            // is CPU-only and the slab would be allocated as CPU → Metal mul_mat_id SIGSEGVs.
            // Fall back to the first non-CPU backend's default buffer type.
            if (wt_buft != nullptr) {
                const char * buft_name = ggml_backend_buft_name(wt_buft);
                if (strcmp(buft_name, "CPU") == 0) {
                    for (size_t bi = 0; bi < backends.size(); ++bi) {
                        auto dev = ggml_backend_get_device(backends[bi].get());
                        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                            auto * dev_buft = ggml_backend_get_default_buffer_type(backends[bi].get());
                            if (dev_buft) {
                                wt_buft = dev_buft;
                                fprintf(stderr, "CGC-PREFILL-STREAM: CPU buft for il=%d kind=%d "
                                        "→ using %s\n", il, kind, ggml_backend_buft_name(wt_buft));
                                break;
                            }
                        }
                    }
                }
            }
            cgc_gather_slab * sl = nullptr;
            if (cgc_db_enable && cgc_db_init) {
                // Double-buffer path: use the pre-filled slab from the current set
                const int si = cgc_db_slab[cgc_db_cur][kind];
                if (si >= 0) {
                    sl = &cache_gather_slab[(size_t) si];
                }
            } else {
                // Original path: allocate + fill synchronously
                sl = cgc_gather_slab_get(kind, exp_bytes, wt_buft);
                if (sl != nullptr) {
                    const int64_t filled = llama_expert_cache_fill_layer_slab(cache, (uint32_t) il,
                            kind, sl->base, exp_bytes);
                    if (filled < 0) {
                        fprintf(stderr, "CGC-PREFILL-STREAM: fill_layer_slab FAILED il=%d kind=%d — "
                                "falling back to original (shrunk) weights, values NOT trustworthy\n",
                                il, kind);
                        sl = nullptr;
                    }
                }
            }
            if (sl == nullptr) {
                continue;
            }
            // Record original state ONCE per (layer, kind) so the graph-build restore lands on
            // the consistent shrunk-tensor state (same mechanism as the wide-union path).
            const bool first_repoint =
                    cache_gather_ne2.find({il, kind}) == cache_gather_ne2.end();
            if (first_repoint) {
                cache_orig[{il, kind}]       = wt->data;
                cache_gather_ne2[{il, kind}] = wt->ne[2];
                cache_gather_orig_buf[{il, kind}] = wt->buffer;
            }
            wt->ne[2]  = (int64_t) n_expert_full;
            wt->data   = sl->base;
            wt->buffer = sl->buf;
            // [M2 debug 2026-09-14] Print tensor state AFTER repoint
            if (m2_state_dbg && il == 0 && kind == 0) {
                fprintf(stderr, "CGC-M2-STATE-AFTER: il=%d kind=%d ntok=%lld "
                        "ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p buf=%p buft=%s "
                        "slab_stride=%zu exp_bytes=%zu\n",
                        il, kind, (long long) n_tokens,
                        (long long) wt->ne[0], (long long) wt->ne[1],
                        (long long) wt->ne[2], (long long) wt->ne[3],
                        wt->nb[0], wt->nb[1], wt->nb[2], wt->nb[3],
                        wt->data, (void *) wt->buffer,
                        wt->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(wt->buffer)) : "null",
                        sl->stride, exp_bytes);
            }
            if (il <= 1 && kind == 0) {
                static int ps_dbg = 0;
                if (ps_dbg < 4) {
                    ps_dbg++;
                    const int64_t filled = (cgc_db_enable && cgc_db_init) ? (int64_t) sl->size : -1;
                    fprintf(stderr, "CGC-PREFILL-STREAM: il=%d kind=%d ntok=%lld "
                            "experts=%u slab=%.2f MiB filled=%lld bytes db=%d\n",
                            il, kind, (long long) n_tokens, n_expert_full,
                            (double) sl->size / (1024.0 * 1024.0), (long long) filled,
                            (int) (cgc_db_enable && cgc_db_init));
                }
            }
        }
        return;
    }

    const uint32_t n_expert = model.hparams.n_expert;

    // [CGC 2026-09-15 S1 slot-table diagnostic] One line per hook call for the first 60 calls:
    // is the GPU table actually present in the capture map, and is it the same tensor the graph
    // built? This separates "capture never happened" from "capture happened but the GPU reads
    // something else", which the CGC-MMID-ASSERT (id_oob) alone cannot distinguish.
    static const bool cgc_s1_dbg = getenv("CGC_S1_DBG") != nullptr;
    if (cgc_s1_dbg && n_tokens <= 2) {
        static int cgc_s1_n = 0;
        if (cgc_s1_n < 80) {
            const auto it_leaf = cache_remap_tensors.find(il);
            const auto it_tbl  = cache_slot_table_tensors.find(il);
            ggml_tensor * tl = it_leaf != cache_remap_tensors.end() ? it_leaf->second : nullptr;
            ggml_tensor * tt = it_tbl  != cache_slot_table_tensors.end() ? it_tbl->second  : nullptr;
            char keys[256];
            int  klen = 0;
            keys[0] = 0;
            for (const auto & kv : cache_slot_table_tensors) {
                klen += snprintf(keys + klen, sizeof(keys) - (size_t) klen, "%d,", kv.first);
                if (klen > (int) sizeof(keys) - 8) {
                    break;
                }
            }
            fprintf(stderr, "CGC-S1: hook t=%s il=%d ne=[%lld,%lld,%lld,%lld] n_tok=%lld n_eu=%lld "
                            "tbl_map=%zu keys=[%s] tbl=%p(tbl_data=%p) leaf=%p\n",
                    t->name, il, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    (long long) n_tokens, (long long) n_expert_used,
                    cache_slot_table_tensors.size(), keys,
                    (void *) tt, tt ? tt->data : nullptr, (void *) tl);
            cgc_s1_n++;
        }
    }

    // L4_SKIP_LAYER0: blk.0 is a full-weight CPU skip-load tensor, not pooled. Its FFN must keep
    // reading the ORIGINAL tensor with the raw expert ids, so write an IDENTITY remap (instead of
    // slot indices) and skip ensure_batch / union recording. Writing slot ids would make mul_mat_id
    // index pool slots that have no adopted region -> layer-0 garbage that corrupts the whole net.
    if (getenv("LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0") != nullptr && il == 0) {
        // [CGC 2026-09-15 S1 slot-table] This path deliberately writes an IDENTITY map (raw expert
        // ids, not slots) because layer 0's FFN reads the full-weight tensor. The GPU table must
        // carry the same identity -- otherwise the graph's get_rows would hand mul_mat_id pool slot
        // indices for a layer whose weights were never repointed at the pool.
        ggml_tensor * cgc_stable0 = cache_slot_table_tensors[il];
        if (cgc_stable0 != nullptr && cgc_stable0->data != nullptr) {
            int32_t * td0 = (int32_t *) cgc_stable0->data;
            for (int64_t e = 0; e < n_expert; ++e) {
                td0[e] = (int32_t) e;
            }
        }
        cgc_s1_expect_dbg("L4", il, n_tokens, n_expert_used, ids, cgc_stable0);
        // layer 0's contract is the IDENTITY map on purpose (full-width weights), so the expected
        // table entry is the raw expert id, not a slot.
        cgc_s1_equiv_dbg("L4", cache, il, n_tokens, n_expert_used, ids, cgc_stable0, true);
        ggml_tensor * remap = cache_remap_tensors[il];
        if (remap != nullptr && remap->data != nullptr) {
            int32_t * rd = (int32_t *) remap->data;
            for (int64_t j = 0; j < n_tokens; ++j) {
                for (int64_t i = 0; i < n_expert_used; ++i) {
                    rd[i + j * n_expert_used] = (int32_t) ids[i + j * n_expert_used];
                }
            }
        }
        return;
    }

    // [CGC routing-aware placement §1/2 2026-08-29] record route frequencies for ALL pool-path
    // steps (fast touch+ZERO and catch-up ensure alike). The stock record_routes call above only
    // fires on n_tokens > pmax large-prefill steps, which never happen under L4 chunked prefill
    // (n_batch=8 <= pmax) — that is why freq stayed empty in the prewarm_hot dead-code era.
    // Pure counting under the cache lock: no pool writes, no deadlock surface (the 2026-08-28
    // GPU deadlock was prewarm_hot's bulk FILLS, not recording). LLAMA_EXPERT_CACHE_ROUTE_RECORD=1
    // opt-in; default off = stock behavior untouched. Consumed by the destructor's
    // LLAMA_EXPERT_CACHE_ROUTE_DUMP (top-K list + coverage verdict).
    {
        static const bool route_record = getenv("LLAMA_EXPERT_CACHE_ROUTE_RECORD") != nullptr;
        if (route_record) {
            llama_expert_cache_record_routes(cache, (uint32_t) il, routes.data(), routes.size());
        }
    }

    // decode: L3 Option A static per-layer slot pool when active and the union fits, else the
    // L3-B per-step gather path.
    const uint32_t n_usable = llama_expert_cache_usable_slots(cache, (uint32_t) il);
    // [CGC M1 union-routable 2026-09-14] Record what the union actually was, and which of the two
    // decode paths it was routed to. The M0 union>slots case (2 GiB) failed M1 2/42, but the union
    // AT the divergent layer was never logged -- the chain "union > usable -> gather -> host buffer
    // -> Metal nil -> wrong values" was inferred from the pool geometry (n_slots=35) plus 234
    // `buffer is nil` dispatches, not observed. This closes that gap.
    //
    // The predicate below is the true pool-vs-gather split: the pool branch returns at its own end
    // (see the "L3-B gather path" block further down), so a false predicate IS the gather path.
    //
    // Note on slots vs usable -- FIXED 2026-09-14. llama_expert_cache_slots_per_layer_l() returns
    // the RAW count, while the last slot may be the reserved ZERO slot
    // (llama_expert_cache_zero_slot() == slots_l - 1, and zero_slot_enabled() is true only under
    // the MTP fast path). So under CGC_VERIFY_DECODE / CGC_DRAFT_DECODE the pool path can hold only
    // usable = n_slots - 1 experts, while the gate used to admit union <= n_slots -- a
    // one-expert-wide window (union == n_slots) that reached ensure_batch and aborted
    // ("distinct experts exceed the usable pool slots", the caps32 abort).
    //
    // That tightening was deliberately withheld while the fallback was broken: routing
    // (usable, n_slots] to the gather path produced WRONG VALUES rather than an abort, so the plan's
    // 6 called the abort "the lesser evil until the Metal slab lands". The slab has landed and is
    // verified (2 GiB: 117/117 bit-identical, 0 nil), so the gate now uses the usable count: the
    // window goes to the gather path, which is correct there. The log below still reports usable
    // (and raw) so the window stays observable.
    //
    // Inert unless CGC_UNION_LOG=1. One block per 8 layer sweeps, ranked by max union.
    static const bool cgc_union_log = getenv("CGC_UNION_LOG") != nullptr;
    static uint32_t cgc_ul_max[128]  = {0};
    static uint32_t cgc_ul_min[128]  = {0};
    static uint64_t cgc_ul_sum[128]  = {0};
    static uint32_t cgc_ul_n[128]    = {0};
    static uint32_t cgc_ul_gath[128] = {0};
    static int      cgc_ul_sweeps    = 0;
    static int      cgc_ul_prev_il   = -1;
    const bool cgc_union_pool = llama_expert_cache_pool_active(cache) && uni.size() <= n_usable;
    if (cgc_union_log && il >= 0 && il < 128) {
        const uint32_t cgc_u = (uint32_t) uni.size();
        if (cgc_ul_n[il] == 0 || cgc_u < cgc_ul_min[il]) {
            cgc_ul_min[il] = cgc_u;
        }
        if (cgc_u > cgc_ul_max[il]) {
            cgc_ul_max[il] = cgc_u;
        }
        cgc_ul_sum[il] += cgc_u;
        cgc_ul_n[il]++;
        if (!cgc_union_pool) {
            cgc_ul_gath[il]++;
        }
        // a sweep ends when the layer index wraps back down
        if (cgc_ul_prev_il >= 0 && il < cgc_ul_prev_il && (++cgc_ul_sweeps % 8) == 0) {
            uint32_t cgc_mx[128];
            for (int l = 0; l < 128; l++) {
                cgc_mx[l] = cgc_ul_max[l];
            }
            for (int rank = 0; rank < 8; rank++) {
                int cgc_best = -1;
                for (int l = 0; l < 128; l++) {
                    if (cgc_ul_n[l] == 0 || cgc_mx[l] == 0) {
                        continue;
                    }
                    if (cgc_best < 0 || cgc_mx[l] > cgc_mx[cgc_best]) {
                        cgc_best = l;
                    }
                }
                if (cgc_best < 0) {
                    break;
                }
                const uint32_t cgc_us  = llama_expert_cache_usable_slots(cache, (uint32_t) cgc_best);
                const uint32_t cgc_raw = llama_expert_cache_slots_per_layer_l(cache, (uint32_t) cgc_best);
                fprintf(stderr,
                        "CGC-UNION: layer=%d union avg=%.1f min=%u max=%u of usable=%u (raw=%u) "
                        "(%.0f%%)%s gather=%u/%u\n",
                        cgc_best,
                        cgc_ul_n[cgc_best] ? (double) cgc_ul_sum[cgc_best] / (double) cgc_ul_n[cgc_best] : 0.0,
                        cgc_ul_min[cgc_best], cgc_ul_max[cgc_best], cgc_us, cgc_raw,
                        cgc_us ? 100.0 * (double) cgc_ul_max[cgc_best] / (double) cgc_us : 0.0,
                        cgc_ul_max[cgc_best] > cgc_us ? "  [WIDE: exceeds usable]" : "",
                        cgc_ul_gath[cgc_best], cgc_ul_n[cgc_best]);
                cgc_mx[cgc_best] = 0;
            }
            for (int l = 0; l < 128; l++) {
                cgc_ul_max[l] = 0;
                cgc_ul_min[l] = 0;
                cgc_ul_sum[l] = 0;
                cgc_ul_n[l]   = 0;
                cgc_ul_gath[l] = 0;
            }
        }
        cgc_ul_prev_il = il;
    }
    if (cgc_union_pool) {
        // [CGC M1 2026-09-14] Un-repoint anything this tensor was left in GATHER state in.
        //
        // The pool branch does not repoint; it relies on the repoint done in graph_get_cb at
        // BUILD time. With a REUSED graph that build never ran, so a tensor left holding the slab
        // (data/buffer = slab, ne[2] = union) would be read here with POOL slot ids -- the slab
        // holds the previous step's sorted union, so this silently reads the wrong experts rather
        // than failing. Putting the recorded pre-gather values back IS the pool state (data = the
        // pool region, buffer = the model buffer, ne[2] = the raw slot count), which is exactly
        // what the gather hook recorded on its first call.
        //
        // Reachable whenever a layer flips gather -> pool between two steps of the same shape
        // (same shape => reuse => no restore), and that is the common case as soon as the pool is
        // smaller than cap x top_k. Verified on the caps config below and on 2 GiB.
        for (int k = 0; k < 4; ++k) {
            auto it = cache_gather_ne2.find({il, k});
            if (it == cache_gather_ne2.end()) {
                continue;
            }
            auto it_ffn = cache_ffn_tensors.find(il);
            ggml_tensor * wt = (it_ffn != cache_ffn_tensors.end() &&
                                (size_t) k < it_ffn->second.size())
                    ? it_ffn->second[(size_t) k] : nullptr;
            if (wt != nullptr) {
                auto ito = cache_orig.find({il, k});
                if (ito != cache_orig.end()) {
                    wt->data = ito->second;
                    cache_orig.erase(ito);
                }
                auto itb = cache_gather_orig_buf.find({il, k});
                if (itb != cache_gather_orig_buf.end()) {
                    wt->buffer = itb->second;
                    cache_gather_orig_buf.erase(itb);
                }
                wt->ne[2] = it->second;
            }
            cache_gather_ne2.erase(it);
        }
        // [CGC MTP fast path] CGC_VERIFY_DECODE / CGC_DRAFT_DECODE (rebuilt 2026-08-28 from
        // [CGC Phase Discrimination 2026-09-08] Use explicit phase marker set by the caller
        // (server-context.cpp / common/speculative.cpp) instead of unreliable n_tokens/seq_pos_max
        // heuristics. Phase is set before each llama_decode() call.
        //   - VERIFY: target context MTP verify batch (can be n_tokens==1 when L4 splits it)
        //   - DRAFT:  MTP draft context single-token decode
        //   - PREFILL / CATCHUP / NORMAL_DECODE / UNKNOWN: exact ensure_batch path, no ZERO-slot
        // UNKNOWN is the safe default: when caller forgets to set phase, we fall back to exact path.
        const cgc_phase_t cgc_phase = cgc_current_phase;
        const bool verify_fast = getenv("CGC_VERIFY_DECODE") != nullptr &&
            cgc_phase == CGC_PHASE_VERIFY &&
            cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT;
        const bool draft_fast = getenv("CGC_DRAFT_DECODE") != nullptr &&
            cgc_phase == CGC_PHASE_DRAFT &&
            cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && n_tokens == 1;
        // [CGC warmup gate 2026-08-30] 短 prompt 的 prefill 餵不飽 pool（55 tok ≈ 7 chunk），
        // decode 冷啟動期大量 cold expert 被 ZERO-slot 讀成零權重 → logits 崩潰成 0000...。
        // 根因是「pool 未餵飽」而非「cold 率本身」（長/短 prompt decode 期 cold 分佈相近，用
        // ratio 門檻會誤傷長 prompt）。確定性修復：n_past 未達暖機門檻（pool 尚未餵飽）時
        // 禁用 fast path，走下方 exact ensure_batch 補槽；n_past 達標後恢復 fast path。長
        // prompt prefill 已餵飽 pool，decode 首步 n_past 就超門檻，完全不受影響。env
        // CGC_WARM_NPAST 可調（預設 2048，從 256 提高以覆蓋更多 prefill 場景）。
        // Note: warmup gate is kept as a second-layer safety net, but the primary phase
        // discrimination now comes from cgc_current_phase set by the caller.
        const long long cgc_n_past = llama_memory_seq_pos_max(get_memory(), 0);
        const char * cgc_warm_env = getenv("CGC_WARM_NPAST");
        const long long cgc_warm_npast = cgc_warm_env ? atoll(cgc_warm_env) : 2048;
        const bool cgc_warm_gate = cgc_n_past < cgc_warm_npast;
        bool cgc_fast_eligible = !cgc_warm_gate;

        // [CGC Phase safety] Only VERIFY and DRAFT phases are allowed to use the fast path
        // (ZERO-slot). PREFILL / CATCHUP / NORMAL_DECODE / UNKNOWN must use exact ensure_batch.
        // This is the primary gate; warmup gate above is secondary.
        if (cgc_phase != CGC_PHASE_VERIFY && cgc_phase != CGC_PHASE_DRAFT) {
            cgc_fast_eligible = false;
        }

        // [診斷] 記錄 phase 判別結果（CGC_PHASE_DBG=1 啟用，只看前 2 層避免日誌過多）
        if (getenv("CGC_PHASE_DBG") != nullptr && il <= 1) {
            const char * phase_names[] = {"UNKNOWN", "PREFILL", "VERIFY", "DRAFT", "CATCHUP", "NORMAL_DECODE"};
            fprintf(stderr, "CGC-PHASE-DBG: il=%d phase=%s n_past=%lld n_tokens=%lld "
                    "warm_gate=%d verify_fast=%d draft_fast=%d fast_eligible=%d\n",
                    il, phase_names[cgc_phase], cgc_n_past, (long long)n_tokens,
                    (int)cgc_warm_gate, (int)verify_fast, (int)draft_fast, (int)cgc_fast_eligible);
        }

        // [CGC Fast-Path Cold Guard] When too many of this step's top-k experts are cold
        // (slot_table == -1), the ZERO-slot contamination exceeds softmax's absorption
        // capacity -> uniform softmax -> deterministic sequential top-k -> 0000 garbage.
        // Disable fast path so cold experts get filled via ensure_batch.
        // [CGC Step-2 weighted cold guard consume 2026-09-06] Decide with the WEIGHTED cold
        // ratio when the logits eval callback measured it this step (sum of routing probs of
        // cold experts / total, written into cgc_weighted_cold_ratio[il] before this hook fires;
        // 1e9 sentinel = unmeasured). The count fallback below treats every cold expert as
        // equally harmful, so a layer whose few cold experts carry tiny routing mass trips the
        // guard and loses the fast path for no quality gain; the weighted metric keeps it on
        // the fast path. When it DOES fall through, ensure_batch below only fills the COLD
        // members of the union (warm members are hit-refreshed), i.e. selective fill, not the
        // whole 256-expert layer.
        if (cgc_fast_eligible && cache != nullptr && il >= 0 &&
            (uint32_t) il < cache->slot_owner.size()) {
            const char * cgc_cold_env = getenv("CGC_FAST_COLD_MAX");
            const double cgc_fast_cold_max = cgc_cold_env ? atof(cgc_cold_env) : 0.30;
            if (cgc_fast_cold_max > 0.0) {
                const int32_t * cgc_st = cache->slot_table.data() + (size_t) il * cache->n_expert;
                // count-based cold set (both the fallback ratio and the guard log need it)
                size_t cgc_cold_count = 0;
                for (size_t i = 0; i < uni.size(); ++i) {
                    uint32_t e = uni[i];
                    if (e < cache->n_expert && cgc_st[e] < 0) {
                        cgc_cold_count++;
                    }
                }
                // weighted ratio (produce side: expert_cache_eval_cb on the layer's gate
                // probs/logits). Unmeasured layers keep the 1e9 sentinel -> count fallback.
                const bool cgc_wr_ok =
                    (size_t) il < cache->cgc_weighted_cold_ratio.size() &&
                    cache->cgc_weighted_cold_ratio[(size_t) il] < 1e9;
                const double cgc_metric = cgc_wr_ok
                    ? cache->cgc_weighted_cold_ratio[(size_t) il]
                    : (uni.size() > 0 ? (double) cgc_cold_count / uni.size() : 0.0);
                if (cgc_metric > cgc_fast_cold_max) {
                    cgc_fast_eligible = false;
                    if (il <= 1) {
                        if (cgc_wr_ok) {
                            fprintf(stderr, "CGC-COLD-GUARD-W: il=%d cold=%zu/%zu weighted=%.1f%% (count=%.1f%%) > %.0f%% -> ensure_batch (selective fill)\n",
                                    il, cgc_cold_count, uni.size(), cgc_metric * 100.0,
                                    (uni.size() > 0 ? (double) cgc_cold_count / uni.size() * 100.0 : 0.0),
                                    cgc_fast_cold_max * 100.0);
                        } else {
                            fprintf(stderr, "CGC-COLD-GUARD: il=%d cold=%zu/%zu (%.0f%% > %.0f%%) -> ensure_batch\n",
                                    il, cgc_cold_count, uni.size(), cgc_metric * 100.0, cgc_fast_cold_max * 100.0);
                        }
                    }
                }
            }
        }
        // [CGC Draft Guard 2026-09-08] Strict cold guard for the MTP draft context.
        // The generic guard above uses CGC_FAST_COLD_MAX=0.30 (default) which almost never
        // fires for draft: draft's union is only 8 experts (1 token x 8), so 1-2 cold =
        // 12-25% < 30% -> draft keeps ZERO-mapping. Draft ZERO skews the proposed token
        // distribution and cascades into verify accept decisions.
        // Fix: draft is a cheap 1-token prediction (<=8 experts fill cost), so fall back to
        // exact ensure_batch on ANY cold expert. Gated by CGC_DRAFT_STRICT (default 1).
        if (cgc_fast_eligible && draft_fast) {
            const char * cgc_ds_env = getenv("CGC_DRAFT_STRICT");
            const bool cgc_draft_strict = cgc_ds_env ? atoi(cgc_ds_env) != 0 : true;
            if (cgc_draft_strict) {
                const int32_t * cgc_st = cache->slot_table.data() + (size_t) il * cache->n_expert;
                size_t cgc_dcold = 0;
                for (size_t i = 0; i < uni.size(); ++i) {
                    const uint32_t e = uni[i];
                    if (e < cache->n_expert && cgc_st[e] < 0) {
                        cgc_dcold++;
                    }
                }
                if (cgc_dcold > 0) {
                    cgc_fast_eligible = false;
                    if (il <= 1) {
                        fprintf(stderr, "CGC-DRAFT-GUARD: il=%d cold=%zu/%zu -> exact ensure_batch (strict draft)\n",
                                il, cgc_dcold, uni.size());
                    }
                }
            }
        }
        // [CGC verify-strict 2026-09-13] The verify context is GROUND TRUTH: it must never read
        // the reserved ZERO slot. The ZERO slot substitutes a zeroed region for a cold expert,
        // i.e. that expert's contribution silently disappears — and HOW MANY disappear depends on
        // how many slots the pool has, which is exactly how the pool size leaked into the values
        // (measured: 4/6/8 GiB diverged at the first multi-token step, 7/39 identically on iq3 and
        // iq4, while every single-token step was bit-identical). CGC_SYNCFILL_COLD (default on)
        // closed that gap by filling the step's selected cold experts first; this prologue makes
        // the invariant STRUCTURAL instead of a default that an A/B can silently flip:
        //   1. fill this step's selected cold experts (blocking, real bytes) when enabled,
        //   2. re-count and, if ANY selected expert is still cold (ensure_slot failure), REFUSE
        //      the fast path so the exact path below runs. The ZERO slot then cannot be read for a
        //      selected expert at all; CGC_VERIFY_STRICT=0 is the only way back to the old way.
        static const bool cgc_syncfill_cold = []() {
            const char * e = getenv("CGC_SYNCFILL_COLD");
            if (e == nullptr) {
                return true;
            }
            // a numeric value is compared explicitly: the old `!= nullptr` style silently treats
            // CGC_SYNCFILL_COLD=0 as ON.
            return e[0] != '0';
        }();
        if ((verify_fast || draft_fast) && cgc_fast_eligible) {
            const char * vs_env = getenv("CGC_VERIFY_STRICT");
            const bool cgc_verify_strict = vs_env ? atoi(vs_env) != 0 : true;
            if (cgc_verify_strict) {
                const int32_t * cst = cache->slot_table.data() + (size_t) il * cache->n_expert;
                size_t cgc_cold_before = 0;
                for (size_t i = 0; i < uni.size(); ++i) {
                    const uint32_t e = uni[i];
                    if (e < cache->n_expert && cst[e] < 0) {
                        cgc_cold_before++;
                    }
                }
                if (cgc_cold_before > 0 && cgc_syncfill_cold) {
                    // [CGC 2026-09-15] This used to be a serial `for` over the cold experts, one
                    // blocking llama_expert_cache_ensure_slot per expert. Measured on prod25+MTP
                    // that path is the MTP bottleneck, not the pool:
                    //
                    //   prewarm req=22902 hit=0 miss=22902   (ensure_slot with count=false)
                    //   file_reads=285279 over ~187 decode steps  = 1525 reads/step
                    //   pread_usec=553.78 s inside a 92.6 s wall  -> fully IO-bound
                    //   draft accept 0.735 / mean len 3.21 -> 6.48 t/s, WORSE than MTP-off (10.43)
                    //
                    // ~122 experts go cold per step and each one costs a blocking ~1.7 ms pread
                    // on the critical path, serialised across experts. The MTP fast path is
                    // supposed to hide exactly this behind the FFN dispatch, and ensure_batch is
                    // built for it: one lock for the whole layer, then ALL misses flattened into
                    // a single job list onto the persistent worker pool
                    // (LLAMA_EXPERT_CACHE_WORKERS, default 8) -- see the "L3 Option A batch:
                    // cross-expert parallel fill" comment in llama-expert-cache.cpp.
                    //
                    // Two further consequences of using the batch path, both intended:
                    //   * it increments n_requests/n_misses, so this cold traffic finally shows
                    //     up in the headline hit rate instead of hiding in the `prewarm` counters
                    //     (where hit=0 by construction, because the loop only ever asks for
                    //     experts whose slot table entry is already -1);
                    //   * it claims its slots under one lock with batch_owned mid-batch protection,
                    //     which is what makes concurrent cross-expert fill safe.
                    //
                    // CGC_SYNCFILL_SERIAL=1 restores the old serial loop for A/B.
                    static const bool cgc_syncfill_serial = getenv("CGC_SYNCFILL_SERIAL") != nullptr;
                    std::vector<uint32_t> cold;
                    cold.reserve(uni.size());
                    for (size_t i = 0; i < uni.size(); ++i) {
                        const uint32_t e = uni[i];
                        if (e < cache->n_expert && cst[e] < 0) {
                            cold.push_back(e);
                        }
                    }
                    const size_t n_cold_filled = cold.size();
                    if (n_cold_filled > 0) {
                        if (cgc_syncfill_serial) {
                            for (uint32_t e : cold) {
                                llama_expert_cache_ensure_slot(cache, (uint32_t) il, e, /*count=*/false);
                            }
                        } else {
                            // defer_decode_protect=false: this is a decode-phase fill, so the
                            // slots it installs are decode-reserved and a later prefill must
                            // defer them (same contract as the normal decode path).
                            llama_expert_cache_ensure_batch(cache, (uint32_t) il, cold.data(),
                                                            cold.size(), /*defer_decode_protect=*/false);
                        }
                    }
                    if (n_cold_filled > 0 && il <= 1) {
                        fprintf(stderr, "CGC-SYNCFILL: %s il=%d cold_filled=%zu (%s)\n",
                                cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "draft" : "verify",
                                il, n_cold_filled, cgc_syncfill_serial ? "serial" : "batch");
                    }
                }
                size_t cgc_cold_after = 0;
                for (size_t i = 0; i < uni.size(); ++i) {
                    const uint32_t e = uni[i];
                    if (e < cache->n_expert && cst[e] < 0) {
                        cgc_cold_after++;
                    }
                }
                if (cgc_cold_after > 0) {
                    cgc_fast_eligible = false;
                    cache->n_verify_strict_refused++;
                    if (il <= 1 || cache->n_verify_strict_refused <= 8) {
                        fprintf(stderr, "CGC-VERIFY-STRICT: %s il=%d cold=%zu/%zu (before=%zu) -> exact path, ZERO-slot refused\n",
                                cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "draft" : "verify",
                                il, cgc_cold_after, uni.size(), cgc_cold_before);
                    }
                }
            }
        }
        static int cgc_warm_dbg = 0;
        if (cgc_warm_dbg < 24 && il <= 1) {
            cgc_warm_dbg++;
            fprintf(stderr, "CGC-WARM %s n_past=%lld warm=%lld fast=%d\n",
                    cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "draft" : "verify",
                    cgc_n_past, cgc_warm_npast, (int) cgc_fast_eligible);
        }
        if ((verify_fast || draft_fast) && cgc_fast_eligible) {
            // zero the reserved ZERO slot once per layer: cold experts (slot table == -1) map to
            // it, so the FFN reads a finite zero contribution instead of an OOB pool row (NaN
            // cascade -> whole-graph corruption).
            llama_expert_cache_zero_reserved_slot(cache, (uint32_t) il);
            // LRU-touch resident experts only (no fill, no wait): keeps hot slots' freshness so
            // the next step's pick_slot does not hand them out to a cold fill. draft_path tag:
            // fast-path telemetry splits verify (ctx_tgt) vs draft (ctx MTP) cold rates.
            llama_expert_cache_touch(cache, (uint32_t) il, uni.data(), uni.size(),
                                     cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP);
            // [CGC Fast-Path Wait 2026-09-06] verify ctx only (ground-truth route; draft is a
            // prediction — waiting there would slow MTP draft without proportional quality gain).
            // For cold experts whose DBUF step-ahead fill is already in flight, wait up to
            // CGC_FAST_WAIT_US (shared deadline, bounded) so the remap below uses the real slot
            // instead of the ZERO-slot. Default OFF (cgc_fast_wait_on()=false) = no-op.
            if (verify_fast) {
                llama_expert_cache_wait_loading(cache, (uint32_t) il, uni.data(), uni.size());
            }
            // [CGC SyncFill 2026-09-09] Residual cold experts: blocking fill instead of
            // ZERO-slot. The ZERO-slot fast path maps cold experts (slot_table == -1) to a
            // zeroed region -> the FFN reads zeros as if that expert contributed nothing,
            // corrupting layer logits (measured: 15+27 十連發 0/10 with L4 pool + fast path
            // vs 10/10 without; v1/v2 oracle FAIL on the L4 path). Fix: for this step's
            // actually-selected experts, synchronously fill the cold ones (usually 0-2 per
            // layer) so the remap below writes their REAL slot. SpAc EMA membership keeps
            // the cold ratio low (<5%); this closes the residual gap exactly. The ZERO-slot
            // reservation is kept as a safe fallback for ensure_slot failure.
            // [CGC default flip 2026-09-11] DEFAULT ON. Off was not a neutral performance choice:
            // it made the output depend on the POOL LAYOUT, because how many experts are cold at
            // a given step depends on how many slots the pool has. Measured with the logits
            // oracle, 4/6/8 GiB against each other at a fixed CGC_POOL_MAX_TOKENS=6:
            //   * every single-token step (no fast path) was BIT-IDENTICAL across pools;
            //   * the first multi-token step -- the MTP verify batch, which IS the fast path --
            //     diverged by 12-24% RELATIVE in the logits, with genuinely different top-4
            //     candidates (e.g. top3 20/22/21 vs 20/22/15). The identical count was exactly
            //     7/39 for iq3 and iq4 alike, i.e. a deterministic divergence at that step,
            //     and the corruption then persisted into every later single-token step.
            // A 12-24% swing cannot be floating-point ordering: those experts were simply read
            // as zero. Turning this on makes every selected expert resident before the remap is
            // written, so the result no longer depends on which slots happen to be warm.
            // CGC_SYNCFILL_COLD=0 restores the legacy ZERO-slot behavior. Note the parsing: the
            // old `e[0] == '1'` / `!= nullptr` style silently treats =0 as ON, so a numeric
            // value is compared explicitly here and anything not starting with '0' means on.
            // [CGC verify-strict 2026-09-13] The selected cold experts were already filled by the
            // verify-strict prologue ABOVE (it runs before the fast-path/eligibility split so that a
            // still-cold expert refuses the fast path instead of being ZERO-mapped). If we reach
            // here, every selected expert is resident — that is now an invariant, not the
            // consequence of a default that an A/B can silently flip off.
            // [CGC STEP_DBG] per-step miss timeline (il==1 fires once per step): cumulative
            // fast-path cold (ZERO-mapped) + ensure_batch (prefill chunk 1 / catch-up) requests
            // and hits. Measured verdict (2026-08-28, steady MTP denseIQ4X seed1): cold stays
            // ~65% across ALL phases (no early concentration) -> structural churn (per-layer
            // route working set >> pool slots), NOT a cold start a prewarm could fix.
            if (il == 1 && getenv("LLAMA_EXPERT_CACHE_STEP_DBG") != nullptr) {
                static uint32_t step_dbg_n = 0;
                if ((step_dbg_n % 20) == 0) {
                    fprintf(stderr, "STEPDBG step=%u phase=%s ntok=%lld fastuni=%zu fastcold=%zu ensure_req=%zu ensure_hit=%zu\n",
                            step_dbg_n,
                            cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "draft" : "verify",
                            (long long) n_tokens,
                            cache->n_fast_union, cache->n_fast_cold,
                            cache->n_requests, cache->n_hits);
                }
                step_dbg_n++;
            }
            // [CGC 2026-09-15 S1 slot-table] Publish the GPU-readable expert->slot table under the
            // SAME mapping this path writes into the leaf below. The two writes are one mapping and
            // are kept adjacent on purpose: if one of the four remap write sites ever published the
            // table under a different mapping, the GPU-computed ids would silently diverge from the
            // host-written ones and only the logits oracle would catch it.
            ggml_tensor * cgc_stable = cache_slot_table_tensors[il];
            if (cgc_stable != nullptr && cgc_stable->data != nullptr) {
                // [CGC 2026-09-15 §8.3] Routed through the one counted publish entry point, exactly
                // like the pool path. This site used to call llama_expert_cache_publish_slot_table
                // directly and print its own clamp count with its own local counter -- two sites
                // counting the same quantity in two different places is how one of them ends up
                // discarding it, which is precisely what the pool site did. See the long note on
                // cgc_publish_slot_table_counted above for why both are funnelled here.
                cgc_publish_slot_table_counted(cache, il, n_tokens, n_expert, cgc_stable,
                        cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP, "fast", ids, n_expert_used);
                if (cgc_s1_dbg) {
                    // [CGC 2026-09-15 S1 diagnostic] The decisive comparison: what the HOOK just
                    // published for THIS step's ids, next to what the CGC-MMID-ASSERT reports the
                    // GPU actually consumed. Equal -> the table is correct and the divergence is
                    // downstream of it (the gather's index). Different -> the GPU is reading a
                    // different buffer than the hook wrote. Without both numbers side by side the
                    // two are indistinguishable, because both surface as an out-of-range id.
                    //
                    // Was `il == 1`. Layer 1 was the interesting layer while the question was the
                    // CPU/Metal boundary (layer 0's FFN runs on CPU); it is NOT the interesting layer
                    // now that the ladder localized the divergence to 10..19. Gating a probe on a
                    // single layer is how a probe ends up proving something about a layer nobody
                    // asked about.
                    cgc_s1_expect_dbg("fast", il, n_tokens, n_expert_used, ids, cgc_stable);
                }
                cgc_s1_equiv_dbg("fast", cache, il, n_tokens, n_expert_used, ids, cgc_stable, false);
            }
            // write the remap leaf: resident -> slot index, cold -> ZERO-slot.
            ggml_tensor * remap = cache_remap_tensors[il];
            if (remap != nullptr && remap->data != nullptr) {
                int32_t * rd = (int32_t *) remap->data;
                // [CGC verify-strict 2026-09-13] Detect any selected expert that still comes back as
                // the ZERO slot: that means its contribution is silently lost. The prologue above
                // makes this unreachable on the default path, so a nonzero count here is a bug
                // signal — and it must never be silent (it is exactly what made the logits depend
                // on the pool size before).
                int32_t n_zero_mapped = 0;
                const int32_t zs = llama_expert_cache_zero_slot(cache, (uint32_t) il);
                for (int64_t j = 0; j < n_tokens; ++j) {
                    for (int64_t i = 0; i < n_expert_used; ++i) {
                        const uint32_t e = (uint32_t) ids[i + j * n_expert_used];
                        const int32_t slot = llama_expert_cache_slot_table_safe(cache, (uint32_t) il, e);
                        rd[i + j * n_expert_used] = slot;
                        if (zs >= 0 && slot == zs) {
                            n_zero_mapped++;
                        }
                    }
                }
                if (n_zero_mapped > 0) {
                    cache->n_zero_mapped_selected += (size_t) n_zero_mapped;
                    fprintf(stderr, "CGC-ZERO-MAPPED: %s il=%d n=%d/%lld -> selected expert(s) read the ZERO slot (contribution lost)\n",
                            cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "draft" : "verify",
                            il, n_zero_mapped, (long long) (n_tokens * n_expert_used));
                }
            }
            // [CGC 2026-09-15 S1 cleanup] The `CGC-FAST:` per-layer print that used to sit here (capped
            // at 20 lines) was a cheap churn proxy whose question is now answered by the counted
            // publish: see cgc_publish_slot_table_counted and n_slot_table_changed.
            // [CGC DBUF spike 2026-09-06] step-ahead refill of this step's ZERO-mapped cold
            // experts (CGC_DBUF=1, default OFF). Verify ctx only (the multi-token ground-truth
            // route; draft ctx duplicates it one token ahead). Safety: this hook runs between
            // GPU segments — seg il+1 (this layer's FFN) is submitted only after we return and
            // its remap references union slots only, while prefetch_slot's LRU victim is by
            // construction a non-union slot (touch() just refreshed union members), so the bg
            // fill cannot tear any read of this decode. prefetch_slot publishes slot_table only
            // after bytes land; a still-loading fill at the NEXT step's hook degrades to the
            // existing cold (ZERO/guard) handling for that one token. This attacks the quality
            // gap that blocks 25 t/s: cold experts seen this step become resident before the
            // next step needs them (step-to-step routing stability ~87%).
            if (verify_fast && cgc_dbuf_on()) {
                llama_expert_cache_dbuf_refill(cache, (uint32_t) il, uni.data(), uni.size());
            }
            return;
        }
        static int cgc_pre_post_n = 0;
        const bool cgc_pp = cgc_pre_post_n < 24 && il <= 2;
        // [CGC Exact Path Debug 2026-09-08] detailed prefill/exact-path tracing: dump every
        // selected expert's slot_table value before/after ensure_batch, and the remap values
        // written. Enabled by CGC_EXACT_PATH_DBG=1; limited to prefill (n_tokens>1) + first
        // 3 layers to keep log volume manageable. This is the diagnostic for the oracle
        // divergence at token 1 (chunk 0 exact path).
        static const bool cgc_exact_dbg = getenv("CGC_EXACT_PATH_DBG") != nullptr;
        const bool cgc_ep = cgc_exact_dbg && n_tokens > 1 && il <= 2;
        if (cgc_pp) {
            cgc_pre_post_n++;
            const int32_t * st0 = llama_expert_cache_slot_table(cache, (uint32_t) il);
            fprintf(stderr, "CGC-PRE: il=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d\n",
                    il, ids[0], st0 ? st0[ids[0]] : -2, ids[1], st0 ? st0[ids[1]] : -2,
                    ids[2], st0 ? st0[ids[2]] : -2, ids[3], st0 ? st0[ids[3]] : -2,
                    ids[4], st0 ? st0[ids[4]] : -2, ids[5], st0 ? st0[ids[5]] : -2,
                    ids[6], st0 ? st0[ids[6]] : -2, ids[7], st0 ? st0[ids[7]] : -2);
        }
        if (cgc_ep) {
            const int32_t * st0 = llama_expert_cache_slot_table(cache, (uint32_t) il);
            fprintf(stderr, "CGC-EXACT-PRE: il=%d n_tokens=%lld n_expert_used=%lld uni_size=%zu\n",
                    il, (long long) n_tokens, (long long) n_expert_used, uni.size());
            for (size_t k = 0; k < uni.size(); ++k) {
                const uint32_t e = uni[k];
                fprintf(stderr, "  uni[%zu]=%u slot=%d owner=%d loading=%d queued=%d\n",
                        k, e, st0 ? st0[e] : -2,
                        (e < cache->n_expert && (uint32_t) il < cache->slot_owner.size()) ? cache->slot_owner[il][st0[e] >= 0 ? st0[e] : 0] : -3,
                        (e < cache->n_expert && (uint32_t) il < cache->slot_loading.size() && st0[e] >= 0) ? cache->slot_loading[il][st0[e]] : -3,
                        (e < cache->n_expert && (uint32_t) il < cache->slot_queued.size() && st0[e] >= 0) ? cache->slot_queued[il][st0[e]] : -3);
            }
        }
        // [CGC P1 prefill-protect 2026-09-12] Tell the cache whether this fill belongs to a
        // PREFILL step. When CGC_PREFILL_PROTECT=1 the cache then defers evicting slots whose
        // owner was filled during decode, instead of churning the working set the next
        // generation needs (measured: 22.2 t/s steady-state decode vs 8.1-8.9 t/s after a
        // prefill). False for every other phase, so the default build is unchanged.
        llama_expert_cache_ensure_batch(cache, (uint32_t) il, uni.data(), uni.size(),
                                        cgc_current_phase == CGC_PHASE_PREFILL);
        llama_expert_cache_drain_layer(cache, (uint32_t) il);
        if (cgc_ep) {
            const int32_t * st1 = llama_expert_cache_slot_table(cache, (uint32_t) il);
            fprintf(stderr, "CGC-EXACT-POST: il=%d\n", il);
            for (size_t k = 0; k < uni.size(); ++k) {
                const uint32_t e = uni[k];
                fprintf(stderr, "  uni[%zu]=%u slot=%d (was %d)\n",
                        k, e, st1 ? st1[e] : -2,
                        (e < cache->n_expert) ? -99 : -2);
            }
        }

        // NOTE: the FFN expert weight tensors were already repointed at the pool regions in
        // graph_get_cb (ffn_moe_topk_remap); the segmented dispatch submits with those pointers.

        // [CGC 2026-09-15 S1 slot-table] Same mapping as the leaf write below (raw slot table, not
        // the *_safe variant). For every expert this path selects, verify-strict has already made it
        // resident, so both variants agree on the ids that are actually consumed.
        ggml_tensor * cgc_stable_e = cache_slot_table_tensors[il];
        if (cgc_stable_e != nullptr && cgc_stable_e->data != nullptr) {
            // [CGC 2026-09-15 §8.3] This site used to discard the return value, and it is the site
            // every MTP-off S1 arm takes -- so the clamp count was computed and never observed.
            cgc_publish_slot_table_counted(cache, il, n_tokens, n_expert, cgc_stable_e,
                    cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP, "pool", ids, n_expert_used);
        }
        cgc_s1_expect_dbg("pool", il, n_tokens, n_expert_used, ids, cgc_stable_e);
        cgc_s1_equiv_dbg("pool", cache, il, n_tokens, n_expert_used, ids, cgc_stable_e, false);
        // write the remap leaf: selected expert id -> slot index
        ggml_tensor * remap = cache_remap_tensors[il];
        if (remap != nullptr && remap->data != nullptr) {
            const int32_t * st = llama_expert_cache_slot_table(cache, (uint32_t) il);
            if (cgc_pp) {
                fprintf(stderr, "CGC-POST: il=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d\n",
                        il, ids[0], st ? st[ids[0]] : -2, ids[1], st ? st[ids[1]] : -2,
                        ids[2], st ? st[ids[2]] : -2, ids[3], st ? st[ids[3]] : -2,
                        ids[4], st ? st[ids[4]] : -2, ids[5], st ? st[ids[5]] : -2,
                        ids[6], st ? st[ids[6]] : -2, ids[7], st ? st[ids[7]] : -2);
            }
            int32_t * rd = (int32_t *) remap->data;
            for (int64_t j = 0; j < n_tokens; ++j) {
                for (int64_t i = 0; i < n_expert_used; ++i) {
                    const uint32_t e = (uint32_t) ids[i + j * n_expert_used];
                    rd[i + j * n_expert_used] = (st != nullptr && e < n_expert) ? st[e] : (int32_t) e;
                }
            }
            // [CGC 2026-09-15] This used to fire unconditionally for the first 40 layers of every
            // run -- synchronous fprintf(stderr) from the decode hook, i.e. inside the hot path,
            // on every single run including production ones. It is now opt-in via CGC_SLOT_DBG
            // (same treatment CGC-M2-DBG got earlier today): the slot table it prints is the
            // input to the remap, and the remap itself is now covered by the always-on
            // CGC-MMID-ASSERT in ggml-metal-ops.cpp, so leaving this on bought nothing that the
            // assertion does not already give us.
            static const bool cgc_slot_dbg = getenv("CGC_SLOT_DBG") != nullptr;
            static int cgc_slot_dbg_n = 0;
            if (cgc_slot_dbg && cgc_slot_dbg_n < 40) {
                cgc_slot_dbg_n++;
                fprintf(stderr, "CGC-SLOT: il=%d st[%d]=%d st[%d]=%d st[%d]=%d st[%d]=%d remap=[%d %d %d %d %d %d %d %d]\n",
                        il, ids[0], st ? st[ids[0]] : -1, ids[1], st ? st[ids[1]] : -1,
                        ids[2], st ? st[ids[2]] : -1, ids[3], st ? st[ids[3]] : -1,
                        rd[0], rd[1], rd[2], rd[3], rd[4], rd[5], rd[6], rd[7]);
            }
            // [CGC Exact Path Debug] dump every remap value written for prefill chunks
            if (cgc_exact_dbg && n_tokens > 1 && il <= 2) {
                fprintf(stderr, "CGC-EXACT-REMAP: il=%d n_tokens=%lld n_expert_used=%lld\n",
                        il, (long long) n_tokens, (long long) n_expert_used);
                for (int64_t j = 0; j < n_tokens; ++j) {
                    fprintf(stderr, "  token[%lld]: ", j);
                    for (int64_t i = 0; i < n_expert_used; ++i) {
                        const uint32_t e = (uint32_t) ids[i + j * n_expert_used];
                        const int32_t slot = rd[i + j * n_expert_used];
                        fprintf(stderr, "e%u->s%d ", e, slot);
                        if (slot < 0) {
                            fprintf(stderr, "[NEGATIVE!] ");
                        }
                    }
                    fprintf(stderr, "\n");
                }
            }
        }

        // B: record this step's per-layer union so process_ubatch can async-prefetch it for the
        // next decode step (temporal locality). The bg fill runs behind the sampler + next step's
        // GPU window, so the next step's ensure_batch hits instead of blocking on disk reads.
        if (cache_step_union.size() < (size_t) model.hparams.n_layer_all) {
            cache_step_union.resize(model.hparams.n_layer_all);
        }
        cache_step_union[(size_t) il] = uni;
        return;
    }

    // L3-B gather path: ensure resident, gather the union into contiguous buffers, repoint the
    // FFN tensors and remap the ids to 0..k-1.
    {
        std::vector<uint32_t> layers(uni.size(), (uint32_t) il);
        llama_expert_cache_ensure(cache, layers.data(), uni.data(), uni.size());

        for (int kind = 0; kind < 4; ++kind) {
            ggml_tensor * wt = cache_ffn_tensors[il][kind];
            if (wt == nullptr || uni.empty()) {
                continue;
            }
            const size_t exp_bytes = ggml_row_size(wt->type, wt->ne[0]) * wt->ne[1];
            // [CGC M1 Metal slab 2026-09-14] A union wider than the slab's fixed capacity would need
            // the multi-pass loop of 3.2, which is not implemented (and is provably unnecessary at
            // cap=8, where the ceiling is exactly 64 -- see cgc_gather_slab_cap). Refuse rather than
            // repoint at a slab that cannot hold the union: that would put ggml_nbytes(wt) past the
            // end of the buffer, which is the "buffer is nil" we are removing. Loud, once per kind.
            if (uni.size() > (size_t) cgc_gather_slab_cap()) {
                static bool cgc_slab_overflow_warned = false;
                if (!cgc_slab_overflow_warned) {
                    cgc_slab_overflow_warned = true;
                    fprintf(stderr, "CGC-GATHER-SLAB: OVERFLOW il=%d kind=%d union=%zu > cap=%u -- "
                            "multi-pass (M1 3.2) not implemented; leaving the original weights in "
                            "place, values for this layer are NOT trustworthy\n",
                            il, kind, uni.size(), cgc_gather_slab_cap());
                }
                continue;
            }
            // The slab is allocated with the EXPERT TENSOR's own buffer type. Not the device
            // default: Metal only gives host-visible storage when `use_shared_buffers && shared`
            // (ggml-metal-device.m:1666), otherwise `all_data` is a virtual address and the CPU
            // fill below writes into nothing. The tensor's buft is the shared one by construction
            // -- the pool path CPU-memcpy's into that same region (pool_region -> pool_ext).
            ggml_backend_buffer_type_t wt_buft = wt->buffer != nullptr
                    ? ggml_backend_buffer_get_type(wt->buffer) : nullptr;
            // [CGC M1 work item 1] With CGC_POOL_SPLIT the tensor's own buffer is a CPU mapping, so
            // its type is NOT host-visible Metal storage and a slab allocated from it would be
            // unwritable by the CPU fill and unreadable by Metal. The pool's buffer is the right
            // type by construction (the pool fill memcpy's into it).
            if (llama_expert_cache_pool_buffer(cache, (uint32_t) il, kind) != nullptr) {
                wt_buft = ggml_backend_buffer_get_type(
                        llama_expert_cache_pool_buffer(cache, (uint32_t) il, kind));
            }
            // [M2 fix 2026-09-14] If expert tensor was skip-loaded (--load-mode none), its buffer
            // is CPU-only and the slab would be allocated as CPU → Metal mul_mat_id SIGSEGVs.
            // Fall back to the first non-CPU backend's default buffer type.
            if (wt_buft != nullptr && strcmp(ggml_backend_buft_name(wt_buft), "CPU") == 0) {
                for (auto & backend : backends) {
                    auto dev = ggml_backend_get_device(backend.get());
                    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                        auto * dev_buft = ggml_backend_get_default_buffer_type(backend.get());
                        if (dev_buft) {
                            wt_buft = dev_buft;
                            break;
                        }
                    }
                }
            }
            uint8_t * slab = nullptr;
            ggml_backend_buffer_t slab_buf = nullptr;
            {
                cgc_gather_slab * sl = cgc_gather_slab_get(kind, exp_bytes, wt_buft);
                if (sl == nullptr) {
                    continue;
                }
                // copied out immediately: an allocation for a later kind moves the vector
                slab     = sl->base;
                slab_buf = sl->buf;
            }
            const int64_t copied = llama_expert_cache_fill(cache, (uint32_t) il, uni.data(),
                    uni.size(), kind, slab, exp_bytes);
            if (copied < 0) {
                continue;
            }
            // [CGC M1 2026-09-14] Record the pre-repoint state ONCE per step, not once per call.
            // The graph-build restore is SKIPPED when the graph is REUSED (same ubatch shape -- see
            // the `else` branch of graph_compute), so two gather steps for one tensor can run back
            // to back with no restore in between. Recording on the second call would then save the
            // SLAB ITSELF as the "original" buffer, and the next restore would set data (a pool
            // pointer) and buffer (the slab) to values from two different states -- the mixed state
            // Metal reports as `buffer is nil`.
            //
            // Measured 2026-09-14, LLAMA_EXPERT_CACHE_LAYER_CAPS=1-39:32 (the wide route is taken on
            // ~7 of 8 steps per layer): 117 nils, exactly one per (layer, kind), every one of them
            // that mixed state (CGC-METAL-NIL: b0data = the slab base, tdata elsewhere, ne[2] = 32 =
            // the raw slot count). Keeping the FIRST values makes the restore land on the
            // consistent pool state instead.
            const bool cgc_first_repoint =
                    cache_gather_ne2.find({il, kind}) == cache_gather_ne2.end();
            if (cgc_first_repoint) {
                cache_orig[{il, kind}] = wt->data;
            }
            // Metal's range check is `ioffs + ggml_nbytes(t) <= buffers[i].size`, and ggml_nbytes
            // scales with ne[2] -- which the loader set to the POOL capacity, not to this union.
            // Leaving it alone is wrong in BOTH directions, which is why the old host buffer could
            // never have worked: with ne[2] > union, ggml_nbytes exceeds the gathered slab and Metal
            // still reports nil; with ne[2] < union, the ids the remap writes below (positions in
            // the sorted union, so in [0, union-1]) exceed the expert count mul_mat_id believes in.
            // Setting ne[2] = union makes both the range check and mul_mat_id agree with what was
            // actually gathered. Restored next graph build via cache_gather_ne2.
            if (cgc_first_repoint) {
                cache_gather_ne2[{il, kind}] = wt->ne[2];
                cache_gather_orig_buf[{il, kind}] = wt->buffer;
            }
            wt->ne[2] = (int64_t) uni.size();
            // Metal resolves a tensor through ITS OWN buffer, not by scanning every buffer:
            //     ggml_backend_buffer_t b = t->view_src ? t->view_src->buffer : t->buffer;
            //     return ggml_metal_buffer_get_id(b->context, t);
            // (ggml-metal-ops.cpp:22-26). So moving `data` without moving `buffer` leaves the
            // lookup computing ioffs against the ORIGINAL allocation -> negative -> Metal still
            // reports "buffer is nil". That is exactly why the first version of this change left
            // the nil count at 468. data, buffer and ne[2] must all move together.
            wt->data   = slab;
            wt->buffer = slab_buf;
            // [CGC M1 Metal slab] Replicate Metal's own range check right here so that a failure
            // is attributable instead of guessed. ggml_metal_buffer_get_id() (ggml-metal-device.m)
            // walks the buffer's sub-ranges and returns the first one where
            //     ioffs + ggml_nbytes(t) <= buffers[i].size,  ioffs = t->data - buffers[i].data
            // and logs "buffer is nil" when none matches. The values below are exactly those
            // operands. Inert unless CGC_SLAB_DBG=1.
            static const bool cgc_slab_dbg = getenv("CGC_SLAB_DBG") != nullptr;
            if (cgc_slab_dbg) {
                const size_t bsize = ggml_backend_buffer_get_size(slab_buf);
                const void * bbase = ggml_backend_buffer_get_base(slab_buf);
                const size_t tsz   = ggml_nbytes(wt);
                const long   offs  = (long) ((const char *) wt->data - (const char *) bbase);
                fprintf(stderr,
                        "CGC-SLAB-CHECK: %s il=%d kind=%d data=%p base=%p size=%zu nbytes=%zu "
                        "offs=%ld fits=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                        wt->name, il, kind, wt->data, bbase, bsize, tsz, offs,
                        (offs >= 0 && (size_t) offs + tsz <= bsize) ? 1 : 0,
                        (long long) wt->ne[0], (long long) wt->ne[1],
                        (long long) wt->ne[2], (long long) wt->ne[3],
                        wt->nb[0], wt->nb[1], wt->nb[2], wt->nb[3]);
            }
        }

        // [CGC 2026-09-15 S1 slot-table] This path uses a DIFFERENT mapping from the others: the
        // weights were gathered into a contiguous per-step buffer, so the ids are union indices
        // (`uidx`), not pool slots. The published table must use the same union-index mapping --
        // publishing the slot table here would point mul_mat_id at pool slots while the FFN reads
        // the gather buffer. Experts outside the union map to 0, exactly as the leaf write does.
        ggml_tensor * cgc_stable_g = cache_slot_table_tensors[il];
        if (cgc_stable_g != nullptr && cgc_stable_g->data != nullptr) {
            int32_t * tdg = (int32_t *) cgc_stable_g->data;
            for (int64_t e = 0; e < n_expert; ++e) {
                tdg[e] = (int32_t) (uidx.count((uint32_t) e) ? uidx[(uint32_t) e] : 0);
            }
        }
        cgc_s1_expect_dbg("gather", il, n_tokens, n_expert_used, ids, cgc_stable_g);
        // Same differential check, but against the mapping this site is actually defined on: the
        // gather path maps a raw expert id to its UNION index, not to a pool slot, so
        // slot_table_safe is the wrong oracle here by construction. The leaf write just below is the
        // authority, and it is literally the same `uidx.count(e) ? uidx[e] : 0` expression.
        if (cgc_stable_g != nullptr && cgc_stable_g->data != nullptr &&
                getenv("CGC_S1_DBG") != nullptr) {
            const int32_t * tbg = (const int32_t *) cgc_stable_g->data;
            int64_t n_mis = 0;
            int32_t fe = -1, fg = 0, fw = 0;
            for (int64_t j = 0; j < n_tokens * n_expert_used; ++j) {
                const uint32_t e = (uint32_t) ids[j];
                const int32_t want = (int32_t) (uidx.count(e) ? uidx[e] : 0);
                if (tbg[e] != want) {
                    if (n_mis == 0) { fe = (int32_t) e; fg = tbg[e]; fw = want; }
                    n_mis++;
                }
            }
            fprintf(stderr, "CGC-S1: EQUIV-gather il=%d ntok=%lld table_vs_uidx=%lld first_e=%d table=%d uidx=%d union=%zu\n",
                    il, (long long) n_tokens, (long long) n_mis, fe, fg, fw, uidx.size());
        }
        ggml_tensor * remap = cache_remap_tensors[il];
        if (remap != nullptr && remap->data != nullptr) {
            int32_t * rd = (int32_t *) remap->data;
            for (int64_t j = 0; j < n_tokens; ++j) {
                for (int64_t i = 0; i < n_expert_used; ++i) {
                    const uint32_t e = (uint32_t) ids[i + j * n_expert_used];
                    const uint32_t slot = uidx.count(e) ? uidx[e] : 0;
                    rd[i + j * n_expert_used] = (int32_t) slot;
                }
            }
        }
    }
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // [CGC M1 work item 1 CGC_POOL_SPLIT] Install this build's expert geometry deterministically.
        //
        // Why here and why derived: with an OWNED pool the expert tensor alternates between two
        // geometries -- the pool's (slots, pool buffer, pool base) for a pool build and the
        // loader's full-width weights for a wide build -- and all three fields must come from the
        // SAME state, because Metal resolves a tensor through its own buffer and bounds
        // ggml_nbytes, which scales with ne[2]. Recording "the previous state" per build cannot
        // work: once a repoint has happened the recorded state IS the pool state, so the next
        // restore writes pool `data` next to a full-width `ne[2]` and the model `buffer` -- the
        // exact operand triple Metal reports as `buffer is nil` (measured). Deriving both sides
        // from stored values makes every build idempotent instead.
        //
        // This callback runs for every MoE weight of every layer in every build, which is what
        // makes it the right place: the wide path builds NO remap leaf, so the remap-leaf callback
        // never fires for it.
        auto cgc_apply_expert_geometry = [&](ggml_tensor * wt, int il_, int kind) {
            if (wt == nullptr || model.expert_cache == nullptr) {
                return;
            }
            // [CGC M2 prefill streaming 2026-09-14] When CGC_PREFILL_STREAM=1 and this is a large
            // prefill chunk (n_tokens > pool_max), the eval hook will fill a whole-layer slab and
            // repoint this tensor at it JUST BEFORE the FFN computes. We must NOT set data/buffer
            // here (graph build), because the slab is allocated per-step and the tensor's own
            // (shrunk) storage is what ggml-alloc sizes against. Leaving it untouched here means
            // the eval hook's repoint is the only geometry change, and the graph-build restore
            // (cache_gather_ne2 / cache_orig) returns the tensor to its shrunk state afterward.
            const bool prefill_stream = cgc_stream_on;  // validated once in the constructor
            if (prefill_stream && (int64_t) ubatch.n_tokens > (int64_t) cgc_pool_max_tokens()) {
                return; // eval hook handles it (whole-layer slab)
            }
            if (!llama_expert_cache_pool_owned(model.expert_cache)) {
                return; // legacy path: the tensor's buffer already points at the pool by construction
            }
            if ((int64_t) ubatch.n_tokens <= (int64_t) cgc_pool_max_tokens()) {
                const uint8_t * base = llama_expert_cache_pool_data(model.expert_cache, (uint32_t) il_, kind);
                ggml_backend_buffer_t pbuf = llama_expert_cache_pool_buffer(model.expert_cache, (uint32_t) il_, kind);
                if (base == nullptr || pbuf == nullptr) {
                    return;
                }
                wt->data   = (void *) base;
                wt->buffer = pbuf;
                wt->ne[2]  = (int64_t) llama_expert_cache_slots_per_layer_l(model.expert_cache, (uint32_t) il_);
            } else {
                void * wdata = nullptr;
                ggml_backend_buffer_t wbuf = nullptr;
                int64_t wne2 = 0;
                if (!llama_expert_cache_pool_get_wide(model.expert_cache, (uint32_t) il_, kind,
                                                      &wdata, &wbuf, &wne2)) {
                    return;
                }
                wt->data   = wdata;
                wt->buffer = wbuf;
                wt->ne[2]  = wne2;
            }
            if (getenv("CGC_POOL_SPLIT_DBG") != nullptr && il_ <= 2) {
                fprintf(stderr, "CGC-POOL-SPLIT-GEOM: il=%d kind=%d ntok=%lld mode=%s ne2=%lld data=%p buf=%p\n",
                        il_, kind, (long long) ubatch.n_tokens,
                        (int64_t) ubatch.n_tokens <= (int64_t) cgc_pool_max_tokens() ? "pool" : "wide",
                        (long long) wt->ne[2], wt->data, (void *) wt->buffer);
            }
        };

        // CGC expert-cache: capture the remap leaf (ffn_moe_topk) and the FFN expert weight
        // tensors (src0 of the mul_mat_id results) so the eval hook can repoint them at the
        // cache pool / gather buffers and write remapped ids. Mirrors build-prod graph_get_cb.
        if (il >= 0 && model.expert_cache_active) {
            if (strcmp(name, "ffn_moe_rn_mask") == 0) {
                // [CGC Step-3] renormalized-routing mask leaf (CGC_RN_ROUTING=1 only): captured
                // so the eval hook can rewrite 0.0/-inf per expert each step before the router
                // softmax reads it. Mirrors the remap-leaf capture just below.
                cache_rn_mask_tensors[il] = cur;
                return;
            }
            // [CGC 2026-09-15 S1 slot-table] `ffn_moe_slots` is the GPU gather's output -- the tensor
            // mul_mat_id actually consumes as ids. Captured so the post-synchronize readback can
            // read it back on the host AFTER the command buffer completed; see the note on
            // cache_slots_out_tensors in the header for why the encode-time probe cannot.
            if (strcmp(name, "ffn_moe_slots") == 0) {
                cache_slots_out_tensors[il] = cur;
            }
            // [CGC 2026-09-15 S1 slot-table] `ffn_moe_slot_table` is the S1 replacement for
            // `ffn_moe_topk_remap` and is built under the same conditions, so it must get the same
            // capture AND the same pool repoint below. The repoint hangs off this branch rather
            // than off the FFN weight nodes, so a table build that skipped it would leave the FFN
            // src0 pointing at the full-width weights while mul_mat_id indexes pool slots.
            const bool cgc_cap_is_remap = strcmp(name, "ffn_moe_topk_remap") == 0;
            const bool cgc_cap_is_table = strcmp(name, "ffn_moe_slot_table") == 0;
            if (cgc_cap_is_remap || cgc_cap_is_table) {
                if (cgc_cap_is_remap) {
                    cache_remap_tensors[il] = cur;
                } else {
                    cache_slot_table_tensors[il] = cur;
                }
                // CGC: point this layer's expert weight tensors at the L4 pool regions up front
                // (the remap leaf is built only for decode, n_tokens == 1). The segmented Metal
                // dispatch (CGC_OA_ASYNC) submits segments whose weights already point at the
                // pool, so the eval hook only needs to ensure the slot data and write the remap
                // ids (it must not repoint after submit — that would be too late).
                if (model.expert_cache != nullptr && llama_expert_cache_pool_active(model.expert_cache)) {
                    auto & ffn = cache_ffn_tensors[il];
                    if (ffn.size() < 4) ffn.resize(4);
                    for (int kind = 0; kind < 4; ++kind) {
                        ggml_tensor * wt = ffn[kind];
                        if (wt == nullptr) {
                            continue;
                        }
                        const uint8_t * base = llama_expert_cache_pool_data(model.expert_cache, (uint32_t) il, kind);
                        if (base == nullptr) {
                            continue;
                        }
                        // [CGC M1 work item 1 CGC_POOL_SPLIT] With an OWNED pool this data-only move is
                        // redundant: the ffn_moe_* capture below installs the full geometry (data +
                        // buffer + ne[2]) for this build, and it runs later in the same build, so it
                        // overwrites this. Kept because it is harmless and it is the whole repoint the
                        // legacy zero-copy path needs (there the tensor's own storage IS the pool).
                        cache_orig[{il, kind}] = wt->data;
                        wt->data = (void *) base;
                    }
                }
            } else if (strcmp(name, "ffn_moe_gate_up") == 0) {
                if (cur->src[0] != nullptr) {
                    auto & ffn = cache_ffn_tensors[il];
                    if (ffn.size() < 4) ffn.resize(4);
                    ffn[3] = cur->src[0];
                    cgc_apply_expert_geometry(ffn[3], il, 3);
                }
            } else if (strcmp(name, "ffn_moe_up") == 0) {
                if (cur->src[0] != nullptr) {
                    auto & ffn = cache_ffn_tensors[il];
                    if (ffn.size() < 4) ffn.resize(4);
                    ffn[1] = cur->src[0];
                    cgc_apply_expert_geometry(ffn[1], il, 1);
                }
            } else if (strcmp(name, "ffn_moe_gate") == 0) {
                if (cur->src[0] != nullptr) {
                    auto & ffn = cache_ffn_tensors[il];
                    if (ffn.size() < 4) ffn.resize(4);
                    ffn[0] = cur->src[0];
                    cgc_apply_expert_geometry(ffn[0], il, 0);
                }
            } else if (strcmp(name, "ffn_moe_down") == 0) {
                // [CGC 2026-09-15 S1 output capture] `cur` here IS the down projection's mul_mat_id
                // result -- the whole layer's expert contribution, before the residual add -- which is
                // exactly the tensor §9.18 ended on. Pinned so the post-synchronize readback in
                // graph_compute reads a LIVE buffer.
                //
                // ggml_set_output is needed HERE and nowhere else in this callback: every other
                // tensor captured above is a leaf that the allocator must keep alive anyway, while
                // ffn_moe_down is an intermediate whose buffer ggml-alloc is free to recycle the
                // moment its consumer has read it. A read of a recycled buffer returns whatever the
                // next occupant happens to be -- a plausible-looking WRONG number, which is the one
                // failure mode this probe cannot afford, because the whole point is to decide whether
                // the two arms' outputs differ.
                //
                // Gated on CGC_S1_OUT_CAP and never on a gate arm: ggml_set_output changes the
                // allocator's layout and is numerically relevant in this codebase (it is implicated
                // in the clobber fixed in §7.4), so it must not ride along on a bit-identical gate
                // arm. It IS applied to every diagnostic arm identically, which is what keeps the
                // two-arm comparison fair -- an asymmetric pin would move the numbers being compared.
                //
                // [CGC 2026-09-15 third fix] The pin is confined to DECODE graphs, and that is a
                // hard requirement, not an optimisation. With the pin applied in EVERY graph both
                // arms died during the load-time warmup decode and served nothing (2/2):
                // `command buffer 8 failed with status 5` /
                // kIOGPUCommandBufferCallbackErrorOutOfMemory, then the CGC_METAL_FAIL_STOP abort.
                // The control arm without the knob started and decoded at 20.2 t/s in the same
                // session, and with the pin moved onto `ubatch.n_tokens <= 8` both arms then ran
                // clean (2/2).
                //
                // The BYTES DO NOT EXPLAIN IT, and that is the part worth writing down -- twice over,
                // because the first version of this comment offered a byte budget (50 MB/layer at a
                // 6144-token chunk) and the second failure refuted even that weaker claim.
                //
                //   pin on EVERY graph, 40 layers  -> died at the warmup decode (2/2), ntok=2
                //   pin on PREFILL only, 11 layers -> died at the warmup decode (2/2), ntok=2
                //   pin on PREFILL only,  4 layers -> ran (2/2)
                //   pin on DECODE  only, 40 layers -> ran (2/2)
                //
                // At the graph that dies the hook reports ntok=2 and ffn_moe_down is [2048, 8, ntok]
                // F32, so eleven pinned layers hold 11 x 131 KB ~= 1.4 MB and four hold 0.5 MB. The
                // regression is therefore driven by the COUNT of pinned tensors -- by
                // ggml_set_output marking them as graph outputs, which changes the scheduler's
                // allocation and split -- and not by their size. So the cost of a pin is not
                // estimable from dimensions, and any reasoning of the form "pinning is cheap because
                // these tensors are small" is refuted by this pair of runs. The working boundary on
                // this machine is between FOUR and ELEVEN pinned layers per graph; the mechanism
                // between those two numbers is NOT established, and the failure is a fail-stop abort
                // rather than a slow path, so it is not something to probe by widening.
                //
                // `ubatch.n_tokens <= 8` rather than `== 1` so that an MTP verify graph -- which
                // decodes several tokens at once -- is still covered, while prefill chunks of
                // thousands are excluded.
                //
                // [CGC 2026-09-15 fourth change] `CGC_S1_OUT_CAP=pre` selects the PREFILL capture
                // instead of the decode one, and it pins only the layers named by CGC_S1_OUT_LAYERS
                // (default "0"). The layer window is not a convenience: the decode capture came back
                // 480/480 differing, layer 0 included, which cannot localize anything because decode
                // step 0 inherits the prefill's KV and the two prefills already disagree. The prefill
                // ladder is the one with an identical input at layer 0, and it costs one window's
                // worth of pinned bytes rather than forty layers' worth -- which, per the boundary
                // above, is the constraint that actually binds.
                const char * cgc_out_cap_env = getenv("CGC_S1_OUT_CAP");
                if (cgc_out_cap_env != nullptr) {
                    const bool cgc_out_pre = strcmp(cgc_out_cap_env, "pre") == 0;
                    const int64_t cgc_out_ntok = (int64_t) ubatch.n_tokens;
                    bool cgc_out_layer_ok = true;
                    if (cgc_out_pre) {
                        // "0" / "0-3" / "0,5,7-9" -- deliberately tiny, no need for a general parser
                        // beyond this. Unparseable input pins layer 0, which is the layer the ladder
                        // starts from, so a typo degrades to a valid run rather than to silence.
                        cgc_out_layer_ok = false;
                        const char * win = getenv("CGC_S1_OUT_LAYERS");
                        if (win == nullptr) {
                            win = "0";
                        }
                        for (const char * p = win; *p != '\0'; ) {
                            while (*p == ',' || *p == ' ') ++p;
                            if (*p == '\0') break;
                            char * end = nullptr;
                            const long lo = strtol(p, &end, 10);
                            long hi = lo;
                            if (end != nullptr && *end == '-') {
                                hi = strtol(end + 1, &end, 10);
                            }
                            if (il >= lo && il <= hi) { cgc_out_layer_ok = true; }
                            if (end == p) break;
                            p = end;
                        }
                    }
                    if ((cgc_out_pre ? (cgc_out_ntok > 1) : (cgc_out_ntok <= 8)) && cgc_out_layer_ok) {
                        cache_down_out_tensors[il] = cur;
                        ggml_set_output(cur);
                    }
                }
                if (cur->src[0] != nullptr) {
                    auto & ffn = cache_ffn_tensors[il];
                    if (ffn.size() < 4) ffn.resize(4);
                    ffn[2] = cur->src[0];
                    cgc_apply_expert_geometry(ffn[2], il, 2);
                }
            }
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
            }
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
