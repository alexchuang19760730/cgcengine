#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-sampler.h"

#include "llama-cgc-canon.h"
#include "llama-cgc-phase.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>

// dedup helpers

// CGC debug: dump tensor stats for MoE intermediate comparison
static void cgc_moe_dump_stats(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    if (ith != 0) {
        // non-zero threads: just copy
        const int64_t n = ggml_nelements(a);
        memcpy(dst->data, a->data, n * ggml_type_size(a->type));
        return;
    }
    const char * name = (const char *)userdata;
    const float * data = (const float *)a->data;
    const int64_t n = ggml_nelements(a);
    float sum = 0, sum_sq = 0, max_val = -1e30f, min_val = 1e30f;
    int64_t argmax = 0;
    for (int64_t i = 0; i < n; i++) {
        float v = data[i];
        sum += v;
        sum_sq += v * v;
        if (v > max_val) { max_val = v; argmax = i; }
        if (v < min_val) { min_val = v; }
    }
    float mean = sum / (float)n;
    float var = sum_sq / (float)n - mean * mean;
    fprintf(stderr, "CGC-MOE-DUMP [%s]: ne=[%ld,%ld,%ld,%ld] n=%ld mean=%.4f std=%.4f min=%.4f max=%.4f argmax=%ld first4=[%.2f,%.2f,%.2f,%.2f]\n",
            name, (long)a->ne[0], (long)a->ne[1], (long)a->ne[2], (long)a->ne[3], (long)n,
            mean, sqrtf(var > 0 ? var : 0), min_val, max_val, (long)argmax,
            data[0], data[1], data[2], data[3]);
    memcpy(dst->data, a->data, n * sizeof(float));
}

static const bool cgc_moe_dump = []{
    const char * e = getenv("CGC_MOE_DUMP");
    return e != nullptr && e[0] == '1';
}();

static ggml_tensor * cgc_moe_dump_node(ggml_context * ctx, ggml_tensor * a, const char * name, int layer) {
    if (!cgc_moe_dump || layer != 0) return a;
    // use a static buffer for the name (layer 0 only, so safe)
    static char buf[128];
    snprintf(buf, sizeof(buf), "%s_L%d", name, layer);
    ggml_tensor * r = ggml_map_custom1(ctx, a, cgc_moe_dump_stats, 1, (void *)buf);
    return r;
}

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// impl

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }

    // TODO: extend llama_ubatch to differentiate between token embeddings and hidden states
    //       for now, we assume that the hidden state is always provided as an embedding
    //       ref: https://github.com/ggml-org/llama.cpp/pull/23643
    if (ubatch->embd) {
        GGML_ASSERT(n_embd == h->ne[0]);

        ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL)) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(self_kq_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        if (self_v_idxs) {
            mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
        }
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        if (self_v_idxs_swa) {
            mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
        }
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);

    if (inp_csa.k_rot && inp_csa.k_rot->buffer) {
        mctx->get_csa()->set_input_k_rot(inp_csa.k_rot);
    }

    if (inp_hca.k_rot && inp_hca.k_rot->buffer) {
        mctx->get_hca()->set_input_k_rot(inp_hca.k_rot);
    }

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    if (inp_attn->self_k_rot) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_kq_mask_swa && inp_attn->self_kq_mask_swa->buffer) {
        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();
    fused_nodes.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    for (auto * tensor : t_sampled) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_probs) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_logits) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_candidates) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    expert_cache_active (params.expert_cache_active),
    expert_cache_decode_max_tokens (params.expert_cache_decode_max_tokens),
    n_gpu_layers     (params.n_gpu_layers),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids);
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q));
        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur, "Qcur", il);
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            cb(Qcur, "Qcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        cb(Kcur, "Kcur", il);
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            cb(Kcur, "Kcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        cb(Vcur, "Vcur", il);
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            cb(Vcur, "Vcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
    GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
    GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
    GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                        cb(tmp, "ffn_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, tmp);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits_raw", il);  // [CGC Step-2/3] logits eval callback key — keep distinct from _biased variant
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    // [CGC Step-3 renormalized routing 2026-09-06] CGC_RN_ROUTING=1 only. Before the router
    // softmax, add a host-writable per-layer mask leaf (0.0 for warm/resident experts,
    // -inf for cold experts, slot_table[e]<0) so softmax renormalizes over the resident
    // experts only. Cold experts then can never be top-k selected, so the ZERO-slot fast
    // path loses no routing mass -> quality degrades gracefully ("fewer active experts")
    // instead of catastrophically (silent mass drop). The eval hook rewrites every mask leaf
    // each step from the current slot_table (graph_compute entry); mirrors the remap leaf
    // mechanism below. Mask only softmax-gated MoE layers in the pool-path token range.
    const char * rn_env = getenv("CGC_RN_ROUTING");
    const bool rn_on = rn_env != nullptr && rn_env[0] != '\0' &&
        probs_in == nullptr && gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX &&
        expert_cache_active && il >= 0 && il < n_layer + n_layer_nextn &&
        n_tokens >= 1 && cgc_is_decode_graph(n_tokens, expert_cache_decode_max_tokens);
    if (rn_on) {
        ggml_tensor * rn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_expert, 1);
        ggml_set_output(rn_mask); // writable from host; prevent allocator overwrite before softmax reads it
        cb(rn_mask, "ffn_moe_rn_mask", il);
        ggml_build_forward_expand(gf, rn_mask);
        logits = ggml_add(ctx0, logits, rn_mask); // [n_expert, n_tokens] += [n_expert, 1] broadcast
        cb(logits, "ffn_moe_logits_masked", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    // CGC expert-cache: remap leaf. An I32 input tensor of the same shape as the top-k ids
    // that the eval hook (expert_cache_on_topk) repoints to cache slot indices (pool path) or
    // 0..k-1 gathered indices (L3-B path) before the FFN mul_mat_id dispatches. Built as an
    // input so its data is writable from the host; the hook fills it via cache_remap_tensors.
    // When active, mul_mat_id below uses remap_ids instead of the raw expert ids so the FFN
    // reads cache-resident (pool / gather) weights; the routing weights (ffn_moe_weights) keep
    // using the raw ids because they index the gating probs by expert.
    ggml_tensor * remap_ids = nullptr;
    const char * dw_env = getenv("LLAMA_EXPERT_CACHE_DISABLE_WRITE");
    // [CGC 2026-09-15 S1: GPU-side slot lookup, CGC_SLOT_TABLE_GPU=1]
    //
    // The remap leaf below is an INPUT tensor that the eval hook hand-writes every step, and it is
    // the reason the dispatcher has to drain the pipeline at every layer boundary (segment i+1 may
    // not be submitted until the hook of segment i has written the leaf). This switch moves the
    // mapping into the graph as a gather:
    //
    //     slots = get_rows(slot_table, selected_experts)      slot_table : I32 [1, n_expert]
    //
    // which is exactly what the hook computes today (llama-context.cpp, expert_cache_on_topk), just
    // executed on the GPU instead of the host.
    //
    // SHAPE: ggml_get_rows asserts src0->ne[2] == ids->ne[1] and yields [src0->ne[0], ids->ne[0]..],
    // so a 1-wide row table plus a flattened ids vector gives [1, k*n_tokens], which reshapes back
    // to [k, n_tokens] in the ids' own memory order. That keeps the table's shape independent of
    // n_tokens -- one [1, n_expert] tensor per layer, reused every step, no per-step shape churn.
    //
    // NO NEW OP: ggml_get_rows already carries an I32 path everywhere it can run --
    // ggml.c ("if (a->type == GGML_TYPE_I32) type = a->type"), ggml-cpu/ops.cpp
    // ("case GGML_TYPE_I32: ggml_compute_forward_get_rows_f32") and the Metal template
    // instantiation "kernel_get_rows_i32" (ggml-metal.metal:9962).
    //
    // SCOPE: exactly the leaf's condition -- every step where the leaf would have been built
    // (single-token decode AND the small multi-token pool steps: MTP verify, gather/union), so the
    // table and the leaf can never disagree about which steps carry a mapping. An earlier revision
    // narrowed it to n_tokens == 1, which was wrong for a subtle reason worth recording:
    // build_moe_ffn's `n_tokens` is NOT ubatch.n_tokens (measured: the hook sees t->ne[1] == 2 while
    // the ubatch is 1), so the decode step never built a table at all -- only the dummy reserve
    // build did. Widening it is also what S2/S3 need, and it depends on this stage passing the
    // bit-identical gate first.
    static const bool cgc_slot_table_gpu = getenv("CGC_SLOT_TABLE_GPU") != nullptr;
    static const bool cgc_s1_dbg = getenv("CGC_S1_DBG") != nullptr;
    // [CGC 2026-09-15 S1 CONTROL] Build the S1 nodes but keep mul_mat_id consuming the host leaf.
    // The full rationale is at the point of use, further down the S1 branch.
    static const bool cgc_s1_keep_leaf = getenv("CGC_S1_KEEP_LEAF") != nullptr;
    // [CGC 2026-09-15 S1 CONTROL] Build the host leaf as well but leave it unconsumed, i.e. the
    // design the header comment on cache_slot_table_tensors describes. See the point of use.
    static const bool cgc_s1_build_leaf = getenv("CGC_S1_BUILD_LEAF") != nullptr;
    // [CGC 2026-09-26 miss mask · step 2 REBUILD] CGC_MISS_MASK=1 adds ONE more gather per MoE
    // layer inside the S1 branch: vmask = get_rows(valid_table, ids_flat), where valid_table is a
    // host-published [1, n_expert] I32 leaf carrying valid[e] = (slot_table[e] >= 0). That is the
    // device-side answer to "which of THIS step's selected experts are placeholders", which is the
    // number per-expert recompute has to be priced against. It can only be taken on the device:
    // which experts were selected is known only there (that is the entire point of
    // CGC_SLOT_TABLE_GPU), while residency is known only on the host -- so the two are intersected
    // here rather than on either side alone.
    //
    // CGC_MISS_MASK is the PRICED arm: two extra nodes per layer, no extra synchronize and no extra
    // command buffer (the gather rides the same segment as the slot gather).
    // CGC_MISS_MASK_DBG (llama-context.cpp) is diagnostic-only: it pays one extra synchronize per
    // step to read the mask back and print it. Never quote throughput from an arm that has it on --
    // the extra synchronize is exactly what the priced arm is argued not to have.
    //
    // REBUILD NOTE (2026-09-26): this code existed on 2026-09-24 and was measured
    // (docs/MISS_MASK_STEP2_2026-09-24.md) but NEVER entered version control and was then lost from
    // the work tree -- `git grep CGC_MISS_MASK HEAD` returns docs only, and `strings` over all 53
    // binaries in src/llama.cpp/build/bin finds zero hits. See
    // docs/STEP23_BLOCKER_AND_RECOVERY_2026-09-26.md §1. That is why it goes in WITH the commit.
    static const bool cgc_miss_mask = getenv("CGC_MISS_MASK") != nullptr;
    // [CGC 2026-09-15 S1 layer-0 gate] The GPU table must be used ONLY for layers whose MoE FFN is
    // actually offloaded to Metal. Layer 0 is not: in the prod25 profile its expert tensors keep
    // their full size (blk.0.ffn_gate_exps = 82M vs 45M for blk.1) and live in the BLAS buffer, so
    // its mul_mat_id is assigned to the CPU backend. The `## SPLIT`/`node #` dump from
    // GGML_SCHED_DEBUG=2 (2026-09-15 20:26/20:27 pair) shows the whole difference between the two
    // arms in one line each:
    //
    //   baseline : node #73 (MUL_MAT_ID) ffn_moe_gate-0 [CPU] ... src[2] = ffn_moe_topk_remap-0 [CPU]
    //   S1       : node #77 (MUL_MAT_ID) ffn_moe_gate-0 [CPU] ... src[2] = CPU#ffn_moe_slots-0# [NULL]
    //
    // i.e. the baseline's ids operand is a CPU-backend graph input (the host writes it straight into
    // the buffer the CPU kernel reads), while S1's ids are computed on MTL0 and therefore need a
    // Metal->CPU cross-backend copy of a RESHAPE view -- which the scheduler records with a NULL
    // backend. That is the only structural difference, and it is the shape of the 20:18 crash
    // (SIGSEGV inside ggml_compute_forward_mul_mat_id on libggml-cpu, a signature absent from every
    // other run of the day). Layers >= 1 are unaffected: node #180 ffn_moe_gate-1 [MTL0] with
    // src[2] = ffn_moe_slots-1 [MTL0], same backend, no copy.
    //
    // So the GPU table is gated to il >= CGC_S1_MIN_IL (default 1); layer 0 keeps the host leaf.
    // That still removes 39 of 40 round trips per decode step, which is the point of S1.
    static const int cgc_s1_min_il = [] {
        const char * e = getenv("CGC_S1_MIN_IL");
        return e != nullptr ? atoi(e) : 1;
    }();
    // [CGC MTP fix] the remap leaf is created for single-token decode AND small multi-token
    // pool steps (speculative/MTP verify, n_tokens <= cgc_pool_max_tokens()): the hook maps
    // expert ids to pool slot indices across ALL tokens of the batch, and the FFN reads the
    // pool regions by slot. Larger prefill batches (n_tokens > pool_max) compute over the
    // full expert weights and MUST keep using the raw top-k ids, because the per-layer expert
    // tensors are shrunk to the bounded pool capacity — a raw-id read against a capacity-slot
    // tensor would go OOB -> NaN -> garbage downstream. Creating the leaf here for such batches
    // would perturb the ggml-alloc buffer layout, so it is only built for the pool-path range.
    // [CGC M1 work item 2 · phase split] This block IS the decode graph, so the condition is now
    // the one shared phase predicate rather than a re-derived `n_tokens <= cap`: cap is only this
    // graph's ceiling (llama-cgc-phase.h), and a step wider than the pool can route takes the
    // prefill graph even when cap would have allowed it. Every other site that asks the same
    // question (the hook's prefill branch, the prewarm/prefetch pool features) asks this function.
    if (expert_cache_active && !(dw_env && dw_env[0]) && n_tokens >= 1 &&
            cgc_is_decode_graph(n_tokens, expert_cache_decode_max_tokens) && il >= 0 && il < n_layer + n_layer_nextn) {
        // [CGC M1 work item 4 · canonical gather order] CGC_CANON_ORDER=1 (or =2 for the identity
        // control) applies ONE permutation to `selected_experts` HERE, before anything downstream
        // consumes it, so every consumer follows without a second, independently computed perm:
        //
        //   * `weights = get_rows(probs, selected_experts)` below is gathered POSITIONALLY and is
        //     consumed positionally by ggml_mul, so it must follow the same perm -- permuting only
        //     the ids would pair expert A's output with expert B's weight (a silent wrong answer,
        //     not a reordering). See docs/M1_WORKITEM4_CANONICAL_GATHER_ORDER_2026-09-17.md §1.
        //   * the S1 gather's own CONT of `selected_experts`, built below in this same block
        //   * the host-written remap leaf, which the hook fills with the ids permuted by the SAME
        //     perm (llama-context.cpp, expert_cache_on_topk). That is why the placement is above the
        //     slot-table/remap branch and not after it: a placement below would hand the GPU an
        //     unpermuted id vector for a permuted weight vector.
        //
        // The perm is built on the HOST (the hook already reads the raw top-k ids there), so the
        // sort key never has to exist on the device: the hook writes the perm into a host-written
        // I32 leaf and this side consumes it as GET_ROWS' index vector.
        //
        // Two structures ARE avoided by construction, because both were measured to abort the
        // reserve build with `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)`: a 1-D root (`perm`) is
        // consumed directly -- the `slot_table` shape, the one proven to schedule -- and the permuted
        // ids are a CONT -> RESHAPE_1D -> GET_ROWS -> RESHAPE_2D chain, which is the S1 chain
        // verbatim. A 2-D perm root reached only through a view does not schedule.
        if (cgc_canon_order_mode() != 0) {
            const int64_t n_sel = (int64_t) n_expert_used * (int64_t) n_tokens;
            ggml_tensor * perm = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_sel);
            ggml_set_output(perm); // one pinned buffer per layer -- same reason as the remap leaf
            cb(perm, "ffn_moe_canon_perm", il);
            ggml_build_forward_expand(gf, perm);

            // SHAPE, measured the hard way: ggml_get_rows treats src0 as a TABLE of rows of width
            // src0->ne[0] and indexes it with src1's values (`result = [a->ne0, b->ne0, b->ne1, b->ne2]`,
            // with `GGML_ASSERT(a->ne[2] == b->ne[1])`). A flat 1-D id vector as src0 is therefore
            // wrong twice: one row of width n_sel (so an index of 0 reads the first n_sel ids), and a
            // result of [n_sel, n_sel] -- measured as
            // `ggml.c:3703 GGML_ASSERT(ggml_nelements(a) == ne0*ne1) failed` from the reshape below,
            // i.e. an abort during model load, before any token. Casting the ids to [1, n_sel] makes
            // every row exactly one id wide, which is precisely the `slot_table` shape the S1 gather
            // already uses and which the assert `a->ne[2] == b->ne[1]` is written for. This is why
            // the index vector must be ABSOLUTE and 1-D: it is consumed row-by-row in flattened order.
            ggml_tensor * sel_c = ggml_cont(ctx0, selected_experts);
            sel_c = ggml_reshape_2d(ctx0, sel_c, 1, n_sel); // [1, n_sel] table, one id per row
            sel_c = ggml_get_rows(ctx0, sel_c, perm);       // [1, n_sel]
            sel_c = ggml_reshape_2d(ctx0, sel_c, n_expert_used, n_tokens);
            cb(sel_c, "ffn_moe_topk_canon", il);
            selected_experts = sel_c;
        }
        if (cgc_slot_table_gpu && il >= cgc_s1_min_il) {
            // table: [1, n_expert], published by the hook, identical for every layer of the same
            // width so the shape (and thus ggml-alloc's sizing) never changes step to step.
            ggml_tensor * slot_table = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_expert);
            ggml_set_output(slot_table); // same reason the leaf is an output: the hook writes it just before the dispatch that reads it
            cb(slot_table, "ffn_moe_slot_table", il);
            ggml_build_forward_expand(gf, slot_table);

            // CONT: `selected_experts` is ggml_argsort_top_k()'s output, i.e. ggml_argsort +
            // ggml_view_4d that keeps nb[1] = n_expert*4 and only cuts ne[0] down to k. It is
            // therefore NOT contiguous, and ggml_reshape_1d asserts ggml_is_contiguous --
            // measured: GGML_ASSERT(ggml_is_contiguous(a)) abort inside the very first
            // graph_reserve build, before any token was generated. One I32->I32 cont makes the
            // row-major [k, n_tokens] order explicit, which is the same order the host leaf writes
            // (rd[i + j*n_expert_used]). It stays on the GPU: ggml-metal-device.m supports_op for
            // GGML_OP_CONT has "case GGML_TYPE_I32: return op->type == GGML_TYPE_I32" and
            // kernel_cpy_i32_i32 is instantiated, so no CPU fallback (which would reintroduce a
            // per-layer sync). Cost is one tiny cpy of k*n_tokens ints per layer, and dropping it
            // (the id vector is already contiguous in the host path) is S2 work.
            // [CGC 2026-09-17 §11.5] The index vector is written by the HOOK, not computed from
            // `selected_experts` on the device. That is the fix for the bug the same day localized,
            // and the reason is worth keeping:
            //
            // The previous chain was CONT(selected_experts) -> RESHAPE_1D(k*T) -> GET_ROWS. The CONT
            // was added because `selected_experts` is `ggml_argsort_top_k`'s output -- a view with
            // nb[1] = n_expert*4, i.e. NOT contiguous -- and `ggml_reshape_1d` asserts
            // `ggml_is_contiguous`. So it existed purely to satisfy an assert, and it did satisfy it
            // while breaking the data: measured 2026-09-17 (pinned readback + hook ids, same run),
            // the vector the gather consumed was correct for token 0 and wrong for every token >= 1
            // on every layer --
            //
            //   hook = [193 105 229 249 220 106 181  84 | 237 163  50 218  20  55 222 212]
            //   idx  = [193 105 229 249 220 106 181  84 | 161 103 250  74 110 121 212  99]
            //
            // i.e. a strided copy that read contiguous k-blocks: token 1's row became ranks 8..15 of
            // token 0's sorted list. Every value stayed in [0, n_expert), so nothing asserted and the
            // gather did exactly what it was told (`gather_vs_table=0` on all 16 positions) -- the
            // corruption is silent by construction. The published table is ~72% non-resident zeros,
            // which is why it surfaced as "mostly zeros with a few plausible slots" rather than as a
            // crash, and why it was read as a mapping/pool problem for several rounds.
            //
            // The hook has had these exact ids all along (`expert_cache_on_topk` derives them from
            // the router and the host-leaf arm -- which is bit-identical to the baseline -- writes
            // them), so the correct index vector never needed the device at all. Writing it into a
            // pinned per-layer tensor keeps the gather (and the table publish) on the same mapping as
            // the leaf by construction, and removes the strided copy from the correctness path.
            //
            // Contract: the hook MUST fill this tensor for every step in which the S1 nodes are built
            // (`cgc_write_ids_leaf` at both remap-leaf sites in llama-context.cpp), exactly like the
            // table. A step that builds the nodes and skips the write would leave the previous step's
            // ids in place -- a legal but wrong vector, which is the failure mode this whole section
            // exists to eliminate.
            // 1-D on purpose, and a graph root consumed DIRECTLY by the gather: that is exactly the
            // shape `slot_table` has (root -> get_rows src1), which is the one proven to deliver host
            // writes to the device. Measured 2026-09-17: a 2-D root consumed through a view read
            // zeros -- the view did not share the root's buffer (`ids_src_data` sat in the arena while
            // the root sat next to `slot_table`), and `ggml_cont(ids_leaf)` folds to `ids_leaf`
            // itself because the source is already contiguous, so the whole chain collapsed onto a
            // view of the root. Flat layout is token-major, i.e. ids[j] is the expert of output row j
            // -- the same order the hook's `ids` array has.
            // [CGC 2026-09-17 r31] MEASURED, and it retires the host-written index vector: a
            // 1-D I32 root written by `expert_cache_on_topk` and consumed directly by this GET_ROWS
            // scheduled fine and the host readback confirmed BOTH the address and the values --
            //
            //   ids_src_data == ids_leaf_data (0x11146c500), ids_src_valid=1
            //   idx = [193 105 229 249 220 106 181 84 | 237 163 50 218 20 55 222 212]   (= hook)
            //
            // -- and yet the gather returned [0 x16]: `gather_vs_table=[15]`, i.e. the device read
            // ZEROS for that index vector while the host mirror held the numbers above. A
            // host-written root consumed by a GPU kernel is therefore not delivered to the device
            // (the table and the remap leaf are consumed by `mul_mat_id`, not by a device GET_ROWS),
            // so that design cannot work no matter how the host fills it. The index vector goes BACK
            // to being device-produced (CONT of `selected_experts`), which is the structure that
            // r26 measured as self-consistent (`gather_vs_table=[0]` on all 16 positions).
            //
            // The "token >= 1 is wrong" reading that motivated the experiment is re-opened and now
            // points at the HOST side: `expert_cache_on_topk` reads the top-k with `ids[i]` linear
            // indexing (see the snapshot there), which is only correct if `t->nb[1] == k*4`. The
            // shapes below are printed so that read can be checked against the tensor's own layout.
            static int cgc_sel_shape_lines = 0;
            if (cgc_sel_shape_lines++ < 4) {
                fprintf(stderr, "CGC-S1: SELSHAPE il=%d sel_ne=[%lld,%lld] sel_nb=[%lld,%lld] "
                        "ops=%d src0_op=%d\n", il,
                        (long long) selected_experts->ne[0], (long long) selected_experts->ne[1],
                        (long long) selected_experts->nb[0], (long long) selected_experts->nb[1],
                        (int) selected_experts->op,
                        selected_experts->src[0] != nullptr ? (int) selected_experts->src[0]->op : -1);
            }
            // Consume the root DIRECTLY. An earlier revision reshaped it to 1-D first (the shape the
            // GET_ROWS of `probs`/`weights` uses), and that aborted the reserve build with
            // `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)`: the get_rows then reached the root only
            // through a view, and a view of a graph root is not assigned a backend by the scheduler.
            // This is the same trap the keep-leaf comment above records for `slot_table`, hit from the
            // other side. A 2-D index tensor is legal here: ggml_get_rows produces [a->ne0, k, T].
            // Keep the node structure that is proven to schedule: CONT -> RESHAPE_1D -> GET_ROWS ->
            // RESHAPE_2D with a 1-D index vector. Only the SOURCE of the copy changes -- from the
            // strided argsort view to the host-written leaf -- so the CONT becomes a contiguous
            // duplicate (a plain memcpy, no strides to get wrong).
            //
            // Two alternative structures aborted the reserve build with
            // `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)`, i.e. a node the scheduler never assigned
            // a backend to:
            //   * reshape_1d(ids_leaf) -> GET_ROWS            (root reached only through a view)
            //   * GET_ROWS(slot_table, ids_leaf) with a 2-D index -> [1,k,T] -> reshape_2d
            // Both had a 2-D root. A 1-D root consumed directly is the `slot_table` shape and
            // schedules; that is what this is.
            ggml_tensor * ids_cont = ggml_cont(ctx0, selected_experts);
            // Pinned on purpose: the post-synchronize readback validates the index vector by reading
            // it AFTER the command buffer completed, which only means something if its buffer is still
            // alive. Measured r25/r26: this pin is safe, and without it the readback degrades to
            // `n/a (index vector buffer recycled)` and silently proves nothing.
            ggml_set_output(ids_cont);
            // [CGC 2026-09-26 miss mask] named so the host can capture it. The mask alone is k flags
            // with no identity: CGC_MISS_MASK_DBG needs the expert ids next to it to print WHICH
            // experts missed, and scripts/check/miss_mask_check.py compares those ids element by
            // element against BATCHDBG. The tensor already exists above (the CONT the slot gather
            // consumes); only the name is new.
            cb(ids_cont, "ffn_moe_ids_cont", il);
            ggml_tensor * ids_flat = ggml_reshape_1d(ctx0, ids_cont, n_expert_used * n_tokens);
            ggml_tensor * slots    = ggml_get_rows(ctx0, slot_table, ids_flat); // I32 [1, k*n_tokens]
            // [CGC 2026-09-15 S1: buffer aliasing] `ggml_set_output` is REQUIRED here for exactly the
            // reason the old leaf needed it, and the measurement is unambiguous. The CGC-MMID-ASSERT
            // provenance field added today (`ids_data=%p`) shows that WITHOUT this flag every layer's
            // gather result collapses onto the SAME address:
            //
            //   ids_name='ffn_moe_slots-1' ids_op=VIEW ids_view_op=GET_ROWS ids_data=0x128179d60
            //   ids_name='ffn_moe_slots-2' ...                            ids_data=0x128179d60
            //   ids_name='ffn_moe_slots-4' ...                            ids_data=0x128179d60
            //   (10 of the 11 layers observed in Backup/cgc_logs/llama_server_20260915_203053.log)
            //
            // Every MoE layer then reads one shared 16-int buffer that at most one layer can have
            // written, which is exactly the observed symptom: `id_oob=16/16` on essentially every
            // layer, with the "value" being whatever F32 tensor last occupied that memory
            // (0x3F4EC2F8 ~ 0.808 = a router probability, 0x7FC00000 = +NaN). The leaf path never had
            // this because `ggml_set_output(remap)` pins one buffer per layer.
            //
            // Marking the GET_ROWS result directly (not just the reshape view) is deliberate: the view
            // is not allocated separately, so the flag has to live on the tensor ggml-alloc actually
            // sizes.
            ggml_set_output(slots);
            slots = ggml_reshape_2d(ctx0, slots, n_expert_used, n_tokens);
            cb(slots, "ffn_moe_slots", il);
            remap_ids = slots;
            // [CGC 2026-09-26 miss mask · step 2 REBUILD] the miss mask itself. Structure is copied
            // node-for-node from `slot_table` two hundred lines up, on purpose: every other shape
            // tried here hit `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)` (a node the scheduler
            // never assigned a backend to) -- notably reshape_1d(root) -> GET_ROWS, and a 2-D index
            // into GET_ROWS. The shape that is proven to schedule is a 1-D [1, n_expert] I32 root
            // consumed DIRECTLY by GET_ROWS with the same 1-D index vector `ids_flat` the slot
            // gather already uses. Reusing `ids_flat` is also what makes the mask MEAN the same
            // thing as the slots: both are sampled at exactly this step's selected experts, in the
            // same flattened order, so mask[i] == 0 says "position i of the ids mul_mat_id consumes
            // is a placeholder" -- which is precisely the per-expert recompute's work list.
            if (cgc_miss_mask) {
                ggml_tensor * valid_table = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_expert);
                // output for the same reason `slot_table` is: a host-written leaf that a device
                // kernel reads must keep its buffer across the dispatch, or the write lands in
                // memory the allocator has already handed to somebody else.
                ggml_set_output(valid_table);
                cb(valid_table, "ffn_moe_valid", il);
                ggml_build_forward_expand(gf, valid_table);
                ggml_tensor * vmask = ggml_get_rows(ctx0, valid_table, ids_flat); // I32 [1, k*n_tokens]
                // pinned so the post-synchronize readback in CGC_MISS_MASK_DBG reads this step's
                // values and not whatever the arena holds afterwards (the failure mode documented
                // on cache_slots_out_tensors in llama-context.h).
                ggml_set_output(vmask);
                cb(vmask, "ffn_moe_missmask", il);
            }
            // [CGC 2026-09-15 S1 CONTROL: CGC_S1_KEEP_LEAF]
            //
            // The bisect left two explanations for the digest divergence that need OPPOSITE fixes,
            // and md5 sets alone cannot separate them:
            //   MAPPING  the GPU-computed ids differ from the host-written ids at some layer;
            //   NUMERICS the ids are identical everywhere, but ADDING these four nodes per layer
            //            (table + CONT + GET_ROWS + VIEW) changes what ggml-backend/sched and
            //            ggml-alloc produce -- a different buffer assignment or a different Metal
            //            kernel/accumulation order -- so the logits move in the last bits.
            // Measured so far: min_il=39 (1 layer), 38 (2) and 20 (20 layers, ~80 extra nodes) are
            // all bit-identical, min_il=6 is stably DIFFERENT and min_il=2 is UNSTABLE. "Stably
            // different" is the part a mapping error explains least well, because a wrong table
            // entry points at another expert's weights -- a large, deterministic change -- whereas a
            // numerically perturbed logit diverges only at the argmax.
            //
            // This flag is the control that separates them. It builds every node the S1 branch
            // builds AND the host leaf, then lets mul_mat_id consume the LEAF. The graph therefore
            // carries exactly the S1 extra nodes while the ids still come from the host, so:
            //   bit-identical to the baseline -> the extra nodes have NO numerical effect, and any
            //                                   divergence in the real arm is the mapping;
            //   still different               -> the extra nodes themselves move the answer, and the
            //                                   mapping was never the question.
            // The leaf is not dead code here: it is the thing being controlled for. Note that this
            // arm is only meaningful against a baseline in the SAME sweep (same build fingerprint),
            // and it must never be quoted for throughput -- it exists to answer one yes/no question.
            if (cgc_s1_keep_leaf || cgc_s1_build_leaf) {
                // Two related experiments, one block, because they share the leaf construction.
                //
                // CGC_S1_KEEP_LEAF=1 -- build the leaf AND let mul_mat_id consume it. The graph then
                // carries the whole S1 node set while the ids come from the host, which is what
                // separates "the ids are wrong" from "the nodes move the answer". Measured: with all
                // 39 layers served it is BIT-IDENTICAL to the baseline (dc055e63, stable), so the
                // nodes are inert; and a per-layer readback of the gather against this leaf showed
                // the ids agree at every layer for the tokens that matter.
                //
                // When the leaf is consumed, nothing else consumes `slots`, so the CONT / GET_ROWS /
                // VIEW chain would be orphaned and `slot_table` would become a graph root with no
                // consumer -- the scheduler never assigns it a backend and ggml-alloc aborts the
                // reserve build with `ggml-alloc.c:623 GGML_ASSERT(buffer_id >= 0)`. Expanding
                // `slots` here keeps the chain live. A tensor is in the graph only if the traversal
                // reaches it, not merely because it was constructed.
                //
                // CGC_S1_BUILD_LEAF=1 -- build the leaf and LEAVE IT UNCONSUMED (ids still come from
                // the gather). This is not a new design: it is exactly what the header comment on
                // cache_slot_table_tensors has always claimed S1 does --
                //     "The leaf itself is still built when this is on: it is simply not consumed.
                //      That keeps every other path bit-identical and makes the S1 A/B a pure
                //      scheduler/dispatch change."
                // The implementation never did it (the branch below is an if/else, so the leaf is
                // NOT built when the table is). With the ids ruled out on one side and the extra
                // nodes ruled out on the other, the one remaining difference between the real S1
                // arm and the keep-leaf control is the ABSENCE of this node, so the documented
                // design is the obvious thing to measure. The hook still writes it (and still
                // counts n_zero_mapped_selected), so this also repairs the silently disabled
                // instrument recorded in eng-gate-0003.
                if (cgc_s1_keep_leaf) {
                    ggml_build_forward_expand(gf, slots);
                }
                ggml_tensor * remap = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_expert_used, n_tokens);
                ggml_set_output(remap);
                cb(remap, "ffn_moe_topk_remap", il);
                ggml_build_forward_expand(gf, remap);
                if (cgc_s1_keep_leaf) {
                    remap_ids = remap;
                }
            }
            if (cgc_s1_dbg) {
                // [CGC 2026-09-17 §11.3] The decisive shape question, printed for EVERY distinct build
                // instead of the first six. The post-synchronize readback proved the gather really does
                // write token 0's k lookups and then mostly zeros, which is what looking up a long index
                // vector whose tail is not this step's experts looks like; the published table is ~72%
                // zeros (non-resident), so garbage indices land on 0 most of the time and occasionally
                // on a resident entry -- exactly the observed mix.
                //
                // What decides it is the pair of shapes, not any value:
                //   selected_experts->ne[1]  the router's own token count (ground truth)
                //   n_tokens                 the local count this block builds the S1 nodes with
                // The code already documents that these can disagree ("build_moe_ffn's n_tokens is NOT
                // ubatch.n_tokens"). If selected_experts->ne[1] > n_tokens, then ids_flat is k long
                // while the consumer reads k*T, and the zeros are explained by construction rather than
                // by any allocator or kernel behaviour.
                //
                // Deduplicated by signature with a count: the distinct shapes are few, and printing one
                // line per layer per step is how the earlier probe managed to look at a shape nobody had
                // questioned (all six of its lines were the 1-token reserve build).
                static std::map<std::string, long long> cgc_s1_sig;
                char sig[256];
                snprintf(sig, sizeof(sig),
                        "n_tokens=%lld sel_ne=[%lld,%lld] ids_flat_ne0=%lld "
                        "slots_ne=[%lld,%lld] slots_nbytes=%zu",
                        (long long) n_tokens, (long long) selected_experts->ne[0],
                        (long long) selected_experts->ne[1], (long long) ids_flat->ne[0],
                        (long long) slots->ne[0], (long long) slots->ne[1],
                        ggml_nbytes(slots));
                long long & n_sig = cgc_s1_sig[sig];
                if (n_sig < 3) {
                    fprintf(stderr, "CGC-S1: SHAPE il=%d %s (occurrence %lld)%s\n",
                            il, sig, n_sig + 1,
                            selected_experts->ne[1] != n_tokens
                                ? "  <-- MISMATCH: the S1 gather is sized from n_tokens, the router has more tokens"
                                : "");
                }
                n_sig++;
            }
        } else {
        ggml_tensor * remap = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_expert_used, n_tokens);
        ggml_set_output(remap); // prevent ggml-alloc from overwriting the hook-written ids before mul_mat_id consumes them
        cb(remap, "ffn_moe_topk_remap", il);
        ggml_build_forward_expand(gf, remap);
        remap_ids = remap;
        }
    }

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    // CGC expert-cache: the mul_mat_id / add_id ops below index the (cache-resident) expert
    // weight tensors, so they must consume the remapped ids (slot indices) when active.
    const char * mmid_raw = getenv("LLAMA_MMID_RAW");
    ggml_tensor * mm_id_ids = (mmid_raw && mmid_raw[0]) ? selected_experts : (remap_ids != nullptr ? remap_ids : selected_experts);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, mm_id_ids, up_exps_s); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);
        gate_up = cgc_moe_dump_node(ctx0, gate_up, "gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, mm_id_ids);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, mm_id_ids, up_exps_s); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, mm_id_ids);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, mm_id_ids, gate_exps_s); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, mm_id_ids);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        up = ggml_clamp(ctx0, up, -limit, limit);
                        cb(up, "ffn_moe_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_moe_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, up);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
                cur = cgc_moe_dump_node(ctx0, cur, "swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    // CGC P0: batch down-combine (Lily-style). One kernel replaces 8 down GEMVs + 7 adds.
    // Only supports: decode (n_tokens==1), Q3_K down weights, no scale/bias on down,
    // weight_after_ffn (standard MoE). Falls back to original path if any condition fails.
    static const bool cgc_down_combine = []{
        const char * e = getenv("CGC_DOWN_COMBINE");
        return e != nullptr && e[0] == '1';
    }();

    // [CGC 2026-09-18] ONE predicate, used by both the audit print and the branch below. It used
    // to be written twice (once as `fuse` inside the audit, once as the `if` condition) -- a
    // latent way for an instrument to describe a decision that the code does not actually make.
    //
    // Two changes from the original, both required for this to fire on the shipped model:
    //   * the type test accepts the three types the model actually uses for ffn_down_exps
    //     (IQ3_S x37 + IQ4_XS x3 on the flagship; the audit prints the per-layer truth);
    //   * `n_tokens == 1` becomes overridable, because under MTP every trunk graph is a VERIFY
    //     graph (n_tokens = 2 or 4) -- with the type gate open but this one closed, the fused
    //     path would still be structurally unreachable.
    // ⚠️ CGC_DC_MULTITOK is read in BOTH this file and ggml-metal-ops.cpp, and the two must stay
    // in step: if this gate admits a shape that the Metal op then rejects, the node ends up in
    // the graph with no implementation behind it.
    static const bool cgc_dc_multitok = []{
        const char * e = getenv("CGC_DC_MULTITOK");
        return e != nullptr && e[0] == '1';
    }();

    // [CGC 2026-09-18] How wide a multi-token shape may be and still take the fused path.
    // 8 is the pool path's own bound (CGC_POOL_MAX_TOKENS, default 8) and covers the two shapes
    // that matter: verify under MTP is n_tokens = 2 or 4. It deliberately EXCLUDES prefill
    // (16 / 2048 / 5632).
    //
    // Measured, and the reason this constant exists: with the bound absent, the audit showed
    // fuse=1 on 37 rows of n_tokens=16 and 74 rows of n_tokens=5632 per run -- i.e. prefill was
    // taking the fused path -- and the A/B's answer_md5_set then DIFFERED between the two arms
    // (87647aec vs 72ca6608), while the single-token pair produced the same digest
    // (72ca6608 == 72ca6608). So the single-token kernel is right and the wide shape is not.
    static constexpr int CGC_DC_MAX_TOKENS = 8;
    const bool cgc_dc_shape_ok = (n_tokens == 1) ||
                                 (cgc_dc_multitok && n_tokens <= CGC_DC_MAX_TOKENS);
    const bool cgc_dc_fuse = cgc_down_combine &&
        cgc_dc_shape_ok &&
        down_exps != nullptr &&
        (down_exps->type == GGML_TYPE_Q3_K ||
         down_exps->type == GGML_TYPE_IQ3_S ||
         down_exps->type == GGML_TYPE_IQ4_XS) &&
        down_exps_s == nullptr && down_exps_b == nullptr &&
        !weight_before_ffn &&
        mm_id_ids != nullptr;

    // [CGC 2026-09-18 DOWN-COMBINE AUDIT] CGC_DOWN_COMBINE_AUDIT=1 prints, once per MoE layer per
    // built graph, the value of every one of the six gate conditions and the resulting decision.
    // Read-only: it does not change the decision below.
    // Why it exists: the gate is an AND of six conditions, and the type condition is the one you
    // cannot see from the outside -- the GGUF says IQ3_S while the fused Metal kernel
    // (ggml-metal.metal:11886 "kernel_mul_mv_id_down_combine_q3_K_f32") is Q3_K-only. This prints
    // what the loader actually produced, per layer, instead of what the file nominally contains.
    if (getenv("CGC_DOWN_COMBINE_AUDIT") != nullptr) {
        const bool fuse = cgc_dc_fuse;   // the SAME predicate the branch below uses, by construction
        fprintf(stderr,
                "CGC-DCAUDIT il=%d flag=%d n_tokens=%lld down_type=%s has_scale=%d has_bias=%d "
                "weight_before_ffn=%d ids=%d => fuse=%d\n",
                il, (int) cgc_down_combine, (long long) n_tokens,
                down_exps != nullptr ? ggml_type_name(down_exps->type) : "(null)",
                (int) (down_exps_s != nullptr), (int) (down_exps_b != nullptr),
                (int) weight_before_ffn, (int) (mm_id_ids != nullptr), (int) fuse);
    }

    if (cgc_dc_fuse) {
        // weights is [1, n_expert_used, n_tokens] → reshape to [n_expert_used, n_tokens]
        ggml_tensor * weights_2d = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        cb(weights_2d, "ffn_moe_weights_2d", il);

        ggml_tensor * moe_out = ggml_mul_mat_id_down_combine(ctx0, down_exps, cur, mm_id_ids, weights_2d);
        cb(moe_out, "ffn_moe_out_down_combine", il);
        // [CGC 2026-09-18] The audit above says the GATE is true; this says the NODE exists.
        // They are different claims and only this one is about the graph. It is printed here
        // (builder time) rather than read out of CGC-GRPH/CGC-GPUOPS because those two live
        // inside the segmented dispatch path -- and the one graph where this fires (the MTP
        // nextn head, il = n_layer) does not appear in either of them.
        if (getenv("CGC_DOWN_COMBINE_AUDIT") != nullptr) {
            fprintf(stderr,
                    "CGC-DCFUSED il=%d op=%d(%s) name=%s out_ne=[%lld,%lld,%lld] "
                    "n_tokens=%lld down_type=%s swiglu_src=%d(%s)\n",
                    il, (int) moe_out->op, ggml_op_name(moe_out->op), moe_out->name,
                    (long long) moe_out->ne[0], (long long) moe_out->ne[1], (long long) moe_out->ne[2],
                    (long long) n_tokens, ggml_type_name(down_exps->type),
                    (int) cur->op, ggml_op_name(cur->op));
        }
        ggml_build_forward_expand(gf, moe_out);
        return moe_out;
    }

    experts = build_lora_mm_id(down_exps, cur, mm_id_ids, down_exps_s); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);
    experts = cgc_moe_dump_node(ctx0, experts, "down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, mm_id_ids);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    //
    // [CGC 2026-09-17 §9.18.9 reduction-order probe] CGC_ADD_ORDER=rev reverses the ASSOCIATION of
    // this left-to-right chain: (((c0+c1)+c2)+...) becomes (((c7+c6)+c5)+...).
    //
    // Why this is the right shape for the question it answers. The open question is whether the
    // pool-path A/B divergence is carried by the REDUCTION ORDER/PRECISION of the FFN aggregation
    // (the "coin flip": a sub-ulp difference that flips the top-k three passes later) or by the two
    // arms genuinely computing different values. Two facts pin the probe down:
    //   * the per-expert row index i is the POSITION in the ids array, and `cur_experts[i]` is the
    //     i-th expert's already-weighted contribution -- so reordering the CHAIN reassociates the
    //     same 8 terms and changes nothing but floating-point summation order. Reordering the IDS
    //     (the obvious alternative) does NOT have that property: the routing weights are gathered
    //     positionally (`weights = get_rows(probs, selected_experts)`) and consumed positionally by
    //     `ggml_mul`, so permuting the ids alone would pair expert A's output with expert B's weight
    //     -- a silent wrong answer, not a reordering.
    //   * fp32 `add` is not associative, so a full reversal is the STRONGEST single test: if even
    //     the reversed chain is bit-identical, then NO canonicalisation of this sum (ascending by
    //     expert id, slot order, or anything else) can change the result either -- every one of them
    //     is a permutation of the very terms whose permutation just measured as inert. If it is NOT
    //     bit-identical, the size of the movement is the order-sensitivity of the reduction, which is
    //     the number a canonical order would be introduced to control.
    // Same treatment on both arms (this is a graph property, not an arm property), default off.
    static const char * cgc_add_order = getenv("CGC_ADD_ORDER");
    static const bool   cgc_add_rev   = cgc_add_order != nullptr && strcmp(cgc_add_order, "rev") == 0;
    // One-shot banner: an arm is only quotable if the log it produced names the arm that produced
    // it. A run whose CGC_ADD_ORDER did not reach the server is otherwise indistinguishable from a
    // run where the reversal simply changed nothing -- the same "absent instrument reads as equal"
    // failure the capture tooling carries a mandatory same-arm control for.
    // RAW fprintf(stderr), and that is measured rather than stylistic. Two attempts to make this
    // banner readable through the logging macros both produced a log in which the arm could not name
    // itself, while the SAME run carried 3847 `CGC-` lines from the ggml-metal probes:
    //   * LLAMA_LOG_INFO  -- absent. (The arm's log is not simply WARN-only: it has `I cmn` / `I srv`
    //     lines from the server's own logger, but zero `llama_context`/`llama_model_loader` lines.)
    //   * LLAMA_LOG_WARN  -- absent too, even though the string is verifiably in the loaded dylib
    //     (`strings build/bin/libllama.0.dylib`) and `llama_log_internal` imposes no level filter.
    // Rather than keep probing which channel survives this harness, the banner uses the channel whose
    // delivery is proven in these very logs: the `[CGC] Shutdown reminder` line at the head of every
    // arm log is a raw stderr write from the engine, with no server prefix and no timestamp. An arm
    // identity has to be readable in the arm's own log, so it rides the channel that arrives.
    static const bool cgc_add_rev_banner = [] {
        fprintf(stderr, "CGC-ADD-ORDER: expert aggregation = %s\n",
                cgc_add_rev ? "REVERSED (reduction-order probe)" : "left-to-right");
        fflush(stderr);
        return true;
    }();
    (void) cgc_add_rev_banner;

    ggml_tensor * moe_out = cur_experts[cgc_add_rev ? hparams.n_expert_used - 1 : 0];

    if (cgc_add_rev) {
        for (int i = (int) hparams.n_expert_used - 2; i >= 0; --i) {
            moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

            ggml_build_forward_expand(gf, moe_out);
            // [CGC 2026-09-18 NAMING] The reduce chain left its intermediates UNNAMED, so the
            // node-level GPU instrument bucketed n_expert_used-1 = 6 nodes per layer into `node`
            // (240 nodes/step -- the single largest unnamed cluster, MEASURED). A name here is
            // free: `cb` is ggml_format_name and nothing else (llama-context.cpp:6866-6872), and
            // no numeric path matches "ffn_moe_add-" (the only name readers are the CGC_VERIFY_OP
            // TIMING predicate at llama-context.cpp:3808-3824 and the capture dispatch at
            // :6944+, both exact strcmp / diagnostic). The LAST term still ends up as
            // `ffn_moe_out` because the cb() after the loop renames it.
            cb(moe_out, "ffn_moe_add", il);
        }
    } else {
        for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
            moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

            ggml_build_forward_expand(gf, moe_out);
            cb(moe_out, "ffn_moe_add", il);
        }
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);
    moe_out = cgc_moe_dump_node(ctx0, moe_out, "moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // select one of the 2 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 2> inps;

    // token embeddings path (ubatch.token != nullptr)
    {
        auto & cur = inps[0];

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }
    }

    // vector embeddings path (ubatch.embd != nullptr)
    {
        auto & cur = inps[1];

        cur = inp->embd;
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // For Granite architecture
    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    if (hparams.f_embedding_scale != 0.0f && (ubatch.token || hparams.n_deepstack_layers == 0)) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "attn_scale");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    GGML_UNUSED(v_cur);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = k;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_base->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_base->build_input_v_rot(ctx0);

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_kq_mask = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);
    inp_raw->self_kq_mask_cnv = inp_raw->self_kq_mask;

    inp_raw->self_k_rot = raw_ctx->build_input_k_rot(ctx0);
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->build_input_k_rot(ctx0);
    inp->inp_hca.k_rot = mctx_cur->get_hca()->build_input_k_rot(ctx0);
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (arch == LLM_ARCH_MODERN_BERT) {
                    // modern bert gte reranker builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, std::vector<uint32_t>> sampling_rows;
    uint32_t n_rows = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampling_rows[ubatch.seq_id[i][0]].push_back(n_rows++);
        }
    }

    res->t_sampled.resize(n_rows, nullptr);
    res->t_sampled_probs.resize(n_rows, nullptr);
    res->t_sampled_logits.resize(n_rows, nullptr);
    res->t_candidates.resize(n_rows, nullptr);

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row to keep the single-output graph static regardless of active samplers
    // multi-output graphs can still vary with the number of output rows
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }

    static const std::vector<uint32_t> dummy_row = { 0 };

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);

        // inactive samplers always work on the first row
        const bool active = it != sampling_rows.end();
        const auto & rows = active ? it->second : dummy_row;
        const int i_out   = active ? 1          : 0;

        for (uint32_t i = 0; i < rows.size(); ++i) {
            ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], rows[i] * logits_t->nb[1]);
            ggml_format_name(logits_seq, "logits_seq_%d_%u", seq_id, i);

            struct llama_sampler_data data = {
                /*.logits       =*/ logits_seq,
                /*.probs        =*/ nullptr,
                /*.sampled      =*/ nullptr,
                /*.candidates   =*/ nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);

            if (data.sampled != nullptr) {
                if (active) {
                    res->t_sampled[rows[i]] = data.sampled;
                }
                outs[1] = data.sampled;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.probs != nullptr) {
                if (active) {
                    res->t_sampled_probs[rows[i]] = data.probs;
                }
                outs[1] = data.probs;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.logits != nullptr) {
                if (active) {
                    res->t_sampled_logits[rows[i]] = data.logits;
                }
                outs[1] = data.logits;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.candidates != nullptr) {
                if (active) {
                    res->t_candidates[rows[i]] = data.candidates;
                }
                outs[1] = data.candidates;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
        }
    }

    // TODO: Call backend_accept after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        if (it == sampling_rows.end()) {
            continue;
        }

        for (uint32_t row : it->second) {
            ggml_tensor * selected_token = res->t_sampled[row];
            if (selected_token != nullptr && sampler->iface->backend_accept) {
                sampler->iface->backend_accept(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
