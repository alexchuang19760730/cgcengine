#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

#ifdef MTP_SUPPORT
    // [CGC MTP fix] compute spec_mtp early so we can skip the default warmup.
    // The default warmup (inside common_init_from_params) builds the graph with
    // embeddings_nextn=false, but the MTP driver later calls llama_set_embeddings_nextn(true)
    // which changes graph topology.  Although sched_need_reserve is set, the warmup graph's
    // buffer layout can leave stale state.  Skipping warmup here and doing a manual warmup
    // after set_embeddings_nextn ensures the graph is built with the correct MTP topology.
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    const bool warmup_skip = spec_mtp;
    const bool warmup_prev = params.warmup;
    if (getenv("CGC_MTP_DBG")) {
        fprintf(stderr, "MTPDBG main: spec_mtp=%d warmup_skip=%d warmup_prev=%d types_sz=%zu\n",
                (int)spec_mtp, (int)warmup_skip, (int)warmup_prev, params.speculative.types.size());
    }
    if (warmup_skip) {
        params.warmup = false;
    }
#else
    const bool spec_mtp = false;
#endif

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;
#ifdef MTP_SUPPORT
    common_speculative_init_result_ptr spec_init;

    // MTP draft: the nextn head lives inside the target model. Create a second context with
    // ctx_type=MTP against the same model (mirrors llama-server via common_speculative_init_result).
    // The MTP context shares KV with the target (cparams.ctx_other = ctx_tgt) and only runs the
    // nextn block, so its decode cost is ~1 layer instead of the full trunk.
#endif

    // no speculative decoding requested: run the plain target loop (C0 baseline arm)
    // note: the default types vector is { COMMON_SPECULATIVE_TYPE_NONE }, not empty
    const bool no_spec = (params.speculative.types.empty() ||
                          (params.speculative.types.size() == 1 &&
                           params.speculative.types[0] == COMMON_SPECULATIVE_TYPE_NONE)) &&
                         !params.speculative.has_dft();

    // TODO: simplify this logic
    if (no_spec) {
        // ctx_dft stays null; the loop skips draft/accept/process
    }
#ifdef MTP_SUPPORT
    else if (spec_mtp) {
        auto params_dft = common_base_params_to_speculative(params);

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        // [CGC §8.86] take SOLE ownership: spec_init->context() borrows, reset() would
        // double-own -> double llama_free at teardown (free_tiny_botch, ASan-confirmed).
        ctx_dft.reset(spec_init->release_context());
        if (ctx_dft == nullptr) {
            LOG_ERR("failed to create MTP context\n");
            return 1;
        }
    }
#endif
    else {
        const auto & params_spec = params.speculative.draft;

        auto params_dft = params;

        params_dft.n_outputs_max = params.n_parallel;
        params_dft.n_outputs_max_per_seq = 1;

        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.draft.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.draft.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        auto cparams = common_context_params_to_llama(params_dft);
        ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
    }

    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = ctx_dft.get();

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = (common_context_can_seq_rm(ctx_tgt)       == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    const bool use_ckpt_dft = !spec_mtp && ctx_dft != nullptr && (common_context_can_seq_rm(ctx_dft.get()) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);
#ifdef MTP_SUPPORT
    // [CGC MTP fix] Qwen3.5 tokenizer has add_bos=false, so tokenize() does not
    // prepend BOS.  The MTP driver needs at least one prompt token to extract the
    // hidden state for pending_h (the carry-over embedding fed to the draft head).
    // Without BOS, inp=[9419] for "Hello", batch_enc would be empty (inp.size()-1=0),
    // process() sees n_tokens=0 and skips pending_h, so the draft gets a zero embedding
    // and produces garbage logits (argmax=0).  Manually prepend BOS when missing.
    {
        const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
        const llama_token bos = llama_vocab_bos(vocab);
        if (bos != LLAMA_TOKEN_NULL && (inp.empty() || inp[0] != bos)) {
            inp.insert(inp.begin(), bos);
        }
        if (getenv("CGC_MTP_DBG")) {
            fprintf(stderr, "MTPDBG tokenize: inp.size=%zu bos=%d eos=%d vals=[", inp.size(),
                    (int)llama_vocab_bos(vocab), (int)llama_vocab_eos(vocab));
            for (size_t i = 0; i < inp.size() && i < 8; ++i) fprintf(stderr, "%d ", inp[i]);
            fprintf(stderr, "]\n");
            fflush(stderr);
        }
    }
#endif

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);
#ifdef MTP_SUPPORT
    if (getenv("CGC_MTP_DBG")) {
        fprintf(stderr, "MTPDBG main: spec=%p warmup_skip=%d warmup_prev=%d ctx_dft=%p\n",
                (void*)spec, (int)warmup_skip, (int)warmup_prev, (void*)params.speculative.draft.ctx_dft);
    }

    // [CGC MTP fix] manual warmup AFTER set_embeddings_nextn has been called by the MTP driver.
    // The default warmup was skipped (params.warmup=false above) to avoid building a graph with
    // embeddings_nextn=false.  Now that the MTP driver has enabled embeddings_nextn on both
    // ctx_tgt (unmasked) and ctx_dft (masked), we safely warm up with the correct graph topology.
    if (warmup_skip && warmup_prev) {
        if (getenv("CGC_MTP_DBG")) {
            fprintf(stderr, "MTPDBG main: entering manual warmup, ctx_tgt=%p ctx_dft=%p\n",
                    (void*)ctx_tgt, (void*)ctx_dft.get());
        }
        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);
        if (bos != LLAMA_TOKEN_NULL) tmp.push_back(bos);
        if (eos != LLAMA_TOKEN_NULL) tmp.push_back(eos);
        if (tmp.empty()) tmp.push_back(0);

        // warm up target context (embeddings_nextn=true, masked=false)
        llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) llama_n_batch(ctx_tgt))));
        llama_memory_clear(llama_get_memory(ctx_tgt), true);
        llama_synchronize(ctx_tgt);
        llama_perf_context_reset(ctx_tgt);

        // warm up draft context (embeddings_nextn=true, masked=true)
        if (ctx_dft != nullptr) {
            llama_decode(ctx_dft.get(), llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) llama_n_batch(ctx_dft.get()))));
            llama_memory_clear(llama_get_memory(ctx_dft.get()), true);
            llama_synchronize(ctx_dft.get());
            llama_perf_context_reset(ctx_dft.get());
        }
    }
#endif

    common_speculative_begin(spec, seq_id, prompt_tgt);

#ifdef MTP_SUPPORT
    // eval the prompt (for MTP the draft context shares KV with the target, so no separate
    // prompt decode is needed -- the driver's process() captures the hidden states)
    // note: use a properly-formed batch (pos/seq_id) -- llama_batch_get_one leaves them
    // null, and the MTP driver's process() dereferences seq_id[k][0] directly.
    {
        llama_batch batch_enc = llama_batch_init(inp.size() - 1, 0, 1);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            common_batch_add(batch_enc, inp[i], (llama_pos) i, { 0 }, i == inp.size() - 2);
        }
        if (getenv("CGC_MTP_DBG")) {
            fprintf(stderr, "MTPDBG prompt_enc: inp.size=%zu batch_enc.n_tokens=%d id_last=%d\n",
                    inp.size(), batch_enc.n_tokens, id_last);
            fflush(stderr);
        }
        llama_decode(ctx_tgt, batch_enc);
        common_speculative_process(spec, batch_enc);
        llama_batch_free(batch_enc);
    }
#endif
    if (!spec_mtp && ctx_dft != nullptr) {
        llama_decode(ctx_dft.get(), llama_batch_get_one(inp.data(), inp.size() - 1));
    }

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    size_t n_draft = 0;

    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // generate a new draft
            if (spec != nullptr) {
                common_speculative_get_draft_params(spec, seq_id) = {
                    /* .drafting   = */ true,
                    /* .n_max      = */ -1,
                    /* .n_past     = */ n_past,
                    /* .id_last    = */ id_last,
                    /* .prompt     = */ &prompt_tgt,
                    /* .result     = */ &draft, // output
                };
                common_speculative_draft(spec);
            }

            // save the original draft size
            n_draft = draft.size();

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            if (!spec_mtp) {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { seq_id }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt);
#ifdef MTP_SUPPORT
            common_speculative_process(spec, batch_tgt);
#endif
        }

        // evaluate the same batch with the draft model
        // (skipped for MTP: the draft context shares KV with the target and the driver's
        //  process() already captured the verify hidden states during the target decode)
        if (!spec_mtp && ctx_dft != nullptr) {
            llama_decode(ctx_dft.get(), batch_tgt);
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < draft.size()) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, draft.size());

            draft = std::move(ids);

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            if (!spec_mtp) {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        if (spec != nullptr) {
            common_speculative_accept(spec, seq_id, ids.size() - 1);
        }

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
            if (!spec_mtp) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, n_past, -1);
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
