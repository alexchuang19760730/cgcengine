#include "llama.h"
// [CGC layer bisect] staging API: llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp
#include "../../src/llama-ext.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// [CGC layer bisect] write every enabled layer-input buffer's rows to
// /tmp/cgc_<tag>_l<il>.f32. The buffer holds the LAST decode call's tokens:
// row j = the j-th token of that call. Used for the bit-level spec-vs-non-spec
// divergence bisect (see speculative-simple.cpp for the mirror dump).
static void cgc_dump_layer_inp(llama_context * ctx, const llama_model * model, const char * tag, int n_rows) {
    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    for (int il = 0; il < n_layer; ++il) {
        const float * p = llama_get_embeddings_layer_inp(ctx, (uint32_t) il);
        if (!p) continue;
        char path[128];
        snprintf(path, sizeof(path), "/tmp/cgc_%s_l%02d.f32", tag, il);
        FILE * f = fopen(path, "wb");
        if (f) { fwrite(p, sizeof(float), (size_t) n_embd * n_rows, f); fclose(f); }
    }
    fprintf(stderr, "CGC-LAYER-DUMP %s: %d layers x %d rows x %d embd\n", tag, n_layer, n_rows, n_embd);
    fflush(stderr);
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [-t n_threads] [-p prompt] [-s seed] [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    int ngl = 99;
    // [EXPERIMENTAL] expert streaming cache budget for MoE weights (0 = off)
    size_t expert_cache_bytes = 0;
    // CGC: read from env var when -expert-cache flag is not provided
    if (expert_cache_bytes == 0) {
        const char * env_ec = getenv("CGC_EXPERT_CACHE_BYTES");
        if (env_ec) expert_cache_bytes = std::stoull(env_ec);
    }
    // model load mode (default AUTO = mmap when supported)
    llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO;
    // number of tokens to predict
    int n_predict = 32;
    // number of CPU threads (0 = llama default)
    int n_threads = 0;
    // optional -p prompt (overrides the positional prompt if both given)
    std::string opt_prompt;
    // accepted for CLI compatibility; greedy sampling is deterministic so it is unused
    uint32_t seed = 0;
    // --ignore-eos: keep generating past the EOG token (for steady-state t/s measurement)
    bool ignore_eos = false;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-no-mmap") == 0 || strcmp(argv[i], "--no-mmap") == 0) {
                load_mode = LLAMA_LOAD_MODE_NONE;
            } else if (strcmp(argv[i], "-expert-cache") == 0) {
                if (i + 1 < argc) {
                    try {
                        expert_cache_bytes = std::stoull(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_threads = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
                if (i + 1 < argc) {
                    opt_prompt = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
                if (i + 1 < argc) {
                    try {
                        seed = (uint32_t) std::stoul(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--ignore-eos") == 0) {
                ignore_eos = true;
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
        if (!opt_prompt.empty()) {
            // -p wins over the positional prompt (matches llama-cli precedence)
            prompt = opt_prompt;
        }
        if (seed != 0) {
            fprintf(stderr, "%s: note: -s/--seed accepted for CLI compatibility; greedy sampling is deterministic, seed unused\n", __func__);
        }
    }

    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    int ret = 0;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    model_params.load_mode = load_mode;
    model_params.expert_cache_bytes = expert_cache_bytes;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    // null-init so the cleanup lambda below can free unconditionally on error paths
    llama_sampler * smpl = nullptr;
    llama_context * ctx  = nullptr;

    auto cgc_cleanup = [&]() {
        if (smpl != nullptr) {
            llama_sampler_free(smpl);
        }
        llama_free(ctx);
        llama_model_free(model);
    };

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        ret = 1;
        cgc_cleanup();
        return ret;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    if (n_threads > 0) {
        ctx_params.n_threads = n_threads;
    }
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;
    // [CGC logit bin] result_norm rows (llama_get_embeddings_ith) for the
    // bit-level comparison vs the speculative path — embeddings default to
    // false, which makes get_embeddings_ith fail.
    if (getenv("CGC_LOGIT_DBG") || getenv("CGC_LAYER_BIN")) {
        ctx_params.embeddings = true;
    }

    ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        ret = 1;
        cgc_cleanup();
        return ret;
    }

    // [CGC layer bisect] enable per-layer input extraction BEFORE the first decode
    // so the graph is built with those outputs and output_reserve() allocates the
    // host buffers (n_embd * n_batch floats per layer). Layers 0..n_layer-1 only —
    // index n_layer (the output-norm input) is served by embeddings/nextn instead.
    if (getenv("CGC_LAYER_BIN")) {
        const int n_layer_dbg = llama_model_n_layer(model);
        for (int il = 0; il < n_layer_dbg; ++il) {
            llama_set_embeddings_layer_inp(ctx, (uint32_t) il, true);
        }
        fprintf(stderr, "SIMP: CGC_LAYER_BIN enabled layer_inp on %d layers\n", n_layer_dbg);
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            ret = 1;
            cgc_cleanup();
            return ret;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            ret = 1;
            cgc_cleanup();
            return ret;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    int n_pos = 0;
    llama_token new_token_id;

    if (!llama_model_has_encoder(model)) {
        // eval the prompt in n_batch-sized chunks: the L4 metal-pool union-fit cap may shrink
        // n_batch below the prompt length (llama-cli chunks the same way).
        const int32_t n_batch_eff = llama_n_batch(ctx);
        // [CGC bit-bisect 2026-08-30 v3] CGC_SIMP_TAIL_SPLIT=1: decode the LAST prefill chunk
        // as [n-1 tokens] + [last token alone (M=1)] — exactly the batch shapes the speculative
        // path uses (prefill tail M=2 [24,25] then tok26 as row0 of a fresh batch with
        // n_past=26). Dumping both sub-batches (simpp / simp) discriminates:
        //   1. simpp(M=2) vs specp(M=2)      — equal => spec prefill itself is clean
        //   2. simp(M=1 tok26) vs spec verify row0 — equal => full-attn kernel is shape-dependent
        //   3. simp(M=1 tok26) vs simp3(M=3 row2)  — the M=3-row2 vs M=1 shape divergence itself
        const bool cgc_tail_split = getenv("CGC_SIMP_TAIL_SPLIT") != nullptr;
        int32_t i_last_chunk = 0;
        int32_t n_eval_last = 0;
        for (int32_t i = 0; i < (int32_t) prompt_tokens.size(); i += n_batch_eff) {
            i_last_chunk = i;
            const int32_t n_eval = std::min<int32_t>(n_batch_eff, (int32_t) prompt_tokens.size() - i);
            const bool is_last_chunk = i + n_eval == (int32_t) prompt_tokens.size();
            if (cgc_tail_split && is_last_chunk && n_eval >= 2) {
                // sub-batch A: [i .. i+n_eval-2] with logits on its last row (mirrors spec's
                // prefill tail: logits only on the final prefill token)
                {
                    const int32_t na = n_eval - 1;
                    llama_batch pb = llama_batch_init(na, 0, 1);
                    for (int32_t j = 0; j < na; ++j) {
                        pb.token[j]   = prompt_tokens[i + j];
                        pb.pos[j]     = i + j;
                        pb.n_seq_id[j] = 1;
                        pb.seq_id[j][0] = 0;
                        pb.logits[j]  = false;
                    }
                    pb.logits[na - 1] = true;
                    pb.n_tokens = na;
                    const int rc = llama_decode(ctx, pb);
                    llama_batch_free(pb);
                    if (rc) {
                        fprintf(stderr, "%s : failed to eval prompt tail-split A\n", __func__);
                        ret = 1;
                        cgc_cleanup();
                        return ret;
                    }
                    if (getenv("CGC_LAYER_BIN")) {
                        fprintf(stderr, "SIMP: tail-split A pos=%d n_eval=%d\n", i, na);
                        cgc_dump_layer_inp(ctx, model, "simpp", na);
                        // [CGC v4 per-chunk] chunk idx 3 == spec's prefill chunk 3 (both M=2
                        // [24,25]) — sc3 vs pc3 is the per-chunk cross-path comparison.
                        char tag[8];
                        snprintf(tag, sizeof(tag), "sc%d", i / n_batch_eff);
                        cgc_dump_layer_inp(ctx, model, tag, na);
                    }
                }
                // sub-batch B: the last prompt token alone (M=1, n_past = i+n_eval-1)
                {
                    const int32_t ib = i + n_eval - 1;
                    llama_batch pb = llama_batch_init(1, 0, 1);
                    pb.token[0]   = prompt_tokens[ib];
                    pb.pos[0]     = ib;
                    pb.n_seq_id[0] = 1;
                    pb.seq_id[0][0] = 0;
                    pb.logits[0]  = true;
                    pb.n_tokens = 1;
                    const int rc = llama_decode(ctx, pb);
                    llama_batch_free(pb);
                    if (rc) {
                        fprintf(stderr, "%s : failed to eval prompt tail-split B\n", __func__);
                        ret = 1;
                        cgc_cleanup();
                        return ret;
                    }
                    n_eval_last = 1;
                    if (getenv("CGC_LAYER_BIN")) {
                        fprintf(stderr, "SIMP: tail-split B (M=1) pos=%d n_eval=1\n", ib);
                        cgc_dump_layer_inp(ctx, model, "simp", 1);
                    }
                }
                continue;
            }
            n_eval_last = n_eval;
            // explicit positions: chunk j sits at KV pos i+j (llama_batch_get_one restarts at 0)
            llama_batch pb = llama_batch_init(n_eval, 0, 1);
            for (int32_t j = 0; j < n_eval; ++j) {
                pb.token[j]   = prompt_tokens[i + j];
                pb.pos[j]     = i + j;
                pb.n_seq_id[j] = 1;
                pb.seq_id[j][0] = 0;
                pb.logits[j]  = false;
            }
            // logits for the chunk's last token (LLAMA_SIMPLE_LOGITS_DBG reads row n_eval-1)
            pb.logits[n_eval - 1] = true;
            pb.n_tokens = n_eval;
            const int rc = llama_decode(ctx, pb);
            llama_batch_free(pb);
            if (rc) {
                fprintf(stderr, "%s : failed to eval prompt chunk %d\n", __func__, i / n_batch_eff);
                ret = 1;
                cgc_cleanup();
                return ret;
            }
            // [CGC v4 per-chunk] dump EVERY prefill chunk (not just the last): tag sc<idx>
            // mirrors spec's pc<idx> — chunks 0..2 are M=8 in BOTH paths, so a divergence
            // found in an early chunk means the tail-chunk comparison was looking at a
            // symptom, not the cause.
            if (getenv("CGC_LAYER_BIN")) {
                char tag[8];
                snprintf(tag, sizeof(tag), "sc%d", i / n_batch_eff);
                fprintf(stderr, "SIMP: chunk %d pos=%d n_eval=%d\n", i / n_batch_eff, i, n_eval);
                cgc_dump_layer_inp(ctx, model, tag, n_eval);
            }
        }
        // [CGC layer bisect] dump the LAST prefill chunk's per-layer inputs.
        // row j = prompt position (i_last_chunk + j); the last row is the last
        // prompt token — the token the speculative path instead computes inside
        // its first verify batch [id_last, d0, d1]. The buffer still holds this
        // chunk's data because no decode has run since.
        if (getenv("CGC_LAYER_BIN") && n_eval_last > 0) {
            fprintf(stderr, "SIMP: last prefill chunk pos=%d n_eval=%d\n", i_last_chunk, n_eval_last);
            cgc_dump_layer_inp(ctx, model, "simp", n_eval_last);
        }
        // debug: dump top-5 logits after the last (prefill) chunk. The logits buffer only holds
        // the LAST chunk's rows, so the row id must be relative to i_last_chunk (llama_get_logits_ith
        // aborts on out-of-range ids — fixed 2026-08-17).
        if (getenv("LLAMA_SIMPLE_LOGITS_DBG") != nullptr) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            const int logits_row = n_prompt - 1 - i_last_chunk;
            const float * logits = llama_get_logits_ith(ctx, logits_row);
            int best[5] = {0,0,0,0,0};
            float bv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for (int v = 0; v < n_vocab; ++v) {
                for (int k = 0; k < 5; ++k) {
                    if (logits[v] > bv[k]) {
                        for (int j = 4; j > k; --j) { bv[j] = bv[j-1]; best[j] = best[j-1]; }
                        bv[k] = logits[v]; best[k] = v;
                        break;
                    }
                }
            }
            fprintf(stderr, "LOGITS top5: %d=%.4f %d=%.4f %d=%.4f %d=%.4f %d=%.4f\n",
                best[0], bv[0], best[1], bv[1], best[2], bv[2], best[3], bv[3], best[4], bv[4]);
            fprintf(stderr, "LOGITS first16: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                logits[0], logits[1], logits[2], logits[3], logits[4], logits[5], logits[6], logits[7],
                logits[8], logits[9], logits[10], logits[11], logits[12], logits[13], logits[14], logits[15]);
            const float * emb = llama_get_embeddings_ith(ctx, logits_row);
            fprintf(stderr, "EMBED postnorm[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                emb ? (double) emb[0] : 0, emb ? (double) emb[1] : 0, emb ? (double) emb[2] : 0, emb ? (double) emb[3] : 0,
                emb ? (double) emb[4] : 0, emb ? (double) emb[5] : 0, emb ? (double) emb[6] : 0, emb ? (double) emb[7] : 0);
        }
        // generation starts from the last prompt token
        new_token_id = prompt_tokens.back();
        batch = llama_batch_get_one(&new_token_id, 1);
        n_pos = n_prompt;
    }

    // [CGC 2026-08-29 bit-identical fix] decoder-only models: the chunked prefill above
    // already decoded ALL prompt tokens (logits on the last chunk's last row), so the
    // first sample must read those logits directly. The old code re-decoded
    // prompt_tokens.back() in the loop, which auto-positioned at n_prompt and
    // DUPLICATED the last prompt token in the KV — the first sample was conditioned
    // on [prompt + dup(last tok)] instead of [prompt], shifting every later position
    // +1 and diverging from the speculative path (verified via CGC_LOGIT_DBG:
    // SIMP pos=28 vs SPEC pos=27 with structurally different logits).
    bool first_sample_prefill = !llama_model_has_encoder(model);

    // CGC: per-token wall timer (decode + sample cycle). CGC_STEP_TIMING=1 prints mean/p50/p90/p99.
    const bool cgc_step_timing = getenv("CGC_STEP_TIMING") != nullptr;
    std::vector<double> cgc_step_ms, cgc_decode_ms, cgc_sample_ms;

    for (; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        const int64_t t_step0 = ggml_time_us();
        // evaluate the current batch with the transformer model
        if (first_sample_prefill) {
            // first sample after prefill: logits already computed (last chunk's
            // last row) — no decode, no n_pos advance (see comment above).
            first_sample_prefill = false;
        } else if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            ret = 1;
            cgc_cleanup();
            return ret;
        } else {
            n_pos += batch.n_tokens;
        }
        const int64_t t_decode_end = ggml_time_us();

        // CGC_LOGIT_DBG: dump the distribution the sampler actually consumes this step
        // (top-2 with hex floats — ULP-level diff vs the speculative path).
        // CGC_LOGIT_BIN=1 additionally writes the full logit row + result_norm to
        // /tmp/cgc_simp_pos<N>{_lg,_nm}.f32 for offline divergence analysis
        // (constant-shift vs per-element noise vs structural).
        if (getenv("CGC_LOGIT_DBG")) {
            const int n_vocab_dbg = llama_vocab_n_tokens(vocab);
            const float * lg = llama_get_logits_ith(ctx, -1);
            int b0 = 0, b1 = 0; float v0 = -1e30f, v1 = -1e30f;
            for (int v = 0; v < n_vocab_dbg; ++v) {
                if (lg[v] > v0) { v1 = v0; b1 = b0; v0 = lg[v]; b0 = v; }
                else if (lg[v] > v1) { v1 = lg[v]; b1 = v; }
            }
            fprintf(stderr, "SIMP pos=%d top2: %d=%a %d=%a\n", n_pos, b0, v0, b1, v1);
            if (getenv("CGC_LOGIT_BIN")) {
                char path[128];
                snprintf(path, sizeof(path), "/tmp/cgc_simp_pos%d_lg.f32", n_pos);
                FILE * f = fopen(path, "wb");
                if (f) { fwrite(lg, sizeof(float), n_vocab_dbg, f); fclose(f); }
                const int n_embd_dbg = llama_model_n_embd(model);
                const float * nm = llama_get_embeddings_ith(ctx, -1);
                if (nm) {
                    snprintf(path, sizeof(path), "/tmp/cgc_simp_pos%d_nm.f32", n_pos);
                    f = fopen(path, "wb");
                    if (f) { fwrite(nm, sizeof(float), n_embd_dbg, f); fclose(f); }
                }
            }
        }

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                if (!ignore_eos) {
                    break;
                }
                // --ignore-eos: keep generating past EOG; the EOG token piece is still printed
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }

        if (cgc_step_timing) {
            cgc_step_ms.push_back((ggml_time_us() - t_step0) / 1000.0);
            cgc_decode_ms.push_back((t_decode_end - t_step0) / 1000.0);
            cgc_sample_ms.push_back((ggml_time_us() - t_decode_end) / 1000.0);
        }
    }

    printf("\n");

    if (cgc_step_timing && !cgc_step_ms.empty()) {
        auto pct = [](std::vector<double> & v, double p) {
            std::sort(v.begin(), v.end());
            return v[std::min(v.size() - 1, (size_t)(p * v.size()))];
        };
        double d_sum = 0, s_sum = 0;
        for (double v : cgc_decode_ms) d_sum += v;
        for (double v : cgc_sample_ms) s_sum += v;
        fprintf(stderr, "CGC-STEP: n=%zu mean=%.3f p50=%.3f p90=%.3f p99=%.3f ms/step | decode=%.3f sample=%.3f\n",
                cgc_step_ms.size(), d_sum / cgc_step_ms.size(), pct(cgc_step_ms, 0.50), pct(cgc_step_ms, 0.90), pct(cgc_step_ms, 0.99),
                d_sum / cgc_step_ms.size(), s_sum / cgc_step_ms.size());
    }

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    ret = 0;
    cgc_cleanup();

    return ret;
}
