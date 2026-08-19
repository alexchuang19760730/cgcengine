#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

    ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        ret = 1;
        cgc_cleanup();
        return ret;
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
        int32_t i_last_chunk = 0;
        for (int32_t i = 0; i < (int32_t) prompt_tokens.size(); i += n_batch_eff) {
            i_last_chunk = i;
            const int32_t n_eval = std::min<int32_t>(n_batch_eff, (int32_t) prompt_tokens.size() - i);
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

    for (; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            ret = 1;
            cgc_cleanup();
            return ret;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
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
    }

    printf("\n");

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
