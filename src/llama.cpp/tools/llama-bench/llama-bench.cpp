#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iterator>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unordered_set>

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "download.h"
#include "fit.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
// [CGC MTP instrument 2026-09-17] The speculative path for the MTP instrument (see
// test_gen_spec below). llama-bench already links llama-common ("common.h" above is the
// CGC-fork's common params), so these two headers are the whole wiring requirement.
//
// 2026-09-17 (later): the instrument is selected by `--spec-type draft-mtp`, a normal CLI option,
// not by the LLAMA_BENCH_SPEC environment variable. The MTP head is an OPTIONAL block inside ONE
// GGUF, and `--spec-type` is already the knob the server exposes for it (run_server.sh), so
// "which configuration am I measuring" must not have to travel through an environment variable
// that only makes sense for a particular file. LLAMA_BENCH_SPEC is kept as a DEPRECATED alias
// (it maps onto the flag and says so on stderr) so that existing drivers keep running.
#include "sampling.h"
#include "speculative.h"
// [CGC MTP instrument 2026-09-18] llama_context_set_cgc_phase — the caller-set phase marker the
// fast path is gated on (see the verify decode in test_gen_spec). Same include, same reason, as
// tools/server/server-context.cpp:19.
#include "../../src/llama-ext.h" // staging API: llama_context_set_cgc_phase (CGC Phase Discrimination)

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

// utils
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

static bool tensor_buft_override_equal(const llama_model_tensor_buft_override& a, const llama_model_tensor_buft_override& b) {
    if (a.pattern != b.pattern) {
        // cString comparison that may be null
        if (a.pattern == nullptr || b.pattern == nullptr) {
            return false;
        }
        if (strcmp(a.pattern, b.pattern) != 0) {
            return false;
        }
    }
    if (a.buft != b.buft) {
        return false;
    }
    return true;
}

static bool vec_tensor_buft_override_equal(const std::vector<llama_model_tensor_buft_override>& a, const std::vector<llama_model_tensor_buft_override>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!tensor_buft_override_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

static bool vec_vec_tensor_buft_override_equal(const std::vector<std::vector<llama_model_tensor_buft_override>>& a, const std::vector<std::vector<llama_model_tensor_buft_override>>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!vec_tensor_buft_override_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

template <class T> static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

template <typename T, typename F> static std::vector<std::string> transform_to_str(const std::vector<T> & values, F f) {
    std::vector<std::string> str_values;
    std::transform(values.begin(), values.end(), std::back_inserter(str_values), f);
    return str_values;
}

template <typename T> static T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T) v.size();
}

template <typename T> static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean   = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev  = std::sqrt(sq_sum / (T) (v.size() - 1) - mean * mean * (T) v.size() / (T) (v.size() - 1));
    return stdev;
}

static std::string get_cpu_info() {
    std::vector<std::string> cpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU || dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            cpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(cpu_list, ", ");
}

static std::string get_gpu_info() {
    std::vector<std::string> gpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(gpu_list, ", ");
}

static std::vector<ggml_backend_dev_t> parse_devices_arg(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    std::string                     trimmed = string_strip(value);
    if (trimmed.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (trimmed == "auto") {
        return devices;
    }

    auto dev_names = string_split<std::string>(trimmed, '/');
    if (dev_names.size() == 1 && string_strip(dev_names[0]) == "none") {
        devices.push_back(nullptr);
        return devices;
    }

    for (auto & name : dev_names) {
        std::string dev_name = string_strip(name);
        if (dev_name.empty()) {
            throw std::invalid_argument("invalid device specification");
        }
        auto * dev = ggml_backend_dev_by_name(dev_name.c_str());
        if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            throw std::invalid_argument(string_format("invalid device: %s", dev_name.c_str()));
        }
        devices.push_back(dev);
    }

    devices.push_back(nullptr);
    return devices;
}

static void register_rpc_server_list(const std::string & servers) {
    auto rpc_servers = string_split<std::string>(servers, ',');
    if (rpc_servers.empty()) {
        throw std::invalid_argument("no RPC servers specified");
    }

    auto * rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        throw std::invalid_argument("failed to find RPC backend");
    }

    using add_rpc_server_fn = ggml_backend_reg_t (*)(const char * endpoint);
    auto * ggml_backend_rpc_add_server_fn = (add_rpc_server_fn) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!ggml_backend_rpc_add_server_fn) {
        throw std::invalid_argument("failed to find RPC add server function");
    }
    for (const auto & server : rpc_servers) {
        auto reg = ggml_backend_rpc_add_server_fn(server.c_str());
        ggml_backend_register(reg);
    }
}

static std::string devices_to_string(const std::vector<ggml_backend_dev_t> & devices) {
    if (devices.empty()) {
        return "auto";
    }

    if (devices.size() == 1 && devices[0] == nullptr) {
        return "none";
    }

    std::vector<std::string> names;
    for (auto * dev : devices) {
        if (dev == nullptr) {
            break;
        }
        names.push_back(ggml_backend_dev_name(dev));
    }

    return join(names, "/");
}

// command line params
enum output_formats { NONE, CSV, JSON, JSONL, MARKDOWN, SQL };

static const char * output_format_str(output_formats format) {
    switch (format) {
        case NONE:
            return "none";
        case CSV:
            return "csv";
        case JSON:
            return "json";
        case JSONL:
            return "jsonl";
        case MARKDOWN:
            return "md";
        case SQL:
            return "sql";
        default:
            GGML_ABORT("invalid output format");
    }
}

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "none") {
        format = NONE;
    } else if (s == "csv") {
        format = CSV;
    } else if (s == "json") {
        format = JSON;
    } else if (s == "jsonl") {
        format = JSONL;
    } else if (s == "md") {
        format = MARKDOWN;
    } else if (s == "sql") {
        format = SQL;
    } else {
        return false;
    }
    return true;
}

static const char * split_mode_str(llama_split_mode mode) {
    switch (mode) {
        case LLAMA_SPLIT_MODE_NONE:
            return "none";
        case LLAMA_SPLIT_MODE_LAYER:
            return "layer";
        case LLAMA_SPLIT_MODE_ROW:
            return "row";
        case LLAMA_SPLIT_MODE_TENSOR:
            return "tensor";
        default:
            GGML_ABORT("invalid split mode");
    }
}

static std::string pair_str(const std::pair<int, int> & p) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d,%d", p.first, p.second);
    return buf;
}

static std::vector<int> parse_int_range(const std::string & s, bool allow_negative = false) {
    // first[-last[(+|*)step]]
    std::regex range_regex(allow_negative
        ? R"(^(-?\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))"
        : R"(^(\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))");

    std::smatch match;
    std::string::const_iterator search_start(s.cbegin());
    std::vector<int> result;
    while (std::regex_search(search_start, s.cend(), match, range_regex)) {
        int  first = std::stoi(match[1]);
        int  last  = match[2].matched ? std::stoi(match[2]) : first;
        char op    = match[3].matched ? match[3].str()[0] : '+';
        int  step  = match[4].matched ? std::stoi(match[4]) : 1;

        for (int i = first; i <= last;) {
            result.push_back(i);

            int prev_i = i;

            if (op == '+') {
                i += step;
            } else if (op == '*') {
                i *= step;
            } else {
                throw std::invalid_argument("invalid range format");
            }

            if (i <= prev_i) {
                throw std::invalid_argument("invalid range");
            }
        }
        search_start = match.suffix().first;
    }

    if (search_start != s.cend()) {
        throw std::invalid_argument("invalid range format");
    }

    return result;
}

struct cmd_params {
    std::vector<std::string>         model;
    std::vector<std::string>         hf_repo;
    std::vector<std::string>         hf_file;
    std::string                      hf_token;
    bool                             offline;
    std::vector<int>                 n_prompt;
    std::vector<int>                 n_gen;
    std::vector<std::pair<int, int>> n_pg;
    std::vector<int>                 n_depth;
    std::vector<int>                 n_batch;
    std::vector<int>                 n_ubatch;
    std::vector<ggml_type>           type_k;
    std::vector<ggml_type>           type_v;
    std::vector<int>                 n_threads;
    std::vector<std::string>         cpu_mask;
    std::vector<bool>                cpu_strict;
    std::vector<int>                 poll;
    std::vector<int>                 n_gpu_layers;
    std::vector<int>                 n_cpu_moe;
    std::vector<size_t>              expert_cache_bytes;
    std::vector<llama_split_mode>    split_mode;
    std::vector<llama_load_mode>     load_mode;
    std::vector<int>                 main_gpu;
    std::vector<bool>                no_kv_offload;
    std::vector<llama_flash_attn_type> flash_attn;
    std::vector<std::vector<ggml_backend_dev_t>> devices;
    std::vector<std::vector<float>>  tensor_split;
    std::vector<std::vector<llama_model_tensor_buft_override>> tensor_buft_overrides;
    std::vector<bool>                embeddings;
    std::vector<bool>                no_op_offload;
    std::vector<bool>                no_host;
    std::vector<size_t>              fit_params_target;
    std::vector<uint32_t>            fit_params_min_ctx;
    ggml_numa_strategy               numa;
    int                              reps;
    ggml_sched_priority              prio;
    int                              delay;
    bool                             verbose;
    bool                             progress;
    bool                             no_warmup;
    output_formats                   output_format;
    output_formats                   output_format_stderr;
    // [CGC MTP instrument 2026-09-17] Speculative decoding for the gen cell. `spec_type` is the CLI
    // surface (--spec-type, same vocabulary as the server); `spec_types` is the resolved form and is
    // what the run actually consults. Empty `spec_types` = plain decode cell, i.e. the shape that
    // stays comparable with upstream's tg row.
    std::vector<std::string>             spec_type;
    int                                  spec_draft_n_max;
    std::vector<common_speculative_type> spec_types;
    // [CGC 2026-09-19] --prompt-file: fill prompt/depth with REAL text instead of std::rand()%n_vocab.
    std::string                          prompt_file;
};

static const cmd_params cmd_params_defaults = {
    /* model                */ { "models/7B/ggml-model-q4_0.gguf" },
    /* hf_repo              */ {},
    /* hf_file              */ {},
    /* hf_token             */ "",
    /* offline              */ false,
    /* n_prompt             */ { 512 },
    /* n_gen                */ { 128 },
    /* n_pg                 */ {},
    /* n_depth              */ { 0 },
    /* n_batch              */ { 2048 },
    /* n_ubatch             */ { 512 },
    /* type_k               */ { GGML_TYPE_F16 },
    /* type_v               */ { GGML_TYPE_F16 },
    /* n_threads            */ { common_cpu_get_num_math() },
    /* cpu_mask             */ { "0x0" },
    /* cpu_strict           */ { false },
    /* poll                 */ { 50 },
    /* n_gpu_layers         */ { -1 },
    /* n_cpu_moe            */ { 0 },
    /* expert_cache_bytes   */ { 0 },
    /* split_mode           */ { LLAMA_SPLIT_MODE_LAYER },
    /* load_mode            */ { LLAMA_LOAD_MODE_AUTO },
    /* main_gpu             */ { 0 },
    /* no_kv_offload        */ { false },
    /* flash_attn           */ { LLAMA_FLASH_ATTN_TYPE_AUTO },
    /* devices              */ { {} },
    /* tensor_split         */ { std::vector<float>(llama_max_devices(), 0.0f) },
    /* tensor_buft_overrides*/ { std::vector<llama_model_tensor_buft_override>{ { nullptr, nullptr } } },
    /* embeddings           */ { false },
    /* no_op_offload        */ { false },
    /* no_host              */ { false },
    /* fit_params_target    */ { 0 },
    /* fit_params_min_ctx   */ { 0 },
    /* numa                 */ GGML_NUMA_STRATEGY_DISABLED,
    /* reps                 */ 5,
    /* prio                 */ GGML_SCHED_PRIO_NORMAL,
    /* delay                */ 0,
    /* verbose              */ false,
    /* progress             */ false,
    /* no_warmup            */ false,
    /* output_format        */ MARKDOWN,
    /* output_format_stderr */ NONE,
    /* spec_type            */ {},
    /* spec_draft_n_max     */ 3,
    /* spec_types           */ {},
    /* prompt_file          */ "",
};

// [CGC 2026-09-19] --prompt-file support for test_prompt().
//
// WHY. `llama-bench` has always filled its prompt AND its depth with `std::rand() % n_vocab`,
// i.e. a uniform draw over the whole vocabulary. That is not what the served path sees: the
// server's prompt is real prose, whose MoE routing is far from uniform. The measured consequence
// (2026-09-18, docs/BENCH_HTTP_PARITY_2026-09-18.md and
// docs/MTP_ROUND_COST_OPT_BACKLOG_2026-09-19.md) is that the uniform draw moves TWO things at
// once, and both in the direction that made llama-bench disagree with HTTP:
//   - cache requests/round 200.3 vs the server's 116.5 (routing spread over ~256 experts instead of
//     a prose-shaped working set), which is 76% of the per-round cost;
//   - draft acceptance 0.8012 vs the server's 0.4654 -- a garbage prompt makes the model's
//     distribution very peaked, so draft and target agree on the top-1 far more often. mean len is
//     1 + n_max*p, so 1+3(0.8012)=3.404 vs 1+3(0.4654)=2.396, i.e. the same cause.
// Feeding the same real text the server gets should therefore remove BOTH halves and let
// llama-bench land on the HTTP number (predicted 2.40/0.185 = 12.97 vs HTTP 12.99).
//
// The tokens are cycled (idx % n) so a short unit can fill an arbitrary --n-depth, which is exactly
// what http_duo.py does when it repeats _PREFILL_UNIT up to the profile's ctx.
static std::string              g_prompt_file;
static std::vector<llama_token> g_prompt_tokens;
static const llama_vocab *      g_prompt_vocab = nullptr;

// [CGC 2026-09-19] --fixed-fill-seed N: reseed std::rand() at the top of EVERY rep.
//
// WHY. `llama-bench` never calls srand(), so the C library's seed is 1 once per PROCESS, not once
// per rep. The rand stream therefore keeps advancing across reps: rep 1, 2, 3, 4 each draw a
// DIFFERENT random depth and a different decode token stream. Since the expert cache lives on the
// model (it is NOT cleared by llama_memory_clear), every rep flushes the pool and re-pays the
// compulsory misses -- llama-bench can never reach steady state, while the HTTP server (same
// prompt, temp 0) converges after rep 1. That is the whole of the bench-vs-http gap.
// 0 keeps the historical behaviour bit-for-bit.
static int g_fixed_fill_seed = 0;

// [CGC 2026-09-19] --warm-skip N: run N generated tokens that are NOT timed.
//
// WHY. A decode t/s from this tool is a window average and the window is whatever `-n` says; the
// plateau is not reached at the start of a generation. Measured on this box with the house decode
// shape: `-d 512 -n 128` reads 7.96 t/s while `-d 512 -n 512` reads 10.22 -- same engine, same pool,
// same everything except how many tokens the average covers. The plateau then had to be recovered
// by hand from the per-rep `samples_ts`, which is not something a reader of this binary's JSON can
// do. This flag puts the window in the command line and makes the reported `n_gen` state the number
// of tokens the figure is actually about.
//
// It is NOT `platform_ts`: that drops a whole REP (a different question, "the first rep is cold"),
// while this drops the first N tokens of EVERY rep.
//
// 0 (default) is bit-identical to the previous behaviour: no extra generation call, the timed
// interval is untouched, `n_gen` is untouched.
static int g_warm_skip = 0;

// [CGC 2026-09-19] -c, --ctx-size N: override the derived n_ctx (n_prompt + n_gen + n_depth).
//
// WHY. That derived value is the only ctx this tool ever ran at, and it is much smaller than the
// server's: the house decode arm `-d 512 -n 192` derives **704** while prod25 serves at **-c 4096**.
// n_ctx is not cosmetic -- it sizes the KV allocation and it drives the engine's own batch clamping
// (`CGC-PHASE-SPLIT: L4 pool capacity=179 -> n_batch 2048 capped to 8`). So every bench-vs-HTTP
// comparison in this project has been taken at two different ctx values, and ctx was never tested
// because the tool could not set it. 0 = the historical derived value.
static int g_ctx_override = 0;

// [CGC 2026-09-19] Env fallbacks for the two window/carrier knobs.
//
// WHY. `scripts/check/paired_ab.py` drives an A/B through `--a-env`/`--b-env`, i.e. environment
// variables -- it has no way to differ two arms by CLI flag. Without these, `--warm-skip` and
// `--ctx-size` cannot be tested by this project's own paired design (AB/BA + median + `--null`
// noise floor), which is the only design that has survived this box's across-launch spread.
// A CLI flag still WINS over the env when both are given.
static int env_int_or(const char * name, int fallback) {
    const char * v = getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    return std::atoi(v);
}

static int warm_skip_value() { return g_warm_skip > 0 ? g_warm_skip : env_int_or("CGC_BENCH_WARM_SKIP", 0); }
static int ctx_size_value()  { return g_ctx_override > 0 ? g_ctx_override : env_int_or("CGC_BENCH_CTX", 0); }

// Returns token `idx` of the real text, or a uniform random token when --prompt-file was not given
// (or could not be read). Bit-identical to the previous behaviour in the fallback case.
static llama_token bench_fill_token(const llama_vocab * vocab, int32_t n_vocab, int idx) {
    if (g_prompt_file.empty()) {
        return std::rand() % n_vocab;
    }
    if (g_prompt_vocab != vocab) {
        g_prompt_tokens.clear();
        g_prompt_vocab = vocab;

        FILE * f = fopen(g_prompt_file.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "llama-bench: --prompt-file '%s' could not be opened; using random fill\n",
                    g_prompt_file.c_str());
            g_prompt_file.clear();
            return std::rand() % n_vocab;
        }
        fseek(f, 0, SEEK_END);
        const long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string text;
        if (sz > 0) {
            text.resize((size_t) sz);
            const size_t got = fread(&text[0], 1, (size_t) sz, f);
            text.resize(got);
        }
        fclose(f);

        g_prompt_tokens.resize(text.size() + 64);
        int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), g_prompt_tokens.data(),
                                   (int32_t) g_prompt_tokens.size(), true, false);
        if (n < 0) { // buffer too small: llama_tokenize returns -(required)
            g_prompt_tokens.resize((size_t) -n);
            n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), g_prompt_tokens.data(),
                               (int32_t) g_prompt_tokens.size(), true, false);
        }
        if (n <= 0) {
            fprintf(stderr, "llama-bench: --prompt-file '%s' tokenized to %d tokens; using random fill\n",
                    g_prompt_file.c_str(), (int) n);
            g_prompt_file.clear();
            g_prompt_tokens.clear();
            return std::rand() % n_vocab;
        }
        g_prompt_tokens.resize((size_t) n);
        fprintf(stderr, "llama-bench: --prompt-file '%s' -> %d real tokens (cycled to fill prompt/depth)\n",
                g_prompt_file.c_str(), (int) n);
    }
    if (g_prompt_tokens.empty()) {
        return std::rand() % n_vocab;
    }
    return g_prompt_tokens[(size_t) idx % g_prompt_tokens.size()];
}

// [CGC 2026-09-19 n-gram control arm] Self-speculation from the token history: it drafts WITHOUT any
// draft-model forward. That is the whole reason it is here -- it is the only arm that can answer
// "is the ~46-59 ms per draft token the DRAFT FORWARD, or the verify path that the extra token
// brings with it?" (docs/POOL_BUDGET_COST_DECOMP_2026-09-18.md §6.1b). Neither k nor the pool size
// can separate those two, because both scale together with k.
//
// ONE definition, three consumers (type validation, spec setup, the measured loop), for the same
// reason the phase predicate has one: a second copy of this list is how the arms stop agreeing on
// what they measured.
static bool is_ngram_spec_type(common_speculative_type t) {
    switch (t) {
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
            return true;
        default:
            return false;
    }
}

static bool is_ngram_spec_arm(const std::vector<common_speculative_type> & types) {
    return !types.empty() && std::all_of(types.begin(), types.end(), is_ngram_spec_type);
}

static void print_usage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  --numa <distribute|isolate|numactl>         numa mode (default: disabled)\n");
    printf("  -r, --repetitions <n>                       number of times to repeat each test (default: %d)\n", cmd_params_defaults.reps);
    printf("  --prio <-1|0|1|2|3>                         process/thread priority (default: %d)\n", cmd_params_defaults.prio);
    printf("  --delay <0...N> (seconds)                   delay between each test (default: %d)\n", cmd_params_defaults.delay);
    printf("  -o, --output <csv|json|jsonl|md|sql>        output format printed to stdout (default: %s)\n", output_format_str(cmd_params_defaults.output_format));
    printf("  -oe, --output-err <csv|json|jsonl|md|sql>   output format printed to stderr (default: %s)\n", output_format_str(cmd_params_defaults.output_format_stderr));
    printf("  --list-devices                              list available devices and exit\n");
    printf("  -v, --verbose                               verbose output\n");
    printf("  --progress                                  print test progress indicators\n");
    printf("  --no-warmup                                 skip warmup runs before benchmarking\n");
    printf("  -fitt, --fit-target <MiB>                   fit model to device memory with this margin per device in MiB (default: off)\n");
    printf("  -fitc, --fit-ctx <n>                        minimum ctx size for --fit-target (default: 4096)\n");
    if (llama_supports_rpc()) {
        printf("  -rpc, --rpc <rpc_servers>                   register RPC devices (comma separated)\n");
    }
    printf("\n");
    printf("test parameters:\n");
    printf("  -m, --model <filename>                            (default: %s)\n", join(cmd_params_defaults.model, ",").c_str());
    printf("  -hf, -hfr, --hf-repo <user>/<model>[:quant]       Hugging Face model repository; quant is optional, case-insensitive\n");
    printf("                                                    default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.\n");
    printf("                                                    example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n");
    printf("                                                    (default: unused)\n");
    printf("  -hff, --hf-file <file>                            Hugging Face model file. If specified, it will override the quant in --hf-repo\n");
    printf("                                                    (default: unused)\n");
    printf("  -hft, --hf-token <token>                          Hugging Face access token\n");
    printf("                                                    (default: value from HF_TOKEN environment variable)\n");
    printf("  --offline                                         Offline mode: forces use of cache, prevents network access\n");
    printf("                                                    (default: disabled)\n");
    printf("  -p, --n-prompt <n>                                (default: %s)\n", join(cmd_params_defaults.n_prompt, ",").c_str());
    printf("  -n, --n-gen <n>                                   (default: %s)\n", join(cmd_params_defaults.n_gen, ",").c_str());
    printf("  -pg <pp,tg>                                       (default: %s)\n", join(transform_to_str(cmd_params_defaults.n_pg, pair_str), ",").c_str());
    printf("  -d, --n-depth <n>                                 (default: %s)\n", join(cmd_params_defaults.n_depth, ",").c_str());
    printf("  -b, --batch-size <n>                              (default: %s)\n", join(cmd_params_defaults.n_batch, ",").c_str());
    printf("  -ub, --ubatch-size <n>                            (default: %s)\n", join(cmd_params_defaults.n_ubatch, ",").c_str());
    printf("  -ctk, --cache-type-k <t>                          (default: %s)\n", join(transform_to_str(cmd_params_defaults.type_k, ggml_type_name), ",").c_str());
    printf("  -ctv, --cache-type-v <t>                          (default: %s)\n", join(transform_to_str(cmd_params_defaults.type_v, ggml_type_name), ",").c_str());
    printf("  -t, --threads <n>                                 (default: %s)\n", join(cmd_params_defaults.n_threads, ",").c_str());
    printf("  -C, --cpu-mask <hex,hex>                          (default: %s)\n", join(cmd_params_defaults.cpu_mask, ",").c_str());
    printf("  --cpu-strict <0|1>                                (default: %s)\n", join(cmd_params_defaults.cpu_strict, ",").c_str());
    printf("  --poll <0...100>                                  (default: %s)\n", join(cmd_params_defaults.poll, ",").c_str());
    printf("  -ngl, --n-gpu-layers <n>                          (default: %s)\n", join(cmd_params_defaults.n_gpu_layers, ",").c_str());
    printf("  -ncmoe, --n-cpu-moe <n>                           (default: %s)\n", join(cmd_params_defaults.n_cpu_moe, ",").c_str());
  printf("  -expert-cache, --expert-cache <bytes>                (default: %s, 0 = off)\n", join(cmd_params_defaults.expert_cache_bytes, ",").c_str());
    printf("  -sm, --split-mode <none|layer|row|tensor>         (default: %s)\n", join(transform_to_str(cmd_params_defaults.split_mode, split_mode_str), ",").c_str());
    printf("  -mg, --main-gpu <i>                               (default: %s)\n", join(cmd_params_defaults.main_gpu, ",").c_str());
    printf("  -nkvo, --no-kv-offload <0|1>                      (default: %s)\n", join(cmd_params_defaults.no_kv_offload, ",").c_str());
    printf("  -fa, --flash-attn <on|off|auto>                   (default: %s)\n", join(transform_to_str(cmd_params_defaults.flash_attn, llama_flash_attn_type_name), ",").c_str());
    printf("  -dev, --device <dev0/dev1/...>                    (default: auto)\n");
    printf("  -lm, --load-mode <auto|none|mmap|mlock|mmap+mlock|dio> (default: %s)\n", join(transform_to_str(cmd_params_defaults.load_mode, llama_load_mode_name), ",").c_str());
    printf("  -mmp, --mmap <0|1>                                (DEPRECATED IN FAVOUR OF --load-mode)\n");
    printf("  -dio, --direct-io <0|1>                           (DEPRECATED IN FAVOUR OF --load-mode)\n");
    printf("  -embd, --embeddings <0|1>                         (default: %s)\n", join(cmd_params_defaults.embeddings, ",").c_str());
    printf("  -ts, --tensor-split <ts0/ts1/..>                  (default: 0)\n");
    printf("  -ot --override-tensor <tensor name pattern>=<buffer type>;...\n");
    printf("                                                    (default: disabled)\n");
    printf("  -nopo, --no-op-offload <0|1>                      (default: 0)\n");
    printf("  --no-host <0|1>                                   (default: %s)\n", join(cmd_params_defaults.no_host, ",").c_str());
    printf("  --spec-type <type>                                speculative decoding for the gen cell. Same names as the server;\n");
    printf("                                                    only 'draft-mtp' is implemented here, and it uses the TARGET\n");
    printf("                                                    model's own in-file MTP block (self-referential draft: no\n");
    printf("                                                    separate draft model file).\n");
    printf("                                                    known types: %s\n", common_speculative_all_types_str());
    printf("                                                    (default: none -- plain decode cell)\n");
    printf("  --spec-draft-n-max <n>                            max draft tokens for --spec-type draft-mtp, 1..16 (default: %d)\n", cmd_params_defaults.spec_draft_n_max);
    printf("                                                    (without --spec-type this is inert)\n");
    printf("  --prompt-file <path>                              fill the prompt/depth with REAL text read from\n");
    printf("                                                    <path> (cycled) instead of std::rand()%%n_vocab.\n");
    printf("                                                    The random fill is NOT what the server sees, and it\n");
    printf("                                                    moves both cache requests/round and draft acceptance.\n");
    printf("  --fixed-fill-seed <n>                             reseed std::rand() with <n> at the top of EVERY\n");
    printf("                                                    rep, so all reps see the same fill stream (and the\n");
    printf("                                                    expert cache can reach steady state). 0 = off,\n");
    printf("                                                    the historical advancing-stream behaviour (default).\n");
    printf("  -c, --ctx-size <n>                                context size (default: n_prompt + n_gen + n_depth).\n");
    printf("  --warm-skip <n>                                   run <n> generated tokens in EVERY rep before the\n");
    printf("                                                    clock starts, and report them out of n_gen. The\n");
    printf("                                                    plateau of a generation is not reached at its first\n");
    printf("                                                    token: -d 512 -n 128 reads 7.96 t/s while -d 512\n");
    printf("                                                    -n 512 reads 10.22 on the same engine. 0 = off.\n");
    printf("\n");
    printf(
        "Multiple values can be given for each parameter by separating them with ','\n"
        "or by specifying the parameter multiple times. Ranges can be given as\n"
        "'first-last' or 'first-last+step' or 'first-last*mult'.\n");
}

static ggml_type ggml_type_from_name(const std::string & s) {
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    if (s == "bf16") {
        return GGML_TYPE_BF16;
    }
    if (s == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (s == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (s == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (s == "q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (s == "q5_1") {
        return GGML_TYPE_Q5_1;
    }
    if (s == "iq4_nl") {
        return GGML_TYPE_IQ4_NL;
    }

    return GGML_TYPE_COUNT;
}

static cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params        params;
    std::string       arg;
    bool              invalid_param = false;
    const std::string arg_prefix    = "--";
    const char        split_delim   = ',';

    params.verbose              = cmd_params_defaults.verbose;
    params.output_format        = cmd_params_defaults.output_format;
    params.output_format_stderr = cmd_params_defaults.output_format_stderr;
    params.reps                 = cmd_params_defaults.reps;
    params.numa                 = cmd_params_defaults.numa;
    params.prio                 = cmd_params_defaults.prio;
    params.delay                = cmd_params_defaults.delay;
    params.progress             = cmd_params_defaults.progress;
    params.no_warmup            = cmd_params_defaults.no_warmup;
    params.offline              = cmd_params_defaults.offline;
    // [CGC MTP instrument 2026-09-17] cmd_params is default-CONSTRUCTED here and then filled field
    // by field (it is not copied from cmd_params_defaults), so every field that has a non-zero
    // default has to be listed. Missing this one made a bare `--spec-type draft-mtp` fail its own
    // range check with "got 0" -- caught by the interface test, not by the compiler.
    params.spec_draft_n_max     = cmd_params_defaults.spec_draft_n_max;

    if (const char * env = getenv("HF_TOKEN")) {
        params.hf_token = env;
    }

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        try {
            if (arg == "-h" || arg == "--help") {
                print_usage(argc, argv);
                exit(0);
            } else if (arg == "-m" || arg == "--model") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.model.insert(params.model.end(), p.begin(), p.end());
            } else if (arg == "-hf" || arg == "-hfr" || arg == "--hf-repo") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.hf_repo.insert(params.hf_repo.end(), p.begin(), p.end());
            } else if (arg == "-hff" || arg == "--hf-file") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.hf_file.insert(params.hf_file.end(), p.begin(), p.end());
            } else if (arg == "-hft" || arg == "--hf-token") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.hf_token = argv[i];
            } else if (arg == "--offline") {
                params.offline = true;
            } else if (arg == "-p" || arg == "--n-prompt") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_prompt.insert(params.n_prompt.end(), p.begin(), p.end());
            } else if (arg == "-n" || arg == "--n-gen") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_gen.insert(params.n_gen.end(), p.begin(), p.end());
            } else if (arg == "-pg") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], ',');
                if (p.size() != 2) {
                    invalid_param = true;
                    break;
                }
                params.n_pg.push_back({ std::stoi(p[0]), std::stoi(p[1]) });
            } else if (arg == "-d" || arg == "--n-depth") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_depth.insert(params.n_depth.end(), p.begin(), p.end());
            } else if (arg == "-b" || arg == "--batch-size") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_batch.insert(params.n_batch.end(), p.begin(), p.end());
            } else if (arg == "-ub" || arg == "--ubatch-size") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_ubatch.insert(params.n_ubatch.end(), p.begin(), p.end());
            } else if (arg == "-ctk" || arg == "--cache-type-k") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<ggml_type> types;
                for (const auto & t : p) {
                    ggml_type gt = ggml_type_from_name(t);
                    if (gt == GGML_TYPE_COUNT) {
                        invalid_param = true;
                        break;
                    }
                    types.push_back(gt);
                }
                if (invalid_param) {
                    break;
                }
                params.type_k.insert(params.type_k.end(), types.begin(), types.end());
            } else if (arg == "-ctv" || arg == "--cache-type-v") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<ggml_type> types;
                for (const auto & t : p) {
                    ggml_type gt = ggml_type_from_name(t);
                    if (gt == GGML_TYPE_COUNT) {
                        invalid_param = true;
                        break;
                    }
                    types.push_back(gt);
                }
                if (invalid_param) {
                    break;
                }
                params.type_v.insert(params.type_v.end(), types.begin(), types.end());
            } else if (arg == "-dev" || arg == "--device") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto combos = string_split<std::string>(argv[i], split_delim);
                for (const auto & combo : combos) {
                    try {
                        params.devices.push_back(parse_devices_arg(combo));
                    } catch (const std::exception & e) {
                        fprintf(stderr, "error: %s\n", e.what());
                        invalid_param = true;
                        break;
                    }
                }
                if (invalid_param) {
                    break;
                }
            } else if (arg == "--list-devices") {
                common_print_available_devices();
                exit(0);
            } else if (arg == "-t" || arg == "--threads") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_threads.insert(params.n_threads.end(), p.begin(), p.end());
            } else if (arg == "-C" || arg == "--cpu-mask") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.cpu_mask.insert(params.cpu_mask.end(), p.begin(), p.end());
            } else if (arg == "--cpu-strict") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.cpu_strict.insert(params.cpu_strict.end(), p.begin(), p.end());
            } else if (arg == "--poll") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.poll.insert(params.poll.end(), p.begin(), p.end());
            } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i], /*allow_negative=*/true);
                params.n_gpu_layers.insert(params.n_gpu_layers.end(), p.begin(), p.end());
            } else if (arg == "-ncmoe" || arg == "--n-cpu-moe") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_cpu_moe.insert(params.n_cpu_moe.end(), p.begin(), p.end());
            } else if (arg == "-expert-cache" || arg == "--expert-cache") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                try {
                    params.expert_cache_bytes.push_back(std::stoull(argv[i]));
                } catch (const std::exception &) {
                    invalid_param = true;
                }
            } else if (llama_supports_rpc() && (arg == "-rpc" || arg == "--rpc")) {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                try {
                    register_rpc_server_list(argv[i]);
                } catch (const std::exception & e) {
                    fprintf(stderr, "error: %s\n", e.what());
                    invalid_param = true;
                    break;
                }
            } else if (arg == "-sm" || arg == "--split-mode") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<llama_split_mode> modes;
                for (const auto & m : p) {
                    llama_split_mode mode;
                    if (m == "none") {
                        mode = LLAMA_SPLIT_MODE_NONE;
                    } else if (m == "layer") {
                        mode = LLAMA_SPLIT_MODE_LAYER;
                    } else if (m == "row") {
                        mode = LLAMA_SPLIT_MODE_ROW;
                    } else if (m == "tensor") {
                        mode = LLAMA_SPLIT_MODE_TENSOR;
                    } else {
                        invalid_param = true;
                        break;
                    }
                    modes.push_back(mode);
                }
                if (invalid_param) {
                    break;
                }
                params.split_mode.insert(params.split_mode.end(), modes.begin(), modes.end());
            } else if (arg == "-lm" || arg == "--load-mode") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<llama_load_mode> modes;
                for (const auto & m : p) {
                    llama_load_mode mode;
                    if (m == "auto") {
                        mode = LLAMA_LOAD_MODE_AUTO;
                    } else if (m == "none") {
                        mode = LLAMA_LOAD_MODE_NONE;
                    } else if (m == "mmap") {
                        mode = LLAMA_LOAD_MODE_MMAP;
                    } else if (m == "mlock") {
                        mode = LLAMA_LOAD_MODE_MLOCK;
                    } else if (m == "mmap+mlock") {
                        mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
                    } else if (m == "dio") {
                        mode = LLAMA_LOAD_MODE_DIRECT_IO;
                    } else {
                        invalid_param = true;
                        break;
                    }
                    modes.push_back(mode);
                }
                if (invalid_param) {
                    break;
                }
                params.load_mode.insert(params.load_mode.end(), modes.begin(), modes.end());
            } else if (arg == "-mg" || arg == "--main-gpu") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.main_gpu = parse_int_range(argv[i]);
            } else if (arg == "-nkvo" || arg == "--no-kv-offload") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.no_kv_offload.insert(params.no_kv_offload.end(), p.begin(), p.end());
            } else if (arg == "--numa") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                std::string value(argv[i]);
                if (value == "distribute" || value == "") {
                    params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
                } else if (value == "isolate") {
                    params.numa = GGML_NUMA_STRATEGY_ISOLATE;
                } else if (value == "numactl") {
                    params.numa = GGML_NUMA_STRATEGY_NUMACTL;
                } else {
                    invalid_param = true;
                    break;
                }
            } else if (arg == "-fa" || arg == "--flash-attn") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<llama_flash_attn_type> types;
                for (const auto & v : p) {
                    llama_flash_attn_type type;
                    if (common_arg_utils::is_truthy(v)) {
                        type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                    } else if (common_arg_utils::is_falsey(v)) {
                        type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                    } else if (common_arg_utils::is_autoy(v)) {
                        type = LLAMA_FLASH_ATTN_TYPE_AUTO;
                    } else {
                        invalid_param = true;
                        break;
                    }
                    types.push_back(type);
                }
                if (invalid_param) {
                    break;
                }
                params.flash_attn.insert(params.flash_attn.end(), types.begin(), types.end());
            } else if (arg == "-mmp" || arg == "--mmap") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                LOG_WRN("DEPRECATED: -mmp and --mmap are deprecated in favour of --load-mode. Please use --load-mode mmap instead.");
                auto p = string_split<bool>(argv[i], split_delim);

                std::vector<llama_load_mode> modes;
                for (const auto & m : p) {
                    llama_load_mode mode;
                    if (m) {
                        mode = LLAMA_LOAD_MODE_MMAP;
                    } else {
                        mode = LLAMA_LOAD_MODE_NONE;
                    }
                    modes.push_back(mode);
                }
                params.load_mode.insert(params.load_mode.end(), modes.begin(), modes.end());
            } else if (arg == "-dio" || arg == "--direct-io") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                LOG_WRN("DEPRECATED: -dio and --direct-io are deprecated in favour of --load-mode. Please use --load-mode dio instead.");
                auto p = string_split<bool>(argv[i], split_delim);

                std::vector<llama_load_mode> modes;
                for (const auto & m : p) {
                    llama_load_mode mode;
                    if (m) {
                        mode = LLAMA_LOAD_MODE_DIRECT_IO;
                    } else {
                        mode = LLAMA_LOAD_MODE_NONE;
                    }
                    modes.push_back(mode);
                }
                params.load_mode.insert(params.load_mode.end(), modes.begin(), modes.end());
            } else if (arg == "-embd" || arg == "--embeddings") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.embeddings.insert(params.embeddings.end(), p.begin(), p.end());
            } else if (arg == "-nopo" || arg == "--no-op-offload") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.no_op_offload.insert(params.no_op_offload.end(), p.begin(), p.end());
            } else if (arg == "--no-host") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.no_host.insert(params.no_host.end(), p.begin(), p.end());
            } else if (arg == "-ts" || arg == "--tensor-split") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                for (auto ts : string_split<std::string>(argv[i], split_delim)) {
                    // split string by ; and /
                    const std::regex           regex{ R"([;/]+)" };
                    std::sregex_token_iterator it{ ts.begin(), ts.end(), regex, -1 };
                    std::vector<std::string>   split_arg{ it, {} };
                    GGML_ASSERT(split_arg.size() <= llama_max_devices());

                    std::vector<float> tensor_split(llama_max_devices());
                    for (size_t i = 0; i < llama_max_devices(); ++i) {
                        if (i < split_arg.size()) {
                            tensor_split[i] = std::stof(split_arg[i]);
                        } else {
                            tensor_split[i] = 0.0f;
                        }
                    }
                    params.tensor_split.push_back(tensor_split);
                }
            } else if (arg == "-ot" || arg == "--override-tensor") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto * value = argv[i];
                /* static */ std::map<std::string, ggml_backend_buffer_type_t> buft_list;
                if (buft_list.empty()) {
                    // enumerate all the devices and add their buffer types to the list
                    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                        auto * dev = ggml_backend_dev_get(i);
                        auto * buft = ggml_backend_dev_buffer_type(dev);
                        if (buft) {
                            buft_list[ggml_backend_buft_name(buft)] = buft;
                        }
                    }
                }
                auto override_group_span_len = std::strcspn(value, ",");
                bool last_group = false;
                do {
                    if (override_group_span_len == 0) {
                        // Adds an empty override-tensors for an empty span
                        params.tensor_buft_overrides.push_back({{}});
                        if (value[override_group_span_len] == '\0') {
                            value = &value[override_group_span_len];
                            last_group = true;
                        } else {
                            value = &value[override_group_span_len + 1];
                            override_group_span_len = std::strcspn(value, ",");
                        }
                        continue;
                    }
                    // Stamps null terminators into the argv
                    // value for this option to avoid the
                    // memory leak present in the implementation
                    // over in arg.cpp. Acceptable because we
                    // only parse these args once in this program.
                    auto * override_group = value;
                    if (value[override_group_span_len] == '\0') {
                        value = &value[override_group_span_len];
                        last_group = true;
                    } else {
                        value[override_group_span_len] = '\0';
                        value = &value[override_group_span_len + 1];
                    }
                    std::vector<llama_model_tensor_buft_override> group_tensor_buft_overrides{};
                    auto override_span_len = std::strcspn(override_group, ";");
                    while (override_span_len > 0) {
                        auto * override = override_group;
                        if (override_group[override_span_len] != '\0') {
                            override_group[override_span_len] = '\0';
                            override_group = &override_group[override_span_len + 1];
                        } else {
                            override_group = &override_group[override_span_len];
                        }
                        auto tensor_name_span_len = std::strcspn(override, "=");
                        if (tensor_name_span_len >= override_span_len) {
                            invalid_param = true;
                            break;
                        }
                        override[tensor_name_span_len] = '\0';
                        auto * tensor_name = override;
                        auto * buffer_type = &override[tensor_name_span_len + 1];
                        if (buft_list.find(buffer_type) == buft_list.end()) {
                            printf("error: unrecognized buffer type '%s'\n", buffer_type);
                            printf("Available buffer types:\n");
                            for (const auto & it : buft_list) {
                                printf("  %s\n", ggml_backend_buft_name(it.second));
                            }
                            invalid_param = true;
                            break;
                        }
                        group_tensor_buft_overrides.push_back({tensor_name, buft_list.at(buffer_type)});
                        override_span_len = std::strcspn(override_group, ";");
                    }
                    if (invalid_param) {
                        break;
                    }
                    group_tensor_buft_overrides.push_back({nullptr,nullptr});
                    params.tensor_buft_overrides.push_back(group_tensor_buft_overrides);
                    override_group_span_len = std::strcspn(value, ",");
                } while (!last_group);
            } else if (arg == "-r" || arg == "--repetitions") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.reps = std::stoi(argv[i]);
            } else if (arg == "--prio") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.prio = (enum ggml_sched_priority) std::stoi(argv[i]);
            } else if (arg == "--delay") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.delay = std::stoi(argv[i]);
            } else if (arg == "-o" || arg == "--output") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                invalid_param = !output_format_from_str(argv[i], params.output_format);
            } else if (arg == "-oe" || arg == "--output-err") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                invalid_param = !output_format_from_str(argv[i], params.output_format_stderr);
            } else if (arg == "-v" || arg == "--verbose") {
                params.verbose = true;
            } else if (arg == "--progress") {
                params.progress = true;
            } else if (arg == "--no-warmup") {
                params.no_warmup = true;
            } else if (arg == "-fitt" || arg == "--fit-target") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                for (const auto & v : p) {
                    params.fit_params_target.push_back(std::stoull(v));
                }
            } else if (arg == "-fitc" || arg == "--fit-ctx") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                for (const auto & v : p) {
                    params.fit_params_min_ctx.push_back(std::stoul(v));
                }
            } else if (arg == "--spec-type") {
                // [CGC MTP instrument 2026-09-17] Long form only: `-st` is --single-turn in this
                // fork, and upstream's --spec-type has no short form either (common/arg.cpp:4088).
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.spec_type.insert(params.spec_type.end(), p.begin(), p.end());
            } else if (arg == "--spec-draft-n-max") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.spec_draft_n_max = std::stoi(argv[i]);
            } else if (arg == "--prompt-file") {
                // [CGC 2026-09-19] Long form only; see the note at the --spec-type branch above.
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.prompt_file = argv[i];
                g_prompt_file      = argv[i];
            } else if (arg == "--fixed-fill-seed") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                g_fixed_fill_seed = std::stoi(argv[i]);
            } else if (arg == "-c" || arg == "--ctx-size") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                g_ctx_override = std::stoi(argv[i]);
            } else if (arg == "--warm-skip") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                g_warm_skip = std::stoi(argv[i]);
            } else {
                invalid_param = true;
                break;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            invalid_param = true;
            break;
        }
    }

    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }

    // [CGC MTP instrument 2026-09-17] Speculative-path selection, resolved ONCE here so that every
    // later consumer (the model load in to_llama_mparams(), the gen cell) reads the same answer.
    // Order: CLI first, then the deprecated environment shim. Nothing REQUIRES an env var -- see the
    // header note at the top of this file.
    if (params.spec_type.empty()) {
        if (getenv("LLAMA_BENCH_SPEC") != nullptr) {
            params.spec_type = { "draft-mtp" };
            fprintf(stderr, "llama-bench: warning: LLAMA_BENCH_SPEC is deprecated; use `--spec-type draft-mtp`\n");
            if (const char * n = getenv("LLAMA_BENCH_SPEC_DRAFT_N_MAX")) {
                params.spec_draft_n_max = std::stoi(n);
                fprintf(stderr, "llama-bench: warning: LLAMA_BENCH_SPEC_DRAFT_N_MAX is deprecated; use `--spec-draft-n-max %s`\n", n);
            }
        }
    }

    try {
        params.spec_types = common_speculative_types_from_names(params.spec_type);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        fprintf(stderr, "       known spec types: %s\n", common_speculative_all_types_str());
        exit(1);
    }

    // 'none' is upstream's way of saying "explicitly disabled" (common_speculative_types_from_names
    // returns exactly {NONE} for it), so it maps onto the empty set here rather than onto a type.
    if (params.spec_types.size() == 1 && params.spec_types[0] == COMMON_SPECULATIVE_TYPE_NONE) {
        params.spec_types.clear();
    }

    // draft-mtp is what this instrument was built for; the n-gram family is accepted as its CONTROL
    // (a real draft without a draft forward). Anything else still stops the run: falling back
    // silently to the plain gen cell would emit a throughput number under a command line that
    // claims it measured the speculative path.
    for (const auto t : params.spec_types) {
        if (t != COMMON_SPECULATIVE_TYPE_DRAFT_MTP && !is_ngram_spec_type(t)) {
            fprintf(stderr, "error: --spec-type %s is not implemented by this instrument "
                            "(draft-mtp, or one ngram-* type as the no-draft-forward control)\n",
                    common_speculative_type_to_str(t).c_str());
            exit(1);
        }
    }

    // Draft depth is a measured knob, so an out-of-range value is an error, not a silent clamp.
    if (!params.spec_types.empty() && (params.spec_draft_n_max < 1 || params.spec_draft_n_max > 16)) {
        fprintf(stderr, "error: --spec-draft-n-max must be in [1,16] (got %d)\n", params.spec_draft_n_max);
        exit(1);
    }

    if (!params.spec_types.empty()) {
        // The number produced by the gen cell changes meaning under this flag, so the run says so
        // before it starts rather than leaving it to whoever reads the CSV afterwards.
        fprintf(stderr, "llama-bench: [spec] %s enabled (draft n_max=%d): the gen cell measures the SPECULATIVE path "
                        "and is NOT comparable with a plain -n decode cell; the draft model is the target model "
                        "itself (its in-file MTP block)\n",
                common_speculative_type_name_str(params.spec_types).c_str(), params.spec_draft_n_max);
        if (is_ngram_spec_arm(params.spec_types)) {
            // The number this cell produces is not a quality number, and saying which cost it
            // measures is the difference between an ablation and a mystery.
            fprintf(stderr, "llama-bench: [spec] n-gram CONTROL arm: the draft comes from the token HISTORY "
                            "and there is NO draft forward. This cell therefore measures the VERIFY path at "
                            "the same batch size (1 + n_max tokens); its m is the control for draft-mtp's m.\n");
        }
    }

    if (!params.hf_repo.empty()) {
        for (size_t i = 0; i < params.hf_repo.size(); i++) {
            common_params p;
            p.hf_token      = params.hf_token;
            p.offline       = params.offline;
            p.model.hf_repo = params.hf_repo[i];
            if (!params.hf_file.empty() && !params.hf_file[i].empty()) {
                p.model.hf_file = params.hf_file[i];
            }

            // only the text model file is needed
            common_models_handler models_handler = common_models_handler_init(p, LLAMA_EXAMPLE_BENCH);
            common_models_handler_apply(models_handler, p);
            if (p.model.path.empty()) {
                fprintf(stderr, "error: failed to download model from HuggingFace\n");
                exit(1);
            }

            params.model.push_back(p.model.path);
        }
    }

    // set defaults
    if (params.model.empty()) {
        params.model = cmd_params_defaults.model;
    }
    if (params.n_prompt.empty()) {
        params.n_prompt = cmd_params_defaults.n_prompt;
    }
    if (params.n_gen.empty()) {
        params.n_gen = cmd_params_defaults.n_gen;
    }
    if (params.n_pg.empty()) {
        params.n_pg = cmd_params_defaults.n_pg;
    }
    if (params.n_depth.empty()) {
        params.n_depth = cmd_params_defaults.n_depth;
    }
    if (params.n_batch.empty()) {
        params.n_batch = cmd_params_defaults.n_batch;
    }
    if (params.n_ubatch.empty()) {
        params.n_ubatch = cmd_params_defaults.n_ubatch;
    }
    if (params.type_k.empty()) {
        params.type_k = cmd_params_defaults.type_k;
    }
    if (params.type_v.empty()) {
        params.type_v = cmd_params_defaults.type_v;
    }
    if (params.n_gpu_layers.empty()) {
        params.n_gpu_layers = cmd_params_defaults.n_gpu_layers;
    }
    if (params.n_cpu_moe.empty()) {
        params.n_cpu_moe = cmd_params_defaults.n_cpu_moe;
    }
    if (params.expert_cache_bytes.empty()) {
        params.expert_cache_bytes = cmd_params_defaults.expert_cache_bytes;
    }
    if (params.split_mode.empty()) {
        params.split_mode = cmd_params_defaults.split_mode;
    }
    if (params.load_mode.empty()) {
        params.load_mode = cmd_params_defaults.load_mode;
    }
    if (params.main_gpu.empty()) {
        params.main_gpu = cmd_params_defaults.main_gpu;
    }
    if (params.no_kv_offload.empty()) {
        params.no_kv_offload = cmd_params_defaults.no_kv_offload;
    }
    if (params.flash_attn.empty()) {
        params.flash_attn = cmd_params_defaults.flash_attn;
    }
    if (params.devices.empty()) {
        params.devices = cmd_params_defaults.devices;
    }
    if (params.tensor_split.empty()) {
        params.tensor_split = cmd_params_defaults.tensor_split;
    }
    if (params.tensor_buft_overrides.empty()) {
        params.tensor_buft_overrides = cmd_params_defaults.tensor_buft_overrides;
    }
    if (params.embeddings.empty()) {
        params.embeddings = cmd_params_defaults.embeddings;
    }
    if (params.no_op_offload.empty()) {
        params.no_op_offload = cmd_params_defaults.no_op_offload;
    }
    if (params.no_host.empty()) {
        params.no_host = cmd_params_defaults.no_host;
    }
    if (params.n_threads.empty()) {
        params.n_threads = cmd_params_defaults.n_threads;
    }
    if (params.cpu_mask.empty()) {
        params.cpu_mask = cmd_params_defaults.cpu_mask;
    }
    if (params.cpu_strict.empty()) {
        params.cpu_strict = cmd_params_defaults.cpu_strict;
    }
    if (params.poll.empty()) {
        params.poll = cmd_params_defaults.poll;
    }
    if (params.fit_params_target.empty()) {
        params.fit_params_target = cmd_params_defaults.fit_params_target;
    }
    if (params.fit_params_min_ctx.empty()) {
        params.fit_params_min_ctx = cmd_params_defaults.fit_params_min_ctx;
    }

    return params;
}

struct cmd_params_instance {
    std::string        model;
    int                n_prompt;
    int                n_gen;
    int                n_depth;
    int                n_batch;
    int                n_ubatch;
    ggml_type          type_k;
    ggml_type          type_v;
    int                n_threads;
    std::string        cpu_mask;
    bool               cpu_strict;
    int                poll;
    int                n_gpu_layers;
    int                n_cpu_moe;
    size_t             expert_cache_bytes;
    llama_split_mode   split_mode;
    llama_load_mode    load_mode;
    int                main_gpu;
    bool               no_kv_offload;
    llama_flash_attn_type flash_attn;
    std::vector<ggml_backend_dev_t> devices;
    std::vector<float> tensor_split;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    bool               embeddings;
    bool               no_op_offload;
    bool               no_host;
    size_t             fit_target;
    uint32_t           fit_min_ctx;
    // [CGC MTP instrument 2026-09-17] Carried on the instance because it decides the MODEL LOAD
    // (whether the MTP block is kept), not just the test loop. The initialisers here exist only so
    // that the aggregate initialisers below stay warning-free; get_cmd_params_instances() assigns
    // both for every instance, so no instance can be built with a stale value.
    std::vector<common_speculative_type> spec_types       = {};
    int                                  spec_draft_n_max = 0;

    llama_model_params to_llama_mparams() const {
        llama_model_params mparams = llama_model_default_params();

        mparams.n_gpu_layers = n_gpu_layers;
        if (!devices.empty()) {
            mparams.devices = const_cast<ggml_backend_dev_t *>(devices.data());
        }
        mparams.split_mode    = split_mode;
        mparams.load_mode     = load_mode;
        mparams.main_gpu      = main_gpu;
        mparams.tensor_split  = tensor_split.data();
        mparams.no_host       = no_host;

        mparams.expert_cache_bytes = expert_cache_bytes;

        // [CGC MTP instrument 2026-09-17] `--spec-type draft-mtp` needs the MTP block to EXIST in
        // the loaded model. The qwen35moe loader creates layers[n_layer].nextn.* with TENSOR_SKIP
        // unless llama_model_params::load_mtp is set (models/qwen35moe.cpp:45), and the draft
        // context that common_speculative_init_from_params builds asserts on those tensors
        // (models/qwen35moe.cpp:566 "MTP block missing nextn.eh_proj"). This is the SAME rule the
        // server applies, so it is written the same way as common.cpp's
        // common_model_params_to_llama() rather than as a second opinion -- it has to be set HERE,
        // at load time, because the test loop runs long after the model is mapped. Observed without
        // it: instant GGML_ASSERT + SIGABRT (rc=-6) at `llama_init_from_model` <-
        // `common_speculative_init_from_params` in 11 s.
        mparams.load_mtp = std::find(spec_types.begin(), spec_types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != spec_types.end();

        if (n_cpu_moe <= 0) {
            if (tensor_buft_overrides.empty()) {
                mparams.tensor_buft_overrides = nullptr;
            } else {
                GGML_ASSERT(tensor_buft_overrides.back().pattern == nullptr &&
                            "Tensor buffer overrides not terminated with empty pattern");
                mparams.tensor_buft_overrides = tensor_buft_overrides.data();
            }
        } else {
            static std::vector<llama_model_tensor_buft_override> merged;
            static std::vector<std::string> patterns;

            merged.clear();
            patterns.clear();

            auto first = tensor_buft_overrides.begin();
            auto last  = tensor_buft_overrides.end();
            if (first != last && (last - 1)->pattern == nullptr) {
                --last;
            }
            merged.insert(merged.end(), first, last);

            patterns.reserve((size_t) n_cpu_moe);
            merged.reserve(merged.size() + (size_t) n_cpu_moe + 1);

            for (int i = 0; i < n_cpu_moe; ++i) {
                patterns.push_back(llm_ffn_exps_block_regex(i));
                merged.push_back({ patterns.back().c_str(),
                                ggml_backend_cpu_buffer_type() });
            }

            merged.push_back({ nullptr, nullptr });

            mparams.tensor_buft_overrides = merged.data();
        }

        return mparams;
    }

    bool equal_mparams(const cmd_params_instance & other) const {
        return model == other.model && n_gpu_layers == other.n_gpu_layers && n_cpu_moe == other.n_cpu_moe &&
               expert_cache_bytes == other.expert_cache_bytes &&
               // load_mtp is derived from spec_types, so it is part of the mparams identity.
               spec_types == other.spec_types &&
               split_mode == other.split_mode &&
               main_gpu == other.main_gpu && tensor_split == other.tensor_split &&
               load_mode == other.load_mode && devices == other.devices && no_host == other.no_host &&
               vec_tensor_buft_override_equal(tensor_buft_overrides, other.tensor_buft_overrides);
    }

    llama_context_params to_llama_cparams() const {
        llama_context_params cparams = llama_context_default_params();

        // [CGC 2026-09-19] -c/--ctx-size overrides the derived value; 0 keeps it bit-for-bit.
        cparams.n_ctx           = ctx_size_value() > 0
                                ? (uint32_t) ctx_size_value()
                                : (uint32_t) (n_prompt + n_gen + n_depth);
        cparams.n_batch         = n_batch;
        cparams.n_ubatch        = n_ubatch;
        cparams.type_k          = type_k;
        cparams.type_v          = type_v;
        cparams.offload_kqv     = !no_kv_offload;
        cparams.flash_attn_type = flash_attn;
        cparams.embeddings      = embeddings;
        cparams.op_offload      = !no_op_offload;
        cparams.swa_full        = false;

        return cparams;
    }
};

static std::vector<cmd_params_instance> get_cmd_params_instances(const cmd_params & params) {
    std::vector<cmd_params_instance> instances;

    // this ordering minimizes the number of times that each model needs to be reloaded
    // clang-format off
    for (const auto & m : params.model)
    for (const auto & fpt : params.fit_params_target)
    for (const auto & fpc : params.fit_params_min_ctx)
    for (const auto & nl : params.n_gpu_layers)
    for (const auto & ncmoe : params.n_cpu_moe)
    for (const auto & ecb : params.expert_cache_bytes)
    for (const auto & sm : params.split_mode)
    for (const auto & lm : params.load_mode)
    for (const auto & mg : params.main_gpu)
    for (const auto & devs : params.devices)
    for (const auto & ts : params.tensor_split)
    for (const auto & ot : params.tensor_buft_overrides)
    for (const auto & noh : params.no_host)
    for (const auto & embd : params.embeddings)
    for (const auto & nopo : params.no_op_offload)
    for (const auto & nb : params.n_batch)
    for (const auto & nub : params.n_ubatch)
    for (const auto & tk : params.type_k)
    for (const auto & tv : params.type_v)
    for (const auto & nkvo : params.no_kv_offload)
    for (const auto & fa : params.flash_attn)
    for (const auto & nt : params.n_threads)
    for (const auto & cm : params.cpu_mask)
    for (const auto & cs : params.cpu_strict)
    for (const auto & nd : params.n_depth)
    for (const auto & pl : params.poll) {
        for (const auto & n_prompt : params.n_prompt) {
            if (n_prompt == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model                 = */ m,
                /* .n_prompt              = */ n_prompt,
                /* .n_gen                 = */ 0,
                /* .n_depth               = */ nd,
                /* .n_batch               = */ nb,
                /* .n_ubatch              = */ nub,
                /* .type_k                = */ tk,
                /* .type_v                = */ tv,
                /* .n_threads             = */ nt,
                /* .cpu_mask              = */ cm,
                /* .cpu_strict            = */ cs,
                /* .poll                  = */ pl,
                /* .n_gpu_layers          = */ nl,
                /* .n_cpu_moe             = */ ncmoe,
                /* .expert_cache_bytes    = */ ecb,
                /* .split_mode            = */ sm,
                /* .load_mode             = */ lm,
                /* .main_gpu              = */ mg,
                /* .no_kv_offload         = */ nkvo,
                /* .flash_attn            = */ fa,
                /* .devices               = */ devs,
                /* .tensor_split          = */ ts,
                /* .tensor_buft_overrides = */ ot,
                /* .embeddings            = */ embd,
                /* .no_op_offload         = */ nopo,
                /* .no_host               = */ noh,
                /* .fit_target            = */ fpt,
                /* .fit_min_ctx           = */ fpc,
            };
            instances.push_back(instance);
        }

        for (const auto & n_gen : params.n_gen) {
            if (n_gen == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model                 = */ m,
                /* .n_prompt              = */ 0,
                /* .n_gen                 = */ n_gen,
                /* .n_depth               = */ nd,
                /* .n_batch               = */ nb,
                /* .n_ubatch              = */ nub,
                /* .type_k                = */ tk,
                /* .type_v                = */ tv,
                /* .n_threads             = */ nt,
                /* .cpu_mask              = */ cm,
                /* .cpu_strict            = */ cs,
                /* .poll                  = */ pl,
                /* .n_gpu_layers          = */ nl,
                /* .n_cpu_moe             = */ ncmoe,
                /* .expert_cache_bytes    = */ ecb,
                /* .split_mode            = */ sm,
                /* .load_mode             = */ lm,
                /* .main_gpu              = */ mg,
                /* .no_kv_offload         = */ nkvo,
                /* .flash_attn            = */ fa,
                /* .devices               = */ devs,
                /* .tensor_split          = */ ts,
                /* .tensor_buft_overrides = */ ot,
                /* .embeddings            = */ embd,
                /* .no_op_offload         = */ nopo,
                /* .no_host               = */ noh,
                /* .fit_target            = */ fpt,
                /* .fit_min_ctx           = */ fpc,
            };
            instances.push_back(instance);
        }

        for (const auto & n_pg : params.n_pg) {
            if (n_pg.first == 0 && n_pg.second == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model                 = */ m,
                /* .n_prompt              = */ n_pg.first,
                /* .n_gen                 = */ n_pg.second,
                /* .n_depth               = */ nd,
                /* .n_batch               = */ nb,
                /* .n_ubatch              = */ nub,
                /* .type_k                = */ tk,
                /* .type_v                = */ tv,
                /* .n_threads             = */ nt,
                /* .cpu_mask              = */ cm,
                /* .cpu_strict            = */ cs,
                /* .poll                  = */ pl,
                /* .n_gpu_layers          = */ nl,
                /* .n_cpu_moe             = */ ncmoe,
                /* .expert_cache_bytes    = */ ecb,
                /* .split_mode            = */ sm,
                /* .load_mode             = */ lm,
                /* .main_gpu              = */ mg,
                /* .no_kv_offload         = */ nkvo,
                /* .flash_attn            = */ fa,
                /* .devices               = */ devs,
                /* .tensor_split          = */ ts,
                /* .tensor_buft_overrides = */ ot,
                /* .embeddings            = */ embd,
                /* .no_op_offload         = */ nopo,
                /* .no_host               = */ noh,
                /* .fit_target            = */ fpt,
                /* .fit_min_ctx           = */ fpc,
            };
            instances.push_back(instance);
        }
    }
    // clang-format on

    // [CGC MTP instrument 2026-09-17] The speculative selection is per-INVOCATION, not per-instance:
    // every instance of one run shares it, and it also has to reach to_llama_mparams() (it decides
    // whether the MTP block is loaded), which is why it is copied onto the instances here instead of
    // being read from the environment at the point of use.
    for (auto & inst : instances) {
        inst.spec_types       = params.spec_types;
        inst.spec_draft_n_max = params.spec_draft_n_max;
    }

    return instances;
}

struct test {
    static const std::string build_commit;
    static const int         build_number;
    const std::string        cpu_info;
    const std::string        gpu_info;
    std::string              model_filename;
    std::string              model_type;
    uint64_t                 model_size;
    uint64_t                 model_n_params;
    int                      n_batch;
    int                      n_ubatch;
    int                      n_threads;
    std::string              cpu_mask;
    bool                     cpu_strict;
    int                      poll;
    ggml_type                type_k;
    ggml_type                type_v;
    int                      n_gpu_layers;
    int                      n_cpu_moe;
    llama_split_mode         split_mode;
    llama_load_mode          load_mode;
    int                      main_gpu;
    bool                     no_kv_offload;
    llama_flash_attn_type    flash_attn;
    std::vector<ggml_backend_dev_t> devices;
    std::vector<float>       tensor_split;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    bool                     embeddings;
    bool                     no_op_offload;
    bool                     no_host;
    size_t                   fit_target;
    uint32_t                 fit_min_ctx;
    int                      n_prompt;
    int                      n_gen;
    int                      n_depth;
    std::string              test_time;
    std::vector<uint64_t>    samples_ns;

    test(const cmd_params_instance & inst, const llama_model * lmodel, const llama_context * ctx) :
        cpu_info(get_cpu_info()),
        gpu_info(get_gpu_info()) {

        model_filename = inst.model;
        char buf[128];
        llama_model_desc(lmodel, buf, sizeof(buf));
        model_type     = buf;
        model_size     = llama_model_size(lmodel);
        model_n_params = llama_model_n_params(lmodel);
        n_batch        = inst.n_batch;
        n_ubatch       = inst.n_ubatch;
        n_threads      = inst.n_threads;
        cpu_mask       = inst.cpu_mask;
        cpu_strict     = inst.cpu_strict;
        poll           = inst.poll;
        type_k         = inst.type_k;
        type_v         = inst.type_v;
        n_gpu_layers   = inst.n_gpu_layers;
        n_cpu_moe      = inst.n_cpu_moe;
        split_mode     = inst.split_mode;
        load_mode      = inst.load_mode;
        main_gpu       = inst.main_gpu;
        no_kv_offload  = inst.no_kv_offload;
        flash_attn     = inst.flash_attn;
        devices        = inst.devices;
        tensor_split   = inst.tensor_split;
        tensor_buft_overrides = inst.tensor_buft_overrides;
        embeddings     = inst.embeddings;
        no_op_offload  = inst.no_op_offload;
        no_host        = inst.no_host;
        fit_target     = inst.fit_target;
        fit_min_ctx    = inst.fit_min_ctx;
        n_prompt       = inst.n_prompt;
        n_gen          = inst.n_gen;
        n_depth        = inst.n_depth;
        // RFC 3339 date-time format
        time_t t       = time(NULL);
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        (void) ctx;
    }

    uint64_t avg_ns() const { return ::avg(samples_ns); }

    uint64_t stdev_ns() const { return ::stdev(samples_ns); }

    std::vector<double> get_ts() const {
        int                 n_tokens = n_prompt + n_gen;
        std::vector<double> ts;
        std::transform(samples_ns.begin(), samples_ns.end(), std::back_inserter(ts),
                       [n_tokens](uint64_t t) { return 1e9 * n_tokens / t; });
        return ts;
    }

    double avg_ts() const { return ::avg(get_ts()); }

    double stdev_ts() const { return ::stdev(get_ts()); }

    // [CGC 2026-09-18] THE PLATFORM VALUE -- the same statistic with rep 1 dropped, which is what
    // this project's standard quotes (`prod_matrix.py` `WARMUP_RULE` = "report the platform value,
    // i.e. drop rep 1"). It is exposed as a FIELD rather than left as a reader-side convention
    // because the convention is invisible to anyone reading this binary's JSON, and the cost of
    // not applying it is large:
    //
    //   * the generation warmup is ONE token (see the warmup block in main), and for a
    //     `-p 0 -d 512` arm -- the house decode shape -- the prompt warmup does not run at all
    //     (`if (t.n_prompt > 0)`), so rep 1 measures a cold pipeline;
    //   * measured: samples [7.82, 9.71, 9.86] -> avg_ts 9.13 vs platform 9.79 (+7.2%);
    //     samples [4.95, 10.71, 11.00] -> avg_ts 8.89 vs platform 10.86 (+22.2%).
    //
    // So `avg_ts` is not wrong, it answers a different question ("mean over all reps, including the
    // cold one"), and a reader who quotes it understates decode by 3-22% while every visible
    // consistency check still passes. `n_kept` is emitted beside it so that a 1-rep run is visibly
    // NOT a platform value instead of quietly looking like one.
    uint64_t n_kept() const { return samples_ns.size() > 1 ? samples_ns.size() - 1 : samples_ns.size(); }

    double platform_ts() const {
        const std::vector<double> ts = get_ts();
        if (ts.size() <= 1) {
            return ::avg(ts);   // 1 rep: nothing to drop, and n_kept says so
        }
        return ::avg(std::vector<double>(ts.begin() + 1, ts.end()));
    }

    static std::string get_backend() {
        std::vector<std::string> backends;
        bool                     rpc_used = false;
        for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
            auto *      reg  = ggml_backend_reg_get(i);
            std::string name = ggml_backend_reg_name(reg);
            if (string_starts_with(name, "RPC")) {
                if (ggml_backend_reg_dev_count(reg) > 0) {
                    rpc_used = true;
                }
            } else {
                if (name != "CPU") {
                    backends.push_back(ggml_backend_reg_name(reg));
                }
            }
        }
        if (rpc_used) {
            backends.push_back("RPC");
        }
        return backends.empty() ? "CPU" : join(backends, ",");
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {
            "build_commit",   "build_number",   "cpu_info",      "gpu_info",       "backends",
            "model_filename", "model_type",     "model_size",    "model_n_params", "n_batch",
            "n_ubatch",       "n_threads",      "cpu_mask",      "cpu_strict",     "poll",
            "type_k",         "type_v",         "n_gpu_layers",  "n_cpu_moe",      "split_mode",
            "main_gpu",       "no_kv_offload",  "flash_attn",    "devices",        "tensor_split",
            "tensor_buft_overrides",            "load_mode",     "embeddings",
            "no_op_offload",  "no_host",        "fit_target",    "fit_min_ctx",
            "n_prompt",       "n_gen",          "n_depth",
            "test_time",      "avg_ns",         "stddev_ns",     "avg_ts",         "stddev_ts",
            "platform_ts",    "n_kept",         "warm_skip",      "ctx_override"
        };
        return fields;
    }

    enum field_type { STRING, BOOL, INT, FLOAT };

    static field_type get_field_type(const std::string & field) {
        if (field == "build_number" || field == "n_batch" || field == "n_ubatch" || field == "n_threads" ||
            field == "poll" || field == "model_size" || field == "model_n_params" || field == "n_gpu_layers" ||
            field == "main_gpu" || field == "n_prompt" || field == "n_gen" || field == "n_depth" || field == "avg_ns" ||
            field == "stddev_ns" || field == "no_op_offload" || field == "n_cpu_moe" ||
            field == "fit_target" || field == "fit_min_ctx" || field == "flash_attn") {
            return INT;
        }
        if (field == "f16_kv" || field == "no_kv_offload" || field == "cpu_strict" ||
            field == "embeddings" || field == "no_host") {
            return BOOL;
        }
        if (field == "avg_ts" || field == "stddev_ts" || field == "platform_ts") {
            return FLOAT;
        }
        if (field == "n_kept" || field == "warm_skip" || field == "ctx_override") {
            return INT;
        }
        if (field == "load_mode") {
            return STRING;
        }
        return STRING;
    }

    std::vector<std::string> get_values() const {
        std::string tensor_split_str;
        std::string tensor_buft_overrides_str;
        int         max_nonzero = 0;
        for (size_t i = 0; i < llama_max_devices(); i++) {
            if (tensor_split[i] > 0) {
                max_nonzero = i;
            }
        }
        for (int i = 0; i <= max_nonzero; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2f", tensor_split[i]);
            tensor_split_str += buf;
            if (i < max_nonzero) {
                tensor_split_str += "/";
            }
        }
        if (tensor_buft_overrides.size() == 1) {
            // Last element of tensor_buft_overrides is always a null pattern
            // so if it is only one element long, it must be a null pattern.
            GGML_ASSERT(tensor_buft_overrides[0].pattern == nullptr);
            tensor_buft_overrides_str += "none";
        } else {
            for (size_t i = 0; i < tensor_buft_overrides.size()-1; i++) {
                // Last element of tensor_buft_overrides is always a null pattern
                if (tensor_buft_overrides[i].pattern == nullptr) {
                    tensor_buft_overrides_str += "none";
                } else {
                    tensor_buft_overrides_str += tensor_buft_overrides[i].pattern;
                    tensor_buft_overrides_str += "=";
                    tensor_buft_overrides_str += ggml_backend_buft_name(tensor_buft_overrides[i].buft);
                }
                if (i + 2 < tensor_buft_overrides.size()) {
                    tensor_buft_overrides_str += ";";
                }
            }
        }
        std::vector<std::string> values = { build_commit,
                                            std::to_string(build_number),
                                            cpu_info,
                                            gpu_info,
                                            get_backend(),
                                            model_filename,
                                            model_type,
                                            std::to_string(model_size),
                                            std::to_string(model_n_params),
                                            std::to_string(n_batch),
                                            std::to_string(n_ubatch),
                                            std::to_string(n_threads),
                                            cpu_mask,
                                            std::to_string(cpu_strict),
                                            std::to_string(poll),
                                            ggml_type_name(type_k),
                                            ggml_type_name(type_v),
                                            std::to_string(n_gpu_layers),
                                            std::to_string(n_cpu_moe),
                                            split_mode_str(split_mode),
                                            std::to_string(main_gpu),
                                            std::to_string(no_kv_offload),
                                            std::to_string((int) flash_attn),
                                            devices_to_string(devices),
                                            tensor_split_str,
                                            tensor_buft_overrides_str,
                                            llama_load_mode_name(load_mode),
                                            std::to_string(embeddings),
                                            std::to_string(no_op_offload),
                                            std::to_string(no_host),
                                            std::to_string(fit_target),
                                            std::to_string(fit_min_ctx),
                                            std::to_string(n_prompt),
                                            std::to_string(n_gen),
                                            std::to_string(n_depth),
                                            test_time,
                                            std::to_string(avg_ns()),
                                            std::to_string(stdev_ns()),
                                            std::to_string(avg_ts()),
                                            std::to_string(stdev_ts()),
                                            std::to_string(platform_ts()),
                                            std::to_string(n_kept()),
                                            std::to_string(warm_skip_value()),
                                            std::to_string(ctx_size_value()) };
        return values;
    }

    std::map<std::string, std::string> get_map() const {
        std::map<std::string, std::string> map;
        auto                               fields = get_fields();
        auto                               values = get_values();
        std::transform(fields.begin(), fields.end(), values.begin(), std::inserter(map, map.end()),
                       std::make_pair<const std::string &, const std::string &>);
        return map;
    }
};

const std::string test::build_commit = llama_commit();
const int         test::build_number = llama_build_number();

struct printer {
    virtual ~printer() {}

    FILE * fout;

    virtual void print_header(const cmd_params & params) { (void) params; }

    virtual void print_test(const test & t) = 0;

    virtual void print_footer() {}
};

struct csv_printer : public printer {
    static std::string escape_csv(const std::string & field) {
        std::string escaped = "\"";
        for (auto c : field) {
            if (c == '"') {
                escaped += "\"";
            }
            escaped += c;
        }
        escaped += "\"";
        return escaped;
    }

    void print_header(const cmd_params & params) override {
        std::vector<std::string> fields = test::get_fields();
        fprintf(fout, "%s\n", join(fields, ",").c_str());
        (void) params;
    }

    void print_test(const test & t) override {
        std::vector<std::string> values = t.get_values();
        std::transform(values.begin(), values.end(), values.begin(), escape_csv);
        fprintf(fout, "%s\n", join(values, ",").c_str());
    }
};

static std::string escape_json(const std::string & value) {
    std::string escaped;
    for (auto c : value) {
        if (c == '"') {
            escaped += "\\\"";
        } else if (c == '\\') {
            escaped += "\\\\";
        } else if (c <= 0x1f) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            escaped += buf;
        } else {
            escaped += c;
        }
    }
    return escaped;
}

static std::string format_json_value(const std::string & field, const std::string & value) {
    switch (test::get_field_type(field)) {
        case test::STRING:
            return "\"" + escape_json(value) + "\"";
        case test::BOOL:
            return value == "0" ? "false" : "true";
        default:
            return value;
    }
}

struct json_printer : public printer {
    bool first = true;

    void print_header(const cmd_params & params) override {
        fprintf(fout, "[\n");
        (void) params;
    }

    void print_fields(const std::vector<std::string> & fields, const std::vector<std::string> & values) {
        assert(fields.size() == values.size());
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "    \"%s\": %s,\n", fields.at(i).c_str(),
                    format_json_value(fields.at(i), values.at(i)).c_str());
        }
    }

    void print_test(const test & t) override {
        if (first) {
            first = false;
        } else {
            fprintf(fout, ",\n");
        }
        fprintf(fout, "  {\n");
        print_fields(test::get_fields(), t.get_values());
        fprintf(fout, "    \"samples_ns\": [ %s ],\n", join(t.samples_ns, ", ").c_str());
        fprintf(fout, "    \"samples_ts\": [ %s ]\n", join(t.get_ts(), ", ").c_str());
        fprintf(fout, "  }");
        fflush(fout);
    }

    void print_footer() override { fprintf(fout, "\n]\n"); }
};

struct jsonl_printer : public printer {
    void print_fields(const std::vector<std::string> & fields, const std::vector<std::string> & values) {
        assert(fields.size() == values.size());
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "\"%s\": %s, ", fields.at(i).c_str(), format_json_value(fields.at(i), values.at(i)).c_str());
        }
    }

    void print_test(const test & t) override {
        fprintf(fout, "{");
        print_fields(test::get_fields(), t.get_values());
        fprintf(fout, "\"samples_ns\": [ %s ],", join(t.samples_ns, ", ").c_str());
        fprintf(fout, "\"samples_ts\": [ %s ]", join(t.get_ts(), ", ").c_str());
        fprintf(fout, "}\n");
        fflush(fout);
    }
};

struct markdown_printer : public printer {
    std::vector<std::string> fields;

    static int get_field_width(const std::string & field) {
        if (field == "model") {
            return -30;
        }
        if (field == "t/s") {
            return 20;
        }
        if (field == "size" || field == "params") {
            return 10;
        }
        if (field == "n_gpu_layers") {
            return 3;
        }
        if (field == "n_threads") {
            return 7;
        }
        if (field == "n_batch") {
            return 7;
        }
        if (field == "n_ubatch") {
            return 8;
        }
        if (field == "type_k" || field == "type_v") {
            return 6;
        }
        if (field == "split_mode") {
            return 6;
        }
        if (field == "load_mode") {
            return 10;
        }
        if (field == "flash_attn") {
            return 3;
        }
        if (field == "devices") {
            return -12;
        }
        if (field == "test") {
            return 15;
        }
        if (field == "no_op_offload") {
            return 4;
        }
        if (field == "no_host") {
            return 4;
        }

        int width = std::max((int) field.length(), 10);

        if (test::get_field_type(field) == test::STRING) {
            return -width;
        }
        return width;
    }

    static std::string get_field_display_name(const std::string & field) {
        if (field == "n_gpu_layers") {
            return "ngl";
        }
        if (field == "split_mode") {
            return "sm";
        }
        if (field == "n_threads") {
            return "threads";
        }
        if (field == "no_kv_offload") {
            return "nkvo";
        }
        if (field == "flash_attn") {
            return "fa";
        }
        if (field == "load_mode") {
            return "lm";
        }
        if (field == "embeddings") {
            return "embd";
        }
        if (field == "no_op_offload") {
            return "nopo";
        }
        if (field == "no_host") {
            return "noh";
        }
        if (field == "devices") {
            return "dev";
        }
        if (field == "tensor_split") {
            return "ts";
        }
        if (field == "tensor_buft_overrides") {
            return "ot";
        }
        if (field == "fit_target") {
            return "fitt";
        }
        if (field == "fit_min_ctx") {
            return "fitc";
        }
        return field;
    }

    void print_header(const cmd_params & params) override {
        // select fields to print
        fields.emplace_back("model");
        fields.emplace_back("size");
        fields.emplace_back("params");
        fields.emplace_back("backend");
        bool is_cpu_backend = test::get_backend().find("CPU") != std::string::npos ||
                              test::get_backend().find("BLAS") != std::string::npos ||
                              test::get_backend().find("ZenDNN") != std::string::npos;
        if (!is_cpu_backend) {
            fields.emplace_back("n_gpu_layers");
        }
        if (params.n_cpu_moe.size() > 1 || params.n_cpu_moe != cmd_params_defaults.n_cpu_moe) {
            fields.emplace_back("n_cpu_moe");
        }
        if (params.n_threads.size() > 1 || params.n_threads != cmd_params_defaults.n_threads || is_cpu_backend) {
            fields.emplace_back("n_threads");
        }
        if (params.cpu_mask.size() > 1 || params.cpu_mask != cmd_params_defaults.cpu_mask) {
            fields.emplace_back("cpu_mask");
        }
        if (params.cpu_strict.size() > 1 || params.cpu_strict != cmd_params_defaults.cpu_strict) {
            fields.emplace_back("cpu_strict");
        }
        if (params.poll.size() > 1 || params.poll != cmd_params_defaults.poll) {
            fields.emplace_back("poll");
        }
        if (params.n_batch.size() > 1 || params.n_batch != cmd_params_defaults.n_batch) {
            fields.emplace_back("n_batch");
        }
        if (params.n_ubatch.size() > 1 || params.n_ubatch != cmd_params_defaults.n_ubatch) {
            fields.emplace_back("n_ubatch");
        }
        if (params.type_k.size() > 1 || params.type_k != cmd_params_defaults.type_k) {
            fields.emplace_back("type_k");
        }
        if (params.type_v.size() > 1 || params.type_v != cmd_params_defaults.type_v) {
            fields.emplace_back("type_v");
        }
        if (params.main_gpu.size() > 1 || params.main_gpu != cmd_params_defaults.main_gpu) {
            fields.emplace_back("main_gpu");
        }
        if (params.split_mode.size() > 1 || params.split_mode != cmd_params_defaults.split_mode) {
            fields.emplace_back("split_mode");
        }
        if (params.no_kv_offload.size() > 1 || params.no_kv_offload != cmd_params_defaults.no_kv_offload) {
            fields.emplace_back("no_kv_offload");
        }
        if (params.flash_attn.size() > 1 || params.flash_attn != cmd_params_defaults.flash_attn) {
            fields.emplace_back("flash_attn");
        }
        if (params.devices.size() > 1 || params.devices != cmd_params_defaults.devices) {
            fields.emplace_back("devices");
        }
        if (params.tensor_split.size() > 1 || params.tensor_split != cmd_params_defaults.tensor_split) {
            fields.emplace_back("tensor_split");
        }
        if (params.tensor_buft_overrides.size() > 1 || !vec_vec_tensor_buft_override_equal(params.tensor_buft_overrides, cmd_params_defaults.tensor_buft_overrides)) {
            fields.emplace_back("tensor_buft_overrides");
        }
        if (params.load_mode.size() > 1 || params.load_mode != cmd_params_defaults.load_mode) {
            fields.emplace_back("load_mode");
        }
        if (params.embeddings.size() > 1 || params.embeddings != cmd_params_defaults.embeddings) {
            fields.emplace_back("embeddings");
        }
        if (params.no_op_offload.size() > 1 || params.no_op_offload != cmd_params_defaults.no_op_offload) {
            fields.emplace_back("no_op_offload");
        }
        if (params.no_host.size() > 1 || params.no_host != cmd_params_defaults.no_host) {
            fields.emplace_back("no_host");
        }
        if (params.fit_params_target.size() > 1 || params.fit_params_target != cmd_params_defaults.fit_params_target) {
            fields.emplace_back("fit_target");
        }
        if (params.fit_params_min_ctx.size() > 1 || params.fit_params_min_ctx != cmd_params_defaults.fit_params_min_ctx) {
            fields.emplace_back("fit_min_ctx");
        }
        fields.emplace_back("test");
        fields.emplace_back("t/s");

        fprintf(fout, "|");
        for (const auto & field : fields) {
            fprintf(fout, " %*s |", get_field_width(field), get_field_display_name(field).c_str());
        }
        fprintf(fout, "\n");
        fprintf(fout, "|");
        for (const auto & field : fields) {
            int width = get_field_width(field);
            fprintf(fout, " %s%s |", std::string(std::abs(width) - 1, '-').c_str(), width > 0 ? ":" : "-");
        }
        fprintf(fout, "\n");
    }

    void print_test(const test & t) override {
        std::map<std::string, std::string> vmap = t.get_map();

        fprintf(fout, "|");
        for (const auto & field : fields) {
            std::string value;
            char        buf[128];
            if (field == "model") {
                value = t.model_type;
            } else if (field == "size") {
                if (t.model_size < 1024 * 1024 * 1024) {
                    snprintf(buf, sizeof(buf), "%.2f MiB", t.model_size / 1024.0 / 1024.0);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f GiB", t.model_size / 1024.0 / 1024.0 / 1024.0);
                }
                value = buf;
            } else if (field == "params") {
                if (t.model_n_params < 1000 * 1000 * 1000) {
                    snprintf(buf, sizeof(buf), "%.2f M", t.model_n_params / 1e6);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f B", t.model_n_params / 1e9);
                }
                value = buf;
            } else if (field == "backend") {
                value = test::get_backend();
            } else if (field == "test") {
                if (t.n_prompt > 0 && t.n_gen == 0) {
                    snprintf(buf, sizeof(buf), "pp%d", t.n_prompt);
                } else if (t.n_gen > 0 && t.n_prompt == 0) {
                    snprintf(buf, sizeof(buf), "tg%d", t.n_gen);
                } else {
                    snprintf(buf, sizeof(buf), "pp%d+tg%d", t.n_prompt, t.n_gen);
                }
                if (t.n_depth > 0) {
                    int len = strlen(buf);
                    snprintf(buf + len, sizeof(buf) - len, " @ d%d", t.n_depth);
                }
                value = buf;
            } else if (field == "t/s") {
                snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.avg_ts(), t.stdev_ts());
                value = buf;
            } else if (vmap.find(field) != vmap.end()) {
                value = vmap.at(field);
            } else {
                assert(false);
                exit(1);
            }

            int width = get_field_width(field);
            if (field == "t/s") {
                // HACK: the utf-8 character is 2 bytes
                width += 1;
            }
            fprintf(fout, " %*s |", width, value.c_str());
        }
        fprintf(fout, "\n");
    }

    void print_footer() override {
        fprintf(fout, "\nbuild: %s (%d)\n", test::build_commit.c_str(), test::build_number);
    }
};

struct sql_printer : public printer {
    static std::string get_sql_field_type(const std::string & field) {
        switch (test::get_field_type(field)) {
            case test::STRING:
                return "TEXT";
            case test::BOOL:
            case test::INT:
                return "INTEGER";
            case test::FLOAT:
                return "REAL";
            default:
                assert(false);
                exit(1);
        }
    }

    void print_header(const cmd_params & params) override {
        std::vector<std::string> fields = test::get_fields();
        fprintf(fout, "CREATE TABLE IF NOT EXISTS llama_bench (\n");
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "  %s %s%s\n", fields.at(i).c_str(), get_sql_field_type(fields.at(i)).c_str(),
                    i < fields.size() - 1 ? "," : "");
        }
        fprintf(fout, ");\n");
        fprintf(fout, "\n");
        (void) params;
    }

    void print_test(const test & t) override {
        fprintf(fout, "INSERT INTO llama_bench (%s) ", join(test::get_fields(), ", ").c_str());
        fprintf(fout, "VALUES (");
        std::vector<std::string> values = t.get_values();
        for (size_t i = 0; i < values.size(); i++) {
            fprintf(fout, "'%s'%s", values.at(i).c_str(), i < values.size() - 1 ? ", " : "");
        }
        fprintf(fout, ");\n");
    }
};

struct ctx_state {
    int depth = 0; // in tokens

    std::vector<uint8_t> buf; // the llama_context state buffer
};

static bool test_prompt(llama_context * ctx, int n_prompt, int n_batch, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;

    while (n_processed < n_prompt) {
        int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0]    = n_processed == 0 && llama_vocab_get_add_bos(vocab)
                       ? llama_vocab_bos(vocab)
                       : bench_fill_token(vocab, n_vocab, n_processed);
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = bench_fill_token(vocab, n_vocab, n_processed + i);
        }
        int res = llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens));
        if (res != 0) {
            fprintf(stderr, "%s: failed to decode prompt batch, res = %d\n", __func__, res);
            return false;
        }
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);
    return true;
}

static bool test_gen(llama_context * ctx, int n_gen, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    llama_token token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;

    // [CGC 2026-09-29 per-token phase split] WHERE THE ~11.6 ms OUT-OF-GRAPH RESIDUAL IS.
    //
    // WHY THIS EXISTS. DECPROF accounts for the in-graph work (`wait + cb + submit` summed over the
    // step's 40 segments = 73.89 ms on the 85.45 ms baseline), leaving ~11.6 ms that has only ever
    // been a RESIDUAL: `end-to-end - DECPROF total`. §16 §五 and §25 §五 both record that nobody
    // instrumented it, and §25's SG-OFFGRAPH block is the largest blank in the 25 t/s
    // decomposition.
    //
    // WHAT THIS LOOP ACTUALLY CONTAINS, which decides what the residual can even be:
    //   * `llama_decode` -- build + alloc + inputs + graph_compute. CGC_PHASE_TIMING already
    //     splits this end (build=0.038 alloc=0.075 inputs=0.017 ms mean, i.e. ~0.14 ms total).
    //   * `llama_synchronize(ctx)` -- a per-token FULL BLOCK on the GPU. This is NOT inside
    //     DECPROF's segment accounting (there is no hook after the step's last segment), so the
    //     step's tail wait can only land here.
    //   * `std::rand() % n_vocab` -- NOT sampling. This loop never reads logits; it fabricates the
    //     next token. So "sampling cost" is not a term in this residual AT ALL, and a
    //     decomposition that budgets for one would be measuring a thing this harness does not do.
    //
    // So the residual's candidates are exactly two: the untimed part of `llama_decode`, and this
    // `llama_synchronize`. The stamps below separate them, per token, and print AFTER the loop so
    // the timed region is not disturbed by the printf itself.
    static const bool cgc_pt_phase = getenv("CGC_PHASE_TIMING") != nullptr;
    int64_t pt_dec = 0, pt_sync = 0, pt_rand = 0, pt_loop0 = 0;
    if (cgc_pt_phase) { pt_loop0 = ggml_time_us(); }

    for (int i = 0; i < n_gen; i++) {
        const int64_t pt_a = cgc_pt_phase ? ggml_time_us() : 0;
        int res = llama_decode(ctx, llama_batch_get_one(&token, 1));
        const int64_t pt_b = cgc_pt_phase ? ggml_time_us() : 0;
        if (res != 0) {
            fprintf(stderr, "%s: failed to decode generation batch, res = %d\n", __func__, res);
            return false;
        }
        llama_synchronize(ctx);
        const int64_t pt_c = cgc_pt_phase ? ggml_time_us() : 0;
        token = std::rand() % n_vocab;
        if (cgc_pt_phase) {
            pt_dec  += pt_b - pt_a;
            pt_sync += pt_c - pt_b;
            pt_rand += ggml_time_us() - pt_c;
        }
    }

    if (cgc_pt_phase && n_gen > 0) {
        const double loop_ms = (double) (ggml_time_us() - pt_loop0) / 1e3;
        const double n       = (double) n_gen;
        fprintf(stderr,
                "CGC-BENCH-PHASE: n_gen=%d loop=%.3f decode=%.3f sync=%.3f rand=%.3f "
                "accounted=%.3f unaccounted=%.3f ms/token\n",
                n_gen, loop_ms / n, (double) pt_dec / 1e3 / n, (double) pt_sync / 1e3 / n,
                (double) pt_rand / 1e3 / n,
                ((double) pt_dec + (double) pt_sync + (double) pt_rand) / 1e3 / n,
                (loop_ms * 1e3 - (double) (pt_dec + pt_sync + pt_rand)) / 1e3 / n);
    }
    return true;
}

// [CGC MTP instrument 2026-09-17] Speculative generation -- the missing half of the MTP instrument.
//
// WHY THIS EXISTS. `llama-bench` is the project's single instrument of record for prefill and
// decode (docs/PROD_MATRIX_STANDARD_*.html), but it has NO speculative support at all: grepping
// tools/llama-bench/ for speculat|spec_type|draft|mtp returns zero hits, and the test_gen below
// generates tokens with std::rand() % n_vocab -- it never looks at logits. So M4's product (the
// MTP gain) had no instrument of record: `--cells decode` measures MTP-OFF by construction,
// because llama-bench never runs the server and therefore never reads the profile's mtp=1.
//
// HOW IT IS SELECTED: `--spec-type draft-mtp` (or none). NOT a cell, and NOT an env var.
// Not a cell, because the existing `decode` cell has to keep its shape bit-for-bit -- it must stay
// comparable with upstream's `tg128 @ d512` row, so it must not grow a spec branch that fires by
// default; registering it as a proper cell (and deciding what it may be quoted next to) belongs in
// prod_matrix.py + the standard document, not here.
// Not an env var, because the MTP head is an optional block inside ONE GGUF and `--spec-type` is
// already the knob the server uses for it: an interface that requires a particular environment
// variable makes the measurement look file-specific when it is not. LLAMA_BENCH_SPEC still works
// as a DEPRECATED alias (see parse_cmd_params) so existing drivers do not break; it is translated
// into the flag, announced on stderr, and absent from every behavioural path below.
//
// SHAPE. Same verbs, same order, as examples/speculative-simple/speculative-simple.cpp:
//   draft -> verify decode -> common_speculative_process -> sample_and_accept_n -> accept.
// Nothing here consumes the sampled ids as output, so there is no EOG handling and no prompt
// bookkeeping -- this measures throughput only.
//
// WHAT CHANGED AFTER THE FIRST REAL RUN (2026-09-17 21:1x-22:1x, all four found by running it):
//   1. argv must come from llama_bench_matrix.forward_argv() -- hand-building it (only `-m`) made
//      every attempt abort in ggml_metal_synchronize with "command buffer 8 failed (status 5,
//      Insufficient Memory)", which looks exactly like an out-of-memory machine and is not one.
//   2. mparams.load_mtp has to be set at LOAD time (see to_llama_mparams): the loader otherwise
//      skips the MTP block and graph_mtp asserts on it.
//   3. The draft context must be handed over: release_context() + draft.ctx_tgt/ctx_dft.
//   4. Partial acceptance uses a common_prompt_checkpoint restore, NOT a trim -- the context
//      reports SEQ_RM_TYPE_FULL, i.e. it cannot remove a partial sequence, and trimming made the
//      next llama_decode return -1 silently. `LLAMA_BENCH_SPEC_OLDPARTIAL=1` keeps the old path
//      for comparison; `LLAMA_BENCH_SPEC_DBG=1` traces every round.
//   5. THE DRAFT CONTEXT MUST BE BUILT BEFORE THE TARGET DECODES ANYTHING. Built late -- which is
//      where it used to be, inside this function -- the first verify batch of the first round
//      fails with ret=-1 and teardown prints CGC-M2-UNREPOINT. It is invisible on short shapes and
//      fatal on the production one. See bench_spec_setup() for the full account.
//
// ONE REAL SIMPLIFICATION remains vs speculative-simple:
//   `dist = nullptr` in the draft params, so the accept step uses the token-id comparison --
//   i.e. the PRE-M4-1 rule. That is the correct control arm: to measure what the rejection
//   rule buys, run this same function twice with CGC_MTP_REJECTION unset / set.
// [CGC MTP instrument 2026-09-18] WHY THE SETUP IS OUTSIDE test_gen_spec -- the ordering fix.
//
// The draft context used to be created inside test_gen_spec(), i.e. at the START OF THE MEASURED
// RUN. That works on a short shape and fails on the production one, with a failure that looks like
// anything but an ordering bug:
//
//     test_gen_spec: verify decode failed: ret=-1 n_tokens=4(pos 0..3) n_ctx=768 n_past=1 draft=3
//     llama_bench: error: failed to run gen
//     CGC-M2-UNREPOINT: teardown restored 120 expert tensor(s) to their model storage before
//                       freeing 6 slab(s) (a second context built from this model would otherwise
//                       read a freed buffer)
//
// The FIRST llama_decode of the FIRST round returns -1, and llama_decode prints nothing for it.
// Three observations pin this on WHEN the second context is built, not on what it is:
//
//   * `-n 16 -d 0` works, and so does every short shape -- the expert cache is never touched
//     enough for the pool to matter.
//   * `-n 128 -d 512` fails, and `-n 128 -d 512 --no-warmup` fails identically. So the trigger is
//     not the warmup: it is the DEPTH PREFILL (`-d 512`), which is what fills the pool with slabs.
//   * llama-server runs MTP fine, and it builds the draft context at STARTUP -- via common.cpp's
//     `--spec-type draft-mtp` handling -- before any prompt is evaluated.
//
// In this file the order for one instance is: target context, warmup prompt/gen, then the per-rep
// DEPTH PREFILL, then the measured run whose generation calls test_gen_spec.
//
// => It is built here, by the caller, right after the target context exists and before anything is
//    decoded through it: the order llama-server uses, and the only one under which the second
//    context cannot disturb the expert-tensor mapping the pool installs. The speculator, the
//    sampler and the loop still run at the measured point, so what is being timed did not move.
//
// HONEST RESULT, measured 2026-09-18 00:22: moving the construction up here did NOT fix the
// failure. The production shape still died on its first verify batch, byte-identically. The real
// cause was the batch POSITION, not the construction order -- see the n_past initialiser in
// test_gen_spec. This ordering is kept because it is harmless and strictly closer to the server.
//
// Ownership: the draft context lives in this state and is freed BEFORE the target context, because
// the speculator's MTP impl holds a pointer to it. See the reset() call next to llama_free(ctx).
struct bench_spec_state {
    // Declaration order is destruction order inverted: `spec` may touch `ctx_dft` while dying, so
    // ctx_dft has to outlive it.
    common_params                                            params;
    common_speculative_init_result_ptr                       init;
    std::unique_ptr<llama_context, void (*)(llama_context *)> ctx_dft {
        nullptr, [](llama_context * c) { if (c != nullptr) { llama_free(c); } } };
    common_speculative_ptr                                   spec;
    common_sampler_ptr                                       smpl;

    void reset() {
        smpl.reset();
        spec.reset();
        ctx_dft.reset();
        init.reset();
    }
};

// Build the MTP draft context and the speculator. Must be called BEFORE the target context decodes
// anything (see the block comment on bench_spec_state). False + a message on stderr on failure.
// [CGC spec-tree 2026-10-03] Defined further down with the tree helpers; declared here so
// bench_spec_setup() can size the speculator to the tree's branch width.
static bool cgc_spec_tree_on();
static int  cgc_spec_tree_width();

static bool bench_spec_setup(bench_spec_state & s, llama_model * model, llama_context * ctx,
                             const std::vector<common_speculative_type> & spec_types,
                             int32_t spec_draft_n_max) {
    // The spec parameters come from the command line (`--spec-type` / `--spec-draft-n-max`), i.e.
    // from the same vocabulary the server uses for the same two knobs in run_server.sh. n_max is
    // the knob which trades draft depth against per-round cost -- the one an A/B actually moves --
    // so it is a flag, not an environment variable, and its range is validated in parse_cmd_params()
    // rather than clamped here.
    s.params.speculative.types       = spec_types;
    s.params.speculative.draft.n_max = spec_draft_n_max;

    // TWO objects, and the order matters. `common_speculative_init_from_params` does NOT return
    // the speculator -- it returns a holder for the MTP draft CONTEXT, and it writes that context
    // into params.speculative.draft.ctx_dft. The speculator itself is then built from those same
    // params by common_speculative_init(). Both must outlive the loop, and they must share ONE
    // common_params instance or the second call will not see what the first built.
    const bool ngram_arm = is_ngram_spec_arm(spec_types);

    // [CGC spec-tree 2026-10-05] The MTP draft context maps its n_seq_max from params.n_parallel
    // (common/common.cpp: `cparams.n_seq_max = params.n_parallel`). The tree's ALT branch drafts
    // on seq 1, so the draft context must accept two sequences; with the default of 1, llama-batch
    // refused the second seq on EVERY second draft step (`init: invalid seq_id[0][0] = 1 >= 1`,
    // measured 2026-10-05). The target context is created before this function runs (its own
    // n_seq_max override lives at the context-creation site), so this only reaches the draft.
    // Chain/default (width 1) keep n_parallel == 1, bit-for-bit.
    if (cgc_spec_tree_on() && cgc_spec_tree_width() >= 2) {
        s.params.n_parallel = std::max(2, cgc_spec_tree_width());
    }

    // The MTP draft CONTEXT only exists for the draft-model impls. The n-gram arm has no draft model
    // at all (common/speculative.cpp:2625-2635 is the list of impls NOT gated on ctx_dft), so asking
    // for one would be an error this arm has to survive rather than a step it has to take.
    if (!ngram_arm) {
        s.init = common_speculative_init_from_params(s.params, model, ctx);
        if (!s.init) {
            fprintf(stderr, "%s: failed to initialise the MTP draft context\n", __func__);
            return false;
        }
    }
    // [CGC MTP instrument 2026-09-17] The draft context has to be HANDED to the speculator, not
    // just created. common_speculative_init() enables DRAFT_MTP only when
    // `params.draft.ctx_dft != nullptr` (speculative.cpp:2633) -- with it null the constructor
    // builds an empty impl list and returns nullptr, which is what "failed to initialise the
    // speculative path" was. Mirrors examples/speculative-simple/speculative-simple.cpp:142
    // (`ctx_dft.reset(spec_init->release_context())`) and :181
    // (`params.speculative.draft.ctx_dft = ctx_dft.get()`). release_context() transfers ownership
    // out of the holder, so the pointer must be owned and freed here.
    //
    // [CGC 2026-09-19] This half is inside the guard too, and it has to be: with the constructor
    // skipped, `s.init` is null and `s.init->release_context()` is a null dereference -- measured as
    // SIGSEGV at KERN_INVALID_ADDRESS 0x0 with a 3-frame crash report whose only reason frame is
    // `common_speculative_init_result::release_context()` called from llama_bench().
    if (!ngram_arm) {
        s.ctx_dft.reset(s.init->release_context());
        if (!s.ctx_dft) {
            fprintf(stderr, "%s: the MTP draft context could not be released\n", __func__);
            return false;
        }
    }
    // [CGC MTP instrument 2026-09-17] BOTH sides have to be handed over, not just the draft one.
    // common_speculative_impl_draft_mtp asserts `ctx_tgt && ctx_dft` (speculative.cpp:1318), so with
    // ctx_tgt left null the next thing you see is an abort inside the impl ctor. Mirrors
    // examples/speculative-simple/speculative-simple.cpp:180-181:
    //     params.speculative.draft.ctx_tgt = ctx_tgt;
    //     params.speculative.draft.ctx_dft = ctx_dft.get();
    s.params.speculative.draft.ctx_tgt = ctx;
    s.params.speculative.draft.ctx_dft = ngram_arm ? nullptr : s.ctx_dft.get();

    // [CGC spec-tree 2026-10-03] A tree arm's branch lives on seq 1; the MTP verifier tracks
    // its hidden rows per sequence (i_batch_beg / verify_h) so accept_from() can read them.
    // The speculator therefore needs n_seq == tree width. Chain and default (gate unset) keep
    // n_seq == 1, bit-for-bit.
    const uint32_t spec_n_seq = cgc_spec_tree_on() ? (uint32_t) cgc_spec_tree_width() : 1u;
    s.spec.reset(common_speculative_init(s.params.speculative, spec_n_seq));
    if (!s.spec) {
        fprintf(stderr, "%s: failed to initialise the speculative path\n", __func__);
        return false;
    }

    // [CGC 2026-09-19 instrument parity] SAMPLING PARITY WITH llama-server.
    //
    // Until this block existed, this tool had NO sampling controls at all -- `params.sampling` was
    // never touched, so it kept common_params_sampling's struct defaults (temp 0.80 / top_p 0.95 /
    // top_k 40 / min_p 0.05), while run_server.sh:1268-1270 launches the server with
    // `--temp ${CGC_SERVER_TEMP:-0.4} --top-p ${CGC_SERVER_TOP_P:-0.8}` and `--top-k 0`.
    //
    // That is not a cosmetic difference. The MTP accept step compares the draft token against the
    // TARGET's sampled token, so the target distribution's sharpness IS the accept rate: a sharper
    // (lower-temperature) target puts more mass on the draft token and accepts more, a flatter one
    // accepts less. Measured 2026-09-19 on the same prompt and carrier (prod25): the server reports
    // mean_len 2.50 while this tool reports 2.02, and that accept term is 2/3 of the 1.37x
    // throughput gap between the two paths -- so the two instruments were not measuring the same
    // quantity. Reading the SAME env names the launcher uses is what makes them comparable; passing
    // one env set to a driver aligns both sides with no new flags.
    if (const char * v = getenv("CGC_SERVER_TEMP")) {
        s.params.sampling.temp = (float) atof(v);
    }
    if (const char * v = getenv("CGC_SERVER_TOP_P")) {
        s.params.sampling.top_p = (float) atof(v);
    }
    if (const char * v = getenv("CGC_SERVER_TOP_K")) {
        s.params.sampling.top_k = atoi(v);
    }
    if (const char * v = getenv("CGC_SERVER_MIN_P")) {
        s.params.sampling.min_p = (float) atof(v);
    }

    // [CGC 2026-09-26 carrier pin] SAMPLING SEED -- the third parity axis, and the one that makes
    // E a property of the CONFIG rather than of the draw.
    //
    // WHY. `common_params_sampling::seed` defaults to LLAMA_DEFAULT_SEED, and
    // llama_sampler_init_dist() resolves that through get_rng_seed(), which returns a fresh
    // std::random_device draw per process (llama-sampler.cpp:340-350; llama.h:1404 -- "seed ==
    // LLAMA_DEFAULT_SEED to use a random seed"). So every launch of this tool sampled a DIFFERENT
    // token stream. The MTP accept step compares the draft token against the TARGET's sampled
    // token, so the accept rate -- and therefore E = 1 + accepted/rounds -- was a per-process
    // draw. Measured 2026-09-26 (k=2, one config, two reps of one pass): E = 0.984 vs 1.442, 46%
    // apart, while the same cell's measured k-dependence is smaller than that. No m/k ranking
    // taken from those artifacts is quotable, and the artifact had no way to say so.
    //
    // WHAT IS PINNED. Two streams, because both can move the accepted count:
    //   * the sampler's (mt19937 seeded from seed_cur) -- which target token the draft meets;
    //   * the C library's, because the generation loop's FIRST token is
    //     `llama_vocab_get_add_bos() ? bos : std::rand() % n_vocab` (test_gen_spec) and this tool
    //     has never called srand(), so with an add_bos-less vocab that start token is a fixed
    //     process-wide stream rather than a free choice. `std::srand` runs ONCE here, before the
    //     prompt/depth fills and before both generation calls, so the whole process -- prompt
    //     fill, untimed --warm-skip generation, timed generation -- becomes a function of
    //     (config, seed). That is what makes a repeated launch an exact repeat instead of a sample.
    //
    // WHY THIS AND NOT GREEDY. temp 0 also makes the token stream deterministic (the dist sampler
    // collapses to argmax), but it CHANGES the accept rate: a sharper target puts more mass on the
    // draft token and accepts more -- that is exactly why the parity block above exists. E feeds
    // S = E/(1+F+m*k_eff) for the DELIVERY regime (run_server.sh: temp 0.4 / top-p 0.8), so the
    // carrier keeps that distribution and removes only the draw. The seed itself is arbitrary; it
    // only has to be the same one in every arm of a comparison.
    //
    // The witness line is printed unconditionally so an artifact can tell "no seed was asked for"
    // from "a seed was asked for and this engine ignored it" -- the failure mode that has cost
    // this project several rounds (CGC_SERVER_* keys that silently never reached the engine).
    const char * seed_env = getenv("CGC_SERVER_SEED");
    const bool   seed_pinned = seed_env && *seed_env;
    if (seed_pinned) {
        s.params.sampling.seed = (uint32_t) std::strtoul(seed_env, nullptr, 10);
        std::srand((unsigned) s.params.sampling.seed);
    }
    fprintf(stderr, "[CGC seed] sampler_seed=%u pinned=%d temp=%.2f top_p=%.2f top_k=%d\n",
            s.params.sampling.seed, seed_pinned ? 1 : 0,
            s.params.sampling.temp, s.params.sampling.top_p, s.params.sampling.top_k);

    s.smpl.reset(common_sampler_init(model, s.params.sampling));
    if (!s.smpl) {
        fprintf(stderr, "%s: failed to initialise the sampler\n", __func__);
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// [CGC spec-tree 2026-10-02] Tree verification gate, width, and the two context
// overrides the design doc's "no engine change" section did not know about.
//
// WHAT THIS IS. The chain verifier (below, the default) verifies ONE drafted path
// [id_last, draft0 .. draftk-1] and, on partial acceptance, throws the whole round away and
// re-verifies (the `n_replay` path). The tree verifier puts the runner-up candidate next to the
// chosen one in the SAME batch as a real branch (design: docs/SPEC_TREE_VERIFY_DESIGN_2026-10-02.md),
// so a rejected first token is a branch choice instead of a lost round.
//
// GATE SEMANTICS, and why "0" exists:
//   unset -> tree OFF and NO context override. The production path, bit-for-bit untouched.
//   "0"   -> tree OFF, but the context overrides below ARE applied. This is the CONTROL arm of a
//            tree A/B: the tree arm and its chain control have to share one context mode, or the
//            comparison moves two variables and the result means nothing.
//   "1"   -> tree ON. Width from CGC_SPEC_TREE_B (default 2); width 1 is the degenerate tree that
//            must equal the chain, which is the T1c equivalence witness.
//
// WHY THE CONTEXT OVERRIDES ARE NOT OPTIONAL (both are context PARAMS, not engine edits, but the
// design's premise table missed them -- see the charter's follow-up note):
//   1. seq ids. A branch is a second seq in the same batch. The bench context comes from
//      llama_context_default_params() with n_seq_max=1 / kv_unified=false, and seq_id 1 is out of
//      range there (llama-context.cpp:2593 rejects it and llama_decode returns -1). kv_unified=true
//      keeps ONE KV stream (no memory doubling) and admits seq ids up to LLAMA_MAX_SEQ, so it is the
//      smallest-footprint way to make the branch legal.
//   2. state rollback. After the walk, the losing branch and the unaccepted suffix must leave the
//      cache. This model is a Gated-DeltaNet hybrid (arch LLM_ARCH_QWEN35MOE): its recurrent half
//      cannot erase a suffix in place (llama-memory-recurrent.cpp:170 -- "models like Mamba or RWKV
//      can't have a state partially erased at the end of the sequence") unless per-token snapshots
//      exist. `n_rs_seq` is upstream's knob for exactly this, and llm_arch_supports_rs_rollback()
//      returns true for this arch (llama-arch.cpp:1002). Without it the trim fails and the stale
//      cells pollute the next round's attention.
static int cgc_spec_tree_env() {
    const char * v = getenv("CGC_SPEC_TREE");
    return v ? atoi(v) : -1;
}

static bool cgc_spec_tree_on() {
    return cgc_spec_tree_env() == 1;
}

static int cgc_spec_tree_width() {
    const char * v = getenv("CGC_SPEC_TREE_B");
    int w = v ? atoi(v) : 2;
    if (w < 1) { w = 1; }
    if (w > 2) { w = 2; }
    return w;
}

static bool cgc_spec_dump_on() {
    return getenv("LLAMA_BENCH_SPEC_DUMP") != nullptr;
}

// [CGC spec-tree 2026-10-06 H2 shape] SECOND BRANCH FROM THE RUNNER-UP CHANNEL (env
// `CGC_SPEC_TREE_ALTBIND=1`，預設關).
//
// 為什麼要這一格：width-2 的 verify 形狀（`W_eff = 2` ⇒ 6 列）在唯一可用的自然實驗裡**只由
// alt-seed 觸發** —— `alt_on = 1` 同時決定了「B 與 A 共批」與「6 列 verify」⇒ 兩者綁在同一個
// 開關上（F1F2F3 §13.3 的第 2 個混淆）。要分開它們，就得有「**6 列但 draft step 沒有共批**」的臂。
//
// 做法**完全不動引擎**：把 seq 0 的 `draft_params::alt` 綁到 `draft_alt`（impl 在
// `common/speculative.cpp:1962-1963` 逐位把 runner-up 寫進去），同時讓 seq 1 的 `alt_branch`
// 不成立 ⇒ alt-seed 不觸發 ⇒ **draft batch 就是普通 3 列鏈**（B 的第一顆不再被塞進 seq 0 的批、
// 也不共用 seq 0 的 hidden row）。`draft_alt` 改由 runner-up 頻道填滿 ⇒ `W_eff` 照樣回 2 ⇒
// verify 形狀（列數）不變。
//
// ⚠ 讀數的語意：branch 1 的**尾巴**在 alt-seed 版是 B 自己的鏈，在這一版是 seq 0 逐位的
// runner-up ⇒ 兩者的**分支策略不同，只有形狀可比**。報告必須寫成「6 列 ＋ runner-up 尾巴」。
// 預設關 ⇒ `dp.alt` 仍是 nullptr、`alt_branch` 仍由 `cgc_alt_on` 決定，既有 cell 一字不動。
static bool cgc_spec_tree_altbind() {
    const char * v = getenv("CGC_SPEC_TREE_ALTBIND");
    if (v == nullptr) { return false; }
    return v[0] != '0';
}

// [CGC spec-tree 2026-10-02] The recurrent-state rollback budget. It has to be >= the deepest
// rollback the tree can ask for (= the tree depth = k = the draft count): llama_memory_recurrent
// ::seq_rm() refuses a rollback larger than n_rs_seq (llama-memory-recurrent.cpp:184). It also
// MULTIPLIES the recurrent state buffer by (1 + n_rs_seq) (:99), i.e. it is a direct, linear memory
// cost -- the first thing to shrink on a box with no headroom, which is why it is an env knob and
// the default is the minimum that covers the depth-1 tree at k=2.
static int cgc_spec_tree_rs() {
    const char * v = getenv("CGC_SPEC_TREE_RS");
    int n = v ? atoi(v) : 2;
    return n < 0 ? 0 : n;
}

static bool test_gen_spec(llama_context * ctx, llama_model * model, int n_gen, int n_threads,
                          bench_spec_state & s, const char * phase) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    const llama_seq_id  seq_id  = 0;

    // [CGC MTP instrument 2026-09-18] The speculator, the draft context and the sampler are built by
    // bench_spec_setup(), which the caller runs BEFORE the target context decodes anything. Only the
    // measured part is left here. Raw aliases so the loop below reads exactly as it did before.
    common_speculative * spec = s.spec.get();
    common_sampler     * smpl = s.smpl.get();

    llama_tokens prompt; // handed to the draft params; never fed to the target (see SHAPE above)

    // [CGC MTP path parity 2026-09-18] `prompt` above is EMPTY, and its emptiness used to reach
    // common_speculative_begin(), which silently disabled the only draft-path health check in the
    // tree. `prompt_probe` is a separate vector so the size handed to begin() can be honest
    // without changing what the (legacy) draft-params field carries -- nothing enabled here reads
    // it anyway (common/speculative.h:61 marks it for removal).
    llama_tokens prompt_probe;
    llama_tokens draft;
    // [CGC spec-tree 2026-10-02] The runner-up for each drafted position, parallel to `draft` (see
    // common_speculative_draft_params::alt). Populated only when the gate is on; left empty
    // otherwise, which is exactly the nullptr path the draft step skips for free.
    llama_tokens draft_alt;
    // [CGC spec-tree 2026-10-02] T1c witness storage: the committed token sequence, dumped when
    // LLAMA_BENCH_SPEC_DUMP is set so a width-1 tree and the chain can be compared bit-for-bit.
    std::vector<llama_token> emitted;
    // [CGC 2026-09-19 instrument parity] Storage for the draft distribution the rejection
    // rule consumes. The server hands this over only when the rule is on
    // (server-context.cpp:3218: `common_sampler_mtp_rejection_on() ? &slot.spec_draft_dist
    // : nullptr`); this tool hardcoded nullptr, so `CGC_MTP_REJECTION=1` was structurally
    // unrepresentable here and any accept-rate comparison against the server would silently
    // compare two different rules. Left empty when the rule is off, which is exactly the
    // server's default path.
    std::vector<common_draft_dist> draft_dist;

    // [CGC MTP instrument 2026-09-17] Checkpoint for partial acceptance. common_context_can_seq_rm()
    // reports COMMON_CONTEXT_SEQ_RM_TYPE_FULL for this context, which in this fork means "only a
    // WHOLE sequence can be removed" -- i.e. partial removal is NOT supported (speculative-simple
    // says it out loud at :195: use_ckpt_tgt => "context does not support partial sequence
    // removal"). Trimming a partial accept anyway made the NEXT llama_decode return -1, silently,
    // on round 3. The reference's answer is to restore a saved state and re-verify the accepted
    // tokens (speculative-simple.cpp:570-593) -- that is what this does.
    //
    // Why the type is NOT probed here: common_context_can_seq_rm() DECODES 2 tokens [0,0] through
    // the full trunk and clears the memory, but it does NOT reset the expert cache -- it fills
    // slots and bumps LRU ticks (speculative-simple.cpp:184-189, "[CGC bit-bisect v5]"). This
    // instrument does not want that side effect before the measured run, and the checkpoint path
    // is strictly safe, so it is used unconditionally. `LLAMA_BENCH_SPEC_OLDPARTIAL=1` selects the
    // old (broken) trim instead, for A/B of the two mechanisms.
    common_prompt_checkpoint ckpt;

    // [CGC spec-tree 2026-10-02] llama_batch_init allocates n_seq_max seq-id slots PER TOKEN. The
    // tree's root carries one seq id per branch ({0,1,...}), so a batch sized for 1 would write the
    // second id out of bounds. Size it for the tree when the gate is on; the chain still needs 1.
    const int batch_n_seq = cgc_spec_tree_on() ? cgc_spec_tree_width() : 1;
    struct llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, batch_n_seq);

    // [CGC spec-tree 2026-10-02] The tree walk implements GREEDY acceptance only; the MTP rejection
    // rule (CGC_MTP_REJECTION) is applied by common_sampler_sample_and_accept_n, which the tree path
    // does not call. Combining the two would silently measure acceptance with the rule OFF, so say
    // it out loud rather than let the artifact imply otherwise.
    if (cgc_spec_tree_on() && common_sampler_mtp_rejection_on()) {
        fprintf(stderr, "[CGC spec-tree] CGC_MTP_REJECTION is set but the tree walk implements greedy "
                        "acceptance only -- the rejection rule is NOT applied on the tree path. "
                        "Unset one of the two to get a meaningful arm.\n");
    }

    llama_token id_last = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;

    // [CGC MTP instrument 2026-09-18] Start at the context's ACTUAL next position, not at 0.
    //
    // The batch positions below are explicit, while test_gen() -- the non-spec path -- hands
    // llama_batch_get_one() to llama and lets it allocate the position. So only THIS path has to
    // know where it is. With `-d 512` the depth prefill has already written pos 0..511 in this rep
    // (llama-bench runs depth, then prompt, then gen), and asking for pos 0 again makes the FIRST
    // llama_decode return -1 -- silently, because llama_decode prints nothing for it. That single
    // mistake is why every short shape worked (`-n 16 -d 0` has no depth run) and every production
    // shape failed, and why moving the draft-context construction earlier did not help.
    //
    // Empty memory returns -1 here, so a shape without a depth run still starts at 0 exactly as
    // before.
    int n_past = (int) llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
    int n_done = 0;

    // [CGC 2026-09-26] The accept witness. Until this existed, "at what accept was this t/s
    // measured" was unanswerable from a bench artifact: the counters existed (CGC-MTP-PERF,
    // gated on an env that the bench path drops) but no line tied them to the run. These three
    // are the SERVER's own quantities, so the two carriers' reports are comparable:
    //   n_rounds  = slot.n_draft_verif_steps  (one per verify decode, replay rounds included)
    //   n_drafted = slot.n_draft_total        (drafts handed to verify, summed per round)
    //   n_acc_drf = slot.n_draft_accepted     (committed drafts, with the replay discount -- the
    //                                          server's `if (spec_is_replay && n_accepted > 0)`)
    // The printed mean_len is the server's `1 + n_draft_accepted / n_draft_verif_steps`.
        size_t n_rounds  = 0;
        size_t n_drafted = 0;
        size_t n_acc_drf = 0;
        // [CGC 2026-10-01] A replay round (partial acceptance -> checkpoint restore -> `continue`,
        // :3167) is a VERIFY decode that commits and emits NOTHING, and it does not call
        // common_speculative_draft (`if (draft.empty())` at :3008 -- the carried `ids` are the next
        // draft). Until this counter existed, the only two instruments disagreed by exactly the
        // replay count and neither said so: `mean_len = 1 + acc/rounds` counts a replay round as if
        // it emitted one token, and common/speculative.cpp's `emit_tok_per_round` divides by
        // calls_draft, i.e. by the ORIGINAL rounds only (k=2, measured: 232 rounds vs 146 draft
        // calls => 1.59x too high). `emit_per_round = n_done/rounds` below is the quantity that maps
        // to machine cost: emitted tokens per verify decode.
        size_t n_replay  = 0;

        // [CGC spec-tree 2026-10-02] The treatment witness. Following the CGC-LEAFSPLIT lesson, this
        // gets its OWN tag rather than lengthening CGC-BENCH-ACCEPT: a new field on a line that 19
        // parsers read is how `splits=` silently broke them (commit 2ab18619c). These say, per rep:
        // whether the tree actually ran, whether a branch (not just the chain path) was taken, how
        // often the path ended short (the rounds the chain would have thrown away as a replay), and
        // whether the KV trim succeeded (it needs recurrent rollback -- see cgc_spec_tree_env()).
        size_t n_tree_rounds       = 0;
        size_t n_tree_branchB      = 0;
        size_t n_tree_path_partial = 0;
        size_t n_tree_trim_fail    = 0;
        size_t n_tree_dkv_retag    = 0;   // [CGC spec-tree 2026-10-06 F2] B 勝且真的重貼 draft KV 的輪數
        size_t n_tree_rows         = 0;   // [CGC spec-tree 2026-10-06 k-axis] 送進 verify batch 的樹列數總和
        // [CGC spec-tree 2026-10-06 H2 shape] 見 cgc_spec_tree_altbind()／cgc_tree_rows：
        // 第二分支的來源與目標列數。兩個都是 per-phase 常數，但放在這裡是因為見證列印在迴圈外。
        const bool cgc_altbind     = cgc_spec_tree_altbind();
        size_t n_rows_unmet        = 0;   // 目標列數沒有被兌現的輪數（不是靜默）
        int    rows_d_v            = 0;   // 最後一輪算出的「每條分支 verify 尾長」上限
        // [CGC spec-tree 2026-10-06 shape·adaptive] 逐輪控制器的狀態（見 cgc_rows_adapt）。
        // `rows_d_adapt` ＝ 下一輪要用的尾長（1 起手，見上面的起手理由）；`rows_req_v` ＝ 最後一輪的
        // 目標列數（見證欄位 rows_req 的讀數；靜態路徑下恰等於 cgc_tree_rows，故既有值不動）。
        // `rows_hist` 是**實際**列數直方圖 —— 平均值看不出「一趟用過幾種形狀」，直方圖才看得出。
        int    rows_d_adapt        = 1;
        int    rows_req_v          = 0;
        size_t n_rows_switch       = 0;   // 目標列數真的換過的次數（靜態形狀 ⇒ 0）
        int    rows_req_prev       = -1;  // 上一輪的目標列數（-1 ＝ 還沒有上一輪）
        std::vector<double> rows_rate_win;
        double rows_rate_sum       = 0.0;
        std::vector<size_t> rows_hist;
        // [CGC spec-tree 2026-10-06 shape·net] 淨收益目標的狀態（見 cgc_rows_net）：每個候選 d
        // 一條 (payoff, 實際列數) 的滑動窗，窗長 = `_WIN`。索引 ＝ d（1..max，0 不用）。
        // 兩條平行 vector 而不是 pair —— 少一個 <utility> 的相依。
        std::vector<std::vector<int>> net_m_win;
        std::vector<std::vector<int>> net_r_win;
        size_t n_net_samples       = 0;   // 已收的樣本數（每一輪一筆）
        size_t n_net_probes        = 0;   // 已插的探測輪數（round-robin 指標）

        // [CGC spec-tree 2026-10-06 F1] BRANCH-B GATE (env `CGC_SPEC_TREE_BGATE=<window>`, 預設 0 = 關).
        //
        // 為什麼要它：width-2 樹的 verify batch 是 6 列（llama-bench.cpp:3306-3381：
        // `[root_b0][tail b0 x k][root_b1][tail b1 x k]`，k = n_max = 2），但 branch B 只有在
        // 「B 真的贏」的輪才換得回 token —— 實測 branchB = 2/53（rep 1）..19/66（rep 2），
        // 而每一輪都要付 B 的 1 個 root ＋ 2 條 tails（實測每輪 verify ≈3.05 個 token 當量，
        // 見 `Backup/spec_tree/SPECTREE_MTPON_HITRATE_VERDICT_2026-10-06.md` §2）。
        //
        // 政策（deterministic，逐 rep 重置）：每 `window` 輪結算一次；整窗 B 勝 0 次 ⇒ 關掉 B
        // 並冷卻 `4*window` 輪（冷卻結束回到探測窗）。預設 0 = 完全維持現行行為
        // （`W_eff` 永遠 2），所以這個 env 不開時 T1c 的 width-1 見證與既有 cell 一字不動。
        // 關掉 B 的方式是讓 seq-1 的 `alt_branch` 不成立 ⇒ 實作不會 seed alt 鏈 ⇒ `draft_alt`
        // 保持空 ⇒ `W_eff` 依 :3274 自然回 1 ⇒ batch 回到 3 列的鏈（不是另一條特殊路徑）。
        static const int cgc_tree_bgate = []() {
            const char * e = getenv("CGC_SPEC_TREE_BGATE");
            const int v = e != nullptr ? atoi(e) : 0;
            return v < 2 ? 0 : v;   // 0 = 關；<2 輪的窗沒有統計意義
        }();
        int    gt_left = cgc_tree_bgate;   // 本窗還剩幾輪
        int    gt_wins = 0;                // 本窗 B 勝次數
        int    gt_cool = 0;                // 冷卻剩餘輪數（>0 ⇒ 本輪不帶 B）
        size_t n_bgate_alt_rounds = 0, n_bgate_alt_wins = 0;

        // [CGC spec-tree 2026-10-06 F1b] PER-ROUND GATE (env `CGC_SPEC_TREE_RGATE=<K>`，預設 0 = 關；
        // 回探間隔 `CGC_SPEC_TREE_RPROBE=<P>`，預設 8)。§8.4 的結論：window 結算的**粒度**才是 13%
        // 覆蓋率的來源（一窗只要贏 1 次，整窗就不關），不是 3 列不省 —— 3 列步實測只要 6 列步的
        // 0.364×（同熱區 0.36-0.41）。這裡改成逐輪：
        //   open 輪：B 勝 ⇒ streak=0（繼續探測）；B 敗 ⇒ streak++；streak ≥ K ⇒ 關 P 輪。
        //   closed 輪：不帶 B（3 列鏈）；數完 P 輪 ⇒ 回探（streak 歸零重數）。
        // 為什麼是「固定間隔回探」而不是永遠關：closed 輪看不到 B 會不會贏（沒有 alt 鏈可勝），
        // 只能每隔 P 輪花 1 輪去問「B 又活了嗎」。
        static const int cgc_tree_rgate = []() {
            const char * e = getenv("CGC_SPEC_TREE_RGATE");
            const int v = e != nullptr ? atoi(e) : 0;
            return v < 2 ? 0 : v;   // K<2 ＝ 每兩輪就關一輪，白花回探成本
        }();
        static const int cgc_tree_rprobe = []() {
            const char * e = getenv("CGC_SPEC_TREE_RPROBE");
            const int v = e != nullptr ? atoi(e) : 0;
            return v < 1 ? 8 : v;
        }();

        // [CGC spec-tree 2026-10-06 k-axis] TREE DEPTH CAP (env `CGC_SPEC_TREE_K=<k>`，預設 0 = 不裁).
        //
        // 為什麼：6 列樹的第二步命中率只有 0.253（`CGC-ORC-ROUND` 的 pos2A 49/194，見
        // F1F2F3_IMPLEMENT_2026-10-06.md §8.3），但每一輪都要為它付 2 列。這裡把**被驗證的深度**
        // 由 k 裁到 K（verify 列數 = W_eff*(1+K)：K=1 ⇒ 4 列），draft 步驟完全不動 —— 兩條鏈照樣
        // 各吐完整的 k 顆，只是尾巴不進 verify batch。預設 0 = 完全不裁（既有行為一字不動）。
        // 為何不用 `--spec-draft-n-max 1` 做 k=1：那條路會連 alt 鏈的 seed 都不發生
        // （speculative.cpp 的 `params.n_max <= result.size()` 早退排在 alt-seed 之前）⇒ `draft_alt`
        // 空 ⇒ `W_eff` 依上面自然回 1 ⇒ 量到的會是 3 列鏈而不是 4 列樹。所以 k 軸必須裁在這裡。
        static const int cgc_tree_k = []() {
            const char * e = getenv("CGC_SPEC_TREE_K");
            const int v = e != nullptr ? atoi(e) : 0;
            return v < 1 ? 0 : v;
        }();

        // [CGC spec-tree 2026-10-06 shape] VERIFY ROW COUNT (env `CGC_SPEC_TREE_ROWS=<n>`，預設 0 = 關).
        //
        // 為什麼：`W_eff` 與 `K` 兩個旋鈕已經能張出列數（W=1 ⇒ 1+k、W=2 ⇒ 2*(1+k)），但**列數本身
        // 不是參數**，而它是「樹形狀」問題唯一的自變數。這一格讓臂直接指定「這一輪 verify 要幾列」：
        // 目標 n ⇒ 每條分支的 verify 尾長 d = floor((n - W_eff) / W_eff)（列數 = W_eff*(1+d)）。
        // 取不到整數時 d 向下取 ⇒ 量到的是**下一個可達形狀**，並在見證上記 `rows_unmet`
        // （可被跳過的閘門不是閘門：不符目標要看得見，不能靜默）。
        // 兩個前提由臂自己負責（本格不代打）：draft 深度要 ≥ d（`LLAMA_BENCH_SPEC_DRAFT_N_MAX`），
        // 且 `CGC_SPEC_TREE_RS` ≥ K（recurrent rollback 的上限，見 cgc_spec_tree_rs()）。
        // 預設 0 ⇒ 這一格完全不存在，既有 cell 一字不動。
        static const int cgc_tree_rows = []() {
            const char * e = getenv("CGC_SPEC_TREE_ROWS");
            const int v = e != nullptr ? atoi(e) : 0;
            return v < 1 ? 0 : v;
        }();

        // [CGC spec-tree 2026-10-06 shape·adaptive] PER-ROUND ADAPTIVE ROW CONTROLLER
        // (env `CGC_SPEC_TREE_ROWS_ADAPT=<dmax>`，預設 0 = 關；`_WIN=<w>` 接受率視窗，預設 4；
        //  `_THR=<pct>` 接受率門檻，預設 50)。
        //
        // 為什麼：`CGC_SPEC_TREE_ROWS` 是**靜態**形狀 —— 一趟裡每一輪都送同一個列數，量的因此是
        // 「形狀 → 表現」的平均；它回答不了「引擎能不能自己挑形狀」。這一格把列數接上一個逐輪的
        // 閉環：**下一輪的尾長 d 由最近幾輪的接受率決定**。狀態是尾長 d（不是列數），因為
        // 列數 = W_eff*(1+d) 隨 W_eff 逐輪變（alt 鏈在不在），而 d 是形狀唯一不變的自變數。
        //
        // 控制律（bang-bang ＋ 視窗平均，寫死、無隱藏狀態）：
        //   rate_t   = 本輪「勝出分支被接受的 draft 數 m」/「勝出分支實際進 verify 的尾長」
        //   mean_t   = 最近 `_WIN` 輪 rate 的算術平均（不足一窗時就是已收的幾輪）
        //   mean_t >  thr ⇒ d += 1（上限 `_ADAPT`）；mean_t < thr ⇒ d -= 1（下限 1）；相等 ⇒ 不動。
        // 起手 d = 1（不是 0）：d=0 的尾巴是空的，`m` 恆為 0 ⇒ 量不到接受率 ⇒ 控制器沒有方向；
        // 「先驗一顆再問」是唯一讓第一個訊號存在的起手式。d=0 因此**不在**自適應的值域裡。
        //
        // ⚠ 臂的責任（本格不代打，與 CGC_SPEC_TREE_ROWS 同）：draft 深度要 ≥ dmax
        // （`LLAMA_BENCH_SPEC_DRAFT_N_MAX`），且 `CGC_SPEC_TREE_RS` ≥ K —— 否則量到的是被 cap
        // 過的形狀，`rows_unmet`／直方圖會看得見。
        //
        // 預設 0 ⇒ 這一格完全不存在（連見證欄位都是 0），既有 cell 一字不動。
        static const int cgc_rows_adapt = []() {
            const char * e = getenv("CGC_SPEC_TREE_ROWS_ADAPT");
            const int v = e != nullptr ? atoi(e) : 0;
            return v < 1 ? 0 : v;
        }();
        static const int cgc_rows_win = []() {
            const char * e = getenv("CGC_SPEC_TREE_ROWS_WIN");
            const int v = e != nullptr ? atoi(e) : 0;
            return v < 1 ? 4 : v;
        }();
        static const int cgc_rows_thr = []() {
            const char * e = getenv("CGC_SPEC_TREE_ROWS_THR");
            const int v = e != nullptr ? atoi(e) : 0;
            return (v < 1 || v > 100) ? 50 : v;
        }();

        // [CGC spec-tree 2026-10-06 shape·net] 第二個目標函數：**每輪淨收益**，取代接受率門檻。
        //   env `CGC_SPEC_TREE_ROWS_OBJ=net`（預設 rate = 上面的門檻版，行為一字不動）
        //   env `CGC_SPEC_TREE_ROWS_COST=<c>`（單位：**百分之一 token／列**，預設 100 = 1.0）
        //
        // 為什麼要第二個目標：門檻版問的是「接受率夠高嗎」，但接受率高不等於划算 —— 深樹的列數
        // 是逐輪都要付的成本，門檻版的回饋迴路只在「比率」上，看不到「這一輪多送的列值不值得」。
        // 淨收益版把兩個都在同一種單位裡比：
        //   payoff_t = m_t              （本輪真的被接受的 draft 數；root **不計** —— 它每一輪都會
        //                               被接受，是常數、不是形狀的功勞，計進去只會加一個常數）
        //   cost_t   = rows_t × COST/100（**實際**列數，不是目標列數 —— 被 draft 深度／cap 砍過的
        //                               輪才不會把「沒兌現的形狀」記成便宜）
        //   net_t    = payoff_t − cost_t（單位：token）
        //
        // 政策（寫死、無隱藏狀態，與門檻版的差別只在**目標**）：
        //   前 2*dmax 輪 round-robin（每個 d 各量兩次）—— 沒有樣本的 d 不能參加評比，否則 argmax
        //   等於「先量到誰就選誰」；之後 d = argmax_d（最近 `_WIN` 輪的 mean net），同分取**較小的
        //   d**（一樣的收益就少花列數）。
        // 預設 OFF ⇒ 既有 cell 一字不動。
        static const bool cgc_rows_net = []() {
            const char * e = getenv("CGC_SPEC_TREE_ROWS_OBJ");
            return e != nullptr && strcmp(e, "net") == 0;
        }();
        static const int cgc_rows_cost = []() {
            const char * e = getenv("CGC_SPEC_TREE_ROWS_COST");
            if (e == nullptr) { return 100; }
            const int v = atoi(e);
            return v < 0 ? 0 : v;
        }();
        int    rg_streak = 0;   // 連續幾輪 B 沒勝（只在 open 輪累積）
        int    rg_closed = 0;   // 還要關幾輪（>0 ⇒ 本輪不帶 B）
        size_t n_rgate_closed_rounds = 0, n_rgate_closes = 0, n_rgate_probe_wins = 0;

        // [CGC oracle 2026-10-06] HEAD-vs-TARGET-ARGMAX probe (LLAMA_BENCH_SPEC_ORACLE). The walk
        // below accepts a draft when it matches the target's SAMPLED token; the same draft has a
        // second verdict available -- does it match the target's ARGMAX at that row? That is the
        // offline question "how good is the in-file MTP head on the engine's own hidden", with the
        // sampler removed from the comparison. pos1 = the round's first draft (branch A), pos2 =
        // branch A's second token in rounds the walk actually reached it; the B columns are the alt
        // branch's first/second token. smp* counts sampled == argmax on the same rows, i.e. the
        // share of the live loss that is the sampler's, not the head's.
        size_t n_orc_rows1 = 0, n_orc_smp1 = 0, n_orc_amA1 = 0, n_orc_amB1 = 0;
        size_t n_orc_rows2 = 0, n_orc_smp2 = 0, n_orc_amA2 = 0;
        size_t n_orc_rows2B = 0, n_orc_amB2 = 0;
        // [CGC oracle 2026-10-06 F1b] PER-ROUND 版：本輪 root 的 target argmax 落在 A 還是 B 子節點。
        // -1 ＝ 本輪拿不到 root logits（不可沿用上一輪殘值，故開走訪前歸零）。
        // 有了它，T1W2 §2 的「B 首勝之後 A 鎖死」就能在**同一條軌跡內**（本輪 winbr/m × 下一輪 amA）
        // 檢定，不必再用跨臂（between-arm）觀察。
        int orc_r_amA = -1, orc_r_amB = -1;
        // [CGC oracle 2026-10-06 F1c] 本輪 carryover 的來源 seq ＝ 上一輪 accept 用的 src_seq_id。
        // `common_speculative_accept(spec, seq_id, m, W_eff==2 ? win_br : -1)` 就是從這個 seq 的
        // verify 列取 hidden 種子寫進 `pending_h`（speculative.cpp:2053-2073）；-1 ＝ 本 rep 還沒有
        // 前一輪（第一輪的種子由 process() 給）。有了逐輪的它，就能把「本輪第二 token 命中」按
        // 「上一輪取自 seq0／seq1」分組，檢定 F1F2F3 §8.6.5 的播種列假說（不必再靠 winbr 平移）。
        int orc_src_seq = -1;
        // 本輪 branch-A 第二步（pos2）的 argmax 逐輪結果：-1 ＝ 本輪沒走到 A 的 level-1 列（無觀測）；
        // 0 ＝ 走到了、但 argmax ≠ A 的續接 token；1 ＝ 命中。補 `CGC-ORACLE` 聚合（pos2A 49/194）
        // 看不到的「按來源 seq 分組」分布。
        int orc_r_p2A  = -1;
        int orc_r_p2B  = -1;

    // [CGC MTP path parity 2026-09-18] THE SIZE OF THIS VECTOR IS LOAD-BEARING. It used to be 0.
    //
    // common_speculative_impl_draft_mtp::begin() (common/speculative.cpp:1455-1471) opens with
    // `if (N <= 0) return;`, and its ONLY body is a warning that fires when
    //
    //     pos_max(ctx_dft) < N - 1
    //
    // i.e. the one check in the tree that asks "did the target's prefill actually reach the DRAFT
    // context on every ubatch -- need_embd / logits=1 on every prompt position?". The server feeds
    // it the real prompt, so the check is live there. llama-bench fed it the EMPTY `prompt` vector
    // above, so in every bench run that check returned at :1457 and could never fire, no matter how
    // degraded the draft path was. This is the same failure shape as the missing CGC_PHASE_VERIFY:
    // the instrument was silent not because the path was healthy, but because it was never asked.
    //
    // With the length supplied, bench says it out loud:
    //     spec begin: ctx_dft pos_max=-1 < N-1=2024 - process() hook may not have run on every
    //                 prefill ubatch (need_embd / logits=1 on every prompt position?).
    // `pos_max = -1`: the draft context held NO positions -- the MTP head was drafting against an
    // empty prefix. Only `prompt.size()` is read, so the faithful analogue of the server's N is
    // "how many tokens this context has already consumed", i.e. n_past.
    //
    // ATTEMPTED FIX, MEASURED AND REVERTED TWICE (do not remove this note): feeding the prefill to
    // the draft context (the way the server does) is a large REGRESSION in bench, both alone
    // (tg [7.62, 10.03, 12.54, 8.81] -> [2.56, 3.90, 2.68, 2.98]) and paired with the KV-lifecycle
    // half (-> [4.44, 7.23, 3.58, 1.36], 5.3x spread). Retained variant:
    // Backup/bench_parity2_20260918/llama-bench.cpp.VARIANT_B_prefillfeed_lifecycle
    // [CGC 2026-09-19 n-gram control arm] WHAT THE HISTORY HAS TO BE for the control to be a control.
    //
    // common_ngram_map_draft() (common/ngram-map.cpp:229-234) RETURNS WITH NO DRAFT when
    // `inp.size() < 2*size_key + size_m`, and it looks its key up in this history. Two degenerate
    // choices are rejected on exactly those grounds:
    //   * empty history -> cur_len = 0 -> zero drafts -> the arm silently becomes a plain per-token
    //     decode, and would "find" that the draft forward costs everything;
    //   * constant run  -> drafts ARE produced, but the verify batch is k+1 copies of ONE token,
    //     which collapses the expert union and makes the verify path look cheaper than the MTP arm's
    //     -- i.e. it manufactures the same answer by the other door.
    // So the history is a deterministic period-48 ramp: 48 < 2*12+48-1 keeps the key inside the window
    // the descending search actually scans (ngram-map.cpp:279-296), and the m-gram it proposes is a
    // run of DIFFERENT tokens, like draft-mtp's.
    const bool ngram_arm = is_ngram_spec_arm(s.params.speculative.types);
    const int  ngram_period = 48;
    llama_token ngram_next = id_last;
    if (ngram_arm) {
        prompt_probe.assign((size_t) std::max(n_past, 1), 0);
        for (size_t i = 0; i < prompt_probe.size(); ++i) {
            prompt_probe[i] = (llama_token) (n_vocab / 4 + (int32_t) (i % (size_t) ngram_period));
        }
        // `sampled` has to CONTINUE the ramp, or the key is simply absent from the history and the
        // round verifies ONE token with no draft -- a PARTIAL null that reads as "the verify path is
        // cheap". This value is re-imposed every round below, for the same reason.
        ngram_next = (llama_token) (n_vocab / 4 +
                (int32_t) (prompt_probe.size() % (size_t) ngram_period));
        id_last = ngram_next;
        if ((int) prompt_probe.size() < 2 * 12 + 48) {
            fprintf(stderr, "%s: the n-gram control arm needs a history of at least 2*size_key+size_m = "
                            "72 tokens, because common_ngram_map_draft() drafts nothing below that; "
                            "this shape supplied %d. Refusing to run: an arm with no draft verifies one "
                            "token and would be read as a cheap verify path (use -d >= 128).\n",
                    __func__, (int) prompt_probe.size());
            llama_batch_free(batch);
            return false;
        }
    } else {
        prompt_probe.assign((size_t) std::max(n_past, 0), id_last);
    }

    common_speculative_begin(spec, seq_id, prompt_probe);

    while (n_done < n_gen) {
        // [CGC spec-tree 2026-10-06 F1] 本輪要不要帶 branch B（`CGC_SPEC_TREE_BGATE` 沒開時恆 true）。
        const bool cgc_alt_on = ((cgc_tree_bgate <= 0) || (gt_cool == 0)) && (rg_closed == 0);
        if (getenv("LLAMA_BENCH_SPEC_DBG") != nullptr) {
            // [CGC spec-tree 2026-10-05 DBG] The round's top state == what the previous round's
            // cleanup left == the state the draft step below is judged against (M-RoPE X < Y).
            llama_memory_t mem_top = llama_get_memory(ctx);
            llama_context * ctx_dft_top = s.ctx_dft.get();
            const int dft_max = ctx_dft_top
                    ? (int) llama_memory_seq_pos_max(llama_get_memory(ctx_dft_top), seq_id) : -2;
            fprintf(stderr, "SPECDBG top: n_past=%d id_last=%d pos_min=%d pos_max=%d dft_pos_max=%d draft=%zu n_done=%d\n",
                    n_past, (int) id_last,
                    (int) llama_memory_seq_pos_min(mem_top, seq_id),
                    (int) llama_memory_seq_pos_max(mem_top, seq_id),
                    dft_max, draft.size(), n_done);
        }
        if (draft.empty()) {
            // The two halves of the checkpoint are taken at DIFFERENT points, exactly as the
            // reference does it: update_pos before the draft (speculative-simple.cpp:421) and
            // update_tgt after it (:452). The order matters for MTP because the draft context
            // SHARES KV with the target, so by the time update_tgt runs the draft step has already
            // written into the same memory -- and ckpt.pos_max stays the PRE-draft boundary.
            // [CGC spec-tree 2026-10-02] UNCONDITIONAL on purpose -- and the reason IS the T1c
            // witness. This was briefly gated on `!cgc_spec_tree_on()` (the tree never replays, so a
            // snapshot is work with no reader). That gate made the two arms differ in more than the
            // accept path, and the width-1 witness then diverged at TOKEN 0 -- where no partial
            // acceptance has happened yet, so the tree's own logic cannot be the cause. A witness
            // that cannot separate "the treatment" from "my own gating" is not a witness. Reverted:
            // both arms snapshot, so the ONLY difference left is the accept/commit path. The tree arm
            // therefore pays a snapshot it never reads -- a cost, not a correctness difference, and
            // the conservative direction for a paired delta.
            ckpt.update_pos(n_past,
                    llama_memory_seq_pos_min(llama_get_memory(ctx), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id));

            if (ngram_arm) {
                // Re-anchored every round ON PURPOSE: the n-gram map has no way to draft against a
                // history that is not appended to, so `sampled` is kept on the ramp instead of
                // following the target's own last token. The target's token stream is synthetic in
                // this instrument anyway (the plain cell fills it with std::rand()), and what this
                // arm measures is COST per verify step, not quality.
                id_last = ngram_next;
            }

            // [CGC 2026-09-19] Two fields differ for the n-gram arm, and both are load-bearing:
            //   * `.prompt` -- the n-gram impl SEARCHES this vector (and drafts nothing when it is
            //     shorter than 2*size_key+size_m). The MTP impl never reads it, so pointing it at
            //     `prompt_probe` cannot move the draft-mtp arm.
            //   * `.n_max`  -- the dispatcher truncates the draft to this (speculative.cpp:2839-2843)
            //     and llama-bench passed -1 = unbounded. An unbounded n-gram draft is up to size_m =
            //     48 tokens, i.e. a 49-token verify batch, which is NOT the "same k" this control
            //     exists to hold. Bounded to spec_draft_n_max it is exactly k, like draft-mtp.
            // [CGC spec-tree 2026-10-02] `draft_alt` mirrors `draft` positionally and the draft step
            // APPENDS (it does not clear). The two are only in lockstep if whatever emptied `draft`
            // emptied `draft_alt` too -- so do it here, at the one point a fresh draft is asked for,
            // where a stale entry cannot survive into the round even if a path forgot to clear.
            // [CGC 2026-10-06 RS6] `draft_dist` is APPENDED by the draft step, exactly
            // like `draft_alt` right below, and it is only ever read by THIS round's accept
            // call -- so it has to be emptied here too. Left uncleared, round N's distributions
            // get read at round N+1's positions: the guard `i < size()` stays true, nothing
            // throws, and accept rate is quietly wrong.
            draft_dist.clear();
            if (cgc_spec_tree_on()) {
                draft_alt.clear();
            }
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ ngram_arm ? s.params.speculative.draft.n_max : -1,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ ngram_arm ? &prompt_probe : &prompt,
                /* .result   = */ &draft,
                // [CGC 2026-09-19 instrument parity] Was a hardcoded nullptr. The server passes
                // storage only when common_sampler_mtp_rejection_on() (env CGC_MTP_REJECTION,
                // common/sampling.cpp:794) is set; mirroring that here is what lets the same env
                // drive both sides. With the env unset this evaluates to nullptr, so the default
                // path is bit-for-bit the previous behaviour.
                /* .dist     = */ common_sampler_mtp_rejection_on() ? &draft_dist : nullptr,
                // [CGC spec-tree 2026-10-05] nullptr for BOTH arms in the default path: the tree's
                // branch 1 is seq 1's own ALT CHAIN (bound below, written through
                // draft_params::alt_branch), not the seq-0 runner-up array. Binding both would have
                // written two chains into one vector -- the interleave that broke the first alt
                // draft test.
                // [CGC spec-tree 2026-10-06 H2 shape] `CGC_SPEC_TREE_ALTBIND=1` 是那條規則的
                // **唯一例外**：seq 0 綁 runner-up 頻道，而 seq 1 的 `alt_branch` 被強制關
                // （下一格）⇒ 任何時刻只有一個寫者（`drafting[1]` 永遠不成立 ⇒ seq 1 的
                // `.result` 不會被推）⇒ interleave 的前提不成立。預設 nullptr。
                /* .alt      = */ cgc_altbind ? &draft_alt : nullptr,
            };
            // [CGC spec-tree 2026-10-05] Bind seq 1 for the tree: the MTP impl drafts branch 1 as
            // the ALT chain seeded from seq 0's runner-up (draft_params::alt_branch), writing the
            // full chain into `draft_alt`. Before this the impl had nowhere to write branch 1 and
            // the tree reused seq 0's tail -- the defeat this replaces.
            if (cgc_spec_tree_on() && cgc_spec_tree_width() >= 2) {
                common_speculative_get_draft_params(spec, 1) = {
                    /* .drafting   = */ false,
                    /* .n_max      = */ -1,
                    /* .n_past     = */ n_past,
                    /* .id_last    = */ id_last,
                    /* .prompt     = */ &prompt,
                    /* .result     = */ &draft_alt,
                    /* .dist       = */ nullptr,
                    /* .alt        = */ nullptr,
                    // [CGC spec-tree 2026-10-06 F1] `alt_branch` 是本輪「帶不帶 B」的唯一開關：
                    // false ⇒ 實作不 seed alt 鏈（speculative.cpp:1956-1976 的 alt-seed 條件就是
                    // 這個欄位）⇒ `draft_alt` 空 ⇒ `W_eff` = 1。
                    // [CGC spec-tree 2026-10-06 H2 shape] ALTBIND 開的時候 alt-seed 一律不觸發
                    // （這一格就是它的閘）。seq 1 的參數仍然綁著：impl 的條件式要
                    // `n_seq >= 2` 與 `dparams.at(1)` 存在，把整格拿掉會改到別的路徑。
                    /* .alt_branch = */ (cgc_alt_on && !cgc_altbind),
                };
            }
            common_speculative_draft(spec);

            // [CGC spec-tree 2026-10-02] Unconditional, same reason as the update_pos call above:
            // gating this per-arm made the two arms differ in more than the accept path.
            ckpt.update_tgt(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }

        // [CGC MTP instrument 2026-09-17] Bound the verify batch by the context. The batch needs
        // 1 + draft.size() free positions, and llama-bench sizes n_ctx = n_prompt + n_gen + n_depth
        // -- so a short `-n` leaves NO headroom (with -n 16 -d 0 the context is 16). Without this
        // guard the loop walks off the end of the context and llama_decode() fails, which showed up
        // as "failed to decode the verify batch" after 6 rounds. The production decode cell
        // (-n 128 -d 512 => n_ctx 640) has headroom, but the guard makes short shapes usable too.
        if (n_past + (int) draft.size() + 1 > (int) llama_n_ctx(ctx)) {
            break;
        }

        // ── verify batch ────────────────────────────────────────────────────────────────
        // chain (default): [id_last, draft0 .. draftN-1], one seq.
        // tree (CGC_SPEC_TREE): root on seq{0..W-1}; branch b on seq b, whose first token is
        //   (b==0 ? draft[0] : alt[0]) and whose tail is the chain's draft[1..]. Branches share
        //   POSITIONS but not seqs, so the causal mask (llama-kv-cache.cpp:1632, seq_has) keeps
        //   them from seeing each other. No engine change is involved -- the batch API already
        //   carries a per-token seq-id array (include/llama.h:262-263).
        const bool tree_on = cgc_spec_tree_on();
        const int  tree_W  = cgc_spec_tree_width();
        // A branch needs a REAL runner-up at position 0. Without one this round is a width-1 tree,
        // which builds the chain batch exactly -- so "no alt" is not a special case, it is the
        // degenerate tree, and it is the shape T1c pins.
        const int  W_eff   = (tree_on && tree_W >= 2 && draft.size() >= 1 &&
                              !draft_alt.empty() && draft_alt[0] >= 0) ? 2 : 1;
        // [CGC spec-tree 2026-10-06 shape] 由目標列數反推每條分支的 verify 尾長（見 cgc_tree_rows）。
        // 列數 = W_eff 個 root ＋ W_eff 條尾 ⇒ 尾長 d = floor((n - W_eff) / W_eff)；n < W_eff ⇒ d=0
        // （只有 root 的退化形狀，量到幾列由見證負責說）。
        // [CGC spec-tree 2026-10-06 shape·adaptive] 控制器開著時尾長改由 `rows_d_adapt`（逐輪狀態）
        // 給，靜態旋鈕整格不讀 ⇒ 預設路徑的 `rows_d` 與 `rows_req` 一字不動。
        const int  rows_d  = (cgc_rows_adapt > 0)
                ? std::max(0, std::min(rows_d_adapt, cgc_rows_adapt))
                : ((cgc_tree_rows > 0) ? std::max(0, (cgc_tree_rows - W_eff) / W_eff) : 0);
        // `rows_req` 這一輪的目標列數：自適應 ＝ W_eff*(1+d)，靜態 ＝ env 原值。
        const int  rows_req = (cgc_rows_adapt > 0) ? W_eff * (1 + rows_d) : cgc_tree_rows;
        rows_d_v = rows_d;
        rows_req_v = rows_req;
        if (tree_on && rows_req_prev >= 0 && rows_req != rows_req_prev) {
            n_rows_switch += 1;
        }
        rows_req_prev = rows_req;
        // [CGC spec-tree 2026-10-06 shape·adaptive] 本輪真的進 verify 的列數（形狀見證那裡填）與
        // 勝出分支的尾長（控制器的分母，tree walk 尾端填）。放在迴圈體是因為兩者都逐輪歸零。
        int rows_actual_v = 0;
        std::vector<int> tail_len;

        // One tree node = one batch slot. `slot` is implicit (== index into tree_nodes == batch
        // index) because slots are appended to the batch in tree_nodes order.
        struct cgc_tree_node {
            llama_token      token  = -1;
            int              parent = -1;
            int              branch = -1;   // -1 = root, else 0..W_eff-1
            std::vector<int> children;
        };
        std::vector<cgc_tree_node> tree_nodes;

        const int P_base = n_past;   // the root's position; the tree cleanup needs it later

        // [CGC spec-tree 2026-10-03] Give every branch seq the SAME memory prefix as the base seq.
        // The allocator couples seqs that share a prefix and REQUIRES their memory positions to stay
        // equal (llama-batch.cpp:323-335); a fresh branch seq has no memory positions at all, so the
        // first width-2 round died with
        //     init: sequence 1 is coupled to 0 in the input batch, but have divereged
        // This is the template's exact step (examples/speculative/speculative.cpp:612-614: seq_keep(0)
        // then seq_cp(0 -> s) for every branch s) before its verify decode. Skipped for W_eff == 1:
        // with no branch there is nothing to copy, and width 1 must stay the chain bit-for-bit (T1c).
        if (tree_on && W_eff > 1) {
            llama_memory_t mem_pre = llama_get_memory(ctx);
            llama_memory_seq_keep(mem_pre, seq_id);
            for (int b = 1; b < W_eff; ++b) {
                llama_memory_seq_cp(mem_pre, seq_id, (llama_seq_id) b, -1, -1);
            }
        }

        common_batch_clear(batch);
        int tree_depth_v = 0;   // [CGC spec-tree 2026-10-06 k-axis] 本輪真的進 verify 的樹深度（draft 深度或其 cap）
        if (tree_on) {
            const int k_full = (int) draft.size();
            const int k      = (cgc_tree_k > 0) ? std::min(k_full, cgc_tree_k) : k_full;
            tree_depth_v = k;
            // [CGC spec-tree 2026-10-03] ROOT PER BRANCH. A single root carrying seq{0..W-1} is
            // rejected by the MTP verifier (common/speculative.cpp process() asserts
            // n_seq_id[k] == 1: it indexes hidden rows per sequence, and a multi-seq token has
            // no single row). Each branch therefore gets its own copy of the root, and each
            // sequence's tokens stay CONTIGUOUS -- process()'s i_batch_beg/i_batch_end spans
            // (and the non-shared catch-up path) assume one contiguous run per seq.
            //
            // Batch order: [root_b0][tail b0: k][root_b1][tail b1: k] ...
            // Node index == batch slot, so the walk below still samples by slot.
            tree_nodes.reserve((size_t) W_eff * (1 + (size_t) k));
            std::vector<int> root_slot (W_eff, -1);
            std::vector<int> first_tail(W_eff, -1);

            auto tree_push = [&](llama_token tok, int branch) -> int {
                cgc_tree_node n;
                n.token  = tok;
                n.parent = -1;
                n.branch = branch;
                tree_nodes.push_back(std::move(n));
                return (int) tree_nodes.size() - 1;
            };

            // [CGC spec-tree 2026-10-05] PER-BRANCH TAILS: branch 0 = the chain's own draft,
            // branch 1 = the alt chain drafted on seq 1 -- its own first token AND its own
            // continuation. Before this, branch 1 reused draft[1..], so every branch-B win died
            // at the second position (accept@2|1 = 0.375 vs 0.90 for branch 0's own chain).
            tail_len.assign((size_t) W_eff, 0);
            for (int b = 0; b < W_eff; ++b) {
                const llama_tokens & tail = (b == 0) ? draft : draft_alt;
                const int cap = (rows_d > 0) ? std::min(k, rows_d) : k;
                tail_len[b] = std::min(cap, (int) tail.size());
                root_slot[b] = tree_push(id_last, -1);
                for (int i = 0; i < tail_len[b]; ++i) {
                    const int s = tree_push(tail[i], b);
                    if (i == 0) {
                        first_tail[b] = s;
                    }
                }
            }

            // [CGC spec-tree 2026-10-06 k-axis] 形狀見證：本輪送進 decode 的列數 = W_eff 個 root
            // ＋ 每條分支自己的尾長（K=1 ⇒ 2+1+1 = 4；未裁 ⇒ 2+2+2 = 6）。列數不靠推論，這條是量出來的。
            {
                size_t rows = (size_t) W_eff;
                for (int b = 0; b < W_eff; ++b) {
                    rows += (size_t) tail_len[b];
                }
                n_tree_rows += rows;
                // [CGC spec-tree 2026-10-06 shape] 目標列數沒有被兌現時記帳（不靜默）。目標改讀
                // 逐輪的 `rows_req`（靜態路徑下恰等於 cgc_tree_rows ⇒ 既有記帳不動）。
                if (rows_req > 0 && rows != (size_t) rows_req) {
                    n_rows_unmet += 1;
                }
                // [CGC spec-tree 2026-10-06 shape·adaptive] 實際列數直方圖：控制器「一趟用過幾種
                // 形狀」的唯一一手證據（tree_rows/rounds_tree 只是平均，平均 6.0 可以是恆 6 也可
                // 以是 4/6/8 混出來的）。桶超界就長大，不靜默丟。
                if ((size_t) rows >= rows_hist.size()) {
                    rows_hist.resize(rows + 1, 0);
                }
                rows_hist[rows] += 1;
                rows_actual_v = (int) rows;
            }

            // Logical links for the walk: it starts at root_0; every branch's first tail hangs
            // off root_0 (the roots are the SAME token -- the duplication exists only so the
            // verifier sees one sequence per row); a tail chains to the next tail of its branch.
            for (int b = 0; b < W_eff; ++b) {
                if (first_tail[b] >= 0) {
                    tree_nodes[root_slot[0]].children.push_back(first_tail[b]);
                }
            }
            for (int b = 0; b < W_eff; ++b) {
                for (int i = 0; i + 1 < tail_len[b]; ++i) {
                    tree_nodes[first_tail[b] + i].children.push_back(first_tail[b] + i + 1);
                }
            }

            // Every branch root sits at the SAME logical position P_base (the prefixes were
            // seq_cp'd, and coupled seqs must keep equal memory positions). n_past advances by
            // exactly one logical position, not W.
            for (int b = 0; b < W_eff; ++b) {
                common_batch_add(batch, id_last, P_base, { (llama_seq_id) b }, true);
            }
            n_past = P_base + 1;
            for (int b = 0; b < W_eff; ++b) {
                for (int i = 0; i < tail_len[b]; ++i) {
                    const int slot = first_tail[b] + i;
                    GGML_ASSERT(slot >= 0 && slot < (int) tree_nodes.size());
                    common_batch_add(batch, tree_nodes[slot].token, P_base + 1 + (llama_pos) i,
                                     { (llama_seq_id) b }, true);
                }
            }
        } else {
            // verify batch: [id_last, draft0 .. draftN-1]
            common_batch_add  (batch, id_last, n_past++, { seq_id }, true);
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch, draft[i], n_past + (llama_pos) i, { seq_id }, true);
            }
        }

        // [CGC MTP instrument 2026-09-18] THE MISSING PHASE CALL. Until this line existed, this
        // tool's MTP never used the production verify path, and nothing said so.
        //
        // The fast path is gated on a CALLER-SET phase, not on the batch shape
        // (llama-context.cpp:6065):
        //     verify_fast = getenv("CGC_VERIFY_DECODE") != nullptr
        //                   && cgc_current_phase == CGC_PHASE_VERIFY
        //                   && ctx_type == DEFAULT
        // `cgc_current_phase` defaults to CGC_PHASE_UNKNOWN, and llama_context_set_cgc_phase had
        // exactly TWO callers in the whole tree: server-context.cpp:3897 and common/speculative.cpp
        // — and the latter only ever touches ctx_dft. So every target-context decode in THIS tool,
        // including this verify batch, fell to the exact ensure_batch path. Measured, same day,
        // same flags, both MTP on:
        //     llama-bench stderr : MTP fast path: calls=395   verify: calls=0     draft: calls=395
        //     llama-server log   : MTP fast path: calls=9497  verify: calls=8819   (18.69 experts/call)
        // `union/calls = 8.00` on the bench side is the tell: all 395 were single-token DRAFT
        // decodes. The tool was not measuring the server's MTP, it was measuring the exact-path
        // fallback — and it prints the same "-p 0 -n 128" either way, so nothing looked wrong.
        //
        // WHY UNCONDITIONAL VERIFY IS CORRECT FOR THIS BATCH: the batch built just above IS the
        // verify round — [id_last, draft0 .. draftN-1]. The server marks exactly the same thing at
        // server-context.cpp:520-524, where it pushes the BASE token into spec_i_batch first and
        // the drafts after it: a 1-token round (zero drafts accepted) is still VERIFY. The phase is
        // allowed to be n_tokens == 1 there (llama-context.cpp:6049).
        //
        // Set-then-decode is the same pairing the server uses (server-context.cpp:3897 -> :3904).
        llama_context_set_cgc_phase(ctx, CGC_PHASE_VERIFY);

        const int ret = llama_decode(ctx, batch);

        // Fail-closed reset, mirroring common/speculative.cpp:1861-1864 ("a missed marker can never
        // accidentally ZERO-map an exact-path batch"). Done BEFORE the error check so the failure
        // path cannot leak a stale VERIFY into the next rep: this context is reused across cells,
        // and test_gen()'s depth prefill (:2289) and single-token step (:2311) deliberately set no
        // phase — a leftover VERIFY would silently move those plain decode cells onto the fast path.
        llama_context_set_cgc_phase(ctx, CGC_PHASE_UNKNOWN);

        if (ret != 0) {
            fprintf(stderr, "%s: verify decode failed: ret=%d n_tokens=%d(pos %d..%d) n_ctx=%d n_past=%d draft=%zu n_batch=%d\n",
                    __func__, ret, (int) batch.n_tokens,
                    batch.n_tokens > 0 ? (int) batch.pos[0] : -1,
                    batch.n_tokens > 0 ? (int) batch.pos[batch.n_tokens - 1] : -1,
                    (int) llama_n_ctx(ctx), n_past, draft.size(), (int) llama_n_batch(ctx));
            llama_batch_free(batch);
            return false;
        }

        if (getenv("LLAMA_BENCH_SPEC_DBG") != nullptr) {
            fprintf(stderr, "SPECDBG round: n_done=%d n_past=%d draft=%zu pos_max=%d\n", n_done, n_past - 1, draft.size(),
                    (int) llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id));
        }

        // the round happened: it was verified whatever it commits (a replay round is still a
        // verification step by the server's accounting, and its drafts were still generated)
        n_rounds  += 1;
        n_drafted += draft.size();

        common_speculative_process(spec, batch);

        // Save the sampler state before sampling. The replay path below has to put it back, or the
        // re-verify samples a DIFFERENT token than the round that was rolled back (the RNG advanced)
        // and the same partial acceptance repeats -- observed as several consecutive
        // "SPECDBG partial-restore: n_past=12 ..." rounds that commit nothing. Both references do
        // this: speculative-simple.cpp:539-542 clones before sampling and moves it back at :588,
        // and llama-server keeps one per slot (server-context.cpp:4218 `common_sampler_copy`).
        common_sampler_ptr smpl_save(common_sampler_clone(smpl));

        std::vector<llama_token> ids;

        if (tree_on) {
            // ── tree walk ───────────────────────────────────────────────────────────────
            // Sample at the current node, accept, then follow the child whose token matches. The
            // first node with no matching child ends the path, and THAT sample is the emitted
            // token. The chain instead rolls the whole round back and re-verifies it (n_replay) --
            // pure waste -- which is what this axis removes. At W_eff==1 the node table is a single
            // chain and this walk must reduce to the chain's accept exactly; that is T1c.
            (void) smpl_save;   // the tree never rolls the sampler back, so the clone is unused here
            ids.reserve(draft.size() + 1);
            int cur     = 0;    // node index == batch slot
            int branchB = 0;
            orc_r_amA   = -1;   // 逐輪歸零（見上面的宣告）
            orc_r_amB   = -1;
            orc_r_p2A   = -1;
            orc_r_p2B   = -1;
            while (true) {
                const llama_token id = common_sampler_sample(smpl, ctx, cur, false);
                if (getenv("LLAMA_BENCH_SPEC_ORACLE") != nullptr) {
                    const float * lg = llama_get_logits_ith(ctx, cur);
                    if (lg != nullptr) {
                        llama_token am = 0;
                        for (int32_t v = 1; v < n_vocab; ++v) {
                            if (lg[v] > lg[(int) am]) { am = (llama_token) v; }
                        }
                        const bool at_root = (cur == 0);
                        const bool at_lv1  = !at_root && !tree_nodes[cur].children.empty();
                        if (at_root) {
                            n_orc_rows1 += 1;
                            n_orc_smp1  += (id == am);
                            orc_r_amA = 0;
                            orc_r_amB = 0;
                        } else if (at_lv1 && tree_nodes[cur].branch != 1) {
                            n_orc_rows2 += 1;
                            n_orc_smp2  += (id == am);
                            orc_r_p2A   = 0;   // 走到 A 的 level-1 列 ⇒ 本輪對 pos2 有觀測（預設未命中）
                        } else if (at_lv1) {
                            n_orc_rows2B += 1;
                            orc_r_p2B   = 0;
                        }
                        for (int c : tree_nodes[cur].children) {
                            if (tree_nodes[c].token != am) { continue; }
                            if (at_root) {
                                if (tree_nodes[c].branch == 1) { n_orc_amB1 += 1; orc_r_amB = 1; }
                                else                           { n_orc_amA1 += 1; orc_r_amA = 1; }
                            } else if (at_lv1) {
                                if (tree_nodes[c].branch == 1) { n_orc_amB2 += 1; orc_r_p2B = 1; }
                                else                           { n_orc_amA2 += 1; orc_r_p2A = 1; }
                            }
                        }
                    }
                }
                common_sampler_accept(smpl, id, true);
                ids.push_back(id);
                int next = -1;
                for (int c : tree_nodes[cur].children) {
                    if (tree_nodes[c].token == id) { next = c; break; }
                }
                if (next < 0) { break; }
                cur = next;
                if (tree_nodes[cur].branch == 1) { branchB = 1; }
            }
            n_tree_rounds += 1;
            if (branchB) { n_tree_branchB += 1; }
            if (cgc_tree_bgate > 0 || cgc_tree_rgate > 0) {
                // 共用的 alt 計數（`alt_rounds` 是「這一輪真的帶了 B」的輪數，任一閘門都算）。
                const bool rg_probe_round = (cgc_tree_rgate > 0) && rg_closed == 0 && rg_streak == 0 &&
                        n_rgate_closes > 0;
                if (cgc_alt_on) {
                    n_bgate_alt_rounds += 1;
                    if (branchB) { gt_wins += 1; n_bgate_alt_wins += 1; }
                }
                if (cgc_tree_bgate > 0) {
                    if (cgc_alt_on) {
                        if (--gt_left <= 0) {
                            if (gt_wins == 0) { gt_cool = cgc_tree_bgate * 4; }  // 整窗全敗 ⇒ 冷卻
                            gt_left = cgc_tree_bgate;
                            gt_wins = 0;
                        }
                    } else {
                        if (--gt_cool <= 0) { gt_cool = 0; gt_left = cgc_tree_bgate; gt_wins = 0; }
                    }
                }
                if (cgc_tree_rgate > 0) {
                    if (cgc_alt_on) {
                        if (branchB) {
                            if (rg_probe_round) { n_rgate_probe_wins += 1; }
                            rg_streak = 0;   // 勝一場就繼續探測（B 有價值時不要關它）
                        } else if (++rg_streak >= cgc_tree_rgate) {
                            rg_closed = cgc_tree_rprobe;
                            rg_streak = 0;
                            n_rgate_closes += 1;
                        }
                    } else {
                        n_rgate_closed_rounds += 1;
                        if (--rg_closed <= 0) { rg_closed = 0; }
                    }
                }
            }
            // [CGC spec-tree 2026-10-06 k-axis] 比的是「真的送進 verify 的深度」；未裁時
            // tree_depth_v == draft.size()，與原本一字不差。
            if ((int) ids.size() - 1 < tree_depth_v) { n_tree_path_partial += 1; }

            const int m      = (int) ids.size() - 1;                       // accepted drafts
            const int win_br = (m == 0) ? 0 : tree_nodes[cur].branch;      // branch of the winner

            // [CGC spec-tree 2026-10-06 shape·adaptive] 控制器結算（見 cgc_rows_adapt）。分母是**勝出
            // 分支實際進 verify 的尾長**（tail_len[win_br]），不是目標列數 —— 被 cap／draft 深度砍過
            // 的輪才不會讓接受率失真。尾長 0（唯一可能：d=0，自適應的值域不含）⇒ 沒有可量的訊號 ⇒
            // 這輪不更新（控制器停在原值，絕不靠猜測推進）。
            if (cgc_rows_adapt > 0 && cgc_rows_net) {
                // ── 淨收益目標（見 cgc_rows_net）────────────────────────────────────────────
                // 樣本記在**這一輪真的用的 d**（rows_d）上，用的是**實際**列數 rows_actual_v。
                if (net_m_win.empty()) {
                    net_m_win.assign((size_t) cgc_rows_adapt + 1, {});
                    net_r_win.assign((size_t) cgc_rows_adapt + 1, {});
                }
                const size_t d_used = (size_t) std::max(1, std::min(rows_d, cgc_rows_adapt));
                net_m_win[d_used].push_back(m);
                net_r_win[d_used].push_back(rows_actual_v);
                while ((int) net_m_win[d_used].size() > cgc_rows_win) {
                    net_m_win[d_used].erase(net_m_win[d_used].begin());
                    net_r_win[d_used].erase(net_r_win[d_used].begin());
                }
                // 政策：每 `dmax` 輪插一輪**探測**（round-robin 走過每個 d），其餘輪**利用**（argmax）。
                // 探測刻意**穿插**而不是全部排在最前面 —— 這個 cell 的頭幾輪 MTP 接受率還沒起來
                // （實測 m 連續為 0），把探測全塞在暖機段會讓每個 d 都量到 0 分 ⇒ argmax 退化成
                // 「列數最少的那個」= 探測本身把答案汙染掉。穿插後每個 d 的樣本才會落在不同階段。
                const bool probe_round = (n_net_samples % (size_t) cgc_rows_adapt) == 0;
                n_net_samples += 1;

                const int before = rows_d_adapt;
                if (probe_round) {
                    // 探測：沒有樣本的 d 不能參加評比，否則 argmax 等於「先量到誰就選誰」。
                    rows_d_adapt = 1 + (int) (n_net_probes % (size_t) cgc_rows_adapt);
                    n_net_probes += 1;
                } else {
                    // 利用：argmax_d mean(net)（定點：100*Σm − COST*Σrows，同一個窗內 n 可比；
                    // 不同 d 的 n 可能不同，故除以各自的 n）。同分取較小的 d。
                    int    best_d = 1;
                    double best_v = 0.0;
                    bool   have   = false;
                    for (int d = 1; d <= cgc_rows_adapt; ++d) {
                        const size_t n = net_m_win[(size_t) d].size();
                        if (n == 0) { continue; }
                        double sm = 0.0, sr = 0.0;
                        for (size_t q = 0; q < n; ++q) {
                            sm += (double) net_m_win[(size_t) d][q];
                            sr += (double) net_r_win[(size_t) d][q];
                        }
                        const double v = (100.0 * sm - (double) cgc_rows_cost * sr) / (double) n;
                        if (!have || v > best_v) { have = true; best_v = v; best_d = d; }
                    }
                    rows_d_adapt = best_d;
                }
                if (getenv("LLAMA_BENCH_SPEC_ROWS_DBG") != nullptr) {
                    fprintf(stderr, "CGC-ROWS-ROUND phase=%s round=%zu obj=net d_used=%zu rows_req=%d "
                            "rows=%d m=%d cost_pct=%d net100=%.1f win=%zu pick=%d\n",
                            phase, n_tree_rounds, d_used, rows_req, rows_actual_v, m,
                            cgc_rows_cost, 100.0 * m - (double) cgc_rows_cost * rows_actual_v,
                            n_net_samples, rows_d_adapt);
                }
                (void) before;
            } else if (cgc_rows_adapt > 0 && win_br >= 0 && win_br < (int) tail_len.size()) {
                const int rows_tail_win = tail_len[win_br];
                if (rows_tail_win > 0) {
                    const double rate = (double) m / (double) rows_tail_win;
                    rows_rate_win.push_back(rate);
                    rows_rate_sum += rate;
                    while ((int) rows_rate_win.size() > cgc_rows_win) {
                        rows_rate_sum -= rows_rate_win.front();
                        rows_rate_win.erase(rows_rate_win.begin());
                    }
                    const double mean = rows_rate_sum / (double) rows_rate_win.size();
                    const double thr  = (double) cgc_rows_thr / 100.0;
                    const int    before = rows_d_adapt;
                    if (mean > thr)      { rows_d_adapt = std::min(cgc_rows_adapt, rows_d_adapt + 1); }
                    else if (mean < thr) { rows_d_adapt = std::max(1, rows_d_adapt - 1); }
                    if (getenv("LLAMA_BENCH_SPEC_ROWS_DBG") != nullptr) {
                        fprintf(stderr, "CGC-ROWS-ROUND phase=%s round=%zu rows_req=%d rows=%d m=%d "
                                "tail=%d rate=%.4f win=%zu mean=%.4f thr=%.2f d=%d->%d\n",
                                phase, n_tree_rounds, rows_req, rows_actual_v, m, rows_tail_win,
                                rate, rows_rate_win.size(), mean, thr, before, rows_d_adapt);
                    }
                }
            }

            // [CGC oracle 2026-10-06 F1b] PER-ROUND 記錄（env `LLAMA_BENCH_SPEC_ORACLE`）。一輪一行 ⇒
            // 離線可算「本輪 winbr/m → 下一輪 amA」的鎖定相關、扣掉暖機輪、並用 alt_on/closed 標出
            // 這一輪到底帶不帶 B（閘門臂才看得出）。只在 root 真的看到 logits 時印，序列才可對齊。
            // [CGC oracle 2026-10-06 F1c] 新欄位一律**追加在尾端**（欄位是契約，見上面的教訓）：
            //   srcseq ＝ 本輪 carryover 來源 seq（上一輪 accept 的 src_seq_id；-1 ＝ 本 rep 第一輪）
            //   p2A/p2B＝ 本輪 branch-A／B 第二步 argmax 結果（-1 無觀測／0 未命中／1 命中）
            if (orc_r_amA >= 0 && getenv("LLAMA_BENCH_SPEC_ORACLE") != nullptr) {
                fprintf(stderr,
                        "CGC-ORC-ROUND phase=%s round=%zu m=%d winbr=%d amA=%d amB=%d alt_on=%d closed=%d "
                        "srcseq=%d p2A=%d p2B=%d\n",
                        phase, n_tree_rounds, m, win_br, orc_r_amA, orc_r_amB,
                        (int) cgc_alt_on, rg_closed, orc_src_seq, orc_r_p2A, orc_r_p2B);
            }

            // ── KV cleanup ──────────────────────────────────────────────────────────────
            // (a) drop the losing branch whole -- whole-seq removal never fails, recurrent included
            // (b) if the winner is branch 1, re-tag its cells onto the base seq, then drop the temp id
            // (c) trim the winning path's suffix. THIS needs partial removal: attention does ranges
            //     natively, but the GDN half needs per-token snapshots (n_rs_seq, see
            //     cgc_spec_tree_env()). The return value is COUNTED, never ignored -- a silent no-op
            //     would leave stale cells visible to the next round's attention, which is exactly the
            //     failure mode the CGC_SEQ_RM_TYPE_FULL note warns about.
            llama_memory_t mem = llama_get_memory(ctx);
            if (W_eff == 2) {
                const llama_seq_id lose = (llama_seq_id) (1 - win_br);
                llama_memory_seq_rm(mem, lose, -1, -1);
                if (win_br != (int) seq_id) {
                    llama_memory_seq_rm(mem, seq_id, -1, -1);
                    llama_memory_seq_cp(mem, (llama_seq_id) win_br, seq_id, -1, -1);
                    llama_memory_seq_rm(mem, (llama_seq_id) win_br, -1, -1);
                }
            }
            if (!llama_memory_seq_rm(mem, seq_id, P_base + m + 1, -1)) {
                n_tree_trim_fail += 1;
            }

            // [CGC spec-tree 2026-10-05] Mirror that trim on the DRAFT context. Its cells live in
            // its own memory ("shared" here means the draft layers read the target's KV; a target-
            // side seq_rm does not remove the cells the draft step wrote), and the impl's own per-
            // round cleanup is skipped under its `!is_mem_shared` guard. A round that accepts
            // NOTHING therefore leaves the REJECTED first draft's cell at P_base+1 -- exactly the
            // position the NEXT draft step has to write (its batch min position is P_base+m+1, so
            // P_base+1 when m == 0); llama-batch then refuses it (M-RoPE X < Y, X == Y) and that
            // round drafts nothing, i.e. it silently measures plain decode. Measured on tree_dbg2
            // 2026-10-05: every m == 0 round (4/35) was followed by exactly one such refusal, at
            // X == Y == 1, 3, 5, 26; rounds with m >= 1 never collide, because their stale cell
            // sits at an ACCEPTED position (inside the committed prefix). Keeping the draft prefix
            // == the committed prefix is exactly what the non-shared path enforces with its own
            // seq_rm before every draft step.
            if (s.ctx_dft) {
                llama_memory_t mem_dft = llama_get_memory(s.ctx_dft.get());
                // [CGC spec-tree 2026-10-06 F2] DRAFT-KV RETAG on a branch-B win (env
                // `CGC_SPEC_TREE_DKV=1`, 預設關 ⇒ 先 A/B 再談轉正）。
                //
                // 為什麼：draft 階段是「seq 0 = A 鏈、seq 1 = B 鏈」寫進 draft ctx 的 KV。B 勝時
                // committed 前綴在 P_base+1.. 是 **B 的** token，但 draft ctx 的 seq 0 在那些位置
                // 存的是 **A 的** token ⇒ 下一輪在 seq 0 上取的 hidden 是在一段與 committed 不一致
                // 的 KV 上算出來的，而 target 側並沒有這個問題（那裡有 lose/retag，:3518-3527）。
                // 把 B 的 cell 重貼到 seq 0（與 target 側同一動作），再走原本的 mirror trim。
                //
                // 只改 draft 側（預測用），committed KV 一個字都不碰 ⇒ 不會動到正確性見證。
                static const bool cgc_tree_dkv = []() {
                    const char * e = getenv("CGC_SPEC_TREE_DKV");
                    if (e == nullptr) { return false; }
                    return e[0] != '0';
                }();
                // ⚠ 第一版用**帶範圍**的 seq_cp（p0=P_base+1）會在 `llama-kv-cache.cpp:502` 斷言：
                // `seq_cp() is only supported for full KV buffers`。讀碼後知道原因：seq0/seq1 在
                // **不同 stream** ⇒ 必須真的搬 buffer，而帶範圍的請求會讓 `is_full=false`
                // （:493-501）。target 側之所以能過，是因為它傳的是 p0=p1=-1（整段）。
                // 所以這裡照 target 側一字不差地做：整段丟掉、整段複製、整段丟掉來源，
                // 再由下面那條 mirror trim 收掉未被接受的尾巴。
                if (cgc_tree_dkv && W_eff == 2 && win_br != (int) seq_id && m >= 1) {
                    llama_memory_seq_rm(mem_dft, seq_id, -1, -1);
                    llama_memory_seq_cp(mem_dft, (llama_seq_id) win_br, seq_id, -1, -1);
                    llama_memory_seq_rm(mem_dft, (llama_seq_id) win_br, -1, -1);
                    n_tree_dkv_retag += 1;
                }
                for (llama_seq_id sq = 0; sq < (llama_seq_id) W_eff; ++sq) {
                    llama_memory_seq_rm(mem_dft, sq, P_base + m + 1, -1);
                }
                // [CGC H2 2026-10-06] SEQ-1 DRAFT RESIDUE (env `CGC_SPEC_TREE_BRESET=1`，預設關).
                //
                // 為什麼要這一格（H2 的第一條**可 bench 側**的假說）：上面那圈只收 `sq < W_eff`。
                // 當 B 被關著（rgate／bgate：`cgc_alt_on == false` ⇒ `W_eff == 1`）時，**seq 1 的 cell
                // 一個都不收** —— 它停在最後一次 B-on 的位置，而 B 再開時 alt-seed 是在
                // `dp1.n_past = dp.n_past`（本輪 seq 0 的 n_past）上寫第一顆 ⇒ **可能落在那些殘留
                // 之後**。兩種後果都與 §10.3 的 M-RoPE 牆同族（llama-batch 拒絕同一位置的重複、
                // 或遮罩把殘留算進 attention）。
                //
                // 這條假說**預測**（寫死，跑之前先註冊）：
                //   1. `h_orcrgate40` 型（B 會關會開）：B-on 的 `pos1 amA` 由 0.486 往 ~0.845 回升，
                //      且 B-off 段內的 0.619 起步（暫態）消失；
                //   2. **B 恆開**的臂（`g_orc_ctrl`／`k_k2b`／`e2_k2`）→ **一字不動**（那裡 W_eff==2，
                //      上面那圈本來就收了 seq 1）⇒ 這是最強的可否證性檢查；
                //   3. 鏈臂（`alt` 從不開）→ 一字不動；`X < Y` 維持 0。
                // 若 (1) 不動而 (2) 也一字不動 ⇒ 殘留假說死，H2 的來源要往 verify 形狀或共用 hidden
                // row 去找（見 §13 的兩個混淆）。
                static const bool cgc_seq1_reset = []() {
                    const char * e = getenv("CGC_SPEC_TREE_BRESET");
                    if (e == nullptr) { return false; }
                    return e[0] != '0';
                }();
                if (cgc_seq1_reset && W_eff == 1) {
                    llama_memory_seq_rm(mem_dft, (llama_seq_id) 1, P_base + m + 1, -1);
                }
            }

            // commit in place -- no checkpoint, no replay.
            // [CGC spec-tree 2026-10-03] The hidden seed must come from the WINNING branch's
            // verify rows: branch 1 was verified on seq 1 while the continuation is seq 0.
            common_speculative_accept(spec, seq_id, (uint16_t) m,
                    W_eff == 2 ? (llama_seq_id) win_br : (llama_seq_id) -1);
            // [CGC oracle 2026-10-06 F1c] 記下「下一輪的 carryover 取自哪個 seq」：accept_from 用
            // src_seq_id=win_br 讀 verify 列當 pending_h 種子；W_eff==1 的鏈走 accept()，來源恆為
            // seq_id(=0)。這是逐輪 F1c 欄位 `srcseq` 的寫入點（值印在**下一輪**那一行）。
            orc_src_seq = (W_eff == 2) ? win_br : (int) seq_id;
            n_acc_drf += m;
            n_past    += m;
            if (getenv("LLAMA_BENCH_SPEC_DBG") != nullptr) {
                llama_memory_t mem_after = llama_get_memory(ctx);
                fprintf(stderr, "SPECDBG tree-cleanup: P_base=%d m=%d win_br=%d bound=%d n_past_new=%d pos_min0=%d pos_max0=%d\n",
                        P_base, m, win_br, P_base + m + 1, n_past,
                        (int) llama_memory_seq_pos_min(mem_after, seq_id),
                        (int) llama_memory_seq_pos_max(mem_after, seq_id));
            }
            n_done    += (int) ids.size();
            id_last    = ids.back();
            draft.clear();
            draft_alt.clear();
            if (cgc_spec_dump_on()) {
                emitted.insert(emitted.end(), ids.begin(), ids.end());
            }
            continue;
        }

        {
            std::vector<int> idxs(draft.size() + 1);
            for (size_t i = 0; i < idxs.size(); ++i) {
                idxs[i] = (int) i;
            }
            ids = common_sampler_sample_and_accept_n(smpl, ctx, idxs, draft, &draft_dist);
        }
        GGML_ASSERT(!ids.empty());

        // partial acceptance: the target decoded more positions than we are keeping, and this
        // context cannot remove a PARTIAL sequence (it reports SEQ_RM_TYPE_FULL -- see the
        // common_prompt_checkpoint comment above). So: restore the checkpoint taken before this
        // batch, and let the accepted tokens become the next draft -- they get re-verified from a
        // consistent state. Mirrors speculative-simple.cpp:570-593 and, for a server-shaped flow,
        // server-context.cpp:4186-4221, including the three things that are easy to get wrong: it
        // does NOT call common_speculative_accept (the tokens are not committed), it does NOT
        // advance n_past/n_done/id_last, and it DOES restore the sampler.
        if (ids.size() - 1 < draft.size() && getenv("LLAMA_BENCH_SPEC_OLDPARTIAL") == nullptr) {
            draft = std::move(ids);

            ckpt.load_tgt(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx), seq_id, ckpt.pos_max + 1, -1);

            // server-context.cpp:4321 discounts one token here: in a replay the accepted tokens
            // were counted as accepted on the round that first produced them (`ids` becomes the
            // next draft), so counting them again would inflate accept.
            n_acc_drf += ids.size() >= 2 ? ids.size() - 2 : 0;
            n_replay  += 1;

            common_sampler_copy(smpl_save.get(), smpl);

            n_past = (int) ckpt.n_tokens;

            if (getenv("LLAMA_BENCH_SPEC_DBG") != nullptr) {
                fprintf(stderr, "SPECDBG partial-restore: n_past=%d ids(as draft)=%zu pos_max=%d mem_pos_max=%d\n",
                        n_past, draft.size(), (int) ckpt.pos_max,
                        (int) llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id));
            }
            continue;
        }

        // The old, WRONG path: trim the unaccepted positions away. Kept behind an env so the two
        // mechanisms can be compared from one binary; it made the next llama_decode return -1.
        if (ids.size() - 1 < draft.size() && getenv("LLAMA_BENCH_SPEC_DBG") != nullptr) {
            fprintf(stderr, "SPECDBG partial-trim (OLDPARTIAL): ids=%zu draft=%zu\n", ids.size(), draft.size());
        }
        if (ids.size() - 1 < draft.size()) {
            llama_memory_seq_rm(llama_get_memory(ctx), seq_id, n_past + (llama_pos) ids.size() - 1, -1);
        }

        // full acceptance (or the OLDPARTIAL fallback): commit the tokens to the speculator, then
        // account for them. Accept is called HERE and not before the partial branch -- the reference
        // has it at speculative-simple.cpp:596, past the checkpoint path, so a restored round never
        // commits anything.
        common_speculative_accept(spec, seq_id, (uint16_t) (ids.size() - 1));

        n_acc_drf += ids.size() - 1;

        n_past += (int) ids.size() - 1;
        if (getenv("LLAMA_BENCH_SPEC_DBG") != nullptr) {
            fprintf(stderr, "SPECDBG chain-commit: n_past_new=%d pos_max=%d ids=%zu draft=%zu\n",
                    n_past, (int) llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id),
                    ids.size(), draft.size());
        }
        n_done += (int) ids.size();
        id_last = ids.back();

        draft.clear();

        // [CGC spec-tree 2026-10-02] chain-side dump, so the T1c witness compares like with like.
        // Replay rounds never reach here (they `continue` above), which is correct: a replay emits
        // nothing, and counting its carried `ids` would make the chain look like it emitted more.
        if (cgc_spec_dump_on()) {
            emitted.insert(emitted.end(), ids.begin(), ids.end());
        }
    }

    llama_synchronize(ctx);
    llama_batch_free(batch);

    // [CGC 2026-09-26] The witness itself. Unconditional on purpose: the neighbours are gated by
    // LLAMA_BENCH_SPEC_DBG / CGC_MTP_PERF, and an env-gated instrument is one the carrier can drop
    // silently -- this repo has already paid for that twice (the missing CGC_PHASE_VERIFY, and the
    // harness-bench route dropping DECPROF). Absence prints NA, never 0, because a 0.000 mean_len
    // and "no rounds happened" are different facts and this project has confused them before.
    // `phase` says which call this was: launchers run the spec loop twice per rep on --warm-skip
    // (once untimed, once timed), so an unlabelled line would be unattributable.
    {
        char s_mean_len[16];
        char s_ratio   [16];
        char s_emit    [16];
        if (n_rounds == 0) {
            snprintf(s_mean_len, sizeof(s_mean_len), "NA");
            snprintf(s_ratio,    sizeof(s_ratio),    "NA");
            snprintf(s_emit,     sizeof(s_emit),     "NA");
        } else {
            // mean_len is retained bit-for-bit (historical series depend on it) but it is NOT
            // emitted-per-round: a replay round adds 1 to the denominator and 0 to the numerator
            // while also emitting nothing. emit_per_round is the honest ratio (also exact against
            // n_done, which is measured, not derived).
            snprintf(s_mean_len, sizeof(s_mean_len), "%.4f", 1.0 + (double) n_acc_drf / (double) n_rounds);
            snprintf(s_ratio,    sizeof(s_ratio),    "%.5f", n_drafted ? (double) n_acc_drf / (double) n_drafted : 0.0);
            snprintf(s_emit,     sizeof(s_emit),     "%.4f", (double) n_done / (double) n_rounds);
        }
        fprintf(stderr,
                "CGC-BENCH-ACCEPT phase=%s rounds=%zu drafted=%zu acc_drafts=%zu mean_len=%s "
                "draft_ratio=%s gen_tokens=%d n_gen=%d replay=%zu emit_per_round=%s\n",
                phase, n_rounds, n_drafted, n_acc_drf, s_mean_len, s_ratio, n_done, n_gen,
                n_replay, s_emit);
    }

    // [CGC spec-tree 2026-10-02] The treatment witness, on its OWN tag (see the counters' note):
    // it must be readable whether the tree ran at all, whether a real branch was taken, how often
    // the path ended short (the rounds the chain would have replayed), and whether the KV trim
    // succeeded. Printed only when the gate was set, so a default run's stderr is unchanged.
    if (cgc_spec_tree_env() >= 0) {
        if (cgc_tree_bgate > 0 || cgc_tree_rgate > 0 || n_tree_dkv_retag > 0) {
            // 既有欄位順序不動，新欄位一律**追加在尾端**（見上面「欄位是契約」的教訓）。
            fprintf(stderr, "CGC-SPEC-TREE-GATE phase=%s bgate=%d alt_rounds=%zu alt_wins=%zu "
                    "dkv_retag=%zu rgate=%d rprobe=%d rclosed=%zu rcloses=%zu rpwin=%zu\n",
                    phase, cgc_tree_bgate, n_bgate_alt_rounds, n_bgate_alt_wins, n_tree_dkv_retag,
                    cgc_tree_rgate, cgc_tree_rprobe, n_rgate_closed_rounds, n_rgate_closes,
                    n_rgate_probe_wins);
        }
        // [CGC spec-tree 2026-10-06 shape·adaptive] 既有欄位順序不動，新欄位一律**追加在尾端**
        // （欄位是契約）。`rows_req` 改讀逐輪的 `rows_req_v`（靜態路徑下恰等於 cgc_tree_rows ⇒ 既有
        // 值一字不動）；`rows_hist`／`rows_kinds` 是「一趟真的用過哪些形狀」的一手證據，
        // `rows_switch` 是靜態形狀的**否定**證據（靜態 ⇒ 0）。
        std::string s_hist;
        size_t n_rows_kinds = 0;
        for (size_t i = 0; i < rows_hist.size(); ++i) {
            if (rows_hist[i] == 0) { continue; }
            char buf[48];
            snprintf(buf, sizeof(buf), "%s%zu:%zu", s_hist.empty() ? "" : ",", i, rows_hist[i]);
            s_hist += buf;
            n_rows_kinds += 1;
        }
        if (s_hist.empty()) { s_hist = "-"; }
        // [CGC spec-tree 2026-10-06 shape·net] 淨收益版的**逐 d 帳本**：`mean(payoff):mean(rows):n`。
        // 刻意報**原料**而不是算好的 net —— 讀者可以用任何 COST 重算 argmax，不必信控制器的算術。
        std::string s_net;
        if (cgc_rows_net && !net_m_win.empty()) {
            for (int d = 1; d <= cgc_rows_adapt; ++d) {
                const size_t n = net_m_win[(size_t) d].size();
                if (n == 0) { continue; }
                double sm = 0.0, sr = 0.0;
                for (size_t q = 0; q < n; ++q) {
                    sm += (double) net_m_win[(size_t) d][q];
                    sr += (double) net_r_win[(size_t) d][q];
                }
                char buf[80];
                snprintf(buf, sizeof(buf), "%s%d:%.3f:%.3f:%zu", s_net.empty() ? "" : ",",
                         d, sm / (double) n, sr / (double) n, n);
                s_net += buf;
            }
        }
        if (s_net.empty()) { s_net = "-"; }
        fprintf(stderr,
                "CGC-SPEC-TREE phase=%s rounds_tree=%zu branchB=%zu path_partial=%zu "
                "trim_fail=%zu replay=%zu tree_rows=%zu tree_k=%d "
                "rows_req=%d rows_d=%d rows_unmet=%zu altbind=%d "
                "adapt=%d adapt_max=%d adapt_thr=%d adapt_win=%d rows_switch=%zu rows_kinds=%zu "
                "rows_hist=%s net_obj=%d net_cost=%d net_pick=%d net_tab=%s\n",
                phase, n_tree_rounds, n_tree_branchB, n_tree_path_partial,
                n_tree_trim_fail, n_replay, n_tree_rows, cgc_tree_k,
                rows_req_v, rows_d_v, n_rows_unmet, cgc_altbind ? 1 : 0,
                cgc_rows_adapt, cgc_rows_adapt, cgc_rows_thr, cgc_rows_win,
                n_rows_switch, n_rows_kinds, s_hist.c_str(),
                cgc_rows_net ? 1 : 0, cgc_rows_cost, rows_d_adapt, s_net.c_str());
    }

    // [CGC oracle 2026-10-06] See the counter block above: head-vs-target-argmax agreement with
    // the sampler removed from the comparison, plus sampled == argmax on the same rows. Own tag,
    // env-gated, so no existing parser sees a new field.
    if (getenv("LLAMA_BENCH_SPEC_ORACLE") != nullptr && n_orc_rows1 > 0) {
        fprintf(stderr,
                "CGC-ORACLE phase=%s pos1_rows=%zu pos1_argmaxA=%zu pos1_argmaxB=%zu "
                "pos1_smp_eq_argmax=%zu | pos2_rowsA=%zu pos2_argmaxA=%zu pos2_smpA=%zu "
                "pos2_rowsB=%zu pos2_argmaxB=%zu\n",
                phase, n_orc_rows1, n_orc_amA1, n_orc_amB1, n_orc_smp1,
                n_orc_rows2, n_orc_amA2, n_orc_smp2, n_orc_rows2B, n_orc_amB2);
    }

    // [CGC spec-tree 2026-10-02] T1c: the committed token sequence, for a bit-for-bit diff between
    // a width-1 tree and the chain. Gated, so it never appears by default.
    if (cgc_spec_dump_on()) {
        fprintf(stderr, "CGC-SPEC-DUMP phase=%s n=%zu ids=", phase, emitted.size());
        for (size_t i = 0; i < emitted.size(); ++i) {
            fprintf(stderr, "%s%d", i ? "," : "", (int) emitted[i]);
        }
        fprintf(stderr, "\n");
    }

    common_speculative_print_stats(spec);
    return true;
}

// [CGC 2026-09-28] ERRORs are never silent, even without -v.
//
// This used to be a NULL sink, so llama-bench discarded EVERY llama.cpp log line -- including
// LLAMA_LOG_ERROR. Measured cost: a `-r 3` arm whose draft chain died on the first step of every
// round printed the bare `llama_decode[0] returned -1` 264 times, while the engine's own reason
// for that -1 (llama-batch.cpp's M-RoPE check, which names X and Y) went into this sink. The log
// showed a symptom with no cause, and the arm's avg_ts -- one speculative rep plus two PLAIN
// DECODE reps -- was read as a k result. A failure that cannot state its reason is a number
// wearing a verdict's clothes, and scripts/check/draft_liveness.py can only classify what the log
// actually contains, so the log has to contain it.
//
// INFO/DEBUG are still dropped; that is what keeps the default cheap enough not to push a tight
// prefill into Metal OOM (the -v concern, not the rare ERROR).
static void llama_error_only_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

static std::unique_ptr<printer> create_printer(output_formats format) {
    switch (format) {
        case NONE:
            return nullptr;
        case CSV:
            return std::unique_ptr<printer>(new csv_printer());
        case JSON:
            return std::unique_ptr<printer>(new json_printer());
        case JSONL:
            return std::unique_ptr<printer>(new jsonl_printer());
        case MARKDOWN:
            return std::unique_ptr<printer>(new markdown_printer());
        case SQL:
            return std::unique_ptr<printer>(new sql_printer());
    }
    GGML_ABORT("fatal error");
}

// satisfies -Wmissing-declarations
int llama_bench(int argc, char ** argv);

int llama_bench(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    // try to set locale for unicode characters in markdown
    std::setlocale(LC_CTYPE, ".UTF-8");

#if !defined(NDEBUG)
    fprintf(stderr, "warning: asserts enabled, performance may be affected\n");
#endif

#if (defined(_MSC_VER) && defined(_DEBUG)) || (!defined(_MSC_VER) && !defined(__OPTIMIZE__))
    fprintf(stderr, "warning: debug build, performance may be affected\n");
#endif

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    fprintf(stderr, "warning: sanitizer enabled, performance may be affected\n");
#endif

    // initialize backends
    ggml_backend_load_all();

    cmd_params params = parse_cmd_params(argc, argv);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        fprintf(stderr, "%s: error: CPU backend is not loaded\n", __func__);
        return 1;
    }
    auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free");

    // initialize llama.cpp
    // [CGC] CGC_KEEP_ERRORS keeps its historical meaning: install NO sink at all, i.e. keep INFO
    // and above (without paying for full --verbose DEBUG output, which itself can push a tight
    // prefill into Metal OOM). The DEFAULT is now "keep ERROR only" -- see
    // llama_error_only_log_callback for why a discarded ERROR is worse than a noisy one.
    if (!params.verbose) {
        if (getenv("CGC_KEEP_ERRORS") != nullptr) {
            fprintf(stderr, "[CGC] CGC_KEEP_ERRORS set: llama.cpp log sink NOT installed "
                            "(INFO and above will be printed)\n");
        } else {
            llama_log_set(llama_error_only_log_callback, NULL);
        }
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    if (!set_process_priority(params.prio)) {
        fprintf(stderr, "%s: error: failed to set process priority\n", __func__);
        return 1;
    }

    // initialize printer
    std::unique_ptr<printer> p     = create_printer(params.output_format);
    std::unique_ptr<printer> p_err = create_printer(params.output_format_stderr);

    if (p) {
        p->fout = stdout;
        p->print_header(params);
    }

    if (p_err) {
        p_err->fout = stderr;
        p_err->print_header(params);
    }

    std::vector<cmd_params_instance> params_instances = get_cmd_params_instances(params);

    llama_model *               lmodel    = nullptr;
    const cmd_params_instance * prev_inst = nullptr;

    // store the llama_context state at the previous depth that we performed a test
    // ref: https://github.com/ggml-org/llama.cpp/pull/16944#issuecomment-3478151721
    ctx_state cstate;

    int  params_idx   = 0;
    auto params_count = params_instances.size();
    for (const auto & inst : params_instances) {
        params_idx++;
        if (params.progress) {
            fprintf(stderr, "llama-bench: benchmark %d/%zu: starting\n", params_idx, params_count);
        }
        auto mparams = inst.to_llama_mparams();
        auto cparams = inst.to_llama_cparams();

        // [CGC spec-tree 2026-10-02] Any CGC_SPEC_TREE value (including "0") selects the context mode
        // the tree needs, so a tree arm and its chain control differ in ONE variable (the gate). The
        // two reasons are in cgc_spec_tree_env() above; both are disclosed costs, not free.
        //
        // n_rs_seq is a LINEAR memory cost -- llama_memory_recurrent allocates `mem_size * (1 +
        // n_rs_seq)` rows (:99) -- so it is the knob to shrink FIRST on a box with no headroom, and
        // the reason it is an env rather than a constant. It must be >= the deepest rollback the
        // tree can ask for, which is the tree depth = k = the draft count (llama-memory-recurrent
        // .cpp:184 refuses a larger rollback). Default 2 = the minimum for the depth-1 tree at the
        // delivery cell's k=2; measure before raising it. 2026-10-02: with 4 (and an 8 GiB pool) the
        // box -- already at ~15.5/16 GB -- went into swap and froze (see the daily log).
        if (cgc_spec_tree_env() >= 0) {
            cparams.kv_unified = true;
            cparams.n_rs_seq   = (uint32_t) cgc_spec_tree_rs();
            // [CGC spec-tree 2026-10-03] kv_unified ALONE IS NOT ENOUGH. A branch is a second seq,
            // and the allocator rejects any seq_id >= the CONTEXT's n_seq_max (llama-batch.cpp:59-63)
            // -- that check reads cparams.n_seq_max, NOT kv_unified (llama-context.cpp:120 takes it
            // from params.n_seq_max, default 1). Measured failure, width-2 arm, first round:
            //     decode: failed to initialize batch   (ret=-1, n_tokens=5 pos 512..514)
            // Width 1 only ever touches seq 0, so it passed and hid this. Both arms get the same
            // value so a tree A/B still moves exactly one variable.
            cparams.n_seq_max   = std::max(2u, (uint32_t) cgc_spec_tree_width());
        }

        bool do_fit = inst.fit_target != cmd_params_defaults.fit_params_target[0] ||
                      inst.fit_min_ctx != cmd_params_defaults.fit_params_min_ctx[0];

        std::vector<float> fit_tensor_split(llama_max_devices(), 0.0f);
        std::vector<llama_model_tensor_buft_override> fit_overrides(llama_max_tensor_buft_overrides(), {nullptr, nullptr});

        if (do_fit) {
            // free the previous model so fit sees full free VRAM
            if (lmodel) {
                llama_model_free(lmodel);
                lmodel    = nullptr;
                prev_inst = nullptr;
            }

            // use default n_gpu_layers and n_ctx so common_fit_params can adjust them
            mparams.n_gpu_layers          = llama_model_default_params().n_gpu_layers;
            mparams.tensor_split          = fit_tensor_split.data();
            mparams.tensor_buft_overrides = fit_overrides.data();
            cparams.n_ctx                 = 0;

            std::vector<size_t> margins(llama_max_devices(), inst.fit_target * 1024 * 1024);

            uint32_t n_ctx_needed = inst.n_prompt + inst.n_gen + inst.n_depth;
            cparams.n_ctx = std::max(cparams.n_ctx, n_ctx_needed);

            common_fit_params(inst.model.c_str(), &mparams, &cparams,
                fit_tensor_split.data(),
                fit_overrides.data(),
                margins.data(),
                inst.fit_min_ctx,
                params.verbose ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
       }

        // keep the same model between tests when possible
        if (!lmodel || !prev_inst || !inst.equal_mparams(*prev_inst)) {
            if (lmodel) {
                llama_model_free(lmodel);
            }

            lmodel = llama_model_load_from_file(inst.model.c_str(), mparams);
            if (lmodel == NULL) {
                fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
                return 1;
            }
            prev_inst = &inst;
        }

        llama_context * ctx = llama_init_from_model(lmodel, cparams);
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, inst.model.c_str());
            llama_model_free(lmodel);
            return 1;
        }

        test t(inst, lmodel, ctx);

        llama_memory_clear(llama_get_memory(ctx), false);

        // cool off before the test
        if (params.delay) {
            std::this_thread::sleep_for(std::chrono::seconds(params.delay));
        }

        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(t.n_threads);
        if (!parse_cpu_mask(t.cpu_mask, tpp.cpumask)) {
            fprintf(stderr, "%s: failed to parse cpu-mask: %s\n", __func__, t.cpu_mask.c_str());
            llama_free(ctx);
            llama_model_free(lmodel);
            exit(1);
        }
        tpp.strict_cpu = t.cpu_strict;
        tpp.poll       = t.poll;
        tpp.prio       = params.prio;

        struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
        if (!threadpool) {
            fprintf(stderr, "%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
            llama_free(ctx);
            llama_model_free(lmodel);
            exit(1);
        }

        llama_attach_threadpool(ctx, threadpool, NULL);

        // [CGC MTP instrument 2026-09-18] Build the speculative side BEFORE the target context
        // decodes anything. The warmup below already touches the expert cache and the per-rep depth
        // prefill definitely does; building the draft context after that is what made every
        // production-shaped run fail on its first verify batch. See bench_spec_state.
        const bool cgc_spec_on = !params.spec_types.empty();
        // [CGC lazy-draft 2026-09-27] The draft context + speculator built by bench_spec_setup()
        // are used ONLY by a GENERATE cell (test_gen_spec). A prompt-only cell (pp, t.n_gen==0)
        // never calls test_gen_spec, so setting the draft context up here reserved ~285 MiB of
        // Metal for nothing -- and that extra working set is what pushed the 2048-token prefill
        // past the Metal working-set limit (command buffer status 5 OOM at il=39; the off arm at
        // the same shape prefilled fine at ~8.2 GB). Build the speculative side only when this
        // test actually generates. The tg cell still sets it up before its depth prefill + gen.
        const bool spec_needed = cgc_spec_on && t.n_gen > 0;
        bench_spec_state spec_state;
        if (spec_needed &&
            !bench_spec_setup(spec_state, lmodel, ctx, params.spec_types, params.spec_draft_n_max)) {
            fprintf(stderr, "%s: error: failed to set up the speculative path\n", __func__);
            llama_free(ctx);
            llama_model_free(lmodel);
            exit(1);
        }

        // warmup run
        if (!params.no_warmup) {
            if (t.n_prompt > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: warmup prompt run\n", params_idx, params_count);
                }
                //test_prompt(ctx, std::min(t.n_batch, std::min(t.n_prompt, 32)), 0, t.n_batch, t.n_threads);
                bool res = test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run prompt warmup\n", __func__);
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    exit(1);
                }
            }
            if (t.n_gen > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: warmup generation run\n", params_idx, params_count);
                }
                bool res = test_gen(ctx, 1, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run gen warmup\n", __func__);
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    exit(1);
                }
            }
        }

        // [CGC 2026-09-19] --warm-skip: derived ONCE, before the loop. It must not be recomputed
        // from t.n_gen inside the loop, because t.n_gen is deliberately left alone there (see the
        // subtract after the loop) and mutating it mid-loop would shrink every later rep.
        const int n_warm_skip = (warm_skip_value() > 0 && t.n_gen > warm_skip_value()) ? warm_skip_value() : 0;

        for (int i = 0; i < params.reps; i++) {
            // [CGC 2026-09-19] Make every rep see the SAME fill stream. See the block comment on
            // g_fixed_fill_seed; 0 (default) keeps the historical advancing-stream behaviour.
            if (g_fixed_fill_seed > 0) {
                std::srand((unsigned) g_fixed_fill_seed);
            }

            llama_memory_clear(llama_get_memory(ctx), false);

            // [CGC C 2026-10-06] ... and clear the DRAFT context at the same point.
            //
            // The draft context is deliberately built ONCE per cell (bench_spec_state), but the
            // upstream reset above only touches the TARGET context. Left alone, `ctx_dft` keeps
            // seq_pos_max = n_gen-1 from the previous rep while the new rep's first draft batch
            // starts at position 0, so llama-batch refuses it (M-RoPE X < Y: seq 0 is Y=0 vs X=32,
            // seq 1 is Y=2 vs X=32) and that round drafts nothing. Reproduced on CPU
            // (-ngl 0 -p 32 -n 32 -r 3, CGC_SPEC_TREE=1 CGC_SPEC_TREE_B=2): one refusal pair at
            // EVERY rep boundary, 4/4.
            //
            // The impl DOES have a repair for exactly this (common/speculative.cpp:1545-1552,
            // added 2026-09-28), but it is unreachable here: it sits behind `begin()`'s
            // `if (N <= 0) return;` where N = prompt.size(), and this tool deliberately hands
            // begin() an EMPTY prompt (llama-bench.cpp:3228, `prompt_probe`). It is also
            // single-seq, so it could not have repaired the tree's seq 1 anyway. Clearing here is
            // the layer that owns "one rep = one fresh generation", and it covers every seq.
            if (spec_state.ctx_dft) {
                llama_memory_clear(llama_get_memory(spec_state.ctx_dft.get()), false);
            }

            if (t.n_depth > 0) {
                bool is_cached = t.n_depth == cstate.depth;

                if (is_cached) {
                    // if previously we have computed at this depth, just restore the state
                    const size_t ret = llama_state_seq_set_data(ctx, cstate.buf.data(), cstate.buf.size(), 0);
                    if (ret == 0) {
                        // if the old state is incompatible with the current context - reprocess from scratch
                        is_cached = false;
                    }
                }

                if (!is_cached) {
                    if (params.progress) {
                        fprintf(stderr, "llama-bench: benchmark %d/%zu: depth run %d/%d\n", params_idx, params_count,
                                i + 1, params.reps);
                    }
                    bool res = test_prompt(ctx, t.n_depth, t.n_batch, t.n_threads);
                    if (!res) {
                        fprintf(stderr, "%s: error: failed to run depth\n", __func__);
                        llama_free(ctx);
                        llama_model_free(lmodel);
                        exit(1);
                    }

                    // store the context state for reuse in later runs
                    cstate.depth = t.n_depth;
                    cstate.buf.resize(llama_state_seq_get_size(ctx, 0));
                    llama_state_seq_get_data(ctx, cstate.buf.data(), cstate.buf.size(), 0);
                } else {
                    if (params.progress) {
                        fprintf(stderr, "llama-bench: benchmark %d/%zu: depth run %d/%d (cached)\n", params_idx, params_count,
                                i + 1, params.reps);
                    }
                }
            }

            uint64_t t_start    = get_time_ns();
            uint64_t t_warm_ns  = 0;   // [CGC 2026-09-19] --warm-skip; subtracted from t_ns below

            if (t.n_prompt > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: prompt run %d/%d\n", params_idx, params_count,
                            i + 1, params.reps);
                }
                bool res = test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run prompt\n", __func__);
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    exit(1);
                }
            }
            if (t.n_gen > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: generation run %d/%d\n", params_idx, params_count,
                            i + 1, params.reps);
                }
                // [CGC MTP instrument 2026-09-17] `--spec-type draft-mtp` selects the speculative
                // generation path (test_gen_spec). It stays a MODE rather than a registered cell so
                // that the existing `decode` cell keeps its shape bit-for-bit -- it must stay
                // comparable with upstream's `tg128 @ d512` row. Absent the flag ⇒ previous
                // behaviour, which is why `--cells decode` measures MTP-off by construction.
                // `cgc_spec_on` and `spec_state` are built ABOVE, before the warmup -- see the block
                // comment on bench_spec_state for why the draft context cannot be built here.
                // [CGC 2026-09-19] --warm-skip: the warm-up generation is a SEPARATE call. It has
                // to be: the plateau is a property of the generation's own progression, so the run
                // cannot be skipped by shortening the depth fill (that is a different state, and it
                // is what `-d` already controls). Its duration is measured here and subtracted from
                // t_ns, and its token count is removed from t.n_gen after the loop.
                if (n_warm_skip > 0) {
                    const uint64_t w0   = get_time_ns();
                    const bool     wres = cgc_spec_on ? test_gen_spec(ctx, lmodel, n_warm_skip, t.n_threads, spec_state, "warm_skip")
                                                      : test_gen(ctx, n_warm_skip, t.n_threads);
                    if (!wres) {
                        fprintf(stderr, "%s: error: failed to run warm-skip gen\n", __func__);
                        llama_free(ctx);
                        llama_model_free(lmodel);
                        exit(1);
                    }
                    t_warm_ns = get_time_ns() - w0;
                }
                bool res = cgc_spec_on ? test_gen_spec(ctx, lmodel, t.n_gen - n_warm_skip, t.n_threads, spec_state, "timed")
                                       : test_gen(ctx, t.n_gen - n_warm_skip, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run gen\n", __func__);
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    exit(1);
                }
            }

            uint64_t t_ns = get_time_ns() - t_start - t_warm_ns;
            t.samples_ns.push_back(t_ns);
        }

        // [CGC 2026-09-19] --warm-skip: the timed region covered only (n_gen - n_warm_skip) tokens,
        // so the reported n_gen has to say so -- otherwise `avg_ts` (= n_tokens / ns) would divide a
        // short interval into a long token count and overstate the result. Done HERE, after the loop,
        // so the loop's own calls all still derived from the original value.
        t.n_gen -= n_warm_skip;

        if (p) {
            p->print_test(t);
            fflush(p->fout);
        }

        if (p_err) {
            p_err->print_test(t);
            fflush(p_err->fout);
        }

        llama_perf_context_print(ctx);

        // [CGC MTP instrument 2026-09-18] Release the draft context BEFORE the target context: the
        // speculator's MTP impl holds a pointer to it, and spec_state's destructor alone would run
        // after this llama_free(ctx). The failure paths above call exit(1), so this is the only
        // place where the order is observable.
        spec_state.reset();

        llama_free(ctx);

        ggml_threadpool_free_fn(threadpool);
    }

    llama_model_free(lmodel);

    if (p) {
        p->print_footer();
    }

    if (p_err) {
        p_err->print_footer();
    }

    llama_backend_free();

    return 0;
}
