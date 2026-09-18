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
};

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

    // This instrument implements draft-mtp and nothing else. Saying so and stopping is the point:
    // falling back silently to the plain gen cell would emit a throughput number under a command
    // line that claims it measured the speculative path.
    for (const auto t : params.spec_types) {
        if (t != COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            fprintf(stderr, "error: --spec-type %s is not implemented by this instrument (only draft-mtp)\n",
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

        cparams.n_ctx           = n_prompt + n_gen + n_depth;
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
            "platform_ts",    "n_kept"
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
        if (field == "n_kept") {
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
                                            std::to_string(n_kept()) };
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
        tokens[0]    = n_processed == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
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

    for (int i = 0; i < n_gen; i++) {
        int res = llama_decode(ctx, llama_batch_get_one(&token, 1));
        if (res != 0) {
            fprintf(stderr, "%s: failed to decode generation batch, res = %d\n", __func__, res);
            return false;
        }
        llama_synchronize(ctx);
        token = std::rand() % n_vocab;
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
    s.init = common_speculative_init_from_params(s.params, model, ctx);
    if (!s.init) {
        fprintf(stderr, "%s: failed to initialise the MTP draft context\n", __func__);
        return false;
    }

    // [CGC MTP instrument 2026-09-17] The draft context has to be HANDED to the speculator, not
    // just created. common_speculative_init() enables DRAFT_MTP only when
    // `params.draft.ctx_dft != nullptr` (speculative.cpp:2633) -- with it null the constructor
    // builds an empty impl list and returns nullptr, which is what "failed to initialise the
    // speculative path" was. Mirrors examples/speculative-simple/speculative-simple.cpp:142
    // (`ctx_dft.reset(spec_init->release_context())`) and :181
    // (`params.speculative.draft.ctx_dft = ctx_dft.get()`). release_context() transfers ownership
    // out of the holder, so the pointer must be owned and freed here.
    s.ctx_dft.reset(s.init->release_context());
    if (!s.ctx_dft) {
        fprintf(stderr, "%s: the MTP draft context could not be released\n", __func__);
        return false;
    }
    // [CGC MTP instrument 2026-09-17] BOTH sides have to be handed over, not just the draft one.
    // common_speculative_impl_draft_mtp asserts `ctx_tgt && ctx_dft` (speculative.cpp:1318), so with
    // ctx_tgt left null the next thing you see is an abort inside the impl ctor. Mirrors
    // examples/speculative-simple/speculative-simple.cpp:180-181:
    //     params.speculative.draft.ctx_tgt = ctx_tgt;
    //     params.speculative.draft.ctx_dft = ctx_dft.get();
    s.params.speculative.draft.ctx_tgt = ctx;
    s.params.speculative.draft.ctx_dft = s.ctx_dft.get();

    s.spec.reset(common_speculative_init(s.params.speculative, 1));
    if (!s.spec) {
        fprintf(stderr, "%s: failed to initialise the speculative path\n", __func__);
        return false;
    }

    s.smpl.reset(common_sampler_init(model, s.params.sampling));
    if (!s.smpl) {
        fprintf(stderr, "%s: failed to initialise the sampler\n", __func__);
        return false;
    }

    return true;
}

static bool test_gen_spec(llama_context * ctx, llama_model * model, int n_gen, int n_threads,
                          bench_spec_state & s) {
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
    llama_tokens draft;

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

    struct llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);

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

    common_speculative_begin(spec, seq_id, prompt);

    while (n_done < n_gen) {
        if (draft.empty()) {
            // The two halves of the checkpoint are taken at DIFFERENT points, exactly as the
            // reference does it: update_pos before the draft (speculative-simple.cpp:421) and
            // update_tgt after it (:452). The order matters for MTP because the draft context
            // SHARES KV with the target, so by the time update_tgt runs the draft step has already
            // written into the same memory -- and ckpt.pos_max stays the PRE-draft boundary.
            ckpt.update_pos(n_past,
                    llama_memory_seq_pos_min(llama_get_memory(ctx), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id));

            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting = */ true,
                /* .n_max    = */ -1,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt,
                /* .result   = */ &draft,
                /* .dist     = */ nullptr,
            };
            common_speculative_draft(spec);

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

        // verify batch: [id_last, draft0 .. draftN-1]
        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch, draft[i], n_past + (llama_pos) i, { seq_id }, true);
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
            fprintf(stderr, "SPECDBG round: n_done=%d n_past=%d draft=%zu\n", n_done, n_past - 1, draft.size());
        }

        common_speculative_process(spec, batch);

        // Save the sampler state before sampling. The replay path below has to put it back, or the
        // re-verify samples a DIFFERENT token than the round that was rolled back (the RNG advanced)
        // and the same partial acceptance repeats -- observed as several consecutive
        // "SPECDBG partial-restore: n_past=12 ..." rounds that commit nothing. Both references do
        // this: speculative-simple.cpp:539-542 clones before sampling and moves it back at :588,
        // and llama-server keeps one per slot (server-context.cpp:4218 `common_sampler_copy`).
        common_sampler_ptr smpl_save(common_sampler_clone(smpl));

        std::vector<int> idxs(draft.size() + 1);
        for (size_t i = 0; i < idxs.size(); ++i) {
            idxs[i] = (int) i;
        }

        auto ids = common_sampler_sample_and_accept_n(smpl, ctx, idxs, draft, nullptr);
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

            common_sampler_copy(smpl_save.get(), smpl);

            n_past = (int) ckpt.n_tokens;

            if (getenv("LLAMA_BENCH_SPEC_DBG") != nullptr) {
                fprintf(stderr, "SPECDBG partial-restore: n_past=%d ids(as draft)=%zu pos_max=%d\n",
                        n_past, draft.size(), (int) ckpt.pos_max);
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

        n_past += (int) ids.size() - 1;
        n_done += (int) ids.size();
        id_last = ids.back();

        draft.clear();
    }

    llama_synchronize(ctx);
    llama_batch_free(batch);

    common_speculative_print_stats(spec);
    return true;
}

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
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
    if (!params.verbose) {
        llama_log_set(llama_null_log_callback, NULL);
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
        bench_spec_state spec_state;
        if (cgc_spec_on &&
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

        for (int i = 0; i < params.reps; i++) {
            llama_memory_clear(llama_get_memory(ctx), false);

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

            uint64_t t_start = get_time_ns();

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
                bool res = cgc_spec_on ? test_gen_spec(ctx, lmodel, t.n_gen, t.n_threads, spec_state)
                                       : test_gen(ctx, t.n_gen, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run gen\n", __func__);
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    exit(1);
                }
            }

            uint64_t t_ns = get_time_ns() - t_start;
            t.samples_ns.push_back(t_ns);
        }

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
