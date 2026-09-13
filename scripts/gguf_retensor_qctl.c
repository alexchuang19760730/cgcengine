// gguf_retensor_qctl.c -- numeric backend for scripts/gguf_retensor.py
//
// WHY THIS EXISTS
// ---------------
// gguf-py's quantizer only implements a handful of types (F16/F32/Q4_0/Q5_0/Q5_1/Q8_0/Q6_K...).
// Asking it for IQ4_XS, IQ2_K, Q2_K or Q3_K raises NotImplementedError -- and hand-written Python
// reimplementations would silently disagree with the kernels that actually produced the model.
// ggml's own encoders are the only ground truth, so this shim links against libggml-base and
// exposes them as a small subprocess API. All numbers are produced here; scripts/gguf_retensor.py
// only does layout (offsets, header fields, backup/restore).
//
// MODES
// -----
//   qctl types
//       Print one row per ggml type:
//         id \t name \t blck_size \t type_size \t to_float \t from_float \t is_quantized
//       "to_float" / "from_float" are 0/1 and tell the caller which conversions are possible.
//       The type table comes from ggml itself, so new types appear here automatically.
//
//   qctl cast <dst_type> <in> <src_type> <out> <nrows> <n_per_row>
//       Generic src -> f32 -> dst conversion. Handles F32/F16/BF16 directly via ggml's row
//       converters and everything else through the quantize/dequantize traits. This is the mode
//       the repacker uses, so it never has to know which types are "special".
//
//   qctl deq   <type> <in> <out.f32> <nrows> <n_per_row>
//   qctl quant <type> <in.f32> <out> <nrows> <n_per_row>
//       The two halves of `cast`, kept for convenience and for micro-benchmarks.
//
// CONVENTIONS
// -----------
//   n_per_row = ggml ne[0]; nrows = product(ne[1:]). With that mapping the ggml data region is
//   byte-identical to numpy's C-order reshape (nrows, n_per_row), which is what makes
//   `arr.tofile()` / `np.fromfile()` valid on either side of the pipe.
//
// BUILD
// -----
//   cc -O2 -o qctl gguf_retensor_qctl.c \
//      -I<repo>/src/llama.cpp/ggml/include \
//      -L<repo>/src/llama.cpp/build/bin -lggml-base -Wl,-rpath,<repo>/src/llama.cpp/build/bin
//   (scripts/gguf_retensor.py builds this automatically on first use.)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"

static void usage(void) {
    fprintf(stderr,
            "usage:\n"
            "  qctl types\n"
            "  qctl cast  <dst_type> <in> <src_type> <out> <nrows> <n_per_row>\n"
            "  qctl deq   <type> <in> <out.f32> <nrows> <n_per_row>\n"
            "  qctl quant <type> <in.f32> <out> <nrows> <n_per_row>\n"
            "\n"
            "type may be a ggml type id or a name (see `qctl types`).\n");
}

static enum ggml_type parse_type(const char * s) {
    char * end = NULL;
    const long id = strtol(s, &end, 10);
    if (end != s && *end == '\0') {
        if (id < 0 || id >= GGML_TYPE_COUNT) {
            fprintf(stderr, "qctl: type id %ld out of range\n", id);
            exit(3);
        }
        return (enum ggml_type) id;
    }
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const enum ggml_type t = (enum ggml_type) i;
        const char * n = ggml_type_name(t);
        if (n != NULL && strcasecmp(n, s) == 0) {
            return t;
        }
    }
    fprintf(stderr, "qctl: unknown type '%s'\n", s);
    exit(3);
}

// src -> f32 for one row-chunk. Only types with a to_float trait can be read back.
static void to_f32(enum ggml_type t, const void * src, float * dst, int64_t nrows, int64_t n_per_row) {
    if (t == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t) nrows * n_per_row * sizeof(float));
        return;
    }
    if (t == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, nrows * n_per_row);
        return;
    }
    if (t == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, nrows * n_per_row);
        return;
    }
    const struct ggml_type_traits * tr = ggml_get_type_traits(t);
    if (tr == NULL || tr->to_float == NULL) {
        fprintf(stderr, "qctl: no to_float for type %s\n", ggml_type_name(t));
        exit(11);
    }
    // quantized types are block-based: convert per row, converting whole blocks.
    const int64_t row = ggml_row_size(t, n_per_row);
    for (int64_t r = 0; r < nrows; ++r) {
        tr->to_float((const char *) src + (size_t) r * row, dst + r * n_per_row, n_per_row);
    }
}

// f32 -> dst for one row-chunk.
static size_t from_f32(enum ggml_type t, const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const int64_t n = nrows * n_per_row;
    if (t == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t) n * sizeof(float));
        return (size_t) n * sizeof(float);
    }
    if (t == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) dst, n);
        return (size_t) n * sizeof(ggml_fp16_t);
    }
    if (t == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(src, (ggml_bf16_t *) dst, n);
        return (size_t) n * sizeof(ggml_bf16_t);
    }
    if (ggml_quantize_requires_imatrix(t)) {
        // Not fatal: ggml's encoders fall back to a default importance when imatrix is NULL, but
        // the result is not the same as a calibrated quantize, so say so out loud.
        fprintf(stderr, "qctl: warning: %s normally requires an importance matrix; "
                        "quantizing without one\n", ggml_type_name(t));
    }
    ggml_quantize_init(t);
    const size_t written = ggml_quantize_chunk(t, src, dst, 0, nrows, n_per_row, NULL);
    if (written != ggml_row_size(t, n_per_row) * (size_t) nrows) {
        fprintf(stderr, "qctl: quantize_chunk wrote %zu, expected %zu\n",
                written, ggml_row_size(t, n_per_row) * (size_t) nrows);
        exit(9);
    }
    return written;
}

static void * read_all(const char * path, size_t nbytes) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) { perror("qctl: open input"); exit(5); }
    void * buf = malloc(nbytes);
    if (buf == NULL) { fprintf(stderr, "qctl: oom (%zu bytes)\n", nbytes); exit(8); }
    const size_t got = fread(buf, 1, nbytes, f);
    fclose(f);
    if (got != nbytes) {
        fprintf(stderr, "qctl: short read %zu of %zu from %s\n", got, nbytes, path);
        exit(6);
    }
    return buf;
}

static void write_all(const char * path, const void * buf, size_t nbytes) {
    FILE * f = fopen(path, "wb");
    if (f == NULL) { perror("qctl: open output"); exit(7); }
    if (fwrite(buf, 1, nbytes, f) != nbytes) { fprintf(stderr, "qctl: short write\n"); exit(10); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(); return 2; }
    const char * mode = argv[1];

    if (strcmp(mode, "types") == 0) {
        printf("id\tname\tblck_size\ttype_size\tto_float\tfrom_float\tis_quantized\n");
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const enum ggml_type t = (enum ggml_type) i;
            const struct ggml_type_traits * tr = ggml_get_type_traits(t);
            if (tr == NULL || tr->type_name == NULL) continue;
            printf("%d\t%s\t%lld\t%zu\t%d\t%d\t%d\n", i, tr->type_name,
                   (long long) ggml_blck_size(t), ggml_type_size(t),
                   tr->to_float   != NULL, tr->from_float_ref != NULL, tr->is_quantized);
        }
        return 0;
    }

    if (strcmp(mode, "cast") == 0) {
        if (argc < 8) { usage(); return 2; }
        const enum ggml_type dst_t = parse_type(argv[2]);
        const char * in_path  = argv[3];
        const enum ggml_type src_t = parse_type(argv[4]);
        const char * out_path = argv[5];
        const int64_t nrows = atoll(argv[6]);
        const int64_t n_per_row = atoll(argv[7]);
        const int64_t n = nrows * n_per_row;

        if (ggml_blck_size(src_t) > 1 && n_per_row % ggml_blck_size(src_t) != 0) {
            fprintf(stderr, "qctl: src %s: n_per_row %lld not a multiple of %lld\n",
                    ggml_type_name(src_t), (long long) n_per_row, (long long) ggml_blck_size(src_t));
            return 4;
        }
        if (ggml_blck_size(dst_t) > 1 && n_per_row % ggml_blck_size(dst_t) != 0) {
            fprintf(stderr, "qctl: dst %s: n_per_row %lld not a multiple of %lld\n",
                    ggml_type_name(dst_t), (long long) n_per_row, (long long) ggml_blck_size(dst_t));
            return 4;
        }

        const size_t src_bytes = ggml_row_size(src_t, n_per_row) * (size_t) nrows;
        const size_t dst_bytes = ggml_row_size(dst_t, n_per_row) * (size_t) nrows;
        void  * src = read_all(in_path, src_bytes);
        float * f32 = malloc((size_t) n * sizeof(float));
        void  * dst = malloc(dst_bytes);
        if (f32 == NULL || dst == NULL) { fprintf(stderr, "qctl: oom\n"); return 8; }

        to_f32(src_t, src, f32, nrows, n_per_row);
        const size_t written = from_f32(dst_t, f32, dst, nrows, n_per_row);
        write_all(out_path, dst, written);

        fprintf(stderr, "qctl: cast %s -> %s  nrows=%lld n_per_row=%lld  %zu -> %zu bytes\n",
                ggml_type_name(src_t), ggml_type_name(dst_t),
                (long long) nrows, (long long) n_per_row, src_bytes, written);
        free(src); free(f32); free(dst);
        ggml_quantize_free();
        return 0;
    }

    if (argc < 7) { usage(); return 2; }
    const enum ggml_type t = parse_type(argv[2]);
    const char * in_path  = argv[3];
    const char * out_path = argv[4];
    const int64_t nrows = atoll(argv[5]);
    const int64_t n_per_row = atoll(argv[6]);
    const int64_t n = nrows * n_per_row;

    if (ggml_blck_size(t) > 1 && n_per_row % ggml_blck_size(t) != 0) {
        fprintf(stderr, "qctl: %s: n_per_row %lld not a multiple of %lld\n",
                ggml_type_name(t), (long long) n_per_row, (long long) ggml_blck_size(t));
        return 4;
    }

    if (strcmp(mode, "deq") == 0) {
        const size_t nbytes = ggml_row_size(t, n_per_row) * (size_t) nrows;
        void  * raw = read_all(in_path, nbytes);
        float * f32 = malloc((size_t) n * sizeof(float));
        if (f32 == NULL) { fprintf(stderr, "qctl: oom\n"); return 8; }
        to_f32(t, raw, f32, nrows, n_per_row);
        write_all(out_path, f32, (size_t) n * sizeof(float));
        fprintf(stderr, "qctl: deq %s nrows=%lld n_per_row=%lld nbytes=%zu\n",
                ggml_type_name(t), (long long) nrows, (long long) n_per_row, nbytes);
        free(raw); free(f32);
        return 0;
    }

    if (strcmp(mode, "quant") == 0) {
        float * f32 = (float *) read_all(in_path, (size_t) n * sizeof(float));
        void  * raw = malloc(ggml_row_size(t, n_per_row) * (size_t) nrows);
        if (raw == NULL) { fprintf(stderr, "qctl: oom\n"); return 8; }
        const size_t written = from_f32(t, f32, raw, nrows, n_per_row);
        write_all(out_path, raw, written);
        fprintf(stderr, "qctl: quant %s nrows=%lld n_per_row=%lld nbytes=%zu\n",
                ggml_type_name(t), (long long) nrows, (long long) n_per_row, written);
        free(f32); free(raw);
        ggml_quantize_free();
        return 0;
    }

    fprintf(stderr, "qctl: unknown mode '%s'\n", mode);
    usage();
    return 3;
}
