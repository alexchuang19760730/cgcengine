// cgc_safetensors.c — 解析 safetensors header (C 实现)
//
// 移植自 turbo-fieldfare 的 Safetensors.swift
// safetensors 格式:
//   [8 bytes LE: header_size]
//   [header_size bytes: JSON]
//   [payload: tensor data]
//
// JSON 每个 tensor entry:
//   "name": { "dtype": "BF16", "shape": [n,m], "data_offsets": [begin, end] }

#include "cgc_repack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

// ============================================================================
// 简易 JSON 解析 (只解析 safetensors header 的扁平结构)
// ============================================================================

typedef struct {
    const char* json;
    size_t len;
    size_t pos;
} json_parser_t;

static void json_skip_ws(json_parser_t* p) {
    while (p->pos < p->len) {
        char c = p->json[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->pos++;
        else break;
    }
}

static bool json_match(json_parser_t* p, char c) {
    json_skip_ws(p);
    if (p->pos < p->len && p->json[p->pos] == c) {
        p->pos++;
        return true;
    }
    return false;
}

static bool json_expect(json_parser_t* p, char c) {
    if (!json_match(p, c)) {
        fprintf(stderr, "[safetensors] JSON parse error at pos %zu: expected '%c' got '%c'\n",
                p->pos, c, p->pos < p->len ? p->json[p->pos] : '?');
        return false;
    }
    return true;
}

// 解析双引号字符串 (输出到 out_buf, 不超过 buf_size-1)
static bool json_parse_string(json_parser_t* p, char* out_buf, size_t buf_size) {
    json_skip_ws(p);
    if (!json_expect(p, '"')) return false;
    size_t i = 0;
    while (p->pos < p->len && p->json[p->pos] != '"') {
        char c = p->json[p->pos++];
        if (c == '\\' && p->pos < p->len) {
            char esc = p->json[p->pos++];
            switch (esc) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                default: c = esc; break;
            }
        }
        if (i + 1 < buf_size) out_buf[i++] = c;
    }
    if (!json_expect(p, '"')) return false;
    out_buf[i] = '\0';
    return true;
}

// 解析数字 (int64 或 double)
static bool json_parse_number(json_parser_t* p, double* out_val) {
    json_skip_ws(p);
    size_t start = p->pos;
    if (p->pos < p->len && (p->json[p->pos] == '-' || p->json[p->pos] == '+')) p->pos++;
    while (p->pos < p->len) {
        char c = p->json[p->pos];
        if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
            p->pos++;
        } else break;
    }
    if (p->pos == start) return false;
    char buf[64];
    size_t n = p->pos - start;
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    memcpy(buf, p->json + start, n);
    buf[n] = '\0';
    *out_val = strtod(buf, NULL);
    return true;
}

static bool json_parse_int(json_parser_t* p, int64_t* out_val) {
    double d;
    if (!json_parse_number(p, &d)) return false;
    *out_val = (int64_t)d;
    return true;
}

// 跳过当前 JSON 值 (用于跳过 __metadata__ 等)
static bool json_skip_value(json_parser_t* p) {
    json_skip_ws(p);
    if (p->pos >= p->len) return false;
    char c = p->json[p->pos];
    if (c == '"') {
        char dummy[16];
        return json_parse_string(p, dummy, sizeof(dummy));
    }
    if (c == '{') {
        int depth = 0;
        while (p->pos < p->len) {
            char ch = p->json[p->pos++];
            if (ch == '{') depth++;
            else if (ch == '}') { depth--; if (depth == 0) return true; }
        }
        return false;
    }
    if (c == '[') {
        int depth = 0;
        while (p->pos < p->len) {
            char ch = p->json[p->pos++];
            if (ch == '[') depth++;
            else if (ch == ']') { depth--; if (depth == 0) return true; }
        }
        return false;
    }
    // number / true / false / null
    double d;
    return json_parse_number(p, &d);
}

// ============================================================================
// 文件读取辅助
// ============================================================================

static uint64_t file_get_size(const char* path) {
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &fad)) return 0;
    return ((uint64_t)fad.nFileSizeHigh << 32) | fad.nFileSizeLow;
#else
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (uint64_t)st.st_size;
#endif
}

static uint64_t read_u64_le(const void* p) {
    const uint8_t* b = (const uint8_t*)p;
    return ((uint64_t)b[0]) | ((uint64_t)b[1] << 8) | ((uint64_t)b[2] << 16) |
           ((uint64_t)b[3] << 24) | ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
}

// ============================================================================
// safetensors header 解析
// ============================================================================

cgc_source_tensor_t* cgc_safetensors_parse(const char* shard_path, int* out_count) {
    *out_count = 0;
    uint64_t file_size = file_get_size(shard_path);
    if (file_size == 0) {
        fprintf(stderr, "[safetensors] file not found or empty: %s\n", shard_path);
        return NULL;
    }

    FILE* f = fopen(shard_path, "rb");
    if (!f) {
        fprintf(stderr, "[safetensors] cannot open: %s\n", shard_path);
        return NULL;
    }

    // 读 header size (8 bytes LE)
    uint8_t header_size_buf[8];
    if (fread(header_size_buf, 1, 8, f) != 8) {
        fprintf(stderr, "[safetensors] failed to read header size\n");
        fclose(f);
        return NULL;
    }
    uint64_t header_size = read_u64_le(header_size_buf);
    if (header_size > (1ULL << 24)) {  // 16MB 上限
        fprintf(stderr, "[safetensors] header too large: %llu\n",
                (unsigned long long)header_size);
        fclose(f);
        return NULL;
    }

    // 读 header JSON
    char* header_json = (char*)malloc(header_size + 1);
    if (!header_json) { fclose(f); return NULL; }
    if (fread(header_json, 1, header_size, f) != header_size) {
        fprintf(stderr, "[safetensors] failed to read header\n");
        free(header_json);
        fclose(f);
        return NULL;
    }
    header_json[header_size] = '\0';
    fclose(f);

    uint64_t payload_base = 8 + header_size;

    // 解析 JSON: { "name1": {..}, "name2": {..}, "__metadata__": {..} }
    json_parser_t p = { header_json, header_size, 0 };
    if (!json_expect(&p, '{')) { free(header_json); return NULL; }

    // 先数有多少个 tensor (粗略,含 __metadata__)
    size_t save_pos = p.pos;
    int entry_count = 0;
    do {
        char name[256];
        if (!json_parse_string(&p, name, sizeof(name))) break;
        if (!json_expect(&p, ':')) break;
        if (!json_skip_value(&p)) break;
        entry_count++;
        json_skip_ws(&p);
        if (p.pos >= p.len || p.json[p.pos] != ',') break;
        p.pos++; // skip comma
    } while (true);
    // 回到开头重新解析
    p.pos = 1;

    cgc_source_tensor_t* tensors = (cgc_source_tensor_t*)calloc(entry_count, sizeof(cgc_source_tensor_t));
    if (!tensors) { free(header_json); return NULL; }
    int n_tensors = 0;

    do {
        char name[256];
        json_skip_ws(&p);
        if (p.pos >= p.len || p.json[p.pos] == '}') break;
        if (!json_parse_string(&p, name, sizeof(name))) break;
        if (!json_expect(&p, ':')) break;

        // 跳过 __metadata__
        if (strcmp(name, "__metadata__") == 0) {
            json_skip_value(&p);
        } else {
            // 解析 tensor entry: { "dtype": "BF16", "shape": [..], "data_offsets": [begin, end] }
            if (!json_expect(&p, '{')) break;
            char dtype_str[16] = {0};
            uint64_t shape[4] = {0,0,0,0};
            int n_dims = 0;
            uint64_t data_begin = 0, data_end = 0;
            bool have_offsets = false;

            do {
                json_skip_ws(&p);
                if (p.pos >= p.len || p.json[p.pos] == '}') break;
                char key[32];
                if (!json_parse_string(&p, key, sizeof(key))) break;
                if (!json_expect(&p, ':')) break;

                if (strcmp(key, "dtype") == 0) {
                    json_parse_string(&p, dtype_str, sizeof(dtype_str));
                } else if (strcmp(key, "shape") == 0) {
                    if (!json_expect(&p, '[')) break;
                    n_dims = 0;
                    do {
                        json_skip_ws(&p);
                        if (p.pos >= p.len || p.json[p.pos] == ']') break;
                        int64_t v;
                        if (json_parse_int(&p, &v) && n_dims < 4) {
                            shape[n_dims++] = (uint64_t)v;
                        }
                        json_skip_ws(&p);
                        if (p.pos < p.len && p.json[p.pos] == ',') p.pos++;
                    } while (true);
                    json_expect(&p, ']');
                } else if (strcmp(key, "data_offsets") == 0) {
                    if (!json_expect(&p, '[')) break;
                    int64_t v;
                    if (json_parse_int(&p, &v)) data_begin = (uint64_t)v;
                    json_expect(&p, ',');
                    if (json_parse_int(&p, &v)) data_end = (uint64_t)v;
                    json_expect(&p, ']');
                    have_offsets = true;
                } else {
                    json_skip_value(&p);
                }

                json_skip_ws(&p);
                if (p.pos < p.len && p.json[p.pos] == ',') p.pos++;
            } while (true);
            json_expect(&p, '}');

            if (have_offsets) {
                cgc_source_tensor_t* t = &tensors[n_tensors++];
                strncpy(t->name, name, sizeof(t->name) - 1);
                strncpy(t->shard_path, shard_path, sizeof(t->shard_path) - 1);
                if (strcmp(dtype_str, "U32") == 0) t->dtype = CGC_DTYPE_U32;
                else if (strcmp(dtype_str, "BF16") == 0) t->dtype = CGC_DTYPE_BF16;
                else if (strcmp(dtype_str, "F16") == 0) t->dtype = CGC_DTYPE_FP16;
                else if (strcmp(dtype_str, "F32") == 0) t->dtype = CGC_DTYPE_FP32;
                else t->dtype = CGC_DTYPE_FP32;
                t->n_dims = n_dims;
                for (int i = 0; i < n_dims && i < 4; ++i) t->shape[i] = shape[i];
                t->absolute_offset = payload_base + data_begin;
                t->size_bytes = data_end - data_begin;
            }
        }

        json_skip_ws(&p);
        if (p.pos < p.len && p.json[p.pos] == ',') p.pos++;
    } while (true);

    free(header_json);
    *out_count = n_tensors;
    return tensors;
}

// ============================================================================
// 扫描目录下所有 .safetensors 文件,合并解析
// ============================================================================

#ifdef _WIN32
static int scan_safetensors_dir(const char* dir, cgc_source_tensor_t** out_tensors, int* out_total) {
    char pattern[MAX_PATH];
    snprintf(pattern, sizeof(pattern), "%s\\*.safetensors", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[safetensors] no .safetensors files in %s\n", dir);
        return -1;
    }
    int total = 0;
    cgc_source_tensor_t* all = NULL;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        char full_path[MAX_PATH];
        snprintf(full_path, sizeof(full_path), "%s\\%s", dir, fd.cFileName);
        int n = 0;
        cgc_source_tensor_t* tensors = cgc_safetensors_parse(full_path, &n);
        if (n > 0 && tensors) {
            all = (cgc_source_tensor_t*)realloc(all, (total + n) * sizeof(cgc_source_tensor_t));
            memcpy(all + total, tensors, n * sizeof(cgc_source_tensor_t));
            total += n;
            free(tensors);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    *out_tensors = all;
    *out_total = total;
    return 0;
}
#else
// POSIX 版本 (未来 Mac/Linux 用)
static int scan_safetensors_dir(const char* dir, cgc_source_tensor_t** out_tensors, int* out_total) {
    // TODO: opendir/readdir
    return -1;
}
#endif

// 公开 API:扫描目录
cgc_source_tensor_t* cgc_safetensors_scan_dir(const char* dir, int* out_count) {
    cgc_source_tensor_t* tensors = NULL;
    int total = 0;
    if (scan_safetensors_dir(dir, &tensors, &total) != 0) {
        *out_count = 0;
        return NULL;
    }
    *out_count = total;
    return tensors;
}
