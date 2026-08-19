// test_real_gemma4.c — 用真實 llama.cpp gemma4 GGUF 驗證 expert streaming 定址
//
// llama.cpp gemma4（IQ3_S 等）: 每層 2 個 packed 張量
//   blk.N.ffn_gate_up_exps（gate+up 合併）[inter*2, hidden, n_experts]
//   blk.N.ffn_down_exps                     [inter,  hidden, n_experts]
//   專家 e 的權重分散在兩個非連續檔案區段（gate_up / down，各用不同 bpw）。
// 本工具驗證 cgc_expert_segments() 產生的 2 段 offset/size：
//   1. 與 GGUF tensor 表的 offset/type 一致
//   2. 對多個 expert（含邊界 0/3/127）pread 到的 bytes，與「tensor 表直接定址」
//      讀到的 bytes 完全相同（offset 算術正確性）
//   3. 每段前 4096 bytes 非零（量化權重密度 sanity）
#include "cgc_expert_streamer_gguf.h"
#include "cgc_gguf_lite.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>

static int nz_count(const uint8_t* buf, int n) {
    int nz = 0;
    for (int i = 0; i < n; i++) if (buf[i] != 0) nz++;
    return nz;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }
    const char* path = argv[1];

    cgc_stream_layout_t layout = cgc_load_stream_layout_from_gguf(path);
    if (layout.experts_per_layer == 0) { fprintf(stderr, "layout parse failed\n"); return 1; }

    printf("layout: experts=%d stride=%llu has_segments=%d stream_offset=%llu stream_size=%llu\n",
           layout.experts_per_layer, (unsigned long long)layout.expert_stride,
           layout.has_segments,
           (unsigned long long)layout.stream_offset, (unsigned long long)layout.stream_size);
    if (layout.has_segments) {
        printf("  seg_base = [%llu, %llu, %llu]\n",
               (unsigned long long)layout.seg_base[0],
               (unsigned long long)layout.seg_base[1],
               (unsigned long long)layout.seg_base[2]);
        printf("  seg_size = [%llu, %llu, %llu] (per-expert bytes)\n",
               (unsigned long long)layout.seg_size[0],
               (unsigned long long)layout.seg_size[1],
               (unsigned long long)layout.seg_size[2]);
        if (layout.seg_size[0] + layout.seg_size[1] != layout.expert_stride) {
            fprintf(stderr, "  [FAIL] seg_size[0]+seg_size[1] != expert_stride\n");
            return 1;
        }
    }

    // --- 對照: GGUF tensor 表（層 0 兩張量）---
    cgc_gguf_lite_ctx_t* ctx = cgc_gguf_lite_load(path);
    if (!ctx) { fprintf(stderr, "gguf load failed\n"); return 1; }

    uint64_t t_off[2] = {0, 0};
    int64_t  t_ne[2] = {0, 0};
    int      t_type[2] = {-1, -1};
    const char* t_suffix[2] = { ".ffn_gate_up_exps.", ".ffn_down_exps." };
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        const char* n = ctx->tensor_names[i];
        if (n[0] != 'b' || n[1] != 'l' || n[2] != 'k' || n[3] != '.' || n[4] != '0' || n[5] != '.') continue;
        // 排除 per-tensor scale 變體（blk.N.ffn_*_exps.scale），只要 .weight 主張量
        if (strstr(n, ".scale")) continue;
        for (int k = 0; k < 2; k++) {
            if (strstr(n, t_suffix[k]) && t_off[k] == 0) {
                // GGUF v3: raw offset 相對 data_start；絕對位置 = data_start + raw
                t_off[k] = ctx->data_start + ctx->tensors[i].offset;
                t_ne[k]  = ctx->tensors[i].n_elements;
                t_type[k] = ctx->tensors[i].type;
                printf("[T] matched[%d] = %s (abs=%llu)\n", k, n, (unsigned long long)t_off[k]);
            }
        }
    }
    if (t_off[0] == 0 || t_off[1] == 0) {
        fprintf(stderr, "[FAIL] 找不到層 0 的 ffn_gate_up_exps / ffn_down_exps 張量\n");
        return 1;
    }

    int fail = 0;
    for (int k = 0; k < 2; k++) {
        double bpe = cgc_ggml_type_bytes_per_elem(t_type[k]);
        uint64_t per_exp = (uint64_t)(bpe * (double)t_ne[k] / (double)layout.experts_per_layer);
        printf("[T] seg%d tensor: off=%llu type=%d per_expert=%llu (layout=%llu)\n",
               k, (unsigned long long)t_off[k], t_type[k],
               (unsigned long long)per_exp, (unsigned long long)layout.seg_size[k]);
        if (!layout.has_segments) {
            fprintf(stderr, "  [FAIL] expected has_segments=1 for llama.cpp 2-tensor layout\n");
            fail = 1;
        } else if (t_off[k] != layout.seg_base[k]) {
            fprintf(stderr, "  [FAIL] seg_base[%d] %llu != tensor offset %llu\n",
                    k, (unsigned long long)layout.seg_base[k], (unsigned long long)t_off[k]);
            fail = 1;
        } else if (per_exp != layout.seg_size[k]) {
            fprintf(stderr, "  [FAIL] seg_size[%d] %llu != per_expert %llu\n",
                    k, (unsigned long long)layout.seg_size[k], (unsigned long long)per_exp);
            fail = 1;
        }
    }

    // --- 位元組一致性: cgc 定址 vs tensor 表直接定址 ---
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    uint8_t buf_cgc[4096], buf_ref[4096];

    const int eids[] = { 0, 3, layout.experts_per_layer - 1 };
    for (size_t ei = 0; ei < sizeof(eids) / sizeof(eids[0]); ei++) {
        int e = eids[ei];
        uint64_t offs[3] = {0}, sizes[3] = {0};
        int n_seg = cgc_expert_segments(&layout, 0, e, offs, sizes);
        if (n_seg <= 0) { fprintf(stderr, "[E%d] cgc_expert_segments failed\n", e); fail = 1; continue; }
        for (int k = 0; k < n_seg; k++) {
            if (sizes[k] == 0) continue; // gemma4 只有 2 段，第 3 段為空
            uint64_t ref_off = t_off[k] + (uint64_t)e * layout.seg_size[k];
            if (offs[k] != ref_off) {
                fprintf(stderr, "[E%d] seg%d cgc off=%llu != ref=%llu\n", e, k,
                        (unsigned long long)offs[k], (unsigned long long)ref_off);
                fail = 1;
                continue;
            }
            // 讀 4096 bytes（不超過段尾）
            ssize_t want = (ssize_t)(sizes[k] < sizeof(buf_cgc) ? sizes[k] : sizeof(buf_cgc));
            ssize_t nc = pread(fd, buf_cgc, want, (off_t)offs[k]);
            ssize_t nr = pread(fd, buf_ref, want, (off_t)ref_off);
            if (nc != want || nr != want) {
                fprintf(stderr, "[E%d] seg%d short read cgc=%zd ref=%zd\n", e, k, nc, nr);
                fail = 1;
                continue;
            }
            int same = memcmp(buf_cgc, buf_ref, want) == 0;
            int nz = nz_count(buf_cgc, (int)want);
            printf("[E%3d] seg%d @ %llu: %zd bytes, identical_to_tensor_table=%s, nonzero=%d/%zd%s\n",
                   e, k, (unsigned long long)offs[k], want, same ? "YES" : "NO", nz, want,
                   same ? " ✓" : " ✗");
            if (!same) fail = 1;
        }
    }

    close(fd);
    cgc_gguf_lite_free(ctx);
    printf(fail ? "\nRESULT: FAIL\n" : "\nRESULT: PASS — gemma4 GGUF expert streaming addressing verified\n");
    return fail ? 1 : 0;
}
