// test_streamer_gemma4.c — 消費端驗證：cgc_expert_streamer create+load_experts
// 對 gemma4 GGUF（合併 gate_up 2 段）載入的 slot bytes 與直接 pread 絕對定址一致。
// 參考：cgc_expert_segments() 產生的絕對 offset（該路徑已對 GGUF 檔驗證 byte-identical）。
#include "cgc_expert_streamer.h"
#include "cgc_expert_streamer_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <gemma4.gguf>\n", argv[0]); return 1; }
    const char* path = argv[1];

    cgc_stream_layout_t layout = cgc_load_stream_layout_from_gguf(path);
    if (layout.experts_per_layer == 0) { fprintf(stderr, "layout parse failed\n"); return 1; }
    printf("layout: experts=%d stride=%llu has_segments=%d\n",
           layout.experts_per_layer, (unsigned long long)layout.expert_stride, layout.has_segments);

    // streamer: 8 slots, pread path (has_segments → 強制 pread)
    cgc_expert_streamer_t* s = cgc_expert_streamer_create(&layout, 8, false, NULL, 0);
    if (!s) { fprintf(stderr, "streamer create failed\n"); return 1; }

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    uint8_t* ref = malloc((size_t)layout.expert_stride);
    uint8_t* got = malloc((size_t)layout.expert_stride);
    if (!ref || !got) { fprintf(stderr, "malloc failed\n"); return 1; }

    cgc_cache_access_ctx_t ctx = {
        .owner_phase = CGC_CACHE_SLOT_DECODE_PROTECTED,
        .control_plane = CGC_CACHE_CONTROL_DECODE,
        .request_id = 1, .decode_step_index = 0,
    };

    const int eids[8] = { 0, 1, 3, 7, 15, 63, 127, 2 }; // 含邊界 0/127
    int fail = 0;

    cgc_cache_result_t r = cgc_expert_streamer_load_experts(s, eids, 8, &ctx);
    printf("load: count=%d hits=%d misses=%d\n", r.count, r.hits, r.misses);
    if (r.misses != 8) { fprintf(stderr, "[FAIL] expected 8 misses on cold load, got %d\n", r.misses); fail = 1; }

    for (int i = 0; i < 8; i++) {
        int e = eids[i];
        if (!r.buffers[i] || r.sizes[i] != layout.expert_stride) {
            fprintf(stderr, "[E%d] bad slot: buf=%p size=%llu\n", e, r.buffers[i],
                    (unsigned long long)r.sizes[i]);
            fail = 1; continue;
        }
        // 參考 bytes: gate_up + down 兩段依序拼接（與 read_expert 相同順序）
        uint64_t offs[3] = {0}, sizes[3] = {0};
        int n_seg = cgc_expert_segments(&layout, 0, e, offs, sizes);
        if (n_seg <= 0) { fprintf(stderr, "[E%d] segments failed\n", e); fail = 1; continue; }
        size_t pos = 0;
        for (int k = 0; k < n_seg; k++) {
            if (sizes[k] == 0) continue;
            if (pread(fd, ref + pos, (size_t)sizes[k], (off_t)offs[k]) != (ssize_t)sizes[k]) {
                fprintf(stderr, "[E%d] ref pread failed\n", e); fail = 1; break;
            }
            pos += (size_t)sizes[k];
        }
        memcpy(got, r.buffers[i], (size_t)layout.expert_stride);
        int same = memcmp(ref, got, (size_t)layout.expert_stride) == 0;
        printf("[E%3d] slot bytes == direct pread (abs addressing): %s\n", e, same ? "YES ✓" : "NO ✗");
        if (!same) fail = 1;
    }

    // hit 路徑：重載同一批 → 全 hit、buffer 指標不變
    cgc_cache_result_t r2 = cgc_expert_streamer_load_experts(s, eids, 8, &ctx);
    printf("reload: count=%d hits=%d misses=%d\n", r2.count, r2.hits, r2.misses);
    if (r2.hits != 8) { fprintf(stderr, "[FAIL] expected 8 hits on reload, got %d\n", r2.hits); fail = 1; }
    for (int i = 0; i < 8; i++) {
        if (r2.buffers[i] != r.buffers[i]) {
            fprintf(stderr, "[FAIL] reload buffer pointer changed for E%d\n", eids[i]);
            fail = 1;
        }
    }

    cgc_cache_telemetry_t tel = cgc_expert_streamer_telemetry(s);
    printf("telemetry: requests=%llu hits=%llu misses=%llu loads=%llu read_bytes=%llu\n",
           (unsigned long long)tel.total_requests, (unsigned long long)tel.total_hits,
           (unsigned long long)tel.total_misses, (unsigned long long)tel.total_loads,
           (unsigned long long)tel.total_read_bytes);

    close(fd);
    cgc_expert_streamer_destroy(s);
    free(ref); free(got);
    printf(fail ? "\nRESULT: FAIL\n" : "\nRESULT: PASS — streamer consumer path verified\n");
    return fail ? 1 : 0;
}
