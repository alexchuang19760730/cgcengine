#include "cgc_pd_scheduler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static uint64_t now_nanos_pd(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int freq_init = 0;
    if (!freq_init) {
        QueryPerformanceFrequency(&freq);
        freq_init = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)(counter.QuadPart * 1000000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

bool cgc_pd_is_prefill_layer(const cgc_pd_layer_assignment_t* a, int layer) {
    if (!a) return false;
    for (int i = 0; i < a->prefill_count; i++) {
        if (a->prefill_layers[i] == layer) return true;
    }
    return false;
}

bool cgc_pd_is_decode_layer(const cgc_pd_layer_assignment_t* a, int layer) {
    if (!a) return false;
    for (int i = 0; i < a->decode_count; i++) {
        if (a->decode_layers[i] == layer) return true;
    }
    return false;
}

int cgc_pd_get_device_for_layer(const cgc_pd_layer_assignment_t* a, int layer) {
    if (!a) return -1;
    if (cgc_pd_is_prefill_layer(a, layer)) return a->prefill_gpu;
    if (cgc_pd_is_decode_layer(a, layer)) return a->decode_gpu;
    return -1;
}

cgc_pd_layer_assignment_t cgc_pd_layer_assignment_by_ratio(int total_layers, double prefill_ratio) {
    cgc_pd_layer_assignment_t a;
    memset(&a, 0, sizeof(a));

    int prefill_count = (int)(total_layers * prefill_ratio);
    int decode_count = total_layers - prefill_count;

    if (prefill_count < 0) prefill_count = 0;
    if (prefill_count > CGC_MAX_PD_LAYERS) prefill_count = CGC_MAX_PD_LAYERS;
    if (decode_count < 0) decode_count = 0;
    if (decode_count > CGC_MAX_PD_LAYERS) decode_count = CGC_MAX_PD_LAYERS;

    for (int i = 0; i < prefill_count; i++) {
        a.prefill_layers[i] = i;
    }
    a.prefill_count = prefill_count;

    for (int i = 0; i < decode_count; i++) {
        a.decode_layers[i] = prefill_count + i;
    }
    a.decode_count = decode_count;

    a.prefill_gpu = 0;
    a.decode_gpu = 1;

    return a;
}

cgc_pd_layer_assignment_t cgc_pd_layer_assignment_custom(const int* prefill, int prefill_count,
                                                          const int* decode, int decode_count) {
    cgc_pd_layer_assignment_t a;
    memset(&a, 0, sizeof(a));

    if (prefill && prefill_count > 0 && prefill_count <= CGC_MAX_PD_LAYERS) {
        memcpy(a.prefill_layers, prefill, prefill_count * sizeof(int));
        a.prefill_count = prefill_count;
    }
    if (decode && decode_count > 0 && decode_count <= CGC_MAX_PD_LAYERS) {
        memcpy(a.decode_layers, decode, decode_count * sizeof(int));
        a.decode_count = decode_count;
    }

    a.prefill_gpu = 0;
    a.decode_gpu = 1;
    return a;
}

static void route_history_record(cgc_pd_route_history_t* h, int layer, const int* expert_ids, int count) {
    if (!h || !expert_ids || count <= 0) return;

    for (int i = 0; i < count; i++) {
        int expert_id = expert_ids[i];

        for (int j = 0; j < h->count; j++) {
            if (h->entries[j].layer == layer && h->entries[j].expert_id == expert_id) {
                h->entries[j].weight += 1.0;
                return;
            }
        }

        if (h->count >= h->max_history) {
            double min_w = 1e30;
            int min_idx = 0;
            for (int j = 0; j < h->count; j++) {
                h->entries[j].weight *= h->decay_factor;
                if (h->entries[j].weight < min_w) {
                    min_w = h->entries[j].weight;
                    min_idx = j;
                }
            }
            h->entries[min_idx].layer = layer;
            h->entries[min_idx].expert_id = expert_id;
            h->entries[min_idx].weight = 1.0;
        } else {
            h->entries[h->count].layer = layer;
            h->entries[h->count].expert_id = expert_id;
            h->entries[h->count].weight = 1.0;
            h->count++;
        }
    }
}

static int route_history_get_most_frequent(cgc_pd_route_history_t* h, int layer, int top_k,
                                             int* out_ids, int max_out) {
    if (!h || !out_ids || top_k <= 0) return 0;

    int count = 0;
    double weights[CGC_MAX_EXPERTS_PER_LAYER];
    int layer_experts[CGC_MAX_EXPERTS_PER_LAYER];
    int layer_count = 0;

    for (int i = 0; i < h->count; i++) {
        if (h->entries[i].layer == layer && layer_count < CGC_MAX_EXPERTS_PER_LAYER) {
            layer_experts[layer_count] = h->entries[i].expert_id;
            weights[layer_count] = h->entries[i].weight;
            layer_count++;
        }
    }

    for (int i = 0; i < layer_count - 1; i++) {
        for (int j = i + 1; j < layer_count; j++) {
            if (weights[j] > weights[i]) {
                double tw = weights[i]; weights[i] = weights[j]; weights[j] = tw;
                int te = layer_experts[i]; layer_experts[i] = layer_experts[j]; layer_experts[j] = te;
            }
        }
    }

    int take = top_k < layer_count ? top_k : layer_count;
    if (take > max_out) take = max_out;

    for (int i = 0; i < take; i++) {
        out_ids[count++] = layer_experts[i];
    }

    return count;
}

static int route_history_predict_next(cgc_pd_route_history_t* h, int layer,
                                        const int* current_experts, int current_count,
                                        int top_k, int* out_ids, int max_out) {
    return route_history_get_most_frequent(h, layer, top_k, out_ids, max_out);
}

cgc_pd_scheduler_t* cgc_pd_scheduler_create(cgc_streamer_pool_t* pool,
                                               const cgc_pd_layer_assignment_t* assignment,
                                               int max_experts_per_layer,
                                               int tile_experts) {
    if (!pool || !assignment) return NULL;

    cgc_pd_scheduler_t* s = (cgc_pd_scheduler_t*)calloc(1, sizeof(cgc_pd_scheduler_t));
    if (!s) return NULL;

    s->streamer_pool = pool;
    memcpy(&s->assignment, assignment, sizeof(cgc_pd_layer_assignment_t));
    s->max_experts_per_layer = max_experts_per_layer > 0 ? max_experts_per_layer : 8;
    s->tile_experts = tile_experts > 0 ? tile_experts : 8;
    s->top_k = 8;
    s->current_phase = CGC_PD_PHASE_IDLE;

    s->route_history.max_history = CGC_MAX_ROUTE_HISTORY;
    s->route_history.decay_factor = 0.98;
    s->route_history.count = 0;

    s->gpu0_cache.max_count = s->max_experts_per_layer * 8;
    s->gpu0_cache.gpu_id = 0;
    s->gpu0_cache.count = 0;

    s->gpu1_cache.max_count = s->max_experts_per_layer * 8;
    s->gpu1_cache.gpu_id = 1;
    s->gpu1_cache.count = 0;

    s->initialized = 1;
    return s;
}

void cgc_pd_scheduler_destroy(cgc_pd_scheduler_t* s) {
    if (!s) return;

    for (int i = 0; i < s->gpu0_cache.count; i++) {
        if (s->gpu0_cache.entries[i].data) {
            free(s->gpu0_cache.entries[i].data);
        }
    }
    for (int i = 0; i < s->gpu1_cache.count; i++) {
        if (s->gpu1_cache.entries[i].data) {
            free(s->gpu1_cache.entries[i].data);
        }
    }

    free(s);
}

void cgc_pd_scheduler_enter_prefill(cgc_pd_scheduler_t* s) {
    if (!s) return;

    fprintf(stderr, "[PDExpertScheduler] Entering PREFILL phase...\n");
    s->current_phase = CGC_PD_PHASE_PREFILL;

    for (int i = 0; i < s->assignment.prefill_count; i++) {
        int layer = s->assignment.prefill_layers[i];
        cgc_expert_streamer_t* streamer = cgc_streamer_pool_get(s->streamer_pool, layer);
        if (!streamer) {
            fprintf(stderr, "[PDExpertScheduler] No streamer for layer %d, skipping\n", layer);
            continue;
        }

        int hot_experts[CGC_MAX_EXPERTS_PER_LAYER];
        int hot_count = route_history_get_most_frequent(&s->route_history, layer,
                                                        s->max_experts_per_layer / 2,
                                                        hot_experts, CGC_MAX_EXPERTS_PER_LAYER);

        if (hot_count == 0) {
            for (int j = 0; j < s->max_experts_per_layer && j < streamer->layout.experts_per_layer; j++) {
                hot_experts[hot_count++] = j;
            }
        }

        cgc_cache_access_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.owner_phase = CGC_CACHE_SLOT_PREFILL_TRANSIENT;
        ctx.control_plane = CGC_CACHE_CONTROL_PREFILL;

        for (int j = 0; j < hot_count; j++) {
            cgc_cache_result_t result = cgc_expert_streamer_load_experts(streamer, &hot_experts[j], 1, &ctx);
            if (result.buffers[0]) {
                void* buf_copy = malloc(result.sizes[0]);
                if (buf_copy) {
                    memcpy(buf_copy, result.buffers[0], result.sizes[0]);

                    if (s->gpu0_cache.count < s->gpu0_cache.max_count) {
                        cgc_pd_cache_entry_t* entry = &s->gpu0_cache.entries[s->gpu0_cache.count++];
                        entry->layer = layer;
                        entry->expert_id = hot_experts[j];
                        entry->data = buf_copy;
                        entry->size = result.sizes[0];
                        entry->last_access = now_nanos_pd();
                        entry->access_count = 1;
                        entry->pinned = true;
                    } else {
                        free(buf_copy);
                    }
                }
                s->expert_loads++;
            }
        }
    }

    fprintf(stderr, "[PDExpertScheduler] Prefill phase ready: %d experts in GPU0 cache\n",
            s->gpu0_cache.count);
}

void cgc_pd_scheduler_switch_to_decode(cgc_pd_scheduler_t* s) {
    if (!s) return;

    fprintf(stderr, "[PDExpertScheduler] Switching to DECODE phase...\n");
    uint64_t t0 = now_nanos_pd();

    s->current_phase = CGC_PD_PHASE_DECODE;

    int gpu0_count = s->gpu0_cache.count;
    memset(s->gpu0_cache.entries, 0, sizeof(s->gpu0_cache.entries));
    s->gpu0_cache.count = 0;
    fprintf(stderr, "[PDExpertScheduler] Released GPU 0 cache (%d experts)\n", gpu0_count);

    int preloaded = 0;

    for (int i = 0; i < s->assignment.decode_count; i++) {
        int layer = s->assignment.decode_layers[i];
        cgc_expert_streamer_t* streamer = cgc_streamer_pool_get(s->streamer_pool, layer);
        if (!streamer) continue;

        int hot_experts[CGC_MAX_EXPERTS_PER_LAYER];
        int hot_count = route_history_get_most_frequent(&s->route_history, layer,
                                                        s->max_experts_per_layer / 2,
                                                        hot_experts, CGC_MAX_EXPERTS_PER_LAYER);

        if (hot_count == 0) {
            for (int j = 0; j < 4 && j < streamer->layout.experts_per_layer; j++) {
                hot_experts[hot_count++] = j;
            }
        }

        cgc_cache_access_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.owner_phase = CGC_CACHE_SLOT_DECODE_PROTECTED;
        ctx.control_plane = CGC_CACHE_CONTROL_DECODE;

        for (int j = 0; j < hot_count; j++) {
            cgc_cache_result_t result = cgc_expert_streamer_load_experts(streamer, &hot_experts[j], 1, &ctx);
            if (result.buffers[0]) {
                void* buf_copy = malloc(result.sizes[0]);
                if (buf_copy) {
                    memcpy(buf_copy, result.buffers[0], result.sizes[0]);

                    if (s->gpu1_cache.count < s->gpu1_cache.max_count) {
                        cgc_pd_cache_entry_t* entry = &s->gpu1_cache.entries[s->gpu1_cache.count++];
                        entry->layer = layer;
                        entry->expert_id = hot_experts[j];
                        entry->data = buf_copy;
                        entry->size = result.sizes[0];
                        entry->last_access = now_nanos_pd();
                        entry->access_count = 1;
                        entry->pinned = true;
                    } else {
                        free(buf_copy);
                    }
                }
                preloaded++;
                s->expert_loads++;
            }
        }
    }

    for (int i = 0; i < s->assignment.decode_count; i++) {
        int layer = s->assignment.decode_layers[i];
        cgc_expert_streamer_t* streamer = cgc_streamer_pool_get(s->streamer_pool, layer);
        if (!streamer) continue;

        int predicted[CGC_MAX_EXPERTS_PER_LAYER];
        int pred_count = route_history_predict_next(&s->route_history, layer,
                                                     NULL, 0,
                                                     s->max_experts_per_layer,
                                                     predicted, CGC_MAX_EXPERTS_PER_LAYER);

        if (pred_count > 0) {
            cgc_expert_streamer_prefetch(streamer, predicted, pred_count);
            s->prefetch_hits++;
        }
    }

    uint64_t elapsed = now_nanos_pd() - t0;
    s->total_prefetch_time_nanos += elapsed;

    fprintf(stderr, "[PDExpertScheduler] Switch complete: preloaded %d experts in %.2f ms\n",
            preloaded, (double)elapsed / 1e6);
}

cgc_pd_phase_t cgc_pd_scheduler_current_phase(const cgc_pd_scheduler_t* s) {
    if (!s) return CGC_PD_PHASE_IDLE;
    return s->current_phase;
}

int cgc_pd_scheduler_process_prefill(cgc_pd_scheduler_t* sched,
                                      const cgc_pd_token_routes_t* token_routes,
                                      int route_count,
                                      cgc_pd_tile_t* out_tiles,
                                      int max_tiles) {
    if (!sched || !token_routes || !out_tiles || route_count <= 0 || max_tiles <= 0) return 0;

    int tile_count = 0;
    int tile_idx = 0;

    for (int t = 0; t < route_count && tile_count < max_tiles; t++) {
        const cgc_pd_token_routes_t* tr = &token_routes[t];
        for (int r = 0; r < tr->route_count; r++) {
            const cgc_pd_expert_route_t* route = &tr->routes[r];
            if (!cgc_pd_is_prefill_layer(&sched->assignment, route->layer)) continue;

            cgc_pd_tile_t* tile = &out_tiles[tile_count++];
            tile->tile_index = tile_idx++;
            tile->layer = route->layer;
            tile->token_start = t;
            tile->token_count = 1;

            int ec = route->expert_count < CGC_MAX_EXPERTS_PER_LAYER ? route->expert_count : CGC_MAX_EXPERTS_PER_LAYER;
            for (int e = 0; e < ec && e < sched->tile_experts; e++) {
                tile->expert_ids[e] = route->expert_ids[e];
            }
            tile->expert_count = ec < sched->tile_experts ? ec : sched->tile_experts;
        }
    }

    sched->prefill_tokens += (uint64_t)route_count;
    return tile_count;
}

cgc_cache_result_t cgc_pd_scheduler_load_prefill_experts(cgc_pd_scheduler_t* sched,
                                                          int layer,
                                                          const int* expert_ids,
                                                          int count) {
    cgc_cache_result_t empty;
    memset(&empty, 0, sizeof(empty));

    if (!sched || !expert_ids || count <= 0) return empty;

    cgc_streamer_pool_t* pool = sched->streamer_pool;
    if (!pool) return empty;

    cgc_cache_access_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.owner_phase = CGC_CACHE_SLOT_PREFILL_TRANSIENT;
    ctx.control_plane = CGC_CACHE_CONTROL_PREFILL;

    return cgc_streamer_pool_load_experts(pool, layer, expert_ids, count, &ctx);
}

int cgc_pd_scheduler_process_decode(cgc_pd_scheduler_t* sched,
                                     const cgc_pd_token_routes_t* token_route,
                                     cgc_pd_tile_t* out_tiles,
                                     int max_tiles) {
    if (!sched || !token_route || !out_tiles || max_tiles <= 0) return 0;

    int tile_count = 0;
    int tile_idx = 0;

    for (int r = 0; r < token_route->route_count && tile_count < max_tiles; r++) {
        const cgc_pd_expert_route_t* route = &token_route->routes[r];
        if (!cgc_pd_is_decode_layer(&sched->assignment, route->layer)) continue;

        cgc_pd_tile_t* tile = &out_tiles[tile_count++];
        tile->tile_index = tile_idx++;
        tile->layer = route->layer;
        tile->token_start = token_route->token_index;
        tile->token_count = 1;

        int ec = route->expert_count < CGC_MAX_EXPERTS_PER_LAYER ? route->expert_count : CGC_MAX_EXPERTS_PER_LAYER;
        for (int e = 0; e < ec && e < sched->tile_experts; e++) {
            tile->expert_ids[e] = route->expert_ids[e];
        }
        tile->expert_count = ec < sched->tile_experts ? ec : sched->tile_experts;
    }

    sched->decode_tokens++;
    return tile_count;
}

cgc_cache_result_t cgc_pd_scheduler_load_decode_experts(cgc_pd_scheduler_t* sched,
                                                        int layer,
                                                        const int* expert_ids,
                                                        int count) {
    cgc_cache_result_t empty;
    memset(&empty, 0, sizeof(empty));

    if (!sched || !expert_ids || count <= 0) return empty;

    cgc_streamer_pool_t* pool = sched->streamer_pool;
    if (!pool) return empty;

    cgc_cache_access_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.owner_phase = CGC_CACHE_SLOT_DECODE_PROTECTED;
    ctx.control_plane = CGC_CACHE_CONTROL_DECODE;

    return cgc_streamer_pool_load_experts(pool, layer, expert_ids, count, &ctx);
}

int cgc_pd_scheduler_trigger_prefetch(cgc_pd_scheduler_t* sched,
                                       const cgc_pd_token_routes_t* current_routes,
                                       int* out_expert_ids,
                                       int max_ids) {
    if (!sched || !out_expert_ids || !current_routes) return 0;

    int count = 0;

    for (int i = 0; i < current_routes->route_count && count < max_ids; i++) {
        const cgc_pd_expert_route_t* route = &current_routes->routes[i];
        if (!cgc_pd_is_decode_layer(&sched->assignment, route->layer)) continue;

        int predicted[CGC_MAX_EXPERTS_PER_LAYER];
        int pred_count = route_history_predict_next(&sched->route_history, route->layer,
                                                     route->expert_ids, route->expert_count,
                                                     sched->top_k, predicted,
                                                     CGC_MAX_EXPERTS_PER_LAYER);

        for (int j = 0; j < pred_count && count < max_ids; j++) {
            out_expert_ids[count++] = predicted[j];
        }
    }

    sched->prefetch_hits++;
    return count;
}

void cgc_pd_scheduler_record_routes(cgc_pd_scheduler_t* sched,
                                      const cgc_pd_token_routes_t* routes) {
    if (!sched || !routes) return;

    for (int i = 0; i < routes->route_count; i++) {
        const cgc_pd_expert_route_t* route = &routes->routes[i];
        route_history_record(&sched->route_history, route->layer,
                             route->expert_ids, route->expert_count);
    }
}

void cgc_pd_scheduler_set_top_k(cgc_pd_scheduler_t* sched, int top_k) {
    if (sched && top_k > 0) sched->top_k = top_k;
}

cgc_pd_scheduler_stats_t cgc_pd_scheduler_get_stats(const cgc_pd_scheduler_t* sched) {
    cgc_pd_scheduler_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    if (!sched) return stats;

    stats.phase = sched->current_phase;
    stats.gpu0_cache_count = sched->gpu0_cache.count;
    stats.gpu1_cache_count = sched->gpu1_cache.count;

    int gpu0_total = sched->gpu0_cache.count;
    int gpu0_hits = 0;
    for (int i = 0; i < gpu0_total; i++) {
        if (sched->gpu0_cache.entries[i].access_count > 1) gpu0_hits++;
    }
    stats.gpu0_hit_rate = gpu0_total > 0 ? (double)gpu0_hits / gpu0_total : 0.0;

    int gpu1_total = sched->gpu1_cache.count;
    int gpu1_hits = 0;
    for (int i = 0; i < gpu1_total; i++) {
        if (sched->gpu1_cache.entries[i].access_count > 1) gpu1_hits++;
    }
    stats.gpu1_hit_rate = gpu1_total > 0 ? (double)gpu1_hits / gpu1_total : 0.0;

    stats.prefill_tokens = sched->prefill_tokens;
    stats.decode_tokens = sched->decode_tokens;
    stats.expert_loads = sched->expert_loads;
    stats.prefetch_hits = sched->prefetch_hits;
    stats.total_prefetch_time_nanos = sched->total_prefetch_time_nanos;
    stats.total_load_time_nanos = sched->total_load_time_nanos;

    return stats;
}

void cgc_pd_scheduler_reset_stats(cgc_pd_scheduler_t* sched) {
    if (!sched) return;

    sched->prefill_tokens = 0;
    sched->decode_tokens = 0;
    sched->expert_loads = 0;
    sched->prefetch_hits = 0;
    sched->total_prefetch_time_nanos = 0;
    sched->total_load_time_nanos = 0;
}
