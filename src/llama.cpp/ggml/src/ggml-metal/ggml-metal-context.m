#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>   // snprintf ([CGC watchdog] capture command)
#include <stdlib.h>  // system, getenv ([CGC watchdog] capture command)
#include <unistd.h>  // getpid, usleep ([CGC watchdog] dump hint + probe poll)

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy; // for async copies

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    // CGC pipelined segment dispatch (CGC_OA_ASYNC): monotonically increasing count of
    // graph-compute segments whose main command buffer finished on the GPU. The sched polls
    // this (via ggml_metal_cgc_done) instead of blocking per segment, so the Metal pipeline
    // stays busy while the CPU writes the remap leaves for the next segments.
    _Atomic int cgc_done;

    // [CGC 2026-08-29 deadlock watchdog] the sched busy-waits on cgc_done (hook_seg);
    // an intermittent freeze (~1-in-4 steady runs) shows it never reaching its target.
    // Tracking: expected = completion handlers attached at COMMIT (fires exactly once per
    // committed buffer); last_progress = updated by every completion; per-cb encode
    // start/done + create timestamps for the LATEST graph_compute. The watchdog thread
    // (CGC_WATCHDOG=1, CGC_WATCHDOG_MS threshold) dumps all cmd buffer statuses when
    // completions stall or an encode worker hangs, then aborts after a 60s sampling window.
    _Atomic int        cgc_expected;
    _Atomic int64_t    cgc_last_progress_us;
    _Atomic int64_t    cgc_last_submit_us;
    _Atomic int        cgc_n_computes;
    _Atomic int64_t    cgc_encode_start_us[GGML_METAL_MAX_COMMAND_BUFFERS + 1];
    _Atomic int64_t    cgc_encode_done_us[GGML_METAL_MAX_COMMAND_BUFFERS + 1];
    int64_t            cgc_cb_create_us[GGML_METAL_MAX_COMMAND_BUFFERS + 1]; // main thread only
    _Atomic bool       cgc_watchdog_stop;
    dispatch_queue_t   cgc_watchdog_queue;
    dispatch_source_t  cgc_watchdog_timer;
    int                cgc_watchdog_ms;       // constant after init
    int64_t            cgc_watchdog_fired_us; // watchdog queue only
    int64_t            cgc_watchdog_dump_us;  // watchdog queue only
    bool               cgc_probe_done;        // watchdog queue only (see cgc_watchdog_probe)

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
};

// [CGC watchdog] defined below (after the init/free section)
static void cgc_watchdog_tick(ggml_metal_t ctx);

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    if (queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    res->dev = dev;
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

            free(res);

            return NULL;
        }
    }

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

    res->capture_compute = 0;
    res->capture_started = false;
    res->capture_scope = nil;

    {
        const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
        if (val) {
            res->capture_compute = atoi(val);
        }
    }

    res->has_error = false;

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    atomic_store_explicit(&res->cgc_done, 0, memory_order_relaxed);

    // [CGC watchdog] opt-in via CGC_WATCHDOG=1 (default off = production unchanged)
    atomic_store_explicit(&res->cgc_expected,        0, memory_order_relaxed);
    atomic_store_explicit(&res->cgc_last_progress_us, 0, memory_order_relaxed);
    atomic_store_explicit(&res->cgc_last_submit_us,   0, memory_order_relaxed);
    atomic_store_explicit(&res->cgc_n_computes,      0, memory_order_relaxed);
    atomic_store_explicit(&res->cgc_watchdog_stop,   false, memory_order_relaxed);
    {
        const char * wd  = getenv("CGC_WATCHDOG");
        const char * wms = getenv("CGC_WATCHDOG_MS");
        res->cgc_watchdog_ms = (wms && wms[0]) ? atoi(wms) : 10000;
        if (wd && atoi(wd) != 0 && res->cgc_watchdog_ms > 0) {
            res->cgc_watchdog_queue = dispatch_queue_create("ggml-metal-wd", DISPATCH_QUEUE_SERIAL);
            res->cgc_watchdog_timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, res->cgc_watchdog_queue);
            GGML_ASSERT(res->cgc_watchdog_timer);
            dispatch_source_set_timer(res->cgc_watchdog_timer,
                    dispatch_time(DISPATCH_TIME_NOW, 500 * NSEC_PER_MSEC),
                    (uint64_t) 500 * NSEC_PER_MSEC, (uint64_t) 100 * NSEC_PER_MSEC);
            ggml_metal_t ctx_wd = res; // captured by the handler block below
            dispatch_source_set_event_handler(res->cgc_watchdog_timer, ^{
                cgc_watchdog_tick(ctx_wd);
            });
            dispatch_resume(res->cgc_watchdog_timer);
            GGML_LOG_WARN("%s: CGC deadlock watchdog ON (threshold %d ms, 60s sampling grace)\n",
                    __func__, res->cgc_watchdog_ms);
        }
    }

    res->pipelines_ext = ggml_metal_pipelines_init();

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    // [CGC watchdog] stop + drain before teardown so the tick cannot touch freed state
    if (ctx->cgc_watchdog_timer) {
        atomic_store_explicit(&ctx->cgc_watchdog_stop, true, memory_order_relaxed);
        dispatch_sync(ctx->cgc_watchdog_queue, ^{}); // wait out any in-flight tick
        dispatch_source_cancel(ctx->cgc_watchdog_timer);
        dispatch_release(ctx->cgc_watchdog_timer);
        ctx->cgc_watchdog_timer = NULL;
        dispatch_release(ctx->cgc_watchdog_queue);
        ctx->cgc_watchdog_queue = NULL;
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

// CGC: wake-poll wait for a Metal command buffer (§8.51, env-gated, default off).
// CGC_WAKE_POLL_US = spin budget in µs before falling back to the blocking wait; 0/absent = off
// (pure blocking waitUntilCompleted). Mirrors turbo's waitForCompletionPolling: spin on the
// command buffer status with sched_yield() between polls; on deadline or Error fall back to the
// blocking wait so real errors are still reported. NOTE: validated as a net loss on llama.cpp's
// graph-level submit model (no per-layer pipeline to overlap, §8.51), kept as a diagnostic env;
// run_n30cache.sh sets it (production profile).
static void cgc_wait_cmd_buf(id<MTLCommandBuffer> cmd_buf) {
    const char * env = getenv("CGC_WAKE_POLL_US");
    const int64_t poll_us = (env && env[0]) ? atoll(env) : 0;
    if (poll_us > 0 && cmd_buf != nil) {
        const int64_t deadline = ggml_time_us() + poll_us;
        for (;;) {
            const MTLCommandBufferStatus status = [cmd_buf status];
            if (status == MTLCommandBufferStatusCompleted ||
                status == MTLCommandBufferStatusError ||
                ggml_time_us() >= deadline) {
                break;
            }
            sched_yield();
        }
    }
    // blocking wait: also reports real errors (fallback on deadline / Error)
    [cmd_buf waitUntilCompleted];
}

int ggml_metal_cgc_done(ggml_metal_t ctx) {
    return atomic_load_explicit(&ctx->cgc_done, memory_order_relaxed);
}

int ggml_metal_cgc_bufs(ggml_metal_t ctx) {
    return ctx->n_cb + 1; // one completion per cmd buffer, n_cb+1 per graph_compute
}

// [CGC 2026-08-29 deadlock watchdog] see struct comment. MTLCommandBufferStatus:
// 0=NotEnqueued 1=Enqueued 2=Committed 3=Scheduled 4=Executing 5=Completed 6=Error
static const char * cgc_cb_status_name(int s) {
    switch (s) {
        case 0: return "NotEnqueued";
        case 1: return "Enqueued";
        case 2: return "Committed";
        case 3: return "Scheduled";
        case 4: return "Executing";
        case 5: return "Completed";
        case 6: return "Error";
        default: return "?";
    }
}

static void cgc_watchdog_dump(ggml_metal_t ctx, int64_t now, int expected, int done, bool enc_stale) {
    @autoreleasepool {
        const int n_cb = ctx->n_cb;
        // [CGC probe] wedge-instant forensics: how long since the LAST completion vs this
        // graph's submission. stale == submit_age => completions stopped the moment this
        // graph was committed; stale << submit_age => progressed then stopped mid-graph.
        const int64_t last_prog = atomic_load_explicit(&ctx->cgc_last_progress_us, memory_order_relaxed);
        const int64_t last_subm = atomic_load_explicit(&ctx->cgc_last_submit_us,   memory_order_relaxed);
        fprintf(stderr,
                "CGC-WATCHDOG: Metal stall: ctx=%p expected=%d done=%d n_cb=%d computes=%d enc_stale=%d pid=%d\n"
                "CGC-WATCHDOG: stale=%lldms (since last completion) submit_age=%lldms wedge_at=%lldms after submit\n"
                "CGC-WATCHDOG: (sample me within the grace window: `sample %d 5 -file /tmp/cgc_sample.txt`)\n",
                (void *) ctx, expected, done, n_cb,
                atomic_load_explicit(&ctx->cgc_n_computes, memory_order_relaxed),
                enc_stale ? 1 : 0, (int) getpid(),
                last_prog > 0 ? (long long) (now - last_prog) / 1000 : -1LL,
                last_subm > 0 ? (long long) (now - last_subm) / 1000 : -1LL,
                (last_prog > 0 && last_subm > 0 && last_prog >= last_subm)
                    ? (long long) (last_prog - last_subm) / 1000 : -1LL,
                (int) getpid());
        for (int i = 0; i <= n_cb; ++i) {
            id<MTLCommandBuffer> cb = ctx->cmd_bufs[i].obj;
            const int64_t create = ctx->cgc_cb_create_us[i];
            const int64_t enc_s  = atomic_load_explicit(&ctx->cgc_encode_start_us[i], memory_order_relaxed);
            const int64_t enc_d  = atomic_load_explicit(&ctx->cgc_encode_done_us[i],  memory_order_relaxed);
            if (cb == nil) {
                fprintf(stderr, "CGC-WATCHDOG: cb[%d] obj=nil (create=%lldms ago)\n", i,
                        create ? (now - create) / 1000 : -1);
                continue;
            }
            const MTLCommandBufferStatus st = [cb status];
            fprintf(stderr, "CGC-WATCHDOG: cb[%d] status=%s(%d) create=%lldms enc_start=%lldms enc_done=%lldms",
                    i, cgc_cb_status_name((int) st), (int) st,
                    create ? (now - create) / 1000 : -1,
                    enc_s  ? (now - enc_s)  / 1000 : -1,
                    enc_d  ? (now - enc_d)  / 1000 : -1);
            if (st == MTLCommandBufferStatusError) {
                NSError * err = [cb error];
                fprintf(stderr, " err=%s", err ? [[err localizedDescription] UTF8String] : "(nil)");
            }
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    }
}

// [CGC probe 2026-08-29] liveness test at stall time. Two probes:
//   A) same-queue: a 16B fill submitted to THE shared mtl_queue — it lands BEHIND the
//      wedged buffers, so ALIVE => the wedge does not block the queue tail (buffer-local
//      dependency); DEAD => the queue is wedged from the stuck position onward.
//   B) fresh-queue: same fill on a brand-new command queue — ALIVE => the DEVICE still
//      processes new work (queue-level wedge; recovery = migrate to a new queue);
//      DEAD => device/kernel-level wedge (only avoid-or-restart).
// Runs at every watchdog dump cadence (15s) while a stall persists, on the watchdog's
// serial queue (blocking <=6s per call). [CGC probe 2026-08-29 repeat-kick fix]
// D3 evidence: a second stall episode in the same process got NO probe (old one-shot
// guard) and hung to the 60s abort. The probe pair is the RECOVERY KICK (new commit
// re-triggers the driver's lost-wakeup scheduler), so it must fire per episode — and
// repeat every 15s within an episode if the first kick does not take. The kernel-state
// capture (system()/sample, ~5s) stays once per episode (cgc_probe_done re-armed on
// recovery) to keep the diagnostic cost bounded.
static void cgc_watchdog_probe(ggml_metal_t ctx) {
    const bool do_capture = !ctx->cgc_probe_done;
    ctx->cgc_probe_done = true;

    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    for (int which = 0; which < 2; ++which) {
        @autoreleasepool {
            id<MTLCommandQueue> q = which == 0 ? queue : [device newCommandQueue];
            if (q == nil) {
                fprintf(stderr, "CGC-WATCHDOG: probe[%s] FAILED to get queue\n", which == 0 ? "same" : "fresh");
                continue;
            }

            const int64_t t0 = ggml_time_us();
            id<MTLBuffer> buf = [device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            {
                id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
                [enc fillBuffer:buf range:NSMakeRange(0, 16) value:0];
                [enc endEncoding];
            }
            [cb commit];

            MTLCommandBufferStatus st = [cb status];
            const int64_t deadline = t0 + 3000000; // 3s
            while (st != MTLCommandBufferStatusCompleted &&
                   st != MTLCommandBufferStatusError &&
                   ggml_time_us() < deadline) {
                usleep(1000);
                st = [cb status];
            }
            const int64_t dt_ms = (ggml_time_us() - t0) / 1000;

            if (st == MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "CGC-WATCHDOG: probe[%s-queue] ALIVE (%lldms)\n", which == 0 ? "same" : "fresh", dt_ms);
            } else if (st == MTLCommandBufferStatusError) {
                fprintf(stderr, "CGC-WATCHDOG: probe[%s-queue] ERROR (%lldms): %s\n",
                        which == 0 ? "same" : "fresh", dt_ms,
                        [[cb error].localizedDescription UTF8String]);
            } else {
                fprintf(stderr, "CGC-WATCHDOG: probe[%s-queue] DEAD (timeout 3000ms, status=%s) — %s\n",
                        which == 0 ? "same" : "fresh", cgc_cb_status_name((int) st),
                        which == 0
                            ? "shared command queue is wedged from the stuck position onward"
                            : "DEVICE/kernel-level wedge: new queues cannot run either");
            }
            [buf release];
            if (which == 1) {
                [q release];
            }
        }
    }

    // [CGC probe] optional kernel-side state capture (CGC_WATCHDOG_CAPTURE=1): GPU scheduler
    // view + memory pressure, ~10s after the wedge — closest we can get to the wedge instant.
    // Once per stall episode (cgc_probe_done re-armed on recovery).
    if (do_capture && getenv("CGC_WATCHDOG_CAPTURE") != NULL) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
            "sh -c 'ioreg -r -d 1 -w 0 -c IOGPUDevice > /tmp/cgc_gpu_probe.txt 2>&1;"
            " memory_pressure > /tmp/cgc_memp_probe.txt 2>&1;"
            " sample %d 5 -file /tmp/cgc_sample_probe.txt > /dev/null 2>&1'",
            (int) getpid());
        const int rc = system(cmd);
        fprintf(stderr, "CGC-WATCHDOG: kernel-state capture %s (ioreg + memory_pressure + sample -> /tmp/cgc_{{gpu,memp,sample}}_probe.txt)\n",
                rc == 0 ? "done" : "failed");
    }
    fflush(stderr);
}

static void cgc_watchdog_tick(ggml_metal_t ctx) {
    if (atomic_load_explicit(&ctx->cgc_watchdog_stop, memory_order_relaxed)) {
        return;
    }
    const int     expected     = atomic_load_explicit(&ctx->cgc_expected, memory_order_relaxed);
    const int     done         = atomic_load_explicit(&ctx->cgc_done,     memory_order_relaxed);
    const int64_t now          = ggml_time_us();
    const int64_t threshold_us = (int64_t) ctx->cgc_watchdog_ms * 1000;

    int64_t last = atomic_load_explicit(&ctx->cgc_last_progress_us, memory_order_relaxed);
    if (last <= 0) {
        last = atomic_load_explicit(&ctx->cgc_last_submit_us, memory_order_relaxed);
    }
    const int64_t stale_us = now - last;

    // encode-stall: an encode worker started but never finished (dispatch_apply hang)
    bool enc_stale = false;
    for (int i = 0; i <= GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        const int64_t s = atomic_load_explicit(&ctx->cgc_encode_start_us[i], memory_order_relaxed);
        const int64_t d = atomic_load_explicit(&ctx->cgc_encode_done_us[i],  memory_order_relaxed);
        if (s > 0 && d == 0 && now - s > threshold_us) {
            enc_stale = true;
            break;
        }
    }

    const bool stalled = (expected > done && stale_us > threshold_us) || enc_stale;

    if (!stalled) {
        if (ctx->cgc_watchdog_fired_us != 0) {
            ctx->cgc_watchdog_fired_us = 0;
            ctx->cgc_watchdog_dump_us  = 0;
            ctx->cgc_probe_done        = false; // re-arm probe+capture for a future episode
            fprintf(stderr, "CGC-WATCHDOG: recovered (progress resumed)\n");
        }
        return;
    }

    if (ctx->cgc_watchdog_fired_us == 0) {
        ctx->cgc_watchdog_fired_us = now;
    }
    if (ctx->cgc_watchdog_dump_us == 0 || now - ctx->cgc_watchdog_dump_us >= 15000000) {
        ctx->cgc_watchdog_dump_us = now;
        cgc_watchdog_dump(ctx, now, expected, done, enc_stale);
        // [CGC probe] run the liveness pair once, right after the first dump
        cgc_watchdog_probe(ctx);
    }
    if (now - ctx->cgc_watchdog_fired_us >= 60000000) {
        GGML_ABORT("CGC-WATCHDOG: Metal stall for %.1fs — aborting (60s sampling grace elapsed)\n",
                   (now - ctx->cgc_watchdog_fired_us) / 1e6);
    }
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    const bool cgc_dbg = getenv("CGC_METAL_DBG") != NULL;
    const int64_t s0 = cgc_dbg ? ggml_time_us() : 0;
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        cgc_wait_cmd_buf(ctx->cmd_buf_last);
        ctx->cmd_buf_last = nil;
        if (cgc_dbg) fprintf(stderr, "CGC-SYNC: wait_last=%dus\n", (int)(ggml_time_us() - s0));
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        const bool cgc_dbg = getenv("CGC_METAL_DBG") != NULL;
        const int64_t a0 = cgc_dbg ? ggml_time_us() : 0;
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_dst release];
        const int64_t a1 = cgc_dbg ? ggml_time_us() : 0;

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        if (cgc_dbg) fprintf(stderr, "CGC-GTA: buf=%dus size=%zu '%s'\n", (int)(a1 - a0), size, tensor->name);
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // [CGC watchdog] per-compute bookkeeping: reset the per-cb timestamps for THIS graph
    atomic_store_explicit(&ctx->cgc_last_submit_us, ggml_time_us(), memory_order_relaxed);
    atomic_fetch_add_explicit(&ctx->cgc_n_computes, 1, memory_order_relaxed);
    for (int i = 0; i <= GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        ctx->cgc_cb_create_us[i] = 0;
        atomic_store_explicit(&ctx->cgc_encode_start_us[i], 0, memory_order_relaxed);
        atomic_store_explicit(&ctx->cgc_encode_done_us[i],  0, memory_order_relaxed);
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            ctx->capture_compute = -1;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;
            ctx->cgc_cb_create_us[n_cb] = ggml_time_us(); // [CGC watchdog]

            // CGC: count this segment's completion so the sched can poll it (CGC_OA_ASYNC
            // pipelined dispatch) without blocking the Metal pipeline
            [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                GGML_UNUSED(cb);
                atomic_fetch_add_explicit(&ctx->cgc_done, 1, memory_order_relaxed);
                atomic_store_explicit(&ctx->cgc_last_progress_us, ggml_time_us(), memory_order_relaxed);
            }];

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;
            ctx->cgc_cb_create_us[cb_idx] = ggml_time_us(); // [CGC watchdog]

            // CGC: count this buffer's completion too so the sched can wait for the WHOLE segment
            // (all n_cb+1 cmd buffers) before firing the top-k hook. Waiting only on the main
            // cmd_buf[n_cb] fired the callback while the argsort (which may land in a secondary
            // buffer) was still running -> stale ids -> garbage remap -> whole-graph corruption.
            [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                GGML_UNUSED(cb);
                atomic_fetch_add_explicit(&ctx->cgc_done, 1, memory_order_relaxed);
                atomic_store_explicit(&ctx->cgc_last_progress_us, ggml_time_us(), memory_order_relaxed);
            }];

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }

        // [CGC drain 2026-08-29] deadlock mitigation experiment (CGC_DRAIN_EVERY=K, 0/absent = off):
        // every K-th graph_compute does a FULL queue drain (waitUntilCompleted on the last-enqueued
        // buffer — FIFO => waits for everything, including the other Metal context's in-flight work).
        // Hypothesis: the intermittent completion wedge builds up in the driver's never-drained
        // firehose (~1900 cb/s for minutes); a periodic quiescent window lets the kernel command
        // queue fully recycle. Cost = one pipeline bubble every K segments (~2-5ms / K).
        {
            static int cgc_drain_every = -1;
            if (cgc_drain_every < 0) {
                const char * env = getenv("CGC_DRAIN_EVERY");
                cgc_drain_every = (env && env[0]) ? atoi(env) : 0;
                if (cgc_drain_every > 0) {
                    GGML_LOG_WARN("%s: CGC periodic queue drain ON (every %d graph_computes)\n",
                            __func__, cgc_drain_every);
                }
            }
            if (cgc_drain_every > 0 &&
                (atomic_load_explicit(&ctx->cgc_n_computes, memory_order_relaxed) % cgc_drain_every) == 0) {
                if (ctx->cmd_buf_last) {
                    const int64_t t0 = ggml_time_us();
                    cgc_wait_cmd_buf(ctx->cmd_buf_last);
                    GGML_LOG_DEBUG("%s: CGC drain took %.1f ms\n", __func__, (ggml_time_us() - t0) / 1000.0);
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        // [CGC watchdog] encode bookkeeping for this cb (start / done)
        atomic_store_explicit(&ctx->cgc_encode_start_us[cb_idx], ggml_time_us(), memory_order_relaxed);

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency,
            ctx->capture_compute,
            ctx->debug_graph,
            ctx->debug_fusion);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
            // [CGC watchdog] a committed buffer's completion handler fires exactly once
            atomic_fetch_add_explicit(&ctx->cgc_expected, 1, memory_order_relaxed);
        }

        // [CGC watchdog] encode done (after commit)
        atomic_store_explicit(&ctx->cgc_encode_done_us[cb_idx], ggml_time_us(), memory_order_relaxed);
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
