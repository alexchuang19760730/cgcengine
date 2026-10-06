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
//
// [CGC 2026-09-18] Raised 8 -> 128 so that CGC_N_CB can ask for a command buffer PER NODE. The
// default stays 8 (run_server.sh's `CGC_SERVER_N_CB`, the §8.93 sweet spot), so nothing moves
// unless a measurement arm sets it: this only enlarges the arrays the encoder indexes.
//
// Why it is needed: `ggml_metal_cgc_gpu_take_cb` attributes each command buffer's GPUStartTime/
// GPUEndTime to the node RANGE that buffer encoded, and the ranges are
// [0, n_nodes_0) + n_cb slices of the rest. With n_nodes_0 = MAX(64, 0.1*N) and n_cb = 8, a
// ~100-node MoE segment gives ONE buffer covering the first 64 nodes -- and the layer's
// topk/gate/up/down live exactly there -- so the best possible split inside it is a node-COUNT
// guess (measured: ffn_moe_gate got 1/64 of its own buffer's duration). With CGC_CB_N_MAIN=1 and
// a large n_cb the slices become one node each and the kind table stops being a guess.
#define GGML_METAL_MAX_COMMAND_BUFFERS 128

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

    // [CGC 2026-10-05 whole-step GPU busy] Monotonic, NEVER reset on this side: the backend reads
    // the totals before/after each step and takes the delta. Why a monotonic fetch_add and not
    // min/max + reset: the first revision of this instrument accumulated in the completion handlers
    // and then `atomic_exchange`d the accumulators, and a handler mid compare-exchange left the
    // min/max wider than the sum (measured: union 1230 ms > busy_sum 746 ms > wait 375 ms --
    // mathematically impossible for one segment; see the note above ggml_metal_cgc_gpu_take).
    // A pure fetch_add has no such window, and a busy SUM is all `gpu_busy_sum / wait` needs.
    // Gated by BOTH CGC_GPU_TIMING_LATE and CGC_GPU_TIMING; off => the handlers never touch these.
    _Atomic int64_t    cgcst_busy_ns;
    _Atomic int64_t    cgcst_n;
    _Atomic int64_t    cgcst_unsup;
    bool               cgc_step_accum;
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

    // [CGC 2026-10-02 overlap fence] Device-side wait so the sched can COMMIT segment i+1 before
    // the host has finished writing segment i's remap leaf, without the racy read.
    //
    // This is option (B) of the CGC_GPU_TIMING note below, implemented instead of inferred: the
    // measured decomposition (2026-10-02, build 054fb22f04a0, ab_interleave p25-gputime vs
    // p25-submit-ahead) puts the WHOLE prize in `wait` -- total 73.54 -> 38.10 ms/step with
    // `wait` 65.20 -> 25.67, while `cb` 2.86 -> 8.35 and `submit` 4.44 -> 4.06 barely move. So the
    // window is the GPU->CPU->GPU round trip at the layer boundary, not GPU execution.
    //
    // Contract: `cgc_fence_arm_v > 0` means "the NEXT ggml_metal_graph_compute must wait for this
    // event value on EVERY command buffer it commits". The host signals after the remap leaf is
    // written. The value is consumed one-shot (reset to 0) so a second graph cannot inherit it.
    //
    // Why EVERY command buffer and not just the first one: a segment's nodes are spread over
    // n_cb+1 command buffers (the main one plus n_cb worker ones), and Metal only guarantees they
    // are *started* in commit order -- it does NOT make them mutually exclusive. The mul_mat_id
    // nodes that consume the remap leaf routinely land in a WORKER buffer (the note on cgc_done
    // below records the mirror-image bug: waiting only on the main buffer fired the hook while the
    // argsort was still running). Fencing only the main buffer therefore left the real consumers
    // unprotected. Measured 2026-10-02, first cut of this fence: CGC-MMID-ASSERT id_oob=1456
    // first=955391402 (garbage ids) -- the exact submit_ahead symptom, and the output was all-NaN.
    //
    // `0` == no fence == upstream behaviour, byte for byte.
    //
    // Why an MTLSharedEvent rather than a spin: `encodeWaitForEvent:value:` parks the GPU without
    // occupying it and without a host round trip, and the read-before-write hazard is removed by
    // construction (the leaf write strictly precedes the signal).
    id<MTLSharedEvent> cgc_fence_ev;
    uint64_t           cgc_fence_arm_v;       // one-shot ARM: wait value for the NEXT graph_compute
    uint64_t           cgc_fence_this_v;      // value in force for the graph_compute in progress
    uint64_t           cgc_fence_sig_v;       // last value the host signalled (must be monotonic)

    // [CGC 2026-09-15 GPU-side timing] No state is kept here on purpose: MTLCommandBuffer
    // records GPUStartTime/GPUEndTime itself, and ggml_metal_cgc_gpu_take() reads them off
    // ctx->cmd_bufs[] at the segment boundary, so this instrument adds no fields, no
    // completion-handler work and no hot-path cost when CGC_GPU_TIMING is unset.
    // Why it exists at all: CGC-DECPROF attributes 91% of a decode step to `wait`, but `wait`
    // is a CPU-side spin on cgc_done, so it cannot tell apart two situations with OPPOSITE
    // fixes:
    //   (A) the GPU really was busy ~72 ms -> n_tokens=1 GEMV is execution/occupancy bound and
    //       the lever is batching a layer's 8 expert GEMVs into one dispatch;
    //   (B) the GPU finished quickly and the window is mostly launch + completion-report
    //       latency -> the lever is removing the GPU->CPU->GPU round trip the segmented
    //       dispatch performs at every layer boundary (the remap leaf).
    // Measure instead of inferring. See ggml_metal_cgc_gpu_take() for the sampling contract.

    struct ggml_cgraph * gf;

    // [CGC 2026-09-18 node-level GPU time] Node-name snapshot for CGC_GPU_NODES.
    //
    // WHY A SNAPSHOT RATHER THAN ctx->gf: ctx->gf holds the CALLER's cgraph, and the segmented
    // dispatcher builds that one in a LOCAL (`ggml-backend.cpp` submit_seg:
    // `struct ggml_cgraph gv = seg_view(s);` then `graph_compute_async(..., &gv)`), so the pointer
    // dangles the instant that call returns -- while the sched-side hook reads the names AFTER that
    // boundary. MEASURED: reading ctx->gf from the hook segfaults inside the load-time warmup
    // (ggml_metal_cgc_node_name did exactly that). Nothing else in this file reads ctx->gf outside
    // graph_compute, which is why the dangle was never hit before. Copying the name POINTERS is
    // sufficient: the ggml_tensor objects themselves live in the model's context for the life of
    // the process, so only the cgraph's node ARRAY is transient.
    //
    // Filled only when the process opted in (CGC_GPU_NODES set), so the default path pays nothing.
    const char * cgc_nm[1024];
    // [CGC 2026-09-18 op-keyed attribution] The OP of the same node. The name-keyed table cannot
    // answer "does a VIEW cost GPU time?" -- a name bucket mixes ops (`ffn_moe_` holds GLU/MUL/
    // SUM_ROWS/VIEW/RESHAPE), and node COUNT is not GPU TIME. With the op available the table can be
    // keyed by OP, and for buffers whose nodes are ALL the same op the per-node cost becomes an
    // exact division instead of a node-count guess. Snapshotted in the same loop as cgc_nm -- same
    // lifetime argument (the ggml_tensor lives in the model context).
    int          cgc_nop[1024];
    int          cgc_nm_n;
    bool         cgc_nm_on;

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

    // [CGC-METAL-FAIL 2026-09-15] rich record of the FIRST command-buffer failure.
    //
    // Why this exists: Metal work is committed asynchronously and only inspected at
    // ggml_metal_synchronize(). Before this change a failed command buffer produced
    // only a GGML_LOG_ERROR line, and the *caller* then read whatever stale bytes
    // happened to be in the output buffer and reported success (HTTP 200 + garbage
    // logits - see the 2026-09-15 "sum=-6.83e38" incident). The record below makes
    // the failure (a) unambiguous, (b) queryable by whoever owns the output tensor,
    // and (c) impossible to lose behind a later failure.
    bool    err_set;                        // first failure recorded
    int     err_cb_idx;                     // which command buffer
    int     err_status;                     // MTLCommandBufferStatus
    int64_t err_at_us;                       // wall clock at detection (NOT err_us: that is a mach macro)
    int64_t err_n_computes;                  // how many graph_computes had been submitted
    char    err_desc[256];                   // localizedDescription, or ""

    // [CGC-METAL-FAIL] fail-stop policy. Default ON: a command-buffer failure aborts
    // the process instead of allowing a caller to observe a stale output buffer.
    // Set CGC_METAL_FAIL_STOP=0 to restore the old log-only behaviour (debugging only).
    bool    fail_stop;
};

// [CGC 2026-09-16 command-buffer lifetime] cmd_buf_last exists so that synchronization can wait on
// "the last buffer this context queued" without having to know which of cmd_bufs[] / cmd_bufs_ext it
// landed in. It used to be a BORROWED pointer: every assignment stored a pointer it did not own
// (graph_compute's per-segment buffers, the get/set/copy async paths) while the owner released it
// underneath -- cmd_bufs[cb_idx].obj is released and replaced on every graph_compute (:1039-1042 and
// :1067-1070) and cmd_bufs_ext is drained with removeAllObjects (:796) -- leaving a window in which
// the pointer refers to a deallocated object and the three readers (synchronize :711, the capture
// path :1001, the drain hook :1175) would message it.
//
// The window is narrow -- the calling thread is both the only mutator and the only reader, and it
// re-points cmd_buf_last later in the same call. Checked, not assumed: the watchdog queue is the
// only other thread that can reach this struct, and it never reads this field (cgc_watchdog_tick /
// _dump / _probe touch cgc_watchdog_* and the per-cb timestamp arrays only). But "narrow" is not an
// ownership argument, and its failure mode is the class this file was already bitten by on
// 2026-09-16 -- ggml_metal_synchronize released a command buffer and *then* read [cmd_buf error],
// turning a nameable GPU OOM into an unreadable SIGSEGV.
//
// Consequence worth stating, because it bounds the fix: since every reader runs on the mutating
// thread, owning the reference is *sufficient* here. A cross-thread reader would additionally need
// its own retain across the read, and any future one has to add it.
//
// So make it owned: retain on assignment, release on re-point. Balanced by cgc_clear_cmd_buf_last,
// by the next cgc_set_cmd_buf_last, and by ggml_metal_free.
static void cgc_set_cmd_buf_last(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf) {
    if (ctx->cmd_buf_last == cmd_buf) {
        return;
    }
    [ctx->cmd_buf_last release];
    ctx->cmd_buf_last = [cmd_buf retain];
}

static void cgc_clear_cmd_buf_last(ggml_metal_t ctx) {
    [ctx->cmd_buf_last release];
    ctx->cmd_buf_last = nil;
}

// [CGC-METAL-FAIL] record the first command-buffer failure. Idempotent: only the
// first failure is kept, so the original cause is never overwritten by cascades.
static void cgc_metal_record_error(ggml_metal_t ctx, int cb_idx, int status, id<MTLCommandBuffer> cmd_buf) {
    if (ctx->err_set) {
        return;
    }

    ctx->err_set        = true;
    ctx->err_cb_idx     = cb_idx;
    ctx->err_status     = status;
    ctx->err_at_us      = ggml_time_us();
    ctx->err_n_computes = (int64_t) atomic_load_explicit(&ctx->cgc_n_computes, memory_order_relaxed);
    ctx->err_desc[0]    = '\0';

    if (cmd_buf != nil && status == MTLCommandBufferStatusError) {
        NSError * err = [cmd_buf error];
        if (err != nil) {
            snprintf(ctx->err_desc, sizeof(ctx->err_desc), "%s", [[err localizedDescription] UTF8String]);
        }
    }

    GGML_LOG_ERROR("CGC-METAL-FAIL: command buffer %d failed with status %d (compute #%lld, %s) desc=%s\n",
            ctx->err_cb_idx, ctx->err_status, (long long) ctx->err_n_computes,
            ctx->fail_stop ? "fail-stop" : "log-only",
            ctx->err_desc[0] ? ctx->err_desc : "(none)");
}

// returns true if the caller must stop consuming GPU output
bool ggml_metal_has_error(ggml_metal_t ctx) {
    return ctx != NULL && ctx->has_error;
}

const char * ggml_metal_error_desc(ggml_metal_t ctx) {
    if (ctx == NULL || !ctx->err_set) {
        return "";
    }

    static char buf[512];
    snprintf(buf, sizeof(buf), "cb=%d status=%d compute#=%lld desc=%s",
            ctx->err_cb_idx, ctx->err_status, (long long) ctx->err_n_computes,
            ctx->err_desc[0] ? ctx->err_desc : "(none)");
    return buf;
}

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

    // [CGC-METAL-FAIL] default ON: never let a caller consume a stale output buffer.
    res->err_set        = false;
    res->err_cb_idx     = -1;
    res->err_status     = 0;
    res->err_at_us      = 0;
    res->err_n_computes = 0;
    res->err_desc[0]    = '\0';
    {
        const char * val = getenv("CGC_METAL_FAIL_STOP");
        res->fail_stop = !(val != NULL && val[0] == '0');
    }
    GGML_LOG_INFO("%s: metal fail-stop = %s (CGC_METAL_FAIL_STOP=0 to disable)\n",
            __func__, res->fail_stop ? "ON" : "OFF");

    res->gf = nil;
    // [CGC 2026-09-18 node-level GPU time] opt-in, read once. The snapshot stays empty unless
    // CGC_GPU_NODES is set, so the default path neither copies nor reads cgc_nm[].
    res->cgc_nm_n  = 0;
    res->cgc_nm_on = getenv("CGC_GPU_NODES") != NULL;
    res->encode_async = nil;
    // `<=`: slot n_cb is the main thread's, and n_cb can be GGML_METAL_MAX_COMMAND_BUFFERS
    for (int i = 0; i <= GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
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
    // [CGC 2026-10-05 whole-step GPU busy] see the struct comment
    atomic_store_explicit(&res->cgcst_busy_ns, 0, memory_order_relaxed);
    atomic_store_explicit(&res->cgcst_n,       0, memory_order_relaxed);
    atomic_store_explicit(&res->cgcst_unsup,   0, memory_order_relaxed);
    res->cgc_step_accum = getenv("CGC_GPU_TIMING_LATE") != NULL &&
                          getenv("CGC_GPU_TIMING")      != NULL;
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

    // [CGC 2026-09-16 command-buffer lifetime] drop the owned reference before the arrays it may
    // alias are torn down. cmd_buf_last holds its own retain (cgc_set_cmd_buf_last), so this is not
    // a use-after-free guard -- it is the release that balances it.
    cgc_clear_cmd_buf_last(ctx);

    // `<=`: slot n_cb is the main thread's, and n_cb can be GGML_METAL_MAX_COMMAND_BUFFERS
    for (int i = 0; i <= GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
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

// [CGC 2026-10-05 whole-step GPU busy] Read the monotonic accumulator; `out` = {busy_ns, n, unsup}.
// Pure read, no reset -- the caller takes deltas between two reads. See the struct comment for why
// this side must not reset (the ABA window that killed the first revision) and why the sum, not the
// span, is what `gpu_busy_sum / wait` needs. Inert unless CGC_GPU_TIMING_LATE is set.
int ggml_metal_cgc_step_busy(ggml_metal_t ctx, int64_t * out) {
    if (ctx == NULL || out == NULL) {
        return 0;
    }
    out[0] = atomic_load_explicit(&ctx->cgcst_busy_ns, memory_order_relaxed);
    out[1] = atomic_load_explicit(&ctx->cgcst_n,       memory_order_relaxed);
    out[2] = atomic_load_explicit(&ctx->cgcst_unsup,   memory_order_relaxed);
    return 3;
}

// [CGC 2026-10-02 overlap fence] ARM: the NEXT ggml_metal_graph_compute's first command buffer
// will `encodeWaitForEvent` for value `v`. One-shot (consumed by that graph_compute). v must be
// strictly increasing across the process lifetime, because MTLSharedEvent::setSignaledValue:
// requires monotonic values -- the caller uses a global counter.
void ggml_metal_cgc_fence_arm(ggml_metal_t ctx, uint64_t v) {
    ctx->cgc_fence_arm_v = v;
}

// SIGNAL: called by the sched AFTER the remap leaf for this boundary has been written (i.e. right
// after the top-k hook returns). This is the whole point: the GPU cannot get past the fence until
// the leaf is in memory, so the read-before-write hazard is removed by construction rather than by
// prediction. Idempotent and monotonic: a value <= the last signalled one is ignored.
void ggml_metal_cgc_fence_signal(ggml_metal_t ctx, uint64_t v) {
    if (ctx->cgc_fence_ev == nil) {
        id<MTLDevice> dev = ggml_metal_device_get_obj(ctx->dev);
        if (dev == nil) {
            return;
        }
        ctx->cgc_fence_ev = [dev newSharedEvent];
    }
    if (ctx->cgc_fence_ev == nil) {
        return;
    }
    if (v > ctx->cgc_fence_sig_v) {
        ctx->cgc_fence_sig_v = v;
        [ctx->cgc_fence_ev setSignaledValue:v];
    }
}

// [CGC 2026-10-02 overlap fence] probe: how many fences this ctx has armed / signalled, and whether
// the event exists. Exposed so the sched can print a "the instrument is alive" line instead of
// inferring it from a t/s difference (the same rule as the CGC_SUBMIT_AHEAD `answer_md5` guard).
void ggml_metal_cgc_fence_stats(ggml_metal_t ctx, int64_t * out) {
    out[0] = ctx->cgc_fence_ev != nil ? 1 : 0;
    out[1] = (int64_t) ctx->cgc_fence_sig_v;
}

// [CGC 2026-09-15 GPU-side timing] Read the GPU start/end that Metal itself recorded for the
// command buffers of the MOST RECENT graph_compute. See the struct comment for why this exists.
//
// Why it reads ctx->cmd_bufs[] instead of sampling in addCompletedHandler: each dispatch segment
// is its own ggml_backend_graph_compute_async call, so at a segment boundary ctx->cmd_bufs[]
// holds exactly that segment's n_cb+1 buffers, and the caller reaches here only after the
// cgc_done poll confirmed all of them completed -- the only point where MTLCommandBuffer
// reports GPUStartTime/GPUEndTime. A first attempt accumulated per-buffer samples in atomics
// from the completion handlers; resetting those accumulators with atomic_exchange while a
// handler was mid compare-exchange left the min/max wider than the sum (measured: union 1230 ms
// > busy_sum 746 ms > wait 375 ms -- mathematically impossible for one segment), i.e. a textbook
// ABA race on the reset. Reading the finished objects directly needs no shared state at all,
// cannot race (the main thread is the only mutator), and costs nothing when the env is unset
// because the caller then never calls this.
//
// `out` must have 7 elements:
//   out[0] = GPU busy sum   (ns)  -- sum over the segment's cmd buffers
//   out[1] = GPU busy union (ns)  -- max(end) - min(start); << out[0] means the buffers overlap
//   out[2] = earliest GPU start (ns, mach absolute clock -- comparable across segments)
//   out[3] = latest   GPU end   (ns, same clock)
//   out[4] = buffers without a usable timestamp (no buffer / not completed / 0.0 / NaN)
//   out[5] = ... of those, slot had NO buffer at all   (structural)
//   out[6] = ... of those, buffer existed but was not Completed (timing / memory pressure)
//            (bad timestamp = out[4] - out[5] - out[6])
// Returns the number of buffers that contributed. All zeros with a nonzero out[4] means the
// platform does not report the timestamps and the line means nothing.
// [CGC 2026-09-27] out[5]/out[6] were added because out[4] alone could not tell the two apart,
// and the two have opposite remedies: a structural nil is a property of the n_main split, while
// a not-completed buffer is a property of the window.
int ggml_metal_cgc_gpu_take(ggml_metal_t ctx, int64_t * out) {
    out[0] = out[1] = out[2] = out[3] = out[4] = 0;
    out[5] = out[6] = 0;
    if (ctx == NULL) {
        return 0;
    }
    int64_t busy = 0;
    int64_t s_min = INT64_MAX;
    int64_t e_max = INT64_MIN;
    int n = 0;
    int unsup = 0;
    int unsup_nil = 0;      // [CGC 2026-09-27] slot 沒有 buffer（結構性：該 worker 分不到節點）
    int unsup_nc   = 0;     // [CGC 2026-09-27] buffer 存在但還沒 Completed（運行時：時序／內存壓力）
    const int n_bufs = ctx->n_cb + 1;
    for (int i = 0; i < n_bufs && i <= GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        id<MTLCommandBuffer> cb = ctx->cmd_bufs[i].obj;
        // ★ 2026-09-27：這兩個條件原本寫在同一個 `||` 裡，共用一個 `unsup` 計數器，
        //   於是在報表上完全分不出「這個 slot 根本沒被用」與「buffer 還沒跑完」——
        //   而這兩者的處置完全不同：前者是結構（n_main 切分造成），後者是時序（窗口／內存壓力）。
        //   實測需要它：同為 n_main=64，ctl64b 是 3.1% 而 off1 是 33.0%，差十倍卻無法歸因。
        if (cb == nil) {
            unsup++;
            unsup_nil++;
            continue;
        }
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            unsup++;
            unsup_nc++;
            continue;
        }
        const CFTimeInterval s = [cb GPUStartTime];
        const CFTimeInterval e = [cb GPUEndTime];
        if (!(s > 0.0) || !(e > s)) {
            unsup++;
            continue;
        }
        const int64_t s_ns = (int64_t) (s * 1e9);
        const int64_t e_ns = (int64_t) (e * 1e9);
        busy += e_ns - s_ns;
        if (s_ns < s_min) { s_min = s_ns; }
        if (e_ns > e_max) { e_max = e_ns; }
        n++;
    }
    if (n > 0) {
        out[0] = busy;
        out[1] = e_max - s_min;
        out[2] = s_min;
        out[3] = e_max;
    }
    out[4] = unsup;          // 總數（向後相容：報表上的 `skipped` 仍讀這個）
    out[5] = unsup_nil;      // 細分：slot 沒有 buffer（結構）
    out[6] = unsup_nc;       // 細分：buffer 未 Completed（時序）；bad-ts = out[4]-out[5]-out[6]
    return n;
}

// [CGC 2026-09-18 node-level GPU time] The SAME per-command-buffer timestamps as
// ggml_metal_cgc_gpu_take(), but NOT collapsed into one segment span: one record per command
// buffer, each carrying the node index range that buffer encoded.
//
// Why the range is derivable and why this needs no new state, no sampling and no
// MTLCounterSampleBuffer: ggml_metal_graph_compute already splits the graph into n_cb+1 command
// buffers with FIXED index ranges (the idx_start/idx_end computation in the encode callback). The
// main thread's buffer -- slot n_cb -- encodes nodes [0, n_nodes_0); worker slot i < n_cb encodes
// [n_nodes_0 + i*n_nodes_per_cb, n_nodes_0 + min(...)). n_cb, n_nodes_0, n_nodes_per_cb and gf are
// all still set on ctx when the sched reads at the segment boundary, i.e. the same instant at which
// ggml_metal_cgc_gpu_take() reads the same finished MTLCommandBuffers.
//
// What it CAN and CANNOT answer. Each buffer is a GROUP of nodes, so the node identity inside a
// buffer is not separated by the clock: what this hands back is a duration per contiguous node
// range. The consumer therefore attributes a buffer's duration across the node KINDS in its range
// (by node count), which is enough to answer a question of the form "does ffn_moe_* carry a large
// share of the GPU time" but NOT "node X took Y ms". Within one segment the buffers overlap, so
// summing their durations is a BUSY measure, the same quantity the per-segment `gpu` field reports.
//
// `out` holds max_cb records of 5 int64: {start_ns, end_ns, first_node, last_node, ok}.
// Returns the number of records written (one per command-buffer slot 0..n_cb, ok=0 for a slot whose
// buffer is missing / not completed / has no usable timestamp). Node NAMES are read separately
// through ggml_metal_cgc_node_name(), so this side never copies strings.
int ggml_metal_cgc_gpu_take_cb(ggml_metal_t ctx, int64_t * out, int max_cb) {
    if (ctx == NULL || out == NULL || max_cb <= 0) {
        return 0;
    }
    const int n_bufs = ctx->n_cb + 1;
    const int p      = ctx->n_nodes_per_cb;
    int n = 0;
    for (int i = 0; i < n_bufs && i <= GGML_METAL_MAX_COMMAND_BUFFERS && n < max_cb; ++i) {
        int64_t * rec = out + 5*n;
        rec[0] = rec[1] = 0;
        rec[4] = 0;
        // the node range this slot encoded -- mirrors the encode callback's idx_start/idx_end
        if (i == ctx->n_cb) {
            rec[2] = 0;
            rec[3] = ctx->n_nodes_0;
        } else {
            rec[2] = ctx->n_nodes_0 + (int64_t) i * p;
            rec[3] = ctx->n_nodes_0 + (int64_t) MIN((i == ctx->n_cb - 1) ? ctx->n_nodes_1 : (i + 1) * p,
                                                    ctx->n_nodes_1);
        }
        id<MTLCommandBuffer> cb = ctx->cmd_bufs[i].obj;
        if (cb != nil && [cb status] == MTLCommandBufferStatusCompleted) {
            const CFTimeInterval s = [cb GPUStartTime];
            const CFTimeInterval e = [cb GPUEndTime];
            if (s > 0.0 && e > s) {
                rec[0] = (int64_t) (s * 1e9);
                rec[1] = (int64_t) (e * 1e9);
                rec[4] = 1;
            }
        }
        n++;
    }
    return n;
}

// Name of node `node_idx` of the graph the MOST RECENT graph_compute was handed. Returns NULL when
// there is no such node. Reads the snapshot taken at graph_compute time -- NOT ctx->gf, which
// dangles by the time the sched-side hook asks (see the cgc_nm struct comment).
const char * ggml_metal_cgc_node_name(ggml_metal_t ctx, int node_idx) {
    if (ctx == NULL || node_idx < 0 || node_idx >= ctx->cgc_nm_n) {
        return NULL;
    }
    return ctx->cgc_nm[node_idx];
}

// [CGC 2026-09-18 op-keyed attribution] Same snapshot, same lifetime, same "the most recent
// graph_compute" semantics as ggml_metal_cgc_node_name above. Returns the ggml_op enum value, or -1
// when there is no such node. Only meaningful while CGC_GPU_NODES is set (that is what fills the
// snapshot); callers glue the two accessors together, so a -1 here means "no node", not "no op".
int ggml_metal_cgc_node_op(ggml_metal_t ctx, int node_idx) {
    if (ctx == NULL || node_idx < 0 || node_idx >= ctx->cgc_nm_n) {
        return -1;
    }
    return ctx->cgc_nop[node_idx];
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
        cgc_clear_cmd_buf_last(ctx);
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
                cgc_metal_record_error(ctx, cb_idx, (int) status, cmd_buf);
                if (ctx->fail_stop) {
                    // [CGC-METAL-FAIL] The graph never ran: every output tensor this compute
                    // was supposed to produce still holds whatever the previous compute left
                    // there. Returning normally would let the caller read that stale data as
                    // a valid result (the 2026-09-15 garbage-logits incident). Stop instead.
                    GGML_ABORT("CGC-METAL-FAIL: command buffer %d failed (status %d, %s) - refusing to return stale output; "
                               "lower Metal memory pressure (expert cache / -ub) or set CGC_METAL_FAIL_STOP=0 to override\n",
                               cb_idx, (int) status, ctx->err_desc[0] ? ctx->err_desc : "no description");
                }
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

                // [CGC 2026-09-16] Record the failure BEFORE releasing anything. This call must stay
                // above the release loop: cgc_metal_record_error() reads [cmd_buf error] to fill
                // err_desc, and `cmd_buf` is released on BOTH lanes below -- explicitly as
                // cmd_bufs_ext[i] (j == i) and again by removeAllObjects, which is exactly the two
                // retains this array holds (addObject + the explicit retain at the add site). So by
                // the time the old code reached it, cmd_buf was already deallocated and [cmd_buf error]
                // was a use-after-free.
                //
                // Measured, not inferred: llama-server-2026-09-16-113207.ips says objc_msgSend <-
                // ggml_metal_synchronize + 0x13e04 (= imageOffset 81412), and that offset is the cbz
                // right after `bl _objc_msgSend$error`; disassembling the pre-fix lib shows the
                // releases at 0x13d7c (loop) and 0x13db8 (removeAllObjects), that msgSend at 0x13e00.
                // The damage was asymmetric: a command buffer failing with kIOGPU...OutOfMemory
                // (status 5) during req2 of prefill250 killed the process with EXC_BAD_ACCESS instead
                // of the intended CGC-METAL-FAIL abort, so the one fact -- GPU OOM -- never printed.
                ctx->has_error = true;
                cgc_metal_record_error(ctx, (int) i, (int) status, cmd_buf);

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                if (ctx->fail_stop) {
                    GGML_ABORT("CGC-METAL-FAIL: extra command buffer %d failed (status %d, %s) - refusing to return stale output; "
                               "set CGC_METAL_FAIL_STOP=0 to override\n",
                               (int) i, (int) status, ctx->err_desc[0] ? ctx->err_desc : "no description");
                }
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    // [CGC 2026-09-15 S1 kernel-side ids capture] Reached only when every command buffer reported
    // Completed -- the failing paths returned above. So the bytes captured by
    // kernel_cgc_ids_capture are now host-visible and can be emitted. Slots are consumed by a
    // cursor, so calling this from each of the many synchronize points a segmented dispatch
    // performs emits each captured value exactly once.
    ggml_metal_cgc_ids_dump();
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
        cgc_set_cmd_buf_last(ctx, cmd_buf);

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
        cgc_set_cmd_buf_last(ctx, cmd_buf);

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
        cgc_set_cmd_buf_last(ctx_src, cmd_buf);

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
    //
    // [CGC 2026-09-18] CGC_CB_N_MAIN is a MEASUREMENT-ONLY override of the MAX(64, ...) floor.
    // That floor is what makes the node-kind attribution coarse, and it is structural rather than
    // cosmetic: every segment's FIRST >= 64 nodes go into one command buffer, and for a MoE layer
    // those are exactly the topk/argsort/gate/up/weights/down nodes that the attribution is about
    // (measured with CGC_GPU_NODES_TRACE: the 64-node buffer's head is
    // `ffn_moe_topk-23 ffn_moe_gate-23 ffn_moe_up-23 ffn_gate-23 ...`). Lowering it together with a
    // large CGC_N_CB turns the n_cb+1 buffers into one-node slices, which is what makes the kind
    // table a measurement instead of a node-count guess. Unset => upstream behaviour, byte for byte.
    const char * cgc_cb_nmain = getenv("CGC_CB_N_MAIN");
    const int n_main = cgc_cb_nmain != NULL && cgc_cb_nmain[0] != '\0'
                     ? MAX(1, atoi(cgc_cb_nmain))
                     : MAX(64, 0.1*gf->n_nodes);

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

        // [CGC 2026-10-02 overlap fence] Consume the one-shot ARM here, at the top, before ANY
        // command buffer exists, and publish it for this whole graph_compute. Reading it at the top
        // is what guarantees the value cannot leak into a later graph: by the time the second graph
        // of the process runs, cgc_fence_arm_v is already 0 again. The value is then put on EACH
        // command buffer in encode_async (see the struct note on why the main buffer alone is not
        // enough). Event creation is lazy and one-time per context.
        ctx->cgc_fence_this_v = ctx->cgc_fence_arm_v;
        ctx->cgc_fence_arm_v  = 0;
        if (ctx->cgc_fence_this_v > 0 && ctx->cgc_fence_ev == nil) {
            id<MTLDevice> fence_dev = ggml_metal_device_get_obj(ctx->dev);
            if (fence_dev != nil) {
                ctx->cgc_fence_ev = [fence_dev newSharedEvent];
            }
        }

        // [CGC 2026-09-18 node-level GPU time] Snapshot the node names while gf is still the live
        // object -- see the struct comment on cgc_nm for why reading ctx->gf from the hook
        // segfaults instead. Taken BEFORE the thread pool is started, so no worker races this.
        if (ctx->cgc_nm_on) {
            ctx->cgc_nm_n = MIN(gf->n_nodes, (int) (sizeof(ctx->cgc_nm) / sizeof(ctx->cgc_nm[0])));
            for (int i = 0; i < ctx->cgc_nm_n; ++i) {
                ctx->cgc_nm[i]  = gf->nodes[i] != NULL ? gf->nodes[i]->name : NULL;
                ctx->cgc_nop[i] = gf->nodes[i] != NULL ? (int) gf->nodes[i]->op : -1;
            }
        }

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
                cgc_clear_cmd_buf_last(ctx);
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

            // [CGC 2026-10-02 overlap fence] The wait is NOT encoded here: this buffer is only ONE
            // of the n_cb+1 that make up the segment, and fence-on-main-alone was measured WRONG
            // (garbage ids -> all-NaN). It is encoded inside encode_async, which runs for every
            // buffer including this one.
            // CGC: count this segment's completion so the sched can poll it (CGC_OA_ASYNC
            // pipelined dispatch) without blocking the Metal pipeline
            [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                GGML_UNUSED(cb);
                atomic_fetch_add_explicit(&ctx->cgc_done, 1, memory_order_relaxed);
                atomic_store_explicit(&ctx->cgc_last_progress_us, ggml_time_us(), memory_order_relaxed);
                // [CGC 2026-10-05 whole-step GPU busy] GPUStartTime/GPUEndTime are only valid once
                // the buffer is COMPLETED, which is exactly here -- and this handler is the only
                // place that sees EVERY buffer of EVERY segment. With the overlap fence on, the CPU
                // skips the per-segment waits, so no backend read point can: the array below is
                // recycled per segment (n_cb+1 = 9 slots) and holds only the tail segment by the time
                // the backend runs. Pure fetch_add, no reset => no ABA window.
                if (ctx->cgc_step_accum) {
                    const CFTimeInterval s = [cb GPUStartTime];
                    const CFTimeInterval e = [cb GPUEndTime];
                    const CFTimeInterval d = e - s;
                    // The 10 s bound is a plausibility filter, not a tuning knob: a single command
                    // buffer in a decode step runs for microseconds, and MEASURED on the first run a
                    // few buffers report absurd spans (the SAME pathology the per-segment accessor
                    // shows: gpu_busy_sum 2.0e8 / 1.4e8 ms) which would poison a whole step's delta.
                    // Rejecting them makes the count visible in `unsup` instead of hiding the bug.
                    if (s > 0.0 && e > s && d < 10.0) {
                        atomic_fetch_add_explicit(&ctx->cgcst_busy_ns,
                                                  (int64_t) (d * 1e9), memory_order_relaxed);
                        atomic_fetch_add_explicit(&ctx->cgcst_n, 1, memory_order_relaxed);
                    } else {
                        atomic_fetch_add_explicit(&ctx->cgcst_unsup, 1, memory_order_relaxed);
                    }
                }
            }];

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        cgc_set_cmd_buf_last(ctx, ctx->cmd_bufs[n_cb].obj);

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
                // [CGC 2026-10-05 whole-step GPU busy] GPUStartTime/GPUEndTime are only valid once
                // the buffer is COMPLETED, which is exactly here -- and this handler is the only
                // place that sees EVERY buffer of EVERY segment. With the overlap fence on, the CPU
                // skips the per-segment waits, so no backend read point can: the array below is
                // recycled per segment (n_cb+1 = 9 slots) and holds only the tail segment by the time
                // the backend runs. Pure fetch_add, no reset => no ABA window.
                if (ctx->cgc_step_accum) {
                    const CFTimeInterval s = [cb GPUStartTime];
                    const CFTimeInterval e = [cb GPUEndTime];
                    const CFTimeInterval d = e - s;
                    // The 10 s bound is a plausibility filter, not a tuning knob: a single command
                    // buffer in a decode step runs for microseconds, and MEASURED on the first run a
                    // few buffers report absurd spans (the SAME pathology the per-segment accessor
                    // shows: gpu_busy_sum 2.0e8 / 1.4e8 ms) which would poison a whole step's delta.
                    // Rejecting them makes the count visible in `unsup` instead of hiding the bug.
                    if (s > 0.0 && e > s && d < 10.0) {
                        atomic_fetch_add_explicit(&ctx->cgcst_busy_ns,
                                                  (int64_t) (d * 1e9), memory_order_relaxed);
                        atomic_fetch_add_explicit(&ctx->cgcst_n, 1, memory_order_relaxed);
                    } else {
                        atomic_fetch_add_explicit(&ctx->cgcst_unsup, 1, memory_order_relaxed);
                    }
                }
            }];

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                cgc_set_cmd_buf_last(ctx, cmd_buf);
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

    // [CGC-METAL-FAIL] Re-check at every exit. A worker thread or an awaited command buffer
    // can have latched a failure while this graph was being encoded - the early-entry check
    // above only covers failures from *previous* calls. Without this, a compute that already
    // knows it failed would still report GGML_STATUS_SUCCESS to the scheduler.
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: returning FAILED - backend latched a command buffer error during this compute (%s)\n",
                __func__, ctx->err_set ? ggml_metal_error_desc(ctx) : "no detail");
        return GGML_STATUS_FAILED;
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
        cgc_set_cmd_buf_last(ctx, cmd_buf);

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
        cgc_set_cmd_buf_last(ctx, cmd_buf);

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

        // [CGC 2026-10-02 overlap fence] Encode the device wait at the HEAD of EVERY buffer of a
        // fenced graph, before any node is encoded into it. Fencing only the main buffer left the
        // mul_mat_id consumers (which live in arbitrary worker buffers) unprotected -- see the
        // struct note. cgc_fence_this_v is written once per graph_compute before any thread runs and
        // is read-only here, so the parallel dispatch_apply workers need no further synchronization.
        if (ctx->cgc_fence_this_v > 0 && ctx->cgc_fence_ev != nil) {
            [cmd_buf encodeWaitForEvent:ctx->cgc_fence_ev value:ctx->cgc_fence_this_v];
        }

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

// [CGC 2026-09-24] Batched speculative decode verify implementation
// Adapted from oMLX bonsai/spec_decode.metal (MIT License)
// Performs accept/reject comparison for K draft tokens across B batch rows on GPU.
bool ggml_metal_spec_decode_verify(
    ggml_metal_t ctx,
    const void * draft,      // [B, K] int32
    const void * target,     // [B, K+1] int32
    void * n_accepted,       // [B] int32 output
    void * committed,        // [B, K+1] int32 output
    int K, int B) {

    if (B <= 0 || K <= 0) return false;

    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        if (!device) return false;

        // Create or retrieve the pipeline state (cached in static for reuse)
        static id<MTLComputePipelineState> cached_pipeline = nil;
        if (!cached_pipeline) {
            NSError *error = nil;
            const char *kernel_src =
                "[[kernel]] void kernel_spec_decode_verify(\n"
                "    const device int* draft [[buffer(0)]],\n"
                "    const device int* target [[buffer(1)]],\n"
                "    device int* n_accepted [[buffer(2)]],\n"
                "    device int* committed [[buffer(3)]],\n"
                "    constant int& K [[buffer(4)]],\n"
                "    constant int& B [[buffer(5)]],\n"
                "    uint b [[thread_position_in_grid]]) {\n"
                "    if (b >= uint(B)) return;\n"
                "    int n = K;\n"
                "    for (int j = 0; j < K; ++j) {\n"
                "        if (draft[b*K + j] != target[b*(K+1) + j]) { n = j; break; }\n"
                "    }\n"
                "    n_accepted[b] = n;\n"
                "    for (int j = 0; j < K + 1; ++j)\n"
                "        committed[b*(K+1)+j] = (j < n) ? draft[b*K+j] : (j == n ? target[b*(K+1)+n] : 0);\n"
                "}\n";
            NSString *src_str = [NSString stringWithUTF8String:kernel_src];
            id<MTLLibrary> lib = [device newLibraryWithSource:src_str options:nil error:&error];
            if (!lib || error) {
                GGML_LOG_ERROR("%s: failed to compile spec_decode_verify library: %s\n",
                    __func__, error ? [[error description] UTF8String] : "unknown");
                return false;
            }
            id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_spec_decode_verify"];
            cached_pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
            [fn release];
            [lib release];
            if (!cached_pipeline || error) {
                GGML_LOG_ERROR("%s: failed to create pipeline: %s\n",
                    __func__, error ? [[error description] UTF8String] : "unknown");
                return false;
            }
        }

        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (!command_buffer) return false;

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:cached_pipeline];

        // Set buffers (index 0-3) and constants (index 4-5)
        [encoder setBuffer:(id<MTLBuffer>)draft   offset:0 atIndex:0];
        [encoder setBuffer:(id<MTLBuffer>)target  offset:0 atIndex:1];
        [encoder setBuffer:(id<MTLBuffer>)n_accepted offset:0 atIndex:2];
        [encoder setBuffer:(id<MTLBuffer>)committed offset:0 atIndex:3];
        [encoder setBytes:&K length:sizeof(int) atIndex:4];
        [encoder setBytes:&B length:sizeof(int) atIndex:5];

        // Dispatch: B threads, one per batch row
        int tg = (int)[cached_pipeline maxTotalThreadsPerThreadgroup];
        if (tg > B) tg = B;
        if (tg < 1) tg = 1;

        // [CGC 2026-09-29 DISPATCH CENSUS v2] this is a real kernel on a real command buffer, but it
        // is the ONE dispatch in the Metal backend that does not go through
        // ggml_metal_encoder_dispatch_threadgroups -- so the census in ggml-metal-ops.cpp cannot see
        // it, not even with the v2 encoder macro (which only rewrites calls in its own translation
        // unit; `dispatchThreads:` is an ObjC message send and no macro can name it). Report it here
        // instead. Counted as `direct` (no graph node is in flight). No-op unless
        // CGC_DISPATCH_CENSUS=1 -- this call is what makes the census a total rather than a bound.
        cgc_dispatch_census_direct("kernel_spec_decode_verify");
        [encoder dispatchThreads:MTLSizeMake(B, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.error != nil) {
            GGML_LOG_ERROR("%s: Metal error: %s\n", __func__, [[command_buffer.error description] UTF8String]);
            return false;
        }
    }

    return true;
}
