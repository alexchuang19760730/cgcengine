#pragma once

// ============================================================================================
// CGC shape knob table -- 2026-09-22
//
// WHY THIS FILE EXISTS. The four axes that still decide whether this engine can reach
// 25 t/s decode are NOT in the expert-cache request path (see
// docs/IOCACHE_CONSTRAINT_ADMISSION_2026-09-22.md §2/#1 #5 #6 #7): they are the step's WIDTH,
// the dense-vs-pool memory split, the graph's command-buffer shape, and the GDN recurrence.
// Today each of them is read ad hoc by whichever translation unit happens to need it -- one
// parser inside `llama-graph.h` for the width, one inside `llama-context.cpp` for the cb count,
// one CLI arg for the pool budget -- so there is no single place that can answer
// "what shape did this process actually run, and was the request even honoured".
//
// That is the failure shape this repo keeps paying for (see the repo's own instrument notes):
// a knob that silently does not land looks exactly like a knob that was measured and found
// useless. These two functions exist to make that distinguishable:
//
//   cgc_shape_knobs()   INPUT   parse every shape env ONCE, cross-validate the axes against the
//                               hard constraints (slot ceiling, memory budget), and record per
//                               axis whether the request was APPLIED / CLAMPED / CONFLICT /
//                               ABSENT. Nothing here is allowed to fail silently.
//   cgc_shape_report_*() OUTPUT print ONE machine-parsable `CGC-SHAPE` line carrying the
//                               REALIZED geometry plus the cache's own counters, including the
//                               QUALITY sentinels, so a searcher can accept a configuration
//                               without having to multiply instruments.
//
// Nothing in this file prints more than one line per phase and nothing here changes arithmetic:
// the report MOVES numbers from the engine to the harness, it does not compute derived
// quantities that would need their own validation.
// ============================================================================================

#include <cstddef>
#include <cstdint>

struct llama_expert_cache;

// How a requested knob value ended up. The harness must treat anything other than APPLIED /
// DEFAULTED as "this arm did not test what it claims to have tested".
enum cgc_shape_status {
    CGC_SHAPE_DEFAULTED = 0,  // no env; the engine default stands
    CGC_SHAPE_APPLIED,        // requested and honoured
    CGC_SHAPE_CLAMPED,        // requested but bounded by a hard constraint -> value differs
    CGC_SHAPE_CONFLICT,       // two env names disagreed; one won and the loser is printed
    CGC_SHAPE_ABSENT,         // the axis has no implementation in this build (request ignored)
    CGC_SHAPE_REFUSED,        // the request is unsafe in this configuration -> default kept
};

const char * cgc_shape_status_name(int s);

struct cgc_shape_cfg {
    // ---- axis A: step width (columns per forward, i.e. M) ------------------------------------
    int      m_req        = 0;        // 0 = not requested
    int      m_val        = 8;        // value this process actually uses
    int      m_status     = CGC_SHAPE_DEFAULTED;
    int      m_conflict   = 0;        // 1 = both CGC_SHAPE_M and CGC_POOL_MAX_TOKENS set, differed
    int      m_other      = 0;        // the losing value, printed so the conflict is checkable
    uint32_t width_realized     = 0;  // the decode graph width after the routable bound
    uint64_t routable_slots     = 0;  // min over pooled layers of usable slots
    uint32_t top_k              = 0;  // n_expert_used
    uint32_t union_at_width     = 0;  // top_k * width_realized: experts demanded in one step
    int      union_fits_pool    = 1;  // union_at_width <= routable_slots
    int      width_limit_hit    = 0;  // the routable bound bound it (cap was higher)

    // ---- axis B: dense-vs-pool memory split ---------------------------------------------------
    // NOT bytes. `llama_model::expert_cache_pool_capacity` is the per-layer SLOT COUNT the loader
    // derived from the byte budget (8 GiB /(41 layers x ~1.391 MiB per slot) = 143). The model does
    // not retain the byte figure, so printing a "bytes" column here would be a second derivation
    // that nobody could check -- the realized quantity we CAN report is the slot count, and it is
    // also the one every derived constraint is written against (union = top_k * M <= routable).
    size_t   pool_cap_slots = 0;      // per-layer slot capacity (0 = pool off)
    uint32_t slots_layer = 0;         // slots per layer as realized by the loader
    uint32_t n_expert    = 0;

    // ---- axis C: graph command-buffer shape ---------------------------------------------------
    int      ncb_req  = 0;
    int      ncb_val  = -1;           // -1 = not requested (ggml default stands)
    int      ncb_status = CGC_SHAPE_DEFAULTED;

    // ---- axis D: GDN recurrence IMPLEMENTATION ------------------------------------------------
    // "Chunked vs step-at-a-time" is not what this axis says, and the file was wrong when it was
    // first written (2026-09-22 corrected). The tree has THREE implementations and the fused one
    // already covers both shapes: ggml_gated_delta_net takes K state snapshots so it serves the
    // T==1 recurrence and the T>1 rolled-up recurrence out of one Metal kernel
    // (`kernel_gated_delta_net_<type>_<nsg>`, function constants ne20/ne30/K). Dispatch is
    // delta-net-base.cpp:430-451: T==1 -> fused if cparams.fused_gdn_ar else the manual AR form,
    // T>1 -> fused if cparams.fused_gdn_ch else the manual chunking form.
    // What an env var can therefore change is WHICH IMPLEMENTATION runs, and turning the fused
    // one off is an ABLATION (expected slower, and outside what the bit-identity anchor covers)
    // -- never a tuning knob.
    int      gdn_req = -1;            // last request seen (either name)
    int      gdn_ar_req = -1;         // CGC_SHAPE_GDN_AR: 1 = keep fused, 0 = ablate, -1 = unset
    int      gdn_ch_req = -1;         // CGC_SHAPE_GDN_CH: same meaning at T > 1
    int      gdn_status = CGC_SHAPE_DEFAULTED;
    int      gdn_fused_ar = -1;       // realized, read back from cparams after the probe
    int      gdn_fused_ch = -1;       // -1 = not resolved yet

    // Set by the graph builder itself (cgc_shape_count_gdn), so a row can PROVE which recurrence
    // implementation actually ran instead of trusting what was requested. This exists because
    // turning fused_gdn_ar/ch off (2026-09-22) left the probe logits byte-identical, and without
    // this pair there was no way to tell "the ablation ran and is numerically identical" from
    // "the ablation never reached the builder".
    int      gdn_saw_fused = 0;       // 1 = build_delta_net took a fused path at least once
    int      gdn_saw_manual = 0;      // 1 = it took the manual AR/chunking path at least once

    // ---- measured, read from the cache itself (final phase only) --------------------------------
    uint64_t req = 0, hits = 0, misses = 0, miss_compulsory = 0, miss_capacity = 0;
    uint64_t evictions = 0;
    uint64_t zero_mapped = 0, verify_refused = 0, inv_violations = 0;
    uint64_t read_bytes = 0, pread_us = 0, fill_wait_us = 0;
    double   resident_mib = 0.0;
    int      has_final = 0;           // 1 = the final counters above were actually read
};

// INPUT side. Idempotent, thread-safe after first call (the parse happens exactly once).
const cgc_shape_cfg & cgc_shape_knobs();

// The two readers. Routing the existing call sites through here is what makes the table the
// single source of truth instead of a second opinion.
uint32_t cgc_shape_pool_max_tokens();
int      cgc_shape_n_cb();

// -1 = unset. 0 asks to DISABLE the fused recurrence (ablation only); 1 keeps the default.
int cgc_shape_gdn_ar_req();
int cgc_shape_gdn_ch_req();

// Called once from the context constructor once the routable geometry is known, so the report
// can say whether the requested WIDTH survived the pool's slot ceiling.
void cgc_shape_note_width(uint32_t width_realized, uint64_t routable_slots, uint32_t top_k,
                          size_t pool_bytes, uint32_t slots_layer, uint32_t n_expert);

// Called once the support probes have run, so the table carries the implementation actually
// chosen instead of the one requested -- the whole defect this axis had.
void cgc_shape_note_gdn(int fused_ar, int fused_ch);

// Called by models/delta-net-base.cpp on every build_delta_net dispatch with fused=0/1, so the
// final line can report which implementations the run actually executed.
void cgc_shape_count_gdn(int fused);

// OUTPUT side. One line each.
void cgc_shape_report_init(const char * profile, size_t pool_cap_slots, uint32_t slots_layer,
                           uint32_t n_expert);
void cgc_shape_report_final(struct llama_expert_cache * cache, size_t pool_cap_slots,
                            uint32_t n_layer_all);
