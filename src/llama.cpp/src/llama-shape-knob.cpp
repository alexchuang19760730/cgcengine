// cgc-shape-knob.cpp -- the two functions (INPUT applies the env, OUTPUT reports what happened).

#include "llama-shape-knob.h"
#include "llama-expert-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// fprintf, not LLAMA_LOG_INFO: the production launcher runs at verbosity 3, where every INFO line
// from this engine is suppressed. This is the same reason every other CGC diagnostic uses stderr
// (measured 2026-09-17: an INFO banner present in the binary and absent from a 900-line log).

const char * cgc_shape_status_name(int s) {
    switch (s) {
        case CGC_SHAPE_APPLIED:   return "APPLIED";
        case CGC_SHAPE_CLAMPED:   return "CLAMPED";
        case CGC_SHAPE_CONFLICT:  return "CONFLICT";
        case CGC_SHAPE_ABSENT:    return "ABSENT";
        case CGC_SHAPE_REFUSED:   return "REFUSED";
        default:                  return "DEFAULTED";
    }
}

namespace {

int env_int(const char * name, int * out) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return 0;
    }
    char * endp = nullptr;
    long v = strtol(e, &endp, 10);
    if (endp == e) {
        return 0;  // junk we cannot parse is NOT silently treated as "unset"; see below
    }
    *out = (int) v;
    return 1;
}

// Same behaviour as the parser this replaces (llama-graph.h had its own digit loop), but the
// unparsable case now has to be visible: an env whose value we cannot read must not become a
// default that later looks like a measurement.
int env_junk(const char * name) {
    const char * e = getenv(name);
    return (e != nullptr && e[0] != '\0');
}

}  // namespace

const cgc_shape_cfg & cgc_shape_knobs() {
    static cgc_shape_cfg cfg;
    static bool done = false;
    if (done) {
        return cfg;
    }
    done = true;

    // ---------------------------------------------------------------- axis A: width M ----------
    // Canonical name stays CGC_POOL_MAX_TOKENS (it is what every run recipe already sets);
    // CGC_SHAPE_M is the alias the shape harness speaks. If both are set they must agree --
    // disagreeing is printed rather than resolved silently.
    int v_cap = 0, v_m = 0;
    const int has_cap = env_int("CGC_POOL_MAX_TOKENS", &v_cap);
    const int has_m   = env_int("CGC_SHAPE_M",         &v_m);

    if (has_m || has_cap) {
        int v = has_m ? v_m : v_cap;
        cfg.m_req  = v;
        cfg.m_val  = v;
        cfg.m_status = CGC_SHAPE_APPLIED;
        if (has_m && has_cap && v_m != v_cap) {
            cfg.m_conflict = 1;
            cfg.m_other    = has_m ? v_cap : v_m;
            cfg.m_status   = CGC_SHAPE_CONFLICT;
        }
        if (cfg.m_val < 2)  { cfg.m_val = 2;  cfg.m_status = CGC_SHAPE_CLAMPED; }
        if (cfg.m_val > 64) { cfg.m_val = 64; cfg.m_status = CGC_SHAPE_CLAMPED; }
    } else if (env_junk("CGC_POOL_MAX_TOKENS") || env_junk("CGC_SHAPE_M")) {
        // Junk value: refuse loudly rather than defaulting. The alternative failure mode is a
        // run that looks like "M=default was measured" when the operator asked for something.
        cfg.m_status = CGC_SHAPE_REFUSED;
        cfg.m_val    = 8;
        fprintf(stderr, "CGC-SHAPE: REFUSED: width env present but not an integer -> keeping 8\n");
    }

    // ---------------------------------------------------------------- axis C: cb count ---------
    int v_ncb = 0;
    if (env_int("CGC_N_CB", &v_ncb)) {
        cfg.ncb_req    = v_ncb;
        cfg.ncb_val    = v_ncb < 1 ? 1 : v_ncb;
        cfg.ncb_status = (cfg.ncb_val == v_ncb) ? CGC_SHAPE_APPLIED : CGC_SHAPE_CLAMPED;
    }

    // ---------------------------------------------------------------- axis D: GDN impl --------
    // See the header comment: the request selects an IMPLEMENTATION, it does not create one.
    // 1 = keep the fused operator (the default, and the only shape the bit-identity anchor
    // covers); 0 = fall back to the manual AR/chunking graphs, which is an ablation whose t/s
    // is NOT comparable to an anchored number. Anything else is refused rather than guessed.
    for (const std::pair<const char *, int *> name_slot : {
            std::pair<const char *, int *>("CGC_SHAPE_GDN_AR", &cfg.gdn_ar_req),
            std::pair<const char *, int *>("CGC_SHAPE_GDN_CH", &cfg.gdn_ch_req)}) {
        int v = 0;
        if (!env_int(name_slot.first, &v)) {
            continue;
        }
        if (v != 0 && v != 1) {
            cfg.gdn_status   = CGC_SHAPE_REFUSED;
            cfg.gdn_req      = v;
            fprintf(stderr, "CGC-SHAPE: REFUSED: %s=%d is not 0 or 1 (0 = disable the fused "
                            "recurrence for ABLATION, 1 = keep it); keeping the default.\n",
                    name_slot.first, v);
            continue;
        }
        *name_slot.second = v;
        cfg.gdn_req  = v;
        cfg.gdn_status = CGC_SHAPE_APPLIED;
    }

    return cfg;
}

uint32_t cgc_shape_pool_max_tokens() {
    return (uint32_t) cgc_shape_knobs().m_val;
}

int cgc_shape_n_cb() {
    return cgc_shape_knobs().ncb_val;
}

int cgc_shape_gdn_ar_req() { return cgc_shape_knobs().gdn_ar_req; }
int cgc_shape_gdn_ch_req() { return cgc_shape_knobs().gdn_ch_req; }

void cgc_shape_note_gdn(int fused_ar, int fused_ch) {
    cgc_shape_cfg & cfg = const_cast<cgc_shape_cfg &>(cgc_shape_knobs());
    cfg.gdn_fused_ar = fused_ar;
    cfg.gdn_fused_ch = fused_ch;

    // The sentence every reading of this axis needs: is the run inside the anchored engine?
    const int ablated = (cfg.gdn_ar_req == 0 || cfg.gdn_ch_req == 0) ? 1 : 0;
    // note= MUST stay one token with no spaces and no '='. The OUTPUT contract of these lines is
    // "key=value pairs separated by spaces; values never contain spaces" (see
    // scripts/check/shape_knob_search.py:parse_shape_line). An earlier revision of this line put a
    // sentence here -- it contained a space AND the substring "K=1", so the parser read a phantom
    // key named `K` and dumped the rest into `_unparsed`, i.e. the row silently lost its meaning.
    // Machine tokens only; the prose lives in docs/GDN_AXIS_RESOLVED_2026-09-22.md.
    fprintf(stderr,
            "CGC-SHAPE v=1 phase=gdn fused_ar=%d fused_ch=%d "
            "req_ar=%d req_ch=%d ablated=%d bitident=%s note=%s\n",
            fused_ar, fused_ch, cfg.gdn_ar_req, cfg.gdn_ch_req, ablated,
            ablated ? "NO" : "yes",
            ablated ? "fusion_off_ablation_not_comparable"
                    : "fused_metal_default_k_snapshot");
}

void cgc_shape_count_gdn(int fused) {
    cgc_shape_cfg & cfg = const_cast<cgc_shape_cfg &>(cgc_shape_knobs());
    if (fused) {
        cfg.gdn_saw_fused = 1;
    } else {
        cfg.gdn_saw_manual = 1;
    }
}

void cgc_shape_note_width(uint32_t width_realized, uint64_t routable_slots, uint32_t top_k,
                          size_t pool_cap_slots, uint32_t slots_layer, uint32_t n_expert) {
    cgc_shape_cfg & cfg = const_cast<cgc_shape_cfg &>(cgc_shape_knobs());
    cfg.width_realized = width_realized;
    cfg.routable_slots = routable_slots;
    cfg.top_k          = top_k;
    cfg.pool_cap_slots = pool_cap_slots;
    cfg.slots_layer    = slots_layer;
    cfg.n_expert       = n_expert;

    cfg.union_at_width = (uint32_t) top_k * width_realized;
    if (routable_slots > 0) {
        cfg.union_fits_pool  = (cfg.union_at_width <= routable_slots) ? 1 : 0;
        cfg.width_limit_hit  = (width_realized < (uint32_t) cfg.m_val) ? 1 : 0;
    }
    if (cfg.width_limit_hit) {
        cfg.m_status = CGC_SHAPE_CLAMPED;
    }
}

void cgc_shape_report_init(const char * profile, size_t pool_cap_slots, uint32_t slots_layer,
                           uint32_t n_expert) {
    const cgc_shape_cfg & c = cgc_shape_knobs();
    fprintf(stderr,
            "CGC-SHAPE v=1 phase=init profile=%s "
            "M_req=%d M=%d M_stat=%s M_other=%d "
            "width=%u routable=%llu topk=%u union=%u fits=%d limit=%d "
            "pool_cap_slots=%zu slots_layer=%u n_expert=%u "
            "ncb_req=%d ncb=%d ncb_stat=%s "
            "gdn_req=%d gdn_stat=%s\n",
            profile ? profile : "-",
            c.m_req, c.m_val, cgc_shape_status_name(c.m_status), c.m_other,
            c.width_realized, (unsigned long long) c.routable_slots, c.top_k,
            c.union_at_width, c.union_fits_pool, c.width_limit_hit,
            pool_cap_slots, slots_layer, n_expert,
            c.ncb_req, c.ncb_val, cgc_shape_status_name(c.ncb_status),
            c.gdn_req, cgc_shape_status_name(c.gdn_status));
}

void cgc_shape_report_final(struct llama_expert_cache * cache, size_t pool_cap_slots,
                            uint32_t n_layer_all) {
    cgc_shape_cfg & cfg = const_cast<cgc_shape_cfg &>(cgc_shape_knobs());
    // The destructor has no access to the model's allocation size, so the value recorded at init
    // stands; only override it when the caller actually knows better.
    if (pool_cap_slots == 0) {
        pool_cap_slots = cfg.pool_cap_slots;
    }
    if (cache != nullptr) {
        cfg.req             = (uint64_t) cache->n_requests;
        cfg.hits            = (uint64_t) cache->n_hits;
        cfg.misses          = (uint64_t) cache->n_misses;
        cfg.miss_compulsory = (uint64_t) cache->n_miss_compulsory;
        cfg.miss_capacity   = (uint64_t) cache->n_miss_capacity;
        cfg.evictions       = (uint64_t) cache->n_evictions;
        cfg.zero_mapped     = (uint64_t) cache->n_zero_mapped_selected;
        cfg.verify_refused  = (uint64_t) cache->n_verify_strict_refused;
        cfg.inv_violations  = (uint64_t) cache->n_batch_inv_violations;
        cfg.read_bytes      = (uint64_t) cache->n_read_bytes.load(std::memory_order_relaxed);
        cfg.pread_us        = (uint64_t) cache->pread_usec.load(std::memory_order_relaxed);
        cfg.fill_wait_us    = (uint64_t) cache->fill_wait_us.load(std::memory_order_relaxed);
        cfg.slots_layer     = cache->n_slots;
        cfg.n_expert        = cache->n_expert;
        cfg.has_final       = 1;
    }
    const double hit_pct = cfg.req ? 100.0 * (double) cfg.hits / (double) cfg.req : 0.0;
    fprintf(stderr,
            "CGC-SHAPE v=1 phase=final "
            "M=%d M_stat=%s width=%u union=%u fits=%d "
            "pool_cap_slots=%zu slots_layer=%u n_layer=%u "
            "req=%llu hits=%llu misses=%llu hit_pct=%.2f compulsory=%llu capacity=%llu evict=%llu "
            "zero_mapped=%llu verify_refused=%llu inv_viol=%llu "
            "read_mib=%.1f pread_us=%llu fill_wait_us=%llu final_counters=%d "
            "gdn_ablated=%d gdn_saw_fused=%d gdn_saw_manual=%d\n",
            cfg.m_val, cgc_shape_status_name(cfg.m_status), cfg.width_realized,
            cfg.union_at_width, cfg.union_fits_pool,
            pool_cap_slots, cfg.slots_layer, n_layer_all,
            (unsigned long long) cfg.req, (unsigned long long) cfg.hits,
            (unsigned long long) cfg.misses, hit_pct,
            (unsigned long long) cfg.miss_compulsory,
            (unsigned long long) cfg.miss_capacity, (unsigned long long) cfg.evictions,
            (unsigned long long) cfg.zero_mapped, (unsigned long long) cfg.verify_refused,
            (unsigned long long) cfg.inv_violations,
            (double) cfg.read_bytes / 1048576.0,
            (unsigned long long) cfg.pread_us, (unsigned long long) cfg.fill_wait_us,
            cfg.has_final,
            // Carried on the SAME row every existing reader already parses, not only on the
            // phase=gdn row: a cell whose fusion was switched off is not measuring the anchored
            // engine, and that has to be visible to whatever row the verdict is taken from.
            (cfg.gdn_ar_req == 0 || cfg.gdn_ch_req == 0) ? 1 : 0,
            // Which implementations ACTUALLY ran, counted by the builder itself. Without these two
            // a "no difference" result is uninterpretable: invisible ablation vs identical numbers.
            cfg.gdn_saw_fused, cfg.gdn_saw_manual);
}
