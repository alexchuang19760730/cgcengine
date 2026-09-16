#!/usr/bin/env python3
"""Append the 2026-09-16 (round 2) lessons from the extended output capture."""
import json
from pathlib import Path

P = Path("agent_harness/engine_loop/traces/lessons.jsonl")

NEW = [
    {
        "type": "lesson",
        "lesson_id": "eng-diag-0027",
        "class": "diagnosis",
        "rule": "The first divergence between the baseline arm and the S1 arm is INSIDE layer 2's gated delta-net core. Layer 0's and layer 1's outputs (`l_out-0`, `l_out-1`) are bit-identical on every properly delimited pass; layer 2's attention INPUT (`attn_norm-2`) and BOTH of its input projections (`z-2`, `gate-2`) are bit-identical; layer 2's attention OUTPUT projection (`linear_attn_out-2`) differs from graph 4 on. So no dense matmul in layer 2 is the divergence point, and neither is anything before it. Narrow eng-diag-0026's window from 'at or before layer 2's router' to 'between z-2/gate-2 and the ssm_out projection' -- i.e. conv1d / delta-rule scan / state.",
        "because": "Measured 2026-09-16 with CGC_TENSOR_CAPTURE extended to five dispatchers (mul_mat, mul_mat_id, flash_attn_ext, bin, norm), two independent cross-arm pairs, after both same-arm controls came back clean (2 arms x 2 processes x 23 graphs x 11 nodes = zero differences). Restricted to graphs 1..23 because the IDS destination caps at 4096 slots and the tail chunks therefore merge several forward passes. What this refutes: eng-diag-0026 left two candidates standing, 'layer 2's attention (q/k/v, rope, KV)' and 'the norm/residual between layers 1 and 2'. The second is now excluded outright (attn_norm-2 is identical, and so is l_out-1), and the first is narrowed to the recurrent core: the layer's dense projections are all identical. Note the deliberate limit: this localises the FIRST divergence, it does not exonerate the pool anywhere else -- layer 2's MoE does differ, but only downstream of this, so it carries no information.",
        "applies_to": ["s1-divergence", "layer2-gated-delta-net", "output-capture"],
    },
    {
        "type": "lesson",
        "lesson_id": "eng-src-0015",
        "class": "source-reading",
        "rule": "A capture placed at the tail of a Metal dispatcher must name `ctx->node(idx + n_fuse - 1)`, not `ctx->node(idx)`. Every dispatcher that can fuse ends by REPOINTING bid_dst at the last node it covered (`ggml-metal-ops.cpp`: `if (n_fuse > 1) bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));`), and the dispatchers that cannot fuse end in a constant `return 1;` -- so the same expression is correct in both cases. Getting this wrong silently mislabels a fused tensor as its first node.",
        "because": "This is what makes it safe to instrument a dispatcher with one line at its end instead of chasing each of its internal paths: for mul_mat alone there are eight `set_buffer(..., bid_dst, ...)` sites and not all of them are on any given call's path. The rule also has a visible consequence worth knowing before trusting a name: when the last node of a fused group is ANONYMOUS, the row is skipped entirely (the matcher refuses an empty name), so a node's presence in the stream depends on the fusion state -- see the coupled lesson on ABSENT not meaning SAME.",
        "applies_to": ["output-capture", "ggml-metal-dispatchers", "fusion"],
    },
    {
        "type": "lesson",
        "lesson_id": "eng-mh-0043",
        "class": "measurement-hygiene",
        "rule": "Row emission order is NOT computation order, and it is not even stable: the same arm, captured twice, emits the same nodes in a different order (measured: `linear_attn_out-2` before `attn_residual-2` in one graph and after it in another, within one process). Therefore any inference of the form 'this row came first, so this node computed first' is reading noise. Localise by the SOURCE chain (the builder's order), which is knowledge, not by the stream, which is not.",
        "because": "The encode loop itself is strictly ascending and deterministic (`for idx < n_nodes: ggml_metal_op_encode(ctx, idx)`), so the instability has to come from the node list the scheduler hands the backend, not from the encode loop -- which also means it is not something an instrument can fix. It was found while trying to explain an apparent reordering, and it retroactively explains a puzzling enumeration result in the same round. The values, by contrast, ARE reproducible (both same-arm controls were clean over 23 graphs x 11 nodes), so the readout is sound; only its ORDER is not. Consequence for tooling: segmenting a stream at a name marker (as ids_capture_diff.py does at `ffn_moe_gate-1`) is robust to this; pairing rows by position would not be.",
        "applies_to": ["output-capture", "ids-capture", "analysis-tooling"],
    },
    {
        "type": "lesson",
        "lesson_id": "eng-mh-0044",
        "class": "measurement-hygiene",
        "rule": "Two ways this capture stream can look like agreement when it is not, and both are now printed rather than inferred. (1) ABSENT is not SAME: a node is captured only when it happens to be the last node of its fused group, and the fusion state depends on node order, so `norm-2` appears 41/41 in the baseline arm and 0/41 in the S1 arm -- a name missing from one log must be reported as not-comparable, never as identical. (2) The last graph chunks are not delimited: the IDS destination is capped at 4096 slots and a pass costs ~114, so the tail merges several forward passes and pairing by name there compares rows from different passes -- bound the analysis (--upto 24 for these runs) before believing any tail divergence.",
        "because": "Both were caught by controls that existed for other reasons: (1) by a per-name row count, after a same-arm control flagged a node at graph 27; (2) by noticing that a node appears 41 times while the segmentation reports only 36 chunks, which means chunk 36 holds ~5 passes. The failure mode they share is the one this project treats as worst: a silent false negative, or its twin, a divergence that is an artifact of the analysis rather than the engine. The instrument was also made to fail loudly here: DST slot exhaustion now emits one WARN instead of truncating quietly.",
        "applies_to": ["output-capture", "ids-capture", "analysis-tooling"],
    },
]

existing = {json.loads(l)["lesson_id"] for l in P.read_text(encoding="utf-8").splitlines() if l.strip()}
with P.open("a", encoding="utf-8") as f:
    for rec in NEW:
        if rec["lesson_id"] in existing:
            print("skip (exists):", rec["lesson_id"])
            continue
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("added:", rec["lesson_id"], rec["class"])
