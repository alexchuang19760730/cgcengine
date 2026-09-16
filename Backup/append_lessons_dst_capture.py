#!/usr/bin/env python3
"""§9.18.6 lessons: the instrument's own control, and the inference it refuted."""
import json
from pathlib import Path

P = Path("agent_harness/engine_loop/traces/lessons.jsonl")
rows = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]
have = {r["lesson_id"] for r in rows}

new = [
    {
        "type": "lesson",
        "lesson_id": "eng-diag-0024",
        "class": "diagnosis",
        "rule": "Before any cross-arm diff from a NEW instrument, run the SAME arm twice and diff it against itself. A readout that is not reproducible under that control cannot support any cross-arm conclusion, no matter how plausible the result looks.",
        "because": "The first version of the tensor-output capture read the node's dst without a memory barrier. Same-arm control (p25-gputime captured twice, diffed against itself): ids SAME 120/120 on every graph, while `ffn_moe_down-1.dst` DIFFERED on EVERY graph. The cross-arm run then showed exactly the pattern the hypothesis predicted -- 'identical ids, different output, therefore the ids point at different weights' -- i.e. an unreproducible readout produced a CONFIDENT FALSE POSITIVE in the direction of the thing under test. With the barrier added, both same-arm controls (baseline arm and S1 arm) came out SAME on every graph, and the cross-arm answer flipped.",
        "counterexample_observed": "same-arm, no barrier: 'ffn_moe_down-1.dst DIFF' on graphs 1..30; same-arm, with ggml_metal_encoder_memory_barrier: 'ffn_moe_down-1.dst SAME' on graphs 1..35. Same binary otherwise, same prompt, same arm.",
        "applies_to": [
            "src/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp",
            "Backup/run_ids_dst_capture.sh",
            "Backup/analyze_dst_capture.py",
        ],
        "superseded_by": None,
    },
    {
        "type": "lesson",
        "lesson_id": "eng-diag-0025",
        "class": "diagnosis",
        "rule": "A second capture stream that must be paired with an existing one has to be emitted in SUBMISSION order across both streams, not appended after the first. And when a capture is placed after a kernel whose output it reads, it needs an explicit memory barrier if the fork allows kernel concurrency.",
        "because": "Two separate traps, both silent. (1) ids_capture_diff.py segments graphs at `ffn_moe_gate-1`, an ids row: had the dst rows been printed after all ids rows, every dst row would land in one final pseudo-graph, get compared against nothing, and be reported as IDENTICAL -- a false negative in the one direction this instrument exists to rule out. Solved with a shared submission sequence number and a merge in the dump. (2) The ids operand is produced MANY nodes upstream of mul_mat_id and has long settled; the dst is produced by the IMMEDIATELY PRECEDING kernel. That asymmetry is the diagnostic signature in the control: ids reproducible, dst not. It points straight at ordering, not at values.",
        "counterexample_observed": "no barrier: ids 120 SAME / dst DIFF every graph, same arm. With barrier: both SAME. The asymmetry was the tell that identified the mechanism.",
        "applies_to": [
            "src/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp",
            "scripts/check/ids_capture_diff.py",
        ],
        "superseded_by": None,
    },
    {
        "type": "lesson",
        "lesson_id": "eng-src-0014",
        "class": "source-reading",
        "rule": "kernel_cgc_ids_capture is a generic 'copy up to `stride` int32 words from buffer 1 into slot N of buffer 2' kernel -- it has nothing ids-specific in it. Pointing the instrument at a tensor OUTPUT therefore needs no new kernel, no new kargs and no new pipeline: only a second destination buffer with a wider stride and a second submission site.",
        "because": "Reading the kernel instead of assuming an ids-capture kernel must be special-cased saved .metal, ggml-metal-impl.h, ggml-metal-device.{h,cpp} and ggml-metal-context.m entirely -- the whole extension is one file (ggml-metal-ops.cpp) plus a three-line env allowlist. It also kept the existing comparator usable unchanged, because the row format is the same.",
        "counterexample_observed": "extension touched: ggml-metal-ops.cpp (new cgc_submit/cgc_dst_capture/capture_init + seq merge), ggml-metal-ops.h (comment), run_server.sh (allowlist). Unchanged: ggml-metal.metal, ggml-metal-impl.h, ggml-metal-device.h/.cpp/.m, ggml-metal-context.m.",
        "applies_to": ["src/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp"],
        "superseded_by": None,
    },
    {
        "type": "lesson",
        "lesson_id": "eng-mh-0042",
        "class": "measurement-hygiene",
        "rule": "The node-name filter for a targeted capture must match the name EXACTLY. Substring matching looks harmless and captures ten layers instead of one.",
        "because": "`ffn_moe_down-1` is a substring of `ffn_moe_down-10` .. `ffn_moe_down-19`, so a substring filter for layer 1 would also capture layers 10-19. The comparator pairs nodes BY NAME and keeps the FIRST occurrence, so the pairing would then depend on which layer's node happened to be encoded first -- a silently wrong comparison rather than an error.",
        "counterexample_observed": "run_server.sh's allowlist drops any CGC_* it does not list, so a wrong or dropped filter value is indistinguishable from 'the node never ran' -- the value is a node NAME, which makes the silent failure mode worse than for a boolean knob.",
        "applies_to": ["src/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp", "scripts/run_server.sh"],
        "superseded_by": None,
    },
    {
        "type": "lesson",
        "lesson_id": "eng-diag-0026",
        "class": "diagnosis",
        "rule": "S1's carrier is NOT layer 1's MoE weights. With the output capture working, layer 1's MoE output (`ffn_moe_down-1.dst`) is bit-identical between the baseline and the S1 arm on every graph where the two arms still process the same token, and layer 0's and layer 1's ids are identical while layers 2..39 differ. The divergence therefore enters after layer 1's MoE output and at or before layer 2's router. Treat REMAP plan 9.18.4 and 9.18.7 as REFUTED / unsupported.",
        "because": "9.18.4 was an elimination argument, and its premise was wrong. It reasoned: on graph 3, layer 1's gate/up/down ids are identical and layer 2's router differs, so under identical input and identical ids layer 1's MoE output must differ, so the carrier is the weight content the ids point at. Direct measurement of that output says it does NOT differ. Since layer 1's MoE output being bit-identical also implies layer 1's attention output is identical (a different attention output would not produce a bit-identical MoE output), the remaining candidates are layer 2's attention path (q/k/v, rope, KV read) or the norm/residual between layers 1 and 2 -- not the expert pool. 9.18.7's 'the thing to fix is residency' rested on 9.18.4, so it loses its only direct evidence.",
        "counterexample_observed": "graph 4 and 5: SAME = [ffn_moe_down-0, ffn_moe_down-1, ffn_moe_gate-0, ffn_moe_gate-1, ffn_moe_up-0, ffn_moe_up-1]; DIFF = layers 2..39 (114 nodes). ffn_moe_down-1.dst = SAME on graphs 1..30 in two independent repeats. Layer 0's router ids being SAME is also what rules out 'the two arms are simply processing different tokens'.",
        "applies_to": [
            "docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md",
            "src/llama.cpp/src/llama-graph.cpp",
            "scripts/check/ids_capture_diff.py",
        ],
        "superseded_by": None,
    },
]

added = [r for r in new if r["lesson_id"] not in have]
with P.open("a", encoding="utf-8") as fh:
    for r in added:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"appended {len(added)} lesson(s): {[r['lesson_id'] for r in added]}")
print("total now:", len(rows) + len(added))
