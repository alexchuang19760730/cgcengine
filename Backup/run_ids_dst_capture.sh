#!/usr/bin/env bash
# [CGC 2026-09-16] §9.18.6: capture a LIST of nodes' OUTPUT tensors on both arms, then localise the
# first divergence by submission order.
#
# HISTORY, because the shape of this file is a record of what went wrong
#   1st version captured ONE node (`ffn_moe_down-1`) and reported "identical ids, DIFFERENT output",
#   which read exactly like a confirmation of §9.18.4. It was an instrument artefact: the copy had no
#   memory barrier, so it read the buffer's previous occupant. The SAME-ARM control (one arm captured
#   twice, diffed against itself) is what caught it -- without that control the false positive would
#   have been published. The barrier is now in the instrument, and the control below is MANDATORY.
#   The result after the fix: layer 1's MoE output is bit-identical, which REFUTES §9.18.4.
#
# WHAT IT DOES NOW
#   The filter takes a comma-separated list (CGC_TENSOR_CAPTURE), and four more dispatchers are
#   instrumented (mul_mat, flash_attn_ext, bin, norm) so one run can walk a layer's chain: the router
#   INPUT, the router LOGITS, the attention output and the MoE output, in submission order.
#
#   NOTE ON THIS MODEL (it changes what "attention" means): Qwen3.6-35B-A3B is a HYBRID stack with
#   `full_attention_interval = 4`, so layers 0,1,2 are gated delta-net (linear attention) and layer 3
#   is the first full-attention layer. The layer the divergence is localised to, layer 2, has NO
#   flash attention: its attention output projection is an ordinary mul_mat (`linear_attn_out-2`).
#   Reaching "layer 2's attention" is therefore the mul_mat hook, not the flash_attn one.
#
# Usage:  bash Backup/run_ids_dst_capture.sh
#         NODES=a,b,c N_PREDICT=12 bash Backup/run_ids_dst_capture.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

# The names are the AUTHORITATIVE ones: they were enumerated once with CGC_TENSOR_CAPTURE='*' (the
# documented use of that mode), because guessing them from the builder costs a whole run and two of my
# guesses were wrong (`ffn_out-*` and `post_moe-*` do not exist; the real names are `ffn_moe_out-*`
# and `l_out-*`, which is what `build_cvec` renaming the residual produces).
#
# Layer 2 is the layer the divergence is localised to, layer 1 is the last one known identical, and
# `l_out-1` is the boundary between them. `ffn_moe_logits_raw-2` is the last numerical value before
# the top-k that picks the experts, and `attn_post_norm-2` is the value the router consumes -- those
# two together split layer 2 at its most informative point.
# `norm-2` is in the default list even though it is NOT comparable across these two arms: it is the
# gated norm inside layer 2's delta-net, and whether it is capturable depends on whether the norm
# dispatcher fused the following MUL -- which depends on node ORDER, and the S1 arm puts it a
# different way round, so the name is absent from that arm entirely. Kept in the list because
# absence is visible (the analyzer prints ABSENT, never "identical") and because the fix for it is
# to instrument the delta-net's own dispatchers, not to drop the node.
# [CGC 2026-09-16 §9.18.6 round 3] `conv_output_raw-2` added, and it is the only NEW comparable point.
# Round 3 instrumented SSM_CONV / GATED_DELTA_NET / SSM_SCAN, and the way the names were obtained is
# worth keeping: diffing two `CGC_TENSOR_CAPTURE='*'` enumerations (the 20:11 log, taken before the
# hooks, vs a fresh one after) yields exactly 60 new names -- 30 x `conv_output_raw-N` and 30 x
# `node_NNN`. SSM_SCAN contributed nothing (this model has no Mamba path). The layers carrying a
# conv are 0,1,2,4,5,6,8,..., i.e. every layer except 3,7,11,... -- `full_attention_interval = 4`,
# so the hook lands exactly where it should. GATED_DELTA_NET's own output is an ANONYMOUS `node_NNN`
# and anonymous names come from ggml's counter, which differs between arms -> not comparable. Hence
# the one usable addition is the short conv's output, which is NAMED and sits BEFORE the delta-rule
# scan: SAME-conv together with DIFF-linear_attn_out puts the divergence inside the scan itself.
# [CGC 2026-09-16 §9.18.6 r3b] `conv_input-2` added -- and this one is free, because the model's own
# builder already names it (`delta-net-base.cpp:473`: `cb(conv_input, "conv_input", il)`), and it is
# the output of a `ggml_concat`, which the `bin` dispatcher (hooked in round 2) already captures.
# Where it sits: `build_conv_state()` builds `conv_input = concat(conv_states, qkv_mixed)` and then
# `ggml_ssm_conv(conv_input, conv_kernel)` produces `conv_output_raw`. Round 3's cross-arm diff put the
# FIRST divergence on `conv_output_raw-2` at graph 4 with `gate-2` SAME. So this one point splits that
# in two: conv_input SAME + conv_output DIFF => the divergence is INSIDE the conv kernel; conv_input
# DIFF => it is upstream, in `conv_states` (the RECURRENT per-layer conv state read back out of the
# KV cache) or in the qkv projection. Note why graph 4 is the first: a conv of kernel_size 4 needs
# three prior steps to fill its state, so the state term is still degenerate before that.
# [CGC 2026-09-16 §9.18.6 r3d] `conv_state_update-2` added: the `ggml_cpy` DESTINATION of
# `delta-net-base.cpp:496`, i.e. the write that persists the per-layer conv recurrence state back
# into the KV cache. Why this one: r3c put the first cross-arm divergence on `conv_input-2`
# (= concat(conv_states, qkv_mixed)) with the differing words inside the `conv_states` half -- the
# state READ BACK, which is the next step's conv operand. `conv_states` and `conv_state_last` are
# views with no kernel of their own, so the write is the only readable point in that cycle. Requires
# the `ggml_metal_op_cpy` hook (CPY/DUP/CONT share one dispatcher, so the NAME FILTER is what keeps
# this list short -- do not widen it to bare `CGC_TENSOR_CAPTURE` names that CPY also emits).
# [CGC 2026-09-16 §9.18.6 r3e] `linear_attn_qkv_mixed-2` added -- the second operand of the concat, and
# the point that decides between the two halves of `conv_input`. r3d showed the divergence words on
# `conv_input-2` sit at a FIXED stride: graph 1 and 2 both had diff_at = [4, 9, 14, 19, 24, 29], so
# `conv_input`'s dim0 is 5 and the differing slot is i0 = 4 in every channel -- the LAST dim0 row.
# Whether that is the tail of `conv_states` or the whole of `qkv_mixed` depends on the conv kernel
# size, which this file does not record. `conv_states` is a view (nothing to capture), so
# `linear_attn_qkv_mixed` -- named at `qwen35moe.cpp:265` and produced by a `mul_mat` -- is the one
# readable point that separates the two cases:
#   SAME  => the differing row is `conv_states`, i.e. the recurrence state read back from the cache
#   DIFF  => it is the q/k/v projection itself, which is DENSE and has no reason to differ across arms
# [CGC 2026-09-16 §9.18.6 r5] The window anchor was the defect. Rounds r1..r4 all read the FIRST 32
# elements of each tensor, and a ggml tensor is ne0-fastest. Measured geometry (T=8):
#   conv_input-2      ne=90112  (ne0=K-1+T=11, ne1=8192)  -> head window spans EVERY token
#   conv_output_raw-2 ne=65536  (ne0=8192, ne1=T)         -> head window IS token 0
#   attn_norm/z/gate/linear_attn_out  ne=16384 (ne0=2048) -> head window IS token 0
# So "conv_input DIFF while the dense projections are SAME" was never a contradiction: the dense nodes
# were reporting on token 0, the OLDEST token, while conv_input reported on all of them. Every SAME in
# rounds r1..r4 is a SAME-AT-TOKEN-0 -- including `l_out-0`, i.e. the reason layer 0 looked exonerated.
# The tail window (TAIL=1, the default here) reads the LAST 32 elements instead, which for these
# shapes is the NEWEST token -- the one the recurrence chain carries into the next step.
#
# r5 therefore restores layer 0 to the watch list, and it is not a retargeting for its own sake: with
# a head window, walking backwards from layer 2 to layer 0 would have been a walk through token-0
# verdicts, which cannot localise anything. `norm-2`, `linear_attn_qkv_mixed-2` and
# `conv_state_update-2` are DROPPED: all three were confirmed never to appear, because each names a
# VIEW (`cb()` naming a reshape/view gives a node with no kernel of its own, and `ggml_cpy` builds a
# NEW node whose name is not the destination's). Capturable means "the output of an op that drives a
# kernel"; having a `cb()` name is necessary, not sufficient.
# [CGC 2026-09-17 00:01 r7] LAYER 1 added, and this is the first localisation since round 1 that is
# not a window reading. r6 replaced the 32-word window with a whole-tensor digest
# (CGC_TENSOR_CAPTURE_HASH=1) and its same-arm control was clean, so its verdicts are about TENSORS:
#   layer 0's twelve nodes ....... SAME through graph 30   (whole tensor, 36 graphs)
#   l_out-1 (layer 1's output) .. DIFF from graph 1        (whole tensor)
#   layer 2's eleven nodes ...... DIFF from graph 1        (whole tensor)
# Layer 0 precedes layer 1 precedes layer 2 in a pass, so "layer 0 identical, layer 1 not" puts the
# introduction point INSIDE LAYER 1 -- a layer this chain never instrumented, because rounds r1..r4
# had judged it identical from a token-0 window. The names below are copied from layer 0's PROVEN set
# (the same twelve names appeared there in r5b), not guessed from the builder.
NODES="${NODES:-l_out-1,ffn_moe_out-1,ffn_moe_weights_norm-1,attn_norm-0,z-0,gate-0,conv_input-0,conv_output_raw-0,linear_attn_out-0,attn_residual-0,attn_post_norm-0,ffn_moe_logits_raw-0,ffn_moe_weights_norm-0,ffn_moe_out-0,l_out-0,attn_norm-1,z-1,gate-1,conv_input-1,conv_output_raw-1,linear_attn_out-1,attn_residual-1,attn_post_norm-1,ffn_moe_logits_raw-1,attn_norm-2,z-2,gate-2,conv_input-2,conv_output_raw-2,linear_attn_out-2,attn_residual-2,attn_post_norm-2,ffn_moe_logits_raw-2,ffn_moe_weights_norm-2,ffn_moe_out-2,l_out-2}"
# 12, not 24: the IDS destination is 4096 slots and a full forward pass costs ~114 of them, so 36
# graphs would silently truncate the tail of the ids stream -- and the ids rows are what carry the
# graph boundaries. Fewer, complete graphs beat more, truncated ones.
N_PREDICT="${N_PREDICT:-12}"
OUT="$ROOT/Backup/phase_decomp"
LOG="$ROOT/Backup/cgc_logs/ids_dst_capture"
mkdir -p "$OUT" "$LOG"
STAMP="$(date +%Y%m%d_%H%M%S)"
JSON="$OUT/ids_dst_capture_${STAMP}.json"

echo "=== §9.18.6 output capture: n_predict=$N_PREDICT  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  nodes: $NODES"
echo "  window: WORDS=${WORDS:-32} TAIL=${TAIL:-1}  (TAIL=1 reads the LAST n elements = the newest"
echo "          token; TAIL=0 reads the FIRST n = token 0. Name it in any result you quote.)"
echo "  arms : ${ARMS:-p25-gputime,p25-slotgpu}   (default = host leaf vs S1 GPU table)"
echo "  thermal at launch: $(python3 scripts/check/thermal_pressure.py)"
echo

# CGC_IDS_CAPTURE must stay 1: its rows carry the graph boundaries the comparator segments by.
#
# [CGC 2026-09-17 §EN-17] MMID=1|2|3 turns on the pre-existing `CGC_MMID_MV_DBG` host-side print in
# ggml_metal_op_mul_mat_id: per MoE node it prints the weights pointer (`src0_data`/`src0_offs`), the
# ids operands' offset, and an FNV-1a fingerprint of the FIRST <=4096 BYTES OF EACH OF THE FIRST 4
# ROWS the ids select. Its own comment states the decision rule we need: "If the hashes match but the
# ids differ -> the remap is wrong. If the ids match but the hashes differ -> the pool contents are
# wrong." MMID=1 is 150 nodes = one full pass (41 layers x 3 tensors = 123) -- i.e. exactly the first
# pool-path pass. It must NOT be set to 0: the parser reads only the first character and anything
# other than '1'/'2' means 4096 nodes.
if [ -n "${MMID:-}" ]; then
    if [ "$MMID" = "0" ]; then
        echo "MMID=0 would be read as 4096 nodes by CGC_MMID_MV_DBG -- refusing" >&2
        exit 2
    fi
    export CGC_MMID_MV_DBG="$MMID"
    echo "  CGC_MMID_MV_DBG=$MMID  (host-side src0 pointer + row fingerprints)"
fi

# [CGC 2026-09-17 §9.18.6 r12] POOL=1 turns on the DEVICE-side pool-row digest (path=POOL): for every
# named node, the rows the ids select are digested on the GPU at the moment the consumer runs. This is
# the measurement the HOST-side probe (MMID=1 above) cannot make on the S1 arm -- MMID reads
# `op->src[2]->data` at encode time, and on that arm the ids are a GPU-computed node, so it reports
# `id_oob_vs_ne02=16/16` with float bit patterns instead of ids. Setting BOTH is the point of the
# round: MMID is the anchor-arm reading, POOL is the one that works on both.
#
# Why the default list is layer 1: with CGC_S1_MIN_IL=1 (the default) layer 1 is the FIRST layer whose
# MoE gather is served by the GPU table, and the causal test (MIN_IL=1 -> l_out-1 vs MIN_IL=2 ->
# ffn_moe_out-2) puts the first divergence exactly there. The three names are the layer's three routed
# matmuls; `ffn_moe_down-1` is the one round 1 captured by hand, `ffn_moe_gate-1` is the row the
# comparators segment graphs by (so it is known to be one). `ffn_moe_up-1` is INFERRED -- if it does
# not exist the comparator prints ABSENT for it, which is visible, and `POOLNODES='*'` enumerates.
#
# ROWS defaults to 8 = n_expert_used for this model, i.e. a whole decode step's ids for one tensor.
# BYTES defaults to the same 4096 the host probe uses, so the two readings stay commensurable.
POOL_LIST="0"
if [ "${POOL:-0}" = "1" ]; then
    POOLNODES="${POOLNODES:-ffn_moe_gate-1,ffn_moe_up-1,ffn_moe_down-1}"
    POOL_LIST="$POOLNODES"
    echo "  CGC_POOL_CAPTURE=$POOL_LIST  rows=${POOLROWS:-8} bytes=${POOLBYTES:-4096}  (device-side row digest)"
fi

CGC_IDS_CAPTURE=1 \
CGC_TENSOR_CAPTURE="$NODES" \
CGC_TENSOR_CAPTURE_WORDS="${WORDS:-32}" \
CGC_TENSOR_CAPTURE_TAIL="${TAIL:-1}" \
CGC_TENSOR_CAPTURE_HASH="${HASH:-0}" \
CGC_POOL_CAPTURE="$POOL_LIST" \
CGC_POOL_CAPTURE_ROWS="${POOLROWS:-8}" \
CGC_POOL_CAPTURE_BYTES="${POOLBYTES:-4096}" \
RUN_REPLAY_BENCH=0 \
python3 scripts/check/decode_sweep.py \
    --profile prod25 \
    --arms "${ARMS:-p25-gputime,p25-slotgpu}" \
    --rounds 1 --warmup 0 --n-predict "$N_PREDICT" \
    --json "$JSON" --force 2>&1 | tail -12
echo

python3 - "$JSON" "$LOG" <<'PY'
import json, shutil, sys
from pathlib import Path
rows = json.load(open(sys.argv[1]))
dest = Path(sys.argv[2])
paths = {}
for r in rows:
    log = r.get("log")
    if log and Path(log).exists():
        tgt = dest / f"{r['tag'].replace(':', '_').replace(';', '_')}_{Path(log).name}"
        shutil.copyfile(log, tgt)
        paths[r["tag"]] = tgt
        n_ids = sum(1 for l in open(tgt, errors="replace") if "CGC-IDS-CAP" in l and ".dst" not in l and ".pool" not in l)
        n_dst = sum(1 for l in open(tgt, errors="replace") if ".dst" in l and "CGC-IDS-CAP" in l)
        # [CGC 2026-09-17 r12] counted separately, and excluded from n_ids above: the ids counter
        # used to be "anything that is not .dst", so POOL rows would have silently inflated it -- the
        # same class of bug as the 256-byte filter, one layer up in the reporting.
        n_pool = sum(1 for l in open(tgt, errors="replace") if ".pool" in l and "CGC-IDS-CAP" in l)
        print(f"  {r['tag']:<14} ids_rows={n_ids:<6} dst_rows={n_dst:<5} pool_rows={n_pool:<5} -> {tgt.name}")
Path(dest / "arms.json").write_text(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
for p in paths.values():
    # [CGC 2026-09-17 r12] WAS `dest / "latest_" + p.name`, which Python parses as
    # `(dest / "latest_") + p.name` -- Path + str -> TypeError. Every run since this line was added
    # died HERE, at the end of this heredoc, so the `latest_*` convenience copies were never made
    # while `arms.json` (written one line above) was: the failure is invisible unless you read the
    # script's own stderr, and the step it skips is exactly the one the NEXT reader reaches for.
    # Found while adding the POOL counter; not caused by it (it is a context line in `git diff`).
    shutil.copyfile(p, dest / ("latest_" + p.name))
PY
echo
# [CGC 2026-09-17] Resolve A/B from the run's OWN arms.json rather than from hardcoded tag names: the
# arm pair is overridable now (ARMS=...), and the `p25-keepleaf` discrimination needs a different pair
# than the default. dict order == arm order, because the sweep writes the rows in arm order.
A="$(python3 -c "import json,sys;d=list(json.load(open(sys.argv[1])).values());print(d[0] if len(d)>0 else '')" "$LOG/arms.json")"
B="$(python3 -c "import json,sys;d=list(json.load(open(sys.argv[1])).values());print(d[1] if len(d)>1 else '')" "$LOG/arms.json")"
if [ -z "$A" ] || [ -z "$B" ]; then
    echo "!! could not resolve both arm logs -- see $LOG"
    exit 1
fi

echo "=== ids diff (re-derives §9.18.3 in the same run) ==="
python3 scripts/check/ids_capture_diff.py "$A" "$B" --all-graphs --max 4 2>&1 | head -16 || true
echo
echo "=== WINDOW CHECK (read this FIRST) ==="
echo "  Fixes the round-1..4 blind spot: the 32-word window is anchored at element 0, and a ggml"
echo "  tensor is ne0-fastest, so on a (2048, T) output the head window IS token 0 -- the OLDEST"
echo "  token -- while on conv_input (ne0 = K-1+T < 32) it spans EVERY token. Verdicts from the two"
echo "  are not comparable, which is how 'dense projections identical' was decided at token 0 alone."
python3 Backup/analyze_capture_nodes.py "$A" "$B" --graphs 0 --windows
echo
echo "=== DECODE graphs ONLY (T=1, derived from ne -- NOT from --upto) ==="
python3 Backup/analyze_capture_nodes.py "$A" "$B" --decode-only --graphs 2
echo
echo "=== ALL graphs (no --upto: the head of the sequence is prefill, so a localisation over it is a"
echo "=== localisation of prefill; kept for contrast, not as the answer) ==="
python3 Backup/analyze_capture_nodes.py "$A" "$B" --graphs 0 | /usr/bin/grep -E "^ *[0-9]+ |^(A|B) graphs|per graph|earliest|=>"
echo
echo "=== POOL rows: the bytes BEHIND the ids (device-side, both arms) ==="
echo "  Read this BEFORE the dst table: it is the only reading that can separate 'the ids point at"
echo "  different experts' from 'the same expert index holds different bytes'. If a POOL row reports"
echo "  SAME for every row while the same node's .dst row reports DIFF, then the gather's two inputs are"
echo "  identical and the divergence is NOT in the pool contents -- look after the gather instead."
if [ "$POOL_LIST" != "0" ]; then
    python3 Backup/compare_pool_row.py "$A" "$B" --max-graphs 4 2>&1 | head -40
else
    echo "  (skipped: POOL=0 -- rerun with POOL=1 POOLNODES='<exact node names>' to produce these rows)"
fi
echo
echo "=== INSTRUMENT CONTROL (mandatory; read this before believing anything above) ==="
echo "  A dst DIFF is only evidence of a VALUE difference if the same arm, captured twice, agrees with"
echo "  itself. Run this script TWICE and diff the two p25-gputime logs:"
echo "    python3 Backup/analyze_capture_nodes.py <run1>-p25-gputime*.log <run2>-p25-gputime*.log"
echo "  If that reports anything other than SAME, the readout is the finding, not the engine."
echo "=== done $(date '+%H:%M:%S') ==="
