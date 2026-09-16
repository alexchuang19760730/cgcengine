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
NODES="${NODES:-l_out-0,l_out-1,attn_norm-2,z-2,gate-2,norm-2,linear_attn_out-2,attn_residual-2,attn_post_norm-2,ffn_moe_logits_raw-2,l_out-2}"
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
echo "  arms : p25-gputime (host leaf) vs p25-slotgpu (S1 GPU table), both with the capture on"
echo "  thermal at launch: $(python3 scripts/check/thermal_pressure.py)"
echo

# CGC_IDS_CAPTURE must stay 1: its rows carry the graph boundaries the comparator segments by.
CGC_IDS_CAPTURE=1 \
CGC_TENSOR_CAPTURE="$NODES" \
CGC_TENSOR_CAPTURE_WORDS=32 \
RUN_REPLAY_BENCH=0 \
python3 scripts/check/decode_sweep.py \
    --profile prod25 \
    --arms p25-gputime,p25-slotgpu \
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
        n_ids = sum(1 for l in open(tgt, errors="replace") if "CGC-IDS-CAP" in l and ".dst" not in l)
        n_dst = sum(1 for l in open(tgt, errors="replace") if ".dst" in l and "CGC-IDS-CAP" in l)
        print(f"  {r['tag']:<14} ids_rows={n_ids:<6} dst_rows={n_dst:<5} -> {tgt.name}")
Path(dest / "arms.json").write_text(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
for p in paths.values():
    shutil.copyfile(p, dest / "latest_" + p.name)
PY
echo
A="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('p25-gputime',''))" "$LOG/arms.json")"
B="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('p25-slotgpu',''))" "$LOG/arms.json")"
if [ -z "$A" ] || [ -z "$B" ]; then
    echo "!! could not resolve both arm logs -- see $LOG"
    exit 1
fi

echo "=== ids diff (re-derives §9.18.3 in the same run) ==="
python3 scripts/check/ids_capture_diff.py "$A" "$B" --all-graphs --max 4 2>&1 | head -16 || true
echo
echo "=== localisation (graphs 1..23 only -- see --upto) ==="
python3 Backup/analyze_capture_nodes.py "$A" "$B" --upto 24 --graphs 6
echo
echo "=== INSTRUMENT CONTROL (mandatory; read this before believing anything above) ==="
echo "  A dst DIFF is only evidence of a VALUE difference if the same arm, captured twice, agrees with"
echo "  itself. Run this script TWICE and diff the two p25-gputime logs:"
echo "    python3 Backup/analyze_capture_nodes.py <run1>-p25-gputime*.log <run2>-p25-gputime*.log"
echo "  If that reports anything other than SAME, the readout is the finding, not the engine."
echo "=== done $(date '+%H:%M:%S') ==="
