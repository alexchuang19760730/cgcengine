#!/usr/bin/env bash
# [CGC 2026-09-16] §9.18.6: capture `ffn_moe_down-1`'s OUTPUT on both arms and diff it.
#
# THE QUESTION THIS ANSWERS (and why it is the decisive one)
#   §9.18.3 established, with the kernel-side ids capture, that the ids `mul_mat_id` consumes are
#   BIT-IDENTICAL between the baseline arm and the S1 arm (39 layers x 117 nodes x 3 graphs), and
#   that the first divergence is `ffn_moe_gate-2` -- i.e. the *next* layer's router.
#   §9.18.4 then argued by elimination: on graph 3, layer 1's gate/up/down ids are identical and
#   layer 2's router differs, so under identical input AND identical ids, layer 1's MoE OUTPUT
#   differs => the carrier is the weight CONTENT the ids point at (the pool / slot contents), not
#   the ids and not the expert->slot mapping.
#
#   That argument is an inference from an absence. §9.18.6 turns it into a measurement: capture the
#   output tensor itself. If `ffn_moe_down-1`'s output differs while its input and its ids are the
#   same, residency is proven to be the carrier and the investigation moves from the mapping layer
#   to the residency / publish layer. Inputs and ids are already known to be identical from §9.18.3,
#   so this single reading closes it.
#
# WHAT IT REUSES (no second instrument, no second comparator)
#   kernel_cgc_ids_capture is a generic "copy up to `stride` int32 words into slot N" kernel -- it
#   has nothing ids-specific in it. The new host entry `cgc_dst_capture()` submits the SAME kernel
#   into the same command buffer with the node's DST buffer bound instead of its ids operand, at a
#   wider stride (32 words vs 8), into its OWN destination buffer. Rows are emitted in submission
#   order through a shared sequence number, so `scripts/check/ids_capture_diff.py` -- which segments
#   graphs by the ids rows and pairs nodes BY NAME -- keeps working untouched. The dst row is named
#   `<node>.dst` so it cannot collide with the ids row of the same node.
#
#   READ THIS BEFORE CHANGING THE RUN SHAPE: the ids destination is exactly at its cap in §9.18
#   (36 graphs x 117 nodes = 4096 slots), so this runner uses a SHORT run (1 request, 24 tokens
#   -> ~25 graphs) and still enables the ids capture, because the ids rows are what carry the graph
#   boundaries. Without them every dst row lands in one pseudo-graph and the comparator would
#   silently compare nothing (a false IDENTICAL -- the worst outcome this project knows).
#
# Usage:  bash Backup/run_ids_dst_capture.sh
#         NODE=ffn_moe_down-1 N_PREDICT=24 bash Backup/run_ids_dst_capture.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
NODE="${NODE:-ffn_moe_down-1}"
N_PREDICT="${N_PREDICT:-24}"
OUT="$ROOT/Backup/phase_decomp"
LOG="$ROOT/Backup/cgc_logs/ids_dst_capture"
mkdir -p "$OUT" "$LOG"
STAMP="$(date +%Y%m%d_%H%M%S)"
JSON="$OUT/ids_dst_capture_${STAMP}.json"

echo "=== §9.18.6 output capture: node=$NODE  n_predict=$N_PREDICT  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  arms: p25-gputime (host leaf) vs p25-slotgpu (S1 GPU table), both with the capture on"
echo "  thermal at launch: $(python3 scripts/check/thermal_pressure.py)"
echo

# CGC_IDS_CAPTURE must stay 1: its rows carry the graph boundaries the comparator segments by.
CGC_IDS_CAPTURE=1 \
CGC_TENSOR_CAPTURE="$NODE" \
CGC_TENSOR_CAPTURE_WORDS=32 \
RUN_REPLAY_BENCH=0 \
python3 scripts/check/decode_sweep.py \
    --profile prod25 \
    --arms p25-gputime,p25-slotgpu \
    --rounds 1 --warmup 0 --n-predict "$N_PREDICT" \
    --json "$JSON" --force 2>&1 | tail -25
echo

# Lift each arm's server log out of the row, then diff.
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
        n_ids_rows = sum(1 for line in open(tgt, errors="replace") if "CGC-IDS-CAP" in line)
        n_dst_rows = sum(1 for line in open(tgt, errors="replace") if ".dst" in line and "CGC-IDS-CAP" in line)
        print(f"  {r['tag']:<14} ids_rows={n_ids_rows:<6} dst_rows={n_dst_rows:<5} -> {tgt.name}")
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

echo "=== ids diff (re-confirms 9.18.3 in the same run) ==="
python3 scripts/check/ids_capture_diff.py "$A" "$B" --all-graphs --max 4 2>&1 | head -20 || true
echo
echo "=== the new reading: $NODE.dst, graph by graph ==="
python3 Backup/analyze_dst_capture.py "$A" "$B" "$NODE"
echo
echo "=== INSTRUMENT CONTROL (read this before believing anything above) ==="
echo "  The same-arm control must be run separately: capture the SAME arm twice and diff it against"
echo "  itself. Until that shows SAME on every graph, a dst DIFF is not evidence of a value"
echo "  difference -- it is evidence about the readout. See Backup/analyze_dst_capture.py."
echo "  Command: bash Backup/run_ids_dst_capture.sh   # twice, then diff the two p25-gputime logs"
echo "=== done $(date '+%H:%M:%S') ==="
