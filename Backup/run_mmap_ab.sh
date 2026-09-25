#!/bin/bash
# [CGC 2026-09-26] Two defects found by reading this script's own output after the 09-26 run:
#   (a) `srv_log` recorded `sorted(ws_seen)[0]` -- the ALPHABETICALLY first candidate log, not the
#       arm's. It printed `llama_server_20260926_000012.log` (the previous arm's) for arm 2. Now all
#       matched logs are recorded, so which log an arm's numbers came from is auditable.
#   (b) `ws_info` counted `recommendedMaxWorkingSetSize  =`, which is a GGML_LOG_INFO line that this
#       configuration does not emit at all -- so the column that was supposed to prove "this arm's
#       Metal output was captured" was structurally 0. Replaced by `metal_init` (`ggml_metal_init`),
#       which is present in both load modes (measured: 2 in the none arm, 1 in the mmap arm).
#   (c) the `# columns:` header was appended on EVERY run while the printer reads the FIRST one, so a
#       later run with a different column set silently misaligns against an older header. The header
#       is now written only when the summary file is created.
#
# [CGC 2026-09-16] Interleaved A/B: does --load-mode mmap remove the prefill250 req2 GPU OOM?
#
# The question, and why it is the right one:
#   The req2 OOM is a GPU-side allocation failure (MTLCommandBuffer status 5,
#   kIOGPUCommandBufferCallbackErrorOutOfMemory) that only fires on the SECOND request of a
#   prefill250 session. [防護 2d] computes why the box is over-committed: with --load-mode none the
#   whole 13030 MiB model is anonymous and non-reclaimable, plus the expert pool, plus the prefill
#   working set, against 16384 MiB.
#
#   --load-mode mmap is the only lever that removes the *cause* rather than the symptom: it makes the
#   model file-backed, so those pages are reclaimable. It used to be unusable because the expert
#   cache's L4 pool is adopted from the expert tensors' storage and written from the CPU, which
#   faults on a read-only mapping. That is fixed in-tree now (the pool gets its own "_Pool" buffer
#   type), so the hypothesis under test is:
#
#       H: with the pool/mmap fix, `--load-mode mmap` survives req1..req3 at ub=6144 (the full-speed
#          configuration that the ub=4096 workaround pays ~22% for).
#
#   Control: none/6144 must still die at req2 -- if it does not, the machine changed and the whole
#   comparison is void. Arms are interleaved and repeated so machine thermal/swap drift cannot be
#   mistaken for an effect (same discipline as the kernel A/Bs).
#
# Usage:  bash Backup/run_mmap_ab.sh
#         ROUNDS=1 bash Backup/run_mmap_ab.sh
#         ARMS_R1="mmap:6144 none:6144" ARMS_R2="none:6144 mmap:6144" bash Backup/run_mmap_ab.sh
#
# ── 2026-09-26 pre-registration, frozen BEFORE the arms ran ────────────────────────────────
# Question this round: the 09-16 verdict -- "mmap survives at no measured width (0/3)" -- was
# measured BEFORE P0/P1/P2 existed (2026-09-24). Does it still hold?
#   Arms: none:5632 and mmap:5632, interleaved and repeated (R2 reversed). 5632 is the CURRENT
#   default width, at which none survives 4/4 historically -- so if the mmap arm dies while the
#   none arm lives, the cause is the load mode, not the width or the machine.
#   PRIMARY criterion: the count of "greater than the recommended max working set size" in the
#   arm's own server log (column ws_warn) -- historically 0 across 119 load_mode=none launches and
#   5/5 in the decidable mmap logs. Secondary: ready / survived across req1..req3.
#   VOID if the none control does not survive: then the box decided, not the design.
#   Box deviation declared: arms start at whatever swap the box has (see the `### before:` lines);
#   the contract's 2048 MiB start line is breached today, so the interlaced control carries the
#   burden of proof that the box can still discriminate.
#
# Arm syntax: <load_mode>:<ub>[:KEY=VAL[,KEY=VAL...]]
#   The optional third field is extra env applied on top of the prefill250 profile for that arm
#   only, e.g.  none:6144:CGC_SERVER_MTP=0  . Without it the arm is exactly the profile + ub
#   override, i.e. the same arm shape used for every measurement recorded before 2026-09-16 17:00.
#   The extra is echoed into the summary's `extra` column, so a surviving arm can never be
#   attributed to the wrong configuration later.
set -uo pipefail
cd /Users/alexchuang/Documents/flashkv-devserver

OUTDIR="${OUTDIR:-Backup/cgc_logs/mmap_ab}"
ROUNDS="${ROUNDS:-2}"
ARMS_R1="${ARMS_R1:-none:4096 mmap:4096 mmap:6144 none:6144}"
ARMS_R2="${ARMS_R2:-mmap:6144 none:6144}"
SUMMARY="$OUTDIR/summary.tsv"
mkdir -p "$OUTDIR"

swap_now() { sysctl -n vm.swapusage 2>/dev/null | sed 's/  */ /g'; }
free_now() {
    vm_stat 2>/dev/null | awk '/Pages free/ {gsub(/\./,"",$3); printf "%.0fMiB", $3*4096/1048576}'
}
wait_idle() {
    for _ in $(seq 1 90); do
        pgrep -f "build/bin/llama-server" >/dev/null 2>&1 || return 0
        sleep 1
    done
    return 1
}

run_arm() { # $1 = round, $2 = load_mode, $3 = ub, $4 = extra env (comma-separated KEY=VAL, may be empty)
    local rnd="$1" lm="$2" ub="$3" ex="${4:-}"
    local tag="r${rnd}_${lm}_ub${ub}"
    [ -n "$ex" ] && tag="${tag}_$(printf '%s' "$ex" | tr '=,' '--')"
    local log="$OUTDIR/${tag}.out"

    wait_idle || { echo "!! leftover llama-server before $tag -- aborting" >&2; return 1; }

    echo "=================================================================="
    echo "### arm $tag   $(date '+%F %T')"
    echo "### before: swap=$(swap_now)  free=$(free_now)"
    echo "=================================================================="

    # Extra env is applied as CGC_SERVER_* on top of the prefill250 profile that
    # run_req2_retest.sh pins. Seeded with a marker so the array is never empty -- bash 3.2
    # (which is /bin/bash on this box) reports "${arr[@]}" on an empty array as an unbound
    # variable under `set -u`.
    local -a eenv=("CGC_MMAP_AB_TAG=$tag")
    if [ -n "$ex" ]; then
        local kv
        local IFS=','
        for kv in $ex; do
            [ -z "$kv" ] && continue
            eenv+=("$kv")
        done
    fi

    local t0
    t0=$(date +%s)
    env "${eenv[@]}" \
        CGC_SERVER_LOAD_MODE="$lm" \
        CGC_SERVER_UBATCH="$ub" \
        CGC_SERVER_BATCH="$ub" \
        OUTDIR="$OUTDIR" \
        bash Backup/run_req2_retest.sh > "$log" 2>&1

    echo "### after:  swap=$(swap_now)  free=$(free_now)"

    /opt/homebrew/bin/python3 - "$log" "$tag" "$lm" "$ub" "$rnd" "$SUMMARY" "$ex" "$t0" <<'PY'
import glob, json, os, re, sys

log, tag, lm, ub, rnd, summary, ex, t0 = sys.argv[1:9]
t0 = int(t0)
lines = open(log, encoding="utf-8", errors="replace").read().splitlines()

reqs, died = {}, {}
for l in lines:
    m = re.match(r"\s*req(\d): OK .*prompt_n=(\d+).*pp_t/s=(\S+)", l)
    if m:
        reqs[m.group(1)] = (m.group(2), m.group(3))
    m = re.match(r"\s*req(\d): DIED after (\S+)s ->\s*(.*)", l)
    if m:
        died[m.group(1)] = (m.group(2), m.group(3)[:60])

res = {}
for l in lines:
    m = re.search(r"RESULT: ready=(\S+) last_request=(\S+) survived=(\S+)", l)
    if m:
        res = {"ready": m.group(1), "last": m.group(2), "survived": m.group(3)}

crashes = []
in_crash = False
for l in lines:
    if "new crash report(s) written by THIS arm" in l:
        in_crash = True
        continue
    if in_crash:
        s = l.strip()
        if not s or s.startswith("("):
            if s.startswith("(pre-existing"):
                in_crash = False
            continue
        crashes.append(s)

# [CGC 2026-09-16] An arm whose before-snapshot could not be read reports attribution as UNAVAILABLE
# rather than listing every report on disk. Surface that as "?" -- a silent 0 would read as
# "this arm wrote no crash report", which is a different and stronger claim than "could not tell".
uncertain = any("crash-report attribution: UNAVAILABLE" in l for l in lines)

# the loader's own md5 table: "#   <md5>  <mtime>  <name>"
md5s = {}
for l in lines:
    m = re.match(r"#\s+([0-9a-f]{32})\s+(\S+ \S+)\s+(\S+)", l)
    if m:
        md5s[m.group(3)] = m.group(1)
a11 = ""
for l in lines:
    m = re.search(r"\[A11 fingerprint\] ([0-9a-f]{32})", l)
    if m:
        a11 = m.group(1)
argv = ""
for l in lines:
    m = re.search(r"\[argv\] (.*)", l)
    if m:
        argv = m.group(1).strip()
common = next((v for k, v in md5s.items() if k.startswith("libllama-common.0.0.")), "")

# ── the criterion this run was pre-registered on: did Metal say its working set was exceeded?
# `recommendedMaxWorkingSetSize  = ...` is an INFO line printed on EVERY launch, so it doubles as
# proof that this arm's Metal output was captured at all; the WARN fires only when
# currentAllocatedSize > recommendedMaxWorkingSetSize (ggml-metal-device.m:1539, compiled in unless
# GGML_METAL_NDEBUG -- verified present in the shipping dylib with strings(1)).
# A missing server log reports "?", never 0: "no warning" and "could not read a log" are different
# claims and this file has already been burned once by conflating them.
srv_logs = set(re.findall(r"(/\S*llama_server_\d[\d_]*\.log)", "\n".join(lines)))
if not srv_logs:
    srv_logs = {p for p in glob.glob("Backup/cgc_logs/llama_server_*.log")
                if os.path.getmtime(p) >= t0 - 5}
ws_warn = ws_info = 0
ws_seen = []
for p in sorted(srv_logs):
    try:
        txt = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    ws_seen.append(p)
    ws_warn += txt.count("greater than the recommended max working set size")
    ws_info += txt.count("ggml_metal_init")
if not ws_seen:
    ws_warn = ws_info = "?"

row = {
    "round": rnd, "arm": tag, "load_mode": lm, "ub": ub, "extra": ex or "-",
    "ws_warn": ws_warn, "metal_init": ws_info,
    "srv_logs": (",".join(os.path.basename(p) for p in ws_seen) if ws_seen else "?"),
    "ready": res.get("ready", "?"), "last": res.get("last", "?"), "survived": res.get("survived", "?"),
    "req1": (f"{reqs['1'][1]}" if "1" in reqs else f"DIED@{died['1'][0]}s" if "1" in died else "-"),
    "req2": (f"{reqs['2'][1]}" if "2" in reqs else f"DIED@{died['2'][0]}s" if "2" in died else "-"),
    "req3": (f"{reqs['3'][1]}" if "3" in reqs else f"DIED@{died['3'][0]}s" if "3" in died else "-"),
    "crashes": "?" if uncertain else len(crashes),
    "crash_names": "attribution-unavailable" if uncertain else (",".join(crashes) or "none"),
    "common_md5": common[:12], "a11": a11,
    "argv_has_mmap": "yes" if re.search(r"--load-mode\s+mmap", argv) else "no",
}
cols = ["round", "arm", "load_mode", "ub", "extra", "ready", "last", "survived",
        "req1", "req2", "req3", "crashes", "ws_warn", "metal_init", "srv_logs",
        "common_md5", "a11", "argv_has_mmap"]
with open(summary, "a", encoding="utf-8") as f:
    f.write("\t".join(str(row[c]) for c in cols) + "\n")

print(f"  -> ready={row['ready']} last={row['last']} survived={row['survived']} "
      f"req1={row['req1']} req2={row['req2']} req3={row['req3']}")
print(f"  -> new crashes={row['crashes']} ({row['crash_names']})")
print(f"  -> working-set WARN x{row['ws_warn']}  (ggml_metal_init seen x{row['metal_init']}, "
      f"logs={row['srv_logs']})")
print(f"  -> argv has --load-mode mmap: {row['argv_has_mmap']}   A11={a11}   common={common[:12]}")
PY
}

# Header only on creation: the printer reads the FIRST `# columns:` line, so appending a second one
# after a column-set change would leave every later row misaligned against the older header.
if [ ! -s "$SUMMARY" ]; then
    echo "# mmap A/B  HEAD=$(git rev-parse --short HEAD)  started $(date '+%F %T')" > "$SUMMARY"
    echo "# columns: $(printf 'round arm load_mode ub extra ready last survived req1 req2 req3 crashes ws_warn metal_init srv_logs common_md5 a11 argv_has_mmap')" >> "$SUMMARY"
fi

for rnd in $(seq 1 "$ROUNDS"); do
    if [ "$rnd" = "1" ]; then arms="$ARMS_R1"; else arms="$ARMS_R2"; fi
    echo
    echo "################ ROUND $rnd ################"
    for a in $arms; do
        # arm syntax: <load_mode>:<ub>[:KEY=VAL[,KEY=VAL...]]
        # `read` leaves the 3rd field empty for the 2-field form, so the old syntax still works.
        # NB: no `local` here -- this loop is at TOP LEVEL, not inside a function, and bash rejects
        # `local` outside one ("can only be used in a function"). It only warns and continues, but a
        # warning per arm in the log is noise that can hide a real one.
        IFS=':' read -r lm ub ex <<< "$a"
        run_arm "$rnd" "$lm" "$ub" "${ex:-}"
    done
done

echo
echo "=================================================================="
echo "SUMMARY ($SUMMARY)"
echo "=================================================================="
/opt/homebrew/bin/python3 - "$SUMMARY" <<'PY'
# The column names come from the `# columns:` comment line, NOT from rows[0]. This file has no
# header ROW -- by construction every non-data line is a `#` comment -- so treating rows[0] as the
# header made the data row its own header: it then reported every real column as "missing" and
# "0 data rows", which is a presentation failure that looks like an experiment failure. The
# `# columns:` line exists precisely so the reader does not have to guess or hardcode the order.
#
# Being driven by that line also makes the printer survive a file written by an older revision:
# old rows have fewer fields, old `# columns:` lines name fewer columns, and zip() stops at the
# shorter side instead of misaligning.
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    lines = f.read().splitlines()

cols_line = next((l for l in lines if l.startswith("# columns:")), "")
hdr = cols_line.split(":", 1)[1].split() if cols_line else []
rows = [l.split("\t") for l in lines if not l.startswith("#") and l.strip()]
rows = [[("" if c == "-" else c) for c in r] for r in rows]

if not hdr:
    print("  (no '# columns:' line -- cannot name the columns; raw file follows)")
    for r in rows:
        print("  " + "  ".join(r))
    raise SystemExit(0)

keep = ["arm", "ready", "last", "survived", "req1", "req2", "req3", "crashes", "common_md5"]
present = [k for k in keep if k in hdr]
missing = [k for k in keep if k not in hdr]
if missing:
    print(f"  (summary columns absent from this file: {missing})")
if not present:
    print("  (none of the requested columns exist; raw header: " + "\t".join(hdr) + ")")
    raise SystemExit(0)

recs = [dict(zip(hdr, r)) for r in rows]
w = {k: max([len(k)] + [len(r.get(k, "")) for r in recs]) for k in present}
print("  " + "  ".join(k.ljust(w[k]) for k in present))
for r in recs:
    print("  " + "  ".join(r.get(k, "").ljust(w[k]) for k in present))
if not recs:
    print("  (0 data rows -- comment lines only)")
PY
