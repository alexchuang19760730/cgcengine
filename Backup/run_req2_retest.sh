#!/bin/bash
# [CGC 2026-09-16] Does the prefill250 req2 SIGSEGV survive the MTP_SUPPORT rebuild?
#
# The question this answers, and ONLY this question:
#   the shipping libllama-common was compiled WITHOUT -DMTP_SUPPORT (proved three ways in
#   lesson eng-gate-0030), and with that macro undefined the Qwen35MoE MTP graph never sets
#   res->t_embd (src/models/qwen35moe.cpp:733-738) and three [CGC MTP fix] blocks in
#   common/speculative.cpp are compiled out (:1523, :1544, :1692). So the binary we have been
#   debugging is NOT the MTP code path the author wrote.
#
#   Arm A (default): P250, one ~2.9k-token prompt, requests until death or req3.
#   Arm B (ZOMBIE=1): same, plus NSZombieEnabled=YES. A freed ObjC object becomes a zombie, so
#     the next message send to it logs the object's CLASS and ADDRESS instead of dying silently
#     in objc_msgSend. This names the dead object -- the mtimes/reading cannot.
#
# Usage:  bash Backup/run_req2_retest.sh                 # arm A
#         ZOMBIE=1 bash Backup/run_req2_retest.sh        # arm B
#
# Proxy note: HTTP_PROXY/http_proxy are set on this host and will silently route 127.0.0.1
# through a local proxy unless bypassed (requests -> ProxyHandler({}), curl -> --noproxy '*').
# Health note: /health answers 200 {"status":"loading model"} DURING model load, so the body is
# the signal, not the status code.
set -uo pipefail
cd /Users/alexchuang/Documents/flashkv-devserver

OUTDIR="${OUTDIR:-Backup/cgc_logs}"
# [CGC 2026-09-16] Create OUTDIR. Without this, `env … run_server.sh > "$OUTDIR/<x>.driver.log" 2>&1 &`
# fails at the REDIRECTION and therefore never executes run_server.sh at all -- no server, no server
# log, and the arm then reports `ready=no last_request=0 survived=n/a`, which is indistinguishable
# from "the server started and died". Measured: the 13:39:34 acceptance arm did exactly that with
# OUTDIR=Backup/cgc_logs/prod_accept (a directory that did not exist), and its report could not even
# be written (`tee: … No such file or directory`).
#
# This is the third defect of the same shape in this file: a harness artifact presented as an
# observation. The other two were newest_ips() (an ordering test read as a difference test) and
# newest_server_log() (globbing OUTDIR while the log lives elsewhere). A harness that cannot write
# its own evidence must say so, not report a result.
mkdir -p "$OUTDIR" || { echo "error: cannot create OUTDIR=$OUTDIR" >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="req2retest${ZOMBIE:+_zombie}${ZOMBIE:-}"
REPORT="$OUTDIR/${TAG}_${STAMP}.txt"
ARMS_FILE="$OUTDIR/${TAG}_${STAMP}.arms"
IPS_DIR="$HOME/Library/Logs/DiagnosticReports"

# [CGC 2026-09-16] The server log is NOT in OUTDIR, and assuming it was cost 240 s per arm.
#
# run_server.sh writes its log to a HARDCODED directory -- "$ROOT/Backup/cgc_logs/llama_server_<ts>.log"
# (scripts/run_server.sh:466-467) -- while OUTDIR is only where THIS script writes its own report,
# envdump, driver.log and .ips snapshot. So globing "$PWD/$OUTDIR" matched nothing for every arm
# driven with a custom OUTDIR, the 240-iteration wait loop below then ran to exhaustion, and every
# such report said `log=<none>`. That reads as "the server produced no log" when the truth is "this
# script looked in the wrong directory" -- the same failure shape as the newest_ips() ordering test
# fixed above: a wrong answer that looks like a real observation.
#
# Measured: the 12:51:59 probe arm took 5m15s, of which ~4m was this spin (its report line was
# `log=<none>  driver=48442`). The spin is not free even when it is harmless: the model sits
# resident and the pool stays warm for those 4 minutes, and this whitepaper is about thermal
# transients, so dead time between load and the first request is a variable nobody wants.
#
# CGC_SERVER_LOG_DIR can override it, but the default is the directory run_server.sh actually uses.
# The `latest` symlink sorts LAST lexically, so the [0-9]* glob keeps selecting a TIMESTAMPED log.
SERVER_LOG_DIR="${CGC_SERVER_LOG_DIR:-$PWD/Backup/cgc_logs}"
newest_server_log() { ls -t "$SERVER_LOG_DIR"/llama_server_[0-9]*.log 2>/dev/null | head -1; }

# [CGC 2026-09-16] "new crash report" has to mean NEW, not NEWEST.
#
# The previous version was `newest_ips() { ls -t "$IPS_DIR"/llama-server-*.ips | head -1; }` compared
# against a before-value, which looks like a difference test but is an ORDERING test: it cannot tell
# "this arm crashed" from "the newest file on disk changed". Measured failure, 12:05: the ub=6144
# control arm reported `new crash report: llama-server-2026-09-16-113649.ips` -- a report written 29
# minutes earlier by a different arm (and its real crash was provable from the server log alone: 4
# lines of status 5 / kIOGPUCommandBufferCallbackErrorOutOfMemory). Attribution has to be by
# identity, so snapshot the SET of report paths before the arm and diff, with the arm's start time as
# a second net because DiagnosticReports can be published after the process dies.
ips_snapshot() { ls -1 "$IPS_DIR"/llama-server-*.ips 2>/dev/null | sort; }
new_ips() {  # $1 = path of the before-snapshot ; $2 = arm start (epoch seconds)
    /opt/homebrew/bin/python3 - "$1" "$2" <<'PY'
import glob, os, sys
snap_path, start = sys.argv[1], int(sys.argv[2])
before = set(open(snap_path, encoding="utf-8", errors="replace").read().split()) \
         if os.path.exists(snap_path) else set()
ips_dir = os.path.expanduser("~/Library/Logs/DiagnosticReports")
new = [p for p in sorted(glob.glob(os.path.join(ips_dir, "llama-server-*.ips")))
       if p not in before or int(os.path.getmtime(p)) >= start]
print(" ".join(new))
PY
}
alive()             { pgrep -f "build/bin/llama-server" >/dev/null 2>&1 && echo yes || echo no; }

send_req() {  # $1 = label
    /opt/homebrew/bin/python3 - "$1" <<'PY'
import json, sys, time, urllib.request
label = sys.argv[1]
para = ("MoE 推論引擎在 16GB 統一記憶體上必須把 expert 權重放在受控的池子裡，"
        "因為一次 prefill 會碰到幾乎全部 256 個 expert，而 decode 一步只碰 8 個。")
prompt = "請閱讀以下內容並用一句話總結重點：\n" + "\n".join([para] * 46)
body = json.dumps({"messages":[{"role":"user","content":prompt}],
                   "max_tokens":24,"temperature":0,"stream":False}).encode()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # MUST bypass HTTP_PROXY
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body,
                             headers={"Content-Type":"application/json"})
t0 = time.time()
try:
    with opener.open(req, timeout=900) as resp:
        d = json.loads(resp.read())
except Exception as e:
    print(f"  {label}: DIED after {time.time()-t0:.1f}s -> {type(e).__name__}: {e}")
    sys.exit(0)
tim = d.get("timings") or {}
pn, pms, pps = tim.get("prompt_n"), tim.get("prompt_ms"), tim.get("prompt_per_second")
print(f"  {label}: OK wall={time.time()-t0:.1f}s prompt_n={pn} "
      f"prompt_ms={pms if pms is None else round(pms,1)} pp_t/s={None if pps is None else round(pps,2)}")
PY
}

{
echo "# prefill250 req2 retest  -- HEAD=$(git rev-parse --short HEAD)"
echo "# started $(date '+%F %T')   report=$REPORT"
echo "# binary under test:"
/opt/homebrew/bin/python3 - <<'PY'
# [CGC 2026-09-16] Do NOT hardcode dylib filenames here.  The SONAME embeds the commit count
# (libllama-common.0.0.<N>.dylib), so every full rebuild renames the artifact and leaves the
# previous one on disk as a stale leftover.  The first version of this header hashed
# libllama-common.0.0.239.dylib by hand; after the 11:34 rebuild the loader mapped ...0.0.275.dylib
# instead, and the header kept reporting the MTP_SUPPORT strings as ABSENT for a library that was
# not even loaded -- a false negative that reads exactly like "MTP is still not compiled in".
# So: ask the dynamic loader what it will actually map.  Walk otool -L, resolve each @rpath entry
# through the symlink chain, and hash/strings THAT file.
import hashlib, os, re, subprocess, time

BIN = "src/llama.cpp/build/bin"

def rpath_loads(root):
    """Breadth-first set of @rpath dylibs the loader will map, starting at `root`."""
    seen, order, queue = set(), [], [root]
    while queue:
        cur = queue.pop(0)
        try:
            raw = subprocess.run(["otool", "-L", cur], capture_output=True, text=True).stdout
        except Exception:
            continue
        for line in raw.splitlines()[1:]:
            m = re.search(r"@rpath/(\S+\.dylib)", line)
            if not m:
                continue
            nm = m.group(1)
            if nm in seen:
                continue
            seen.add(nm)
            order.append(os.path.join(BIN, nm))
            if os.path.exists(os.path.join(BIN, nm)):
                queue.append(os.path.join(BIN, nm))
    return order

server = os.path.join(BIN, "llama-server")
impl   = os.path.join(BIN, "libllama-server-impl.dylib")
targets, seen_real = [server], set()
for f in [impl] + rpath_loads(impl if os.path.exists(impl) else server):
    real = os.path.realpath(f)
    if real in seen_real:
        continue
    seen_real.add(real)
    targets.append(f)
# ggml-metal is dlopen'd transitively by ggml-backend, so otool may not reach it; pin it explicitly.
gm = os.path.realpath(os.path.join(BIN, "libggml-metal.0.dylib"))
if os.path.exists(gm) and gm not in seen_real:
    targets.append(gm)

print("# binary under test (resolved through the symlink chain the loader uses):")
loaded = {}
for f in targets:
    try:
        real = os.path.realpath(f)
        h = hashlib.md5(open(real, "rb").read()).hexdigest()
        mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(real)))
        tag = "" if os.path.basename(f) == os.path.basename(real) else f"  (symlink {os.path.basename(f)})"
        print(f"#   {h}  {mt}  {os.path.basename(real)}{tag}")
        loaded[os.path.basename(real)] = real
    except Exception as e:
        print(f"#   !! {f}: {e}")

# positive control on the *loaded* common lib: is the MTP_SUPPORT-only code in it?
common = [v for k, v in loaded.items() if k.startswith("libllama-common.0.0.")]
d = common[0] if common else None
tests = ["adding speculative implementation 'draft-mtp'",   # UNGUARDED -> must be present
         "MTPDBG mtp_ctor: BEFORE set_nextn",              # guard MTP_SUPPORT -> 3 of these
         "MTPDBG process: n_tokens=",
         "MTPDBG mtp_ctor: AFTER set_nextn"]
print("# MTP_SUPPORT evidence (strings -a on the common lib the loader maps"
      f"{'' if d else ' -- NO COMMON LIB RESOLVED'}):")
if d is None:
    print("#   !! cannot evaluate: no libllama-common.0.0.*.dylib was resolved")
else:
    out = subprocess.run(["strings", "-a", d], capture_output=True, text=True).stdout
    for i, needle in enumerate(tests):
        where = "UNGUARDED  (control, must be present)" if i == 0 else "GUARDED by MTP_SUPPORT"
        print(f"#   {'present' if needle in out else 'ABSENT ':>7}  {where:<34} {needle}")
    guarded = sum(1 for n in tests[1:] if n in out)
    ctrl    = tests[0] in out
    verdict = "MTP_SUPPORT=ON (compiled in)" if guarded == 3 else "MTP_SUPPORT=OFF (compiled out)"
    print(f"#   => guarded {guarded}/3, control {'present' if ctrl else 'MISSING'}"
          f"  => {verdict}")
    if not ctrl:
        print("#   !! the unguarded control string is missing too -- this is not a MTP_SUPPORT"
              " verdict, it is a broken/stripped artifact; do not quote the line above")
PY

echo "# arm: profile=prefill250 (nothing else set), requests req1..req3, gate unset"
echo "# ZOMBIE=${ZOMBIE:-0}  (1 => NSZombieEnabled=YES + MallocStackLogging=1)"
echo

# ---- what IS this arm, per the script's own resolved env (not per the label) ---------------------
env CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=prefill250 ./scripts/run_server.sh \
    > "$OUTDIR/${TAG}_${STAMP}.envdump" 2>&1
/opt/homebrew/bin/python3 - "$OUTDIR/${TAG}_${STAMP}.envdump" <<'PY'
import sys, hashlib
L = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
for l in L:
    if l.startswith(("[mode]", "[perf]")):
        print("  " + l.strip()[:150])
rows = [l for l in L if l.startswith(("ENV ", "ARG ", "CGCENV "))]
keep = ("PROFILE", "BATCH", "UBATCH", "PREFILL_STREAM", "SLAB", "CTX", "BUDGET", "MTP", "DENSE_IQ4X")
for l in rows:
    if any(k in l for k in keep):
        print("  " + l.strip()[:150])
argv = " ".join(l[4:].strip() for l in rows if l.startswith("ARG "))
print("  [argv] " + argv[:190])
norm = [l for l in rows if not l.startswith("CGCENV LOG") and "free=" not in l]
print(f"  [A11 fingerprint] {hashlib.md5(chr(10).join(norm).encode()).hexdigest()}")
PY

echo

# [CGC 2026-09-16] THE THERMAL PRECONDITION -- layer 2 of "250 的設定" (whitepaper §10.10.4).
#
# §10 established with powermetrics sample-by-sample that prefill throughput is set by the DVFS step
# the thermal governor has picked, and that the governor only returns to the top step (1470 MHz)
# after the machine has been left alone. §3.3 measured it directly: 0/60/90/120 s of quiet ->
# 122.39/133.58/127.82/124.99 t/s, while 150/180/240 s -> 294.05/300.09/307.84 t/s, and NO
# intermediate value was ever sampled -- the cliff between 120 s and 150 s is that steep.
#
# That quiet interval is not a setting in run_server.sh. It is a property of how this harness is
# DRIVEN, which is why it was missing for so long: every knob in the profile looked right.
#
# What that missing quiet interval cost (2026-09-16): the four "final binary" acceptance arms
# returned 118.30-154.77 t/s, while an arm preceded by ~60 min of no GPU prefill returned
# 268.09/273.55/256.61 t/s (and a later one, forced through this block, 254.29/282.38/265.27 t/s).
# Same binary, same A11 fingerprint a9c9dc104f057bf640e0132a83e01c12, same resolved config. The
# difference was never in the artifact.
#
# BUT -- and this is why the block below MEASURES instead of assuming -- the ">= 150 s quiet" reading
# of §10.10.4 layer 2 is FALSE as a sufficient condition, and this harness measured that itself.
# The four slow arms were NOT 111/94/144 s apart: those are differences of report stamps, and each
# arm's own run lasts ~90 s. Reading the server logs (creation of arm N+1's log minus mtime of arm
# N's) the real quiet intervals were 21 / 8 / 55 s. And when the same harness was driven
# cold-then-immediately again at 14:31/14:35 the arm with ZERO seconds of quiet returned
# 289.86 t/s on req1 -- faster than the arm that had been given 180 s. So the operative variable is
# the machine's ACCUMULATED heat budget, not the interval before this launch: one launch (~40 s of
# prefill) does not exhaust it, ~3 consecutive launches do, and the resulting plateau was ~185 t/s
# in one sequence and ~130 t/s in another -- two levels this harness cannot yet tell apart.
#
# swap is excluded as the explanation, on the same data: the 289.86 t/s arm launched at swap
# used=4975.06M while the 188.35 t/s arm launched at used=4549.69M -- the ordering is inverted.
#
# Which leaves the honest labelling below: this harness cannot enforce the precondition (nothing
# non-root can see the governor), so it CLASSIFIES the arm and refuses to let a hot-state number be
# read as a delivery number.
#
# There is no non-root instrument for the governor's current step: `pmset -g therm` answers only
# "No thermal warning level has been recorded", and the DVFS residency needs root powermetrics
# (§3.5, §10.1). The acceptance protocol therefore has to carry a root powermetrics capture whose
# verdict is the DVFS step, not the wall clock -- whitepaper §11.14.
#
# [CGC 2026-09-16 15:1x] CORRECTION to the paragraph that stood here. It read "No non-root
# instrument can see the governor", and that is FALSE. The counterexample is one key:
#
#     notifyutil -g com.apple.system.thermalpressurelevel
#
# which is the SAME notify(3) key powermetrics' thermal sampler reads to print its
# `Current pressure level` line -- no privileges, ~11 ms of CPU per read (measured: 20 reads in
# 0.225 s), IOKit scale:
#     0 = Nominal   1 = Moderate   2 = Heavy   3 = Trapping   4 = Sleeping
# The old paragraph is kept rather than deleted (CONVENTIONS E) because the way it was wrong is
# the instructive part: the instrument existed all along, and the search had stopped at
# "powermetrics needs root" -- an inference about one tool, written down as a fact about the system.
#
# So this block does four things:
#   (a) runs the layer-2 protocol wait, idempotently and reported;
#   (b) computes the quiet interval since the PREVIOUS arm ended, from a state file, and
#       CLASSIFIES this arm COLD-STATE or HOT-STATE against COLD_QUIET;
#   (c) READS the pressure level in-band -- before launch and at every request boundary -- so the
#       condition travels with the number instead of being reconstructed afterwards;
#   (d) prints the machine state that a previous round failed to record.
# The classification is the point: a hot-state number that is silently presented as a delivery
# number is the exact failure this file was written to fix (see the newest_ips() note above --
# same shape, different variable).
IDLE_BEFORE="${IDLE_BEFORE:-180}"
COLD_QUIET="${COLD_QUIET:-1800}"
STATE_FILE="${CGC_ARM_STATE:-$OUTDIR/.last_arm_end}"

# In-band thermal pressure reading. Unreadable returns '?', NEVER a silent 0: a 0 that means "could
# not read" is indistinguishable from the condition being satisfied, which is the single worst
# confusion available for this particular variable.
#
# [CGC 2026-09-16 15:2x] This definition was LOST in an earlier edit -- the three call sites landed
# and this did not, because it was submitted in the same message as another edit to this same file
# and the second write clobbered the first. The arms that ran in between therefore printed
# `[thermal pressure] before req1 = ` with an EMPTY value: `command not found` goes to stderr,
# which the tee captures, but the substituted string is empty and reads like a missing reading
# rather than a missing instrument. Lesson recorded in CONVENTIONS B31 and lessons.jsonl
# eng-tool-0038: never submit two edits to the same file in one message. The launch-level readings
# that the 15:18/15:19 arms DID produce came from the calling shell, not from here, which is why
# those arms' verdicts survive the defect -- but the in-band ones did not.
_thermal_level() {
    local raw
    raw="$(notifyutil -g com.apple.system.thermalpressurelevel 2>/dev/null | awk '{print $NF}')"
    case "$raw" in
        0) echo "0/NOMINAL"  ;;
        1) echo "1/MODERATE" ;;
        2) echo "2/HEAVY"    ;;
        3) echo "3/TRAPPING" ;;
        4) echo "4/SLEEPING" ;;
        *) echo "?/UNREADABLE" ;;
    esac
}

_now="$(date +%s)"; _since=0; _known=0
if [ -f "$STATE_FILE" ]; then
    _prev="$(cat "$STATE_FILE" 2>/dev/null || echo 0)"
    case "$_prev" in ''|*[!0-9]*) _prev=0 ;; esac
    if [ "$_prev" -gt 0 ]; then _since=$(( _now - _prev )); _known=1; fi
fi
if [ "$_known" = 0 ]; then
    THERMAL_LABEL="UNKNOWN-STATE"
elif [ "$_since" -ge "$COLD_QUIET" ]; then
    THERMAL_LABEL="COLD-STATE"
else
    THERMAL_LABEL="HOT-STATE"
fi
echo "# [thermal label] $THERMAL_LABEL -- quiet since previous arm ended = ${_since}s (COLD needs >= ${COLD_QUIET}s)"
case "$THERMAL_LABEL" in
    COLD-STATE)
        echo "#   this arm is a DELIVERY-GRADE cold sample; its prefill t/s may be quoted (whitepaper §11.14)."
        ;;
    HOT-STATE)
        echo "#   !! NOT a cold sample. The machine's heat budget may not have recovered. Report this"
        echo "#      arm's t/s as a hot-state observation; do NOT quote it as the delivery number."
        echo "#      (Measured: the same binary/config/fingerprint gave 254.29/282.38/265.27 t/s COLD and"
        echo "#       167.41-200.58 t/s HOT. The number is a property of the machine, not of the artifact.)"
        ;;
    UNKNOWN-STATE)
        echo "#   ?? no state file at $STATE_FILE -- the quiet interval is UNKNOWN, so this arm makes no"
        echo "#      thermal claim in either direction. Start a known sequence by removing the guesswork:"
        echo "#      the harness writes this file at the end of every arm, so a second arm will be labelled."
        ;;
esac
case "$IDLE_BEFORE" in
    0|0[!0-9]*|[!0-9]*)
        echo "# [thermal precondition] IDLE_BEFORE=${IDLE_BEFORE} -- protocol wait NOT APPLIED."
        echo "#   (Note: a 0 s wait does NOT by itself make a hot sample -- and it does not make a cold"
        echo "#    one either: measured 289.86 t/s on req1 with ZERO quiet. The label above is the claim.)"
        ;;
    *)
        echo "# [thermal precondition] layer 2 of §10.10.4: requiring >= ${IDLE_BEFORE}s quiet before launch"
        if [ "$(alive)" = "yes" ]; then
            echo "#   !! a llama-server is ALREADY resident -- this arm would inherit its heat, and the"
            echo "#      quiet interval cannot start until it is gone. Precondition NOT satisfied."
        fi
        _q0="$(date +%s)"
        echo "#   quiet from $(date '+%F %T') ..."
        sleep "$IDLE_BEFORE"
        echo "#   ... to $(date '+%F %T')  (quiet = $(( $(date +%s) - _q0 ))s)"
        ;;
esac
# Record the state variable a previous round left unrecorded, and which is therefore still not
# excluded: swap is monotonic within a session and never recovers. If a cold arm is STILL slow with
# this line showing the same swap as the hot arms, then swap cannot be the explanation -- which is
# what makes this line worth having rather than decorative.
echo "# [machine state] $(sysctl -n vm.swapusage 2>/dev/null)"
echo "# [machine state] load:$(sysctl -n vm.loadavg 2>/dev/null)"
T_PRELAUNCH="$(_thermal_level)"
echo "# [thermal pressure] before launch = $T_PRELAUNCH   (notifyutil -g com.apple.system.thermalpressurelevel)"
echo

BEFORE_LOG="$(newest_server_log)"
ips_snapshot > "$OUTDIR/${TAG}_${STAMP}.ips_before"
ARM_START_EPOCH="$(date +%s)"

unset LLAMA_EXPERT_CACHE_BATCH_INVARIANT
ZENV=()
if [ "${ZOMBIE:-0}" = "1" ]; then
    ZENV=(NSZombieEnabled=YES MallocStackLogging=1 MallocScribble=1)
    echo "  zombie instrumentation: ${ZENV[*]}"
fi
# NB: /bin/bash on macOS is 3.2, where "${ZENV[@]}" on an EMPTY array is an unbound-variable
# error under `set -u`. ${ZENV[@]+"${ZENV[@]}"} expands to nothing when the array is empty.
env ${ZENV[@]+"${ZENV[@]}"} CGC_SERVER_PROFILE=prefill250 ./scripts/run_server.sh \
    > "$OUTDIR/${TAG}_${STAMP}.driver.log" 2>&1 &
DRIVER=$!

SLOG=""
for i in $(seq 1 240); do
    N="$(newest_server_log)"
    if [ -n "$N" ] && [ "$N" != "$BEFORE_LOG" ]; then SLOG="$N"; break; fi
    sleep 1
done
# [CGC 2026-09-16] Say it out loud. The old silent fallthrough turned a wrong-directory lookup into
# a 4-minute delay plus a `log=<none>` that looked like a server-side fact.
if [ -z "$SLOG" ]; then
    echo "  !! no NEW server log appeared in $SERVER_LOG_DIR within 240s (before: ${BEFORE_LOG:-<none>})"
    echo "     -- continuing; this report's log= field will read <none>."
fi
echo "  log=${SLOG:-<none>}  driver=$DRIVER"

LAST=""; READY=no
for i in $(seq 1 420); do
    LAST="$(curl --noproxy '*' -s --max-time 3 http://127.0.0.1:8080/health 2>/dev/null || true)"
    case "$LAST" in
        *'"status":"ok"'*) echo "  health OK after ${i}s"; READY=yes; break ;;
    esac
    kill -0 "$DRIVER" 2>/dev/null || { echo "  !! driver exited early (last health: $LAST)"; break; }
    sleep 1
done

SURVIVED="n/a"; REQS="0"; T_LEVELS=""; T_MAX="0"
if [ "$READY" = yes ]; then
    for r in req1 req2 req3; do
        _L0="$(_thermal_level)"
        echo "  [thermal pressure] before $r = $_L0"
        send_req "$r"; REQS="$r"
        _L1="$(_thermal_level)"
        echo "  [thermal pressure] after  $r = $_L1"
        T_LEVELS="${T_LEVELS}${r}:${_L0}->${_L1} "
        # Numeric high-water mark over the arm, so a mid-arm excursion to Heavy/Very Heavy is not
        # hidden by two endpoint readings that happen to be Nominal.
        case "${_L0%%/*}" in 0|1|2|3|4) [ "${_L0%%/*}" -gt "$T_MAX" ] && T_MAX="${_L0%%/*}" ;; esac
        case "${_L1%%/*}" in 0|1|2|3|4) [ "${_L1%%/*}" -gt "$T_MAX" ] && T_MAX="${_L1%%/*}" ;; esac
        st="$(alive)"; echo "  alive after $r: $st"
        [ "$st" = yes ] || break
    done
    SURVIVED="$(alive)"
fi

# [CGC 2026-09-16] ReportCrash publishes the .ips AFTER the process dies, so one check immediately
# after the arm can miss the very report the arm caused. Measured publish latency: the 12:46:00 mmap
# arm died at ~12:46:56 and its report (llama-server-2026-09-16-124700.ips) has mtime 12:47:00 --
# about 4 s, against this script's bare `sleep 3`.
#
# That arm's report DID list it, but partly by luck: a since-removed 240 s wait (newest_server_log
# globing the wrong directory) delayed the check far past the publish. Removing that wait is right,
# and it exposes this race, so make the wait explicit and bounded instead of implicit.
#
# No miss has been observed yet: after the 240 s spin was removed, all four crashing arms at 13:23-
# 13:26 attributed their own report correctly (3 mmap + 1 ub=6144). This is prevention, not a fix
# for an observed failure -- but the 4 s publish latency says the margin is thin.
#
# Cost control: only arms that look like they died pay the poll, and it stops as soon as a report
# lands, so a healthy arm still pays 3 s.
IPS_POLL=0
[ "$SURVIVED" != "yes" ] && IPS_POLL=1
if [ -n "$SLOG" ] && grep -qE "CGC-METAL-FAIL|ggml_abort|failed with status" "$SLOG" 2>/dev/null; then
    IPS_POLL=1
fi
NEWIPS=""
IPS_CANNOT=""
if [ ! -f "$OUTDIR/${TAG}_${STAMP}.ips_before" ]; then
    # [CGC 2026-09-16] No before-set means the difference test has no left-hand side. Listing every
    # report on disk instead would be worse than listing nothing: measured at 13:39:34, an absent
    # snapshot made this section claim EIGHT pre-existing reports (12:47 through 13:35, written by
    # other arms) as "written by THIS arm". Say you cannot tell.
    IPS_CANNOT="before-snapshot ${OUTDIR}/${TAG}_${STAMP}.ips_before does not exist"
elif [ "$IPS_POLL" = "1" ]; then
    for _ in $(seq 1 30); do
        sleep 2
        NEWIPS="$(new_ips "$OUTDIR/${TAG}_${STAMP}.ips_before" "$ARM_START_EPOCH")"
        [ -n "$NEWIPS" ] && break
    done
    echo "  (crash-report poll: $( [ -n "$NEWIPS" ] && echo "report found" || echo "none within 60s" ))"
else
    sleep 3
    NEWIPS="$(new_ips "$OUTDIR/${TAG}_${STAMP}.ips_before" "$ARM_START_EPOCH")"
fi

if [ "$SURVIVED" = yes ]; then
    pkill -TERM -f "build/bin/llama-server" 2>/dev/null
    for i in $(seq 1 90); do pgrep -f "build/bin/llama-server" >/dev/null 2>&1 || break; sleep 1; done
    pgrep -f "build/bin/llama-server" >/dev/null 2>&1 && pkill -9 -f "build/bin/llama-server"
    sleep 2
fi
kill -TERM "$DRIVER" 2>/dev/null; sleep 1; kill -9 "$DRIVER" 2>/dev/null; sleep 2

echo
echo "  ===== RESULT: ready=$READY last_request=$REQS survived=$SURVIVED ====="
if [ -n "$IPS_CANNOT" ]; then
    echo "  crash-report attribution: UNAVAILABLE -- $IPS_CANNOT"
    echo "    (listing nothing on purpose: with no before-set, every report on disk looks new,"
    echo "     which is how the 13:39:34 run came to claim 8 pre-existing reports as its own.)"
elif [ -n "$NEWIPS" ]; then
    echo "  new crash report(s) written by THIS arm:"
    for f in $NEWIPS; do echo "    $(basename "$f")"; done
    echo "  (pre-existing reports are not listed; the before-snapshot is"
    echo "   $OUTDIR/${TAG}_${STAMP}.ips_before)"
else
    echo "  new crash report: none"
fi
echo "$TAG|$SLOG|$SURVIVED|${NEWIPS:-none}|$READY|$REQS" >> "$ARMS_FILE"

echo
echo "  --- zombie / objc diagnostics found in the server log ---"
/opt/homebrew/bin/python3 - "$SLOG" <<'PY'
import sys
try: L = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
except Exception as e: print(f"    unreadable: {e}"); sys.exit(0)
hits = [l for l in L if any(k in l for k in
        ("zombie", "Zombie", "deallocated", "malloc", "MallocStackLogging", "SIGABRT",
         "abort", "BATCH-INVARIANT", "CGC-SYNC", "blocker-B stats"))]
if not hits:
    print("    (none)")
for l in hits[-25:]:
    print("    " + l.strip()[:170])
print("  --- last 5 log lines ---")
for l in L[-5:]:
    print("    " + l.strip()[:170])
PY

echo
echo "  --- server log request timeline ---"
/opt/homebrew/bin/python3 - "$SLOG" <<'PY'
import sys, re
try: L = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
except Exception: sys.exit(0)
for i, l in enumerate(L, 1):
    if re.search(r"launch_slot_\S+ .*processing task|print_timing|slot released|slot .* released|prompt eval time|CGC-PREFILL-STREAM", l):
        print(f"    {i:>5} | {l.strip()[:155]}")
PY
} 2>&1 | tee -a "$REPORT"
echo; echo "=== report: $REPORT ==="

# [CGC 2026-09-16] Close the thermal bookkeeping: record when this arm's sustained prefill ended, so
# the NEXT arm can compute its quiet interval and label itself COLD/HOT instead of being believed.
# Written after the tee so the file's mtime means "this arm is done", which is the interval that
# matters -- the loading phase is IO/CPU, not the prefill that drives the governor.
date +%s > "$STATE_FILE" 2>/dev/null || true
