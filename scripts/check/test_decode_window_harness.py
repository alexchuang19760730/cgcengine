#!/usr/bin/env python3
"""Self-test for the harness verdicts.

The verdict logic is the deliverable here -- a report that silently says CONFIRMED when the effect
is inside the box's own drift would be worse than no report. So each branch is driven with a
synthetic state that must land on a specific verdict, including the ones that must NOT conclude.

Run: python3 scripts/check/test_decode_window_harness.py

Two defects this suite was written to catch, both found by it:
  * BASELINE_AGREE_PCT was declared and never used, so a 0.0% baseline spread made any 1% wobble
    count as "moved" -- launch noise rounded into a conclusion.
  * the CONFIRMED branch read the arm's step via _step() on a summary row (always None), which made
    CONFIRMED unreachable: a verdict function that could only ever say no.
"""
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "h", os.path.join(HERE, "decode_window_harness.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def rep(ntok1=95.0, ntok4=150.0, mpt=100.0, layers=None):
    return {"ms_per_token": mpt, "tokens": 92,
            "steps": {"1": {"total": ntok1, "wait": ntok1 - 20, "cb": 20.0, "n": 9},
                      "4": {"total": ntok4, "wait": ntok4 - 30, "cb": 30.0, "n": 30}},
            "layers": layers or {}}


def lane_a(b1=None, b0=None, b1b=None, ok=True):
    st = "done" if ok else "interrupted"
    return {"lane_a": {
        "b1": {"status": st, "bitident": "1", "reps": {"rep2_warm": b1 or rep()},
               "server_log": "b1.log"},
        "b0": {"status": st, "bitident": "0", "reps": {"rep2_warm": b0 or rep()},
               "server_log": "b0.log"},
        "b1b": {"status": st, "bitident": "1", "reps": {"rep2_warm": b1b or rep()},
                "server_log": "b1b.log"}}}


def show(name, state):
    v = h.verdict_lane_a(state)
    kinds = [k for k, _ in v["verdicts"]]
    print(f"\n--- {name}\n    kinds: {kinds}")
    for k, t in v["verdicts"]:
        print(f"      [{k}] {t[:100]}")
    return kinds


fails = []


def expect(name, kinds, want_contains, want_absent=()):
    for w in want_contains:
        if w not in kinds:
            fails.append(f"{name}: expected {w} in {kinds}")
    for a in want_absent:
        if a in kinds:
            fails.append(f"{name}: did NOT expect {a} in {kinds}")


# 1. control flat, verify step much faster -> CONFIRMED
k = show("control flat + verify much faster (expected CONFIRMED)",
         lane_a(rep(ntok4=152.0), rep(ntok1=94.0, ntok4=110.0), rep(ntok4=150.0)))
expect("confirmed", k, ["CONFIRMED"], ["REFUTED", "NOT SUPPORTED"])

# 2. both moved -> REFUTED
k = show("both control and test moved (expected REFUTED)",
         lane_a(rep(), rep(ntok1=70.0, ntok4=110.0), rep()))
expect("refuted", k, ["REFUTED"], ["CONFIRMED"])

# 3. nothing moved -> NOT SUPPORTED
k = show("nothing moved (expected NOT SUPPORTED)",
         lane_a(rep(), rep(ntok1=95.0, ntok4=150.0), rep()))
expect("not supported", k, ["NOT SUPPORTED"], ["CONFIRMED"])

# 4. baselines 23.5% apart, effect inside that spread -> underpowered, so NOT SUPPORTED (and said
#    as underpowered rather than as evidence against the hypothesis)
k = show("baselines drifted apart, effect inside (expected NOT SUPPORTED, not CONFIRMED)",
         lane_a(rep(ntok4=150.0), rep(ntok4=140.0), rep(ntok4=190.0)))
expect("drifted baselines", k, ["NOT SUPPORTED", "CAVEAT"], ["CONFIRMED", "REFUTED"])
if not any("BEYOND" not in t and "TEST" in t for _, t in h.verdict_lane_a(
        lane_a(rep(ntok4=150.0), rep(ntok4=140.0), rep(ntok4=190.0)))["verdicts"]):
    fails.append("drifted baselines: the TEST line should say 'within baseline spread'")
if not any("underpowered" in t for _, t in h.verdict_lane_a(
        lane_a(rep(ntok4=150.0), rep(ntok4=140.0), rep(ntok4=190.0)))["verdicts"]):
    fails.append("an inside-drift result must say it is underpowered, not negative")

# 4b. baselines agree, effect inside the 5% noise floor -> NOT SUPPORTED, not CONFIRMED
k = show("effect inside the 5% floor (expected NOT SUPPORTED, not CONFIRMED)",
         lane_a(rep(ntok4=150.0), rep(ntok4=143.0), rep(ntok4=151.0)))
expect("inside floor", k, ["NOT SUPPORTED"], ["CONFIRMED", "REFUTED", "INCONCLUSIVE"])

# 4c. the floor must not swallow a real effect: 40% faster on a 0.0% spread -> CONFIRMED
k = show("large effect on a tight pair (expected CONFIRMED)",
         lane_a(rep(ntok4=150.0), rep(ntok4=90.0), rep(ntok4=150.0)))
expect("tight large effect", k, ["CONFIRMED"], ["REFUTED", "INCONCLUSIVE", "NOT SUPPORTED"])

# 5. verify slower -> PARTIAL
k = show("verify slower (expected PARTIAL)",
         lane_a(rep(ntok4=150.0), rep(ntok4=260.0), rep(ntok4=152.0)))
expect("partial", k, ["PARTIAL"], ["CONFIRMED"])

# 6. incomplete -> nothing concluded
k = show("interrupted arm (expected INCOMPLETE only)", lane_a(ok=False))
expect("incomplete", k, ["INCOMPLETE"], ["CONFIRMED", "REFUTED", "NOT SUPPORTED"])

# ---- lane B: residue concentration + recovery
def lb(rep4_wait, rep5_wait, rep4_rate=101.0, rep5_rate=84.0, cb4=4.6, cb5=4.6, warm=65.0):
    # Residue/recovery are driven by ms/token (rep4_rate / rep5_rate), because that is the clock the
    # harness judges on; the waits are carried along to prove they are no longer the criterion.
    # Warm baseline in this fixture is (84.0 + 90.0) / 2 = 87.0 ms/token.
    def layers(delta_each, n_top=0, top_delta=0.0):
        d = {L: {"wait": warm, "n": 5} for L in range(40)}
        for L in range(40):
            d[L]["wait"] = warm + delta_each
        for L in range(n_top):
            d[L]["wait"] = warm + top_delta
        return d
    return {"lane_b": {"status": "done", "engine_digest": {}, "reps": {
        "rep1_cold": {"ms_per_token": 118.0, "tokens": 92, "steps": {"1": {"total": 110, "wait": 77, "cb": 20, "n": 5}}},
        "rep2_warm": {"ms_per_token": 84.0, "tokens": 92, "steps": {"1": {"total": 76, "wait": warm, "cb": 4.8, "n": 5}}, "layers": layers(0)},
        "rep3_warm": {"ms_per_token": 90.0, "tokens": 92, "steps": {"1": {"total": 78, "wait": warm + 3, "cb": 3.7, "n": 5}}, "layers": layers(0)},
        "thrash_slab2000": {"ms_per_token": 110.0, "tokens": 8, "steps": {}},
        "rep4_after_thrash": {"ms_per_token": rep4_rate, "tokens": 92, "steps": {"1": {"total": 94, "wait": rep4_wait, "cb": cb4, "n": 5}}, "layers": layers(2.0, n_top=5, top_delta=14.0)},
        "rep5_after_thrash2": {"ms_per_token": rep5_rate, "tokens": 92, "steps": {"1": {"total": 76, "wait": rep5_wait, "cb": cb5, "n": 5}}}}}}


vb = h.verdict_lane_b(lb(rep4_wait=79.0, rep5_wait=66.0))
kinds = [k for k, _ in vb["verdicts"]]
print("\n--- lane B: concentrated residue + second pass recovers")
for k, t in vb["verdicts"]:
    print(f"      [{k}] {t[:100]}")
expect("laneB concentrated/recovered", kinds, ["concentrated", "recovered"])

vb = h.verdict_lane_b(lb(rep4_wait=79.0, rep5_wait=79.0, rep5_rate=106.0))
kinds = [k for k, _ in vb["verdicts"]]
print("\n--- lane B: residue does NOT clear on a second pass (rate still 106 vs warm 87)")
for k, t in vb["verdicts"]:
    print(f"      [{k}] {t[:100]}")
expect("laneB not recovered", kinds, ["not recovered"])

# Real data lesson: rep5's WAIT was 5.6% off warm while its RATE was back on baseline. Judging on
# `wait` said "not recovered" and would have sent someone to re-warm a pool that was already fine.
vb = h.verdict_lane_b(lb(rep4_wait=79.0, rep5_wait=59.5, rep5_rate=84.0))
kinds = [k for k, _ in vb["verdicts"]]
print("\n--- lane B: wait off by 5.6% but rate on baseline -> must be recovered")
for k, t in vb["verdicts"]:
    print(f"      [{k}] {t[:100]}")
expect("laneB wait-vs-rate", kinds, ["recovered"], ["not recovered"])

# ... and a residue too small to engineer away must not get a concentration shape.
vb = h.verdict_lane_b(lb(rep4_wait=64.0, rep5_wait=64.0, rep4_rate=87.5, rep5_rate=87.0))
kinds = [k for k, _ in vb["verdicts"]]
print("\n--- lane B: +0.6% residue -> no material residue, and no 'concentrated' claim")
for k, t in vb["verdicts"]:
    print(f"      [{k}] {t[:100]}")
expect("laneB immaterial", kinds, ["no material residue", "recovered"], ["concentrated", "global"])

vb = h.verdict_lane_b({"lane_b": {"status": "interrupted"}})
kinds = [k for k, _ in vb["verdicts"]]
print(f"\n--- lane B interrupted: {kinds}")
expect("laneB incomplete", kinds, ["INCOMPLETE"])

print("\n=== HTML generation on an empty state ===")
html = h.write_html({"lane_a": {}, "lane_b": {}, "log": ["x"]})
print("  length:", len(html), " has INCOMPLETE:", "INCOMPLETE" in html,
      " has </html>:", html.rstrip().endswith("</html>"))
if not html.rstrip().endswith("</html>"):
    fails.append("empty-state HTML is not closed")
if "INCOMPLETE" not in html:
    fails.append("empty-state HTML should say INCOMPLETE, not imply a result")

# A populated report must carry the arms, the per-layer residue table and both verdicts -- an empty
# report is not evidence that the renderer works.
print("\n=== HTML generation on a populated state ===")
full = lane_a(rep(ntok4=152.0), rep(ntok4=110.0), rep(ntok4=150.0))
full["lane_b"] = lb(rep4_wait=79.0, rep5_wait=66.0)["lane_b"]
html = h.write_html(full)
for needle, why in (("b1b", "the third (bracketing) arm row"),
                    ("CONFIRMED", "the lane A verdict"),
                    ("concentrated", "the lane B residue verdict"),
                    ("recovered", "the lane B recovery verdict"),
                    ("thrash_slab2000", "the thrash request row"),
                    ("after-thrash wait", "the per-layer residue table"),
                    ("engine digests", "the provenance line")):
    if needle not in html:
        fails.append(f"populated HTML is missing {why} ({needle!r})")
print("  length:", len(html), " closed:", html.rstrip().endswith("</html>"))
if not html.rstrip().endswith("</html>"):
    fails.append("populated HTML is not closed")

# Provenance must resolve, otherwise the report cannot say which build produced it.
print("\n=== provenance ===")
dg = h.digest()
print(" ", dg)
for want in ("llama-server", "libllama.0.dylib", "libggml-metal.0.dylib"):
    if want not in dg:
        fails.append(f"digest() did not resolve {want}")

# An arm measured on a build the oracle gate never passed must not be reported as data: another
# session can rebuild libllama under us, and the digest would then name a build no gate covers.
print("\n=== anchor guard ===")
ok, why = h.anchor_ok(dg)
print("  on-disk build:", ok, "|", why)
if not ok:
    fails.append(f"the on-disk build should match the anchor, got: {why}")
ok, why = h.anchor_ok({**dg, "libllama.0.dylib": "deadbeefdeadbeef"})
print("  rebuilt libllama:", ok, "|", why)
if ok:
    fails.append("a changed libllama digest must fail the anchor check")
ok, why = h.anchor_ok({k: v for k, v in dg.items() if k != "llama-server"})
print("  unresolved artefact:", ok, "|", why)
if ok:
    fails.append("an unresolvable artefact must fail the anchor check")
import os as _os
_os.environ["PROBE_ANCHOR"] = "none"
ok, _ = h.anchor_ok({"libllama.0.dylib": "deadbeefdeadbeef"})
del _os.environ["PROBE_ANCHOR"]
print("  PROBE_ANCHOR=none bypass:", ok)
if not ok:
    fails.append("PROBE_ANCHOR=none must bypass the anchor check")

# 4d. no ntok=1 plain steps (MTP on) -> the control is UNAVAILABLE, said as a design gap. Real data
#     looks like this: every base step is a 4-wide verify batch, so there is nothing to compare the
#     verify step against.
def rep_no_plain(ntok4=150.0, mpt=100.0):
    return {"ms_per_token": mpt, "tokens": 92,
            "steps": {"4": {"total": ntok4, "wait": ntok4 - 30, "cb": 30.0, "n": 30}}, "layers": {}}


k = show("no plain step to control with (expected DESIGN GAP)",
         lane_a(rep_no_plain(ntok4=152.0), rep_no_plain(ntok4=140.0), rep_no_plain(ntok4=150.0)))
expect("no control", k, ["DESIGN GAP", "CAVEAT"], ["CONFIRMED"])
kv = h.verdict_lane_a(lane_a(rep_no_plain(ntok4=152.0), rep_no_plain(ntok4=140.0),
                             rep_no_plain(ntok4=150.0)))["verdicts"]
if not any("UNAVAILABLE, not as a pass" in t for _, t in kv):
    fails.append("a missing control must be labelled unavailable, not silently skipped")
if any("missing data" in t for _, t in kv):
    fails.append("a shape that does not exist is a design gap, not missing data")

# 4e. no control AND the test moved -> must not be CONFIRMED (a global change would look identical)
k = show("no control + test moved (expected INCONCLUSIVE, not CONFIRMED)",
         lane_a(rep_no_plain(ntok4=152.0), rep_no_plain(ntok4=90.0), rep_no_plain(ntok4=150.0)))
expect("no control moved", k, ["INCONCLUSIVE"], ["CONFIRMED", "REFUTED"])

# ... and the verdict must refuse to report numbers from such an arm.
mismatched = lane_a()
mismatched["lane_a"]["b0"] = {"status": "anchor-mismatch",
                              "anchor": "engine differs from the M1/M2/M3 anchor: {...}"}
bv = h.verdict_lane_a(mismatched)
kinds = [k for k, _ in bv["verdicts"]]
print("  lane A verdict on a mismatched build:", kinds)
if kinds != ["INCOMPLETE"]:
    fails.append(f"a mismatched build must yield only INCOMPLETE, got {kinds}")
if not bv["verdicts"][0][1].startswith("lane A was measured on a build"):
    fails.append("the mismatch verdict should name the cause, not say 'has not finished'")
bv = h.verdict_lane_b({"lane_b": {"status": "anchor-mismatch", "anchor": "x != y"}})
kinds = [k for k, _ in bv["verdicts"]]
if kinds != ["INCOMPLETE"]:
    fails.append(f"lane B on a mismatched build must yield only INCOMPLETE, got {kinds}")

print("\n=== the report keeps itself current, not only at the end of a run ===")
import tempfile
_tip = tempfile.mkdtemp()
# Redirect BOTH outputs: the real harness may be running against the default STATE, and a self-test
# that overwrote a live run's state would be the same class of defect as everything else here.
h.STATE = os.path.join(_tip, "s.json")
h.REPORT_OUT = os.path.join(_tip, "r.html")
h.LOG_LINES.append("[test] waiting: a reason that must be visible in the report")
h.save({"lane_a": {}, "lane_b": {}})
_rp = os.path.join(_tip, "r.html")
_ok = os.path.exists(_rp)
_text = open(_rp).read() if _ok else ""
print("  a state change wrote a report:", _ok,
      "| it carries the waiting reason:", "a reason that must be visible" in _text)
if not _ok:
    fails.append("save() must refresh the report")
if "a reason that must be visible" not in _text:
    fails.append("the report must carry the log lines, or 'why is there no output' stays unanswerable")
if "INCOMPLETE" not in _text:
    fails.append("a report with no measurement must say INCOMPLETE")
if h.report_path() != _rp:
    fails.append(f"report_path() must follow REPORT_OUT; got {h.report_path()}")
print("  report_path() follows REPORT_OUT:", h.report_path() == _rp)
print("  (this test never writes the default docs/ path)")

print("\n=== launch() must NOT put run_server.sh into dump-only mode ===")
# CGC_DUMP_ENV=1 makes run_server.sh print the resolved env/argv and exit 0 WITHOUT starting a
# server (run_server.sh:762). The first harness set it, so every launch "failed" in seconds and the
# symptom looked exactly like a parallel session killing the launcher. It was never killed.
_d = tempfile.mkdtemp()
os.makedirs(os.path.join(_d, "scripts"), exist_ok=True)
_fake = os.path.join(_d, "scripts", "run_server.sh")
open(_fake, "w").write('#!/bin/bash\necho "DUMP=$CGC_DUMP_ENV"\nexit 0\n')
os.chmod(_fake, 0o755)
h.ROOT = _d
h.LAUNCHER_LOG = os.path.join(_d, "launcher.log")
os.environ["CGC_DUMP_ENV"] = "1"          # as if the caller's shell exported it
_t0 = time.time()
h.launch({})
os.environ.pop("CGC_DUMP_ENV", None)
_said = open(h.LAUNCHER_LOG).read().strip()
print(f"  launcher saw: {_said!r}  (bailed in {time.time() - _t0:.1f}s, not 26 min)")
if "DUMP=0" not in _said:
    fails.append("launch() must force CGC_DUMP_ENV=0, or run_server.sh never starts a server")
if time.time() - _t0 > 30:
    fails.append("a launcher that exits immediately must not burn the long timeouts")

print("\n=== drift lane: does interleaving give item 1 resolution? ===")
# Two ARMS THAT ARE IDENTICAL BY CONSTRUCTION must agree within 5%, or no effect measured against
# them means anything. The real first run's baselines differed by 17.5%.

def ld(pairs):
    """pairs = [(b1_step, b1b_step), ...] one tuple per interleaved block; b0 sits between them."""
    runs = []
    for block, (a, c) in enumerate(pairs):
        for name, bi, step in (("b1", "1", a), ("b0", "0", a * 0.9), ("b1b", "1", c)):
            runs.append({"name": name, "bitident": bi, "block": block, "index": len(runs),
                         "reversed": bool(block % 2), "ts": "00:00:00", "free_mb": 9000,
                         "status": "done",
                         "reps": {"rep2_warm": {"ms_per_token": step, "tokens": 92,
                                  "steps": {"4": {"total": step, "cb": 10.0,
                                                   "wait": step - 15, "n": 39}},
                                  "layers": {}}}})
    return {"lane_drift": runs}


dv = h.verdict_drift(ld([(150.0, 150.0), (152.0, 151.0), (148.0, 149.0)]))
kinds = [k for k, _ in dv["verdicts"]]
print("\n--- drift lane: tight pairs (expected controls-drift + resolved)")
for k, t in dv["verdicts"]:
    print(f"      [{k}] {t[:110]}")
expect("drift tight", kinds, ["interleaving controls drift", "resolved"],
       ["interleaving insufficient", "NOT SUPPORTED"])
if len(dv["blocks"]) != 3:
    fails.append(f"drift lane should pair 3 blocks, got {len(dv['blocks'])}")

# The real failure mode: identical arms bouncing by +-20% between blocks. The MEAN ratio can still
# look fine (the bias averages out) -- the spread is what says the comparison is unusable.
dv = h.verdict_drift(ld([(166.0, 139.0), (140.0, 168.0), (150.0, 150.0)]))
kinds = [k for k, _ in dv["verdicts"]]
print("\n--- drift lane: wide pairs (expected insufficient + NOT SUPPORTED)")
for k, t in dv["verdicts"]:
    print(f"      [{k}] {t[:110]}")
expect("drift wide", kinds, ["interleaving insufficient", "NOT SUPPORTED"], ["resolved"])

# One pair is not a spread.
dv = h.verdict_drift(ld([(150.0, 150.0)]))
kinds = [k for k, _ in dv["verdicts"]]
print(f"\n--- drift lane: a single block -> {kinds}")
expect("drift single", kinds, ["INCOMPLETE"], ["resolved"])

dv = h.verdict_drift({"lane_drift": [], "lane_drift_error": "launch failed repeatedly"})
kinds = [k for k, _ in dv["verdicts"]]
print(f"--- drift lane: stopped early -> {kinds}")
expect("drift stopped", kinds, ["INCOMPLETE"])
if not any("launch failed repeatedly" in t for _, t in dv["verdicts"]):
    fails.append("a stopped drift lane must carry its own reason, not just say 'no arms'")

print("\n=== foreign-process classification (a wrapper shell is not a llama) ===")
# The first version matched any argv token, so a parallel agent's polling shell
# `zsh -c '... pgrep -f "build/bin/llama-bench" ...'` held the guard shut permanently. Classification
# must use the executable, while still catching the launcher's `env CGC_... /path/llama-server` form.
_cases = [
    ("/bin/zsh -c if [ -n x ]; then ...; eval 'sleep 150; pgrep -f \"build/bin/llama-bench\"'",
     "zsh", "a polling shell that only mentions the name"),
    ("/bin/bash -c cd /repo && ./bin/llama-cli -m x", "bash", "another agent's wrapper"),
    ("/Users/x/repo/src/llama.cpp/build/bin/llama-bench -m m.gguf -p 512",
     "llama-bench", "a real bench"),
    ("env CGC_SERVER_MTP=1 CGC_X=2 /repo/build/bin/llama-server -m m.gguf",
     "llama-server", "the launcher's env-wrapped server"),
]
for _args, _want, _why in _cases:
    _got = h.argv_exe(_args)
    _is_llama = _got in h.LLAMA_NAMES
    print(f"  {_why:34s} -> exe={_got!r} llama={_is_llama}")
    if _got != _want:
        fails.append(f"argv_exe({_why!r}) = {_got!r}, expected {_want!r}")
    _should = _want in ("llama-bench", "llama-server")
    if _is_llama is not _should:
        fails.append(f"classification of {_why!r} should be llama={_should}")

print("\n=== kernel-family eligibility: is item 1's premise true for THESE weights? ===")
# Source-level, not timing: if the expert types are on neither small-batch list, CGC_MM_BITIDENT is
# inert for exactly the tensors item 1 measures. This gate has to be able to say both yes and no.
_open = h.gguf_tensors
_tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
_tmp.write(b"GGUF")
_tmp.close()


def fam_state(rows, path=_tmp.name):
    h.gguf_tensors = lambda p: rows
    try:
        return h.verdict_family({"model_path": path})
    finally:
        h.gguf_tensors = _open


def rows_of(spec):
    """spec = [(name, type, ne00, MiB)] -> the 4-tuple shape the GGUF reader returns, spaced in file."""
    out, off = [], 0
    for name, tname, ne00, mib in spec:
        out.append((name, tname, ne00, off))
        off += int(mib * 2**20)
    out.append(("<eof>", "<eof>", 0, off))
    return out


# The measured model, in its real proportions: 123 expert tensors whose dominant types
# (IQ2_S / IQ3_S / IQ4_XS) are admitted by NEITHER list, plus 3 that are (Q2_K x2, Q3_K x1).
_spec = ([("blk.%d.ffn_gate_exps.weight" % i, "IQ2_S", 2048, 40) for i in range(78)]
         + [("blk.%d.ffn_up_exps.weight" % i, "IQ3_S", 2048, 40) for i in range(39)]
         + [("blk.%d.ffn_down_exps.weight" % i, "IQ4_XS", 512, 40) for i in range(3)]
         + [("blk.0.ffn_gate_exps.weight", "Q2_K", 2048, 40),
            ("blk.1.ffn_gate_exps.weight", "Q2_K", 2048, 40),
            ("blk.2.ffn_gate_exps.weight", "Q3_K", 2048, 40),
            ("blk.0.attn_q.weight", "Q8_0", 2048, 2)])
_vf = fam_state(rows_of(_spec))
print("--- real proportions (expected premise bounded, ~2.4% of expert bytes)")
for k, t in _vf["verdicts"]:
    print(f"      [{k}] {t[:140]}")
expect("family bounded", [k for k, _ in _vf["verdicts"]], ["premise bounded"], ["premise void"])
if abs(_vf["eligible_share"] - 3 / 123) > 0.005:
    fails.append(f"expert byte share should be ~2.4% (3 of 123), got {_vf['eligible_share']:.3f}")
_q8 = [r for r in _vf["rows"] if r["type"] == "Q8_0"][0]
if _q8["eligible"] != 1 or _q8["expert"] != 0:
    fails.append("a non-expert Q8_0 tensor must show eligible=1 while expert count stays 0")

# Experts with no tensor on either list at all -> the premise is void, not merely small.
_vf = fam_state(rows_of([("blk.%d.ffn_gate_exps.weight" % i, "IQ2_S", 2048, 40) for i in range(9)]))
expect("family void", [k for k, _ in _vf["verdicts"]], ["premise void"], ["premise bounded"])

# ...while experts that ARE Q4_K make it bounded by 100%: the gate has to be able to say that too.
_vf = fam_state(rows_of([("blk.%d.ffn_gate_exps.weight" % i, "Q4_K", 2048, 40) for i in range(20)]))
print("--- experts that ARE Q4_K (expected premise bounded, 100%)")
for k, t in _vf["verdicts"]:
    print(f"      [{k}] {t[:140]}")
expect("family all eligible", [k for k, _ in _vf["verdicts"]], ["premise bounded"], ["premise void"])
if _vf["eligible_share"] < 0.99:
    fails.append(f"all-Q4_K experts should be ~100% eligible, got {_vf['eligible_share']:.3f}")

# ne00 % 128 != 0 is part of the guard, so a listed type with a bad row length is still ineligible.
_vf = fam_state(rows_of([("blk.0.ffn_gate_exps.weight", "Q4_K", 100, 40)]))
expect("family ne00 guard", [k for k, _ in _vf["verdicts"]], ["premise void"], ["premise bounded"])

_vf = h.verdict_family({})
expect("family no model", [k for k, _ in _vf["verdicts"]], ["INCOMPLETE"], ["premise void"])

# The two instruments cross-checked. Neither the ceiling nor the A/B can reach this conclusion alone:
# a 2.4% ceiling with a 15% measured effect means the measurement is not the mechanism.
def fam_cross(rows, b1, b0):
    st = {"model_path": _tmp.name, "lane_a": {}}
    for nm, v in (("b1", b1), ("b0", b0)):
        st["lane_a"][nm] = {"reps": {"rep2_warm": {"steps": {"4": {"total": v}}}}}
    h.gguf_tensors = lambda p: rows
    try:
        return h.verdict_family(st)
    finally:
        h.gguf_tensors = _open


def _real_rows():
    return rows_of(_spec)


_vf = fam_cross(_real_rows(), b1=166.41, b0=140.55)
print("--- 2.4% ceiling vs a 15% measured effect (expected not attributable)")
for k, t in _vf["verdicts"]:
    print(f"      [{k}] {t[:140]}")
expect("family cross excludes", [k for k, _ in _vf["verdicts"]], ["not attributable"], [])

_vf = fam_cross(rows_of([("blk.%d.ffn_gate_exps.weight" % i, "Q4_K", 2048, 40)
                        for i in range(20)]), b1=166.41, b0=140.55)
print("--- 100% ceiling vs the same effect (expected NO not-attributable)")
for k, t in _vf["verdicts"]:
    print(f"      [{k}] {t[:140]}")
expect("family cross allows", [k for k, _ in _vf["verdicts"]], ["premise bounded"],
       ["not attributable"])

print("\n=== drift lane: a block that flips inside itself ===")
# The measured case: 337.2 then 133.6 in ONE block with cb identical. Presenting this as "not enough
# blocks" would be the wrong next action; the shape itself is the problem.
_dv = h.verdict_drift(ld([(337.2, 133.6), (340.0, 135.0)]))
_kinds = [k for k, _ in _dv["verdicts"]]
for k, t in _dv["verdicts"]:
    print(f"      [{k}] {t[:120]}")
expect("drift flip", _kinds, ["regime faster than one arm", "not settled by this shape"], ["resolved"])

print("\n" + ("ALL VERDICT TESTS PASSED" if not fails else "FAILURES:\n  " + "\n  ".join(fails)))
sys.exit(1 if fails else 0)
