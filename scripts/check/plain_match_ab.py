#!/usr/bin/env python3
"""plain_match: does the batch-verify path compute the SAME thing as per-token decode?

The question
------------
`plain_match = False` has been recorded since `MTP_HEAD_PROVENANCE_GATE_2026-09-14.md` §5 and never
resolved. It matters twice over: it is a CORRECTNESS claim (the batch-verify logits differ from the
per-token ones, which is the same class of defect as the token>=1 wrong-expert read already fixed
once) and it is the top of `MTP_2X_BOUNDARY_2026-09-19.md` §3 -- if divergence explains >=15pp of the
rejection, then accept 0.465 is not a property of the head and there is a free path to 0.6+.

The test, and why greedy is the right instrument
------------------------------------------------
Same server binary, same checkpoint bytes, same prompt, temperature 0 (the server's own
CGC_FORCE_TEMP0=1 pins temperature AND seed, server-common.cpp:1356), once with MTP ON and once with
MTP OFF. Under greedy decoding the target's argmax logits ARE the output, so:

  * identical token ids  -> the verify path computes the same target distribution; accept 0.465 is
    about the DRAFT, not a verify defect;
  * differing ids at position p -> the verify path's logits differ from per-token decode's from p
    onward, i.e. a real bug, and the divergence rate is its size.

Determinism is measured, not assumed: each arm is run twice and the two identical-arm runs must
agree, otherwise "the arms differ" is unfalsifiable (two runs of the same thing differ too).

The MTP=0 carrier: `run_server.sh:83,154-160` selects $Q36 for MTP=0, and $Q36 is a symlink to the
denseIQ4X MTP carrier (established 2026-09-19). This script re-checks it per arm by realpath+size, so
a future edit that splits them stops the comparison instead of silently measuring two models.

The control arm (P0-1, 2026-09-19)
---------------------------------
An MTP on/off comparison is only a comparison if the arms differ in ONE thing. Two knobs in
run_server.sh are engine-wide -- a matmul KERNEL choice and a background-prefetch policy -- but used
to be exported only inside `if [ "$SERVER_MTP" = "1" ]`, so the MTP=0 arm ran a different kernel and
any output difference had two candidate causes. run_server.sh now hoists those two so a caller can
pass them to BOTH arms; every arm here carries EQUALISE below, and the record keeps the resolved
`layer_caps=` and `pool capacity=` per arm so the geometry claim is checkable rather than asserted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/Users/alexchuang/Documents/flashkv-devserver")
sys.path.insert(0, str(ROOT / "scripts" / "check"))
import thermal_pressure as thermal  # noqa: E402
import server_window as SW  # noqa: E402  (the ONE box probe: scripts/check/server_window.py)

PROMPT = ("用條列方式說明快取置換策略的取捨，並比較 LRU 與成本感知淘汰在長序列推論下的差異。"
          "請具體舉例，不要只列名詞。")


def digest(path: Path):
    p = Path(path)
    if not p.exists():
        return None
    st = p.stat()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(65536))
        if st.st_size > 131072:
            f.seek(-65536, 2)
            h.update(f.read(65536))
    return {"size": st.st_size, "head_tail_hash": h.hexdigest()[:16]}


def engine_digest() -> dict:
    """md5 of the linked engine artifacts, recorded with every run.

    Why this is not optional here: the first version of this experiment produced PLAIN_MATCH TRUE at
    02:57 and the same arms produce FALSE at 04:0x, and the ONLY difference found between the two
    settings is that libllama was rebuilt in between (mtime 03:39:45, i.e. after the first run and
    before the second). Every environment variable, argument, prompt and geometry line is identical.
    Without a digest per run that flip is unattributable; with one it is a fact about build identity
    rather than a mystery.
    """
    out = {}
    for name in ("libllama.0.0.279.dylib", "libggml-metal.0.19.0.dylib", "libggml-base.0.19.0.dylib",
                 "llama-server"):
        p = ROOT / "src" / "llama.cpp" / "build" / "bin" / name
        if not p.exists():
            continue
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[name] = {"md5": h.hexdigest()[:16], "mtime": int(p.stat().st_mtime)}
    return out


def newest_log(after: float) -> Path | None:
    logs = sorted((ROOT / "Backup" / "cgc_logs").glob("llama_server_*.log"),
                  key=lambda p: p.stat().st_mtime)
    for p in reversed(logs):
        if p.stat().st_mtime >= after:
            return p
    return None


# [CGC 2026-09-19 P0-1] What must be equal across arms, and what may differ.
#
# EQUALISE is applied to BOTH arms. Each entry is a knob the engine reads through getenv() at a
# point that has nothing to do with a verify/draft batch, so leaving it on one arm only would make a
# difference unattributable:
#   CGC_MM_BITIDENT=1                -> ggml-metal-ops.cpp:2470, chooses the mul_mat kernel for M<=8
#                                       (decode's GEMV is M=1, inside that range). bit-identical pillar 1.
#   CGC_SERVER_NO_PREFETCH=1         -> llama-context.cpp:2058, background slot prefetch, plain decode.
#   CGC_SERVER_LAYER_CAPS=40-40:256  -> pool geometry: without it MTP=1 gets 40-40:256 and MTP=0 gets
#                                       nothing, i.e. the arms hold different per-layer slot budgets.
#
# NOT equalised, and INERT by construction: CGC_VERIFY_DECODE / CGC_DRAFT_DECODE / CGC_WARM_NPAST.
# They only gate the ZERO-slot fast path, which llama-context.cpp:6154-6161 forces off unless the
# phase marker is VERIFY or DRAFT -- and only a speculative batch ever sets those markers.
#
# The rejected route, kept so nobody re-discovers it: CGC_SERVER_MTP=1 + CGC_SERVER_MTP_N_MAX=0 (the
# k=0 control that needs no hoist and shares the launcher branch) ABORTS the server. The MTP draft
# impl emits ONE draft token even at n_max=0 because the clamp is checked AFTER the push
# (common/speculative.cpp:1795 push, :1820 clamp), while the server reserves 1+n_max = 1 output rows
# (common_speculative_get_output_limits, common/speculative.cpp:2597). The resulting 2-token verify
# batch trips GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) (llama-context.cpp:2961) and the
# process dies with `Abort trap: 6` -- observed 2026-09-19 03:49,
# Backup/cgc_logs/llama_server_20260919_034940.log. That is an engine defect in its own right and it
# is NOT fixed here (it needs a rebuild of libllama-common, which this box shares with other sessions).
EQUALISE = {
    "CGC_MM_BITIDENT": "1",
    "CGC_SERVER_NO_PREFETCH": "1",
    "CGC_SERVER_LAYER_CAPS": "40-40:256",
}

ARM_RECIPES = {
    # launcher-branch identity -> the env that defines the arm
    "1":   {"CGC_SERVER_MTP": "1", **EQUALISE},
    "on":  {"CGC_SERVER_MTP": "1", **EQUALISE},
    "0":   {"CGC_SERVER_MTP": "0", **EQUALISE},
    "off": {"CGC_SERVER_MTP": "0", **EQUALISE},
    # [2026-09-19 attribution arm] Same as `on` but with the VERIFY/DRAFT ZERO-slot fast path OFF, so
    # the verify batch runs the SAME exact ensure_batch path a plain decode step runs. This is the
    # arm that decides what carries a plain_match divergence: if `on` != `off` but `on-exact` ==
    # `off`, the carrier is the fast path (a documented approximation, and the one the M1/M2 oracle
    # gate checks for); if `on-exact` still diverges, the carrier is the batch math itself and the
    # divergence is not explainable by the fast path.
    "on-exact": {"CGC_SERVER_MTP": "1", **EQUALISE,
                 "CGC_SERVER_VERIFY_DECODE": "0", "CGC_SERVER_DRAFT_DECODE": "0"},
    # [2026-09-19 bisect arms] The MTP-off arms exist to answer a question the first equalised run
    # raised: the MTP-off arm's text CHANGED when EQUALISE was applied (262 chars and identical to
    # the MTP-on arm in the legacy config; 278 chars and divergent now, same prompt, same greedy).
    # So one of the three equalisers -- or the binary -- moves plain-decode values. These four arms
    # isolate it, one knob at a time, against `off-plain` (exactly what run_server.sh's MTP=0 default
    # exports: nothing):
    "off-plain": {"CGC_SERVER_MTP": "0"},
    "off-bitid": {"CGC_SERVER_MTP": "0", "CGC_MM_BITIDENT": "1"},
    "off-nopf":  {"CGC_SERVER_MTP": "0", "CGC_SERVER_NO_PREFETCH": "1"},
    "off-caps":  {"CGC_SERVER_MTP": "0", "CGC_SERVER_LAYER_CAPS": "40-40:256"},
}
ON_ARMS = {"1", "on"}
K0_ARMS = {"0", "off"}   # name kept from the k=0 design; it is the MTP-off control arm
ARM_ARMS = {k: ("on" if k in ON_ARMS else "off" if k in K0_ARMS else k) for k in ARM_RECIPES}

# Refuse the aborting configuration up front instead of measuring a server that dies mid-request.
if os.environ.get("CGC_SERVER_MTP_N_MAX") is not None:
    raise SystemExit(
        "refusing to run: CGC_SERVER_MTP_N_MAX is set. With --spec-type draft-mtp the server reserves\n"
        "1+n_max outputs while the MTP draft impl emits one draft token even at n_max=0, so n_max=0\n"
        "aborts at llama-context.cpp:2961 and n_max>=1 makes this a different arm than intended.\n"
        "Equalise the arms with EQUALISE instead (see the block above).")


def launch(mtp: str, port: int, pool_gb: float, args):
    # Ask the shared probe before loading 13 GB + an 8 GiB pool. Gated on the FIRST launch of this
    # process only: re-checking before arm 2 would fail on the page cache arm 1 just filled, i.e.
    # over-block the very multi-arm runs this guard exists to make attributable. Refuses by default;
    # CGC_WINDOW_OVERRIDE=1 proceeds but records the condition with the run instead of hiding it.
    SW.require_first(port=port, need_mb=float(os.environ.get("CGC_WINDOW_NEED_MB", "8000")),
                     where=f"plain_match arm {mtp}")
    env = dict(os.environ)
    recipe = ARM_RECIPES.get(mtp)
    if recipe is None:
        raise SystemExit(f"unknown arm {mtp!r}: use 1/on (MTP on) or 0/off (MTP off). Both carry "
                         f"EQUALISE, so the arms differ only in whether a speculative batch exists.")
    env.update({
        **recipe,
        "CGC_FORCE_TEMP0": "1",                     # pins temperature AND seed to greedy server-side
        "CGC_SERVER_PORT": str(port),
        "CGC_SERVER_EXPERT_CACHE_BYTES": str(int(pool_gb * 1024 ** 3)),
        "CGC_PREFLIGHT_KILL": "0",                  # do not hunt other sessions' processes
        # prod25 = the production SERVING profile (the one the recorded accept 0.465 / 12.99 t/s
        # numbers came from). CGC_PREFILL_STREAM is required by the profile's own shape: without it
        # a wide prompt aborts with GPU OOM (llama_bench_matrix's `prod25` note).
        "CGC_SERVER_PROFILE": args.profile,
        "CGC_PREFILL_STREAM": "1",
    })
    t0 = time.time()
    log = ROOT / "Backup" / "plain_match_launch.log"
    f = open(log, "ab")
    p = subprocess.Popen(["bash", str(ROOT / "scripts" / "run_server.sh")],
                         env=env, cwd=str(ROOT), stdout=f, stderr=f)
    return p, log, t0


def wait_health(port: int, proc, timeout=420) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def decode(port: int, n_predict: int, seed: int):
    # [2026-09-19] `return_tokens` was believed unsupported here: `variants_nonempty()` saw empty
    # `tokens` arrays and the comparison fell back to rendered text. It IS supported --
    # `tools/server/server-schema.cpp:34` registers the field and `server-context.cpp:2004` fills
    # `slot.generated_tokens` when it is set. The client simply never asked, so this driver was
    # comparing characters (where one differing token shifts every later position) instead of ids.
    # Asking for the ids is a strict improvement: the divergence is then measured in the same unit
    # the claim is about.
    body = json.dumps({
        "prompt": PROMPT,
        "n_predict": n_predict,
        "temperature": 0,
        "top_k": 1,
        "seed": seed,
        "cache_prompt": False,
        "stream": False,
        "return_tokens": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def variants_nonempty(ids) -> bool:
    """True only when every rep returned a plausible token array (>= 8 ids).

    The threshold is not cosmetic: a server that returns `tokens: []` must NOT be read as "the two
    arms agreed", and one that returned a single id could not support a 128-token claim either.
    """
    return bool(ids) and all(isinstance(x, list) and len(x) >= 8 for x in ids)


def run_arm(mtp: str, args) -> dict:
    proc, launch_log, t0 = launch(mtp, args.port, args.pool_gb, args)
    try:
        if not wait_health(args.port, proc):
            return {"mtp": mtp, "error": "server never became healthy"}
        slog = newest_log(t0)
        model = None
        if slog is not None:
            txt = slog.read_text(errors="replace")
            m = re.search(r"model path\s*=\s*(\S+)", txt) or re.search(r"llama_model_loader: - .*", txt)
            if m:
                model = m.group(1)
        sampler = thermal.Sampler()
        with sampler:
            outs = [decode(args.port, args.n_predict, 0) for _ in range(args.reps)]
        ids = [o.get("tokens") or [] for o in outs]
        texts = [o.get("content") or "" for o in outs]
        rec = {
            "mtp": mtp,
            "engine_digest": engine_digest(),
            "arm": ARM_ARMS.get(mtp, mtp),
            "recipe": mtp,
            "arm_env": ARM_RECIPES.get(mtp, {}),
            "reps": args.reps,
            "n_predict": args.n_predict,
            "ids": ids,
            "texts": texts,
            "within_arm_identical": all(ids[0] == x for x in ids) and all(texts[0] == t for t in texts),
            "n_tokens": [len(x) for x in ids],
            # The id array is the STRONGER evidence, but this server build does not return it
            # (measured 2026-09-19: `tokens` absent, ids all empty, so the first version of this
            # driver compared two empty lists, found them equal and reported PLAIN_MATCH TRUE on
            # ZERO data -- a green light from an empty array, the exact defect class this repo
            # polices). The comparison therefore falls back to the rendered text and says so.
            "compared_by": "ids" if variants_nonempty(ids) else "text",
            "server_log": str(slog) if slog else None,
            "thermal": sampler.result,
            "timings": [{k: v for k, v in (o.get("timings") or {}).items()
                         if k in ("prompt_n", "predicted_n", "prompt_ms", "predicted_ms",
                                  "draft_n", "draft_n_accepted")} for o in outs],
        }
        # the carrier identity check: MTP=0 must serve the SAME bytes, or this is not a comparison
        # Carrier identity: WHICH bytes this arm served. `run_server.sh:83,154-160` picks $Q36 for
        # MTP=0 and $Q36_MTP for MTP=1, and the whole comparison is void if those are not the same
        # file. Measured 2026-09-19: $Q36 is a symlink to the denseIQ4X MTP carrier, so both
        # realpaths, sizes and head/tail hashes match -- but that is checked here, per arm, not
        # assumed. The path is taken from the launcher's own log (it prints one line per launch),
        # not from a regex over a server log that no longer echoes the argv.
        if slog is not None:
            txt = slog.read_text(errors="replace")
            m = re.search(r"^.*?(-m|--model)\s+(\S+\.gguf)", txt, re.M)
            if m:
                p = Path(m.group(2))
                rec["carrier"] = {"path": m.group(2), "realpath": str(p.resolve()),
                                  **(digest(p) or {})}
        if "carrier" not in rec and launch_log.exists():
            m = re.findall(r"(/\S+\.gguf)", launch_log.read_text(errors="replace"))
            if m:
                p = Path(m[-1])
                rec["carrier"] = {"path": str(p), "realpath": str(p.resolve()), **(digest(p) or {})}
        # Geometry equality evidence (P0-1): the resolved layer caps (launcher banner) and the pool
        # capacity the engine actually built (server log) are recorded per arm, so "both arms ran the
        # same geometry" is read off the artifacts instead of asserted.
        if launch_log.exists():
            caps = re.findall(r"layer_caps=(\S+)", launch_log.read_text(errors="replace"))
            if caps:
                rec["layer_caps"] = caps[-1]
        if slog is not None:
            m = re.search(r"CGC-PHASE-SPLIT: L4 pool capacity=(\d+)", slog.read_text(errors="replace"))
            if m:
                rec["pool_capacity"] = int(m.group(1))
        return rec
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except Exception:
            proc.kill()
        time.sleep(4)
        # [CGC 2026-09-19 P0-2] NO pattern kill here. `pkill -f build/bin/llama-server` is
        # pid-blind: on this shared box it hunts every parallel session's server, and an earlier
        # find of this bug (http_duo.py:31,285) is already recorded in this repo. proc is owned by
        # this driver and terminate()+wait()+kill() is the complete teardown; the pattern kill was
        # belt-and-braces that could only ever hit somebody else's process.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--pool-gb", type=float, default=8.0)
    ap.add_argument("--profile", default="prod25")
    ap.add_argument("--order", default="1,0,1,0",
                    help="interleaved arms; each entry is one server start. '1'/'on' = MTP on, "
                         "'0'/'off' = MTP off. BOTH carry EQUALISE (kernel choice, prefetch policy, "
                         "layer caps), so the only difference left is the speculative batch; "
                         "ARM_RECIPES documents what is equalised and what is inert.")
    # Artifacts land in Backup/ (gitignored, survives /tmp reaping): the 09-18 matrix lost 36 runs
    # because its outputs lived in /tmp only (POOL_BUDGET_COST_DECOMP §7-3).
    ap.add_argument("--json", default=str(ROOT / "Backup" / "phase_decomp" / "plain_match.json"))
    args = ap.parse_args()

    arms = [a.strip() for a in args.order.split(",") if a.strip()]
    runs = []
    for mtp in arms:
        print(f"--- arm {mtp} env={ARM_RECIPES.get(mtp)} ---", flush=True)
        r = run_arm(mtp, args)
        print(f"    tokens={r.get('n_tokens')} within-arm identical={r.get('within_arm_identical')} "
              f"draft={[t.get('draft_n_accepted') for t in r.get('timings', [])]}"
              f"/{[t.get('draft_n') for t in r.get('timings', [])]}"
              f"{'  ERROR: ' + r['error'] if r.get('error') else ''}", flush=True)
        runs.append(r)

    res = {"prompt": PROMPT, "args": vars(args), "runs": runs}
    for r in runs:
        r["text_len"] = len(r["texts"][0]) if r.get("texts") else 0
    on = [r for r in runs if r.get("arm") == "on" and r.get("ids")]
    off = [r for r in runs if r.get("arm") == "off" and r.get("ids")]
    on_exact = [r for r in runs if r.get("arm") == "on-exact" and r.get("ids")]

    def compare(x, y):
        """Compare two arms on ids when the build returns them, else on rendered text.

        The text fallback is weaker and is labelled as such: tokenization is not injective (two
        token sequences can render the same, which HIDES a difference), but the reverse cannot
        happen, so a text DIFFERENCE is always a real token difference.
        """
        by_ids = variants_nonempty(x["ids"]) and variants_nonempty(y["ids"])
        a = x["ids"][0] if by_ids else x["texts"][0]
        b = y["ids"][0] if by_ids else y["texts"][0]
        n = min(len(a), len(b))
        first = next((i for i in range(n) if a[i] != b[i]), None)
        div = sum(1 for i in range(n) if a[i] != b[i])
        return {"n_compared": n, "compared_by": "ids" if by_ids else "text",
                "len_a": len(a), "len_b": len(b), "first_divergence_pos": first,
                "diverging_positions": div,
                "divergence_pct": round(100.0 * div / n, 2) if n else None, "enough": n >= 8}

    if on and off:
        # Carrier identity is the BYTES, not the path: run_server.sh picks $Q36 for MTP=0 and $Q36_MTP
        # for MTP=1, and $Q36 is a symlink, so a JSON dump of the whole carrier record differs on
        # `path` alone and reported same_carrier=False for two arms serving the same file. Compare
        # the resolved identity instead (realpath + size + head/tail hash).
        def carrier_id(r):
            c = r.get("carrier") or {}
            return json.dumps({k: c.get(k) for k in ("realpath", "size", "head_tail_hash")},
                              sort_keys=True)
        carriers = {carrier_id(r) for r in on + off}
        c = compare(on[0], off[0])
        arm_det = all(r.get("within_arm_identical") for r in runs if r.get("texts"))
        # A verdict is only issued on >= 8 compared units, and never on an empty comparison.
        res["plain_match"] = {
            **c,
            "verdict": ("NO DATA: nothing comparable was returned, so nothing is claimed"
                        if not c["enough"] else
                        "PLAIN_MATCH TRUE: the two arms produced identical output, so the verify path computes "
                        "the same target distribution as per-token decode" if c["diverging_positions"] == 0 else
                        f"PLAIN_MATCH FALSE: {c['diverging_positions']}/{c['n_compared']} positions differ, "
                        f"first at {c['first_divergence_pos']} -- the verify path's logits/argmax differ from "
                        "per-token decode's from there on"),
            "within_arm_deterministic": arm_det,
            "same_carrier": len(carriers) == 1,
            "carriers": sorted(carriers),
            "accept": [{k: v for k, v in (t or {}).items() if k.startswith("draft_")}
                       for r in runs for t in (r.get("timings") or [])],
        }
        print("\n" + json.dumps(res["plain_match"], indent=2, ensure_ascii=False))

    # What carries the divergence: the verify batch with the ZERO-slot fast path ON vs the SAME batch
    # with it OFF (on-exact). Equal to `off` => the carrier is the fast path; still divergent => the
    # carrier is the batch math, and the fast path is not the explanation.
    if on and on_exact:
        c2 = compare(on[0], on_exact[0])
        res["fast_path_attribution"] = {
            **c2,
            # Reading, in the direction the comparison actually runs: `on` and `on-exact` are the
            # SAME arm except for the ZERO-slot fast path. Identical output => the fast path is
            # exonerated and the divergence from `off` lives in the verify batch itself. Different
            # output => the fast path is the carrier.
            "verdict": ("NO DATA" if not c2["enough"] else
                        "FAST PATH EXONERATED: turning the VERIFY/DRAFT ZERO-slot fast path off changes "
                        "nothing, so the divergence from the MTP-off arm is not the pool approximation "
                        "-- it lives in the batch/verify path itself" if c2["diverging_positions"] == 0 else
                        f"FAST PATH IS THE CARRIER: {c2['diverging_positions']}/{c2['n_compared']} positions "
                        f"differ with the fast path on vs off (first at {c2['first_divergence_pos']})"),
        }
        print("\n" + json.dumps(res["fast_path_attribution"], indent=2, ensure_ascii=False))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
