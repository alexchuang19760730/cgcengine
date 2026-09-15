#!/usr/bin/env python3
"""Anti-knife-edge evaluation matrix: (model) x (pool size).

WHY THIS EXISTS
---------------
Single greedy questions on this stack are NOT a measurement. Two runs of the same question
can land in different trajectories (answer vs echo/loop) because a ~0.02 logit offset —
which comes from which MoE path/layout a step took — is enough to flip a near-tie about ten
tokens in. Ranking pool sizes on one greedy question measures luck.

`flip_rate.py` already makes that uncertainty explicit (N greedy repeats + M fixed seeds +
per-question flip statistics + suite variance). What was missing is the **model dimension**:
the same harness must be runnable for iq3 and iq4 without hand-editing anything, because the
quality and the correct chat template differ per model.

This driver adds that, plus the operational hygiene the earlier round was missing:

  * EXCLUSIVITY - refuses to measure while another llama-server is running. Earlier pool
    numbers were taken while other servers were live, which corrupts both the memory and the
    speed readings. Use --kill-existing to clear them explicitly.
  * RESUME      - a combo whose result JSON already exists is skipped, so a long sweep
    survives an interruption instead of restarting from zero.
  * TEMPLATE    - launches with `enable_thinking=false` (see below) and records the
    `[chat] model_kind=` line the server actually chose, so a silent template mismatch is
    visible in the results rather than assumed away.
  * SPEED       - records RSS / free% / decode tok/s next to the pass rate, because the pool
    floor is a quality x speed trade, not a quality alone.

TEMPLATE NOTE (2026-09-11, verified): the GGUF embedded template gates its no-think scaffold
on `enable_thinking`:

    {%- if enable_thinking is defined and enable_thinking is false %}
        '<think>\n\n</think>\n\n'   <- closed   |   {%- else %} '<think>\n'  <- unclosed

If the launcher does not pass it, every generation prompt ends with an UNCLOSED marker and
the model degenerates into a deterministic prompt echo that never stops
(`finish=length`, no answer). Verified on a live IQ4_XS 8GB-pool server: content was the user
text repeated to the token cap.

`--chat-template-kwargs '{"enable_thinking": false}'` is therefore MANDATORY, and it must be
set at LAUNCH: this build does NOT read `chat_template_kwargs` from the request body -- three
request variants (none / enable_thinking=false / +assistant_prefill) came back byte-identical.
A client cannot repair this; see docs/CHAT_SCAFFOLD_ROOTCAUSE_2026-09-11.md §7.

`assistant_prefill` is MANDATORY TOO (2026-09-11) and is why the default kwargs carry both keys.
`run_server.sh` only applies its own anchor when chat-template-kwargs is EMPTY, so a harness that
passes `enable_thinking` alone silently DISABLES the anchor. On iq4 that produced
`completion_tokens: 1` with empty content at every pool size -- which reads like a pool-quality
bug and is not one.

Override with --kwargs '' only if a template genuinely does not need it.

Usage:
    # smoke test the harness end-to-end on both models, one question each
    python3 scripts/check/knifeedge_matrix.py --pilot

    # real sweep
    python3 scripts/check/knifeedge_matrix.py --models iq3,iq4 --pools 4,6,8 \\
        --per-profile 2 --greedy-repeats 3 --repeats 5 --temperature 0.4

    # add a config dimension to the A/B (repeatable)
    python3 scripts/check/knifeedge_matrix.py --models iq3 --pools 8 \\
        --extra-env CGC_LOOP_GUARD=1
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
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GGUF = os.path.join(ROOT, "models", "gguf")
SERVER_MATCH = "build/bin/llama-server"
RESULT_DIR = os.path.join(ROOT, "Backup", "knifeedge_matrix")
DEFAULT_PORT = 8080

# Per-model launch facts. `mtp` selects whether to ask run_server.sh for the MTP draft head;
# `path` (optional, absolute) overrides models/gguf/<file>.
#
# The template is chosen by CGC_SERVER_PROFILE in run_server.sh, NOT by model kind: with no
# profile and no explicit overrides, EVERY model gets the same Qwen3-nothink-ChatML.jinja. Any
# note here claiming "iq3 -> embedded template" was wrong; two entries can only ever differ by
# their weights. gate_model_identity() enforces that they actually do.
MODELS = {
    "iq3": {
        "file": "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
        "mtp": "1",
        "topk": 8,
        "note": "Nail IQ3_XXS denseIQ4X MTP carrier (13.6 GB). In this checkout the name is a "
                "symlink -- see gate_model_identity: it must not point at another entry's file.",
    },
    "iq3ext": {
        # The real 13 GB IQ3_XXS-denseIQ4X lives on the external drive; the internal disk has no
        # room for it next to the 18.2 GB IQ4_XS (10 GiB free, 13 GB needed). Read speed there is
        # a measured 86 MB/s, so a small pool can be filled from it at run time.
        "file": "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
        "path": "/Volumes/AlexZhuang/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
        "mtp": "1",
        "topk": 8,
        "note": "real IQ3_XXS-denseIQ4X MTP, run straight from the external drive",
    },
    "iq4": {
        "file": "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        "mtp": "1",
        "topk": 8,
        "note": "vanilla IQ4_XS (18.2 GB). NOT on disk in this checkout -- see the model-id line, "
                "which prints exists=False rather than silently comparing something else.",
    },
    "ornith": {
        # A different BASE, not a different quantisation of the same base: Ornith 1.5 35B is an
        # abliterated MLX-bf16 fine-tune, requantised here to a mixed q3_K/q4_K trunk plus a q8_0
        # MTP layer (123 expert tensors: q3_K=90, q4_K=30, q8_0=3).
        "file": "Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Compact-v2D-lite.gguf",
        "mtp": "1",
        "topk": 8,
        "note": "17.6 GB, arch qwen35moe, block_count 41 + nextn_predict_layers 1, 256 experts, "
                "top_k 8, n_ff_exp 512 -> 589,824 B/expert for the q4_K kinds (the same per-slot "
                "geometry as the Edge0 Q4_0 entry, so a given pool maps to a comparable slot "
                "count) and 450,560 B/expert for q3_K. It carries its own head (fingerprint "
                "2e35283d4a95c0bc, 21 tensors, 1372.7 MiB, types F32=9/Q8_0=12, router alive) -- "
                "no graft, so this is a THIRD independently-paired (base, head) rather than a "
                "second carrier for an existing pair.",
    },
}


def model_path(kind):
    """Absolute path for a configured model: explicit `path` wins, else models/gguf/<file>."""
    return MODELS[kind].get("path") or os.path.join(GGUF, MODELS[kind]["file"])

# `buffer is nil` -- the signature of the L3-B gather path repointing a Metal-resident FFN
# tensor at a HOST std::vector. One hit invalidates the combo AND explains numeric drift.
NIL_RE = re.compile(r"buffer is nil")


def model_identity(path):
    """Fingerprint a model file well enough to prove two entries are the SAME weights.

    realpath alone is not enough (a copy under another name defeats it), and size alone is not
    either (two quantisations of one base can land on nearby sizes). realpath + size + a hash of
    the GGUF header/first tensors is: a re-quantised file cannot share all three.
    """
    rp = os.path.realpath(path)
    rec = {"path": path, "realpath": rp, "exists": os.path.exists(rp)}
    if not rec["exists"]:
        return rec
    st = os.stat(rp)
    h = hashlib.sha256()
    with open(rp, "rb") as f:
        h.update(f.read(1 << 20))
    rec.update(size=st.st_size, head_sha256=h.hexdigest()[:16])
    return rec


def gate_model_identity(models):
    """Refuse to measure a 'model matrix' whose columns are secretly the same file.

    Measured failure this gate exists for (2026-09-11): `models/gguf/Nail-...-IQ3_XXS-denseIQ4X\
    .gguf` was a symlink re-pointed at `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (the real IQ3 GGUF was
    not on the internal disk), so the iq3 and iq4 columns loaded the SAME 18.2 GB file. The
    oracle dumps for the two 'models' came out byte-identical (md5 0040238f...), and every
    iq3-vs-iq4 number agreed exactly -- which reads as a beautiful result and was in fact no
    cross-model evidence at all. Same class of defect as the false-passing cap probe: the
    measurement cannot detect that it is not measuring what it claims.
    """
    ids = {m: model_identity(model_path(m)) for m in models}
    problems = []
    by_fp = {}
    for m, rec in ids.items():
        print(f"[model-id] {m}: realpath={rec['realpath']} exists={rec['exists']}"
              + (f" size={rec['size']}" if rec["exists"] else ""), flush=True)
        print(f"           name={MODELS[m]['file']}", flush=True)
        if not rec["exists"]:
            problems.append(f"{m}: file does not exist ({rec['realpath']})")
            continue
        fp = (rec["size"], rec["head_sha256"])
        if fp in by_fp:
            problems.append(
                f"{m} and {by_fp[fp]} resolve to the SAME weights "
                f"(realpath={rec['realpath']}, size={rec['size']}, head={rec['head_sha256']}). "
                f"A cross-model comparison would be two measurements of one file.")
        else:
            by_fp[fp] = m
    if problems:
        print("\nREFUSING to measure: model identity check failed.", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("  Fix: make each configured model name resolve to its own GGUF "
              "(a symlink is fine, but it must not point at another entry's file).", file=sys.stderr)
        return False
    return True


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def running_servers():
    """PIDs of REAL llama-server processes (not bash wrappers that merely mention it).

    `pgrep -f` also matches the desktop-client wrappers whose command line happens to contain
    the binary path, which would make the exclusivity gate refuse forever. So verify that the
    first argv token actually is the server binary before counting a pid as a server.
    """
    pids = []
    for p in sh("pgrep", "-f", SERVER_MATCH).split():
        if not p.strip():
            continue
        cmd = sh("ps", "-o", "command=", "-p", p).strip()
        if cmd and cmd.split(" ", 1)[0].endswith(SERVER_MATCH):
            pids.append(p)
    return pids


def kill_servers():
    subprocess.run(["pkill", "-9", "-f", SERVER_MATCH], capture_output=True)
    time.sleep(3)


def mem_free_pct():
    m = re.search(r"free percentage:\s*(\d+)%", sh("memory_pressure", "-Q"))
    return int(m.group(1)) if m else -1


def server_pid():
    pids = running_servers()
    return pids[0] if pids else None


def rss_gb():
    p = server_pid()
    if not p:
        return 0.0
    out = sh("ps", "-o", "rss=", "-p", p).strip()
    return round(int(out) / 1048576, 2) if out else 0.0


def health(port, timeout=3):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return b"ok" in r.read()
    except Exception:  # noqa: BLE001 - harness
        return False


# Canonical probe. The point is not the arithmetic: it is that a working scaffold makes the
# model produce a short answer containing 42, while a broken one echoes the prompt, leaks a
# think marker, or returns empty. Whatever the pool size, EVERY config must pass this or the
# numbers below it are measuring the template, not the cache.
PROBE_PROMPT = "15+27 等於多少？請只輸出答案"


def preflight(port, timeout=150.0):
    payload = {"model": "local",
               "messages": [{"role": "user", "content": PROBE_PROMPT}],
               "temperature": 0.0, "max_tokens": 64}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001 - harness
        return {"ok": False, "error": str(e)[:200], "content": ""}
    ch = (body.get("choices") or [{}])[0]
    content = (ch.get("message", {}).get("content") or "")
    finish = ch.get("finish_reason", "?")
    norm = re.sub(r"\s+", "", content)
    echoes = norm.count("15+27") > 1 or re.sub(r"\s+", "", PROBE_PROMPT) in norm
    leak = ("<think" in content) or ("think>" in content) or ("</think" in content)
    empty = content.strip() == ""
    answers_42 = "42" in content
    return {"ok": bool(answers_42 and not leak), "content": content[:240], "finish": finish,
            "answers_42": answers_42, "echoes_prompt": echoes, "scaffold_leak": leak,
            "empty": empty}


# ---------------------------------------------------------------------------
# GATES. A combo only counts as "same mechanism, just slower" if it passes ALL of these.
# Each gate exists because the corresponding failure was silent before.
# ---------------------------------------------------------------------------


def apply_template_probe(port, prompt=None, timeout=30.0):
    """Render the generation prompt the server would really use and check the scaffold is CLOSED.

    Checking the *answer* is not enough: a degenerating model can still stumble onto "42".
    This asks the server to render the template itself, so an unclosed `<think>` (the
    enable_thinking gate never satisfied) is caught even when the output happens to look fine.
    """
    body = {"messages": [{"role": "user", "content": prompt or PROBE_PROMPT}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/apply-template",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return {"available": False, "ok": None, "error": f"http {e.code}"}
    except Exception as e:  # noqa: BLE001 - harness
        return {"available": False, "ok": None, "error": str(e)[:120]}
    text = d.get("prompt", "") or ""
    last_open, last_close = text.rfind("<think"), text.rfind("</think>")
    unclosed = last_open != -1 and last_open > last_close
    return {"available": True, "ok": not unclosed, "unclosed_think": unclosed,
            "tail": text[-140:]}


def scan_nil(paths):
    """Grep for `buffer is nil` -- the signature of the L3-B gather path repointing a
    Metal-resident FFN tensor at a HOST std::vector. Any hit means the combo's numbers are
    invalid AND that the pool path silently switched. Must never be ignored."""
    hits = []
    for p in paths:
        if not p:
            continue
        try:
            txt = open(p, "rb").read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - harness
            continue
        n = len(NIL_RE.findall(txt))
        if n:
            hits.append({"file": os.path.basename(p), "count": n,
                         "n_blocks": len(set(re.findall(r"blk\.\d+", txt)))})
    return {"ok": not hits, "hits": hits}


# `CGC-UNION: layer=1 union avg=38.2 min=16 max=64 of usable=34 (188%)  [WIDE: exceeds usable] gather=3/9`
# The `gather=N/M` tail is the only place the engine reports that the WIDE-union route was
# actually taken (llama-context.cpp, CGC_UNION_LOG). The WIDE marker is required in the pattern:
# a narrow layer can also print gather>0 for other reasons, and those would prove nothing about
# the wide-union path this gate is about.
GATHER_RE = re.compile(r"CGC-UNION: layer=(\d+).*?\[WIDE[^\]]*\].*?gather=(\d+)/(\d+)")


def scan_gather_evidence(paths):
    """Did the WIDE-union route actually run, and how many steps went through it?

    Why this exists: `union > slots` used to be a hard FAIL because the only route out was the
    L3-B gather path, which repointed a Metal-resident FFN tensor at HOST memory and produced
    `buffer is nil` (measured 2 GiB, 2026-09-13: 468 nil, M1 2/42, first divergence at step 2,
    that step's sum off by 55%). That is no longer the case -- the gather path now fills a Metal
    slab (docs/M1_POOL_GRAPH_DECOUPLE_PLAN_2026-09-14.md 3.1) -- so the criterion becomes
    "routable": EITHER the pool is arithmetically guaranteed to hold the union, OR the wide-union
    route ran and produced no nil. Arithmetic alone would mark a correct 2 GiB cell as broken;
    nil alone would pass a cell that never exercised the path, which is the failure mode that
    hid this for two days.
    """
    wide_layers, steps, total = 0, 0, 0
    for p in paths:
        if not p:
            continue
        try:
            txt = open(p, "rb").read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - harness
            continue
        for _layer, hit, tot in GATHER_RE.findall(txt):
            total += int(tot)
            steps += int(hit)
            if int(hit) > 0:
                wide_layers += 1
    return {"wide_layers": wide_layers, "gather_steps": steps, "total_steps": total,
            "exercised": wide_layers > 0}


def union_fit_gate(facts, cap, topk, paths=None):
    """Is the pool EITHER guaranteed to hold the worst-case union, OR proven routable?

    The decode hook takes the pool path only when `union <= slots_per_layer`; otherwise it falls
    through to the wide-union route. A ubatch of `cap` tokens selects at most `cap * topk`
    distinct experts, so `cap * topk <= n_slots - 1` (slot 0 is the reserved ZERO slot) is the
    exact condition under which that fallback is unreachable -- for ANY model and ANY pool size.

    Reported as ok=None when the launch log did not expose n_slots (e.g. --attach to a server
    someone else started): unknown is not the same as passing.
    """
    # The binding number is the MINIMUM per-layer cap: the hook compares the union against the
    # cap of the layer it is currently in, so n_slots (or the average) would overstate the room
    # and could approve a cap that still overflows some layer. Fall back to n_slots only when
    # the log predates the min= field, and say which was used.
    src = "min_layer_slots" if facts.get("min_layer_slots") else "n_slots"
    try:
        n_slots = int(facts.get(src))
    except (TypeError, ValueError):
        return {"ok": None, "cap": cap, "topk": topk, "n_slots": None, "source": None,
                "union_max": cap * topk,
                "error": "neither min per-layer slots nor n_slots in the launch log"}
    union_max = cap * topk
    usable = n_slots - 1
    fit = bool(union_max <= usable)
    ev = scan_gather_evidence(paths or [])
    # mode is reported so a PASS cannot be mistaken for the other reason: `pool` means the
    # fallback was unreachable by arithmetic, `gather` means it was reached and stayed clean.
    if fit:
        mode = "pool"
    elif ev["exercised"]:
        mode = "gather"
    else:
        # Not routable by arithmetic AND no evidence the wide route ran. Two causes, and they
        # need different fixes: either the cell genuinely never reaches the wide route (then the
        # arithmetic is what matters and this is a real FAIL), or the evidence was never
        # recorded (CGC_UNION_LOG off -> `layers=0 steps=0/0`). The harness now sets that flag
        # unconditionally, so seeing this mode means the fallback really was not exercised.
        mode = "unreachable"
    return {"ok": bool(fit or ev["exercised"]), "mode": mode, "fit": fit,
            "cap": cap, "topk": topk, "n_slots": n_slots,
            "source": src, "usable_slots": usable, "union_max": union_max,
            "headroom": usable - union_max, "gather": ev}


# Floor for `CGC_POOL_MAX_TOKENS`. Two independent lower bounds:
#   * arithmetic -- cap*topk must fit the pool (see union_fit_gate);
#   * liveness   -- n_batch is clamped to cap, and the MTP verify batch needs n_batch above
#                   n_keep_tail: measured, cap=4 aborted at llama-batch.cpp:609
#                   GGML_ASSERT(n_ubatch > n_keep_tail) during the startup warmup decode.
MIN_CAP = 6


# Floor for `CGC_POOL_MAX_TOKENS`. Two independent lower bounds:
#   * arithmetic -- cap*topk must fit the pool (see union_fit_gate);
#   * liveness   -- n_batch is clamped to cap, and the MTP verify batch needs n_batch above
#                   n_keep_tail: measured, cap=4 aborted at llama-batch.cpp:609
#                   GGML_ASSERT(n_ubatch > n_keep_tail) in the startup warmup decode.
MIN_CAP = 6


# The feasibility prediction is a model of the LOADER (`compute_l4_pool_capacity` and
# `cgc_layer_cap`), so its provenance has to include the loader. Deliberately NOT merged into
# NUMERIC_SOURCES: that tuple stamps the oracle path, and a file added there would invalidate every
# existing reference (they carry no hash for it, so a diff would read as "numerics changed").
# Different question, different stamp.
POOL_GEOMETRY_SOURCES = (
    "src/llama.cpp/src/llama-model-loader.cpp",   # compute_l4_pool_capacity -> the slot count
    "src/llama.cpp/src/llama-expert-cache.cpp",   # cgc_layer_cap / LAYER_CAPS resolution
    "src/llama.cpp/src/llama-expert-cache.h",
)


def pool_geometry_stamp():
    """Fingerprint the code that decides how many slots a (model, pool) cell has."""
    digests, h = {}, hashlib.sha256()
    for rel in POOL_GEOMETRY_SOURCES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        blob = open(p, "rb").read()
        digests[rel] = hashlib.sha256(blob).hexdigest()[:16]
        h.update(rel.encode())
        h.update(blob)
    return {"source_digest": h.hexdigest(), "sources": digests,
            "last_commit": _git_out(["log", "-1", "--format=%h %ct %s", "--"]
                                    + list(POOL_GEOMETRY_SOURCES)),
            "head": _git_out(["rev-parse", "--short", "HEAD"]),
            "dirty": bool(_git_out(["status", "--short", "--"]
                                   + list(POOL_GEOMETRY_SOURCES)))}


BINARY_DIR = os.path.join(ROOT, "src", "llama.cpp", "build", "bin")


def binary_stamp():
    """Content-based identity of the server and the libraries it loads.

    size + a hash of the head, deliberately NOT mtime: a rebuild that changes nothing produces a
    fresh mtime over identical bytes, and calling that "different" would refuse comparisons that
    are perfectly valid. Cheap (64 KiB per file, once per cell).

    Narrowed to `llama-server` + `libllama*` + `libggml*`: the other `*-impl.dylib` files are
    separate tools (bench/quantize/perplexity) that cannot change what the server computes, and
    including them would split geometry groups every time an unrelated tool was relinked.
    """
    import glob
    out = {}
    paths = [os.path.join(BINARY_DIR, "llama-server")]
    for pat in ("libllama*.dylib", "libggml*.dylib"):
        paths += sorted(glob.glob(os.path.join(BINARY_DIR, pat)))
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            st = os.stat(p)
            with open(p, "rb") as f:
                head = f.read(1 << 16)
        except OSError:
            continue
        out[os.path.basename(p)] = {"size": st.st_size,
                                    "head": hashlib.sha256(head).hexdigest()[:16]}
    return out


def _as_int(v):
    """`read_launch_facts` returns strings; a stamp must not split groups over int-vs-str."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def record_provenance(kind, gb, args, cap=None, facts=None, extra_env=None):
    """The (pool, engine, weights) fingerprint that makes a result comparable -- or not.

    Stored INSIDE every result record. Without it the only way to tell whether two rows describe
    the same computation was to remember, and this project has paid for that twice: a cached cap
    that described an older loader (106 vs 143 slots, silently reused), and an oracle reference
    that described an older engine (M1 0/53 that read as a pool regression for hours).
    """
    ee = list((args.extra_env if extra_env is None else extra_env) or [])
    env = effective_launch_env(kind, ee)
    ec_state = expert_cache_state(ee, launch_pool_bytes(args, gb))
    pool = {"gb": int(gb), "bytes": launch_pool_bytes(args, gb), "cap": cap,
            "expert_cache": ec_state}
    if ec_state in GROUND_TRUTH_STATES:
        # Say it in the record, not just in the filename: a ground-truth row read out of
        # context (a glob over combo_*.json) must still be recognisable as a control.
        pool["ground_truth"] = True
    try:
        cell = feasibility_cell(kind, gb, cap, ee)
        pool.update({"min_layer_slots": cell["min_layer_slots"],
                     "usable_slots": cell["usable_slots"], "cap_max": cell["cap_max"],
                     "floor": cell["floor"], "verdict": cell["verdict"]})
    except Exception as e:  # noqa: BLE001 - stamping must not be able to kill a measurement
        pool["geometry_error"] = str(e)
    if facts:
        # What the engine REPORTED at load. If this disagrees with the prediction above, the row
        # says so -- the prediction is the model, this is the measurement.
        launched_slots = _as_int(facts.get("min_layer_slots"))
        pool.update({"launched_min_layer_slots": launched_slots,
                     "launched_n_slots": _as_int(facts.get("n_slots"))})
        # Per-row geometry mismatch flag: predicted (from pool_feasibility.py arithmetic) vs
        # measured (what the engine actually reported at load). This used to be visible only via
        # a one-shot cap probe; now every row carries it so a stale predictor is caught on the
        # first measurement that disagrees, not on the next probe run.
        predicted = pool.get("min_layer_slots")
        if predicted is not None and launched_slots is not None:
            mismatch = int(predicted) != int(launched_slots)
            pool["geometry_mismatch"] = mismatch
            if mismatch:
                pool["geometry_mismatch_detail"] = (
                    f"predicted {predicted} slots/layer vs launched {launched_slots}")
        else:
            pool["geometry_mismatch"] = None  # cannot tell (no prediction or no measurement)
    ss, gs = source_stamp(), pool_geometry_stamp()
    return {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "pool": pool,
            "engine": {"source_digest": ss["source_digest"], "sources": ss["sources"],
                       "geometry_digest": gs["source_digest"], "geometry_sources": gs["sources"],
                       "head": ss["head"], "last_commit": ss["last_commit"],
                       "dirty": ss["dirty"], "binary": binary_stamp()},
            "weights": model_stamp(model_path(kind)),
            "launch": {"mtp": str(env.get("CGC_SERVER_MTP", "1")).lower()
                               in ("1", "true", "on", "yes"),
                       "spec_n_max": _env_int(env, "CGC_SERVER_MTP_N_MAX", 3),
                       "layer_caps": env.get("CGC_SERVER_LAYER_CAPS") or "default",
                       "mode": "M2_stream" if getattr(args, "m2", False) else "standard",
                       "extra_env": sorted(ee)},
            # The SHAPE of the measurement, not of the engine. Two rows measured with different
            # suites are not the same experiment even when the pool, engine and weights match --
            # a 1-question smoke run and a 48-question suite both land in `iq3_pool8gb`.
            "suite": {"profiles": getattr(args, "profiles", None),
                      "per_profile": getattr(args, "per_profile", None),
                      "greedy_repeats": getattr(args, "greedy_repeats", None),
                      "repeats": getattr(args, "repeats", None),
                      "temperature": getattr(args, "temperature", None),
                      "max_tokens": getattr(args, "max_tokens", None)},
            # Informational only, NEVER part of the comparability key: a harness edit (a print, a
            # new gate) must not retroactively declare every past measurement incomparable.
            "harness": {"script_digest": hashlib.sha256(
                open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]}}


def record_geometry_key(rec):
    """Everything that must MATCH for two rows to be comparable, or None when unstamped.

    The POOL SIZE is deliberately absent: it is the variable under test. The cap IS present,
    because it clamps n_batch and therefore decides the prefill step partition (docs/
    CAP_INVARIANCE_LOCALIZATION_2026-09-14.md) -- so two rows at different caps are different
    computations even when everything else matches.
    """
    prov = rec.get("provenance")
    if not prov:
        return None
    eng = prov.get("engine") or {}
    w = prov.get("weights") or {}
    L = prov.get("launch") or {}
    binary = tuple(sorted((k, (v or {}).get("size"), (v or {}).get("head"))
                          for k, v in (eng.get("binary") or {}).items()))
    return {"engine": eng.get("source_digest"),
            "pool_geometry": eng.get("geometry_digest"),
            "binary": binary,
            "weights": (w.get("realpath"), w.get("size"), w.get("digest")),
            "cap": rec.get("pool_cap", (prov.get("pool") or {}).get("cap")),
            "mode": L.get("mode", "standard"),
            "mtp": L.get("mtp"), "spec_n_max": L.get("spec_n_max"),
            "layer_caps": L.get("layer_caps"), "extra_env": tuple(L.get("extra_env") or ()),
            "suite": tuple(sorted(((prov.get("suite") or {}).items()))) }


def _short_val(v, width=52):
    """Keep a refusal readable: the binary stamp alone is a 38-element tuple."""
    s = repr(v)
    if len(s) <= width:
        return s
    if isinstance(v, (tuple, list, dict)):
        return f"{type(v).__name__} of {len(v)}: {s[:width - 24]}…"
    return s[:width - 1] + "…"


def key_fingerprint(key):
    """Short, stable label for one geometry group (so blocks can be named in the output)."""
    if key is None:
        return "unattributed"
    blob = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


# Human-readable names for the fields of `record_geometry_key`. A refusal that says "something
# differs" is useless; it has to name which of these moved and to what.
KEY_FIELDS = (("engine", "numerics code (source digest)"),
              ("pool_geometry", "pool-geometry code (loader/cache digest)"),
              ("binary", "built binary"),
              ("weights", "weights identity"),
              ("cap", "CGC_POOL_MAX_TOKENS"),
              ("mode", "prefill mode (standard / M2_stream)"),
              ("mtp", "MTP"),
              ("spec_n_max", "spec-draft-n-max"),
              ("layer_caps", "LAYER_CAPS"),
              ("extra_env", "extra env"),
              ("suite", "suite shape (profiles / questions / repeats / temperature / max_tokens)"))


def comparability_verdict(recs):
    """Group rows per model by geometry and say whether they may be compared at all.

    states:
      OK            one geometry group -> pool size is the only variable
      UNATTRIBUTED  one group, but it carries no provenance (pre-2026-09-14 records) -> nothing can
                    be certified; re-measure with the current harness to make it comparable
      REFUSED       two or more groups -> the rows are NOT comparable, and the caller must say so
                    instead of printing them as one table
    """
    by_model = {}
    for r in recs:
        by_model.setdefault(r.get("model") or "?", []).append(r)
    out = {}
    for model, rows in by_model.items():
        groups = {}
        for r in rows:
            key = record_geometry_key(r)
            fp = key_fingerprint(key)
            g = groups.setdefault(fp, {"key": key, "fingerprint": fp, "rows": []})
            g["rows"].append(r)
        if len(groups) == 1:
            only = next(iter(groups.values()))
            state = "UNATTRIBUTED" if only["key"] is None else "OK"
        else:
            state = "REFUSED"
        # Name every field whose value differs between groups, with the rows on each side.
        # A group with NO provenance differs from everything by construction, so it is reported as
        # that fact instead of as a list of nine fields whose real cause is one: nobody recorded
        # them. (Otherwise the refusal drowns in its own noise, starting with the binary tuple.)
        differences = []
        if state == "REFUSED":
            glist = list(groups.values())
            stamped = [g for g in glist if g["key"] is not None]
            unstamped = [g["fingerprint"] for g in glist if g["key"] is None]
            if unstamped:
                differences.append(
                    f"group(s) {unstamped} carry NO provenance, so none of their fields can be "
                    f"compared with anything (records written before the stamp existed)"
                    + (f" -- re-measure them if they are meant to sit beside the "
                       f"{len(stamped)} stamped group(s)" if stamped else ""))
            for field, human in KEY_FIELDS:
                seen = {g["fingerprint"]: (g["key"] or {}).get(field) for g in stamped}
                if len(seen) > 1 and len({repr(v) for v in seen.values()}) > 1:
                    parts = ", ".join(f"{fp}: {_short_val(val)}" for fp, val in seen.items())
                    differences.append(f"{human} differs -> {parts}")
        out[model] = {"state": state, "groups": list(groups.values()),
                      "differences": differences}
    return out


def _cap_probe_provenance(kind, args):
    """Everything that must still hold for a cached cap-probe record to describe THIS tree."""
    env = effective_launch_env(kind, args.extra_env)
    return {"pool_geometry": pool_geometry_stamp(),
            "model": model_stamp(model_path(kind)),
            "layer_caps": env.get("CGC_SERVER_LAYER_CAPS") or "default",
            "mtp": str(env.get("CGC_SERVER_MTP", "1")).lower() in ("1", "true", "on", "yes"),
            "spec_n_max": _env_int(env, "CGC_SERVER_MTP_N_MAX", 3),
            "extra_env": sorted(args.extra_env or [])}


def _recorded_slots(rec):
    """The MEASURED per-layer slot count in a cap-probe record, or None."""
    uf = rec.get("union_fit") or {}
    if uf.get("n_slots"):
        return uf["n_slots"]
    for t in reversed(rec.get("tried") or []):
        tuf = t.get("union_fit") or {}
        if tuf.get("n_slots"):
            return tuf["n_slots"]
    return None


def reconcile_cap_probe(kind, gb, args, rec, source="cache"):
    """Cross-check a MEASURED cap-probe slot count against the arithmetic prediction.

    Why this is necessary and not paranoia: `cap_probe_iq3_pool8gb.json` (recorded 2026-09-11) says
    **106** slots while the prediction -- and every 2026-09-14 launch log -- says **143**. It was
    being reused verbatim, so the whole sweep's cap was derived from a geometry this engine no
    longer has, and the number looked perfectly plausible in the record. Worse, the same stale
    number is what `--cap-invariance` would then compare across pools.

    The rule is: the MEASUREMENT wins (it is the engine); the mismatch is what has to be explained,
    and the two explanations need opposite actions:

      MATCH          -> reuse is sound
      STALE_RECORD   -> the record's code / weights / launch env moved since it was written
                        (or it carries no provenance at all). Discard and re-probe.
      PREDICTOR_STALE-> the record (or the launch that just happened) matches this tree exactly, so
                        the ARITHMETIC in pool_feasibility.py is what no longer describes the
                        loader. Every feasibility verdict rests on it -> stop.
      NO_EVIDENCE    -> the record holds no measured slot count at all.
    """
    out = {"source": source, "ok": None, "verdict": "UNKNOWN"}
    try:
        cell = feasibility_cell(kind, gb, rec.get("cap"), list(args.extra_env or []))
    except Exception as e:  # noqa: BLE001 - a cross-check must never be why a run dies
        out.update({"error": f"no prediction available ({e})"})
        return out
    measured = _recorded_slots(rec)
    out.update({"predicted_min_layer_slots": cell["min_layer_slots"],
                "predicted_usable_slots": cell["usable_slots"],
                "predicted_cap_max": cell["cap_max"],
                "measured_min_layer_slots": measured,
                "record": os.path.join(RESULT_DIR, f"cap_probe_{kind}_pool{gb}gb.json")})
    if measured is None:
        out.update({"ok": False, "verdict": "NO_EVIDENCE",
                    "cause": "the record contains no measured slot count (its probe never reached a "
                             "loaded server)"})
        out["error"] = (f"CAP-PROBE UNVERIFIABLE: {out['record']} has no measured slot count, so "
                        f"its cap={rec.get('cap')} cannot be checked against the arithmetic "
                        f"(predicted {cell['min_layer_slots']} slots). Re-probing.")
        return out
    if int(measured) == int(cell["min_layer_slots"]):
        out.update({"ok": True, "verdict": "MATCH"})
        return out

    prov = rec.get("provenance")
    cur = _cap_probe_provenance(kind, args)
    if source == "probe" or not prov:
        verdict = "PREDICTOR_STALE" if source == "probe" else "STALE_RECORD"
        cause = ("the launch that produced this number just happened from this tree, so "
                 "pool_feasibility.py no longer describes the loader"
                 if source == "probe" else
                 "the record carries NO provenance, so nothing ties it to this tree "
                 "(records written before 2026-09-14 have none)")
    else:
        diffs = []
        pg = prov.get("pool_geometry") or {}
        if pg.get("source_digest") != cur["pool_geometry"]["source_digest"]:
            diffs.append(f"pool-geometry code changed (record digest {pg.get('source_digest') or '?'}"
                         f", now {cur['pool_geometry']['source_digest']}; newest commit over those "
                         f"paths then: {pg.get('last_commit') or '?'})")
        pm, cm = prov.get("model") or {}, cur["model"] or {}
        if (pm.get("size"), pm.get("digest")) != (cm.get("size"), cm.get("digest")):
            diffs.append(f"weights differ (record {pm.get('realpath') or '?'} "
                         f"size={pm.get('size')}, now {cm.get('realpath') or '?'} "
                         f"size={cm.get('size')})")
        for k in ("layer_caps", "mtp", "spec_n_max", "extra_env"):
            if prov.get(k) != cur.get(k):
                diffs.append(f"{k} changed (record {prov.get(k)!r} vs now {cur.get(k)!r})")
        if diffs:
            verdict, cause = "STALE_RECORD", "; ".join(diffs)
        else:
            verdict = "PREDICTOR_STALE"
            cause = ("the record's provenance matches this tree exactly, so the engine code did "
                     "not move -- the arithmetic in pool_feasibility.py is what is wrong")
    out.update({"ok": False, "verdict": verdict, "cause": cause,
                "provenance_recorded": prov is not None})
    out["error"] = (f"CAP-PROBE MISMATCH ({verdict}): measured {measured} slots vs predicted "
                    f"{cell['min_layer_slots']} for {kind} pool{gb}GB. Cause: {cause}. "
                    f"Both numbers and the record path are in the artifact; the MEASURED one wins "
                    f"(it is the engine).")
    return out


def cap_probe_reusable(check):
    """May a cached cap-probe record be reused? Only when it reconciles (see reconcile_cap_probe)."""
    return check.get("ok") is True


def geometry_conflict_error(rec, allow=False):
    """The refusal message for a FRESH probe that contradicts the arithmetic, or None.

    Kept separate from main() so the fatal path is testable without launching anything: this is
    the branch that decides "do not measure at all", and it has to be pinned like any other
    decision (the whole reason this gate exists is that a wrong number otherwise looks fine).
    """
    gc = rec.get("geometry_conflict")
    if not gc or allow:
        return None
    return (f"REFUSING to start the sweep: the cap probe measured "
            f"{gc.get('measured_min_layer_slots')} slots where pool_feasibility.py predicts "
            f"{gc.get('predicted_min_layer_slots')}.\n"
            f"  {gc.get('cause')}\n"
            f"  Record: {gc.get('record')}\n"
            f"  Fix the arithmetic (or the loader) and re-run; --allow-geometry-drift records "
            f"the disagreement and measures anyway, with the conflicting geometry stamped "
            f"into every result.")


def probe_pool_cap(kind, gb, args):
    """Pick ONE `CGC_POOL_MAX_TOKENS` that keeps the SMALLEST pool on the pool path.

    Generic across models: it only uses the observed `buffer is nil` signal, so a new GGUF with
    a different expert count or expert width needs no new hand-tuned constant.

    A single cap for every pool is mandatory, not a convenience: the cap clamps n_batch and
    therefore decides the ubatch shape and the prefill chunking. Letting each pool pick its own
    cap would compare different computations, not different pool sizes. Measured here: 8GB with
    pmax=8 vs 8GB with pmax=6 agreed on argmax only 15.4% of steps -- the cap IS a quality knob.
    """
    cache_path = os.path.join(RESULT_DIR, f"cap_probe_{kind}_pool{gb}gb.json")
    if os.path.exists(cache_path) and not args.force:
        rec = json.load(open(cache_path, encoding="utf-8"))
        check = reconcile_cap_probe(kind, gb, args, rec, source="cache")
        if check.get("ok") is None:
            # No prediction available (GGUF unreadable / module missing): the cache can still be
            # used, but say that it was NOT verified rather than implying it was.
            print(f"[cap-probe] {kind} pool{gb}GB cached -> cap={rec.get('cap')} "
                  f"(nil={rec.get('nil')}); NOT cross-checked ({check.get('error')})", flush=True)
            return rec
        if cap_probe_reusable(check):
            strong = bool(rec.get("provenance"))
            print(f"[cap-probe] {kind} pool{gb}GB cached -> cap={rec.get('cap')} "
                  f"(nil={rec.get('nil')}); geometry verified: "
                  f"{check['measured_min_layer_slots']} slots = predicted "
                  f"{check['predicted_min_layer_slots']}"
                  + ("" if strong else "  [legacy record: no provenance, so the AGREEMENT is what "
                                       "it stands on]"), flush=True)
            rec["verified_slots"] = {
                "measured": check["measured_min_layer_slots"],
                "predicted": check["predicted_min_layer_slots"],
                "basis": ("recorded provenance matches this tree" if strong else
                          "legacy record: no provenance, agrees with the arithmetic"),
                "pool_geometry_digest_now": pool_geometry_stamp()["source_digest"]}
            return rec
        # NOT reusable. Do not raise here: a stale cache is repaired by re-probing, which is
        # ground truth. What must never happen again is using it SILENTLY.
        print(f"[cap-probe] {kind} pool{gb}GB cache REFUSED\n  {check.get('error')}\n"
              f"  Re-probing instead of reusing it: the cap must come from a geometry this "
              f"engine actually has. (--force always re-probes; this refusal is the same action, "
              f"taken because the cached number is provably wrong or unattributable.)",
              file=sys.stderr, flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    topk = MODELS[kind].get("topk", 8)
    rec = {"kind": kind, "pool_gb": gb, "topk": topk, "cap": None, "tried": [],
           "min_cap": MIN_CAP}
    cap = min(8, args.pool_cap_max)
    for _ in range(3):
        log_path = os.path.join(RESULT_DIR, f"capprobe_{kind}_pool{gb}gb_cap{cap}.log")
        print(f"[cap-probe] {kind} pool{gb}GB trying cap={cap} (union <= {cap * topk})",
              flush=True)
        pre = tuple(running_servers())
        kill_servers()
        if running_servers():
            rec["tried"].append({"cap": cap, "error": "foreign server would not clear"})
            break
        launch(model_path(kind), MODELS[kind]["mtp"], launch_pool_bytes(args, gb), args.port,
               args.kwargs, list(args.extra_env) + [f"CGC_POOL_MAX_TOKENS={cap}"], log_path)
        took = wait_health(args.port, model_path(kind), args.health_timeout,
                           pre_pids=pre)
        if took is None:
            rec["tried"].append({"cap": cap, "error": "server down, crashed, or a foreign "
                                                     "server owns the port"})
            kill_servers()
            break
        # Send real traffic BEFORE scanning: `buffer is nil` only appears once a decode actually
        # builds an expert union. Without this the too-large cap is rejected on arithmetic alone
        # and the log looks clean -- observation is the stronger evidence, arithmetic is what
        # makes the result generalisable to a prompt that happens not to overflow.
        ask_probe(args.port, timeout=min(args.health_timeout, 240.0))
        facts = read_launch_facts(log_path)
        nil = scan_nil([log_path, facts.get("log_path")])
        fit = union_fit_gate(facts, cap, topk)
        rec["tried"].append({"cap": cap, "nil": nil["hits"], "union_fit": fit,
                             "health_s": round(took, 1)})
        kill_servers()
        # A cap only counts as safe when the LAUNCH LOG proves the invariant. `ok is not False`
        # was too weak: a server that never loaded reports no nil AND no n_slots, which read as
        # a pass. Require the arithmetic proof (`ok is True`), not the absence of evidence.
        if nil["ok"] and fit.get("ok") is True:
            rec["cap"] = cap
            rec["nil"] = nil["hits"]
            rec["union_fit"] = fit
            break
        if fit.get("ok") is False:
            # The MEASURED pool cannot hold cap*topk experts, so the gather path is reachable.
            # Jump straight to the largest cap that provably fits (not merely cap-1).
            nxt = min(cap - 1, fit["usable_slots"] // topk)
            if nxt < MIN_CAP:
                rec["error"] = (f"pool {gb}GB holds {fit['usable_slots']} usable slots: cap "
                                f"{nxt}*{topk} would fit but cap < {MIN_CAP} aborts at "
                                f"llama-batch.cpp:609 (GGML_ASSERT(n_ubatch > n_keep_tail)) "
                                f"during warmup. This pool is too small for topk={topk}.")
                break
            cap = nxt
            continue
        cap -= 1
        if cap < MIN_CAP:
            rec["error"] = f"no tried cap >= {MIN_CAP} produced a clean pool path"
            break
    # Stamp the record and reconcile it with the arithmetic, in the same write. Both halves are
    # needed: without the stamp the next run cannot tell whether a mismatch means "this file is
    # old" or "the loader changed"; without the reconciliation a wrong number is only visible if
    # someone happens to compare it by hand -- which is exactly how 106 vs 143 survived.
    rec["provenance"] = _cap_probe_provenance(kind, args)
    check = reconcile_cap_probe(kind, gb, args, rec, source="probe")
    rec["reconciled"] = check
    if check.get("ok") is False and cap_probe_reusable(check) is False:
        # The MEASURED geometry and the arithmetic model disagree, and this launch came from the
        # current tree -- so the arithmetic every feasibility verdict rests on is wrong. Record it
        # as a conflict and let main() refuse the sweep: launching cells under a geometry model we
        # have just falsified is how "the pool broke numerics" verdicts get manufactured.
        # The measured cap itself is kept (it IS a measurement); what is refused is acting on it.
        rec["geometry_conflict"] = check
    json.dump(rec, open(cache_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if check.get("ok") is True:
        print(f"[cap-probe] geometry reconciled: measured "
              f"{check['measured_min_layer_slots']} slots = predicted "
              f"{check['predicted_min_layer_slots']}")
    elif check.get("ok") is False:
        print(f"[cap-probe] GEOMETRY CONFLICT\n  {check.get('error')}", file=sys.stderr, flush=True)
    if rec["cap"] is None and not rec.get("geometry_conflict"):
        print(f"[cap-probe] WARNING: no cap kept pool{gb}GB on the pool path; "
              f"the matrix will still run but the nil gate will fail.", file=sys.stderr,
              flush=True)
    else:
        print(f"[cap-probe] {kind} -> cap={rec['cap']} for ALL pools "
              f"(union {rec['cap'] * topk} <= usable {rec.get('union_fit', {}).get('usable_slots')})",
              flush=True)
    return rec


# Source files that DECIDE the numerics. A reference oracle is a snapshot of the engine these
# files describe, so a change here invalidates it. Deliberately narrow: harness/probe scripts
# are excluded because editing them cannot move a single logit.
NUMERIC_SOURCES = (
    "src/llama.cpp/src/llama-expert-cache.cpp",
    "src/llama.cpp/src/llama-expert-cache.h",
    "src/llama.cpp/src/llama-context.cpp",
    "src/llama.cpp/src/llama-graph.cpp",
)


def _git_out(argv):
    """Run git in the repo root; '' on any failure (git absent, not a repo, timeout)."""
    try:
        return subprocess.run(["git"] + list(argv), cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001 - harness
        return ""


def source_stamp():
    """Fingerprint the code that decides the numerics, plus WHO last commit-touched it.

    Both halves are load-bearing and neither is sufficient on its own:
      * a commit-only stamp misses UNCOMMITTED work, and that is not hypothetical -- the live
        state on 2026-09-13 is exactly it: HEAD's newest commit over these paths is 09-11
        22:54, while the three expert-cache/context files were then edited 09-13 00:31-01:35
        and the binary rebuilt 01:36. A commit-only rule would call that tree unchanged.
      * a content-only stamp cannot say which commit introduced the change, so a stale
        reference could not be explained, only rejected.
    """
    digests, missing, h = {}, [], hashlib.sha256()
    for rel in NUMERIC_SOURCES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            missing.append(rel)
            continue
        blob = open(p, "rb").read()
        digests[rel] = hashlib.sha256(blob).hexdigest()[:16]
        h.update(rel.encode())
        h.update(blob)
    return {"source_digest": h.hexdigest(),
            "sources": digests,
            "missing": missing,
            "last_commit": _git_out(["log", "-1", "--format=%h %ct %s", "--"] +
                                    list(NUMERIC_SOURCES)),
            "head": _git_out(["rev-parse", "--short", "HEAD"]),
            "dirty": bool(_git_out(["status", "--short", "--"] + list(NUMERIC_SOURCES)))}


def _oracle_meta(path):
    """Read an oracle's sidecar as a dict, or None. Tolerant of the legacy plain-cap form."""
    if not path:
        return None
    p = path + ".cap"
    if not os.path.exists(p):
        return None
    try:
        raw = open(p, encoding="utf-8").read().strip()
    except Exception:  # noqa: BLE001 - harness
        return None
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return {"cap": raw}          # legacy: the file held the bare cap, e.g. "6"
    if isinstance(d, dict):
        return d
    return {"cap": str(d)}           # legacy: a bare JSON number


def _cap_sidecar(path):
    """Read the cap (CGC_POOL_MAX_TOKENS) recorded beside an oracle dump, or None if absent.

    The sidecar exists because the cap is NOT a neutral knob: it clamps n_batch and therefore
    fixes the ubatch shape and the prefill chunking (see probe_pool_cap's docstring -- 8 GB at
    pmax=8 vs pmax=6 agreed on argmax only 15.4% of steps). Comparing a dump taken at one cap
    against a reference taken at another yields a large, entirely SPURIOUS M1/M2 failure that
    reads as "the pool broke numerics". Measured 2026-09-13: a cap=8 dump vs a cap=6 reference
    scored M1 3/98 and M2 12/98 while the two caps' dumps are byte-identical within their own
    cap. Guarding against it beats explaining it afterwards.
    """
    m = _oracle_meta(path)
    if not m or m.get("cap") is None:
        return None
    return str(m["cap"])


MODEL_SAMPLE_BYTES = 4 << 20        # 4 MiB from the head + 4 MiB from the tail


def _model_digest_full():
    """Full 13 GB sha256 is opt-in: it evicts the page cache the harness depends on."""
    return os.environ.get("CGC_ORACLE_MODEL_DIGEST", "").strip().lower() == "full"


def model_stamp(model_file):
    """Identity of the WEIGHTS an oracle dump was produced from.

    The source stamp records the CODE but not the MODEL, so a reference taken from one GGUF
    could be compared against another -- and that is not hypothetical: this matrix's `iq3`
    column was silently resolving to the IQ4_XS file, so two columns measured a single model
    and nothing in the record showed it. The model dimension is the whole point of this harness.

    Cost/benefit: realpath+size+inode+mtime are free and already catch that observed failure
    (IQ4_XS is 18.2 GB vs IQ3's 13.66 GB). A sampled digest (head+tail 4 MiB) additionally makes
    a SAME-SIZE swap detectable, because a re-quantized file differs in its GGUF header/KV block
    and in its tail. A full sha256 is available via CGC_ORACLE_MODEL_DIGEST=full, but not the
    default: reading 13.7 GB evicts the page cache, and page-cache state (7721 MiB/s hot vs
    1225 MiB/s cold) is exactly what these measurements turn on.
    """
    if not model_file:
        return None
    try:
        real = os.path.realpath(model_file)
        st = os.stat(real)
    except OSError as e:
        return {"path": model_file, "error": str(e)[:200]}
    rec = {"path": model_file, "realpath": real, "size": st.st_size,
           "mtime": int(st.st_mtime), "inode": st.st_ino,
           "digest": None, "digest_mode": None}
    try:
        if _model_digest_full():
            h = hashlib.sha256()
            with open(real, "rb") as f:
                for chunk in iter(lambda: f.read(8 << 20), b""):
                    h.update(chunk)
            rec.update({"digest": h.hexdigest(), "digest_mode": "full"})
        else:
            h = hashlib.sha256()
            with open(real, "rb") as f:
                h.update(f.read(MODEL_SAMPLE_BYTES))
                if st.st_size > 2 * MODEL_SAMPLE_BYTES:
                    f.seek(st.st_size - MODEL_SAMPLE_BYTES)
                    h.update(f.read(MODEL_SAMPLE_BYTES))
            rec.update({"digest": h.hexdigest(), "digest_mode": "sampled(head+tail)"})
    except OSError as e:
        rec["digest_error"] = str(e)[:200]
    return rec


def _model_guard(ref_path, model_file):
    """Refuse to compare when the reference came from different WEIGHTS."""
    if not model_file:
        return None
    ref = (_oracle_meta(ref_path) or {}).get("model")
    cur = model_stamp(model_file)
    if not isinstance(ref, dict) or ref.get("error"):
        return {"ok": None, "model_unknown": True, "current_model": cur,
                "error": (f"MODEL IDENTITY UNKNOWN: {os.path.basename(ref_path)} records no "
                          f"weights identity, so it could have come from any GGUF and comparing "
                          f"logits across two models is meaningless. Rebuild the reference, or "
                          f"pass --allow-model-mismatch to compare anyway.")}
    same = (ref.get("realpath") == cur.get("realpath")
            and ref.get("size") == cur.get("size")
            and ref.get("digest") == cur.get("digest"))
    if same:
        return None
    return {"ok": False, "model_mismatch": {"ref": ref, "candidate": cur},
            "error": (f"MODEL MISMATCH: {os.path.basename(ref_path)} was produced from "
                      f"{ref.get('realpath')} ({ref.get('size')} B, "
                      f"{ref.get('digest_mode')}={str(ref.get('digest'))[:12]}), but this run "
                      f"measures {cur.get('realpath')} ({cur.get('size')} B, "
                      f"{cur.get('digest_mode')}={str(cur.get('digest'))[:12]}). Any M1/M2 "
                      f"number would compare two different models. Regenerate the reference "
                      f"for this GGUF, or pass --allow-model-mismatch if that is deliberate.")}


# [CGC 2026-09-15] How the expert cache participates in a dump. This is the field that was
# missing, and its absence is why the gate could report M1 19/19 for a year while the engine was
# wrong: every reference on this box was produced by `run_server.sh`, which ALWAYS passes
# `-expert-cache <budget>` (scripts/run_server.sh:587). So "reference vs candidate" was
# expert-cache-ON vs expert-cache-ON -- a pool-size invariance check wearing the costume of a
# correctness check. Only the states in GROUND_TRUTH_STATES may be compared against as truth.
EXPERT_CACHE_ON = "on"
EXPERT_CACHE_OFF = "off"                 # no cache object at all (budget 0)
EXPERT_CACHE_OFF_NOGATHER = "off_nogather"   # cache exists, hook + skip_load both off
EXPERT_CACHE_NOHOOK = "on_nohook"        # cache exists, skip_load on, hook off (4th arm)
GROUND_TRUTH_STATES = (EXPERT_CACHE_OFF, EXPERT_CACHE_OFF_NOGATHER)


def expert_cache_state(extra_env=None, pool_bytes=None):
    """Classify how the expert cache participates, from the launch knobs.

    Read from the LAUNCH, not from the label: a dump called `ref_*_NOCACHE*` that was actually
    launched with a budget is worse than no dump at all, because it looks like a control.
    """
    try:
        if pool_bytes is not None and int(pool_bytes) == 0:
            return EXPERT_CACHE_OFF
    except (TypeError, ValueError):
        pass
    for kv in extra_env or []:
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if k == "LLAMA_EXPERT_CACHE_NOGATHER" and v not in ("", "0"):
            return EXPERT_CACHE_OFF_NOGATHER
        if k == "LLAMA_EXPERT_CACHE_NOHOOK" and v not in ("", "0"):
            return EXPERT_CACHE_NOHOOK
    return EXPERT_CACHE_ON


def _write_oracle_meta(path, cap, model_file=None, launch_sig=None, expert_cache=None):
    """Write the provenance sidecar: cap + source stamp + weights identity + LAUNCH KNOBS.

    `launch_sig` records the launch-time environment that shapes the computation (extra env vars,
    MTP on/off). It is not decoration: the cap is not the only knob that changes the numerics, and
    a dump whose label says only model/pool/cap would be silently reused across, say, MTP on and
    MTP off -- the reuse check would see the same cap, the same engine stamp and the same weights
    and conclude it already had the dump it needed.

    `expert_cache` is the 2026-09-15 addition: on / off / off_nogather / on_nohook. It decides
    whether this dump may ever be used as GROUND TRUTH (see _truth_guard). Written as "unknown"
    rather than omitted when the caller does not say, so "never recorded" and "recorded as off"
    stay distinguishable -- an unlabelled dump must not silently pass as a control.
    """
    meta = {"cap": str(cap) if cap is not None else "default",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stamp": source_stamp(),
            "expert_cache": expert_cache if expert_cache is not None else "unknown"}
    if launch_sig is not None:
        meta["launch"] = launch_sig
    m = model_stamp(model_file)
    if m is not None:
        meta["model"] = m
    with open(path + ".cap", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")


def _cap_guard(ref_path, dump_path):
    """Return an error record when the reference and the dump were taken at different caps."""
    cap_ref, cap_new = _cap_sidecar(ref_path), _cap_sidecar(dump_path)
    if cap_ref and cap_new and cap_ref != cap_new:
        return {"ok": False,
                "cap_mismatch": {"ref": cap_ref, "candidate": cap_new},
                "error": (f"CAP MISMATCH: reference was produced at cap={cap_ref}, this dump at "
                          f"cap={cap_new}. The cap clamps n_batch, so these are different "
                          f"computations -- any M1/M2 number would be a shape artifact, not a "
                          f"pool difference. Re-run with --pool-cap {cap_ref}.")}
    return None


def _stamp_guard(ref_path):
    """Refuse to compare when the reference predates a change to the numerics sources.

    Why this exists: on 2026-09-13 a reference captured 21:58 was silently compared against an
    engine fixed at 22:54, and the result (M1 0/53) read as a pool regression for hours. A stale
    reference cannot be detected from the numbers -- it fails loud, looks real, and points at
    the wrong subsystem. So provenance is checked BEFORE the comparison, not after.
    """
    rec = (_oracle_meta(ref_path) or {}).get("stamp")
    cur = source_stamp()
    if not isinstance(rec, dict):
        return {"ok": None, "stale_unknown": True, "current_source_stamp": cur,
                "error": (f"REFERENCE PROVENANCE UNKNOWN: {os.path.basename(ref_path)} has no "
                          f"source stamp, so it is impossible to tell which expert-cache code "
                          f"produced it. Rebuild it (this driver now stamps every reference), "
                          f"or pass --allow-stale-oracle to compare anyway.")}
    same_digest = rec.get("source_digest") == cur["source_digest"]
    same_commit = rec.get("last_commit") == cur["last_commit"]
    if same_digest and same_commit:
        return None
    changed = sorted(r for r in NUMERIC_SOURCES
                     if rec.get("sources", {}).get(r) != cur["sources"].get(r))
    kind = "content_changed" if not same_digest else "commit_changed"
    detail = (f"changed sources: {', '.join(os.path.basename(c) for c in changed) or '(none)'}"
              if not same_digest else
              "file contents are identical; only the newest commit over these paths moved")
    if kind == "content_changed":
        ask = ("Rebuild the reference from the current tree, then re-run the comparison so the "
               "pool-size verdict describes ONE engine.")
    else:
        ask = ("The numerics are unchanged, so a comparison is still meaningful -- pass "
               "--allow-stale-oracle once you have confirmed that, or refresh the reference.")
    return {"ok": False, "stale": True, "stale_kind": kind,
            "ref_source_stamp": rec, "current_source_stamp": cur, "changed_sources": changed,
            "error": (f"STALE REFERENCE ({kind}): {os.path.basename(ref_path)} was produced from "
                      f"different engine code than the tree about to be measured. "
                      f"ref last_commit = {rec.get('last_commit') or '?'} | "
                      f"current last_commit = {cur['last_commit'] or '?'} | {detail}. "
                      f"Any M1/M2 number here describes the code drift, not the pool size. {ask}")}


def _truth_guard(ref_path):
    """Refuse to use an expert-cache-ON dump as ground truth.

    This is the guard that did not exist and whose absence made every "M1 19/19" on this box
    unfalsifiable. A dump produced with the expert cache active can only ever prove that two
    cache-ON runs agree with each other; it carries no information about whether either agrees
    with the model. Measured 2026-09-14/15: the same harness reported M1/M2/M3 = 19/19 for
    4/6/8 GiB while a hand-run A/B against a cache-OFF baseline disagreed at step 0
    (argmax 271 / logit 14.22 vs argmax 198 / logit 27.18).

    Deliberately NOT bypassable by --allow-stale-oracle or --allow-model-mismatch: those two
    flags answer "is this reference old / is it the same file?", which is a different question
    from "is this reference a control?". Conflating them is exactly how the 19/19 happened.
    """
    meta = _oracle_meta(ref_path) or {}
    ec = meta.get("expert_cache")
    if ec is None or ec == "unknown":
        return {"ok": None, "truth_unknown": True, "ref_expert_cache": ec,
                "error": (f"REFERENCE HAS NO EXPERT-CACHE STATE: {os.path.basename(ref_path)} "
                          f"does not record whether the expert cache was active, so it cannot be "
                          f"used as ground truth. Re-dump it with this driver (it now stamps "
                          f"expert_cache=on|off|off_nogather|on_nohook), or use it only for a "
                          f"RELATIVE pool-size comparison (omit --oracle-abs).")}
    if ec not in GROUND_TRUTH_STATES:
        return {"ok": False, "not_ground_truth": True, "ref_expert_cache": ec,
                "accepted_states": list(GROUND_TRUTH_STATES),
                "error": (f"REFERENCE IS NOT GROUND TRUTH: {os.path.basename(ref_path)} was "
                          f"produced with expert_cache={ec}. Comparing a candidate against a "
                          f"cache-ON dump measures agreement between two cache-ON runs, not "
                          f"correctness -- a broken expert cache that is broken the same way in "
                          f"both runs scores 100%. Produce a control with --no-expert-cache "
                          f"(budget 0) or LLAMA_EXPERT_CACHE_NOGATHER=1 and point --oracle-ref "
                          f"at that.")}
    return None


def _oracle_guard(ref_path, dump_path, model_file=None, allow_stale=False,
                  allow_model_mismatch=False, absolute=False):
    """Return an error record when the comparison would be meaningless, else None.

    Four independent preconditions, each of which has already produced a large and entirely
    spurious "the pool broke numerics" verdict on this box: the cap (ubatch shape), the source
    provenance (engine version), the weights identity (which model), and -- when `absolute` --
    whether the reference is a CONTROL at all (expert cache off).
    """
    if absolute:
        bad = _truth_guard(ref_path)
        if bad:
            return bad
    bad = _cap_guard(ref_path, dump_path)
    if bad:
        return bad
    bad = _model_guard(ref_path, model_file)
    if bad and not allow_model_mismatch:
        return bad
    if bad:
        mm = bad.get("model_mismatch")
        if mm:
            print(f"[oracle-guard] WARNING -- model mismatch accepted via "
                  f"--allow-model-mismatch: ref={(mm.get('ref') or {}).get('realpath')} vs "
                  f"now={(mm.get('candidate') or {}).get('realpath')}",
                  file=sys.stderr, flush=True)
        else:
            print("[oracle-guard] WARNING -- unknown model identity accepted via "
                  "--allow-model-mismatch", file=sys.stderr, flush=True)
    bad = _stamp_guard(ref_path)
    if bad and not allow_stale:
        return bad
    if bad:
        print(f"[oracle-guard] WARNING -- stale reference accepted via --allow-stale-oracle: "
              f"{bad.get('stale_kind', bad.get('stale_unknown'))}", file=sys.stderr, flush=True)
    return None


def _run_oracle_compare(a_path, b_path, label):
    """Run the M1/M2/M3 comparison of two dumps; return the report dict, or None on failure.

    Extracted so every caller (pool-size gate, its final re-compare, and the cap-invariance gate)
    goes through ONE comparison implementation. The metrics' definitions -- and the rule that M1
    and M2 are never merged -- live in cgc_logits_oracle_compare.py.
    """
    rep = os.path.join(RESULT_DIR, f"oraclecmp_{label}.json")
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "scripts", "check", "cgc_logits_oracle_compare.py"),
                    "--a", a_path, "--b", b_path, "--report", rep],
                   cwd=ROOT, capture_output=True)
    if not os.path.exists(rep):
        return None
    try:
        return json.load(open(rep, encoding="utf-8"))
    except Exception:  # noqa: BLE001 - harness
        return None


def oracle_gate(port, dump_path, ref_path, label, timeout, allow_stale=False,
                model_file=None, allow_model_mismatch=False, absolute=False):
    """Send the deterministic probe, then compare this combo's logits oracle against a
    reference using the TWO independent metrics (never merged).

    M1 numeric identity  = same row hashes (bit-identical logits)
    M2 decision agreement = same argmax token
    Pool sizes may only differ if M1 says so; 'M2 passed' alone does NOT mean 'no difference'.

    `absolute=True` marks the comparison as CORRECTNESS rather than INVARIANCE: the reference
    must be a control (expert cache off -- see _truth_guard), and the verdict is recorded as
    `comparison: "absolute"` so the two kinds of 19/19 can never be read as the same number
    again.
    """
    if not (dump_path and os.path.exists(dump_path)):
        return {"ok": None, "error": "no oracle dump produced"}
    if not (ref_path and os.path.exists(ref_path)):
        return {"ok": None, "error": f"reference oracle missing: {ref_path}"}
    bad = _oracle_guard(ref_path, dump_path, model_file, allow_stale, allow_model_mismatch,
                        absolute=absolute)
    if bad:
        if absolute:
            bad.setdefault("comparison", "absolute")
        return bad
    ask_probe(port, timeout)
    time.sleep(1.0)
    rep = os.path.join(RESULT_DIR, f"oraclecmp_{label}.json")
    d = _run_oracle_compare(ref_path, dump_path, label)
    if d is None:
        return {"ok": None, "error": "oracle compare produced no report"}
    m1, m2 = d["metrics"]["numeric_identity"], d["metrics"]["decision_agreement"]
    return {"ok": (m1["rate"] == 1.0 and m2["rate"] == 1.0),
            "m1_numeric_identity": f"{m1['equal']}/{m1['n']}",
            "m2_decision_agreement": f"{m2['equal']}/{m2['n']}",
            "cross_tab": d["cross_tab"],
            "comparison": "absolute" if absolute else "relative",
            "report": rep}


def oracle_recompare(dump_path, ref_path, label, allow_stale=False, model_file=None,
                     allow_model_mismatch=False, absolute=False):
    """Re-compare the COMPLETE dump and print the verdict; returns the report dict or None.

    `oracle_gate` deliberately runs first (its comment: the dump's early ubatches must line up
    with the reference run's). But the template and preflight probes that follow keep APPENDING
    to the same JSONL, so the comparison it performs measures only a PREFIX of the file -- on
    2026-09-13 it compared 53 of the eventual 103 step keys. That silently understates the gate
    instead of failing it. The finished file is a strict superset of what was compared, so
    re-comparing here is free and strictly stronger; no extra request is sent.
    """
    if not (dump_path and os.path.exists(dump_path) and ref_path
            and os.path.exists(ref_path)):
        return None
    if _oracle_guard(ref_path, dump_path, model_file, allow_stale, allow_model_mismatch,
                     absolute=absolute):
        return None
    rep = os.path.join(RESULT_DIR, f"oraclecmp_{label}_final.json")
    d = _run_oracle_compare(ref_path, dump_path, f"{label}_final")
    if d is None:
        return None
    m1, m2 = d["metrics"]["numeric_identity"], d["metrics"]["decision_agreement"]
    m3 = d["metrics"]["topk_set_agreement"]
    out = {"ok": (m1["rate"] == 1.0 and m2["rate"] == 1.0),
           "m1_numeric_identity": f"{m1['equal']}/{m1['n']}",
           "m2_decision_agreement": f"{m2['equal']}/{m2['n']}",
           "m3_topk_set_agreement": f"{m3['equal']}/{m3['n']}",
           "n_compared": m1["n"],
           "cross_tab": d["cross_tab"],
           "report": rep}
    print(f"[{label}] gate-oracle-final {'PASS' if out['ok'] else 'CHECK'}  "
          f"M1(bit-identical)={out['m1_numeric_identity']} "
          f"M2(argmax)={out['m2_decision_agreement']} "
          f"M3(topk)={out['m3_topk_set_agreement']} (n={out['n_compared']})", flush=True)
    return out


# ============================================================================================
# CAP INVARIANCE -- a first-class gate (2026-09-14)
#
# What it proves: the SAME pool, dumped at TWO different `CGC_POOL_MAX_TOKENS`, must produce the
# same logits. This is the "reduction order / pool layout independence" claim -- the cap changes
# the union width (cap x topk), therefore which experts share the pool and in which order they are
# summed, while the pool SIZE is held fixed. A pool size may only ever differ in speed; a cap
# change may not differ at all, because both caps are legal configurations of the same engine.
#
# Why it has to be a harness gate and not a hand-run: the failure it looks for is invisible in
# every cheaper signal. Measured 2026-09-14 by hand -- cap=5 on an 8 GiB pool was 13-15% FASTER
# than cap=8 and scored M1 3/82 against the cap=8 reference. Nothing in the speed number, the nil
# scan, the union-fit arithmetic or the template gate distinguishes that from a clean config; only
# the logits do. The earlier hand procedure also could not be repeated cheaply, so the +13% lever
# sat unused for want of a gate rather than for want of a measurement.
#
# ALIGNMENT (this is the part that makes it valid): the dump's `step` is a per-process counter of
# dump CALLS, i.e. of ubatches, and the cap clamps n_batch -- so the two caps split one prompt into
# a different number of ubatches and number the same sequence position DIFFERENTLY. Keying on
# (step, token_idx) across two caps would compare unrelated rows and manufacture a verdict out of
# chunking. `pmax` (llama_memory_seq_pos_max) is the absolute sequence position, so the gate
# realigns both dumps on it and keeps only single-token (decode) rows -- in a multi-token dump
# every row carries the same ubatch-wide pmax, so no per-position key exists for them.
# ============================================================================================

CAP_INVARIANCE_DEFAULT_CAPS = "6,8"


def _oracle_rows(path):
    """Count usable rows in a dump (a row is a non-empty JSON line)."""
    n = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    return n


def _realign_decode(dump, out_path, only_ctx=None):
    """Rewrite a dump so rows are keyed by the SEQUENCE POSITION of the token they describe.

    Why the raw keys cannot be used across two caps, in three parts -- each one measured while
    building this:

      1. `step` is a per-process counter of dump CALLS, i.e. of ubatches, and the cap clamps
         n_batch. Two caps therefore split one prompt into a different number of ubatches and
         number the same position differently.
      2. Dropping multi-token rows ("only compare decode steps") does not work either: with MTP on
         the base context runs VERIFY batches, so the bulk of its rows have n_tokens == 1+n_max.
         Measured on the existing 8 GiB cap=8 dump: 117 rows -> 21 "decode" rows, of which only 2
         were the base model's. The comparison would have been of the draft head, not the model.
      3. Positions repeat across REQUESTS (dump_seq never resets), so position alone collides.

    The fix: in a causal batch the rows cover contiguous positions ending at `pmax` (= the max
    sequence position after the ubatch), so row `token_idx` of an n-token dump sits at
    `pmax - (n-1) + token_idx`. That is a per-token position, valid for prefill chunks, single
    decode steps and verify batches alike.

    The second key slot is the OCCURRENCE index of that position (`token_idx` in the comparison
    script's key), NOT a request number. Two things make a position repeat inside one dump, both
    of them normal: a second request restarts at position 0, and an MTP draft rejection ROLLS
    BACK the memory so the same position is recomputed. An earlier version tried to detect
    request boundaries from a decreasing position -- that is unsound here, because how many
    rollbacks happen depends on the draft accept pattern, i.e. on the very numerics under test,
    so the two dumps would have been split into different numbers of "requests" (measured: 6 vs
    8) and only part of each would have aligned. Counting occurrences instead is canonical: for
    two dumps running the same requests in the same order, the k-th computation of position p
    aligns with the k-th computation of position p, whatever the chunking.

    Returns a stats dict, not a bare count, because how many rows were KEPT is part of the claim:
    a gate that silently compared three steps would pass vacuously.
    """
    stats = {"kept": 0, "bad_json": 0, "rows": 0, "ctx": {}, "repeat_rows": 0,
             "dropped_ctx": 0, "first_pos": None, "last_pos": None}
    occ = {}
    try:
        fin = open(dump, encoding="utf-8")
    except OSError:
        return stats
    with fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            if obj.get("pmax") is None:
                continue
            ctx = obj.get("ctx_type", "DEF")
            if only_ctx is not None and ctx != only_ctx:
                stats["dropped_ctx"] += 1
                continue
            n_tok = max(1, int(obj.get("n_tokens", 1)))
            idx = int(obj.get("token_idx", 0))
            pos = int(obj["pmax"]) - (n_tok - 1) + idx
            k = occ.get((ctx, pos), 0)
            occ[(ctx, pos)] = k + 1
            if k > 0:
                stats["repeat_rows"] += 1
            obj["step"] = pos
            obj["token_idx"] = k
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["ctx"][ctx] = stats["ctx"].get(ctx, 0) + 1
            if stats["first_pos"] is None or pos < stats["first_pos"]:
                stats["first_pos"] = pos
            stats["last_pos"] = pos
    return stats


def _first_divergence(a_path, b_path):
    """Smallest aligned position where two realigned dumps differ at the M1 (bits) level.

    This is the actionable half of a cap verdict: 'the logits differ' says something is wrong,
    'they first differ at position 12 of request 0, argmax 1234 -> 1235' says where to look.
    """
    def load(path):
        out = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    out[(int(o["step"]), int(o.get("token_idx", 0)), o.get("ctx_type", "DEF"))] = o
        except (OSError, json.JSONDecodeError):
            pass
        return out
    a, b = load(a_path), load(b_path)
    for k in sorted(set(a) & set(b)):
        if a[k].get("row_fnv1a64") != b[k].get("row_fnv1a64"):
            return {"pos": k[0], "occurrence": k[1], "ctx": k[2],
                    "argmax_a": a[k].get("argmax_token"), "argmax_b": b[k].get("argmax_token"),
                    "sum_a": a[k].get("sum"), "sum_b": b[k].get("sum")}
    return None


def step_partition(log_path, il=0):
    """Per-forward token counts actually used by the prefill, from the CGC-IDS hook.

    This exists because a cap-invariance FAIL is otherwise ambiguous. `CGC_POOL_MAX_TOKENS` is
    clamped onto `n_batch`, and `n_ubatch` follows from it, so the cap does not merely relabel
    pool slots -- it changes HOW MANY TOKENS EACH FORWARD PASS PROCESSES. Measured 2026-09-14
    (docs/CAP_INVARIANCE_LOCALIZATION_2026-09-14.md): three arms that differ only in that
    partition (cap5 -> 5,5,5...; cap8 -> 8,8,8...; cap8 with `-ub 5` -> 5,3,5,3...) produce three
    DIFFERENT completions, while the same cap across pools 2/4/6/8/10 GiB is bit-identical.
    So the partition is the variable that moves the numerics, and recording it turns a bare FAIL
    into a statement about which knob actually changed.

    Only the prefill matters here: the decode tail is single-token in every arm by construction.
    Returns the run-length-compressed sequence plus its length.
    """
    import re
    pat = re.compile(r"CGC-IDS: ctx=(\S+) pmax=(\d+) il=(\d+) ntok=(\d+) ")
    seq = []
    try:
        with open(log_path, errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if not m or int(m.group(3)) != il:
                    continue
                step = (int(m.group(2)), int(m.group(4)))
                if not seq or seq[-1] != step:
                    seq.append(step)
    except OSError:
        return {"available": False, "reason": "log not readable"}
    if not seq:
        return {"available": False, "reason": "no CGC-IDS rows (CGC_UNION_LOG off?)"}
    pre = []
    for _pmax, ntok in seq:
        if ntok == 1:
            break
        pre.append(ntok)
    runs = []
    for n in pre:
        if runs and runs[-1][0] == n:
            runs[-1][1] += 1
        else:
            runs.append([n, 1])
    return {"available": True, "prefill_ntok": pre, "prefill_steps": len(pre),
            "compressed": ",".join(f"{n}x{c}" for n, c in runs),
            "largest_step": (max(pre) if pre else None)}


def _capinv_guard(a_path, b_path, model_file=None):
    """Preconditions for comparing two dumps taken at DIFFERENT caps.

    Deliberately NOT `_oracle_guard`: that one refuses a cap mismatch, which is exactly the
    variable under test here. What must still hold is everything that would make the verdict
    describe something other than the cap -- the same engine code (stamp) and the same weights
    (model), checked on BOTH dumps, because one stale side poisons the pair as thoroughly as two.
    """
    for path in (a_path, b_path):
        bad = _stamp_guard(path)
        if bad:
            return {"ok": None, "stale": True, "which": os.path.basename(path), "error": bad["error"]}
        bad = _model_guard(path, model_file)
        if bad:
            return {"ok": None, "model_mismatch": True, "which": os.path.basename(path),
                    "error": bad["error"]}
    return None


def cap_invariance_dump(kind, gb, cap, args, occ=0):
    """Produce (or legitimately reuse) one oracle dump at exactly one cap, one pool size.

    `occ` is the occurrence of this cap within the requested pair list: non-zero only when the
    caller asks for the same cap twice (`--cap-invariance-caps 8,8`), which is the gate's own
    DETERMINISM control -- two independent runs at one cap must agree bit-for-bit, otherwise an
    M1 failure between two different caps cannot be attributed to the cap at all.

    Reuse is allowed only when the dump's OWN sidecar proves it is still this computation: same
    cap, same engine stamp, same weights. A cached dump is a claim about the past, so it is
    trusted only as far as its provenance can be checked -- otherwise the gate would report last
    night's verdict about last night's binary.
    """
    label = f"capinv_{kind}_pool{gb}gb_cap{cap}" + (f"_r{occ}" if occ else "")
    dump = os.path.join(RESULT_DIR, f"oracle_{label}.jsonl")
    launch_log = os.path.join(RESULT_DIR, f"launch_{label}.log")
    model_file = model_path(kind)
    rec = {"label": label, "model": kind, "pool_gb": gb, "cap": cap, "dump": dump}
    extra = list(args.extra_env) + ["CGC_UNION_LOG=1", f"CGC_POOL_MAX_TOKENS={cap}"]
    # The MTP flag reaches run_server.sh as CGC_SERVER_MTP, which launch() sets and then lets
    # extra_env override -- so an MTP-off run is expressed as `--extra-env CGC_SERVER_MTP=0`. It
    # belongs in the signature because the MTP context adds its own dumps and changes the base
    # context's batch shape (verify batches instead of single-token decodes).
    launch_sig = {"extra_env": sorted(extra), "mtp": MODELS[kind]["mtp"]}

    if os.path.exists(dump) and _oracle_rows(dump) > 0 and not args.force:
        stale = _stamp_guard(dump)
        bad_model = _model_guard(dump, model_file)
        same_cap = _cap_sidecar(dump) == str(cap)
        prev_sig = (_oracle_meta(dump) or {}).get("launch")
        same_launch = prev_sig == launch_sig
        if same_cap and same_launch and stale is None and bad_model is None:
            rows = _oracle_rows(dump)
            print(f"[capinv] {kind} pool{gb}GB cap={cap}: reusing existing dump ({rows} rows; "
                  f"cap/stamp/weights verified)", flush=True)
            rec.update({"up": True, "reused": True, "rows": rows})
            # The partition is a property of the LAUNCH, so it survives from the cached run's log.
            if os.path.exists(launch_log):
                rec["partition"] = step_partition(launch_log)
            return rec
        why = ([f"cached cap={_cap_sidecar(dump)}"] if not same_cap else [])
        if not same_launch:
            why.append(f"cached launch={prev_sig} vs {launch_sig}")
        if stale is not None:
            why.append(stale.get("stale_kind", "stale"))
        if bad_model is not None:
            why.append("different weights")
        print(f"[capinv] {kind} pool{gb}GB cap={cap}: re-dumping ({'; '.join(why)})", flush=True)

    if args.attach:
        rec.update({"up": False, "skipped": "attach_cannot_set_cap",
                    "error": "--attach cannot set CGC_POOL_MAX_TOKENS: the cap is decided at "
                             "launch, so a cap-invariance pair needs two launches."})
        return rec

    if os.path.exists(dump):
        os.remove(dump)
    os.path.exists(dump + ".cap") and os.remove(dump + ".cap")
    extra += [f"CGC_LOGITS_ORACLE_DUMP={dump}", "CGC_LOGITS_ORACLE_TOPN=8"]

    # PRE-LAUNCH FEASIBILITY: an infeasible cap is exactly the case this gate exists for. Without
    # it, asking for cap 5 on a pool whose cap_max is 4 would launch, abort or fall back, and the
    # resulting dump would be compared as if it described cap 5.
    feas_ok, g_feas = feasibility_gate(kind, gb, args, cap, extra, where="capinv")
    if not feas_ok:
        rec.update({"up": False,
                    "skipped": "infeasible_" + g_feas["verdict"].lower().replace("-", "_"),
                    "feasibility": g_feas,
                    "error": g_feas["why"]})
        return rec

    kill_servers()
    if running_servers():
        rec.update({"up": False, "skipped": "foreign_server",
                    "error": f"another process owns the port: {running_servers()}"})
        return rec
    free_before = mem_free_pct()
    if free_before < args.min_free_pct:
        rec.update({"up": False, "skipped": "low_memory", "free_before_pct": free_before,
                    "error": f"machine {free_before}% free < --min-free-pct {args.min_free_pct}%"})
        return rec

    pre = tuple(running_servers())
    print(f"[capinv] {kind} pool{gb}GB cap={cap}: launching {MODELS[kind]['file']} mtp="
          f"{MODELS[kind]['mtp']} free={free_before}%", flush=True)
    launch(model_file, MODELS[kind]["mtp"], launch_pool_bytes(args, gb), args.port, args.kwargs,
           extra, launch_log)
    took = wait_health(args.port, model_file, args.health_timeout, pre_pids=pre)
    if took is None:
        tail = ""
        try:
            tail = open(launch_log, "rb").read().decode("utf-8", "replace")[-800:]
        except Exception:  # noqa: BLE001
            pass
        kill_servers()
        rec.update({"up": False, "launch_tail": tail,
                    "error": "server died, timed out, or a foreign server owns the port"})
        return rec

    facts = read_launch_facts(launch_log)
    # Real traffic: the dump only contains rows for ubatches that actually ran.
    ask_probe(args.port, timeout=min(args.timeout, 240.0))
    time.sleep(1.0)
    _write_oracle_meta(dump, cap, model_file=model_file, launch_sig=launch_sig,
                       expert_cache=expert_cache_state(extra, launch_pool_bytes(args, gb)))
    rows = _oracle_rows(dump)
    nil = scan_nil([launch_log, facts.get("log_path")])
    # Record the PREFILL STEP PARTITION, not just the requested cap: the cap's real effect on the
    # numerics is via n_batch/n_ubatch (see step_partition), so the runs that differ only in
    # pool size but share a partition are the ones whose comparison is meaningful.
    part = step_partition(facts.get("log_path") or launch_log)
    rec.update({"up": True, "reused": False, "rows": rows, "health_s": round(took, 1),
                "rss_gb": rss_gb(), "nil": nil, "launch_facts": facts, "partition": part})
    print(f"[capinv] {kind} pool{gb}GB cap={cap}: healthy in {took:.0f}s  rows={rows}  "
          f"rss={rec['rss_gb']}GB  nil={'ok' if nil.get('ok') else 'FAIL'}  "
          f"partition={part.get('compressed', 'n/a')}", flush=True)
    kill_servers()
    return rec


def cap_invariance_gate(kind, gb, caps, args):
    """Same pool, every pair of caps: dump each, realign on position, compare M1/M2/M3.

    The verdict never merges M1 and M2 (see cgc_logits_oracle_compare.py): 'M2 agreed' is not 'no
    difference', and this gate exists precisely because greedy decoding is chaotic enough that a
    layout-induced drift can sit at 100% decision agreement right up to the token where it does
    not.
    """
    label = f"capinv_{kind}_pool{gb}gb"
    out_json = os.path.join(RESULT_DIR, f"{label}.json")
    rec = {"label": label, "model": kind, "pool_gb": gb, "caps": list(caps),
           "dumps": [], "pairs": []}

    seen_caps = {}
    plan = []
    for cap in caps:
        plan.append((cap, seen_caps.get(cap, 0)))
        seen_caps[cap] = seen_caps.get(cap, 0) + 1
    dumps = [cap_invariance_dump(kind, gb, cap, args, occ) for cap, occ in plan]
    rec["dumps"] = dumps
    for d in dumps:
        if not d.get("up"):
            rec.update({"ok": None, "verdict": "INCOMPLETE",
                        "error": f"cap={d['cap']} produced no dump ({d.get('error')})"})
            json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            print(f"[{label}] INCOMPLETE: cap={d['cap']} produced no dump - {d.get('error')}",
                  flush=True)
            return rec

    model_file = model_path(kind)
    for i in range(len(dumps)):
        for j in range(i + 1, len(dumps)):
            a, b = dumps[i], dumps[j]
            pair = {"cap_a": a["cap"], "cap_b": b["cap"], "a": a["dump"], "b": b["dump"]}
            guard = _capinv_guard(a["dump"], b["dump"], model_file)
            if guard:
                pair.update({"ok": None, "verdict": "REFUSED", "error": guard["error"]})
                rec["pairs"].append(pair)
                continue
            plabel = f"{label}_cap{a['cap']}v{b['cap']}"
            ra = os.path.join(RESULT_DIR, f"realign_{a['label']}.jsonl")
            rb = os.path.join(RESULT_DIR, f"realign_{b['label']}.jsonl")
            rad = os.path.join(RESULT_DIR, f"realign_{a['label']}_def.jsonl")
            rbd = os.path.join(RESULT_DIR, f"realign_{b['label']}_def.jsonl")
            pair["realign"] = {"a": _realign_decode(a["dump"], ra),
                               "b": _realign_decode(b["dump"], rb),
                               "a_def": _realign_decode(a["dump"], rad, only_ctx="DEF"),
                               "b_def": _realign_decode(b["dump"], rbd, only_ctx="DEF")}
            pair["realigned"] = {"a": ra, "b": rb, "a_def": rad, "b_def": rbd}
            # TWO comparisons, and the verdict comes from the DEF-only one:
            #   DEF = the target model's own context. This is what the cap is supposed to be
            #         inert to, so it is the primary claim.
            #   all = every context together, draft head (MTP) included. Reported alongside because
            #         the draft head has its own layout sensitivity, but it must not be allowed to
            #         speak for the model -- a cap change that only moves the draft head is a
            #         different finding from one that moves the target.
            d_def = _run_oracle_compare(rad, rbd, f"{plabel}_def")
            d_all = _run_oracle_compare(ra, rb, f"{plabel}_all")
            if d_def is None:
                pair.update({"ok": None, "verdict": "ERROR", "error": "compare produced no report"})
                rec["pairs"].append(pair)
                continue
            m1 = d_def["metrics"]["numeric_identity"]
            m2 = d_def["metrics"]["decision_agreement"]
            m3 = d_def["metrics"]["topk_set_agreement"]
            n_common, n_a, n_b = d_def["n_common"], d_def["n_a"], d_def["n_b"]
            denom = min(n_a, n_b) if min(n_a, n_b) else 0
            coverage = (n_common / denom) if denom else 0.0
            m1_full, m2_full = m1["rate"] == 1.0, m2["rate"] == 1.0
            # Verdict, in the order the evidence deserves to be weighed:
            #
            #  * DIVERGENCE stands on its own. A non-full M1 is positive evidence that the cap
            #    moved the numerics, and it is not weakened by imperfect alignment -- imperfect
            #    alignment is in fact its CONSEQUENCE: once a token differs, the two generations
            #    diverge and their later positions stop overlapping. (An earlier version required
            #    >=95% coverage for any verdict, which made the gate silent in exactly the case it
            #    exists for: measured 94% coverage on a pair whose M1 was 2/29.)
            #  * INERTNESS is the claim that needs coverage and sample size: 'the cap changes
            #    nothing' is only supported if the full aligned sequence was actually compared.
            #    So the thresholds gate the PASS direction only, where the degenerate failure mode
            #    is a vacuous pass on almost nothing.
            if not m1_full:
                verdict, ok = ("FAIL", False) if not m2_full else ("DRIFT", False)
            elif n_common < args.capinv_min_steps or coverage < args.capinv_min_coverage:
                verdict, ok = "INCONCLUSIVE", None
            else:
                verdict, ok = "PASS", True
            pair["first_divergence"] = _first_divergence(rad, rbd)
            # WHICH KNOB MOVED? A FAIL here is only informative once this is answered, because a
            # cap change does two things at once: it re-labels pool slots AND (via the n_batch
            # clamp) it changes the prefill step partition. Measured 2026-09-14: at a FIXED cap the
            # result is bit-identical across pools 2/4/6/8/10 GiB; across caps the partition
            # changes and so does the output. So when the partitions differ, this pair is
            # comparing two different forward passes rather than two pool layouts.
            part_a, part_b = a.get("partition") or {}, b.get("partition") or {}
            both_part = bool(part_a.get("available")) and bool(part_b.get("available"))
            same_part = both_part and part_a.get("compressed") == part_b.get("compressed")
            if not both_part:
                attribution = "unknown: prefill step partition not recorded for both arms"
            elif same_part:
                attribution = ("pool layout / slab geometry: prefill step partition is IDENTICAL "
                               f"({part_a.get('compressed')}), so the divergence is NOT the partition")
            else:
                attribution = ("prefill step partition: "
                               f"cap{pair['cap_a']}={part_a.get('compressed')} vs "
                               f"cap{pair['cap_b']}={part_b.get('compressed')} -- the two caps "
                               "chunk the prompt differently, so this compares two forward passes")
            pair["partition"] = {"a": part_a.get("compressed"), "b": part_b.get("compressed"),
                                 "a_steps": part_a.get("prefill_steps"),
                                 "b_steps": part_b.get("prefill_steps"), "same": same_part}
            pair["attribution"] = attribution
            pair.update({"ok": ok, "verdict": verdict, "ctx": "DEF", "n_common": n_common,
                         "coverage": round(coverage, 4),
                         "repeats": {"a": pair["realign"]["a_def"].get("repeat_rows"),
                                     "b": pair["realign"]["b_def"].get("repeat_rows")},
                         "pos_range": {"a": [pair["realign"]["a_def"].get("first_pos"),
                                             pair["realign"]["a_def"].get("last_pos")],
                                       "b": [pair["realign"]["b_def"].get("first_pos"),
                                             pair["realign"]["b_def"].get("last_pos")]},
                         "rows": {"a": pair["realign"]["a_def"].get("kept"),
                                  "b": pair["realign"]["b_def"].get("kept")},
                         "m1_numeric_identity": f"{m1['equal']}/{m1['n']}",
                         "m2_decision_agreement": f"{m2['equal']}/{m2['n']}",
                         "m3_topk_set_agreement": f"{m3['equal']}/{m3['n']}",
                         "cross_tab": d_def["cross_tab"],
                         "report": os.path.join(RESULT_DIR, f"oraclecmp_{plabel}_def.json"),
                         "diff_examples": d_def.get("diff_examples", [])[:3]})
            if d_all is not None:
                am1 = d_all["metrics"]["numeric_identity"]
                am2 = d_all["metrics"]["decision_agreement"]
                am3 = d_all["metrics"]["topk_set_agreement"]
                pair["all_ctx"] = {"n_common": d_all["n_common"],
                                  "m1_numeric_identity": f"{am1['equal']}/{am1['n']}",
                                  "m2_decision_agreement": f"{am2['equal']}/{am2['n']}",
                                  "m3_topk_set_agreement": f"{am3['equal']}/{am3['n']}",
                                  "m1_full": am1["rate"] == 1.0, "m2_full": am2["rate"] == 1.0}
            rec["pairs"].append(pair)

    verdicts = [p.get("verdict") for p in rec["pairs"]]
    rec["ok"] = all(p.get("ok") is True for p in rec["pairs"]) if rec["pairs"] else None
    rec["verdict"] = ("PASS" if rec["ok"] else
                      ("INCONCLUSIVE" if all(v in ("INCONCLUSIVE", "REFUSED", "ERROR")
                                             for v in verdicts) else
                       ("DRIFT" if "DRIFT" in verdicts else "FAIL")))
    json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    for p in rec["pairs"]:
        if p.get("verdict") in ("PASS", "DRIFT", "FAIL"):
            ac = p.get("all_ctx") or {}
            fd = p.get("first_divergence") or {}
            where = (f"  first divergence at pos={fd['pos']} occ={fd['occurrence']} "
                     f"({fd['ctx']}) argmax {fd['argmax_a']} -> {fd['argmax_b']}" if fd else "")
            print(f"[{label}] cap{p['cap_a']} vs cap{p['cap_b']}: {p['verdict']}  "
                  f"DEF M1(bit-identical)={p['m1_numeric_identity']} "
                  f"M2(argmax)={p['m2_decision_agreement']} "
                  f"M3(topk)={p['m3_topk_set_agreement']} "
                  f"n={p['n_common']} coverage={p['coverage']:.1%}  | "
                  f"ALL(ctx) M1={ac.get('m1_numeric_identity', '-')} "
                  f"M2={ac.get('m2_decision_agreement', '-')} n={ac.get('n_common', '-')}"
                  f"{where}", flush=True)
            if p.get("verdict") in ("FAIL", "DRIFT"):
                print(f"[{label}]   attribution: {p['attribution']}", flush=True)
        elif p.get("verdict") == "INCONCLUSIVE" and p.get("m1_numeric_identity"):
            # Report the numbers even when the pair cannot be judged: "too few steps" is a
            # statement about the ALIGNMENT, not about the logits, and hiding the observed M1/M2
            # would make an unmatched pair indistinguishable from a clean one.
            print(f"[{label}] cap{p['cap_a']} vs cap{p['cap_b']}: INCONCLUSIVE "
                  f"(aligned DEF steps n={p['n_common']} of {p.get('rows', {}).get('a')}/"
                  f"{p.get('rows', {}).get('b')} realigned rows; coverage "
                  f"{p['coverage']:.0%} < {args.capinv_min_coverage:.0%} or n < "
                  f"{args.capinv_min_steps}) -- INDICATIVE ONLY, not a verdict: "
                  f"M1={p['m1_numeric_identity']} M2={p['m2_decision_agreement']}. The two dumps "
                  f"did not run the same requests in the same order, so their positions overlap "
                  f"only in part; check pos_range/repeats in the record.",
                  flush=True)
        else:
            print(f"[{label}] cap{p['cap_a']} vs cap{p['cap_b']}: {p.get('verdict')} - "
                  f"{p.get('error', '')}", flush=True)
    print(f"[{label}] VERDICT {rec['verdict']}  (wrote {out_json})", flush=True)
    return rec


def _store_capinv(records):
    """Merge this run's cap-invariance records into the on-disk store, keyed by (label, caps).

    Accumulating rather than overwriting is not a convenience: the natural workflow is to re-run
    `--cap-invariance-caps 8,8` as a determinism control right after a 6-vs-8 run, and with a
    single-record file that control silently deletes the FAIL it was supposed to qualify.
    Re-running the SAME (label, caps) still replaces that entry, so a run stays re-runnable.
    """
    path = os.path.join(RESULT_DIR, "cap_invariance.json")
    store = []
    if os.path.exists(path):
        try:
            prev = json.load(open(path, encoding="utf-8"))
            store = prev if isinstance(prev, list) else (prev.get("runs") or [])
        except Exception:  # noqa: BLE001 - harness
            store = []
    keep = []
    for r in store:
        replaced = any(r.get("label") == n.get("label")
                       and tuple(r.get("caps") or ()) == tuple(n.get("caps") or ())
                       for n in records)
        if not replaced:
            keep.append(r)
    out = keep + list(records)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return out


def print_capinv_table(records):
    """Consolidated cap-invariance verdicts, one row per (model x pool x cap pair)."""
    print()
    print("=" * 118)
    print("CAP INVARIANCE -- same pool, different CGC_POOL_MAX_TOKENS: a cap change must be "
          "numerically inert")
    print("-" * 118)
    print("%-26s %5s %5s %5s %8s %11s %11s %11s %9s %11s" % (
        "config", "pool", "capA", "capB", "verdict", "M1 DEF", "M2 DEF", "M3 DEF",
        "cover", "M1 all-ctx"))
    print("-" * 118)
    for r in records:
        if not r.get("pairs"):
            print("%-26s   %s" % (r.get("label"), r.get("error") or "no pair compared"))
            continue
        for p in r["pairs"]:
            ac = p.get("all_ctx") or {}
            print("%-26s %4sG %5s %5s %8s %11s %11s %11s %9s %11s" % (
                r.get("label"), r.get("pool_gb"), p.get("cap_a"), p.get("cap_b"),
                p.get("verdict"), p.get("m1_numeric_identity", "-"),
                p.get("m2_decision_agreement", "-"), p.get("m3_topk_set_agreement", "-"),
                (f"{p['coverage']:.0%}" if isinstance(p.get("coverage"), float) else "-"),
                ac.get("m1_numeric_identity", "-")))
    print("-" * 118)
    print("DEF = the target model's own context (the primary claim); all-ctx includes the MTP")
    print("      draft head, whose layout sensitivity is a separate finding.")
    print("PASS         = M1 full: the cap changed the pool LAYOUT and the logits did not move.")
    print("DRIFT        = M1 not full but M2 full: same decisions, different bits. NOT a pass --")
    print("               greedy decoding is chaotic, so this can flip at any later token.")
    print("FAIL         = M1 and M2 both differ.")
    print("INCONCLUSIVE = too few position-aligned decode steps to judge (see 'cover').")
    print("cover        = aligned positions / smaller dump. Alignment is by sequence position,")
    print("               because the dump's own step counter counts UBATCHES and the cap decides")
    print("               the chunking -- keying on it across two caps compares unrelated rows.")
    print("=" * 118)


def ask_probe(port, timeout=180.0):
    payload = {"model": "local", "messages": [{"role": "user", "content": PROBE_PROMPT}],
               "temperature": 0.0, "max_tokens": 48}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except Exception:  # noqa: BLE001 - harness
        pass


def launch_pool_bytes(args, gb):
    """Expert-cache budget, in bytes, that `launch()` should hand to run_server.sh.

    `--no-expert-cache` is the ground-truth arm: budget 0 means `-expert-cache 0`, which leaves
    `model->expert_index` empty, so no cache object is created and no hook is installed -- the
    stock full-resident forward pass. Every other launch keeps the pool at `gb` GiB.
    """
    if getattr(args, "no_expert_cache", False):
        return 0
    try:
        return int(gb) * 1024 ** 3
    except (TypeError, ValueError):
        return 0


def launch(model_file, mtp, pool_bytes, port, kwargs, extra_env, log_path):
    """Start run_server.sh in its own session so it survives this driver dying.

    `model_file` is an ABSOLUTE path (see model_path) -- run_server.sh accepts a model outside
    its MODEL_ROOT, which is what lets the external-drive IQ3 be measured without a 13 GB copy.
    """
    env = dict(os.environ)
    env.update({
        "CGC_DETACHED": "1",
        "_CGC_DETACHED_MARKER": "1",          # stop run_server.sh from forking again
        "CGC_SERVER_MODEL": model_file,
        "CGC_SERVER_EXPERT_CACHE_BYTES": str(pool_bytes),
        "CGC_SERVER_PORT": str(port),
        "CGC_SERVER_MTP": mtp,
    })
    if kwargs is not None:
        env["CGC_SERVER_CHAT_TEMPLATE_KWARGS"] = kwargs
    for kv in extra_env or []:
        k, _, v = kv.partition("=")
        env[k.strip()] = v.strip()
    log = open(log_path, "wb")
    subprocess.Popen(["bash", "scripts/run_server.sh"], cwd=ROOT, env=env,
                     stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                     start_new_session=True)


def server_procs():
    """[(pid, cmdline)] for real llama-server processes (see running_servers for the filter)."""
    out = []
    for p in sh("pgrep", "-f", SERVER_MATCH).split():
        if not p.strip():
            continue
        cmd = sh("ps", "-o", "command=", "-p", p).strip()
        if cmd and cmd.split(" ", 1)[0].endswith(SERVER_MATCH):
            try:
                out.append((int(p), cmd))
            except ValueError:
                pass
    return out


def port_owner(port):
    """PID holding a socket on `port`, or None when it is free / ambiguous."""
    pids = sorted({p for p in sh("lsof", "-ti", f":{port}").split() if p.strip()})
    if len(pids) != 1:
        return None
    try:
        return int(pids[0])
    except ValueError:
        return None


def wait_health(port, model_file, timeout, grace=60.0, pre_pids=()):
    """Seconds until OUR server answers /health, or None if it never came up.

    "No server process yet" is NOT death: `run_server.sh` needs seconds before its child is
    exec'd, and an early poll would abort the wait immediately -- which once made every combo
    report "server died" while the server was still loading.

    Identity is taken from the COMMAND LINE (exact model path + --port), never from a pid.
    Two measured failures forced that: `run_server.sh` prints `[detach] server PID=N` where N is
    not always the server's own pid, so a pid check rejected our own healthy servers; and on a
    shared checkout another agent respawns servers, so a FOREIGN already-loaded server can
    answer /health seconds after our launch and silently be measured instead of ours.
    """
    abs_model = model_file            # absolute; see launch()
    pre = set(pre_pids)
    t0 = time.time()
    seen = False
    while time.time() - t0 < timeout:
        procs = server_procs()
        ours = [p for p, c in procs if abs_model in c and f"--port {port}" in c]
        # A foreign llama-server anywhere means we cannot attribute /health to our config.
        foreign = [p for p, _ in procs if p not in ours]
        owner = port_owner(port)
        if not foreign and len(ours) == 1 and owner == ours[0] and ours[0] not in pre \
                and health(port):
            # Four independent facts: nobody else is loaded, exactly one process matches our
            # model AND port, that process OWNS the port, and it is newer than our launch.
            # Command-line matching alone was not enough on a shared checkout: the other agent
            # runs this same model on this same port, and a probe accepted their already-loaded
            # server 10 s after launch while our launch log still said "模型載入中".
            return time.time() - t0
        if procs:
            seen = True
        elif seen or (time.time() - t0) > grace:
            return None
        time.sleep(5)
    return None


FACT_PATTERNS = ((r"\[chat\]\s+model_kind=(\S+)", "model_kind"),
                 (r"\[chat\]\s+template_file=(\S+)", "template_file"),
                 (r"\[chat\]\s+template_kwargs=(\S+)", "template_kwargs"),
                 (r"\[chat\]\s+profile=(\S+)", "profile"),
                 (r"-expert-cache\s+(\d+)", "budget_bytes"),
                 (r"\[start\].*budget=(\d+)B", "budget_bytes"),
                 (r"Soft Pool init: L0=(\d+) L1=(\d+) \(n_slots=(\d+)", "pool"),
                 (r"n_slots=(\d+)", "n_slots"),
                 (r"LAYER_CAPS per-layer caps: total (\d+) slots", "total_slots"),
                 (r"min (\d+)/layer", "min_layer_slots"),
                 (r"\[guard\]\s+memory_mode=(\S+) class=(\S+)", "guard"))


def _read_text(path):
    try:
        return open(path, "rb").read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - harness
        return ""


def read_launch_facts(log_path):
    """Pull the actually-chosen template/model_kind/pool geometry out of the launch log.

    Two logs, one fact set. `run_server.sh` writes the [chat]/[log] headers to its own stdout,
    while the C++ side (n_slots, LAYER_CAPS min) goes to the log file it names in `[log]`. Both
    are needed: the pool path gate is arithmetic on those C++ numbers, so missing them made the
    union-fit gate answer "unknown" for every combo.

    The `[log]` path is cut at the first non-ASCII byte: the line ends with a Chinese
    parenthetical, and a naive `\S+` swallowed it, so the nil scan opened a nonexistent file and
    reported a clean `[]` forever.
    """
    facts = {}
    txt = _read_text(log_path)
    m = re.search(r"\[log\]\s+(\S+)", txt)
    if m:
        facts["log_path"] = re.split(r"[^\x20-\x7e]", m.group(1))[0]

    for source in (txt, _read_text(facts.get("log_path", ""))):
        if not source:
            continue
        for pat, key in FACT_PATTERNS:
            if key in facts:
                continue
            mm = re.search(pat, source)
            if mm:
                facts[key] = mm.groups() if len(mm.groups()) > 1 else mm.group(1)
    return facts


# ---------------------------------------------------------------------------------------------
# PRE-LAUNCH FEASIBILITY GATE (docs/POOL_FEASIBILITY_MATRIX_2026-09-14.html)
#
# A cell whose `cap_max` sits below the reference cap can only be measured at a SMALLER cap -- and
# the cap decides the prefill step partition. Comparing such a cell against a reference-cap cell
# therefore compares two different forward passes, not two pool layouts, while the resulting number
# looks exactly like a pool result. Measured 2026-09-14, one build: cap 8 across pools 4/6/8 GiB is
# bit-identical (M1 117/117) AND cap 6 vs cap 8 in that same batch is M1 3/103.
#
# Refusing at launch is the only place that mistake can be caught before it becomes a table row.
# The arithmetic itself lives in scripts/check/pool_feasibility.py (one definition, also usable
# standalone with `--validate` against the measured cap-probe records); this file only decides
# WHICH cell is being launched, because the cap / MTP / spec-n-max / layer-caps all arrive from
# the environment, and the env-merge order here mirrors `launch()`'s exactly.
# ---------------------------------------------------------------------------------------------
_FEAS_MOD = None
_FEAS_GEOM: dict = {}


def _feasibility():
    """`pool_feasibility` holds the arithmetic; import it lazily.

    It loads THIS file to read MODELS, so a module-level import would be circular. Loading it by
    path under its own name is what its CLI already does.
    """
    global _FEAS_MOD
    if _FEAS_MOD is None:
        import importlib.util
        path = os.path.join(ROOT, "scripts", "check", "pool_feasibility.py")
        spec = importlib.util.spec_from_file_location("cgc_pool_feasibility", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FEAS_MOD = mod
    return _FEAS_MOD


def pool_geometry(kind):
    """GGUF tensor-table geometry for one configured model, cached on (path, size, mtime).

    Tensor table only -- no weights are read and no server is started, so this costs milliseconds
    even for a 16 GB GGUF. Keyed on the file's identity so a re-quantised file at the same path is
    not described by the previous geometry.
    """
    path = model_path(kind)
    try:
        st = os.stat(path)
    except OSError as e:
        raise RuntimeError(f"{path}: {e}") from e
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if _FEAS_GEOM.get("key") != key:
        _FEAS_GEOM.clear()
        _FEAS_GEOM["key"] = key
        _FEAS_GEOM["geom"] = _feasibility().gguf_geometry(path)
    return _FEAS_GEOM["geom"]


def effective_launch_env(kind, extra_env):
    """The knobs run_server.sh will actually see, merged the way `launch()` merges them.

    Only CGC_*/LLAMA_EXPERT_CACHE_* matter here: those are what move the pool geometry and the
    batch shape. `extra_env` wins over the ambient environment because `launch()` sets its own
    values first and then applies `extra_env` on top -- which is how a caller expresses "MTP off"
    (`--extra-env CGC_SERVER_MTP=0`).
    """
    env = {k: v for k, v in os.environ.items()
           if k.startswith(("CGC_", "LLAMA_EXPERT_CACHE_"))}
    env["CGC_SERVER_MTP"] = MODELS[kind]["mtp"]
    for kv in extra_env or []:
        k, _, v = kv.partition("=")
        if k.strip():
            env[k.strip()] = v.strip()
    return env


def _env_int(env, key, default):
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError):
        return default


def feasibility_cell(kind, gb, cap=None, extra_env=None):
    """Pre-launch verdict for one (model, pool[, cap]) cell. Arithmetic, not a measurement.

    Same verdict vocabulary as pool_feasibility, except the reference cap is replaced by the cap
    this cell will actually be launched at:

      USABLE         it runs at the reference cap -> a cross-cell difference is pool-only
      PARTITION-REF  it runs, but the cap has to drop -> the difference includes the prefill STEP
                     PARTITION, so the cell can inform partition sensitivity only
      UNUSABLE       no cap satisfies P1 (capacity) and P2 (survival) at the same time
    """
    feas = _feasibility()
    geom = pool_geometry(kind)
    env = effective_launch_env(kind, extra_env)
    mtp = str(env.get("CGC_SERVER_MTP", "1")).lower() in ("1", "true", "on", "yes")
    spec = _env_int(env, "CGC_SERVER_MTP_N_MAX", feas.DEFAULT_SPEC_N_MAX)
    layer_caps = env.get("CGC_SERVER_LAYER_CAPS") or None
    topk = MODELS[kind].get("topk") or geom["topk"] or 8
    if cap is None and env.get("CGC_POOL_MAX_TOKENS"):
        cap = _env_int(env, "CGC_POOL_MAX_TOKENS", None)

    r = feas.evaluate(geom, int(gb) * 1024 ** 3, topk, mtp, spec, layer_caps)
    ref = feas.REF_CAP
    requested = ref if cap is None else int(cap)
    cap_max, floor = r["cap_max"], r["floor"]

    if cap_max < floor:
        verdict = "UNUSABLE"
        why = (f"cap_max {cap_max} < floor {floor}: P1 and P2 cannot both hold at any cap "
               f"({r['usable_slots']} usable slots, n_rs_seq={r['n_rs_seq']})")
    elif requested > cap_max:
        verdict = "PARTITION-REF"
        why = (f"cap {requested} x topk {topk} = {requested * topk} > usable {r['usable_slots']} "
               f"(cap_max {cap_max}): the engine has to fall back to cap {cap_max}, so this cell's "
               f"forward pass is partitioned differently from one that keeps cap {requested}")
    elif requested < floor:
        verdict = "UNUSABLE"
        why = (f"cap {requested} < floor {floor} (n_rs_seq={r['n_rs_seq']}): "
               f"GGML_ASSERT(n_ubatch > n_keep_tail) aborts")
    elif requested >= ref:
        verdict = "USABLE"
        why = f"runs at cap {requested} (reference cap {ref}); a cross-cell diff is pool-only"
    else:
        verdict = "USABLE"
        why = (f"runs at cap {requested} by request (< reference {ref}); comparable only with "
               f"cells launched at that same cap")

    r.update({"model": kind, "pool_gb": int(gb), "cap_requested": requested,
              "cap_is_default": requested == ref, "cap_source": "cli/env" if cap is not None
              else "engine default",
              "cap_max": cap_max, "floor": floor, "topk": topk, "mtp": mtp,
              "spec_n_max": spec, "verdict": verdict, "why": why,
              "ref_cap_ok": cap_max >= ref, "path": model_path(kind)})
    # Convert numpy scalar types (uint64 etc.) to Python natives so the cell is JSON-serializable.
    for k, v in list(r.items()):
        if hasattr(v, "item") and not isinstance(v, (dict, list)):
            try:
                r[k] = v.item()
            except Exception:  # noqa: BLE001
                pass
    return r


def feasibility_gate(kind, gb, args, cap=None, extra_env=None, where="run"):
    """Refuse a launch that cannot produce a comparable number. Returns (ok, cell).

    `ok=True, cell=None` means "no verdict available" (a predictor must never be the reason a run
    dies, so an unreadable GGUF warns and continues). `ok=False` means the caller must not launch.
    """
    if getattr(args, "no_feasibility_gate", False):
        return True, None
    if getattr(args, "attach", False):
        # --attach measures a server somebody else launched: its cap and its pool are not ours to
        # know, so a verdict about "the cap this cell will launch at" would be a verdict about a
        # launch that is not happening.
        print(f"[{where}] feasibility: skipped (--attach: the running server's cap/pool are fixed "
              f"outside this process)", flush=True)
        return True, None
    try:
        cell = feasibility_cell(kind, gb, cap, extra_env)
    except Exception as e:  # noqa: BLE001 - harness: never fatal
        print(f"[{where}] feasibility: no pre-launch verdict (cannot read the GGUF geometry: "
              f"{e}); launching anyway", file=sys.stderr, flush=True)
        return True, None
    if cell["verdict"] == "USABLE":
        note = "" if cell["cap_is_default"] else " (non-default cap: compare only at the same cap)"
        print(f"[{where}] feasibility USABLE  cap={cell['cap_requested']} "
              f"cap_max={cell['cap_max']} slots={cell['usable_slots']} floor={cell['floor']}{note}",
              flush=True)
        return True, cell

    override = (getattr(args, "allow_partition_ref", False)
                if cell["verdict"] == "PARTITION-REF" else
                getattr(args, "allow_unusable", False))
    head = f"[{where}] feasibility {cell['verdict']}  {kind}@{gb}GiB cap={cell['cap_requested']}"
    if override:
        print(f"{head}\n  {cell['why']}\n  continuing because the override flag was given; the "
              f"record carries verdict={cell['verdict']} and must NOT be read as a pool "
              f"comparison", file=sys.stderr, flush=True)
        return True, cell
    print(f"{head}\n  {cell['why']}\n  REFUSING to launch: this cell cannot produce a number "
          f"comparable with a reference-cap cell.\n  Use --allow-partition-ref (if you want the "
          f"partition sensitivity measured) or --no-feasibility-gate to disable the check; see "
          f"docs/POOL_FEASIBILITY_MATRIX_2026-09-14.html.", file=sys.stderr, flush=True)
    return False, cell


def print_feasibility_matrix(models, pools, args, caps=None):
    """Up-front table: the verdict for every (model, pool) BEFORE anything is launched."""
    caps = caps or {}
    default_cap = None if args.pool_cap in ("auto", "off") else _env_int(
        {"CGC_POOL_MAX_TOKENS": args.pool_cap}, "CGC_POOL_MAX_TOKENS", None)
    print("\nPOOL FEASIBILITY (pre-launch: GGUF tensor table + the engine's own arithmetic)")
    print(f"{'model':<12} {'pool':>5} {'slots':>6} {'cap_max':>8} {'floor':>6} {'cap':>4} "
          f"{'verdict':>14}  why")
    print("-" * 118)
    for kind in models:
        for gb in pools:
            try:
                r = feasibility_cell(kind, gb, caps.get(kind, default_cap), args.extra_env)
            except Exception as e:  # noqa: BLE001 - a table row must not kill the run
                print(f"{kind[:12]:<12} {gb:>4}G {'-':>6} {'-':>8} {'-':>6} {'-':>4} "
                      f"{'UNKNOWN':>14}  {e}")
                continue
            print(f"{kind[:12]:<12} {gb:>4}G {r['usable_slots']:>6} {r['cap_max']:>8} "
                  f"{r['floor']:>6} {str(r['cap_requested']):>4} {r['verdict']:>14}  "
                  f"{r['why'][:70]}")
    print("-" * 118)
    print("USABLE runs at the reference cap (cross-cell diff = pool only) | PARTITION-REF runs "
          "only at a smaller cap (diff includes step partition) | UNUSABLE no cap fits")
    print()


def cap_for(args, kind):
    """Resolved `CGC_POOL_MAX_TOKENS` for this model, shared by EVERY pool (or None = leave it
    to the build's default, which is iq3-independent but not pool-independent)."""
    resolved = getattr(args, "pool_cap_resolved", None)
    if isinstance(resolved, dict):
        return resolved.get(kind)
    if args.pool_cap in ("auto", "off"):
        return None
    try:
        return int(args.pool_cap)
    except (TypeError, ValueError):
        return None


def combo_label(kind, gb, tag):
    return f"{kind}_pool{gb}gb{('_' + tag) if tag else ''}"


def run_combo(kind, gb, args):
    model = MODELS[kind]
    # A ground-truth dump written over the cache-ON dump of the same (model, pool) would destroy
    # the very evidence it exists to judge -- and the two files have the same default name. So
    # --no-expert-cache forces a distinct label unless the caller named one explicitly.
    tag = args.tag
    if getattr(args, "no_expert_cache", False) and not tag:
        tag = "nocache"
    label = combo_label(kind, gb, tag)
    out_json = os.path.join(RESULT_DIR, f"knifeedge_{label}.json")
    # The cap this cell will launch with, resolved up front because the resume check compares the
    # cached record's geometry -- and the cap is part of that geometry (it clamps n_batch).
    cap = cap_for(args, kind)
    if os.path.exists(out_json) and not args.force:
        # RESUME reads the COMBO record, not the suite output: `knifeedge_*.json` is flip_rate's
        # summary and carries no `up`/`gates`, so gating resume on it meant no full run could ever
        # be skipped -- the check always fell through to REDO. The combo record is the harness's
        # own measurement record, and it now also carries the provenance that decides whether a
        # cached measurement still describes this tree.
        prev_path = os.path.join(RESULT_DIR, f"combo_{label}.json")
        prev = None
        if os.path.exists(prev_path):
            try:
                prev = json.load(open(prev_path, encoding="utf-8"))
            except Exception:  # noqa: BLE001 - harness
                prev = None
        if prev is not None:
            # A combo that never came up is NOT a result. Caching it as one is the same trap as
            # the false-passing cap probe: the next run skips the cell and the table shows a
            # config that was never measured. Measured case (2026-09-11): run_server.sh's memory
            # guard refused to start (`free=15%<40%`), the stub `{"up": false}` was written, and
            # the following run printed SKIP for it.
            why = None
            if not prev.get("up"):
                why = f"it never came up (up={prev.get('up')!r}, skipped={prev.get('skipped')!r})"
            elif not (prev.get("gates") or prev.get("skipped") is None):
                why = "it has no gate evidence"
            elif prev.get("gates_only"):
                why = "it is a --gates-only run, not a quality measurement"
            elif prev.get("skipped"):
                why = f"it was skipped ({prev.get('skipped')!r})"
            else:
                # PROVENANCE: a cached measurement that cannot prove WHICH pool geometry, engine
                # and weights produced it is exactly the 106-vs-143 trap one layer up -- the
                # number is plausible and unattributable.
                prev_key = record_geometry_key(prev)
                want = record_geometry_key({"pool_cap": cap, "provenance": record_provenance(
                    kind, gb, args, cap=cap)})
                if prev_key is None:
                    why = ("it carries no provenance, so nothing ties it to this tree -- re-run "
                           "once with the current harness to make it attributable")
                elif prev_key != want:
                    moved = [human for f, human in KEY_FIELDS
                             if prev_key.get(f) != want.get(f)]
                    why = (f"it was measured under a different geometry ({', '.join(moved)}); the "
                           f"pool-size comparison would mix two computations")
            if why is None:
                print(f"[{label}] SKIP (complete, attributable result exists; --force to redo)",
                      flush=True)
                return prev
            print(f"[{label}] REDO: the cached result is not reusable -- {why}; re-running",
                  flush=True)

    launch_log = os.path.join(RESULT_DIR, f"launch_{label}.log")
    # Oracle dump is env-gated at launch; kept out of the result name so a rerun overwrites it.
    oracle_dump = os.path.join(RESULT_DIR, f"oracle_{label}.jsonl")
    extra = list(args.extra_env)
    # M2 prefill-stream mode: whole-layer slab gather, has its own oracle (L2 divergence vs std)
    # getattr, not args.m2: run_combo is also driven by callers that build their own Namespace
    # (feasibility_gate_selftest.py), and an AttributeError there reads as "the gate crashed"
    # rather than "the flag is off". Off-by-default is the correct meaning of an absent flag.
    if getattr(args, "m2", False):
        extra.append("CGC_PREFILL_STREAM=1")
        extra.append("CGC_GATHER_SLAB_CAP=256")
    # The union-routable gate must not depend on an OPTIONAL diagnostic flag the caller happened
    # to pass: on 2026-09-14 the 2 GiB cell scored M1/M2/M3 117/117 while the gate said FAIL
    # mode=unreachable simply because CGC_UNION_LOG was not set, so the `gather=N/M` evidence it
    # reads had never been written. "Unknown" and "not routable" must not look the same -- the
    # harness therefore always asks the engine for the evidence it is about to judge. (Without
    # this the gate can only ever pass a config whose FIRST version was measured by hand.)
    extra.append("CGC_UNION_LOG=1")
    # One stamp for every record this cell writes. Computed HERE (before the launch) so the same
    # object can go into the failure records too -- a cell that died still has to say which
    # geometry it died under.
    prov = record_provenance(kind, gb, args, cap=cap)
    # PRE-LAUNCH FEASIBILITY: before anything is killed or started. A cell that cannot run at
    # this cap does not produce a pool measurement, so it must not become a table row -- and per
    # the same rule as the low-memory refusal below, a refusal is NOT a result and is not written
    # to out_json (a cached stub would then be SKIPped by later runs).
    if getattr(args, "no_expert_cache", False):
        # Ground-truth arm: there is no pool to be feasible. The gate's whole question
        # ("can a cap satisfy capacity AND survival?") is undefined at budget 0, and asking it
        # would refuse the very cell that has to exist for --oracle-abs to be usable at all.
        g_feas = {"verdict": "GROUND_TRUTH", "why": "expert cache off (budget 0); no pool "
                                                    "geometry to validate", "pool_gb": gb}
        feas_ok = True
    else:
        feas_ok, g_feas = feasibility_gate(kind, gb, args, cap, args.extra_env, where=label)
    if not feas_ok:
        return {"label": label, "model": kind, "pool_gb": gb, "up": False,
                "skipped": "infeasible_" + g_feas["verdict"].lower().replace("-", "_"),
                "feasibility": g_feas}
    if cap is not None:
        # ONE cap for every pool so the ubatch shape and the chunking are identical across pool
        # sizes -- otherwise a pool "difference" is partly a shape difference (see probe_pool_cap).
        extra.append(f"CGC_POOL_MAX_TOKENS={cap}")
    if args.oracle_ref or args.dump_oracle:
        if os.path.exists(oracle_dump):
            os.remove(oracle_dump)
        extra += [f"CGC_LOGITS_ORACLE_DUMP={oracle_dump}", "CGC_LOGITS_ORACLE_TOPN=8"]
        # Record provenance next to the dump so a future comparison can PROVE it is looking at
        # the same computation: the cap (which clamps n_batch) AND the state of the code that
        # decides the numerics. Without this, forgetting --pool-cap silently yields reference
        # and candidate computed under different ubatch shapes, and an expert-cache edit makes
        # an old reference describe an engine that no longer exists -- in both cases the
        # resulting M1 collapse looks exactly like a real pool-induced numeric regression.
        # expert_cache is part of the computation's identity, not a label: without it a
        # cache-ON dump can be handed to --oracle-abs and scored as if it were a control.
        _write_oracle_meta(oracle_dump, cap if cap is not None else args.pool_cap_max,
                           model_file=model_path(kind),
                           launch_sig={"extra_env": sorted(args.extra_env or []),
                                       "mtp": MODELS[kind]["mtp"],
                                       "expert_cache": expert_cache_state(
                                           args.extra_env, launch_pool_bytes(args, gb))},
                           expert_cache=expert_cache_state(args.extra_env,
                                                           launch_pool_bytes(args, gb)))
    if args.attach:
        # Shared checkout: someone else's server may already own the port. Measure it
        # read-only (no launch, no kill) instead of fighting over the machine.
        print(f"[{label}] ATTACH to already-running server on port {args.port}", flush=True)
        free_before = free_after = mem_free_pct()
        took, facts = 0.0, {"attached": True}
    else:
        print(f"[{label}] launching {model['file']} pool={gb}GB mtp={model['mtp']}", flush=True)
        kill_servers()
        if running_servers():
            print(f"[{label}] REFUSING: another agent's server will not clear "
                  f"({running_servers()}); its /health would be measured instead of ours.",
                  file=sys.stderr, flush=True)
            return {"label": label, "model": kind, "pool_gb": gb, "up": False,
                    "skipped": "foreign_server"}
        free_before = mem_free_pct()
        # MEMORY PRECONDITION. run_server.sh refuses to start full-MTP below 40% free, so
        # launching under that is guaranteed to produce nothing but a launch log full of guard
        # errors -- and (before the skip-rule fix) a cached stub that later runs would SKIP.
        # Check it here so the failure is loud, named, and not written as a measurement.
        if free_before < args.min_free_pct:
            print(f"[{label}] REFUSING to launch: machine is {free_before}% free, below the "
                  f"{args.min_free_pct}% precondition (run_server.sh's memory guard would "
                  f"reject it: 'startup blocked by memory guard').\n"
                  f"  This is a MACHINE precondition, not a result. Free memory (a reboot clears "
                  f"the compressor + swap that repeated 13-18 GB model loads leave behind) and "
                  f"re-run; nothing was measured for this cell.", file=sys.stderr, flush=True)
            return {"label": label, "model": kind, "pool_gb": gb, "up": False,
                    "skipped": "low_memory", "free_before_pct": free_before}
        pre = tuple(running_servers())
        launch(model_path(kind), model["mtp"], launch_pool_bytes(args, gb), args.port, args.kwargs,
               extra, launch_log)
        took = wait_health(args.port, model_path(kind), args.health_timeout, pre_pids=pre)

    if took is None:
        print(f"[{label}] FAILED to become healthy (server died, timed out, or a foreign "
              f"server owns port {args.port})", flush=True)
        tail = ""
        try:
            tail = open(launch_log, "rb").read().decode("utf-8", "replace")[-800:]
        except Exception:  # noqa: BLE001
            pass
        rec = {"label": label, "model": kind, "pool_gb": gb, "up": False,
               "feasibility": g_feas, "provenance": prov,
               "free_before_pct": free_before, "launch_tail": tail}
        json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        kill_servers()
        return rec

    if not args.attach:
        facts = read_launch_facts(launch_log)
    # Re-stamp with the facts the engine reported at load: the record then carries both the
    # PREDICTED slot geometry and the MEASURED one, and a disagreement is visible per row.
    prov = record_provenance(kind, gb, args, cap=cap, facts=facts)
    rss = rss_gb()
    free_after = mem_free_pct()
    print(f"[{label}] healthy in {took:.0f}s  rss={rss}GB free={free_before}%->{free_after}%  "
          f"kind={facts.get('model_kind')} template={facts.get('template_file', '(embedded)')}",
          flush=True)

    # GATE: numeric invariance vs the reference oracle. Run this BEFORE anything else talks to
    # the server so the dump's early ubatches line up with the reference run's.
    g_oracle = None
    g_oracle_final = None
    if args.oracle_ref:
        g_oracle = oracle_gate(args.port, oracle_dump, args.oracle_ref, label, args.timeout,
                               allow_stale=args.allow_stale_oracle,
                               model_file=model_path(kind),
                               allow_model_mismatch=args.allow_model_mismatch,
                               absolute=getattr(args, "oracle_abs", False))
        print(f"[{label}] gate-oracle  {'PASS' if g_oracle.get('ok') else 'CHECK'}  "
              f"M1(bit-identical)={g_oracle.get('m1_numeric_identity')} "
              f"M2(argmax)={g_oracle.get('m2_decision_agreement')} "
              f"{g_oracle.get('error', '')}", flush=True)
    elif args.dump_oracle:
        # Producing a reference must exercise the SAME probe sequence a comparison cell will.
        # `ask_probe()` lives inside oracle_gate and is what appends those steps to the dump.
        # Without this call the reference is a strict PREFIX of every candidate, so a
        # prefix-matching compare reports a perfect 60/60 while the remaining keys are never
        # tested -- the same silent understatement as §6 of POOLSIZE_INVARIANCE_2026-09-13.md,
        # just moved into the reference instead of the comparison.
        ask_probe(args.port, args.timeout)
        time.sleep(1.0)
        print(f"[{label}] dump-oracle: probe sent, so the reference covers the same steps as "
              f"a comparison cell", flush=True)

    # GATE: the rendered generation prompt must be a CLOSED scaffold. A degenerating model can
    # still stumble onto the right answer, so the answer check alone is not sufficient.
    g_template = apply_template_probe(args.port)
    print(f"[{label}] gate-template {'PASS' if g_template.get('ok') else 'FAIL' if g_template.get('ok') is False else 'n/a'}  "
          f"unclosed_think={g_template.get('unclosed_think')} "
          f"tail={g_template.get('tail', '')[-70:]!r}", flush=True)

    # GATE: `buffer is nil` = the pool path silently fell through to the L3-B gather path on a
    # GPU-resident layer. Invalidates the combo AND explains any numeric drift.
    g_nil = scan_nil([launch_log, facts.get("log_path")])
    print(f"[{label}] gate-buffer-nil {'PASS' if g_nil['ok'] else 'FAIL'}  {g_nil['hits']}", flush=True)

    # GATE: is the broken path unreachable BY CONSTRUCTION? `union_max <= usable_slots` is the
    # arithmetic guarantee, independent of which experts this particular prompt happens to route
    # to -- so it holds for every model and every pool size, not just the probes here.
    g_union = union_fit_gate(facts, cap if cap is not None else args.pool_cap_max,
                             MODELS[kind].get("topk", 8),
                             paths=[launch_log, facts.get("log_path")])
    _ev = g_union.get("gather") or {}
    print(f"[{label}] gate-union-routable "
          f"{'PASS' if g_union['ok'] else 'FAIL' if g_union['ok'] is False else 'n/a'}  "
          f"mode={g_union.get('mode')} cap={g_union['cap']} x topk={g_union['topk']} = "
          f"{g_union['union_max']} vs usable={g_union.get('usable_slots')} "
          f"(n_slots={g_union.get('n_slots')}); wide route: layers={_ev.get('wide_layers')} "
          f"steps={_ev.get('gather_steps')}/{_ev.get('total_steps')} "
          f"{g_union.get('error', '')}", flush=True)

    # PREFLIGHT: refuse to produce numbers from a broken prompt scaffold. This is the gate
    # that was missing when the earlier pool curve / flip_rate runs were silently measuring
    # a template defect instead of the expert cache.
    if args.preflight != "off":
        pf = preflight(args.port, timeout=args.preflight_timeout)
        rec_pf = {k: pf.get(k) for k in ("ok", "answers_42", "echoes_prompt",
                                         "scaffold_leak", "empty", "finish", "error")}
        print(f"[{label}] preflight {'PASS' if pf.get('ok') else 'FAIL'}  "
              f"42={pf.get('answers_42')} echo={pf.get('echoes_prompt')} "
              f"leak={pf.get('scaffold_leak')} empty={pf.get('empty')} "
              f"finish={pf.get('finish')}  content={pf.get('content','')[:90]!r}", flush=True)
        # The probes above appended to the same dump the oracle gate compared, so that verdict
        # only covered a prefix. Re-compare the finished file before anything is recorded.
        g_oracle_final = oracle_recompare(oracle_dump, args.oracle_ref, label,
                                         allow_stale=args.allow_stale_oracle,
                                         model_file=model_path(kind),
                                         allow_model_mismatch=args.allow_model_mismatch,
                                         absolute=getattr(args, "oracle_abs", False))
        if not pf.get("ok"):
            msg = (f"[{label}] preflight FAILED -- the prompt scaffold is broken, so any "
                   f"quality number from this config would be measuring the template, "
                   f"not the pool. Fix the scaffold first (see "
                   f"docs/CHAT_SCAFFOLD_ROOTCAUSE_2026-09-11.md).")
            if args.preflight == "abort":
                print(msg, file=sys.stderr, flush=True)
                # Keep the gates in the record even on abort. The oracle/template/nil verdicts
                # were already measured and they are exactly what you need when preflight fails:
                # dropping them here once threw away the M1/M2 comparison precisely in the case
                # where the config is already suspect.
                rec = {"label": label, "model": kind, "pool_gb": gb, "up": True,
                       "preflight": rec_pf,
                       "gates": {"template": g_template, "buffer_nil": g_nil,
                                 "union_fit": g_union, "oracle": g_oracle,
                                 "oracle_final": g_oracle_final},
                       "pool_cap": cap, "feasibility": g_feas, "provenance": prov,
                       "free_before_pct": free_before,
                       "free_after_pct": free_after, "rss_gb": rss,
                       "launch_facts": facts, "extra_env": args.extra_env,
                       "skipped": "preflight_failed"}
                json.dump(rec, open(os.path.join(RESULT_DIR, f"combo_{label}.json"), "w",
                                    encoding="utf-8"), indent=2, ensure_ascii=False)
                if not args.attach:
                    kill_servers()
                return rec
            print(msg, file=sys.stderr, flush=True)
    else:
        rec_pf = None
        # No preflight traffic, but the template probe still grew the dump; re-compare it.
        g_oracle_final = oracle_recompare(oracle_dump, args.oracle_ref, label,
                                         allow_stale=args.allow_stale_oracle,
                                         model_file=model_path(kind),
                                         allow_model_mismatch=args.allow_model_mismatch,
                                         absolute=getattr(args, "oracle_abs", False))

    if args.gates_only:
        print(f"[{label}] --gates-only: skipping the quality suite", flush=True)
        if not args.attach:
            kill_servers()
        rec = {"label": label, "model": kind, "pool_gb": gb, "up": True,
               "preflight": rec_pf,
               "gates": {"template": g_template, "buffer_nil": g_nil, "union_fit": g_union,
                         "oracle": g_oracle,
                         "oracle_final": g_oracle_final},
               "pool_cap": cap, "gates_only": True, "feasibility": g_feas,
               "provenance": prov,
               "gates_ok": (bool(g_nil["ok"]) and g_template.get("ok") is not False
                            and g_union.get("ok") is not False),
               "free_before_pct": free_before, "free_after_pct": free_after,
               "rss_gb": rss, "launch_facts": facts, "extra_env": args.extra_env,
               "attached": bool(args.attach)}
        json.dump(rec, open(os.path.join(RESULT_DIR, f"combo_{label}.json"), "w",
                            encoding="utf-8"), indent=2, ensure_ascii=False)
        return rec

    cmd = [sys.executable, os.path.join(ROOT, "scripts", "check", "flip_rate.py"),
           "--base-url", f"http://127.0.0.1:{args.port}/v1", "--model", "local",
           "--label", label, "--profiles", args.profiles,
           "--per-profile", str(args.per_profile),
           "--greedy-repeats", str(args.greedy_repeats),
           "--repeats", str(args.repeats),
           "--temperature", str(args.temperature),
           "--max-tokens", str(args.max_tokens),
           "--timeout", str(args.timeout),
           "--output", out_json]
    subprocess.run(cmd, cwd=ROOT)

    # Re-scan for `buffer is nil` AFTER the traffic above. The early scan runs before any request
    # has been served, so an empty result there only means "no decode built a union yet" -- it
    # cannot clear a combo. This second scan is the one that can actually observe the failure.
    g_nil_post = scan_nil([launch_log, facts.get("log_path")])
    if not g_nil_post["ok"]:
        if g_nil["ok"]:
            print(f"[{label}] gate-buffer-nil FAIL (post-traffic) {g_nil_post['hits']}",
                  flush=True)
        g_nil = g_nil_post

    if not args.attach:
        kill_servers()
    rec = {"label": label, "model": kind, "pool_gb": gb, "up": True,
           "preflight": rec_pf,
           "gates": {"template": g_template, "buffer_nil": g_nil, "union_fit": g_union,
                     "oracle": g_oracle,
                     "oracle_final": g_oracle_final},
           "pool_cap": cap, "feasibility": g_feas, "provenance": prov,
           "gates_ok": (bool(g_nil["ok"]) and g_template.get("ok") is not False
                        and g_union.get("ok") is not False),
           "free_before_pct": free_before, "free_after_pct": free_after,
           "rss_gb": rss, "launch_facts": facts, "extra_env": args.extra_env,
           "attached": bool(args.attach)}
    if os.path.exists(out_json):
        sess = json.load(open(out_json, encoding="utf-8"))
        rec.update({k: sess.get(k) for k in
                    ("greedy_all_pass_rate", "greedy_unstable_questions",
                     "seeded_flip_questions", "suite_rate_per_seed", "suite_rate_mean",
                     "suite_rate_std", "suite_rate_min", "suite_rate_max",
                     "tps_mean", "tps_std", "tps_max", "n_questions")})
        # Stamp the SUITE's own output too. `knifeedge_*.json` is the artifact read months later,
        # and until now it was the one file in the tree that described a measurement without
        # saying which pool geometry, engine and weights produced it -- so two of them could be
        # lined up side by side with no way to tell whether that was legitimate.
        try:
            sess["provenance"] = prov
            sess["model"] = kind
            sess["pool_gb"] = gb
            sess["pool_cap"] = cap
            json.dump(sess, open(out_json, "w", encoding="utf-8"), indent=2,
                      ensure_ascii=False)
        except Exception as e:  # noqa: BLE001 - harness
            print(f"[{label}] WARNING: could not stamp {out_json}: {e}", file=sys.stderr,
                  flush=True)
    json.dump(rec, open(os.path.join(RESULT_DIR, f"combo_{label}.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    return rec


def print_matrix(records):
    print()
    print("=" * 140)
    print("%-26s %5s %7s %8s %8s %7s %7s %8s %5s %6s %6s %6s %14s" % (
        "config", "rss", "free%", "greedy", "seeded", "std", "flips", "tok/s",
        "cap", "tmpl", "nil", "geom", "M1/M2"))
    print("-" * 146)
    for r in records:
        if not r.get("up"):
            # Name the REASON. A cell that never launched because the MACHINE was out of memory
            # is not evidence about the model, and printing it as "did not come up" invites
            # exactly that misreading (and, before the skip-rule fix, a cached stub).
            sk = r.get("skipped") or ""
            fe = r.get("feasibility") or {}
            why = {"low_memory": "machine below --min-free-pct -> NOT measured, nothing cached",
                   "foreign_server": "another process owns the port -> NOT measured"}.get(sk)
            if why is None and sk.startswith("infeasible"):
                # The reason must name the ARITHMETIC, not just "refused". A refused cell is
                # otherwise indistinguishable from a cell nobody got to, and the whole point of
                # the gate is that this number could never have been a pool measurement.
                why = (f"{fe.get('verdict', sk)} at launch: cap_max={fe.get('cap_max')} "
                       f"< cap {fe.get('cap_requested')} (usable slots {fe.get('usable_slots')}, "
                       f"floor {fe.get('floor')}) -> REFUSED, not a pool result; "
                       f"--allow-partition-ref to measure it as partition sensitivity")
            if why is None:
                why = f"server did not come up ({r.get('model')})"
            print("%-26s   -- %s" % (r["label"], why))
            continue
        g = r.get("gates") or {}
        gt = (g.get("template") or {}).get("ok")
        gts = "n/a" if gt is None else ("ok" if gt else "FAIL")
        gn = (g.get("buffer_nil") or {}).get("ok")
        gns = "n/a" if gn is None else ("ok" if gn else "FAIL")
        go = g.get("oracle") or {}
        mpair = f"{go.get('m1_numeric_identity','-')}/{go.get('m2_decision_agreement','-')}"
        # Per-row geometry mismatch: predicted vs launched slots/layer. This is the flag that
        # catches a stale pool_feasibility.py on the FIRST measurement, not on the next probe.
        prov = r.get("provenance") or {}
        gm = (prov.get("pool") or {}).get("geometry_mismatch")
        gms = "n/a" if gm is None else ("MISMATCH" if gm else "ok")
        print("%-26s %5s %7s %8s %8s %7s %7s %8s %5s %6s %6s %6s %14s" % (
            r["label"], r.get("rss_gb"), r.get("free_after_pct"),
            r.get("greedy_all_pass_rate"), r.get("suite_rate_mean"),
            r.get("suite_rate_std"), r.get("seeded_flip_questions"),
            r.get("tps_mean"), r.get("pool_cap", "-"), gts, gns, gms, mpair))
    print("=" * 108)
    print("greedy   = all-%d-greedy-repeats pass rate (determinism side)" % args_greedy_global)
    print("seeded   = mean suite pass rate over fixed seeds (sampling side)")
    print("std      = spread of the per-seed suite rate -> the uncertainty the pool-floor")
    print("           decision has to survive. A pool ranking INSIDE one std is not a ranking.")
    print("cap      = CGC_POOL_MAX_TOKENS actually launched; MUST be identical down the column")
    print("           for a given model, else the pool sizes are not the only variable")
    print("flips    = questions whose verdict changes across seeds")
    print("tok/s    = mean decode tok/s over every request in the session")
    # Same grouping rule as --summary, applied to the sweep that just finished: a table that mixes
    # two geometries is not a comparison, and printing it without saying so is how one gets read
    # as one. (Within one sweep this normally says OK; it exists to stay true when it does not.)
    for model, v in comparability_verdict(records).items():
        if v["state"] == "REFUSED":
            print(f"GEOMETRY: {model} rows span {len(v['groups'])} groups -> NOT comparable: "
                  + "; ".join(v["differences"]))
        elif v["state"] == "OK":
            print(f"GEOMETRY: {model} rows share one geometry group "
                  f"({v['groups'][0]['fingerprint']}) -> comparable")
        else:
            print(f"GEOMETRY: {model} rows are UNATTRIBUTED (no provenance) -> comparability "
                  f"unverified")


def summary_table():
    """One row per (model x pool) with all four gates, assembled from the result records.

    Read the GEOMETRY GROUPS first, then the columns together. Rows are only laid out next to
    each other when the provenance says they were measured under the same engine, weights, launch
    env and cap; otherwise they are printed in separate blocks and the comparison is REFUSED,
    naming which of those moved. That is the same rule the cap probe and the oracle guard already
    enforce at launch, applied to the table decisions actually get made from.

    Read the columns together, not one at a time:
      cap must be IDENTICAL down a model's rows, or the pools were not the only variable;
      union-fit is the arithmetic reason the gather path is unreachable;
      nil is that prediction OBSERVED (nil PASS with union-fit n/a proves nothing);
      M1/M2 must stay apart -- M2 agreement on its own is not "no difference".
    """
    import glob
    recs, seen = [], {}
    # Read BOTH artifacts. `knifeedge_*.json` is the suite's own output (the file people actually
    # open months later) and `combo_*.json` is the harness record that carries the gates; they
    # describe the same cell, so they are merged by label rather than listed twice.
    for pat in ("combo_*.json", "knifeedge_*.json"):
        for p in sorted(glob.glob(os.path.join(RESULT_DIR, pat))):
            try:
                r = json.load(open(p, encoding="utf-8"))
            except Exception:  # noqa: BLE001 - harness
                continue
            base = os.path.basename(p)
            label = r.get("label") or base.split("_", 1)[1][:-len(".json")]
            r.setdefault("label", label)
            if label in seen:
                for k, v in r.items():
                    seen[label].setdefault(k, v)
                continue
            seen[label] = r
            recs.append(r)
    if not recs:
        print("no result records yet")
        return
    order = {"iq3": 0, "iq4": 1}
    recs.sort(key=lambda r: (order.get(r.get("model"), 9), r.get("pool_gb", 0)))
    verdict = comparability_verdict(recs)
    print("=" * 134)
    print("%-16s %4s %6s %6s %6s %9s %6s %6s %10s %12s %7s" % (
        "config", "cap", "rss", "free%", "tmpl", "union-fit", "nil", "geom", "M1 bit-id",
        "M2 argmax", "preflt"))
    print("-" * 134)

    def mark(v):
        return {True: "PASS", False: "FAIL", None: "n/a"}.get(v, "-")

    def pct(v):
        """M1/M2 come back either as a ratio or as '7/39' depending on the compare script."""
        if v is None:
            return "n/a"
        if isinstance(v, str):
            mm = re.match(r"^(\d+)\s*/\s*(\d+)$", v.strip())
            if mm and int(mm.group(2)):
                return f"{v} ({int(mm.group(1)) / int(mm.group(2)):.1%})"
            return v
        try:
            return f"{float(v):.1%}"
        except (TypeError, ValueError):
            return str(v)

    def row_line(r):
        g = r.get("gates") or {}
        t = (g.get("template") or {}).get("ok")
        n = (g.get("buffer_nil") or {}).get("ok")
        u = (g.get("union_fit") or {})
        o = (g.get("oracle") or {})
        pf = (r.get("preflight") or {}).get("ok")
        uf = mark(u.get("ok"))
        if u.get("headroom") is not None:
            uf += f" ({u['union_max']}<={u.get('usable_slots')}, {u['headroom']:+d})"
        # Per-row geometry mismatch: predicted vs launched slots/layer. Catches a stale
        # pool_feasibility.py on the first measurement, not on the next probe run.
        prov = r.get("provenance") or {}
        gm = (prov.get("pool") or {}).get("geometry_mismatch")
        gms = "n/a" if gm is None else ("MISMATCH" if gm else "ok")
        return "%-16s %4s %6s %6s %6s %-9s %6s %6s %10s %12s %7s" % (
            r.get("label"), r.get("pool_cap", "-"), r.get("rss_gb"),
            r.get("free_after_pct"), mark(t), uf, mark(n), gms,
            pct(o.get("m1_numeric_identity")), pct(o.get("m2_decision_agreement")), mark(pf))

    def geom_line(key, rows):
        """Name what produced a group, so the grouping itself is auditable rather than assumed."""
        if key is None:
            return ("  (no provenance: recorded before the stamp existed -> comparability "
                    "unverified)")
        eng = ((rows[0].get("provenance") or {}).get("engine") or {})
        w = key["weights"] or (None, None, None)
        return (f"  engine={str(key['engine'])[:8]} geometry={str(key['pool_geometry'])[:8]} "
                f"head={eng.get('head') or '?'}{' (dirty)' if eng.get('dirty') else ''} "
                f"weights={os.path.basename(str(w[0] or '?'))}@{w[1]} "
                f"cap={key['cap']} mtp={key['mtp']} spec_n_max={key['spec_n_max']} "
                f"layer_caps={key['layer_caps']} extra_env={list(key['extra_env']) or '-'}")

    for model in [m for m in verdict]:
        v = verdict[model]
        if v["state"] == "REFUSED":
            print(f"{model}: REFUSED -- {len(v['groups'])} geometry group(s); rows from different "
                  f"groups are NOT comparable and are printed separately below, with no "
                  f"cross-group claim made:")
            for d in v["differences"]:
                print(f"    - {d}")
        elif v["state"] == "UNATTRIBUTED":
            print(f"{model}: UNATTRIBUTED -- one group, but it carries no provenance, so its rows "
                  f"cannot be certified comparable (re-measure with the current harness).")
        for g in v["groups"]:
            print(f"[{model} group {g['fingerprint']}]  {len(g['rows'])} row(s)")
            print(geom_line(g["key"], g["rows"]))
            for r in sorted(g["rows"], key=lambda r: r.get("pool_gb", 0)):
                print(row_line(r))
    print("-" * 126)
    for model, v in verdict.items():
        if v["state"] == "OK":
            print(f"{model}: one geometry group and a single cap in it -> pool size is the only "
                  f"variable. OK")
        elif v["state"] == "UNATTRIBUTED":
            print(f"{model}: comparability UNVERIFIED (no provenance) -- do not read a pool "
                  f"ranking off these rows as though it were one experiment.")
        else:
            print(f"{model}: comparison REFUSED across {len(v['groups'])} geometry groups -- "
                  f"nothing in this table may be read as a pool ranking until the rows in "
                  f"different groups are re-measured under one tree.")
    # Cap invariance rides along in the same summary: it is the gate that says whether the `cap`
    # column above may be read as a speed-only knob, so leaving it in a separate mode would hide
    # the one verdict that qualifies every other row in this table.
    ci_path = os.path.join(RESULT_DIR, "cap_invariance.json")
    if os.path.exists(ci_path):
        try:
            ci = json.load(open(ci_path, encoding="utf-8"))
        except Exception:  # noqa: BLE001 - harness
            ci = []
        print()
        print("CAP INVARIANCE (same pool, different CGC_POOL_MAX_TOKENS)")
        print("-" * 140)
        print("%-30s %6s %6s %6s %8s %10s %12s %9s %7s" % (
            "config", "pool", "capA", "capB", "verdict", "M1 bit-id", "M2 argmax", "cover",
            "partn"))
        for r in ci:
            if not r.get("pairs"):
                print("%-30s   %s" % (r.get("label"), r.get("error") or "no pair compared"))
                continue
            for p in r["pairs"]:
                part = p.get("partition") or {}
                print("%-30s %5sG %6s %6s %8s %10s %12s %9s %7s" % (
                    r.get("label"), r.get("pool_gb"), p.get("cap_a"), p.get("cap_b"),
                    p.get("verdict"), p.get("m1_numeric_identity", "-"),
                    p.get("m2_decision_agreement", "-"),
                    (f"{p['coverage']:.0%}" if isinstance(p.get("coverage"), float) else "-"),
                    ("same" if part.get("same") else
                     ("diff" if part.get("a") is not None else "n/a"))))
        print("-" * 140)
        # The cap moves TWO things at once: it re-labels pool slots AND, via the n_batch clamp, it
        # changes the prefill step partition. Only the second one was measured to move the logits
        # (fixed cap across pools 2/4/6/8/10 GiB = bit-identical; different partitions = different
        # completions). `partn` says which of the two a row is actually about.
        print("partn = prefill step partition, same/diff. diff => the two caps ran different forward "
              "passes, so the row is NOT a statement about pool layout.")
        print("PASS = logits did not move (only meaningful when partn=same) | "
              "DRIFT = same decisions, different bits (NOT a pass) | FAIL = both differ")
    print("tmpl = scaffold closed | union-fit = cap*topk <= min_slots-1 (arithmetic) | "
          "nil = observed) | preflt = bare-prompt answer")
    print("=" * 126)


def main():
    global args_greedy_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="iq3,iq4")
    ap.add_argument("--pools", default="4,6,8", help="comma list of GB")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--profiles", default="qa-zh,math,coding,reasoning")
    ap.add_argument("--per-profile", type=int, default=2)
    ap.add_argument("--greedy-repeats", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--min-free-pct", type=float, default=40.0,
                    help="refuse to launch a combo when the machine is below this %% free. "
                         "Mirrors run_server.sh's memory guard (full-MTP needs >=40%%); a cell "
                         "skipped for this reason is a MACHINE precondition, not a result")
    ap.add_argument("--health-timeout", type=float, default=240.0)
    # BOTH keys are needed, and one string covers both models:
    #   enable_thinking:false  -> the GGUF embedded template closes its no-think scaffold
    #                             (otherwise it emits an unclosed <think> and the model echoes).
    #   assistant_prefill      -> the custom jinja starts the assistant turn after a short anchor.
    # Without the anchor iq4 emits EOS after ONE token (`completion_tokens: 1`, content ""),
    # identically at 4GB and 8GB -- an anchor problem masquerading as a pool problem. It has to
    # be set HERE because run_server.sh only applies its own anchor when chat-template-kwargs is
    # EMPTY, and this harness always passes some, so it was suppressing the anchor. iq3 ignores
    # keys its template does not use, so one value is safe for both.
    ap.add_argument("--kwargs",
                    default='{"enable_thinking": false, "assistant_prefill": "答："}',
                    help="--chat-template-kwargs value; pass '' to omit")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="extra KEY=VAL for run_server.sh (repeatable)")
    ap.add_argument("--tag", default="", help="suffix for the label (A/B configs)")
    ap.add_argument("--force", action="store_true", help="redo combos that already have results")
    ap.add_argument("--kill-existing", action="store_true",
                    help="kill a pre-existing llama-server instead of refusing")
    ap.add_argument("--attach", action="store_true",
                    help="measure the server already on --port (no launch, no kill); for "
                         "shared checkouts where another agent owns the machine")
    ap.add_argument("--preflight", choices=("abort", "warn", "off"), default="abort",
                    help="gate on a canonical probe before measuring (default abort)")
    ap.add_argument("--preflight-timeout", type=float, default=150.0)
    ap.add_argument("--pilot", action="store_true",
                    help="1 profile x 1 question x 2 greedy x 2 seeds (smoke test)")
    ap.add_argument("--oracle-ref", default=None,
                    help="reference logits oracle JSONL; enables the M1/M2 numeric-invariance "
                         "gate for every combo (a pool size may differ ONLY in speed, so M1 "
                         "numeric identity must be full -- M2 alone is not proof)")
    ap.add_argument("--oracle-abs", action="store_true",
                    help="ABSOLUTE correctness gate: --oracle-ref must be a CONTROL (expert cache "
                         "OFF), and the verdict is recorded as comparison=absolute. Without this "
                         "the gate is only a RELATIVE invariance check -- two cache-ON runs "
                         "agreeing with each other. A cache-ON reference is REFUSED here and "
                         "--allow-stale-oracle / --allow-model-mismatch do NOT override it.")
    ap.add_argument("--no-expert-cache", action="store_true",
                    help="ground-truth arm: launch with -expert-cache 0, so no cache object and "
                         "no hook exist and the forward pass is the stock full-resident one. "
                         "Use with --dump-oracle to produce the reference that --oracle-abs "
                         "demands. Slow (no bounded residency) and NOT a benchmark configuration.")
    ap.add_argument("--dump-oracle", action="store_true",
                    help="write the logits oracle dump WITHOUT comparing it to anything. This is "
                         "how a reference is produced: previously the dump was only written as a "
                         "side effect of --oracle-ref, so making a fresh reference required "
                         "pointing at some other dump first -- a chicken-and-egg that made the "
                         "reference's own provenance accidental.")
    ap.add_argument("--m2", action="store_true",
                    help="M2 prefill-stream mode: sets CGC_PREFILL_STREAM=1 CGC_GATHER_SLAB_CAP=256, "
                         "uses ref_<model>_pool8gb_M2.jsonl as default oracle ref, and records "
                         "mode=M2_stream in provenance. M2 has its own oracle because it changes "
                         "the prefill chunk size (L2 divergence vs standard path).")
    ap.add_argument("--pool-cap", default="auto",
                    help="CGC_POOL_MAX_TOKENS, IDENTICAL for every pool (default 'auto' = probe "
                         "the smallest pool until it reports no `buffer is nil`, then reuse that "
                         "cap everywhere; 'off' leaves the build default). One shared cap is "
                         "required because the cap clamps n_batch and so the ubatch shape.")
    ap.add_argument("--pool-cap-max", type=int, default=8,
                    help="largest cap 'auto' may pick (build default is 8)")
    # PRE-LAUNCH FEASIBILITY GATE (default ON). Rationale + evidence:
    # docs/POOL_FEASIBILITY_MATRIX_2026-09-14.html. The short version: a cell whose cap has to
    # drop is not a pool measurement, and the number it produces is indistinguishable from one.
    ap.add_argument("--no-feasibility-gate", action="store_true",
                    help="disable the pre-launch (model, pool, cap) feasibility check that refuses "
                         "PARTITION-REF / UNUSABLE cells")
    ap.add_argument("--allow-partition-ref", action="store_true",
                    help="launch PARTITION-REF cells anyway (the cap has to drop below the "
                         "reference, so the measurement includes the prefill step partition). "
                         "The verdict is still recorded next to the result")
    ap.add_argument("--allow-unusable", action="store_true",
                    help="launch UNUSABLE cells anyway (no cap satisfies capacity AND survival; "
                         "the server is expected to abort or overflow the union)")
    ap.add_argument("--allow-geometry-drift", action="store_true",
                    help="measure even when the cap probe's MEASURED slot count disagrees with "
                         "pool_feasibility.py's arithmetic. Off by default: the arithmetic is what "
                         "decides whether a cell is comparable at all, so a falsified predictor "
                         "must be fixed before cells are launched (or at least acknowledged)")
    ap.add_argument("--feasibility-only", action="store_true",
                    help="print the pre-launch feasibility table for --models x --pools and exit "
                         "(no launch, no GPU)")
    ap.add_argument("--cap-invariance", action="store_true",
                    help="run ONLY the cap-invariance gate for every (model x pool): dump the logits "
                         "oracle once per cap, realign both dumps on sequence position, and compare "
                         "M1/M2/M3. This is the gate that decides whether a cap change is "
                         "numerically inert -- i.e. whether the pool LAYOUT can be treated as a "
                         "speed-only knob. Measured 2026-09-14 by hand: an 8 GiB pool at cap=5 was "
                         "13-15%% faster than cap=8 and M1 3/82 -- a real lever that no cheaper "
                         "signal (speed, nil, union-fit, template) can distinguish from a clean run.")
    ap.add_argument("--cap-invariance-caps", default=CAP_INVARIANCE_DEFAULT_CAPS,
                    help=f"comma list of caps to compare against each other (default "
                         f"{CAP_INVARIANCE_DEFAULT_CAPS!r}: the harness floor and the build "
                         f"default). Two launches per (model x pool) per cap; dumps are cached "
                         f"and reused while their provenance still matches (see --force).")
    ap.add_argument("--capinv-min-coverage", type=float, default=0.95,
                    help="minimum fraction of position-aligned steps two cap dumps must share "
                         "before the pair may be certified INERT (default 0.95). Gates the PASS "
                         "direction only: a non-full M1 is divergence evidence and is reported as "
                         "FAIL/DRIFT however partial the overlap, because divergent generations "
                         "stop sharing positions BY CONSTRUCTION once a token differs.")
    ap.add_argument("--capinv-min-steps", type=int, default=16,
                    help="absolute floor on the number of POSITION-ALIGNED base-context (DEF) steps "
                         "required to certify a pair INERT (default 16). Coverage alone is not "
                         "enough -- two dumps that both lost the same rows to bad alignment would "
                         "agree perfectly on the little that is left -- but the floor has to sit "
                         "under what the probe actually produces: with MTP on, a 48-token probe "
                         "yields only ~30 base-context rows, because most batches are 4-token "
                         "verify steps rather than 1-token decodes.")
    ap.add_argument("--allow-model-mismatch", action="store_true",
                    help="compare against an oracle reference taken from a DIFFERENT GGUF. Off "
                         "by default and rarely what you want: logits from two models are not "
                         "comparable, so an M1 failure would be guaranteed and meaningless. "
                         "This matrix's `iq3` column once silently resolved to the IQ4_XS file, "
                         "so two columns measured one model -- the model guard exists for that.")
    ap.add_argument("--allow-stale-oracle", action="store_true",
                    help="compare against an oracle reference even when its provenance stamp "
                         "shows it was produced from older expert-cache/context code. Off by "
                         "default: a stale reference fails loudly and looks like a real pool "
                         "regression (measured 2026-09-13: M1 0/53 that looked like a bug for "
                         "hours). Use only when you have confirmed the drift is numerics-neutral.")
    ap.add_argument("--summary", action="store_true",
                    help="print the consolidated (model x pool) gate table from the records on "
                         "disk and exit. Warns when a model was measured with more than one cap.")
    ap.add_argument("--gates-only", action="store_true",
                    help="run only the correctness gates (template / union-fit / buffer-nil / "
                         "oracle M1-M2) and skip the quality suite. Use this to establish that "
                         "pool sizes differ ONLY in speed before spending time on quality.")
    ap.add_argument("--preflight-only", action="store_true",
                    help="probe the server on --port and exit (is its scaffold sane?)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (models/pools/kwargs/facts) and exit")
    args = ap.parse_args()

    # M2 mode: use the M2-specific reference oracle by default. M2 changes the prefill
    # chunk size (L2 divergence vs standard path), so it must NOT be compared against the
    # standard-path reference. The M2 reference lives at ref_<model>_pool8gb_M2.jsonl.
    if args.m2 and not args.oracle_ref and not args.dump_oracle:
        m2_ref = os.path.join(RESULT_DIR, f"ref_{args.models}_pool8gb_M2.jsonl")
        if os.path.exists(m2_ref):
            args.oracle_ref = m2_ref
            print(f"[m2] using M2 reference oracle: {m2_ref}", flush=True)
        else:
            print(f"[m2] WARNING: no M2 reference oracle at {m2_ref}; "
                  f"run with --m2 --dump-oracle on 8GB pool to create it", flush=True)

    if args.summary:
        summary_table()
        return 0

    if args.pilot:
        args.models, args.pools = "iq3,iq4", "8"
        args.profiles, args.per_profile = "qa-zh", 1
        args.greedy_repeats, args.repeats = 2, 2
        args.max_tokens, args.timeout = 64, 120.0

    args_greedy_global = args.greedy_repeats

    # EXCLUSIVITY: earlier rounds were measured with other servers live, which corrupts
    # memory and speed readings. Refuse by default instead of silently contaminating.
    if args.preflight_only:
        pf = preflight(args.port, timeout=args.preflight_timeout)
        print(json.dumps(pf, indent=2, ensure_ascii=False))
        d = os.path.join(RESULT_DIR, "preflight.json")
        os.makedirs(RESULT_DIR, exist_ok=True)
        json.dump(pf, open(d, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"wrote {d}")
        return 0 if pf.get("ok") else 1

    if args.dry_run:
        print_feasibility_matrix([m.strip() for m in args.models.split(",") if m.strip()],
                                 [int(p) for p in args.pools.split(",") if p.strip()], args)
        print(f"models      : {args.models}\npools       : {args.pools}\n"
              f"kwargs      : {args.kwargs!r}\npreflight   : {args.preflight}\n"
              f"attach      : {args.attach}\nextra_env   : {args.extra_env}\n"
              f"result dir  : {RESULT_DIR}")
        for m in (x.strip() for x in args.models.split(",") if x.strip()):
            f = model_path(m) if m in MODELS else "?"
            print(f"  {m:8s} exists={os.path.exists(f)}  {f}")
        print(f"running servers now: {running_servers() or 'none'}")
        if args.pool_cap == "auto":
            print(f"pool-cap    : auto (will probe the smallest pool "
                  f"{min(int(p) for p in args.pools.split(',') if p.strip())}GB per model "
                  f"up to cap-max={args.pool_cap_max})")
        else:
            print(f"pool-cap    : {args.pool_cap}")
        return 0

    existing = running_servers()
    if existing and args.attach:
        print(f"attaching to existing llama-server(s) {existing} (--attach)", flush=True)
    elif existing:
        if args.kill_existing:
            print(f"killing pre-existing llama-server(s): {existing}", flush=True)
            kill_servers()
        else:
            print(f"REFUSING to measure: {len(existing)} llama-server(s) already running "
                  f"(pids {existing}).\nAnother agent may be running its own pool sweep; "
                  f"results would be contaminated.\nRe-run with --kill-existing to clear them.",
                  file=sys.stderr)
            return 2

    os.makedirs(RESULT_DIR, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    pools = [int(p) for p in args.pools.split(",") if p.strip()]
    for m in models:
        if m not in MODELS:
            print(f"unknown model {m!r}; known: {', '.join(MODELS)}", file=sys.stderr)
            return 2
        p = model_path(m)
        if not os.path.exists(p):
            print(f"WARNING: model file missing for {m}: {p}", file=sys.stderr)

    # The whole grid's verdicts, before a single server is started. Cheap (GGUF tensor tables
    # only), and it turns "why was that cell refused?" into something readable up front. Ahead of
    # the model-identity gate on purpose: inspecting one model's grid is legitimate even when two
    # configured columns turn out to be the same file.
    if args.feasibility_only:
        print_feasibility_matrix(models, pools, args)
        return 0

    # Before any cap probing or launching: prove the columns really are different files. A
    # matrix of one duplicated model produces perfectly self-consistent numbers and zero
    # cross-model evidence, so this must be a hard stop, not a warning.
    if not gate_model_identity(models):
        return 2

    print(f"matrix: models={models} pools={pools} kwargs={args.kwargs!r} "
          f"tag={args.tag!r} extra_env={args.extra_env}", flush=True)

    if args.cap_invariance:
        caps = [int(c) for c in args.cap_invariance_caps.split(",") if c.strip()]
        if len(caps) < 2:
            print(f"--cap-invariance needs at least two caps; got {caps}", file=sys.stderr)
            return 2
        if any(c < MIN_CAP for c in caps):
            print(f"WARNING: cap < {MIN_CAP} aborts at llama-batch.cpp:609 "
                  f"(GGML_ASSERT(n_ubatch > n_keep_tail)); caps={caps}", file=sys.stderr)
        print(f"cap invariance: caps={caps} pools={pools} (one launch per cap per pool)",
              flush=True)
        records = []
        for kind in models:
            for gb in pools:
                records.append(cap_invariance_gate(kind, gb, caps, args))
        store = _store_capinv(records)
        print_capinv_table(records)
        print(f"\nwrote {os.path.join(RESULT_DIR, 'cap_invariance.json')} "
              f"({len(store)} stored run(s); --summary prints all of them)")
        return 0 if all(r.get("ok") is True for r in records) else 1

    # Resolve ONE pool cap per model BEFORE any measurement, so that no pool size can fall into
    # the L3-B gather path (the 4GB `buffer is nil` bug) and so that every pool is measured with
    # the same ubatch shape. This is the harness half of "4/6/8GB must differ only in speed";
    # the M1/M2 oracle gate is the verification half.
    args.pool_cap_resolved = {}
    if args.attach:
        for kind in models:
            args.pool_cap_resolved[kind] = None
        print("[cap] --attach: cannot probe, using the server's own cap", flush=True)
    elif args.pool_cap == "off":
        for kind in models:
            args.pool_cap_resolved[kind] = None
        print("[cap] --pool-cap off: leaving the build default (NOT pool-independent)", flush=True)
    elif args.pool_cap != "auto":
        for kind in models:
            args.pool_cap_resolved[kind] = int(args.pool_cap)
        print(f"[cap] fixed cap={args.pool_cap} for every pool", flush=True)
    else:
        smallest = min(pools)
        for kind in models:
            rec = probe_pool_cap(kind, smallest, args)
            # The probe just MEASURED the slots; if they disagree with the arithmetic that every
            # feasibility verdict in this sweep rests on (`cap_max`, hence PARTITION-REF vs
            # UNUSABLE), stop here -- while nothing has been measured yet.
            conflict = geometry_conflict_error(rec, args.allow_geometry_drift)
            if conflict:
                print(f"\n{conflict}", file=sys.stderr)
                return 2
            args.pool_cap_resolved[kind] = rec.get("cap")

    # Every cell's verdict, WITH the caps the sweep resolved above -- this is the table that says
    # which cells the gate below will refuse and why.
    print_feasibility_matrix(models, pools, args, args.pool_cap_resolved)

    records = []
    for kind in models:
        for gb in pools:
            records.append(run_combo(kind, gb, args))

    json.dump(records, open(os.path.join(RESULT_DIR, "matrix.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print_matrix(records)
    print(f"\nwrote {os.path.join(RESULT_DIR, 'matrix.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
