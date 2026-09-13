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
        "note": "vanilla IQ4_XS (18.2 GB)",
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
        print(f"[cap-probe] {kind} pool{gb}GB cached -> cap={rec.get('cap')} "
              f"(nil={rec.get('nil')})", flush=True)
        return rec

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
        launch(model_path(kind), MODELS[kind]["mtp"], gb * 1024 ** 3, args.port,
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
    json.dump(rec, open(cache_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if rec["cap"] is None:
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


def _write_oracle_meta(path, cap, model_file=None):
    """Write the provenance sidecar: cap + source stamp + weights identity + timestamp."""
    meta = {"cap": str(cap) if cap is not None else "default",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stamp": source_stamp()}
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


def _oracle_guard(ref_path, dump_path, model_file=None, allow_stale=False,
                  allow_model_mismatch=False):
    """Return an error record when the comparison would be meaningless, else None.

    Three independent preconditions, each of which has already produced a large and entirely
    spurious "the pool broke numerics" verdict on this box: the cap (ubatch shape), the source
    provenance (engine version) and the weights identity (which model).
    """
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
    return None


def oracle_gate(port, dump_path, ref_path, label, timeout, allow_stale=False,
                model_file=None, allow_model_mismatch=False):
    """Send the deterministic probe, then compare this combo's logits oracle against a
    reference using the TWO independent metrics (never merged).

    M1 numeric identity  = same row hashes (bit-identical logits)
    M2 decision agreement = same argmax token
    Pool sizes may only differ if M1 says so; 'M2 passed' alone does NOT mean 'no difference'.
    """
    if not (dump_path and os.path.exists(dump_path)):
        return {"ok": None, "error": "no oracle dump produced"}
    if not (ref_path and os.path.exists(ref_path)):
        return {"ok": None, "error": f"reference oracle missing: {ref_path}"}
    bad = _oracle_guard(ref_path, dump_path, model_file, allow_stale, allow_model_mismatch)
    if bad:
        return bad
    ask_probe(port, timeout)
    time.sleep(1.0)
    rep = os.path.join(RESULT_DIR, f"oraclecmp_{label}.json")
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "scripts", "check", "cgc_logits_oracle_compare.py"),
                    "--a", ref_path, "--b", dump_path, "--report", rep],
                   cwd=ROOT, capture_output=True)
    if not os.path.exists(rep):
        return {"ok": None, "error": "oracle compare produced no report"}
    d = json.load(open(rep, encoding="utf-8"))
    m1, m2 = d["metrics"]["numeric_identity"], d["metrics"]["decision_agreement"]
    return {"ok": (m1["rate"] == 1.0 and m2["rate"] == 1.0),
            "m1_numeric_identity": f"{m1['equal']}/{m1['n']}",
            "m2_decision_agreement": f"{m2['equal']}/{m2['n']}",
            "cross_tab": d["cross_tab"],
            "report": rep}


def oracle_recompare(dump_path, ref_path, label, allow_stale=False, model_file=None,
                     allow_model_mismatch=False):
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
    if _oracle_guard(ref_path, dump_path, model_file, allow_stale, allow_model_mismatch):
        return None
    rep = os.path.join(RESULT_DIR, f"oraclecmp_{label}_final.json")
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "scripts", "check", "cgc_logits_oracle_compare.py"),
                    "--a", ref_path, "--b", dump_path, "--report", rep],
                   cwd=ROOT, capture_output=True)
    if not os.path.exists(rep):
        return None
    d = json.load(open(rep, encoding="utf-8"))
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
    label = combo_label(kind, gb, args.tag)
    out_json = os.path.join(RESULT_DIR, f"knifeedge_{label}.json")
    if os.path.exists(out_json) and not args.force:
        prev = json.load(open(out_json, encoding="utf-8"))
        # A combo that never came up is NOT a result. Caching it as one is the same trap as the
        # false-passing cap probe: the next run skips the cell and the table shows a config that
        # was never measured. Measured case (2026-09-11): run_server.sh's memory guard refused to
        # start (`free=15%<40%`), the stub `{"up": false}` was written, and the following run
        # printed SKIP for it. Require evidence that the server actually ran and was measured.
        if prev.get("up") and (prev.get("gates") or prev.get("skipped") is None):
            print(f"[{label}] SKIP (complete result exists; --force to redo)", flush=True)
            return prev
        print(f"[{label}] REDO: the cached result is not a measurement "
              f"(up={prev.get('up')!r}, gates={bool(prev.get('gates'))}); re-running",
              flush=True)

    launch_log = os.path.join(RESULT_DIR, f"launch_{label}.log")
    # Oracle dump is env-gated at launch; kept out of the result name so a rerun overwrites it.
    oracle_dump = os.path.join(RESULT_DIR, f"oracle_{label}.jsonl")
    extra = list(args.extra_env)
    # The union-routable gate must not depend on an OPTIONAL diagnostic flag the caller happened
    # to pass: on 2026-09-14 the 2 GiB cell scored M1/M2/M3 117/117 while the gate said FAIL
    # mode=unreachable simply because CGC_UNION_LOG was not set, so the `gather=N/M` evidence it
    # reads had never been written. "Unknown" and "not routable" must not look the same -- the
    # harness therefore always asks the engine for the evidence it is about to judge. (Without
    # this the gate can only ever pass a config whose FIRST version was measured by hand.)
    extra.append("CGC_UNION_LOG=1")
    cap = cap_for(args, kind)
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
        _write_oracle_meta(oracle_dump, cap if cap is not None else args.pool_cap_max,
                           model_file=model_path(kind))
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
        launch(model_path(kind), model["mtp"], gb * 1024 ** 3, args.port, args.kwargs,
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
               "free_before_pct": free_before, "launch_tail": tail}
        json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        kill_servers()
        return rec

    if not args.attach:
        facts = read_launch_facts(launch_log)
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
                               allow_model_mismatch=args.allow_model_mismatch)
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
                                         allow_model_mismatch=args.allow_model_mismatch)
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
                       "pool_cap": cap,
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
                                         allow_model_mismatch=args.allow_model_mismatch)

    if args.gates_only:
        print(f"[{label}] --gates-only: skipping the quality suite", flush=True)
        if not args.attach:
            kill_servers()
        rec = {"label": label, "model": kind, "pool_gb": gb, "up": True,
               "preflight": rec_pf,
               "gates": {"template": g_template, "buffer_nil": g_nil, "union_fit": g_union,
                         "oracle": g_oracle,
                         "oracle_final": g_oracle_final},
               "pool_cap": cap, "gates_only": True,
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
           "pool_cap": cap,
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
    json.dump(rec, open(os.path.join(RESULT_DIR, f"combo_{label}.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    return rec


def print_matrix(records):
    print()
    print("=" * 132)
    print("%-26s %5s %7s %8s %8s %7s %7s %8s %5s %6s %6s %14s" % (
        "config", "rss", "free%", "greedy", "seeded", "std", "flips", "tok/s",
        "cap", "tmpl", "nil", "M1/M2"))
    print("-" * 138)
    for r in records:
        if not r.get("up"):
            # Name the REASON. A cell that never launched because the MACHINE was out of memory
            # is not evidence about the model, and printing it as "did not come up" invites
            # exactly that misreading (and, before the skip-rule fix, a cached stub).
            why = {"low_memory": "machine below --min-free-pct -> NOT measured, nothing cached",
                   "foreign_server": "another process owns the port -> NOT measured"}.get(
                       r.get("skipped"), f"server did not come up ({r.get('model')})")
            print("%-26s   -- %s" % (r["label"], why))
            continue
        g = r.get("gates") or {}
        gt = (g.get("template") or {}).get("ok")
        gts = "n/a" if gt is None else ("ok" if gt else "FAIL")
        gn = (g.get("buffer_nil") or {}).get("ok")
        gns = "n/a" if gn is None else ("ok" if gn else "FAIL")
        go = g.get("oracle") or {}
        mpair = f"{go.get('m1_numeric_identity','-')}/{go.get('m2_decision_agreement','-')}"
        print("%-26s %5s %7s %8s %8s %7s %7s %8s %5s %6s %6s %14s" % (
            r["label"], r.get("rss_gb"), r.get("free_after_pct"),
            r.get("greedy_all_pass_rate"), r.get("suite_rate_mean"),
            r.get("suite_rate_std"), r.get("seeded_flip_questions"),
            r.get("tps_mean"), r.get("pool_cap", "-"), gts, gns, mpair))
    print("=" * 108)
    print("greedy   = all-%d-greedy-repeats pass rate (determinism side)" % args_greedy_global)
    print("seeded   = mean suite pass rate over fixed seeds (sampling side)")
    print("std      = spread of the per-seed suite rate -> the uncertainty the pool-floor")
    print("           decision has to survive. A pool ranking INSIDE one std is not a ranking.")
    print("cap      = CGC_POOL_MAX_TOKENS actually launched; MUST be identical down the column")
    print("           for a given model, else the pool sizes are not the only variable")
    print("flips    = questions whose verdict changes across seeds")
    print("tok/s    = mean decode tok/s over every request in the session")


def summary_table():
    """One row per (model x pool) with all four gates, assembled from the combo records.

    Read the columns together, not one at a time:
      cap must be IDENTICAL down a model's rows, or the pools were not the only variable;
      union-fit is the arithmetic reason the gather path is unreachable;
      nil is that prediction OBSERVED (nil PASS with union-fit n/a proves nothing);
      M1/M2 must stay apart -- M2 agreement on its own is not "no difference".
    """
    import glob
    recs = []
    for p in sorted(glob.glob(os.path.join(RESULT_DIR, "combo_*.json"))):
        try:
            recs.append(json.load(open(p, encoding="utf-8")))
        except Exception:  # noqa: BLE001 - harness
            continue
    if not recs:
        print("no combo records yet")
        return
    order = {"iq3": 0, "iq4": 1}
    recs.sort(key=lambda r: (order.get(r.get("model"), 9), r.get("pool_gb", 0)))
    print("=" * 126)
    print("%-16s %4s %6s %6s %6s %9s %6s %10s %12s %7s" % (
        "config", "cap", "rss", "free%", "tmpl", "union-fit", "nil", "M1 bit-id",
        "M2 argmax", "preflt"))
    print("-" * 126)

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

    for r in recs:
        g = r.get("gates") or {}
        t = (g.get("template") or {}).get("ok")
        n = (g.get("buffer_nil") or {}).get("ok")
        u = (g.get("union_fit") or {})
        o = (g.get("oracle") or {})
        pf = (r.get("preflight") or {}).get("ok")
        m1 = o.get("m1_numeric_identity")
        m2 = o.get("m2_decision_agreement")
        uf = mark(u.get("ok"))
        if u.get("headroom") is not None:
            uf += f" ({u['union_max']}<={u.get('usable_slots')}, {u['headroom']:+d})"
        print("%-16s %4s %6s %6s %6s %-9s %6s %10s %12s %7s" % (
            r.get("label"), r.get("pool_cap", "-"), r.get("rss_gb"),
            r.get("free_after_pct"), mark(t), uf, mark(n),
            pct(m1), pct(m2), mark(pf)))
    print("-" * 126)
    caps = {}
    for r in recs:
        caps.setdefault(r.get("model"), set()).add(r.get("pool_cap"))
    bad = [m for m, s in caps.items() if len(s) > 1]
    if bad:
        print(f"WARNING: these models were measured with MORE THAN ONE cap {bad} -- "
              f"the pool sizes are not the only variable; the rows are not comparable.")
    else:
        print("cap is identical within each model -> pool size is the only variable. OK")
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
    ap.add_argument("--dump-oracle", action="store_true",
                    help="write the logits oracle dump WITHOUT comparing it to anything. This is "
                         "how a reference is produced: previously the dump was only written as a "
                         "side effect of --oracle-ref, so making a fresh reference required "
                         "pointing at some other dump first -- a chicken-and-egg that made the "
                         "reference's own provenance accidental.")
    ap.add_argument("--pool-cap", default="auto",
                    help="CGC_POOL_MAX_TOKENS, IDENTICAL for every pool (default 'auto' = probe "
                         "the smallest pool until it reports no `buffer is nil`, then reuse that "
                         "cap everywhere; 'off' leaves the build default). One shared cap is "
                         "required because the cap clamps n_batch and so the ubatch shape.")
    ap.add_argument("--pool-cap-max", type=int, default=8,
                    help="largest cap 'auto' may pick (build default is 8)")
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

    # Before any cap probing or launching: prove the columns really are different files. A
    # matrix of one duplicated model produces perfectly self-consistent numbers and zero
    # cross-model evidence, so this must be a hard stop, not a warning.
    if not gate_model_identity(models):
        return 2

    print(f"matrix: models={models} pools={pools} kwargs={args.kwargs!r} "
          f"tag={args.tag!r} extra_env={args.extra_env}", flush=True)

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
            args.pool_cap_resolved[kind] = rec.get("cap")

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
