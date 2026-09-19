#!/usr/bin/env python3
"""One timeline: M1/M2/M3 readings and on/off attribution, segmented by engine digest.

The question this exists to answer
---------------------------------
"An engine change moved M1/M2/M3 -- did attribution move with it?" That question cannot be asked of
a flat list of results, because the two numbers live in different files and, more importantly, they
are only comparable inside one binary generation. A digest IS the generation. So the timeline is
segmented by `engine_digest`, and each segment is asked three separate things:

  1. What did the gate read? (only runs with `comparable: true` count as readings)
  2. Is the on/off difference attributable to the flag on this binary? (only a NULL CELL says so)
  3. If attribution is unknown here, is that because nobody ran the control, or because it ran and
     its arms failed? Those are different holes and they need different repairs.

The thing it must not do
------------------------
Not fuse readings across digests into one series, and not let "no attribution evidence" read as
"attribution fine". On this corpus that second trap is the live one: 112 gate summaries exist and
two attribution products do. Most binaries therefore have a reader rating and NO answer to the
attribution question -- which the report says out loud rather than leaving blank.

Usage
-----
    python3 scripts/check/attribution_timeline.py show
    python3 scripts/check/attribution_timeline.py show --md docs/ATTRIBUTION_TIMELINE.md
    python3 scripts/check/attribution_timeline.py selftest
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check"
GATE_DIR = ROOT / "Backup" / "m123_oracle_gate"
ATTR_GLOBS = ("Backup/phase_decomp/*null_cell*.json",
              "Backup/phase_decomp/*onset*.json",
              "Backup/phase_decomp/*/*null_cell*.json")

# -------------------------------------------------------------------------------------------------
# The OTHER line's measurement products, folded in by SHAPE rather than by filename.
#
# The two lines answer different questions about the same binary: this line asks whether an on/off
# difference is attributable (null cell) and what the oracle gate read (M1/M2/M3); the other line
# decomposes WHERE a step's time goes (KIND x OP, per-layer gap tables), sweeps the pool geometry,
# and records accept. A digest can have one and not the other, and that asymmetry is the thing worth
# seeing: "M1/M2/M3 identical" says nothing about where the time went, and a decomposition says
# nothing about whether the numerics moved.
# -------------------------------------------------------------------------------------------------
CAPABILITY_GLOBS = ("Backup/knifeedge_matrix/*.json",
                    "Backup/phase_decomp/attn_moe_split*.json",
                    "Backup/phase_decomp/*/attn_moe_split.json",
                    "Backup/phase_decomp/oracle_*.json",
                    # the same family is written here by the compare tool (51 already live under
                    # knifeedge_matrix/); without this line a freshly stamped product is invisible
                    # and the 0/186 cannot move no matter how correct the stamp is
                    "Backup/phase_decomp/oraclecmp*.json",
                    "Backup/phase_decomp/en_*.json",
                    "Backup/phase_decomp/m1_speed*.json",
                    "Backup/cgc_logs/mtp_accept_ab*.json",
                    "Backup/cgc_logs/cap_sweep*.json",
                    "Backup/cgc_logs/union_probe*.json")

# capability names as used in the report
CAP_READING = "m1m2m3-reading"
CAP_ATTRIBUTION = "onoff-attribution"
CAP_DECOMP = "kindxop-decomposition"
CAP_ACCEPT = "mtp-accept"
CAP_GEOMETRY = "pool-geometry"
CAP_LATENCY = "step-latency"


def classify_product(doc, path: str) -> tuple[str | None, dict | None]:
    """(capability, digest) for one product, decided by its SHAPE.

    Deliberately shape-based: the two lines name files differently and always will, so a filename
    list would silently stop covering new products. Returns (None, None) for a product whose shape
    is not recognised, and the caller reports those by name rather than dropping them -- an
    unreadable product is a hole, not a zero.
    """
    def dig(d):
        # `provenance`/`binary` are the names the other line's products already use (mtp_accept_ab
        # carries them). Accepting them is not cosmetic: without it, 186 capability products fold in
        # with no identity and cannot be attached to any generation.
        for k in ("engine_digest", "digest", "engine", "provenance", "engine_provenance",
                  "binary"):
            if isinstance(d, dict) and d.get(k):
                return d[k]
        return None

    # a matrix-style product: a LIST of row dicts, or a dict carrying one
    rows = None
    if isinstance(doc, list) and doc and isinstance(doc[0], dict):
        rows = doc
    elif isinstance(doc, dict):
        for k in ("rows", "results", "combo", "combos"):
            if isinstance(doc.get(k), list) and doc[k] and isinstance(doc[k][0], dict):
                rows = doc[k]
                break
    if rows is not None:
        keys = set().union(*(set(r) for r in rows[:5]))
        d = next((dig(r) for r in rows if dig(r)), None)
        if {"m1_numeric_identity"} & keys or "cross_tab" in keys or "comparable" in keys:
            return CAP_READING, d
        if {"accept", "requests"} & keys:
            return CAP_ACCEPT, d
        if {"slots", "capacity", "pool_gb", "geometry"} & keys:
            return CAP_GEOMETRY, d
        return None, None

    if not isinstance(doc, dict):
        return None, None
    keys = set(doc)
    if "null_cells" in keys or "onoff_separated" in keys:
        return CAP_ATTRIBUTION, dig(doc)
    if "m1_numeric_identity" in keys or "cross_tab" in keys:
        return CAP_READING, dig(doc)
    if "n_tables" in keys or ({"tables", "kinds", "ops"} & keys):
        return CAP_DECOMP, dig(doc)
    if {"accept", "requests"} <= keys:
        return CAP_ACCEPT, dig(doc)
    if {"mean_len", "step_ms"} & keys or "cb" in keys or "gap" in keys:
        return CAP_LATENCY, dig(doc)
    if {"slots", "capacity", "pool_gb"} & keys:
        return CAP_GEOMETRY, dig(doc)
    return None, None


def collect_capabilities(paths=None) -> tuple[list[dict], list[str]]:
    """(capability rows, unrecognised product names). Coverage is returned, never dropped."""
    files: list[str] = []
    for pat in (paths or CAPABILITY_GLOBS):
        files.extend(glob.glob(str(ROOT / pat)))
    out, unknown = [], []
    for f in sorted(set(files)):
        try:
            doc = json.loads(Path(f).read_text())
        except Exception:
            unknown.append(Path(f).name + " (unreadable)")
            continue
        cap, digest = classify_product(doc, Path(f).name)
        if cap is None:
            unknown.append(Path(f).name)
            continue
        key, disp = digest_key(digest)
        out.append({"when": Path(f).stat().st_mtime, "file": Path(f).name, "capability": cap,
                    "digest_key": key, "digest_disp": disp})
    return out, unknown


def _load(name: str, filename: str):
    """Load a sibling by path -- the convention in this directory (there is no package)."""
    spec = importlib.util.spec_from_file_location(name, CHECK / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def digest_key(digest) -> tuple[str | None, str]:
    """(key, display). The key is libllama's md5 when present: it is the file that carries the
    numerics. Falling back to a hash of the whole dict keeps an unusual digest from merging with
    another one -- two generations must never fuse into one row."""
    if isinstance(digest, str):
        # A binary path or a commit id: readable as a label, hashed as a key. It can never merge
        # with an md5-keyed generation, which is the property that matters.
        base = digest.rsplit("/", 1)[-1][:24] or "string-id"
        return hashlib.sha256(digest.encode()).hexdigest(), base
    if not isinstance(digest, dict) or not digest:
        return None, "no-digest"
    # Delegate to the writer's own rule instead of keeping a third copy of it. `stamp()` nests the
    # md5s under artifacts.<name>.md5 -- one level deeper than the older blocks -- so this reader
    # classified the newly stamped products as opaque and the 0/186 did not move. Writer and reader
    # are now tested against each other in both modules' selftests.
    return _load("engine_identity", "engine_identity.py").key_of(digest)
    # A provenance blob with no hash is NOT an identity. Hashing it would look harmless and is not:
    # it fragments every unknown-provenance product into its own fake "generation", so the report
    # grows rows that are not generations and hides the real finding (these products cannot be
    # placed). Refusing them makes the gap countable instead. (An earlier version hashed them and
    # produced a dozen `aggregate` rows -- the selftest now pins the refusal.)
    return None, "opaque-provenance"


def collect_gates(paths=None) -> list[dict]:
    """Every gate summary, classified by the SAME rule the round runner uses."""
    mw = _load("mw", "m123_gate_window.py")
    out = []
    for f in sorted(glob.glob(str(GATE_DIR / "summary_*.json")) if not paths else paths):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        r = mw.verdict_from_summary(d)
        key, disp = digest_key(d.get("engine_digest"))
        out.append({"when": Path(f).stat().st_mtime, "file": Path(f).name, "tag": d.get("tag"),
                    "profile": d.get("profile"), "comparable": r.get("comparable"),
                    "published_reading": bool(r.get("published_reading")),
                    "m1": r.get("m1"), "m2": r.get("m2"), "m3": r.get("m3"),
                    "n": r.get("n_compared"), "digest_key": key, "digest_disp": disp,
                    "tree": d.get("tree"), "config_diffs": len(r.get("config_diffs") or [])})
    return out


def collect_attribution(paths=None) -> list[dict]:
    """Every sweep / null-cell product, classified by the SAME rule the round runner uses."""
    mw = _load("mw", "m123_gate_window.py")
    files: list[str] = []
    for pat in (paths or ATTR_GLOBS):
        files.extend(glob.glob(str(ROOT / pat)))
    out = []
    seen = set()
    for f in sorted(set(files)):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "arms" not in d:
            continue
        r = mw.verdict_from_sweep(d)
        key, disp = digest_key(r.get("engine_digest"))
        # a product with no arms that ran carries no attribution evidence at all
        ran = [a for a in d.get("arms", []) if not a.get("error")]
        out.append({"when": Path(f).stat().st_mtime, "file": Path(f).name,
                    "onoff_separated": r.get("onoff_separated"),
                    "null_cell_status": r.get("null_cell_status"),
                    "sweep_verdict": r.get("sweep_verdict"),
                    "onset": r.get("onset_positions"), "arms_ran": len(ran),
                    "digest_key": key, "digest_disp": disp})
        seen.add(f)
    return out


def attribution_of(items: list[dict]) -> tuple[str, str]:
    """Fold this digest's attribution evidence into one state, conflicts included.

    UNKNOWN has three distinct causes and they are returned verbatim, because "we never ran the
    control" and "we ran it and it could not launch" need different work.
    """
    if not items:
        return "UNKNOWN", "no null-cell product for this digest"
    known = [i for i in items if i["onoff_separated"] is not None]
    if not known:
        why = "; ".join(sorted({str(i.get("null_cell_status")) for i in items}))
        return "UNKNOWN", f"product(s) exist but none carries a null cell ({why})"
    yes = [i for i in known if i["onoff_separated"]]
    if len(yes) == len(known):
        return "ATTRIBUTABLE", f"{len(yes)}/{len(known)} control(s) clean"
    if not yes:
        return "NOT-ATTRIBUTABLE", f"{len(known)} control(s) divergent"
    return "CONFLICTING", f"{len(yes)}/{len(known)} clean -- the same digest disagrees with itself"


def build(gates: list[dict], attrs: list[dict], caps: list[dict] | None = None) -> dict:
    """Pure: segment every series by digest and answer the three questions per segment."""
    caps = caps or []
    keys: dict[str, dict] = {}

    def seg(key, disp, when):
        s = keys.setdefault(key, {"key": key, "disp": disp, "first": when, "last": when,
                                  "gates": [], "attrs": [], "caps": []})
        s["first"] = min(s["first"], when)
        s["last"] = max(s["last"], when)
        return s

    for g in gates:
        if not g["digest_key"]:
            continue
        seg(g["digest_key"], g["digest_disp"], g["when"])["gates"].append(g)
    for a in attrs:
        if not a["digest_key"]:
            continue
        seg(a["digest_key"], a["digest_disp"], a["when"])["attrs"].append(a)
    for c in caps:
        if not c["digest_key"]:
            continue
        seg(c["digest_key"], c["digest_disp"], c["when"])["caps"].append(c)

    order = sorted(keys.values(), key=lambda s: s["first"])
    for s in order:
        s["gates"].sort(key=lambda x: x["when"])
        s["attrs"].sort(key=lambda x: x["when"])
        s["readings"] = [g for g in s["gates"] if g["published_reading"]]
        s["void"] = [g for g in s["gates"] if not g["published_reading"]]
        s["reading"] = (s["readings"][-1] if s["readings"] else None)
        s["attribution"], s["attribution_why"] = attribution_of(s["attrs"])
        s["capabilities"] = sorted({c["capability"] for c in s["caps"]})
        holes = []
        if not s["readings"]:
            holes.append("no comparable M1/M2/M3 reading")
        # The cross-line questions: the two lines measure different things about the same binary, so
        # a digest carrying one capability says nothing about the other. Naming the missing one is
        # the whole point of folding both lines into one timeline.
        if s["readings"] and CAP_DECOMP not in s["capabilities"]:
            holes.append("no per-layer KIND x OP decomposition on this binary (where the time went)")
        if CAP_DECOMP in s["capabilities"] and not s["readings"]:
            holes.append("decomposition exists but no comparable oracle reading on this binary")
        if s["attribution"] == "UNKNOWN":
            holes.append("attribution unknown: " + s["attribution_why"])
        if any(g["comparable"] is False for g in s["gates"]):
            holes.append(f"{len(s['void'])} void comparison(s) on this digest (recorded, not read)")
        s["holes"] = holes

    transitions = []
    for a, b in zip(order, order[1:]):
        transitions.append({
            "from": a["disp"], "to": b["disp"],
            "reading_before": (a["reading"] or {}).get("m1"),
            "reading_after": (b["reading"] or {}).get("m1"),
            "attribution_before": a["attribution"], "attribution_after": b["attribution"],
            "attribution_changed": (a["attribution"] != b["attribution"]
                                    and "UNKNOWN" not in (a["attribution"], b["attribution"])),
            "attribution_unknowable": "UNKNOWN" in (a["attribution"], b["attribution"]),
        })
    return {"segments": order, "transitions": transitions}


def render(rep: dict, md: bool = False) -> str:
    segs, trs = rep["segments"], rep["transitions"]
    L: list[str] = []
    if md:
        L.append("# M1/M2/M3 與 on/off 歸因：按 engine digest 分段的時間軸\n")
        L.append(f"（產於 {time.strftime('%Y-%m-%d %H:%M')}；讀數只在同一 digest 內可比）\n")
        L.append("| digest | 首次 | 讀數 | M1/M2/M3 | 歸因 | 兩條線已行使的能力 | 這一顆答不出來的事 |")
        L.append("|---|---|---|---|---|---|---|")
        for s in segs:
            r = s["reading"]
            rd = (f"{r['m1']} / {r['m2']} / {r['m3']}" if r else "—")
            L.append(f"| `{s['disp']}` | {time.strftime('%m-%d %H:%M', time.localtime(s['first']))} "
                     f"| {len(s['readings'])}/{len(s['gates'])} | {rd} | "
                     f"**{s['attribution']}** | {', '.join(s['capabilities']) or '—'} | "
                     f"{'; '.join(s['holes']) or '—'} |")
    else:
        L.append(f"segments (engine generations): {len(segs)}   transitions: {len(trs)}")
        L.append("")
        L.append(f"{'digest':<18}{'first':<13}{'read':<7}{'M1/M2/M3':<24}{'attribution':<18}"
                 f"{'capabilities':<34}holes")
        for s in segs:
            r = s["reading"]
            rd = (f"{r['m1']} / {r['m2']} / {r['m3']}" if r else "—")
            L.append(f"{s['disp']:<18}{time.strftime('%m-%d %H:%M', time.localtime(s['first'])):<13}"
                     f"{str(len(s['readings'])) + '/' + str(len(s['gates'])):<7}{rd:<24}"
                     f"{s['attribution']:<18}{','.join(s['capabilities'])[:33]:<34}"
                     f"{'; '.join(s['holes'])[:60]}")
    L.append("")
    L.append("## 每次引擎改動：歸因有沒有跟著變")
    if not trs:
        L.append("（只有一個 digest，沒有可比較的改動）")
    for t in trs:
        moved = ("改動" if t["attribution_changed"] else
                 "**答不出來**" if t["attribution_unknowable"] else "沒變")
        L.append(f"- `{t['from']}` → `{t['to']}`：歸因 {t['attribution_before']} → "
                 f"{t['attribution_after']}（{moved}）；M1 {t['reading_before']} → "
                 f"{t['reading_after']}")
    holes = [s for s in segs if s["holes"]]
    L.append("")
    L.append(f"## 缺口：{len(holes)}/{len(segs)} 個 digest 有東西答不出來")
    for s in holes:
        L.append(f"- `{s['disp']}`：{'; '.join(s['holes'])}")
    return "\n".join(L)


def cmd_show(args) -> int:
    gates, attrs = collect_gates(), collect_attribution()
    caps, unknown = collect_capabilities()
    rep = build(gates, attrs, caps)
    txt = render(rep, md=bool(args.md))
    print(txt)
    if args.md:
        Path(args.md).write_text(txt + "\n")
        print(f"\nmd -> {args.md}")
    if args.json:
        Path(args.json).write_text(json.dumps({"segments": rep["segments"],
                                               "transitions": rep["transitions"]}, indent=1))
        print(f"json -> {args.json}")
    # Coverage, so an empty-looking timeline can be told apart from a broken collector.
    print(f"\ncoverage: {len(gates)} gate summaries, {len(attrs)} attribution product(s), "
          f"{len(caps)} capability product(s) read")
    by_cap: dict[str, int] = {}
    for c in caps:
        by_cap[c["capability"]] = by_cap.get(c["capability"], 0) + 1
    print(f"  by capability: {by_cap}")
    placeable = [c for c in caps if c["digest_key"]]
    unplaceable = [c for c in caps if not c["digest_key"]]
    print(f"  placeable on a generation: {len(placeable)}/{len(caps)}")
    if unplaceable:
        print(f"[coverage] {len(unplaceable)} capability product(s) carry no usable engine identity and "
              f"cannot be joined to a binary at all (provenance without a hash is not an identity): "
              f"{sorted({c['file'] for c in unplaceable})[:5]}")
    if unknown:
        print(f"[coverage] {len(unknown)} product(s) matched a capability glob but their SHAPE was "
              f"not recognised, so they carry no capability here: {unknown[:6]}"
              f"{'...' if len(unknown) > 6 else ''}")
    no_key = sum(1 for g in gates if not g["digest_key"])
    if no_key:
        print(f"[coverage] {no_key} summary file(s) carry no engine digest and cannot be placed on "
              f"the timeline at all")
    return 0


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            fails.append(name)

    def gate(when, key, disp, m1="9/9", comp=True, m3="9/9"):
        return {"when": when, "digest_key": key, "digest_disp": disp, "comparable": comp,
                "published_reading": bool(comp), "m1": m1, "m2": "9/9", "m3": m3, "n": 9,
                "tag": disp, "profile": "prefill250"}

    def attr(when, key, disp, sep, status="compared"):
        return {"when": when, "digest_key": key, "digest_disp": disp, "onoff_separated": sep,
                "null_cell_status": status, "sweep_verdict": "fixed-onset", "onset": [None, 95],
                "arms_ran": 3, "file": disp + ".json"}

    print("a digest with a clean control is ATTRIBUTABLE; one with a diverging control is not")
    rep = build([gate(1, "AAA", "aaa"), gate(2, "BBB", "bbb")],
                [attr(1, "AAA", "aaa", True), attr(2, "BBB", "bbb", False)])
    a, b = rep["segments"]
    expect("AAA attributable", a["attribution"], "ATTRIBUTABLE")
    expect("BBB not attributable", b["attribution"], "NOT-ATTRIBUTABLE")
    expect("the transition reports that attribution moved",
           rep["transitions"][0]["attribution_changed"], True)

    print("\nM1/M2/M3 with no control is UNKNOWN -- and it must not read as fine")
    rep = build([gate(1, "AAA", "aaa")], [])
    expect("attribution is UNKNOWN", rep["segments"][0]["attribution"], "UNKNOWN")
    expect("and it is listed as a hole",
           any("attribution unknown" in h for h in rep["segments"][0]["holes"]), True)
    expect("a transition into UNKNOWN is flagged as unanswerable",
           build([gate(1, "AAA", "a"), gate(2, "BBB", "b")],
                 [attr(1, "AAA", "a", True)])["transitions"][0]["attribution_unknowable"], True)
    expect("and it is NOT reported as a change",
           build([gate(1, "AAA", "a"), gate(2, "BBB", "b")],
                 [attr(1, "AAA", "a", True)])["transitions"][0]["attribution_changed"], False)

    print("\nthe three causes of UNKNOWN are distinguished")
    _, why = attribution_of([])
    expect("no product at all", "no null-cell product" in why, True)
    _, why = attribution_of([attr(1, "AAA", "a", None, "requested-but-arms-failed")])
    expect("control ran but its arms failed", "requested-but-arms-failed" in why, True)
    st, _ = attribution_of([attr(1, "AAA", "a", True), attr(2, "AAA", "a", False)])
    expect("a digest whose controls disagree is CONFLICTING", st, "CONFLICTING")

    print("\nvoid comparisons are recorded but never counted as readings")
    rep = build([gate(1, "AAA", "aaa", m1="1/9", comp=False)], [])
    s = rep["segments"][0]
    expect("no reading on that digest", s["reading"], None)
    expect("it is listed as a hole", any("no comparable M1/M2/M3 reading" in h for h in s["holes"]),
           True)
    expect("and the void run is named as such",
           any("void comparison" in h for h in s["holes"]), True)

    print("\nthe other line's products must be folded in by SHAPE, not by filename")
    expect("a list-of-rows matrix with m1 is a reading",
           classify_product([{"tag": "a", "m1_numeric_identity": "9/9", "comparable": True}],
                            "matrix.json")[0], CAP_READING)
    expect("an accept product is MTP accept",
           classify_product({"accept": 0.58, "requests": []}, "x.json")[0], CAP_ACCEPT)
    expect("a KIND x OP table is a decomposition",
           classify_product({"n_tables": 928, "tables": []}, "attn_moe_split.json")[0], CAP_DECOMP)
    expect("a per-layer table with gap/cb is step latency",
           classify_product({"cb": 14.9, "mean_len": 2.4}, "x.json")[0], CAP_LATENCY)
    expect("a null-cell product stays attribution",
           classify_product({"null_cells": [], "arms": []}, "x.json")[0], CAP_ATTRIBUTION)
    expect("an unrecognised shape returns None (and is reported, not dropped)",
           classify_product({"hello": 1}, "x.json")[0], None)
    expect("a product whose shape is unknown is not counted as a capability",
           collect_capabilities(["/nonexistent/*.json"])[0], [])

    print("\na product stamped by engine_identity must be PLACEABLE (writer and reader shapes agree)")
    ei = _load("engine_identity", "engine_identity.py")
    stamped = ei.stamp({"cross_tab": {"num_eq_dec_eq": 884}, "n_common": 884})
    cap, dig = classify_product(stamped, "oraclecmp_x.json")
    key, disp = digest_key(dig)
    expect("classified as an M1/M2/M3 reading", cap, CAP_READING)
    expect("and it joins the gate's generation (this is the 0/186 fix)", disp, ei.identity()["label"])
    expect("a product with NO identity is still unplaceable", digest_key(None)[0], None)
    expect("and opaque provenance is still refused, not hashed into a fake generation",
           digest_key({"note": "hi"})[0], None)

    print("\na digest with a reading but no decomposition must say so")
    rep = build([gate(1, "AAA", "aaa")], [],
                [{"when": 1, "digest_key": "AAA", "digest_disp": "aaa",
                  "capability": CAP_ACCEPT, "file": "x.json"}])
    s = rep["segments"][0]
    expect("the capability is listed", s["capabilities"], [CAP_ACCEPT])
    expect("the missing decomposition is a named hole",
           any("KIND x OP decomposition" in h for h in s["holes"]), True)
    rep = build([], [], [{"when": 1, "digest_key": "AAA", "digest_disp": "aaa",
                          "capability": CAP_DECOMP, "file": "y.json"}])
    expect("and decomposition without a reading is the mirror-image hole",
           any("decomposition exists but no comparable" in h for h in rep["segments"][0]["holes"]),
           True)
    rep = build([gate(1, "AAA", "aaa")], [],
                [{"when": 1, "digest_key": "AAA", "digest_disp": "aaa",
                  "capability": CAP_DECOMP, "file": "y.json"}])
    expect("both present -> neither hole",
           [h for h in rep["segments"][0]["holes"] if "decomposition" in h], [])

    print("\ntwo generations must never fuse into one row")
    rep = build([gate(1, "AAA", "aaa"), gate(2, "BBB", "bbb")], [])
    expect("two segments, not one", len(rep["segments"]), 2)
    expect("the older digest keeps its own first-seen",
           rep["segments"][0]["first"] < rep["segments"][1]["first"], True)
    expect("a summary with no digest cannot be placed", digest_key(None)[0], None)
    expect("an unusual digest does not merge with another",
           digest_key({"x": {"md5": "1"}})[0] != digest_key({"y": {"md5": "1"}})[0], True)
    expect("a provenance block with an md5 under another key still identifies a generation",
           digest_key({"build": {"md5": "abc"}})[0], "build:abc")
    expect("a binary path is usable as an identity", digest_key("/x/y/libllama.dylib")[1],
           "libllama.dylib")
    expect("opaque provenance is NOT a generation", digest_key({"a": 1, "b": "x"})[0], None)
    expect("and it says why", digest_key({"a": 1})[1], "opaque-provenance")
    expect("a provenance dict WITH a hash still identifies",
           digest_key({"build": {"md5": "abc"}})[0], "build:abc")
    expect("a path identity can never collide with an md5 identity",
           digest_key("/x/y/libllama.dylib")[0] != digest_key({"libllama.dylib": {"md5": "z"}})[0],
           True)
    expect("the other line's products carry provenance/binary, so it is accepted",
           classify_product({"accept": 0.5, "requests": [], "provenance": {"md5": "abc"}},
                            "m.json")[1], {"md5": "abc"})

    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)}): {fails}")
        return 1
    print("SELFTEST PASS -- readings and attribution are joined per digest, and UNKNOWN stays UNKNOWN")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("show")
    s.add_argument("--md", default=None)
    s.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return selftest() if args.cmd == "selftest" else cmd_show(args)


if __name__ == "__main__":
    sys.exit(main())
