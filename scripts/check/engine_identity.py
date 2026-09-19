#!/usr/bin/env python3
"""Stamp a measurement product with the binary that produced it. One identity, one place.

The defect this fixes
--------------------
`attribution_timeline.py` folds both lines' products into one timeline keyed by engine digest, and
found **0 of 186** capability products carry a usable identity: `oraclecmp_*.json` (51 of them) has
only `a_path`/`b_path`, `matrix.json` and `attn_moe_split.json` have nothing, and only
`mtp_accept_ab.json` carries `provenance`/`binary`. A reading that cannot name its binary cannot be
compared with anything, so all of it is unfalsifiable-by-construction.

Two functions already compute an engine digest (`m123_oracle_gate.py`, `plain_match_ab.py`) and
`engine_freeze.py` computes a third, richer one that also names the SOURCE. This module does not add
a fourth: it loads `engine_freeze` and reuses its `ARTIFACTS` list and hashing, then writes the
result in a shape every consumer can read.

Two hash families on purpose
----------------------------
The gate summaries key on **md5**; `engine_freeze` keys on **sha256**[:24]. The same file has both,
and neither implies the other, so the stamped block carries both. Without that, a newly stamped
product would not join an existing gate summary -- which is precisely the join this exists to make.

What it will not do
-------------------
**No retrofitting.** The identity of an already-written product is not recoverable: its dump path
does not name the binary. Stamping it now would attribute the file to whichever binary happens to be
on disk, i.e. invent provenance. So the audit reports the old products as unplaceable, and they
become placeable from the next run on.

Usage
-----
    python3 scripts/check/engine_identity.py show
    python3 scripts/check/engine_identity.py audit 'Backup/phase_decomp/*.json'
    python3 scripts/check/engine_identity.py selftest

    from engine_identity import stamp          # in a product's writer
    doc = stamp(doc)                           # adds doc["engine"]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check"
# libllama carries the numerics; its md5 is what the gate summaries already key on.
PRIMARY = "libllama.0.0.279.dylib"
# The writers that call stamp(). They are part of this module's test surface: the first end-to-end
# run of the oracle-compare hook raised NameError because that file used `os` without importing it --
# after the whole comparison had already run, so the product silently never got written. The unit
# selftest could not see it (it tested THIS module, not the consumers), so the consumers are now
# checked statically below.
CONSUMERS = ("attn_moe_split.py", "cgc_logits_oracle_compare.py")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CHECK / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_dir() -> Path:
    """Where the built artifacts live. `engine_freeze` already resolves this; reuse it."""
    ef = _load("engine_freeze", "engine_freeze.py")
    for attr in ("BIN_DIR", "ARTIFACT_DIR"):
        if hasattr(ef, attr):
            return Path(getattr(ef, attr))
    return ROOT / "src" / "llama.cpp" / "build" / "bin"


def identity(bin_dir: Path | None = None) -> dict:
    """The binary this process is (or would be) measuring with: md5 + sha256 + size per artifact."""
    ef = _load("engine_freeze", "engine_freeze.py")
    d = Path(bin_dir) if bin_dir else artifact_dir()
    out: dict = {"artifacts": {}, "artifact_dir": str(d)}
    for name in ef.ARTIFACTS:
        p = d / name
        if not p.exists():
            continue
        out["artifacts"][name] = {"md5": _md5_file(p), "sha256": ef._sha256_file(p)[:24],
                                  "size": p.stat().st_size}
    if PRIMARY in out["artifacts"]:
        out["label"] = out["artifacts"][PRIMARY]["md5"][:16]
    elif out["artifacts"]:
        first = sorted(out["artifacts"])[0]
        out["label"] = out["artifacts"][first]["md5"][:16]
    else:
        out["label"] = None
        out["why_empty"] = f"no artifacts found under {d}: a product stamped here would name nothing"
    return out


def stamp(doc: dict, key: str = "engine", bin_dir: Path | None = None) -> dict:
    """Add the identity block to a product dict. In-process only, by construction.

    Called at WRITE time in the producing process. There is deliberately no `stamp_file(path)`:
    re-stamping an old file would attribute it to whatever is on disk now.
    """
    doc[key] = identity(bin_dir)
    return doc


def _join_key(md5) -> str:
    """The joinable unit: the 16-hex prefix, because that is what the gate summaries store.

    Full (32) and truncated (16) md5 name the same binary but are different strings, and keying on
    the wrong one splits one generation into two rows that look like two binaries. Observed live: the
    first stamped product produced a fake `ATTRIBUTABLE -> UNKNOWN` transition against the very gate
    summaries it exists to join. Both shapes are normalised here so they cannot disagree.
    """
    return str(md5)[:16]


def key_of(block) -> tuple[str | None, str]:
    """(key, display) for a stamped block OR a gate-style digest dict -- the READER half of stamp().

    Why this exists: the consumer did not understand its own writer's shape. `stamp()` nests the
    md5s under `artifacts.<name>.md5`, one level deeper than every digest block that existed before,
    so `attribution_timeline` classified a freshly stamped product as opaque provenance -- still
    0 of 186 placeable after the writers were fixed. A reader that does not recognise the writer's
    shape turns a fix into a silent no-op, which is why the two are tested against each other here.
    """
    if isinstance(block, dict):
        # Priority matters, and the order below is the whole reason this is one function rather than
        # a loop: a gate summary block lists EVERY artifact, so `sorted()` alone picks
        # libggml-base (it sorts before libllama) and silently re-keys every gate row to the wrong
        # binary -- two generations appear where there is one. libllama carries the numerics and is
        # checked first, exactly as the previous reader did.
        for k in (PRIMARY, "libllama.dylib"):
            v = block.get(k)
            if isinstance(v, dict) and v.get("md5"):
                return _join_key(v["md5"]), _join_key(v["md5"])
        arts = block.get("artifacts")
        if isinstance(arts, dict):
            for name in [PRIMARY] + sorted(arts):
                a = arts.get(name)
                if isinstance(a, dict) and a.get("md5"):
                    return _join_key(a["md5"]), _join_key(a["md5"])
        for k, v in sorted(block.items()):
            if isinstance(v, dict) and v.get("md5"):
                # A gate-style block keys on the BARE md5 when the artifact is the primary one, so
                # gate summaries and freshly stamped products land on the SAME generation. Prefixing
                # here would have split them into two rows that look like two binaries -- the join
                # this module exists to make, broken by a display detail.
                if k in (PRIMARY, "libllama.dylib"):
                    return _join_key(v["md5"]), _join_key(v["md5"])
                return f"{k}:{_join_key(v['md5'])}", _join_key(v["md5"])
    return None, "opaque-provenance"


# --------------------------------------------------------------------------- audit
def has_identity(doc) -> tuple[bool, str]:
    """(found, where) for one product: any block carrying a hash for a named artifact."""
    def scan(d, path, depth=0):
        if depth > 3:
            return None
        if isinstance(d, dict):
            # The dict ITSELF may be the digest block ({"md5": ...}), which is the shape
            # engine_identity stamps. Testing only its values missed exactly that case -- caught by
            # this module's own selftest, not by reading the code.
            if d.get("md5") or d.get("sha256"):
                return path
            for k, v in d.items():
                if isinstance(v, dict):
                    if v.get("md5") or v.get("sha256"):
                        return f"{path}.{k}"
                    hit = scan(v, f"{path}.{k}", depth + 1)
                    if hit:
                        return hit
                elif isinstance(v, str) and len(v) in (32, 64) and all(
                        c in "0123456789abcdef" for c in v.lower()):
                    return f"{path}.{k}"
        return None

    for top in ("engine", "engine_digest", "provenance", "binary"):
        if isinstance(doc, dict) and doc.get(top):
            v = doc[top]
            if isinstance(v, str):
                return True, top
            hit = scan(v, top)
            if hit:
                return True, hit
    return False, ""


def audit(paths: list[str]) -> dict:
    rep = {"files": 0, "products": 0, "with_identity": 0, "without": 0, "examples_without": []}
    files: list[str] = []
    for pat in paths:
        files.extend(glob.glob(str(pat)))
    for f in sorted(set(files)):
        try:
            doc = json.loads(Path(f).read_text())
        except Exception:
            continue
        rep["files"] += 1
        ok, where = has_identity(doc)
        rep["products"] += 1
        if ok:
            rep["with_identity"] += 1
        else:
            rep["without"] += 1
            if len(rep["examples_without"]) < 10:
                rep["examples_without"].append(Path(f).name)
    return rep


def cmd_audit(args) -> int:
    rep = audit(args.paths or [str(ROOT / "Backup" / "knifeedge_matrix" / "*.json"),
                              str(ROOT / "Backup" / "phase_decomp" / "*.json"),
                              str(ROOT / "Backup" / "phase_decomp" / "*" / "attn_moe_split.json")])
    print(f"products: {rep['products']}   with an engine identity: {rep['with_identity']}   "
          f"WITHOUT: {rep['without']}")
    for n in rep["examples_without"]:
        print(f"  no identity: {n}")
    print("\n(no retrofitting: an already-written product's binary is not recoverable, so these "
          "become placeable from the next run on, not now)")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))
        print(f"json -> {args.json}")
    return 0


def consumer_import_check(paths: list[Path] | None = None) -> list[tuple[str, str]]:
    """Static: every name a consumer's `stamp(...)` call uses must be bound in that same file.

    A missing import there is a runtime `NameError` at write time -- the worst place for it, because
    the measurement has already been paid for by then. Returns [(file, why)] for offenders; empty is
    the only good answer. Falsifiable by construction: it is fed a deliberately broken file below.
    """
    import ast
    import builtins

    files = [Path(p) for p in paths] if paths else [CHECK / n for n in CONSUMERS]
    problems: list[tuple[str, str]] = []
    for p in files:
        if not p.exists():
            problems.append((p.name, "missing file"))
            continue
        text = p.read_text()
        tree = ast.parse(text)
        # Everything bound anywhere in the module: imports, defs, assignments, args, loop/with/except
        # targets. Deliberately permissive -- this must never cry wolf on a legal indirection.
        # Module dunders are always bound (both real consumers use `__file__`); the checker's own
        # selftest fixture uses it too, so this allowance is tested rather than assumed.
        bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__",
                                      "__spec__", "__loader__", "__builtins__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for al in node.names:
                    bound.add(al.asname or al.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for al in node.names:
                    bound.add(al.asname or al.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
        # The unit is the whole BLOCK that stamps, not the stamp(...) call expression. The real defect
        # was `os.path.join(...)` on the statement BEFORE the stamp call -- a narrower check saw
        # nothing, which is why this checker's own negative test exists.
        def has_stamp(n):
            return any(isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "stamp"
                       for c in ast.walk(n))

        blocks = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and has_stamp(n)]
        if not blocks and has_stamp(tree):
            blocks = [tree]          # a module-level stamp call: the whole module is the block
        if not blocks:
            problems.append((p.name, "no stamp() call -- this file no longer identifies its product"))
        else:
            seen: set[str] = set()
            for blk in blocks:
                for sub in ast.walk(blk):
                    if (isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
                            and sub.id not in bound and sub.id not in seen):
                        seen.add(sub.id)
                        problems.append(
                            (p.name, f"the stamp() path uses {sub.id!r}, not bound in this file"))
            if "engine_identity" not in text:
                problems.append(
                    (p.name, "stamp() does not load engine_identity (a second stamping point?)"))
    return problems


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}")
        if not ok:
            fails.append(name)

    print("identity must reuse engine_freeze's artifact list, not invent a fourth one")
    ef = _load("engine_freeze", "engine_freeze.py")
    idn = identity()
    expect("artifact names come from engine_freeze",
           set(idn["artifacts"]) <= set(ef.ARTIFACTS), True)
    expect("the primary artifact is present", PRIMARY in idn["artifacts"], True)
    expect("label is the primary md5 prefix", idn["label"],
           idn["artifacts"][PRIMARY]["md5"][:16])

    print("\nboth hash families, because the two lines key on different ones")
    a = idn["artifacts"][PRIMARY]
    expect("md5 present (what the gate summaries carry)", len(a["md5"]), 32)
    expect("sha256 present (what engine_freeze uses)", len(a["sha256"]), 24)
    expect("md5 and sha256 are different strings", a["md5"][:24] != a["sha256"], True)
    expect("size is recorded", a["size"] > 0, True)

    print("\nthe stamp must be additive and must not mutate its input")
    doc = {"accept": 0.5}
    out = stamp(dict(doc))
    expect("the original is untouched", "engine" in doc, False)
    expect("the stamped copy carries the block", "engine" in out, True)
    expect("and the block names a binary", out["engine"]["label"], idn["label"])

    print("\nthe reader half: the block this module writes must be readable by its consumers")
    st = stamp({"accept": 0.5})["engine"]
    k, disp = key_of(st)
    expect("the stamped block gives the same label as identity()", disp, idn["label"])
    expect("and keys on the 16-hex prefix a gate summary stores", k,
           idn["artifacts"][PRIMARY]["md5"][:16])
    expect("a gate-style flat block keys on that same prefix (existing rows unmoved)",
           key_of({"libllama.0.0.279.dylib": {"md5": "a" * 32}}), ("a" * 16, "a" * 16))
    expect("an opaque block refuses rather than inventing a generation",
           key_of({"note": "hi"}), (None, "opaque-provenance"))
    # The defect this pins, live: gate summaries store the md5 TRUNCATED to 16 hex chars, stamp()
    # stores it full (32), and keying on the raw string split one binary into two rows -- a fake
    # "ATTRIBUTABLE -> UNKNOWN" transition on the very digest that was meant to be joined.
    gate_side = key_of({PRIMARY: {"md5": idn["artifacts"][PRIMARY]["md5"][:16]}})
    stamped_side = key_of(st)
    expect("a truncated gate block and a full-md5 stamped block share one generation",
           gate_side, stamped_side)

    print("\nthe audit must count products, not files, and name offenders")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "stamped.json").write_text(json.dumps(stamp({"accept": 0.5})))
        (p / "gate_like.json").write_text(json.dumps(
            {"m1_numeric_identity": "9/9", "engine_digest": {"libllama.0.0.279.dylib":
                                                             {"md5": "aab787412e572550"}}}))
        (p / "bare.json").write_text(json.dumps({"accept": 0.5, "n_common": 884}))
        rep = audit([str(p / "*.json")])
        expect("all three counted", rep["products"], 3)
        expect("two carry identity (ours + the gate shape)", rep["with_identity"], 2)
        expect("one does not", rep["without"], 1)
        expect("and it is named", rep["examples_without"], ["bare.json"])
        expect("a provenance block with a hash counts",
               has_identity({"provenance": {"md5": "abc"}})[0], True)
        expect("a provenance block WITHOUT a hash does not",
               has_identity({"provenance": {"note": "hi"}})[0], False)
        expect("a bare 64-hex string counts as identity",
               has_identity({"binary": "a" * 64})[0], True)

    print("\nan empty artifact dir must say so instead of stamping a label of None")
    import tempfile as tf2
    with tf2.TemporaryDirectory() as d2:
        e = identity(Path(d2))
        expect("no label", e["label"], None)
        expect("and a reason", "stamped here would name nothing" in e.get("why_empty", ""), True)

    print("\nthe consumers are test surface: an unbound name in their stamp() call is a NameError at")
    print("write time, i.e. after the measurement was already paid for")
    bad = consumer_import_check()
    expect("every consumer's stamp() call uses only names bound in its own file", bad, [])
    with tempfile.TemporaryDirectory() as dc:
        dcp = Path(dc)
        (dcp / "good.py").write_text(
            "import os\n"
            "import importlib.util\n"
            "def w(d):\n"
            "    _ei = importlib.util.spec_from_file_location('engine_identity', __file__)\n"
            "    p = os.path.join('scripts', 'engine_identity.py')\n"
            "    return _ei.stamp(d, p)\n")
        (dcp / "bad.py").write_text(
            "import importlib.util\n"
            "def w(d):\n"
            "    _ei = importlib.util.spec_from_file_location('engine_identity', 'x')\n"
            "    p = os.path.join('scripts', 'engine_identity.py')\n"
            "    return _ei.stamp(d, p)\n")
        (dcp / "unrelated.py").write_text("def w(d):\n    return d\n")
        good = consumer_import_check([dcp / "good.py"])
        broken = consumer_import_check([dcp / "bad.py"])
        silent = consumer_import_check([dcp / "unrelated.py"])
        expect("a legal consumer passes", good, [])
        expect("the missing import is caught, and named",
               [w for _, w in broken], ["the stamp() path uses 'os', not bound in this file"])
        expect("a file that stopped stamping is caught too",
               [w for _, w in silent],
               ["no stamp() call -- this file no longer identifies its product"])

    print()
    if fails:
        print(f"SELFTEST FAIL ({len(fails)}): {fails}")
        return 1
    print("SELFTEST PASS -- one stamping point, both hash families, and no retrofitting")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("show")
    sub.add_parser("consumers")
    a = sub.add_parser("audit")
    a.add_argument("paths", nargs="*")
    a.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "show":
        print(json.dumps(identity(), indent=1))
        return 0
    if args.cmd == "consumers":
        bad = consumer_import_check()
        for f, why in bad:
            print(f"  BROKEN {f}: {why}")
        print(f"consumers checked: {len(CONSUMERS)}   broken: {len(bad)}")
        return 1 if bad else 0
    return cmd_audit(args)


if __name__ == "__main__":
    sys.exit(main())
