#!/usr/bin/env python3
"""engine_freeze.py -- pin (engine source state) <-> (build artifacts), and check the pin later.

WHY THIS EXISTS
---------------
Every speed and identity number in this repo is a number about a *build*, and until now nothing
recorded which build. The bill came due on 2026-09-19: the same probe, same env, same argv
produced PLAIN_MATCH TRUE at 02:57 and FALSE at 04:0x, libllama had been rebuilt in between, and
the artifact that had been linked at 02:54 was already gone from disk -- so the difference was
unbisectable after the fact. `libllama.0.0.279.dylib` is a *version number*, not an identity: it
stays `0.0.279` across rebuilds, so "the same dylib" can mean two different binaries.

`engine_digest()` in the drivers names the artifacts. This tool names the SOURCE, records both
together under a tag, and can re-check the pair afterwards:

  freeze --tag X     write one JSON: source digest + artifact digests + the flags that decide
                     them + HEAD + the dirty source list. Everything a reader needs to say
                     "these numbers are about build X".
  verify --tag X     re-check a recorded freeze against the tree as it is NOW, per section, so
                     drift is attributed ("source moved" vs "artifact replaced") instead of
                     collapsing into "the number changed".

What it does NOT claim: hermetic bit-reproducibility. A clean-dir rebuild is not attempted (this
build dir is shared with parallel sessions and is the production artifact). The claim is exactly
the one the drivers need -- "this artifact is the build of this source state, and you can prove I
did not quietly move either half" -- and `verify` is what makes that claim falsifiable.

Exit codes: 0 = ok/match, 1 = drift, 2 = usage/IO error.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "src" / "llama.cpp"
BUILD_BIN = ENGINE / "build" / "bin"
# The artifacts the launcher can load. Listed individually (no glob) so a missing one is visible
# instead of silently absent from the digest.
ARTIFACTS = ("llama-server", "libllama.0.0.279.dylib", "libllama-common.0.dylib",
             "libggml-metal.0.19.0.dylib", "libggml-base.0.19.0.dylib")
# Flags that decide the artifact's contents. A rebuild with different flags is a different build
# even from identical sources, so they belong inside the freeze, not in a human's memory.
CACHE_KEYS = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS", "LLAMA_BUILD_SERVER")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_digest() -> dict:
    """Content digest over every non-ignored file under `src/llama.cpp` except the build dir.

    `git ls-files -co --exclude-standard` is deliberate: tracked PLUS untracked-but-not-ignored.
    This tree carries uncommitted engine edits from parallel sessions, so a pure `HEAD` hash would
    describe a source state nobody built.
    """
    rel = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "--", "src/llama.cpp"],
                         cwd=str(ROOT), capture_output=True, text=True, check=True).stdout.split()
    files = sorted(r for r in rel if "/build/" not in r and not r.startswith("src/llama.cpp/build/"))
    h = hashlib.sha256()
    n_bytes = 0
    for r in files:
        p = ROOT / r
        if not p.is_file():
            continue
        st = p.stat()
        # path + size + content: a rename, a truncation and an edit all move the digest, and none
        # of them can be confused with each other.
        h.update(r.encode())
        h.update(str(st.st_size).encode())
        h.update(bytes([0]))
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        n_bytes += st.st_size
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/llama.cpp"],
                           cwd=str(ROOT), capture_output=True, text=True).stdout.splitlines()
    dirty_src = sorted(l[3:].strip() for l in dirty
                       if l.strip() and not l.startswith("??") and "/build/" not in l)
    return {"sha256": h.hexdigest(), "n_files": len(files), "n_bytes": n_bytes,
            "head": head, "dirty_sources": dirty_src}


def artifact_digest() -> dict:
    out = {}
    for name in ARTIFACTS:
        p = BUILD_BIN / name
        if not p.exists():
            out[name] = None
            continue
        st = p.stat()
        out[name] = {"sha256": _sha256_file(p)[:24], "size": st.st_size,
                     "mtime": int(st.st_mtime)}
    return out


def build_flags() -> dict:
    cache = ENGINE / "build" / "CMakeCache.txt"
    flags = {}
    if cache.exists():
        for line in cache.read_text(errors="replace").splitlines():
            k, _, v = line.partition("=")
            if k.split(":")[0] in CACHE_KEYS:
                flags[k.split(":")[0]] = v
    return flags


def snapshot() -> dict:
    return {"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": source_digest(), "flags": build_flags(), "artifacts": artifact_digest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("freeze", "verify", "show"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--dir", default=str(ROOT / "Backup" / "engine_freeze"))
    args = ap.parse_args()

    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    path = d / f"{tag}.json"

    if args.cmd == "show":
        print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "freeze":
        snap = snapshot()
        snap["note"] = ("source<->artifact pin; `engine_freeze.py verify --tag "
                        f"{tag}` re-checks it")
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=2))
        print(f"FROZEN {path}")
        s, a = snap["source"], snap["artifacts"]
        print(f"  source  sha256={s['sha256'][:24]} files={s['n_files']} bytes={s['n_bytes']}")
        print(f"  head    {s['head']}  dirty_sources={len(s['dirty_sources'])}")
        for r in s["dirty_sources"]:
            print(f"            {r}")
        print(f"  flags   {snap['flags']}")
        for k, v in a.items():
            print(f"  {k:34s} {'MISSING' if v is None else v['sha256'] + '  ' + str(v['size'])}")
        return 0

    # verify
    if not path.exists():
        print(f"ERROR: no freeze at {path}", file=sys.stderr)
        return 2
    rec = json.loads(path.read_text())
    now = snapshot()
    bad = 0
    for section in ("source", "flags", "artifacts"):
        if section == "source":
            r, n = rec[section], now[section]
            keys = ("sha256", "head", "dirty_sources")
            diffs = [f"{k}: frozen={r.get(k)!r}  now={n.get(k)!r}" for k in keys
                     if r.get(k) != n.get(k)]
        elif section == "flags":
            r, n = rec["flags"], now["flags"]
            diffs = [f"{k}: frozen={r.get(k)!r}  now={n.get(k)!r}"
                     for k in sorted(set(r) | set(n)) if r.get(k) != n.get(k)]
        else:
            r, n = rec["artifacts"], now["artifacts"]
            diffs = []
            for k in sorted(set(r) | set(n)):
                rv, nv = r.get(k) or {}, n.get(k) or {}
                if rv.get("sha256") != nv.get("sha256"):
                    diffs.append(f"{k}: frozen={rv.get('sha256')!r} (size {rv.get('size')})  "
                                 f"now={nv.get('sha256')!r} (size {nv.get('size')})")
        status = "MATCH" if not diffs else "DRIFT"
        if diffs:
            bad += len(diffs)
        print(f"{status:5s}  {section}")
        for x in diffs:
            print(f"         {x}")
    print(f"\n{'FROZEN STATE HOLDS' if not bad else f'DRIFTED ({bad} difference(s))'}: {path}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
