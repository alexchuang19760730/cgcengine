#!/usr/bin/env python3
"""knifeedge.identity — 由 split_module.py 從 scripts/check/knifeedge_matrix.py 機械拆出。

這個模組的內容逐位元組來自原檔（含註解），除了 map 裡 declared_edits 明列、
且由 check_module_split.py 逐字重建驗過的那幾處。來源修訂與對帳見
agent_harness/shared/knifeedge_split_map.json。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from .anchor import GGUF, ROOT
from .constants import MODELS, MODEL_SAMPLE_BYTES, NUMERIC_SOURCES, POOL_GEOMETRY_SOURCES



def model_path(kind):
    """Absolute path for a configured model: explicit `path` wins, else models/gguf/<file>."""
    return MODELS[kind].get("path") or os.path.join(GGUF, MODELS[kind]["file"])



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
