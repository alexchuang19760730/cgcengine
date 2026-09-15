#!/usr/bin/env python3
"""Schema + integrity check for the engine-loop trace records.

Two jobs, deliberately in one file:

1. SCHEMA. A small, dependency-free validator for the subset of JSON Schema the three schema files
   actually use. There is no `jsonschema` on this machine's managed interpreter, and adding a
   dependency to the trace pipeline would mean the check silently stops running the first time
   someone emits traces in a clean checkout. A check that can be skipped is not a check.

2. INTEGRITY. The rules that make the records MEAN something across files, which no per-record
   schema can express:
     - build == null            => usable_as_evidence == false
       A measurement whose binaries are unknown cannot be compared to anything, not even to itself
       after a rebuild. This is the rule that stops a historical row from being quoted.
     - a generation happened    => answer_md5 is non-empty
       Otherwise "the answer was stable" is vacuously true.
     - every id resolves        (decision.evidence[].episode_id, superseded_by, supersedes,
                                 lesson.counterexample_observed)
     - refuted                  => superseded_by is set
       Recorded because the plan's risk R3 is training on a hypothesis that was already killed:
       "double-buffering the remap leaf recovers the x1.78" is refuted, and if it entered the SFT
       set as a positive example the model would learn to re-walk that path.

exit 0 = clean, 1 = at least one error, 2 = only warnings were found.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

TRACES_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(TRACES_DIR, "schema")

TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
    "null": type(None),
}


def _type_ok(value, spec) -> bool:
    names = spec if isinstance(spec, list) else [spec]
    for n in names:
        py = TYPES.get(n)
        if py is None:
            continue
        if n == "integer" and isinstance(value, bool):
            continue
        if n == "number" and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False


def check(value, spec: dict, path: str, errs: list, warns: list) -> None:
    """Validate `value` against the schema subset. Collects instead of raising, so one run reports
    every problem rather than the first."""
    if not isinstance(spec, dict):
        return
    t = spec.get("type")
    if t is not None and not _type_ok(value, t):
        errs.append(f"{path}: expected type {t}, got {type(value).__name__}")
        return
    if "const" in spec and value != spec["const"]:
        errs.append(f"{path}: expected const {spec['const']!r}, got {value!r}")
    if "enum" in spec and value not in spec["enum"]:
        errs.append(f"{path}: {value!r} not in enum {spec['enum']}")
    if isinstance(value, str):
        if "pattern" in spec and not re.search(spec["pattern"], value):
            errs.append(f"{path}: {value!r} does not match {spec['pattern']}")
        if "minLength" in spec and len(value) < spec["minLength"]:
            errs.append(f"{path}: string shorter than minLength {spec['minLength']}")
    if isinstance(value, list):
        if "minItems" in spec and len(value) < spec["minItems"]:
            errs.append(f"{path}: array shorter than minItems {spec['minItems']}")
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            for i, v in enumerate(value):
                check(v, item_spec, f"{path}[{i}]", errs, warns)
    if isinstance(value, dict):
        for req in spec.get("required", []):
            if req not in value:
                errs.append(f"{path}: missing required field '{req}'")
        props = spec.get("properties", {})
        if spec.get("additionalProperties") is False:
            extra = [k for k in value if k not in props]
            if extra:
                errs.append(f"{path}: unexpected field(s) {extra} "
                            f"(schema sets additionalProperties: false)")
        elif isinstance(spec.get("additionalProperties"), dict):
            for k, v in value.items():
                if k not in props:
                    check(v, spec["additionalProperties"], f"{path}.{k}", errs, warns)
        for k, sub in props.items():
            if k in value:
                check(value[k], sub, f"{path}.{k}", errs, warns)


def load_schemas() -> dict:
    out = {}
    for name in ("episode", "decision", "lesson"):
        p = os.path.join(SCHEMA_DIR, f"{name}.schema.json")
        if not os.path.exists(p):
            print(f"  [warn] missing schema {p}", file=sys.stderr)
            continue
        out[name] = json.load(open(p))
    return out


def rule_build_null_implies_unusable(rec: dict, errs: list) -> None:
    if rec.get("build") is None and rec.get("usable_as_evidence") is True:
        errs.append(f"{rec.get('episode_id')}: build == null but usable_as_evidence == true "
                    f"(a number whose binaries are unknown is not evidence)")


def rule_generation_needs_digest(rec: dict, errs: list) -> None:
    """Only the kinds that actually generate text have to show a digest. A llama-bench cell reports
    tokens/s and never sees the output text, so requiring a digest there would force the emitter to
    invent one — the opposite of what this pipeline is for."""
    obs = rec.get("obs") or {}
    if obs.get("kind") not in ("sweep", "interleave"):
        return
    med = (obs.get("decode_tps") or {}).get("median")
    if isinstance(med, (int, float)) and not obs.get("answer_md5"):
        errs.append(f"{rec.get('episode_id')}: a generation happened (decode_tps.median={med}) but "
                    f"answer_md5 is empty, so answer_stable is vacuous")


def rule_refuted_needs_supersede(rec: dict, errs: list) -> None:
    if rec.get("judgement") == "refuted" and not rec.get("superseded_by"):
        errs.append(f"{rec.get('decision_id')}: judgement == refuted requires superseded_by "
                    f"(otherwise a killed hypothesis can be read as still-held)")


def validate_file(path: str, quiet: bool = False, episodes_index: dict | None = None):
    schemas = load_schemas()
    if not os.path.exists(path):
        print(f"  [error] {path} does not exist")
        return 1
    errs: list = []
    warns: list = []
    recs = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append(f"line {lineno}: not JSON ({e})")
                continue
            recs.append(rec)
            kind = rec.get("type")
            spec = schemas.get(kind)
            if spec is None:
                errs.append(f"line {lineno}: unknown record type {kind!r}")
                continue
            check(rec, spec, f"line {lineno} ({kind})", errs, warns)

    # per-type cross-record rules
    eps = [r for r in recs if r.get("type") == "episode"]
    decs = [r for r in recs if r.get("type") == "decision"]
    less = [r for r in recs if r.get("type") == "lesson"]

    for r in eps:
        rule_build_null_implies_unusable(r, errs)
        rule_generation_needs_digest(r, errs)
    for r in decs:
        rule_refuted_needs_supersede(r, errs)

    ids = [r.get("episode_id") for r in eps]
    for dup in {i for i in ids if ids.count(i) > 1}:
        errs.append(f"duplicate episode_id {dup}")

    # The same rule for the other two record types. It was written for episodes only -- the check
    # sat next to the episode rules and was never generalised -- so two lessons shared
    # `eng-gate-0006` and two more shared `eng-bound-0001` while this validator stayed green.
    # A record id is a KEY: duplicates silently break `supersedes`/`superseded_by` resolution and
    # make any "the lesson about X" reference ambiguous.
    for rtype, key in (("decision", "decision_id"), ("lesson", "lesson_id")):
        rids = [r.get(key) for r in recs if r.get("type") == rtype]
        for dup in {i for i in rids if rids.count(i) > 1}:
            errs.append(f"duplicate {key} {dup}")

    # id resolution against the episodes index (when we have one)
    if episodes_index:
        known = set(episodes_index)
        for r in decs:
            for ev in r.get("evidence") or []:
                eid = ev.get("episode_id")
                if eid and eid not in known:
                    errs.append(f"{r.get('decision_id')}: evidence references unknown "
                                f"episode_id {eid}")
        for r in less:
            ce = r.get("counterexample_observed")
            if ce and ce.startswith("eng-") and ce not in known:
                warns.append(f"{r.get('lesson_id')}: counterexample_observed {ce} resolves to no "
                             f"episode in the index")

    dec_ids = {r.get("decision_id") for r in decs}
    les_ids = {r.get("lesson_id") for r in less}
    for r in decs:
        for key in ("supersedes",):
            for ref in r.get(key) or []:
                if ref not in dec_ids:
                    errs.append(f"{r.get('decision_id')}: {key} -> unknown decision {ref}")
        sb = r.get("superseded_by")
        if sb and dec_ids and sb not in dec_ids:
            warns.append(f"{r.get('decision_id')}: superseded_by {sb} not in this file "
                         f"(may live in another traces file)")
    for r in less:
        sb = r.get("superseded_by")
        if sb and les_ids and sb not in les_ids:
            warns.append(f"{r.get('lesson_id')}: superseded_by {sb} not in this file")
        for p in r.get("applies_to") or []:
            if not p.startswith("$") and not os.path.exists(os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(TRACES_DIR))), p)):
                warns.append(f"{r.get('lesson_id')}: applies_to path does not exist: {p}")

    if not quiet:
        kinds = {}
        for r in recs:
            kinds[r.get("type")] = kinds.get(r.get("type"), 0) + 1
        print(f"\n  {os.path.basename(path)}: {len(recs)} records {kinds}")
    for w in warns:
        print(f"    WARN  {w}")
    for e in errs:
        print(f"    ERROR {e}")
    if not quiet:
        print(f"  -> {'OK' if not errs else f'{len(errs)} error(s)'}"
              f"{f', {len(warns)} warning(s)' if warns else ''}")
    if errs:
        return 1
    return 2 if warns else 0


def validate_dir(traces_dir: str, quiet: bool = False) -> int:
    ep_path = os.path.join(traces_dir, "episodes.jsonl")
    index = {}
    if os.path.exists(ep_path):
        with open(ep_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        index[json.loads(line)["episode_id"]] = True
                    except (json.JSONDecodeError, KeyError):
                        pass
    worst = 0
    found = 0
    for name in ("episodes.jsonl", "decisions.jsonl", "lessons.jsonl"):
        p = os.path.join(traces_dir, name)
        if not os.path.exists(p):
            if not quiet:
                print(f"  (no {name} yet)")
            continue
        found += 1
        worst = max(worst, validate_file(p, quiet=quiet, episodes_index=index))
    if not found:
        print(f"  [error] no trace files found in {traces_dir}")
        return 1
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="JSONL files; defaults to the whole traces dir")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.files:
        idx = {}
        ep = os.path.join(args.dir or TRACES_DIR, "episodes.jsonl")
        if os.path.exists(ep):
            with open(ep, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            idx[json.loads(line)["episode_id"]] = True
                        except Exception:
                            pass
        return max(validate_file(f, quiet=args.quiet, episodes_index=idx) for f in args.files)
    return validate_dir(args.dir or TRACES_DIR, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
