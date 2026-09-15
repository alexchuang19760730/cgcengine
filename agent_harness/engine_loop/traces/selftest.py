#!/usr/bin/env python3
"""Prove that traces/validate.py REJECTS. A validator that has only ever returned OK has not been
shown to do anything.

This is CONVENTIONS.md B7 ("a gate must first be shown to reject something") applied to the trace
pipeline itself. Each case below is a violation that the pipeline exists to prevent, injected into a
copy of a real record, and the assertion is that validate.py exits 1.

The cases are not hypothetical: every one of them is a mistake that would silently poison the SFT
set. `build_null_but_usable` in particular is the rule that stops a historical row from being quoted;
`refuted_without_supersede` is the rule that stops a killed hypothesis from being trained on as a
positive example.

    selftest.py            run all cases; exit 1 if any violation is NOT caught
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def first_record(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    raise SystemExit(f"selftest: {path} has no records; run emit_episodes.py first")


def cases() -> list:
    ep = first_record(os.path.join(HERE, "episodes.jsonl"))
    dec = first_record(os.path.join(HERE, "decisions.jsonl"))
    les = first_record(os.path.join(HERE, "lessons.jsonl"))
    numeric = next((e for e in
                    (json.loads(l) for l in open(os.path.join(HERE, "episodes.jsonl")) if l.strip())
                    if isinstance(((e.get("obs") or {}).get("decode_tps") or {}).get("median"),
                                  (int, float))
                    and (e.get("obs") or {}).get("kind") in ("sweep", "interleave")), ep)
    return [
        ("build_null_but_usable",
         "a measurement whose binaries are unknown must not be evidence",
         "episode", {**ep, "build": None, "usable_as_evidence": True}),
        ("missing_required_field",
         "dropping `gate` must not pass silently",
         "episode", {k: v for k, v in ep.items() if k != "gate"}),
        ("bad_episode_id_pattern",
         "a hand-typed id breaks the timestamp convention (ids come from the log filename)",
         "episode", {**ep, "episode_id": "2026-09-15-slotgpu"}),
        ("generation_without_digest",
         "a generation with no answer digest makes answer_stable vacuous",
         "episode", {**numeric, "obs": {**numeric["obs"], "answer_md5": [], "answer_stable": True}}),
        ("refuted_without_supersede",
         "a killed hypothesis must name what replaced it, or it can be read as still-held",
         "decision", {**dec, "judgement": "refuted", "superseded_by": None}),
        ("evidence_unknown_episode",
         "evidence pointing at a non-existent run turns a decision into an assertion",
         "decision", {**dec, "evidence": [{"episode_id": "eng-19700101-000000-ghost-r1",
                                           "reading": "a reading long enough to pass minLength"}]}),
        ("unknown_extra_field",
         "an undocumented field is a convention nobody agreed to; it must not slip in",
         "lesson", {**les, "confidence": "high"}),
        ("bad_lesson_class",
         "class is what makes a lesson injectable at the right moment",
         "lesson", {**les, "class": "vibes"}),
        # Duplicate ids were the one integrity rule that had NO enforcement at all -- only
        # episode_id was checked, because the check was written next to the episode rules and never
        # generalised. Two lessons shipped sharing `eng-gate-0006` and two more sharing
        # `eng-bound-0001`, and validate.py stayed green. An id is a KEY: a duplicate silently
        # breaks every `supersedes`/`superseded_by` reference and makes "the lesson about X"
        # ambiguous, which is exactly the failure mode D4 exists to prevent.
        ("duplicate_lesson_id",
         "two different lessons under one id make every reference to it ambiguous",
         "lesson", [les, {**les, "rule": "A different rule that claims the same id as its twin."}]),
        ("duplicate_decision_id",
         "a supersede chain cannot resolve if the id it names is not unique",
         "decision", [dec, {**dec, "conclusion": "A different conclusion under the same decision id."}]),
    ]


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="trace_selftest_")
    missed = []
    for name, why, kind, rec in cases():
        p = os.path.join(tmp, f"{kind}_{name}.jsonl")
        # `rec` is one record, or a list of them for cases whose violation is a RELATION between
        # records (a duplicate id only exists in the presence of its twin).
        recs = rec if isinstance(rec, list) else [rec]
        with open(p, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        r = subprocess.run([sys.executable, os.path.join(HERE, "validate.py"), p,
                            "--quiet", "--dir", HERE],
                           capture_output=True, text=True)
        ok = r.returncode == 1
        print(f"  {'rejected' if ok else 'ACCEPTED'}  {name:28s}  {why}")
        if not ok:
            missed.append(name)
            print("      validate.py said:", (r.stdout or r.stderr).strip()[:220])
    total = len(cases())
    print(f"\n  {total - len(missed)}/{total} injected violations rejected")
    if missed:
        print(f"  NOT CAUGHT: {missed} — the validator does not enforce what it claims")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
