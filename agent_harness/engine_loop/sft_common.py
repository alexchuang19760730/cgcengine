#!/usr/bin/env python3
"""Shared plumbing for the two SFT projections (sft_pi/, sft_prime/).

WHY A SHARED MODULE AND NOT TWO COPIES
--------------------------------------
Both projections read the same three record files and must render an observation the SAME way,
because the plan's whole point (§6.2) is that "the criteria the model sees at inference time are
byte-identical to the criteria it was trained on". Two renderers would drift on the first edit,
and the drift would be invisible: both datasets would still build, both would still look right,
and only the trained model would be wrong. So there is one renderer.

WHAT "RECONSTRUCTED" MEANS HERE (read this before trusting a sample)
--------------------------------------------------------------------
The traces are records, not transcripts. An `episode` is one arm's observation; a `decision`
cites episodes and states an `action`. Neither contains "the tool call the agent issued".
`sft_pi` therefore RECONSTRUCTS a tool-call trajectory from (decision, cited episodes): the arm
invocation is generated from the episode's own `arm.env`, and it is labelled as reconstructed in
the sample's `_provenance`. That is a weaker claim than "this is what happened" and it is the
accurate one.

`system` IS byte-identical to CONVENTIONS.md -- read from the file at build time rather than
pasted, so there is exactly one copy of the criteria in the repo. The sha256 of the bytes used
is recorded in every output file so a dataset can be traced back to one revision of the charter.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent          # agent_harness/engine_loop
TRACES = HERE / "traces"
CONVENTIONS = HERE.parent / "CONVENTIONS.md"

# The trace files are the authoritative record; validate.py is what makes them trustworthy.
EPISODES = TRACES / "episodes.jsonl"
DECISIONS = TRACES / "decisions.jsonl"
LESSONS = TRACES / "lessons.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing trace file: {path} (run traces/emit_episodes.py first)")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path} has no records -- refusing to build a dataset from nothing")
    return rows


def charter() -> tuple[str, str]:
    """The system prompt, as (text, sha256[:16]). Read, never pasted -- one copy of the criteria."""
    text = CONVENTIONS.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def render_obs(ep: dict) -> str:
    """One episode -> the observation block a reader (or a model) gets.

    Numbers are printed as they are recorded, and the two gates that decide whether a number may
    be QUOTED are printed next to it (`usable_as_evidence`, `caveats`). Omitting those is how a
    dataset teaches a model to quote exactly the rows the harness refuses to quote.
    """
    obs = ep.get("obs") or {}
    lines = [f"episode_id : {ep['episode_id']}", f"profile    : {ep['profile']}",
             f"arm        : {ep['arm']['name']}  ({ep['arm']['declared_purpose']})",
             f"goal       : {ep['goal']}"]
    if ep["arm"].get("usable_for_throughput") is False:
        lines.append("NOTE       : this arm is NOT quotable for throughput (probe or upper-bound)")
    build = ep.get("build")
    lines.append("build      : " + (
        ", ".join(f"{k}={v}" for k, v in sorted(build.items())) if build
        else "null  <-- NO FINGERPRINT: this row is NOT comparable to anything"))
    if obs.get("decode_tps"):
        t = obs["decode_tps"]
        lines.append(f"decode_tps : median={t.get('median')} min={t.get('min')} max={t.get('max')} "
                     f"n_rounds={t.get('n_rounds')}")
    if obs.get("prefill_tps_median") is not None:
        lines.append(f"prefill_tps: median={obs['prefill_tps_median']}")
    if obs.get("answer_md5") is not None:
        lines.append(f"answer_md5 : {obs['answer_md5']}   stable={obs.get('answer_stable')}")
    if obs.get("pool"):
        lines.append("pool       : " + json.dumps(obs["pool"], ensure_ascii=False, sort_keys=True))
    if obs.get("phase_ms"):
        lines.append("phase_ms   : " + json.dumps(obs["phase_ms"], ensure_ascii=False, sort_keys=True))
    if obs.get("asserts"):
        lines.append("asserts    : " + json.dumps(obs["asserts"], ensure_ascii=False, sort_keys=True))
    g = ep.get("gate") or {}
    lines.append(f"gate       : {g.get('name')} verdict={g.get('verdict')} ref={g.get('ref')}")
    lines.append(f"quotable   : usable_as_evidence={ep.get('usable_as_evidence')} caveats={ep.get('caveats')}")
    return "\n".join(lines)


def arm_invocation(ep: dict) -> str:
    """The command that WOULD have produced this episode, from the episode's own arm.env.

    Reconstructed, not recorded. Kept here (not in the builder) so both projections agree on it.
    """
    env = ep["arm"].get("env") or {}
    envs = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
    return (f"python3 scripts/check/decode_sweep.py --profile {ep['profile']} "
            f"--arms {ep['arm']['name']} --rounds 3 --warmup 0"
            + (f"   # env: {envs}" if envs else ""))


def render_state_for_decision(dec: dict, by_ep: dict[str, dict]) -> str:
    """The state a decision was taken on: the question, and every reading it leaned on."""
    out = [f"QUESTION: {dec['question']}", "", "EVIDENCE ON HAND:"]
    for ev in dec["evidence"]:
        eid = ev.get("episode_id")
        tag = f"[{eid}]" if eid else f"[artifact {ev.get('artifact')}]"
        out.append(f"- {tag} {ev['reading']}")
        if eid and eid in by_ep:
            ep = by_ep[eid]
            b = ep.get("build")
            out.append(f"    build={'same-as-above' if b else 'null'}  quotable={ep.get('usable_as_evidence')}")
    if dec.get("ruled_out"):
        out += ["", "ALREADY RULED OUT (do not re-walk these):"]
        for r in dec["ruled_out"]:
            out.append(f"- claim: {r['claim']}")
            out.append(f"  why false: {r['why_false']}")
    return "\n".join(out)


def split(rows: list[dict], val_frac: float, seed: int = 42) -> tuple[list, list]:
    """Deterministic split. Seeded on PURPOSE: a dataset that reshuffles every build cannot be
    diffed against the previous one, and 'the valid set changed' would be indistinguishable from
    'the data changed'."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, round(len(rows) * val_frac)) if len(rows) > 1 else 0
    return rows[n_val:], rows[:n_val]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(out_dir: Path, train: list, valid: list, extra: str = "") -> None:
    print(f"  train={len(train)} valid={len(valid)}  -> {out_dir}")
    if extra:
        print(f"  {extra}")
