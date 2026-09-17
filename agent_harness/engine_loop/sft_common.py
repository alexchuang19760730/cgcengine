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

WHICH PART OF THE CHARTER GOES IN `system`, AND WHY IT IS RECORDED
------------------------------------------------------------------
`system` is read from CONVENTIONS.md at build time -- one copy of the criteria, never pasted.
HOW MUCH of it is included depends on `--system`:

    full   the whole file, byte-identical  (what PLAN §6.2's prose asks for)
    head   the first `--head-chars` CHARACTERS  (what PLAN §6.2's table calls "摘要")
    none   omitted

Note the unit: `--head-chars` counts CHARACTERS, because that is what a Python slice does. The
flag used to be named `--head-bytes` while slicing characters -- on a CJK document those differ
by ~1.7x, so the name was wrong by a factor of 1.7 and nothing said so. It survives as a
deprecated alias that warns.

PLAN §6.2 is internally inconsistent about this: its table says `system=CONVENTIONS.md 摘要`,
its prose says the system prompt "就是 `CONVENTIONS.md`" and byte-identical. Both readings are
in the plan, so neither builder picks one for you -- the choice is a flag, it is recorded in
`PROVENANCE.json` beside the data, and `--check` reads that record rather than assuming it.

THE BUILD FINGERPRINT
---------------------
An earlier revision of this docstring claimed the charter's sha256 "is recorded in every output
file". It was not: `grep` for it returned 0 hits. What is true now is that every output DIRECTORY
carries a `PROVENANCE.json` recording the builder, the invocation, and the sha256[:16] of every
authoritative input. Per-directory rather than per-row on purpose: the same charter bytes repeated
in 120 rows buy nothing that one record beside them does not.

That record is what makes `--check` able to tell three states apart, which are otherwise
indistinguishable from the outside:

    content differs because the ARTIFACT is stale (inputs advanced, rebuild required)
    content differs because you asked for a DIFFERENT build (the invocation changed)
    content is identical but the CHARTER changed under it (charter is not an input hash)

Only the first is drift. Lumping all three into "differs" is how a stale dataset gets rebuilt with
the wrong flags and nobody notices.
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

PROVENANCE_NAME = "PROVENANCE.json"
SPLIT_SEED = 42


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
    return text, sha16(text.encode("utf-8"))


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def inputs_fingerprint(paths: tuple[Path, ...] = (EPISODES, DECISIONS, LESSONS)) -> dict[str, str]:
    """sha256[:16] of the inputs this output actually reads, so "which revision is this from" is
    answerable.

    Pass the exact set. Recording a file the builder does NOT read turns somebody else's unrelated
    edit into an `INPUTS ADVANCED` failure -- measured 2026-09-17: `sft_pi` derives from episodes +
    decisions only, and recording `lessons.jsonl` as well made appending a lesson (which `sft_pi`
    cannot see) report that `sft_pi` had gone stale. A check that fires when nothing changed is a
    check that stops being read.
    """
    return {p.name: sha16(p.read_bytes()) for p in paths}


def system_text(full: str, mode: str, head_chars: int) -> str:
    """full | head (first `head_chars` CHARACTERS) | none. See the module docstring on the unit."""
    if mode == "full":
        return full
    if mode == "none":
        return ""
    return full[:head_chars]


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


def split(rows: list[dict], val_frac: float, seed: int = SPLIT_SEED) -> tuple[list, list]:
    """Deterministic split. Seeded on PURPOSE: a dataset that reshuffles every build cannot be
    diffed against the previous one, and 'the valid set changed' would be indistinguishable from
    'the data changed'."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, round(len(rows) * val_frac)) if len(rows) > 1 else 0
    return rows[n_val:], rows[:n_val]


def jsonl_text(rows: list[dict]) -> str:
    """The exact bytes `write_jsonl` would put on disk -- so `--check` can compare without I/O."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonl_text(rows), encoding="utf-8")


def report(out_dir: Path, train: list, valid: list, extra: str = "") -> None:
    print(f"  train={len(train)} valid={len(valid)}  -> {out_dir}")
    if extra:
        print(f"  {extra}")


# --------------------------------------------------------------------------------------
# provenance: recording the invocation, and reconciling it with the current one
# --------------------------------------------------------------------------------------

def provenance_path(out_dir: Path) -> Path:
    return out_dir / PROVENANCE_NAME


def load_provenance(out_dir: Path) -> dict | None:
    p = provenance_path(out_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{p} is not valid JSON ({e}) -- delete it and rebuild")


def write_provenance(out_dir: Path, rec: dict) -> None:
    provenance_path(out_dir).write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reconcile(rec: dict | None, *, builder: str, mode, head_chars, val_frac, check: bool,
              default_head_chars: int) -> tuple[dict, list[str]]:
    """Effective invocation + lines that MUST be printed (never a silent default).

    `mode` / `head_chars` / `val_frac` are None when the caller did not pass the flag -- the
    distinction between "not given" and "given the same value as the default" is what lets a bare
    run reproduce the artifact instead of silently switching its system mode.

    `default_head_chars` is passed in rather than held here so each builder owns its own default
    (the two projections do not share one).
    """
    notes: list[str] = []
    if rec is None:
        eff = {
            "system": mode or "full",
            "head_chars": default_head_chars if head_chars is None else head_chars,
            "val_frac": 0.15 if val_frac is None else val_frac,
        }
        if check:
            notes.append(f"no {PROVENANCE_NAME}: the invocation that produced these files was never "
                         f"recorded, so checking against the CURRENT flags only "
                         f"({_fmt(eff)}) -- a mismatch here does not prove drift")
        return eff, notes

    if rec.get("builder") != builder:
        raise SystemExit(f"{PROVENANCE_NAME} says it was written by {rec.get('builder')!r}, not "
                         f"{builder!r} -- the two projections must not write into each other's "
                         f"directory")

    recorded = rec["invocation"]
    explicit = {"system": mode, "head_chars": head_chars, "val_frac": val_frac}

    if check:
        conflicts = {k: v for k, v in explicit.items() if v is not None and v != recorded[k]}
        if conflicts:
            raise SystemExit(
                "--check verifies the artifact AS BUILT, so it cannot be combined with flags that "
                f"contradict {PROVENANCE_NAME}: {_fmt(conflicts)} vs recorded {_fmt(recorded)}.\n"
                f"    Rebuild with the flags you want, or drop them to check as-built.")
        notes.append(f"using the recorded invocation: {_fmt(recorded)}")
        return dict(recorded), notes

    if all(v is None for v in explicit.values()):
        notes.append(f"no flags given -> using the recorded invocation ({_fmt(recorded)}); "
                     f"this reproduces the artifact on disk")
        return dict(recorded), notes

    eff = {k: (v if v is not None else recorded[k]) for k, v in explicit.items()}
    if eff != dict(recorded):
        notes.append(f"!! this rebuild CHANGES the invocation: {_fmt(recorded)} -> {_fmt(eff)}")
    return eff, notes


def _fmt(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(d.items()))


def check_artifacts(out_dir: Path, train: list, valid: list, rec: dict | None, builder: str,
                    sys_text: str) -> tuple[list[str], list[str]]:
    """(problems, notes): every way the artifacts on disk disagree with a re-derivation.

    The thing that must match is the SYSTEM BYTES THE DATASET ACTUALLY CARRIES, not the sha of the
    whole charter. Those are the same only under `--system full`; under `head` a charter edit past
    the truncation point, and under `none` any charter edit at all, leave the dataset bit-for-bit
    unchanged. Failing the check in those cases would be crying wolf on the most common edit there
    is (touching CONVENTIONS.md), and a check that fires when nothing changed stops being read.
    """
    problems: list[str] = []
    notes: list[str] = []
    if rec is None:
        problems.append(f"{PROVENANCE_NAME}: missing -- the files cannot be traced to an invocation "
                        f"or to a revision of the inputs")
    else:
        if rec.get("builder") != builder:
            problems.append(f"{PROVENANCE_NAME}: builder is {rec.get('builder')!r}, expected {builder!r}")
        cur = inputs_fingerprint()
        for name, was in sorted(rec.get("input_sha256_16", {}).items()):
            now = cur.get(name)
            if now != was:
                problems.append(f"{name}: INPUTS ADVANCED since this artifact was built "
                                f"({was} -> {now}) -- this is staleness, not a different build")
        sys_now = sha16(sys_text.encode("utf-8"))
        if rec.get("system_sha256_16") != sys_now:
            problems.append(f"system prompt bytes differ from the recorded build "
                            f"({rec.get('system_sha256_16')} -> {sys_now}) under "
                            f"--system {rec.get('invocation', {}).get('system')}")
        _, ch_now = charter()
        if rec.get("charter_sha256_16") != ch_now:
            mode = rec.get("invocation", {}).get("system")
            if mode == "full":
                notes.append(f"CONVENTIONS.md changed ({rec.get('charter_sha256_16')} -> {ch_now}); "
                             f"under --system full that is the system prompt, so it is reported above")
            else:
                notes.append(f"CONVENTIONS.md changed ({rec.get('charter_sha256_16')} -> {ch_now}), but "
                             f"this dataset uses --system {mode}, so the bytes it carries are unchanged "
                             f"-- already verified above. The charter still governs `sft_pi` at run "
                             f"time; it just does not appear in these files.")
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        p = out_dir / name
        if not p.exists():
            problems.append(f"{name}: missing")
            continue
        have = p.read_text(encoding="utf-8")
        want = jsonl_text(rows)
        if have != want:
            problems.append(f"{name}: content differs from a re-derivation "
                            f"({have.count(chr(10))} rows on disk -> {want.count(chr(10))} derived)")
    return problems, notes
