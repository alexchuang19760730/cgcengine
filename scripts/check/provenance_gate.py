#!/usr/bin/env python3
"""Commit gate: a measurement product must be able to answer "was the box busy?".

WHY THIS EXISTS
---------------
`server_window.py audit-products` measured the state of this repo's evidence on 2026-09-22: **893
measured objects across 514 products, and 3 of them carried a window block.** The other 890 cannot
answer the only question that decides whether a reading is a finding or a coincidence. In
`server_window`'s own words: a launch into a busy box measures the neighbour, the spread on identical
configs here was 17% and 2.5x on a bad day, and *the failure is invisible in the number itself*. So
it has to be caught in the file, at the moment the file enters the repo.

`target25_report.py` stamps engine digest + window into its own artifact. This gate is the other half
of that contract: it refuses to let the NEXT artifact in without them.

WHAT IT ENFORCES
----------------
For every tracked `*.json` that contains a measured object (`server_window.MEASURED_KEYS`):

  1. every measured object carries `window` with a `class`   -- the busy-box question is answerable
  2. the product carries an engine digest somewhere          -- the reading has a named binary

And for the machine-written reports in `docs/` whose producer is registered below:

  3. both halves of the pair enter the repository in the same commit -- a report staged without its
     sidecar is numbers whose evidence is elsewhere, and a sidecar staged without its report is
     evidence attached to nothing. The registry is checked for completeness against the tree, so a
     new producer cannot quietly write artifacts that nobody judges (`producers_writing_html_docs`).

Rule: **no NEW offender, and the offender count may not rise.** The pre-existing offenders are in
the baseline and are TOLERATED, because a gate that blocks hundreds of files on day one is a gate
somebody `--no-verify`s away, and then nothing is gated. Equal is fine, fewer is reported as
progress, and `--update` re-baselines deliberately -- a separate command, so it cannot happen by
accident inside a commit.

Keys are REPO-RELATIVE. The baseline ships with the repo and this repo has several worktrees
(`flashkv0516`, `flashkv-devserver`, `flashkv-autobuild`, ...); absolute keys would make every
offender look new from the second worktree onward.

WHAT IT REUSES, AND WHAT THAT COSTS
-----------------------------------
`server_window.audit_products` is imported: it is THE definition of "can this product answer the
busy-box question", and a second definition of one question is how this line has already shipped two
answers to "how much memory is free". The digest key names are duplicated from
`attribution_timeline.classify_product` (that function lives on another line and its list is not
importable) -- a NAMED coupling, recorded here so a change there is a known edit here.

HONEST BOUNDARIES
-----------------
- **JSON only.** A report's provenance belongs in its sidecar JSON; requiring the pair is the
  author's job, not a text match on HTML. An HTML artifact with no sidecar is invisible here.
- **Key matching**, inherited from `audit_products`: a product recording the same evidence under a
  different name counts as missing. A false positive costs a one-line fix; a false negative costs
  weeks, so the gate errs toward naming the file.
- **`git ls-files`, not a glob.** It judges what git is being asked to keep, so untracked scratch
  under `Backup/` cannot move the count.
- **A rename enters a new path**, so moving a legacy offender is judged like adding one. That is rule
  1 read literally rather than a special case: carry the evidence in the same commit, or leave the
  file out.
- It says nothing about whether a number is CORRECT -- only whether the file records the conditions
  it was taken under. Those are different claims, and only the second is cheap enough to check on
  every commit.

TWO MODES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
--------------------------------------------------
`check --staged` is the COMMIT gate (`check_build_tracked.sh` 檢查 12). Every `*.json` this commit
ADDS or MODIFIES must carry both halves, or the commit is refused. It reads each blob **out of the
index, not the worktree** -- otherwise a file staged bare and then fixed in the worktree would pass
while the commit kept the bare one, which is the discipline 檢查 11 already applies to its baseline.
`check` is the repo-wide audit versus the baseline: the legacy offenders are tolerated there so the
gate could ship at all. The commit gate needs no baseline and no tolerance, which is why it also
works while the audit is red.

Usage
-----
    python3 scripts/check/provenance_gate.py check --staged   # the commit gate (0 pass / 1 block / 2 cannot judge)
    python3 scripts/check/provenance_gate.py check            # repo-wide audit vs the baseline
    python3 scripts/check/provenance_gate.py check --json
    python3 scripts/check/provenance_gate.py update           # re-baseline from a real scan
    python3 scripts/check/provenance_gate.py selftest

WHAT THE PAIRING RULE CANNOT SEE (named, not hidden)
---------------------------------------------------
- It judges `docs/` HTML artifacts **whose producer is registered**. A report written to another
  directory (`Backup/`, `/tmp`) or by a producer nobody registered is invisible -- the second half of
  that is why an unregistered producer in a commit is itself refused.
- Both members must be in the same commit, so a run whose HTML is byte-identical and whose sidecar
  changed (or the reverse) is also refused. That is the strict reading of "a pair moves together",
  and it is deliberate: a split pair is one nobody can check (`git add docs/X.*` stages both).
- A sidecar deleted in a commit that does not touch its report is not caught (`--diff-filter=ACMR`
  excludes deletions from the staged set).
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = Path(__file__).resolve().parent
BASELINE = CHECK / "provenance_gate_baseline.json"
CORPUS = "git ls-files '*.json'"

# The names `attribution_timeline.classify_product` accepts as a product's identity. Kept in sync by
# hand; see the docstring.
DIGEST_KEYS = ("engine_digest", "digest", "engine", "provenance", "engine_provenance", "binary")

# ---------------------------------------------------------------- the report/sidecar registry
# `docs/` holds 77 `*.html` files and almost all of them are hand-written prose. A rule that demanded
# a sidecar from a whitepaper is a rule somebody bypasses, so an artifact is judged here only because
# a TOOL is known to write it. A registry's dangerous direction is going stale -- a new producer
# nobody registered writes artifacts nobody checks -- so its completeness is checked against the tree
# instead of trusted (`producers_writing_html_docs`), and an unregistered producer that is part of the
# commit is refused.
#
# `evidence` is not decoration: it is where the claim "this tool writes a sidecar" can be checked.
REPORT_PRODUCERS = (
    {"tool": "scripts/check/target25_report.py", "glob": "docs/TARGET25_REPORT_*.html",
     "sidecar": "<stem>.json",
     "evidence": "writes the sidecar in the same run (target25_report.py:636)"},
    # The glob is the producer's OWN naming rule (`report_path()`: `DECODE_WINDOW_%Y-%m-%d.html`), not
    # `DECODE_WINDOW_*.html`: that wider one judged `DECODE_WINDOW_HARNESS_20260919_1745.html`, which is
    # hand-curated prose ABOUT the harness (its own text says so) and has no sidecar by construction.
    # Over-matching a width is not conservative here -- it demands evidence from an artifact that was
    # never measured, which is the same defect class as the 77 hand-written whitepapers in docs/.
    # Known blind spot: a report written through `--out` under another name escapes this rule.
    {"tool": "scripts/check/decode_window_harness.py",
     "glob": "docs/DECODE_WINDOW_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].html",
     "sidecar": "<stem>.json",
     "evidence": "sidecar written in the same call as the HTML (decode_window_harness.py write_report / "
                 "autoreport); the one dated report from before that was backfilled from its own text"},
)
# Producers whose `docs/*.html` mention is not a report write. The scan's two line exclusions
# (comment lines, stderr/print lines) already remove the 12 decoys in this tree; what is left is THIS
# file, whose selftest fixtures write `docs/TARGET25_REPORT_t.html` into a throwaway repo. An
# exemption is a decision, so it is recorded here with its reason, and the selftest fails if an entry
# goes stale (the file no longer mentions one) -- a silent allow-list is how a completeness check
# stops being one.
REPORT_EXEMPT: dict[str, str] = {
    "scripts/check/provenance_gate.py":
        "the docs/*.html paths in it are selftest fixtures written to a temp repo, not a producer",
}
DOCS_HTML_PATH = re.compile(r"(?:os\.path\.join\s*\([^)]*[\"']docs[\"']|ROOT\s*/\s*[\"']docs[\"']"
                            r"|[\"']docs/)")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CHECK / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_digest(obj) -> str | None:
    """The first digest-ish key anywhere in the product, or None. Recursive on purpose: the two lines
    put their identity at different depths (`engine_digest` at the top, `binary` per row), and a
    top-level-only check would fail products that do carry one."""
    if isinstance(obj, dict):
        for k in DIGEST_KEYS:
            if obj.get(k):             # a non-empty value; `{}` or "" is not an identity
                return k
        return next((d for v in obj.values() if (d := find_digest(v))), None)
    if isinstance(obj, list):
        return next((d for v in obj if (d := find_digest(v))), None)
    return None


def key_of(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def judge(doc, sw, path: Path) -> dict:
    """One product's reasons, with the window half taken from the repo's own audit. Shared by both
    modes so the commit gate and the repo-wide audit cannot drift into two answers."""
    rep = sw.audit_products([str(path)])
    measured = rep["measured_objects"]
    reasons = []
    if measured and rep["without"] > 0:
        reasons.append(f"{rep['without']} measured object(s) with no window block")
    # a window without an identity is its own defect: the reading cannot be attached to a
    # binary, which is the 0/186 hole the other line hit.
    if measured and find_digest(doc) is None:
        reasons.append("no engine digest anywhere in the product")
    return {"measured": measured, "without": rep["without"],
            "examples": [e["where"] for e in rep["examples_without"]], "reasons": reasons}


def inspect(path: Path, sw) -> dict:
    """`judge` on a file on disk, keeping "not JSON at all" as its own verdict rather than a reason:
    the two are handled differently (see `scan`)."""
    try:
        doc = json.loads(path.read_text(errors="replace"))
    except Exception as e:
        return {"unreadable": f"{type(e).__name__}: {str(e)[:70]}"}
    return judge(doc, sw, path)


def scan(paths: list[Path], sw, repo: Path) -> tuple[dict, dict, dict]:
    """(totals, offenders, unparseable), all keyed by repo-relative path.

    An unparseable tracked `.json` is reported by NAME and counted separately, but it is **not an
    offender**. Measured, not assumed: 6 of the 7 in this repo are vendored JSONC configs
    (`pyrightconfig.json`, `tools/ui/tsconfig.json` in three llama.cpp trees) that were never
    products, plus one 12-byte `No results` file. A gate whose baseline is padded with those is a
    gate people learn to ignore. The residual hole is named in the docstring: a truncated
    measurement dump passes here -- but it appears as a NEW unparseable file, so it is not silent."""
    totals = {"files": 0, "measured_objects": 0, "with_window": 0, "without": 0,
              "unreadable": 0, "not_a_product": 0}
    offenders: dict[str, dict] = {}
    unparseable: dict[str, str] = {}
    for p in paths:
        v = inspect(p, sw)
        k = key_of(p, repo)
        if "unreadable" in v:
            totals["unreadable"] += 1
            unparseable[k] = v["unreadable"]
            continue
        if v["measured"] == 0:
            totals["not_a_product"] += 1
            continue
        totals["files"] += 1
        totals["measured_objects"] += v["measured"]
        totals["without"] += v["without"]
        totals["with_window"] += v["measured"] - v["without"]
        if v["reasons"]:
            offenders[k] = {"reasons": v["reasons"], "examples": v["examples"][:6],
                            "measured": v["measured"]}
    return totals, offenders, unparseable


def tracked_json(repo: Path) -> list[Path]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "*.json"],
                         capture_output=True, text=True).stdout
    return sorted(repo / p for p in out.split("\0") if p.strip())


def staged_json(repo: Path) -> tuple[list[str], str | None]:
    """What this commit adds or modifies that this gate judges:

      * every `*.json`                             -- the product rules (window + digest)
      * every `*.py`                               -- the producer registry's completeness
      * the `*.html` of a registered producer      -- the other half of a report pair

    Returns `(names, error)`: an empty list and a non-empty error are different answers, and the
    caller must not read the first as a pass (`scan` makes the same distinction for an absent miss
    axis). The diff asks for everything and filters here, because `-- '*.json'` alone would hide both
    the report half of every pair and the tool that writes it.
    """
    r = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--name-only",
                        "--diff-filter=ACMR", "-z"], capture_output=True, text=True)
    if r.returncode != 0:
        return [], (r.stderr.strip() or f"git diff --cached exited {r.returncode}")
    names = [p for p in r.stdout.split("\0") if p.strip()]
    return sorted(p for p in names
                  if p.endswith((".json", ".py"))
                  or any(fnmatch.fnmatch(p, prod["glob"]) for prod in REPORT_PRODUCERS)), None


def staged_blob(repo: Path, rel: str) -> str | None:
    """The INDEX copy of `rel` -- what the commit will actually contain."""
    r = subprocess.run(["git", "-C", str(repo), "show", f":{rel}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def docs_html_writer_lines(text: str) -> list[int]:
    """Line numbers that construct a path into `docs/` ending in `.html` (= a report producer).

    Deliberately a text scan and not an import or an AST walk: the thing to catch is a producer whose
    shape nobody anticipated. The two exclusions are measured, not guessed -- the naive `docs` +
    `.html` match hit 14 lines in this tree and 12 of them were docstring/help-text mentions (the
    stderr hint at `knifeedge/feasibility.py:209` is an f-string, not a path).
    """
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if ".html" not in line or line.lstrip().startswith("#"):
            continue
        if "stderr" in line or "print(" in line:
            continue
        if DOCS_HTML_PATH.search(line):
            out.append(i)
    return out


def producers_writing_html_docs(repo: Path) -> dict[str, list[int]]:
    """`{repo-relative file: [lines]}` for every `scripts/**/*.py` that writes a report into `docs/`."""
    out: dict[str, list[int]] = {}
    for p in sorted((repo / "scripts").rglob("*.py")):
        hits = docs_html_writer_lines(p.read_text(errors="replace"))
        if hits:
            out[key_of(p, repo)] = hits
    return out


def registered_tools() -> set[str]:
    return {p["tool"] for p in REPORT_PRODUCERS} | set(REPORT_EXEMPT)


def pair_member(rel: str, repo: Path) -> tuple[str, dict] | None:
    """`(the other half of the pair, its producer)` when `rel` is one member of a registered pair.

    Both directions, because a pair split across commits is a pair nobody can check: the report alone
    asserts numbers whose evidence is elsewhere, and the sidecar alone is evidence attached to
    nothing. A `.json` counts as a member only when its `.html` is really there -- otherwise every
    `*.json` in this repo would look like an orphaned sidecar.
    """
    stem, ext = Path(rel).with_suffix(""), Path(rel).suffix.lower()
    if ext == ".html":
        for prod in REPORT_PRODUCERS:
            if fnmatch.fnmatch(rel, prod["glob"]):
                return str(stem.with_suffix(".json")), prod
        return None
    if ext == ".json":
        other = str(stem.with_suffix(".html"))
        for prod in REPORT_PRODUCERS:
            if fnmatch.fnmatch(other, prod["glob"]) and (repo / other).exists():
                return other, prod
    return None


def in_head(repo: Path, rel: str) -> bool:
    """Is `rel` already committed at HEAD?

    The pair rule needs this, and it is a different question from "does it exist on disk": an
    untracked file on disk is not in the repository, so a pair whose other half is only untracked is
    still a split. It is what separates "this commit tears a pair apart" from "this commit supplies
    the half the repository was missing" -- and the second is the repair, not the defect.
    """
    return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"HEAD:{rel}"],
                          capture_output=True).returncode == 0


def pair_audit(repo: Path, tracked: set[str] | None = None) -> dict:
    """Per registered producer: its artifacts on disk and the ones with no sidecar beside them.

    `unpaired` counts artifacts, NOT evidence: the two `DECODE_WINDOW_*.html` in this repo were
    written before the harness wrote sidecars, and their sidecars cannot be reconstructed honestly
    (the run is gone). They are named by `new_unpaired` as legacy and tolerated exactly like the
    legacy window/digest offenders -- the rule is "no NEW unpaired artifact", because a rule that
    demanded a fabricated sidecar would be teaching the wrong lesson.
    """
    out = {}
    for prod in REPORT_PRODUCERS:
        arts = sorted(key_of(p, repo) for p in repo.glob(prod["glob"]))
        unpaired = [a for a in arts if not (repo / Path(a).with_suffix(".json")).exists()]
        out[prod["tool"]] = {
            "glob": prod["glob"], "artifacts": len(arts),
            "tracked_artifacts": [a for a in arts if tracked is None or a in tracked],
            "unpaired": unpaired,
            "evidence": prod["evidence"],
        }
    return out


def unpaired_reports(audit: dict) -> list[str]:
    return sorted(a for v in audit.values() for a in v["unpaired"])


def new_unpaired(base_legacy: list[str], audit: dict) -> list[str]:
    """Unpaired artifacts the baseline did not know about. THIS is what fails a run."""
    return [a for a in unpaired_reports(audit) if a not in set(base_legacy)]


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def compare(base_offenders: dict, offenders: dict) -> tuple[dict, list[str], list[str]]:
    """(new, fixed, legacy). `new` decides pass/fail; `fixed` is progress; `legacy` is tolerated."""
    new = {k: v for k, v in offenders.items() if k not in base_offenders}
    fixed = sorted(k for k in base_offenders if k not in offenders)
    legacy = sorted(k for k in offenders if k in base_offenders)
    return new, fixed, legacy


def new_unparseable(base_unparseable: dict, unparseable: dict) -> list[str]:
    """An unparseable file that the baseline did not know about. A WARNING, not a failure -- see the
    docstring: the 7 in this repo are vendored JSONC configs, and failing on those is how a gate
    loses its audience. It is still named, so a truncated dump cannot enter silently."""
    return sorted(k for k in unparseable if k not in base_unparseable)


def cmd_check(args) -> int:
    repo = Path(args.repo).resolve()
    baseline_path = Path(args.baseline)
    base = load_baseline(baseline_path)
    if not base.get("offenders"):
        # Fail closed on the BASELINE, not on the corpus: a missing baseline would otherwise report
        # every tolerated file as a new offender and produce a wall of text nobody reads.
        print(f"CANNOT JUDGE: no usable baseline at {baseline_path}")
        print(f"  run: python3 {Path(__file__).name} update --baseline {baseline_path}")
        return 2
    sw = _load("sw", "server_window.py")
    paths = tracked_json(repo)
    totals, offenders, unparseable = scan(paths, sw, repo)
    new, fixed, legacy = compare(base["offenders"], offenders)
    new_unp = new_unparseable(base.get("unparseable", {}), unparseable)
    pairs = pair_audit(repo, set(key_of(p, repo) for p in paths))
    legacy_unpaired = base.get("unpaired_reports", [])
    fresh_unpaired = new_unpaired(legacy_unpaired, pairs)
    unregistered = sorted(set(producers_writing_html_docs(repo)) - registered_tools())
    rep = {"repo": str(repo), "corpus": f"{len(paths)} tracked *.json", "totals": totals,
           "offender_count": len(offenders), "baseline_count": len(base["offenders"]),
           "new": new, "fixed": fixed, "legacy_count": len(legacy),
           "unparseable": {"count": len(unparseable), "new": new_unp},
           "report_pairs": pairs, "unpaired_reports": {"new": fresh_unpaired,
                                                          "legacy": legacy_unpaired},
           "unregistered_producers": unregistered, "baseline": str(baseline_path)}
    if args.json:
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        return 1 if (new or unregistered or fresh_unpaired) else 0

    print(f"provenance gate: {len(paths)} tracked *.json")
    print(f"  measurement products : {totals['files']}  "
          f"({totals['not_a_product']} file(s) carry no measured object)")
    print(f"  measured objects     : {totals['measured_objects']}")
    print(f"  with a window block  : {totals['with_window']}")
    print(f"  offenders            : {len(offenders)}  (baseline {rep['baseline_count']})")
    print(f"  corpus note          : {CORPUS} -- tracked only. `server_window.py audit-products`")
    print(f"                         scans untracked `Backup/` scratch too, so its object count is")
    print(f"                         larger; the two numbers answer one question over two corpora.")
    if unparseable:
        print(f"  not parseable as JSON: {len(unparseable)} (not offenders -- vendored JSONC"
              f" configs and the like; counted so the number cannot drift)")
        for k in new_unp:
            print(f"      WARNING new unparseable file: {k}")
    print(f"  registered report pairs       : {len(pairs)} producer(s), "
          f"{sum(v['artifacts'] for v in pairs.values())} artifact(s), "
          f"{len(fresh_unpaired)} NEW unpaired ({len(legacy_unpaired)} legacy)")
    for a in fresh_unpaired:
        print(f"      NEW unpaired: {a} -- stage its sidecar, or take it out of the tree")
    for tool, v in sorted(pairs.items()):
        if any(a in legacy_unpaired for a in v["unpaired"]):
            print(f"      {tool}: {len(v['unpaired'])} of {v['artifacts']} unpaired, all legacy "
                  f"(predate the sidecar) -- e.g. {v['unpaired'][0]}")
            print(f"        ({v['evidence']})")
    if unregistered:
        print(f"\nFAIL: {len(unregistered)} producer(s) write reports into docs/ without being "
              f"registered, so their artifacts are judged by nobody:")
        for f in unregistered:
            print(f"  {f} -- add it to REPORT_PRODUCERS (with its sidecar rule) or to REPORT_EXEMPT "
                  f"(with a reason)")
    if fixed:
        print(f"\nFIXED since the baseline ({len(fixed)}) -- `update` locks the lower count:")
        for p in fixed[:10]:
            print(f"  - {p}")
        if len(fixed) > 10:
            print(f"  ... and {len(fixed) - 10} more")
    if legacy:
        print(f"\nlegacy offenders tolerated by the baseline: {len(legacy)} "
              f"({baseline_path.name})")
    if new:
        print(f"\nFAIL: {len(new)} NEW offender(s). A reading with no window or no digest cannot be"
              f" attributed -- add the evidence, or keep the file out of the commit.")
        for p, v in sorted(new.items())[:15]:
            print(f"  {p}")
            for r in v.get("reasons", []):
                print(f"      {r}")
            for e in v.get("examples", [])[:3]:
                print(f"      at {e}")
        if len(new) > 15:
            print(f"  ... and {len(new) - 15} more")
    if fresh_unpaired:
        print(f"\nFAIL: {len(fresh_unpaired)} unpaired report artifact(s) that the baseline does not "
              f"know about. A report without its sidecar is numbers whose evidence is elsewhere:")
        for a in fresh_unpaired:
            print(f"  {a} -- expected next to it: {Path(a).with_suffix('.json')}")
    if new or unregistered or fresh_unpaired:
        return 1
    print("\nPASS: no new offender; the offender count did not rise, and every report producer is "
          "registered.")
    return 0


def cmd_check_staged(args) -> int:
    repo = Path(args.repo).resolve()
    names, err = staged_json(repo)
    if err:
        print(f"CANNOT JUDGE: git diff --cached failed: {err}")
        return 2
    if not names:
        print("provenance gate (staged): this commit adds or modifies nothing this gate judges "
              "(no *.json, no registered report artifact).")
        return 0
    sw = _load("sw", "server_window.py")
    blobs: dict[str, str] = {}
    for rel in names:
        text = staged_blob(repo, rel)
        if text is None:
            print(f"CANNOT JUDGE: could not read the staged blob for {rel}")
            return 2
        blobs[rel] = text

    # (a) a registered pair moves together or not at all -- with one exception that is the repair
    # rather than a hole in it: when the other half is ALREADY COMMITTED, staging this half completes
    # the pair. Demanding both halves inside one commit would have made `backfill` (giving a report
    # published before sidecars existed the sidecar it should have had) impossible to land, which is
    # the exact shape the rule was written to force. `split` therefore means "the repository ends up
    # with one half and not the other", not "one half is in this diff".
    split: dict[str, dict] = {}
    completed: dict[str, str] = {}
    for rel in names:
        m = pair_member(rel, repo)
        if m and m[0] not in blobs:
            other, prod = m
            if in_head(repo, other):
                completed[rel] = other
                continue
            split[rel] = {"producer": prod["tool"], "missing": other,
                          "sidecar_rule": prod["sidecar"], "evidence": prod["evidence"]}

    # (b) a NEW report producer must be registered, or its artifacts are judged by nobody
    new_producers = {rel: docs_html_writer_lines(blobs[rel]) for rel in names
                     if rel.endswith(".py") and rel not in registered_tools()}
    new_producers = {k: v for k, v in new_producers.items() if v}

    tmp = Path(tempfile.mkdtemp(prefix="pgate_staged_"))
    offenders: dict[str, dict] = {}
    unreadable: dict[str, str] = {}
    products: list[str] = []
    for rel in names:
        if not rel.endswith(".json"):       # the product rules are about JSON; .py is rule (b) above
            continue
        p = tmp / rel          # same relative layout, so `examples` come back repo-relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(blobs[rel], errors="replace")
        v = inspect(p, sw)
        if "unreadable" in v:
            unreadable[rel] = v["unreadable"]
            continue
        if v["measured"] == 0:
            continue
        products.append(rel)
        if v["reasons"]:
            offenders[rel] = {"reasons": v["reasons"], "measured": v["measured"],
                              "examples": [e.replace(str(tmp) + "/", "")
                                           for e in v["examples"][:6]]}
    if args.json:
        print(json.dumps({"repo": str(repo), "mode": "staged", "staged_json": names,
                          "measurement_products": products, "offenders": offenders,
                          "split_pairs": split, "completed_pairs": completed,
                          "unregistered_producers": new_producers,
                          "unparseable": unreadable}, indent=1, ensure_ascii=False))
        return 1 if (offenders or split or new_producers) else 0

    n_json = sum(1 for p in names if p.endswith(".json"))
    n_py = sum(1 for p in names if p.endswith(".py"))
    print(f"provenance gate (staged): {len(names)} judged path(s) added or modified by this commit"
          f" ({n_json} *.json, {n_py} *.py, {len(names) - n_json - n_py} report artifact(s))")
    print(f"  measurement products among them : {len(products)}")
    if unreadable:
        print(f"  not parseable as JSON           : {len(unreadable)} (named, not offenders --"
              f" vendored JSONC configs and the like)")
        for k, why in sorted(unreadable.items()):
            print(f"      {k}: {why}")
    for rel, other in sorted(completed.items()):
        print(f"  completes the pair: {rel} -- its {other} is already committed, so this commit "
              f"supplies the half the repository was missing")
    if split:
        print(f"\nFAIL: {len(split)} registered report pair(s) split by this commit. Both halves "
              f"carry the reading -- the report quotes the numbers, the sidecar carries the evidence "
              f"-- so they enter the repository together:")
        for p, v in sorted(split.items()):
            print(f"  {p}")
            print(f"      its {v['sidecar_rule']} is not in this commit: {v['missing']}")
            print(f"      producer {v['producer']} -- {v['evidence']}")
    if new_producers:
        print(f"\nFAIL: {len(new_producers)} new producer(s) write a report into docs/ but are not "
              f"registered, so nothing judges the artifacts they will write:")
        for p, ls in sorted(new_producers.items()):
            print(f"  {p} (lines {', '.join(str(x) for x in ls[:4])})")
            print(f"      add it to REPORT_PRODUCERS with its sidecar rule, or to REPORT_EXEMPT with "
                  f"a reason")
    if not (offenders or split or new_producers):
        print("\nPASS: every measurement product in this commit carries a window block and an"
              " engine digest, and every registered report pair moved together.")
        return 0
    if offenders:
        print(f"\nFAIL: {len(offenders)} measurement product(s) in this commit cannot be attributed"
              f" -- add the evidence, or keep the file out of the commit.")
        for p, v in sorted(offenders.items()):
            print(f"  {p}  ({v['measured']} measured object(s))")
            for why in v["reasons"]:
                print(f"      {why}")
            for e in v["examples"][:3]:
                print(f"      at {e}")
    return 1


def cmd_update(args) -> int:
    repo = Path(args.repo).resolve()
    sw = _load("sw", "server_window.py")
    paths = tracked_json(repo)
    totals, offenders, unparseable = scan(paths, sw, repo)
    doc = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Offenders and unpaired reports that existed BEFORE this gate was wired in, or before"
                 " the producing tool wrote sidecars. Tolerated so the gate can ship; the counts may"
                 " only go down, and `update` is a deliberate command so a re-baseline cannot happen"
                 " by accident inside a commit. `unpaired_reports` are artifacts whose sidecar cannot"
                 " be reconstructed honestly -- the run that wrote them is gone, so they are named,"
                 " not backfilled."),
        "corpus": CORPUS,
        "counts": totals,
        "offenders": dict(sorted(offenders.items())),
        "unparseable": dict(sorted(unparseable.items())),
        "unpaired_reports": unpaired_reports(pair_audit(repo)),
    }
    Path(args.baseline).write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"baseline -> {args.baseline}")
    print(f"  {len(offenders)} offender(s) of {totals['files']} measurement product(s); "
          f"{totals['without']} measured object(s) cannot answer the busy-box question")
    up = doc["unpaired_reports"]
    print(f"  {len(up)} legacy unpaired report(s): {', '.join(up) or 'none'}")
    if unparseable:
        print(f"  {len(unparseable)} tracked *.json do not parse; recorded separately so a later"
              f" one can be told apart from these")
    return 0


def selftest() -> int:
    """The four ways this gate could be worthless: passing an empty corpus, treating a missing key as
    a zero, judging a file it was not asked to judge, and failing open when the baseline is gone."""
    bad = 0
    total = 0

    def check(name, cond):
        # Counted, never typed: the printed denominator went stale twice on this line already, and a
        # selftest whose "N/N" is hand-maintained cannot be trusted to report its own coverage.
        nonlocal bad, total
        total += 1
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        bad += 0 if cond else 1

    sw = _load("sw", "server_window.py")
    tmp = Path(tempfile.mkdtemp(prefix="pgate_"))

    def write(name: str, doc) -> Path:
        p = tmp / name
        p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
        return p

    # `rows`/`arms`/`requests` are themselves measured keys, so a CONTAINER carrying one is a
    # measured object too. That trap has now bitten twice on this line (an artifact that recorded
    # the box state still could not answer for half its objects), so both shapes are fixtures.
    leafwin = {"tps": 11.0, "window": {"class": "clean"}}
    leafbare = {"tps": 9.0}
    w = {"class": "clean"}
    nwin = write("nwin.json", {"rows": [leafbare], "window": w, "engine_digest": {"x": "y"}})
    ncont = write("ncont.json", {"rows": [leafwin], "engine_digest": {"x": "y"}})
    okprod = write("ok.json", {"rows": [leafwin], "window": w, "engine_digest": {"x": "y"}})
    nodig = write("nodig.json", {"rows": [leafwin], "window": w})
    cfg = write("cfg.json", {"profile": "prod25", "cells": ["decode"]})
    broken = write("broken.json", "{not json")

    _, o, _u = scan([nwin], sw, tmp)
    check("a measured LEAF with no window block is an offender", "nwin.json" in o)
    check("...and the reason names the window",
          "no window block" in o["nwin.json"]["reasons"][0])
    _, o, _u = scan([ncont], sw, tmp)
    check("a CONTAINER carrying a measured key needs a window too", "ncont.json" in o)
    check("...and the offender is the container, counted once",
          "1 measured object(s)" in o["ncont.json"]["reasons"][0])
    t, o, _u = scan([okprod], sw, tmp)
    check("a product with window on every measured object passes", "ok.json" not in o)
    check("...and both the container and the leaf are counted", t["measured_objects"] == 2)
    _, o, _u = scan([nodig], sw, tmp)
    check("a window without a digest is an offender too", "nodig.json" in o)
    check("...for the digest reason, not the window one",
          o["nodig.json"]["reasons"] == ["no engine digest anywhere in the product"])
    t, o, _u = scan([cfg], sw, tmp)
    check("a config file with no measured object is not judged", "cfg.json" not in o)
    check("...and is not counted as a measurement product", t["files"] == 0)
    t, o, u = scan([broken], sw, tmp)
    check("an unparseable tracked .json is named", "broken.json" in u)
    check("...but is NOT an offender (vendored JSONC configs were 6 of the 7 in this repo)",
          "broken.json" not in o)
    check("...and is not counted as a product either", t["files"] == 0 and t["unreadable"] == 1)
    check("a NEW unparseable file is raised as a warning while a baselined one is not",
          new_unparseable({"broken.json": "x"}, u) == []
          and new_unparseable({}, u) == ["broken.json"])

    check("find_digest finds a top-level identity",
          find_digest({"engine_digest": {"a": 1}}) == "engine_digest")
    check("find_digest finds one nested inside rows",
          find_digest({"rows": [{"binary": "x"}]}) == "binary")
    check("find_digest accepts the other line's name",
          find_digest({"provenance": {"b": 2}}) == "provenance")
    check("an empty identity is not an identity", find_digest({"digest": {}}) is None)

    new, fixed, legacy = compare({"nwin.json": {}}, {"nwin.json": {}, "ok.json": {}})
    check("a file not in the baseline is new", "ok.json" in new)
    check("a baselined file is tolerated", legacy == ["nwin.json"])
    check("a baselined file that is now clean is reported as progress",
          compare({"nwin.json": {}}, {"ok.json": {}})[1] == ["nwin.json"])

    check("an empty corpus yields zero offenders, not a pass on nothing",
          scan([], sw, tmp)[1] == {})

    check("keys are repo-relative (the baseline ships and is shared by several worktrees)",
          key_of(tmp / "nwin.json", tmp) == "nwin.json")
    check("a path outside the repo is not silently dropped",
          key_of(Path("/etc/hosts"), tmp) == "/etc/hosts")

    check("no new offender -> 0, one new offender -> 1",
          (not compare({"nwin.json": {}}, {"nwin.json": {}})[0])
          and bool(compare({"nwin.json": {}}, {"nwin.json": {}, "ok.json": {}})[0]))

    # the corpus and the shipped baseline: a gate that is installed but has nothing to judge
    check("this repo's corpus is not empty", len(tracked_json(ROOT)) > 0)
    base_doc = load_baseline(BASELINE)
    check("the shipped baseline was generated from a real corpus",
          bool(base_doc.get("offenders")) and base_doc.get("corpus") == CORPUS)
    def quiet_check(baseline: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            return cmd_check(argparse.Namespace(repo=str(ROOT), baseline=baseline, json=False))

    check("a missing baseline is refused, not treated as an empty one",
          quiet_check(str(tmp / "absent.json")) == 2)
    check("the shipped baseline covers the shipped corpus (the gate is green today)",
          quiet_check(str(BASELINE)) == 0)

    # ---- staged mode: the commit gate. "Judge the INDEX, not the worktree" can only be tested
    # against a real index, so this fixture is a throwaway git repo.
    gd = Path(tempfile.mkdtemp(prefix="pgate_git_"))

    def git(*a):
        return subprocess.run(["git", "-C", str(gd), *a], capture_output=True, text=True, check=True)

    def run_staged() -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_check_staged(argparse.Namespace(repo=str(gd), json=False))
        return rc, buf.getvalue()

    git("init", "-q")
    (gd / "bare.json").write_text(json.dumps({"rows": [leafbare]}))
    (gd / "clean.json").write_text(json.dumps({"rows": [leafwin], "window": w,
                                               "engine_digest": {"x": "y"}}))
    (gd / "cfgfile.json").write_text(json.dumps({"profile": "prod25"}))
    (gd / "unstaged.json").write_text(json.dumps({"rows": [leafbare]}))
    git("add", "bare.json", "clean.json", "cfgfile.json")
    names, err = staged_json(gd)
    check("staged mode sees exactly the *.json in the index",
          names == ["bare.json", "cfgfile.json", "clean.json"] and err is None)
    check("...and ignores a modified-but-unstaged file", "unstaged.json" not in names)
    rc, out = run_staged()
    check("a staged product with no window blocks the commit",
          rc == 1 and "bare.json" in out and "no window block" in out)
    check("a staged product carrying both halves passes", "clean.json" not in out)
    check("a staged file with no measured object is not judged", "cfgfile.json" not in out)
    (gd / "bare.json").write_text(json.dumps({"rows": [leafwin], "window": w,
                                             "engine_digest": {"x": "y"}}))
    rc, out = run_staged()
    check("fixing it in the WORKTREE does not unblock the commit (the index still holds the bare one)",
          rc == 1 and "bare.json" in out)
    git("add", "bare.json")
    rc, out = run_staged()
    check("staging the fix does unblock it", rc == 0 and "PASS" in out)
    git("rm", "--cached", "-q", "bare.json", "clean.json", "cfgfile.json")
    rc, out = run_staged()
    check("an empty staged set passes but SAYS it judged nothing",
          rc == 0 and "nothing this gate judges" in out)
    with contextlib.redirect_stdout(io.StringIO()):
        rc = cmd_check_staged(argparse.Namespace(repo=str(tmp / "not-a-repo"), json=False))
    check("a failing git diff is refused, not read as `nothing to judge`", rc == 2)

    # ---- the report/sidecar pairing: registered pairs move together
    check("a registered report resolves to its sidecar",
          pair_member("docs/TARGET25_REPORT_x.html", ROOT)[0] == "docs/TARGET25_REPORT_x.json")
    check("an unregistered html is NOT judged (docs/ is mostly hand-written prose)",
          pair_member("docs/MY_NOTES.html", ROOT) is None)
    check("a plain *.json is not an orphaned sidecar for a report that never existed",
          pair_member("docs/STANDALONE.json", ROOT) is None)
    check("the .json half of a real pair resolves back to its .html",
          pair_member("docs/TARGET25_REPORT_20260922_184209.json", ROOT)[0]
          == "docs/TARGET25_REPORT_20260922_184209.html")

    def run_staged_case(paths: dict[str, str]) -> tuple[int, str]:
        for rel in paths:
            (gd / rel).parent.mkdir(parents=True, exist_ok=True)
        for rel, body in paths.items():
            (gd / rel).write_text(body)
        git("add", "-A", "--", *paths)
        rc_, out_ = run_staged()
        git("reset", "-q", "--", *paths)
        return rc_, out_

    rc, out = run_staged_case({"docs/TARGET25_REPORT_t.html": "<html>x</html>"})
    check("a staged report with no sidecar in the same commit blocks",
          rc == 1 and "split by this commit" in out and "TARGET25_REPORT_t.json" in out)
    check("...and the message names the producer, so the fix is a one-liner",
          "target25_report.py" in out)
    rc, out = run_staged_case({"docs/TARGET25_REPORT_t.json": json.dumps(
        {"rows": [leafwin], "window": w, "engine_digest": {"x": "y"}})})
    check("a staged sidecar with no report blocks too (the pair is symmetric)",
          rc == 1 and "TARGET25_REPORT_t.html" in out)
    rc, out = run_staged_case({"docs/TARGET25_REPORT_t.html": "<html>x</html>",
                               "docs/TARGET25_REPORT_t.json": json.dumps(
                                   {"rows": [leafwin], "window": w, "engine_digest": {"x": "y"}})})
    check("both halves in one commit pass", rc == 0 and "PASS" in out)
    rc, out = run_staged_case({"docs/HAND_WRITTEN_HTML_NOTES.html": "<html>prose</html>"})
    check("a hand-written html is not judged (no producer is registered for it)",
          rc == 0 and "nothing this gate judges" in out)

    # The repair shape: the report is ALREADY COMMITTED and the sidecar arrives now. Blocking this
    # would make "give the old report its evidence" impossible to land -- the rule's own intent.
    legacy_html = "docs/DECODE_WINDOW_2026-09-19.html"
    (gd / "docs").mkdir(parents=True, exist_ok=True)
    (gd / legacy_html).write_text("<html>a report published before sidecars existed</html>")
    git("add", "--", legacy_html)
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "the report, alone")
    rc, out = run_staged_case({legacy_html.replace(".html", ".json"): json.dumps(
        {"rows": [leafwin], "window": w, "engine_digest": {"x": "y"}})})
    check("a sidecar arriving after its report is committed COMPLETES the pair, not splits it",
          rc == 0 and "completes the pair" in out)
    check("...and that sidecar is still judged as a product (completing a pair is not a bypass)",
          "measurement products among them : 1" in out)

    # ---- the registry's completeness: a producer nobody registered is a FAIL, not a silent hole
    check("the scan fires on the shape a producer really uses",
          docs_html_writer_lines('out = ROOT / "docs" / f"X_{stamp}.html"') == [1])
    check("...and on the os.path.join shape",
          docs_html_writer_lines('p = os.path.join(ROOT, "docs", f"Y_{d}.html")') == [1])
    check("...but not on a docstring mention",
          docs_html_writer_lines('"""see docs/Z_2026-09-22.html for the readings"""') == [])
    check("...nor on a comment",
          docs_html_writer_lines('# docs/W_2026-09-22.html asked this question') == [])
    check("...nor on a stderr hint (the one decoy the naive rule hit)",
          docs_html_writer_lines('print(f"docs/V_2026-09-14.html.", file=sys.stderr)') == [])
    rc, out = run_staged_case({"scripts/check/brand_new_report.py":
                               'out = ROOT / "docs" / f"NEW_{stamp}.html"\n'})
    check("an unregistered new producer in the commit blocks",
          rc == 1 and "brand_new_report.py" in out and "REPORT_PRODUCERS" in out)
    rc, out = run_staged_case({"scripts/check/not_a_producer.py": "x = 1\n"})
    check("...and an ordinary new script does not", rc == 0 and "PASS" in out)

    found = set(producers_writing_html_docs(ROOT))
    check("the shipped registry covers every producer in THIS tree (stale registry -> red selftest)",
          found <= registered_tools())
    check("...and it found the two producers it claims",
          {"scripts/check/decode_window_harness.py", "scripts/check/target25_report.py"} <= found)
    check("every exemption names a file that exists, with a reason",
          bool(REPORT_EXEMPT) and all((ROOT / k).exists() and v.strip()
                                      for k, v in REPORT_EXEMPT.items()))
    check("no exemption is stale (an exempted file still has to mention a docs/*.html)",
          set(REPORT_EXEMPT) <= found)

    # ---- an unpaired artifact that the baseline does not know about is what FAILS a run
    fake = {"prod": {"unpaired": ["docs/DECODE_WINDOW_2026-09-19.html"]}}
    check("a legacy unpaired artifact is tolerated",
          new_unpaired(["docs/DECODE_WINDOW_2026-09-19.html"], fake) == [])
    check("a NEW unpaired artifact is not",
          new_unpaired([], fake) == ["docs/DECODE_WINDOW_2026-09-19.html"])
    check("unpaired_reports() flattens the per-producer lists",
          unpaired_reports({"a": {"unpaired": ["x"]}, "b": {"unpaired": ["y"]}}) == ["x", "y"])
    live = pair_audit(ROOT)
    check(f"NOTHING in this tree is unpaired today (the rule's own self-test; found "
          f"{unpaired_reports(live)})", not unpaired_reports(live))
    check("the shipped baseline covers today's unpaired artifacts (nothing NEW, or the gate is red)",
          new_unpaired(load_baseline(BASELINE).get("unpaired_reports", []), live) == [])
    check("...and the audit records tracked vs on-disk separately",
          all("tracked_artifacts" in v for v in live.values()))

    # ---- the glob has to be the producer's naming rule, not `DECODE_WINDOW_*.html`: the wider one
    # judged hand-curated prose ABOUT the harness and demanded evidence from an artifact nobody
    # measured (the same defect as demanding sidecars from the 77 hand-written docs/).
    check("a dated report is a producer artifact",
          pair_member("docs/DECODE_WINDOW_2026-09-19.html", ROOT)[0] == "docs/DECODE_WINDOW_2026-09-19.json")
    check("prose about the harness that merely shares the prefix is NOT judged",
          pair_member("docs/DECODE_WINDOW_HARNESS_20260919_1745.html", ROOT) is None)

    # ---- the backfill: a report published before sidecars existed can be given one by COPYING its
    # own text, and must be refused when there is no evidence in it to copy.
    # Two lines as the harness printed them on 2026-09-19, trimmed to what the backfill reads: the
    # digest line, one arm row, one raw window reading (the em dash is a missing swap value).
    digest_fixture = (
        "<html><p>engine digests: {&quot;llama-server&quot;: &quot;deadbeefdeadbeef&quot;}</p>"
        "<table><tr><td>b1</td><td>1</td><td>102.45</td>"
        "<td>llama_server_20260919_154509.log</td></tr></table>"
        "<table><tr><td>16:41:52</td><td>920</td><td>\u2014</td></tr></table></html>")
    dwh = _load("dwh", "decode_window_harness.py")
    bf = dwh.backfill_doc(digest_fixture, "docs/DECODE_WINDOW_2026-09-19.html")
    check("a backfilled sidecar carries the report's own digests, not a run's",
          bf["engine_digest"] == {"llama-server": "deadbeefdeadbeef"}
          and "NOTHING here was re-measured" in bf["source"])
    check("...and the rows and window readings it found",
          bf["rows"] == [{"lane": "A", "arm": "b1", "bitident": 1, "ms_per_token": 102.45,
                          "log": "llama_server_20260919_154509.log", "window": bf["window"]}]
          and bf["window"]["samples"] == [{"ts": "16:41:52", "free_mb": 920, "swap_mb": None}])
    check("...and a window block, so it answers the busy-box question",
          bf["window"]["class"] == "unknown" and "backfilled" in bf["window"]["why"])
    check("...and its rows carry that block too (a container-only block reports half unanswerable)",
          bool(bf["rows"]) and all("window" in r for r in bf["rows"]))
    bfp = write("backfilled.json", bf)
    audit = sw.audit_products([str(bfp)])
    check("a backfilled sidecar passes the shared busy-box audit (nothing left answerless)",
          find_digest(bf) == "engine_digest"
          and audit["with_window"] == audit["measured_objects"] > 0 and audit["without"] == 0)
    check("a report with no digest line gets NO sidecar (nothing to copy, so nothing is invented)",
          dwh.backfill_doc("<html><p>no evidence here</p></html>", "x.html") is None)

    print(f"selftest: {total - bad}/{total} passed")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    # `selftest` is both a subcommand and a flag on purpose: `server_window.py` takes the subcommand
    # form and the other tools in this driver's family take `--selftest`, and a tool whose selftest
    # you cannot remember how to run is one nobody runs.
    for name in ("check", "update", "selftest"):
        s = sub.add_parser(name)
        s.add_argument("--repo", default=str(ROOT))
        s.add_argument("--baseline", default=str(BASELINE))
        s.add_argument("--json", action="store_true")
        s.add_argument("--staged", action="store_true",
                       help="check: judge the *.json this commit adds or modifies (the commit gate)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest or args.cmd == "selftest":
        return selftest()
    if args.cmd == "check":
        return cmd_check_staged(args) if args.staged else cmd_check(args)
    if args.cmd == "update":
        return cmd_update(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
