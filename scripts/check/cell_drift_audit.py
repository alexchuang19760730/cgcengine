#!/usr/bin/env python3
"""Audit cell provenance drift across every saved bench artifact (the CI half of the single-mouth
contract). Scans JSON artifacts and classifies each arm by whether the declared warm-skip was
ACTUALLY applied -- a nominal `--warm-skip 64` does not imply it took effect.

WHY THIS EXISTS
---------------
A scan on 2026-09-25 found arms that nominally carried warm-skip but whose tg n_gen was not reduced
("名義有、實際沒有"): quoting those as hot-cache numbers silently mixes cold and hot regimes. This
script turns that into a red/green light a CI gate can consume.

VERDICTS
  applied       tg n_gen == card_gen - warm_skip  (or new artifact's warm_skip_applied=True)
  not_applied   warm-skip declared (>0) but tg n_gen was NOT reduced  -> CI RED
  unknown       old format without a recorded warm_skip, or ambiguous (needs the parent run)
  not_required  the arm did not request warm-skip

USAGE
    python3 scripts/check/cell_drift_audit.py --roots Backup
    python3 scripts/check/cell_drift_audit.py --roots Backup --json /tmp/drift.json
    python3 scripts/check/cell_drift_audit.py --selftest
EXIT: 1 if any arm is not_applied (red). --strict also reddens unknown.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_contract as cc  # noqa: E402

# Artifacts larger than this are almost certainly full dumps, not bench summaries; skip them.
_MAX_JSON_BYTES = 15_000_000


def classify_arm(arm, card_gen: int):
    """Return (verdict, tag, why) or None if this dict is not a matrix arm."""
    if not isinstance(arm, dict) or not isinstance(arm.get("rows"), list):
        return None
    tag = str(arm.get("tag") or arm.get("profile") or "?")
    tg = [r for r in arm["rows"]
          if int(r.get("n_prompt", 0)) == 0 and int(r.get("n_gen", 0)) > 0]

    # New artifacts carry the self-proving field.
    if "warm_skip_applied" in arm:
        wa, ws = arm.get("warm_skip_applied"), arm.get("warm_skip")
        if wa is True:
            return ("applied", tag, "warm_skip_applied=True")
        if wa is False:
            return ("not_applied", tag, f"warm_skip_applied=False (nominal ws={ws})")
        if ws:
            return ("unknown", tag, f"ws={ws} but no tg conclusion")
        return ("not_required", tag, "warm-skip not requested")

    # Intermediate / old artifacts: derive from tg n_gen.
    ws = arm.get("warm_skip")
    if not ws:
        return ("unknown", tag, "artifact does not record warm_skip (old format; intent unknown)")
    ws = int(ws)
    if not tg:
        return ("unknown", tag, f"ws={ws} but no tg row")
    ng = [int(r.get("n_gen", -1)) for r in tg]
    expected = card_gen - ws
    if all(n == expected for n in ng):
        return ("applied", tag, f"tg n_gen={ng[0]} == {expected}")
    if all(n == card_gen for n in ng):
        return ("not_applied", tag, f"ws={ws} nominal but tg n_gen={ng[0]} not reduced (want {expected})")
    return ("unknown", tag, f"ws={ws} tg n_gen={ng} (neither {expected} nor {card_gen})")


def scan(roots: list[str]):
    contract = cc.load_contract()
    card_gen = int(contract["cell"]["gen"])
    order = ["applied", "not_applied", "unknown", "not_required"]
    counts = {k: 0 for k in order}
    details: list[dict] = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.json")):
            try:
                if p.stat().st_size > _MAX_JSON_BYTES:
                    continue
                data = json.loads(p.read_text())
            except Exception:
                continue
            arms = data if isinstance(data, list) else [data]
            for arm in arms:
                r = classify_arm(arm, card_gen)
                if r is None:
                    continue
                verdict, tag, why = r
                counts[verdict] += 1
                details.append({"file": str(p), "tag": tag, "verdict": verdict, "why": why})
    return counts, details


def _selftest() -> int:
    G = 128
    cases = [
        ("new applied", {"rows": [{"n_prompt": 0, "n_gen": 64}],
                         "warm_skip_applied": True, "warm_skip": 64}, "applied"),
        ("new not applied", {"rows": [{"n_prompt": 0, "n_gen": 128}],
                             "warm_skip_applied": False, "warm_skip": 64}, "not_applied"),
        ("new no warm", {"rows": [{"n_prompt": 0, "n_gen": 128}],
                         "warm_skip_applied": None}, "not_required"),
        ("mid applied", {"rows": [{"n_prompt": 0, "n_gen": 64}], "warm_skip": 64}, "applied"),
        ("mid not applied (run3)", {"rows": [{"n_prompt": 0, "n_gen": 128}],
                                    "warm_skip": 64}, "not_applied"),
        ("old unknown", {"rows": [{"n_prompt": 0, "n_gen": 128}]}, "unknown"),
        ("not an arm", {"foo": 1}, None),
    ]
    fails = 0
    for name, arm, want in cases:
        got = classify_arm(arm, G)
        gv = got[0] if got else None
        ok = gv == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {gv}")
        fails += not ok
    if fails:
        print(f"SELFTEST FAIL ({fails})")
        return 1
    print("selftest 7/7 PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roots", nargs="+", default=["Backup"])
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--show-unknown", action="store_true")
    ap.add_argument("--strict", action="store_true", help="also treat unknown as red")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()

    counts, details = scan(args.roots)
    print("cell provenance drift audit:")
    for k in ("applied", "not_applied", "unknown", "not_required"):
        print(f"  {k:<13} {counts[k]}")
    for d in details:
        if d["verdict"] == "not_applied" or (args.show_unknown and d["verdict"] == "unknown"):
            print(f"  [{d['verdict']}] {d['file']} :: {d['tag']} -- {d['why']}")
    if args.json_path:
        Path(args.json_path).write_text(json.dumps({"counts": counts, "details": details},
                                                    indent=2, ensure_ascii=False))
        print(f"json -> {args.json_path}")
    red = counts["not_applied"] > 0 or (args.strict and counts["unknown"] > 0)
    if red:
        print("RESULT: RED — quoted numbers above are not trustworthy (see not_applied).")
    else:
        print("RESULT: GREEN (use --strict to also fail on unknown).")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
