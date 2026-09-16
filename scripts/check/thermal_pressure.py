#!/usr/bin/env python3
"""Read the OS thermal pressure level, in band, next to whatever is being measured.

WHY THIS EXISTS. The decode baseline is not reproducible: the same arm, same settings,
same day produced 6.6 / 9.75 / 10.24 / 16.17 / 6.95 t/s -- a 2.4x spread (see
`.workbuddy/memory/2026-09-16.md` section U). Without a reading taken at the same moment,
that spread cannot be attributed, so any "Nx faster" claim about decode is unfalsifiable:
a 1.5x win and a thermal slump are the same observation.

WHY THIS INSTRUMENT. `com.apple.system.thermalpressurelevel` is a notify(3) key the OS
publishes; it is the SAME key `powermetrics`' thermal sampler reads, it needs no root, and
it costs ~2 ms (measured: 12 calls in 0.02 s -- cheaper than the round trip to ask the
server, so it can go inside a per-round loop).

The scale, and the separation it gave on prefill (2026-09-16, request level, zero overlap):

    0 Nominal   1 Moderate   2 Heavy   3 Trapping   4 Sleeping

  * read 0 at launch  ->  6/6 runs >= 250 t/s   (253.42 - 271.64)
  * read 1 or 2       ->  0/21 runs >= 250 t/s  (104.88 - 211.65)

The criterion is the level AT LAUNCH, not over the whole run: one arm read 0 at launch,
drifted to 1 then 2 mid-run, and its three requests still came in at 257.41 / 253.42 / 262.64.
See `docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html` and lesson `eng-gate-0038`.

THE TRAP THIS MODULE IS BUILT AROUND (measured 2026-09-16, macOS on this box):

    $ notifyutil -g com.apple.system.thermalpressurelevel        # the real key
    com.apple.system.thermalpressurelevel 0                      # exit 0
    $ notifyutil -g com.apple.system.this.key.does.not.exist     # a key that is not there
    com.apple.system.this.key.does.not.exist 0                   # exit 0, same shape

`notifyutil -g` answers `0` -- and exits 0 -- for a key that DOES NOT EXIST. `-v` and `-q`
do not change that (verified). So neither the exit code nor the output shape separates
"present and Nominal" from "absent". A parser that trusts them reports `NOMINAL` on a
machine where the instrument is missing entirely, and that is the same failure family as a
silent zero: "we did not measure" wearing the costume of "we measured, and it was fine".

What this module does about it, and what it cannot:
  * It refuses any key it has not been verified against (`SUPPORTED_KEYS`) instead of
    inventing a reading -- this kills the typo class, which is the one it can kill.
  * It CANNOT detect a supported key being absent on some machine. On such a machine the
    reading would be a permanent `NOMINAL`. The mitigation is out of band and already exists
    in this project: the key's liveness was established by its measured response to load
    (0 at rest, 2 under sustained load, ~47 s to return to 0). **A permanently-0 series is
    the signature to be suspicious of**, and the way to test it is a load test, not a parser.
  * It never folds an unknown into a number: unreadable is `None` / `UNREADABLE`.

DELIBERATELY NOT A GATE. This module records; it does not refuse to run. Decode has no
measured level -> throughput mapping yet -- establishing one is the point of recording it
everywhere first. Do not start gating decode numbers on a level until the separation has
been measured the way the prefill separation was.
"""

from __future__ import annotations

import subprocess
import sys
import time

KEY = "com.apple.system.thermalpressurelevel"

# Only keys whose behaviour has actually been observed on this box. The tuple exists so a
# typo (or an optimistic caller) cannot turn into a plausible-looking reading -- see the
# docstring, where the tool's own absent-key behaviour makes that otherwise undetectable.
SUPPORTED_KEYS = (KEY,)

# The labels are the kernel's own, spelled out; they are what a reader can sanity-check
# against `powermetrics` output without knowing this file exists.
LABELS = {0: "NOMINAL", 1: "MODERATE", 2: "HEAVY", 3: "TRAPPING", 4: "SLEEPING"}

UNREADABLE = "UNREADABLE"


def level(key: str = KEY):
    """The level as an int 0..4, or None when it could not be read.

    None is returned for: a key outside `SUPPORTED_KEYS`, notifyutil missing (non-macOS),
    non-zero exit, output that does not name the key we asked for, or a non-integer value.
    Every one of those is "we do not know", never "it is 0". The one case this cannot catch
    -- a supported key that is absent on this machine -- is documented in the module
    docstring rather than papered over here.
    """
    if key not in SUPPORTED_KEYS:
        return None
    try:
        out = subprocess.run(["notifyutil", "-g", key],
                             capture_output=True, text=True, timeout=5)
    except Exception:  # noqa: BLE001 - a missing tool must not kill a measurement
        return None
    if out.returncode != 0:
        return None
    parts = out.stdout.split()
    if len(parts) < 2 or parts[0] != key:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def label(lv):
    """`NOMINAL` / ... for a level, `UNREADABLE` for None or anything out of range."""
    if lv is None:
        return UNREADABLE
    return LABELS.get(lv, "UNKNOWN-%r" % (lv,))


def stamp(key: str = KEY) -> dict:
    """One JSON-safe reading: `{level, label, t}`. `level` is None when unreadable."""
    lv = level(key)
    return {"level": lv, "label": label(lv), "t": time.strftime("%H:%M:%S")}


def worst(stamps) -> dict:
    """The highest level among readings -- the one that bounds the claim.

    An unreadable reading is NOT treated as worst-case and NOT ignored silently: if every
    reading failed, the result is an UNREADABLE stamp with level None.
    """
    lv = [s.get("level") for s in stamps if isinstance(s, dict)]
    known = [x for x in lv if x is not None]
    if not known:
        return {"level": None, "label": UNREADABLE, "t": ""}
    m = max(known)
    return {"level": m, "label": label(m), "t": ""}


def histogram(stamps) -> dict:
    """`{label: count}` over readings -- how much of a run was spent hot.

    A median level hides the shape: 3 rounds at 0 then 3 at 2 has the same median as six
    rounds at 1, and they are not the same experiment. Report the counts.
    """
    out: dict = {}
    for s in stamps:
        if isinstance(s, dict):
            out[s.get("label", UNREADABLE)] = out.get(s.get("label", UNREADABLE), 0) + 1
    return out


def _selftest() -> int:
    """Prove the failure mode: an unreadable key must NOT come back as 0.

    A parser that maps "no answer" onto a plausible number is worse than one that crashes --
    it turns "we did not measure" into "we measured Nominal", and that is exactly how the
    2.4x decode spread became invisible in the first place.
    """
    bad = 0

    def check(name, ok, detail=""):
        nonlocal bad
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
        if not ok:
            bad += 1

    check("label(None) is UNREADABLE", label(None) == UNREADABLE, label(None))
    check("every documented level has a label",
          all(label(i) not in (UNREADABLE, "UNKNOWN") for i in range(5)))
    check("label() does not invent a label for out-of-range",
          label(9).startswith("UNKNOWN"), label(9))

    # The load-bearing one. `notifyutil -g <bogus>` answers `0` and exits 0, so this cannot
    # be caught by inspecting the output -- it has to be refused before the call.
    bogus = "com.apple.system.this.key.does.not.exist"
    check("unsupported key -> None (never 0)", level(bogus) is None, repr(level(bogus)))
    check("a supported-shape key is still read",
          KEY in SUPPORTED_KEYS)

    s = stamp(bogus)
    check("stamp() of a refused key is labelled UNREADABLE",
          s["level"] is None and s["label"] == UNREADABLE, repr(s))
    check("worst() of only-unreadable is UNREADABLE, not 0",
          worst([s, s])["level"] is None, repr(worst([s, s])))
    check("worst() takes the max of known readings",
          worst([{"level": 0}, {"level": 2}, {"level": None}])["level"] == 2)
    check("histogram() counts every reading including unreadable",
          histogram([{"label": "NOMINAL"}, {"label": "NOMINAL"}, {"label": UNREADABLE}])
          == {"NOMINAL": 2, UNREADABLE: 1})

    live = level()
    check("the real key reads an int 0..4 on this machine (or is honestly unreadable)",
          live is None or 0 <= live <= 4,
          f"read {label(live)}" + (f" ({live})" if live is not None else ""))

    print()
    print(f"  {10 - bad}/10 checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    lv = level()
    print(f"{KEY} {label(lv)}" + (f" ({lv})" if lv is not None else " (level unknown)"))
