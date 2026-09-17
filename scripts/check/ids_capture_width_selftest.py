#!/usr/bin/env python3
"""Prove the id-capture READERS survive the full-width record -- and that "0 differences" can be
told apart from "not looked at".

WHY THIS EXISTS (2026-09-17, §9.18.9)
------------------------------------
`CGC_IDS_STRIDE` was 8 while the submitted width is `ne20*ne21` (= top_k * n_tokens), and the printer
used the stride, so every multi-token row printed the first 8 ids -- which, the operand being
ne0-fastest, ARE token 0. Every "the ids are identical" verdict taken from a T>1 pass was therefore
a verdict about one token; widening the record moved an A/B's "first divergence" from pass 3 to
pass 0 and from 114/120 to 117/120 MoE nodes (docs/ROUTING_TRACE_2026-09-17.md §10).

The instrument is fixed. These readers are the other half: they must PARSE a 16- or 64-word row,
and a difference confined to token >= 1 must be VISIBLE through them. A reader that quietly sliced
the list to 8 would reproduce the original defect one layer up, and nothing else in the pipeline
would notice.

The second property is the one that actually bit: a "no difference" summary that does not name what
it compared. `ids_capture_diff.py` compares ONE graph by default, so `TOTAL differing nodes: 0`
after graph 0 of a 35-graph run reads as "nothing in the run differs". Case 3 pins that wording.

WHAT IT DOES
------------
Builds synthetic captures in a temp dir (nothing else is touched, no GPU, no model, no build) and
runs the three readers over them as subprocesses, asserting on their OUTPUT -- because the failure
mode this guards is a plausible-looking report, not an exception.

The two `Backup/` readers are SKIPPED (not failed) when absent: `Backup/` is gitignored, so a fresh
clone legitimately has them missing. Skips are counted separately and never reported as passes.

Usage:
    python3 scripts/check/ids_capture_width_selftest.py [--keep]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDS_DIFF = os.path.join(REPO, "scripts", "check", "ids_capture_diff.py")
POOL_ROW = os.path.join(REPO, "Backup", "compare_pool_row.py")
NODES = os.path.join(REPO, "Backup", "analyze_capture_nodes.py")

TOK0 = [2, 105, 7, 9, 5, 106, 1, 84]
TOK1_A = [8, 0, 50, 4, 11, 12, 13, 14]
TOK1_B = [0, 103, 0, 74, 11, 12, 13, 14]      # 7 of 8 are slot 0 -- the observed S1 shape
G1 = [(5, 100, 77, 9999, 4096), (9, 200, 66, 8888, 4096)]
G2 = [(7, 111, 55, 7777, 4096), (3, 222, 44, 6666, 4096)]


# --- fixture builders --------------------------------------------------------------------------

def mv(slot, name, words):
    return (f"CGC-IDS-CAP slot={slot} path=MV n_ids={len(words)} name={name} "
            f"ids=[{','.join(map(str, words))}]\n")


def dst(slot, name, words, fuse=1, ne=2048, off=0):
    return (f"CGC-IDS-CAP slot={slot} path=DST n_ids={len(words)} name={name} "
            f"ids=[{','.join(map(str, words))}] fuse={fuse} ne={ne} off={off}\n")


def pool(slot, name, groups, own, exp, probe=4096, nsel=16, frm="MV"):
    w = [len(groups)]
    for g in groups:
        w += list(g)
    o = "[" + ",".join(map(str, own)) + "]"
    e = "[" + ",".join(map(str, exp)) + "]"
    return (f"CGC-IDS-CAP slot={slot} path=POOL n_ids={len(w)} name={name} "
            f"ids=[{','.join(map(str, w))}] rows={len(groups)} probe={probe} nsel={nsel} "
            f"from={frm} own={o} exp={e}\n")


def graph(g, tok1, n_tokens=2, l_out=None, words=None):
    """One pass. `n_tokens` is what makes the record wide: 2 -> 16 words, 8 -> 64 words.

    `words` overrides the whole record, so a case can park a difference at a chosen WORD index --
    which is the only way to show that the far end of a 64-word row is still being read.
    """
    if words is None:
        words = list(TOK0) + list(tok1)
        words += [100 + 10 * t + i for t in range(2, n_tokens) for i in range(8)]
    base = 10 * g
    lout = [1, 2, 3, 4] if l_out is None else l_out
    return (mv(base + 1, "ffn_moe_gate-1", words) +
            mv(base + 2, "ffn_moe_down-1", words) +
            dst(base + 3, "attn_norm-1.dst", [1, 2, 3, 4]) +
            dst(base + 4, "l_out-1.dst", lout) +
            pool(base + 5, "ffn_moe_gate-1.pool", G1, own=[5, 9], exp=[5, 9]) +
            pool(base + 6, "ffn_moe_down-1.pool", G2, own=[7, 3], exp=[7, 3]))


def wide_words(n_tokens=8, mutate=None):
    words = list(TOK0) + list(TOK1_A)
    words += [100 + 10 * t + i for t in range(2, n_tokens) for i in range(8)]
    if mutate is not None:
        words[mutate] = 0
    return words


def fmt(ids):
    return "[" + ",".join(map(str, ids)) + "]"


# --- harness -----------------------------------------------------------------------------------

class T:
    def __init__(self):
        self.n = self.fail = self.skip = 0

    def run(self, script, *argv):
        r = subprocess.run([sys.executable, script, *argv], capture_output=True, text=True, cwd=D)
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    def check(self, name, cond, detail=""):
        self.n += 1
        if cond:
            print(f"  PASS  {name}")
        else:
            self.fail += 1
            print(f"  FAIL  {name}")
            if detail:
                for l in str(detail).splitlines()[:8]:
                    print(f"          {l}")

    def case(self, name, fn, needs=None):
        if needs is not None and not os.path.exists(needs):
            self.skip += 1
            print(f"  SKIP  {name}   (missing {os.path.relpath(needs, REPO)} -- Backups are not in "
                  f"git, so this is a skip, not a pass)")
            return
        print(f"  ---- {name}")
        fn()


D = ""


def resolve(repo):
    """Point the three reader paths at `repo`. Overridable because the default is derived from
    __file__ -- and getting that wrong once made this whole test report `0/0 checks passed,
    6 skipped` with rc=0, i.e. green having tested nothing. See the zero-check guard below."""
    global REPO, IDS_DIFF, POOL_ROW, NODES
    REPO = os.path.abspath(repo)
    IDS_DIFF = os.path.join(REPO, "scripts", "check", "ids_capture_diff.py")
    POOL_ROW = os.path.join(REPO, "Backup", "compare_pool_row.py")
    NODES = os.path.join(REPO, "Backup", "analyze_capture_nodes.py")


def main(argv=None):
    global D
    av = list(argv if argv is not None else sys.argv[1:])
    keep = "--keep" in av
    if "--repo" in av:
        resolve(av[av.index("--repo") + 1])
    D = tempfile.mkdtemp(prefix="ids_width_")
    try:
        return _run(keep)
    finally:
        if keep:
            print(f"\ntemp dir kept: {D}")
        else:
            shutil.rmtree(D, ignore_errors=True)


def _run(keep):
    tok1_only = open(os.path.join(D, "tok1_A.log"), "w")
    tok1_only.write(graph(0, TOK1_A) + graph(1, TOK1_A))
    tok1_only.close()
    open(os.path.join(D, "tok1_B.log"), "w").write(graph(0, TOK1_A) + graph(1, TOK1_B))

    # T=8: 64-word records, with the difference parked at WORD 40 (= token 5's first id). A reader
    # that sliced to 8 words could not reach it even for token 1, so this is the sharpest form of
    # "the far end of the record is still being read".
    w_a = wide_words()
    w_b = wide_words(mutate=40)
    open(os.path.join(D, "t8_A.log"), "w").write(graph(0, None, words=w_a))
    open(os.path.join(D, "t8_B.log"), "w").write(graph(0, None, words=w_b))

    open(os.path.join(D, "dst_B.log"), "w").write(
        graph(0, TOK1_A) + graph(1, TOK1_A, l_out=[9, 9, 9, 9]))

    T_ = T()

    def c1():
        rc, out = T_.run(IDS_DIFF, os.path.join(D, "tok1_A.log"), os.path.join(D, "tok1_B.log"),
                         "--all-graphs")
        T_.check("a token-1-only difference is REPORTED (2 nodes in graph 1)",
                 "graph 1: 6 nodes compared -> 2 DIFF" in out and
                 "TOTAL differing nodes: 2" in out, out)
        T_.check("the differing row is printed at full width (16 ids)",
                 "ids=[2,105,7,9,5,106,1,84,0,103,0,74,11,12,13,14]" in out, out)
        T_.check("rc=1 when a difference exists", rc == 1, f"rc={rc}")

    def c2():
        rc, out = T_.run(IDS_DIFF, os.path.join(D, "tok1_A.log"), os.path.join(D, "tok1_A.log"),
                         "--all-graphs")
        T_.check("identical wide records -> 0 differences, rc=0",
                 "TOTAL differing nodes: 0" in out and rc == 0, out)

    def c3():
        rc, out = T_.run(IDS_DIFF, os.path.join(D, "tok1_A.log"), os.path.join(D, "tok1_B.log"))
        T_.check("the default scope is NAMED in the total (graph 0 ONLY)",
                 "graph 0 ONLY" in out, out)
        T_.check("... and the un-compared graphs are called out",
                 "1 were NOT compared" in out and "--all-graphs" in out, out)

    def c4():
        rc, out = T_.run(IDS_DIFF, os.path.join(D, "t8_A.log"), os.path.join(D, "t8_B.log"),
                         "--all-graphs")
        T_.check("a 64-word record is parsed whole",
                 "n_ids=64" in out, out)
        T_.check("... and the A row is printed as all 64 words, not a prefix",
                 f"ids={fmt(w_a)}" in out, out)
        T_.check("a difference at WORD 40 (token 5) is visible",
                 "graph 0: 6 nodes compared -> 2 DIFF" in out and
                 "TOTAL differing nodes: 2" in out, out)

    def c5():
        rc, out = T_.run(POOL_ROW, os.path.join(D, "tok1_A.log"), os.path.join(D, "tok1_B.log"),
                         "--max-graphs", "0", "--quiet-same")
        T_.check("compare_pool_row parses wide MV rows (unmatched=0)",
                 "unmatched=0" in out, out)
        T_.check("... and the POOL verdict is untouched by the MV width",
                 "SAME=4" in out and "DIFF=0" in out, out)

    def c6():
        rc, out = T_.run(NODES, os.path.join(D, "tok1_A.log"), os.path.join(D, "dst_B.log"),
                         "--graphs", "0")
        T_.check("analyze_capture_nodes runs on wide MV rows and excludes POOL rows out loud",
                 "EXCLUDED from this comparison" in out, out)
        row = [l.split() for l in out.splitlines() if l.strip()]
        T_.check("... and its per-graph table says graph 1 has 1 DST DIFF, 0 ids DIFF",
                 ["1", "2", "0", "1", "1"] in row, out)

    print(f"id-capture width selftest  (repo {REPO})")
    print(f"  fixtures in {D}\n")
    T_.case("1. ids_capture_diff --all-graphs sees a token-1 divergence", c1, IDS_DIFF)
    T_.case("2. ids_capture_diff control: identical input", c2, IDS_DIFF)
    T_.case("3. ids_capture_diff: the scope of a zero is stated", c3, IDS_DIFF)
    T_.case("4. 64-word (T=8) records", c4, IDS_DIFF)
    T_.case("5. compare_pool_row with wide MV rows", c5, POOL_ROW)
    T_.case("6. analyze_capture_nodes with wide MV rows", c6, NODES)

    print(f"\n{T_.n - T_.fail}/{T_.n} checks passed, {T_.skip} skipped")
    if T_.n == 0:
        print("!! NOTHING WAS TESTED (0 checks). Every reader was missing, so this run proved "
              "nothing -- reported as FAILURE, because a green 'nothing ran' is the failure mode "
              "this whole file exists to catch.")
        return 1
    return 1 if T_.fail else 0


if __name__ == "__main__":
    sys.exit(main())
