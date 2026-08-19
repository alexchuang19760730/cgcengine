#!/usr/bin/env python3
"""Parse Q36IMP layer-importance tables + routing trace, output pruning candidates."""
import re
import statistics
from collections import defaultdict

lines = open("/tmp/q36_imp_run.log").read().splitlines()
tables = []  # (tokens, [layer lines])
cur_tok = None
for l in lines:
    if l.startswith("Q36IMP tokens="):
        cur_tok = int(l.split("tokens=")[1].split()[0])
        tables.append((cur_tok, []))
    elif cur_tok is not None and l.startswith("Q36IMP L"):
        tables[-1][1].append(l)
tok, layer_lines = tables[-1]
print("last table tokens:", tok, "| layer lines:", len(layer_lines))

rows = {}
for ln in layer_lines:
    p = ln.split()
    if len(p) < 5:
        continue
    L = int(p[1][1:])
    kind = p[2]
    # 只信任 rmsA/rmsM/rmsH（log 可能被並發輸出污染 rel 欄位）
    m = re.search(r"rmsA=([0-9.]+)", ln)
    m2 = re.search(r"rmsM=([0-9.]+)", ln)
    m3 = re.search(r"rmsH=([0-9.]+)", ln)
    if not (m and m2 and m3):
        continue
    rmsA, rmsM, rmsH = float(m[1]), float(m2[1]), float(m3[1])
    relA = rmsA / rmsH if rmsH > 0 else 0
    relM = rmsM / rmsH if rmsH > 0 else 0
    rows[L] = (kind, rmsA, rmsM, relA, relM, relA + relM)
missing = [i for i in range(40) if i not in rows]
print("parsed:", len(rows), "missing:", missing)

distinct = defaultdict(set)
for ln in open("/tmp/q36_imp_trace.csv"):
    parts = ln.strip().split(",")
    if len(parts) < 5:
        continue
    layer = int(parts[0].split("_")[1].split(".")[0])
    distinct[layer].update(int(x) for x in parts[4].split())

ranked = []
for L in range(40):
    kind, rmsA, rmsM, relA, relM, relT = rows[L]
    ranked.append((relT, L, kind, rmsA, rmsM, len(distinct.get(L, set()))))
ranked.sort()
print(f"\n{'L':>3} {'kind':>3} {'relT':>6} {'rmsA':>6} {'rmsM':>6} {'distinct':>7}")
for relT, L, kind, rmsA, rmsM, d in ranked:
    print(f"{L:>3} {kind:>3} {relT:>6.3f} {rmsA:>6.4f} {rmsM:>6.4f} {d:>7}")

cands = [r for r in ranked if r[2] == "DN"][:10]
print("\n=== 候選 10 層（最小 relT 的 DeltaNet） ===")
print([r[1] for r in cands], "| relT:", round(cands[0][0], 3), "-", round(cands[-1][0], 3))
print("DN:", sum(1 for r in ranked if r[2] == "DN"), "GA:", sum(1 for r in ranked if r[2] == "GA"))

uniform = sorted([r[1] for r in ranked if r[2] == "DN" and r[1] % 3 == 1])
print("\n均勻刪除候選（DN 層 index%3==1）:", uniform)
print("兩者交集:", sorted(set([r[1] for r in cands]) & set(uniform)))

ts = [r[0] for r in ranked]
print("\nrelT 統計: min=%.3f max=%.3f mean=%.3f std=%.3f" % (min(ts), max(ts), statistics.mean(ts), statistics.pstdev(ts)))
