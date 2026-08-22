import re, collections, sys

src = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rt.err'
dst = sys.argv[2] if len(sys.argv) > 2 else '/Users/alexchuang/Documents/flashkv0516/profiles/short_union.pin'
import os
os.makedirs(os.path.dirname(dst), exist_ok=True)

per_layer = collections.defaultdict(set)
for ln in open(src).read().splitlines():
    m = re.search(r'CGC-ROUTE step il=(\d+) ntok=\d+ ids\[0\.\.7\]=(.+)', ln)
    if not m:
        continue
    il = int(m.group(1))
    for x in m.group(2).split()[:8]:
        per_layer[il].add(int(x))

n_layers = max(per_layer) + 1
with open(dst, 'w') as f:
    for il in range(n_layers):
        ids = sorted(per_layer.get(il, set()))
        f.write(' '.join(str(e) for e in ids) + '\n')
print(f"PIN_PROFILE written: {dst} ({n_layers} layers)")
