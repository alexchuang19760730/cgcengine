import re, collections, sys

path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rt.err'
lines = open(path).read().splitlines()
per_layer = collections.defaultdict(set)
per_layer_ge174 = collections.defaultdict(set)
freq = collections.defaultdict(collections.Counter)
nsteps = 0
for ln in lines:
    m = re.search(r'CGC-ROUTE step il=(\d+) ntok=\d+ ids\[0\.\.7\]=(.+)', ln)
    if not m:
        continue
    il = int(m.group(1))
    ids = [int(x) for x in m.group(2).split()[:8]]
    per_layer[il].update(ids)
    for e in ids:
        if e >= 174:
            per_layer_ge174[il].add(e)
        freq[il][e] += 1

n_layers = len(per_layer)
nsteps = max((freq[il].total() // 8 for il in range(n_layers)), default=1)
uni_sizes = sorted((len(v) for v in per_layer.values()))
print(f"steps={nsteps} layers={n_layers}")
print(f"每層 union 大小: min={uni_sizes[0]} median={uni_sizes[len(uni_sizes)//2]} max={uni_sizes[-1]}")
print(f"每層 union >= 87(slot數) 的層數: {sum(1 for s in uni_sizes if s >= 87)}/{n_layers}")

miss_sizes = sorted(len(v) for v in per_layer_ge174.values())
print(f"每層 miss(>=174) 唯一數: min={miss_sizes[0]} median={miss_sizes[len(miss_sizes)//2]} max={miss_sizes[-1]}")

# working set: experts appearing in >= 80% of steps per layer
ws = sorted(sum(1 for _, c in freq[il].items() if c >= int(nsteps * 0.8)) for il in range(n_layers))
print(f"80%工作集(專家出現在>=80% steps): min={ws[0]} median={ws[len(ws)//2]} max={ws[-1]}")

# how many of the current 87-slot pool (0..86) are actually used? (pool is identity 0..n_slots-1)
used_low = sum(1 for il in range(n_layers) if any(e < 87 for e in per_layer[il]))
print(f"使用到 0..86 低槽位範圍的層數: {used_low}/{n_layers}")

# Would a profile pinning the union (capped at 87) eliminate all misses?
need = sum(1 for il in range(n_layers) if len(per_layer[il]) <= 87)
print(f"union<=87 (可完全預填) 的層數: {need}/{n_layers}")
