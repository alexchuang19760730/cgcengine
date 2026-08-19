#!/bin/bash
# r3q4la_e4 -> r3q4la_ga_e4 (GatedAttn q/k/v/o 4-bit, same as r4 §13.102)
set -e
cd /Users/alexchuang/Documents/flashkv0516
PARENT=qwen36-r3q4la_e4
NAME=r3q4la_ga_e4
SRC=prime-agent-worktrees/$PARENT.gturbo
DST=prime-agent-worktrees/qwen36-$NAME.gturbo
TMPBIN=/tmp/r3q4la_ga_e4_weights.bin

echo "=== [1/4] verify packed ==="
./.venv-cgc/bin/python3 -c "
import sys; sys.path.insert(0, 'prime-agent-worktrees/qwen36-repack')
from pathlib import Path
from resident_writer import read_resident_bin
e = read_resident_bin(Path('$TMPBIN'))['entries']
packed = {k: v for k, v in e.items() if v.get('dtype') == 0}
bad = {k: v for k, v in packed.items() if v.get('scaleSize', 0) == 0}
assert not bad, f'{len(bad)} broken packed tensors'
ga = [k for k in e if 'self_attn' in k and k.endswith('_proj.weight') and e[k]['dtype'] == 0]
print(f'  packed={len(packed)} broken=0, GA q4={len(ga)}')
"

echo "=== [2/4] assemble ==="
rm -rf "$DST"
mkdir -p "$DST/packed_experts"
cp -R "$SRC/tokenizer" "$DST/tokenizer"
cp -R "$SRC/profiles" "$DST/profiles" 2>/dev/null || true
for f in "$SRC"/packed_experts/*; do
  ln -f "$f" "$DST/packed_experts/$(basename "$f")"
done
cp "$SRC/manifest.json" "$DST/manifest.json"
cp "$SRC/config.json" "$DST/config.json" 2>/dev/null || true
mv "$TMPBIN" "$DST/model_weights.bin"

echo "=== [3/4] manifest sha + receipt ==="
./.venv-cgc/bin/python3 - << PYEOF
import hashlib, json
from pathlib import Path
base = Path("prime-agent-worktrees")
src_dir = base / "$PARENT.gturbo"
d = base / "qwen36-$NAME.gturbo"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

mf = json.load(open(d / "manifest.json"))
size = (d / "model_weights.bin").stat().st_size
sha = sha256_file(d / "model_weights.bin")
files = mf.get("files")
if isinstance(files, dict):
    files["model_weights.bin"] = {"size": size, "sha256": sha}
elif isinstance(files, list):
    for f in files:
        if f.get("path", "").endswith("model_weights.bin"):
            f["size"] = size; f["sha256"] = sha
json.dump(mf, open(d / "manifest.json", "w"), indent=2)

receipt = json.loads(json.dumps(json.load(open(src_dir / "verified-install.json"))))
receipt["modelDirectoryPath"] = str(d.resolve())
receipt["files"]["manifest.json"] = {"size": (d / "manifest.json").stat().st_size,
                                     "sha256": sha256_file(d / "manifest.json")}
receipt["files"]["model_weights.bin"] = {"size": size, "sha256": sha}
receipt["manifestSha256"] = sha256_file(d / "manifest.json")
json.dump(receipt, open(d / "verified-install.json", "w"), indent=2)
print(f"  manifest+receipt updated (model_weights {size/1e6:.0f}MB)")
PYEOF

echo "=== [4/4] verify ==="
ls "$DST/tokenizer" >/dev/null && echo "  tokenizer OK"
du -h "$DST/model_weights.bin" | awk '{print "  resident: " $1}'
ls "$DST/packed_experts" | wc -l | awk '{print "  experts: " $1}'
df -h / | tail -1
