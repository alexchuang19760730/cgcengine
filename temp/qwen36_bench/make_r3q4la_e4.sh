#!/bin/bash
# §13.112: r3q4_e4 → r3q4la_e4（linear_attn qkv/z/out 4-bit，r3 家族對齊 r4）
# 目標: resident 3254 → ~2545MB，驗證 §13.110 mmap 縮小效應在 r3 轉正
set -euo pipefail
cd /Users/alexchuang/Documents/flashkv0516
PARENT=qwen36-r3q4_e4
NAME=r3q4la_e4
SRC=prime-agent-worktrees/$PARENT.gturbo
DST=prime-agent-worktrees/qwen36-$NAME.gturbo
TMPBIN=/tmp/r3q4la_e4.bin
PY=.venv-cgc/bin/python3

echo "=== [1/4] apply linear_attn q4 ==="
"$PY" prime-agent-worktrees/qwen36-repack/apply_linear_attn_q4.py \
  "$SRC/model_weights.bin" "$TMPBIN" 2>&1 | tail -4

echo "=== [2/4] verify metadata（0 broken packed）==="
"$PY" - << PYEOF
import sys; sys.path.insert(0, 'prime-agent-worktrees/qwen36-repack')
from pathlib import Path
from resident_writer import read_resident_bin
e = read_resident_bin(Path("$TMPBIN"))['entries']
packed = {k: v for k, v in e.items() if v.get('dtype') == 0}
bad = {k: v for k, v in packed.items() if v.get('scaleSize', 0) == 0}
la = [k for k in e if 'linear_attn' in k and k.endswith('.weight') and e[k]['dtype'] == 0]
assert not bad, f'{len(bad)} broken packed tensors'
assert len(la) == 90, f'linear_attn q4 = {len(la)} (expect 90)'
print(f'  packed={len(packed)} broken=0 linear_attn_q4={len(la)} OK')
PYEOF

echo "=== [3/4] assemble（tokenizer 先行 + experts hardlink + mv）==="
rm -rf "$DST"
mkdir -p "$DST/packed_experts"
cp -R "$SRC/tokenizer" "$DST/tokenizer"
cp -R "$SRC/profiles" "$DST/profiles"
for f in "$SRC"/packed_experts/*; do
  ln -f "$f" "$DST/packed_experts/$(basename "$f")"
done
cp "$SRC/manifest.json" "$DST/manifest.json"
cp "$SRC/config.json" "$DST/config.json" 2>/dev/null || true
mv "$TMPBIN" "$DST/model_weights.bin"

echo "=== [4/4] manifest sha + receipt ==="
"$PY" - << PYEOF
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

echo "=== verify ==="
ls "$DST/tokenizer" >/dev/null && echo "  tokenizer OK"
ls -la "$DST/model_weights.bin" | awk '{print "  resident: " $5/1e6 " MB"}'
ls "$DST/packed_experts" | wc -l | awk '{print "  experts hardlinked: " $1}'
df -h /System/Volumes/Data | tail -1
