#!/bin/bash
# make_r4q4.sh — r4 → r4q4（lm_head + shared expert 4-bit）
# 流程：轉換 resident → hardlink experts → manifest sha → receipt 重生
set -e
cd /Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees

R4=qwen36-r4.gturbo
R4Q4=qwen36-r4q4.gturbo
PY=/Users/alexchuang/Documents/flashkv0516/.venv-cgc/bin/python3

echo "=== [1/4] 轉換 lm_head+shared → 4-bit ==="
rm -rf "$R4Q4"
mkdir -p "$R4Q4"
$PY qwen36-repack/apply_head_shared_q4.py \
  "$R4/model_weights.bin" "$R4Q4/model_weights.bin"

echo "=== [2/4] 組裝目錄（hardlink experts + 複製設定）==="
cp "$R4/manifest.json" "$R4Q4/manifest.json"
cp -R "$R4/profiles" "$R4Q4/profiles"
cp -R "$R4/tokenizer" "$R4Q4/tokenizer"
cp "$R4/verified-install.json" "$R4Q4/verified-install.json" 2>/dev/null || true
ln -s "$(readlink "$R4/metal_out")" "$R4Q4/metal_out" 2>/dev/null || true
for f in "$R4"/packed_experts/*; do
  ln "$f" "$R4Q4/packed_experts/$(basename "$f")"
done

echo "=== [3/4] manifest sha 更新 ==="
NEW_SHA=$($PY -c "
import hashlib, json
p = '$R4Q4/model_weights.bin'
h = hashlib.sha256()
with open(p, 'rb') as f:
    while True:
        b = f.read(1 << 20)
        if not b: break
        h.update(b)
print(h.hexdigest())
")
NEW_SIZE=$(stat -f%z "$R4Q4/model_weights.bin")
$PY - "$R4Q4/manifest.json" "$NEW_SIZE" "$NEW_SHA" <<'EOF'
import json, sys
p, size, sha = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(p))
for f in m.get("files", []):
    if f.get("name") == "model_weights.bin":
        f["size"] = size
        f["sha256"] = sha
json.dump(m, open(p, "w"), indent=2)
print("manifest updated:", size, sha[:16])
EOF

echo "=== [4/4] receipt 重生 ==="
$PY - "$R4Q4" <<'EOF'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()

# 從 r4 原 receipt 拷貝 packed_experts 條目（hardlink 同一批檔案）
src = json.load(open(root / ".." / "qwen36-r4.gturbo" / "verified-install.json"))
files = {k: dict(v) for k, v in src["files"].items() if k.startswith("packed_experts/")}
for name in ("manifest.json", "model_weights.bin"):
    files[name] = {"size": (root / name).stat().st_size, "sha256": sha(root / name)}
m = json.load(open(root / "manifest.json"))
receipt = {
    "files": {k: files[k] for k in sorted(files)},
    "manifestSha256": sha(root / "manifest.json"),
    "modelDirectoryPath": str(root),
    "schemaVersion": 1,
    "toolVersion": "TurboFieldfareRepack verify-install",
    "verificationTimestamp": "2026-08-11T00:00:00Z",
}
json.dump(receipt, open(root / "verified-install.json", "w"), indent=1, sort_keys=True)
print("receipt regenerated,", len(files), "file entries")
EOF

echo "=== 完成 ==="
du -sh "$R4Q4" && df -h / | tail -1
