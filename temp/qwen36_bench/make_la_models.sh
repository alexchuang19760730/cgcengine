#!/bin/bash
# make_la_models.sh — 一鍵產出 linear_attn q4 模型變體（§13.101 教訓全內建）
#
# 用法:
#   make_la_models.sh <parent_dir> <variant_name> [apply flags]
#     parent_dir   = 父模型 .gturbo 目錄 (linear_attn 為 fp16, 例 qwen36-r3q4.gturbo)
#     variant_name = 產出名 (例 r3q4la_outonly → qwen36-r3q4la_outonly.gturbo)
#     apply flags  = 透傳給 apply_linear_attn_q4.py (例 --keep-qkv --keep-z)
#
# 內建 §13.101 兩個踩坑防護:
#   1. tokenizer/profiles 在 resident bin 之前複製 → set -e 中止也不會漏
#   2. resident bin 用 mv 非 cp → 不會雙份佔磁碟 (ENOSPC)
#   3. receipt 從父模型深拷貝 (packed_experts 條目不變) + 更新 model_weights/manifest
#   4. 組裝後自動驗證: 0 broken metadata / tokenizer 齊全 / experts 數量一致
set -euo pipefail
cd /Users/alexchuang/Documents/flashkv0516

PARENT="${1:?usage: make_la_models.sh <parent_dir> <variant_name> [apply flags]}"
NAME="${2:?usage: make_la_models.sh <parent_dir> <variant_name> [apply flags]}"
shift 2
FLAGS=("$@")

BASE=prime-agent-worktrees
SRC=$BASE/$PARENT
DST=$BASE/qwen36-${NAME}.gturbo
TMPBIN=/tmp/${NAME}.bin
REPACK=prime-agent-worktrees/qwen36-repack/apply_linear_attn_q4.py

[ -f "$SRC/model_weights.bin" ] || { echo "ERROR: $SRC 不存在"; exit 1; }

echo "=== [1/5] apply: $PARENT -> $TMPBIN (flags: ${FLAGS[*]:-all-q4}) ==="
./.venv-cgc/bin/python3 "$REPACK" "$SRC/model_weights.bin" "$TMPBIN" "${FLAGS[@]}" 2>&1 | tail -1

echo "=== [2/5] metadata 驗證 (0 broken) ==="
./.venv-cgc/bin/python3 -c "
import sys; sys.path.insert(0, 'prime-agent-worktrees/qwen36-repack')
from pathlib import Path
from resident_writer import read_resident_bin
e = read_resident_bin(Path('$TMPBIN'))['entries']
packed = {k: v for k, v in e.items() if v.get('dtype') == 0}
bad = {k: v for k, v in packed.items() if v.get('scaleSize', 0) == 0}
assert not bad, f'{len(bad)} broken packed tensors'
print(f'  packed={len(packed)} broken=0 OK')
"

echo "=== [3/5] assemble $DST (tokenizer 先行 + experts hardlink + mv) ==="
rm -rf "$DST"
mkdir -p "$DST/packed_experts"
cp -R "$SRC/tokenizer" "$DST/tokenizer"                      # ① tokenizer 最先
cp -R "$SRC/profiles" "$DST/profiles" 2>/dev/null || true    # ② profiles
for f in "$SRC"/packed_experts/*; do                          # ③ experts hardlink
  ln -f "$f" "$DST/packed_experts/$(basename "$f")"
done
cp "$SRC/manifest.json" "$DST/manifest.json"                  # ④ manifest
mv "$TMPBIN" "$DST/model_weights.bin"                         # ⑤ mv 不複製

echo "=== [4/5] manifest sha + receipt ==="
./.venv-cgc/bin/python3 - << PYEOF
import hashlib, json
from pathlib import Path
base = Path("prime-agent-worktrees")
src_dir = base / "$PARENT"
d = base / "qwen36-${NAME}.gturbo"

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

# receipt 從父模型深拷貝: packed_experts 條目不變 (hardlink), 只換 manifest/model_weights
receipt = json.loads(json.dumps(json.load(open(src_dir / "verified-install.json"))))
receipt["modelDirectoryPath"] = str(d.resolve())
receipt["files"]["manifest.json"] = {"size": (d / "manifest.json").stat().st_size,
                                     "sha256": sha256_file(d / "manifest.json")}
receipt["files"]["model_weights.bin"] = {"size": size, "sha256": sha}
receipt["manifestSha256"] = sha256_file(d / "manifest.json")
json.dump(receipt, open(d / "verified-install.json", "w"), indent=2)
print(f"  resident {size/1e6:.0f}MB, receipt {len(receipt['files'])} entries")
PYEOF

echo "=== [5/5] 組裝後驗證 ==="
N_TOK=$(ls "$DST/tokenizer" | wc -l | tr -d ' ')
N_SRC=$(ls "$SRC/tokenizer" | wc -l | tr -d ' ')
N_EXP=$(ls "$DST/packed_experts" | wc -l | tr -d ' ')
N_EXPSRC=$(ls "$SRC/packed_experts" | wc -l | tr -d ' ')
[ "$N_TOK" = "$N_SRC" ] && [ "$N_TOK" -gt 0 ] || { echo "ERROR: tokenizer 不齊 ($N_TOK vs $N_SRC)"; exit 1; }
[ "$N_EXP" = "$N_EXPSRC" ] || { echo "ERROR: experts 數量不符 ($N_EXP vs $N_EXPSRC)"; exit 1; }
[ -f "$DST/model_weights.bin" ] || { echo "ERROR: model_weights.bin 缺失"; exit 1; }
python3 -c "import json; json.load(open('$DST/verified-install.json'))" || { echo "ERROR: receipt 不是合法 JSON"; exit 1; }
echo "  tokenizer=$N_TOK OK, experts=$N_EXP OK, resident=$(stat -f%z "$DST/model_weights.bin" | awk '{printf "%.0fMB", $1/1e6}') OK, receipt OK"
echo "=== $NAME done ==="
