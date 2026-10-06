#!/usr/bin/env bash
# spec-tree alt_branch 路径验证（用户选项 b2）
# 复现 exp-spectree-mtpon-default-2026-10-06 的 ON/OFF 臂配对，确认 alt_branch 绕开 M-RoPE X<Y 墙
# 但 16GB 盒下 net 回歸（brief 已证 ON 6.15 vs OFF 10.38 = -40.8%；RGATE 打平 0.97x）。
#
# 步骤 A（配置级，已证）：ON 臂加 CGC_SPEC_TREE_RGATE=3;CGC_SPEC_TREE_RPROBE=40
#   => 实跑 e_rgate3p40.json = decode 11.0 t/s / 10.48±1.82（charter h_orcrgate40），
#      对照 OFF ~10.82 → 0.97× 贴线打平（无需改码）。本脚本 ON 臂已默认带此配置。
#
# 入口：scripts/check/harness.py bench（prod-new 基底，harness 会强制武装 swap 三支柱，
#        故 swap 混淆已被控制——这正是与 prod25 路径不同的地方）。
# 前置：binary 含 common/speculative.cpp 的 alt_branch + accept_from（nm 已确认）。
#
# 用法：bash scripts/check/run_spectree_verify.sh
set -u
cd "$(dirname "$0")/../.." || exit 1

HARNESS=scripts/check/harness.py
CHARTER=scripts/check/charters/exp-spectree-mtpon-default-2026-10-06.yaml
OUT=/tmp/spectree_verify

mkdir -p "$OUT"

# ON 臂：真 spec-tree MTP（alt_branch 走 seq1 绕 M-RoPE X<Y）+ caps>=17 + n_max=2 + MTP_PERF
ON_ARM='prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:140;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;LLAMA_BENCH_SPEC=1;LLAMA_BENCH_SPEC_DRAFT_N_MAX=2;CGC_SPEC_TREE=1;CGC_SPEC_TREE_B=2;CGC_SPEC_TREE_RS=2;CGC_MTP_PERF=1;CGC_SPEC_TREE_RGATE=3;CGC_SPEC_TREE_RPROBE=40'
# OFF 臂：plain decode、sampling(默认0.4)、seed 钉 1
OFF_ARM='prod-new:CGC_SERVER_SEED=1;CGC_SERVER_MTP=0'

echo "=== [ON] spec-tree MTP-on (alt_branch) ==="
python3 "$HARNESS" bench --arm "$ON_ARM" --charter "$CHARTER" --json "$OUT/on.json" 2>&1 | tail -40
echo "ON json -> $OUT/on.json"

echo
echo "=== [OFF] plain MTP-off (control) ==="
python3 "$HARNESS" bench --arm "$OFF_ARM" --charter "$CHARTER" --json "$OUT/off.json" 2>&1 | tail -40
echo "OFF json -> $OUT/off.json"

echo
echo "=== summary (tg decode t/s) ==="
for f in on off; do
  python3 - "$OUT/$f.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
# harness bench 产物可能是单臂 dict 或 list；尽量取 decode tg
def dig(x):
    if isinstance(x,dict):
        for k in ('decode_tps_median','tg','tps','decode_tps'):
            if k in x: return x[k]
        if 'rows' in x:
            for r in x['rows']:
                if 'decode_tps_median' in r: return r['decode_tps_median']
    if isinstance(x,list):
        for it in x:
            v=dig(it)
            if v is not None: return v
    return None
print(sys.argv[1], '->', dig(d))
PY
done
