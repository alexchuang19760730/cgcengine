#!/usr/bin/env python3
# Generate the llama-quantize --tensor-type-file for the qwen36 denseIQ4X + headIQ2 carrier
# (Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf -> ...-denseIQ4X-headIQ2.gguf, doc §⑮ 29.85 t/s).
#
# Policy (2026-08-28, replicating the 2026-08-26 carrier):
#   - 250 dense Q6_K tensors (attn_gate/attn_qkv/ssm_out/attn_k/q/v/output + ffn_*_shexp,
#     excluding blk.39's 7 Q8_0 full-attn tensors and token_embd.weight) -> IQ4_XS
#   - output.weight (MTP head, Q6_K) -> IQ2_S (default) or kept Q6_K (--keep-head)
#   - every OTHER tensor is pinned to its CURRENT type (anchored ^name$ regex) so the
#     ftype-default mixture logic can never touch them -> byte-copy, bit-identical by
#     construction (753 tensors total, 251 requantized / 250 with --keep-head, rest copied).
#
# --keep-head (2026-08-29): keep output.weight at Q6_K. Motivation: the headIQ2 carrier
# showed a DETERMINISTIC degenerate tail (seed 1: last ~140 of 1101 tokens -> 0000) while
# the first 86.5% stayed readable; head IQ2_S is the prime suspect (shared target+draft
# lm_head). Costs ~+1GB size / head matmul ~2.5x bytes — expected <2% t/s.
#
# Usage: python3 scripts/gen_denseiq4x_tt.py [--keep-head] [model.gguf] [out.tt]
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src/llama.cpp/gguf-py'))
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

keep_head = '--keep-head' in sys.argv
argv = [a for a in sys.argv[1:] if a != '--keep-head']
model = argv[0] if argv else 'models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf'
out = argv[1] if len(argv) > 1 else 'scripts/qwen36_denseiq4x.tt'

r = GGUFReader(model)
lines = []
n_dense = 0
for t in r.tensors:
    cur = GGMLQuantizationType(int(t.tensor_type)).name
    name = t.name.decode() if isinstance(t.name, bytes) else t.name
    is_expert = ('ffn_down_exps' in name or 'ffn_gate_exps' in name or 'ffn_up_exps' in name)
    if int(t.tensor_type) == GGMLQuantizationType.Q6_K and name != 'token_embd.weight' and not is_expert:
        if name == 'output.weight':
            tgt = 'Q6_K' if keep_head else 'IQ2_S'
            if not keep_head:
                n_dense += 1
        else:
            tgt = 'IQ4_XS'
            n_dense += 1
    else:
        tgt = cur
    lines.append('^' + name.replace('.', r'\.') + '$=' + tgt)

with open(out, 'w') as f:
    f.write('\n'.join(lines) + '\n')
head_note = 'Q6_K kept' if keep_head else 'IQ2_S'
print(f'{out}: {len(lines)} entries, {n_dense} requantized (dense IQ4_XS; output.weight {head_note})')
