#!/usr/bin/env python3
# Generate the llama-quantize --tensor-type-file for the qwen36 denseIQ4X + headIQ2 carrier
# (Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf -> ...-denseIQ4X-headIQ2.gguf, doc §⑮ 29.85 t/s).
#
# Policy (2026-08-28, replicating the 2026-08-26 carrier):
#   - 250 dense Q6_K tensors (attn_gate/attn_qkv/ssm_out/attn_k/q/v/output + ffn_*_shexp,
#     excluding blk.39's 7 Q8_0 full-attn tensors and token_embd.weight) -> IQ4_XS
#   - output.weight (MTP head, Q6_K) -> IQ2_S
#   - every OTHER tensor is pinned to its CURRENT type (anchored ^name$ regex) so the
#     ftype-default mixture logic can never touch them -> byte-copy, bit-identical by
#     construction (753 tensors total, 251 requantized, 502 byte-copied).
#
# Usage: python3 scripts/gen_denseiq4x_tt.py [model.gguf] [out.tt]
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src/llama.cpp/gguf-py'))
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

model = sys.argv[1] if len(sys.argv) > 1 else 'models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf'
out = sys.argv[2] if len(sys.argv) > 2 else 'scripts/qwen36_denseiq4x.tt'

r = GGUFReader(model)
lines = []
n_dense = 0
for t in r.tensors:
    cur = GGMLQuantizationType(int(t.tensor_type)).name
    name = t.name.decode() if isinstance(t.name, bytes) else t.name
    is_expert = ('ffn_down_exps' in name or 'ffn_gate_exps' in name or 'ffn_up_exps' in name)
    if int(t.tensor_type) == GGMLQuantizationType.Q6_K and name != 'token_embd.weight' and not is_expert:
        tgt = 'IQ2_S' if name == 'output.weight' else 'IQ4_XS'
        n_dense += 1
    else:
        tgt = cur
    lines.append('^' + name.replace('.', r'\.') + '$=' + tgt)

with open(out, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'{out}: {len(lines)} entries, {n_dense} requantized (250 dense IQ4_XS + output.weight IQ2_S expected 251)')
