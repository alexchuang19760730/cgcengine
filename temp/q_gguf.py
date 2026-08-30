import sys
sys.path.insert(0, '/Users/alexchuang/Documents/flashkv0516/temp/llama_routeB/llama-src/temp/llama_roadB/llama.cpp-master/gguf-py')
from gguf import GGUFReader

r = GGUFReader('/Users/alexchuang/Documents/flashkv0516/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf')

want = ['attn_k.weight','attn_v.weight','attn_q.weight','attn_output.weight',
        'attn_qkv_gate.weight','ssm_out.weight','attn_qkv.weight',
        'ffn_gate_inp.weight','ffn_gate_shexp.weight']
for t in r.tensors:
    name = t.name
    if 'mtp' in name:
        continue
    for w in want:
        if name.endswith(w):
            print(name, t.shape.tolist(), t.tensor_type.name)
            break

print('--- mtp block (blk.40) ---')
for t in r.tensors:
    if 'blk.40' not in t.name:
        continue
    if any(s in t.name for s in ['attn_q.weight','attn_output','ffn_gate_inp','ffn_up_exps','ffn_down_exps','eh_proj','attn_qkv']):
        print(t.name, t.shape.tolist(), t.tensor_type.name)
