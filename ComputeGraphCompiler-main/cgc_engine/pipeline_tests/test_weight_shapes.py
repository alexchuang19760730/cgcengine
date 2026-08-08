import sys

from cgc_engine.model_parsers import GGUFParser
import torch

print('=== 权重映射和形状验证 ===')
print()

# 解析模型
parser = GGUFParser('/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf')
parsed_model = parser.parse_model()
weights = parser.load_weights()

# 检查关键权重
print('关键权重形状验证:')
print(f'隐藏层维度: {parsed_model.hidden_dim}')
print(f'头维度: {parsed_model.head_dim}')
print(f'KV头数: {parsed_model.metadata.get("num_kv_heads")}')
print()

# 查找特定权重
def find_weight(name):
    for w in weights:
        if name in w.name:
            return w
    return None

# 检查 token_embd
embd = find_weight('token_embd.weight')
print(f'token_embd.weight: shape={embd.shape}')
print(f'  期望形状: [vocab_size, hidden_dim] = [{parsed_model.vocab_size}, {parsed_model.hidden_dim}]')
print(f'  转置后: {embd.tensor.T.shape}')
print()

# 检查 attn_q
q_proj = find_weight('blk.0.attn_q.weight')
print(f'blk.0.attn_q.weight: shape={q_proj.shape}')
print(f'  期望形状: [num_heads * head_dim, hidden_dim] = [{parsed_model.num_heads * parsed_model.head_dim}, {parsed_model.hidden_dim}]')
print(f'  转置后: {q_proj.tensor.T.shape}')
print()

# 检查 attn_k（GQA）
k_proj = find_weight('blk.0.attn_k.weight')
print(f'blk.0.attn_k.weight: shape={k_proj.shape}')
print(f'  期望形状: [num_kv_heads * head_dim, hidden_dim] = [{parsed_model.metadata.get("num_kv_heads") * parsed_model.head_dim}, {parsed_model.hidden_dim}]')
print(f'  转置后: {k_proj.tensor.T.shape}')
print()

# 检查 output
output = find_weight('output.weight')
print(f'output.weight: shape={output.shape}')
print(f'  期望形状: [vocab_size, hidden_dim] = [{parsed_model.vocab_size}, {parsed_model.hidden_dim}]')
print(f'  转置后: {output.tensor.T.shape}')
