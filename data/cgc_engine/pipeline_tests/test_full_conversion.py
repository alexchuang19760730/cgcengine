import sys

from cgc_engine.model_parsers import GGUFParser, parsed_model_to_pytorch
import torch

print('=== GGUF → PyTorch 转换测试 ===')
print()

# 1. 解析模型
print('1️⃣ 解析 GGUF 模型...')
parser = GGUFParser('/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf')
parsed_model = parser.parse_model()

print(f'   ✅ 模型类型: {parsed_model.model_type}')
print(f'   ✅ 词表大小: {parsed_model.vocab_size}')
print(f'   ✅ 隐藏层维度: {parsed_model.hidden_dim}')
print(f'   ✅ 层数: {parsed_model.num_layers}')
print(f'   ✅ 注意力头数: {parsed_model.num_heads}')
print(f'   ✅ KV头数: {parsed_model.metadata.get("num_kv_heads")}')
print(f'   ✅ 头维度: {parsed_model.head_dim}')

# 2. 加载权重
print('\n2️⃣ 加载权重...')
weights = parser.load_weights()
print(f'   ✅ 加载了 {len(weights)} 个权重')

# 显示一些权重信息
print('\n   部分权重信息:')
for w in weights[:5]:
    print(f'      {w.name}: shape={w.shape}')

# 3. 转换为 PyTorch 模型
print('\n3️⃣ 转换为 PyTorch 模型...')
try:
    model = parsed_model_to_pytorch(parsed_model, weights)
    print(f'   ✅ 成功创建 PyTorch 模型')
    print(f'   ✅ 模型类型: {type(model).__name__}')
    
    # 检查模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f'   ✅ 总参数: {total_params / 1e9:.2f}B')
    
    # 测试前向传播
    print('\n4️⃣ 测试前向传播...')
    input_ids = torch.randint(0, parsed_model.vocab_size, (1, 32))
    with torch.no_grad():
        output = model(input_ids)
    print(f'   ✅ 输出形状: {output.shape}')
    print(f'   ✅ 输出范围: [{output.min().item():.2f}, {output.max().item():.2f}]')
    
except Exception as e:
    print(f'   ❌ 转换失败: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
