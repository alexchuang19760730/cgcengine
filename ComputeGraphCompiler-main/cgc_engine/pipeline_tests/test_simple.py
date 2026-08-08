import torch
import time

print('=' * 60)
print('MagiCompiler Phase 1 测试')
print('=' * 60)

# 测试 1: CUDA Graph 基础功能
print('\n[测试 1] CUDA Graph 基础功能')

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 512).cuda()
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.linear(x))

model = SimpleModel()
input_tensor = torch.randn(1, 128, 512).cuda()

# Eager 模式
with torch.no_grad():
    for _ in range(3):
        model(input_tensor)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        model(input_tensor)
torch.cuda.synchronize()
eager_time = (time.time() - start) * 1000 / 100
print(f'Eager: {eager_time:.2f} ms/iter')

# CUDA Graph 模式
graph = torch.cuda.CUDAGraph()
output_placeholder = torch.empty_like(model(input_tensor))
with torch.cuda.graph(graph):
    output_placeholder.copy_(model(input_tensor))

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    graph.replay()
torch.cuda.synchronize()
graph_time = (time.time() - start) * 1000 / 100
print(f'Graph: {graph_time:.2f} ms/iter | Speedup: {eager_time/graph_time:.2f}x')

# 测试 2: torch.compile
print('\n[测试 2] torch.compile 功能')
compiled_model = torch.compile(model, mode='reduce-overhead', fullgraph=True)

with torch.no_grad():
    for _ in range(3):
        compiled_model(input_tensor)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        compiled_model(input_tensor)
torch.cuda.synchronize()
compile_time = (time.time() - start) * 1000 / 100
print(f'Compile: {compile_time:.2f} ms/iter')

print('\n' + '=' * 60)
print('测试完成!')
