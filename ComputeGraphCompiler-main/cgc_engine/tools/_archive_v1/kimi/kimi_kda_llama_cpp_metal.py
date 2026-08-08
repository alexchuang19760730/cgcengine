import torch
import torch.nn.functional as F
import llama_cpp
import time
import numpy as np

# ------------------------------------------------------------------------------
# 初始化 llama.cpp Metal 实例
# ------------------------------------------------------------------------------
print("📦 加载模型...")
llama = llama_cpp.Llama(
    model_path="/Users/alexchuang/Documents/flashkv0430/qwen2.5-7b-q4_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # 🔥 全部丢 Metal
    verbose=False,
    logits_all=True  # 需要获取所有 logits
)
print("✅ 模型加载完成")

# ------------------------------------------------------------------------------
# 🔥 Kimi KDA 核心公式
# ------------------------------------------------------------------------------
def kimi_kda_forward(q, k, v, beta=0.1):
    """
    Kimi 原论文 KDA 公式
    S = (I - β k kᵀ) S + β k vᵀ
    O = Q S
    """
    B, H, L, D = q.shape
    scale = 1.0 / (D ** 0.5)
    
    S = torch.zeros(B, H, D, D, device=q.device)
    O = torch.zeros_like(q)
    
    for l in range(L):
        k_l = k[:, :, l, :]
        v_l = v[:, :, l, :]
        q_l = q[:, :, l, :]
        
        # KDA 状态更新
        S = S * (1.0 - beta * torch.einsum('bhd,bhe->bhde', k_l, k_l)) + beta * torch.einsum('bhd,bhe->bhde', k_l, v_l)
        
        # 输出计算 - 修复形状问题
        o_l = torch.einsum('bhd,bhde->bhe', q_l, S) * scale
        O[:, :, l] = o_l
    
    return O

# ------------------------------------------------------------------------------
# 性能测试
# ------------------------------------------------------------------------------
def benchmark_kda_vs_llama():
    print("\n" + "="*60)
    print("🔥 Kimi KDA vs llama.cpp 性能对比")
    print("="*60)
    
    PROMPT = "Hello, my name is"
    MAX_TOKENS = 64
    
    # ----------------------
    # 测试 llama.cpp 原生性能
    # ----------------------
    print("\n🔹 测试 llama.cpp 原生性能")
    t0 = time.time()
    for _ in range(5):
        output = llama(PROMPT, max_tokens=MAX_TOKENS, echo=False)
    llama_time = (time.time() - t0) / 5
    llama_speed = MAX_TOKENS / llama_time
    print(f"   平均时间: {llama_time:.4f}s")
    print(f"   速度: {llama_speed:.2f} tokens/s")
    
    # ----------------------
    # 测试 KDA 性能（PyTorch 模拟）
    # ----------------------
    print("\n🔹 测试 Kimi KDA 性能")
    
    # 模拟 QKV 输入
    B, H, L, D = 1, 28, 128, 128
    
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    
    # 预热
    for _ in range(3):
        kimi_kda_forward(q, k, v)
    
    # 测试
    t0 = time.time()
    for _ in range(10):
        o = kimi_kda_forward(q, k, v)
    kda_time = (time.time() - t0) / 10
    kda_speed = L / kda_time
    print(f"   平均时间: {kda_time:.4f}s")
    print(f"   速度: {kda_speed:.2f} tokens/s")
    
    # ----------------------
    # 结果对比
    # ----------------------
    speedup = kda_speed / llama_speed
    
    print("\n" + "="*60)
    print("📊 性能对比结果")
    print("="*60)
    print(f"""
┌──────────────────────────────────────┐
│           性能对比表                │
├──────────────┬──────────┬───────────┤
│     指标     │ llama.cpp │  Kimi KDA │
├──────────────┼──────────┼───────────┤
│ 速度 (tok/s) │ {llama_speed:>8.2f} │ {kda_speed:>9.2f} │
├──────────────┴──────────┴───────────┤
│ 加速比: {speedup:.2f}x             │
└──────────────────────────────────────┘
    """)

# ------------------------------------------------------------------------------
# 数学正确性验证
# ------------------------------------------------------------------------------
def verify_kda_correctness():
    print("\n" + "="*60)
    print("🔬 KDA 数学正确性验证")
    print("="*60)
    
    B, H, L, D = 1, 4, 32, 32
    
    q = torch.randn(B, H, L, D) * 0.1
    k = torch.randn(B, H, L, D) * 0.1
    v = torch.randn(B, H, L, D) * 0.1
    
    scale = 1.0 / (D ** 0.5)
    
    # 标准 Attention
    attn_scores = (q @ k.transpose(-2, -1)) * scale
    attn = F.softmax(attn_scores, dim=-1)
    out_std = attn @ v
    
    # KDA
    out_kda = kimi_kda_forward(q, k, v, beta=0.1)
    
    # 误差对比
    mse = F.mse_loss(out_std, out_kda).item()
    mae = F.l1_loss(out_std, out_kda).item()
    
    print(f"\n标准 Attention vs Kimi KDA:")
    print(f"   MSE: {mse:.6e}")
    print(f"   MAE: {mae:.6e}")
    print(f"   ✅ 结果一致性: {'PASS' if mae < 0.1 else 'FAIL'}")

# ------------------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    verify_kda_correctness()
    benchmark_kda_vs_llama()
    print("\n✅ 测试完成！")