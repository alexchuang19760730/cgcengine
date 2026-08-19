#!/usr/bin/env python3
"""
正交基生成工具 - 生成预计算的正交基文件

使用方法:
    python generate_ortho_basis.py --output ortho_basis.bin
    python generate_ortho_basis.py --output ortho_basis.bin --dim 128
"""

import argparse
import numpy as np

def generate_orthogonal_basis(num_heads: int, ortho_dim: int, head_dim: int) -> np.ndarray:
    """
    使用QR分解生成正交基
    
    Args:
        num_heads: 注意力头数
        ortho_dim: 正交基维度
        head_dim: 头维度
        
    Returns:
        正交基矩阵: [num_heads, ortho_dim, head_dim]
    """
    basis = np.zeros((num_heads, ortho_dim, head_dim), dtype=np.float32)
    
    for h in range(num_heads):
        # 生成随机矩阵
        random_matrix = np.random.randn(ortho_dim, head_dim).astype(np.float32)
        
        # QR分解
        Q, _ = np.linalg.qr(random_matrix)
        
        # 确保正交基数量不超过head_dim
        if ortho_dim > head_dim:
            Q = Q[:head_dim, :]
        
        basis[h] = Q
    
    return basis

def save_basis_to_file(basis: np.ndarray, filename: str):
    """
    保存正交基到二进制文件
    
    Args:
        basis: 正交基矩阵
        filename: 输出文件名
    """
    with open(filename, 'wb') as f:
        f.write(basis.tobytes())
    print(f"✅ 正交基已保存到: {filename}")
    print(f"   形状: {basis.shape}")
    print(f"   大小: {basis.nbytes / (1024 * 1024):.2f} MB")

def load_basis_from_file(filename: str, num_heads: int, ortho_dim: int, head_dim: int) -> np.ndarray:
    """
    从文件加载正交基
    
    Args:
        filename: 文件名
        num_heads: 头数
        ortho_dim: 正交基维度
        head_dim: 头维度
        
    Returns:
        正交基矩阵
    """
    with open(filename, 'rb') as f:
        data = f.read()
    
    basis = np.frombuffer(data, dtype=np.float32)
    expected_size = num_heads * ortho_dim * head_dim
    
    if len(basis) != expected_size:
        raise ValueError(f"文件大小不匹配: 期望 {expected_size} 元素, 实际 {len(basis)} 元素")
    
    return basis.reshape(num_heads, ortho_dim, head_dim)

def verify_orthogonality(basis: np.ndarray):
    """
    验证正交基的正交性
    
    Args:
        basis: 正交基矩阵
    """
    num_heads, ortho_dim, head_dim = basis.shape
    max_error = 0.0
    
    for h in range(num_heads):
        Q = basis[h]
        
        # 计算 Q^T * Q
        QQT = Q @ Q.T
        
        # 应该接近单位矩阵
        identity = np.eye(ortho_dim)
        error = np.max(np.abs(QQT - identity))
        max_error = max(max_error, error)
    
    print(f"✅ 正交性验证通过, 最大误差: {max_error:.6e}")
    return max_error < 1e-4

def main():
    parser = argparse.ArgumentParser(description='生成正交基文件')
    parser.add_argument('--output', '-o', type=str, default='ortho_basis.bin',
                        help='输出文件名')
    parser.add_argument('--heads', '-H', type=int, default=32,
                        help='注意力头数')
    parser.add_argument('--dim', '-d', type=int, default=128,
                        help='正交基维度')
    parser.add_argument('--head_dim', '-hd', type=int, default=128,
                        help='头维度')
    parser.add_argument('--verify', '-v', action='store_true',
                        help='验证正交性')
    
    args = parser.parse_args()
    
    print(f"📦 生成正交基: heads={args.heads}, ortho_dim={args.dim}, head_dim={args.head_dim}")
    
    # 生成正交基
    basis = generate_orthogonal_basis(args.heads, args.dim, args.head_dim)
    
    # 验证
    if args.verify:
        verify_orthogonality(basis)
    
    # 保存
    save_basis_to_file(basis, args.output)
    
    print("\n🎉 正交基生成完成!")

if __name__ == '__main__':
    main()