import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """单头 causal self-attention 计算。
    原理：Q K^T / sqrt(dk) 计算相似度，mask 屏蔽未来 token，softmax 得权重，权重 * V 聚合信息。
    GPT-3 相同：用于 decoder-only，确保自回归（不看未来）。"""
    dk = q.size(-1)  # dk = head_dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
    
    if mask is not None:
        scores = scores + mask  # mask: -inf for future positions
    
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output

# 测试入口（小维度模拟）
if __name__ == "__main__":
    B, N, dk = 2, 4, 8  # batch=2, seq=4, dim=8
    q = torch.randn(B, N, dk)
    k = torch.randn(B, N, dk)
    v = torch.randn(B, N, dk)
    
    # Causal mask: 上三角 -inf (不含对角线)
    mask = torch.triu(torch.ones(N, N) * float('-inf'), diagonal=1)
    mask = mask.unsqueeze(0).expand(B, -1, -1)  # 广播到 batch
    
    output = causal_self_attention(q, k, v, mask)
    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output[0,0,:]: {output[0,0,:]}")  # 第一位置输出（只 attend 自己）
    
    assert output.shape == (B, N, dk), "Shape mismatch!"
    print("Test passed.")
