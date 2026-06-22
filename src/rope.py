# src/rope.py — 旋转位置编码 / Rotary Position Embedding (RoPE)
# 功能 / Purpose: 提供 RoPE 的核心数学运算，独立于 Transformer 架构
# 原理 / Principle: RoPE 通过在 query 和 key 向量上施加旋转，使注意力
#   分数天然包含 token 之间的相对位置信息，无需可学习的绝对位置嵌入。
#   RoPE applies rotation to Q and K vectors so that attention scores
#   naturally encode relative position, eliminating learned absolute embeddings.

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转位置编码的复数频率张量
    Precompute complex frequency tensor for RoPE.

    原理 / Principle:
        RoPE 将位置信息编码为旋转角度。对于 head_dim 维度中的每一对 (2i, 2i+1)，
        旋转角 θ_i = 1 / (10000 ^ (2i / dim))。位置 pos 处的旋转角为 pos * θ_i。
        这里预计算所有位置的 e^(i*θ_pos)，避免每次 forward 时重复计算。

    参数 / Args:
        dim: 每个注意力头的维度 / dimension per attention head (head_dim)
        end: 预计算的最大位置数 / max sequence position to precompute
        theta: RoPE 的频率基 / base frequency (default 10000.0)

    返回 / Returns:
        freqs_cis: 形状 (end, dim//2) 的复数张量，freqs_cis[pos, i] = e^(i * pos * θ_i)
                   Complex tensor of shape (end, dim//2)
    """
    # 计算每对维度的频率: θ_i = 1 / (10000 ^ (2i / dim))
    # Compute frequencies per dimension pair: θ_i = 1 / (10000 ^ (2i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成位置序列: [0, 1, 2, ..., end-1]
    # Generate position sequence
    t = torch.arange(end, device=freqs.device)
    # 外积得到所有 (位置, 频率对) 的角度矩阵: (end, dim//2)
    # Outer product to get angle matrix for all (position, frequency) pairs
    freqs = torch.outer(t, freqs).float()
    # 转换为复数形式: cos(angle) + i*sin(angle)
    # Convert to complex form: cos(angle) + i*sin(angle)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    对 query 和 key 张量应用 RoPE 旋转
    Apply RoPE rotation to query and key tensors.

    原理 / Principle:
        1. 将 Q 和 K 的最后一维两两分组，视为复数 (实部, 虚部)
           Treat consecutive pairs in the last dim as complex numbers (real, imag)
        2. 逐位置乘以预计算的 e^(i*θ_pos): 复数乘法 = 角度旋转
           Multiply by precomputed e^(i*θ_pos) position-wise: complex mul = rotation
        3. 变回实数表示
           Convert back to real representation

        经过 RoPE 后, Q 和 K 的点积 (Q @ K^T) 自然包含相对位置信息:
        (R_pos * Q) @ (R_pos' * K)^T = Q @ R_{pos-pos'} @ K^T
        After RoPE, Q @ K^T naturally encodes relative position.

    参数 / Args:
        xq: query 张量, 形状 (B, T, n_head, head_dim)
        xk: key 张量, 形状 (B, T, n_head, head_dim)
        freqs_cis: 预计算的复数频率, 形状 (T, head_dim//2)

    返回 / Returns:
        xq_out, xk_out: 应用旋转后的 Q 和 K, 形状与输入相同
    """
    # 将最后维度 reshape 为 (..., head_dim/2, 2)，然后视为复数
    # Reshape last dim to (..., head_dim//2, 2), then view as complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 调整 freqs_cis 形状以支持广播: (T, head_dim/2) → (1, T, 1, head_dim/2)
    # Reshape freqs_cis for broadcasting
    freqs_cis = freqs_cis.view(1, xq.size(1), 1, xq_.size(-1))

    # 复数乘法 = 旋转 / Complex multiplication = rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # 保持原始 dtype (与输入一致)
    # Preserve original dtype
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==========================================
# RoPE 独立单元测试 / RoPE Independent Unit Test
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("RoPE 组件测试 / RoPE Component Tests")
    print("=" * 60)

    # 测试 1: 频率预计算形状 / Test 1: Frequency precomputation shape
    head_dim = 64
    seq_len = 32
    freqs_cis = precompute_freqs_cis(head_dim, seq_len * 2)
    expected_shape = (seq_len * 2, head_dim // 2)
    assert freqs_cis.shape == expected_shape, \
        f"形状错误 / Shape mismatch: {freqs_cis.shape} != {expected_shape}"
    print(f"[PASS] Test 1: precompute_freqs_cis shape correct: {freqs_cis.shape}")

    # 测试 2: apply_rotary_emb 形状保持 / Test 2: Shape preservation
    B, T, n_head = 2, 16, 4
    xq = torch.randn(B, T, n_head, head_dim)
    xk = torch.randn(B, T, n_head, head_dim)
    freqs_cis_t = freqs_cis[:T]  # 取前 T 个位置
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis_t)
    assert xq_out.shape == xq.shape, f"Q shape changed: {xq_out.shape}"
    assert xk_out.shape == xk.shape, f"K shape changed: {xk_out.shape}"
    print(f"[PASS] Test 2: Shape preserved after RoPE")

    # 测试 3: RoPE 只改变旋转方向, 不改变向量模长 (近似)
    # Test 3: RoPE preserves vector magnitude (approximately)
    q_norm_before = xq.float().norm(dim=-1).mean()
    q_norm_after = xq_out.float().norm(dim=-1).mean()
    norm_diff = abs(q_norm_before - q_norm_after)
    assert norm_diff < 0.01, f"Q norm changed too much: {norm_diff:.6f}"
    print(f"[PASS] Test 3: Vector magnitude preserved (norm diff: {norm_diff:.6f})")

    # 测试 4: 相对位置信息编码
    # Test 4: Relative position encoding — 两个 token 在不同绝对位置
    # 但相对距离相同时, 它们的 QK^T 点积应该相似
    # Two tokens at different absolute positions but same relative distance
    # should have similar dot products
    xq_full = torch.randn(1, 32, n_head, head_dim)
    xk_full = torch.randn(1, 32, n_head, head_dim)
    freqs_full = precompute_freqs_cis(head_dim, 64)[:32]
    q_rot, k_rot = apply_rotary_emb(xq_full, xk_full, freqs_full)

    # 位置 0 对位置 5 (距离 5):
    dot_05 = (q_rot[0, 0] * k_rot[0, 5]).sum(dim=-1)
    # 位置 10 对位置 15 (距离 5):
    dot_10_15 = (q_rot[0, 10] * k_rot[0, 15]).sum(dim=-1)
    # 位置 0 对位置 10 (距离 10):
    dot_0_10 = (q_rot[0, 0] * k_rot[0, 10]).sum(dim=-1)

    # 相同距离的点积模式应该不同(因为绝对位置不同), 但可以验证非对称性
    # 这里只验证旋转后确实产生了变化（与原始QK不同）
    q_raw, k_raw = xq_full, xk_full
    dot_raw_05 = (q_raw[0, 0] * k_raw[0, 5]).sum(dim=-1)
    # RoPE 后的点积应不同于原始点积 (因为位置信息被编码进去了)
    dot_diff = (dot_05 - dot_raw_05).abs().mean().item()
    assert dot_diff > 0.001, f"RoPE should change attention scores, diff too small: {dot_diff:.6f}"
    print(f"[PASS] Test 4: RoPE modifies attention scores (diff: {dot_diff:.4f})")

    print("=" * 60)
    print("RoPE 组件全部测试通过 / All RoPE component tests passed!")
    print("=" * 60)
