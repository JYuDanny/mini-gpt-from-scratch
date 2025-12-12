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
    print("Test passed -- causal_self_attentions.")


class MultiHeadAttention(nn.Module):
    """多头 causal self-attention 类，严格如 GPT-3。
    原理：将 d_model 分成 h 头，每头独立计算 attention，然后 concat + 线性融合。
    这允许模型在不同子空间捕捉多方面依赖（e.g., 语法、语义）。"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 投影矩阵：一次性线性层，GPT-3 类似偏置可选（这里无 bias）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """前向传播：输入 x (B, N, d)，输出同形。
        原理：投影 QKV，reshape 多头，计算 attention，concat + proj。"""
        B, N, d = x.shape
        
        # 投影 Q, K, V
        qkv = self.qkv_proj(x)  # (B, N, 3*d)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, N, dk)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 (B, h, N, dk)
        
        # 计算 scores
        dk = self.head_dim
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(B, self.num_heads, N, N)  # 广播到 heads
            scores = scores + mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 输出
        head_out = torch.matmul(attn_weights, v)  # (B, h, N, dk)
        head_out = head_out.permute(0, 2, 1, 3).reshape(B, N, d)  # concat
        output = self.out_proj(head_out)
        return output

# 测试入口（追加到 model.py 底部）
if __name__ == "__main__":
    d_model, num_heads = 16, 2
    attn = MultiHeadAttention(d_model, num_heads)
    
    B, N = 2, 4
    x = torch.randn(B, N, d_model)
    
    # Causal mask
    mask = torch.triu(torch.ones(N, N) * float('-inf'), diagonal=1)
    mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
    
    output = attn(x, mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output[0,0,:]: {output[0,0,:]}")
    
    assert output.shape == x.shape, "Shape mismatch!"
    print("Test passed -- MultiHeadAttention class.")


class Block(nn.Module):
    """Transformer Block 类，严格如 GPT-3 的 decoder block。
    原理：Attention + residual + LN，然后 FFN + residual + LN。
    Residual 防梯度消失，LN 稳定分布，dropout 防 overfit。"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),  # GPT-3 用 bias
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """前向：post-norm 变体（原 Transformer），GPT-3 用 pre-norm 但原理类似。
        原理：x + attn(norm(x)) 残差连接，稳定深层训练。"""
        attn_out = self.attn(x, mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.ln2(x)
        return x

# 测试入口（追加到 model.py 底部）
if __name__ == "__main__":
    d_model, num_heads, d_ff = 16, 2, 64  # 小参数测试
    block = Block(d_model, num_heads, d_ff)
    
    B, N = 2, 4
    x = torch.randn(B, N, d_model)
    
    # Causal mask
    mask = torch.triu(torch.ones(N, N) * float('-inf'), diagonal=1)
    mask = mask.unsqueeze(0).expand(B, -1, -1)
    
    output = block(x, mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output[0,0,:]: {output[0,0,:]}")
    
    assert output.shape == x.shape, "Shape mismatch!"
    print("Test passed -- Block class.")


class GPT(nn.Module):
    """完整 GPT 模型类，严格如 GPT-3 的 decoder-only 架构。
    原理：Embedding + Positional Encoding + 多层 Block + LM Head。
    自回归训练：forward 计算 logits，generate 逐 token 采样。"""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, 
                 d_ff: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)  # Learned PE，如 GPT-3
        self.layers = nn.ModuleList([Block(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 无 bias，如 GPT-3 常见
        
        # 初始化权重（GPT-3 用类似 He init，但 mini 用默认）
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """前向：输入 token IDs (B, N)，输出 logits (B, N, vocab)。
        原理：emb + pos，逐层 Block（带 mask），final norm + head。
        Causal mask 全局，确保自回归。"""
        B, N = idx.shape
        assert N <= self.block_size, f"Sequence {N} exceeds block_size {self.block_size}"
        
        tok_emb = self.token_emb(idx)  # (B, N, d)
        pos = torch.arange(0, N, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)  # (N, d)
        x = tok_emb + pos_emb
        
        # Causal mask
        mask = torch.triu(torch.ones(N, N, device=idx.device) * float('-inf'), diagonal=1)
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, N, vocab)
        return logits

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """生成：自回归采样新 token。
        原理：循环 forward 取最后 logit，softmax / temp 采样，append。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # 截取最后 block_size
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 最后位置，scale
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 测试入口（追加到 model.py 底部）
if __name__ == "__main__":
    vocab_size = 50257  # 从 tokenizer
    model = GPT(vocab_size=vocab_size, d_model=16, num_heads=2, num_layers=2, 
                d_ff=64, block_size=8, dropout=0.1)
    
    B, N = 2, 4
    idx = torch.randint(0, vocab_size, (B, N))  # 随机 token IDs
    
    logits = model(idx)
    print(f"Input shape: {idx.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Sample logits[0,0,:5]: {logits[0,0,:5]}")  # 前5个值
    
    # 生成测试
    gen_idx = model.generate(idx, max_new_tokens=3, temperature=1.0)
    print(f"Generated shape: {gen_idx.shape}")
    
    assert logits.shape == (B, N, vocab_size), "Logits shape mismatch!"
    assert gen_idx.shape == (B, N + 3), "Generate length mismatch!"
    print("Test passed -- GPT class.")
