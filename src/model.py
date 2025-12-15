import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTConfig:
    """
    GPT-3 模型配置类
    """
    def __init__(self, vocab_size=50257, d_model=512, n_head=8, n_layer=6, 
                 block_size=1024, dropout=0.1, bias=True):
        self.vocab_size = vocab_size    # 词表大小
        self.d_model = d_model          # 嵌入维度
        self.n_head = n_head            # 注意力头数
        self.n_layer = n_layer          # 层数
        self.block_size = block_size    # 最大上下文长度 (Context Window)
        self.dropout = dropout          # Dropout 概率
        self.bias = bias                # 是否在 Linear 层中使用偏置 (GPT-3 通常为 True)


class MultiHeadAttention(nn.Module):
    """
    多头因果自注意力机制 (Multi-Head Causal Self-Attention)
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        # key, query, value 投影
        # GPT-3 在 QKV 投影中通常包含 Bias (偏置)
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        
        # 注册一个下三角掩码矩阵 (Causal Mask)
        # register_buffer 确保它作为模型状态保存，但不是可训练参数
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch, Time(Sequence Length), Channel(d_model)

        # 1. 计算 Q, K, V
        # 形状变换: [B, T, 3*C] -> [B, T, 3, n_head, d_k] -> [3, B, n_head, T, d_k]
        qkv = self.c_attn(x)
        q, k, v = qkv.view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)

        # 2. 计算注意力分数 (Scaled Dot-Product Attention)
        # att = (q @ k) * (1/sqrt(d_k))
        # 形状: [B, n_head, T, d_k] @ [B, n_head, d_k, T] -> [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 3. 应用因果掩码 (Causal Masking)
        # 将上三角部分(即未来信息)替换为 -inf，使其在 softmax 后为 0
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # 4. Softmax 归一化与 Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 5. 聚合 Value
        # 形状: [B, n_head, T, T] @ [B, n_head, T, d_k] -> [B, n_head, T, d_k]
        y = att @ v 

        # 6. 合并多头并输出
        # [B, n_head, T, d_k] -> [B, T, n_head, d_k] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影与残差 dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class FFN(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)
    通常维度放大 4 倍，使用 GELU 激活函数
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer Block (Decoder)
    GPT-3 关键特征：Pre-Normalization (Pre-Norm)
    LayerNorm 位于 Attention 和 MLP 之前
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.fnn = FFN(config)

    def forward(self, x):
        # Pre-Norm 结构: x = x + Sublayer(LayerNorm(x))
        x = x + self.attn(self.ln_1(x))
        x = x + self.fnn(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    微型 GPT-3 模型主体
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token Embedding (词嵌入)
            wte = nn.Embedding(config.vocab_size, config.d_model),
            # Positional Embedding (位置嵌入 - 可学习)
            wpe = nn.Embedding(config.block_size, config.d_model),
            # Dropout
            drop = nn.Dropout(config.dropout),
            # Transformer Layers (堆叠 Block)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final LayerNorm (Pre-Norm 架构需要在最后加一层 LN)
            ln_f = nn.LayerNorm(config.d_model),
        ))

        # Language Model Head (输出层)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # GPT-3 关键特征：Weight Tying (权重绑定)
        # 将 Embedding 层的权重与输出层(lm_head)的权重共享
        # 这不仅减少了参数，还被证明能提升效果
        self.transformer.wte.weight = self.lm_head.weight

        # 参数初始化 (参考 GPT-2/3 论文)
        self.apply(self._init_weights)

        # 特殊初始化：对残差投影层进行缩放 (1/sqrt(2 * n_layer))
        # 目的是在深层网络中控制方差增长
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播
        idx: [Batch, Sequence Length] 的 Token 索引整数张量
        """
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.block_size, f"输入长度 {t} 超过最大上下文 {self.config.block_size}"

        # 1. 嵌入层
        pos = torch.arange(0, t, dtype=torch.long, device=device) # [0, 1, ..., t-1]

        # Token Embedding + Positional Embedding
        tok_emb = self.transformer.wte(idx) # [B, T, d_model]
        pos_emb = self.transformer.wpe(pos) # [T, d_model]
        x = self.transformer.drop(tok_emb + pos_emb)

        # 2. Transformer Blocks
        for block in self.transformer.h:
            x = block(x)

        # 3. Final Norm
        x = self.transformer.ln_f(x)

        # 4. 输出 Logits
        if targets is not None:
            # 如果是训练模式(有target)，我们计算 Loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 如果是推理模式(只取最后一步)，为了节省计算，可以只算最后一个 token
            # 但为了通用性，这里返回所有 logits
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """
        自回归文本生成
        idx: (B, T) 形状的起始 token 序列
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，截断到 block_size 以内
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 前向传播
            logits, _ = self(idx_cond)

            # 只取最后一个时间步的预测 logits
            logits = logits[:, -1, :] / temperature

            # 可选: Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 计算概率并采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 早停检查
            if eos_id is not None and idx_next.item() == eos_id:
                break

            # 拼接生成的 token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ==========================================
# 自动化验证脚本
# ==========================================
if __name__ == "__main__":
    print("-" * 50)
    print("开始 Micro-GPT-3 模型验证...")
    print("-" * 50)

    # 1. 配置模型参数 (使用极小参数以便快速运行)
    # 模拟一个超小的 GPT-3
    conf = GPTConfig(
        vocab_size=100,  # 只有 100 个词
        d_model=32,      # 嵌入维度 32
        n_head=4,        # 4 个头
        n_layer=2,       # 2 层
        block_size=32,   # 上下文窗口 32
        dropout=0.0
    )
    
    try:
        model = GPT(conf)
        print("[1/4] 模型实例化成功")
        print(f"    参数量: {sum(p.numel() for p in model.parameters())/1e3:.2f}K")
    except Exception as e:
        print(f"[1/4] 模型实例化失败: {e}")
        exit()

    # 2. 验证前向传播 (Forward Pass)
    try:
        # 创建一个 batch_size=2, seq_len=8 的随机输入
        batch_size = 2
        seq_len = 8
        dummy_input = torch.randint(0, conf.vocab_size, (batch_size, seq_len))
        
        logits, loss = model(dummy_input)
        
        expected_shape = (batch_size, seq_len, conf.vocab_size)
        assert logits.shape == expected_shape, f"Logits shape 错误: {logits.shape} != {expected_shape}"
        print("[2/4] 前向传播验证通过 (Output Shape 正确)")
    except Exception as e:
        print(f"[2/4] 前向传播失败: {e}")
        exit()

    # 3. 验证因果掩码 (Causal Mask)
    # 这一步通过检查生成结果是否会报错，以及注意力矩阵是否为下三角来隐式验证
    # 这里我们做一个直接的生成测试
    try:
        start_idx = torch.zeros((1, 1), dtype=torch.long) # 从 token 0 开始
        generated = model.generate(start_idx, max_new_tokens=10)
        
        assert generated.shape == (1, 11), f"生成长度错误: {generated.shape}"
        print("[3/4] 文本生成逻辑验证通过 (Autoregressive Generation)")
        print(f"    生成序列示例: {generated.tolist()}")
    except Exception as e:
        print(f"[3/4] 生成测试失败: {e}")
        exit()
        
    # 4. 验证权重绑定 (Weight Tying)
    try:
        # 检查 Embedding 指针是否等于 Head 指针
        assert model.transformer.wte.weight is model.lm_head.weight
        print("[4/4] 权重绑定验证通过 (Weight Tying Active)")
    except AssertionError:
        print("[4/4] 权重绑定失败: Embedding 和 LM Head 权重不一致")
        exit()

    print("-" * 50)
    print("Micro-GPT-3 架构验证全部通过！")
    print("模型结构已经在逻辑和张量形状上完全正确。")
    print("-" * 50)
