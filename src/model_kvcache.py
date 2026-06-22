import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    """
    GPT-3 模型配置类
    """
    def __init__(self, vocab_size=50257, d_model=512, n_head=8, n_layer=6,
                 block_size=1024, dropout=0.1, bias=True, **kwargs):
        # **kwargs 接收所有多余的参数，防止报错
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias

    def to_dict(self):
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'block_size': self.block_size,
            'dropout': self.dropout,
            'bias': self.bias
        }


class CausalSelfAttention(nn.Module):
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

    def forward(self, x, layer_past=None):
        B, T, C = x.size() # batch_size, sequence_length, embedding_dim

        # 1. 计算 Q, K, V
        # 形状变换: [B, T, 3*C] -> [B, T, 3, n_head, d_k] -> [3, B, n_head, T, d_k]
        qkv = self.c_attn(x)
        # q, k, v = qkv.view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.split(self.d_model, dim=2)
        # 转换为多头形状: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # --- KV-Cache 核心逻辑 ---
        if layer_past is not None:
            past_k, past_v = layer_past
            # 在时间维度(dim=2)上进行拼接
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # 保存当前的完整 K, V 以便传给下一步
        present = (k, v)
        # ------------------------

        # 2. 计算注意力分数 (Scaled Dot-Product Attention)
        # att = (q @ k) * (1/sqrt(d_k))
        # 形状: [B, n_head, T, d_k] @ [B, n_head, d_k, T_total] -> [B, n_head, T, T_total]
        # 如果是推理(使用了cache)，T=1，T_total=历史长度+1
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 3. 应用因果掩码 (Causal Masking)
        # 将上三角部分(即未来信息)替换为 -inf，使其在 softmax 后为 0
        # 只有在没有 past (即训练或第一步推理) 时才需要 mask
        # 如果有 past，说明我们在生成第 N 个词，它有权看到之前所有的词，不需要遮蔽
        if layer_past is None:
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

        return y, present # 返回输出和缓存


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
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)
    def forward(self, x, layer_past=None):
        # Pre-Norm 结构: x = x + Sublayer(LayerNorm(x))
        # 注意这里接收 layer_past 并接收返回值 present
        attn_out, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, present


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

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        """
        前向传播
        idx: [Batch, Sequence Length] 的 Token 索引整数张量
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"输入长度 {t} 超过最大上下文 {self.config.block_size}"

        # --- 1. 位置编码调整 ---
        if past_kv is not None:
            # 如果有缓存，说明是推理的中途
            # 过去已经处理了多少个 token？
            past_length = past_kv[0][0].size(2) 
            # 当前的位置应该是从 past_length 开始
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
        else:
            # 训练阶段或推理的第一步，从 0 开始
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 确保位置没有越界
        if pos.max() >= self.config.block_size:
             raise ValueError(f"Position index {pos.max()} out of range")

        # --- 2. Embedding ---
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)

        # --- 3. Transformer Blocks 逐层传递 Cache ---
        new_kv = []
        for i, block in enumerate(self.transformer.h):
            # 取出对应层的旧 cache
            layer_past = past_kv[i] if past_kv is not None else None

            # 前向传播
            x, layer_present = block(x, layer_past=layer_past)

            # 如果开启了 cache 模式，收集新的 cache
            if use_cache:
                new_kv.append(layer_present)

        # 4. Final Norm
        x = self.transformer.ln_f(x)

        # 5. 输出 Logits
        if targets is not None:
            # 如果是训练模式(有target)，我们计算 Loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 如果是推理模式(只取最后一步)，为了节省计算，可以只算最后一个 token
            # 但为了通用性，这里返回所有 logits
            logits = self.lm_head(x)
            loss = None

        return logits, loss, (new_kv if use_cache else None)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """
        自回归文本生成
        idx: (B, T) 形状的起始 token 序列
        """
        # 初始化
        past_kv = None

        for _ in range(max_new_tokens):
            # 位置安全检查：如果 idx 的长度已经达到了 block_size，就不能再往后生成了
            if idx.size(1) >= self.config.block_size:
                print(f"\nWarning: Reached context limit ({self.config.block_size}). Stopping generation.")
                break
            # 如果是第一次迭代(past_kv is None)，我们需要把完整的 prompt 传进去来计算初始的 KV
            # 如果不是第一次(past_kv 有值)，我们只需要传刚刚生成的最后一个 token
            if past_kv is None:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            else:
                idx_cond = idx[:, -1:] # 只取最后一个 token (B, 1)

            # 前向传播，开启 use_cache=True
            logits, _, past_kv = self(idx_cond, past_kv=past_kv, use_cache=True)

            # 这里的 logits 已经是 [B, 1, vocab_size]，只取最后一个时间步的预测 logits
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

            # 拼接生成的 token 到完整序列中
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ==========================================
# KV-Cache 增强版自动化验证脚本
# ==========================================
if __name__ == "__main__":
    # 强制使用 CPU 进行确定性测试，避免 CUDA 带来的微小浮点误差干扰验证
    device = 'cpu'

    print("-" * 60)
    print("开始 Mini-GPT (KV-Cache版) 模型验证...")
    print("-" * 60)

    # 1. 配置模型参数
    conf = GPTConfig(
        vocab_size=100,
        d_model=64,      # 稍微加大一点维度，确保矩阵运算不退化
        n_head=4,
        n_layer=2,
        block_size=32,
        dropout=0.0      # 关闭 Dropout 以确保确定性比较
    )

    try:
        model = GPT(conf).to(device)
        model.eval() # 必须开启 eval 模式，关闭 Dropout
        print("[1/5] 模型实例化成功")
    except Exception as e:
        print(f"[1/5] 模型实例化失败: {e}")
        exit()

    # 2. 验证标准前向传播 (不带 Cache)
    try:
        batch_size = 2
        seq_len = 10
        dummy_input = torch.randint(0, conf.vocab_size, (batch_size, seq_len)).to(device)

        # 这里的 use_cache=False 模拟训练时的行为
        logits, _, _ = model(dummy_input, use_cache=False)
        expected_shape = (batch_size, seq_len, conf.vocab_size)
        assert logits.shape == expected_shape, f"Logits shape 错误: {logits.shape}"
        print("[2/5] 标准前向传播验证通过 (Training Mode)")
    except Exception as e:
        print(f"[2/5] 前向传播失败: {e}")
        exit()

    # 3. 验证 KV-Cache 数值一致性 (核心验证)
    # 目标：证明 `[t1, t2, t3]` 一次性算出的结果，和 `[t1, t2]` 算出 cache 后再喂入 `[t3]` 的结果完全一样
    try:
        # 构造一个随机输入 [B, T]
        x = torch.randint(0, conf.vocab_size, (1, 10)).to(device)

        # --- 方式 A: 无 Cache (基准) ---
        # 模拟一次性输入所有 token
        logits_ref, _, _ = model(x, use_cache=False)
        # 我们关注最后一个 token 的预测结果
        last_logit_ref = logits_ref[:, -1, :] 

        # --- 方式 B: 有 Cache (分步) ---
        # 步骤 1: 先输入前 9 个 token，生成 Cache
        x_prefix = x[:, :-1] # 前 9 个
        _, _, past_kv = model(x_prefix, use_cache=True)

        # 步骤 2: 输入第 10 个 token，并带上 Cache
        x_last = x[:, -1:]   # 第 10 个
        logits_step, _, _ = model(x_last, past_kv=past_kv, use_cache=True)
        # 此时输出应该是 (B, 1, Vocab)，直接取出来
        last_logit_step = logits_step[:, -1, :]

        # --- 比较 ---
        # 计算最大绝对误差
        diff = (last_logit_ref - last_logit_step).abs().max().item()

        if diff > 1e-5:
            raise AssertionError(f"数值不一致! Max Diff: {diff:.6f}\n"
                                 "这意味着 KV-Cache 的拼接逻辑或位置编码处理有误。")

        print(f"[3/5] KV-Cache 数值一致性验证通过 (Diff: {diff:.2e})")
        print("    证明：分步推理结果与一次性计算结果完全吻合。")

    except Exception as e:
        print(f"[3/5] KV-Cache 验证失败: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # 4. 验证 generate 函数流程
    try:
        start_idx = torch.zeros((1, 1), dtype=torch.long).to(device)
        # 生成 5 个新 token
        generated = model.generate(start_idx, max_new_tokens=5)

        assert generated.shape == (1, 6), f"生成形状错误: {generated.shape}"
        print("[4/5] 生成函数流程验证通过 (generate() with Cache)")
    except Exception as e:
        print(f"[4/5] 生成函数测试失败: {e}")
        exit()

    # 5. 验证权重绑定
    try:
        assert model.transformer.wte.weight is model.lm_head.weight
        print("[5/5] 权重绑定验证通过")
    except AssertionError:
        print("[5/5] 权重绑定失败")
        exit()

    print("-" * 60)
    print("Mini-GPT (with KV-cache) 架构验证全部通过！")
    print("模型结构在逻辑和张量形状上完全正确。")
    print("-" * 60)
