# src/model.py — 统一的 Mini-GPT 模型 / Unified Mini-GPT Model
# 架构 / Architecture: Decoder-only Transformer (GPT-3 风格 / GPT-3 style)
#
# 组件分离 / Component Separation:
#   位置编码:  src/rope.py  — RoPE 旋转位置编码 (数学纯函数)
#   模型架构:  src/model.py — Transformer 主体 (GPTConfig + Block + GPT)
#   Position encoding is in src/rope.py (pure math), architecture is here.
#
# 支持的运行模式 / Supported Modes:
#   pos_enc_type="rope"    → RoPE 旋转位置编码 + KV-Cache 推理加速 (当前默认)
#   pos_enc_type="learned" → 可学习绝对位置编码 + KV-Cache 推理加速
#   pos_enc_type="learned" 且 use_cache=False → 纯基础 GPT (无缓存, 教学用)
#
# 数据流总览 / Data Flow Overview:
#   Token IDs → Token Embedding + Position Encoding
#     → [Block × n_layer: Pre-LN → CausalAttn → + → Pre-LN → FFN → +]
#     → Final LayerNorm → LM Head → Logits

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 模型配置 / Model Configuration
# ============================================================
class GPTConfig:
    """
    GPT 模型配置类 / GPT Model Configuration

    原理 / Principle:
        所有模型超参数集中管理，避免分散在代码各处。
        **kwargs 接收多余参数避免因配置字典有多余字段而报错。
        All hyperparameters are centralized; **kwargs absorbs extras to avoid
        errors from extra config fields.
    """
    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        n_head=8,
        n_layer=6,
        block_size=1024,
        dropout=0.1,
        bias=True,
        pos_enc_type="rope",  # 位置编码类型: "rope" | "learned"
        # Position encoding type: "rope" (Rotary) or "learned" (absolute wpe)
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias
        self.pos_enc_type = pos_enc_type

    def to_dict(self):
        """导出为字典，用于 checkpoint 保存和推理恢复"""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'block_size': self.block_size,
            'dropout': self.dropout,
            'bias': self.bias,
            'pos_enc_type': self.pos_enc_type,
        }


# ============================================================
# 2. 多头因果自注意力 / Multi-Head Causal Self-Attention
# ============================================================
class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力机制 (Multi-Head Causal Self-Attention)

    原理 / Principle:
        每个 token 通过 Q (query) 查询所有历史 token 的 K (key)，
        按相似度加权聚合 V (value)，实现"关注上文"的能力。
        因果掩码 (causal mask) 确保 token i 只能看到位置 ≤ i 的信息。

    支持的两种位置编码 / Two supported position encodings:
        - learned (绝对位置编码): QKV 投影后直接计算注意力。
          位置信息来自 wpe Embedding，在此层不做额外处理。
        - rope (旋转位置编码): QKV 投影后、注意力计算前，
          对 Q 和 K 施加 RoPE 旋转，使注意力天然编码相对位置。
          RoPE is applied to Q and K before attention to encode relative position.
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0, \
            f"d_model ({config.d_model}) 必须能被 n_head ({config.n_head}) 整除"

        # Q, K, V 投影矩阵 — 一次性计算三者，提高效率
        # Single linear layer computes Q, K, V in one shot for efficiency
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # 输出投影 — 将多头拼接结果映射回 d_model 维度
        # Output projection — maps concatenated multi-head result back to d_model
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropout 正则化 — 防止过拟合 / Dropout for regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.pos_enc_type = config.pos_enc_type

        # 注册因果掩码 (下三角矩阵) — 不是可训练参数, 但随模型保存
        # Register causal mask (lower triangular) — buffer, not trainable
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x, freqs_cis=None, layer_past=None):
        """
        前向传播 / Forward Pass

        参数 / Args:
            x:          输入张量 (B, T, d_model) / input tensor
            freqs_cis:  预计算的 RoPE 频率 (仅 pos_enc_type="rope" 时需要)
                        Precomputed RoPE frequencies (only for rope mode)
            layer_past: 上一轮的 (K, V) 缓存, 用于 KV-Cache 推理
                        Past (K, V) cache for KV-Cache inference

        返回 / Returns:
            y:       注意力输出 (B, T, d_model) / attention output
            present: 当前轮的 (K, V), 用于下一步缓存 / current (K, V) for next cache step
        """
        B, T, C = x.size()

        # -------------------------------------------------------
        # Step 1: QKV 投影 / Project to Q, K, V
        # -------------------------------------------------------
        # 单次 Linear 投影后 split, 比三次独立投影效率更高
        # Single projection then split — more efficient than 3 separate projections
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # 变形为多头形状 / Reshape to multi-head layout
        # (B, T, C) → (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim)
        k = k.view(B, T, self.n_head, head_dim)
        v = v.view(B, T, self.n_head, head_dim)

        # -------------------------------------------------------
        # Step 2: RoPE 旋转位置编码 (仅 rope 模式)
        # -------------------------------------------------------
        if self.pos_enc_type == "rope" and freqs_cis is not None:
            # 从 src.rope 导入 — 独立的位置编码组件
            # Imported from src.rope — standalone position encoding component
            try:
                from src.rope import apply_rotary_emb
            except ImportError:
                from rope import apply_rotary_emb
            # 在 transpose 前应用 (保持 (B, T, n_head, head_dim) 形状)
            # Apply before transpose (preserves (B, T, n_head, head_dim) layout)
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # 转置为注意力标准形状 / Transpose to standard attention layout
        # (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # -------------------------------------------------------
        # Step 3: KV-Cache 拼接 / Concatenate with past KV
        # -------------------------------------------------------
        if layer_past is not None:
            past_k, past_v = layer_past
            # 沿时间维度拼接: 历史 K + 当前 K
            # Concatenate along time dim: past K + current K
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # 保存当前完整 K, V 用于下一步推理 / Save for next generation step
        present = (k, v)

        # -------------------------------------------------------
        # Step 4: 缩放点积注意力 / Scaled Dot-Product Attention
        # -------------------------------------------------------
        # att = softmax(Q @ K^T / sqrt(d_k)) @ V
        # 除以 sqrt(d_k) 防止点积方差随维度增大, 保持 softmax 梯度稳定
        # Scaling prevents dot product variance from growing with dimension,
        # keeping softmax gradients stable
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # -------------------------------------------------------
        # Step 5: 因果掩码 / Causal Masking
        # -------------------------------------------------------
        # 只在无缓存时 (训练或推理第一步) 需要掩码
        # Mask only needed when there's no cache (training or first inference step)
        # 有缓存时, 当前 token 有权看到全部历史, 不需要遮蔽
        if layer_past is None:
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        # -------------------------------------------------------
        # Step 6: Softmax + Dropout + 聚合 Value
        # -------------------------------------------------------
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # (B, n_head, T, T_total) @ (B, n_head, T_total, head_dim) → (B, n_head, T, head_dim)
        y = att @ v

        # -------------------------------------------------------
        # Step 7: 合并多头 / Merge Multi-Head
        # -------------------------------------------------------
        # (B, n_head, T, head_dim) → (B, T, n_head, head_dim) → (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影 + Dropout
        y = self.resid_dropout(self.c_proj(y))

        return y, present


# ============================================================
# 3. 前馈神经网络 / Feed-Forward Network (FFN)
# ============================================================
class FFN(nn.Module):
    """
    前馈神经网络 (Position-wise Feed-Forward Network)

    原理 / Principle:
        这是一个作用于每个位置独立的两层 MLP:
        d_model → 4×d_model → d_model.
        升维 (expansion) 提供更多非线性容量, 捕捉 token 内的特征变换;
        降维 (projection) 压缩回原始维度, 适配残差连接。

    激活函数 / Activation: GELU (Gaussian Error Linear Unit)
        GELU = x * Φ(x), 其中 Φ 是标准正态分布 CDF。
        比 ReLU 更平滑, 负值微小平滑通过而非截断为 0, 梯度流更好。
        Smoother than ReLU — small negative values pass through rather than
        being clipped to zero, giving better gradient flow.
    """

    def __init__(self, config):
        super().__init__()
        # 升维层 / Expansion layer: d_model → 4×d_model
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        # 降维层 / Projection layer: 4×d_model → d_model
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        前向传播 / Forward Pass
        x: (B, T, d_model) → y: (B, T, d_model)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ============================================================
# 4. Transformer Block / 解码器层
# ============================================================
class Block(nn.Module):
    """
    Transformer 解码器层 (Decoder Block)

    架构 / Architecture: Pre-Norm (GPT-3 风格)
        x = x + Attention(LayerNorm(x))    ← 自注意子层 / Self-Attention sublayer
        x = x + FFN(LayerNorm(x))          ← 前馈子层 / Feed-Forward sublayer

    为什么 Pre-Norm / Why Pre-Norm:
        LayerNorm 放在子层之前 (而非之后) 使得训练更稳定,
        深层网络的梯度流动更平滑, 已成为 GPT-3 及后续模型的标准做法。
        Placing LayerNorm before each sublayer stabilizes training for
        deep networks. Standard practice since GPT-3.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)   # 注意力子层前的 LayerNorm
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)   # FFN 子层前的 LayerNorm
        self.ffn = FFN(config)

    def forward(self, x, freqs_cis=None, layer_past=None):
        """
        前向传播 / Forward Pass

        参数 / Args:
            x:          输入 (B, T, d_model)
            freqs_cis:  RoPE 频率 (仅 rope 模式)
            layer_past: KV 缓存 (仅推理时使用)

        返回 / Returns:
            x:       输出 (B, T, d_model)
            present: KV 缓存 (用于下一步推理)
        """
        # Pre-Norm Attention: LN → Attention → 残差连接
        attn_out, present = self.attn(self.ln_1(x), freqs_cis=freqs_cis, layer_past=layer_past)
        x = x + attn_out
        # Pre-Norm FFN: LN → FFN → 残差连接
        x = x + self.ffn(self.ln_2(x))
        return x, present


# ============================================================
# 5. GPT 模型主体 / GPT Model Body
# ============================================================
class GPT(nn.Module):
    """
    微型 GPT 语言模型 / Mini GPT Language Model

    架构总览 / Architecture Overview:
        Token IDs [B, T]
          ├── Token Embedding (wte)     [B, T, d_model]
          ├── Position Encoding (wpe 或 RoPE)  [B, T, d_model]
          ├── Dropout
          ├── Block × n_layer  (Pre-Norm Attention + FFN)
          ├── Final LayerNorm
          └── LM Head (weight-tied with wte)  →  Logits [B, T, vocab_size]

    关键设计 / Key Design Decisions:
        1. 权重绑定 (Weight Tying): lm_head.weight = wte.weight
           共享嵌入层和输出层的权重, 减少约 30% 参数量,
           对小模型尤其重要, 可有效防止过拟合。
        2. Pre-Norm: LayerNorm 在子层之前, 训练更稳定。
        3. 残差投影缩放: c_proj 的初始化被缩放, 控制深层方差。
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # --- 嵌入层 / Embeddings ---
        # Token Embedding: 将 token ID 映射到 d_model 维向量
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.d_model),
        ))

        # --- 位置编码 / Position Encoding ---
        if config.pos_enc_type == "learned":
            # 可学习的绝对位置嵌入 / Learned absolute position embedding
            # 形状 (block_size, d_model), 每个位置一个独立的可学习向量
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.d_model)
        elif config.pos_enc_type == "rope":
            # RoPE: 预计算频率表 (buffer, 不参与训练)
            # RoPE: precompute frequency table as a buffer (not trained)
            head_dim = config.d_model // config.n_head
            try:
                from src.rope import precompute_freqs_cis
            except ImportError:
                from rope import precompute_freqs_cis
            # 预计算 2×block_size 以支持外推
            # Precompute 2×block_size for potential extrapolation
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(head_dim, config.block_size * 2)
            )
        else:
            raise ValueError(f"未知位置编码类型 / Unknown pos_enc_type: {config.pos_enc_type}")

        # --- 输出层 / LM Head ---
        # bias=False + 权重绑定 / bias=False + weight tying
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定 (Weight Tying): 嵌入层和输出层共享权重
        self.transformer.wte.weight = self.lm_head.weight

        # --- 初始化 / Initialization ---
        self.apply(self._init_weights)

        # 残差投影层特殊缩放初始化
        # Special scaled init for residual projection layers
        # 原理: 在深层网络中, 如果 c_proj 用标准初始化,
        # 每个 Block 的残差支路会逐层放大信号, 导致训练不稳定。
        # 缩小 c_proj 的初始方差可以抵消这种累积效应。
        # Principle: standard init causes residual branches to amplify
        # signals layer by layer. Scaling down c_proj variance counters this.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        """权重初始化 / Weight Initialization — N(0, 0.02)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_pos_encoding(self, device, t, past_length=0):
        """
        获取位置编码 / Get position encoding

        根据 pos_enc_type 的类型返回不同的位置编码:
        - "learned": 返回位置嵌入向量 [T, d_model]
        - "rope":    返回预计算的复数频率 [T, head_dim//2]

        参数:
            device:      设备 (cpu/cuda)
            t:           当前输入长度
            past_length: 已缓存的 token 数量 (推理时 > 0)
        """
        if self.config.pos_enc_type == "learned":
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
            if pos.max() >= self.config.block_size:
                raise ValueError(f"位置索引 {pos.max()} 超出 block_size {self.config.block_size}")
            return self.transformer['wpe'](pos)
        elif self.config.pos_enc_type == "rope":
            # 从预计算的频率表中切片当前位置的 RoPE 频率
            return self.freqs_cis[past_length: past_length + t].to(device)
        else:
            raise ValueError(f"未知 pos_enc_type: {self.config.pos_enc_type}")

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        """
        前向传播 / Forward Pass

        参数 / Args:
            idx:       Token ID 张量 (B, T)
            targets:   目标 token ID (B, T), 用于计算 loss; 推理时为 None
            past_kv:   历史 KV 缓存, 列表长度 = n_layer, 每个元素是 (K, V) 元组
            use_cache: 是否返回新的 KV 缓存

        返回 / Returns:
            logits:   预测 logits (B, T, vocab_size)
            loss:     交叉熵损失 (训练时) 或 None (推理时)
            new_kv:   新的 KV 缓存列表 (use_cache=True 时) 或 None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"输入长度 {t} 超过 block_size {self.config.block_size}"

        # --- 1. 位置编码 / Position Encoding ---
        past_length = past_kv[0][0].size(2) if past_kv is not None else 0
        pos_enc = self._get_pos_encoding(device, t, past_length)

        # --- 2. Token Embedding + 位置编码 / Token Embedding + Position Encoding ---
        tok_emb = self.transformer.wte(idx)  # (B, T, d_model)

        if self.config.pos_enc_type == "learned":
            # 可学习位置编码: tok_emb + pos_emb
            x = self.transformer.drop(tok_emb + pos_enc)
            freqs_cis = None  # 不需要 RoPE 频率
        elif self.config.pos_enc_type == "rope":
            # RoPE: 位置编码在 Attention 层内部通过旋转 Q,K 施加
            # 这里只加 token embedding, freqs_cis 传入 Attention 层
            x = self.transformer.drop(tok_emb)
            freqs_cis = pos_enc
        else:
            raise ValueError(f"未知 pos_enc_type: {self.config.pos_enc_type}")

        # --- 3. Transformer Blocks / 逐层传递 ---
        new_kv = []
        for i, block in enumerate(self.transformer.h):
            # 取出对应层的 KV 缓存
            layer_past = past_kv[i] if past_kv is not None else None

            # 前向传播: 传入 RoPE 频率 (仅 rope 模式) 和 KV 缓存 (仅推理时)
            x, layer_present = block(x, freqs_cis=freqs_cis, layer_past=layer_past)

            if use_cache:
                new_kv.append(layer_present)

        # --- 4. 最终 LayerNorm ---
        x = self.transformer.ln_f(x)

        # --- 5. LM Head → Logits ---
        if targets is not None:
            # 训练模式: 计算全部位置的 logits + loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # 推理模式: 只返回 logits
            logits = self.lm_head(x)
            loss = None

        return logits, loss, (new_kv if use_cache else None)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """
        自回归文本生成 / Autoregressive Text Generation

        原理 / Principle:
            1. 输入 prompt token 序列 → 前向传播 → 取最后一个位置的 logits
            2. logits / temperature 缩放温度 → 控制生成多样性
            3. 可选 top-k 过滤: 只保留概率最高的 k 个 token
            4. softmax → multinomial 采样 → 得到下一个 token
            5. 拼接新 token → 如果启用 KV-Cache, 只对新 token 计算 (效率 O(T) 而非 O(T²))
            6. 重复直到达到 max_new_tokens 或遇到 EOS

        参数 / Args:
            idx:             起始 token 序列 (B, T)
            max_new_tokens:  最多生成多少个新 token
            temperature:     采样温度 (>1 更随机, <1 更确定)
            top_k:           Top-K 过滤参数
            eos_id:          EOS token ID, 遇到则停止

        返回 / Returns:
            idx: 完整的 token 序列 (prompt + 生成的 tokens)
        """
        past_kv = None

        for _ in range(max_new_tokens):
            # 检查是否达到 context 上限
            if idx.size(1) >= self.config.block_size:
                break

            # KV-Cache 优化: 第一步传全部 prompt, 后续只传最新 token
            # KV-Cache optimization: first step passes full prompt,
            # subsequent steps pass only the latest token
            if past_kv is None:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            else:
                idx_cond = idx[:, -1:]  # (B, 1)

            # 前向传播 (带 KV-Cache)
            logits, _, past_kv = self(idx_cond, past_kv=past_kv, use_cache=True)

            # 只取最后一个时间步的 logits
            logits = logits[:, -1, :] / temperature

            # Top-K 过滤: 将概率较低的 token 设为 -inf
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Softmax → 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # EOS 早停
            if eos_id is not None and idx_next.item() == eos_id:
                break

            # 拼接新 token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================
# 6. 自动化验证脚本 / Automated Verification Suite
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(1337)
    device = 'cpu'
    n_test_passed = 0
    n_test_total = 0

    def check(condition, msg):
        """辅助: 断言并打印结果 / Helper: assert and print result"""
        global n_test_passed, n_test_total
        n_test_total += 1
        if condition:
            print(f"  [PASS] {msg}")
            n_test_passed += 1
        else:
            print(f"  [FAIL] {msg}")
            exit(1)

    def section(title):
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")

    print("=" * 60)
    print("  Mini-GPT 统一模型验证 / Unified Model Verification")
    print("=" * 60)

    # ================================================================
    # 测试 1: 基础 GPT (learned PE, 无 cache)
    # ================================================================
    section("Test 1: Learned PE — Basic GPT (no cache)")

    conf1 = GPTConfig(
        vocab_size=100, d_model=32, n_head=4, n_layer=2,
        block_size=32, dropout=0.0, pos_enc_type="learned"
    )
    model1 = GPT(conf1).to(device).eval()
    check(hasattr(model1.transformer, 'wpe'), "wpe (learned PE) exists")

    x1 = torch.randint(0, 100, (2, 10)).to(device)
    logits, loss, kv = model1(x1, use_cache=False)
    check(logits.shape == (2, 10, 100), f"Logits shape correct: {logits.shape}")
    check(kv is None, "KV cache is None when use_cache=False")
    check(model1.transformer.wte.weight is model1.lm_head.weight, "Weight tying active")

    # 生成测试
    start1 = torch.zeros((1, 1), dtype=torch.long)
    gen1 = model1.generate(start1, max_new_tokens=5)
    check(gen1.shape == (1, 6), f"Generation shape correct: {gen1.shape}")

    # ================================================================
    # 测试 2: Learned PE + KV-Cache 一致性
    # ================================================================
    section("Test 2: Learned PE + KV-Cache Consistency")

    conf2 = GPTConfig(
        vocab_size=100, d_model=64, n_head=4, n_layer=2,
        block_size=32, dropout=0.0, pos_enc_type="learned"
    )
    model2 = GPT(conf2).to(device).eval()

    # 一次性输入全部 10 个 token / Parallel forward (all 10 tokens at once)
    x_full = torch.randint(0, 100, (1, 10)).to(device)
    logits_ref, _, _ = model2(x_full, use_cache=False)
    last_logit_ref = logits_ref[:, -1, :]

    # 分步输入: 前 9 个 → 第 10 个 (带 cache)
    # Sequential: first 9 → then 10th (with cache)
    x_prefix = x_full[:, :9]
    _, _, past_kv = model2(x_prefix, use_cache=True)
    x_last = x_full[:, 9:]
    logits_step, _, _ = model2(x_last, past_kv=past_kv, use_cache=True)
    last_logit_step = logits_step[:, -1, :]

    diff = (last_logit_ref - last_logit_step).abs().max().item()
    check(diff < 1e-5, f"KV-cache consistency (diff: {diff:.2e})")

    # ================================================================
    # 测试 3: RoPE + KV-Cache 一致性
    # ================================================================
    section("Test 3: RoPE + KV-Cache Consistency")

    conf3 = GPTConfig(
        vocab_size=100, d_model=64, n_head=4, n_layer=2,
        block_size=32, dropout=0.0, pos_enc_type="rope"
    )
    model3 = GPT(conf3).to(device).eval()
    check(not hasattr(model3.transformer, 'wpe'), "wpe removed in rope mode")
    check(hasattr(model3, 'freqs_cis'), "freqs_cis buffer exists in rope mode")

    # 一次性并行 / Parallel forward
    x_full3 = torch.randint(0, 100, (1, 10)).to(device)
    logits_ref3, _, _ = model3(x_full3, use_cache=False)
    last_logit_ref3 = logits_ref3[:, -1, :]

    # 分步 / Sequential with cache
    _, _, past_kv3 = model3(x_full3[:, :9], use_cache=True)
    logits_step3, _, _ = model3(x_full3[:, 9:], past_kv=past_kv3, use_cache=True)
    last_logit_step3 = logits_step3[:, -1, :]

    diff3 = (last_logit_ref3 - last_logit_step3).abs().max().item()
    check(diff3 < 1e-5, f"RoPE + KV-cache consistency (diff: {diff3:.2e})")

    # ================================================================
    # 测试 4: 生成冒烟测试 (RoPE 模式)
    # ================================================================
    section("Test 4: Generation Smoke Test (RoPE mode)")
    start4 = torch.zeros((1, 1), dtype=torch.long).to(device)
    gen4 = model3.generate(start4, max_new_tokens=5)
    check(gen4.shape == (1, 6), f"RoPE generation shape: {gen4.shape}")

    # ================================================================
    # 测试 5: 训练 loss 计算
    # ================================================================
    section("Test 5: Training Loss (RoPE mode)")
    x5 = torch.randint(0, 100, (2, 10)).to(device)
    y5 = torch.randint(0, 100, (2, 10)).to(device)
    _, loss5, _ = model3(x5, targets=y5)
    check(loss5 is not None, "Loss computed for training")
    check(loss5.item() > 0, f"Loss > 0 (value: {loss5.item():.4f})")

    # ================================================================
    # 结果汇总
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  验证完成 / Verification Complete: {n_test_passed}/{n_test_total} PASS")
    if n_test_passed == n_test_total:
        print("  全部测试通过 / All tests passed!")
    else:
        print(f"  {n_test_total - n_test_passed} 项测试失败 / tests failed!")
    print(f"{'=' * 60}")
