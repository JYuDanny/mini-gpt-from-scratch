# src/model.py — 统一的 Mini-GPT 模型 / Unified Mini-GPT Model
# 架构 / Architecture: Decoder-only Transformer (LLaMA 风格 / LLaMA-style)
#
# 组件分离 / Component Separation:
#   位置编码:  src/rope.py  — RoPE 旋转位置编码 (数学纯函数)
#   模型架构:  src/model.py — Transformer 主体 (GPTConfig + Block + GPT)
#
# 支持的运行模式 / Supported Modes:
#   pos_enc_type="rope"    → RoPE 旋转位置编码 + KV-Cache (默认)
#   pos_enc_type="learned" → 可学习绝对位置编码 + KV-Cache
#
# 与 GPT-3 的架构差异 / Architecture differences from GPT-3:
#   - RMSNorm (替代 LayerNorm): 去均值, 只做缩放, 更快更轻量
#     RMSNorm replaces LayerNorm — drops mean subtraction, scale only, faster
#   - SwiGLU (替代 GELU FFN): 门控激活, 计算量相近但效果更好
#     SwiGLU replaces GELU FFN — gated activation, similar FLOPs, better quality
#   - Flash Attention: 使用 F.scaled_dot_product_attention 融合算子
#     Fused kernel via F.scaled_dot_product_attention

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
    """
    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        n_head=8,
        n_layer=6,
        block_size=1024,
        dropout=0.1,
        bias=False,                        # SwiGLU + RMSNorm 通常不使用 bias
        pos_enc_type="rope",               # "rope" | "learned"
        norm_type="rmsnorm",               # "rmsnorm" | "layernorm"
        ffn_type="swiglu",                 # "swiglu" | "gelu"
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
        self.norm_type = norm_type
        self.ffn_type = ffn_type

    def to_dict(self):
        return {
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_head': self.n_head, 'n_layer': self.n_layer,
            'block_size': self.block_size, 'dropout': self.dropout,
            'bias': self.bias, 'pos_enc_type': self.pos_enc_type,
            'norm_type': self.norm_type, 'ffn_type': self.ffn_type,
        }


# ============================================================
# 2. RMSNorm — Root Mean Square Layer Normalization
# ============================================================
class RMSNorm(nn.Module):
    """
    RMS 归一化 / Root Mean Square Normalization

    原理 / Principle:
        RMSNorm(x) = x / sqrt(mean(x²) + ε) × γ

        与 LayerNorm 的关键区别 / Key differences from LayerNorm:
        - 不减去均值 (no mean subtraction): 论文证明减去均值对效果不重要
        - 仅一个可学习缩放参数 γ (no bias β): 更少参数, 更快计算
        - 等价于对输入向量做 L2 归一化后缩放
        - LLaMA, Mistral, Gemma 等主流模型均使用 RMSNorm

    公式 / Formula:
        rms = sqrt(mean(x²) + eps)
        output = x / rms * weight
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算 RMS: sqrt(mean(x²))
        # 使用 float32 进行中间计算以保证精度 / Use float32 for precision
        x_fp32 = x.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        # 归一化 + 缩放
        return (x_fp32 / rms * self.weight.float()).to(x.dtype)


def _create_norm(config):
    """工厂函数: 根据配置创建归一化层 / Factory: create norm layer from config"""
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model)
    else:
        return nn.LayerNorm(config.d_model)


# ============================================================
# 3. 多头因果自注意力 / Multi-Head Causal Self-Attention
# ============================================================
class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力机制 (Multi-Head Causal Self-Attention)

    原理 / Principle:
        支持两种位置编码 (RoPE / learned PE) 和 KV-Cache 推理加速。
        使用 PyTorch 的 F.scaled_dot_product_attention 实现 Flash Attention,
        该函数自动选择最优后端 (FlashAttention-2, Memory-Efficient Attention, 或朴素实现)。
        Uses F.scaled_dot_product_attention which auto-selects the best backend
        (FlashAttention-2, Memory-Efficient, or vanilla).
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0

        # QKV 投影: 一次性计算三者 / Single projection for Q, K, V
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.pos_enc_type = config.pos_enc_type

    def forward(self, x, freqs_cis=None, layer_past=None):
        """
        前向传播 / Forward Pass

        参数 / Args:
            x:          (B, T, d_model)
            freqs_cis:  RoPE 复数频率 (仅 rope 模式)
            layer_past: 历史 KV 缓存 (推理时)

        返回 / Returns:
            y:       (B, T, d_model)
            present: 当前 KV 缓存
        """
        B, T, C = x.size()

        # --- Step 1: QKV 投影 / QKV Projection ---
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # (B, T, C) → (B, T, n_head, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim)
        k = k.view(B, T, self.n_head, head_dim)
        v = v.view(B, T, self.n_head, head_dim)

        # --- Step 2: RoPE (仅 rope 模式) / RoPE (rope mode only) ---
        if self.pos_enc_type == "rope" and freqs_cis is not None:
            try:
                from src.rope import apply_rotary_emb
            except ImportError:
                from rope import apply_rotary_emb
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # --- Step 3: KV-Cache / Concatenate past KV ---
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present = (k, v)

        # --- Step 4: Flash Attention / Fused Attention ---
        # F.scaled_dot_product_attention 原理 / Principle:
        #   PyTorch 2.0+ 内置的融合注意力算子, 自动选择后端:
        #   - CUDA + Turing+: FlashAttention-2 (HBM I/O 优化, O(N²) 显存 → O(N))
        #   - 其他: Memory-Efficient Attention (xformers 风格)
        #   - 回退: 标准实现
        #   Auto-selects backend: FlashAttention-2, Mem-Efficient, or vanilla.
        #
        # 因果掩码策略 / Causal mask strategy:
        #   - 训练或推理第一步 (layer_past is None): is_causal=True
        #     自动生成下三角 mask, 每个 token 只能看到自己和前面的 token。
        #   - 后续推理步 (layer_past 存在, 此时 T=1): is_causal=False
        #     因为只输入了一个新 token, 且已有全部历史的 K,V, 无需 mask。
        is_causal = (layer_past is None)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # --- Step 5: 合并多头 / Merge Heads ---
        # (B, n_head, T, head_dim) → (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影 + 残差 dropout
        y = self.resid_dropout(self.c_proj(y))

        return y, present


# ============================================================
# 4. 前馈神经网络 / Feed-Forward Network (SwiGLU)
# ============================================================
class FFN(nn.Module):
    """
    SwiGLU 前馈神经网络 / SwiGLU Feed-Forward Network

    原理 / Principle:
        SwiGLU = (SiLU(W_gate · x) ⊙ (W_up · x)) · W_down

        与 GELU-FFN 的对比 / Comparison with GELU-FFN:
        - GELU-FFN:  x → W1 → GELU → W2 → x  (2 个权重矩阵, 单路径)
        - SwiGLU:    x → gate ⊙ up → W_down  (3 个权重矩阵, 双路径)

        门控机制 (Gating): gate 支路先过 SiLU (Swish) 激活, 然后与 up 支路
        逐元素相乘。这等价于一个可学习的"开关", 控制哪些信息通过。
        Gate branch passes through SiLU activation, then element-wise multiplies
        with the up branch — a learnable "switch" controlling information flow.

        参数量平衡 / Parameter count balancing:
        - GELU: 2 × d_model × 4d_model = 8 × d_model²
        - SwiGLU: 3 × d_model × hidden_dim
        - 令 hidden_dim ≈ 8/3 × d_model 保持参数规模近似
          hidden_dim ≈ 8/3 × d_model keeps param count roughly equal

    为什么 SwiGLU 更好 / Why SwiGLU is better:
        PaLM, LLaMA 等实验证明 SwiGLU 在下游任务上一致优于 GELU,
        且计算量相近。门控机制提供了更强的非线性表达能力。
        PaLM, LLaMA show consistent gains over GELU with similar FLOPs.
    """

    def __init__(self, config):
        super().__init__()
        # hidden_dim = 8/3 * d_model, 取整到 256 的倍数以便 GPU 高效计算
        # hidden_dim = 8/3 * d_model, rounded to multiple of 256 for efficiency
        if config.ffn_type == "swiglu":
            hidden_dim = int(8 * config.d_model / 3)
            hidden_dim = max(256, ((hidden_dim + 255) // 256) * 256)

            self.w_gate = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
            self.w_up   = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
            self.w_down = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        else:
            # GELU FFN (回退 / fallback): d_model → 4d_model → d_model
            self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)
        self.ffn_type = config.ffn_type

    def forward(self, x):
        """
        前向传播 / Forward Pass
        x: (B, T, d_model) → y: (B, T, d_model)
        """
        if self.ffn_type == "swiglu":
            # SiLU(gate) ⊙ up → down
            # SiLU = x * sigmoid(x), 光滑的 Swish 激活
            gate = F.silu(self.w_gate(x))
            up = self.w_up(x)
            x = self.w_down(gate * up)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)

        x = self.dropout(x)
        return x


# ============================================================
# 5. Transformer Block / 解码器层
# ============================================================
class Block(nn.Module):
    """
    Transformer 解码器层 (Decoder Block)

    Pre-Norm 架构 / Pre-Norm Architecture:
        x = x + Attention(Norm(x))
        x = x + FFN(Norm(x))
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = _create_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _create_norm(config)
        self.ffn = FFN(config)

    def forward(self, x, freqs_cis=None, layer_past=None):
        attn_out, present = self.attn(self.ln_1(x), freqs_cis=freqs_cis, layer_past=layer_past)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, present


# ============================================================
# 6. GPT 模型主体 / GPT Model Body
# ============================================================
class GPT(nn.Module):
    """
    微型 GPT 语言模型 / Mini GPT Language Model

    架构总览 / Architecture:
        Token IDs [B, T]
          → Token Embedding + Position Encoding
          → Block × n_layer (RMSNorm → Attn → + → RMSNorm → SwiGLU → +)
          → Final RMSNorm
          → LM Head (weight-tied)
          → Logits [B, T, vocab_size]
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=_create_norm(config),
        ))

        # 位置编码 / Position Encoding
        if config.pos_enc_type == "learned":
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.d_model)
        elif config.pos_enc_type == "rope":
            head_dim = config.d_model // config.n_head
            try:
                from src.rope import precompute_freqs_cis
            except ImportError:
                from rope import precompute_freqs_cis
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(head_dim, config.block_size * 2)
            )
        else:
            raise ValueError(f"未知 pos_enc_type: {config.pos_enc_type}")

        # LM Head (权重绑定 / weight tying)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化 / Initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w_down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        # RMSNorm 的 weight 初始化为 1.0 (默认), 无需额外处理

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_pos_encoding(self, device, t, past_length=0):
        if self.config.pos_enc_type == "learned":
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
            if pos.max() >= self.config.block_size:
                raise ValueError(f"位置 {pos.max()} 超出 block_size {self.config.block_size}")
            return self.transformer['wpe'](pos)
        elif self.config.pos_enc_type == "rope":
            return self.freqs_cis[past_length: past_length + t].to(device)
        else:
            raise ValueError(f"未知 pos_enc_type: {self.config.pos_enc_type}")

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"输入长度 {t} 超过 block_size {self.config.block_size}"

        past_length = past_kv[0][0].size(2) if past_kv is not None else 0
        pos_enc = self._get_pos_encoding(device, t, past_length)

        tok_emb = self.transformer.wte(idx)
        if self.config.pos_enc_type == "learned":
            x = self.transformer.drop(tok_emb + pos_enc)
            freqs_cis = None
        else:
            x = self.transformer.drop(tok_emb)
            freqs_cis = pos_enc

        new_kv = []
        for i, block in enumerate(self.transformer.h):
            layer_past = past_kv[i] if past_kv is not None else None
            x, layer_present = block(x, freqs_cis=freqs_cis, layer_past=layer_past)
            if use_cache:
                new_kv.append(layer_present)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss, (new_kv if use_cache else None)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        past_kv = None
        for _ in range(max_new_tokens):
            if idx.size(1) >= self.config.block_size:
                break
            if past_kv is None:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            else:
                idx_cond = idx[:, -1:]
            logits, _, past_kv = self(idx_cond, past_kv=past_kv, use_cache=True)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if eos_id is not None and idx_next.item() == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================
# 7. 自动化验证脚本 / Automated Verification Suite
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(1337)
    device = 'cpu'
    n_test_passed = 0
    n_test_total = 0

    def check(condition, msg):
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
    print("  Mini-GPT 模型验证 / Model Verification")
    print(f"  RMSNorm + SwiGLU + Flash Attention")
    print("=" * 60)

    # ================================================================
    # Test 1: 基础模型 (learned PE + LayerNorm + GELU, 向后兼容)
    # ================================================================
    section("Test 1: Learned PE + LayerNorm + GELU (backward compat)")

    conf1 = GPTConfig(
        vocab_size=100, d_model=32, n_head=4, n_layer=2,
        block_size=32, dropout=0.0,
        pos_enc_type="learned", norm_type="layernorm", ffn_type="gelu"
    )
    model1 = GPT(conf1).to(device).eval()
    check(hasattr(model1.transformer, 'wpe'), "wpe exists")
    check(isinstance(model1.transformer.ln_f, nn.LayerNorm), "LayerNorm used")

    x1 = torch.randint(0, 100, (2, 10)).to(device)
    logits, loss, kv = model1(x1, use_cache=False)
    check(logits.shape == (2, 10, 100), f"Logits shape: {logits.shape}")
    check(model1.transformer.wte.weight is model1.lm_head.weight, "Weight tying")

    start1 = torch.zeros((1, 1), dtype=torch.long)
    gen1 = model1.generate(start1, max_new_tokens=5)
    check(gen1.shape == (1, 6), f"Generation: {gen1.shape}")

    # ================================================================
    # Test 2: Learned PE + KV-Cache 一致性 (GELU + LayerNorm)
    # ================================================================
    section("Test 2: Learned PE + KV-Cache Consistency")

    conf2 = GPTConfig(
        vocab_size=100, d_model=64, n_head=4, n_layer=2,
        block_size=32, dropout=0.0,
        pos_enc_type="learned", norm_type="layernorm", ffn_type="gelu"
    )
    model2 = GPT(conf2).to(device).eval()

    x_full = torch.randint(0, 100, (1, 10)).to(device)
    logits_ref, _, _ = model2(x_full, use_cache=False)
    last_logit_ref = logits_ref[:, -1, :]

    _, _, past_kv = model2(x_full[:, :9], use_cache=True)
    logits_step, _, _ = model2(x_full[:, 9:], past_kv=past_kv, use_cache=True)
    last_logit_step = logits_step[:, -1, :]

    diff = (last_logit_ref - last_logit_step).abs().max().item()
    check(diff < 1e-5, f"KV-cache consistency (diff: {diff:.2e})")

    # ================================================================
    # Test 3: RoPE + RMSNorm + SwiGLU + KV-Cache 一致性 (新默认)
    # ================================================================
    section("Test 3: RoPE + RMSNorm + SwiGLU + KV-Cache Consistency")

    conf3 = GPTConfig(
        vocab_size=100, d_model=64, n_head=4, n_layer=2,
        block_size=32, dropout=0.0,
        pos_enc_type="rope", norm_type="rmsnorm", ffn_type="swiglu"
    )
    model3 = GPT(conf3).to(device).eval()
    check(not hasattr(model3.transformer, 'wpe'), "wpe removed (rope mode)")
    check(hasattr(model3, 'freqs_cis'), "freqs_cis buffer exists")
    check(isinstance(model3.transformer.ln_f, RMSNorm), "RMSNorm is final norm")
    check(model3.transformer.h[0].ffn.ffn_type == "swiglu", "SwiGLU is FFN type")

    x_full3 = torch.randint(0, 100, (1, 10)).to(device)
    logits_ref3, _, _ = model3(x_full3, use_cache=False)
    last_logit_ref3 = logits_ref3[:, -1, :]

    _, _, past_kv3 = model3(x_full3[:, :9], use_cache=True)
    logits_step3, _, _ = model3(x_full3[:, 9:], past_kv=past_kv3, use_cache=True)
    last_logit_step3 = logits_step3[:, -1, :]

    diff3 = (last_logit_ref3 - last_logit_step3).abs().max().item()
    check(diff3 < 1e-5, f"RoPE + KV-cache consistency (diff: {diff3:.2e})")

    # ================================================================
    # Test 4: Flash Attention 训练正确性 (无 NaN)
    # ================================================================
    section("Test 4: Flash Attention — Training Forward (no NaN)")

    conf4 = GPTConfig(
        vocab_size=100, d_model=64, n_head=4, n_layer=2,
        block_size=32, dropout=0.1,   # 训练模式 with dropout
        pos_enc_type="rope", norm_type="rmsnorm", ffn_type="swiglu"
    )
    model4 = GPT(conf4).to(device)
    model4.train()  # 训练模式: dropout 激活

    x4 = torch.randint(0, 100, (2, 16)).to(device)
    y4 = torch.randint(0, 100, (2, 16)).to(device)
    _, loss4, _ = model4(x4, targets=y4)
    check(not torch.isnan(loss4), f"Loss not NaN: {loss4.item():.4f}")
    check(not torch.isinf(loss4), f"Loss not Inf: {loss4.item():.4f}")

    # 反向传播: 验证梯度正常 / Backward: verify gradients are healthy
    loss4.backward()
    grad_norm = sum(p.grad.norm().item() for p in model4.parameters() if p.grad is not None)
    check(not math.isnan(grad_norm), f"Gradients not NaN (norm: {grad_norm:.4f})")
    check(grad_norm < 100, f"Gradients not exploding (norm: {grad_norm:.4f})")

    # ================================================================
    # Test 5: 生成冒烟测试
    # ================================================================
    section("Test 5: Generation Smoke Test")
    model4.eval()
    start5 = torch.zeros((1, 1), dtype=torch.long).to(device)
    gen5 = model4.generate(start5, max_new_tokens=5)
    check(gen5.shape == (1, 6), f"Generation shape: {gen5.shape}")

    # ================================================================
    # 结果
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  {n_test_passed}/{n_test_total} PASS")
    if n_test_passed == n_test_total:
        print("  All tests passed!")
    else:
        print(f"  {n_test_total - n_test_passed} FAILED!")
    print(f"{'=' * 60}")
