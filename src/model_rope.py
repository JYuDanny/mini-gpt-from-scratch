import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. RoPE 核心辅助函数
# ==========================================
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转位置编码的复数频率张量
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # (end, dim//2)
    # 转换为复数形式: cos(x) + i*sin(x)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    应用 RoPE 旋转
    xq, xk 形状: (B, T, n_head, head_dim)
    freqs_cis 形状: (T, head_dim/2)
    """
    # 将最后维度变形成复数形式 (..., head_dim/2, 2) -> (..., head_dim/2) complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 调整 freqs_cis 形状以支持广播: (T, head_dim/2) -> (1, T, 1, head_dim/2)
    freqs_cis = freqs_cis.view(1, xq.size(1), 1, xq_.size(-1))

    # 复数乘法 (旋转)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# ==========================================
# 2. 模型定义
# ==========================================
class GPTConfig:
    def __init__(self, vocab_size=50257, d_model=512, n_head=8, n_layer=6,
                 block_size=1024, dropout=0.1, bias=True, **kwargs):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        # 注意：RoPE 不需要 mask 这里的 bias 变量，但为了因果遮蔽仍需保留下三角 mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, freqs_cis, layer_past=None):
        B, T, C = x.size()

        # 1. QKV 投影
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # 2. 变形为 (B, T, n_head, head_dim) 以便应用 RoPE
        # 注意这里不再先 transpose(1,2)，为了方便后续 view_as_complex
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim)
        k = k.view(B, T, self.n_head, head_dim)
        v = v.view(B, T, self.n_head, head_dim)

        # 3. 应用 RoPE (只对当前的 q, k 进行旋转)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 4. 转换回 Multi-head attention 标准形状 (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. KV-Cache 拼接 (注意：k 已经是旋转过的了)
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present = (k, v)

        # 6. Attention 计算
        # (B, nh, T, hs) x (B, nh, hs, T_total) -> (B, nh, T, T_total)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if layer_past is None:
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present

class FFN(nn.Module):
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
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x, freqs_cis, layer_past=None):
        # 必须透传 freqs_cis
        attn_out, present = self.attn(self.ln_1(x), freqs_cis=freqs_cis, layer_past=layer_past)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, present

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            # [RoPE 修改点 1] 移除了 wpe (Positional Embedding)
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

        # 特殊初始化：对残差投影层进行缩放 (1/sqrt(2 * n_layer))
        # 目的是在深层网络中控制方差增长，防止残差信号逐层放大
        # Special init: scale residual projection layers to control variance
        # growth in deep networks, preventing residual signal amplification
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # [RoPE 修改点 2] 预计算 RoPE 频率表
        head_dim = config.d_model // config.n_head
        # 为了支持外推，我们计算 2 倍 block_size 的长度，或者直接用 block_size
        self.register_buffer("freqs_cis", precompute_freqs_cis(head_dim, config.block_size * 2))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        device = idx.device
        b, t = idx.size()

        # [RoPE 修改点 3] 动态获取对应的频率
        if past_kv is not None:
            # 推理模式：只取当前这一步对应的频率
            past_length = past_kv[0][0].size(2)
            freqs_cis = self.freqs_cis[past_length : past_length + t]
        else:
            # 训练模式：取前 t 个频率
            freqs_cis = self.freqs_cis[:t]

        # 1. Embedding (不再加 wpe)
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        # 2. Forward pass with RoPE
        new_kv = []
        for i, block in enumerate(self.transformer.h):
            layer_past = past_kv[i] if past_kv is not None else None
            # 传入 freqs_cis
            x, layer_present = block(x, freqs_cis, layer_past=layer_past)
            if use_cache:
                new_kv.append(layer_present)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
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


# ==========================================
# 3. RoPE 正确性验证与单元测试
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(1337)
    device = 'cpu' # 使用 CPU 验证以避免非确定性

    print("=" * 60)
    print("Running RoPE & KV-Cache Implementation Checks...")
    print("=" * 60)

    # 1. 实例化测试
    config = GPTConfig(
        vocab_size=100, d_model=64, n_head=4, n_layer=2, 
        block_size=32, dropout=0.0 # 关闭 dropout
    )
    model = GPT(config).to(device)
    model.eval()

    # 检查 wpe 是否已删除
    if hasattr(model.transformer, 'wpe'):
        print("[FAIL] 'wpe' (Absolute Embedding) still exists in model.transformer!")
        exit()
    else:
        print("[PASS] Absolute position embedding 'wpe' successfully removed.")

    # 检查 freqs_cis 是否存在
    if hasattr(model, 'freqs_cis'):
        print(f"[PASS] RoPE frequencies 'freqs_cis' initialized with shape {model.freqs_cis.shape}.")
    else:
        print("[FAIL] 'freqs_cis' not found.")
        exit()

    # 2. RoPE + KV-Cache 一致性测试 (核心测试)
    # 目的：验证 "一次性输入整个序列" 和 "使用Cache逐个输入" 结果是否完全一致
    # 如果 RoPE 的 slicing 逻辑错了，或者相对位置计算错了，这里必然报错

    print("\n[Running Consistency Check] Parallel Forward vs. Step-by-Step with Cache...")

    seq_len = 10
    x = torch.randint(0, 100, (1, seq_len)).to(device) # Batch=1, Len=10

    # --- 场景 A: Parallel (训练模式) ---
    # 模拟一次性把句子喂给模型
    with torch.no_grad():
        logits_parallel, _, _ = model(x, use_cache=False)
    # 取出最后一个 token 的 logit
    last_logit_parallel = logits_parallel[:, -1, :]

    # --- 场景 B: Sequential (推理模式) ---
    # 1. 先喂前 9 个
    x_past = x[:, :-1]
    with torch.no_grad():
        _, _, past_kv = model(x_past, use_cache=True)

    # 2. 再喂第 10 个 (带上 cache)
    # 关键点：此时模型内部必须正确识别这是第 10 个位置，并使用 index=9 的 freqs_cis
    x_current = x[:, -1:]
    with torch.no_grad():
        logits_step, _, _ = model(x_current, past_kv=past_kv, use_cache=True)
    last_logit_step = logits_step[:, -1, :]

    # 比较两者差异
    diff = (last_logit_parallel - last_logit_step).abs().max().item()

    if diff < 1e-5:
        print(f"[PASS] Consistency Check Passed! Max Diff: {diff:.2e}")
        print("       这证明 RoPE 旋转角度在 Cache 模式下是对齐的。")
    else:
        print(f"[FAIL] Consistency Check Failed. Max Diff: {diff}")
        print("       提示：请检查 forward 中 freqs_cis 的切片逻辑。")
        exit()

    # 3. 生成函数冒烟测试
    print("\n[Running Generation Smoke Test]...")
    try:
        idx = torch.zeros((1, 1), dtype=torch.long)
        out = model.generate(idx, max_new_tokens=5)
        print(f"[PASS] Generation successful. Output shape: {out.shape}")
    except Exception as e:
        print(f"[FAIL] Generation crashed: {e}")
        exit()

    print("=" * 60)
    print("All RoPE checks passed. Your Mini-GPT is now powered by Rotary Embeddings!")
    print("=" * 60)
