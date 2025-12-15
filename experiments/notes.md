# mini-gpt 项目解读

```shell
conda create -n mini-gpt python=3.11 -y

conda activate mini-gpt

nvidia-smi

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126

pip install tiktoken tqdm numpy

pip install datasets

python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU count:', torch.cuda.device_count())"
```

## Part 1. Tokenizer and Dataset

### 1.1 Tokenizer Wrapper

`src/tokenizer.py`

Tokenizer wrapper。这部分代码会封装 tiktoken（OpenAI 的 BPE tokenizer），因为 TinyShakespeare 是英文，我们用 GPT-2 的预训练 tokenizer（vocab_size ~50k，支持英文子词分词）。这能模拟大模型的 token 处理，而非简单字符级。

Tokenizer 是 LLM 的入口，将 raw text 转成数字序列（tokens），供 embedding 用。为什么 BPE？英文单词变体多（如 "run", "running"），字符级 vocab 太小（~256）会导致长序列（内存爆），词级 vocab 太大（百万级，稀疏）。BPE 平衡：从字节开始，迭代合并高频对（如 "th" → 新 token），最终 vocab ~50k。输出中，"Hello" 是 15496（常见词直接一个 token），"world!" 是 995 + 0（"world" + "!" 分开）。assert 检查无损 round-trip，确保模型输入输出一致。这在训练中关键：loss 计算基于 token IDs。实际大模型如 GPT-4 用更大 vocab + 自定义 BPE 来优化压缩率（tokens/字 更低）。

### 1.2 Dataset Class

`src/dataset.py`

这部分实现数据加载器，用于将 TinyShakespeare 文本转成 PyTorch Dataset，便于训练时的 batching 和 shifting（自回归关键）。我们严格遵循 GPT-3 的流程：tokenize 后，创建 (context, target) pairs，其中 target 是 context 右移一位（next-token prediction）。代码会用 DataLoader 包装，确保高效迭代。
代码放 src/dataset.py。我们会用 torch.utils.data.Dataset 基类，保持 mini 但不缺省：支持 block_size（上下文长度，GPT-3 用 2048，但我们从小开始如 128），随机采样序列片段（像 GPT-3 的训练方式，避免顺序 bias）。

## Part 2. Transformer Framework

实现 Multi-Head Attention，包括 Causal Self-Attention。这是 Transformer 的核心，严格遵循 GPT-3 的设计（scaled dot-product attention，多头，causal mask 用于自回归）。我们不缺省任何部分：包括投影、mask、softmax、dropout（虽 mini，但加 dropout 以匹配 GPT-3 的训练稳定）。
节奏：今天分成两小部分 - (1) 单头 Attention 函数（基础），(2) MultiHeadAttention 类（完整）。代码放 src/model.py（从这里开始建模型文件）。每部分后验证。
用小维度测试（d_model=16, heads=2），确保 shape 和 mask 工作。

### 2.1 Causal Attention Function

`src/model.py`

Causal self-attention 是 GPT-3 decoder 的基石。Q (query) 表示“当前 token 想找什么”，K (key) 是“所有 token 的钥匙”，V (value) 是“内容”。scores = Q K^T / sqrt(dk) 计算 cosine 相似（除 sqrt(dk) 防 variance 过大，导致 softmax 梯度小）。Mask 用上三角 -inf，确保位置 i 只见 1..i（自回归：生成时不泄露未来）。Softmax 转概率，output = attn_weights V 聚合上下文。输出中，第一位置只 self-attend（mask 挡住后位），后续位置渐多信息。这模拟生成：prompt "Hello" 时，只用末尾 hidden 预测 next。GPT-3 用此堆栈多层，捕捉层级依赖（浅层语法，深层语义）。随机测试确认数值稳定，无 nan。

### 2.2 Multihead Attention Class

`src/model.py MultiHeadAttention class`

MultiHeadAttention 类，整合了多头机制、投影层和 dropout，严格匹配 GPT-3 的实现（多头并行子空间，bias=False 以简化但不影响本质，dropout for 训练稳定）。我们用 Linear 层高效投影 QKV（一次性计算），reshape/permute 处理 heads。

MultiHeadAttention 是 GPT-3 attention 的精确实现：单 Linear 投影 QKV（效率高，参数 3*d^2），reshape 到 (B, h, N, dk) 允许并行计算 h 个头（torch matmul 自动 batch）。每个头在 d_model/h 子空间独立 attend，捕捉不同关系（e.g., head1 关注邻近 token，head2 远距）。Mask 扩展到 heads（unsqueeze(1)），确保所有头 causal。Dropout 在 attn_weights 后防 overfit（如 GPT-3 的 0.1 rate）。Out_proj 融合 heads（参数 d^2），输出同输入 shape，便于残差。测试中，output[0,0,:] 只从 x[0,0,:] 投影而来（mask 挡住 [1:]），体现自回归：生成时，new token 只用历史。GPT-3 用 96 层这种 attention 堆栈，参数爆炸但梯度流畅。这在 mini 模型中也能看到：小 d_model 时，heads 多样化提升表达力。

### 2.3 Transformer Block Class

`src/model.py Block class`

严格遵循 GPT-3 的 decoder-only 架构：每个 block 是 attention + residual + norm + FFN + residual + norm，多层堆栈，位置编码用 learned embedding（GPT-3 用 absolute learned PE）。

Transformer Block 是 GPT-3 的基本单元（96层堆栈）。Attention 捕捉依赖，FFN（position-wise MLP，d_ff~4*d_model）扩充非线性容量（GELU 如 GPT-3，近似 ReLU 但平滑梯度）。Residual (x + sublayer(x)) 来自 ResNet，允许深层（防 vanishing gradients：梯度直通）。LayerNorm 标准化每个 token 的特征（mean=0, var=1），防 internal covariate shift，GPT-3 用 RMSNorm 变体但原理同（无 mean，只 scale）。Post-norm (sublayer 后 norm) 如原论文，pre-norm (前) 如 GPT 更稳定，但 mini 无差。Dropout 在 attn/FFN/residual 后随机丢特征，防 co-adaptation，提升泛化。测试中，output 保持 shape，值变化反映信息聚合；多层时，这累积抽象表示（低层词级，高层语义）。GPT-3 参数主要在此（per-layer ~ d^2 * 12）。

### 2.4 GPT Class

`src/model.py GPT class`

完成模型主体，严格遵循 GPT-3 的 decoder-only 流程：token + pos embedding，堆栈多层 Block，causal mask 在 forward 中生成（全局应用），LM head 线性投影到 vocab（GPT-3 常共享 embedding 权重，但 mini 先独立以简单）。我们加 generate 方法模拟推理。

GPT 类封装 GPT-3 完整流程：token_emb 将 IDs 转向量（one-hot * weight），pos_emb 加绝对位置信号（learned，如 GPT-3，防 attention permutation-invariant）。Layers 堆栈 Block，逐层精炼表示（累积上下文）。Mask 在 forward 统一生成，确保所有层 causal（生成时不需额外 mask）。Final LN 稳定输出分布，lm_head 投影到 vocab logits（B,N,vocab），训练时用所有位置 loss。Generate 模拟推理：截取 sliding window (block_size)，取末尾 logit，temperature scale 控制多样性（1.0=标准，>1 随机，<1 保守）。测试中，logits 值小因初始化（防 exploding），generate 扩展 3 token 证明 autoregressive 循环工作。GPT-3 参数 ~175B 主在此（layers * (attn + ffn)），mini 规模小但原理同：训练后，能从 prompt 生成 coherent 文本。

## Part 3. Train + Generate

### 3.1 Train

`train/train.py`

实现训练脚本（train.py）和生成函数完善。严格遵循 GPT-3 训练流程：用 AdamW optimizer，cross-entropy loss（忽略 -100 labels，但 mini 无需），warmup scheduler 可选（mini 简化）。生成函数已在 GPT 类，我们在 train.py 加 eval 生成样例。

完整 train.py 镜像 GPT-3 pretrain：train loop per-batch 优化（zero_grad→loss→backward→step），掌控变量如 loss (CE: -log P(true_token)) 量化预测准度，flatten 并行所有位置）。Eval 用 no_grad 防 grad 累积，平均 val_loss 监控泛化（低=好，高=overfit）。Checkpoint 保存 state_dict（权重），如 GPT-3 checkpointing 防中断。Generate 在 eval 用低 temp 更 deterministic，展示学到模式：初始乱因随机 init，epoch 后捕捉 n-gram/shakespeare 风格。Wandb 可视化 loss 曲线，帮 debug（如 plateau 加 lr decay）。这核心 GPT-3：规模化迭代 + eval = 能力涌现。Mini 版参数 ~10M，训 5 epoch 后 loss ~3，能生成短 coherent 文本。

`train/train_gpt.py`

训练流程是 GPT-3 的核心：自回归 loss 教模型 P(next|context)，AdamW 通过梯度更新权重，残差/LN 确保深层稳定。全局步 global_step 掌控 eval 时机，val_loss 作为 early-stop 代理（虽无实现，但 best 保存基于它）。进度条 update(1) 确保视觉流畅。loss 下降表示模型学到 n-gram 模式（高 loss=随机预测，低=上下文相关）。Generated 样例可视化进步：初始 perplexity 高 (e^loss ~ thousands)，后期低 (~20)，体现涌现（mini 规模有限，但原理同）。如果 loss 爆炸，init_weights 防之；不降，调 lr。

### 3.2 Generate

`generate.py`

generate.py 解耦推理：load_state_dict 恢复权重，eval() 关 dropout。Generate 循环 forward 只取末尾 logit，multinomial 采样模拟 beam/greedy（GPT-3 用 top-p/top-k 扩展，但 mini 简单）。Temperature scale logits 防 greedy (temp=0=argmax, >1=随机)。这体现 LLM 部署：训练后推理高效，低内存。
