# mini-gpt 项目解读

```shell
conda create -n mini-gpt python=3.11 -y

conda activate mini-gpt

nvidia-smi

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126

pip install tiktoken tqdm numpy

python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU count:', torch.cuda.device_count())"
```

## Part 1. Tokenizer and Dataset

### 1.1 Tokenizer Wrapper

`src/tokenizer.py`

Tokenizer wrapper。这部分代码会封装 tiktoken（OpenAI 的 BPE tokenizer），因为 TinyShakespeare 是英文，我们用 GPT-2 的预训练 tokenizer（vocab_size ~50k，支持英文子词分词）。这能模拟大模型的 token 处理，而非简单字符级。

原理深入：Tokenizer 是 LLM 的入口，将 raw text 转成数字序列（tokens），供 embedding 用。为什么 BPE？英文单词变体多（如 "run", "running"），字符级 vocab 太小（~256）会导致长序列（内存爆），词级 vocab 太大（百万级，稀疏）。BPE 平衡：从字节开始，迭代合并高频对（如 "th" → 新 token），最终 vocab ~50k。输出中，"Hello" 是 15496（常见词直接一个 token），"world!" 是 995 + 0（"world" + "!" 分开）。assert 检查无损 round-trip，确保模型输入输出一致。这在训练中关键：loss 计算基于 token IDs。实际大模型如 GPT-4 用更大 vocab + 自定义 BPE 来优化压缩率（tokens/字 更低）。

### 1.2 Dataset Class

`src/dataset.py`

这部分实现数据加载器，用于将 TinyShakespeare 文本转成 PyTorch Dataset，便于训练时的 batching 和 shifting（自回归关键）。我们严格遵循 GPT-3 的流程：tokenize 后，创建 (context, target) pairs，其中 target 是 context 右移一位（next-token prediction）。代码会用 DataLoader 包装，确保高效迭代。
代码放 src/dataset.py。我们会用 torch.utils.data.Dataset 基类，保持 mini 但不缺省：支持 block_size（上下文长度，GPT-3 用 2048，但我们从小开始如 128），随机采样序列片段（像 GPT-3 的训练方式，避免顺序 bias）。

## Part 2. Multi-Head Attention

实现 Multi-Head Attention，包括 Causal Self-Attention。这是 Transformer 的核心，严格遵循 GPT-3 的设计（scaled dot-product attention，多头，causal mask 用于自回归）。我们不缺省任何部分：包括投影、mask、softmax、dropout（虽 mini，但加 dropout 以匹配 GPT-3 的训练稳定）。
节奏：今天分成两小部分 - (1) 单头 Attention 函数（基础），(2) MultiHeadAttention 类（完整）。代码放 src/model.py（从这里开始建模型文件）。每部分后验证。
用小维度测试（d_model=16, heads=2），确保 shape 和 mask 工作。

### Causal Attention Function


