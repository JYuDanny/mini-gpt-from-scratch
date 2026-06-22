# Mini-GPT from Scratch / 从零训练 Mini-GPT

> 从零构建一个微型 GPT 语言模型，严格遵循 GPT-3 的 Decoder-only Transformer 架构。
> 本项目兼顾**教学**与**实战**：代码中有完整的中文教学注释，帮助理解每个组件的原理与数据流动。
>
> A tiny GPT language model built from scratch, strictly following GPT-3's decoder-only transformer architecture.
> This project serves both **education** and **practice**: complete Chinese teaching annotations in code explain the principles and data flow of every component.

---

## 目录 / Table of Contents

- [项目概览 / Overview](#项目概览--overview)
- [硬件要求 / Hardware Requirements](#硬件要求--hardware-requirements)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [项目结构 / Project Structure](#项目结构--project-structure)
- [训练数据 / Training Data](#训练数据--training-data)
- [模型配置 / Configuration](#模型配置--configuration)
- [模型验证 / Model Verification](#模型验证--model-verification)
- [模型版本 / Model Variants](#模型版本--model-variants)
- [参考资料 / References](#参考资料--references)

---

## 项目概览 / Overview

本项目从零实现了完整的 GPT 风格语言模型训练管线：分词器 (tokenizer)、数据加载 (data loader)、模型架构 (model architecture)、训练循环 (training loop) 和推理生成 (inference)。核心目标是亲身理解大语言模型在底层是如何工作的。

| 组件 / Component | 实现 / Implementation |
|-----------|---------------|
| 分词器 / Tokenizer | GPT-2 BPE (tiktoken, ~50K vocab) |
| 位置编码 / Position Encoding | RoPE (旋转位置编码 / Rotary Position Embedding) |
| 注意力机制 / Attention | 多头因果自注意力 + KV-Cache / Multi-Head Causal Self-Attention |
| 激活函数 / FFN Activation | GELU (与 GPT-3 一致) |
| 归一化 / Normalization | Pre-Norm LayerNorm |
| 权重绑定 / Weight Tying | LM Head 与 Token Embedding 共享权重 |
| 默认规模 / Default Size | ~45M 参数 (可配置 / configurable) |

### 模型版本 / Model Variants

三个渐进式实现，每个均可独立运行和自测：

| 文件 / File | 位置编码 / Pos Encoding | KV-Cache | 用途 / Purpose |
|------|-------------------|----------|---------|
| `model.py` | 可学习绝对位置编码 / Learned absolute | 无 / No | 基线 GPT-3 复刻 |
| `model_kvcache.py` | 可学习绝对位置编码 / Learned absolute | 有 / Yes | 高效推理 |
| `model_rope.py` | RoPE 旋转位置编码 | 有 / Yes | 更好的长度泛化 / Better length generalization |

当前训练脚本默认使用 `model_rope.py`。

---

## 硬件要求 / Hardware Requirements

| 项目 | 规格 |
|------|------|
| 操作系统 / OS | Windows 11 |
| GPU | NVIDIA RTX 4060 8GB |
| RAM | 16GB |
| 训练数据 / Training Data | TinyStories (~200MB), 中文维基 (~1.5GB) 等 |

---

## 快速开始 / Quick Start

### 1. 环境配置 / Environment Setup

```bash
# 创建并激活 conda 环境 (一次性操作 / one-time setup)
conda create -n mini-gpt python=3.11 -y
conda activate mini-gpt

# 安装 PyTorch (CUDA 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装其他依赖 / Install other dependencies
pip install tiktoken tqdm numpy datasets pyyaml

# 验证安装 / Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 2. 准备训练数据 / Prepare Training Data

```bash
python src/prepare_data.py
```

此命令下载数据集并保存分词后的二进制文件到 `data/` 目录。详见 [训练数据](#训练数据--training-data) 章节。

### 3. 训练模型 / Train the Model

```bash
# 使用默认配置训练 / Train with default config
python train.py --config configs/mini_gpt.yaml

# 覆盖特定参数 / Override parameters
python train.py --config configs/mini_gpt.yaml --lr 1e-4 --batch_size 16

# 从 checkpoint 续训 / Resume from checkpoint
python train.py --config configs/mini_gpt.yaml --resume checkpoints/mini_gpt/ckpt_latest.pt
```

### 4. 生成文本 / Generate Text

```bash
# 交互式输入 prompt / Interactive prompt
python generate_rope.py --ckpt checkpoints/mini_gpt_rope/ckpt_best.pt

# 指定 prompt / With a specific prompt
python generate_rope.py --ckpt checkpoints/mini_gpt_rope/ckpt_best.pt --prompt "Once upon a time"
```

---

## 项目结构 / Project Structure

```
mini-gpt-from-scratch/
├── AGENTS.md                  # AI 助手工作规范 / AI assistant instructions
├── README.md                  # 项目文档 (本文件)
├── requirements.txt           # Python 依赖列表
├── train.py                   # 训练入口脚本 / Training entry point
├── generate.py                # 推理脚本 (KV-cache 模型)
├── generate_rope.py           # 推理脚本 (RoPE 模型)
├── configs/
│   └── mini_gpt.yaml          # 训练与模型配置文件
├── src/
│   ├── model.py               # 基础 GPT (绝对位置编码, 无缓存)
│   ├── model_kvcache.py       # GPT + KV-Cache 推理加速
│   ├── model_rope.py          # GPT + RoPE + KV-Cache (当前主力)
│   ├── dataset.py             # 二进制数据集加载器 (numpy memmap)
│   ├── prepare_data.py        # 数据下载与分词预处理
│   └── tokenizer.py           # GPT-2 BPE 分词器封装
└── experiments/
    ├── transformer.md         # Transformer 理论深度解析
    └── notes.md               # 开发日志与代码走读
```

---

## 训练数据 / Training Data

本项目支持多种训练数据集，以扩展模型的能力边界（不局限于故事续写）。数据文件存储在 `data/` 目录，**不纳入 Git 版本控制**。

### 可用数据集 / Available Datasets

| 数据集 / Dataset | 规模 / Size | 语言 / Language | 内容类型 / Content | 用途 / Use Case |
|------|-------|----------|----------|----------|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | ~200MB 分词后 | 英文 / English | 儿童故事 / Children's stories | 快速验证架构、故事续写 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 极大 (TB级) | 英文 / English | 高质量网页文本 / High-quality web text | 通用知识、广泛能力 |
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) | ~180MB | 英文 / English | 维基百科文章 / Wikipedia articles | 知识密集型、事实性文本 |
| [Chinese Wikipedia / 中文维基百科](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered) | ~1.5GB | 中文 / Chinese | 中文维基百科 / Chinese Wikipedia | 中文语言能力 |
| [C4 (Colossal Clean Crawled Corpus)](https://huggingface.co/datasets/allenai/c4) | 极大 (TB级) | 英文 / English | 清洗后的网页文本 / Cleaned web text | 通用预训练 |
| [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) | ~40GB | 英文 / English | Reddit 高质量链接文本 | 通用预训练 (GPT-2 复现) |
| [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) | ~5GB | 英文 / English | 未出版书籍 / Unpublished books | 长文本、叙事能力 |
| [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | 极大 (600B+ tokens) | 英文 / English | RedPajama 清洗子集 | 大规模通用预训练 |

### 数据准备流程 / Data Preparation

1. 选择数据集，修改 `src/prepare_data.py` 中的配置
2. 确认数据集下载后硬盘空间充足
3. 运行 `python src/prepare_data.py` 完成下载和分词
4. 分词后的 `.bin` 文件自动保存在 `data/` 目录

> **注意 / Note:** `data/` 和 `checkpoints/` 目录已在 `.gitignore` 中排除。

---

## 模型配置 / Configuration

训练通过 `configs/mini_gpt.yaml` 配置。配置优先级：命令行参数 > YAML 文件 > 代码默认值。

| 分类 / Section | 参数 / Parameter | 默认值 / Default | 说明 / Description |
|---------|-----------|---------|-------------|
| model | `d_model` | 768 | 嵌入维度 / Embedding dimension |
| model | `n_layer` | 6 | Transformer 层数 / Number of layers |
| model | `n_head` | 6 | 注意力头数 / Number of attention heads |
| model | `block_size` | 256 | 最大上下文长度 / Max context length |
| model | `dropout` | 0.1 | Dropout 比例 |
| optimizer | `learning_rate` | 3e-4 | 峰值学习率 / Peak learning rate |
| optimizer | `weight_decay` | 1e-2 | 权重衰减 / Weight decay |
| trainer | `max_iters` | 20000 | 总训练步数 / Total training steps |
| trainer | `eval_interval` | 1000 | 评估间隔 / Steps between evaluations |

---

## 模型验证 / Model Verification

每个模型文件内置自测，用于验证架构和张量形状的正确性：

```bash
python src/model.py            # 基础 GPT 架构测试 / Base GPT architecture test
python src/model_kvcache.py    # KV-cache 一致性测试 / KV-cache consistency test
python src/model_rope.py       # RoPE + KV-cache 一致性测试 / RoPE + KV-cache consistency test
python src/tokenizer.py        # 分词器往返测试 / Tokenizer roundtrip test
```

---

## 参考资料 / References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3) — Brown et al., 2020
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
