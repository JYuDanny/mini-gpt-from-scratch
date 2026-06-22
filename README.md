# Mini-GPT from Scratch / 从零训练 Mini-GPT

> 从零构建一个微型 GPT 语言模型，采用 LLaMA 风格架构 (RoPE + RMSNorm + SwiGLU + Flash Attention)。
> 本项目兼顾**教学**与**实战**：代码中有完整的中文教学注释，帮助理解每个组件的原理与数据流动。
>
> A tiny GPT language model built from scratch with LLaMA-style architecture.
> This project serves both **education** and **practice**: complete Chinese teaching annotations in code.

---

## 目录 / Table of Contents

- [项目概览 / Overview](#项目概览--overview)
- [硬件要求 / Hardware Requirements](#硬件要求--hardware-requirements)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [项目结构 / Project Structure](#项目结构--project-structure)
- [训练数据 / Training Data](#训练数据--training-data)
- [模型配置 / Configuration](#模型配置--configuration)
- [训练效率优化 / Training Optimizations](#训练效率优化--training-optimizations)
- [模型验证 / Model Verification](#模型验证--model-verification)
- [参考资料 / References](#参考资料--references)

---

## 项目概览 / Overview

本项目从零实现了完整的 GPT 风格语言模型训练管线：分词、数据加载、模型架构、训练循环和推理生成。

| 组件 / Component | 实现 / Implementation |
|-----------|---------------|
| 分词器 / Tokenizer | GPT-2 BPE (tiktoken, ~50K vocab) |
| 位置编码 / Position Encoding | RoPE (旋转位置编码 / Rotary Position Embedding) |
| 归一化 / Normalization | RMSNorm (默认) / LayerNorm (可选) |
| 注意力 / Attention | Flash Attention (`F.scaled_dot_product_attention`) + KV-Cache |
| FFN 激活 / FFN Activation | SwiGLU (默认) / GELU (可选) |
| 权重绑定 / Weight Tying | LM Head 与 Token Embedding 共享权重 |
| 训练优化 / Training Opt | AMP (bfloat16) + torch.compile + 梯度累积 + TensorBoard |

架构默认值约 ~81M 参数 (d_model=768, n_layer=6, n_head=6)，可通过 `configs/mini_gpt.yaml` 调整。

### 架构对比 / Architecture Comparison

| 特性 / Feature | 本项目默认 / Our Default | GPT-3 | LLaMA |
|------|------|------|------|
| 位置编码 | RoPE | Learned Absolute | RoPE |
| 归一化 | RMSNorm | LayerNorm | RMSNorm |
| FFN 激活 | SwiGLU | GELU | SwiGLU |
| 注意力 | Flash Attention | Manual | Flash Attention |
| Bias | No | Yes | No |

---

## 硬件要求 / Hardware Requirements

| 项目 | 规格 |
|------|------|
| 操作系统 / OS | Windows 11 |
| GPU | NVIDIA RTX 4060 8GB |
| RAM | 16GB |
| 训练数据 / Training Data | TinyStories (~200MB), WikiText-103 (~25MB 分词后) |

---

## 快速开始 / Quick Start

### 1. 环境配置 / Environment Setup

```bash
conda create -n mini-gpt python=3.11 -y
conda activate mini-gpt

# 安装 PyTorch (CUDA 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装其他依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 2. 准备训练数据 / Prepare Training Data

```bash
# 查看可用数据集
python src/prepare_data.py --list

# 下载 WikiText-103 (默认, 百科文本)
python src/prepare_data.py --dataset wikitext103

# 下载 TinyStories (故事文本, 快速收敛)
python src/prepare_data.py --dataset tiny_stories
```

### 3. 训练模型 / Train the Model

```bash
# 使用默认配置训练 (RoPE + RMSNorm + SwiGLU + Flash Attention)
python train.py --config configs/mini_gpt.yaml

# 覆盖特定参数
python train.py --config configs/mini_gpt.yaml --lr 1e-4 --batch_size 16

# 从 checkpoint 续训
python train.py --config configs/mini_gpt.yaml --resume checkpoints/mini_gpt/ckpt_latest.pt

# 查看训练曲线
tensorboard --logdir=logs
```

### 4. 生成文本 / Generate Text

```bash
# 交互式输入 prompt
python generate_rope.py --ckpt checkpoints/mini_gpt/ckpt_best.pt --prompt "The history of"

# 指定参数
python generate_rope.py --ckpt checkpoints/mini_gpt/ckpt_best.pt --prompt "Once upon a time" --temperature 0.7 --max_new_tokens 300
```

---

## 项目结构 / Project Structure

```
mini-gpt-from-scratch/
├── AGENTS.md                  # AI 助手工作规范
├── README.md                  # 项目文档 (本文件)
├── requirements.txt           # Python 依赖列表
├── train.py                   # 训练入口 (含 AMP / compile / 梯度累积 / TensorBoard)
├── generate.py                # 推理脚本 (learned PE 模式)
├── generate_rope.py           # 推理脚本 (RoPE 模式, 默认)
├── configs/
│   └── mini_gpt.yaml          # 完整训练配置 (模型/优化器/AMP/compile/日志)
├── src/
│   ├── model.py               # 统一 GPT 模型 (RoPE/Learned PE + RMSNorm/LayerNorm + SwiGLU/GELU)
│   ├── rope.py                # RoPE 旋转位置编码组件 (独立纯函数)
│   ├── dataset.py             # 二进制数据集加载器 (numpy memmap)
│   ├── prepare_data.py        # 多数据集下载与分词 (TinyStories / WikiText-103)
│   └── tokenizer.py           # GPT-2 BPE 分词器封装
└── experiments/
    ├── transformer.md         # Transformer 理论深度解析
    └── notes.md               # 开发日志与代码走读
```

---

## 训练数据 / Training Data

数据文件存储在 `data/` 目录，**不纳入 Git 版本控制**。

| 数据集 / Dataset | 分词后大小 | 语言 | 内容类型 | 用途 |
|------|-------|------|----------|------|
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) | ~25MB | 英文 | 维基百科文章 | 知识密集型、百科体 (当前默认) |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | ~200MB | 英文 | 儿童故事 | 快速收敛、故事续写 |

数据下载: `python src/prepare_data.py --dataset <name>`，新增数据集在 `prepare_data.py` 的 `DATASET_CONFIGS` 字典中添加。

---

## 模型配置 / Configuration

配置文件 `configs/mini_gpt.yaml`。优先级：命令行参数 > YAML 文件 > 代码默认值。

| 分类 | 参数 | 默认值 | 说明 |
|---------|-----------|---------|-------------|
| **model** | `d_model` | 768 | 嵌入维度 |
| | `n_layer` | 6 | Transformer 层数 |
| | `n_head` | 6 | 注意力头数 |
| | `block_size` | 256 | 最大上下文长度 |
| | `dropout` | 0.1 | Dropout 比例 |
| | `pos_enc_type` | `rope` | 位置编码: `rope` / `learned` |
| | `norm_type` | `rmsnorm` | 归一化: `rmsnorm` / `layernorm` |
| | `ffn_type` | `swiglu` | 前馈网络: `swiglu` / `gelu` |
| **optimizer** | `learning_rate` | 3e-4 | 峰值学习率 (warmup + cosine decay) |
| | `weight_decay` | 1e-2 | 权重衰减 |
| | `gradient_accumulation_steps` | 1 | 梯度累积 (1=不累积) |
| **trainer** | `max_iters` | 20000 | 总训练步数 |
| | `eval_interval` | 1000 | 评估间隔 (步数) |
| | `warmup_iters` | 100 | 预热步数 |
| **amp** | `enabled` | true | 混合精度训练 (bfloat16) |
| **compile** | `enabled` | true | torch.compile 加速 (需要 CUDA) |
| **logging** | `log_dir` | `logs` | TensorBoard 日志目录 |

---

## 训练效率优化 / Training Optimizations

| 优化 / Optimization | 原理 / Principle | 预期收益 |
|------|------|------|
| **AMP (bfloat16)** | 自动混合精度，计算密集型操作用低精度，其他用 FP32 | 1.5-2× 吞吐，显存降 30-40% |
| **torch.compile** | PyTorch 2.0+ 图编译，融合算子减少 kernel launch 开销 | 20-50% 加速 (CUDA only) |
| **梯度累积** | 累积 N 步梯度后一次性更新，等效 batch = batch_size × N | 突破显存限制 |
| **TensorBoard** | 实时记录 loss/LR/tokens/s | 追踪实验对比 |

---

## 模型验证 / Model Verification

每个模块内置自测，验证架构和张量形状的正确性：

```bash
python src/rope.py             # RoPE 组件测试 (4项)
python src/model.py            # 统一模型测试 (16项: learned PE / RoPE / KV-Cache / FlashAttn / 梯度)
python src/tokenizer.py        # 分词器往返测试
```

---

## 参考资料 / References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3) — Brown et al., 2020
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
