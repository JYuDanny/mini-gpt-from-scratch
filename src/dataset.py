# src/dataset.py — 二进制数据集加载器 / Binary Dataset Loader
# 功能 / Purpose: 使用 numpy memmap (内存映射) 加载大型二进制 token 数据，
# 支持数据文件大于 RAM 的情况，通过按需分页读取的方式实现
# Function: Load large binary token data via numpy memmap (memory-mapping),
# supports datasets larger than RAM through on-demand paged reading

import torch
import numpy as np
import os
from torch.utils.data import Dataset


class BinaryDataset(Dataset):
    """
    二进制数据集加载器 / Binary Dataset Loader

    核心原理 / Core Principle:
        使用 numpy.memmap 实现内存映射 (memory-mapping)。
        文件不会真正完全加载到 RAM 中，而是由操作系统按需从磁盘分页读取。
        这使得我们可以在 16GB RAM 的机器上训练 100GB+ 的数据集。
        Uses numpy.memmap for memory-mapping. The file is NOT fully loaded into RAM;
        instead, the OS pages it from disk on demand. This enables training on
        100GB+ datasets on a 16GB RAM machine.

    自回归训练原理 / Autoregressive Training Principle:
        语言模型训练的核心任务是"给定前文，预测下一个 token"。
        x = tokens[idx : idx+block_size]      # 输入序列 (上下文)
        y = tokens[idx+1 : idx+block_size+1]  # 目标序列 (右移一位)
        即: 用前 block_size 个 token 预测紧随其后的 block_size 个 token。
        LM training task: "given context, predict the next token".
    """

    def __init__(self, data_dir, block_size, split="train", data_file=None):
        """
        参数 / Args:
            data_dir: 数据文件所在目录 / Directory containing data files
            block_size: 上下文窗口大小 (最大序列长度) / Context window size (max sequence length)
            split: 数据划分 (train/val) / Data split (train/val)
            data_file: 直接指定文件名 (可选, 覆盖 split 参数) / Directly specify filename (optional, overrides split)
        """
        self.block_size = block_size

        # --- 定位数据文件 / Locate data file ---
        # 如果指定了 data_file, 直接使用 / If data_file is specified, use it directly
        if data_file is not None:
            filename = os.path.join(data_dir, data_file)
            if not os.path.exists(filename):
                filename = data_file  # 尝试作为绝对/相对路径 / Try as absolute/relative path
        else:
            # 否则按 split 命名规则查找 / Otherwise use split naming convention
            filename = os.path.join(data_dir, f"{split}.bin")

        # 回退机制: 如果没有 train.bin, 尝试 val.bin / Fallback: if no train.bin, try val.bin
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, trying fallback...")
            if split == "train":
                # 如果是训练划分缺失, 尝试常见的数据文件名
                fallbacks = [
                    os.path.join(data_dir, "wikitext103.bin"),
                    os.path.join(data_dir, "tiny_stories.bin"),
                ]
                for fb in fallbacks:
                    if os.path.exists(fb):
                        print(f"  Using fallback: {fb}")
                        filename = fb
                        break

        if not os.path.exists(filename):
            local_files = []
            if os.path.isdir(data_dir):
                local_files = [f for f in os.listdir(data_dir) if f.endswith(".bin")]
            raise FileNotFoundError(
                f"找不到数据文件 / Data file not found: {filename}\n"
                f"数据目录下可用的 .bin 文件 / Available .bin files: {local_files}\n"
                f"请先运行 / Please run first:\n"
                f"  python src/prepare_data.py --list    # 查看可用数据集 / List datasets\n"
                f"  python src/prepare_data.py --dataset <name>  # 下载指定数据集 / Download"
            )

        # --- 文件信息 / File info ---
        file_size_bytes = os.path.getsize(filename)

        # 计算总 token 数 / Calculate total token count
        # uint16 = 2 bytes per token, 所以 token 数 = 文件大小 / 2
        # uint16 = 2 bytes per token, so token count = file_size / 2
        total_tokens = file_size_bytes // 2

        # --- 创建内存映射 (不加载到 RAM) / Create memory map (not loaded into RAM) ---
        # mode='r' 表示只读, shape 指定数组形状
        # mode='r' means read-only, shape specifies array dimensions
        self.data = np.memmap(filename, dtype=np.uint16, mode="r", shape=(total_tokens,))
        self.filename = os.path.basename(filename)

        print(f"加载数据集 / Loaded dataset: {self.filename}")
        print(f"  总 token 数 / Total tokens: {total_tokens / 1e6:.2f}M")
        print(f"  文件大小 / File size: {file_size_bytes / 1024 / 1024:.2f} MB")

    def __len__(self):
        """
        返回可用训练样本数 / Returns number of available training samples.
        每 block_size 个 token 产生一个训练样本, 最后一个样本的 target 需要
        多一个后续 token, 所以总样本数 = total_tokens - block_size。
        """
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        取一个训练样本 / Fetch a single training sample.

        参数 / Args:
            idx: 数据集索引, 同时也是 token 序列的起始位置
                 In large-scale training, this index doubles as the starting position
                 in the token stream, providing random access to the corpus.

        返回 / Returns:
            (x, y): 输入和目标 / input and target token sequences
        """
        start = idx
        end = start + self.block_size + 1  # +1 因为 y 比 x 多一个未来 token

        # 从 memmap 中切片 (非常快, 操作系统按需分页)
        # Slice from memmap (very fast, OS pages on demand)
        # 必须转为 int64 (long), 因为 PyTorch Embedding 需要 long 类型索引
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))

        # 自回归训练数据格式 / Autoregressive training data format:
        # x: 前 block_size 个 token 作为输入上下文
        # y: 后 block_size 个 token 作为预测目标 (整体右移一位)
        x = chunk[:-1]
        y = chunk[1:]

        return x, y
