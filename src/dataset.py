# src/dataset.py
import torch
import numpy as np
import os
from torch.utils.data import Dataset

class BinaryDataset(Dataset):
    """
    通用的大规模二进制数据集加载器。
    使用 numpy.memmap 实现内存映射，允许在小内存机器上训练超大数据集。
    """
    def __init__(self, data_dir, block_size, split='train'):
        self.block_size = block_size

        # 寻找 .bin 文件
        filename = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(filename):
            # 如果没有 split.bin，尝试找 train.bin 作为回退
            print(f"Warning: {filename} not found, falling back to train.bin")
            filename = os.path.join(data_dir, "train.bin")

        if not os.path.exists(filename):
            raise FileNotFoundError(f"No data file found in {data_dir}")

        # 获取文件大小
        file_size_bytes = os.path.getsize(filename)

        # 计算总 token 数 (uint16 占 2 字节)
        total_tokens = file_size_bytes // 2

        # 创建内存映射 (不会真正加载到 RAM，像虚拟内存一样读取)
        # mode='r' 表示只读
        self.data = np.memmap(filename, dtype=np.uint16, mode='r', shape=(total_tokens,))

        print(f"Loaded dataset from {filename}")
        print(f"Total tokens: {total_tokens / 1e6:.2f}M")

    def __len__(self):
        # 我们返回可选的样本数量
        # 实际上是 total_tokens - block_size，因为最后一个样本需要后续的 token 做 target
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # 从 memmap 中切片，非常快
        # 必须转为 int64 (long)，因为 PyTorch Embedding 层需要 long 类型

        # 这里的 idx 是 dataset 的索引。
        # 在大规模训练中，通常我们随机取一段，而不是按顺序 idx
        # 但为了兼容 DataLoader 的标准接口，我们保留 idx

        # 这里的逻辑稍微 tricky：
        # 如果 dataset 非常大，DataLoader 传入的 idx 可能会很大。
        # 我们可以直接用这个 idx 作为起始位置。

        start = idx
        end = start + self.block_size + 1

        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))

        # x 是输入，y 是目标 (右移一位)
        x = chunk[:-1]
        y = chunk[1:]

        return x, y

# --- 适配训练脚本的修改 ---
# 现在的 Dataset 很大，不能像莎士比亚那样 idx 从 0 到 len 遍历一遍。
# 大模型训练通常是“无限采样”的。
# 为了让你的 train_gpt.py 依然能用 DataLoader，我们可以在 dataset 内部做一个小调整，
# 或者在 train_gpt.py 里使用 RandomSampler。
# 最简单的普适性改法是：让 __len__ 返回一个“足够大”的虚数，或者真实的长度。
# 如果返回真实长度 (比如 10亿)，tqdm 进度条可能会显示不完，但不影响训练。
