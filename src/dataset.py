import torch
from torch.utils.data import Dataset
from src.tokenizer import Tokenizer  # 导入上一步的 tokenizer
from src.utils import download_tinyshakespeare  # 导入下载函数

class ShakespeareDataset(Dataset):
    """Shakespeare 数据集类，用于自回归训练。
    原理：将整个文本 tokenize 成一个长序列，然后随机采样 block_size 的上下文片段。
    GPT-3 类似：训练时用 next-token loss，target = data[1:], data[:-1] 是 input。
    这模拟 autoregressive generation：模型基于前 t token 预测 t+1。"""
    
    def __init__(self, block_size: int = 128, train_split: float = 0.9):
        """初始化：下载数据，tokenize，分 train/val。
        block_size: 最大上下文长度 (GPT-3 用 2048，但 mini 用小值)。
        train_split: 训练/验证拆分比例。"""
        data_path = download_tinyshakespeare()
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokenizer = Tokenizer()
        self.tokens = tokenizer.encode(text)  # 整个文本的 token IDs 列表
        self.vocab_size = tokenizer.vocab_size
        
        # 分 train/val
        n = len(self.tokens)
        self.train_data = torch.tensor(self.tokens[:int(n * train_split)], dtype=torch.long)
        self.val_data = torch.tensor(self.tokens[int(n * train_split):], dtype=torch.long)
        
        self.block_size = block_size
        self.is_train = True  # 默认 train，外部可切换

    def set_train(self, is_train: bool = True):
        """切换 train/val 模式。"""
        self.is_train = is_train

    def __len__(self):
        """数据集长度：总 token 数减 block_size（确保有足够片段）。"""
        data = self.train_data if self.is_train else self.val_data
        return len(data) - self.block_size

    def __getitem__(self, idx: int):
        """获取一个样本：x (context), y (target)。
        原理：随机 idx 采样 block_size+1 的 chunk，然后 x = chunk[:-1], y = chunk[1:]。
        这严格如 GPT-3：并行计算所有位置的 loss，但 causal mask 确保不看未来。"""
        data = self.train_data if self.is_train else self.val_data
        chunk = data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# 测试入口
if __name__ == "__main__":
    dataset = ShakespeareDataset(block_size=8)  # 小 block 测试
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Train length: {len(dataset.train_data)}, Val length: {len(dataset.val_data)}")
    
    # 获取一个样本
    x, y = dataset[0]
    print(f"Sample x: {x.tolist()}")
    print(f"Sample y: {y.tolist()}")
    
    # 解码检查
    tokenizer = Tokenizer()
    print(f"Decoded x: {tokenizer.decode(x.tolist())}")
    print(f"Decoded y: {tokenizer.decode(y.tolist())}")
    
    assert len(x) == len(y) == dataset.block_size, "Length mismatch!"
    assert all(x[1:] == y[:-1]), "Shift mismatch!"
    print("Test passed.")
