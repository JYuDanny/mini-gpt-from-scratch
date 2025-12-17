# prepare_data.py
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# --- 配置 ---
# 选项 1: TinyStories (推荐，收敛极快，适合验证架构)
dataset_name = "roneneldan/TinyStories"
subset_name = None 

# 选项 2: FineWeb-Edu (高质量通用语料，建议只取一小部分，否则跑不动)
# dataset_name = "HuggingFaceFW/fineweb-edu"
# subset_name = "sample-10BT"

# 输出文件名
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

def process(dataset_name):
    # 1. 加载分词器
    enc = tiktoken.get_encoding("gpt2")

    # 2. 下载数据集 (使用 streaming=True 以防内存溢出)
    print(f"Loading dataset: {dataset_name}...")
    # split="train" 表示只下载训练集
    # streaming=True 允许我们在下载的同时处理，不需要一次性加载到内存
    dataset = load_dataset(dataset_name, name=subset_name, split="train", streaming=True)

    # 3. 预处理循环
    # 我们将数据保存为 uint16 (如果你词表 < 65535) 以节省空间
    # GPT-2 词表 50257，正好可以用 uint16 (2 bytes per token)
    arr_len = 0
    arr = []
    total_tokens = 0

    # 为了演示，我们限制处理的样本数量，防止你硬盘爆了
    # 如果你想跑全量，可以把 max_samples 设得非常大
    max_samples = 600_000  # TinyStories 约有 200万+ 条，这里取 1/10 足够你玩了

    filename = os.path.join(output_dir, "train.bin")

    print(f"Processing and saving to {filename}...")

    # 使用 tqdm 显示进度
    with open(filename, "wb") as f:
        for idx, item in tqdm(enumerate(dataset), total=max_samples):
            if idx >= max_samples:
                break

            text = item['text'] # 大多数数据集的文本字段叫 'text'

            # 编码: text -> list of integers
            ids = enc.encode(text, allowed_special={'<|endoftext|>'})
            ids.append(enc.eot_token) # 每个样本末尾加上结束符

            # 转为 numpy uint16 并写入文件
            # 直接写入文件流，节省内存
            data = np.array(ids, dtype=np.uint16)
            f.write(data.tobytes())

            total_tokens += len(ids)

    print(f"Done! Saved {total_tokens} tokens to {filename}")
    print(f"File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    process(dataset_name)
