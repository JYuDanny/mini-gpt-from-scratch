# prepare_data.py — 训练数据预处理 / Training Data Preprocessing
# 功能 / Purpose: 从 HuggingFace 下载数据集 → GPT-2 分词 → 保存为紧凑的二进制文件 (.bin)
# Function: Download dataset from HuggingFace → tokenize with GPT-2 → save as compact binary (.bin)
# 核心原理 / Core principle: streaming=True 流式处理，避免全量加载到内存，适合超大数据集
# Core principle: streaming=True avoids loading entire dataset into RAM, ideal for large datasets

import os
import numpy as np
import tiktoken
import argparse
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# 数据集配置表 / Dataset Configuration Table
# 添加新数据集只需在此字典中新增一项
# To add a new dataset, simply add one entry to this dictionary
# ============================================================
# 每个数据集配置项说明:
#   - name: HuggingFace 数据集名称 / HuggingFace dataset identifier
#   - subset: 子集名 / subset name (None 表示不指定 / None means no subset)
#   - split: 数据划分 / data split (通常为 "train" / usually "train")
#   - text_key: 文本字段名 / name of the text field in each sample
#   - max_samples: 处理上限 / max number of samples to process
#   - description: 数据集简要说明 / brief description
DATASET_CONFIGS = {
    "tiny_stories": {
        "name": "roneneldan/TinyStories",
        "subset": None,
        "split": "train",
        "text_key": "text",
        "max_samples": 600_000,
        "description": "英文儿童故事 / English children's stories — 快速收敛、适合验证架构 / fast convergence, ideal for verifying architecture"
    },
    "wikitext103": {
        "name": "Salesforce/wikitext",
        "subset": "wikitext-103-v1",
        "split": "train",
        "text_key": "text",
        "max_samples": 200_000,
        "description": "英文维基百科文章 / English Wikipedia articles — 知识密集、百科体、词汇多样 / knowledge-dense, encyclopedic style, rich vocabulary"
    },
}

# ============================================================
# 输出配置 / Output Configuration
# ============================================================
OUTPUT_DIR = "data"                # 输出目录 / output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tokenizer():
    """
    加载 GPT-2 分词器 / Load GPT-2 tokenizer
    原理 / Principle: tiktoken 是 OpenAI 发布的 BPE (Byte Pair Encoding) 分词器，
    将文本拆分为子词单元 (subword tokens)，词表大小约 50,257。
    BPE 从字节开始，迭代合并高频字节对，在编码效率和 OOV 问题间取得平衡。
    """
    enc = tiktoken.get_encoding("gpt2")
    return enc


def process_dataset(dataset_key, output_name=None):
    """
    处理单个数据集: 下载 → 分词 → 写二进制文件
    Process a single dataset: download → tokenize → write binary file

    参数 / Args:
        dataset_key: 数据集在 DATASET_CONFIGS 中的键名 / key name in DATASET_CONFIGS
        output_name: 输出文件名 (不含扩展名) / output filename without extension
                     默认使用 dataset_key 作为文件名
    """
    if dataset_key not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"未知数据集 / Unknown dataset: '{dataset_key}'. 可用选项 / Available: {available}")

    cfg = DATASET_CONFIGS[dataset_key]
    output_name = output_name or dataset_key
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.bin")

    # -------------------------------------------------------
    # 1. 加载分词器 / Load tokenizer
    # -------------------------------------------------------
    enc = load_tokenizer()
    eot_token = enc.eot_token  # EOT (End of Text) 结束符，用于分隔不同样本

    # -------------------------------------------------------
    # 2. 流式下载数据集 / Stream download dataset
    # -------------------------------------------------------
    # streaming=True 原理 / principle:
    #   不会一次性下载整个数据集到内存，而是按需逐条获取，
    #   适合处理 GB 甚至 TB 级别的超大数据集。
    print(f"\n{'='*60}")
    print(f"数据集 / Dataset: {dataset_key} — {cfg['description']}")
    print(f"HuggingFace ID: {cfg['name']}")
    print(f"预计处理样本数 / Max samples: {cfg['max_samples']:,}")
    print(f"输出文件 / Output: {output_path}")
    print(f"{'='*60}\n")

    print(f"[1/3] 正在流式下载数据集 / Streaming dataset...")
    dataset = load_dataset(
        cfg["name"],
        name=cfg["subset"],
        split=cfg["split"],
        streaming=True
    )

    # -------------------------------------------------------
    # 3. 分词并写入二进制文件 / Tokenize and write binary
    # -------------------------------------------------------
    # 为什么用 uint16 / Why uint16:
    #   GPT-2 词表大小为 50257，正好在 uint16 的表示范围 (0~65535) 内。
    #   每个 token 仅占 2 字节 (bytes)，相比 int32 (4 bytes) 或 int64 (8 bytes) 最节省磁盘。
    #
    # 为什么用二进制流写入而非 numpy 数组拼接 / Why binary streaming vs numpy array concat:
    #   逐条编码后直接追写到文件，避免在内存中维护不断增长的数组，
    #   这是处理超大数据集时的标准做法。
    print(f"[2/3] 开始分词并写入 / Tokenizing and writing...")

    total_tokens = 0
    total_samples = 0
    skipped_empty = 0

    with open(output_path, "wb") as f:
        pbar = tqdm(enumerate(dataset), total=cfg["max_samples"], desc="处理中 / Processing")

        for idx, item in pbar:
            if idx >= cfg["max_samples"]:
                break

            text = item.get(cfg["text_key"], "")
            if not text or not text.strip():
                skipped_empty += 1
                continue

            # BPE 分词 / BPE tokenization
            # 原理 / Principle: enc.encode() 将原始文本拆分为子词 token ID 序列
            # 例如: "hello world" → [31373, 995]
            ids = enc.encode(text, allowed_special={"<|endoftext|>"})

            # 每个样本末尾追加 EOT (End of Text) 标记
            # 原理 / Principle: EOT token 告诉模型"本条文本结束，下一条是独立内容"
            # 没有 EOT 模型可能错误地把两条文本的边界当作文本连续性来学习
            ids.append(eot_token)

            # 转 uint16 并直接写入磁盘 / Convert to uint16 and write directly to disk
            data = np.array(ids, dtype=np.uint16)
            f.write(data.tobytes())

            total_tokens += len(ids)
            total_samples += 1

            pbar.set_postfix(tokens=f"{total_tokens/1e6:.1f}M")

    # -------------------------------------------------------
    # 4. 输出统计信息 / Print statistics
    # -------------------------------------------------------
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n[3/3] 完成 / Done!")
    print(f"  处理样本数 / Samples processed: {total_samples:,}")
    print(f"  跳过空样本 / Empty samples skipped: {skipped_empty}")
    print(f"  总 token 数 / Total tokens:     {total_tokens:,} ({total_tokens/1e6:.2f}M)")
    print(f"  文件大小 / File size:           {file_size_mb:.2f} MB")
    print(f"  输出路径 / Output:              {output_path}")
    print()


def main():
    """
    主入口 / Main entry point
    用法 / Usage:
        python src/prepare_data.py                        # 使用默认配置 (wikitext103)
        python src/prepare_data.py --dataset tiny_stories  # 处理 TinyStories
        python src/prepare_data.py --dataset wikitext103   # 处理 WikiText-103
        python src/prepare_data.py --list                  # 列出所有可用数据集
    """
    parser = argparse.ArgumentParser(
        description="下载并预处理训练数据 / Download and preprocess training data"
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext103",
        help=f"数据集名称 / Dataset name. 可用选项 / Available: {list(DATASET_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出文件名(不含扩展名) / Output filename without extension"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有可用数据集 / List all available datasets"
    )
    args = parser.parse_args()

    # 列出所有可用数据集
    if args.list:
        print("\n可用数据集 / Available Datasets:")
        print("-" * 60)
        for key, cfg in DATASET_CONFIGS.items():
            print(f"  {key:20s} | {cfg['description']}")
            print(f"  {'':20s} | HuggingFace: {cfg['name']}")
            print(f"  {'':20s} | 最大样本数 / Max: {cfg['max_samples']:,}")
            print()
        return

    # 处理指定数据集
    process_dataset(args.dataset, args.output)

    # 提示：如果同时需要 TinyStories，可以再次运行
    if args.dataset != "tiny_stories":
        print("提示 / Hint: 需要故事续写数据？运行 / Need story data? Run:")
        print("  python src/prepare_data.py --dataset tiny_stories")
        print()


if __name__ == "__main__":
    main()
