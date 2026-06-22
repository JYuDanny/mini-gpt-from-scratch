# --- generate.py ---
import time
import torch
import argparse
import os
import sys

from src.tokenizer import Tokenizer
from src.model import GPT, GPTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="GPT Text Generation")

    # 终端指定参数
    parser.add_argument('--ckpt', type=str, default='checkpoints/mini_gpt_raw/ckpt_best.pt', help='模型 checkpoint 路径')
    parser.add_argument('--prompt', type=str, default='', help='提示文本 (留空则手动输入)')
    parser.add_argument('--num_samples', type=int, default=1, help='生成样本数量')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='生成最大长度')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')
    parser.add_argument('--top_k', type=int, default=200, help='Top-K 采样')
    parser.add_argument('--device', type=str, default='auto', help='设备')

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 设备设置
    if args.device == 'auto':
        if torch.cuda.is_available(): device = 'cuda'
        elif torch.backends.mps.is_available(): device = 'mps'
        else: device = 'cpu'
    else:
        device = args.device
    print(f"Running inference on: {device}")

    # 2. 加载 Checkpoint
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint file {args.ckpt} not found.")
        sys.exit(1)

    print(f"Loading model from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)

    # 3. 自动恢复配置 (关键步骤)
    # train.py 在模型文件中保存了 'config' 字段
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 兼容未保存config的情况 (fallback)
        print("Warning: Config not found in checkpoint, utilizing default GPTConfig.")
        config = GPTConfig()

    # 4. 初始化模型并加载权重
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print(f"Model loaded. Layers: {config.n_layer}, Model dim: {config.d_model}, Block size: {config.block_size}")

    tokenizer = Tokenizer()

    # 5. 获取 Prompt
    start_text = args.prompt
    if start_text == '':
        start_text = input("Enter prompt: ")

    # 6. 生成
    start_ids = tokenizer.encode(start_text)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    print("\n--- Generating ---")
    time1 = time.perf_counter()
    with torch.no_grad():
        for k in range(args.num_samples):
            y = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_id=tokenizer.enc.eot_token,
            )
            print(f"Sample {k+1}:")
            print(tokenizer.decode(y[0].tolist()))
            print("-" * 50)
    time2 = time.perf_counter()
    print(f'\nTotal time: {time2 - time1:.4f}s')

if __name__ == "__main__":
    main()
