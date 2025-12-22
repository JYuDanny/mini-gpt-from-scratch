# --- generate.py ---
import time
import torch
import argparse
import os
import sys

from src.tokenizer import Tokenizer
from src.model_rope import GPT, GPTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="GPT Text Generation")

    # 终端指定参数
    parser.add_argument('--ckpt', type=str, default='checkpoints/mini_gpt_rope/ckpt_best.pt', help='模型 checkpoint 路径')
    parser.add_argument('--prompt', type=str, default='', help='提示文本 (留空则手动输入)')
    parser.add_argument('--num_samples', type=int, default=1, help='生成样本数量')
    parser.add_argument('--max_new_tokens', type=int, default=300, help='生成最大长度')
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

    # --- 关键修改开始 ---

    # 4. 初始化模型
    model = GPT(config)

    # 4.1 处理 State Dict (清洗权重)
    state_dict = checkpoint['model']

    # 剔除不需要加载的键
    # (1) freqs_cis: 它是 Buffer，应该由当前模型根据当前配置重新计算，不要加载旧的
    # (2) ...可以根据需要剔除其他临时变量
    unwanted_prefix = '_orig_mod.' # 如果你用了 torch.compile，会有这个前缀
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    if "freqs_cis" in state_dict:
        print("Info: Dropping 'freqs_cis' from checkpoint to allow re-computation.")
        del state_dict["freqs_cis"]

    # 4.2 加载权重
    # 使用 strict=False，允许 checkpoint 里少一些 buffer (比如 freqs_cis)
    # 但如果有形状不匹配的 Linear 层权重，依然会报错，这是我们要的
    keys = model.load_state_dict(state_dict, strict=False)

    # 验证关键权重是否丢失 (过滤掉 freqs_cis 的缺失警告)
    missing_keys = [k for k in keys.missing_keys if "freqs_cis" not in k]
    unexpected_keys = [k for k in keys.unexpected_keys if "freqs_cis" not in k]

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")

    # --- 关键修改结束 ---

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
