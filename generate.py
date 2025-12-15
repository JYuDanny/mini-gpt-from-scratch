import time
import torch

from src.tokenizer import Tokenizer
from src.model import GPT, GPTConfig

def main():
    """独立生成脚本，加载 checkpoint 生成文本。
    原理：eval 模式下 autoregressive 采样，temperature 控制随机性。"""

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Running test on: {device}")
    checkpoint_path = 'checkpoints/best_model.pt'

    # 配置需匹配训练（使用 GPTConfig 类初始化）
    config = GPTConfig(
        vocab_size=50257,  # 从 tokenizer
        d_model=768,
        n_head=6,
        n_layer=6,
        block_size=128,
        dropout=0.1,
        bias=True  # 匹配新 model.py 默认
    )

    model = GPT(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = Tokenizer()
    prompt_text = input("Enter prompt (e.g., First Citizen:): ")
    prompt = torch.tensor([tokenizer.encode(prompt_text)]).to(device)

    max_new_tokens = 300
    temperature = 0.7
    time1 = time.perf_counter()
    generated = model.generate(
        prompt,
        max_new_tokens,
        temperature,
        eos_id=tokenizer.enc.eot_token,  # 引入结束符，tiktoken 的结束符为 eot_token
    )
    time2 = time.perf_counter()
    print(f"Generated: {tokenizer.decode(generated[0].tolist())}")
    print(f"\nTotal time: {time2 - time1:.4f}s")

if __name__ == "__main__":
    main()
