import torch

from src.tokenizer import Tokenizer
from src.model import GPT, GPTConfig

def main():
    """独立生成脚本，加载 checkpoint 生成文本。
    原理：eval 模式下 autoregressive 采样，temperature 控制随机性。"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'checkpoints/best_model.pt'

    # 配置需匹配训练（使用 GPTConfig 类初始化）
    config = GPTConfig(
        vocab_size=50257,  # 从 tokenizer
        d_model=384,
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

    max_new_tokens = 200
    temperature = 0.7
    generated = model.generate(prompt, max_new_tokens, temperature)
    print(f"Generated: {tokenizer.decode(generated[0].tolist())}")

if __name__ == "__main__":
    main()
