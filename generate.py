import torch

from src.tokenizer import Tokenizer
from src.model import GPT

def main():
    """独立生成脚本，加载 checkpoint 生成文本。
    原理：eval 模式下 autoregressive 采样，temperature 控制随机性。"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'checkpoints/best_model.pt'

    # 配置需匹配训练（严格掌控）
    vocab_size = 50257  # 从 tokenizer
    d_model = 384
    num_heads = 6
    num_layers = 6
    d_ff = 4 * d_model
    block_size = 128
    dropout = 0.1

    model = GPT(vocab_size, d_model, num_heads, num_layers, d_ff, block_size, dropout)
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
