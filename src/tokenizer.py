import tiktoken

class Tokenizer:
    """Tokenizer wrapper 使用 tiktoken 的 GPT-2 编码器，用于英文文本分词。
    原理：BPE (Byte Pair Encoding) 将常见子词合并，减少 vocab_size，提高效率。
    例如，"hello world" → [31373, 995]，而不是字符级。"""
    
    def __init__(self):
        # 初始化 GPT-2 tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab  # ~50257

    def encode(self, text: str) -> list[int]:
        """将文本编码成 token IDs 列表。
        原理：BPE 先转字节，再合并高频对，避免 OOV (out-of-vocab) 问题。"""
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int]) -> str:
        """将 token IDs 解码回文本。
        原理：逆 BPE，合并回原字符串，确保无损。"""
        return self.enc.decode(tokens)

# 测试入口
if __name__ == "__main__":
    tokenizer = Tokenizer()
    text = "Hello, world! This is a test."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    assert text == decoded, "Encode-decode mismatch!"
    print("Test passed.")
