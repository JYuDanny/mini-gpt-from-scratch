import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

from src.dataset import ShakespeareDataset
from src.tokenizer import Tokenizer
from src.model import GPT

def train():
    """完整训练函数，严格如 GPT-3：CE loss + AdamW + eval + checkpoint。
    原理：train loop 优化参数，val loop 监控 overfit，generate eval 质量。"""
    
    # 配置（严格掌控变量：所有 hyperparam 显式定义）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64  # 4060 可调至 64，如果 OOM 降至 16
    block_size = 128
    d_model = 384
    num_heads = 6
    num_layers = 6
    d_ff = 4 * d_model
    dropout = 0.1
    learning_rate = 3e-4
    epochs = 5  # 实际训练用，多 epoch 观察收敛
    eval_interval = 200  # 每 X 步 eval + generate
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 数据（掌控：dataset 实例化，loader batch_size 匹配）
    dataset = ShakespeareDataset(block_size=block_size)
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size  # 严格从 tokenizer 获取
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset.set_train(False)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataset.set_train(True)
    
    # 模型（掌控：所有 param 传入，to device）
    model = GPT(vocab_size, d_model, num_heads, num_layers, d_ff, block_size, dropout)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    model.train()
    
    # 修改：全局进度条，覆盖所有 steps
    total_steps = epochs * len(train_loader)  # 总 steps 计算
    global_pbar = tqdm(total=total_steps, desc="Training", unit="step")  # 一个动态条
    
    global_step = 0  # 全局 step 计数器
    for epoch in range(epochs):
        for batch in train_loader:
            global_step += 1
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            logits = model(x)  # (B, N, vocab_size)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))  # flatten 计算
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新全局条（显示 epoch/step/loss）
            global_pbar.set_description(f"Epoch {epoch+1}/{epochs}, Step {global_step}/{total_steps}")
            global_pbar.set_postfix(train_loss=loss.item())
            global_pbar.update(1)
            
            if global_step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, vocab_size)
                print(f"Step {global_step}: val_loss={val_loss:.4f}")
                
                # 生成样例 eval
                model.eval()
                prompt = torch.tensor([tokenizer.encode("First Citizen:")]).to(device)
                generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
                print(f"Generated: {tokenizer.decode(generated[0].tolist())}")
                model.train()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
    
    global_pbar.close()  # 关闭条
    return model, tokenizer

def evaluate(model: GPT, loader: DataLoader, device: str, vocab_size: int) -> float:
    """Eval 函数：计算 val loss。
    原理：no_grad 节省内存，平均 batch loss。"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)

if __name__ == "__main__":
    model, tokenizer = train()
    # 最终生成测试
    model.eval()
    prompt = torch.tensor([tokenizer.encode("First Citizen:")]).to(model.token_emb.weight.device)
    generated = model.generate(prompt, max_new_tokens=100, temperature=0.7)
    print(f"Final Generated: {tokenizer.decode(generated[0].tolist())}")
