import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import wandb  # 新增：导入 wandb

from src.dataset import ShakespeareDataset
from src.tokenizer import Tokenizer
from src.model import GPT

def train():
    """完整训练函数，严格如 GPT-3：CE loss + AdamW + eval + checkpoint + wandb log。"""
    
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    block_size = 128
    d_model = 384
    num_heads = 6
    num_layers = 6
    d_ff = 4 * d_model
    dropout = 0.1
    learning_rate = 3e-4
    epochs = 5
    eval_interval = 200
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 新增：初始化 wandb，project 名自定义，config 自动记录所有 hyperparams
    wandb.init(project="mini-gpt-from-scratch", config={
        "batch_size": batch_size,
        "block_size": block_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "device": device
    })
    
    # 数据
    dataset = ShakespeareDataset(block_size=block_size)
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset.set_train(False)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataset.set_train(True)
    
    # 模型
    model = GPT(vocab_size, d_model, num_heads, num_layers, d_ff, block_size, dropout)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(train_loss=loss.item())
            
            if (step + 1) % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, vocab_size)
                print(f"Step {step+1}: val_loss={val_loss:.4f}")
                
                # 新增：log 到 wandb（train_loss, val_loss, step）
                wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "step": step + 1, "epoch": epoch + 1})
                
                # 生成样例并 log
                model.eval()
                prompt = torch.tensor([tokenizer.encode("First Citizen:")]).to(device)
                generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
                gen_text = tokenizer.decode(generated[0].tolist())
                print(f"Generated: {gen_text}")
                
                # 新增：log 生成文本到 wandb（作为 table 或 text）
                wandb.log({"generated_sample": wandb.Html(f"<p>{gen_text}</p>")})  # 用 Html 格式美观显示
                
                model.train()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
                    # 新增：保存模型到 wandb artifact（云备份）
                    wandb.save(os.path.join(checkpoint_dir, 'best_model.pt'))
    
    wandb.finish()  # 新增：结束 run，上传所有数据
    return model, tokenizer

def evaluate(model: GPT, loader: DataLoader, device: str, vocab_size: int) -> float:
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
    # 最终生成测试（可选 log 到 wandb，但已 finish）
    model.eval()
    prompt = torch.tensor([tokenizer.encode("First Citizen:")]).to(model.token_emb.weight.device)
    generated = model.generate(prompt, max_new_tokens=100, temperature=0.7)
    print(f"Final Generated: {tokenizer.decode(generated[0].tolist())}")
