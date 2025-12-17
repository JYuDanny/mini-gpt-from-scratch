import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import math

from src.dataset import BinaryDataset
from src.tokenizer import Tokenizer
from src.model import GPT, GPTConfig

# --- 新增：简单的学习率调度器 ---
def get_lr(it, max_iters, learning_rate, warmup_iters=100, min_lr=1e-5):
    # 1. 线性预热 (Warmup)
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2. 余弦衰减 (Cosine Decay)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Running test on: {device}")

    # --- 配置参数 ---
    batch_size = 32
    block_size = 128  # 256 依然不会 OOM，但训练速度严重下降
    d_model = 512
    n_head = 4
    n_layer = 6
    dropout = 0.1
    bias = True

    learning_rate = 2e-4  # 稍微调高初始 LR，依靠调度器控制
    max_iters = 30000      # 我们改用迭代次数控制，而不是 Epoch，这样更直观
    eval_interval = 500
    eval_iters = 50       # 每次验证只跑 50 个 batch，防止卡顿
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 数据准备 ---
    train_dataset = BinaryDataset(data_dir='data', block_size=block_size, split='train')
    val_dataset = train_dataset
    tokenizer = Tokenizer()

    # 必须设置 shuffle=True，这样 DataLoader 会随机读取 dataset 中的位置
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 验证集可以不 shuffle，但建议随机采一些
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # --- 模型初始化 ---
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model, n_head=n_head, n_layer=n_layer,
        block_size=block_size, dropout=dropout, bias=bias
    )
    model = GPT(config)
    model.to(device)

    # 打印参数量
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    best_val_loss = float('inf')
    model.train()

    # 使用 iter 手动控制 loader，实现无限循环直到 max_iters
    train_iter = iter(train_loader)

    # --- 训练循环 (使用 tqdm 包装 range) ---
    pbar = tqdm(range(max_iters), desc="Training")

    for step in pbar:
        # 1. 获取 Batch (如果读完了就重新开始)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # 2. 更新学习率 (LR Schedule)
        lr = get_lr(step, max_iters, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 3. 前向传播与反向传播
        _, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()

        # 4. 梯度裁剪 (Gradient Clipping) - 关键技巧！
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # 5. 更新进度条显示的 Loss
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # 6. 定期评估 (Eval)
        if (step + 1) % eval_interval == 0:
            # 使用 tqdm.write 防止进度条错位
            tqdm.write(f"\n--- Step {step+1}: Evaluating ---")

            val_loss = evaluate(model, val_loader, device, eval_iters)
            tqdm.write(f"Validation Loss: {val_loss:.4f}")

            # 生成测试
            model.eval()
            ctx = torch.tensor([tokenizer.encode("Once upon a time,")]).to(device)
            gen = model.generate(ctx, max_new_tokens=50, temperature=0.8)
            decoded = tokenizer.decode(gen[0].tolist())
            tqdm.write(f"Generate sample: {decoded.strip()}\n")
            model.train()

            # 保存模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
                tqdm.write("Saved best model.")

    return model, tokenizer

@torch.no_grad()
def evaluate(model, loader, device, eval_iters):
    """
    优化的评估函数：只跑 eval_iters 个 batch，而不是跑完整个验证集
    """
    model.eval()
    losses = torch.zeros(eval_iters)
    loader_iter = iter(loader)

    for k in range(eval_iters):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader) # 如果验证集太小，循环读取
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        losses[k] = loss.item()

    model.train()
    return losses.mean().item()

if __name__ == "__main__":
    train()
