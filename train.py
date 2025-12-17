# --- train.py ---
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import math
import argparse
import yaml
import sys

from src.dataset import BinaryDataset
from src.tokenizer import Tokenizer
from src.model import GPT, GPTConfig

# --- 简单的学习率调度器 ---
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

def get_config():
    """
    配置加载逻辑：
    1. 定义命令行参数
    2. 如果指定了 --config，先加载 YAML
    3. 用命令行参数覆盖 YAML 中的同名参数
    """
    parser = argparse.ArgumentParser(description="Train GPT Model")
    
    # 配置文件路径
    parser.add_argument('--config', type=str, default=None, help='Path to .yaml config file')
    
    # --- 定义所有可能的命令行参数 (默认值设为 None) ---
    # 只有设为 None，我们才知道用户到底有没有在命令行里输入这个参数
    # 如果用户没输，就用 YAML 里的；如果输了，就覆盖 YAML 里的。
    
    # System
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    
    # Data
    parser.add_argument('--batch_size', type=int, default=None)
    
    # Model
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)

    # Training
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_iters', type=int, default=None)

    args = parser.parse_args()

    # --- 1. 初始化默认配置字典 (兜底) ---
    config = {
        'system': {'out_dir': 'checkpoints', 'device': 'auto', 'resume': None},
        'data': {'data_dir': 'data', 'batch_size': 64, 'num_workers': 0},
        'model': {'block_size': 128, 'd_model': 512, 'n_layer': 6, 'n_head': 8, 'dropout': 0.1, 'bias': True},
        'optimizer': {'learning_rate': 3e-4, 'weight_decay': 1e-2},
        'trainer': {'eval_interval': 1000, 'eval_iters': 50, 'max_iters': 30000}
    }

    # --- 2. 加载 YAML 并更新 ---
    if args.config is not None:
        print(f"Loading config from {args.config}...")
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)

        # 递归更新字典 (这里做一个简单的深度更新)
        for section, params in yaml_config.items():
            if section in config:
                config[section].update(params)
            else:
                config[section] = params # 新增的 section

    # --- 3. 命令行参数覆盖 (Override) ---
    # 这种映射有点繁琐，但在没有引入 Hydra 等重型库之前，这是最清晰的方法
    if args.out_dir: config['system']['out_dir'] = args.out_dir
    if args.resume:  config['system']['resume'] = args.resume
    if args.device:  config['system']['device'] = args.device
    if args.batch_size: config['data']['batch_size'] = args.batch_size
    if args.n_layer: config['model']['n_layer'] = args.n_layer
    if args.d_model: config['model']['d_model'] = args.d_model
    if args.lr:      config['optimizer']['learning_rate'] = args.lr
    if args.max_iters: config['trainer']['max_iters'] = args.max_iters

    return config

def train():
    # 获取最终配置字典
    cfg = get_config()

    # 方便调用，提取一些变量
    sys_cfg = cfg['system']
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    opt_cfg = cfg['optimizer']
    train_cfg = cfg['trainer']

    # 1. Device Setup
    if sys_cfg['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = sys_cfg['device']
    print(f"Running on: {device}")

    os.makedirs(sys_cfg['out_dir'], exist_ok=True)

    # 2. Dataset
    tokenizer = Tokenizer()
    train_dataset = BinaryDataset(data_dir=data_cfg['data_dir'], block_size=model_cfg['block_size'], split='train')
    val_dataset = train_dataset # 简化

    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True, num_workers=data_cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=data_cfg['batch_size'], shuffle=True, num_workers=data_cfg['num_workers'])

    # 3. Model
    # 这里的关键是：vocab_size 来自 tokenizer，其他来自 config
    model_args = model_cfg.copy()
    model_args['vocab_size'] = tokenizer.vocab_size

    # GPTConfig 接收 **kwargs，所以我们可以直接传入字典
    gpt_conf = GPTConfig(**model_args)
    model = GPT(gpt_conf)
    model.to(device)

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=opt_cfg['learning_rate'], weight_decay=opt_cfg['weight_decay'])

    # 初始化训练状态变量 (默认为从头训练)
    iter_num = 0
    best_val_loss = float('inf')

    # --- 断点续训逻辑 (Resume) ---
    if sys_cfg['resume'] is not None:
        ckpt_path = sys_cfg['resume']
        if os.path.exists(ckpt_path):
            print(f"Resuming training from {ckpt_path}...")
            # map_location 确保在 cpu/cuda 之间迁移时不会报错
            checkpoint = torch.load(ckpt_path, map_location=device)

            # 1. 恢复模型权重
            # 注意：这里的模型架构配置必须与 checkpoint 里的匹配，否则会报形状不匹配错误
            model.load_state_dict(checkpoint['model'])

            # 2. 恢复优化器状态
            # 包含动量(momentum)和自适应学习率的历史信息，对 AdamW 尤为重要
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 3. 恢复标量状态
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']

            print(f"Loaded checkpoint '{ckpt_path}' (iter {iter_num}, best_loss {best_val_loss:.4f})")
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    model.train()
    iter_num = 0
    max_iters = train_cfg['max_iters']
    best_val_loss = float('inf')

    train_iter = iter(train_loader)
    pbar = tqdm(range(max_iters), desc="Training")

    for step in pbar:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        lr = get_lr(step, max_iters, opt_cfg['learning_rate'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        _, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        if (step + 1) % train_cfg['eval_interval'] == 0:
            val_loss = evaluate(model, val_loader, device, train_cfg['eval_iters'])
            tqdm.write(f"\nStep {step+1} | Val Loss: {val_loss:.4f}")

            # 保存 Checkpoint
            # 重点：我们保存整个 cfg 字典，这样以后能完全复现
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': gpt_conf, # 保存 GPTConfig 对象，方便推理加载
                'full_config': cfg, # 保存完整的 YAML 配置，方便人类查看
                'iter_num': step + 1,
                'best_val_loss': best_val_loss,
            }

            torch.save(checkpoint, os.path.join(sys_cfg['out_dir'], 'ckpt_latest.pt'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(sys_cfg['out_dir'], 'ckpt_best.pt'))


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
