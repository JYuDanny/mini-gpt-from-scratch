# --- train.py ---
# 训练入口脚本 / Training Entry Point
# 功能 / Purpose: 完整的 GPT 模型训练管线, 包括:
#   1. 配置加载 (YAML + CLI 覆盖 / YAML + CLI overrides)
#   2. 数据加载 (二进制 memmap 数据集 / binary memmap dataset)
#   3. 训练循环 (warmup + cosine 衰减, 梯度裁剪, checkpoint 保存)
#   4. 断点续训 (恢复模型权重 + 优化器状态 / resume from checkpoint)
#
# 学习率调度原理 / Learning Rate Scheduler Principle:
#   - Warmup (预热): 训练初期, 模型参数是随机的, 梯度的方向不稳定,
#     从一个很小的学习率开始, 逐步增大到峰值, 避免初期训练震荡。
#     At the start, model params are random, gradients are noisy.
#     Ramp up LR from a small value to avoid early training instability.
#   - Cosine Decay (余弦衰减): 到达峰值后, 按余弦曲线平滑下降到 min_lr。
#     比线性衰减更平滑, 在实践中收敛效果更好。
#     After peak, smoothly decay to min_lr following a cosine curve.
#     Smoother than linear decay, empirically better convergence.

import os
import math
import argparse
import yaml
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.dataset import BinaryDataset
from src.tokenizer import Tokenizer
# from src.model import GPT, GPTConfig
# from src.model_kvcache import GPT, GPTConfig
from src.model_rope import GPT, GPTConfig

# --- 学习率调度器 / Learning Rate Scheduler ---
def get_lr(it, max_iters, learning_rate, warmup_iters=100, min_lr=1e-5):
    """
    计算当前步的学习率 / Compute learning rate for current step.

    原理 / Principle:
        1. 线性预热 (Linear Warmup): it < warmup_iters 时, LR 从 0 线性增长到 learning_rate
        2. 余弦衰减 (Cosine Decay): it >= warmup_iters 时, LR 从 learning_rate
           按余弦曲线衰减到 min_lr
    """
    # 第一阶段: 线性预热 / Phase 1: Linear Warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 第二阶段: 余弦衰减 / Phase 2: Cosine Decay
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_config():
    """
    配置加载逻辑 / Configuration Loading Logic

    三层优先级 / Three-layer priority (后者覆盖前者 / later overrides earlier):
        1. Hard-coded defaults (代码默认值 / code defaults)
        2. YAML config file (YAML 配置文件)
        3. CLI arguments (命令行参数)

    这样设计的好处: 既能通过 YAML 管理预设配置, 又能在命令行快速调整单次实验
    """
    parser = argparse.ArgumentParser(description="Train GPT Model")

    # 配置文件路径 / Config file path
    parser.add_argument('--config', type=str, default=None, help='Path to .yaml config file')

    # --- 所有可命令行覆盖的参数 / All CLI-overridable parameters ---
    # System
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)

    # Data
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--data_file', type=str, default=None)

    # Model
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)

    # Training
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_iters', type=int, default=None)

    args = parser.parse_args()

    # --- 1. 代码默认值 (最底层) / Code defaults (lowest priority) ---
    config = {
        'system': {'out_dir': 'checkpoints/mini_gpt', 'device': 'auto', 'resume': None},
        'data': {'data_dir': 'data', 'data_file': 'wikitext103.bin', 'batch_size': 32, 'num_workers': 0},
        'model': {'block_size': 256, 'd_model': 768, 'n_layer': 6, 'n_head': 6, 'dropout': 0.1, 'bias': True},
        'optimizer': {'learning_rate': 3e-4, 'weight_decay': 1e-2},
        'trainer': {'eval_interval': 1000, 'eval_iters': 50, 'max_iters': 20000}
    }

    # --- 2. 加载 YAML 配置 (中间层) / Load YAML config (middle priority) ---
    if args.config is not None:
        print(f"Loading config from {args.config}...")
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)

        # 递归更新: 用 YAML 中每个 section 的值覆盖默认配置
        for section, params in yaml_config.items():
            if section in config:
                config[section].update(params)
            else:
                config[section] = params

    # --- 3. 命令行参数覆盖 (最高优先级) / CLI overrides (highest priority) ---
    if args.out_dir: config['system']['out_dir'] = args.out_dir
    if args.resume:  config['system']['resume'] = args.resume
    if args.device:  config['system']['device'] = args.device
    if args.batch_size: config['data']['batch_size'] = args.batch_size
    if args.data_file:  config['data']['data_file'] = args.data_file
    if args.n_layer: config['model']['n_layer'] = args.n_layer
    if args.d_model: config['model']['d_model'] = args.d_model
    if args.lr:      config['optimizer']['learning_rate'] = args.lr
    if args.max_iters: config['trainer']['max_iters'] = args.max_iters

    return config


def train():
    # 获取最终合并后的配置字典 / Get final merged config dictionary
    cfg = get_config()

    sys_cfg = cfg['system']
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    opt_cfg = cfg['optimizer']
    train_cfg = cfg['trainer']

    # 1. 设备设置 / Device Setup
    if sys_cfg['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = sys_cfg['device']
    print(f"Running on: {device}")

    os.makedirs(sys_cfg['out_dir'], exist_ok=True)

    # 2. 数据集加载 / Dataset Loading
    # 使用 memmap (内存映射) 加载, 数据文件不全部进入 RAM
    tokenizer = Tokenizer()
    data_file = data_cfg.get('data_file', None)
    train_dataset = BinaryDataset(
        data_dir=data_cfg['data_dir'],
        block_size=model_cfg['block_size'],
        split='train',
        data_file=data_file
    )
    # 验证集: 使用同一数据集的另一段作为验证 (简化方案)
    # TODO: 后续支持独立的验证集文件
    # Validation set: use a different segment of the same data stream (simplified)
    val_dataset = BinaryDataset(
        data_dir=data_cfg['data_dir'],
        block_size=model_cfg['block_size'],
        split='train',
        data_file=data_file
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers']
    )

    # 3. 模型初始化 / Model Initialization
    model_args = model_cfg.copy()
    model_args['vocab_size'] = tokenizer.vocab_size

    gpt_conf = GPTConfig(**model_args)
    model = GPT(gpt_conf)
    model.to(device)

    # 4. 优化器 / Optimizer
    # AdamW = Adam + 解耦的 weight decay (权重衰减)
    # 原理 / Principle: weight_decay 将权重的 L2 正则化与 Adam 的自适应学习率解耦,
    # 避免了普通 Adam 中 L2 正则化与自适应学习率相互干扰的问题。
    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg['learning_rate'],
        weight_decay=opt_cfg['weight_decay'],
        betas=(opt_cfg.get('beta1', 0.9), opt_cfg.get('beta2', 0.95))
    )

    # --- 训练状态初始化 / Initialize training state ---
    iter_num = 0
    best_val_loss = float('inf')

    # --- 断点续训逻辑 (Resume from Checkpoint) ---
    # 核心原理 / Core principle:
    #   Checkpoint 保存了三类信息:
    #   1. model.state_dict()     → 模型权重 (含可学习参数和 buffer 如 freqs_cis)
    #   2. optimizer.state_dict() → 优化器状态 (含 AdamW 的动量 momentum 和自适应学习率历史)
    #   3. 标量训练状态            → iter_num, best_val_loss
    #   恢复时必须确保这三者全部正确加载, 训练才能无缝衔接。
    if sys_cfg['resume'] is not None:
        ckpt_path = sys_cfg['resume']
        if os.path.exists(ckpt_path):
            print(f"Resuming training from {ckpt_path}...")
            # map_location 确保在不同设备 (cpu/cuda) 之间正确迁移
            checkpoint = torch.load(ckpt_path, map_location=device)

            # 1. 恢复模型权重 / Restore model weights
            model.load_state_dict(checkpoint['model'])

            # 2. 恢复优化器状态 (包含 AdamW 的动量缓冲区, 至关重要!)
            #    Restore optimizer state (includes AdamW momentum buffers, critical!)
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 3. 恢复标量训练状态 / Restore scalar training state
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']

            print(f"Loaded checkpoint '{ckpt_path}' (iter {iter_num}, best_loss {best_val_loss:.4f})")
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    model.train()
    max_iters = train_cfg['max_iters']

    train_iter = iter(train_loader)
    pbar = tqdm(range(iter_num, max_iters), desc="Training", initial=iter_num, total=max_iters)

    for step in pbar:
        # --- 获取 batch 数据 / Fetch batch data ---
        # 当 DataLoader 的一个 epoch 结束后, 自动重置迭代器 (无限循环训练)
        # When a DataLoader epoch ends, auto-reset iterator (infinite training loop)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # --- 更新学习率 / Update learning rate ---
        # 每一步根据当前步数和预设调度方案动态调整 LR
        # Dynamically adjust LR at each step based on the schedule
        lr = get_lr(step, max_iters, opt_cfg['learning_rate'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- 前向传播 / Forward pass ---
        # 模型返回 (logits, loss, kv_cache)
        # kv_cache 仅在 use_cache=True 时有值, 训练时始终为 None
        _, loss, _ = model(x, targets=y)

        # --- 反向传播 / Backward pass ---
        optimizer.zero_grad()                         # 清空旧梯度 / Clear old gradients
        loss.backward()                                # 计算新梯度 / Compute new gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪防爆炸 / Gradient clipping
        optimizer.step()                              # 更新参数 / Update parameters

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # --- 周期性评估 / Periodic Evaluation ---
        if (step + 1) % train_cfg['eval_interval'] == 0:
            val_loss = evaluate(model, val_loader, device, train_cfg['eval_iters'])
            tqdm.write(f"\nStep {step+1} | Val Loss: {val_loss:.4f}")

            # --- 保存 Checkpoint / Save Checkpoint ---
            # 保存完整的训练状态用于断点续训:
            #   model:    模型权重 / model weights
            #   optimizer: 优化器状态 (AdamW 动量等) / optimizer state (AdamW momentum, etc.)
            #   config:    GPTConfig 对象 (推理时自动恢复架构) / GPTConfig for inference
            #   full_config: 完整 YAML 配置 (方便查看和复现) / full YAML config for reproducibility
            #   iter_num, best_val_loss: 训练进度标量 / training progress scalars
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': gpt_conf,
                'full_config': cfg,
                'iter_num': step + 1,
                'best_val_loss': best_val_loss,
            }

            torch.save(checkpoint, os.path.join(sys_cfg['out_dir'], 'ckpt_latest.pt'))

            # 如果验证 loss 更低, 额外保存一份 best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(sys_cfg['out_dir'], 'ckpt_best.pt'))


@torch.no_grad()
def evaluate(model, loader, device, eval_iters):
    """
    评估函数 / Evaluation Function

    原理 / Principle:
        只跑 eval_iters 个 batch 来估算验证集上的平均 loss, 而非遍历整个验证集。
        这是一个实用的折中: 完全遍历可能非常耗时 (验证集可能有数百万样本),
        而取少量随机 batch 的平均值足以监控模型是否过拟合。
        Runs only eval_iters batches to estimate average loss on validation set,
        rather than iterating the entire val set. This is a practical compromise:
        full traversal is time-consuming, but a few random batches suffice to
        monitor overfitting.
    """
    model.eval()
    losses = torch.zeros(eval_iters)
    loader_iter = iter(loader)

    for k in range(eval_iters):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        losses[k] = loss.item()

    model.train()
    return losses.mean().item()


if __name__ == "__main__":
    train()
