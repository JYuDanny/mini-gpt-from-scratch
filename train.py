# --- train.py ---
# 训练入口脚本 / Training Entry Point
# 功能 / Purpose: 完整的 GPT 模型训练管线, 包括:
#   1. 配置加载 (YAML + CLI 覆盖 / YAML + CLI overrides)
#   2. 数据加载 (二进制 memmap 数据集 / binary memmap dataset)
#   3. 训练循环 (warmup + cosine 衰减, 梯度裁剪, checkpoint 保存)
#   4. 断点续训 (恢复模型权重 + 优化器状态 / resume from checkpoint)
#   5. 训练效率优化: 混合精度 (AMP), 梯度累积, torch.compile, TensorBoard
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
#
# 混合精度训练原理 / Mixed Precision Training (AMP) Principle:
#   自动混合精度 (Automatic Mixed Precision) 在矩阵乘法等计算密集型操作
#   中使用 FP16/BF16 (低精度, 速度快/省显存), 在需要精度的操作中自动
#   保持 FP32。GradScaler 防止低精度下的小梯度变成 0 (梯度下溢)。
#   Uses FP16/BF16 for compute-heavy ops (fast, low VRAM), keeps FP32
#   where needed. GradScaler prevents gradient underflow.
#
# 梯度累积原理 / Gradient Accumulation Principle:
#   每次前向/反向传播后不立即更新参数, 而是累积 N 步的梯度后一次性更新。
#   等效 batch = batch_size × N, 在不增加显存的前提下模拟大批量训练。
#   Accumulate gradients for N steps before updating. Effective batch size
#   = batch_size × N, simulating large-batch training without extra VRAM.

import os
import math
import time
import argparse
import yaml
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.dataset import BinaryDataset
from src.tokenizer import Tokenizer
from src.model import GPT, GPTConfig


# ============================================================
# 学习率调度器 / Learning Rate Scheduler
# ============================================================
def get_lr(it, max_iters, learning_rate, warmup_iters=100, min_lr=1e-5):
    """
    计算当前步的学习率 / Compute learning rate for current step.

    原理 / Principle:
        1. Linear Warmup: 训练初期 LR 从 0 线性增长到 learning_rate,
           避免初期梯度震荡 / ramp up from small LR to avoid early instability.
        2. Cosine Decay: 到达峰值后按余弦曲线衰减到 min_lr,
           平滑下降比线性衰减收敛更好 / smoother than linear decay.
    """
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================
# 配置加载 / Configuration Loading
# ============================================================
def get_config():
    """
    配置加载逻辑 / Configuration Loading Logic

    三层优先级 / Three-layer priority (后者覆盖前者 / later overrides earlier):
        1. Code defaults (代码默认值)
        2. YAML config file (YAML 配置文件)
        3. CLI arguments (命令行参数)
    """
    parser = argparse.ArgumentParser(description="Train GPT Model")

    parser.add_argument('--config', type=str, default=None, help='YAML config file path')

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

    # --- 1. 代码默认值 / Code defaults ---
    config = {
        'system': {'out_dir': 'checkpoints/mini_gpt', 'device': 'auto', 'resume': None},
        'data': {'data_dir': 'data', 'data_file': 'wikitext103.bin', 'batch_size': 32, 'num_workers': 0},
        'model': {'block_size': 256, 'd_model': 768, 'n_layer': 6, 'n_head': 6,
                  'dropout': 0.1, 'bias': True, 'pos_enc_type': 'rope'},
        'optimizer': {'learning_rate': 3e-4, 'weight_decay': 1e-2, 'beta1': 0.9, 'beta2': 0.95,
                      'gradient_accumulation_steps': 1},
        'trainer': {'eval_interval': 1000, 'eval_iters': 50, 'max_iters': 20000,
                    'warmup_iters': 100, 'min_lr': 1e-5},
        'amp': {'enabled': True, 'dtype': 'bfloat16'},
        'compile': {'enabled': True},
        'logging': {'log_dir': 'logs', 'log_interval': 10},
    }

    # --- 2. YAML 配置覆盖 / YAML config override ---
    if args.config is not None:
        print(f"Loading config from {args.config}...")
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        for section, params in yaml_config.items():
            if section in config:
                if isinstance(config[section], dict) and isinstance(params, dict):
                    config[section].update(params)
                else:
                    config[section] = params
            else:
                config[section] = params

    # --- 3. CLI 参数覆盖 / CLI overrides ---
    if args.out_dir:       config['system']['out_dir'] = args.out_dir
    if args.resume:        config['system']['resume'] = args.resume
    if args.device:        config['system']['device'] = args.device
    if args.batch_size:    config['data']['batch_size'] = args.batch_size
    if args.data_file:     config['data']['data_file'] = args.data_file
    if args.n_layer:       config['model']['n_layer'] = args.n_layer
    if args.d_model:       config['model']['d_model'] = args.d_model
    if args.n_head:        config['model']['n_head'] = args.n_head
    if args.dropout:       config['model']['dropout'] = args.dropout
    if args.lr:            config['optimizer']['learning_rate'] = args.lr
    if args.max_iters:     config['trainer']['max_iters'] = args.max_iters

    return config


# ============================================================
# TensorBoard 日志 / TensorBoard Logger
# ============================================================
def setup_logging(cfg):
    """
    初始化 TensorBoard 日志记录器 / Initialize TensorBoard logger

    原理 / Principle:
        SummaryWriter 将标量 (loss, lr, tokens/s) 写入日志文件。
        运行 `tensorboard --logdir=logs` 可在浏览器中实时查看训练曲线。
        SummaryWriter writes scalars to log files. Run
        `tensorboard --logdir=logs` to view training curves in browser.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = cfg['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging enabled: {log_dir}")
        return writer
    except ImportError:
        print("TensorBoard not available (pip install tensorboard)")
        return None


# ============================================================
# 评估函数 / Evaluation Function
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, eval_iters, scaler, amp_cfg, compile_cfg):
    """
    评估函数 / Evaluation Function

    取 eval_iters 个随机 batch 估算验证集平均 loss。
    少量 batch 足以监控过拟合趋势, 无需遍历全量验证集。
    Average loss over eval_iters random batches — sufficient for
    monitoring overfitting without full val set traversal.
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

        # 评估时也用 AMP 加速 (仅前向, 不需要 GradScaler)
        if amp_cfg.get('enabled') and device == 'cuda':
            dtype = torch.bfloat16 if amp_cfg.get('dtype') == 'bfloat16' else torch.float16
            with torch.autocast(device_type='cuda', dtype=dtype):
                _, loss, _ = model(x, targets=y)
        else:
            _, loss, _ = model(x, targets=y)

        losses[k] = loss.item()

    model.train()
    return losses.mean().item()


# ============================================================
# 主训练函数 / Main Training Function
# ============================================================
def train():
    cfg = get_config()

    sys_cfg = cfg['system']
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    opt_cfg = cfg['optimizer']
    train_cfg = cfg['trainer']
    amp_cfg = cfg.get('amp', {'enabled': False, 'dtype': 'float16'})
    compile_cfg = cfg.get('compile', {'enabled': False})
    logging_cfg = cfg.get('logging', {})

    # 1. 设备设置 / Device Setup
    if sys_cfg['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = sys_cfg['device']
    print(f"Running on: {device}")

    os.makedirs(sys_cfg['out_dir'], exist_ok=True)

    # 2. 数据集加载 / Dataset Loading
    tokenizer = Tokenizer()
    data_file = data_cfg.get('data_file', None)
    train_dataset = BinaryDataset(
        data_dir=data_cfg['data_dir'], block_size=model_cfg['block_size'],
        split='train', data_file=data_file
    )
    val_dataset = BinaryDataset(
        data_dir=data_cfg['data_dir'], block_size=model_cfg['block_size'],
        split='train', data_file=data_file
    )

    train_loader = DataLoader(
        train_dataset, batch_size=data_cfg['batch_size'],
        shuffle=True, num_workers=data_cfg['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=data_cfg['batch_size'],
        shuffle=True, num_workers=data_cfg['num_workers']
    )

    # 3. 模型初始化 / Model Initialization
    model_args = model_cfg.copy()
    model_args['vocab_size'] = tokenizer.vocab_size
    gpt_conf = GPTConfig(**model_args)
    model = GPT(gpt_conf)
    model.to(device)

    # --- torch.compile 加速 / torch.compile acceleration ---
    # 原理 / Principle:
    #   torch.compile 将 PyTorch 模型的计算图编译为优化的 Triton/C++ kernel,
    #   减少 Python 解释器开销和 kernel launch 次数, 通常提速 20-50%。
    #   torch.compile compiles the model graph into optimized kernels,
    #   reducing Python overhead and kernel launch count.
    if compile_cfg.get('enabled') and hasattr(torch, 'compile') and device == 'cuda':
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model, mode=compile_cfg.get('mode', 'reduce-overhead'))
            print("  Model compiled successfully.")
        except Exception as e:
            print(f"  torch.compile failed: {e}. Continuing without compile.")
    elif compile_cfg.get('enabled'):
        if device != 'cuda':
            print("  torch.compile disabled: only supported on CUDA. Continuing without.")
        else:
            print("  torch.compile not available (requires PyTorch 2.0+). Continuing without.")

    # 4. 优化器 / Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg['learning_rate'],
        weight_decay=opt_cfg['weight_decay'],
        betas=(opt_cfg.get('beta1', 0.9), opt_cfg.get('beta2', 0.95))
    )

    # --- 混合精度训练 / Mixed Precision Training ---
    # 原理 / Principle:
    #   GradScaler 在反向传播前将 loss 乘以一个缩放因子, 防止 FP16/BF16
    #   下的小梯度变成 0 (梯度下溢)。反向传播后再除以该因子恢复真实梯度。
    #   GradScaler scales loss before backward to prevent small FP16/BF16
    #   gradients from vanishing to zero (gradient underflow).
    use_amp = amp_cfg.get('enabled', False) and device == 'cuda'
    amp_dtype = torch.bfloat16 if amp_cfg.get('dtype') == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
    # 注意 / Note: bfloat16 不需要 GradScaler (指数位与 FP32 相同, 无下溢风险)
    #   bfloat16 doesn't need GradScaler (same exponent range as FP32, no underflow)
    if use_amp and amp_dtype == torch.bfloat16:
        scaler = torch.amp.GradScaler('cuda', enabled=False)

    if use_amp:
        dtype_name = amp_cfg.get('dtype', 'float16')
        print(f"AMP enabled: {dtype_name} (GradScaler: {scaler.is_enabled()})")

    # --- TensorBoard 日志 / TensorBoard Logging ---
    writer = setup_logging(cfg) if logging_cfg.get('log_dir') else None

    # --- 训练状态 / Training State ---
    iter_num = 0
    best_val_loss = float('inf')

    # 5. 断点续训 / Resume from Checkpoint
    if sys_cfg['resume'] is not None:
        ckpt_path = sys_cfg['resume']
        if os.path.exists(ckpt_path):
            print(f"Resuming training from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 恢复 scaler 状态 (FP16 AMP 的缩放因子历史)
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])

            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            print(f"Loaded checkpoint (iter {iter_num}, best_loss {best_val_loss:.4f})")
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    model.train()
    max_iters = train_cfg['max_iters']
    warmup_iters = train_cfg.get('warmup_iters', 100)
    min_lr = train_cfg.get('min_lr', 1e-5)
    grad_accum_steps = opt_cfg.get('gradient_accumulation_steps', 1)
    log_interval = logging_cfg.get('log_interval', 10)

    effective_batch = data_cfg['batch_size'] * grad_accum_steps
    print(f"Gradient accumulation: {grad_accum_steps} steps "
          f"(effective batch={effective_batch})")

    train_iter = iter(train_loader)
    pbar = tqdm(range(iter_num, max_iters), desc="Training",
                initial=iter_num, total=max_iters)

    # 训练速度统计 / Throughput statistics
    tokens_processed = 0
    time_start = time.perf_counter()

    for step in pbar:
        # --- 获取 batch / Fetch batch ---
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # --- 学习率调度 / LR Scheduling ---
        lr = get_lr(step, max_iters, opt_cfg['learning_rate'],
                    warmup_iters=warmup_iters, min_lr=min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- 前向 + 反向传播 (AMP + 梯度累积) / Forward + Backward ---
        if use_amp:
            # AMP 前向: 计算密集型操作自动使用低精度
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                _, loss, _ = model(x, targets=y)
                # 梯度累积: loss 除以累积步数, 使得总梯度量级与单步一致
                loss = loss / grad_accum_steps
        else:
            _, loss, _ = model(x, targets=y)
            loss = loss / grad_accum_steps

        # 反向传播: loss 缩放 (仅 FP16) + 计算梯度
        scaler.scale(loss).backward()

        # --- 每 grad_accum_steps 步更新一次参数 ---
        # 原理 / Principle:
        #   不立即更新, 而是累积梯度。累积 N 步后, 梯度已经平均了 N 个 batch
        #   的信息, 等效于 batch_size × N 的效果。
        #   Accumulate gradients for N steps before updating, averaging
        #   information from N batches for effective larger batch size.
        if (step + 1) % grad_accum_steps == 0:
            # 梯度裁剪 (在 scaler 缩放状态下进行)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 参数更新 (scaler 自动处理 FP16 的缩放恢复)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # --- 吞吐量统计 / Throughput statistics ---
        tokens_processed += x.numel()
        pbar.set_postfix(
            loss=f"{loss.item() * grad_accum_steps:.4f}",
            lr=f"{lr:.2e}"
        )

        # --- TensorBoard 日志 / TensorBoard logging ---
        if writer and (step + 1) % log_interval == 0:
            elapsed = time.perf_counter() - time_start
            tokens_per_sec = tokens_processed / max(elapsed, 0.001)
            writer.add_scalar('train/loss', loss.item() * grad_accum_steps, step)
            writer.add_scalar('train/lr', lr, step)
            writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)

        # --- 周期性评估 / Periodic Evaluation ---
        if (step + 1) % train_cfg['eval_interval'] == 0:
            val_loss = evaluate(
                model, val_loader, device,
                train_cfg['eval_iters'], scaler, amp_cfg, compile_cfg
            )
            elapsed = time.perf_counter() - time_start
            tokens_per_sec = tokens_processed / max(elapsed, 0.001)

            tqdm.write(
                f"\nStep {step+1} | Val Loss: {val_loss:.4f} | "
                f"Tokens/sec: {tokens_per_sec:.0f}"
            )

            if writer:
                writer.add_scalar('val/loss', val_loss, step + 1)
                writer.add_scalar('val/tokens_per_sec', tokens_per_sec, step + 1)

            # --- 保存 Checkpoint / Save Checkpoint ---
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'config': gpt_conf,
                'full_config': cfg,
                'iter_num': step + 1,
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, os.path.join(sys_cfg['out_dir'], 'ckpt_latest.pt'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(sys_cfg['out_dir'], 'ckpt_best.pt'))
                tqdm.write(f"  -> New best model saved! (val_loss={best_val_loss:.4f})")

    # 训练结束 / Training Complete
    if writer:
        writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
