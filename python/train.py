"""DMC (Deep Monte Carlo) 训练 — DouZero 原生算法.

核心思路:
  - 自对弈跑完整 episode, 得到每个位置的 ADP reward
  - 对该位置每个 action 的 Q(s,a) 做 MSE(Q(s,a), episode_return) 回归
  - 不需要 target network, 不需要 TD bootstrapping — 纯 MC

DMC 比 DQN 稳定得多, 因为:
  - 没有 bootstrap 偏差
  - 没有 moving target
  - 每个样本的 target 就是实际 ADP

用法:
  python3 train.py                          # 默认训练循环
  python3 train.py --games-per-iter 50      # 每轮 50 局
  python3 train.py --iterations 20          # 训练 20 轮
  python3 train.py --resume checkpoint.pt   # 从 checkpoint 继续
"""
import os
import sys
import time
import copy
import pickle
import argparse
import numpy as np
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# 输出重定向到文件时保持行缓冲, 否则进度看不到
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim

from model import DMCModel, export_to_onnx
from selfplay import run_episode, clear_onnx_cache

DEVICE = 'cpu'  # CPU 推理自对弈, MPS 收益小开销大
POSITIONS = ['landlord', 'landlord_up', 'landlord_down']

# 输出目录
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'models_v2')


# ---------------------------------------------------------------------------
# 数据收集
# ---------------------------------------------------------------------------

def collect_batch(models, n_games, eps, device,
                  baseline_dir=None, hero_positions=None):
    """跑 n_games 局, 返回 {position: [(z, x, action_idx, reward), ...]}.

    直接用 PyTorch CPU 推理自对弈 — 不需要 ONNX 导出.
    训练用 device 梯度, 推理用 CPU (MPS 对小 batch 有 overhead).

    baseline_dir: 若提供, hero_positions 用候选 models (PyTorch),
                  其他位置用 baseline_dir 的 ONNX. 让训练信号
                  变成「打赢固定参考」而非「打赢自己」.
    """
    batch_data = {p: [] for p in POSITIONS}
    stats = {'ll_win': 0, 'total_steps': 0, 'total_reward': 0.0}

    # 模型已经在 device 上, predict 内部会处理 device transfer
    for _ in range(n_games):
        traj, rewards, winner, steps = run_episode(
            models=models, eps=eps, device=device,
            baseline_dir=baseline_dir, hero_positions=hero_positions)
        stats['total_steps'] += steps
        stats['total_reward'] += rewards['landlord']
        if winner == 'landlord':
            stats['ll_win'] += 1

        for pos in POSITIONS:
            for step in traj[pos]:
                batch_data[pos].append({
                    'z': step['z'],           # (5, 162)
                    'x': step['x'],           # (x_dim,)
                    'reward': rewards[pos],
                })

    return batch_data, stats


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """按位置分的经验回放池."""

    def __init__(self, max_size=100000):
        self.buffers = {p: deque(maxlen=max_size) for p in POSITIONS}

    def add(self, position, item):
        self.buffers[position].append(item)

    def add_batch(self, batch_data):
        for pos in POSITIONS:
            for item in batch_data[pos]:
                self.add(pos, item)

    def sample(self, position, batch_size):
        buf = self.buffers[position]
        n = min(batch_size, len(buf))
        indices = np.random.choice(len(buf), n, replace=False)
        return [buf[i] for i in indices]

    def size(self, position=None):
        if position:
            return len(self.buffers[position])
        return sum(len(b) for b in self.buffers.values())


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def train_iteration(models, optimizers, buffer, args, device):
    """一轮训练: 收集经验 -> 各位置训练 -> 返回统计."""

    # 1. 收集经验 (PyTorch CPU 推理)
    # baseline_dir + hero_positions: 非 hero 位置用固定 ONNX 基线,
    # 训练信号从"打赢自己"变成"打赢基线"
    t0 = time.time()
    baseline_dir = getattr(args, 'baseline_dir', None)
    hero_positions = getattr(args, 'hero_positions', None)
    batch_data, stats = collect_batch(
        models, args.games_per_iter, args.eps, device,
        baseline_dir=baseline_dir, hero_positions=hero_positions)
    buffer.add_batch(batch_data)
    collect_time = time.time() - t0

    # 2. 每个位置训练
    train_stats = {}
    for pos in POSITIONS:
        model = models.get_model(pos)

        # 跳过 frozen 位置
        if pos in getattr(args, 'freeze_set', set()):
            model.eval()
            train_stats[pos] = {'loss': 0.0, 'n_samples': len(batch_data[pos])}
            continue

        model.train()
        pos_losses = []
        n_batches = args.train_batches_per_iter

        for _ in range(n_batches):
            batch = buffer.sample(pos, args.batch_size)
            if not batch:
                continue

            # 构造 batch tensors — 取每个样本选中 action 的 obs
            # z: (B, 5, 162), x: (B, x_dim)
            z_list, x_list, r_list = [], [], []
            for item in batch:
                z_list.append(item['z'])                 # (5, 162)
                x_list.append(item['x'])                 # (x_dim,)
                r_list.append(item['reward'])

            z_t = torch.from_numpy(np.stack(z_list)).float().to(device)
            x_t = torch.from_numpy(np.stack(x_list)).float().to(device)
            r_t = torch.tensor(r_list, dtype=torch.float32, device=device)

            values = model(z_t, x_t).squeeze(-1)
            # Huber loss 更稳定 (对 outlier reward 不敏感)
            loss = nn.functional.smooth_l1_loss(values, r_t)

            optimizers[pos].zero_grad()
            loss.backward()
            # 梯度裁剪防爆
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizers[pos].step()

            pos_losses.append(loss.item())

        model.eval()
        train_stats[pos] = {
            'loss': np.mean(pos_losses) if pos_losses else 0.0,
            'n_samples': len(batch_data[pos]),
        }

    return stats, train_stats, collect_time


def evaluate(models, n_games, tmp_dir, ref_dir=None):
    """评估当前模型 — 用固定参考对手, 而不是自对弈.

    为什么不用自对弈 LL_WR: 冻结地主时农民变弱会让 LL_WR 升高,
    这个指标会把变差的模型判为变好.

    返回 (综合胜率, 当地主胜率, 当农民胜率).
    综合胜率 = X vs 参考对手, 两种角色各一半的平均胜率.
    """
    # 导出当前权重到临时目录
    for pos in POSITIONS:
        m = models.get_model(pos)
        m_cpu = copy.deepcopy(m).cpu()
        export_to_onnx(m_cpu, os.path.join(tmp_dir, f'{pos}.onnx'), opset=14)
        del m_cpu

    # 必须清缓存, 否则会用到上一轮的旧 session
    clear_onnx_cache(tmp_dir)

    if ref_dir is None:
        ref_dir = os.path.join(os.path.dirname(__file__), 'models')

    from arena import arena, single_policy
    res = arena(single_policy(tmp_dir), single_policy(ref_dir), n_games)
    return res['overall'], res['as_landlord'], res['as_farmer']


def save_checkpoint(models, optimizers, iteration, stats, path):
    """保存训练 checkpoint."""
    ckpt = {
        'iteration': iteration,
        'stats': stats,
        'timestamp': datetime.now().isoformat(),
    }
    for pos in POSITIONS:
        ckpt[f'model_{pos}'] = models.get_model(pos).state_dict()
        ckpt[f'opt_{pos}'] = optimizers[pos].state_dict()
    torch.save(ckpt, path)
    print(f'  Checkpoint saved: {path}')


def load_checkpoint(models, optimizers, path, device):
    """加载训练 checkpoint.

    注意: optimizer.load_state_dict 会把存档里的 lr 一并恢复,
    从而覆盖命令行 --lr/--freeze 的配置. 所以这里先记下当前 lr,
    加载后再写回, 保证命令行参数优先.
    """
    ckpt = torch.load(path, map_location=device)
    for pos in POSITIONS:
        models.get_model(pos).load_state_dict(ckpt[f'model_{pos}'])
        if optimizers:
            wanted_lr = optimizers[pos].param_groups[0]['lr']
            optimizers[pos].load_state_dict(ckpt[f'opt_{pos}'])
            # 恢复命令行指定的 lr (不被存档覆盖)
            for g in optimizers[pos].param_groups:
                g['lr'] = wanted_lr
    print(f'  Checkpoint loaded: {path} (iter={ckpt["iteration"]})')
    return ckpt['iteration']


def export_models(models, export_dir, device):
    """导出三个模型为 ONNX."""
    os.makedirs(export_dir, exist_ok=True)
    for pos in POSITIONS:
        model = models.get_model(pos)
        model_cpu = copy.deepcopy(model).cpu()
        path = os.path.join(export_dir, f'{pos}.onnx')
        export_to_onnx(model_cpu, path, opset=14)
        # 移回原设备
        del model_cpu
    print(f'  Exported to {export_dir}/')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DMC Training for DouZero')
    parser.add_argument('--iterations', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--games-per-iter', type=int, default=50,
                        help='每轮自对弈局数')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='训练 batch size')
    parser.add_argument('--train-batches-per-iter', type=int, default=20,
                        help='每轮每个位置训练多少个 batch')
    parser.add_argument('--lr', type=float, default=3e-6,
                        help='学习率 (默认 3e-6: 微调已收敛模型, 高 lr 会过拟合)')
    parser.add_argument('--lr-landlord', type=float, default=None,
                        help='地主学习率 (默认=lr)')
    parser.add_argument('--freeze', type=str, default=None,
                        help='冻结位置, e.g. "landlord" or "landlord_up,landlord_down"')
    parser.add_argument('--eps', type=float, default=0.05,
                        help='epsilon-greedy 探索率')
    parser.add_argument('--eps-decay', type=float, default=0.995,
                        help='每轮 epsilon 衰减')
    parser.add_argument('--buffer-size', type=int, default=300000,
                        help='经验回放池大小')
    parser.add_argument('--eval-every', type=int, default=5,
                        help='每几轮评估一次')
    parser.add_argument('--eval-games', type=int, default=30,
                        help='评估时跑多少局')
    parser.add_argument('--export-every', type=int, default=0,
                        help='每几轮导出 ONNX (0=仅结束时导出)')
    parser.add_argument('--ckpt-every', type=int, default=10,
                        help='每几轮保存 checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='初始 ONNX 模型目录')
    parser.add_argument('--export-dir', type=str, default=None,
                        help='导出 ONNX 目录')
    parser.add_argument('--min-improvement', type=float, default=0.02,
                        help='导出所需的最小 arena 提升 (默认 0.02, '
                             '低于此值不导出, 避免用变差的模型覆盖)')
    parser.add_argument('--baseline-dir', type=str, default=None,
                        help='对手 ONNX 目录. 配合 --hero-positions 使用: '
                             '非 hero 位置由它决策 (默认 models/). '
                             '评估时也用它作为参考对手.')
    parser.add_argument('--hero-positions', type=str, default=None,
                        help='候选模型占据的位置 (逗号分隔, e.g. "landlord" '
                             '或 "landlord_up,landlord_down"). '
                             '其余位置用 --baseline-dir 的 ONNX. '
                             '默认 None = 三个位置都用候选 (传统自对弈). '
                             '提供后, 训练信号从「打赢自己」变「打赢基线」, '
                             '更接近真实强度提升.')
    args = parser.parse_args()
    if args.hero_positions:
        args.hero_positions = set(args.hero_positions.split(','))
    else:
        args.hero_positions = None

    device = DEVICE
    export_dir = args.export_dir or EXPORT_DIR

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    print(f'=== DMC Training ===')
    print(f'Device: {device}')
    print(f'Iterations: {args.iterations}')
    print(f'Games/iter: {args.games_per_iter}')
    print(f'Batch size: {args.batch_size}')
    print(f'LR: {args.lr}')
    print(f'Epsilon: {args.eps} -> decay {args.eps_decay}')
    print()

    # 加载模型 (自对弈推理用 CPU, MPS 对小 tensor 开销大)
    print('Loading ONNX weights...')
    models = DMCModel()
    models.load_onnx_all(args.model_dir)
    models.eval()

    # Optimizer — 可选冻结某些位置
    args.freeze_set = set()
    if args.freeze:
        args.freeze_set = set(args.freeze.split(','))
        print(f'Frozen positions: {args.freeze_set}')

    optimizers = {}
    for pos in POSITIONS:
        lr = args.lr_landlord if pos == 'landlord' and args.lr_landlord else args.lr
        if pos in args.freeze_set:
            lr = 0.0
            # 冻结参数
            for p in models.get_model(pos).parameters():
                p.requires_grad = False
        optimizers[pos] = optim.Adam(
            models.get_model(pos).parameters(), lr=lr)
        print(f'  {pos}: lr={lr}')

    # Resume
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(models, optimizers, args.resume, device)

    # Replay buffer
    buffer = ReplayBuffer(max_size=args.buffer_size)

    # 临时目录: 每轮导出当前权重, 供 arena 评估用
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='dmc_eval_')

    # Training loop
    # best_wr 从 0.5 起步: 只有真正打赢基线才算 "best".
    # 若从 0.0 起步, 第一次评估(哪怕只有 0.45, 比基线差)也会被存成 best.
    best_wr = 0.5
    history = []
    eps = args.eps

    try:
        for it in range(start_iter, args.iterations):
            t0 = time.time()

            # 收集 + 训练
            stats, train_stats, collect_time = train_iteration(
                models, optimizers, buffer, args, device)

            total_time = time.time() - t0
            ll_wr = stats['ll_win'] / args.games_per_iter
            avg_r = stats['total_reward'] / args.games_per_iter

            print(f'Iter {it+1}/{args.iterations} | '
                  f'LL_WR={ll_wr:.2f} avg_R={avg_r:.2f} | '
                  f'loss: L={train_stats["landlord"]["loss"]:.4f} '
                  f'U={train_stats["landlord_up"]["loss"]:.4f} '
                  f'D={train_stats["landlord_down"]["loss"]:.4f} | '
                  f'buf={buffer.size()} | '
                  f'eps={eps:.3f} | '
                  f'{total_time:.0f}s (collect {collect_time:.0f}s)')

            history.append({
                'iter': it + 1,
                'll_wr': ll_wr,
                'avg_reward': avg_r,
                'losses': {p: train_stats[p]['loss'] for p in POSITIONS},
            })

            # 评估 — 用固定参考对手的 arena 胜率 (自对弈 LL_WR 会误导)
            if (it + 1) % args.eval_every == 0:
                wr, wr_ll, wr_fa = evaluate(
                    models, args.eval_games, tmp_dir, args.baseline_dir)
                print(f'  >> Arena vs baseline: 综合={wr:.3f} '
                      f'(地主 {wr_ll:.3f} / 农民 {wr_fa:.3f})')
                history[-1]['arena_wr'] = wr
                history[-1]['arena_as_landlord'] = wr_ll
                history[-1]['arena_as_farmer'] = wr_fa

                if wr > best_wr:
                    best_wr = wr
                    save_checkpoint(models, optimizers, it + 1,
                                    {'arena_wr': wr, 'best': True},
                                    os.path.join(CKPT_DIR, 'best.pt'))
                    print(f'  >> New best (arena_wr={wr:.3f})')

            # Checkpoint
            if (it + 1) % args.ckpt_every == 0:
                save_checkpoint(models, optimizers, it + 1,
                                {'ll_wr': ll_wr, 'avg_r': avg_r},
                                os.path.join(CKPT_DIR, 'latest.pt'))

            # Export ONNX (仅当 --export-every > 0 时)
            if args.export_every > 0 and (it + 1) % args.export_every == 0:
                export_models(models, export_dir, device)

            # Decay epsilon
            eps = max(0.01, eps * args.eps_decay)

    except KeyboardInterrupt:
        print('\n\nInterrupted by user. Saving checkpoint...')
        save_checkpoint(models, optimizers, -1,
                        {'interrupted': True},
                        os.path.join(CKPT_DIR, 'interrupted.pt'))
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 保存 checkpoint (总是保存, 便于事后分析)
    save_checkpoint(models, optimizers, args.iterations,
                    {'final': True},
                    os.path.join(CKPT_DIR, 'final.pt'))

    hist_path = os.path.join(CKPT_DIR, 'history.pkl')
    with open(hist_path, 'wb') as f:
        pickle.dump(history, f)

    # 只在确实超过基线时才导出 ONNX.
    # 否则会把变差的模型写进 export_dir, 覆盖掉好的.
    print('\n=== Training Complete ===')
    print(f'Best arena win rate vs baseline: {best_wr:.3f}')
    print(f'Training history saved to {hist_path}')

    if best_wr >= 0.5 + args.min_improvement:
        export_models(models, export_dir, device)
        print(f'Exported models to {export_dir}/')
    else:
        print(f'  NOT exported: 未超过基线 {0.5 + args.min_improvement:.2f} '
              f'(需要 +{args.min_improvement:.2f} 才算真实提升).')
        print(f'  当前最优仍是: models/ (游戏默认配置, 无需改动).')
        print(f'  checkpoint 在 {CKPT_DIR}/final.pt, 可事后分析.')


if __name__ == '__main__':
    main()
