"""自对弈收集 trajectory.

用 Env 跑完整局, 三家都用同一个模型 (或不同 checkpoint) 决策,
收集每个位置的 (obs, action_idx, episode_return) 供 DMC 训练.

用法:
  python3 selfplay.py                     # 跑 10 局 demo
  python3 selfplay.py -n 100              # 跑 100 局
  python3 selfplay.py -n 100 -o replay.pkl  # 保存经验
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
import onnxruntime as ort
from model import DMCModel, predict, load_onnx_weights, X_DIMS, SEQ_LEN, LSTM_INPUT
from env.env import Env, get_obs

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

_onnx_sessions = {}

# 限制 onnxruntime 线程数. 默认会用满所有核心 (实测 955% CPU),
# 长时间训练会拖慢机器上其他工作. 小 batch 推理用 2 线程足够.
ORT_NUM_THREADS = int(os.environ.get('ORT_NUM_THREADS', '2'))


def _make_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_NUM_THREADS
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so)


def _onnx_sess(position, model_dir=None):
    """ONNX 推理 session (比 PyTorch MPS 快)."""
    if model_dir is None:
        model_dir = MODEL_DIR
    key = (position, model_dir)
    if key not in _onnx_sessions:
        _onnx_sessions[key] = _make_session(
            os.path.join(model_dir, f'{position}.onnx'))
    return _onnx_sessions[key]


def register_onnx_session(position, model_dir, sess):
    """注册外部 ONNX session (避免重复加载)."""
    _onnx_sessions[(position, model_dir)] = sess


def clear_onnx_cache(model_dir=None):
    """清空 ONNX session 缓存.

    重新导出模型到同一目录后必须调用, 否则会用到旧的 session.
    """
    if model_dir is None:
        _onnx_sessions.clear()
    else:
        for k in [k for k in _onnx_sessions if k[1] == model_dir]:
            del _onnx_sessions[k]


def predict_onnx(obs, position, model_dir=None):
    """用 ONNX 推理 (自对弈中更快)."""
    sess = _onnx_sess(position, model_dir)
    return sess.run(['values'], {
        'z': obs['z_batch'], 'x': obs['x_batch']})[0].flatten()


def run_episode(models=None, eps=0.0, device='cpu', model_dir=None,
                use_onnx=None):
    """跑一局自对弈.

    Args:
        models: DMCModel (PyTorch 推理) 或 None (用 ONNX)
        eps: epsilon-greedy 探索率
        device: torch device (当 models 不为 None 时)
        model_dir: ONNX 模型目录 (当 models 为 None 时)
        use_onnx: 强制用 ONNX 推理

    Returns:
        trajectories, rewards, winner, num_steps
    """
    if use_onnx is None:
        use_onnx = (models is None)

    env = Env(objective='adp')
    env.reset()

    trajectories = {p: [] for p in ['landlord', 'landlord_up', 'landlord_down']}

    done = False
    num_steps = 0
    max_steps = 200

    while not done and num_steps < max_steps:
        pos = env._acting_player_position
        obs = get_obs(env.infoset)

        if np.random.random() < eps:
            action_idx = np.random.randint(len(obs['legal_actions']))
        elif use_onnx:
            values = predict_onnx(obs, pos, model_dir)
            action_idx = int(np.argmax(values))
        else:
            # PyTorch 推理 — 用 CPU (MPS 对小 tensor 有 overhead)
            model = models.get_model(pos) if hasattr(models, 'get_model') else models[pos]
            was_training = model.training
            model.eval()
            values = predict(model, obs, 'cpu')
            if was_training:
                model.train()
            action_idx = int(np.argmax(values))

        action = obs['legal_actions'][action_idx]

        # 只存选中 action 的特征行 (省内存: 每条 ~1KB vs ~10KB)
        # 特征是 0/1 二值, 存 uint8 无损, 比 float32 省 4x
        trajectories[pos].append({
            'z': obs['z_batch'][action_idx].astype(np.uint8),   # (5, 162)
            'x': obs['x_batch'][action_idx].astype(np.uint8),   # (x_dim,)
            'action_idx': action_idx,
            'num_legal': len(obs['legal_actions']),
        })

        obs_next, reward, done, _ = env.step(action)
        num_steps += 1

    winner = env._game_winner
    bomb_num = env._game_bomb_num

    if winner == 'landlord':
        rewards = {'landlord': 2.0 ** bomb_num,
                   'landlord_up': -2.0 ** bomb_num,
                   'landlord_down': -2.0 ** bomb_num}
    else:
        rewards = {'landlord': -2.0 ** bomb_num,
                   'landlord_up': 2.0 ** bomb_num,
                   'landlord_down': 2.0 ** bomb_num}

    return trajectories, rewards, winner, num_steps


def collect_episodes(models=None, n_episodes=10, eps=0.0, device='cpu',
                     model_dir=None, progress_every=10):
    """跑 n_episodes 局, 返回所有 trajectory 和统计."""
    all_data = []
    stats = {'landlord_win': 0, 'farmer_win': 0,
             'total_steps': 0, 'total_reward': 0.0}

    t0 = time.time()
    for i in range(n_episodes):
        traj, rewards, winner, steps = run_episode(
            models, eps, device, model_dir)

        stats['total_steps'] += steps
        stats['total_reward'] += rewards['landlord']
        if winner == 'landlord':
            stats['landlord_win'] += 1
        else:
            stats['farmer_win'] += 1

        all_data.append({
            'trajectories': traj,
            'rewards': rewards,
            'winner': winner,
        })

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            ll_wr = stats['landlord_win'] / (i + 1)
            avg_r = stats['total_reward'] / (i + 1)
            avg_steps = stats['total_steps'] / (i + 1)
            print(f'  [{i+1}/{n_episodes}] LL_WR={ll_wr:.2f} '
                  f'avg_R={avg_r:.2f} avg_steps={avg_steps:.0f} '
                  f'({elapsed:.0f}s)')

    return all_data, stats


def episodes_to_batch(episodes, position):
    """把 episodes 转成该位置的 (obs, action_idx, reward) 列表."""
    samples = []
    for ep in episodes:
        reward = ep['rewards'][position]
        for step in ep['trajectories'][position]:
            samples.append({
                'z_batch': step['z_batch'][step['action_idx']:step['action_idx']+1],
                'x_batch': step['x_batch'][step['action_idx']:step['action_idx']+1],
                'reward': reward,
            })
    return samples


def save_replay(path, episodes):
    """保存经验到 pickle."""
    with open(path, 'wb') as f:
        pickle.dump(episodes, f)
    print(f'Saved {len(episodes)} episodes to {path}')


def load_replay(path):
    """加载经验."""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=10)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--model-dir', type=str, default=None,
                        help='ONNX model dir (default: models/)')
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    print(f'Running {args.num} self-play episodes (eps={args.eps})...')
    episodes, stats = collect_episodes(
        models=None, n_episodes=args.num, eps=args.eps,
        model_dir=args.model_dir)

    ll_wr = stats['landlord_win'] / args.num
    avg_r = stats['total_reward'] / args.num
    avg_steps = stats['total_steps'] / args.num
    print(f'\nResults: LL_WR={ll_wr:.2f}, avg_reward={avg_r:.2f}, '
          f'avg_steps={avg_steps:.0f}')

    if args.output:
        save_replay(args.output, episodes)
