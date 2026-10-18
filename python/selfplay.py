"""自对弈收集 trajectory.

用 Env 跑完整局, 收集每个位置的 (obs, action_idx, episode_return) 供 DMC 训练.
hero_positions 里的位置用候选模型 (PyTorch), 其余位置用 baseline ONNX.
"""
import os
import numpy as np
import onnxruntime as ort

from model import predict
from env.env import Env, get_obs

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

_onnx_sessions = {}

# 批量训练时限制线程数, 否则 onnxruntime 默认吃满所有核心 (实测 955% CPU)
ORT_NUM_THREADS = int(os.environ.get('ORT_NUM_THREADS', '2'))

POSITIONS = ['landlord', 'landlord_up', 'landlord_down']


def _onnx_sess(position, model_dir=None):
    model_dir = model_dir or MODEL_DIR
    key = (position, model_dir)
    if key not in _onnx_sessions:
        so = ort.SessionOptions()
        so.intra_op_num_threads = ORT_NUM_THREADS
        so.inter_op_num_threads = 1
        _onnx_sessions[key] = ort.InferenceSession(
            os.path.join(model_dir, f'{position}.onnx'), sess_options=so)
    return _onnx_sessions[key]


def clear_onnx_cache(model_dir=None):
    """清空 ONNX session 缓存 (重新导出模型到同一目录后必须调用)."""
    if model_dir is None:
        _onnx_sessions.clear()
    else:
        for k in [k for k in _onnx_sessions if k[1] == model_dir]:
            del _onnx_sessions[k]


def predict_onnx(obs, position, model_dir=None):
    sess = _onnx_sess(position, model_dir)
    return sess.run(['values'], {
        'z': obs['z_batch'], 'x': obs['x_batch']})[0].flatten()


def run_episode(models=None, eps=0.0, model_dir=None,
                baseline_dir=None, hero_positions=None):
    """跑一局.

    models: DMCModel; None 时全部位置走 model_dir 的 ONNX.
    eps: epsilon-greedy 探索率 (只作用于 hero 位置).
    baseline_dir + hero_positions: hero 用候选模型, 其他位置用该目录 ONNX.
    只记录 hero 位置的 trajectory (baseline 的决策不是学习信号).

    Returns: trajectories, rewards, winner, num_steps
    """
    hero_positions = hero_positions or set(POSITIONS)
    env = Env(objective='adp')
    env.reset()

    trajectories = {p: [] for p in POSITIONS}
    done = False
    num_steps = 0

    while not done and num_steps < 200:
        pos = env._acting_player_position
        obs = get_obs(env.infoset)
        is_hero = pos in hero_positions

        if is_hero and eps > 0 and np.random.random() < eps:
            action_idx = np.random.randint(len(obs['legal_actions']))
        elif not is_hero and baseline_dir is not None:
            action_idx = int(np.argmax(predict_onnx(obs, pos, baseline_dir)))
        elif models is None:
            action_idx = int(np.argmax(predict_onnx(obs, pos, model_dir)))
        else:
            model = models.get_model(pos)
            model.eval()
            action_idx = int(np.argmax(predict(model, obs, 'cpu')))

        if is_hero:
            # 只存选中 action 的那一行, uint8 省 4x 内存 (特征本来就是 0/1)
            trajectories[pos].append({
                'z': obs['z_batch'][action_idx].astype(np.uint8),
                'x': obs['x_batch'][action_idx].astype(np.uint8),
                'action_idx': action_idx,
            })

        _, _, done, _ = env.step(obs['legal_actions'][action_idx])
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
