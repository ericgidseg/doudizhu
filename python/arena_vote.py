"""集成 (vote) 模式竞技场 — 评估游戏实际使用的 AI.

游戏里 AI_STYLE='vote' (见 ai_douzero.py):
  三个模型 (ADP / WP / SL) 各自推理, Q 值 min-max 归一化后取平均.

所以单模型对比不能反映游戏实际强度, 必须按 vote 方式评估.

用法:
  python3 arena_vote.py -x models_ft -y models -n 600
  python3 arena_vote.py -x models_ft -y models -n 600 --full   # 三方都换
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import onnxruntime as ort
from env.env import Env, get_obs

BASE = os.path.dirname(__file__)

# 游戏默认的三个模型目录 (见 ai_douzero.py)
DEFAULT_DIRS = {
    'adp': os.path.join(BASE, 'models'),
    'wp': os.path.join(BASE, 'models_wp_backup'),
    'sl': os.path.join(BASE, 'models_sl'),
}

# 限制 onnxruntime 线程数, 否则默认吃满所有核心 (实测 955% CPU)
ORT_NUM_THREADS = int(os.environ.get('ORT_NUM_THREADS', '2'))

_sessions = {}


def _sess(d, pos):
    key = (d, pos)
    if key not in _sessions:
        so = ort.SessionOptions()
        so.intra_op_num_threads = ORT_NUM_THREADS
        so.inter_op_num_threads = 1
        _sessions[key] = ort.InferenceSession(
            os.path.join(d, f'{pos}.onnx'), so, providers=['CPUExecutionProvider'])
    return _sessions[key]


def clear_cache():
    _sessions.clear()


def vote_values(obs, pos, dirs):
    """按游戏的 vote 逻辑算 Q 值: 各自 min-max 归一化后平均."""
    inp = {'z': obs['z_batch'], 'x': obs['x_batch']}
    vals = []
    for d in dirs:
        v = _sess(d, pos).run(['values'], inp)[0].flatten()
        rng = v.max() - v.min()
        vals.append((v - v.min()) / max(rng, 1e-9))
    return np.mean(vals, axis=0)


def play_episode(ll_dirs, fa_dirs, max_steps=200):
    """地主用 ll_dirs 的集成, 农民用 fa_dirs 的集成."""
    env = Env(objective='adp')
    env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        pos = env._acting_player_position
        obs = get_obs(env.infoset)
        dirs = ll_dirs if pos == 'landlord' else fa_dirs
        v = vote_values(obs, pos, dirs)
        _, _, done, _ = env.step(obs['legal_actions'][int(np.argmax(v))])
        steps += 1
    return getattr(env, '_game_winner', 'farmer')


def arena_vote(x_dirs, y_dirs, n_games):
    """X 集成 vs Y 集成, 两种角色各一半."""
    x_wins = ll_wins = ll_games = fa_wins = fa_games = 0
    for i in range(n_games):
        if i % 2 == 0:
            w = play_episode(x_dirs, y_dirs)
            ll_games += 1
            if w == 'landlord':
                x_wins += 1; ll_wins += 1
        else:
            w = play_episode(y_dirs, x_dirs)
            fa_games += 1
            if w == 'farmer':
                x_wins += 1; fa_wins += 1
    return {
        'overall': x_wins / max(n_games, 1),
        'as_landlord': ll_wins / max(ll_games, 1),
        'as_farmer': fa_wins / max(fa_games, 1),
    }


def make_dirs(adp_dir=None, wp_dir=None, sl_dir=None):
    d = dict(DEFAULT_DIRS)
    if adp_dir: d['adp'] = adp_dir
    if wp_dir:  d['wp'] = wp_dir
    if sl_dir:  d['sl'] = sl_dir
    return [d['adp'], d['wp'], d['sl']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-x', '--dir-x', default=None,
                   help='X 的 ADP 模型目录 (默认 models/)')
    p.add_argument('-y', '--dir-y', default=None,
                   help='Y 的 ADP 模型目录 (默认 models/)')
    p.add_argument('-n', '--num-games', type=int, default=600)
    p.add_argument('--full', action='store_true',
                   help='X 的三个模型全部换成 --dir-x')
    p.add_argument('--wp-x', default=None, help='X 的 WP 目录')
    p.add_argument('--sl-x', default=None, help='X 的 SL 目录')
    args = p.parse_args()

    def resolve(d):
        if d is None:
            return None
        return d if os.path.isabs(d) else os.path.join(BASE, d)

    x_adp = resolve(args.dir_x)
    y_adp = resolve(args.dir_y)

    if args.full:
        x_dirs = make_dirs(x_adp, x_adp, x_adp)
    else:
        x_dirs = make_dirs(x_adp, resolve(args.wp_x), resolve(args.sl_x))
    y_dirs = make_dirs(y_adp)

    print(f'X 集成: {[os.path.basename(os.path.dirname(d)) or d for d in x_dirs]}')
    print(f'Y 集成: {[os.path.basename(os.path.dirname(d)) or d for d in y_dirs]}')
    print(f'局数: {args.num_games}\n')

    clear_cache()
    r = arena_vote(x_dirs, y_dirs, args.num_games)
    print(f'X 综合胜率   : {r["overall"]:.3f}   (0.5 = 势均力敌)')
    print(f'X 当地主胜率 : {r["as_landlord"]:.3f}')
    print(f'X 当农民胜率 : {r["as_farmer"]:.3f}')
    print()
    if r['overall'] > 0.52:
        print('=> X 更强')
    elif r['overall'] < 0.48:
        print('=> X 更弱')
    else:
        print('=> 势均力敌')


if __name__ == '__main__':
    main()
