"""固定对手竞技场 — 评估模型强弱.

不能用自对弈 LL_WR: 地主被冻结时农民变弱 → 地主胜率反而升高,
所以 LL_WR 升高可能是模型变差的信号.

正确做法: 候选 vs 固定参考, 两种角色各打一半, 算综合胜率.
  综合胜率 > 0.5 才说明候选真的更强.

两种评估口径:
  单模型  python3 arena.py -x models_v3 -y models -n 1000
  vote    python3 arena.py -x models_v3 -y models -n 600 --vote
          (按游戏实际策略: ADP+WP+SL 三模型 Q 值归一化后投票; --full 三个全换)

用法:
  python3 arena.py -x models_v3 -y models -n 1000
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# 批量评估限制线程数, 否则 onnxruntime 默认吃满所有核心 (实测 955% CPU).
# 必须在 import selfplay 之前设置 (它在模块加载时读取).
os.environ.setdefault('ORT_NUM_THREADS', '2')

import onnxruntime as ort
from env.env import Env, get_obs
from selfplay import predict_onnx, clear_onnx_cache

BASE = os.path.dirname(__file__)
ORT_NUM_THREADS = int(os.environ['ORT_NUM_THREADS'])

# 游戏默认的三个投票模型 (见 ai_douzero.py)
GAME_DIRS = {'adp': os.path.join(BASE, 'models'),
             'wp': os.path.join(BASE, 'models_wp_backup'),
             'sl': os.path.join(BASE, 'models_sl')}

_vote_sessions = {}


def _vote_values(obs, pos, dirs):
    """按游戏的 vote 逻辑算 Q 值: 各自 min-max 归一化后平均."""
    inp = {'z': obs['z_batch'], 'x': obs['x_batch']}
    vals = []
    for d in dirs:
        key = (d, pos)
        if key not in _vote_sessions:
            so = ort.SessionOptions()
            so.intra_op_num_threads = ORT_NUM_THREADS
            so.inter_op_num_threads = 1
            _vote_sessions[key] = ort.InferenceSession(
                os.path.join(d, f'{pos}.onnx'), so, providers=['CPUExecutionProvider'])
        v = _vote_sessions[key].run(['values'], inp)[0].flatten()
        vals.append((v - v.min()) / max(v.max() - v.min(), 1e-9))
    return np.mean(vals, axis=0)


def single_policy(model_dir):
    """单模型 argmax 策略."""
    return lambda obs, pos: int(np.argmax(predict_onnx(obs, pos, model_dir)))


def vote_policy(dirs):
    """三模型投票策略."""
    return lambda obs, pos: int(np.argmax(_vote_values(obs, pos, dirs)))


def play_episode(ll_policy, fa_policy, max_steps=200):
    """地主用 ll_policy, 两个农民都用 fa_policy, 跑一局."""
    env = Env(objective='adp')
    env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        pos = env._acting_player_position
        obs = get_obs(env.infoset)
        policy = ll_policy if pos == 'landlord' else fa_policy
        _, _, done, _ = env.step(obs['legal_actions'][policy(obs, pos)])
        steps += 1
    return env._game_winner


def arena(policy_x, policy_y, n_games):
    """X vs Y, 两种角色各一半. 返回 X 的综合/分角色胜率."""
    x_wins = ll_wins = ll_games = fa_wins = fa_games = 0

    for i in range(n_games):
        if i % 2 == 0:              # X 当地主
            winner = play_episode(policy_x, policy_y)
            ll_games += 1
            if winner == 'landlord':
                x_wins += 1
                ll_wins += 1
        else:                       # X 当农民 (Y 当地主)
            winner = play_episode(policy_y, policy_x)
            fa_games += 1
            if winner == 'farmer':
                x_wins += 1
                fa_wins += 1

    return {
        'overall': x_wins / max(n_games, 1),
        'as_landlord': ll_wins / max(ll_games, 1),
        'as_farmer': fa_wins / max(fa_games, 1),
        'n_as_landlord': ll_games,
        'n_as_farmer': fa_games,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--dir-x', type=str, required=True, help='候选模型目录')
    parser.add_argument('-y', '--dir-y', type=str, default='models', help='参考模型目录')
    parser.add_argument('-n', '--num-games', type=int, default=1000)
    parser.add_argument('--vote', action='store_true',
                        help='按游戏实际的 vote 集成评估 (而不是单模型)')
    parser.add_argument('--full', action='store_true',
                        help='vote 模式下 X 的三个模型全部换成 --dir-x')
    parser.add_argument('--wp-x', default=None, help='vote 模式下 X 的 WP 目录')
    parser.add_argument('--sl-x', default=None, help='vote 模式下 X 的 SL 目录')
    args = parser.parse_args()

    def resolve(d):
        return d if os.path.isabs(d) else os.path.join(BASE, d)

    dir_x, dir_y = resolve(args.dir_x), resolve(args.dir_y)

    if args.vote:
        x_dirs = [dir_x] * 3 if args.full else [
            dir_x,
            resolve(args.wp_x) if args.wp_x else GAME_DIRS['wp'],
            resolve(args.sl_x) if args.sl_x else GAME_DIRS['sl']]
        y_dirs = [dir_y, GAME_DIRS['wp'], GAME_DIRS['sl']]
        policy_x, policy_y = vote_policy(x_dirs), vote_policy(y_dirs)
        print(f'X 集成: {x_dirs}')
        print(f'Y 集成: {y_dirs}')
    else:
        clear_onnx_cache(dir_x)
        clear_onnx_cache(dir_y)
        policy_x, policy_y = single_policy(dir_x), single_policy(dir_y)
        print(f'X (candidate): {dir_x}')
        print(f'Y (reference): {dir_y}')

    print(f'Games: {args.num_games}\n')
    res = arena(policy_x, policy_y, args.num_games)

    print(f'X 综合胜率   : {res["overall"]:.3f}   (0.5 = 势均力敌)')
    print(f'X 当地主胜率 : {res["as_landlord"]:.3f}   ({res["n_as_landlord"]} 局)')
    print(f'X 当农民胜率 : {res["as_farmer"]:.3f}   ({res["n_as_farmer"]} 局)')
    print()
    if res['overall'] > 0.52:
        print('=> X 更强')
    elif res['overall'] < 0.48:
        print('=> X 更弱')
    else:
        print('=> 势均力敌 (差异在噪声范围内)')


if __name__ == '__main__':
    main()
