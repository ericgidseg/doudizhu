"""固定对手竞技场 — 正确评估模型强弱.

为什么不能用自对弈 LL_WR:
  地主被冻结时, 农民变弱 → 地主胜率反而升高.
  所以自对弈 LL_WR 升高可能是模型变差的信号, 不能用来选模型.

正确做法: 候选模型 vs 固定参考模型, 两种角色各打一半, 算候选的综合胜率.
  综合胜率 > 0.5 才说明候选真的更强.

用法:
  python3 arena.py -x models_v3 -y models -n 1000
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.env import Env, get_obs
from selfplay import predict_onnx, clear_onnx_cache

# 限制 onnxruntime 线程数: 默认会用满所有核心 (实测 955% CPU),
# 长时间跑会明显影响机器其他工作.
_NUM_THREADS = int(os.environ.get('ORT_NUM_THREADS', '2'))


def run_episode_mixed(landlord_dir, farmer_dir, eps=0.0, max_steps=200):
    """地主用 landlord_dir, 两个农民都用 farmer_dir, 跑一局.

    返回 (winner, bomb_num). winner ∈ {'landlord', 'farmer'}.
    """
    env = Env(objective='adp')
    env.reset()

    done = False
    steps = 0
    while not done and steps < max_steps:
        pos = env._acting_player_position
        obs = get_obs(env.infoset)

        if eps > 0 and np.random.random() < eps:
            idx = np.random.randint(len(obs['legal_actions']))
        else:
            d = landlord_dir if pos == 'landlord' else farmer_dir
            values = predict_onnx(obs, pos, d)
            idx = int(np.argmax(values))

        _, _, done, _ = env.step(obs['legal_actions'][idx])
        steps += 1

    return getattr(env, '_game_winner', 'farmer'), env._game_bomb_num


def arena(dir_x, dir_y, n_games, eps=0.0):
    """X vs Y 竞技场, 两种角色各一半.

    Returns dict:
      overall      X 的综合胜率 (0.5 = 势均力敌)
      as_landlord  X 当地主时的胜率
      as_farmer    X 当农民时的胜率
      n_*          各角色的局数
    """
    x_wins = 0
    ll_wins = ll_games = 0
    fa_wins = fa_games = 0

    for i in range(n_games):
        if i % 2 == 0:
            # X 当地主
            winner, _ = run_episode_mixed(dir_x, dir_y, eps)
            ll_games += 1
            if winner == 'landlord':
                x_wins += 1
                ll_wins += 1
        else:
            # X 当农民 (Y 当地主)
            winner, _ = run_episode_mixed(dir_y, dir_x, eps)
            fa_games += 1
            if winner == 'farmer':
                x_wins += 1
                fa_wins += 1

    return {
        'overall': x_wins / max(n_games, 1),
        'as_landlord': ll_wins / max(ll_games, 1),
        'as_farmer': fa_wins / max(fa_games, 1),
        'n_games': n_games,
        'n_as_landlord': ll_games,
        'n_as_farmer': fa_games,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--dir-x', type=str, required=True,
                        help='候选模型目录')
    parser.add_argument('-y', '--dir-y', type=str, default='models',
                        help='参考模型目录 (默认: models)')
    parser.add_argument('-n', '--num-games', type=int, default=1000)
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    dir_x = os.path.join(base, args.dir_x) if not os.path.isabs(args.dir_x) else args.dir_x
    dir_y = os.path.join(base, args.dir_y) if not os.path.isabs(args.dir_y) else args.dir_y

    # 确保读到的是最新导出的文件
    clear_onnx_cache(dir_x)
    clear_onnx_cache(dir_y)

    print(f'X (candidate): {dir_x}')
    print(f'Y (reference): {dir_y}')
    print(f'Games: {args.num_games}\n')

    res = arena(dir_x, dir_y, args.num_games)

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
