"""配对 A/B: 农民用 ai_douzero.decide() vs 农民用 vote 集成, 谁赢?

修复 encoding.py 后, decide() 当农民时第一次拿到正确特征.
地主固定 vote 集成 (env.env.get_obs + 三模型投票), 两种农民策略对比.

用法: python3 bench_farmer_decide.py -n 400
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault('ORT_NUM_THREADS', '2')

from env.env import Env, get_obs as env_get_obs
import ai_douzero as A
from arena_vote import vote_values, make_dirs, clear_cache
from bench_mcts import build_room, decide_to_dz, POS2SEAT

BASE_SEED = 81800001


def play_one(seed, farmer_mode, dirs):
    """地主 vote 固定, 农民 farmer_mode ∈ {'decide','vote'}. 返回地主是否赢."""
    np.random.seed(seed)
    env = Env(objective='adp')
    env.reset()
    A.USE_MCTS = False
    while not env._env.game_over:
        iset = env.infoset
        pos = iset.player_position
        if pos == 'landlord':
            obs = env_get_obs(iset)
            v = vote_values(obs, pos, dirs)
            action = iset.legal_actions[int(np.argmax(v))]
        elif farmer_mode == 'decide':
            room = build_room(env, 25.0)
            seat = POS2SEAT[pos]
            action = decide_to_dz(A.decide(room, seat), env)
        else:
            obs = env_get_obs(iset)
            v = vote_values(obs, pos, dirs)
            action = iset.legal_actions[int(np.argmax(v))]
        env.step(action)
    return 1 if env._game_winner == 'landlord' else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--num-games', dest='n', type=int, default=400)
    args = p.parse_args()

    clear_cache()
    dirs = make_dirs()

    wins_decide = wins_vote = 0
    diffs = []
    t0 = time.time()
    for i in range(args.n):
        seed = BASE_SEED + i
        # r = 地主赢? 农民弱 -> 地主赢
        r_d = play_one(seed, 'decide', dirs)
        r_v = play_one(seed, 'vote', dirs)
        wins_decide += r_d
        wins_vote += r_v
        diffs.append(r_d - r_v)
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{args.n}] LL_WR(农民decide)={wins_decide/(i+1):.3f} '
                  f'LL_WR(农民vote)={wins_vote/(i+1):.3f} ({time.time()-t0:.0f}s)')

    diffs = np.array(diffs)
    md = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(args.n)

    print(f'\n=== 结果 ({args.n} 局配对) ===')
    print(f'地主胜率 (农民 decide): {wins_decide/args.n:.4f}')
    print(f'地主胜率 (农民 vote)  : {wins_vote/args.n:.4f}')
    print(f'配对差值 (decide-vote): {md:+.4f}  标准误 {se:.4f}')
    if se > 0:
        z = md / se
        print(f'z = {z:+.2f}')
        if z > 1.96:
            print('=> decide 农民显著弱于 vote 农民 (地主更易赢)')
        elif z < -1.96:
            print('=> decide 农民显著强于 vote 农民')
        else:
            print('=> 噪声内')
    both = int((diffs == 0).sum())
    print(f'结果相同的局: {both}/{args.n} ({both/args.n:.1%})')


if __name__ == '__main__':
    main()
