"""决定性实验: 把 Q值 / rollout / 混合 分别当策略打完整局, 看谁真赢.

之前两次测量结论矛盾:
  - 「rollout 与 Q 值相关性 +0.04」-> 推断 rollout 更差
  - 「rollout 比 Q 值更接近 vote 真值」-> 推断 rollout 更好
两个都是间接推断. 这个实验直接比: 谁打赢谁.

用法: python3 bench_policy_duel.py -n 200
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('ORT_NUM_THREADS', '2')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import ai_douzero as A
from env.env import Env, get_obs
from arena_vote import vote_values, make_dirs, clear_cache

BASE_SEED = 42424242


def play(seed, ll_policy, dirs, mcts_budget=2.5):
    """地主用 ll_policy, 农民固定用 vote. 返回地主是否赢.

    ll_policy:
      'q'      - 纯 Q 值 argmax (MCTS 关)
      'mcts'   - Q 值 + MCTS rollout 混合 (MCTS 开)
      'vote'   - 纯 vote 集成 (基准)
    """
    np.random.seed(seed)
    env = Env(objective='adp')
    env.reset()

    if ll_policy == 'q':
        A.USE_MCTS = False
    elif ll_policy == 'mcts':
        A.USE_MCTS = True
        A.MCTS_BUDGET = mcts_budget

    from bench_mcts import build_room, decide_to_dz

    while not env._env.game_over:
        pos = env._acting_player_position
        obs = get_obs(env.infoset)
        if pos == 'landlord':
            if ll_policy == 'vote':
                v = vote_values(obs, pos, dirs)
                action = obs['legal_actions'][int(np.argmax(v))]
            else:
                action = decide_to_dz(A.decide(build_room(env, 25.0), 0), env)
        else:
            v = vote_values(obs, pos, dirs)
            action = obs['legal_actions'][int(np.argmax(v))]
        env.step(action)

    return 1 if env._game_winner == 'landlord' else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--num-games', type=int, default=200)
    p.add_argument('--budget', type=float, default=2.5)
    args = p.parse_args()

    clear_cache()
    dirs = make_dirs()

    print(f'地主三种策略对决, 农民固定 vote 集成, {args.num_games} 局')
    print(f'MCTS 预算 {args.budget}s\n')

    res = {}
    for pol in ('q', 'mcts', 'vote'):
        wins = 0
        t0 = time.time()
        for i in range(args.num_games):
            wins += play(BASE_SEED + i, pol, dirs, args.budget)
            if (i + 1) % 25 == 0:
                print(f'  [{pol:>4}] {i+1}/{args.num_games} '
                      f'LL_WR={wins/(i+1):.3f} ({time.time()-t0:.0f}s)')
        res[pol] = wins / args.num_games
        print(f'  [{pol:>4}] 最终 LL_WR = {res[pol]:.4f}\n')

    print('=== 汇总 ===')
    base = res['vote']
    for pol in ('vote', 'q', 'mcts'):
        n = args.num_games
        se = np.sqrt(res[pol] * (1 - res[pol]) / n)
        print(f'  {pol:>5}: {res[pol]:.4f} ± {se:.4f}')
    print()
    print(f'q    - vote = {res["q"]-base:+.4f}')
    print(f'mcts - q    = {res["mcts"]-res["q"]:+.4f}   <- MCTS 的净效果')
    print(f'mcts - vote = {res["mcts"]-base:+.4f}')


if __name__ == '__main__':
    main()
