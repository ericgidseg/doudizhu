"""other_hand_cards bug 的配对 A/B 验证.

同一副牌, 地主分别用「修复前」和「修复后」的 ai_douzero 决策, 农民固定.
配对设计消除牌运方差.

用法: python3 bench_otherhand.py -n 200
"""
import os
import sys
import time
import argparse
import importlib.util
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(line_buffering=True)

# 必须在 import onnxruntime 之前设好, 否则 session 会吃满所有核心
os.environ.setdefault('ORT_NUM_THREADS', '2')

from env.env import Env, get_obs
from arena_vote import vote_values, make_dirs, clear_cache
from bench_mcts import build_room, decide_to_dz

BASE_SEED = 77700000


def load_variant(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.USE_MCTS = False      # 隔离变量: 只比较特征修复的影响
    return mod


def play_one(seed, ai, time_left, dirs):
    np.random.seed(seed)
    env = Env(objective='adp')
    env.reset()
    while not env._env.game_over:
        pos = env._acting_player_position
        obs = get_obs(env.infoset)
        if pos == 'landlord':
            action = decide_to_dz(ai.decide(build_room(env, time_left), 0), env)
        else:
            v = vote_values(obs, pos, dirs)
            action = obs['legal_actions'][int(np.argmax(v))]
        env.step(action)
    return 1 if env._game_winner == 'landlord' else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--num-games', type=int, default=200)
    p.add_argument('--time-left', type=float, default=25.0)
    args = p.parse_args()

    clear_cache()
    dirs = make_dirs()

    # 从当前 ai_douzero.py 现场生成「修复前」变体 (只改 other_hand_cards 一处),
    # 这样不用维护一份重复的源码副本.
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, 'ai_douzero.py')).read()
    fixed_pat = ('    all_played = [c for lst in played_by_pos.values() for c in lst]\n'
                 '    for c in my_dz + all_played:')
    buggy_pat = ('    all_played = played_cards_dz.get(seat, [])\n'
                 '    for c in my_dz + all_played:')
    assert fixed_pat in src, 'ai_douzero.py 里找不到已修复的 other_hand_cards 代码'
    bpath = os.path.join(here, '_ai_buggy_tmp.py')
    open(bpath, 'w').write(src.replace(fixed_pat, buggy_pat))

    buggy = load_variant(bpath, 'ai_buggy')
    fixed = load_variant(os.path.join(here, 'ai_douzero.py'), 'ai_fixed')

    wb = wf = 0
    diffs = []
    t0 = time.time()
    for i in range(args.num_games):
        seed = BASE_SEED + i
        rb = play_one(seed, buggy, args.time_left, dirs)
        rf = play_one(seed, fixed, args.time_left, dirs)
        wb += rb; wf += rf
        diffs.append(rf - rb)
        if (i + 1) % 50 == 0:
            print(f'    [{i+1}/{args.num_games}] 修复前={wb/(i+1):.3f} '
                  f'修复后={wf/(i+1):.3f} ({time.time()-t0:.0f}s)')

    diffs = np.array(diffs)
    n = args.num_games
    se = diffs.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    md = diffs.mean()

    print(f'\n=== other_hand_cards 修复效果 ({n} 局配对) ===')
    print(f'修复前 地主胜率: {wb/n:.4f}')
    print(f'修复后 地主胜率: {wf/n:.4f}')
    print(f'配对差值: {md:+.4f}  标准误 {se:.4f}')
    if se > 0:
        z = md / se
        print(f'z = {z:+.2f}  (|z|>1.96 为显著)')
        print('=> 修复显著提升' if z > 1.96 else
              ('=> 修复显著变差' if z < -1.96 else '=> 差异不显著'))
    same = int((diffs == 0).sum())
    print(f'两版结果相同的局: {same}/{n} ({same/n:.1%})')


if __name__ == '__main__':
    main()
