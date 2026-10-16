"""配对 A/B 测试 — 同一副牌下比较两种地主策略.

为什么需要配对:
  bench_mcts.py 每局重新洗牌, 牌运方差极大 (地主胜率 std ~0.04/100局),
  要检测 3% 的差异需要上千局.
  配对测试用同一个 seed 发同一副牌, 只换地主的策略, 方差大幅下降,
  同样局数能测出更小的差异.

用法:
  python3 bench_paired.py -n 200                    # MCTS 开 vs 关
  python3 bench_paired.py -n 200 --a vote --b decide
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(line_buffering=True)

# 必须在 import onnxruntime 之前设好, 否则 session 会吃满所有核心 (实测 ~980% CPU)
os.environ.setdefault('ORT_NUM_THREADS', '2')

from env.env import Env, get_obs
import ai_douzero as A
from arena_vote import vote_values, make_dirs, clear_cache
from bench_mcts import build_room, decide_to_dz

BASE_SEED = 20261016


def play_one(seed, ll_mode, mcts_on, old_sampler, time_left, dirs):
    """用 seed 发牌, 地主按 ll_mode 决策, 农民用 vote. 返回地主是否赢.

    old_sampler: None=不改; True/False=设 A.USE_OLD_SAMPLER.
    belief 采样器只在 MCTS 路径 (_mcts_pick) 里用到, 所以比采样器必须 mcts_on.
    """
    np.random.seed(seed)
    env = Env(objective='adp')
    env.reset()
    A.USE_MCTS = mcts_on
    if old_sampler is not None:
        A.USE_OLD_SAMPLER = old_sampler

    while not env._env.game_over:
        pos = env._acting_player_position
        obs = get_obs(env.infoset)
        if pos == 'landlord':
            if ll_mode == 'decide':
                action = decide_to_dz(A.decide(build_room(env, time_left), 0), env)
            else:
                v = vote_values(obs, pos, dirs)
                action = obs['legal_actions'][int(np.argmax(v))]
        else:
            v = vote_values(obs, pos, dirs)
            action = obs['legal_actions'][int(np.argmax(v))]
        env.step(action)

    return 1 if env._game_winner == 'landlord' else 0


def paired(a_cfg, b_cfg, n, time_left, dirs, verbose=True):
    """同一副牌跑 A/B 两种配置. 返回 (wr_a, wr_b, 配对差值统计)."""
    wins_a = wins_b = 0
    diffs = []
    t0 = time.time()
    for i in range(n):
        seed = BASE_SEED + i
        ra = play_one(seed, *a_cfg, time_left, dirs)
        rb = play_one(seed, *b_cfg, time_left, dirs)

        wins_a += ra
        wins_b += rb
        diffs.append(ra - rb)
        if verbose and (i + 1) % 25 == 0:
            print(f'    [{i+1}/{n}] A={wins_a/(i+1):.3f} B={wins_b/(i+1):.3f} '
                  f'({time.time()-t0:.0f}s)')

    diffs = np.array(diffs)          # diffs[i] = A 赢? - B 赢?  (A 相对 B 的优势)
    wr_a, wr_b = wins_a / n, wins_b / n
    mean_d = diffs.mean()
    # 配对差值的标准误: 只由「A赢B输」和「B赢A输」的不一致局决定, 比独立样本小得多
    se = diffs.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return wr_a, wr_b, mean_d, se, diffs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--num-games', type=int, default=200)
    p.add_argument('--a', default='decide', choices=['decide', 'vote'])
    p.add_argument('--b', default='decide', choices=['decide', 'vote'])
    p.add_argument('--a-mcts', type=int, default=0)
    p.add_argument('--b-mcts', type=int, default=1)
    p.add_argument('--a-old-sampler', type=int, default=None,
                   help='1=用旧 belief 采样器, 0=用 Gumbel, 不传=不改')
    p.add_argument('--b-old-sampler', type=int, default=None)
    p.add_argument('--time-left', type=float, default=25.0)
    args = p.parse_args()

    clear_cache()
    dirs = make_dirs()

    a_os = None if args.a_old_sampler is None else bool(args.a_old_sampler)
    b_os = None if args.b_old_sampler is None else bool(args.b_old_sampler)
    a_cfg = (args.a, bool(args.a_mcts), a_os)
    b_cfg = (args.b, bool(args.b_mcts), b_os)

    def _tag(mcts, os_):
        s = f'MCTS={"on" if mcts else "off"}'
        if os_ is not None:
            s += f', 采样器={"旧" if os_ else "Gumbel"}'
        return s

    print(f'A = {args.a} ({_tag(args.a_mcts, a_os)})')
    print(f'B = {args.b} ({_tag(args.b_mcts, b_os)})')
    print(f'同一副牌配对测试, {args.num_games} 局\n')

    wr_a, wr_b, md, se, diffs = paired(
        a_cfg, b_cfg, args.num_games, args.time_left, dirs)

    print(f'\n=== 结果 ({args.num_games} 局配对) ===')
    print(f'A 地主胜率: {wr_a:.4f}')
    print(f'B 地主胜率: {wr_b:.4f}')
    # md = A - B (见 paired() 里 diffs 的定义)
    print(f'配对差值 (A-B): {md:+.4f}  标准误 {se:.4f}')
    if se > 0:
        z = md / se
        print(f'z = {z:+.2f}  (>1.96 或 <-1.96 为显著)')
        if abs(z) < 1.96:
            print('=> 差异不显著 (噪声内)')
        elif z > 0:
            print('=> A 显著更好')
        else:
            print('=> B 显著更好')
    both = int(((diffs == 0).sum()))
    print(f'两种策略结果相同的局: {both}/{len(diffs)} '
          f'({both/len(diffs):.1%}) — 越高说明差异越集中在少数关键局')


if __name__ == '__main__':
    main()
