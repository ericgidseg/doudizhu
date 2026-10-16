"""测 top-1 选对率: 排序相关性高, 不代表决策选得对.

之前测出 rollout 排序相关性 +0.56 > Q值 +0.30, 但 MCTS 实测更差.
可能的解释: 决策只关心 top-1 是否选对, 而排序相关性和 top-1 准确率
可以差很远 —— 尤其是 Q 值把最好的动作排在前列时, rollout 一扰动
就可能把它换掉.

用法: python3 bench_top1.py --cases 12 --samples 80
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('ORT_NUM_THREADS', '2')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import ai_douzero as A
from env.env import Env, get_obs
from bench_mcts import build_room
from arena_vote import vote_values, make_dirs, clear_cache
from bench_policy_value import playout, ORDER

_dirs = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=int, default=12)
    p.add_argument('--samples', type=int, default=80)
    p.add_argument('--actions', type=int, default=4)
    args = p.parse_args()

    global _dirs
    clear_cache()
    _dirs = make_dirs()

    hit_q = hit_mcts = 0
    hit_heur = hit_nn = 0
    total = 0

    for seed in range(40):
        if total >= args.cases:
            break
        np.random.seed(seed)
        env = Env(objective='adp')
        env.reset()
        # 走到中局
        for _ in range(12):
            if env._env.game_over:
                break
            pos = env._acting_player_position
            o = get_obs(env.infoset)
            v = vote_values(o, pos, _dirs)
            env.step(o['legal_actions'][int(np.argmax(v))])
        if env._env.game_over or env._acting_player_position != 'landlord':
            continue
        iset = env.infoset
        if len(iset.legal_actions) < 4:
            continue

        room = build_room(env, 25.0)
        obs = get_obs(iset)
        base = vote_values(obs, 'landlord', _dirs)

        # 用「真实对手手牌 + 随机置换」构造采样.
        # 不用 _build_belief: 它的先验太均匀, 采出来的对手手牌偏弱,
        # 导致地主随便怎么打都赢 (真值全是 1.00), 区分不出动作好坏.
        # 用真实手牌 + 同点数内的随机置换, 既保证对手强度真实, 又有采样多样性.
        opp = ['landlord_down', 'landlord_up']
        true_hands = {p: list(env._env.info_sets[p].player_hand_cards)
                      for p in opp}
        np.random.seed(seed + 999)
        samples = []
        for _ in range(args.samples):
            h = {}
            for p in opp:
                h[p] = list(true_hands[p])
                # 同点数牌随机打乱 (4 张 5 谁拿哪张不影响, 但换个位置)
                np.random.shuffle(h[p])
            samples.append(h)

        top = list(np.argsort(base)[-args.actions:][::-1])
        truth, heur_v, nn_v = [], [], []
        for i in top:
            act = sorted(iset.legal_actions[i])
            vals = {'vote': [], 'heur': [], 'nn': []}
            for h in samples:
                for pol in vals:
                    hands = {p: list(v) for p, v in h.items()}
                    hands['landlord'] = sorted(
                        c for c in iset.player_hand_cards if c not in act)
                    if not hands['landlord']:
                        vals[pol].append(1)
                        continue
                    w = playout(hands, 'landlord_down', act, 'landlord',
                                list(iset.card_play_action_seq) + [act],
                                iset.bomb_num, iset.three_landlord_cards,
                                {p: list(iset.played_cards[p]) for p in ORDER},
                                pol)
                    vals[pol].append(1 if w == 'landlord' else 0)
            truth.append(np.mean(vals['vote']))
            heur_v.append(np.mean(vals['heur']))
            nn_v.append(np.mean(vals['nn']))

        # 真实最优动作 (按 vote 打出来的胜率)
        if max(truth) - min(truth) < 0.15:
            continue        # 差距太小, 不构成有效判别
        total += 1
        best = int(np.argmax(truth))
        hit_q += int(np.argmax([base[i] for i in top]) == best)
        hit_heur += int(np.argmax(heur_v) == best)
        hit_nn += int(np.argmax(nn_v) == best)
        print(f'  case {total}: 极差={max(truth)-min(truth):.3f}  '
              f'真值最优=第{best+1}名  '
              f'Q选第{int(np.argmax([base[i] for i in top]))+1} '
              f'heur选第{int(np.argmax(heur_v))+1} '
              f'nn选第{int(np.argmax(nn_v))+1}')

    print()
    print(f'=== top-1 选对率 ({total} 个有效局面) ===')
    for name, h in (('Q值', hit_q), ('启发式rollout', hit_heur),
                    ('神经网络rollout', hit_nn)):
        print(f'  {name:>16}: {h}/{total} = {h/max(total,1):.0%}')


if __name__ == '__main__':
    main()
