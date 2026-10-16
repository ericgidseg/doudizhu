"""决定性实验: Q值 / 启发式rollout / 神经网络rollout, 哪个更能预测动作的「真实价值」?

真实价值 = 用游戏实际最强策略 (vote 三模型集成) 打完整局的胜率.

方法: 取一个真实局面, 从 belief 采样 N 组对手手牌,
对每个候选动作 a:
  V_strong(a) = 用 vote 集成打完整局的地主胜率   <- 当作真值
  V_heur(a)   = 用启发式 rollout 打完整局的胜率
  V_nn(a)     = 用神经网络 rollout 打完整局的胜率
  Q(a)        = 网络直接输出的价值

然后看谁与 V_strong 的排序最一致.

用法: python3 bench_policy_value.py [--seed 0] [--step 6] [--samples 25]
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
from env.game import InfoSet
from bench_mcts import build_room
from arena_vote import vote_values, make_dirs, clear_cache

ORDER = ['landlord', 'landlord_down', 'landlord_up']
_dirs = None


def playout(hands, turn_pos, last_move, last_mover, seq, bombs,
            landlord_cards, played, policy, max_steps=100):
    """三家打完整局, 返回胜方位置. policy: 'heur' | 'nn' | 'vote'."""
    hands = {p: sorted(list(h)) for p, h in hands.items()}
    played = {p: list(v) for p, v in played.items()}
    seq = list(seq)
    cur_last = sorted(last_move)
    cur_last_pos = last_mover
    consecutive_pass = 0

    for _ in range(max_steps):
        hand = hands[turn_pos]
        if not hand:
            return turn_pos
        effective_last = [] if cur_last_pos == turn_pos else cur_last
        legal = A._build_legal_moves(hand, effective_last)
        if not legal:
            legal = [[]]

        if policy == 'heur':
            act = A._heuristic_pick(hand, legal, cur_last_pos, turn_pos, hands)
        else:
            other = []
            for p, h in hands.items():
                if p != turn_pos:
                    other += h
            iset = InfoSet(turn_pos)
            iset.player_hand_cards = sorted(hand)
            iset.other_hand_cards = sorted(other)
            iset.legal_actions = legal
            iset.last_move = list(effective_last)
            iset.last_move_dict = {p: (list(cur_last) if p == cur_last_pos else [])
                                   for p in ORDER}
            iset.num_cards_left_dict = {p: len(h) for p, h in hands.items()}
            iset.played_cards = played
            iset.card_play_action_seq = seq
            iset.bomb_num = bombs
            iset.three_landlord_cards = list(landlord_cards)
            try:
                obs = get_obs(iset)
                if policy == 'vote':
                    v = vote_values(obs, turn_pos, _dirs)
                else:
                    v = A._sess(turn_pos).run(
                        ['values'], {'z': obs['z_batch'],
                                     'x': obs['x_batch']})[0].flatten()
                act = list(legal[int(np.argmax(v))])
            except Exception:
                act = []

        if act:
            for c in act:
                if c in hands[turn_pos]:
                    hands[turn_pos].remove(c)
            seq.append(sorted(act))
            played.setdefault(turn_pos, []).extend(act)
            cur_last = sorted(act)
            cur_last_pos = turn_pos
            consecutive_pass = 0
            if (len(act) == 4 and len(set(act)) == 1) or sorted(act) == [20, 30]:
                bombs += 1
            if not hands[turn_pos]:
                return turn_pos
        else:
            seq.append([])
            consecutive_pass += 1
            if consecutive_pass >= 2:
                cur_last = []
                consecutive_pass = 0
                turn_pos = cur_last_pos
                continue

        turn_pos = ORDER[(ORDER.index(turn_pos) + 1) % 3]

    return cur_last_pos


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--step', type=int, default=6, help='先走几步找中局')
    p.add_argument('--samples', type=int, default=25)
    p.add_argument('--actions', type=int, default=4)
    args = p.parse_args()

    global _dirs
    clear_cache()
    _dirs = make_dirs()

    # ---- 找一个真实局面 ----
    np.random.seed(args.seed)
    env = Env(objective='adp')
    env.reset()
    for _ in range(args.step):
        if env._env.game_over:
            break
        pos = env._acting_player_position
        o = get_obs(env.infoset)
        v = vote_values(o, pos, _dirs)
        env.step(o['legal_actions'][int(np.argmax(v))])

    if env._env.game_over or env._acting_player_position != 'landlord':
        print(f'seed={args.seed} 第 {args.step} 步不是地主行动, 换 seed 试试')
        return

    iset = env.infoset
    room = build_room(env, 25.0)
    obs = get_obs(iset)
    base = vote_values(obs, 'landlord', _dirs)

    print(f'局面: seed={args.seed} 第 {args.step} 步, 地主 '
          f'{len(iset.player_hand_cards)} 张, 合法动作 {len(iset.legal_actions)} 个')
    print(f'农民剩牌: down={iset.num_cards_left_dict["landlord_down"]} '
          f'up={iset.num_cards_left_dict["landlord_up"]}')
    print()

    # ---- belief 采样对手手牌 ----
    played_by_pos = {p: list(iset.played_cards[p]) for p in ORDER}
    beliefs, pool, _ = A._build_belief(room, 0, {0: 'landlord', 1: 'landlord_down',
                                                 2: 'landlord_up'},
                                       played_by_pos, iset.player_hand_cards)
    opp = ['landlord_down', 'landlord_up']
    sizes = [iset.num_cards_left_dict[p] for p in opp]

    np.random.seed(args.seed + 999)
    samples = []
    for _ in range(args.samples):
        h = A._sample_hands(pool, beliefs, opp, sizes, 'landlord',
                            sorted(iset.player_hand_cards))
        samples.append(h)

    # ---- 对每个候选动作测三种价值 ----
    top = list(np.argsort(base)[-args.actions:][::-1])
    rows = []
    t0 = time.time()
    for rank, i in enumerate(top):
        act = sorted(iset.legal_actions[i])
        vals = {}
        for policy in ('vote', 'heur', 'nn'):
            w = []
            for h in samples:
                hands = {p: list(v) for p, v in h.items()}
                hands['landlord'] = sorted(c for c in iset.player_hand_cards
                                           if c not in act)
                if not hands['landlord']:
                    w.append(1)
                    continue
                winner = playout(hands, 'landlord_down', act, 'landlord',
                                 list(iset.card_play_action_seq) + [act],
                                 iset.bomb_num, iset.three_landlord_cards,
                                 {p: list(iset.played_cards[p]) for p in ORDER},
                                 policy)
                w.append(1 if winner == 'landlord' else 0)
            vals[policy] = float(np.mean(w))
        rows.append({'rank': rank + 1, 'act': act, 'Q': float(base[i]), **vals})
        print(f'  动作 {rank+1}/{len(top)} 测完 ({time.time()-t0:.0f}s)')

    print()
    print(f'{"Q排名":<6}{"动作":<16}{"Q值":>8}{"真实(vote)":>12}'
          f'{"启发式":>9}{"神经网络":>10}')
    for r in rows:
        print(f'{r["rank"]:<6}{str(r["act"]):<16}{r["Q"]:>8.3f}'
              f'{r["vote"]:>12.3f}{r["heur"]:>9.3f}{r["nn"]:>10.3f}')

    truth = [r['vote'] for r in rows]
    print()
    print('与「真实价值」的排序相关性 (Spearman, 越高越准):')
    for k in ('Q', 'heur', 'nn'):
        c = spearman([r[k] for r in rows], truth)
        print(f'  {k:>5}: {c:+.3f}')
    print()
    print(f'真实价值极差: {max(truth)-min(truth):.3f}  (动作间差距)')
    print(f'耗时 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
