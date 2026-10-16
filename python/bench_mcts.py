"""对比 MCTS 开/关对棋力的影响.

地主用 ai_douzero.decide() (可开关 MCTS), 农民用固定 vote 集成.
统计地主胜率 —— 地主胜率越高说明 MCTS 越有用.

用法:
  python3 bench_mcts.py -n 200            # 跑 200 局
  python3 bench_mcts.py -n 200 --both     # 开关各跑一遍对比
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# 重定向到文件时也能实时看到进度
sys.stdout.reconfigure(line_buffering=True)

from env.env import Env, get_obs
from cards import hand_to_dz
import ai_douzero as A
from arena_vote import vote_values, make_dirs, clear_cache

RANK = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
        11: 'J', 12: 'Q', 13: 'K', 14: 'A', 17: '2', 20: 'SJ', 30: 'BJ'}
POS2SEAT = {'landlord': 0, 'landlord_down': 1, 'landlord_up': 2}


def build_room(env, time_left=25.0):
    """把 Env 的完美信息局面转成 ai_douzero.decide 需要的 room 结构."""
    iset = env.infoset
    players = [{'hand': []} for _ in range(3)]
    for p, s in POS2SEAT.items():
        cards = sorted(env._env.info_sets[p].player_hand_cards)
        # id 用「排序后手牌的下标」, decide_to_dz 按同样规则反查
        players[s]['hand'] = [
            {'id': i, 'rank': RANK[c], 'suit': ''} for i, c in enumerate(cards)
        ]

    # 已出牌: 必须传真实数据, 否则 AI 看到的是「谁都没出过牌」的假局面
    # (decide 用它算 other_hand_cards / 记牌器, 直接影响 MCTS 的 belief 采样)
    # 同样是牌对象列表 (decide 内部走 hand_to_dz)
    played_by = {POS2SEAT[p]: [{'rank': RANK[c], 'suit': ''} for c in v]
                 for p, v in env._env.played_cards.items()}

    # passCount: 从出牌序列尾部数连续的空动作 (=pass)
    seq = list(env._env.card_play_action_seq)
    pass_count = 0
    for a in reversed(seq):
        if len(a) == 0:
            pass_count += 1
        else:
            break
    # 一手牌出完后要清零: 只有「上一手非空动作之后」的 pass 才算
    pass_count = min(pass_count, 2)

    return {
        'players': players,
        'landlordSeat': 0,
        'phase': 'playing',
        'action_seq_dz': seq,
        'played_by': played_by,
        'last_move_dz': list(iset.last_move),
        'last_move_dict_dz': {POS2SEAT[p]: list(v)
                              for p, v in iset.last_move_dict.items()},
        'passCount': pass_count,
        'bomb_num': iset.bomb_num,
        # bottom 要的是牌对象 (hand_to_dz 会取 ['rank']), 不是点数
        'bottom': [{'rank': RANK[c], 'suit': ''}
                   for c in env._env.three_landlord_cards],
        'turnDeadline': time.time() + time_left,
    }


def decide_to_dz(decision, env):
    """把 decide() 返回的 {'type','ids'} 转成 DouZero 点数列表.

    decide() 里的 id 是 build_room 里我们编的「手牌下标」(见 build_room),
    所以这里按同样的顺序映射回点数.
    """
    if decision['type'] == 'pass':
        return []
    hand = sorted(env.infoset.player_hand_cards)
    out = [hand[i] for i in decision['ids'] if 0 <= i < len(hand)]
    return sorted(out)


def play_games(mcts_on, n_games, time_left=25.0, verbose=False):
    """地主用 decide(), 农民用 vote 集成. 返回地主胜率."""
    A.USE_MCTS = mcts_on
    ll_wins = 0
    t0 = time.time()

    for g in range(n_games):
        env = Env(objective='adp')
        env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            pos = env._acting_player_position
            if pos == 'landlord':
                d = A.decide(build_room(env, time_left), 0)
                action = decide_to_dz(d, env)
            else:
                obs = get_obs(env.infoset)
                v = vote_values(obs, pos, make_dirs())
                action = obs['legal_actions'][int(np.argmax(v))]

            _, _, done, _ = env.step(action)
            steps += 1

        if env._game_winner == 'landlord':
            ll_wins += 1

        if verbose and (g + 1) % 20 == 0:
            print(f'    [{g+1}/{n_games}] LL_WR={ll_wins/(g+1):.3f} '
                  f'({time.time()-t0:.0f}s)')

    return ll_wins / n_games


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--num-games', type=int, default=200)
    p.add_argument('--both', action='store_true',
                   help='开关各跑一遍 (对比)')
    p.add_argument('--time-left', type=float, default=25.0,
                   help='模拟每步剩余时间 (默认 25s, 服务器上限 30s)')
    args = p.parse_args()

    clear_cache()

    if args.both:
        print(f'=== MCTS 关 ({args.num_games} 局) ===')
        wr_off = play_games(False, args.num_games, args.time_left, verbose=True)
        print(f'  LL_WR = {wr_off:.3f}\n')

        print(f'=== MCTS 开 ({args.num_games} 局) ===')
        wr_on = play_games(True, args.num_games, args.time_left, verbose=True)
        print(f'  LL_WR = {wr_on:.3f}\n')

        print('=== 结论 ===')
        diff = wr_on - wr_off
        print(f'MCTS 关: {wr_off:.3f}')
        print(f'MCTS 开: {wr_on:.3f}')
        print(f'差值: {diff:+.3f}')
        if abs(diff) < 0.03:
            print('=> 差异在噪声内, MCTS 无显著效果')
        elif diff > 0:
            print('=> MCTS 有帮助')
        else:
            print('=> MCTS 反而有害')
    else:
        A.USE_MCTS = True
        wr = play_games(True, args.num_games, args.time_left, verbose=True)
        print(f'LL_WR = {wr:.3f}')


if __name__ == '__main__':
    main()
