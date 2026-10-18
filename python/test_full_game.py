"""模拟完整一局: DouZero 三家互打, 验证管线端到端 (复用 server 的房间流程).

用法: python3 test_full_game.py
"""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cards as C
import ai_douzero as AI
import server as S

# 别把测试对局写进真实回放目录
S.REPLAY_DIR = tempfile.mkdtemp(prefix='ddz_test_replays_')

N2R = C.DZ_TO_RANK


def fmt(nums):
    return ' '.join(N2R.get(n, str(n)) for n in sorted(nums))


def play_game(verbose=True):
    room = S.new_room()
    for i in range(3):
        room['players'].append(S.new_player(None, i, f'P{i}', bot=True))
    S.start_game(room)

    step = 0
    while room['phase'] != 'done' and step < 300:
        step += 1
        seat = room['turn']
        act = AI.decide(room, seat)
        pos = {room['landlordSeat']: 'landlord',
               (room['landlordSeat'] + 1) % 3: 'landlord_down',
               (room['landlordSeat'] + 2) % 3: 'landlord_up'}[seat] \
            if room['phase'] == 'playing' else '?'
        if room['phase'] == 'bidding':
            if verbose:
                print(f'{step:2}. seat{seat} 叫分 {act["bid"]}')
            S.apply_bid(room, seat, act['bid'])
        elif act['type'] == 'play':
            cards_ = [c for c in room['players'][seat]['hand'] if c['id'] in set(act['ids'])]
            if verbose:
                print(f'{step:2}. seat{seat}({pos}) {C.analyze(cards_)["typeName"]}: '
                      f'{fmt(C.hand_to_dz(cards_))} (剩{len(room["players"][seat]["hand"]) - len(cards_)})')
            S.apply_play(room, seat, act['ids'])
        else:
            if verbose:
                print(f'{step:2}. seat{seat}({pos}) pass')
            S.apply_pass(room, seat)

    if room['phase'] != 'done':
        return None
    # 谁先出完谁赢
    for i, p in enumerate(room['players']):
        if not p['hand']:
            return 'landlord' if i == room['landlordSeat'] else 'farmer'
    return None


if __name__ == '__main__':
    random.seed(42)
    print('=== DouZero 三家自打 ===')
    print('胜方:', play_game())
