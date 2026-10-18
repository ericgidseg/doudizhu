"""ai_douzero 的回归测试: 牌型规则 + decide() 端到端跑完整局.

用法: python3 test_ai.py
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

RULES = [
    # (牌, 期望牌型)
    (['3', '4'], C.TYPE.INVALID),
    (['3', '4', '5'], C.TYPE.INVALID),
    (['3', '3', '4', '4'], C.TYPE.INVALID),
    (['3'], C.TYPE.SINGLE),
    (['3', '3'], C.TYPE.PAIR),
    (['SJ', 'BJ'], C.TYPE.ROCKET),
    (['3', '4', '5', '6', '7'], C.TYPE.CHAIN),
    (['10', 'J', 'Q', 'K', 'A'], C.TYPE.CHAIN),
    (['J', 'Q', 'K', 'A', '2'], C.TYPE.INVALID),          # 顺子不含 2
    (['3', '3', '4', '4', '5', '5'], C.TYPE.PAIR_CHAIN),
    (['3', '3', '3', '4', '4', '4'], C.TYPE.AIRPLANE),
    (['3', '3', '3', '4', '4', '4', '5', '6'], C.TYPE.AIRPLANE_1),
    (['3', '3', '3', '4', '4', '4', '5', '5', '6', '6'], C.TYPE.AIRPLANE_2),
    (['3', '3', '3', '3'], C.TYPE.BOMB),
    (['3', '3', '3', '4'], C.TYPE.TRIPLE_1),
    (['3', '3', '3', '4', '4'], C.TYPE.TRIPLE_2),
    (['3', '3', '3', '3', '4', '5'], C.TYPE.FOUR_2),
    (['3', '3', '3', '3', '4', '4', '5', '5'], C.TYPE.FOUR_2_2),
]


def test_rules():
    for ranks, want in RULES:
        hand = [{'id': i, 'rank': r, 'suit': ''} for i, r in enumerate(ranks)]
        got = C.analyze(hand)['type']
        assert got == want, f'{ranks}: got {C.TYPE_NAMES[got]}, want {C.TYPE_NAMES[want]}'
    print(f'  rules: PASS ({len(RULES)} cases)')


def test_decide_games():
    """三家都用 decide() 打完整局, 校验每一步出牌都合法."""
    for seed in range(3):
        random.seed(seed)
        room = S.new_room()
        for i in range(3):
            room['players'].append(S.new_player(None, i, f'P{i}', bot=True))
        S.start_game(room)
        for _ in range(300):
            seat = room['turn']
            act = AI.decide(room, seat)
            if room['phase'] == 'bidding':
                S.apply_bid(room, seat, act['bid'])
            elif act['type'] == 'play':
                ids = act['ids']
                cards_ = [c for c in room['players'][seat]['hand'] if c['id'] in set(ids)]
                ana = C.analyze(cards_)
                assert ana['type'] != C.TYPE.INVALID, f'seed{seed}: 非法出牌 {cards_}'
                if room['lastPlay'] and room['lastPlay']['seat'] != seat:
                    assert C.can_beat(room['lastPlay']['ana'], ana), \
                        f'seed{seed}: 压不过上家 {ana["typeName"]}'
                S.apply_play(room, seat, ids)
            else:
                assert room['lastPlay'] and room['lastPlay']['seat'] != seat, \
                    f'seed{seed}: 领出时不能 pass'
                S.apply_pass(room, seat)
            if room['phase'] == 'done':
                break
        assert room['phase'] == 'done', f'seed{seed}: 局没打完'
    print('  decide_games: PASS (3 局, 每步出牌均合法)')


if __name__ == '__main__':
    print('=== AI Tests ===')
    test_rules()
    test_decide_games()
    print('\nAll tests passed.')
