"""用真实 replay 验证 DouZero ONNX 模型的决策是否合理.

用法:
    python3 test_replay.py <replay.json> [upto_step] [seat]

replay 文件是 server.py 保存的格式
(默认目录 ~/Library/Application Support/doudizhu/replays/).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cards as C
import ai_douzero as AI


def build_room(r, upto_step):
    """把 replay 的前 upto_step 步重放进一个 server 房间字典."""
    room = {'id': r.get('roomId', '0'), 'players': [], 'phase': 'playing',
            'landlordSeat': r['landlordSeat'], 'bottom': r['bottom'],
            'currentBid': 3, 'turn': -1, 'passCount': 0, 'lastPlay': None,
            'played': [], 'action_seq_dz': [],
            'played_by': {0: [], 1: [], 2: []},
            'last_move_dz': [], 'last_move_dict_dz': {0: [], 1: [], 2: []},
            'bomb_num': 0, 'replay': None}

    # 手牌 = 初始手牌 (+ 地主底牌) - 自己已出的牌
    hands = {}
    for p in r['players']:
        hand = list(p['hand'])
        if p['seat'] == r['landlordSeat']:
            hand += r['bottom']
        hands[p['seat']] = hand
    for p in r['players']:
        room['players'].append({'name': p['name'], 'ws': None, 'seat': p['seat'],
                                'hand': hands[p['seat']], 'score': 0,
                                'bot': p.get('bot', False), 'token': None})

    for a in r['actions'][:upto_step]:
        if a['type'] == 'bid':
            continue
        seat = a['seat']
        if a['type'] == 'pass':
            room['action_seq_dz'].append([])
            room['last_move_dict_dz'][seat] = []
            room['passCount'] += 1
            if room['passCount'] >= 2:
                room['lastPlay'] = None
                room['last_move_dz'] = []
                room['passCount'] = 0
            continue
        cards_ = [c for c in room['players'][seat]['hand']
                  if c['id'] in {x['id'] for x in a['cards']}]
        dz = C.hand_to_dz(cards_)
        room['players'][seat]['hand'] = [
            c for c in room['players'][seat]['hand'] if c not in cards_]
        room['played'].extend(cards_)
        room['played_by'][seat].extend(cards_)
        room['action_seq_dz'].append(dz)
        room['last_move_dz'] = dz
        room['last_move_dict_dz'][seat] = dz
        room['lastPlay'] = {'seat': seat, 'ana': C.analyze(cards_)}
        room['passCount'] = 0
        if a['typeName'] in ('炸弹', '王炸'):
            room['bomb_num'] += 1

    # 轮到谁: 按出牌顺序推 (地主先出), 数到 action 数
    room['turn'] = (r['landlordSeat'] + len(room['action_seq_dz'])) % 3
    return room


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    seat = int(sys.argv[3]) if len(sys.argv) > 3 else None

    with open(path) as f:
        r = json.load(f)
    upto = int(sys.argv[2]) if len(sys.argv) > 2 else len(r['actions'])
    room = build_room(r, upto)
    seat = room['turn'] if seat is None else seat
    room['turn'] = seat

    pos = {room['landlordSeat']: 'landlord',
           (room['landlordSeat'] + 1) % 3: 'landlord_down',
           (room['landlordSeat'] + 2) % 3: 'landlord_up'}[seat]
    print(f'seat{seat} ({pos}) hand:',
          ' '.join(c['rank'] for c in C.sort_cards(room['players'][seat]['hand'])))
    print('DouZero decides...')
    act = AI.decide(room, seat)
    if act['type'] == 'play':
        played = [c for c in room['players'][seat]['hand'] if c['id'] in set(act['ids'])]
        print('  ->', C.analyze(played)['typeName'],
              ' '.join(c['rank'] for c in C.sort_cards(played)))
    else:
        print('  -> pass')


if __name__ == '__main__':
    main()
