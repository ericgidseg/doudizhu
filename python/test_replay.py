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

from env.game import InfoSet
from env.move_generator import MovesGener
from env.move_detector import get_move_type
import env.move_selector as ms
import env.move_detector as md
from ai import decide

# DouZero 牌编码: 3-14, 17=2, 20=小王, 30=大王
R2N = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
       'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17, 'SJ': 20, 'BJ': 30}
N2R = {v: k for k, v in R2N.items()}


def hand2nums(hand):
    return sorted(R2N[c['rank']] for c in hand)


def build_legal(my_hand, last_move, pass_count, last_seat, seat):
    """构造合法动作集合 — 与 ai_douzero._build_legal_moves 同口径."""
    if not last_move or last_seat == seat or pass_count >= 2:
        return MovesGener(my_hand).gen_moves()
    mt = get_move_type(last_move)
    t, ln = mt['type'], mt.get('len', 1)
    mg = MovesGener(my_hand)
    gen_map = {
        md.TYPE_1_SINGLE: mg.gen_type_1_single,
        md.TYPE_2_PAIR: mg.gen_type_2_pair,
        md.TYPE_3_TRIPLE: mg.gen_type_3_triple,
        md.TYPE_6_3_1: mg.gen_type_6_3_1,
        md.TYPE_7_3_2: mg.gen_type_7_3_2,
        md.TYPE_8_SERIAL_SINGLE: lambda: mg.gen_type_8_serial_single(repeat_num=ln),
        md.TYPE_9_SERIAL_PAIR: lambda: mg.gen_type_9_serial_pair(repeat_num=ln),
        md.TYPE_10_SERIAL_TRIPLE: lambda: mg.gen_type_10_serial_triple(repeat_num=ln),
        md.TYPE_11_SERIAL_3_1: lambda: mg.gen_type_11_serial_3_1(repeat_num=ln),
        md.TYPE_12_SERIAL_3_2: lambda: mg.gen_type_12_serial_3_2(repeat_num=ln),
        md.TYPE_13_4_2: mg.gen_type_13_4_2,
        md.TYPE_14_4_22: mg.gen_type_14_4_22,
    }
    filt_map = {
        md.TYPE_1_SINGLE: ms.filter_type_1_single,
        md.TYPE_2_PAIR: ms.filter_type_2_pair,
        md.TYPE_3_TRIPLE: ms.filter_type_3_triple,
        md.TYPE_6_3_1: ms.filter_type_6_3_1,
        md.TYPE_7_3_2: ms.filter_type_7_3_2,
        md.TYPE_8_SERIAL_SINGLE: ms.filter_type_8_serial_single,
        md.TYPE_9_SERIAL_PAIR: ms.filter_type_9_serial_pair,
        md.TYPE_10_SERIAL_TRIPLE: ms.filter_type_10_serial_triple,
        md.TYPE_11_SERIAL_3_1: ms.filter_type_11_serial_3_1,
        md.TYPE_12_SERIAL_3_2: ms.filter_type_12_serial_3_2,
        md.TYPE_13_4_2: ms.filter_type_13_4_2,
        md.TYPE_14_4_22: ms.filter_type_14_4_22,
    }
    if t == md.TYPE_4_BOMB:
        legal = ms.filter_type_4_bomb(
            mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb(), last_move)
    elif t == md.TYPE_5_KING_BOMB:
        legal = []
    else:
        legal = filt_map[t](gen_map[t](), last_move)
        legal += mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
    legal += [[]]
    for m in legal:
        m.sort()
    return legal


def build_infoset(r, upto_step, seat):
    """重建 upto_step 之前的状态, 构造 seat 的 InfoSet."""
    played = []
    played_by = {0: [], 1: [], 2: []}
    action_seq = []
    last_move = []
    last_move_dict = {0: [], 1: [], 2: []}
    last_seat = -1
    bomb_num = 0
    pass_count = 0

    for i, a in enumerate(r['actions'][:upto_step]):
        if a['type'] == 'bid':
            continue
        if a['type'] == 'pass':
            action_seq.append([])
            last_move_dict[a['seat']] = []
            pass_count += 1
            continue
        cards = [R2N[c['rank']] for c in a['cards']]
        played.extend(cards)
        played_by[a['seat']].extend(cards)
        action_seq.append(cards)
        last_move = cards
        last_move_dict[a['seat']] = cards
        last_seat = a['seat']
        pass_count = 0
        if a['typeName'] in ('炸弹', '王炸'):
            bomb_num += 1

    my_hand_all = list(r['players'][seat]['hand'])
    if seat == r['landlordSeat']:
        my_hand_all += r['bottom']
    my_hand = [R2N[c['rank']] for c in my_hand_all]
    for c in played_by[seat]:
        my_hand.remove(c)

    # other_hand = 全副牌 - 我手 - 三家已出
    all_cards = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 4 + [17] * 4 + [20, 30]
    for c in my_hand + played:
        all_cards.remove(c)

    landlord_seat = r['landlordSeat']
    seat2pos = {landlord_seat: 'landlord',
                (landlord_seat + 1) % 3: 'landlord_down',
                (landlord_seat + 2) % 3: 'landlord_up'}
    pos = seat2pos[seat]

    num_left = {seat2pos[s]: len(r['players'][s]['hand']) - len(played_by[s]) +
                (3 if s == landlord_seat else 0)
                for s in range(3)}

    legal = build_legal(my_hand, last_move, pass_count, last_seat, seat)

    infoset = InfoSet(pos)
    infoset.player_hand_cards = sorted(my_hand)
    infoset.other_hand_cards = sorted(all_cards)
    infoset.legal_actions = legal
    infoset.last_move = last_move if pass_count < 2 else []
    infoset.last_move_dict = {seat2pos[s]: last_move_dict[s] for s in range(3)}
    infoset.num_cards_left_dict = num_left
    infoset.played_cards = {seat2pos[s]: played_by[s] for s in range(3)}
    infoset.card_play_action_seq = action_seq
    infoset.bomb_num = bomb_num
    infoset.three_landlord_cards = [R2N[c['rank']] for c in r['bottom']]
    return infoset, legal


def nums2cards(nums):
    return [N2R.get(n, str(n)) for n in nums]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    upto = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    seat = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    r = json.load(open(path))
    if upto < 0:
        upto = len(r['actions'])
    infoset, legal = build_infoset(r, upto, seat)
    print(f'seat{seat} ({infoset.player_position}) hand:',
          nums2cards(infoset.player_hand_cards))
    print('legal actions:', len(legal))
    print('DouZero decides...')
    act = decide(infoset)
    print('  ->', nums2cards(act) if act else 'pass')
