"""
DouZero 状态编码 — 完全对齐 env.py 的 _get_obs_* 实现.
牌编码: 点数编号 3-14(3-K,A), 17=2, 20=小王, 30=大王.
每个牌集合编成 54 维向量: 4×13 矩阵按列展开 + 2 joker 位.
"""
import numpy as np
from collections import Counter
from itertools import combinations

# ---- 常量(对齐 DouZero) ----
Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}
NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

MIN_SINGLE_CARDS = 5
MIN_PAIRS = 3
MIN_TRIPLES = 2


def cards2array(list_cards):
    """牌集合 -> 54 维向量 (4×13 矩阵按列 + 2 joker)."""
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)
    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))


def _process_action_seq(sequence, length=15):
    """保留最近 15 步动作, 不足则前面补空."""
    sequence = sequence[-length:]
    if len(sequence) < length:
        empty = [[] for _ in range(length - len(sequence))]
        empty.extend(sequence)
        sequence = empty
    return sequence


def _action_seq_list2array(action_seq_list):
    """15 步动作 -> 5×162 矩阵 (每 3 步一行)."""
    arr = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        arr[row, :] = cards2array(list_cards)
    return arr.reshape(5, 162)


def _get_one_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1
    return one_hot


def _get_one_hot_bomb(bomb_num):
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot


class InfoSet:
    """完美信息集——服务器知道所有牌, 编码时再裁剪成不完美视角."""
    def __init__(self, player_position, player_hand_cards,
                 other_hand_cards, legal_actions, last_move,
                 last_move_dict, num_cards_left_dict, played_cards,
                 card_play_action_seq, bomb_num, three_landlord_cards=None):
        self.player_position = player_position  # 'landlord'|'landlord_up'|'landlord_down'
        self.player_hand_cards = player_hand_cards
        self.other_hand_cards = other_hand_cards
        self.legal_actions = legal_actions
        self.last_move = last_move
        self.last_move_dict = last_move_dict
        self.num_cards_left_dict = num_cards_left_dict
        self.played_cards = played_cards
        self.card_play_action_seq = card_play_action_seq
        self.bomb_num = bomb_num
        self.three_landlord_cards = three_landlord_cards or []


def get_obs(infoset):
    """返回 {x_batch, z_batch, legal_actions} 供 ONNX 模型推理."""
    if infoset.player_position == 'landlord':
        return _obs_landlord(infoset)
    elif infoset.player_position == 'landlord_up':
        return _obs_landlord_up(infoset)
    else:
        return _obs_landlord_down(infoset)


def _obs_landlord(infoset):
    n = len(infoset.legal_actions)
    rep = lambda a: np.repeat(a[np.newaxis, :], n, axis=0)

    my_hand = rep(cards2array(infoset.player_hand_cards))
    other = rep(cards2array(infoset.other_hand_cards))
    last = rep(cards2array(infoset.last_move))

    my_action = np.zeros(my_hand.shape)
    for j, act in enumerate(infoset.legal_actions):
        my_action[j, :] = cards2array(act)

    lu_num = rep(_get_one_hot_array(infoset.num_cards_left_dict['landlord_up'], 17))
    ld_num = rep(_get_one_hot_array(infoset.num_cards_left_dict['landlord_down'], 17))
    lu_played = rep(cards2array(infoset.played_cards['landlord_up']))
    ld_played = rep(cards2array(infoset.played_cards['landlord_down']))
    bomb = rep(_get_one_hot_bomb(infoset.bomb_num))

    x_batch = np.hstack((my_hand, other, last, lu_played, ld_played,
                         lu_num, ld_num, bomb, my_action))
    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :], n, axis=0)

    return {'position': 'landlord',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions}


def _obs_landlord_up(infoset):
    n = len(infoset.legal_actions)
    rep = lambda a: np.repeat(a[np.newaxis, :], n, axis=0)

    my_hand = rep(cards2array(infoset.player_hand_cards))
    other = rep(cards2array(infoset.other_hand_cards))
    last = rep(cards2array(infoset.last_move))

    my_action = np.zeros(my_hand.shape)
    for j, act in enumerate(infoset.legal_actions):
        my_action[j, :] = cards2array(act)

    last_lord = rep(cards2array(infoset.last_move_dict['landlord']))
    lord_num = rep(_get_one_hot_array(infoset.num_cards_left_dict['landlord'], 20))
    lord_played = rep(cards2array(infoset.played_cards['landlord']))

    last_mate = rep(cards2array(infoset.last_move_dict['landlord_down']))
    mate_num = rep(_get_one_hot_array(infoset.num_cards_left_dict['landlord_down'], 17))
    mate_played = rep(cards2array(infoset.played_cards['landlord_down']))

    bomb = rep(_get_one_hot_bomb(infoset.bomb_num))

    x_batch = np.hstack((my_hand, other, last, last_lord,
                         lord_played, lord_num, mate_played,
                         last_mate, mate_num, bomb, my_action))
    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :], n, axis=0)

    return {'position': 'landlord_up',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions}


def _obs_landlord_down(infoset):
    n = len(infoset.legal_actions)
    rep = lambda a: np.repeat(a[np.newaxis, :], n, axis=0)

    my_hand = rep(cards2array(infoset.player_hand_cards))
    other = rep(cards2array(infoset.other_hand_cards))
    last = rep(cards2array(infoset.last_move))

    my_action = np.zeros(my_hand.shape)
    for j, act in enumerate(infoset.legal_actions):
        my_action[j, :] = cards2array(act)

    last_lord = rep(cards2array(infoset.last_move_dict['landlord']))
    lord_num = rep(_get_one_hot_array(infoset.num_cards_left_dict['landlord'], 20))
    lord_played = rep(cards2array(infoset.played_cards['landlord']))

    last_mate = rep(cards2array(infoset.last_move_dict['landlord_up']))
    mate_num = rep(_get_one_hot_array(infoset.num_cards_left_dict['landlord_up'], 17))
    mate_played = rep(cards2array(infoset.played_cards['landlord_up']))

    bomb = rep(_get_one_hot_bomb(infoset.bomb_num))

    x_batch = np.hstack((my_hand, other, last, last_lord,
                         lord_played, lord_num, mate_played,
                         last_mate, mate_num, bomb, my_action))
    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :], n, axis=0)

    return {'position': 'landlord_down',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions}


# ---- DouZero 合法动作生成 (move_generator.py 等价物) ----

def _gen_serial(cards, min_serial, repeat=1, repeat_num=0):
    """生成连牌 (顺子/连对/飞机主体)."""
    if repeat_num < min_serial:
        repeat_num = 0
    single = sorted(set(cards))
    seqs = []
    moves = []
    start = i = 0
    longest = 1
    while i < len(single):
        if i + 1 < len(single) and single[i + 1] - single[i] == 1:
            longest += 1
            i += 1
        else:
            seqs.append((start, longest))
            i += 1
            start = i
            longest = 1
    for s, l in seqs:
        if l < min_serial:
            continue
        lst = single[s: s + l]
        if repeat_num == 0:
            for steps in range(min_serial, l + 1):
                for idx in range(0, l - steps + 1):
                    moves.append(sorted(lst[idx: idx + steps] * repeat))
        else:
            for idx in range(0, l - repeat_num + 1):
                moves.append(sorted(lst[idx: idx + repeat_num] * repeat))
    return moves


def gen_legal_moves(cards_list):
    """生成所有合法出牌组合 (含 pass)."""
    cards_dict = Counter(cards_list)
    singles = [[i] for i in cards_dict]
    pairs = [[k, k] for k, v in cards_dict.items() if v >= 2]
    triples = [[k, k, k] for k, v in cards_dict.items() if v >= 3]
    bombs = [[k] * 4 for k, v in cards_dict.items() if v == 4]
    rocket = [[20, 30]] if 20 in cards_dict and 30 in cards_dict else []

    moves = singles + pairs + triples + bombs + rocket

    # 三带一 / 三带二
    for t in singles:
        for tr in triples:
            if t[0] != tr[0]:
                moves.append(tr + t)
    for p in pairs:
        for tr in triples:
            if p[0] != tr[0]:
                moves.append(tr + p)

    # 顺子 / 连对 / 飞机
    moves += _gen_serial(cards_list, MIN_SINGLE_CARDS, repeat=1)
    pair_vals = [k for k, v in cards_dict.items() if v >= 2]
    moves += _gen_serial(pair_vals, MIN_PAIRS, repeat=2)
    triple_vals = [k for k, v in cards_dict.items() if v >= 3]
    moves += _gen_serial(triple_vals, MIN_TRIPLES, repeat=3)

    # 飞机带翅
    serial_3 = _gen_serial(triple_vals, MIN_TRIPLES, repeat=3)
    for s3 in serial_3:
        s3_set = set(s3)
        rest = [c for c in cards_list if c not in s3_set]
        for combo in combinations(rest, len(s3_set)):
            moves.append(s3 + list(combo))
        pair_rest = [k for k, v in cards_dict.items() if v >= 2 and k not in s3_set]
        for combo in combinations(pair_rest, len(s3_set)):
            moves.append(s3 + [c for c in combo for _ in range(2)])

    # 四带二
    for fc, v in cards_dict.items():
        if v == 4:
            rest = [c for c in cards_list if c != fc]
            for combo in combinations(rest, 2):
                moves.append([fc] * 4 + list(combo))
            pair_rest = [k for k, vv in cards_dict.items() if k != fc and vv >= 2]
            for combo in combinations(pair_rest, 2):
                moves.append([fc] * 4 + [c for c in combo for _ in range(2)])

    return moves


def gen_beat_moves(cards_list, last_move_type, last_move_rank, last_move_len):
    """生成能压过 last_move 的组合."""
    moves = gen_legal_moves(cards_list)
    beats = []
    for m in moves:
        t, r, l = analyze(m)
        if t is None:
            continue
        if t == 'bomb' or t == 'rocket':
            beats.append(m)  # 炸/王炸压一切非炸
        elif t == last_move_type and l == last_move_len and r > last_move_rank:
            beats.append(m)
    beats.append([])  # pass
    return beats


# ---- 牌型分析 ----
RANK_ORDER = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7, 11:8, 12:9, 13:10, 14:11, 17:12, 20:13, 30:14}

def analyze(cards):
    """返回 (type, rank, len) 或 (None,None,None)."""
    n = len(cards)
    cnt = Counter(cards)
    vals = sorted(cnt)
    if n == 0:
        return ('pass', 0, 0)
    if n == 1:
        return ('single', RANK_ORDER[cards[0]], 1)
    if n == 2:
        if set(cards) == {20, 30}:
            return ('rocket', 99, 2)
        if len(cnt) == 1:
            return ('pair', RANK_ORDER[cards[0]], 2)
        return (None, None, None)
    if n == 3 and len(cnt) == 1:
        return ('triple', RANK_ORDER[cards[0]], 3)
    if n == 4:
        if len(cnt) == 1:
            return ('bomb', RANK_ORDER[cards[0]], 4)
        if len(cnt) == 2:
            # 三带一
            for v, c in cnt.items():
                if c == 3:
                    return ('tri1', RANK_ORDER[v], 4)
                if c == 1:
                    return ('tri1', RANK_ORDER[vals[0] if cnt[vals[0]]==3 else vals[1]], 4)
        return (None, None, None)
    # 顺子
    if n >= 5 and len(cnt) == n:
        if all(vals[i+1] - vals[i] == 1 for i in range(len(vals)-1)) and max(vals) < 17:
            return ('chain', RANK_ORDER[max(vals)], n)
    # 连对
    if n >= 6 and n % 2 == 0 and all(v == 2 for v in cnt.values()):
        if all(vals[i+1] - vals[i] == 1 for i in range(len(vals)-1)) and max(vals) < 17:
            return ('pairchain', RANK_ORDER[max(vals)], n)
    # 三带二
    if n == 5 and len(cnt) == 2:
        for v, c in cnt.items():
            if c == 3:
                return ('tri2', RANK_ORDER[v], 5)
    # 飞机
    if n >= 6:
        triple_vals = sorted([v for v, c in cnt.items() if c == 3])
        if len(triple_vals) >= 2:
            for i in range(len(triple_vals)):
                run = [triple_vals[i]]
                for j in range(i+1, len(triple_vals)):
                    if triple_vals[j] - run[-1] == 1:
                        run.append(triple_vals[j])
                    else:
                        break
                if len(run) >= 2 and max(run) < 17:
                    rest_n = n - len(run) * 3
                    if rest_n == 0:
                        return ('airplane', RANK_ORDER[max(run)], n)
                    if rest_n == len(run):
                        return ('air1', RANK_ORDER[max(run)], n)
                    if rest_n == len(run) * 2:
                        return ('air2', RANK_ORDER[max(run)], n)
    # 四带二
    if n == 6:
        for v, c in cnt.items():
            if c == 4:
                return ('four2', RANK_ORDER[v], 6)
    if n == 8:
        for v, c in cnt.items():
            if c == 4:
                return ('four22', RANK_ORDER[v], 8)
    return (None, None, None)
