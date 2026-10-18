"""DouZero 状态编码 — 与 env/env.py 的 _get_obs_* 逐字段一致.

牌编码: 点数编号 3-14(3-K,A), 17=2, 20=小王, 30=大王.
每个牌集合编成 54 维向量: 4×13 矩阵按列展开 + 2 joker 位.

x_batch 字段顺序 (改这里必须同步 env/env.py, 否则静默错位):
  landlord:      my_hand, other, last, up_played, down_played,
                 up_num, down_num, bomb, my_action
  landlord_up/down: my_hand, other, lord_played, mate_played, last,
                 last_lord, last_mate, lord_num, mate_num, bomb, my_action
"""
import numpy as np
from collections import Counter

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}
NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}


def cards2array(list_cards):
    """牌集合 -> 54 维向量 (4×13 矩阵按列 + 2 joker)."""
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)
    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    for card, num_times in Counter(list_cards).items():
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


def _one_hot(num, max_num):
    one_hot = np.zeros(max_num)
    one_hot[num - 1] = 1
    return one_hot


def _bomb_one_hot(bomb_num):
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot


def get_obs(infoset):
    """返回 {x_batch, z_batch, legal_actions} 供 ONNX 模型推理."""
    n = len(infoset.legal_actions)
    rep = lambda a: np.repeat(a[np.newaxis, :], n, axis=0)

    my_hand = rep(cards2array(infoset.player_hand_cards))
    other = rep(cards2array(infoset.other_hand_cards))
    last = rep(cards2array(infoset.last_move))

    my_action = np.zeros(my_hand.shape)
    for j, act in enumerate(infoset.legal_actions):
        my_action[j, :] = cards2array(act)

    bomb = rep(_bomb_one_hot(infoset.bomb_num))
    pos = infoset.player_position

    if pos == 'landlord':
        lu_played = rep(cards2array(infoset.played_cards['landlord_up']))
        ld_played = rep(cards2array(infoset.played_cards['landlord_down']))
        lu_num = rep(_one_hot(infoset.num_cards_left_dict['landlord_up'], 17))
        ld_num = rep(_one_hot(infoset.num_cards_left_dict['landlord_down'], 17))
        x_batch = np.hstack((my_hand, other, last, lu_played, ld_played,
                             lu_num, ld_num, bomb, my_action))
    else:
        mate = 'landlord_down' if pos == 'landlord_up' else 'landlord_up'
        lord_played = rep(cards2array(infoset.played_cards['landlord']))
        mate_played = rep(cards2array(infoset.played_cards[mate]))
        last_lord = rep(cards2array(infoset.last_move_dict['landlord']))
        last_mate = rep(cards2array(infoset.last_move_dict[mate]))
        lord_num = rep(_one_hot(infoset.num_cards_left_dict['landlord'], 20))
        mate_num = rep(_one_hot(infoset.num_cards_left_dict[mate], 17))
        x_batch = np.hstack((my_hand, other, lord_played, mate_played,
                             last, last_lord, last_mate,
                             lord_num, mate_num, bomb, my_action))

    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :], n, axis=0)

    return {'position': pos,
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions}
