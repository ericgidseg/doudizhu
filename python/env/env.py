"""DouZero 环境包装 + 特征编码 (官方布局, ONNX 权重按此训练).

x_batch 字段顺序:
  landlord:      my_hand, other, last, up_played, down_played,
                 up_num(17), down_num(17), bomb(15), my_action
  landlord_up/down: my_hand, other, lord_played, mate_played, last,
                 last_lord, last_mate, lord_num(20), mate_num(17), bomb, my_action

改这里必须同步 encoding.py, 并跑 test_encoding_consistency.py.
"""
from collections import Counter
import numpy as np

from .game import GameEnv

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])


class Env:
    """DouZero 多智能体包装: 用 dummy players 隔离环境与决策者."""

    def __init__(self, objective):
        self.objective = objective
        self.players = {pos: DummyAgent(pos)
                        for pos in ('landlord', 'landlord_up', 'landlord_down')}
        self._env = GameEnv(self.players)
        self.infoset = None

    def reset(self):
        self._env.reset()

        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'landlord': _deck[:20],
                          'landlord_up': _deck[20:37],
                          'landlord_down': _deck[37:54],
                          'three_landlord_cards': _deck[17:20]}
        for key in card_play_data:
            card_play_data[key].sort()

        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset
        return get_obs(self.infoset)

    def step(self, action):
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        if self._game_over:
            return None, self._get_reward(), True, {}
        return get_obs(self.infoset), 0.0, False, {}

    def _get_reward(self):
        """地主赢为正, 农民赢为负; adp 按炸弹数翻倍, logadp 线性."""
        bomb_num = self._game_bomb_num
        if self._game_winner == 'landlord':
            sign = 1.0
        else:
            sign = -1.0
        if self.objective == 'adp':
            return sign * 2.0 ** bomb_num
        if self.objective == 'logadp':
            return sign * (bomb_num + 1.0)
        return sign

    @property
    def _game_infoset(self):
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        return self._env.acting_player_position

    @property
    def _game_over(self):
        return self._env.game_over


class DummyAgent(object):
    """环境把动作告诉它, 轮到它时原样返回."""

    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        self.action = action


def get_obs(infoset):
    """从 infoset 编码出 {x_batch, z_batch, legal_actions} (按位置分三种特征)."""
    n = len(infoset.legal_actions)
    rep = lambda a: np.repeat(a[np.newaxis, :], n, axis=0)

    my_handcards = rep(_cards2array(infoset.player_hand_cards))
    other_handcards = rep(_cards2array(infoset.other_hand_cards))
    last_action = rep(_cards2array(infoset.last_move))

    my_action_batch = np.zeros(my_handcards.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    bomb_num = rep(_get_one_hot_bomb(infoset.bomb_num))
    pos = infoset.player_position

    if pos == 'landlord':
        x_batch = np.hstack((
            my_handcards, other_handcards, last_action,
            rep(_cards2array(infoset.played_cards['landlord_up'])),
            rep(_cards2array(infoset.played_cards['landlord_down'])),
            rep(_get_one_hot_array(infoset.num_cards_left_dict['landlord_up'], 17)),
            rep(_get_one_hot_array(infoset.num_cards_left_dict['landlord_down'], 17)),
            bomb_num, my_action_batch))
    else:
        mate = 'landlord_down' if pos == 'landlord_up' else 'landlord_up'
        x_batch = np.hstack((
            my_handcards, other_handcards,
            rep(_cards2array(infoset.played_cards['landlord'])),
            rep(_cards2array(infoset.played_cards[mate])),
            last_action,
            rep(_cards2array(infoset.last_move_dict['landlord'])),
            rep(_cards2array(infoset.last_move_dict[mate])),
            rep(_get_one_hot_array(infoset.num_cards_left_dict['landlord'], 20)),
            rep(_get_one_hot_array(infoset.num_cards_left_dict[mate], 17)),
            bomb_num, my_action_batch))

    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :], n, axis=0)

    return {'position': pos,
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions}


def _get_one_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1
    return one_hot


def _cards2array(list_cards):
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


def _action_seq_list2array(action_seq_list):
    """15 步历史动作 -> 5×162 (每 3 步一行, 喂 LSTM)."""
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    return action_seq_array.reshape(5, 162)


def _process_action_seq(sequence, length=15):
    """保留最近 15 步, 不足前面补空."""
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


def _get_one_hot_bomb(bomb_num):
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot
