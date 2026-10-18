"""DouZero 环境包装 (官方布局, ONNX 权重按此训练).

特征编码的唯一定义在 ../encoding.py — 训练 (selfplay / arena / test_train)
和线上推理 (ai_douzero) 必须走同一份实现. 此前这里曾复制一份编码逻辑,
和 encoding.py 单边演化出过 x_batch 静默错位 (shape 相同不报错, 只掉胜率),
所以现在直接从 encoding.py 导入, 本模块不再保留副本.
"""
import numpy as np

from .game import GameEnv
import encoding

# 兼容旧引用名: _cards2array / _get_one_hot_array 等此前是本模块私有函数,
# 统一转发到 encoding.py 的同一实现 (不新建别名副本, 保证单点维护).
get_obs = encoding.get_obs
_cards2array = encoding.cards2array
_get_one_hot_array = encoding._one_hot
_get_one_hot_bomb = encoding._bomb_one_hot
_action_seq_list2array = encoding._action_seq_list2array
_process_action_seq = encoding._process_action_seq
Card2Column = encoding.Card2Column
NumOnes2Array = encoding.NumOnes2Array

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

