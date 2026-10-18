"""DouZero 斗地主游戏引擎 + InfoSet 定义.

GameEnv 管理一局完整游戏 (三家手牌/出牌历史/炸弹数/胜负),
供 selfplay/train 做自对弈. InfoSet 同时被 ai_douzero.decide() 直接构造.
"""
from copy import deepcopy
from . import move_detector as md, move_selector as ms
from .move_generator import MovesGener

bombs = [[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6],
         [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10],
         [11, 11, 11, 11], [12, 12, 12, 12], [13, 13, 13, 13], [14, 14, 14, 14],
         [17, 17, 17, 17], [20, 30]]


class GameEnv(object):

    def __init__(self, players):
        self.players = players
        self.reset()

    def reset(self):
        self.card_play_action_seq = []
        self.three_landlord_cards = None
        self.game_over = False
        self.acting_player_position = None
        self.last_move_dict = {'landlord': [], 'landlord_up': [], 'landlord_down': []}
        self.played_cards = {'landlord': [], 'landlord_up': [], 'landlord_down': []}
        self.last_move = []
        self.info_sets = {'landlord': InfoSet('landlord'),
                          'landlord_up': InfoSet('landlord_up'),
                          'landlord_down': InfoSet('landlord_down')}
        self.bomb_num = 0

    def card_play_init(self, card_play_data):
        for pos in ('landlord', 'landlord_up', 'landlord_down'):
            self.info_sets[pos].player_hand_cards = card_play_data[pos]
        self.three_landlord_cards = card_play_data['three_landlord_cards']
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def game_done(self):
        if any(not self.info_sets[pos].player_hand_cards
               for pos in ('landlord', 'landlord_up', 'landlord_down')):
            self.winner = ('landlord' if not self.info_sets['landlord'].player_hand_cards
                           else 'farmer')
            self.game_over = True

    def get_winner(self):
        return self.winner

    def get_bomb_num(self):
        return self.bomb_num

    def step(self):
        action = self.players[self.acting_player_position].act(self.game_infoset)
        assert action in self.game_infoset.legal_actions

        if action in bombs:
            self.bomb_num += 1

        self.last_move_dict[self.acting_player_position] = action.copy()
        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)
        self.played_cards[self.acting_player_position] += action

        if self.acting_player_position == 'landlord' and action:
            for card in action:
                if card in self.three_landlord_cards:
                    self.three_landlord_cards.remove(card)

        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()

    def get_last_move(self):
        seq = self.card_play_action_seq
        if not seq:
            return []
        return seq[-2] if not seq[-1] else seq[-1]

    def get_acting_player_position(self):
        if self.acting_player_position is None:
            self.acting_player_position = 'landlord'
        else:
            self.acting_player_position = {
                'landlord': 'landlord_down',
                'landlord_down': 'landlord_up',
                'landlord_up': 'landlord',
            }[self.acting_player_position]
        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        if action:
            hand = self.info_sets[self.acting_player_position].player_hand_cards
            for card in action:
                hand.remove(card)
            hand.sort()

    def get_legal_card_play_actions(self):
        pos = self.acting_player_position
        mg = MovesGener(self.info_sets[pos].player_hand_cards)
        rival = self.get_last_move()
        rival_info = md.get_type(rival)
        rival_type = rival_info['type']
        # 序列类牌型的 len 是组数 (连对 3 对 = 3), 由 detector 给出
        rival_len = rival_info.get('len', 1)

        gen = {
            md.TYPE_1_SINGLE: mg.gen_type_1_single,
            md.TYPE_2_PAIR: mg.gen_type_2_pair,
            md.TYPE_3_TRIPLE: mg.gen_type_3_triple,
            md.TYPE_6_3_1: mg.gen_type_6_3_1,
            md.TYPE_7_3_2: mg.gen_type_7_3_2,
            md.TYPE_8_SERIAL_SINGLE: lambda: mg.gen_type_8_serial_single(rival_len),
            md.TYPE_9_SERIAL_PAIR: lambda: mg.gen_type_9_serial_pair(rival_len),
            md.TYPE_10_SERIAL_TRIPLE: lambda: mg.gen_type_10_serial_triple(rival_len),
            md.TYPE_11_SERIAL_3_1: lambda: mg.gen_type_11_serial_3_1(rival_len),
            md.TYPE_12_SERIAL_3_2: lambda: mg.gen_type_12_serial_3_2(rival_len),
            md.TYPE_13_4_2: mg.gen_type_13_4_2,
            md.TYPE_14_4_22: mg.gen_type_14_4_22,
        }
        filt = {
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

        if rival_type == md.TYPE_0_PASS:
            moves = mg.gen_moves()
        elif rival_type == md.TYPE_5_KING_BOMB:
            moves = []
        elif rival_type == md.TYPE_4_BOMB:
            moves = ms.filter_type_4_bomb(
                mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb(), rival)
        else:
            moves = filt[rival_type](gen[rival_type](), rival)
            moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

        if rival:                       # 上家出牌 -> 可以 pass
            moves = moves + [[]]
        for m in moves:
            m.sort()
        return moves

    def get_infoset(self):
        iset = self.info_sets[self.acting_player_position]
        iset.legal_actions = self.get_legal_card_play_actions()
        iset.bomb_num = self.bomb_num
        iset.last_move = self.get_last_move()
        iset.last_move_dict = self.last_move_dict
        iset.num_cards_left_dict = {
            pos: len(self.info_sets[pos].player_hand_cards)
            for pos in ('landlord', 'landlord_up', 'landlord_down')}
        iset.other_hand_cards = [
            c for pos in ('landlord', 'landlord_up', 'landlord_down')
            if pos != self.acting_player_position
            for c in self.info_sets[pos].player_hand_cards]
        iset.played_cards = self.played_cards
        iset.three_landlord_cards = self.three_landlord_cards
        iset.card_play_action_seq = self.card_play_action_seq
        return deepcopy(iset)


class InfoSet(object):
    """当前决策点可见的全部信息 (由 env 或 ai_douzero.decide 填充)."""

    def __init__(self, player_position):
        self.player_position = player_position
        self.player_hand_cards = None      # 我的手牌
        self.num_cards_left_dict = None    # {position: 剩余张数}
        self.three_landlord_cards = None   # 底牌
        self.card_play_action_seq = None   # 历史动作序列
        self.other_hand_cards = None       # 另外两家的手牌并集
        self.legal_actions = None          # 合法动作
        self.last_move = None              # 最近一手有效出牌
        self.last_move_dict = None         # {position: 该家最近一手}
        self.played_cards = None           # {position: 已出牌}
        self.bomb_num = None               # 已出炸弹数
