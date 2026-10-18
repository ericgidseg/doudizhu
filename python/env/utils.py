import itertools

# 顺子/连对/飞机的最短长度
MIN_SINGLE_CARDS = 5
MIN_PAIRS = 3
MIN_TRIPLES = 2

# action types
TYPE_0_PASS = 0
TYPE_1_SINGLE = 1
TYPE_2_PAIR = 2
TYPE_3_TRIPLE = 3
TYPE_4_BOMB = 4
TYPE_5_KING_BOMB = 5
TYPE_6_3_1 = 6
TYPE_7_3_2 = 7
TYPE_8_SERIAL_SINGLE = 8
TYPE_9_SERIAL_PAIR = 9
TYPE_10_SERIAL_TRIPLE = 10
TYPE_11_SERIAL_3_1 = 11
TYPE_12_SERIAL_3_2 = 12
TYPE_13_4_2 = 13
TYPE_14_4_22 = 14
TYPE_15_WRONG = 15

TYPE_NAMES = {
    TYPE_0_PASS: '不出', TYPE_1_SINGLE: '单张', TYPE_2_PAIR: '对子',
    TYPE_3_TRIPLE: '三张', TYPE_4_BOMB: '炸弹', TYPE_5_KING_BOMB: '王炸',
    TYPE_6_3_1: '三带一', TYPE_7_3_2: '三带二', TYPE_8_SERIAL_SINGLE: '顺子',
    TYPE_9_SERIAL_PAIR: '连对', TYPE_10_SERIAL_TRIPLE: '飞机',
    TYPE_11_SERIAL_3_1: '飞机带单', TYPE_12_SERIAL_3_2: '飞机带对',
    TYPE_13_4_2: '四带二', TYPE_14_4_22: '四带二对', TYPE_15_WRONG: '无效牌型',
}


def select(cards, num):
    """从 cards 里选 num 张的所有组合."""
    return [list(i) for i in itertools.combinations(cards, num)]
