"""斗地主牌型判定/比较 —— 对齐 public 前端 cards.js 的规则.

牌对象: {id, rank, suit}  rank ∈ 3..2, SJ(小王), BJ(大王)
内部计算用 DouZero 点数: 3-14, 17=2, 20=小王, 30=大王
"""
from collections import Counter

RANK_ORDER = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'SJ', 'BJ']
RANK_VALUE = {r: i for i, r in enumerate(RANK_ORDER)}
V2 = RANK_VALUE['2']
SUITS = ['♠', '♥', '♣', '♦']

TYPE = type('TYPE', (), dict(
    INVALID=0, SINGLE=1, PAIR=2, TRIPLE=3, TRIPLE_1=4, TRIPLE_2=5,
    CHAIN=6, PAIR_CHAIN=7, AIRPLANE=8, AIRPLANE_1=9, AIRPLANE_2=10,
    FOUR_2=11, FOUR_2_2=12, BOMB=13, ROCKET=14,
))
TYPE_NAMES = {
    0: '无效牌型', 1: '单张', 2: '对子', 3: '三张', 4: '三带一', 5: '三带二',
    6: '顺子', 7: '连对', 8: '飞机', 9: '飞机带单', 10: '飞机带对',
    11: '四带二', 12: '四带二对', 13: '炸弹', 14: '王炸',
}

# DouZero 点数映射 (env 里用 17=2, 20=X, 30=D)
RANK_TO_DZ = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
              'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 17, 'SJ': 20, 'BJ': 30}
DZ_TO_RANK = {v: k for k, v in RANK_TO_DZ.items()}


def create_deck():
    deck = []
    cid = 0
    for suit in SUITS:
        for rank in RANK_ORDER[:-2]:        # 跳过大小王
            deck.append({'id': cid, 'rank': rank, 'suit': suit})
            cid += 1
    deck.append({'id': cid, 'rank': 'SJ', 'suit': ''})
    deck.append({'id': cid + 1, 'rank': 'BJ', 'suit': ''})
    return deck


def shuffle(arr):
    import random
    arr = arr[:]
    for i in range(len(arr) - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def sort_cards(cards):
    return sorted(cards, key=lambda c: (-RANK_VALUE[c['rank']], c['id']))


def _ok(t, n, main):
    return {'type': t, 'typeName': TYPE_NAMES[t], 'len': n, 'main': main}


def analyze(cards):
    """返回 {type, typeName, len, main}."""
    n = len(cards)
    if n == 0:
        return {'type': TYPE.INVALID, 'typeName': TYPE_NAMES[0], 'len': 0, 'main': -1}

    g = Counter(RANK_VALUE[c['rank']] for c in cards)
    keys = sorted(g)
    bad = {'type': TYPE.INVALID, 'typeName': TYPE_NAMES[0], 'len': n, 'main': -1}

    if n == 2 and keys == [RANK_VALUE['SJ'], RANK_VALUE['BJ']]:
        return _ok(TYPE.ROCKET, n, float('inf'))
    if n == 1:
        return _ok(TYPE.SINGLE, n, keys[0])
    if n == 4 and len(keys) == 1:
        return _ok(TYPE.BOMB, n, keys[0])
    if len(keys) == 1:
        if n == 2 and keys[0] < RANK_VALUE['SJ']:
            return _ok(TYPE.PAIR, n, keys[0])
        if n == 3:
            return _ok(TYPE.TRIPLE, n, keys[0])
        return bad

    triple = [k for k in keys if g[k] == 3]
    four = [k for k in keys if g[k] == 4]

    # 三带一 / 三带二
    if len(triple) == 1 and not four:
        t = triple[0]
        rest = [k for k in keys if k != t]
        if n == 4:
            return _ok(TYPE.TRIPLE_1, n, t)
        if n == 5 and len(rest) == 1 and g[rest[0]] == 2:
            return _ok(TYPE.TRIPLE_2, n, t)

    # 四带二 / 四带二对
    if len(four) == 1:
        f = four[0]
        rest = [k for k in keys if k != f]
        if n == 6 and sum(g[k] for k in rest) == 2:
            return _ok(TYPE.FOUR_2, n, f)
        if n == 8 and len(rest) == 2 and all(g[k] == 2 for k in rest):
            return _ok(TYPE.FOUR_2_2, n, f)
        return bad

    def consecutive(vals):
        return all(vals[i] == vals[i - 1] + 1 for i in range(1, len(vals)))

    # 顺子 ≥5 张 / 连对 ≥3 对 / 飞机 ≥2 组, 均不含 2 和王
    if keys[-1] < V2 and consecutive(keys):
        if n >= 5 and n == len(keys):
            return _ok(TYPE.CHAIN, n, keys[-1])
        if n >= 6 and n == len(keys) * 2 and all(g[k] == 2 for k in keys):
            return _ok(TYPE.PAIR_CHAIN, n, keys[-1])
        if n >= 6 and all(g[k] == 3 for k in keys):
            return _ok(TYPE.AIRPLANE, n, keys[-1])

    # 飞机带翅: 枚举连续三张段作为主体, 剩下的凑 m 张单 / m 对
    cand = [k for k in triple if k < V2]
    for end in range(len(cand)):
        for start in range(end - 1, -1, -1):
            seg = cand[start:end + 1]
            if not consecutive(seg):
                break
            m = len(seg)
            rest = [k for k in keys if k not in seg]
            rest_count = sum(g[k] for k in rest)
            if rest_count == m:
                return _ok(TYPE.AIRPLANE_1, n, seg[-1])
            if len(rest) == m and all(g[k] == 2 for k in rest):
                return _ok(TYPE.AIRPLANE_2, n, seg[-1])
    return bad


def can_beat(prev, nxt):
    """prev 为上家 analyze 结果 (None 表示领出), nxt 能否压过."""
    if nxt['type'] == TYPE.INVALID:
        return False
    if prev is None:
        return True
    if nxt['type'] == TYPE.ROCKET:
        return True
    if prev['type'] == TYPE.ROCKET:
        return False
    if nxt['type'] == TYPE.BOMB and prev['type'] != TYPE.BOMB:
        return True
    if prev['type'] == TYPE.BOMB:
        return nxt['type'] == TYPE.BOMB and nxt['main'] > prev['main']
    if nxt['type'] != prev['type'] or nxt['len'] != prev['len']:
        return False
    return nxt['main'] > prev['main']


# ---- DouZero 编码辅助 ----

def hand_to_dz(hand):
    """{id,rank,suit}[] -> DouZero 点数列表"""
    return sorted(RANK_TO_DZ[c['rank']] for c in hand)


def dz_to_cards(dz_nums, hand):
    """DouZero 点数列表 -> 从手牌中匹配出的 {id,rank,suit}[]"""
    result = []
    used = set()
    for n in dz_nums:
        target_rank = DZ_TO_RANK[n]
        for c in hand:
            if c['id'] not in used and c['rank'] == target_rank:
                result.append(c)
                used.add(c['id'])
                break
    return result
