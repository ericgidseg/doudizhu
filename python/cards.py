"""斗地主牌型判定/比较/提示 —— 对齐 cards.js 的语义.

牌对象: {id, rank, suit}  rank ∈ 3..2, SJ(小王), BJ(大王)
内部计算用 DouZero 点数: 3-14, 17=2, 20=小王, 30=大王
"""
from collections import Counter

RANK_ORDER = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','SJ','BJ']
RANK_VALUE = {r: i for i, r in enumerate(RANK_ORDER)}
V2 = RANK_VALUE['2']
SUITS = ['♠','♥','♣','♦']

# 牌型枚举 (和 cards.js TYPE 一致)
TYPE = type('TYPE', (), dict(
    INVALID=0, SINGLE=1, PAIR=2, TRIPLE=3, TRIPLE_1=4, TRIPLE_2=5,
    CHAIN=6, PAIR_CHAIN=7, AIRPLANE=8, AIRPLANE_1=9, AIRPLANE_2=10,
    FOUR_2=11, FOUR_2_2=12, BOMB=13, ROCKET=14,
))
TYPE_NAMES = {
    1: '单张', 2: '对子', 3: '三张', 4: '三带一', 5: '三带二',
    6: '顺子', 7: '连对', 8: '飞机', 9: '飞机带单', 10: '飞机带对',
    11: '四带二', 12: '四带二对', 13: '炸弹', 14: '王炸', 0: '无效牌型',
}

# DouZero 点数映射 (env 里用 17=2, 20=X, 30=D)
RANK_TO_DZ = {'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,
              'J':11,'Q':12,'K':13,'A':14,'2':17,'SJ':20,'BJ':30}
DZ_TO_RANK = {v:k for k,v in RANK_TO_DZ.items()}


def create_deck():
    deck = []
    cid = 0
    for suit in SUITS:
        for rank in RANK_ORDER:
            if rank in ('SJ', 'BJ'):
                continue
            deck.append({'id': cid, 'rank': rank, 'suit': suit})
            cid += 1
    deck.append({'id': cid, 'rank': 'SJ', 'suit': ''}); cid += 1
    deck.append({'id': cid, 'rank': 'BJ', 'suit': ''})
    return deck


def shuffle(arr):
    import random
    arr = arr[:]
    for i in range(len(arr)-1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def val(card):
    return RANK_VALUE[card['rank']]


def sort_cards(cards):
    return sorted(cards, key=lambda c: (-val(c), c['id']))


def group_by_val(cards):
    g = {}
    for c in cards:
        v = val(c)
        g.setdefault(v, []).append(c)
    return g


def is_consecutive(vals):
    return all(vals[i] == vals[i-1] + 1 for i in range(1, len(vals)))


def analyze(cards):
    """返回 {type, typeName, len, main}"""
    bad = {'type': TYPE.INVALID, 'typeName': TYPE_NAMES[0],
           'len': len(cards) if cards else 0, 'main': -1}
    if not cards:
        return bad
    n = len(cards)
    g = group_by_val(cards)
    keys = sorted(g.keys())
    ok = lambda t, m: {'type': t, 'typeName': TYPE_NAMES[t], 'len': n, 'main': m}

    if n == 2 and keys == [RANK_VALUE['SJ'], RANK_VALUE['BJ']]:
        return ok(TYPE.ROCKET, float('inf'))
    if n == 1:
        return ok(TYPE.SINGLE, keys[0])
    if n == 4 and len(keys) == 1:
        return ok(TYPE.BOMB, keys[0])
    if len(keys) == 1:
        if n == 2 and keys[0] < RANK_VALUE['SJ']:
            return ok(TYPE.PAIR, keys[0])
        if n == 3:
            return ok(TYPE.TRIPLE, keys[0])
        return bad

    # 三带
    for t in keys:
        if len(g[t]) == 3:
            rest = [k for k in keys if k != t]
            if n == 4:
                return ok(TYPE.TRIPLE_1, t)
            if n == 5 and len(rest) == 1 and len(g[rest[0]]) == 2:
                return ok(TYPE.TRIPLE_2, t)

    # 四带
    for f in keys:
        if len(g[f]) == 4:
            rest = [k for k in keys if k != f]
            if n == 6 and sum(len(g[k]) for k in rest) == 2:
                return ok(TYPE.FOUR_2, f)
            if n == 8 and len(rest) == 2 and all(len(g[k]) == 2 for k in rest):
                return ok(TYPE.FOUR_2_2, f)

    # 顺子/连对/纯飞机 (所有点 < 2 且连续)
    if all(k < V2 for k in keys) and is_consecutive(keys):
        if n == len(keys):
            return ok(TYPE.CHAIN, keys[-1])
        if n == len(keys) * 2 and all(len(g[k]) == 2 for k in keys):
            return ok(TYPE.PAIR_CHAIN, keys[-1])
        if all(len(g[k]) == 3 for k in keys):
            return ok(TYPE.AIRPLANE, keys[-1])

    # 飞机带翅: 允许"一段连续三同张 + 散牌翅膀", 散牌不必连续
    # 例如 444+333+8+6 (2连飞机带2单翅) 或 444+333+88+66 (带2对翅)
    if all(k < V2 for k in keys):
        # 先找所有"三同张"的连续 run
        triple_ranks = sorted(k for k in keys if len(g[k]) == 3)
        if len(triple_ranks) >= 2:
            # 找最长连续段
            max_run = []
            cur = [triple_ranks[0]]
            for k in triple_ranks[1:]:
                if k - cur[-1] == 1:
                    cur.append(k)
                else:
                    if len(cur) > len(max_run):
                        max_run = cur
                    cur = [k]
            if len(cur) > len(max_run):
                max_run = cur
            if len(max_run) >= 2:
                m = len(max_run)
                rest = [k for k in keys if k not in max_run]
                rest_count = sum(len(g[k]) for k in rest)
                if rest_count == m:
                    return ok(TYPE.AIRPLANE_1, max_run[-1])
                if rest_count == m * 2 and all(len(g[k]) == 2 for k in rest):
                    return ok(TYPE.AIRPLANE_2, max_run[-1])
    return bad


def can_beat(prev, nxt):
    """prev 为上家 analyze 结果 (None 表示领出), nxt 能否压过."""
    if nxt['type'] == TYPE.INVALID:
        return False
    if prev is None:
        return True                  # 领出: 任何合法牌型都行
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
    nums = list(dz_nums)
    result = []
    used = set()
    for n in nums:
        target_rank = DZ_TO_RANK[n]
        for c in hand:
            if c['id'] not in used and c['rank'] == target_rank:
                result.append(c)
                used.add(c['id'])
                break
    return result


def hint_play(hand, last_ana):
    """提示: 找最小能压的牌."""
    # 简单贪心: 枚举所有组合找最小能压的
    from itertools import combinations
    n = len(hand)
    best = None
    for size in range(1, n + 1):
        for combo in combinations(hand, size):
            ana = analyze(list(combo))
            if ana['type'] != TYPE.INVALID and can_beat(last_ana, ana):
                if best is None or ana['main'] < best['ana']['main']:
                    best = {'ids': [c['id'] for c in combo], 'ana': ana}
    return best['ids'] if best else []
