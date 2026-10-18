"""锁定 encoding.get_obs 的 x_batch 字段布局 — 防止静默错位.

背景: 曾有一份 env/env.py 里的编码副本与 encoding.py 单边跑错过, 两边
shape 完全相同 (373 / 484), 不报任何错, 只是模型拿到错位的特征向量,
表现为莫名掉胜率. 合并后唯一定义在 encoding.py, 但这个风险还在 ——
任何人重排 x_batch 的 hstack 顺序, 依然不会报错, 只会悄悄变弱.

所以这里直接断言每个字段块在 x_batch 里的切片: 手工构造一个字段互不
重叠的 infoset (玩家 A K Q, 上家 AA, 下家 KK...), 检查每段的牌点是否
落在预期字段上. 想改布局的人会先在这里看到 FAIL, 而不是在几小时的
训练后才发现胜率不对.

用法: python3 test_encoding_consistency.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encoding import get_obs
from env.game import InfoSet

# A K Q -> 列索引 11/10/9. cards2array 是 matrix.flatten('F') 列优先展开,
# 点数 v 的第 i 张落在偏移 v*4 + i 处 (不是 i*13+v).
COL_A, COL_K, COL_Q = 11, 10, 9

# landlord 的 x_batch 布局 (每段宽, 单位: 元素)
W = {'my_hand': 54, 'other': 54, 'last': 54, 'up_played': 54, 'down_played': 54,
     'up_num': 17, 'down_num': 17, 'bomb': 15, 'my_action': 54}


def block54(points):
    """{点数: 张数} -> 54 维 one-hot 期望向量."""
    v = np.zeros(54, dtype=np.float32)
    for pt, cnt in points.items():
        for i in range(cnt):
            v[pt * 4 + i] = 1.0
    return v


def make_infoset():
    """字段互不重叠: 我方 A K Q; other=AA(上家)+KK(下家);
    up_played=A, down_played=K; last=K; my_action=KK."""
    iset = InfoSet('landlord')
    iset.player_hand_cards = [14, 13, 12]                  # A K Q
    iset.other_hand_cards = [14, 14, 13, 13]               # AA + KK
    iset.legal_actions = [[13, 13], [14], [12], [14, 13]]
    iset.last_move = [13]
    iset.last_move_dict = {'landlord': [12], 'landlord_up': [14], 'landlord_down': [13]}
    iset.num_cards_left_dict = {'landlord': 20, 'landlord_up': 2, 'landlord_down': 2}
    iset.played_cards = {'landlord': [12], 'landlord_up': [14], 'landlord_down': [13]}
    iset.card_play_action_seq = [[12], [14], [13], [12]]
    iset.bomb_num = 0
    iset.three_landlord_cards = []
    return iset


def main():
    obs = get_obs(make_infoset())
    x = obs['x_batch'][0]

    # 按W 的顺序切片
    seg, off = {}, 0
    for name, w in W.items():
        seg[name] = x[off:off + w]
        off += w
    assert off == x.shape[0], f'x_dim 应为 {off}, 实际 {x.shape[0]}'

    checks = [
        ('my_hand',     seg['my_hand'],    block54({COL_A: 1, COL_K: 1, COL_Q: 1})),
        ('last',        seg['last'],       block54({COL_K: 1})),
        ('up_played',   seg['up_played'],  block54({COL_A: 1})),
        ('down_played', seg['down_played'], block54({COL_K: 1})),
        ('my_action',   seg['my_action'],  block54({COL_K: 2})),
        ('up_num',      seg['up_num'],     np.eye(17, dtype=np.float32)[1]),   # 2 张
        ('down_num',    seg['down_num'],   np.eye(17, dtype=np.float32)[1]),
        ('bomb',        seg['bomb'],       np.eye(15, dtype=np.float32)[0]),
        # other 是两个各 2 张的点: 列优先下 v*4+i 命中前两格
        ('other.AA',    seg['other'][COL_A * 4:COL_A * 4 + 2], np.ones(2, dtype=np.float32)),
        ('other.KK',    seg['other'][COL_K * 4:COL_K * 4 + 2], np.ones(2, dtype=np.float32)),
    ]

    ok = True
    for name, actual, expected in checks:
        good = np.array_equal(np.asarray(actual), np.asarray(expected))
        if not good:
            ok = False
        print(f'  {"PASS" if good else "FAIL"}  {name}')

    # z_batch: 最近一手 (card_play_action_seq 最后一项 = Q) 应落在最后一行第三列
    z = obs['z_batch'][0]
    if not np.array_equal(z[4, 108:162], block54({COL_Q: 1})):
        ok = False
        print('  FAIL  z.last')
    else:
        print('  PASS  z.last')

    print(f'x_dim={x.shape[0]}  legal_actions={len(obs["legal_actions"])}')
    if not ok:
        print('\n布局与注释不符 — 改 x_batch 顺序时必须同步本测试的断言')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
