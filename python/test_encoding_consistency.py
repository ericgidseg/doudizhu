"""验证 encoding.get_obs 与 env.env.get_obs 对同一 infoset 产出完全一致.

修复前: encoding.py 的 landlord_up / landlord_down x_batch 字段顺序与
env/env.py 不一致, 会让 ONNX 模型拿到错位的特征向量 (shape 相同不报错).

用法: python3 test_encoding_consistency.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.env import Env, get_obs as env_get_obs
from encoding import get_obs as enc_get_obs


def compare(pos_label):
    env = Env(objective='adp')
    env.reset()
    iset = env.infoset
    # 强制构造指定位置的 infoset (env 只给当前 acting player, 我们换位置重算)
    from env.game import InfoSet
    if iset.player_position != pos_label:
        return None
    a = env_get_obs(iset)
    b = enc_get_obs(iset)
    same_x = np.array_equal(a['x_batch'], b['x_batch'])
    same_z = np.array_equal(a['z_batch'], b['z_batch'])
    return same_x, same_z, a['x_batch'].shape


def main():
    # env 的 acting player 总是从 landlord 开始, 打几步让位置轮转
    results = {}
    env = Env(objective='adp')
    env.reset()
    for step in range(20):
        iset = env.infoset
        pos = iset.player_position
        if pos not in results:
            a = env_get_obs(iset)
            b = enc_get_obs(iset)
            results[pos] = (
                np.array_equal(a['x_batch'], b['x_batch']),
                np.array_equal(a['z_batch'], b['z_batch']),
                a['x_batch'].shape,
            )
        # 随机走一步
        actions = iset.legal_actions
        env.step(actions[np.random.randint(len(actions))])
        if env._env.game_over:
            break

    ok = True
    for pos in ['landlord', 'landlord_up', 'landlord_down']:
        if pos not in results:
            print(f'{pos}: not reached, skipped')
            continue
        same_x, same_z, shape = results[pos]
        status = 'PASS' if (same_x and same_z) else 'FAIL'
        if status == 'FAIL':
            ok = False
        print(f'{pos}: x_batch={shape} x_equal={same_x} z_equal={same_z}  [{status}]')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
