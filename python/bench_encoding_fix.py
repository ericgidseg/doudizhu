"""配对 A/B: 农民用修复前 vs 修复后 encoding.py 编码 x_batch, 谁赢?

控制变量: 地主永远走 env.env.get_obs (正确编码) + vote 集成 argmax.
两组对比只换农民的编码器; 推理都用同一组 ONNX 权重 (models/), argmax 选动作.

用法: python3 bench_encoding_fix.py -n 400
"""
import os
import sys
import time
import argparse
import importlib.util
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault('ORT_NUM_THREADS', '2')

import onnxruntime as ort
from env.env import Env, get_obs as env_get_obs
from arena_vote import vote_values, make_dirs, clear_cache
import encoding as enc_new

spec = importlib.util.spec_from_file_location(
    'encoding_old', '/tmp/encoding_old.py')
enc_old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc_old)

BASE_SEED = 90900001
_sessions = {}


def sess(pos):
    if pos not in _sessions:
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ['ORT_NUM_THREADS'])
        so.inter_op_num_threads = 1
        _sessions[pos] = ort.InferenceSession(
            f'models/{pos}.onnx', so, providers=['CPUExecutionProvider'])
    return _sessions[pos]


def farmer_pick(iset, encoder):
    """用给定 encoder 编码, ADP 模型 argmax 选动作."""
    obs = encoder(iset)
    v = sess(iset.player_position).run(['values'], {
        'z': obs['z_batch'], 'x': obs['x_batch']})[0].flatten()
    return iset.legal_actions[int(np.argmax(v))]


def play_one(seed, farmer_encoder, dirs):
    """地主 vote, 农民用指定 encoder + argmax. 返回地主是否赢."""
    np.random.seed(seed)
    env = Env(objective='adp')
    env.reset()
    while not env._env.game_over:
        iset = env.infoset
        pos = iset.player_position
        if pos == 'landlord':
            obs = env_get_obs(iset)
            v = vote_values(obs, pos, dirs)
            action = iset.legal_actions[int(np.argmax(v))]
        else:
            action = farmer_pick(iset, farmer_encoder)
        env.step(action)
    return 1 if env._game_winner == 'landlord' else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--num-games', dest='n', type=int, default=400)
    args = p.parse_args()

    clear_cache()
    dirs = make_dirs()

    wins_old = wins_new = 0   # wins_* = 地主胜率 (农民弱 -> 地主赢)
    diffs = []
    t0 = time.time()
    for i in range(args.n):
        seed = BASE_SEED + i
        r_old = play_one(seed, enc_old.get_obs, dirs)   # 农民走旧编码
        r_new = play_one(seed, enc_new.get_obs, dirs)   # 农民走新编码
        wins_old += r_old
        wins_new += r_new
        # diffs[i] = 地主在"农民用旧编码"那局赢? - 在"农民用新编码"那局赢?
        # 若修复让农民变强, 则新编码下地主胜率更低 -> diff > 0
        diffs.append(r_old - r_new)
        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{args.n}] LL_WR(农民旧)={wins_old/(i+1):.3f} '
                  f'LL_WR(农民新)={wins_new/(i+1):.3f} ({time.time()-t0:.0f}s)')

    diffs = np.array(diffs)
    wr_old, wr_new = wins_old / args.n, wins_new / args.n
    md = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(args.n)

    print(f'\n=== 结果 ({args.n} 局配对) ===')
    print(f'地主胜率 (农民用旧编码): {wr_old:.4f}')
    print(f'地主胜率 (农民用新编码): {wr_new:.4f}')
    print(f'配对差值 (旧-新): {md:+.4f}  标准误 {se:.4f}')
    if se > 0:
        z = md / se
        print(f'z = {z:+.2f}  (>1.96 为显著)')
        if z > 1.96:
            print('=> 修复显著: 新编码的农民更强 (地主胜率下降)')
        elif z < -1.96:
            print('=> 反向: 旧编码的农民反而更强?')
        else:
            print('=> 噪声内, 无显著差异')
    both = int((diffs == 0).sum())
    print(f'两编码结果相同的局: {both}/{args.n} ({both/args.n:.1%})')


if __name__ == '__main__':
    main()
