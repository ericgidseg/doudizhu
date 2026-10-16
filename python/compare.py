"""对比两组 ONNX 模型的自对弈胜率.

用法:
  python3 compare.py                              # models/ vs models_v2/
  python3 compare.py -a models/ -b models_v2/ -n 200
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import onnxruntime as ort
from env.env import Env, get_obs
from selfplay import run_episode, predict_onnx


def evaluate_dir(model_dir, n_games, label=''):
    """评估一组 ONNX 模型的自对弈表现."""
    print(f'\n=== Evaluating {label or model_dir} ({n_games} games) ===')
    ll_wins = 0
    total_reward = 0.0
    total_steps = 0

    for i in range(n_games):
        _, rewards, winner, steps = run_episode(
            models=None, eps=0.0, model_dir=model_dir)
        total_reward += rewards['landlord']
        total_steps += steps
        if winner == 'landlord':
            ll_wins += 1

        if (i + 1) % max(n_games // 5, 10) == 0:
            print(f'  [{i+1}/{n_games}] LL_WR={ll_wins/(i+1):.2f} '
                  f'avg_R={total_reward/(i+1):.2f}')

    ll_wr = ll_wins / n_games
    avg_r = total_reward / n_games
    avg_steps = total_steps / n_games
    print(f'  Final: LL_WR={ll_wr:.2f} avg_R={avg_r:.2f} '
          f'avg_steps={avg_steps:.0f}')
    return ll_wr, avg_r


def cross_evaluate(dir_a, dir_b, n_games, label_a='A', label_b='B'):
    """A 做地主, B 做农民, 看谁的模型更强.

    通过混合模型跑自对弈: landlord 用 dir_a, farmers 用 dir_b.
    """
    print(f'\n=== Cross: {label_a}(LL) vs {label_b}(Farmers) ({n_games} games) ===')
    ll_wins = 0
    total_reward = 0.0

    for i in range(n_games):
        env = Env(objective='adp')
        env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            pos = env._acting_player_position
            obs = get_obs(env.infoset)

            if pos == 'landlord':
                values = predict_onnx(obs, pos, dir_a)
            else:
                values = predict_onnx(obs, pos, dir_b)

            action_idx = int(np.argmax(values))
            action = obs['legal_actions'][action_idx]
            _, _, done, _ = env.step(action)
            steps += 1

        winner = env._game_winner
        bomb = env._game_bomb_num
        r = 2.0 ** bomb if winner == 'landlord' else -(2.0 ** bomb)
        total_reward += r
        if winner == 'landlord':
            ll_wins += 1

    ll_wr = ll_wins / n_games
    avg_r = total_reward / n_games
    print(f'  {label_a} as LL: LL_WR={ll_wr:.2f} avg_R={avg_r:.2f}')
    return ll_wr, avg_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--dir-a', type=str, default='models',
                        help='模型 A 目录 (默认: models/)')
    parser.add_argument('-b', '--dir-b', type=str, default='models_v2',
                        help='模型 B 目录 (默认: models_v2/)')
    parser.add_argument('-n', '--num-games', type=int, default=100)
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    dir_a = os.path.join(base, args.dir_a)
    dir_b = os.path.join(base, args.dir_b)

    print(f'Model A: {dir_a}')
    print(f'Model B: {dir_b}')

    # 各自自对弈
    ll_a, r_a = evaluate_dir(dir_a, args.num_games, 'Model A (baseline)')
    ll_b, r_b = evaluate_dir(dir_b, args.num_games, 'Model B (trained)')

    # 交叉对比: A 地主 vs B 农民
    cross_evaluate(dir_a, dir_b, args.num_games, 'A', 'B')
    # B 地主 vs A 农民
    cross_evaluate(dir_b, dir_a, args.num_games, 'B', 'A')

    print(f'\n=== Summary ===')
    print(f'A selfplay: LL_WR={ll_a:.2f} avg_R={r_a:.2f}')
    print(f'B selfplay: LL_WR={ll_b:.2f} avg_R={r_b:.2f}')


if __name__ == '__main__':
    main()
